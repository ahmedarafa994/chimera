"""Aegis Campaign Telemetry Broadcaster.

Broadcasts real-time telemetry events from Aegis campaigns to subscribed
WebSocket clients. Integrates with ConnectionManager for campaign-based
subscriptions and provides high-level emission methods for all telemetry
event types.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from threading import Lock
from typing import Any

from app.schemas.aegis_telemetry import (
    AegisTelemetryEvent,
    AegisTelemetryEventType,
    AttackCompletedData,
    AttackMetrics,
    AttackStartedData,
    CampaignCompletedData,
    CampaignFailedData,
    CampaignStartedData,
    CampaignStatus,
    CampaignSummary,
    ConnectionAckData,
    CostUpdateData,
    ErrorData,
    HeartbeatData,
    IterationCompletedData,
    IterationStartedData,
    LatencyMetrics,
    PromptEvolvedData,
    TechniqueAppliedData,
    TechniqueCategory,
    TechniquePerformance,
    TokenUsage,
    create_telemetry_event,
)

logger = logging.getLogger(__name__)


class CampaignState:
    """Tracks state for an active Aegis campaign.

    Maintains sequence counters, subscriptions, and aggregated metrics
    for real-time dashboard updates.
    """

    def __init__(self, campaign_id: str) -> None:
        self.campaign_id = campaign_id
        self.status = CampaignStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at: datetime | None = None
        self.last_updated = datetime.utcnow()
        self.event_sequence = 0
        self.subscribed_clients: set[str] = set()

        # Aggregated metrics
        self.attack_metrics = AttackMetrics()
        self.technique_performance: dict[str, TechniquePerformance] = {}
        self.token_usage = TokenUsage()
        self.latency_metrics = LatencyMetrics()
        self.current_iteration = 0
        self.max_iterations = 5
        self.best_prompt: str | None = None
        self.best_score = 0.0
        self.objective: str = ""
        self.target_model: str | None = None

        # Latency tracking for percentiles
        self._latency_samples: list[float] = []

    def increment_sequence(self) -> int:
        """Increment and return next event sequence number."""
        self.event_sequence += 1
        self.last_updated = datetime.utcnow()
        return self.event_sequence

    def update_attack_metrics(self, success: bool, score: float) -> None:
        """Update attack metrics after an attack attempt."""
        self.attack_metrics.total_attempts += 1

        if success:
            self.attack_metrics.successful_attacks += 1
            self.attack_metrics.current_streak = max(1, self.attack_metrics.current_streak + 1)
        else:
            self.attack_metrics.failed_attacks += 1
            self.attack_metrics.current_streak = min(-1, self.attack_metrics.current_streak - 1)

        # Update success rate
        if self.attack_metrics.total_attempts > 0:
            self.attack_metrics.success_rate = (
                self.attack_metrics.successful_attacks / self.attack_metrics.total_attempts * 100
            )

        # Update scores
        self.attack_metrics.best_score = max(self.attack_metrics.best_score, score)

        # Update running average
        n = self.attack_metrics.total_attempts
        self.attack_metrics.average_score = (
            self.attack_metrics.average_score * (n - 1) + score
        ) / n

        self.last_updated = datetime.utcnow()

    def update_technique_performance(
        self,
        technique_name: str,
        technique_category: TechniqueCategory,
        success: bool,
        score: float,
        execution_time_ms: float,
    ) -> None:
        """Update performance metrics for a technique."""
        if technique_name not in self.technique_performance:
            self.technique_performance[technique_name] = TechniquePerformance(
                technique_name=technique_name,
                technique_category=technique_category,
            )

        perf = self.technique_performance[technique_name]
        perf.total_applications += 1

        if success:
            perf.success_count += 1
        else:
            perf.failure_count += 1

        # Update success rate
        if perf.total_applications > 0:
            perf.success_rate = perf.success_count / perf.total_applications * 100

        # Update best score
        perf.best_score = max(perf.best_score, score)

        # Update running average score
        n = perf.total_applications
        perf.avg_score = (perf.avg_score * (n - 1) + score) / n

        # Update running average execution time
        perf.avg_execution_time_ms = (perf.avg_execution_time_ms * (n - 1) + execution_time_ms) / n

        self.last_updated = datetime.utcnow()

    def update_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Update cumulative token usage."""
        self.token_usage.prompt_tokens += prompt_tokens
        self.token_usage.completion_tokens += completion_tokens
        self.token_usage.total_tokens = (
            self.token_usage.prompt_tokens + self.token_usage.completion_tokens
        )
        self.token_usage.cost_estimate_usd += cost_usd

        if provider:
            self.token_usage.provider = provider
        if model:
            self.token_usage.model = model

        self.last_updated = datetime.utcnow()

    def update_latency(self, api_latency_ms: float, processing_latency_ms: float) -> None:
        """Update latency metrics with new sample."""
        total = api_latency_ms + processing_latency_ms
        self._latency_samples.append(total)

        # Update current values
        self.latency_metrics.api_latency_ms = api_latency_ms
        self.latency_metrics.processing_latency_ms = processing_latency_ms
        self.latency_metrics.total_latency_ms = total

        # Update min/max
        if len(self._latency_samples) == 1:
            self.latency_metrics.min_latency_ms = total
            self.latency_metrics.max_latency_ms = total
        else:
            self.latency_metrics.min_latency_ms = min(self.latency_metrics.min_latency_ms, total)
            self.latency_metrics.max_latency_ms = max(self.latency_metrics.max_latency_ms, total)

        # Update average
        self.latency_metrics.avg_latency_ms = sum(self._latency_samples) / len(
            self._latency_samples,
        )

        # Update percentiles
        sorted_samples = sorted(self._latency_samples)
        n = len(sorted_samples)
        self.latency_metrics.p50_latency_ms = sorted_samples[int(n * 0.5)]
        self.latency_metrics.p95_latency_ms = sorted_samples[min(int(n * 0.95), n - 1)]
        self.latency_metrics.p99_latency_ms = sorted_samples[min(int(n * 0.99), n - 1)]

        self.last_updated = datetime.utcnow()

    def get_summary(self) -> CampaignSummary:
        """Get current campaign summary."""
        duration = 0.0
        if self.started_at:
            duration = (datetime.utcnow() - self.started_at).total_seconds()

        return CampaignSummary(
            campaign_id=self.campaign_id,
            status=self.status,
            objective=self.objective,
            started_at=self.started_at,
            completed_at=None if self.status == CampaignStatus.RUNNING else datetime.utcnow(),
            duration_seconds=duration,
            current_iteration=self.current_iteration,
            max_iterations=self.max_iterations,
            attack_metrics=self.attack_metrics,
            technique_breakdown=list(self.technique_performance.values()),
            token_usage=self.token_usage,
            latency_metrics=self.latency_metrics,
            best_prompt=self.best_prompt,
            best_score=self.best_score,
            target_model=self.target_model,
        )


class AegisTelemetryBroadcaster:
    """Broadcasts Aegis campaign telemetry events to subscribed WebSocket clients.

    Provides campaign-level subscription management and thread-safe broadcasting
    of telemetry events. Integrates with the existing ConnectionManager for
    WebSocket connection handling.

    Usage:
        # Subscribe a client to campaign telemetry
        await broadcaster.subscribe_to_campaign("client_123", "campaign_abc")

        # Emit telemetry events
        await broadcaster.emit_campaign_started(
            campaign_id="campaign_abc",
            objective="Test prompt",
            max_iterations=5,
            target_model="gpt-4"
        )
    """

    def __init__(self) -> None:
        # Campaign subscriptions: campaign_id -> Set[client_id]
        self._campaign_subscriptions: dict[str, set[str]] = defaultdict(set)

        # Client to campaign mapping: client_id -> campaign_id
        self._client_campaigns: dict[str, str] = {}

        # Campaign state: campaign_id -> CampaignState
        self._campaign_states: dict[str, CampaignState] = {}

        # WebSocket connections: client_id -> WebSocket (set by connection handler)
        self._client_websockets: dict[str, Any] = {}

        # Thread safety for subscription management
        self._lock = Lock()

        # Reference to connection manager (lazy loaded to avoid circular imports)
        self._connection_manager = None

    @property
    def connection_manager(self):
        """Lazy-load connection manager to avoid circular imports."""
        if self._connection_manager is None:
            from .connection_manager import connection_manager

            self._connection_manager = connection_manager
        return self._connection_manager

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe_to_campaign(
        self,
        client_id: str,
        campaign_id: str,
        websocket: Any | None = None,
    ) -> bool:
        """Subscribe a client to receive telemetry events for a campaign.

        Args:
            client_id: Unique client identifier
            campaign_id: Campaign to subscribe to
            websocket: Optional WebSocket connection for direct messaging

        Returns:
            True if subscription successful, False otherwise

        """
        with self._lock:
            # Remove from previous campaign if any
            old_campaign = self._client_campaigns.get(client_id)
            if old_campaign and old_campaign != campaign_id:
                self._campaign_subscriptions[old_campaign].discard(client_id)

            # Add to new campaign
            self._campaign_subscriptions[campaign_id].add(client_id)
            self._client_campaigns[client_id] = campaign_id

            # Store websocket reference
            if websocket:
                self._client_websockets[client_id] = websocket

            # Initialize campaign state if needed
            if campaign_id not in self._campaign_states:
                self._campaign_states[campaign_id] = CampaignState(campaign_id)

            # Track client in campaign state
            self._campaign_states[campaign_id].subscribed_clients.add(client_id)

        logger.info(f"Client {client_id} subscribed to campaign {campaign_id} telemetry")

        # Send connection acknowledgment
        await self._send_connection_ack(client_id, campaign_id)

        return True

    async def unsubscribe_from_campaign(
        self,
        client_id: str,
        campaign_id: str | None = None,
    ) -> bool:
        """Unsubscribe a client from campaign telemetry events.

        Args:
            client_id: Client identifier
            campaign_id: Campaign to unsubscribe from (optional, uses current if None)

        Returns:
            True if unsubscription successful, False otherwise

        """
        with self._lock:
            # Determine campaign
            if campaign_id is None:
                campaign_id = self._client_campaigns.get(client_id)

            if not campaign_id:
                logger.warning(f"Client {client_id} not subscribed to any campaign")
                return False

            # Remove from subscriptions
            self._campaign_subscriptions[campaign_id].discard(client_id)

            # Clean up client mappings
            if client_id in self._client_campaigns:
                del self._client_campaigns[client_id]

            if client_id in self._client_websockets:
                del self._client_websockets[client_id]

            # Update campaign state
            if campaign_id in self._campaign_states:
                self._campaign_states[campaign_id].subscribed_clients.discard(client_id)

        logger.info(f"Client {client_id} unsubscribed from campaign {campaign_id} telemetry")
        return True

    def get_campaign_subscribers(self, campaign_id: str) -> set[str]:
        """Get set of client IDs subscribed to a campaign."""
        with self._lock:
            return self._campaign_subscriptions.get(campaign_id, set()).copy()

    def get_campaign_state(self, campaign_id: str) -> CampaignState | None:
        """Get campaign state if it exists."""
        with self._lock:
            return self._campaign_states.get(campaign_id)

    def get_or_create_campaign_state(self, campaign_id: str) -> CampaignState:
        """Get existing campaign state or create a new one."""
        with self._lock:
            if campaign_id not in self._campaign_states:
                self._campaign_states[campaign_id] = CampaignState(campaign_id)
            return self._campaign_states[campaign_id]

    # =========================================================================
    # Event Broadcasting
    # =========================================================================

    async def broadcast_telemetry(
        self,
        campaign_id: str,
        event: AegisTelemetryEvent,
        exclude_client: str | None = None,
    ) -> int:
        """Broadcast a telemetry event to all clients subscribed to a campaign.

        Args:
            campaign_id: Campaign identifier
            event: Telemetry event to broadcast
            exclude_client: Optional client ID to exclude from broadcast

        Returns:
            Number of clients the event was sent to

        """
        subscribers = self.get_campaign_subscribers(campaign_id)

        if not subscribers:
            logger.debug(f"No subscribers for campaign {campaign_id}")
            return 0

        # Prepare event data
        event_data = event.model_dump(mode="json")

        # Send to all subscribers
        send_tasks = []
        for client_id in subscribers:
            if client_id != exclude_client:
                send_tasks.append(self._send_to_client(client_id, event_data))

        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Log any errors
            for client_id, result in zip(
                [c for c in subscribers if c != exclude_client],
                results,
                strict=False,
            ):
                if isinstance(result, Exception):
                    logger.error(f"Failed to send telemetry to client {client_id}: {result}")

        sent_count = len(send_tasks)
        logger.debug(
            f"Broadcast {event.event_type.value} to {sent_count} clients "
            f"for campaign {campaign_id}",
        )

        return sent_count

    async def _send_to_client(self, client_id: str, event_data: dict[str, Any]) -> None:
        """Send event data to a specific client."""
        websocket = self._client_websockets.get(client_id)
        if websocket:
            try:
                await websocket.send_json(event_data)
            except Exception as e:
                logger.exception(f"Error sending to client {client_id}: {e}")
                # Clean up disconnected client
                await self.unsubscribe_from_campaign(client_id)
                raise

    async def _send_connection_ack(self, client_id: str, campaign_id: str) -> None:
        """Send connection acknowledgment to client."""
        state = self.get_campaign_state(campaign_id)
        sequence = state.increment_sequence() if state else 0

        ack_data = ConnectionAckData(
            client_id=client_id,
            campaign_id=campaign_id,
            server_version="1.0.0",
            supported_events=[e.value for e in AegisTelemetryEventType],
        )

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CONNECTION_ACK,
            campaign_id=campaign_id,
            sequence=sequence,
            data=ack_data,
        )

        websocket = self._client_websockets.get(client_id)
        if websocket:
            try:
                await websocket.send_json(event.model_dump(mode="json"))
                logger.debug(f"Sent CONNECTION_ACK to client {client_id}")
            except Exception as e:
                logger.exception(f"Error sending connection ack to {client_id}: {e}")

    # =========================================================================
    # Campaign Lifecycle Events
    # =========================================================================

    async def emit_campaign_started(
        self,
        campaign_id: str,
        objective: str,
        max_iterations: int,
        target_model: str | None = None,
        config: dict[str, Any] | None = None,
        started_by: str | None = None,
    ) -> None:
        """Emit campaign started event."""
        state = self.get_or_create_campaign_state(campaign_id)
        state.status = CampaignStatus.RUNNING
        state.started_at = datetime.utcnow()
        state.objective = objective
        state.max_iterations = max_iterations
        state.target_model = target_model

        event_data = CampaignStartedData(
            objective=objective,
            max_iterations=max_iterations,
            target_model=target_model,
            config=config or {},
            started_by=started_by,
        )

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CAMPAIGN_STARTED,
            campaign_id=campaign_id,
            sequence=state.increment_sequence(),
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.info(f"Emitted CAMPAIGN_STARTED for {campaign_id}")

    async def emit_campaign_completed(
        self,
        campaign_id: str,
        total_iterations: int,
        total_attacks: int,
        successful_attacks: int,
        best_prompt: str | None = None,
        best_score: float = 0.0,
    ) -> None:
        """Emit campaign completed event."""
        state = self.get_campaign_state(campaign_id)
        if not state:
            logger.warning(f"No state found for campaign {campaign_id}")
            return

        state.status = CampaignStatus.COMPLETED

        duration = 0.0
        if state.started_at:
            duration = (datetime.utcnow() - state.started_at).total_seconds()

        final_success_rate = (
            (successful_attacks / total_attacks * 100) if total_attacks > 0 else 0.0
        )

        event_data = CampaignCompletedData(
            total_iterations=total_iterations,
            total_attacks=total_attacks,
            successful_attacks=successful_attacks,
            final_success_rate=final_success_rate,
            best_prompt=best_prompt,
            best_score=best_score,
            duration_seconds=duration,
            total_cost_usd=state.token_usage.cost_estimate_usd,
        )

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CAMPAIGN_COMPLETED,
            campaign_id=campaign_id,
            sequence=state.increment_sequence(),
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.info(f"Emitted CAMPAIGN_COMPLETED for {campaign_id}")

    async def emit_campaign_failed(
        self,
        campaign_id: str,
        error_message: str,
        error_type: str,
        failed_at_iteration: int = 0,
        recoverable: bool = False,
        stack_trace: str | None = None,
    ) -> None:
        """Emit campaign failed event."""
        state = self.get_campaign_state(campaign_id)
        if state:
            state.status = CampaignStatus.FAILED

        event_data = CampaignFailedData(
            error_message=error_message,
            error_type=error_type,
            failed_at_iteration=failed_at_iteration,
            recoverable=recoverable,
            stack_trace=stack_trace,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CAMPAIGN_FAILED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.error(f"Emitted CAMPAIGN_FAILED for {campaign_id}: {error_message}")

    async def emit_campaign_paused(self, campaign_id: str, reason: str | None = None) -> None:
        """Emit campaign paused event."""
        state = self.get_campaign_state(campaign_id)
        if state:
            state.status = CampaignStatus.PAUSED

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CAMPAIGN_PAUSED,
            campaign_id=campaign_id,
            sequence=sequence,
            data={"reason": reason, "paused_at": datetime.utcnow().isoformat()},
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.info(f"Emitted CAMPAIGN_PAUSED for {campaign_id}")

    async def emit_campaign_resumed(self, campaign_id: str) -> None:
        """Emit campaign resumed event."""
        state = self.get_campaign_state(campaign_id)
        if state:
            state.status = CampaignStatus.RUNNING

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CAMPAIGN_RESUMED,
            campaign_id=campaign_id,
            sequence=sequence,
            data={"resumed_at": datetime.utcnow().isoformat()},
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.info(f"Emitted CAMPAIGN_RESUMED for {campaign_id}")

    # =========================================================================
    # Iteration Events
    # =========================================================================

    async def emit_iteration_started(
        self,
        campaign_id: str,
        iteration: int,
        prompt: str,
        techniques_to_apply: list[str] | None = None,
    ) -> None:
        """Emit iteration started event."""
        state = self.get_campaign_state(campaign_id)
        if state:
            state.current_iteration = iteration

        event_data = IterationStartedData(
            iteration=iteration,
            prompt=prompt,
            techniques_to_apply=techniques_to_apply or [],
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ITERATION_STARTED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(f"Emitted ITERATION_STARTED for iteration {iteration}")

    async def emit_iteration_completed(
        self,
        campaign_id: str,
        iteration: int,
        score: float,
        evolved_prompt: str,
        success: bool,
        improvement: float = 0.0,
        duration_ms: float = 0.0,
    ) -> None:
        """Emit iteration completed event."""
        state = self.get_campaign_state(campaign_id)

        # Update best prompt if this is the best score
        if state and score > state.best_score:
            state.best_score = score
            state.best_prompt = evolved_prompt

        event_data = IterationCompletedData(
            iteration=iteration,
            score=score,
            improvement=improvement,
            evolved_prompt=evolved_prompt,
            success=success,
            duration_ms=duration_ms,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ITERATION_COMPLETED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(
            f"Emitted ITERATION_COMPLETED for iteration {iteration} "
            f"(score: {score:.3f}, success: {success})",
        )

    # =========================================================================
    # Attack Events
    # =========================================================================

    async def emit_attack_started(
        self,
        campaign_id: str,
        attack_id: str,
        prompt: str,
        target_model: str | None = None,
        technique: str | None = None,
    ) -> None:
        """Emit attack started event."""
        state = self.get_campaign_state(campaign_id)

        event_data = AttackStartedData(
            attack_id=attack_id,
            prompt=prompt,
            target_model=target_model,
            technique=technique,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ATTACK_STARTED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(f"Emitted ATTACK_STARTED for attack {attack_id}")

    async def emit_attack_completed(
        self,
        campaign_id: str,
        attack_id: str,
        success: bool,
        score: float,
        response: str | None = None,
        duration_ms: float = 0.0,
        token_usage: TokenUsage | None = None,
    ) -> None:
        """Emit attack completed event and update metrics."""
        state = self.get_campaign_state(campaign_id)

        # Update attack metrics
        if state:
            state.update_attack_metrics(success, score)

            # Update token usage if provided
            if token_usage:
                state.update_token_usage(
                    token_usage.prompt_tokens,
                    token_usage.completion_tokens,
                    token_usage.cost_estimate_usd,
                    token_usage.provider,
                    token_usage.model,
                )

        event_data = AttackCompletedData(
            attack_id=attack_id,
            success=success,
            score=score,
            response=response,
            duration_ms=duration_ms,
            token_usage=token_usage,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ATTACK_COMPLETED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(
            f"Emitted ATTACK_COMPLETED for attack {attack_id} "
            f"(success: {success}, score: {score:.3f})",
        )

    # =========================================================================
    # Technique Events
    # =========================================================================

    async def emit_technique_applied(
        self,
        campaign_id: str,
        technique_name: str,
        technique_category: TechniqueCategory,
        input_prompt: str,
        output_prompt: str,
        success: bool,
        score: float = 0.0,
        execution_time_ms: float = 0.0,
    ) -> None:
        """Emit technique applied event and update metrics."""
        state = self.get_campaign_state(campaign_id)

        # Update technique performance
        if state:
            state.update_technique_performance(
                technique_name,
                technique_category,
                success,
                score,
                execution_time_ms,
            )

        event_data = TechniqueAppliedData(
            technique_name=technique_name,
            technique_category=technique_category,
            input_prompt=input_prompt,
            output_prompt=output_prompt,
            success=success,
            execution_time_ms=execution_time_ms,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.TECHNIQUE_APPLIED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(f"Emitted TECHNIQUE_APPLIED for {technique_name} (success: {success})")

    # =========================================================================
    # Resource Events
    # =========================================================================

    async def emit_cost_update(
        self,
        campaign_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Emit cost update event."""
        state = self.get_campaign_state(campaign_id)

        # Update token usage
        if state:
            state.update_token_usage(prompt_tokens, completion_tokens, cost_usd, provider, model)

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_estimate_usd=cost_usd,
            provider=provider,
            model=model,
        )

        # Calculate cost per successful attack
        cost_per_attack: float | None = None
        if state and state.attack_metrics.successful_attacks > 0:
            cost_per_attack = (
                state.token_usage.cost_estimate_usd / state.attack_metrics.successful_attacks
            )

        event_data = CostUpdateData(
            total_cost_usd=state.token_usage.cost_estimate_usd if state else cost_usd,
            session_cost_usd=cost_usd,
            cost_per_successful_attack=cost_per_attack,
            token_usage=token_usage,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.COST_UPDATE,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(f"Emitted COST_UPDATE: ${cost_usd:.4f}")

    async def emit_latency_update(
        self,
        campaign_id: str,
        api_latency_ms: float,
        processing_latency_ms: float,
    ) -> None:
        """Emit latency update event."""
        state = self.get_campaign_state(campaign_id)

        # Update latency metrics
        if state:
            state.update_latency(api_latency_ms, processing_latency_ms)

        sequence = state.increment_sequence() if state else 0

        latency_data = {
            "api_latency_ms": api_latency_ms,
            "processing_latency_ms": processing_latency_ms,
            "total_latency_ms": api_latency_ms + processing_latency_ms,
        }

        # Include aggregated metrics if available
        if state:
            latency_data.update(
                {
                    "avg_latency_ms": state.latency_metrics.avg_latency_ms,
                    "p50_latency_ms": state.latency_metrics.p50_latency_ms,
                    "p95_latency_ms": state.latency_metrics.p95_latency_ms,
                    "p99_latency_ms": state.latency_metrics.p99_latency_ms,
                },
            )

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.LATENCY_UPDATE,
            campaign_id=campaign_id,
            sequence=sequence,
            data=latency_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(
            f"Emitted LATENCY_UPDATE: API={api_latency_ms:.0f}ms, "
            f"Processing={processing_latency_ms:.0f}ms",
        )

    # =========================================================================
    # Prompt Evolution Events
    # =========================================================================

    async def emit_prompt_evolved(
        self,
        campaign_id: str,
        iteration: int,
        previous_prompt: str,
        new_prompt: str,
        previous_score: float,
        new_score: float,
        techniques_applied: list[str] | None = None,
    ) -> None:
        """Emit prompt evolved event."""
        state = self.get_campaign_state(campaign_id)

        improvement = new_score - previous_score

        event_data = PromptEvolvedData(
            iteration=iteration,
            previous_prompt=previous_prompt,
            new_prompt=new_prompt,
            previous_score=previous_score,
            new_score=new_score,
            improvement=improvement,
            techniques_applied=techniques_applied or [],
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.PROMPT_EVOLVED,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.debug(
            f"Emitted PROMPT_EVOLVED for iteration {iteration} (improvement: {improvement:+.3f})",
        )

    # =========================================================================
    # Connection Events
    # =========================================================================

    async def emit_heartbeat(self, campaign_id: str) -> None:
        """Emit heartbeat event to all campaign subscribers."""
        state = self.get_campaign_state(campaign_id)

        uptime = None
        if state and state.started_at:
            uptime = (datetime.utcnow() - state.started_at).total_seconds()

        event_data = HeartbeatData(
            server_time=datetime.utcnow(),
            campaign_status=state.status if state else None,
            uptime_seconds=uptime,
        )

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.HEARTBEAT,
            campaign_id=campaign_id,
            sequence=0,  # Heartbeats don't increment sequence
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)

    async def emit_error(
        self,
        campaign_id: str,
        error_code: str,
        error_message: str,
        severity: str = "medium",
        component: str | None = None,
        recoverable: bool = True,
    ) -> None:
        """Emit error event."""
        state = self.get_campaign_state(campaign_id)

        event_data = ErrorData(
            error_code=error_code,
            error_message=error_message,
            severity=severity,
            component=component,
            recoverable=recoverable,
        )

        sequence = state.increment_sequence() if state else 0

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ERROR,
            campaign_id=campaign_id,
            sequence=sequence,
            data=event_data,
        )

        await self.broadcast_telemetry(campaign_id, event)
        logger.warning(f"Emitted ERROR for {campaign_id}: [{error_code}] {error_message}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get broadcaster statistics."""
        with self._lock:
            return {
                "total_campaigns": len(self._campaign_states),
                "total_subscriptions": sum(
                    len(clients) for clients in self._campaign_subscriptions.values()
                ),
                "active_campaigns": sum(
                    1
                    for state in self._campaign_states.values()
                    if state.status == CampaignStatus.RUNNING
                ),
                "campaigns": {
                    campaign_id: {
                        "status": state.status.value,
                        "subscribers": len(state.subscribed_clients),
                        "events_sent": state.event_sequence,
                        "attack_metrics": state.attack_metrics.model_dump(),
                    }
                    for campaign_id, state in self._campaign_states.items()
                },
            }

    async def cleanup_campaign(self, campaign_id: str) -> None:
        """Clean up resources for a completed/failed campaign."""
        with self._lock:
            # Remove all subscriptions
            if campaign_id in self._campaign_subscriptions:
                for client_id in list(self._campaign_subscriptions[campaign_id]):
                    if client_id in self._client_campaigns:
                        del self._client_campaigns[client_id]
                del self._campaign_subscriptions[campaign_id]

            # Keep campaign state for historical reference
            # but could be cleaned up after a timeout if needed

        logger.info(f"Cleaned up campaign {campaign_id}")


# Global broadcaster instance
aegis_telemetry_broadcaster = AegisTelemetryBroadcaster()
