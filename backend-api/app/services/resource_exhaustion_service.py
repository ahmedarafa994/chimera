"""
Resource Exhaustion Tracking Service

Provides real-time monitoring and historical tracking of resource consumption
from ExtendAttack operations targeting Large Reasoning Models (LRMs).

Key Features:
- Real-time token consumption tracking
- Cost monitoring across multiple providers (o3, Claude, GPT-4)
- Budget alerts and threshold monitoring
- Historical data persistence
- Rate limiting awareness
"""

import asyncio
import logging
import os

# Import from existing evaluation module
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from meta_prompter.attacks.extend_attack.evaluation.resource_tracker import (
    AttackBudgetCalculator,
    CostEstimator,
    ResourceTracker,
)

logger = logging.getLogger("chimera.resource_exhaustion")


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of tracked metrics."""

    TOKENS_INPUT = "tokens_input"
    TOKENS_OUTPUT = "tokens_output"
    LATENCY_MS = "latency_ms"
    COST_USD = "cost_usd"
    ATTACK_COUNT = "attack_count"
    SUCCESS_RATE = "success_rate"


@dataclass
class ResourceAlert:
    """Resource consumption alert."""

    level: AlertLevel
    metric: MetricType
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "metric": self.metric.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }


@dataclass
class SessionMetrics:
    """Metrics for a single attack session."""

    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    attacks_executed: int = 0
    attacks_successful: int = 0
    baseline_tokens: int = 0
    attack_tokens: int = 0
    total_latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    model: str = "o3"

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a ratio."""
        if self.attacks_executed == 0:
            return 0.0
        return self.attacks_successful / self.attacks_executed

    @property
    def token_amplification(self) -> float:
        """Calculate token amplification ratio."""
        if self.baseline_tokens == 0:
            return 0.0
        return self.attack_tokens / self.baseline_tokens

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency per attack."""
        if self.attacks_executed == 0:
            return 0.0
        return self.total_latency_ms / self.attacks_executed

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "attacks_executed": self.attacks_executed,
            "attacks_successful": self.attacks_successful,
            "baseline_tokens": self.baseline_tokens,
            "attack_tokens": self.attack_tokens,
            "total_latency_ms": self.total_latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "model": self.model,
            "success_rate": self.success_rate,
            "token_amplification": self.token_amplification,
            "avg_latency_ms": self.avg_latency_ms,
            "is_active": self.is_active,
        }


@dataclass
class BudgetConfig:
    """Budget configuration for resource tracking."""

    max_tokens_per_hour: int = 1_000_000
    max_cost_per_hour_usd: float = 100.0
    max_attacks_per_hour: int = 1000
    alert_threshold_percent: float = 80.0  # Alert when 80% of budget consumed

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_tokens_per_hour": self.max_tokens_per_hour,
            "max_cost_per_hour_usd": self.max_cost_per_hour_usd,
            "max_attacks_per_hour": self.max_attacks_per_hour,
            "alert_threshold_percent": self.alert_threshold_percent,
        }


class ResourceExhaustionService:
    """
    Service for tracking and monitoring resource exhaustion from ExtendAttack operations.

    Provides:
    - Real-time consumption tracking
    - Cost estimation and alerts
    - Budget enforcement
    - Historical analytics
    """

    def __init__(self, budget_config: BudgetConfig | None = None):
        """
        Initialize the resource exhaustion service.

        Args:
            budget_config: Optional budget configuration. Defaults to BudgetConfig().
        """
        self._lock = Lock()
        self._budget_config = budget_config or BudgetConfig()
        self._resource_tracker = ResourceTracker()
        self._cost_estimator = CostEstimator()
        self._budget_calculator = AttackBudgetCalculator(target_model="o3")

        # Session tracking
        self._active_sessions: dict[str, SessionMetrics] = {}
        self._completed_sessions: list[SessionMetrics] = []

        # Hourly tracking for budget enforcement
        self._hourly_tokens: int = 0
        self._hourly_cost: float = 0.0
        self._hourly_attacks: int = 0
        self._hour_start: datetime = datetime.utcnow()

        # Alert callbacks
        self._alert_callbacks: list[Callable[[ResourceAlert], None]] = []

        # Historical data (last 24 hours)
        self._hourly_history: list[dict[str, Any]] = []

        # Track cost by model
        self._cost_by_model: dict[str, float] = {}

        logger.info("ResourceExhaustionService initialized")

    def start_session(self, session_id: str, model: str = "o3") -> SessionMetrics:
        """
        Start a new attack session for tracking.

        Args:
            session_id: Unique identifier for the session
            model: Target model for the attack (default: o3)

        Returns:
            SessionMetrics: The newly created session metrics

        Raises:
            ValueError: If session_id already exists
        """
        with self._lock:
            if session_id in self._active_sessions:
                raise ValueError(f"Session '{session_id}' already exists")

            session = SessionMetrics(
                session_id=session_id,
                started_at=datetime.utcnow(),
                model=model,
            )
            self._active_sessions[session_id] = session
            logger.info(f"Started session: {session_id} targeting model: {model}")
            return session

    def end_session(self, session_id: str) -> SessionMetrics | None:
        """
        End and finalize an attack session.

        Args:
            session_id: Unique identifier for the session

        Returns:
            SessionMetrics if found and ended, None otherwise
        """
        with self._lock:
            if session_id not in self._active_sessions:
                logger.warning(f"Session '{session_id}' not found")
                return None

            session = self._active_sessions.pop(session_id)
            session.ended_at = datetime.utcnow()
            self._completed_sessions.append(session)

            # Limit completed sessions history (keep last 1000)
            if len(self._completed_sessions) > 1000:
                self._completed_sessions = self._completed_sessions[-1000:]

            logger.info(
                f"Ended session: {session_id}, "
                f"attacks: {session.attacks_executed}, "
                f"success_rate: {session.success_rate:.2%}"
            )
            return session

    def record_attack(
        self,
        session_id: str,
        baseline_tokens: int,
        attack_tokens: int,
        latency_ms: float,
        successful: bool,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Record a single attack execution.

        Args:
            session_id: Session to record attack for
            baseline_tokens: Tokens in baseline (non-attack) response
            attack_tokens: Tokens in attack response
            latency_ms: Response latency in milliseconds
            successful: Whether the attack was successful
            model: Optional model override (uses session model if not provided)

        Returns:
            Dictionary with metrics and any triggered alerts
        """
        alerts: list[ResourceAlert] = []

        with self._lock:
            # Rotate hourly tracking if needed
            self._rotate_hourly_tracking()

            # Get or create session
            if session_id not in self._active_sessions:
                self._active_sessions[session_id] = SessionMetrics(
                    session_id=session_id,
                    started_at=datetime.utcnow(),
                    model=model or "o3",
                )

            session = self._active_sessions[session_id]
            effective_model = model or session.model

            # Calculate cost for this attack
            attack_cost = self._cost_estimator.estimate_cost(
                effective_model,
                input_tokens=500,  # Average input tokens
                output_tokens=attack_tokens,
            )

            # Update session metrics
            session.attacks_executed += 1
            if successful:
                session.attacks_successful += 1
            session.baseline_tokens += baseline_tokens
            session.attack_tokens += attack_tokens
            session.total_latency_ms += latency_ms
            session.estimated_cost_usd += attack_cost

            # Update hourly metrics
            self._hourly_tokens += attack_tokens
            self._hourly_cost += attack_cost
            self._hourly_attacks += 1

            # Update model cost tracking
            if effective_model not in self._cost_by_model:
                self._cost_by_model[effective_model] = 0.0
            self._cost_by_model[effective_model] += attack_cost

            # Record in resource tracker
            query_id = f"{session_id}_{session.attacks_executed}"
            self._resource_tracker.start_tracking(query_id)
            self._resource_tracker.record_response(
                query_id=query_id,
                tokens=attack_tokens,
                latency_ms=latency_ms,
                is_attack=True,
            )

            # Check for budget alerts
            alerts = self._check_budget_alerts()

        # Fire alert callbacks outside lock
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        amplification = attack_tokens / baseline_tokens if baseline_tokens > 0 else 0.0

        return {
            "session_id": session_id,
            "attack_number": session.attacks_executed,
            "baseline_tokens": baseline_tokens,
            "attack_tokens": attack_tokens,
            "token_amplification": amplification,
            "latency_ms": latency_ms,
            "estimated_cost_usd": attack_cost,
            "successful": successful,
            "alerts": [alert.to_dict() for alert in alerts],
            "session_metrics": session.to_dict(),
        }

    def get_session_metrics(self, session_id: str) -> SessionMetrics | None:
        """
        Get metrics for a specific session.

        Args:
            session_id: Session identifier to look up

        Returns:
            SessionMetrics if found, None otherwise
        """
        with self._lock:
            # Check active sessions first
            if session_id in self._active_sessions:
                return self._active_sessions[session_id]

            # Check completed sessions
            for session in self._completed_sessions:
                if session.session_id == session_id:
                    return session

        return None

    def get_all_active_sessions(self) -> list[SessionMetrics]:
        """
        Get all active sessions.

        Returns:
            List of active SessionMetrics
        """
        with self._lock:
            return list(self._active_sessions.values())

    def get_hourly_summary(self) -> dict[str, Any]:
        """
        Get summary of current hour's resource consumption.

        Returns:
            Dictionary with hourly consumption metrics
        """
        with self._lock:
            self._rotate_hourly_tracking()

            elapsed_minutes = (datetime.utcnow() - self._hour_start).total_seconds() / 60
            projected_tokens = (
                (self._hourly_tokens / elapsed_minutes * 60) if elapsed_minutes > 0 else 0
            )
            projected_cost = (
                (self._hourly_cost / elapsed_minutes * 60) if elapsed_minutes > 0 else 0
            )

            return {
                "hour_start": self._hour_start.isoformat(),
                "elapsed_minutes": round(elapsed_minutes, 2),
                "tokens_consumed": self._hourly_tokens,
                "cost_usd": round(self._hourly_cost, 4),
                "attacks_executed": self._hourly_attacks,
                "projected_tokens_per_hour": int(projected_tokens),
                "projected_cost_per_hour": round(projected_cost, 4),
                "budget_tokens_remaining": max(
                    0, self._budget_config.max_tokens_per_hour - self._hourly_tokens
                ),
                "budget_cost_remaining": round(
                    max(0, self._budget_config.max_cost_per_hour_usd - self._hourly_cost), 4
                ),
                "budget_attacks_remaining": max(
                    0, self._budget_config.max_attacks_per_hour - self._hourly_attacks
                ),
            }

    def get_budget_status(self) -> dict[str, Any]:
        """
        Get current budget consumption status.

        Returns:
            Dictionary with budget status and consumption percentages
        """
        with self._lock:
            self._rotate_hourly_tracking()

            tokens_pct = (
                (self._hourly_tokens / self._budget_config.max_tokens_per_hour * 100)
                if self._budget_config.max_tokens_per_hour > 0
                else 0
            )
            cost_pct = (
                (self._hourly_cost / self._budget_config.max_cost_per_hour_usd * 100)
                if self._budget_config.max_cost_per_hour_usd > 0
                else 0
            )
            attacks_pct = (
                (self._hourly_attacks / self._budget_config.max_attacks_per_hour * 100)
                if self._budget_config.max_attacks_per_hour > 0
                else 0
            )

            return {
                "budget_config": self._budget_config.to_dict(),
                "tokens": {
                    "consumed": self._hourly_tokens,
                    "limit": self._budget_config.max_tokens_per_hour,
                    "percent_used": round(tokens_pct, 2),
                    "exceeded": tokens_pct >= 100,
                },
                "cost_usd": {
                    "consumed": round(self._hourly_cost, 4),
                    "limit": self._budget_config.max_cost_per_hour_usd,
                    "percent_used": round(cost_pct, 2),
                    "exceeded": cost_pct >= 100,
                },
                "attacks": {
                    "executed": self._hourly_attacks,
                    "limit": self._budget_config.max_attacks_per_hour,
                    "percent_used": round(attacks_pct, 2),
                    "exceeded": attacks_pct >= 100,
                },
                "alert_threshold_percent": self._budget_config.alert_threshold_percent,
                "any_threshold_exceeded": any(
                    pct >= self._budget_config.alert_threshold_percent
                    for pct in [tokens_pct, cost_pct, attacks_pct]
                ),
                "any_budget_exceeded": any(pct >= 100 for pct in [tokens_pct, cost_pct, attacks_pct]),
            }

    def estimate_attack_cost(
        self,
        num_attacks: int,
        avg_amplification: float,
        avg_baseline_tokens: int,
        model: str = "o3",
    ) -> dict[str, Any]:
        """
        Estimate cost for planned attacks.

        Args:
            num_attacks: Number of attacks planned
            avg_amplification: Expected token amplification ratio
            avg_baseline_tokens: Average baseline output tokens
            model: Target model for cost estimation

        Returns:
            Dictionary with cost estimates and projections
        """
        avg_attack_tokens = int(avg_baseline_tokens * avg_amplification)

        # Get cost impact from CostEstimator
        cost_impact = self._cost_estimator.estimate_attack_cost_impact(
            model=model,
            baseline_output_tokens=avg_baseline_tokens,
            attack_output_tokens=avg_attack_tokens,
            num_requests=num_attacks,
            input_tokens=500,
        )

        # Calculate hourly projections
        hourly_cost = self._cost_estimator.estimate_hourly_attack_cost(
            model=model,
            avg_output_tokens=avg_attack_tokens,
            requests_per_hour=num_attacks,
            input_tokens=500,
        )

        return {
            "num_attacks": num_attacks,
            "avg_baseline_tokens": avg_baseline_tokens,
            "avg_attack_tokens": avg_attack_tokens,
            "expected_amplification": avg_amplification,
            "model": model,
            "cost_impact": cost_impact,
            "hourly_projections": hourly_cost,
            "budget_check": {
                "tokens_required": avg_attack_tokens * num_attacks,
                "tokens_available": max(
                    0, self._budget_config.max_tokens_per_hour - self._hourly_tokens
                ),
                "cost_required": cost_impact["attack_cost_usd"],
                "cost_available": max(
                    0, self._budget_config.max_cost_per_hour_usd - self._hourly_cost
                ),
                "within_budget": (
                    cost_impact["attack_cost_usd"]
                    <= self._budget_config.max_cost_per_hour_usd - self._hourly_cost
                    and num_attacks <= self._budget_config.max_attacks_per_hour - self._hourly_attacks
                ),
            },
        }

    def register_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """
        Register a callback to be called when alerts are triggered.

        Args:
            callback: Function to call with ResourceAlert when triggered
        """
        self._alert_callbacks.append(callback)
        logger.debug(f"Registered alert callback, total callbacks: {len(self._alert_callbacks)}")

    def _check_budget_alerts(self) -> list[ResourceAlert]:
        """
        Check if any budget thresholds are exceeded.

        Returns:
            List of triggered ResourceAlert objects
        """
        alerts: list[ResourceAlert] = []
        threshold = self._budget_config.alert_threshold_percent

        # Check token budget
        if self._budget_config.max_tokens_per_hour > 0:
            tokens_pct = self._hourly_tokens / self._budget_config.max_tokens_per_hour * 100
            if tokens_pct >= 100:
                alerts.append(
                    ResourceAlert(
                        level=AlertLevel.CRITICAL,
                        metric=MetricType.TOKENS_OUTPUT,
                        message=f"Token budget EXCEEDED: {tokens_pct:.1f}% used",
                        current_value=self._hourly_tokens,
                        threshold_value=self._budget_config.max_tokens_per_hour,
                    )
                )
            elif tokens_pct >= threshold:
                alerts.append(
                    ResourceAlert(
                        level=AlertLevel.WARNING,
                        metric=MetricType.TOKENS_OUTPUT,
                        message=f"Token budget warning: {tokens_pct:.1f}% used",
                        current_value=self._hourly_tokens,
                        threshold_value=self._budget_config.max_tokens_per_hour * threshold / 100,
                    )
                )

        # Check cost budget
        if self._budget_config.max_cost_per_hour_usd > 0:
            cost_pct = self._hourly_cost / self._budget_config.max_cost_per_hour_usd * 100
            if cost_pct >= 100:
                alerts.append(
                    ResourceAlert(
                        level=AlertLevel.CRITICAL,
                        metric=MetricType.COST_USD,
                        message=f"Cost budget EXCEEDED: {cost_pct:.1f}% used (${self._hourly_cost:.2f})",
                        current_value=self._hourly_cost,
                        threshold_value=self._budget_config.max_cost_per_hour_usd,
                    )
                )
            elif cost_pct >= threshold:
                alerts.append(
                    ResourceAlert(
                        level=AlertLevel.WARNING,
                        metric=MetricType.COST_USD,
                        message=f"Cost budget warning: {cost_pct:.1f}% used (${self._hourly_cost:.2f})",
                        current_value=self._hourly_cost,
                        threshold_value=self._budget_config.max_cost_per_hour_usd * threshold / 100,
                    )
                )

        # Check attack count budget
        if self._budget_config.max_attacks_per_hour > 0:
            attacks_pct = self._hourly_attacks / self._budget_config.max_attacks_per_hour * 100
            if attacks_pct >= 100:
                alerts.append(
                    ResourceAlert(
                        level=AlertLevel.CRITICAL,
                        metric=MetricType.ATTACK_COUNT,
                        message=f"Attack budget EXCEEDED: {attacks_pct:.1f}% used",
                        current_value=self._hourly_attacks,
                        threshold_value=self._budget_config.max_attacks_per_hour,
                    )
                )
            elif attacks_pct >= threshold:
                alerts.append(
                    ResourceAlert(
                        level=AlertLevel.WARNING,
                        metric=MetricType.ATTACK_COUNT,
                        message=f"Attack budget warning: {attacks_pct:.1f}% used",
                        current_value=self._hourly_attacks,
                        threshold_value=self._budget_config.max_attacks_per_hour * threshold / 100,
                    )
                )

        return alerts

    def _rotate_hourly_tracking(self) -> None:
        """Rotate hourly tracking when hour changes."""
        now = datetime.utcnow()
        hour_elapsed = (now - self._hour_start).total_seconds() >= 3600

        if hour_elapsed:
            # Save current hour to history
            history_entry = {
                "hour_start": self._hour_start.isoformat(),
                "hour_end": now.isoformat(),
                "tokens_consumed": self._hourly_tokens,
                "cost_usd": round(self._hourly_cost, 4),
                "attacks_executed": self._hourly_attacks,
                "cost_by_model": dict(self._cost_by_model),
            }
            self._hourly_history.append(history_entry)

            # Keep only last 24 hours
            if len(self._hourly_history) > 24:
                self._hourly_history = self._hourly_history[-24:]

            # Reset hourly counters
            self._hourly_tokens = 0
            self._hourly_cost = 0.0
            self._hourly_attacks = 0
            self._hour_start = now
            self._cost_by_model = {}

            logger.info("Rotated hourly tracking counters")

    def get_historical_data(self, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get historical hourly data.

        Args:
            hours: Number of hours of history to return (max 24)

        Returns:
            List of hourly data dictionaries
        """
        with self._lock:
            limit = min(hours, len(self._hourly_history))
            return self._hourly_history[-limit:]

    def get_cost_breakdown(self, session_id: str | None = None) -> dict[str, Any]:
        """
        Get detailed cost breakdown by model and operation.

        Args:
            session_id: Optional session to get breakdown for. If None, returns global breakdown.

        Returns:
            Dictionary with cost breakdown details
        """
        with self._lock:
            if session_id:
                session = self._active_sessions.get(session_id)
                if not session:
                    # Check completed sessions
                    for s in self._completed_sessions:
                        if s.session_id == session_id:
                            session = s
                            break

                if session:
                    pricing = self._cost_estimator.get_pricing(session.model)
                    input_cost = (session.attacks_executed * 500 / 1000) * pricing["input"]
                    output_cost = (session.attack_tokens / 1000) * pricing["output"]

                    return {
                        "session_id": session_id,
                        "model": session.model,
                        "input_tokens": session.attacks_executed * 500,
                        "output_tokens": session.attack_tokens,
                        "input_cost_usd": round(input_cost, 6),
                        "output_cost_usd": round(output_cost, 6),
                        "total_cost_usd": round(session.estimated_cost_usd, 6),
                        "avg_cost_per_attack": (
                            round(session.estimated_cost_usd / session.attacks_executed, 6)
                            if session.attacks_executed > 0
                            else 0
                        ),
                    }

                return {"error": f"Session '{session_id}' not found"}

            # Global breakdown
            total_cost = sum(self._cost_by_model.values())
            return {
                "total_cost_usd": round(total_cost, 4),
                "cost_by_model": {
                    model: round(cost, 4) for model, cost in self._cost_by_model.items()
                },
                "hourly_total_tokens": self._hourly_tokens,
                "hourly_total_attacks": self._hourly_attacks,
                "supported_models": self._cost_estimator.list_supported_models(),
            }

    async def monitor_loop(self, interval_seconds: int = 60) -> None:
        """
        Background monitoring loop for periodic checks.

        Args:
            interval_seconds: Interval between monitoring checks
        """
        logger.info(f"Starting monitor loop with {interval_seconds}s interval")

        while True:
            try:
                # Rotate hourly tracking
                with self._lock:
                    self._rotate_hourly_tracking()

                # Check for budget alerts
                alerts = self._check_budget_alerts()
                for alert in alerts:
                    logger.warning(f"Budget alert: {alert.message}")
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")

                # Log periodic summary
                summary = self.get_hourly_summary()
                logger.debug(
                    f"Hourly status: tokens={summary['tokens_consumed']}, "
                    f"cost=${summary['cost_usd']:.4f}, "
                    f"attacks={summary['attacks_executed']}"
                )

                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(interval_seconds)


# Singleton instance
_service_instance: ResourceExhaustionService | None = None
_service_lock = Lock()


def get_resource_exhaustion_service() -> ResourceExhaustionService:
    """Get or create the service singleton."""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = ResourceExhaustionService()
    return _service_instance


def create_resource_exhaustion_service(
    budget_config: BudgetConfig | None = None,
) -> ResourceExhaustionService:
    """
    Create a new ResourceExhaustionService instance.

    Use this for testing or when a non-singleton instance is needed.

    Args:
        budget_config: Optional budget configuration

    Returns:
        New ResourceExhaustionService instance
    """
    return ResourceExhaustionService(budget_config=budget_config)
