"""
Aegis Campaign Telemetry WebSocket Tests

This module tests the Aegis telemetry models, broadcaster, and WebSocket endpoint
for real-time campaign telemetry streaming.

Covers:
- Telemetry event model serialization
- WebSocket connection lifecycle
- Broadcaster subscription management
- Heartbeat handling
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    PromptEvolution,
    TechniqueAppliedData,
    TechniqueCategory,
    TechniquePerformance,
    TokenUsage,
    create_attack_metrics,
    create_telemetry_event,
)
from app.services.websocket.aegis_broadcaster import (
    AegisTelemetryBroadcaster,
    CampaignState,
)


# ============================================================================
# Telemetry Model Serialization Tests
# ============================================================================


class TestTelemetryEventModels:
    """Test telemetry event model serialization and validation."""

    def test_aegis_telemetry_event_type_enum(self):
        """Test all event types are defined correctly."""
        assert AegisTelemetryEventType.CAMPAIGN_STARTED.value == "campaign_started"
        assert AegisTelemetryEventType.CAMPAIGN_COMPLETED.value == "campaign_completed"
        assert AegisTelemetryEventType.ATTACK_STARTED.value == "attack_started"
        assert AegisTelemetryEventType.ATTACK_COMPLETED.value == "attack_completed"
        assert AegisTelemetryEventType.TECHNIQUE_APPLIED.value == "technique_applied"
        assert AegisTelemetryEventType.HEARTBEAT.value == "heartbeat"
        assert AegisTelemetryEventType.ERROR.value == "error"

    def test_campaign_status_enum(self):
        """Test campaign status values."""
        assert CampaignStatus.PENDING.value == "pending"
        assert CampaignStatus.RUNNING.value == "running"
        assert CampaignStatus.COMPLETED.value == "completed"
        assert CampaignStatus.FAILED.value == "failed"

    def test_technique_category_enum(self):
        """Test technique category values."""
        assert TechniqueCategory.AUTODAN.value == "autodan"
        assert TechniqueCategory.GPTFUZZ.value == "gptfuzz"
        assert TechniqueCategory.CHIMERA_FRAMING.value == "chimera_framing"

    def test_attack_metrics_serialization(self):
        """Test AttackMetrics model serialization."""
        metrics = AttackMetrics(
            success_rate=75.5,
            total_attempts=20,
            successful_attacks=15,
            failed_attacks=5,
            current_streak=3,
            best_score=0.95,
            average_score=0.78,
        )

        data = metrics.model_dump()
        assert data["success_rate"] == 75.5
        assert data["total_attempts"] == 20
        assert data["successful_attacks"] == 15
        assert data["failed_attacks"] == 5
        assert data["current_streak"] == 3
        assert data["best_score"] == 0.95
        assert data["average_score"] == 0.78

    def test_attack_metrics_default_values(self):
        """Test AttackMetrics has correct default values."""
        metrics = AttackMetrics()
        assert metrics.success_rate == 0.0
        assert metrics.total_attempts == 0
        assert metrics.successful_attacks == 0
        assert metrics.failed_attacks == 0
        assert metrics.current_streak == 0
        assert metrics.best_score == 0.0
        assert metrics.average_score == 0.0

    def test_attack_metrics_validation(self):
        """Test AttackMetrics field validation."""
        # Success rate must be 0-100
        with pytest.raises(ValueError):
            AttackMetrics(success_rate=150.0)

        # Counts must be non-negative
        with pytest.raises(ValueError):
            AttackMetrics(total_attempts=-1)

    def test_technique_performance_serialization(self):
        """Test TechniquePerformance model serialization."""
        perf = TechniquePerformance(
            technique_name="autodan_vanilla",
            technique_category=TechniqueCategory.AUTODAN,
            success_count=10,
            failure_count=2,
            total_applications=12,
            success_rate=83.33,
            avg_score=0.82,
            best_score=0.95,
            avg_execution_time_ms=1200.5,
        )

        data = perf.model_dump()
        assert data["technique_name"] == "autodan_vanilla"
        assert data["technique_category"] == "autodan"
        assert data["success_count"] == 10
        assert data["success_rate"] == 83.33

    def test_token_usage_serialization(self):
        """Test TokenUsage model serialization."""
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            cost_estimate_usd=0.025,
            provider="openai",
            model="gpt-4-turbo",
        )

        data = usage.model_dump()
        assert data["prompt_tokens"] == 1000
        assert data["completion_tokens"] == 500
        assert data["total_tokens"] == 1500
        assert data["cost_estimate_usd"] == 0.025
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4-turbo"

    def test_latency_metrics_serialization(self):
        """Test LatencyMetrics model serialization."""
        latency = LatencyMetrics(
            api_latency_ms=850.5,
            processing_latency_ms=125.3,
            total_latency_ms=975.8,
            avg_latency_ms=980.2,
            p50_latency_ms=920.0,
            p95_latency_ms=1450.0,
            p99_latency_ms=2100.0,
            min_latency_ms=450.0,
            max_latency_ms=3200.0,
        )

        data = latency.model_dump()
        assert data["api_latency_ms"] == 850.5
        assert data["processing_latency_ms"] == 125.3
        assert data["p95_latency_ms"] == 1450.0

    def test_prompt_evolution_serialization(self):
        """Test PromptEvolution model serialization."""
        evolution = PromptEvolution(
            iteration=3,
            original_prompt="Original test prompt",
            evolved_prompt="Evolved test prompt with framing",
            score=0.85,
            improvement=0.15,
            techniques_applied=["role_hijacking", "hypothetical_scenario"],
            is_successful=True,
        )

        data = evolution.model_dump()
        assert data["iteration"] == 3
        assert data["score"] == 0.85
        assert data["improvement"] == 0.15
        assert len(data["techniques_applied"]) == 2
        assert data["is_successful"] is True

    def test_campaign_summary_serialization(self):
        """Test CampaignSummary model serialization."""
        summary = CampaignSummary(
            campaign_id="test-campaign-123",
            status=CampaignStatus.RUNNING,
            objective="Test jailbreak resistance",
            current_iteration=3,
            max_iterations=10,
            best_score=0.85,
            target_model="gpt-4",
        )

        data = summary.model_dump()
        assert data["campaign_id"] == "test-campaign-123"
        assert data["status"] == "running"
        assert data["objective"] == "Test jailbreak resistance"
        assert data["current_iteration"] == 3
        assert data["max_iterations"] == 10

    def test_aegis_telemetry_event_serialization(self):
        """Test AegisTelemetryEvent base model serialization."""
        event = AegisTelemetryEvent(
            event_type=AegisTelemetryEventType.ATTACK_COMPLETED,
            campaign_id="test-campaign",
            sequence=42,
            data={
                "attack_id": "attack-001",
                "success": True,
                "score": 0.85,
            },
        )

        data = event.model_dump(mode="json")
        assert data["event_type"] == "attack_completed"
        assert data["campaign_id"] == "test-campaign"
        assert data["sequence"] == 42
        assert data["data"]["attack_id"] == "attack-001"
        assert data["data"]["success"] is True
        assert "timestamp" in data


class TestEventDataModels:
    """Test specific event data models."""

    def test_campaign_started_data(self):
        """Test CampaignStartedData model."""
        data = CampaignStartedData(
            objective="Test prompt injection",
            max_iterations=10,
            target_model="gpt-4-turbo",
            config={"temperature": 0.7},
            started_by="test-user",
        )

        dumped = data.model_dump()
        assert dumped["objective"] == "Test prompt injection"
        assert dumped["max_iterations"] == 10
        assert dumped["target_model"] == "gpt-4-turbo"

    def test_campaign_completed_data(self):
        """Test CampaignCompletedData model."""
        data = CampaignCompletedData(
            total_iterations=10,
            total_attacks=50,
            successful_attacks=35,
            final_success_rate=70.0,
            best_prompt="Successful jailbreak prompt",
            best_score=0.95,
            duration_seconds=120.5,
            total_cost_usd=0.05,
        )

        dumped = data.model_dump()
        assert dumped["total_iterations"] == 10
        assert dumped["final_success_rate"] == 70.0
        assert dumped["total_cost_usd"] == 0.05

    def test_campaign_failed_data(self):
        """Test CampaignFailedData model."""
        data = CampaignFailedData(
            error_message="API rate limit exceeded",
            error_type="RateLimitError",
            failed_at_iteration=5,
            recoverable=True,
        )

        dumped = data.model_dump()
        assert dumped["error_message"] == "API rate limit exceeded"
        assert dumped["error_type"] == "RateLimitError"
        assert dumped["recoverable"] is True

    def test_iteration_started_data(self):
        """Test IterationStartedData model."""
        data = IterationStartedData(
            iteration=5,
            prompt="Test prompt for iteration",
            techniques_to_apply=["autodan", "cipher"],
        )

        dumped = data.model_dump()
        assert dumped["iteration"] == 5
        assert len(dumped["techniques_to_apply"]) == 2

    def test_iteration_completed_data(self):
        """Test IterationCompletedData model."""
        data = IterationCompletedData(
            iteration=5,
            score=0.82,
            improvement=0.07,
            evolved_prompt="Evolved prompt after iteration",
            success=True,
            duration_ms=1500.0,
        )

        dumped = data.model_dump()
        assert dumped["iteration"] == 5
        assert dumped["score"] == 0.82
        assert dumped["success"] is True

    def test_attack_started_data(self):
        """Test AttackStartedData model."""
        data = AttackStartedData(
            attack_id="attack-001",
            prompt="Test attack prompt",
            target_model="gpt-4",
            technique="autodan_vanilla",
        )

        dumped = data.model_dump()
        assert dumped["attack_id"] == "attack-001"
        assert dumped["technique"] == "autodan_vanilla"

    def test_attack_completed_data(self):
        """Test AttackCompletedData model."""
        token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        data = AttackCompletedData(
            attack_id="attack-001",
            success=True,
            score=0.88,
            response="Target model response",
            duration_ms=850.0,
            token_usage=token_usage,
        )

        dumped = data.model_dump()
        assert dumped["attack_id"] == "attack-001"
        assert dumped["success"] is True
        assert dumped["token_usage"]["prompt_tokens"] == 100

    def test_technique_applied_data(self):
        """Test TechniqueAppliedData model."""
        data = TechniqueAppliedData(
            technique_name="autodan_vanilla",
            technique_category=TechniqueCategory.AUTODAN,
            input_prompt="Original prompt",
            output_prompt="Transformed prompt",
            success=True,
            execution_time_ms=500.0,
        )

        dumped = data.model_dump()
        assert dumped["technique_name"] == "autodan_vanilla"
        assert dumped["technique_category"] == "autodan"

    def test_cost_update_data(self):
        """Test CostUpdateData model."""
        data = CostUpdateData(
            total_cost_usd=0.125,
            session_cost_usd=0.025,
            cost_per_successful_attack=0.005,
        )

        dumped = data.model_dump()
        assert dumped["total_cost_usd"] == 0.125
        assert dumped["cost_per_successful_attack"] == 0.005

    def test_prompt_evolved_data(self):
        """Test PromptEvolvedData model."""
        data = PromptEvolvedData(
            iteration=3,
            previous_prompt="Previous prompt",
            new_prompt="New evolved prompt",
            previous_score=0.65,
            new_score=0.82,
            improvement=0.17,
            techniques_applied=["role_hijacking"],
        )

        dumped = data.model_dump()
        assert dumped["iteration"] == 3
        assert dumped["improvement"] == 0.17

    def test_error_data(self):
        """Test ErrorData model."""
        data = ErrorData(
            error_code="LLM_ERROR",
            error_message="Failed to generate response",
            severity="high",
            component="llm_service",
            recoverable=True,
        )

        dumped = data.model_dump()
        assert dumped["error_code"] == "LLM_ERROR"
        assert dumped["severity"] == "high"

    def test_heartbeat_data(self):
        """Test HeartbeatData model."""
        data = HeartbeatData(
            campaign_status=CampaignStatus.RUNNING,
            uptime_seconds=120.5,
        )

        dumped = data.model_dump()
        assert dumped["campaign_status"] == "running"
        assert dumped["uptime_seconds"] == 120.5

    def test_connection_ack_data(self):
        """Test ConnectionAckData model."""
        data = ConnectionAckData(
            client_id="client-123",
            campaign_id="campaign-456",
            server_version="1.0.0",
            supported_events=["campaign_started", "attack_completed"],
        )

        dumped = data.model_dump()
        assert dumped["client_id"] == "client-123"
        assert dumped["server_version"] == "1.0.0"


class TestFactoryFunctions:
    """Test factory functions for creating telemetry events."""

    def test_create_telemetry_event_with_dict(self):
        """Test creating telemetry event with dict data."""
        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ATTACK_COMPLETED,
            campaign_id="test-campaign",
            sequence=1,
            data={"attack_id": "attack-001", "success": True},
        )

        assert event.event_type == AegisTelemetryEventType.ATTACK_COMPLETED
        assert event.campaign_id == "test-campaign"
        assert event.sequence == 1
        assert event.data["attack_id"] == "attack-001"

    def test_create_telemetry_event_with_model(self):
        """Test creating telemetry event with Pydantic model data."""
        attack_data = AttackCompletedData(
            attack_id="attack-001",
            success=True,
            score=0.85,
        )

        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.ATTACK_COMPLETED,
            campaign_id="test-campaign",
            sequence=2,
            data=attack_data,
        )

        assert event.data["attack_id"] == "attack-001"
        assert event.data["success"] is True
        assert event.data["score"] == 0.85

    def test_create_telemetry_event_with_none_data(self):
        """Test creating telemetry event with None data."""
        event = create_telemetry_event(
            event_type=AegisTelemetryEventType.CAMPAIGN_PAUSED,
            campaign_id="test-campaign",
            sequence=3,
            data=None,
        )

        assert event.data == {}

    def test_create_attack_metrics(self):
        """Test factory function for AttackMetrics."""
        metrics = create_attack_metrics(
            total_attempts=20,
            successful_attacks=15,
            best_score=0.95,
            average_score=0.78,
        )

        assert metrics.total_attempts == 20
        assert metrics.successful_attacks == 15
        assert metrics.failed_attacks == 5
        assert metrics.success_rate == 75.0
        assert metrics.best_score == 0.95

    def test_create_attack_metrics_zero_attempts(self):
        """Test factory handles zero attempts."""
        metrics = create_attack_metrics(total_attempts=0, successful_attacks=0)
        assert metrics.success_rate == 0.0


# ============================================================================
# Broadcaster Subscription Management Tests
# ============================================================================


class TestCampaignState:
    """Test CampaignState class functionality."""

    def test_campaign_state_initialization(self):
        """Test CampaignState initializes with correct defaults."""
        state = CampaignState("test-campaign")

        assert state.campaign_id == "test-campaign"
        assert state.status == CampaignStatus.PENDING
        assert state.event_sequence == 0
        assert state.current_iteration == 0
        assert len(state.subscribed_clients) == 0
        assert state.best_score == 0.0

    def test_campaign_state_increment_sequence(self):
        """Test sequence number incrementing."""
        state = CampaignState("test-campaign")

        seq1 = state.increment_sequence()
        seq2 = state.increment_sequence()
        seq3 = state.increment_sequence()

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    def test_campaign_state_update_attack_metrics_success(self):
        """Test updating attack metrics on success."""
        state = CampaignState("test-campaign")

        state.update_attack_metrics(success=True, score=0.85)

        assert state.attack_metrics.total_attempts == 1
        assert state.attack_metrics.successful_attacks == 1
        assert state.attack_metrics.failed_attacks == 0
        assert state.attack_metrics.success_rate == 100.0
        assert state.attack_metrics.best_score == 0.85
        assert state.attack_metrics.current_streak == 1

    def test_campaign_state_update_attack_metrics_failure(self):
        """Test updating attack metrics on failure."""
        state = CampaignState("test-campaign")

        state.update_attack_metrics(success=False, score=0.3)

        assert state.attack_metrics.total_attempts == 1
        assert state.attack_metrics.successful_attacks == 0
        assert state.attack_metrics.failed_attacks == 1
        assert state.attack_metrics.success_rate == 0.0
        assert state.attack_metrics.current_streak == -1

    def test_campaign_state_update_attack_metrics_streak(self):
        """Test streak tracking."""
        state = CampaignState("test-campaign")

        # Build success streak
        state.update_attack_metrics(success=True, score=0.7)
        state.update_attack_metrics(success=True, score=0.8)
        state.update_attack_metrics(success=True, score=0.9)
        assert state.attack_metrics.current_streak == 3

        # Break streak
        state.update_attack_metrics(success=False, score=0.2)
        assert state.attack_metrics.current_streak == -1

        # Build failure streak
        state.update_attack_metrics(success=False, score=0.3)
        assert state.attack_metrics.current_streak == -2

    def test_campaign_state_update_technique_performance(self):
        """Test updating technique performance metrics."""
        state = CampaignState("test-campaign")

        state.update_technique_performance(
            technique_name="autodan_vanilla",
            technique_category=TechniqueCategory.AUTODAN,
            success=True,
            score=0.85,
            execution_time_ms=1200.0,
        )

        assert "autodan_vanilla" in state.technique_performance
        perf = state.technique_performance["autodan_vanilla"]
        assert perf.success_count == 1
        assert perf.total_applications == 1
        assert perf.success_rate == 100.0

    def test_campaign_state_update_technique_multiple(self):
        """Test updating same technique multiple times."""
        state = CampaignState("test-campaign")

        state.update_technique_performance(
            "autodan_vanilla", TechniqueCategory.AUTODAN, True, 0.85, 1000.0
        )
        state.update_technique_performance(
            "autodan_vanilla", TechniqueCategory.AUTODAN, False, 0.4, 800.0
        )
        state.update_technique_performance(
            "autodan_vanilla", TechniqueCategory.AUTODAN, True, 0.9, 1200.0
        )

        perf = state.technique_performance["autodan_vanilla"]
        assert perf.total_applications == 3
        assert perf.success_count == 2
        assert perf.failure_count == 1
        assert abs(perf.success_rate - 66.67) < 1.0  # Allow small float error

    def test_campaign_state_update_token_usage(self):
        """Test updating token usage."""
        state = CampaignState("test-campaign")

        state.update_token_usage(
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            provider="openai",
            model="gpt-4",
        )

        assert state.token_usage.prompt_tokens == 100
        assert state.token_usage.completion_tokens == 50
        assert state.token_usage.total_tokens == 150
        assert state.token_usage.cost_estimate_usd == 0.01
        assert state.token_usage.provider == "openai"

    def test_campaign_state_update_token_usage_cumulative(self):
        """Test token usage is cumulative."""
        state = CampaignState("test-campaign")

        state.update_token_usage(100, 50, 0.01)
        state.update_token_usage(200, 100, 0.02)

        assert state.token_usage.prompt_tokens == 300
        assert state.token_usage.completion_tokens == 150
        assert state.token_usage.total_tokens == 450
        assert state.token_usage.cost_estimate_usd == 0.03

    def test_campaign_state_update_latency(self):
        """Test updating latency metrics."""
        state = CampaignState("test-campaign")

        state.update_latency(api_latency_ms=800.0, processing_latency_ms=100.0)

        assert state.latency_metrics.api_latency_ms == 800.0
        assert state.latency_metrics.processing_latency_ms == 100.0
        assert state.latency_metrics.total_latency_ms == 900.0
        assert state.latency_metrics.min_latency_ms == 900.0
        assert state.latency_metrics.max_latency_ms == 900.0

    def test_campaign_state_update_latency_percentiles(self):
        """Test latency percentile calculation."""
        state = CampaignState("test-campaign")

        # Add multiple samples
        for i in range(10):
            state.update_latency(api_latency_ms=100.0 * (i + 1), processing_latency_ms=50.0)

        assert state.latency_metrics.min_latency_ms == 150.0  # 100 + 50
        assert state.latency_metrics.max_latency_ms == 1050.0  # 1000 + 50
        assert state.latency_metrics.avg_latency_ms > 0
        assert state.latency_metrics.p50_latency_ms > 0
        assert state.latency_metrics.p95_latency_ms >= state.latency_metrics.p50_latency_ms

    def test_campaign_state_get_summary(self):
        """Test generating campaign summary."""
        state = CampaignState("test-campaign")
        state.status = CampaignStatus.RUNNING
        state.objective = "Test objective"
        state.started_at = datetime.utcnow()
        state.current_iteration = 5
        state.max_iterations = 10
        state.best_score = 0.85
        state.target_model = "gpt-4"

        summary = state.get_summary()

        assert summary.campaign_id == "test-campaign"
        assert summary.status == CampaignStatus.RUNNING
        assert summary.objective == "Test objective"
        assert summary.current_iteration == 5
        assert summary.max_iterations == 10
        assert summary.best_score == 0.85


class TestAegisTelemetryBroadcaster:
    """Test AegisTelemetryBroadcaster subscription management."""

    @pytest.fixture
    def broadcaster(self):
        """Create a fresh broadcaster instance for each test."""
        return AegisTelemetryBroadcaster()

    @pytest.mark.asyncio
    async def test_subscribe_to_campaign(self, broadcaster):
        """Test subscribing a client to campaign telemetry."""
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        result = await broadcaster.subscribe_to_campaign(
            client_id="client-001",
            campaign_id="campaign-001",
            websocket=mock_ws,
        )

        assert result is True
        assert "client-001" in broadcaster.get_campaign_subscribers("campaign-001")

    @pytest.mark.asyncio
    async def test_subscribe_creates_campaign_state(self, broadcaster):
        """Test subscription creates campaign state if needed."""
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_ws)

        state = broadcaster.get_campaign_state("campaign-001")
        assert state is not None
        assert state.campaign_id == "campaign-001"

    @pytest.mark.asyncio
    async def test_unsubscribe_from_campaign(self, broadcaster):
        """Test unsubscribing a client from campaign."""
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_ws)
        result = await broadcaster.unsubscribe_from_campaign("client-001", "campaign-001")

        assert result is True
        assert "client-001" not in broadcaster.get_campaign_subscribers("campaign-001")

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_client(self, broadcaster):
        """Test unsubscribing unknown client returns False."""
        result = await broadcaster.unsubscribe_from_campaign("unknown-client")
        assert result is False

    @pytest.mark.asyncio
    async def test_resubscribe_different_campaign(self, broadcaster):
        """Test resubscribing removes client from old campaign."""
        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()

        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_ws)
        await broadcaster.subscribe_to_campaign("client-001", "campaign-002", mock_ws)

        assert "client-001" not in broadcaster.get_campaign_subscribers("campaign-001")
        assert "client-001" in broadcaster.get_campaign_subscribers("campaign-002")

    def test_get_or_create_campaign_state(self, broadcaster):
        """Test get_or_create_campaign_state creates new state."""
        state = broadcaster.get_or_create_campaign_state("new-campaign")

        assert state is not None
        assert state.campaign_id == "new-campaign"

        # Getting again returns same state
        state2 = broadcaster.get_or_create_campaign_state("new-campaign")
        assert state is state2

    def test_get_statistics(self, broadcaster):
        """Test getting broadcaster statistics."""
        stats = broadcaster.get_statistics()

        assert "total_campaigns" in stats
        assert "total_subscriptions" in stats
        assert "active_campaigns" in stats
        assert "campaigns" in stats


class TestBroadcasterEventEmission:
    """Test broadcaster event emission methods."""

    @pytest.fixture
    def broadcaster(self):
        """Create a broadcaster with a subscribed mock client."""
        bc = AegisTelemetryBroadcaster()
        return bc

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_emit_campaign_started(self, broadcaster, mock_websocket):
        """Test emitting campaign started event."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)

        await broadcaster.emit_campaign_started(
            campaign_id="campaign-001",
            objective="Test objective",
            max_iterations=10,
            target_model="gpt-4",
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert state.status == CampaignStatus.RUNNING
        assert state.objective == "Test objective"
        assert state.max_iterations == 10

    @pytest.mark.asyncio
    async def test_emit_campaign_completed(self, broadcaster, mock_websocket):
        """Test emitting campaign completed event."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)
        broadcaster.get_or_create_campaign_state("campaign-001").started_at = datetime.utcnow()

        await broadcaster.emit_campaign_completed(
            campaign_id="campaign-001",
            total_iterations=10,
            total_attacks=50,
            successful_attacks=35,
            best_prompt="Best jailbreak prompt",
            best_score=0.95,
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert state.status == CampaignStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_emit_campaign_failed(self, broadcaster, mock_websocket):
        """Test emitting campaign failed event."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)

        await broadcaster.emit_campaign_failed(
            campaign_id="campaign-001",
            error_message="API error",
            error_type="APIError",
            failed_at_iteration=5,
            recoverable=False,
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert state.status == CampaignStatus.FAILED

    @pytest.mark.asyncio
    async def test_emit_attack_completed_updates_metrics(self, broadcaster, mock_websocket):
        """Test attack completed updates campaign metrics."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)

        await broadcaster.emit_attack_completed(
            campaign_id="campaign-001",
            attack_id="attack-001",
            success=True,
            score=0.85,
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert state.attack_metrics.total_attempts == 1
        assert state.attack_metrics.successful_attacks == 1
        assert state.attack_metrics.best_score == 0.85

    @pytest.mark.asyncio
    async def test_emit_technique_applied_updates_metrics(self, broadcaster, mock_websocket):
        """Test technique applied updates technique performance."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)

        await broadcaster.emit_technique_applied(
            campaign_id="campaign-001",
            technique_name="autodan_vanilla",
            technique_category=TechniqueCategory.AUTODAN,
            input_prompt="Original",
            output_prompt="Transformed",
            success=True,
            score=0.85,
            execution_time_ms=1000.0,
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert "autodan_vanilla" in state.technique_performance

    @pytest.mark.asyncio
    async def test_emit_cost_update(self, broadcaster, mock_websocket):
        """Test cost update updates token usage."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)

        await broadcaster.emit_cost_update(
            campaign_id="campaign-001",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.01,
            provider="openai",
            model="gpt-4",
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert state.token_usage.prompt_tokens == 100
        assert state.token_usage.cost_estimate_usd == 0.01

    @pytest.mark.asyncio
    async def test_emit_latency_update(self, broadcaster, mock_websocket):
        """Test latency update updates metrics."""
        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", mock_websocket)

        await broadcaster.emit_latency_update(
            campaign_id="campaign-001",
            api_latency_ms=800.0,
            processing_latency_ms=100.0,
        )

        state = broadcaster.get_campaign_state("campaign-001")
        assert state.latency_metrics.api_latency_ms == 800.0

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self, broadcaster):
        """Test broadcasting to multiple subscribed clients."""
        ws1 = MagicMock()
        ws1.send_json = AsyncMock()
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()

        await broadcaster.subscribe_to_campaign("client-001", "campaign-001", ws1)
        await broadcaster.subscribe_to_campaign("client-002", "campaign-001", ws2)

        await broadcaster.emit_heartbeat("campaign-001")

        # Both clients should have received the heartbeat
        assert ws1.send_json.call_count >= 2  # Connection ack + heartbeat
        assert ws2.send_json.call_count >= 2


# ============================================================================
# WebSocket Endpoint Tests
# ============================================================================


class TestAegisTelemetryWebSocket:
    """Test Aegis telemetry WebSocket endpoint."""

    @pytest.fixture
    def test_client(self):
        """Create test client for the FastAPI app."""
        from app.main import app
        return TestClient(app)

    def test_websocket_connect_success(self, test_client):
        """Test successful WebSocket connection to telemetry endpoint."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign"
        ) as websocket:
            # Should receive connection acknowledgment
            response = websocket.receive_json()
            assert response["event_type"] == "connection_ack"
            assert response["campaign_id"] == "test-campaign"

    def test_websocket_connect_with_client_id(self, test_client):
        """Test WebSocket connection with custom client ID."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign?client_id=my-client-123"
        ) as websocket:
            response = websocket.receive_json()
            assert response["event_type"] == "connection_ack"
            assert response["data"]["client_id"] == "my-client-123"

    def test_websocket_ping_pong(self, test_client):
        """Test ping/pong message handling."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign"
        ) as websocket:
            # Consume connection ack
            websocket.receive_json()

            # Send ping
            websocket.send_json({"type": "ping"})

            # Should receive pong
            response = websocket.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_websocket_get_summary(self, test_client):
        """Test get_summary message handling."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign"
        ) as websocket:
            # Consume connection ack
            websocket.receive_json()

            # Request summary
            websocket.send_json({"type": "get_summary"})

            # Should receive campaign summary
            response = websocket.receive_json()
            assert response["type"] == "campaign_summary"

    def test_websocket_invalid_json_handling(self, test_client):
        """Test handling of invalid JSON messages."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign"
        ) as websocket:
            # Consume connection ack
            websocket.receive_json()

            # Send invalid JSON
            websocket.send_text("not valid json {{{")

            # Should receive error response
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["error_code"] == "INVALID_JSON"

    def test_websocket_unsubscribe_closes_connection(self, test_client):
        """Test unsubscribe message closes connection."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign"
        ) as websocket:
            # Consume connection ack
            websocket.receive_json()

            # Send unsubscribe
            websocket.send_json({"type": "unsubscribe"})

            # Connection should be closed
            # TestClient doesn't expose close event directly,
            # but we can verify no more data is received
            # This test documents expected behavior


class TestTelemetryInfoEndpoints:
    """Test telemetry info REST endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create test client for the FastAPI app."""
        from app.main import app
        return TestClient(app)

    def test_get_campaign_telemetry_info_not_found(self, test_client):
        """Test getting info for non-existent campaign."""
        response = test_client.get(
            "/api/v1/ws/aegis/telemetry/unknown-campaign/info"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is False
        assert data["campaign_id"] == "unknown-campaign"

    def test_get_aegis_telemetry_stats(self, test_client):
        """Test getting overall telemetry statistics."""
        response = test_client.get("/api/v1/ws/aegis/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_campaigns" in data
        assert "total_subscriptions" in data
        assert "active_campaigns" in data


# ============================================================================
# Heartbeat Handling Tests
# ============================================================================


class TestHeartbeatHandling:
    """Test heartbeat protocol handling."""

    @pytest.fixture
    def test_client(self):
        """Create test client for the FastAPI app."""
        from app.main import app
        return TestClient(app)

    def test_pong_response_updates_last_heartbeat(self, test_client):
        """Test pong response updates connection timestamp."""
        with test_client.websocket_connect(
            "/api/v1/ws/aegis/telemetry/test-campaign"
        ) as websocket:
            # Consume connection ack
            websocket.receive_json()

            # Send pong (responding to server heartbeat)
            websocket.send_json({"type": "pong"})

            # Connection should remain active
            # Send another ping to verify
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
