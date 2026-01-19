"""
WebSocket Service Package

Provides real-time event streaming for Deep Team + AutoDAN sessions
and Aegis campaign telemetry.

Components:
- models: Pydantic event models and type definitions
- connection_manager: WebSocket connection lifecycle management
- broadcaster: High-level event broadcasting API
- aegis_broadcaster: Aegis campaign telemetry broadcaster
- integrations: Service integration wrappers (AutoDAN, GPTFuzz)
"""

from .aegis_broadcaster import AegisTelemetryBroadcaster, aegis_telemetry_broadcaster
from .broadcaster import EventBroadcaster, event_broadcaster
from .connection_manager import ConnectionManager, connection_manager
from .integrations import (
                                AutoDANWebSocketIntegration,
                                GPTFuzzWebSocketIntegration,
                                WebSocketIntegration,
                                autodan_websocket,
                                gptfuzz_websocket,
)
from .models import (
                                AgentStatus,
                                AgentStatusChangedEvent,
                                AgentType,
                                ErrorOccurredEvent,
                                EvaluationCompleteEvent,
                                EventType,
                                GenerationCompleteEvent,
                                GenerationProgressEvent,
                                GenerationStartedEvent,
                                RefinementSuggestedEvent,
                                SessionCompletedEvent,
                                SessionFailedEvent,
                                SessionStartedEvent,
                                SessionStatus,
                                WebSocketEvent,
                                create_websocket_event,
)

__all__ = [
    # Aegis telemetry broadcaster
    "AegisTelemetryBroadcaster",
    # Agent types and status
    "AgentStatus",
    "AgentStatusChangedEvent",
    "AgentType",
    "AutoDANWebSocketIntegration",
    # Connection management
    "ConnectionManager",
    # Event types
    "ErrorOccurredEvent",
    "EvaluationCompleteEvent",
    "EventBroadcaster",
    "EventType",
    "GPTFuzzWebSocketIntegration",
    # Event models
    "GenerationCompleteEvent",
    "GenerationProgressEvent",
    "GenerationStartedEvent",
    "RefinementSuggestedEvent",
    "SessionCompletedEvent",
    "SessionFailedEvent",
    "SessionStartedEvent",
    "SessionStatus",
    "WebSocketEvent",
    "WebSocketIntegration",
    "aegis_telemetry_broadcaster",
    # Service integrations
    "autodan_websocket",
    "connection_manager",
    # Broadcasting
    "create_websocket_event",
    "event_broadcaster",
    "gptfuzz_websocket",
]
