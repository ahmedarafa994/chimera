"""
WebSocket Service Package

Provides real-time event streaming for Deep Team + AutoDAN sessions.

Components:
- models: Pydantic event models and type definitions
- connection_manager: WebSocket connection lifecycle management
- broadcaster: High-level event broadcasting API
- integrations: Service integration wrappers (AutoDAN, GPTFuzz)
"""

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
    "AgentStatus",
    "AgentStatusChangedEvent",
    "AgentType",
    "AutoDANWebSocketIntegration",
    "ConnectionManager",
    "ErrorOccurredEvent",
    "EvaluationCompleteEvent",
    "EventBroadcaster",
    # Event types
    "EventType",
    "GPTFuzzWebSocketIntegration",
    "GenerationCompleteEvent",
    "GenerationProgressEvent",
    "GenerationStartedEvent",
    "RefinementSuggestedEvent",
    "SessionCompletedEvent",
    "SessionFailedEvent",
    "SessionStartedEvent",
    "SessionStatus",
    # Event models
    "WebSocketEvent",
    "WebSocketIntegration",
    # Service integrations
    "autodan_websocket",
    # Connection management
    "connection_manager",
    "create_websocket_event",
    # Broadcasting
    "event_broadcaster",
    "gptfuzz_websocket"
]
