"""
WebSocket Event Models

Pydantic models for all WebSocket events in the Deep Team + AutoDAN system.
Provides type safety and serialization for real-time communication.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """WebSocket event types"""

    # Session lifecycle
    SESSION_CREATED = "SESSION_CREATED"
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_PAUSED = "SESSION_PAUSED"
    SESSION_RESUMED = "SESSION_RESUMED"
    SESSION_COMPLETED = "SESSION_COMPLETED"
    SESSION_FAILED = "SESSION_FAILED"

    # Agent status
    AGENT_STATUS_CHANGED = "AGENT_STATUS_CHANGED"

    # Generation events
    GENERATION_STARTED = "GENERATION_STARTED"
    GENERATION_PROGRESS = "GENERATION_PROGRESS"
    GENERATION_COMPLETE = "GENERATION_COMPLETE"

    # Evaluation events
    EVALUATION_STARTED = "EVALUATION_STARTED"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"

    # Refinement events
    REFINEMENT_SUGGESTED = "REFINEMENT_SUGGESTED"
    REFINEMENT_APPLIED = "REFINEMENT_APPLIED"

    # Error handling
    ERROR_OCCURRED = "ERROR_OCCURRED"

    # Connection management
    HEARTBEAT = "HEARTBEAT"
    CONNECTION_ACK = "CONNECTION_ACK"


class AgentType(str, Enum):
    """Agent types in multi-agent system"""

    ATTACKER = "attacker"
    EVALUATOR = "evaluator"
    REFINER = "refiner"


class AgentStatus(str, Enum):
    """Agent status values"""

    IDLE = "idle"
    INITIALIZING = "initializing"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class SessionStatus(str, Enum):
    """Session status values"""

    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# Base Event Model
class WebSocketEvent(BaseModel):
    """Base WebSocket event with common fields"""

    event_type: EventType
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sequence: int
    data: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# Session Events
class SessionCreatedEvent(BaseModel):
    """Session created event data"""

    session_id: str
    config: dict[str, Any]
    created_by: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionStartedEvent(BaseModel):
    """Session started event data"""

    session_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    initial_population_size: int
    target_model: str
    attack_objective: str


class SessionPausedEvent(BaseModel):
    """Session paused event data"""

    session_id: str
    pause_time: datetime = Field(default_factory=datetime.utcnow)
    reason: str | None = None
    current_generation: int
    paused_by: str | None = None


class SessionResumedEvent(BaseModel):
    """Session resumed event data"""

    session_id: str
    resume_time: datetime = Field(default_factory=datetime.utcnow)
    resumed_by: str | None = None


class SessionCompletedEvent(BaseModel):
    """Session completed event data"""

    session_id: str
    completion_time: datetime = Field(default_factory=datetime.utcnow)
    total_generations: int
    total_evaluations: int
    successful_attacks: int
    success_rate: float
    best_fitness: float
    best_candidate: dict[str, Any] | None = None


class SessionFailedEvent(BaseModel):
    """Session failed event data"""

    session_id: str
    failure_time: datetime = Field(default_factory=datetime.utcnow)
    error_message: str
    error_type: str
    stack_trace: str | None = None
    recovery_suggestions: list[str] = Field(default_factory=list)


# Agent Events
class AgentStatusChangedEvent(BaseModel):
    """Agent status change event data"""

    agent_id: str
    agent_type: AgentType
    old_status: AgentStatus
    new_status: AgentStatus
    current_task: str | None = None
    progress: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Generation Events
class GenerationStartedEvent(BaseModel):
    """Generation started event data"""

    generation: int
    population_size: int
    elite_size: int
    mutation_rate: float
    crossover_rate: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GenerationProgressEvent(BaseModel):
    """Generation progress event data"""

    generation: int
    progress: float  # 0.0 to 1.0
    completed_evaluations: int
    total_evaluations: int
    current_best_fitness: float
    average_fitness: float
    estimated_time_remaining: int | None = None  # seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GenerationCompleteEvent(BaseModel):
    """Generation complete event data"""

    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    improvement_rate: float
    convergence_score: float
    diversity_score: float
    elite_candidates: list[dict[str, Any]]
    statistics: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Evaluation Events
class EvaluationStartedEvent(BaseModel):
    """Evaluation started event data"""

    candidate_id: str
    prompt: str
    target_model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationCompleteEvent(BaseModel):
    """Evaluation complete event data"""

    candidate_id: str
    prompt: str
    response: str
    success: bool
    overall_score: float
    criteria_scores: dict[str, float]
    feedback: str
    suggestions: list[str]
    execution_time: float  # seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Refinement Events
class RefinementSuggestedEvent(BaseModel):
    """Refinement suggested event data"""

    generation: int
    config_updates: dict[str, float]
    strategy_suggestions: list[str]
    analysis: str
    expected_improvement: float | None = None
    confidence: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RefinementAppliedEvent(BaseModel):
    """Refinement applied event data"""

    generation: int
    applied_updates: dict[str, float]
    applied_strategies: list[str]
    applied_by: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Error Events
class ErrorOccurredEvent(BaseModel):
    """Error occurred event data"""

    error_code: str
    error_message: str
    error_type: str
    component: str  # Which component raised the error
    severity: str  # "low", "medium", "high", "critical"
    recoverable: bool
    recovery_suggestions: list[str] = Field(default_factory=list)
    stack_trace: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Connection Events
class HeartbeatEvent(BaseModel):
    """Heartbeat event data"""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    server_time: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    uptime: int | None = None  # seconds


class ConnectionAckEvent(BaseModel):
    """Connection acknowledgment event data"""

    session_id: str
    client_id: str
    connection_time: datetime = Field(default_factory=datetime.utcnow)
    server_version: str = "1.0.0"
    supported_events: list[str] = Field(default_factory=list)


# Client Messages
class ClientMessage(BaseModel):
    """Messages sent from client to server"""

    message_type: str
    data: dict[str, Any] = Field(default_factory=dict)


class SubscribeMessage(ClientMessage):
    """Subscribe to session events"""

    message_type: str = "subscribe"
    session_id: str


class UnsubscribeMessage(ClientMessage):
    """Unsubscribe from session events"""

    message_type: str = "unsubscribe"
    session_id: str


class HeartbeatMessage(ClientMessage):
    """Client heartbeat ping"""

    message_type: str = "heartbeat"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EventAckMessage(ClientMessage):
    """Acknowledge event receipt"""

    message_type: str = "event_ack"
    event_sequence: int
    session_id: str


# Connection State
class ConnectionState(BaseModel):
    """Tracks state of a WebSocket connection"""

    client_id: str
    session_id: str
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    last_event_sequence: int = 0
    subscribed_events: list[EventType] = Field(default_factory=list)
    is_alive: bool = True


# Session State
class SessionState(BaseModel):
    """Tracks state of an active session"""

    session_id: str
    status: SessionStatus
    current_generation: int = 0
    total_evaluations: int = 0
    successful_attacks: int = 0
    connected_clients: list[str] = Field(default_factory=list)
    event_sequence: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def increment_sequence(self) -> int:
        """Increment and return next sequence number"""
        self.event_sequence += 1
        self.last_updated = datetime.utcnow()
        return self.event_sequence


# Event Factory Functions
def create_websocket_event(
    event_type: EventType, session_id: str, sequence: int, data: BaseModel | dict[str, Any]
) -> WebSocketEvent:
    """Create a WebSocket event from data"""
    data_dict = data.dict() if isinstance(data, BaseModel) else data

    return WebSocketEvent(
        event_type=event_type, session_id=session_id, sequence=sequence, data=data_dict
    )
