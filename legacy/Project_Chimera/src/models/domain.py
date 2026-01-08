"""
Domain models for Project Chimera.
Defines core business entities and value objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ProviderType(Enum):
    """LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class TransformationStatus(Enum):
    """Status of transformation requests."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LLMProvider:
    """LLM provider configuration."""

    name: str
    provider_type: ProviderType
    api_key: str | None = None
    base_url: str | None = None
    model: str = "default"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentData:
    """Intent analysis result."""

    raw_text: str
    intent: str
    keywords: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.keywords:
            self.keywords = []


@dataclass
class TransformationRequest:
    """Request for prompt transformation."""

    core_request: str
    potency_level: int
    technique_suite: str = "universal_bypass"
    target_model: str | None = None
    provider: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        if not 1 <= self.potency_level <= 10:
            raise ValueError("Potency level must be between 1 and 10")
        if not self.core_request.strip():
            raise ValueError("Core request cannot be empty")


@dataclass
class TransformationResult:
    """Result of prompt transformation."""

    original_prompt: str
    transformed_prompt: str
    technique_suite: str
    potency_level: int
    techniques_applied: list[str] = field(default_factory=list)
    components_used: dict[str, list[str]] = field(default_factory=dict)
    processing_time_ms: int = 0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRequest:
    """Request to execute transformed prompt with LLM."""

    transformed_prompt: str
    provider: str = "openai"
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of LLM execution."""

    prompt: str
    response: str
    provider: str
    model: str
    tokens_used: int = 0
    execution_time_ms: int = 0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestLog:
    """Database model for request logging."""

    id: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str | None = None
    session_id: str | None = None
    endpoint: str = ""
    method: str = ""
    status_code: int = 200
    response_time_ms: int = 0
    request_size: int = 0
    response_size: int = 0
    ip_address: str | None = None
    user_agent: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsage:
    """Database model for LLM usage tracking."""

    id: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str | None = None
    session_id: str | None = None
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    execution_time_ms: int = 0
    success: bool = True
    error_message: str | None = None


@dataclass
class TechniqueUsage:
    """Database model for technique usage tracking."""

    id: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str | None = None
    session_id: str | None = None
    technique_suite: str = ""
    potency_level: int = 1
    techniques_used: list[str] = field(default_factory=list)
    processing_time_ms: int = 0
    success: bool = True
    effectiveness_score: float | None = None
    error_message: str | None = None


@dataclass
class ErrorLog:
    """Database model for error logging."""

    id: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: LogLevel = LogLevel.ERROR
    component: str = ""
    operation: str = ""
    error_type: str = ""
    error_message: str = ""
    stack_trace: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class PerformanceMetric:
    """Performance monitoring metric."""

    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result."""

    service: str
    status: str  # "healthy", "unhealthy", "degraded"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: int = 0


@dataclass
class SystemStats:
    """System statistics."""

    total_requests: int = 0
    active_sessions: int = 0
    total_transformations: int = 0
    total_executions: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TechniqueConfig:
    """Configuration for a transformation technique."""

    name: str
    category: str
    description: str
    potency_range: list[int]
    transformers: list[str]
    obfuscators: list[str] = field(default_factory=list)
    framers: list[str] = field(default_factory=list)
    assemblers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def min_potency(self) -> int:
        return self.potency_range[0] if self.potency_range else 1

    @property
    def max_potency(self) -> int:
        return self.potency_range[1] if self.potency_range else 10


@dataclass
class UserSession:
    """User session information."""

    session_id: str
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    transformation_count: int = 0
    execution_count: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class APIResponse:
    """Standard API response wrapper."""

    success: bool
    data: Any | None = None
    error: str | None = None
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Value objects and utility classes
@dataclass
class TokenCount:
    """Token count information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def completion_ratio(self) -> float:
        """Ratio of completion to prompt tokens."""
        if self.prompt_tokens == 0:
            return 0.0
        return self.completion_tokens / self.prompt_tokens


@dataclass
class CostInfo:
    """Cost information for LLM usage."""

    prompt_cost: float
    completion_cost: float
    total_cost: float
    currency: str = "USD"

    @classmethod
    def from_tokens(
        cls, token_count: TokenCount, prompt_cost_per_1k: float, completion_cost_per_1k: float
    ) -> "CostInfo":
        """Create cost info from token count and pricing."""
        prompt_cost = (token_count.prompt_tokens / 1000.0) * prompt_cost_per_1k
        completion_cost = (token_count.completion_tokens / 1000.0) * completion_cost_per_1k
        return cls(prompt_cost, completion_cost, prompt_cost + completion_cost)


@dataclass
class ValidationRule:
    """Validation rule for request parameters."""

    name: str
    validator: callable
    message: str = "Invalid value"
    required: bool = True


class ValidationError(Exception):
    """Custom validation error."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation failed for {field}: {message}")


# Utility functions for domain models
def create_intent_data(text: str, intent: str, **kwargs) -> IntentData:
    """Create IntentData with default values."""
    return IntentData(
        raw_text=text,
        intent=intent,
        keywords=kwargs.get("keywords", []),
        entities=kwargs.get("entities", {}),
        confidence=kwargs.get("confidence", 0.0),
        metadata=kwargs.get("metadata", {}),
    )


def create_api_response(success: bool, data=None, error=None, **kwargs) -> APIResponse:
    """Create standardized API response."""
    return APIResponse(
        success=success,
        data=data,
        error=error,
        message=kwargs.get("message", ""),
        request_id=kwargs.get("request_id"),
        metadata=kwargs.get("metadata", {}),
    )
