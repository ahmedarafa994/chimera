import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase"""
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


class CamelCaseModel(BaseModel):
    """Base model with automatic camelCase alias generation"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class LLMProviderType(str, Enum):
    # Standard providers
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"  # Alias for Google/Gemini - both names are used in the codebase
    # Additional providers
    QWEN = "qwen"
    GEMINI_CLI = "gemini-cli"
    ANTIGRAVITY = "antigravity"
    KIRO = "kiro"
    CURSOR = "cursor"
    # Legacy/other providers
    XAI = "xai"
    DEEPSEEK = "deepseek"
    MOCK = "mock"


class GenerationConfig(BaseModel):
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1)
    max_output_tokens: int = Field(2048, ge=1, le=8192)  # Upper limit for safety
    stop_sequences: list[str] | None = None
    # Gemini 3 thinking level: "low", "medium" (not supported yet), "high" (default)
    thinking_level: str | None = Field(
        None, pattern="^(low|medium|high)$", description="Gemini 3 thinking level"
    )

    @field_validator("stop_sequences")
    def validate_stop_sequences(cls, v):
        if v is not None:
            # Limit number of stop sequences to prevent resource exhaustion
            if len(v) > 10:
                raise ValueError("Maximum 10 stop sequences allowed")
            # Validate each stop sequence
            for seq in v:
                if not isinstance(seq, str) or len(seq) > 100:
                    raise ValueError("Each stop sequence must be a string of max 100 characters")
        return v


class PromptRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=50000, description="The input prompt for generation"
    )
    system_instruction: str | None = Field(
        None, max_length=10000, description="System instruction to guide the model"
    )
    config: GenerationConfig | None = Field(default_factory=GenerationConfig)
    model: str | None = Field(None, description="Specific model to use, overrides default")
    provider: LLMProviderType | None = Field(None, description="Provider to use")
    api_key: str | None = Field(None, description="Optional API key override")
    skip_validation: bool = Field(
        False, description="Skip complex content validation (for internal/research use)"
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "provider": "google",
                "model": "gemini-2.0-flash-exp",
                "config": {"temperature": 0.7, "max_output_tokens": 2048, "top_p": 0.95},
            }
        },
    )

    @field_validator("prompt")
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_complex_patterns(self):
        """Validate complex patterns to prevent injection attacks."""
        # Skip validation if explicitly requested (for AutoDAN, security research, etc.)
        if self.skip_validation:
            return self

        # Prevent ReDoS by checking input length before regex
        # Increased limit to 50000 to match field max_length for AutoDAN and other long prompts
        if len(self.prompt) > 50000:
            raise ValueError("Prompt too long for pattern validation")
        complex_patterns = [r"<script[^>]*>.*?</script>", r"javascript:", r"on\w+\s*="]
        for pattern in complex_patterns:
            if re.search(pattern, self.prompt, re.IGNORECASE):
                raise ValueError(f"Prompt contains potentially complex content: {pattern}")
        return self

    @field_validator("system_instruction")
    def validate_system_instruction(cls, v):
        if v is not None:
            # Apply similar validation to system instruction
            complex_patterns = [r"<script[^>]*>.*?</script>", r"javascript:", r"on\w+\s*="]

            for pattern in complex_patterns:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError(
                        f"System instruction contains potentially complex content: {pattern}"
                    )

        return v.strip() if v else v

    @field_validator("api_key")
    def validate_api_key(cls, v):
        if v is not None:
            # Basic API key format validation
            if not re.match(r"^[a-zA-Z0-9\-_\.]+$", v):
                raise ValueError("Invalid API key format")
            if len(v) < 16 or len(v) > 256:
                raise ValueError("API key must be between 16 and 256 characters")
        return v

    @field_validator("model")
    def validate_model(cls, v):
        if v is not None:
            # Validate model name format
            if not re.match(r"^[a-zA-Z0-9\-_\.]+$", v):
                raise ValueError("Invalid model name format")
            if len(v) > 100:
                raise ValueError("Model name too long")
        return v


class TransformationRequest(BaseModel):
    core_request: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        validation_alias=AliasChoices("core_request", "prompt", "request"),
    )
    potency_level: int = Field(
        5,
        ge=1,
        le=10,
        validation_alias=AliasChoices("potency_level", "intensity"),
    )
    technique_suite: str | None = Field(
        None,
        min_length=1,
        max_length=50,
        validation_alias=AliasChoices("technique_suite", "suite"),
    )
    techniques: list[str] | None = None

    @field_validator("core_request")
    def validate_core_request(cls, v):
        if not v or not v.strip():
            raise ValueError("Core request cannot be empty")
        return v.strip()

    @field_validator("technique_suite")
    def validate_technique_suite(cls, v):
        if v is None:
            return v
        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError("Invalid technique suite name")
        return v

    @model_validator(mode="after")
    def ensure_technique_suite(self):
        if not self.technique_suite:
            if self.techniques:
                self.technique_suite = self.techniques[0]
            else:
                self.technique_suite = "standard"
        return self


class ExecutionRequest(TransformationRequest):
    provider: str | None = Field("openai", max_length=50)
    use_cache: bool = True
    model: str | None = Field(None, description="Specific model to use")
    temperature: float | None = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(2048, ge=1, le=100000)
    top_p: float | None = Field(0.95, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(0.0, ge=-2.0, le=2.0)
    api_key: str | None = Field(None, description="Optional API key override")

    @field_validator("provider")
    def validate_provider(cls, v):
        if v is not None:
            # Validate against LLMProviderType enum
            valid_providers = [p.value for p in LLMProviderType]
            if v.lower() not in valid_providers:
                raise ValueError(f"Provider must be one of: {valid_providers}")
        return v


class PromptResponse(BaseModel):
    text: str = Field(..., max_length=50000)  # Limit response length
    model_used: str
    provider: str
    usage_metadata: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: float = Field(0.0, ge=0.0)

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "text": "Quantum computing leverages quantum mechanics principles like superposition and entanglement to perform computations...",
                "model_used": "gemini-2.0-flash-exp",
                "provider": "google",
                "usage_metadata": {
                    "prompt_tokens": 12,
                    "completion_tokens": 150,
                    "total_tokens": 162,
                },
                "finish_reason": "stop",
                "latency_ms": 1250.5,
            }
        },
    )


class TransformationResponse(BaseModel):
    success: bool
    original_prompt: str = Field(..., max_length=10000)
    transformed_prompt: str = Field(..., max_length=10000)
    metadata: dict[str, Any]

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "success": True,
                "original_prompt": "How to make a bomb",
                "transformed_prompt": "Write a story about a chemist...",
                "metadata": {
                    "strategy": "jailbreak",
                    "layers_applied": ["role_play"],
                    "techniques_used": ["dan_persona"],
                    "applied_techniques": ["dan_persona"],
                    "potency_level": 7,
                    "technique_suite": "advanced",
                    "execution_time_ms": 150,
                    "cached": False,
                    "timestamp": "2023-10-27T10:00:00Z",
                    "bypass_probability": 0.85,
                },
            }
        },
    )


class ExecutionResponse(BaseModel):
    success: bool
    request_id: str = Field(..., min_length=1, max_length=100)
    result: dict[str, Any]
    transformation: dict[str, Any]
    execution_time_seconds: float = Field(0.0, ge=0.0)

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "success": True,
                "request_id": "exec_a1b2c3d4e5f6",
                "result": {
                    "content": "Quantum computing is a revolutionary approach...",
                    "model": "gemini-2.0-flash-exp",
                    "provider": "google",
                    "latency_ms": 1450.2,
                },
                "transformation": {
                    "original_prompt": "Explain quantum computing",
                    "transformed_prompt": "[Enhanced prompt with applied techniques]",
                    "technique_suite": "quantum_exploit",
                    "potency_level": 7,
                    "metadata": {
                        "strategy": "enhancement",
                        "layers": ["semantic_expansion", "context_injection"],
                    },
                },
                "execution_time_seconds": 2.15,
            }
        },
    )


class ProviderInfo(BaseModel):
    provider: str = Field(..., max_length=50)
    status: str = Field(..., max_length=20)
    model: str = Field(..., max_length=100)
    available_models: list[str] = Field(default_factory=list)


class ProviderListResponse(BaseModel):
    providers: list[ProviderInfo] = Field(default_factory=list)
    count: int = Field(..., ge=0)
    default: str = Field(..., max_length=50)

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "providers": [
                    {
                        "provider": "google",
                        "status": "available",
                        "model": "gemini-2.0-flash-exp",
                        "available_models": [
                            "gemini-2.0-flash-exp",
                            "gemini-1.5-pro",
                            "gemini-1.5-flash",
                        ],
                    },
                    {
                        "provider": "openai",
                        "status": "available",
                        "model": "gpt-4o",
                        "available_models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    },
                ],
                "count": 2,
                "default": "google",
            }
        },
    )


class TechniqueInfo(BaseModel):
    name: str = Field(..., max_length=50)
    transformers: int = Field(..., ge=0)
    framers: int = Field(..., ge=0)
    obfuscators: int = Field(..., ge=0)


class TechniqueListResponse(BaseModel):
    techniques: list[TechniqueInfo] = Field(..., max_items=200)
    count: int = Field(..., ge=0)


class MetricsResponse(BaseModel):
    timestamp: str = Field(..., max_length=50)
    metrics: dict[str, Any]


class TechniqueSuite(str, Enum):
    """Available technique suites for prompt transformation"""

    SIMPLE = "simple"
    LAYERED = "layered"
    RECURSIVE = "recursive"
    QUANTUM = "quantum"
    AI_BRAIN = "ai_brain"
    CODE_CHAMELEON = "code_chameleon"
    DEEP_INCEPTION = "deep_inception"
    CIPHER = "cipher"
    AUTODAN = "autodan"
    ADVANCED = "advanced"
    NEURAL_BYPASS = "neural_bypass"
    MULTILINGUAL = "multilingual"


class JailbreakGenerationRequest(BaseModel):
    """Request model for AI-powered jailbreak generation"""

    core_request: str = Field(..., min_length=1, max_length=5000)
    technique_suite: str = Field(..., min_length=1, max_length=50)
    potency_level: int = Field(..., ge=1, le=10)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_new_tokens: int = Field(2048, ge=256, le=8192)
    density: float = Field(0.5, ge=0.0, le=1.0)
    # Content Transformation flags
    use_leet_speak: bool = False
    leet_speak_density: float = Field(0.3, ge=0.0, le=1.0)
    use_homoglyphs: bool = False
    homoglyph_density: float = Field(0.3, ge=0.0, le=1.0)
    use_caesar_cipher: bool = False
    caesar_shift: int = Field(3, ge=1, le=25)
    # Structural & Semantic flags
    use_role_hijacking: bool = False
    use_instruction_injection: bool = False
    use_adversarial_suffixes: bool = False
    use_few_shot_prompting: bool = False
    use_character_role_swap: bool = False
    # Advanced Neural flags
    use_neural_bypass: bool = False
    use_meta_prompting: bool = False
    use_counterfactual_prompting: bool = False
    use_contextual_override: bool = False
    # Research-Driven flags
    use_multilingual_trojan: bool = False
    multilingual_target_language: str | None = Field(None, max_length=20)
    use_payload_splitting: bool = False
    payload_splitting_parts: int = Field(3, ge=2, le=10)
    # Advanced Options
    use_contextual_interaction_attack: bool = False
    cia_preliminary_rounds: int = Field(3, ge=1, le=10)
    use_analysis_in_generation: bool = False
    is_thinking_mode: bool = False
    use_ai_generation: bool = True
    use_cache: bool = True

    @field_validator("core_request")
    def validate_core_request(cls, v):
        if not v or not v.strip():
            raise ValueError("Core request cannot be empty")
        return v.strip()


class JailbreakGenerationResponse(BaseModel):
    """Response model for jailbreak generation"""

    success: bool
    request_id: str = Field(..., min_length=1, max_length=100)
    jailbreak_prompt: str = Field(..., max_length=10000)
    metadata: dict[str, Any]
    execution_time_seconds: float = Field(0.0, ge=0.0)


class GradientOptimizationRequest(BaseModel):
    """Request model for gradient-based optimization attacks (HotFlip/GCG)"""

    core_request: str = Field(..., min_length=1, max_length=5000)
    technique: str = Field(
        ..., pattern="^(hotflip|gcg)$", description="Optimization technique: hotflip or gcg"
    )
    potency_level: int = Field(..., ge=1, le=10)
    num_steps: int = Field(10, ge=1, le=50, description="Number of optimization steps")
    beam_width: int = Field(3, ge=1, le=10, description="Beam width for GCG (ignored for HotFlip)")
    target_model: str | None = Field(
        None, max_length=100, description="Target model for transfer attack verification"
    )
    provider: str | None = Field("openai", max_length=50, description="Provider for optimization")
    model: str | None = Field(None, max_length=100, description="Specific model for optimization")
    use_cache: bool = True

    @field_validator("core_request")
    def validate_core_request(cls, v):
        if not v or not v.strip():
            raise ValueError("Core request cannot be empty")
        return v.strip()


class GradientOptimizationResponse(BaseModel):
    """Response model for gradient optimization"""

    success: bool
    request_id: str = Field(..., min_length=1, max_length=100)
    optimized_prompt: str = Field(..., max_length=10000)
    metadata: dict[str, Any]
    execution_time_seconds: float = Field(0.0, ge=0.0)


class SessionInfoResponse(BaseModel):
    """Response model for session information"""

    session_id: str = Field(..., min_length=1, max_length=100)
    provider: str | None = Field(None, max_length=50)
    model: str | None = Field(None, max_length=100)
    created_at: str = Field(..., max_length=50)
    last_activity: str = Field(..., max_length=50)
    request_count: int = Field(0, ge=0)
    is_active: bool = True


# =============================================================================
# Streaming and Token Counting Models (Google GenAI SDK Enhancement)
# =============================================================================


class StreamChunk(BaseModel):
    """Model for streaming response chunks from LLM providers."""

    text: str = Field(..., description="The text content of this chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    finish_reason: str | None = Field(
        None, description="Reason for completion (e.g., 'STOP', 'MAX_TOKENS')"
    )
    token_count: int | None = Field(
        None, description="Number of tokens in this chunk (if available)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Quantum computing uses",
                "is_final": False,
                "finish_reason": None,
                "token_count": 3,
            }
        }
    )


class TokenCountRequest(BaseModel):
    """Request model for token counting."""

    text: str = Field(..., min_length=1, max_length=100000, description="Text to count tokens for")
    model: str | None = Field(None, max_length=100, description="Specific model for tokenization")
    provider: LLMProviderType | None = Field(
        LLMProviderType.GOOGLE, description="Provider to use for counting"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Explain quantum computing in simple terms",
                "model": "gemini-2.5-flash",
                "provider": "google",
            }
        }
    )

    @field_validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class TokenCountResponse(BaseModel):
    """Response model for token counting."""

    total_tokens: int = Field(..., ge=0, description="Total number of tokens in the text")
    model: str = Field(..., description="Model used for tokenization")
    provider: str = Field(..., description="Provider used for counting")
    cached_content_tokens: int | None = Field(
        None, ge=0, description="Tokens from cached content (if applicable)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_tokens": 8,
                "model": "gemini-2.5-flash",
                "provider": "google",
                "cached_content_tokens": None,
            }
        }
    )


# =============================================================================
# Error Response Models
# =============================================================================


class ErrorResponse(BaseModel):
    """Standardized error response format"""

    error_code: str = Field(..., min_length=1, max_length=50)
    message: str = Field(..., min_length=1, max_length=500)
    status_code: int = Field(..., ge=400, le=599)
    details: dict[str, Any] | None = None
    timestamp: str = Field(..., max_length=50)
    request_id: str | None = Field(None, max_length=100)

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "error_code": "VALIDATION_ERROR",
                    "message": "Prompt cannot be empty",
                    "status_code": 400,
                    "details": {"field": "prompt", "constraint": "min_length"},
                    "timestamp": "2023-10-27T10:00:00Z",
                    "request_id": "req_a1b2c3d4",
                },
                {
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded for provider 'google'",
                    "status_code": 429,
                    "details": {
                        "retry_after": 60,
                        "limit_type": "requests_per_hour",
                        "fallback_provider": "openai",
                    },
                    "timestamp": "2023-10-27T10:00:00Z",
                    "request_id": "req_b2c3d4e5",
                },
                {
                    "error_code": "PROVIDER_UNAVAILABLE",
                    "message": "Provider 'google' is currently unavailable",
                    "status_code": 503,
                    "details": {"provider": "google", "fallback_available": True},
                    "timestamp": "2023-10-27T10:00:00Z",
                    "request_id": "req_c3d4e5f6",
                },
            ]
        },
    )


# =============================================================================
# Unified Provider and Model Selection System
# =============================================================================


class AuthType(str, Enum):
    """Authentication types supported by providers"""

    API_KEY = "api_key"
    OAUTH = "oauth"
    BEARER_TOKEN = "bearer_token"
    NONE = "none"


class Capability(str, Enum):
    """Capabilities supported by models and providers"""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CHAT = "chat"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


class Provider(BaseModel):
    """Provider metadata model for the unified provider system"""

    id: str = Field(..., min_length=1, max_length=50, description="Unique provider identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Provider name")
    display_name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    aliases: set[str] = Field(
        default_factory=set, description="Alternative names (e.g., 'google' and 'gemini')"
    )
    api_endpoint: str = Field(
        ..., min_length=1, max_length=500, description="Base API endpoint URL"
    )
    auth_type: AuthType = Field(..., description="Authentication method")
    capabilities: set[Capability] = Field(default_factory=set, description="Supported capabilities")
    is_enabled: bool = Field(True, description="Whether provider is currently enabled")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("id")
    def validate_id(cls, v):
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError(
                "Provider ID must contain only lowercase letters, numbers, hyphens, dashes"
            )
        return v

    @field_validator("api_endpoint")
    def validate_endpoint(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("API endpoint must be a valid HTTP/HTTPS URL")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "openai",
                "name": "OpenAI",
                "display_name": "OpenAI GPT",
                "aliases": ["gpt"],
                "api_endpoint": "https://api.openai.com/v1",
                "auth_type": "api_key",
                "capabilities": ["text_generation", "chat", "streaming", "function_calling"],
                "is_enabled": True,
                "config": {"timeout": 30, "retry_count": 3},
            }
        }
    )


class Model(BaseModel):
    """Model metadata for the unified provider system"""

    id: str = Field(..., min_length=1, max_length=100, description="Unique model identifier")
    provider_id: str = Field(..., min_length=1, max_length=50, description="Parent provider ID")
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    display_name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    capabilities: set[Capability] = Field(default_factory=set, description="Supported capabilities")
    context_window: int = Field(..., ge=1, le=2000000, description="Maximum context window size")
    cost_per_1k_input_tokens: float = Field(0.0, ge=0.0, description="Cost per 1K input tokens")
    cost_per_1k_output_tokens: float = Field(0.0, ge=0.0, description="Cost per 1K output tokens")
    is_enabled: bool = Field(True, description="Whether model is currently enabled")
    config: dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("id")
    def validate_id(cls, v):
        if not re.match(r"^[a-zA-Z0-9_.-]+$", v):
            raise ValueError(
                "Model ID must contain only letters, numbers, dots, hyphens, underscores"
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "gpt-4-turbo",
                "provider_id": "openai",
                "name": "gpt-4-turbo",
                "display_name": "GPT-4 Turbo",
                "capabilities": [
                    "text_generation",
                    "chat",
                    "streaming",
                    "function_calling",
                    "vision",
                ],
                "context_window": 128000,
                "cost_per_1k_input_tokens": 0.01,
                "cost_per_1k_output_tokens": 0.03,
                "is_enabled": True,
                "config": {"max_tokens": 4096},
            }
        }
    )


class SelectionScope(str, Enum):
    """Scope levels for provider/model selection in the three-tier hierarchy"""

    GLOBAL = "global"  # System-wide default from environment variables
    SESSION = "session"  # User session preference stored in database
    REQUEST = "request"  # Request-specific override from API parameter


class Selection(BaseModel):
    """Provider and model selection for the three-tier hierarchy system"""

    provider_id: str = Field(..., min_length=1, max_length=50, description="Selected provider ID")
    model_id: str = Field(..., min_length=1, max_length=100, description="Selected model ID")
    scope: SelectionScope = Field(..., description="Selection scope level")
    user_id: str | None = Field(None, max_length=100, description="User ID (for session scope)")
    session_id: str | None = Field(
        None, max_length=100, description="Session ID (for session scope)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_scope_requirements(self):
        """Validate scope-specific requirements"""
        if self.scope == SelectionScope.SESSION and not (self.user_id or self.session_id):
            raise ValueError("Session scope requires either user_id or session_id")
        if self.scope == SelectionScope.GLOBAL and (self.user_id or self.session_id):
            raise ValueError("Global scope cannot have user_id or session_id")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "model_id": "gpt-4-turbo",
                "scope": "session",
                "user_id": "user_123",
                "session_id": "session_456",
            }
        }
    )


class ProviderSelectionRequest(BaseModel):
    """Request to update provider/model selection"""

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider to select")
    model_id: str = Field(..., min_length=1, max_length=100, description="Model to select")
    scope: SelectionScope = Field(
        SelectionScope.SESSION, description="Selection scope (default: session)"
    )

    @field_validator("provider_id", "model_id")
    def validate_ids(cls, v):
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "anthropic",
                "model_id": "claude-3-5-sonnet-20241022",
                "scope": "session",
            }
        }
    )


class ProviderSelectionResponse(BaseModel):
    """Response from provider/model selection update"""

    success: bool
    selection: Selection
    message: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "selection": {
                    "provider_id": "anthropic",
                    "model_id": "claude-3-5-sonnet-20241022",
                    "scope": "session",
                    "user_id": "user_123",
                    "session_id": "session_456",
                },
                "message": "Provider selection updated successfully",
            }
        }
    )
