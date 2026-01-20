"""DeepTeam Integration Module.

This module provides comprehensive integration with the DeepTeam red-teaming framework
for adversarial AI security testing, including:

- Jailbreak prompt generation with multiple attack strategies
- AutoDAN and AutoDAN-Turbo genetic algorithm-based attacks
- PAIR, TAP, Crescendo, and Gray Box attack strategies
- Real-time streaming and WebSocket support
- Lifelong learning capabilities
"""

# Core service layer
from .config import (
    AttackConfig,
    AttackType,
    DeepTeamConfig,
    PresetConfig,
    RedTeamSessionConfig,
    RiskAssessmentResult,
    VulnerabilityConfig,
    VulnerabilityType,
    get_preset_config,
)

# Service layer
from .jailbreak_service import (  # Request/Response models; Service; WebSocket events; Helpers
    GenerationCompleteEvent,
    GenerationErrorEvent,
    GenerationProgressEvent,
    GenerationStartEvent,
    JailbreakBatchRequest,
    JailbreakGenerateRequest,
    JailbreakGenerateResponse,
    JailbreakService,
    PromptGeneratedEvent,
    StrategiesResponse,
    StrategyInfo,
    WebSocketEvent,
    WebSocketHandler,
    configure_jailbreak_service,
    get_jailbreak_service,
    stream_to_sse,
)

# Core prompt generator
from .prompt_generator import (  # Strategies; Config models; Enums; Prompt models; Callbacks; Main generator; Factory
    AttackStrategy,
    AttackStrategyConfig,
    AttackStrategyType,
    AutoDANStrategy,
    AutoDANTurboStrategy,
    CrescendoStrategy,
    GeneratedPrompt,
    GenerationCallbacks,
    GenerationProgress,
    GenerationResult,
    GenerationStatus,
    GeneratorConfig,
    GrayBoxStrategy,
    JailbreakPromptGenerator,
    PAIRStrategy,
    StrategyFactory,
    TAPStrategy,
    VulnerabilityCategory,
)
from .service import DeepTeamService

__all__ = [
    "AttackConfig",
    # Strategies
    "AttackStrategy",
    # Config models
    "AttackStrategyConfig",
    # Enums
    "AttackStrategyType",
    "AttackType",
    "AutoDANStrategy",
    "AutoDANTurboStrategy",
    "CrescendoStrategy",
    "DeepTeamConfig",
    # Core service
    "DeepTeamService",
    # Prompt models
    "GeneratedPrompt",
    # Callbacks
    "GenerationCallbacks",
    "GenerationCompleteEvent",
    "GenerationErrorEvent",
    "GenerationProgress",
    "GenerationProgressEvent",
    "GenerationResult",
    "GenerationStartEvent",
    "GenerationStatus",
    "GeneratorConfig",
    "GrayBoxStrategy",
    "JailbreakBatchRequest",
    # Request/Response models
    "JailbreakGenerateRequest",
    "JailbreakGenerateResponse",
    # Main generator
    "JailbreakPromptGenerator",
    # Service
    "JailbreakService",
    "PAIRStrategy",
    "PresetConfig",
    "PromptGeneratedEvent",
    "RedTeamSessionConfig",
    "RiskAssessmentResult",
    "StrategiesResponse",
    # Factory
    "StrategyFactory",
    "StrategyInfo",
    "TAPStrategy",
    "VulnerabilityCategory",
    "VulnerabilityConfig",
    "VulnerabilityType",
    # WebSocket events
    "WebSocketEvent",
    "WebSocketHandler",
    "configure_jailbreak_service",
    "get_jailbreak_service",
    "get_preset_config",
    # Helpers
    "stream_to_sse",
]

__version__ = "1.0.0"
