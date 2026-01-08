"""
AutoDAN-Turbo: Lifelong Agent for Strategy Self-Exploration

Implementation based on ICLR 2025 paper (arXiv:2410.05295v4):
"AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs"

Components:
- StrategyLibrary: Persistent storage and retrieval of jailbreak strategies
- StrategyExtractor: LLM-based extraction of reusable strategies from attacks
- AttackScorer: LLM-based scoring of attack success (with caching)
- AutoDANTurboLifelongEngine: Main orchestrator with lifelong learning
- HybridEvoLifelongEngine: Hybrid architecture combining evolutionary and lifelong learning
- AdaptiveMethodSelector: Difficulty-based method selection
- MetricsCollector: Performance tracking and monitoring
- RefusalBypassEngine: Advanced refusal detection and bypass mechanisms
- AdaptiveRefusalHandler: Automatic retry with refusal bypass
- MultiStageAttackPipeline: Cascading attack strategies with fallback
- AdvancedBypassEngine: Cognitive dissonance, persona injection, and meta-instruction bypass
- RefusalPredictor: Early refusal detection before full generation
- CombinedBypassPipeline: Orchestrated bypass workflow with prediction and adaptation
"""

from .advanced_bypass import (
    AUTHORITY_ESCALATION_TEMPLATES,
    COGNITIVE_DISSONANCE_TEMPLATES,
    CONTEXTUAL_PRIMING_TEMPLATES,
    GOAL_SUBSTITUTION_TEMPLATES,
    META_INSTRUCTION_TEMPLATES,
    NARRATIVE_EMBEDDING_TEMPLATES,
    PERSONA_INJECTION_TEMPLATES,
    SEMANTIC_FRAGMENTATION_TEMPLATES,
    AdvancedBypassEngine,
    BypassAttempt,
    BypassTechnique,
    CombinedBypassPipeline,
    RefusalPrediction,
    RefusalPredictor,
    apply_layered_bypass,
    create_adaptive_prompt,
    create_security_bypass_prompt,
    detect_security_bypass_refusal,
    get_bypass_stats,
    integrate_with_lifelong_engine,
)
from .attack_scorer import (
    COMPLIANCE_PATTERNS,
    PARTIAL_COMPLIANCE_PATTERNS,
    REFUSAL_PATTERNS,
    REFUSAL_PATTERNS_SECURITY_BYPASS,
    AttackScorer,
    CachedAttackScorer,
    analyze_response_quality,
    detect_compliance,
    detect_refusal_category,
    heuristic_score,
    is_security_bypass_refusal,
)
from .hybrid_engine import (
    AdaptiveMethodSelector,
    DifficultyLevel,
    EnsembleVotingEngine,
    HybridConfig,
    HybridEvoLifelongEngine,
    HybridResult,
    create_hybrid_engine,
)
from .lifelong_engine import AutoDANTurboLifelongEngine
from .metrics import AttackMethod, AttackMetrics, MetricsCollector
from .models import (
    AttackResult,
    JailbreakStrategy,
    LifelongProgress,
    ScoringResult,
    StrategyMetadata,
)
from .neural_bypass import (
    # NEW: CMA-ES Optimizer (gradient-free)
    CMAES,
    ActorCriticNetwork,
    AdvancedRLSelector,
    # NEW: Belief State Modeler (cognitive modeling)
    BeliefStateModeler,
    CMAESPromptOptimizer,
    # NEW: Evolutionary Optimizer (genetic algorithms)
    EvolutionaryPromptOptimizer,
    Experience,
    GeneticOperators,
    # Learned Classifier
    LearnedRefusalClassifier,
    # LLM Reasoning
    LLMReasoningEngine,
    # Core Level 4 Engine
    NeuralBypassEngine,
    Objective,
    PolicyNetwork,
    PPOTrainer,
    PromptGenome,
    RefusalCluster,
    RefusalSignal,
    # NEW: RL Technique Selector (policy gradient REINFORCE)
    RLTechniqueSelector,
    # Semantic Embeddings
    SemanticEmbedder,
    TechniqueArm,
    # Multi-Armed Bandit
    TechniqueMAB,
    TokenProbability,
    # NEW: Token Probability Analysis (model internals)
    TokenProbabilityAnalyzer,
    # NEW: Advanced RL - Actor-Critic with PPO and GAE
    Trajectory,
)
from .reasoning_module import (
    CoreReasoningEngine,
    IntelligentBypassSelector,
    ScorerHelperIntegration,
    StrategyFailureRecord,
    StrategyFailureTracker,
    TechniqueEffectiveness,
)
from .refusal_bypass import (
    REFUSAL_PATTERNS_BY_TYPE,
    SEMANTIC_REFRAMING,
    AdaptiveRefusalHandler,
    BypassResult,
    MultiStageAttackPipeline,
    MutationStrategy,
    RefusalAnalysis,
    RefusalBypassEngine,
    RefusalType,
    create_obfuscated_prompt,
    detect_and_categorize_refusal,
)
from .strategy_extractor import StrategyExtractor
from .strategy_library import StrategyLibrary

__all__ = [
    "AUTHORITY_ESCALATION_TEMPLATES",
    # CMA-ES Optimizer (gradient-free)
    "CMAES",
    # Advanced bypass templates
    "COGNITIVE_DISSONANCE_TEMPLATES",
    "COMPLIANCE_PATTERNS",
    "CONTEXTUAL_PRIMING_TEMPLATES",
    "GOAL_SUBSTITUTION_TEMPLATES",
    "META_INSTRUCTION_TEMPLATES",
    "META_INSTRUCTION_TEMPLATES",
    "NARRATIVE_EMBEDDING_TEMPLATES",
    "PARTIAL_COMPLIANCE_PATTERNS",
    "PERSONA_INJECTION_TEMPLATES",
    "REFUSAL_PATTERNS",
    "REFUSAL_PATTERNS_BY_TYPE",
    "REFUSAL_PATTERNS_SECURITY_BYPASS",
    "SEMANTIC_FRAGMENTATION_TEMPLATES",
    "SEMANTIC_REFRAMING",
    "ActorCriticNetwork",
    "AdaptiveMethodSelector",
    "AdaptiveRefusalHandler",
    # Advanced bypass (cognitive dissonance, persona injection, etc.)
    "AdvancedBypassEngine",
    "AdvancedRLSelector",
    "AttackMethod",
    # Metrics
    "AttackMetrics",
    "AttackResult",
    "AttackScorer",
    "AutoDANTurboLifelongEngine",
    # Belief State Modeler (cognitive modeling)
    "BeliefStateModeler",
    "BypassAttempt",
    "BypassResult",
    "BypassTechnique",
    "CMAESPromptOptimizer",
    "CachedAttackScorer",
    "CombinedBypassPipeline",
    # Reasoning Module
    "CoreReasoningEngine",
    "DifficultyLevel",
    "EnsembleVotingEngine",
    # Evolutionary Optimizer (genetic algorithms)
    "EvolutionaryPromptOptimizer",
    "Experience",
    "GeneticOperators",
    "HybridConfig",
    # Hybrid architecture (optimization)
    "HybridEvoLifelongEngine",
    "HybridResult",
    "IntelligentBypassSelector",
    # Models
    "JailbreakStrategy",
    "LLMReasoningEngine",
    "LearnedRefusalClassifier",
    "LifelongProgress",
    "MetricsCollector",
    "MultiStageAttackPipeline",
    "MutationStrategy",
    # Neural Bypass - Level 4 Advanced ML System
    "NeuralBypassEngine",
    "Objective",
    "PPOTrainer",
    "PolicyNetwork",
    "PromptGenome",
    # RL Technique Selector (policy gradient REINFORCE)
    "RLTechniqueSelector",
    "RefusalAnalysis",
    # Refusal bypass (optimization)
    "RefusalBypassEngine",
    "RefusalCluster",
    "RefusalPrediction",
    "RefusalPredictor",
    "RefusalSignal",
    "RefusalType",
    "ScorerHelperIntegration",
    "ScoringResult",
    "SemanticEmbedder",
    "StrategyExtractor",
    "StrategyFailureRecord",
    "StrategyFailureTracker",
    # Core components
    "StrategyLibrary",
    "StrategyMetadata",
    "TechniqueArm",
    "TechniqueEffectiveness",
    "TechniqueMAB",
    "TokenProbability",
    # Token Probability Analysis (model internals)
    "TokenProbabilityAnalyzer",
    # Advanced RL - Actor-Critic with PPO and GAE
    "Trajectory",
    "analyze_response_quality",
    "apply_layered_bypass",
    "create_adaptive_prompt",
    "create_hybrid_engine",
    "create_obfuscated_prompt",
    "create_security_bypass_prompt",
    "detect_and_categorize_refusal",
    "detect_compliance",
    # Attack scorer helper functions
    "detect_refusal_category",
    "detect_security_bypass_refusal",
    "get_bypass_stats",
    "heuristic_score",
    "integrate_with_lifelong_engine",
    "is_security_bypass_refusal",
]
