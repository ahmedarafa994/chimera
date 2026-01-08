"""
Unified Mathematical Framework for ExtendAttack + AutoDAN Integration.

This module synthesizes two adversarial attack methodologies:
- ExtendAttack: Token extension via poly-base ASCII obfuscation (resource exhaustion)
- AutoDAN: Reasoning-based adversarial prompt mutation (jailbreak)

Combined Attack Objectives:
1. Token Amplification: L(Y') >> L(Y)
2. Reasoning Chain Extension: R(Q') >> R(Q)
3. Resource Depletion: Cost(Y') >> Cost(Y)
4. Attack Success Rate: ASR(Q') > threshold
5. Accuracy Preservation: Acc(A') ≈ Acc(A)

Mathematical Notation:
- Q: Original query
- Q': Adversarial query (after combined transformation)
- Q_e: Query after ExtendAttack obfuscation
- Q_a: Query after AutoDAN mutation
- ρ: ExtendAttack obfuscation ratio ∈ [0, 1]
- μ: AutoDAN mutation strength ∈ [0, 1]
- α: Attack vector weight (0 = pure ExtendAttack, 1 = pure AutoDAN)
- β: Resource exhaustion priority ∈ [0, 1]
- γ: Jailbreak success priority ∈ [0, 1]

Example Usage:
    >>> from meta_prompter.unified import (
    ...     UnifiedAttackParameters,
    ...     CombinedAttackMetrics,
    ...     UnifiedFitnessCalculator,
    ...     AttackComposer,
    ...     ParetoOptimizer,
    ...     UnifiedAttackConfig,
    ... )
    >>>
    >>> # Configure unified attack
    >>> params = UnifiedAttackParameters(
    ...     obfuscation_ratio=0.5,
    ...     mutation_strength=0.5,
    ...     attack_vector_weight=0.5,
    ... )
    >>>
    >>> # Calculate unified fitness
    >>> calculator = UnifiedFitnessCalculator()
    >>> metrics = CombinedAttackMetrics(...)
    >>> fitness = calculator.calculate_unified_fitness(metrics)
"""

from meta_prompter.unified.engine import (
    CombinedAttackEngine,
    CombinedResult,
    MutatedResult,
    MutationEngine,
    MutationType,
    ObfuscatedResult,
    ObfuscationEngine,
)
from meta_prompter.unified.math_framework import (
    AttackComposer,
    AttackVector,
    CombinedAttackMetrics,
    MutationOptimizer,
    OptimizationObjective,
    ReasoningChainExtender,
    TokenExtensionOptimizer,
    UnifiedAttackParameters,
    UnifiedFitnessCalculator,
    calculate_entropy,
    calculate_perplexity_estimate,
    log_amplification,
    normalize_score,
)
from meta_prompter.unified.models import (
    AttackPhase,
    CompositionStrategy,
    MultiVectorAttackPlan,
    ResourceExhaustionTarget,
    TargetType,
    UnifiedAttackConfig,
    UnifiedAttackResult,
)
from meta_prompter.unified.optimization import (
    AdaptiveParameterController,
    ConstraintSatisfaction,
    OptimizationMethod,
    OptimizationResult,
    ParetoOptimizer,
)

__all__ = [
    # Math Framework - Core Types
    "AttackVector",
    "OptimizationObjective",
    # Math Framework - Parameters and Metrics
    "UnifiedAttackParameters",
    "CombinedAttackMetrics",
    # Math Framework - Calculators
    "UnifiedFitnessCalculator",
    "TokenExtensionOptimizer",
    "ReasoningChainExtender",
    "MutationOptimizer",
    # Math Framework - Composers
    "AttackComposer",
    # Math Framework - Utility Functions
    "calculate_entropy",
    "calculate_perplexity_estimate",
    "normalize_score",
    "log_amplification",
    # Optimization - Types
    "OptimizationMethod",
    "OptimizationResult",
    # Optimization - Optimizers
    "ParetoOptimizer",
    "AdaptiveParameterController",
    "ConstraintSatisfaction",
    # Models - Enums
    "AttackPhase",
    "TargetType",
    "CompositionStrategy",
    # Models - Configuration
    "UnifiedAttackConfig",
    # Models - Results
    "UnifiedAttackResult",
    "MultiVectorAttackPlan",
    "ResourceExhaustionTarget",
    # Engine - Main Classes
    "CombinedAttackEngine",
    "ObfuscationEngine",
    "MutationEngine",
    # Engine - Result Types
    "ObfuscatedResult",
    "MutatedResult",
    "CombinedResult",
    # Engine - Enums
    "MutationType",
]

__version__ = "1.0.0"
