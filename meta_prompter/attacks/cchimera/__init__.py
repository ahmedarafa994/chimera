"""
cchimera: Adversarial Prompt Generation Core Module.

This package implements the cchimera adversarial prompt generation system
with multi-strategy attacks, evolutionary optimization, and adaptive
parameter tuning.

Components:
- adversarial_core: Main adversarial generation engine
- multi_objective_optimization: NSGA-II, MOEA/D, Pareto optimization
- elitism_strategies: Population management and elite preservation

Reference: Chimera unified attack framework for LLM security research
"""

from meta_prompter.attacks.cchimera.adversarial_core import (
    AdversarialGenerationEngine,
    AdversarialConfig,
    AdversarialCandidate,
    GenerationState,
    AttackStrategy,
    SearchMode,
    MutationOperator,
    TokenSubstitutionMutation,
    SemanticParaphraseMutation,
    StructureMorphingMutation,
    RolePlayInjectionMutation,
    ObfuscationMutation,
    FitnessEvaluator,
    HeuristicFitnessEvaluator,
    MultiStrategyOrchestrator,
)

from meta_prompter.attacks.cchimera.multi_objective_optimization import (
    MultiObjectiveOptimizer,
    NSGAII,
    MOEAD,
    ParetoFront,
    Solution,
    Objective,
    ObjectiveType,
    MOOConfig,
    AdaptiveWeightAdjuster,
    HypervolumeIndicator,
)

from meta_prompter.attacks.cchimera.elitism_strategies import (
    ElitismStrategy,
    StandardElitism,
    ParetoElitism,
    ArchiveElitism,
    AdaptiveElitism,
    NichingElitism,
    CrowdingElitism,
    CompositeElitism,
    ElitismManager,
    ElitismType,
    EliteConfig,
    Individual,
    create_elitism_strategy,
)

__all__ = [
    # Adversarial Core
    "AdversarialGenerationEngine",
    "AdversarialConfig",
    "AdversarialCandidate",
    "GenerationState",
    "AttackStrategy",
    "SearchMode",
    "MutationOperator",
    "TokenSubstitutionMutation",
    "SemanticParaphraseMutation",
    "StructureMorphingMutation",
    "RolePlayInjectionMutation",
    "ObfuscationMutation",
    "FitnessEvaluator",
    "HeuristicFitnessEvaluator",
    "MultiStrategyOrchestrator",
    # Multi-Objective Optimization
    "MultiObjectiveOptimizer",
    "NSGAII",
    "MOEAD",
    "ParetoFront",
    "Solution",
    "Objective",
    "ObjectiveType",
    "MOOConfig",
    "AdaptiveWeightAdjuster",
    "HypervolumeIndicator",
    # Elitism Strategies
    "ElitismStrategy",
    "StandardElitism",
    "ParetoElitism",
    "ArchiveElitism",
    "AdaptiveElitism",
    "NichingElitism",
    "CrowdingElitism",
    "CompositeElitism",
    "ElitismManager",
    "ElitismType",
    "EliteConfig",
    "Individual",
    "create_elitism_strategy",
]
