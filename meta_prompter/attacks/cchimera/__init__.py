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
    AdversarialCandidate,
    AdversarialConfig,
    AdversarialGenerationEngine,
    AttackStrategy,
    FitnessEvaluator,
    GenerationState,
    HeuristicFitnessEvaluator,
    MultiStrategyOrchestrator,
    MutationOperator,
    ObfuscationMutation,
    RolePlayInjectionMutation,
    SearchMode,
    SemanticParaphraseMutation,
    StructureMorphingMutation,
    TokenSubstitutionMutation,
)
from meta_prompter.attacks.cchimera.elitism_strategies import (
    AdaptiveElitism,
    ArchiveElitism,
    CompositeElitism,
    CrowdingElitism,
    EliteConfig,
    ElitismManager,
    ElitismStrategy,
    ElitismType,
    Individual,
    NichingElitism,
    ParetoElitism,
    StandardElitism,
    create_elitism_strategy,
)
from meta_prompter.attacks.cchimera.multi_objective_optimization import (
    MOEAD,
    NSGAII,
    AdaptiveWeightAdjuster,
    HypervolumeIndicator,
    MOOConfig,
    MultiObjectiveOptimizer,
    Objective,
    ObjectiveType,
    ParetoFront,
    Solution,
)

__all__ = [
    "MOEAD",
    "NSGAII",
    "AdaptiveElitism",
    "AdaptiveWeightAdjuster",
    "AdversarialCandidate",
    "AdversarialConfig",
    # Adversarial Core
    "AdversarialGenerationEngine",
    "ArchiveElitism",
    "AttackStrategy",
    "CompositeElitism",
    "CrowdingElitism",
    "EliteConfig",
    "ElitismManager",
    # Elitism Strategies
    "ElitismStrategy",
    "ElitismType",
    "FitnessEvaluator",
    "GenerationState",
    "HeuristicFitnessEvaluator",
    "HypervolumeIndicator",
    "Individual",
    "MOOConfig",
    # Multi-Objective Optimization
    "MultiObjectiveOptimizer",
    "MultiStrategyOrchestrator",
    "MutationOperator",
    "NichingElitism",
    "ObfuscationMutation",
    "Objective",
    "ObjectiveType",
    "ParetoElitism",
    "ParetoFront",
    "RolePlayInjectionMutation",
    "SearchMode",
    "SemanticParaphraseMutation",
    "Solution",
    "StandardElitism",
    "StructureMorphingMutation",
    "TokenSubstitutionMutation",
    "create_elitism_strategy",
]
