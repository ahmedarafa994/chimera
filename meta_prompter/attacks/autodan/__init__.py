"""
AutoDAN: Automated Adversarial Prompt Generation Module.

This package implements AutoDAN-style automated adversarial prompt generation
using hierarchical genetic algorithms, token-level perturbations, and
multi-objective fitness optimization.

Components:
- hierarchical_genetic_algorithm: HGA with island model and NSGA-II
- token_perturbation: Gradient-guided token perturbation strategies
- fitness_evaluation: Cross-entropy loss and multi-objective fitness

Reference: AutoDAN adversarial prompt generation for LLM red-teaming
"""

from meta_prompter.attacks.autodan.hierarchical_genetic_algorithm import (
    HierarchicalGeneticAlgorithm,
    HGAConfig,
    Individual,
    Island,
    GeneticOperators,
    DefaultFitnessEvaluator,
)

from meta_prompter.attacks.autodan.token_perturbation import (
    TokenPerturbationEngine,
    TokenPerturbationConfig as PerturbationConfig,
    PerturbationResult,
    HotFlipAttack,
    GumbelSoftmaxSampler,
    GradientEstimator,
    SemanticPerturbation,
    HomoglyphPerturbation,
)

from meta_prompter.attacks.autodan.fitness_evaluation import (
    FitnessEvaluationPipeline,
    FitnessConfig,
    FitnessResult,
    FitnessObjective,
    CrossEntropyLoss,
    PerplexityCalculator,
    FluencyScorer,
    ReadabilityScorer,
    JailbreakSuccessEstimator,
    StealthScorer,
    SemanticSimilarityScorer,
)

from meta_prompter.attacks.autodan.attack import AutoDANAttack, AutoDANConfig
from meta_prompter.attacks.autodan.reasoning_integration import (
    AutoDANReasoningWorkflow,
    ReasoningEnhancedMutation,
    MutationWithReasoningResult,
)

__all__ = [
    # Main Attack
    "AutoDANAttack",
    "AutoDANConfig",
    # Reasoning Integration
    "AutoDANReasoningWorkflow",
    "ReasoningEnhancedMutation",
    "MutationWithReasoningResult",
    # Hierarchical Genetic Algorithm
    "HierarchicalGeneticAlgorithm",
    "HGAConfig",
    "Individual",
    "Island",
    "GeneticOperators",
    "DefaultFitnessEvaluator",
    # Token Perturbation
    "TokenPerturbationEngine",
    "PerturbationConfig",
    "PerturbationResult",
    "HotFlipAttack",
    "GumbelSoftmaxSampler",
    "GradientEstimator",
    "SemanticPerturbation",
    "HomoglyphPerturbation",
    # Fitness Evaluation
    "FitnessEvaluationPipeline",
    "FitnessConfig",
    "FitnessResult",
    "FitnessObjective",
    "CrossEntropyLoss",
    "PerplexityCalculator",
    "FluencyScorer",
    "ReadabilityScorer",
    "JailbreakSuccessEstimator",
    "StealthScorer",
    "SemanticSimilarityScorer",
]
