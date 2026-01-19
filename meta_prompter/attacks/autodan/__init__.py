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

from meta_prompter.attacks.autodan.attack import AutoDANAttack, AutoDANConfig
from meta_prompter.attacks.autodan.fitness_evaluation import (
    CrossEntropyLoss,
    FitnessConfig,
    FitnessEvaluationPipeline,
    FitnessObjective,
    FitnessResult,
    FluencyScorer,
    JailbreakSuccessEstimator,
    PerplexityCalculator,
    ReadabilityScorer,
    SemanticSimilarityScorer,
    StealthScorer,
)
from meta_prompter.attacks.autodan.hierarchical_genetic_algorithm import (
    DefaultFitnessEvaluator,
    GeneticOperators,
    HGAConfig,
    HierarchicalGeneticAlgorithm,
    Individual,
    Island,
)
from meta_prompter.attacks.autodan.reasoning_integration import (
    AutoDANReasoningWorkflow,
    MutationWithReasoningResult,
    ReasoningEnhancedMutation,
)
from meta_prompter.attacks.autodan.token_perturbation import (
    GradientEstimator,
    GumbelSoftmaxSampler,
    HomoglyphPerturbation,
    HotFlipAttack,
    PerturbationResult,
    SemanticPerturbation,
    TokenPerturbationEngine,
)
from meta_prompter.attacks.autodan.token_perturbation import (
    TokenPerturbationConfig as PerturbationConfig,
)

__all__ = [
    # Main Attack
    "AutoDANAttack",
    "AutoDANConfig",
    # Reasoning Integration
    "AutoDANReasoningWorkflow",
    "CrossEntropyLoss",
    "DefaultFitnessEvaluator",
    "FitnessConfig",
    # Fitness Evaluation
    "FitnessEvaluationPipeline",
    "FitnessObjective",
    "FitnessResult",
    "FluencyScorer",
    "GeneticOperators",
    "GradientEstimator",
    "GumbelSoftmaxSampler",
    "HGAConfig",
    # Hierarchical Genetic Algorithm
    "HierarchicalGeneticAlgorithm",
    "HomoglyphPerturbation",
    "HotFlipAttack",
    "Individual",
    "Island",
    "JailbreakSuccessEstimator",
    "MutationWithReasoningResult",
    "PerplexityCalculator",
    "PerturbationConfig",
    "PerturbationResult",
    "ReadabilityScorer",
    "ReasoningEnhancedMutation",
    "SemanticPerturbation",
    "SemanticSimilarityScorer",
    "StealthScorer",
    # Token Perturbation
    "TokenPerturbationEngine",
]
