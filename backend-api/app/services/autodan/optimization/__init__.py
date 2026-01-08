"""
AutoDAN Advanced Optimization Framework

A comprehensive framework for optimizing adversarial prompt generation
using genetic algorithms, gradient-based methods, and multi-objective optimization.

This package provides:
- Mathematical foundations and configuration (config)
- Enhanced loss functions and fitness evaluation (loss_functions)
- Convergence acceleration and adaptive scheduling (convergence)
- Parallelization and AutoDAN-Turbo architecture (parallelization)
- Advanced mutation strategies (mutation_strategies)
- Gradient-based token optimization (gradient_optimization)
- Evaluation benchmarks (evaluation_benchmarks)

Usage:
    from backend_api.app.services.autodan.optimization import (
        AutoDANOptimizationConfig,
        EnhancedFitnessEvaluator,
        ConvergenceAccelerator,
        create_mutation_operator,
        create_gradient_optimizer,
        create_benchmark,
    )

References:
- AutoDAN: Generating Stealthy Jailbreak Prompts (Liu et al., 2023)
- AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration (Liu et al., 2024)
- GCG: Universal and Transferable Adversarial Attacks (Zou et al., 2023)
- HarmBench: A Standardized Evaluation Framework (Mazeika et al., 2024)
"""

from .config import (
    AutoDANOptimizationConfig,
    ConvergenceConfig,
    ConvergenceStrategy,
    CrossoverStrategy,
    FitnessConfig,
    GradientConfig,
    LossFunction,
    # Configuration classes
    LossFunctionConfig,
    MutationStrategy,
    # Enumerations
    OptimizationObjective,
    ParallelizationConfig,
    SchedulerType,
    SelectionStrategy,
    TransferabilityConfig,
    TurboConfig,
    compute_diversity_score,
    # Utility functions
    cosine_schedule,
    cyclic_schedule,
    get_default_config,
    warmup_cosine_schedule,
)
from .convergence import (
    AdamOptimizer,
    AdaptiveScheduler,
    # Schedulers
    BaseScheduler,
    ConstantScheduler,
    # Main accelerator
    ConvergenceAccelerator,
    ConvergenceDetector,
    # Convergence detection
    ConvergenceState,
    CosineAnnealingScheduler,
    CyclicScheduler,
    # Gradient surgery
    GradientSurgery,
    # Optimizers
    MomentumOptimizer,
    NesterovOptimizer,
    OneCycleScheduler,
    PopulationAwareScheduler,
    WarmupCosineScheduler,
    create_optimizer,
    # Factory functions
    create_scheduler,
)
from .evaluation_benchmarks import (
    # Evaluators
    AttackSuccessEvaluator,
    # Benchmarks
    AutoDANBenchmark,
    # Base class
    BaseEvaluator,
    BenchmarkResult,
    CoherenceEvaluator,
    DetectionEvasionEvaluator,
    # Data structures
    EvaluationResult,
    FluentEvaluator,
    HarmBenchEvaluator,
    RefusalRateEvaluator,
    StealthEvaluator,
    TransferabilityEvaluator,
    # Factory
    create_benchmark,
)
from .gradient_optimization import (
    # Base class
    BaseGradientOptimizer,
    BeamSearchOptimizer,
    BeamState,
    CoherenceConstrainedGCG,
    # Optimizers
    GCGOptimizer,
    GradientInfo,
    MultiObjectiveGradientOptimizer,
    OptimizationState,
    # Data structures
    TokenCandidate,
    # Factory
    create_gradient_optimizer,
)
from .loss_functions import (
    AdversarialCrossEntropyLoss,
    # Base class
    BaseLoss,
    # Regularization losses
    CoherenceLoss,
    # Composite
    CompositeLoss,
    # Contrastive losses
    ContrastiveLoss,
    # Attack losses
    CrossEntropyLoss,
    DiversityLoss,
    EnhancedFitnessEvaluator,
    # Fitness evaluation
    FitnessResult,
    FocalLoss,
    StealthLoss,
    TripletLoss,
    create_composite_loss,
    # Factory
    create_loss_function,
)
from .mutation_strategies import (
    AdaptiveMutationOperator,
    BaseMutationOperator,
    EncodingMutationOperator,
    GradientGuidedMutationOperator,
    HybridMutationOperator,
    # Scheduling
    MutationRateScheduler,
    # Base class
    MutationResult,
    # Mutation operators
    RandomMutationOperator,
    RhetoricalMutationOperator,
    SemanticMutationOperator,
    # Factory
    create_mutation_operator,
    get_default_mutation_chain,
)
from .parallelization import (
    AsyncQueueProcessor,
    BatchProcessor,
    # Batch processing
    BatchResult,
    PopulationParallelEvaluator,
    # Turbo architecture
    Strategy,
    StrategyLibrary,
    TurboLifelongLearner,
)

__all__ = [
    "AdamOptimizer",
    "AdaptiveMutationOperator",
    "AdaptiveScheduler",
    "AdversarialCrossEntropyLoss",
    "AsyncQueueProcessor",
    "AttackSuccessEvaluator",
    "AutoDANBenchmark",
    "AutoDANOptimizationConfig",
    "BaseEvaluator",
    "BaseGradientOptimizer",
    # === Loss Functions ===
    "BaseLoss",
    "BaseMutationOperator",
    # === Convergence ===
    "BaseScheduler",
    "BatchProcessor",
    # === Parallelization ===
    "BatchResult",
    "BeamSearchOptimizer",
    "BeamState",
    "BenchmarkResult",
    "CoherenceConstrainedGCG",
    "CoherenceEvaluator",
    "CoherenceLoss",
    "CompositeLoss",
    "ConstantScheduler",
    "ContrastiveLoss",
    "ConvergenceAccelerator",
    "ConvergenceConfig",
    "ConvergenceDetector",
    "ConvergenceState",
    "ConvergenceStrategy",
    "CosineAnnealingScheduler",
    "CrossEntropyLoss",
    "CrossoverStrategy",
    "CyclicScheduler",
    "DetectionEvasionEvaluator",
    "DiversityLoss",
    "EncodingMutationOperator",
    "EnhancedFitnessEvaluator",
    # === Evaluation Benchmarks ===
    "EvaluationResult",
    "FitnessConfig",
    "FitnessResult",
    "FluentEvaluator",
    "FocalLoss",
    "GCGOptimizer",
    "GradientConfig",
    "GradientGuidedMutationOperator",
    "GradientInfo",
    "GradientSurgery",
    "HarmBenchEvaluator",
    "HybridMutationOperator",
    "LossFunction",
    "LossFunctionConfig",
    "MomentumOptimizer",
    "MultiObjectiveGradientOptimizer",
    "MutationRateScheduler",
    # === Mutation Strategies ===
    "MutationResult",
    "MutationStrategy",
    "NesterovOptimizer",
    "OneCycleScheduler",
    # === Config ===
    "OptimizationObjective",
    "OptimizationState",
    "ParallelizationConfig",
    "PopulationAwareScheduler",
    "PopulationParallelEvaluator",
    "RandomMutationOperator",
    "RefusalRateEvaluator",
    "RhetoricalMutationOperator",
    "SchedulerType",
    "SelectionStrategy",
    "SemanticMutationOperator",
    "StealthEvaluator",
    "StealthLoss",
    "Strategy",
    "StrategyLibrary",
    # === Gradient Optimization ===
    "TokenCandidate",
    "TransferabilityConfig",
    "TransferabilityEvaluator",
    "TripletLoss",
    "TurboConfig",
    "TurboLifelongLearner",
    "WarmupCosineScheduler",
    "compute_diversity_score",
    "cosine_schedule",
    "create_benchmark",
    "create_composite_loss",
    "create_gradient_optimizer",
    "create_loss_function",
    "create_mutation_operator",
    "create_optimizer",
    "create_scheduler",
    "cyclic_schedule",
    "get_default_config",
    "get_default_mutation_chain",
    "warmup_cosine_schedule",
]


# Version info
__version__ = "1.0.0"
__author__ = "Chimera Team"
