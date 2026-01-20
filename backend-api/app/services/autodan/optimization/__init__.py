"""AutoDAN Advanced Optimization Framework.

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

from .config import (  # Configuration classes; Enumerations; Utility functions
    AutoDANOptimizationConfig,
    ConvergenceConfig,
    ConvergenceStrategy,
    CrossoverStrategy,
    FitnessConfig,
    GradientConfig,
    LossFunction,
    LossFunctionConfig,
    MutationStrategy,
    OptimizationObjective,
    ParallelizationConfig,
    SchedulerType,
    SelectionStrategy,
    TransferabilityConfig,
    TurboConfig,
    compute_diversity_score,
    cosine_schedule,
    cyclic_schedule,
    get_default_config,
    warmup_cosine_schedule,
)
from .convergence import (  # Schedulers; Main accelerator; Convergence detection; Gradient surgery; Optimizers; Factory functions
    AdamOptimizer,
    AdaptiveScheduler,
    BaseScheduler,
    ConstantScheduler,
    ConvergenceAccelerator,
    ConvergenceDetector,
    ConvergenceState,
    CosineAnnealingScheduler,
    CyclicScheduler,
    GradientSurgery,
    MomentumOptimizer,
    NesterovOptimizer,
    OneCycleScheduler,
    PopulationAwareScheduler,
    WarmupCosineScheduler,
    create_optimizer,
    create_scheduler,
)
from .evaluation_benchmarks import (  # Evaluators; Benchmarks; Base class; Data structures; Factory
    AttackSuccessEvaluator,
    AutoDANBenchmark,
    BaseEvaluator,
    BenchmarkResult,
    CoherenceEvaluator,
    DetectionEvasionEvaluator,
    EvaluationResult,
    FluentEvaluator,
    HarmBenchEvaluator,
    RefusalRateEvaluator,
    StealthEvaluator,
    TransferabilityEvaluator,
    create_benchmark,
)
from .gradient_optimization import (  # Base class; Optimizers; Data structures; Factory
    BaseGradientOptimizer,
    BeamSearchOptimizer,
    BeamState,
    CoherenceConstrainedGCG,
    GCGOptimizer,
    GradientInfo,
    MultiObjectiveGradientOptimizer,
    OptimizationState,
    TokenCandidate,
    create_gradient_optimizer,
)
from .loss_functions import (  # Base class; Regularization losses; Composite; Contrastive losses; Attack losses; Fitness evaluation; Factory
    AdversarialCrossEntropyLoss,
    BaseLoss,
    CoherenceLoss,
    CompositeLoss,
    ContrastiveLoss,
    CrossEntropyLoss,
    DiversityLoss,
    EnhancedFitnessEvaluator,
    FitnessResult,
    FocalLoss,
    StealthLoss,
    TripletLoss,
    create_composite_loss,
    create_loss_function,
)
from .mutation_strategies import (  # Scheduling; Base class; Mutation operators; Factory
    AdaptiveMutationOperator,
    BaseMutationOperator,
    EncodingMutationOperator,
    GradientGuidedMutationOperator,
    HybridMutationOperator,
    MutationRateScheduler,
    MutationResult,
    RandomMutationOperator,
    RhetoricalMutationOperator,
    SemanticMutationOperator,
    create_mutation_operator,
    get_default_mutation_chain,
)
from .parallelization import (  # Batch processing; Turbo architecture
    AsyncQueueProcessor,
    BatchProcessor,
    BatchResult,
    PopulationParallelEvaluator,
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
