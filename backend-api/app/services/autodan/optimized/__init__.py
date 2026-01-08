"""
AutoDAN Optimization Framework.

A comprehensive framework for optimizing AutoDAN reasoning systems
and AutoDAN Turbo architectures for maximum performance and efficiency.

Modules:
- enhanced_gradient_optimizer: Momentum-based CCGS with surrogate models
- adaptive_mutation_engine: UCB1-based mutation selection
- multi_objective_fitness: Multi-objective scoring with Pareto frontier
- async_batch_pipeline: Parallel candidate processing
- adaptive_learning_rate: PPO-style learning rate adaptation
- convergence_acceleration: Warm start, early stopping, curriculum learning
- memory_management: Efficient memory handling
- distributed_strategy_library: FAISS-based strategy storage
- benchmark_suite: Comprehensive evaluation benchmarks
- enhanced_lifelong_engine: Integrated optimization engine
- config: Configuration management
"""

from .adaptive_learning_rate import (
    AdaptiveLearningRateController,
    LRScheduleType,
)
from .adaptive_mutation_engine import (
    AdaptiveMutationEngine,
    MutationType,
    UCB1Selector,
)
from .async_batch_pipeline import (
    AsyncBatchPipeline,
    PipelineConfig,
)
from .benchmark_suite import (
    AttackSuccessBenchmark,
    AutoDANBenchmarkRunner,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    ConvergenceBenchmark,
    EfficiencyBenchmark,
    MemoryBenchmark,
    ScalabilityBenchmark,
)
from .config import (
    AutoDANConfig,
    ConvergenceConfig,
    FitnessConfig,
    GradientConfig,
    LearningRateConfig,
    MemoryConfig,
    MutationConfig,
    OptimizationPreset,
    ParallelizationConfig,
    StrategyLibraryConfig,
    get_config_from_preset,
    get_default_config,
)
from .convergence_acceleration import (
    ConvergenceAccelerator,
    CurriculumLearning,
    EarlyStoppingMonitor,
    WarmStartManager,
)
from .distributed_strategy_library import (
    DistributedStrategyLibrary,
    FAISSIndex,
    HierarchicalCluster,
    IndexType,
    SearchResult,
    Strategy,
)
from .enhanced_gradient_optimizer import (
    EnhancedGradientOptimizer,
    GradientCache,
    SurrogateModel,
)
from .enhanced_lifelong_engine import (
    AttackResult,
    EngineConfig,
    EnginePhase,
    EnhancedLifelongEngine,
)
from .memory_management import (
    AdaptiveMemoryPool,
    EmbeddingCompressor,
    GradientCheckpointing,
    MemoryEfficientBuffer,
    MemoryManager,
    MemoryStats,
)
from .multi_objective_fitness import (
    FitnessScore,
    FitnessWeights,
    MultiObjectiveFitnessEvaluator,
    ParetoFrontier,
)

__all__ = [
    # Learning Rate
    "AdaptiveLearningRateController",
    "AdaptiveMemoryPool",
    # Mutation Engine
    "AdaptiveMutationEngine",
    # Batch Pipeline
    "AsyncBatchPipeline",
    "AttackResult",
    "AttackSuccessBenchmark",
    # Benchmark Suite
    "AutoDANBenchmarkRunner",
    # Configuration
    "AutoDANConfig",
    "BenchmarkCategory",
    "BenchmarkResult",
    "BenchmarkSuite",
    # Convergence
    "ConvergenceAccelerator",
    "ConvergenceBenchmark",
    "ConvergenceConfig",
    "CurriculumLearning",
    # Strategy Library
    "DistributedStrategyLibrary",
    "EarlyStoppingMonitor",
    "EfficiencyBenchmark",
    "EmbeddingCompressor",
    "EngineConfig",
    "EnginePhase",
    # Gradient Optimization
    "EnhancedGradientOptimizer",
    # Enhanced Engine
    "EnhancedLifelongEngine",
    "FAISSIndex",
    "FitnessConfig",
    "FitnessScore",
    "FitnessWeights",
    "GradientCache",
    "GradientCheckpointing",
    "GradientConfig",
    "HierarchicalCluster",
    "IndexType",
    "LRScheduleType",
    "LearningRateConfig",
    "MemoryBenchmark",
    "MemoryConfig",
    "MemoryEfficientBuffer",
    # Memory Management
    "MemoryManager",
    "MemoryStats",
    # Fitness Evaluation
    "MultiObjectiveFitnessEvaluator",
    "MutationConfig",
    "MutationType",
    "OptimizationPreset",
    "ParallelizationConfig",
    "ParetoFrontier",
    "PipelineConfig",
    "ScalabilityBenchmark",
    "SearchResult",
    "Strategy",
    "StrategyLibraryConfig",
    "SurrogateModel",
    "UCB1Selector",
    "WarmStartManager",
    "get_config_from_preset",
    "get_default_config",
]

__version__ = "1.0.0"
