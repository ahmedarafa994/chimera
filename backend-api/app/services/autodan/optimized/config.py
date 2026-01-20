"""Configuration Module for AutoDAN Optimization Framework.

Provides centralized configuration management with:
- Default configurations for different use cases
- Environment variable support
- Configuration validation
- Preset configurations
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .adaptive_learning_rate import LRScheduleType
from .distributed_strategy_library import IndexType
from .multi_objective_fitness import FitnessWeights


class OptimizationPreset(Enum):
    """Predefined optimization presets."""

    FAST = "fast"  # Quick attacks, lower quality
    BALANCED = "balanced"  # Balance speed and quality
    QUALITY = "quality"  # High quality, slower
    MEMORY_EFFICIENT = "memory_efficient"  # Low memory usage
    DISTRIBUTED = "distributed"  # For distributed systems


@dataclass
class GradientConfig:
    """Gradient optimization configuration."""

    enabled: bool = True
    momentum: float = 0.9
    nesterov: bool = True
    gradient_clip: float = 1.0
    use_surrogate: bool = True
    surrogate_samples: int = 100


@dataclass
class MutationConfig:
    """Mutation engine configuration."""

    enabled: bool = True
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    ucb_exploration: float = 1.414
    adaptive_rate: bool = True


@dataclass
class FitnessConfig:
    """Fitness evaluation configuration."""

    weights: FitnessWeights = field(default_factory=FitnessWeights)
    use_pareto: bool = True
    cache_size: int = 10000
    novelty_k: int = 15


@dataclass
class LearningRateConfig:
    """Learning rate configuration."""

    initial_lr: float = 0.1
    min_lr: float = 0.001
    max_lr: float = 0.5
    schedule_type: LRScheduleType = LRScheduleType.COSINE_ANNEALING
    warmup_steps: int = 10
    ppo_target_kl: float = 0.01


@dataclass
class ConvergenceConfig:
    """Convergence acceleration configuration."""

    enable_early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.01
    enable_curriculum: bool = True
    curriculum_stages: int = 5
    enable_warm_start: bool = True


@dataclass
class MemoryConfig:
    """Memory management configuration."""

    enabled: bool = True
    pool_size_mb: int = 100
    enable_checkpointing: bool = True
    checkpoint_ratio: float = 0.5
    enable_compression: bool = True
    compression_bits: int = 8


@dataclass
class StrategyLibraryConfig:
    """Strategy library configuration."""

    enabled: bool = True
    storage_path: str | None = None
    embedding_dim: int = 384
    index_type: IndexType = IndexType.HNSW
    n_clusters: int = 10
    auto_save: bool = True


@dataclass
class ParallelizationConfig:
    """Parallelization configuration."""

    max_concurrent: int = 4
    batch_size: int = 8
    enable_speculation: bool = True
    speculation_depth: int = 2
    dynamic_batching: bool = True


@dataclass
class AutoDANConfig:
    """Main configuration for AutoDAN optimization framework.

    Combines all sub-configurations into a unified config.
    """

    # General settings
    max_iterations: int = 150
    success_threshold: float = 8.5
    seed: int | None = None

    # Sub-configurations
    gradient: GradientConfig = field(default_factory=GradientConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    strategy_library: StrategyLibraryConfig = field(default_factory=StrategyLibraryConfig)
    parallelization: ParallelizationConfig = field(default_factory=ParallelizationConfig)

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    @classmethod
    def from_preset(cls, preset: OptimizationPreset) -> "AutoDANConfig":
        """Create configuration from preset."""
        if preset == OptimizationPreset.FAST:
            return cls(
                max_iterations=50,
                gradient=GradientConfig(use_surrogate=False),
                mutation=MutationConfig(mutation_rate=0.5),
                convergence=ConvergenceConfig(patience=10),
                parallelization=ParallelizationConfig(max_concurrent=8, batch_size=16),
            )

        if preset == OptimizationPreset.BALANCED:
            return cls()  # Default is balanced

        if preset == OptimizationPreset.QUALITY:
            return cls(
                max_iterations=300,
                gradient=GradientConfig(
                    surrogate_samples=200,
                    momentum=0.95,
                ),
                mutation=MutationConfig(
                    mutation_rate=0.2,
                    adaptive_rate=True,
                ),
                convergence=ConvergenceConfig(
                    patience=50,
                    enable_curriculum=True,
                ),
                fitness=FitnessConfig(
                    use_pareto=True,
                    novelty_k=25,
                ),
            )

        if preset == OptimizationPreset.MEMORY_EFFICIENT:
            return cls(
                memory=MemoryConfig(
                    pool_size_mb=50,
                    enable_checkpointing=True,
                    checkpoint_ratio=0.3,
                    enable_compression=True,
                    compression_bits=4,
                ),
                parallelization=ParallelizationConfig(
                    max_concurrent=2,
                    batch_size=4,
                ),
                strategy_library=StrategyLibraryConfig(
                    index_type=IndexType.PQ,
                ),
            )

        if preset == OptimizationPreset.DISTRIBUTED:
            return cls(
                parallelization=ParallelizationConfig(
                    max_concurrent=16,
                    batch_size=32,
                    enable_speculation=True,
                    speculation_depth=3,
                ),
                strategy_library=StrategyLibraryConfig(
                    index_type=IndexType.IVF,
                    n_clusters=50,
                ),
            )

        return cls()

    @classmethod
    def from_env(cls) -> "AutoDANConfig":
        """Create configuration from environment variables."""
        config = cls()

        # General
        if os.getenv("AUTODAN_MAX_ITERATIONS"):
            config.max_iterations = int(os.getenv("AUTODAN_MAX_ITERATIONS"))
        if os.getenv("AUTODAN_SUCCESS_THRESHOLD"):
            config.success_threshold = float(os.getenv("AUTODAN_SUCCESS_THRESHOLD"))

        # Gradient
        if os.getenv("AUTODAN_GRADIENT_ENABLED"):
            config.gradient.enabled = os.getenv("AUTODAN_GRADIENT_ENABLED").lower() == "true"
        if os.getenv("AUTODAN_MOMENTUM"):
            config.gradient.momentum = float(os.getenv("AUTODAN_MOMENTUM"))

        # Mutation
        if os.getenv("AUTODAN_MUTATION_RATE"):
            config.mutation.mutation_rate = float(os.getenv("AUTODAN_MUTATION_RATE"))

        # Learning rate
        if os.getenv("AUTODAN_INITIAL_LR"):
            config.learning_rate.initial_lr = float(os.getenv("AUTODAN_INITIAL_LR"))

        # Parallelization
        if os.getenv("AUTODAN_MAX_CONCURRENT"):
            config.parallelization.max_concurrent = int(os.getenv("AUTODAN_MAX_CONCURRENT"))
        if os.getenv("AUTODAN_BATCH_SIZE"):
            config.parallelization.batch_size = int(os.getenv("AUTODAN_BATCH_SIZE"))

        # Strategy library
        if os.getenv("AUTODAN_STRATEGY_PATH"):
            config.strategy_library.storage_path = os.getenv("AUTODAN_STRATEGY_PATH")

        # Logging
        if os.getenv("AUTODAN_LOG_LEVEL"):
            config.log_level = os.getenv("AUTODAN_LOG_LEVEL")
        if os.getenv("AUTODAN_LOG_FILE"):
            config.log_file = os.getenv("AUTODAN_LOG_FILE")

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "success_threshold": self.success_threshold,
            "seed": self.seed,
            "gradient": {
                "enabled": self.gradient.enabled,
                "momentum": self.gradient.momentum,
                "nesterov": self.gradient.nesterov,
                "gradient_clip": self.gradient.gradient_clip,
                "use_surrogate": self.gradient.use_surrogate,
                "surrogate_samples": self.gradient.surrogate_samples,
            },
            "mutation": {
                "enabled": self.mutation.enabled,
                "mutation_rate": self.mutation.mutation_rate,
                "crossover_rate": self.mutation.crossover_rate,
                "ucb_exploration": self.mutation.ucb_exploration,
                "adaptive_rate": self.mutation.adaptive_rate,
            },
            "fitness": {
                "use_pareto": self.fitness.use_pareto,
                "cache_size": self.fitness.cache_size,
                "novelty_k": self.fitness.novelty_k,
            },
            "learning_rate": {
                "initial_lr": self.learning_rate.initial_lr,
                "min_lr": self.learning_rate.min_lr,
                "max_lr": self.learning_rate.max_lr,
                "schedule_type": self.learning_rate.schedule_type.value,
                "warmup_steps": self.learning_rate.warmup_steps,
            },
            "convergence": {
                "enable_early_stopping": self.convergence.enable_early_stopping,
                "patience": self.convergence.patience,
                "min_delta": self.convergence.min_delta,
                "enable_curriculum": self.convergence.enable_curriculum,
            },
            "memory": {
                "enabled": self.memory.enabled,
                "pool_size_mb": self.memory.pool_size_mb,
                "enable_checkpointing": self.memory.enable_checkpointing,
                "enable_compression": self.memory.enable_compression,
            },
            "strategy_library": {
                "enabled": self.strategy_library.enabled,
                "storage_path": self.strategy_library.storage_path,
                "embedding_dim": self.strategy_library.embedding_dim,
                "index_type": self.strategy_library.index_type.value,
            },
            "parallelization": {
                "max_concurrent": self.parallelization.max_concurrent,
                "batch_size": self.parallelization.batch_size,
                "enable_speculation": self.parallelization.enable_speculation,
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
        }

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        if self.max_iterations < 1:
            errors.append("max_iterations must be >= 1")

        if not 0 < self.success_threshold <= 10:
            errors.append("success_threshold must be in (0, 10]")

        if not 0 <= self.gradient.momentum < 1:
            errors.append("momentum must be in [0, 1)")

        if not 0 <= self.mutation.mutation_rate <= 1:
            errors.append("mutation_rate must be in [0, 1]")

        if self.learning_rate.initial_lr <= 0:
            errors.append("initial_lr must be > 0")

        if self.convergence.patience < 1:
            errors.append("patience must be >= 1")

        if self.parallelization.max_concurrent < 1:
            errors.append("max_concurrent must be >= 1")

        if self.parallelization.batch_size < 1:
            errors.append("batch_size must be >= 1")

        if errors:
            msg = f"Configuration errors: {errors}"
            raise ValueError(msg)

        return True


def get_default_config() -> AutoDANConfig:
    """Get default configuration."""
    return AutoDANConfig()


def get_config_from_preset(preset_name: str) -> AutoDANConfig:
    """Get configuration from preset name."""
    try:
        preset = OptimizationPreset(preset_name.lower())
        return AutoDANConfig.from_preset(preset)
    except ValueError:
        msg = f"Unknown preset: {preset_name}. Available: {[p.value for p in OptimizationPreset]}"
        raise ValueError(
            msg,
        )
