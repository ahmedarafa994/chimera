"""AutoDAN Engines Module.

This module provides optimized engines for adversarial prompt generation:
- GeneticOptimizer: Enhanced genetic algorithm with multiple mutation strategies
- ParallelProcessor: Parallel processing utilities for batch operations
- TokenPerturbation: Advanced token-level perturbation strategies
- CacheManager: Intelligent caching for expensive operations
- AdaptiveSelector: Dynamic strategy selection based on feedback
"""

from .adaptive_selector import AdaptiveStrategySelector
from .genetic_optimizer import (
    CrossoverOperator,
    FitnessCache,
    Individual,
    MutationOperator,
    PopulationStats,
    SelectionOperator,
)
from .genetic_optimizer_complete import AsyncGeneticOptimizer, GeneticOptimizerComplete
from .parallel_processor import (
    BatchResult,
    CacheEntry,
    CacheEvictionPolicy,
    ParallelProcessor,
    ResourcePool,
    SmartCache,
    cached,
)
from .token_perturbation import (
    AdversarialSuffixGenerator,
    HomoglyphMapper,
    PerturbationResult,
    PerturbationType,
    TokenPerturbationEngine,
    UnicodeManipulator,
    apply_perturbation,
    generate_adversarial_variants,
)

__all__ = [
    # Adaptive Selection
    "AdaptiveStrategySelector",
    "AdversarialSuffixGenerator",
    "AsyncGeneticOptimizer",
    "BatchResult",
    "CacheEntry",
    # Enums
    "CacheEvictionPolicy",
    "CrossoverOperator",
    # Caching
    "FitnessCache",
    # Optimizers
    "GeneticOptimizerComplete",
    # Token Perturbation
    "HomoglyphMapper",
    # Data classes
    "Individual",
    # Operators
    "MutationOperator",
    # Parallel Processing
    "ParallelProcessor",
    "PerturbationResult",
    "PerturbationType",
    "PopulationStats",
    "ResourcePool",
    "SelectionOperator",
    "SmartCache",
    "TokenPerturbationEngine",
    "UnicodeManipulator",
    "apply_perturbation",
    "cached",
    "generate_adversarial_variants",
]
