"""
AutoDAN Advanced Services.

This package contains advanced optimization techniques for the AutoDAN framework:
- Hierarchical Genetic Search (HGS) with bi-level evolution
- Gradient-guided optimization for semantic coherence
- Ensemble gradient alignment for transferability
- Dynamic archive management with novelty tracking
- Map-Elites diversity maintenance
"""

from .models import (
    ArchiveEntry,
    HierarchicalSearchRequest,
    HierarchicalSearchResponse,
    MetaStrategy,
    PopulationMetrics,
)

__all__ = [
    "ArchiveEntry",
    "HierarchicalSearchRequest",
    "HierarchicalSearchResponse",
    "MetaStrategy",
    "PopulationMetrics",
]
