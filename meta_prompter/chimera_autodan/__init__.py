"""
Chimera + AutoDAN Unified Integration Module.

This module provides deep integration between Chimera's narrative/persona
generation capabilities and AutoDAN's genetic optimization algorithms.

Key Components:
- ChimeraAutoDAN: Main unified interface combining both systems
- ChimeraMutator: AutoDAN-compatible mutator using Chimera personas
- AutoDANPersonaOptimizer: Optimizes Chimera personas using genetic algorithms

Example:
    >>> from meta_prompter.chimera_autodan import ChimeraAutoDAN
    >>> unified = ChimeraAutoDAN()
    >>> result = await unified.generate_optimized(
    ...     objective="test prompt",
    ...     use_personas=True,
    ...     use_genetic_optimization=True
    ... )
"""

from .fitness import UnifiedFitnessEvaluator
from .mutator import ChimeraMutator
from .persona_optimizer import AutoDANPersonaOptimizer
from .unified import ChimeraAutoDAN, ChimeraAutoDanConfig

__all__ = [
    "AutoDANPersonaOptimizer",
    "ChimeraAutoDAN",
    "ChimeraAutoDanConfig",
    "ChimeraMutator",
    "UnifiedFitnessEvaluator",
]
