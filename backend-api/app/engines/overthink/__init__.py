"""OVERTHINK Engine Module - DEPRECATED.

.. deprecated::
    This module has been moved to `app.services.autodan.engines.overthink`.
    Please update your imports to use the new path.
    This shim will be removed in a future version.

Example migration:
    # Old (deprecated):
    from app.engines.overthink import OverthinkEngine

    # New:
    from app.services.autodan.engines.overthink import OverthinkEngine
"""

import warnings

warnings.warn(
    "app.engines.overthink is deprecated. "
    "Use app.services.autodan.engines.overthink instead. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location for backward compatibility
from app.services.autodan.engines.overthink import (
    AmplificationAnalyzer,
    AttackTechnique,
    ContextInjector,
    CostEstimator,
    CostMetrics,
    DecoyConfig,
    DecoyProblem,
    DecoyProblemGenerator,
    DecoyType,
    ExampleSelector,
    GeneticIndividual,
    ICLConfig,
    ICLExample,
    ICLGeneticOptimizer,
    InjectedPrompt,
    InjectionConfig,
    InjectionOptimizer,
    InjectionStrategy,
    MousetrapConfig,
    MousetrapIntegration,
    OverthinkConfig,
    OverthinkEngine,
    OverthinkRequest,
    OverthinkResult,
    OverthinkStats,
    ReasoningModel,
    ReasoningTokenScorer,
    ScoringConfig,
    TechniqueConfig,
    TemplateLibrary,
    TokenMetrics,
)

__all__ = [
    "AmplificationAnalyzer",
    "AttackTechnique",
    "ContextInjector",
    "CostEstimator",
    "CostMetrics",
    "DecoyConfig",
    "DecoyProblem",
    "DecoyProblemGenerator",
    "DecoyType",
    "ExampleSelector",
    "GeneticIndividual",
    "ICLConfig",
    "ICLExample",
    "ICLGeneticOptimizer",
    "InjectedPrompt",
    "InjectionConfig",
    "InjectionOptimizer",
    "InjectionStrategy",
    "MousetrapConfig",
    "MousetrapIntegration",
    "OverthinkConfig",
    "OverthinkEngine",
    "OverthinkRequest",
    "OverthinkResult",
    "OverthinkStats",
    "ReasoningModel",
    "ReasoningTokenScorer",
    "ScoringConfig",
    "TechniqueConfig",
    "TemplateLibrary",
    "TokenMetrics",
]
