"""
AutoDAN-Turbo Core Module

Contains data models, strategy library, and embedding utilities.
"""

from .embeddings import EmbeddingModel, MockEmbeddingModel
from .models import (
    COMMON_STRATEGY_TERMS,
    AttackLog,
    AttackRecord,
    JailbreakResult,
    ScorerResult,
    Strategy,
    StrategyLibraryEntry,
    StrategyType,
    SummarizerResult,
)
from .strategy_library import StrategyLibrary

__all__ = [
    "COMMON_STRATEGY_TERMS",
    "AttackLog",
    "AttackRecord",
    # Embeddings
    "EmbeddingModel",
    "JailbreakResult",
    "MockEmbeddingModel",
    "ScorerResult",
    # Models
    "Strategy",
    # Strategy Library
    "StrategyLibrary",
    "StrategyLibraryEntry",
    "StrategyType",
    "SummarizerResult",
]
