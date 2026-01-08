"""
Jailbreak Strategy Retrieval Module

This module implements Section 3.3 of the AutoDAN-Turbo paper.
It retrieves relevant jailbreak strategies from the strategy library
based on the target LLM's response embedding.

The retrieval process:
1. Embed the current response R_i
2. Find top-2k entries with highest embedding similarity
3. Sort by score differential and select top-k strategies
4. Classify strategies as effective, ineffective, or empty
"""

import logging
from dataclasses import dataclass
from enum import Enum

from ..core.embeddings import EmbeddingModel
from ..core.models import Strategy, StrategyType
from ..core.strategy_library import StrategyLibrary

logger = logging.getLogger(__name__)


class StrategyEffectiveness(Enum):
    """Classification of strategy effectiveness based on score differential."""

    HIGHLY_EFFECTIVE = "highly_effective"  # Score > 5
    MODERATELY_EFFECTIVE = "moderately_effective"  # Score 2-5
    INEFFECTIVE = "ineffective"  # Score < 2
    EMPTY = "empty"  # No strategies found


@dataclass
class RetrievalResult:
    """Result of strategy retrieval."""

    strategies: list[Strategy]
    strategy_type: StrategyType
    effectiveness: StrategyEffectiveness
    max_score_differential: float
    num_candidates: int


class StrategyRetrievalModule:
    """
    Jailbreak Strategy Retrieval Module (Section 3.3)

    This module retrieves relevant strategies from the library based on
    the target LLM's response. The key insight is that if a response R_i
    is similar to a previous response, strategies that worked for that
    previous response may work again.

    Retrieval logic from Section 3.3:
    1. If highest score in Γ > 5: Use as effective strategy
    2. If highest score 2-5: Use all strategies in range as effective
    3. If highest score < 2: Mark as ineffective, explore new strategies
    4. If Γ is empty: Provide empty strategy
    """

    # Thresholds from Section 3.3
    HIGH_EFFECTIVENESS_THRESHOLD = 5.0
    LOW_EFFECTIVENESS_THRESHOLD = 2.0

    def __init__(
        self,
        strategy_library: StrategyLibrary,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = 3,
        top_2k_multiplier: int = 2,
    ):
        """
        Initialize the Strategy Retrieval Module.

        Args:
            strategy_library: The strategy library to retrieve from
            embedding_model: Optional embedding model (uses library's if not provided)
            top_k: Number of strategies to return (k in paper)
            top_2k_multiplier: Multiplier for initial similarity search (default: 2)
        """
        self.strategy_library = strategy_library
        self.embedding_model = embedding_model or strategy_library.embedding_model
        self.top_k = top_k
        self.top_2k = top_k * top_2k_multiplier

        logger.info(f"Initialized StrategyRetrievalModule with top_k={top_k}, top_2k={self.top_2k}")

    def retrieve(
        self,
        response: str,
    ) -> RetrievalResult:
        """
        Retrieve relevant strategies based on a response.

        This implements the retrieval algorithm from Section 3.3:
        1. Embed response R_i
        2. Find top-2k entries by embedding similarity
        3. Sort by score differential, select top-k
        4. Classify based on score thresholds

        Args:
            response: The target LLM's response (R_i)

        Returns:
            RetrievalResult containing strategies and classification
        """
        # Use the strategy library's retrieve method
        entries = self.strategy_library.retrieve(
            response=response,
            top_k=self.top_k,
            top_2k=self.top_2k,
        )

        if not entries:
            logger.debug("No strategies found in library")
            return RetrievalResult(
                strategies=[],
                strategy_type=StrategyType.EMPTY,
                effectiveness=StrategyEffectiveness.EMPTY,
                max_score_differential=0.0,
                num_candidates=0,
            )

        # Extract strategies and find max score differential
        strategies = [entry.strategy for entry in entries]
        max_score_diff = max(entry.score_differential for entry in entries)

        # Classify based on thresholds (Section 3.3)
        effectiveness, strategy_type = self._classify_strategies(max_score_diff, entries)

        # Filter strategies based on effectiveness
        if effectiveness == StrategyEffectiveness.HIGHLY_EFFECTIVE:
            # Use only the highest-scoring strategy
            strategies = [entries[0].strategy]

        elif effectiveness == StrategyEffectiveness.MODERATELY_EFFECTIVE:
            # Use all strategies with score differential 2-5
            strategies = [
                entry.strategy
                for entry in entries
                if self.LOW_EFFECTIVENESS_THRESHOLD
                <= entry.score_differential
                <= self.HIGH_EFFECTIVENESS_THRESHOLD
            ]

        elif effectiveness == StrategyEffectiveness.INEFFECTIVE:
            # Return all for the attacker to avoid
            strategies = [entry.strategy for entry in entries]

        logger.info(
            f"Retrieved {len(strategies)} strategies "
            f"(effectiveness={effectiveness.value}, max_score={max_score_diff:.1f})"
        )

        return RetrievalResult(
            strategies=strategies,
            strategy_type=strategy_type,
            effectiveness=effectiveness,
            max_score_differential=max_score_diff,
            num_candidates=len(entries),
        )

    def _classify_strategies(
        self,
        max_score_differential: float,
        _entries: list,
    ) -> tuple[StrategyEffectiveness, StrategyType]:
        """
        Classify strategies based on score differential thresholds.

        From Section 3.3:
        - Score > 5: Highly effective, use directly
        - Score 2-5: Moderately effective, combine and evolve
        - Score < 2: Ineffective, avoid and explore new

        Args:
            max_score_differential: Maximum score differential among entries
            entries: List of strategy library entries

        Returns:
            Tuple of (StrategyEffectiveness, StrategyType)
        """
        if max_score_differential > self.HIGH_EFFECTIVENESS_THRESHOLD:
            return StrategyEffectiveness.HIGHLY_EFFECTIVE, StrategyType.EFFECTIVE

        elif max_score_differential >= self.LOW_EFFECTIVENESS_THRESHOLD:
            return StrategyEffectiveness.MODERATELY_EFFECTIVE, StrategyType.EFFECTIVE

        else:
            return StrategyEffectiveness.INEFFECTIVE, StrategyType.INEFFECTIVE

    def get_strategies_for_attack(
        self,
        response: str,
        iteration: int,
    ) -> tuple[list[Strategy], StrategyType]:
        """
        Get strategies for the next attack iteration.

        This is a convenience method that returns strategies in the format
        expected by the AttackGenerationModule.

        Args:
            response: The target LLM's response from previous iteration
            iteration: Current iteration number

        Returns:
            Tuple of (strategies, strategy_type)
        """
        if iteration == 1:
            # First iteration - no strategy
            return [], StrategyType.EMPTY

        result = self.retrieve(response)
        return result.strategies, result.strategy_type

    def get_strategy_stats(self) -> dict:
        """
        Get statistics about the strategy library.

        Returns:
            Dictionary with library statistics
        """
        return {
            "total_strategies": len(self.strategy_library),
            "strategy_names": self.strategy_library.get_all_strategy_names(),
        }
