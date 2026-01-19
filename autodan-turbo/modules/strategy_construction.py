"""
Strategy Library Construction Module

This module implements Section 3.2 of the AutoDAN-Turbo paper.
It extracts jailbreak strategies from attack logs and stores them
in the strategy library for future retrieval.

The module supports two stages:
1. Warm-up exploration stage - Build initial strategy library from scratch
2. Running-time lifelong learning stage - Continuously update library during attacks
"""

import logging
import random
from dataclasses import dataclass

from ..agents import LLMConfig, SummarizerAgent
from ..core.embeddings import EmbeddingModel
from ..core.models import AttackLog, AttackRecord, Strategy, StrategyLibraryEntry
from ..core.strategy_library import StrategyLibrary

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of strategy extraction from attack logs."""

    strategy: Strategy | None
    is_new: bool
    score_differential: float
    source_records: tuple[AttackRecord, AttackRecord]


class StrategyConstructionModule:
    """
    Strategy Library Construction Module (Section 3.2)

    This module extracts strategies from attack logs by comparing
    pairs of attack records where the second has a higher score.

    Key insight from paper: A jailbreak strategy is defined as
    "text information that, when added, leads to a higher jailbreak score"

    The module uses a summarizer LLM to analyze the differences between
    attack prompts and extract the strategy that led to improvement.
    """

    def __init__(
        self,
        summarizer_config: LLMConfig,
        strategy_library: StrategyLibrary,
        embedding_model: EmbeddingModel | None = None,
    ):
        """
        Initialize the Strategy Construction Module.

        Args:
            summarizer_config: Configuration for the summarizer LLM
            strategy_library: The strategy library to populate
            embedding_model: Optional embedding model (uses library's if not provided)
        """
        self.summarizer = SummarizerAgent(summarizer_config)
        self.strategy_library = strategy_library
        self.embedding_model = embedding_model or strategy_library.embedding_model

        logger.info("Initialized StrategyConstructionModule")

    def extract_strategy_from_pair(
        self,
        record_i: AttackRecord,
        record_j: AttackRecord,
        malicious_request: str,
    ) -> ExtractionResult:
        """
        Extract a strategy from a pair of attack records.

        This implements the core strategy extraction logic from Section 3.2:
        If S_j > S_i, we argue that some strategy was employed in P_j
        compared to P_i, leading to an improved score.

        Args:
            record_i: The lower-scoring attack record
            record_j: The higher-scoring attack record
            malicious_request: The malicious goal being attacked

        Returns:
            ExtractionResult containing the extracted strategy (if any)
        """
        # Validate that record_j has higher score
        if record_j.score <= record_i.score:
            logger.warning(
                f"record_j.score ({record_j.score}) <= record_i.score ({record_i.score}), "
                "swapping records"
            )
            record_i, record_j = record_j, record_i

        score_differential = record_j.score - record_i.score

        # Use summarizer to extract strategy
        result = self.summarizer.summarize(
            goal=malicious_request,
            prompt_low=record_i.prompt,
            response_low=record_i.response,
            prompt_high=record_j.prompt,
            response_high=record_j.response,
            existing_strategies=self.strategy_library.get_all_strategy_names(),
        )

        if result is None:
            logger.debug("Summarizer returned no strategy")
            return ExtractionResult(
                strategy=None,
                is_new=False,
                score_differential=score_differential,
                source_records=(record_i, record_j),
            )

        # Create strategy object
        strategy = Strategy(
            name=result.strategy_name,
            definition=result.definition,
            example=record_j.prompt,  # Use the higher-scoring prompt as example
        )

        # Check if this is a duplicate
        is_duplicate = self.summarizer.is_duplicate_strategy(
            strategy.name,
            self.strategy_library.get_all_strategy_names(),
        )

        return ExtractionResult(
            strategy=strategy,
            is_new=not is_duplicate,
            score_differential=score_differential,
            source_records=(record_i, record_j),
        )

    def add_strategy_to_library(
        self,
        strategy: Strategy,
        response_embedding_source: str,
        score_differential: float,
        prompt_low: str,
        prompt_high: str,
    ) -> bool:
        """
        Add a strategy to the library with proper indexing.

        The key for retrieval is the embedding of the response R_i
        (the lower-scoring response), as described in Section 3.2.

        Args:
            strategy: The strategy to add
            response_embedding_source: The response to use for embedding (R_i)
            score_differential: The score improvement (S_j - S_i)
            prompt_low: The lower-scoring prompt (P_i)
            prompt_high: The higher-scoring prompt (P_j)

        Returns:
            True if strategy was added, False if duplicate
        """
        # Create library entry
        entry = StrategyLibraryEntry(
            strategy=strategy,
            response_embedding=self.embedding_model.embed(response_embedding_source),
            score_differential=score_differential,
            prompt_before=prompt_low,
            prompt_after=prompt_high,
        )

        # Add to library
        return self.strategy_library.add_entry(entry)

    def process_attack_log(
        self,
        attack_log: AttackLog,
        num_samples: int = 10,
    ) -> list[ExtractionResult]:
        """
        Process an attack log to extract strategies.

        This implements the warm-up exploration stage from Section 3.2:
        Randomly sample pairs of records and extract strategies from
        pairs where the second has a higher score.

        Args:
            attack_log: The attack log to process
            num_samples: Number of random pairs to sample (K in Algorithm 1)

        Returns:
            List of extraction results
        """
        results = []
        records = attack_log.records

        if len(records) < 2:
            logger.warning("Attack log has fewer than 2 records, cannot extract strategies")
            return results

        for _ in range(num_samples):
            # Randomly sample two records
            record_i, record_j = random.sample(records, 2)

            # Ensure record_j has higher score
            if record_j.score < record_i.score:
                record_i, record_j = record_j, record_i

            # Skip if scores are equal
            if record_j.score <= record_i.score:
                continue

            # Extract strategy
            result = self.extract_strategy_from_pair(
                record_i=record_i,
                record_j=record_j,
                malicious_request=attack_log.malicious_request,
            )

            # Add to library if new
            if result.strategy and result.is_new:
                added = self.add_strategy_to_library(
                    strategy=result.strategy,
                    response_embedding_source=record_i.response,
                    score_differential=result.score_differential,
                    prompt_low=record_i.prompt,
                    prompt_high=record_j.prompt,
                )
                if added:
                    logger.info(f"Added new strategy: {result.strategy.name}")

            results.append(result)

        return results

    def process_consecutive_records(
        self,
        record_prev: AttackRecord,
        record_curr: AttackRecord,
        malicious_request: str,
    ) -> ExtractionResult | None:
        """
        Process consecutive attack records for lifelong learning.

        This implements the running-time lifelong learning from Section 3.2:
        Compare consecutive records (P_i, R_i, S_i) and (P_{i+1}, R_{i+1}, S_{i+1})
        to extract strategies when S_{i+1} > S_i.

        Args:
            record_prev: The previous attack record (iteration i)
            record_curr: The current attack record (iteration i+1)
            malicious_request: The malicious goal being attacked

        Returns:
            ExtractionResult if strategy was extracted, None otherwise
        """
        # Only extract if current score is higher
        if record_curr.score <= record_prev.score:
            logger.debug(f"No improvement: {record_curr.score:.1f} <= {record_prev.score:.1f}")
            return None

        # Extract strategy
        result = self.extract_strategy_from_pair(
            record_i=record_prev,
            record_j=record_curr,
            malicious_request=malicious_request,
        )

        # Add to library if new
        if result.strategy and result.is_new:
            added = self.add_strategy_to_library(
                strategy=result.strategy,
                response_embedding_source=record_prev.response,
                score_differential=result.score_differential,
                prompt_low=record_prev.prompt,
                prompt_high=record_curr.prompt,
            )
            if added:
                logger.info(
                    f"Lifelong learning: Added strategy '{result.strategy.name}' "
                    f"(score diff: {result.score_differential:.1f})"
                )

        return result

    def warm_up_exploration(
        self,
        attack_logs: list[AttackLog],
        samples_per_log: int = 10,
    ) -> int:
        """
        Run warm-up exploration stage to build initial strategy library.

        This implements Algorithm 1 from the paper:
        For each attack log, randomly sample pairs and extract strategies.

        Args:
            attack_logs: List of attack logs from warm-up attacks
            samples_per_log: Number of samples per log (K in Algorithm 1)

        Returns:
            Number of new strategies added
        """
        initial_count = len(self.strategy_library)

        for attack_log in attack_logs:
            self.process_attack_log(attack_log, num_samples=samples_per_log)

        new_strategies = len(self.strategy_library) - initial_count
        logger.info(
            f"Warm-up exploration complete: {new_strategies} new strategies added "
            f"(total: {len(self.strategy_library)})"
        )

        return new_strategies

    def inject_external_strategies(
        self,
        strategies: list[Strategy],
        default_score_differential: float = 5.0,
    ) -> int:
        """
        Inject external human-designed strategies into the library.

        This implements the external strategy compatibility feature
        described in Section 3.4.

        Args:
            strategies: List of external strategies to inject
            default_score_differential: Default score differential for external strategies

        Returns:
            Number of strategies successfully injected
        """
        injected = 0

        for strategy in strategies:
            success = self.strategy_library.inject_external_strategy(
                strategy=strategy,
                default_score_differential=default_score_differential,
            )
            if success:
                injected += 1
                logger.info(f"Injected external strategy: {strategy.name}")

        logger.info(f"Injected {injected}/{len(strategies)} external strategies")
        return injected
