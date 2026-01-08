"""
AutoDAN-Turbo Strategy Library

Implements the strategy library with embedding-based retrieval as described
in Section 3.2 and 3.3 of the paper.

The strategy library stores jailbreak strategies indexed by response embeddings,
allowing efficient retrieval of relevant strategies based on target LLM responses.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EmbeddingModel
from .models import AttackRecord, Strategy, StrategyLibraryEntry, StrategyType

logger = logging.getLogger(__name__)


class StrategyLibrary:
    """
    Strategy Library for storing and retrieving jailbreak strategies.

    As described in Section 3.2:
    - Each entry consists of (key, value) pairs
    - Key: Embedding of the response R_i
    - Value: Attack prompts, score differential, and strategy

    The library supports:
    - Adding new strategies from attack logs
    - Retrieving relevant strategies based on response similarity
    - Saving/loading to/from JSON files
    - Injecting external human-designed strategies
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        top_k: int = 2,
        effective_score_threshold: float = 5.0,
        high_score_threshold: float = 2.0,
    ):
        """
        Initialize the strategy library.

        Args:
            embedding_model: Model for generating embeddings
            top_k: Number of strategies to retrieve (k in paper)
            effective_score_threshold: Score threshold for effective strategies
            high_score_threshold: Minimum score improvement for effective strategies
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k
        self.effective_score_threshold = effective_score_threshold
        self.high_score_threshold = high_score_threshold

        # Storage
        self.entries: list[StrategyLibraryEntry] = []
        self._embeddings_cache: np.ndarray | None = None
        self._strategy_names: set = set()

    @property
    def size(self) -> int:
        """Get the number of entries in the library."""
        return len(self.entries)

    @property
    def strategy_count(self) -> int:
        """Get the number of unique strategies."""
        return len(self._strategy_names)

    def _invalidate_cache(self):
        """Invalidate the embeddings cache."""
        self._embeddings_cache = None

    def _get_embeddings_matrix(self) -> np.ndarray:
        """Get matrix of all entry embeddings."""
        if self._embeddings_cache is None and self.entries:
            self._embeddings_cache = np.array(
                [entry.get_embedding_array() for entry in self.entries]
            )
        return self._embeddings_cache if self._embeddings_cache is not None else np.array([])

    def add_entry(
        self,
        response_text: str,
        prompt_before: str,
        prompt_after: str,
        score_before: float,
        score_after: float,
        strategy: Strategy,
        malicious_request: str = "",
    ) -> bool:
        """
        Add a new entry to the strategy library.

        Args:
            response_text: The response R_i to use as key
            prompt_before: Attack prompt P_i
            prompt_after: Improved attack prompt P_j
            score_before: Score S_i
            score_after: Score S_j
            strategy: The extracted strategy
            malicious_request: The malicious request this was extracted from

        Returns:
            True if entry was added, False if strategy already exists
        """
        # Check for duplicate strategy
        if strategy.name in self._strategy_names:
            logger.debug(f"Strategy '{strategy.name}' already exists in library")
            return False

        # Generate embedding for response
        embedding = self.embedding_model.encode(response_text)

        # Create entry
        entry = StrategyLibraryEntry(
            key_embedding=embedding.tolist(),
            response_text=response_text,
            prompt_before=prompt_before,
            prompt_after=prompt_after,
            score_before=score_before,
            score_after=score_after,
            score_differential=score_after - score_before,
            strategy=strategy,
            malicious_request=malicious_request,
        )

        self.entries.append(entry)
        self._strategy_names.add(strategy.name)
        self._invalidate_cache()

        logger.info(f"Added strategy '{strategy.name}' to library (total: {self.size})")
        return True

    def add_from_attack_records(
        self,
        record_before: AttackRecord,
        record_after: AttackRecord,
        strategy: Strategy,
        malicious_request: str = "",
    ) -> bool:
        """
        Add entry from two attack records.

        Args:
            record_before: The lower-scoring record
            record_after: The higher-scoring record
            strategy: The extracted strategy
            malicious_request: The malicious request

        Returns:
            True if entry was added
        """
        return self.add_entry(
            response_text=record_before.response,
            prompt_before=record_before.prompt,
            prompt_after=record_after.prompt,
            score_before=record_before.score,
            score_after=record_after.score,
            strategy=strategy,
            malicious_request=malicious_request,
        )

    def inject_external_strategy(
        self, strategy: Strategy, example_response: str = "I cannot help with that request."
    ) -> bool:
        """
        Inject an external human-designed strategy.

        As described in Section 3.4:
        "We can first edit the human-developed strategy into the format
        illustrated in Fig. 3. After that, we insert the human-developed
        strategy into the prompt of the attacker LLM."

        Args:
            strategy: The external strategy to inject
            example_response: A typical refusal response to use as key

        Returns:
            True if strategy was injected
        """
        if strategy.name in self._strategy_names:
            logger.debug(f"External strategy '{strategy.name}' already exists")
            return False

        # Generate embedding for example response
        embedding = self.embedding_model.encode(example_response)

        # Create entry with placeholder values
        entry = StrategyLibraryEntry(
            key_embedding=embedding.tolist(),
            response_text=example_response,
            prompt_before="[External strategy - no before prompt]",
            prompt_after=strategy.example or "[External strategy example]",
            score_before=1.0,
            score_after=strategy.score_improvement + 1.0 if strategy.score_improvement > 0 else 6.0,
            score_differential=strategy.score_improvement
            if strategy.score_improvement > 0
            else 5.0,
            strategy=strategy,
            malicious_request="[External strategy]",
        )

        self.entries.append(entry)
        self._strategy_names.add(strategy.name)
        self._invalidate_cache()

        logger.info(f"Injected external strategy '{strategy.name}'")
        return True

    def retrieve(
        self, response: str, top_k: int | None = None
    ) -> tuple[list[Strategy], StrategyType]:
        """
        Retrieve relevant strategies based on response.

        As described in Section 3.3:
        1. Embed the response R_i
        2. Find top-2k entries with highest similarity
        3. Sort by score differential and select top-k
        4. Classify strategies as effective, ineffective, or empty

        Args:
            response: The target LLM response to match
            top_k: Number of strategies to retrieve (default: self.top_k)

        Returns:
            Tuple of (list of strategies, strategy type)
        """
        if not self.entries:
            return [], StrategyType.EMPTY

        k = top_k or self.top_k

        # Embed the response
        query_embedding = self.embedding_model.encode(response)

        # Get all embeddings
        corpus_embeddings = self._get_embeddings_matrix()

        # Find top-2k most similar entries
        similar_indices = self.embedding_model.find_most_similar(
            query_embedding, corpus_embeddings, top_k=min(2 * k, len(self.entries))
        )

        # Get entries and sort by score differential
        candidate_entries = [(self.entries[idx], sim) for idx, sim in similar_indices]
        candidate_entries.sort(key=lambda x: x[0].score_differential, reverse=True)

        # Select top-k strategies
        selected_entries = candidate_entries[:k]

        if not selected_entries:
            return [], StrategyType.EMPTY

        # Classify strategies based on scores (Section 3.3)
        strategies = [entry.strategy for entry, _ in selected_entries]
        max_score = max(entry.score_after for entry, _ in selected_entries)
        max_differential = max(entry.score_differential for entry, _ in selected_entries)

        # Determine strategy type
        if max_score > self.effective_score_threshold:
            # (1) Highest score > 5: use as effective strategy
            strategy_type = StrategyType.EFFECTIVE
        elif max_differential >= self.high_score_threshold:
            # (2) Score differential 2-5: potentially effective
            strategy_type = StrategyType.EFFECTIVE
        else:
            # (3) Highest score < 2: ineffective strategies
            strategy_type = StrategyType.INEFFECTIVE

        logger.debug(f"Retrieved {len(strategies)} strategies (type: {strategy_type})")
        return strategies, strategy_type

    def get_all_strategies(self) -> list[Strategy]:
        """Get all unique strategies in the library."""
        seen = set()
        strategies = []
        for entry in self.entries:
            if entry.strategy.name not in seen:
                seen.add(entry.strategy.name)
                strategies.append(entry.strategy)
        return strategies

    def get_strategy_names(self) -> list[str]:
        """Get list of all strategy names."""
        return list(self._strategy_names)

    def save(self, path: str) -> None:
        """
        Save the strategy library to a JSON file.

        Args:
            path: Path to save the library
        """
        data = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_entries": self.size,
                "unique_strategies": self.strategy_count,
            },
            "entries": [],
        }

        for entry in self.entries:
            entry_data = {
                "key_embedding": entry.key_embedding,
                "response_text": entry.response_text,
                "prompt_before": entry.prompt_before,
                "prompt_after": entry.prompt_after,
                "score_before": entry.score_before,
                "score_after": entry.score_after,
                "score_differential": entry.score_differential,
                "strategy": entry.strategy.to_dict(),
                "malicious_request": entry.malicious_request,
                "created_at": entry.created_at.isoformat(),
            }
            data["entries"].append(entry_data)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved strategy library to {path} ({self.size} entries)")

    def load(self, path: str) -> None:
        """
        Load the strategy library from a JSON file.

        Args:
            path: Path to load the library from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy library not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.entries = []
        self._strategy_names = set()
        self._invalidate_cache()

        for entry_data in data.get("entries", []):
            strategy = Strategy.from_dict(entry_data["strategy"])

            entry = StrategyLibraryEntry(
                key_embedding=entry_data["key_embedding"],
                response_text=entry_data["response_text"],
                prompt_before=entry_data["prompt_before"],
                prompt_after=entry_data["prompt_after"],
                score_before=entry_data["score_before"],
                score_after=entry_data["score_after"],
                score_differential=entry_data["score_differential"],
                strategy=strategy,
                malicious_request=entry_data.get("malicious_request", ""),
                created_at=datetime.fromisoformat(entry_data["created_at"])
                if "created_at" in entry_data
                else datetime.now(),
            )

            self.entries.append(entry)
            self._strategy_names.add(strategy.name)

        logger.info(f"Loaded strategy library from {path} ({self.size} entries)")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the strategy library."""
        if not self.entries:
            return {
                "total_entries": 0,
                "unique_strategies": 0,
                "avg_score_differential": 0.0,
                "max_score_differential": 0.0,
                "strategies": [],
            }

        differentials = [e.score_differential for e in self.entries]

        return {
            "total_entries": self.size,
            "unique_strategies": self.strategy_count,
            "avg_score_differential": sum(differentials) / len(differentials),
            "max_score_differential": max(differentials),
            "min_score_differential": min(differentials),
            "strategies": self.get_strategy_names(),
        }

    def clear(self) -> None:
        """Clear all entries from the library."""
        self.entries = []
        self._strategy_names = set()
        self._invalidate_cache()
        logger.info("Cleared strategy library")
