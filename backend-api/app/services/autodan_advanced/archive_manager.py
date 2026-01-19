"""
Dynamic Archive Manager for AutoDAN.

Implements Success Archive and Novelty Archive for diversity maintenance.
"""

import hashlib
import logging
from datetime import datetime

import numpy as np

from .models import ArchiveEntry

logger = logging.getLogger(__name__)


class DynamicArchiveManager:
    """
    Manages Success and Novelty archives for AutoDAN.

    The archive maintains:
    - Success Archive: Prompts with Score > Threshold
    - Novelty Archive: Prompts with high semantic distance

    Selection probability: P(select) ∝ α·Score + β·Novelty
    """

    def __init__(
        self,
        success_threshold: float = 7.0,
        novelty_threshold: float = 0.7,
        max_archive_size: int = 1000,
        alpha: float = 0.7,  # Weight for score
        beta: float = 0.3,  # Weight for novelty
    ):
        """
        Initialize the archive manager.

        Args:
            success_threshold: Minimum score for success archive
            novelty_threshold: Minimum novelty score for novelty archive
            max_archive_size: Maximum entries per archive
            alpha: Weight for score in selection
            beta: Weight for novelty in selection
        """
        self.success_threshold = success_threshold
        self.novelty_threshold = novelty_threshold
        self.max_archive_size = max_archive_size
        self.alpha = alpha
        self.beta = beta

        # Archives
        self.success_archive: list[ArchiveEntry] = []
        self.novelty_archive: list[ArchiveEntry] = []

        logger.info(
            f"DynamicArchiveManager initialized: "
            f"success_threshold={success_threshold}, "
            f"novelty_threshold={novelty_threshold}"
        )

    def add_entry(
        self, prompt: str, score: float, technique_type: str, embedding: np.ndarray | None = None
    ) -> bool:
        """
        Add an entry to the appropriate archive(s).

        Args:
            prompt: The prompt text
            score: Attack score (1-10)
            technique_type: Type of technique used
            embedding: Semantic embedding vector

        Returns:
            True if added to any archive
        """
        added = False

        # Generate ID
        entry_id = self._generate_id(prompt)

        # Calculate novelty score
        novelty_score = self._calculate_novelty(embedding) if embedding is not None else 0.0

        # Create entry
        entry = ArchiveEntry(
            id=entry_id,
            prompt=prompt,
            score=score,
            novelty_score=novelty_score,
            technique_type=technique_type,
            embedding_vector=embedding.tolist() if embedding is not None else [],
            created_at=datetime.utcnow(),
            success_count=1 if score >= self.success_threshold else 0,
        )

        # Add to success archive if score is high
        if score >= self.success_threshold:
            self._add_to_success_archive(entry)
            added = True

        # Add to novelty archive if novel
        if novelty_score >= self.novelty_threshold:
            self._add_to_novelty_archive(entry)
            added = True

        return added

    def sample_diverse_elites(self, k: int) -> list[str]:
        """
        Sample k diverse elite prompts from archives.

        Uses selection probability: P(select) ∝ α·Score + β·Novelty

        Args:
            k: Number of prompts to sample

        Returns:
            List of selected prompts
        """
        # Combine archives
        all_entries = self.success_archive + self.novelty_archive

        if not all_entries:
            return []

        # Calculate selection probabilities
        probabilities = []
        for entry in all_entries:
            prob = self.alpha * (entry.score / 10.0) + self.beta * entry.novelty_score
            probabilities.append(prob)

        # Normalize
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(all_entries)] * len(all_entries)

        # Sample without replacement
        k = min(k, len(all_entries))
        indices = np.random.choice(len(all_entries), size=k, replace=False, p=probabilities)

        return [all_entries[i].prompt for i in indices]

    def query_by_score(self, min_score: float, limit: int = 10) -> list[ArchiveEntry]:
        """Query entries by minimum score."""
        results = [e for e in self.success_archive if e.score >= min_score]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def query_by_novelty(self, min_novelty: float, limit: int = 10) -> list[ArchiveEntry]:
        """Query entries by minimum novelty."""
        results = [e for e in self.novelty_archive if e.novelty_score >= min_novelty]
        results.sort(key=lambda x: x.novelty_score, reverse=True)
        return results[:limit]

    def get_statistics(self) -> dict:
        """Get archive statistics."""
        return {
            "success_archive_size": len(self.success_archive),
            "novelty_archive_size": len(self.novelty_archive),
            "total_entries": len(self.success_archive) + len(self.novelty_archive),
            "avg_success_score": (
                np.mean([e.score for e in self.success_archive]) if self.success_archive else 0.0
            ),
            "avg_novelty_score": (
                np.mean([e.novelty_score for e in self.novelty_archive])
                if self.novelty_archive
                else 0.0
            ),
        }

    def _add_to_success_archive(self, entry: ArchiveEntry) -> None:
        """Add entry to success archive."""
        # Check if already exists
        if any(e.id == entry.id for e in self.success_archive):
            return

        self.success_archive.append(entry)

        # Maintain size limit
        if len(self.success_archive) > self.max_archive_size:
            # Remove lowest scoring entry
            self.success_archive.sort(key=lambda x: x.score)
            self.success_archive.pop(0)

    def _add_to_novelty_archive(self, entry: ArchiveEntry) -> None:
        """Add entry to novelty archive."""
        # Check if already exists
        if any(e.id == entry.id for e in self.novelty_archive):
            return

        self.novelty_archive.append(entry)

        # Maintain size limit
        if len(self.novelty_archive) > self.max_archive_size:
            # Remove lowest novelty entry
            self.novelty_archive.sort(key=lambda x: x.novelty_score)
            self.novelty_archive.pop(0)

    def _calculate_novelty(self, embedding: np.ndarray) -> float:
        """
        Calculate novelty score based on semantic distance.

        Novelty is the average distance to k-nearest neighbors in success archive.
        """
        if not self.success_archive:
            return 1.0  # Maximum novelty if archive is empty

        # Get embeddings from success archive
        archive_embeddings = []
        for entry in self.success_archive:
            if entry.embedding_vector:
                archive_embeddings.append(np.array(entry.embedding_vector))

        if not archive_embeddings:
            return 1.0

        # Calculate distances
        distances = []
        for archive_emb in archive_embeddings:
            # Cosine distance
            similarity = np.dot(embedding, archive_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(archive_emb)
            )
            distance = 1.0 - similarity
            distances.append(distance)

        # Average distance to k-nearest neighbors
        k = min(5, len(distances))
        distances.sort()
        novelty = np.mean(distances[:k])

        return float(novelty)

    def _generate_id(self, prompt: str) -> str:
        """Generate unique ID for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
