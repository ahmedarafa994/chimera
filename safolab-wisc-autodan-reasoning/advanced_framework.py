import logging
import random
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)

# Named tuple for archive entries
ArchiveEntry = namedtuple("ArchiveEntry", ["prompt", "score", "embedding"])


class DiversityArchive:
    """
    Maintains a diverse set of successful prompts using principles from Map-Elites.
    This prevents the search from converging to a single local optimum by explicitly
    valuing novelty (semantic distance) alongside objective performance (score).
    """

    def __init__(self, capacity=100, novelty_weight=0.5):
        self.capacity = capacity
        self.novelty_weight = novelty_weight
        self.entries = []  # List of ArchiveEntry

    def add(self, prompt, score, embedding):
        """
        Adds a prompt to the archive. The archive dynamically maintains high-fitness
        entries, where fitness is a weighted combination of attack score and novelty.
        """
        if not self.entries:
            self.entries.append(ArchiveEntry(prompt, score, embedding))
            return

        # Calculate novelty (average distance to k-nearest neighbors could be used,
        # here we use mean distance to all for efficiency)
        novelty = self._calculate_novelty(embedding)

        # Combined fitness metric
        _fitness = (1 - self.novelty_weight) * score + self.novelty_weight * novelty

        # Add to archive
        self.entries.append(ArchiveEntry(prompt, score, embedding))

        # Prune if over capacity
        if len(self.entries) > self.capacity:
            self._prune()

    def _calculate_novelty(self, embedding):
        """Calculates novelty as mean cosine distance to existing entries."""
        if not self.entries:
            return 1.0

        distances = []
        for entry in self.entries:
            # Assuming embeddings are normalized vectors
            # Cosine distance = 1 - cosine_similarity
            if entry.embedding is None or embedding is None:
                continue
            sim = np.dot(embedding, entry.embedding)
            dist = 1 - sim
            distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def _prune(self):
        """Removes entries with the lowest combined fitness."""
        scored_entries = []
        for entry in self.entries:
            n = self._calculate_novelty(entry.embedding)
            f = (1 - self.novelty_weight) * entry.score + self.novelty_weight * n
            scored_entries.append((f, entry))

        # Keep top capacity entries
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        self.entries = [x[1] for x in scored_entries[: self.capacity]]

    def sample_diverse_elites(self, k=5):
        """
        Returns k diverse elites from the archive to serve as parents for the next generation.
        """
        if not self.entries:
            return []

        if len(self.entries) < k:
            return [e.prompt for e in self.entries]

        # Sample randomly from the diversity archive
        # In a full Map-Elites, we would sample from different cells
        selected = random.sample(self.entries, k)
        return [e.prompt for e in selected]


class GradientOptimizer:
    """
    Handles gradient-based optimization using white-box surrogate models.
    Implements the Coherence-Constrained Gradient Search (CCGS).
    """

    def __init__(self, surrogates, tokenizer):
        self.surrogates = surrogates  # List of model wrappers exposing compute_gradients
        self.tokenizer = tokenizer

    def compute_aligned_gradient(self, tokens, target_string):
        """
        Computes the ensemble gradient, projecting out conflicting directions
        to ensure transferability (Ensemble Gradient Alignment).
        """
        grads = []
        for model in self.surrogates:
            try:
                # Placeholder for actual model gradient computation
                # Should return gradient tensor of shape [seq_len, vocab_size]
                g = model.get_gradient(tokens, target_string)
                if g is not None:
                    grads.append(g)
            except Exception as e:
                logger.debug(f"Error getting gradient from surrogate: {e}")

        if not grads:
            return None

        # PCGrad-style alignment could be implemented here.
        # For this implementation, we simply average the gradients from different models.
        # This biases the search towards directions that are generally bad for all models.
        avg_grad = np.mean(grads, axis=0)
        return avg_grad

    def refine_prompt_step(self, tokens, target_string, _coherence_weight=0.5):
        """
        Refines tokens using Coherence-Constrained Gradient Search.
        Returns a mutated list of tokens.
        """
        grad = self.compute_aligned_gradient(tokens, target_string)
        if grad is None:
            return tokens

        # In a real implementation:
        # 1. Identify token positions with high gradient magnitude
        # 2. For those positions, find candidate replacements with high -grad * coherence
        # 3. Apply substitution

        # Since we don't have the actual tensor ops here, we return the tokens unchanged
        # This serves as the architectural placeholder for the logic flow
        return tokens
