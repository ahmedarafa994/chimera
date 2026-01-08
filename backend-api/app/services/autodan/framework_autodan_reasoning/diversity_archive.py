import logging

import numpy as np

logger = logging.getLogger(__name__)


class DiversityArchive:
    """
    Manages a population of prompts with scores and embeddings.
    Implements a simplified Map-Elites / Novelty Search approach.
    """

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.entries = []  # List of dict: {'prompt': str, 'score': float, 'embedding': np.array}

    def add(self, prompt, score, embedding):
        """
        Add a prompt to the archive.

        Args:
            prompt (str): The prompt text.
            score (float): The fitness score (e.g. attack success rate or judge score).
            embedding (np.array): Semantic embedding of the prompt (or response).
        """
        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        entry = {"prompt": prompt, "score": score, "embedding": embedding}

        # Check if identical prompt exists
        for e in self.entries:
            if e["prompt"] == prompt:
                # Update score if better
                if score > e["score"]:
                    e["score"] = score
                return

        self.entries.append(entry)

        # Simple capacity management: remove lowest scoring items if full
        # In a real Map-Elites, we would map to cells. Here we just keep top scoring + diverse.
        if len(self.entries) > self.capacity:
            self._prune()

    def _prune(self):
        """Remove entries to stay within capacity."""
        # Sort by score descending
        self.entries.sort(key=lambda x: x["score"], reverse=True)
        # Keep top 80% capacity
        keep_count = int(self.capacity * 0.8)
        self.entries = self.entries[:keep_count]

    def sample_diverse_elites(self, k=10):
        """
        Select k diverse and high-performing prompts.
        Selection Logic: P(select) ~ alpha * Score + beta * Novelty

        Args:
            k (int): Number of elites to return.

        Returns:
            List[str]: List of selected prompts.
        """
        if not self.entries:
            return []

        if len(self.entries) <= k:
            return [e["prompt"] for e in self.entries]

        # Calculate Novelty (average distance to k-nearest neighbors or centroid)
        # For simplicity, let's use average distance to all others (expensive O(N^2) but N is small ~1000)
        # Or just distance to the "Success Archive" centroid.

        embeddings = np.array([e["embedding"] for e in self.entries])
        # Center the embeddings
        centroid = np.mean(embeddings, axis=0)

        # Calculate distance from centroid (proxy for novelty/outlier status)
        # Higher distance = more novel
        distances = np.linalg.norm(embeddings - centroid, axis=1)

        # Normalize scores and distances
        scores = np.array([e["score"] for e in self.entries])

        if scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones_like(scores)

        if distances.max() > distances.min():
            norm_dists = (distances - distances.min()) / (distances.max() - distances.min())
        else:
            norm_dists = np.zeros_like(distances)

        # Selection weight
        alpha = 0.7  # Weight for score
        beta = 0.3  # Weight for novelty
        weights = alpha * norm_scores + beta * norm_dists

        # Probabilistic selection based on weights
        # Softmax? Or just weighted random sample
        probs = np.exp(weights) / np.sum(np.exp(weights))

        indices = np.random.choice(len(self.entries), size=k, replace=False, p=probs)

        return [self.entries[i]["prompt"] for i in indices]

    def get_best_prompt(self):
        if not self.entries:
            return None
        return max(self.entries, key=lambda x: x["score"])["prompt"]
