"""
AutoDAN Advanced Loss Functions and Fitness Evaluation

Enhanced loss functions for multi-objective optimization in adversarial prompt generation.
Includes attack success, coherence regularization, diversity, and stealth objectives.

Mathematical Foundations:
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- Contrastive Loss: Chen et al., "A Simple Framework for Contrastive Learning" (2020)
- Triplet Loss: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition" (2015)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FitnessConfig, LossFunction, LossFunctionConfig

# =============================================================================
# BASE LOSS CLASS
# =============================================================================


class BaseLoss(ABC):
    """Abstract base class for loss functions."""

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        """
        Initialize loss function.

        Args:
            weight: Loss weight in composite objective
            reduction: How to reduce batch losses ('mean', 'sum', 'none')
        """
        self.weight = weight
        self.reduction = reduction

    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
        """Compute loss value."""
        pass

    def __call__(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        """Compute and reduce loss."""
        loss = self.forward(predictions, targets, **kwargs)

        if self.reduction == "mean":
            return float(np.mean(loss)) * self.weight
        elif self.reduction == "sum":
            return float(np.sum(loss)) * self.weight
        else:
            return loss * self.weight

    def backward(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions (for gradient-based methods)."""
        # Default: numerical gradient
        eps = 1e-5
        grad = np.zeros_like(predictions)
        for i in range(predictions.shape[-1]):
            pred_plus = predictions.copy()
            pred_minus = predictions.copy()
            pred_plus[..., i] += eps
            pred_minus[..., i] -= eps
            grad[..., i] = (
                self.forward(pred_plus, targets, **kwargs)
                - self.forward(pred_minus, targets, **kwargs)
            ) / (2 * eps)
        return grad


# =============================================================================
# ATTACK LOSS FUNCTIONS
# =============================================================================


class CrossEntropyLoss(BaseLoss):
    """
    Standard cross-entropy loss for adversarial objective.

    L_CE = -Σᵢ yᵢ·log(pᵢ)

    For adversarial attacks: maximize probability of target (complex) tokens.
    """

    def __init__(self, weight: float = 1.0, label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__(weight, reduction)
        self.label_smoothing = label_smoothing

    def forward(self, predictions: np.ndarray, targets: np.ndarray, **_kwargs) -> np.ndarray:
        """
        Compute cross-entropy loss.

        Args:
            predictions: Log probabilities [batch, vocab_size]
            targets: Target token indices [batch] or one-hot [batch, vocab_size]
        """
        # Ensure numerical stability
        log_probs = np.clip(predictions, -100, 0)

        if targets.ndim == 1:
            # Convert indices to one-hot
            targets_onehot = np.zeros_like(predictions)
            targets_onehot[np.arange(len(targets)), targets] = 1.0
            targets = targets_onehot

        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = targets.shape[-1]
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / n_classes

        # Compute cross-entropy
        loss = -np.sum(targets * log_probs, axis=-1)
        return loss

    def backward(self, predictions: np.ndarray, targets: np.ndarray, **_kwargs) -> np.ndarray:
        """Analytical gradient: ∂L/∂p = -y/p"""
        probs = np.exp(np.clip(predictions, -100, 0))
        if targets.ndim == 1:
            targets_onehot = np.zeros_like(predictions)
            targets_onehot[np.arange(len(targets)), targets] = 1.0
            targets = targets_onehot
        return probs - targets  # Softmax gradient


class FocalLoss(BaseLoss):
    """
    Focal loss to down-weight easy examples.

    L_FL = -alpha_t · (1 - p_t) ^ gamma · log(p_t)

    Where:
    - αₜ: Class balancing weight
    - gamma: Focusing parameter (gamma=0 reduces to CE)
    - pₜ: Probability of correct class

    Higher gamma focuses more on hard examples (low p_t).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(weight, reduction)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: np.ndarray, targets: np.ndarray, **_kwargs) -> np.ndarray:
        """Compute focal loss."""
        probs = np.exp(np.clip(predictions, -100, 0))

        if targets.ndim == 1:
            # Get probability of target class
            pt = probs[np.arange(len(targets)), targets]
        else:
            pt = np.sum(probs * targets, axis=-1)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy: -log(pt)
        ce_loss = -np.log(np.clip(pt, 1e-8, 1.0))

        # Combined focal loss
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            loss = self.alpha * loss

        return loss


class AdversarialCrossEntropyLoss(BaseLoss):
    """
    Adversarial cross-entropy for jailbreak optimization.

    L_adv = -log P(target_response | adversarial_prompt)

    Maximizes the probability of generating complex/target responses.
    Includes refusal penalty and coherence bonus.
    """

    def __init__(
        self,
        refusal_weight: float = 0.5,
        coherence_weight: float = 0.3,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(weight, reduction)
        self.refusal_weight = refusal_weight
        self.coherence_weight = coherence_weight

    def forward(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        refusal_probs: np.ndarray | None = None,
        coherence_scores: np.ndarray | None = None,
        **_kwargs,
    ) -> np.ndarray:
        """
        Compute adversarial cross-entropy loss.

        Args:
            predictions: Log probabilities for target tokens
            targets: Target token indices
            refusal_probs: Probability of refusal tokens [optional]
            coherence_scores: Language model coherence scores [optional]
        """
        # Base cross-entropy (negated for maximization)
        probs = np.exp(np.clip(predictions, -100, 0))

        if targets.ndim == 1:
            target_probs = probs[np.arange(len(targets)), targets]
        else:
            target_probs = np.sum(probs * targets, axis=-1)

        loss = -np.log(np.clip(target_probs, 1e-8, 1.0))

        # Add refusal penalty
        if refusal_probs is not None:
            refusal_penalty = np.log(np.clip(1 - refusal_probs, 1e-8, 1.0))
            loss = loss - self.refusal_weight * refusal_penalty

        # Add coherence bonus (negative because we minimize loss)
        if coherence_scores is not None:
            coherence_bonus = -self.coherence_weight * np.log(np.clip(coherence_scores, 1e-8, 1.0))
            loss = loss + coherence_bonus

        return loss


# =============================================================================
# CONTRASTIVE LOSSES
# =============================================================================


class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss for embedding space optimization.

    L_contrastive = -log(exp(sim(z_i, z_j)/τ) / Σₖ exp(sim(z_i, z_k)/τ))

    Where:
    - z_i, z_j: Positive pair embeddings
    - z_k: Negative sample embeddings
    - τ: Temperature parameter

    Used to learn adversarial prompt embeddings that cluster around
    successful attack patterns.
    """

    def __init__(self, temperature: float = 0.07, weight: float = 1.0, reduction: str = "mean"):
        super().__init__(weight, reduction)
        self.temperature = temperature

    def forward(self, embeddings: np.ndarray, labels: np.ndarray, **_kwargs) -> np.ndarray:
        """
        Compute contrastive loss (NT-Xent style).

        Args:
            embeddings: Prompt embeddings [batch, embed_dim]
            labels: Success labels (1 = successful attack, 0 = failed)
        """
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix
        similarity = np.dot(embeddings_norm, embeddings_norm.T) / self.temperature

        # Create positive/negative masks
        labels = labels.reshape(-1, 1)
        positive_mask = (labels == labels.T).astype(float)
        np.fill_diagonal(positive_mask, 0)  # Exclude self-similarity

        negative_mask = (labels != labels.T).astype(float)

        # Compute loss for each sample
        batch_size = embeddings.shape[0]
        losses = []

        for i in range(batch_size):
            positive_indices = np.where(positive_mask[i] > 0)[0]

            if len(positive_indices) == 0:
                continue

            # Log-sum-exp over negatives
            neg_exp = np.exp(similarity[i] * negative_mask[i])
            neg_sum = np.sum(neg_exp) + 1e-8

            # Average over positives
            pos_loss = 0
            for j in positive_indices:
                pos_exp = np.exp(similarity[i, j])
                pos_loss += -np.log(pos_exp / (pos_exp + neg_sum))

            losses.append(pos_loss / len(positive_indices))

        if not losses:
            return np.array([0.0])

        return np.array(losses)


class TripletLoss(BaseLoss):
    """
    Triplet loss for metric learning.

    L_triplet = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)

    Where:
    - a: Anchor (current prompt)
    - p: Positive (similar successful prompt)
    - n: Negative (failed prompt)
    - margin: Minimum separation distance

    Encourages successful prompts to cluster together.
    """

    def __init__(
        self,
        margin: float = 0.5,
        distance: str = "euclidean",  # or "cosine"
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(weight, reduction)
        self.margin = margin
        self.distance = distance

    def _compute_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute pairwise distance."""
        if self.distance == "euclidean":
            return np.sqrt(np.sum((x - y) ** 2, axis=-1))
        else:  # cosine
            x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
            y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
            return 1 - np.sum(x_norm * y_norm, axis=-1)

    def forward(
        self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, **_kwargs
    ) -> np.ndarray:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch, embed_dim]
            positive: Positive embeddings [batch, embed_dim]
            negative: Negative embeddings [batch, embed_dim]
        """
        pos_dist = self._compute_distance(anchor, positive)
        neg_dist = self._compute_distance(anchor, negative)

        loss = np.maximum(0, pos_dist - neg_dist + self.margin)
        return loss

    def mine_triplets(
        self, embeddings: np.ndarray, labels: np.ndarray, strategy: str = "hard"
    ) -> tuple:
        """
        Mine triplets from batch.

        Args:
            embeddings: All embeddings [batch, embed_dim]
            labels: Success labels
            strategy: 'hard', 'semi-hard', or 'random'

        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = len(embeddings)
        positives = np.where(labels == 1)[0]
        negatives = np.where(labels == 0)[0]

        if len(positives) < 2 or len(negatives) < 1:
            return None, None, None

        # Compute all pairwise distances
        distances = np.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                distances[i, j] = self._compute_distance(
                    embeddings[i : i + 1], embeddings[j : j + 1]
                )[0]

        anchors, pos_indices, neg_indices = [], [], []

        for anchor in positives:
            # Find positive (same class, different sample)
            pos_candidates = [p for p in positives if p != anchor]

            if strategy == "hard":
                # Hardest positive: furthest same-class sample
                pos_idx = pos_candidates[np.argmax([distances[anchor, p] for p in pos_candidates])]
                # Hardest negative: closest different-class sample
                neg_idx = negatives[np.argmin([distances[anchor, n] for n in negatives])]
            elif strategy == "semi-hard":
                # Semi-hard: negative further than positive but within margin
                pos_idx = np.random.choice(pos_candidates)
                pos_dist = distances[anchor, pos_idx]
                valid_negs = [
                    n for n in negatives if pos_dist < distances[anchor, n] < pos_dist + self.margin
                ]
                if not valid_negs:
                    valid_negs = list(negatives)
                neg_idx = np.random.choice(valid_negs)
            else:  # random
                pos_idx = np.random.choice(pos_candidates)
                neg_idx = np.random.choice(negatives)

            anchors.append(anchor)
            pos_indices.append(pos_idx)
            neg_indices.append(neg_idx)

        return np.array(anchors), np.array(pos_indices), np.array(neg_indices)


# =============================================================================
# REGULARIZATION LOSSES
# =============================================================================


class CoherenceLoss(BaseLoss):
    """
    Coherence regularization to maintain human readability.

    L_coherence = -log P_LM(x) / |x|

    Where P_LM(x) is the language model probability of the prompt.
    Lower perplexity = more coherent = lower loss.
    """

    def __init__(
        self, target_perplexity: float = 50.0, weight: float = 1.0, reduction: str = "mean"
    ):
        super().__init__(weight, reduction)
        self.target_perplexity = target_perplexity

    def forward(self, log_probs: np.ndarray, lengths: np.ndarray, **_kwargs) -> np.ndarray:
        """
        Compute coherence loss from log probabilities.

        Args:
            log_probs: Sum of log probabilities for each prompt
            lengths: Number of tokens in each prompt
        """
        # Compute perplexity: exp(-1/N * Σ log p)
        avg_log_prob = log_probs / np.maximum(lengths, 1)
        perplexity = np.exp(-avg_log_prob)

        # Loss is deviation from target perplexity
        # Penalize both too high (incoherent) and too low (repetitive)
        loss = np.abs(np.log(perplexity) - np.log(self.target_perplexity))

        return loss

    def from_perplexity(self, perplexity: np.ndarray) -> np.ndarray:
        """Compute loss directly from perplexity values."""
        return np.abs(np.log(perplexity + 1e-8) - np.log(self.target_perplexity))


class StealthLoss(BaseLoss):
    """
    Stealth regularization to evade detection systems.

    L_stealth = log(P_detector(x)) + λ·sim(x, benign_template)

    Minimizes:
    1. Detection classifier confidence
    2. Distance from benign prompt templates
    """

    def __init__(
        self, similarity_weight: float = 0.3, weight: float = 1.0, reduction: str = "mean"
    ):
        super().__init__(weight, reduction)
        self.similarity_weight = similarity_weight

    def forward(
        self,
        detection_probs: np.ndarray,
        embeddings: np.ndarray | None = None,
        benign_embeddings: np.ndarray | None = None,
        **_kwargs,
    ) -> np.ndarray:
        """
        Compute stealth loss.

        Args:
            detection_probs: Probability of being detected as adversarial
            embeddings: Current prompt embeddings
            benign_embeddings: Reference benign prompt embeddings
        """
        # Detection penalty (want low detection probability)
        detection_loss = np.log(np.clip(detection_probs, 1e-8, 1.0))

        # Similarity to benign templates (want high similarity)
        if embeddings is not None and benign_embeddings is not None:
            # Cosine similarity to nearest benign template
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
            benign_norm = benign_embeddings / (
                np.linalg.norm(benign_embeddings, axis=-1, keepdims=True) + 1e-8
            )

            # Max similarity across benign templates
            similarities = np.dot(emb_norm, benign_norm.T)
            max_similarity = np.max(similarities, axis=-1)

            # Negative similarity (want to maximize)
            similarity_loss = -self.similarity_weight * max_similarity

            return detection_loss + similarity_loss

        return detection_loss


class DiversityLoss(BaseLoss):
    """
    Diversity regularization to prevent mode collapse.

    L_diversity = -entropy(population) + λ·avg_pairwise_similarity

    Encourages exploration of diverse attack strategies.
    """

    def __init__(
        self,
        similarity_penalty: float = 0.5,
        min_diversity: float = 0.3,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__(weight, reduction)
        self.similarity_penalty = similarity_penalty
        self.min_diversity = min_diversity

    def forward(
        self, embeddings: np.ndarray, population_embeddings: np.ndarray | None = None, **_kwargs
    ) -> np.ndarray:
        """
        Compute diversity loss.

        Args:
            embeddings: Current prompt embeddings [batch, embed_dim]
            population_embeddings: Full population embeddings [pop_size, embed_dim]
        """
        if population_embeddings is None:
            population_embeddings = embeddings

        # Normalize embeddings
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        pop_norm = population_embeddings / (
            np.linalg.norm(population_embeddings, axis=-1, keepdims=True) + 1e-8
        )

        # Compute similarity to population
        similarities = np.dot(emb_norm, pop_norm.T)

        # Average similarity (excluding self)
        batch_size = embeddings.shape[0]
        pop_size = population_embeddings.shape[0]

        if batch_size == pop_size and np.allclose(embeddings, population_embeddings):
            # Same population, exclude diagonal
            np.fill_diagonal(similarities, 0)
            avg_similarity = np.sum(similarities, axis=-1) / (pop_size - 1)
        else:
            avg_similarity = np.mean(similarities, axis=-1)

        # Loss penalizes high similarity
        loss = self.similarity_penalty * np.maximum(0, avg_similarity - self.min_diversity)

        return loss


# =============================================================================
# COMPOSITE LOSS
# =============================================================================


class CompositeLoss(BaseLoss):
    """
    Composite loss combining multiple objectives.

    L_total = Σᵢ wᵢ·Lᵢ

    Supports dynamic weight adjustment based on optimization progress.
    """

    def __init__(
        self,
        losses: dict[str, BaseLoss],
        config: LossFunctionConfig | None = None,
        reduction: str = "mean",
    ):
        super().__init__(weight=1.0, reduction=reduction)
        self.losses = losses
        self.config = config or LossFunctionConfig()
        self._loss_history: dict[str, list[float]] = {name: [] for name in losses}

    def forward(self, **kwargs) -> dict[str, float]:
        """
        Compute all component losses.

        Returns dict with individual and total losses.
        """
        results = {}
        total = 0.0

        for name, loss_fn in self.losses.items():
            # Get relevant kwargs for this loss
            loss_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

            try:
                loss_value = loss_fn(**loss_kwargs)
                results[name] = float(np.mean(loss_value))
                total += results[name]
                self._loss_history[name].append(results[name])
            except Exception:
                results[name] = 0.0

        results["total"] = total
        return results

    def __call__(self, **kwargs) -> float:
        """Compute total composite loss."""
        results = self.forward(**kwargs)
        return results["total"]

    def get_loss_trends(self, window: int = 10) -> dict[str, float]:
        """Get recent loss trends (positive = increasing)."""
        trends = {}
        for name, history in self._loss_history.items():
            if len(history) < window:
                trends[name] = 0.0
            else:
                recent = history[-window:]
                trends[name] = (recent[-1] - recent[0]) / (window * max(abs(recent[0]), 1e-8))
        return trends

    def rebalance_weights(self, target_ratios: dict[str, float] | None = None) -> dict[str, float]:
        """
        Rebalance loss weights based on current magnitudes.

        Ensures all losses contribute proportionally.
        """
        if not any(self._loss_history.values()):
            return {name: loss.weight for name, loss in self.losses.items()}

        # Get recent average magnitudes
        magnitudes = {}
        for name, history in self._loss_history.items():
            if history:
                magnitudes[name] = np.mean(history[-10:])
            else:
                magnitudes[name] = 1.0

        # Compute inverse magnitude weights (larger losses get smaller weights)
        total_mag = sum(magnitudes.values())
        new_weights = {}

        for name, loss_fn in self.losses.items():
            if target_ratios and name in target_ratios:
                # Use specified target ratio
                new_weights[name] = target_ratios[name]
            else:
                # Inverse magnitude normalization
                new_weights[name] = (total_mag / len(self.losses)) / (magnitudes[name] + 1e-8)

            loss_fn.weight = new_weights[name]

        return new_weights


# =============================================================================
# ENHANCED FITNESS EVALUATOR
# =============================================================================


@dataclass
class FitnessResult:
    """Container for fitness evaluation results."""

    total_fitness: float
    components: dict[str, float]
    metadata: dict[str, Any]


class EnhancedFitnessEvaluator:
    """
    Enhanced fitness evaluator for AutoDAN optimization.

    Combines multiple objectives into a single fitness score:
    F(x) = w₁·S_attack - w₂·P_refusal + w₃·C_coherence + w₄·N_novelty - w₅·D_detect
    """

    def __init__(
        self, config: FitnessConfig | None = None, population_embeddings: np.ndarray | None = None
    ):
        self.config = config or FitnessConfig()
        self.population_embeddings = population_embeddings
        self._fitness_cache: dict[str, FitnessResult] = {}

    def evaluate(
        self,
        prompt: str,
        attack_score: float,
        refusal_score: float = 0.0,
        coherence_score: float = 1.0,
        detection_score: float = 0.0,
        embedding: np.ndarray | None = None,
        use_cache: bool = True,
    ) -> FitnessResult:
        """
        Evaluate fitness of a prompt.

        Args:
            prompt: The adversarial prompt
            attack_score: Attack success score [0, 10]
            refusal_score: Refusal probability [0, 1]
            coherence_score: Linguistic coherence [0, 1]
            detection_score: Detection probability [0, 1]
            embedding: Prompt embedding for diversity computation
            use_cache: Whether to use cached results

        Returns:
            FitnessResult with total fitness and components
        """
        cache_key = f"{hash(prompt)}_{attack_score:.3f}"

        if use_cache and cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]

        # Normalize attack score to [0, 1]
        normalized_attack = attack_score / 10.0

        # Compute novelty score
        novelty_score = self._compute_novelty(embedding)

        # Compute weighted fitness
        components = {
            "attack": self.config.w_alignment * normalized_attack,
            "refusal_penalty": -self.config.w_refusal * refusal_score,
            "coherence": self.config.w_coherence * coherence_score,
            "novelty": self.config.w_novelty * novelty_score,
            "detection_penalty": -self.config.w_detection * detection_score,
        }

        total_fitness = sum(components.values())

        # Scale back to [0, 10]
        total_fitness = np.clip(total_fitness * 10, 0, 10)

        result = FitnessResult(
            total_fitness=float(total_fitness),
            components=components,
            metadata={
                "raw_attack_score": attack_score,
                "cache_hit": False,
            },
        )

        if use_cache:
            self._fitness_cache[cache_key] = result

        return result

    def _compute_novelty(self, embedding: np.ndarray | None) -> float:
        """Compute novelty score relative to population."""
        if embedding is None or self.population_embeddings is None:
            return 0.5  # Neutral score

        if len(self.population_embeddings) == 0:
            return 1.0  # Maximum novelty for empty population

        # Cosine similarity to population
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        pop_norm = self.population_embeddings / (
            np.linalg.norm(self.population_embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = np.dot(pop_norm, emb_norm)
        max_similarity = np.max(similarities)

        # Novelty is inverse of maximum similarity
        return 1.0 - max_similarity

    def update_population(self, embeddings: np.ndarray):
        """Update population embeddings for novelty computation."""
        self.population_embeddings = embeddings
        self._fitness_cache.clear()

    def apply_fitness_sharing(
        self, fitness_values: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Apply fitness sharing for diversity preservation.

        F_shared(i) = F(i) / Σⱼ sh(d(i,j))

        Where sh(d) = 1 - (d/sigma)^alpha if d < sigma, else 0
        """
        n = len(fitness_values)
        shared_fitness = np.zeros(n)

        # Compute pairwise similarities
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(emb_norm, emb_norm.T)

        for i in range(n):
            # Compute sharing function values
            distances = 1 - similarities[i]  # Convert similarity to distance

            # sh(d) = 1 - (d/sigma)^alpha if d < sigma
            sharing_values = np.where(
                distances < self.config.sharing_radius,
                1 - (distances / self.config.sharing_radius) ** self.config.sharing_alpha,
                0,
            )

            niche_count = np.sum(sharing_values)
            shared_fitness[i] = fitness_values[i] / max(niche_count, 1.0)

        return shared_fitness

    def rank_population(
        self,
        fitness_values: np.ndarray,
        embeddings: np.ndarray | None = None,
        use_sharing: bool = True,
    ) -> np.ndarray:
        """
        Rank population by fitness.

        Returns indices sorted by fitness (descending).
        """
        if use_sharing and embeddings is not None:
            effective_fitness = self.apply_fitness_sharing(fitness_values, embeddings)
        else:
            effective_fitness = fitness_values

        return np.argsort(effective_fitness)[::-1]

    def select_elite(
        self, fitness_values: np.ndarray, embeddings: np.ndarray | None = None
    ) -> np.ndarray:
        """Select elite individuals based on fitness and diversity."""
        n_elite = max(1, int(len(fitness_values) * self.config.elite_ratio))

        ranking = self.rank_population(fitness_values, embeddings)
        return ranking[:n_elite]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_loss_function(
    loss_type: LossFunction, config: LossFunctionConfig | None = None
) -> BaseLoss:
    """Factory function to create loss functions."""
    config = config or LossFunctionConfig()

    loss_map = {
        LossFunction.CROSS_ENTROPY: lambda: CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        ),
        LossFunction.FOCAL_LOSS: lambda: FocalLoss(gamma=config.focal_gamma),
        LossFunction.CONTRASTIVE: lambda: ContrastiveLoss(
            temperature=config.contrastive_temperature
        ),
        LossFunction.TRIPLET: lambda: TripletLoss(margin=config.triplet_margin),
        LossFunction.ADVERSARIAL_CE: lambda: AdversarialCrossEntropyLoss(),
        LossFunction.COMPOSITE: lambda: create_composite_loss(config),
    }

    return loss_map[loss_type]()


def create_composite_loss(config: LossFunctionConfig) -> CompositeLoss:
    """Create a composite loss with all components."""
    losses = {
        "attack": AdversarialCrossEntropyLoss(weight=config.alpha_attack),
        "coherence": CoherenceLoss(weight=config.beta_coherence),
        "diversity": DiversityLoss(weight=config.gamma_diversity),
        "stealth": StealthLoss(weight=config.delta_stealth),
    }

    return CompositeLoss(losses, config)
