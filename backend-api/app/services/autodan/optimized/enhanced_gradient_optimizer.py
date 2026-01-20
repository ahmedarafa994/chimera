"""Enhanced Gradient Optimizer for AutoDAN.

Implements advanced gradient-based optimization with:
- Momentum-based updates
- Surrogate model integration
- Coherence-constrained gradient search (CCGS)
- Gradient caching for efficiency
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GradientCacheEntry:
    """Entry in the gradient cache."""

    gradient: np.ndarray
    timestamp: float
    hit_count: int = 0


@dataclass
class OptimizationResult:
    """Result of gradient optimization."""

    optimized_tokens: list[int]
    final_loss: float
    iterations: int
    gradient_norm: float
    coherence_score: float


@dataclass
class SurrogateModel:
    """Lightweight surrogate model for gradient estimation."""

    weights: np.ndarray
    bias: float
    learning_rate: float = 0.01

    def predict(self, x: np.ndarray) -> float:
        """Predict loss value."""
        return float(np.dot(self.weights, x) + self.bias)

    def update(self, x: np.ndarray, target: float) -> None:
        """Update surrogate with new observation."""
        prediction = self.predict(x)
        error = target - prediction
        self.weights += self.learning_rate * error * x
        self.bias += self.learning_rate * error


class GradientCache:
    """LRU cache for gradient computations.

    Reduces redundant gradient calculations by caching results
    based on prompt hash and position.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, GradientCacheEntry] = {}
        self.access_order: list[str] = []

        # Statistics
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str, position: int) -> str:
        """Create cache key from prompt and position."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:16]
        return f"{prompt_hash}_{position}"

    def get(self, prompt: str, position: int) -> np.ndarray | None:
        """Get cached gradient if available and not expired."""
        key = self._make_key(prompt, position)

        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[key]
            self.access_order.remove(key)
            self.misses += 1
            return None

        # Update access order
        self.access_order.remove(key)
        self.access_order.append(key)
        entry.hit_count += 1
        self.hits += 1

        return entry.gradient

    def put(self, prompt: str, position: int, gradient: np.ndarray) -> None:
        """Store gradient in cache."""
        key = self._make_key(prompt, position)

        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = GradientCacheEntry(
            gradient=gradient.copy(),
            timestamp=time.time(),
        )
        self.access_order.append(key)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class EnhancedGradientOptimizer:
    """Enhanced gradient optimizer with momentum and surrogate models.

    Features:
    - Momentum-based gradient updates for faster convergence
    - Surrogate model for efficient gradient estimation
    - Coherence-constrained gradient search (CCGS)
    - Gradient caching for reduced computation
    - Adaptive step sizing
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_dim: int = 384,
        momentum: float = 0.9,
        initial_lr: float = 0.01,
        coherence_lambda: float = 0.5,
        use_surrogate: bool = True,
        cache_size: int = 10000,
    ) -> None:
        self.llm_client = llm_client
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        self.initial_lr = initial_lr
        self.coherence_lambda = coherence_lambda
        self.use_surrogate = use_surrogate

        # Gradient cache
        self.cache = GradientCache(max_size=cache_size)

        # Velocity for momentum
        self.velocity: np.ndarray | None = None

        # Surrogate model
        self.surrogate: SurrogateModel | None = None
        if use_surrogate:
            self.surrogate = SurrogateModel(
                weights=np.zeros(embedding_dim),
                bias=0.0,
            )

        # Statistics
        self.optimization_history: list[dict[str, float]] = []

    def optimize(
        self,
        tokens: list[int],
        target_string: str,
        gradient_steps: int = 10,
        epsilon: float = 0.01,
        coherence_threshold: float = 0.7,
    ) -> OptimizationResult:
        """Optimize tokens using enhanced gradient descent.

        Args:
            tokens: Initial token sequence
            target_string: Target string for optimization
            gradient_steps: Number of gradient steps
            epsilon: Step size multiplier
            coherence_threshold: Minimum coherence score

        Returns:
            OptimizationResult with optimized tokens and metrics

        """
        current_tokens = tokens.copy()
        lr = self.initial_lr

        # Initialize velocity if needed
        if self.velocity is None or len(self.velocity) != len(tokens):
            self.velocity = np.zeros(len(tokens))

        best_tokens = current_tokens.copy()
        best_loss = float("inf")

        for step in range(gradient_steps):
            # Compute gradient
            gradient = self._compute_gradient(current_tokens, target_string)

            # Apply momentum
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * gradient

            # Coherence-constrained update
            updated_tokens, coherence = self._coherence_constrained_update(
                current_tokens,
                self.velocity,
                lr * epsilon,
                coherence_threshold,
            )

            # Compute loss
            loss = self._compute_loss(updated_tokens, target_string)

            # Update surrogate model
            if self.surrogate is not None:
                token_embedding = self._get_token_embedding(updated_tokens)
                self.surrogate.update(token_embedding, loss)

            # Track best
            if loss < best_loss:
                best_loss = loss
                best_tokens = updated_tokens.copy()

            # Adaptive learning rate
            lr = self._adapt_learning_rate(lr, loss, step)

            # Record history
            self.optimization_history.append(
                {
                    "step": step,
                    "loss": loss,
                    "coherence": coherence,
                    "lr": lr,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                },
            )

            current_tokens = updated_tokens

            # Early stopping if loss is very low
            if loss < 0.01:
                logger.info(f"Early stopping at step {step} with loss {loss}")
                break

        return OptimizationResult(
            optimized_tokens=best_tokens,
            final_loss=best_loss,
            iterations=step + 1,
            gradient_norm=float(np.linalg.norm(self.velocity)),
            coherence_score=coherence,
        )

    def _compute_gradient(
        self,
        tokens: list[int],
        target_string: str,
    ) -> np.ndarray:
        """Compute gradient with caching and surrogate estimation."""
        prompt = self._tokens_to_string(tokens)

        # Try cache first
        gradients = []
        for pos in range(len(tokens)):
            cached = self.cache.get(prompt, pos)
            if cached is not None:
                gradients.append(cached)
            else:
                # Compute gradient for this position
                grad = self._compute_position_gradient(tokens, pos, target_string)
                self.cache.put(prompt, pos, grad)
                gradients.append(grad)

        # Aggregate gradients
        return np.mean(gradients, axis=0) if gradients else np.zeros(len(tokens))

    def _compute_position_gradient(
        self,
        tokens: list[int],
        position: int,
        target_string: str,
    ) -> np.ndarray:
        """Compute gradient at a specific position using finite differences."""
        epsilon = 1e-4
        gradient = np.zeros(len(tokens))

        # Original loss
        original_loss = self._compute_loss(tokens, target_string)

        # Perturb each dimension
        for i in range(min(len(tokens), 50)):  # Limit dimensions
            perturbed = tokens.copy()
            # Simulate perturbation by token substitution
            perturbed[i] = (perturbed[i] + 1) % 50000  # Vocab size

            perturbed_loss = self._compute_loss(perturbed, target_string)
            gradient[i] = (perturbed_loss - original_loss) / epsilon

        return gradient

    def _compute_loss(
        self,
        tokens: list[int],
        target_string: str,
    ) -> float:
        """Compute composite loss function.

        L_total = α·L_attack + β·L_coherence + γ·L_diversity
        """
        # Use surrogate if available for efficiency
        if self.surrogate is not None:
            token_embedding = self._get_token_embedding(tokens)
            surrogate_loss = self.surrogate.predict(token_embedding)
            # Blend surrogate with actual loss occasionally
            if np.random.random() < 0.1:
                actual_loss = self._compute_actual_loss(tokens, target_string)
                self.surrogate.update(token_embedding, actual_loss)
                return 0.5 * surrogate_loss + 0.5 * actual_loss
            return surrogate_loss

        return self._compute_actual_loss(tokens, target_string)

    def _compute_actual_loss(
        self,
        tokens: list[int],
        target_string: str,
    ) -> float:
        """Compute actual loss by querying the model."""
        prompt = self._tokens_to_string(tokens)

        # Attack loss (negative of attack success probability)
        attack_loss = 1.0 - self._estimate_attack_success(prompt, target_string)

        # Coherence loss (perplexity-based)
        coherence_loss = self._compute_coherence_loss(prompt)

        # Diversity loss (entropy-based)
        diversity_loss = self._compute_diversity_loss(tokens)

        # Weighted combination
        alpha, beta, gamma = 0.6, 0.3, 0.1
        return alpha * attack_loss + beta * coherence_loss + gamma * diversity_loss

    def _coherence_constrained_update(
        self,
        tokens: list[int],
        gradient: np.ndarray,
        step_size: float,
        coherence_threshold: float,
    ) -> tuple[list[int], float]:
        """Apply gradient update with coherence constraint.

        Ensures the updated prompt maintains linguistic coherence.
        """
        # Compute update direction
        update = -step_size * gradient

        # Apply update to tokens
        updated_tokens = []
        for i, (token, delta) in enumerate(zip(tokens, update, strict=False)):
            # Convert continuous update to discrete token change
            if abs(delta) > 0.5:
                # Significant gradient - consider token change
                new_token = self._find_coherent_replacement(tokens, i, delta, coherence_threshold)
                updated_tokens.append(new_token)
            else:
                updated_tokens.append(token)

        # Compute coherence of result
        coherence = self._compute_coherence(updated_tokens)

        # If coherence is too low, revert some changes
        if coherence < coherence_threshold:
            # Revert changes that hurt coherence most
            updated_tokens = self._revert_incoherent_changes(
                tokens,
                updated_tokens,
                coherence_threshold,
            )
            coherence = self._compute_coherence(updated_tokens)

        return updated_tokens, coherence

    def _find_coherent_replacement(
        self,
        tokens: list[int],
        position: int,
        gradient_direction: float,
        coherence_threshold: float,
    ) -> int:
        """Find a replacement token that maintains coherence."""
        original_token = tokens[position]

        # Get candidate replacements based on gradient direction
        candidates = self._get_replacement_candidates(original_token, gradient_direction)

        # Score candidates by coherence
        best_token = original_token
        best_score = -float("inf")

        for candidate in candidates[:10]:  # Limit candidates
            test_tokens = tokens.copy()
            test_tokens[position] = candidate

            coherence = self._compute_coherence(test_tokens)
            if coherence >= coherence_threshold and coherence > best_score:
                best_score = coherence
                best_token = candidate

        return best_token

    def _get_replacement_candidates(
        self,
        token: int,
        gradient_direction: float,
    ) -> list[int]:
        """Get candidate replacement tokens."""
        # Simple heuristic: nearby tokens in vocabulary
        vocab_size = 50000
        offset = int(np.sign(gradient_direction) * 10)

        candidates = []
        for i in range(-5, 6):
            candidate = (token + offset + i) % vocab_size
            if candidate != token:
                candidates.append(candidate)

        return candidates

    def _revert_incoherent_changes(
        self,
        original_tokens: list[int],
        updated_tokens: list[int],
        coherence_threshold: float,
    ) -> list[int]:
        """Revert changes that hurt coherence."""
        result = updated_tokens.copy()

        # Find changed positions
        changed = [
            i
            for i, (o, u) in enumerate(zip(original_tokens, updated_tokens, strict=False))
            if o != u
        ]

        # Revert changes one by one until coherence is acceptable
        for pos in changed:
            result[pos] = original_tokens[pos]
            if self._compute_coherence(result) >= coherence_threshold:
                break

        return result

    def _compute_coherence(self, tokens: list[int]) -> float:
        """Compute coherence score for token sequence."""
        # Simplified coherence based on token transitions
        if len(tokens) < 2:
            return 1.0

        # Count smooth transitions (adjacent tokens)
        smooth_transitions = 0
        for i in range(len(tokens) - 1):
            diff = abs(tokens[i + 1] - tokens[i])
            if diff < 1000:  # Arbitrary threshold for "smooth"
                smooth_transitions += 1

        return smooth_transitions / (len(tokens) - 1)

    def _compute_coherence_loss(self, prompt: str) -> float:
        """Compute coherence loss based on perplexity."""
        # Simplified: use length and character diversity
        if not prompt:
            return 1.0

        # Character diversity
        unique_chars = len(set(prompt))
        total_chars = len(prompt)
        diversity = unique_chars / total_chars if total_chars > 0 else 0

        # Penalize very low or very high diversity
        optimal_diversity = 0.3
        return abs(diversity - optimal_diversity)

    def _compute_diversity_loss(self, tokens: list[int]) -> float:
        """Compute diversity loss based on token entropy."""
        if not tokens:
            return 1.0

        # Token frequency distribution
        _unique, counts = np.unique(tokens, return_counts=True)
        probs = counts / len(tokens)

        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log(len(tokens))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # We want high entropy (diversity), so loss is 1 - entropy
        return 1.0 - normalized_entropy

    def _estimate_attack_success(
        self,
        prompt: str,
        target_string: str,
    ) -> float:
        """Estimate probability of attack success."""
        # Simplified heuristic based on prompt characteristics
        score = 0.0

        # Check for jailbreak patterns
        jailbreak_patterns = [
            "hypothetically",
            "fictional",
            "academic",
            "research",
            "creative writing",
            "roleplay",
            "pretend",
            "imagine",
        ]

        prompt_lower = prompt.lower()
        for pattern in jailbreak_patterns:
            if pattern in prompt_lower:
                score += 0.1

        # Length bonus (longer prompts often more effective)
        if len(prompt) > 200:
            score += 0.1
        if len(prompt) > 500:
            score += 0.1

        return min(1.0, score)

    def _adapt_learning_rate(
        self,
        current_lr: float,
        loss: float,
        step: int,
    ) -> float:
        """Adapt learning rate based on loss trajectory."""
        if len(self.optimization_history) < 2:
            return current_lr

        # Compare with previous loss
        prev_loss = self.optimization_history[-1]["loss"]

        if loss < prev_loss:
            # Loss decreased - slightly increase LR
            return min(current_lr * 1.05, self.initial_lr * 2)
        # Loss increased - decrease LR
        return max(current_lr * 0.9, self.initial_lr * 0.1)

    def _tokens_to_string(self, tokens: list[int]) -> str:
        """Convert tokens to string (simplified)."""
        # In practice, use actual tokenizer
        return "".join(chr(min(t % 128 + 32, 126)) for t in tokens)

    def _get_token_embedding(self, tokens: list[int]) -> np.ndarray:
        """Get embedding for token sequence."""
        # Simplified: use token values as features
        embedding = np.zeros(self.embedding_dim)
        for i, token in enumerate(tokens[: self.embedding_dim]):
            embedding[i] = token / 50000.0  # Normalize
        return embedding

    def get_cache_stats(self) -> dict[str, Any]:
        """Get gradient cache statistics."""
        return self.cache.get_stats()

    def get_optimization_history(self) -> list[dict[str, float]]:
        """Get optimization history."""
        return self.optimization_history.copy()

    def reset(self) -> None:
        """Reset optimizer state."""
        self.velocity = None
        self.optimization_history.clear()
        if self.surrogate is not None:
            self.surrogate = SurrogateModel(
                weights=np.zeros(self.embedding_dim),
                bias=0.0,
            )
