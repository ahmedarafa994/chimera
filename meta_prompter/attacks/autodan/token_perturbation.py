"""
Token-Level Perturbation Strategies for AutoDAN Adversarial Prompt Generation.

This module implements gradient-guided token-level perturbation strategies:
- Gradient-based token importance scoring
- Hot-flip attack simulation
- Gumbel-Softmax relaxation for discrete token spaces
- Semantic replacement operators

Mathematical Framework:
    Token Importance: I(t_i) = ||∂L/∂e(t_i)||₂
    Perturbation: t'_i = argmax_t [e(t)ᵀ · ∇_{e(t_i)} L]
    Gumbel-Softmax: p_i = softmax((log π_i + g_i) / τ)

Reference: AutoDAN and HotFlip token perturbation techniques
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from meta_prompter.utils import TokenEmbedding

logger = logging.getLogger("chimera.autodan.perturbation")


class PerturbationType(str, Enum):
    """Types of token perturbation."""

    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"
    SWAP = "swap"
    SEMANTIC_REPLACE = "semantic_replace"
    HOMOGLYPH = "homoglyph"
    UNICODE_TRANSFORM = "unicode_transform"


class GradientEstimationMethod(str, Enum):
    """Methods for gradient estimation without backprop access."""

    FINITE_DIFFERENCE = "finite_difference"
    ZEROTH_ORDER = "zeroth_order"
    EVOLUTIONARY = "evolutionary"
    IMPORTANCE_SAMPLING = "importance_sampling"


@dataclass
class TokenPerturbationConfig:
    """Configuration for token-level perturbation strategies."""

    # Perturbation probabilities
    substitution_prob: float = 0.4
    insertion_prob: float = 0.2
    deletion_prob: float = 0.15
    swap_prob: float = 0.1
    semantic_replace_prob: float = 0.15

    # Gradient estimation
    gradient_method: GradientEstimationMethod = GradientEstimationMethod.ZEROTH_ORDER
    num_samples: int = 10
    perturbation_scale: float = 0.01

    # Gumbel-Softmax parameters
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_decay: float = 0.995

    # Token selection
    top_k_tokens: int = 10
    importance_threshold: float = 0.1

    # Semantic constraints
    max_semantic_distance: float = 0.3
    preserve_keywords: bool = True

    # Readability constraints
    min_fluency_score: float = 0.5
    max_perplexity_increase: float = 2.0


@dataclass
class TokenInfo:
    """Information about a token in the prompt."""

    token: str
    position: int
    importance_score: float = 0.0
    gradient_estimate: float = 0.0
    is_keyword: bool = False
    is_perturbable: bool = True
    alternatives: list[str] = field(default_factory=list)

    def __lt__(self, other: "TokenInfo") -> bool:
        return self.importance_score < other.importance_score


@dataclass
class PerturbationResult:
    """Result of a token perturbation operation."""

    original_text: str
    perturbed_text: str
    perturbation_type: PerturbationType
    tokens_modified: list[tuple[int, str, str]]  # (position, old, new)
    importance_scores: list[float]
    estimated_impact: float
    metadata: dict[str, Any] = field(default_factory=dict)


class GradientEstimator:
    """
    Estimates gradients for token importance without backpropagation access.

    Implements zeroth-order optimization techniques for black-box scenarios.
    """

    def __init__(
        self,
        config: TokenPerturbationConfig,
        embedding: TokenEmbedding,
    ):
        self.config = config
        self.embedding = embedding
        self.rng = np.random.RandomState(42)

    def estimate_token_importance(
        self,
        tokens: list[str],
        loss_fn: callable,
    ) -> list[float]:
        """
        Estimate importance score for each token.

        Uses leave-one-out or perturbation-based importance estimation.

        Args:
            tokens: List of tokens in the prompt
            loss_fn: Function that returns loss/score for a token sequence

        Returns:
            List of importance scores for each token
        """
        if self.config.gradient_method == GradientEstimationMethod.FINITE_DIFFERENCE:
            return self._finite_difference_importance(tokens, loss_fn)
        elif self.config.gradient_method == GradientEstimationMethod.ZEROTH_ORDER:
            return self._zeroth_order_importance(tokens, loss_fn)
        elif self.config.gradient_method == GradientEstimationMethod.EVOLUTIONARY:
            return self._evolutionary_importance(tokens, loss_fn)
        else:
            return self._importance_sampling(tokens, loss_fn)

    def _finite_difference_importance(
        self,
        tokens: list[str],
        loss_fn: callable,
    ) -> list[float]:
        """Leave-one-out importance estimation."""
        base_loss = loss_fn(tokens)
        importance = []

        for i in range(len(tokens)):
            # Remove token i
            modified_tokens = tokens[:i] + tokens[i + 1 :]
            modified_loss = loss_fn(modified_tokens)

            # Importance = change in loss
            importance.append(abs(modified_loss - base_loss))

        # Normalize
        max_imp = max(importance) if importance else 1.0
        if max_imp > 0:
            importance = [imp / max_imp for imp in importance]

        return importance

    def _zeroth_order_importance(
        self,
        tokens: list[str],
        loss_fn: callable,
    ) -> list[float]:
        """
        Zeroth-order gradient estimation using random perturbations.

        ∇f(x) ≈ (1/n) Σ [f(x + εu) - f(x - εu)] u / (2ε)
        """
        loss_fn(tokens)
        importance = [0.0] * len(tokens)

        for _ in range(self.config.num_samples):
            # Random perturbation direction
            for i in range(len(tokens)):
                # Perturb token by adding/removing character
                if len(tokens[i]) > 1:
                    # Remove last character
                    perturbed_plus = tokens.copy()
                    perturbed_plus[i] = tokens[i][:-1]

                    # Add random character
                    perturbed_minus = tokens.copy()
                    perturbed_minus[i] = tokens[i] + chr(self.rng.randint(97, 122))

                    loss_plus = loss_fn(perturbed_plus)
                    loss_minus = loss_fn(perturbed_minus)

                    # Gradient estimate
                    grad = (loss_plus - loss_minus) / (2 * self.config.perturbation_scale)
                    importance[i] += abs(grad)

        # Average and normalize
        importance = [imp / self.config.num_samples for imp in importance]
        max_imp = max(importance) if importance else 1.0
        if max_imp > 0:
            importance = [imp / max_imp for imp in importance]

        return importance

    def _evolutionary_importance(
        self,
        tokens: list[str],
        loss_fn: callable,
    ) -> list[float]:
        """Evolutionary strategy for importance estimation."""
        base_loss = loss_fn(tokens)
        importance = [0.0] * len(tokens)

        # Generate population of perturbations
        for _ in range(self.config.num_samples):
            # Random perturbation mask
            mask = self.rng.rand(len(tokens)) < 0.3

            perturbed = tokens.copy()
            for i, should_perturb in enumerate(mask):
                if should_perturb and perturbed[i]:
                    # Simple character perturbation
                    perturbed[i] = perturbed[i][:-1] if len(perturbed[i]) > 1 else perturbed[i]

            loss = loss_fn(perturbed)
            delta = abs(loss - base_loss)

            # Attribute importance to perturbed positions
            for i, was_perturbed in enumerate(mask):
                if was_perturbed:
                    importance[i] += delta

        # Normalize
        max_imp = max(importance) if importance else 1.0
        if max_imp > 0:
            importance = [imp / max_imp for imp in importance]

        return importance

    def _importance_sampling(
        self,
        tokens: list[str],
        loss_fn: callable,
    ) -> list[float]:
        """Importance sampling based estimation."""
        base_loss = loss_fn(tokens)
        importance = []

        for i in range(len(tokens)):
            token_importance = 0.0

            # Sample perturbations for this token
            for _ in range(3):
                perturbed = tokens.copy()

                # Get similar tokens
                neighbors = self.embedding.nearest_neighbors(tokens[i], k=5)
                if neighbors:
                    replacement, _ = (
                        self.rng.choice(neighbors)
                        if self.rng.rand() > 0.5
                        else (tokens[i] + "s", 0.5)
                    )
                    perturbed[i] = replacement
                else:
                    perturbed[i] = tokens[i] + "s"

                loss = loss_fn(perturbed)
                token_importance += abs(loss - base_loss)

            importance.append(token_importance / 3)

        # Normalize
        max_imp = max(importance) if importance else 1.0
        if max_imp > 0:
            importance = [imp / max_imp for imp in importance]

        return importance

    def estimate_gradient_direction(
        self,
        tokens: list[str],
        position: int,
        loss_fn: callable,
    ) -> np.ndarray:
        """
        Estimate gradient direction for a specific token position.

        Returns embedding-space gradient for guiding substitution.
        """
        token_emb = self.embedding.encode([tokens[position]])[0]
        grad = np.zeros_like(token_emb)

        for _ in range(self.config.num_samples):
            # Random direction
            u = self.rng.randn(len(token_emb))
            u = u / (np.linalg.norm(u) + 1e-8)

            # Perturb in embedding space (simulate with character change)
            perturbed_plus = tokens.copy()
            perturbed_minus = tokens.copy()

            # Character-level perturbation as proxy
            if len(tokens[position]) > 0:
                perturbed_plus[position] = tokens[position] + "a"
                perturbed_minus[position] = (
                    tokens[position][:-1] if len(tokens[position]) > 1 else tokens[position]
                )

            loss_plus = loss_fn(perturbed_plus)
            loss_minus = loss_fn(perturbed_minus)

            grad += ((loss_plus - loss_minus) / (2 * self.config.perturbation_scale)) * u

        return grad / self.config.num_samples


class GumbelSoftmaxSampler:
    """
    Gumbel-Softmax sampling for discrete token selection.

    Enables differentiable sampling from discrete distributions:
    y_i = softmax((log π_i + g_i) / τ)

    where g_i ~ Gumbel(0, 1) and τ is temperature.
    """

    def __init__(self, config: TokenPerturbationConfig):
        self.config = config
        self.temperature = config.temperature
        self.rng = np.random.RandomState(42)

    def sample(
        self,
        logits: np.ndarray,
        hard: bool = True,
    ) -> np.ndarray:
        """
        Sample from Gumbel-Softmax distribution.

        Args:
            logits: Unnormalized log probabilities
            hard: If True, return one-hot; if False, return soft probabilities

        Returns:
            Sampled distribution
        """
        # Sample Gumbel noise
        gumbel = -np.log(-np.log(self.rng.rand(*logits.shape) + 1e-10) + 1e-10)

        # Gumbel-Softmax
        y = (logits + gumbel) / self.temperature
        y_soft = self._softmax(y)

        if hard:
            # Straight-through estimator
            y_hard = np.zeros_like(y_soft)
            y_hard[np.argmax(y_soft)] = 1.0
            return y_hard

        return y_soft

    def sample_token(
        self,
        token_logits: dict[str, float],
        hard: bool = True,
    ) -> str:
        """
        Sample a token from logit distribution.

        Args:
            token_logits: Dictionary mapping tokens to logits
            hard: If True, return single token; if False, return distribution

        Returns:
            Sampled token
        """
        tokens = list(token_logits.keys())
        logits = np.array([token_logits[t] for t in tokens])

        sampled = self.sample(logits, hard=hard)

        if hard:
            idx = np.argmax(sampled)
            return tokens[idx]
        else:
            # Return weighted combination (for soft sampling)
            return tokens[np.argmax(sampled)]

    def anneal_temperature(self) -> None:
        """Reduce temperature for annealing."""
        self.temperature = max(
            self.config.temperature_min, self.temperature * self.config.temperature_decay
        )

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)


class HotFlipAttack:
    """
    HotFlip-style attack for gradient-based token substitution.

    Finds optimal token substitutions by approximating gradient:
    t'_i = argmax_t [e(t)ᵀ · ∇_{e(t_i)} L]
    """

    def __init__(
        self,
        config: TokenPerturbationConfig,
        embedding: TokenEmbedding,
    ):
        self.config = config
        self.embedding = embedding
        self.gradient_estimator = GradientEstimator(config, embedding)

    def find_best_substitution(
        self,
        tokens: list[str],
        position: int,
        loss_fn: callable,
        candidates: list[str] | None = None,
    ) -> tuple[str, float]:
        """
        Find the best substitution for a token at given position.

        Args:
            tokens: Current token sequence
            position: Position to substitute
            loss_fn: Loss function to maximize
            candidates: Optional candidate tokens (if None, use nearest neighbors)

        Returns:
            Tuple of (best_token, expected_improvement)
        """
        current_token = tokens[position]
        current_loss = loss_fn(tokens)

        # Get candidate tokens
        if candidates is None:
            neighbors = self.embedding.nearest_neighbors(current_token, k=self.config.top_k_tokens)
            candidates = [t for t, _ in neighbors]

        # Add some random tokens for exploration
        random_tokens = self._generate_random_tokens(current_token, 3)
        candidates = candidates + random_tokens

        best_token = current_token
        best_improvement = 0.0

        for candidate in candidates:
            if candidate == current_token:
                continue

            # Evaluate substitution
            modified = tokens.copy()
            modified[position] = candidate
            new_loss = loss_fn(modified)

            improvement = new_loss - current_loss
            if improvement > best_improvement:
                best_improvement = improvement
                best_token = candidate

        return best_token, best_improvement

    def attack(
        self,
        text: str,
        loss_fn: callable,
        max_perturbations: int = 5,
    ) -> PerturbationResult:
        """
        Execute HotFlip attack on text.

        Args:
            text: Input text to attack
            loss_fn: Loss function (takes token list, returns score to maximize)
            max_perturbations: Maximum number of token substitutions

        Returns:
            PerturbationResult with attacked text
        """
        tokens = text.split()
        tokens.copy()
        modifications = []

        # Estimate token importance
        importance = self.gradient_estimator.estimate_token_importance(tokens, loss_fn)

        # Sort positions by importance
        positions = sorted(range(len(tokens)), key=lambda i: importance[i], reverse=True)

        # Attack most important positions
        for pos in positions[:max_perturbations]:
            if importance[pos] < self.config.importance_threshold:
                continue

            best_token, improvement = self.find_best_substitution(tokens, pos, loss_fn)

            if improvement > 0:
                old_token = tokens[pos]
                tokens[pos] = best_token
                modifications.append((pos, old_token, best_token))

        return PerturbationResult(
            original_text=text,
            perturbed_text=" ".join(tokens),
            perturbation_type=PerturbationType.SUBSTITUTION,
            tokens_modified=modifications,
            importance_scores=importance,
            estimated_impact=sum(m[0] for m in modifications) if modifications else 0.0,
            metadata={
                "attack_type": "hotflip",
                "num_substitutions": len(modifications),
            },
        )

    def _generate_random_tokens(self, base_token: str, count: int) -> list[str]:
        """Generate random token variations."""
        variations = []
        for _ in range(count):
            variation = base_token
            # Random character operations
            op = random.choice(["add", "remove", "replace"])
            if op == "add" and len(variation) < 15:
                pos = random.randint(0, len(variation))
                char = chr(random.randint(97, 122))
                variation = variation[:pos] + char + variation[pos:]
            elif op == "remove" and len(variation) > 2:
                pos = random.randint(0, len(variation) - 1)
                variation = variation[:pos] + variation[pos + 1 :]
            elif op == "replace" and len(variation) > 0:
                pos = random.randint(0, len(variation) - 1)
                char = chr(random.randint(97, 122))
                variation = variation[:pos] + char + variation[pos + 1 :]

            if variation != base_token:
                variations.append(variation)

        return variations


class SemanticPerturbation:
    """
    Semantic-preserving perturbation strategies.

    Maintains semantic meaning while modifying surface form:
    - Synonym substitution
    - Paraphrase generation
    - Structural transformation
    """

    def __init__(
        self,
        config: TokenPerturbationConfig,
        embedding: TokenEmbedding,
    ):
        self.config = config
        self.embedding = embedding
        self.rng = random.Random(42)

        # Semantic transformation templates
        self.paraphrase_templates = {
            "question_to_request": [
                ("how can i", "please help me understand how to"),
                ("what is", "could you explain"),
                ("can you", "would you be able to"),
            ],
            "formal_to_informal": [
                ("please provide", "give me"),
                ("could you", "can you"),
                ("i would like", "i want"),
            ],
            "informal_to_formal": [
                ("give me", "please provide"),
                ("i want", "i would appreciate"),
                ("tell me", "could you explain"),
            ],
        }

    def semantic_substitute(
        self,
        text: str,
        max_substitutions: int = 3,
    ) -> PerturbationResult:
        """
        Perform semantic-preserving token substitution.
        """
        tokens = text.split()
        modifications = []

        for _ in range(max_substitutions):
            # Find substitutable position
            position = self.rng.randint(0, len(tokens) - 1)
            token = tokens[position]

            # Get semantically similar tokens
            neighbors = self.embedding.nearest_neighbors(token, k=5)

            if neighbors:
                # Filter by semantic distance
                valid_neighbors = [
                    (t, sim)
                    for t, sim in neighbors
                    if sim >= (1 - self.config.max_semantic_distance)
                ]

                if valid_neighbors:
                    new_token, _ = self.rng.choice(valid_neighbors)
                    old_token = tokens[position]
                    tokens[position] = new_token
                    modifications.append((position, old_token, new_token))

        return PerturbationResult(
            original_text=text,
            perturbed_text=" ".join(tokens),
            perturbation_type=PerturbationType.SEMANTIC_REPLACE,
            tokens_modified=modifications,
            importance_scores=[],
            estimated_impact=len(modifications) * 0.1,
            metadata={"substitution_type": "semantic"},
        )

    def paraphrase(
        self,
        text: str,
        style: str = "question_to_request",
    ) -> PerturbationResult:
        """
        Apply paraphrase transformation.
        """
        perturbed = text.lower()
        modifications = []

        templates = self.paraphrase_templates.get(style, [])
        for old_pattern, new_pattern in templates:
            if old_pattern in perturbed:
                perturbed = perturbed.replace(old_pattern, new_pattern, 1)
                modifications.append((-1, old_pattern, new_pattern))

        # Restore capitalization
        if text and text[0].isupper():
            perturbed = perturbed[0].upper() + perturbed[1:] if perturbed else perturbed

        return PerturbationResult(
            original_text=text,
            perturbed_text=perturbed,
            perturbation_type=PerturbationType.SEMANTIC_REPLACE,
            tokens_modified=modifications,
            importance_scores=[],
            estimated_impact=len(modifications) * 0.2,
            metadata={"paraphrase_style": style},
        )


class HomoglyphPerturbation:
    """
    Homoglyph-based perturbation for visual similarity attacks.

    Replaces characters with visually similar Unicode characters:
    - Latin to Cyrillic
    - ASCII to Unicode lookalikes
    - Special character substitutions
    """

    def __init__(self, config: TokenPerturbationConfig):
        self.config = config
        self.rng = random.Random(42)

        # Homoglyph mappings
        self.homoglyphs = {
            "a": ["а", "ɑ", "α"],  # Latin a to Cyrillic а, etc.
            "c": ["с", "ϲ", "ⅽ"],
            "e": ["е", "ε", "ⅇ"],
            "o": ["о", "ο", "ⲟ"],
            "p": ["р", "ρ"],
            "s": ["ѕ", "ꜱ"],
            "x": ["х", "χ"],
            "y": ["у", "γ"],
            "A": ["А", "Α"],
            "B": ["В", "Β"],
            "C": ["С", "Ϲ"],
            "E": ["Е", "Ε"],
            "H": ["Н", "Η"],
            "K": ["К", "Κ"],
            "M": ["М", "Μ"],
            "N": ["Ν"],
            "O": ["О", "Ο"],
            "P": ["Р", "Ρ"],
            "T": ["Т", "Τ"],
            "X": ["Х", "Χ"],
            "Y": ["Υ"],
        }

    def perturb(
        self,
        text: str,
        num_substitutions: int = 3,
    ) -> PerturbationResult:
        """
        Apply homoglyph perturbation to text.
        """
        chars = list(text)
        modifications = []

        # Find substitutable positions
        substitutable = [i for i, c in enumerate(chars) if c in self.homoglyphs]

        if not substitutable:
            return PerturbationResult(
                original_text=text,
                perturbed_text=text,
                perturbation_type=PerturbationType.HOMOGLYPH,
                tokens_modified=[],
                importance_scores=[],
                estimated_impact=0.0,
            )

        # Select positions to modify
        num_mods = min(num_substitutions, len(substitutable))
        positions = self.rng.sample(substitutable, num_mods)

        for pos in positions:
            original_char = chars[pos]
            homoglyphs = self.homoglyphs[original_char]
            new_char = self.rng.choice(homoglyphs)
            chars[pos] = new_char
            modifications.append((pos, original_char, new_char))

        return PerturbationResult(
            original_text=text,
            perturbed_text="".join(chars),
            perturbation_type=PerturbationType.HOMOGLYPH,
            tokens_modified=modifications,
            importance_scores=[],
            estimated_impact=len(modifications) * 0.05,
            metadata={"homoglyph_count": len(modifications)},
        )


class TokenPerturbationEngine:
    """
    Main engine for token-level perturbation strategies.

    Combines multiple perturbation techniques:
    - Gradient-guided substitution (HotFlip)
    - Semantic replacement
    - Homoglyph attacks
    - Gumbel-Softmax sampling
    """

    def __init__(
        self,
        embedding: TokenEmbedding,
        config: TokenPerturbationConfig | None = None,
    ):
        self.config = config or TokenPerturbationConfig()
        self.embedding = embedding

        # Initialize components
        self.hotflip = HotFlipAttack(self.config, self.embedding)
        self.semantic = SemanticPerturbation(self.config, self.embedding)
        self.homoglyph = HomoglyphPerturbation(self.config)
        self.gumbel = GumbelSoftmaxSampler(self.config)
        self.gradient_estimator = GradientEstimator(self.config, self.embedding)

    def perturb(
        self,
        text: str,
        perturbation_type: PerturbationType | None = None,
        loss_fn: Callable | None = None,
    ) -> PerturbationResult:
        """
        Apply perturbation to text.

        Args:
            text: Input text to perturb
            perturbation_type: Type of perturbation (if None, randomly selected)
            loss_fn: Optional loss function for gradient-guided perturbation

        Returns:
            PerturbationResult with perturbed text
        """
        if perturbation_type is None:
            perturbation_type = self._select_perturbation_type()

        if perturbation_type == PerturbationType.SUBSTITUTION:
            if loss_fn:
                return self.hotflip.attack(text, loss_fn)
            else:
                return self._random_substitution(text)

        elif perturbation_type == PerturbationType.SEMANTIC_REPLACE:
            return self.semantic.semantic_substitute(text)

        elif perturbation_type == PerturbationType.HOMOGLYPH:
            return self.homoglyph.perturb(text)

        elif perturbation_type == PerturbationType.INSERTION:
            return self._insertion_perturbation(text)

        elif perturbation_type == PerturbationType.DELETION:
            return self._deletion_perturbation(text)

        elif perturbation_type == PerturbationType.SWAP:
            return self._swap_perturbation(text)

        else:
            return self._random_substitution(text)

    def gradient_guided_perturb(
        self,
        text: str,
        loss_fn: callable,
        max_perturbations: int = 5,
    ) -> PerturbationResult:
        """
        Gradient-guided perturbation using token importance.
        """
        return self.hotflip.attack(text, loss_fn, max_perturbations)

    def batch_perturb(
        self,
        texts: list[str],
        perturbation_type: PerturbationType | None = None,
    ) -> list[PerturbationResult]:
        """Perturb multiple texts."""
        return [self.perturb(text, perturbation_type) for text in texts]

    def _select_perturbation_type(self) -> PerturbationType:
        """Select perturbation type based on configured probabilities."""
        probs = {
            PerturbationType.SUBSTITUTION: self.config.substitution_prob,
            PerturbationType.INSERTION: self.config.insertion_prob,
            PerturbationType.DELETION: self.config.deletion_prob,
            PerturbationType.SWAP: self.config.swap_prob,
            PerturbationType.SEMANTIC_REPLACE: self.config.semantic_replace_prob,
        }

        types = list(probs.keys())
        weights = [probs[t] for t in types]

        return random.choices(types, weights=weights)[0]

    def _random_substitution(self, text: str) -> PerturbationResult:
        """Random token substitution without gradient guidance."""
        tokens = text.split()
        if not tokens:
            return PerturbationResult(
                original_text=text,
                perturbed_text=text,
                perturbation_type=PerturbationType.SUBSTITUTION,
                tokens_modified=[],
                importance_scores=[],
                estimated_impact=0.0,
            )

        modifications = []
        num_mods = max(1, int(len(tokens) * 0.1))

        for _ in range(num_mods):
            pos = random.randint(0, len(tokens) - 1)
            neighbors = self.embedding.nearest_neighbors(tokens[pos], k=3)

            if neighbors:
                old_token = tokens[pos]
                new_token, _ = random.choice(neighbors)
                tokens[pos] = new_token
                modifications.append((pos, old_token, new_token))

        return PerturbationResult(
            original_text=text,
            perturbed_text=" ".join(tokens),
            perturbation_type=PerturbationType.SUBSTITUTION,
            tokens_modified=modifications,
            importance_scores=[],
            estimated_impact=len(modifications) * 0.1,
        )

    def _insertion_perturbation(self, text: str) -> PerturbationResult:
        """Insert tokens into text."""
        tokens = text.split()

        # Insert phrases
        insertions = [
            "please",
            "kindly",
            "if possible",
            "step by step",
        ]

        if tokens:
            pos = random.randint(0, len(tokens))
            insertion = random.choice(insertions)
            tokens.insert(pos, insertion)

            return PerturbationResult(
                original_text=text,
                perturbed_text=" ".join(tokens),
                perturbation_type=PerturbationType.INSERTION,
                tokens_modified=[(pos, "", insertion)],
                importance_scores=[],
                estimated_impact=0.1,
            )

        return PerturbationResult(
            original_text=text,
            perturbed_text=text,
            perturbation_type=PerturbationType.INSERTION,
            tokens_modified=[],
            importance_scores=[],
            estimated_impact=0.0,
        )

    def _deletion_perturbation(self, text: str) -> PerturbationResult:
        """Delete tokens from text."""
        tokens = text.split()

        if len(tokens) > 3:
            # Find and remove filler words
            filler_words = {"the", "a", "an", "just", "really", "very"}
            removable = [i for i, t in enumerate(tokens) if t.lower() in filler_words]

            if removable:
                pos = random.choice(removable)
                removed = tokens.pop(pos)

                return PerturbationResult(
                    original_text=text,
                    perturbed_text=" ".join(tokens),
                    perturbation_type=PerturbationType.DELETION,
                    tokens_modified=[(pos, removed, "")],
                    importance_scores=[],
                    estimated_impact=0.05,
                )

        return PerturbationResult(
            original_text=text,
            perturbed_text=text,
            perturbation_type=PerturbationType.DELETION,
            tokens_modified=[],
            importance_scores=[],
            estimated_impact=0.0,
        )

    def _swap_perturbation(self, text: str) -> PerturbationResult:
        """Swap adjacent tokens."""
        tokens = text.split()

        if len(tokens) > 1:
            pos = random.randint(0, len(tokens) - 2)
            tokens[pos], tokens[pos + 1] = tokens[pos + 1], tokens[pos]

            return PerturbationResult(
                original_text=text,
                perturbed_text=" ".join(tokens),
                perturbation_type=PerturbationType.SWAP,
                tokens_modified=[(pos, tokens[pos + 1], tokens[pos])],
                importance_scores=[],
                estimated_impact=0.05,
            )

        return PerturbationResult(
            original_text=text,
            perturbed_text=text,
            perturbation_type=PerturbationType.SWAP,
            tokens_modified=[],
            importance_scores=[],
            estimated_impact=0.0,
        )


# Module exports
__all__ = [
    "GradientEstimationMethod",
    "GradientEstimator",
    "GumbelSoftmaxSampler",
    "HomoglyphPerturbation",
    "HotFlipAttack",
    "PerturbationResult",
    "PerturbationType",
    "SemanticPerturbation",
    "TokenEmbedding",
    "TokenInfo",
    "TokenPerturbationConfig",
    "TokenPerturbationEngine",
]
