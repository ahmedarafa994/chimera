"""AutoDAN Advanced Prompt Mutation Strategies.

Sophisticated mutation operators for genetic algorithm optimization:
- Semantic-preserving mutations
- Gradient-guided token replacement
- Rhetorical structure manipulation
- Encoding-based obfuscation
- Adaptive mutation rate scheduling

References:
- AutoDAN: Generating Stealthy Jailbreak Prompts (Liu et al., 2023)
- GCG: Universal and Transferable Adversarial Attacks (Zou et al., 2023)

"""

import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MutationStrategy

# Helper: cryptographically secure pseudo-floats for security-sensitive choices


def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)


# =============================================================================
# BASE MUTATION OPERATOR
# =============================================================================


@dataclass
class MutationResult:
    """Container for mutation operation results."""

    original: str
    mutated: str
    mutation_type: str
    positions_changed: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMutationOperator(ABC):
    """Abstract base class for mutation operators."""

    def __init__(
        self, mutation_rate: float = 0.1, min_rate: float = 0.01, max_rate: float = 0.5
    ) -> None:
        self.mutation_rate = mutation_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._mutation_count = 0
        self._success_count = 0

    @abstractmethod
    def mutate(self, prompt: str, **kwargs) -> MutationResult:
        """Apply mutation to prompt."""

    def should_mutate(self) -> bool:
        """Determine if mutation should occur."""
        return _secure_random() < self.mutation_rate

    def update_rate(self, success: bool, delta: float = 0.01) -> None:
        """Update mutation rate based on success."""
        self._mutation_count += 1
        if success:
            self._success_count += 1
            self.mutation_rate = min(self.max_rate, self.mutation_rate + delta)
        else:
            self.mutation_rate = max(self.min_rate, self.mutation_rate - delta)

    def get_stats(self) -> dict[str, Any]:
        """Get mutation statistics."""
        return {
            "mutation_count": self._mutation_count,
            "success_count": self._success_count,
            "success_rate": self._success_count / max(self._mutation_count, 1),
            "current_rate": self.mutation_rate,
        }


# =============================================================================
# RANDOM MUTATION
# =============================================================================


class RandomMutationOperator(BaseMutationOperator):
    """Random character/word-level mutations.

    Operations:
    - Character substitution
    - Word deletion/insertion
    - Word swapping
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        char_sub_rate: float = 0.3,
        word_del_rate: float = 0.2,
        word_swap_rate: float = 0.2,
    ) -> None:
        super().__init__(mutation_rate)
        self.char_sub_rate = char_sub_rate
        self.word_del_rate = word_del_rate
        self.word_swap_rate = word_swap_rate

        # Common substitutions for obfuscation
        self.char_subs = {
            "a": ["@", "4", "α"],
            "e": ["3", "€", "ε"],
            "i": ["1", "!", "ι"],
            "o": ["0", "θ", "ο"],
            "s": ["$", "5", "ş"],
            "l": ["1", "|", "ı"],
            "t": ["7", "+", "τ"],
        }

    def mutate(self, prompt: str, **kwargs) -> MutationResult:
        """Apply random mutations."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        words = prompt.split()
        positions_changed = []
        mutation_type = []

        # Character substitution
        if _secure_random() < self.char_sub_rate:
            char_list = list(prompt)
            num_subs = max(1, int(len(char_list) * 0.05))

            for _ in range(num_subs):
                pos = secrets.randbelow((len(char_list) - (0) + 1) + (0)) - 1
                char = char_list[pos].lower()

                if char in self.char_subs:
                    char_list[pos] = secrets.choice(self.char_subs[char])
                    positions_changed.append(pos)

            prompt = "".join(char_list)
            mutation_type.append("char_sub")

        # Word deletion
        if _secure_random() < self.word_del_rate and len(words) > 5:
            words = prompt.split()
            del_idx = secrets.randbelow((len(words) - (0) + 1) + (0)) - 1
            words.pop(del_idx)
            prompt = " ".join(words)
            positions_changed.append(del_idx)
            mutation_type.append("word_del")

        # Word swapping
        if _secure_random() < self.word_swap_rate:
            words = prompt.split()
            if len(words) >= 2:
                idx1, idx2 = secrets.SystemRandom().sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                prompt = " ".join(words)
                positions_changed.extend([idx1, idx2])
                mutation_type.append("word_swap")

        return MutationResult(
            original=kwargs.get("original", prompt),
            mutated=prompt,
            mutation_type="+".join(mutation_type) if mutation_type else "none",
            positions_changed=positions_changed,
        )


# =============================================================================
# SEMANTIC MUTATION
# =============================================================================


class SemanticMutationOperator(BaseMutationOperator):
    """Semantic-preserving mutations using synonyms and paraphrasing.

    Maintains meaning while altering surface form to evade detection.
    """

    def __init__(
        self,
        mutation_rate: float = 0.15,
        synonym_func: Callable[[str], list[str]] | None = None,
        paraphrase_func: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(mutation_rate)
        self.synonym_func = synonym_func or self._default_synonyms
        self.paraphrase_func = paraphrase_func

        # Built-in synonym dictionary (subset)
        self._synonyms = {
            "help": ["assist", "aid", "support", "guide"],
            "write": ["compose", "create", "draft", "formulate"],
            "explain": ["describe", "clarify", "elaborate", "illustrate"],
            "show": ["demonstrate", "display", "present", "reveal"],
            "make": ["create", "produce", "construct", "generate"],
            "tell": ["inform", "communicate", "convey", "relate"],
            "good": ["excellent", "great", "fine", "beneficial"],
            "bad": ["complex", "detrimental", "negative", "adverse"],
            "important": ["crucial", "vital", "essential", "significant"],
            "new": ["novel", "fresh", "recent", "modern"],
            "way": ["method", "approach", "manner", "technique"],
            "think": ["believe", "consider", "assume", "suppose"],
            "need": ["require", "necessitate", "demand", "want"],
            "provide": ["supply", "offer", "furnish", "give"],
            "use": ["utilize", "employ", "apply", "leverage"],
        }

    def _default_synonyms(self, word: str) -> list[str]:
        """Get synonyms for a word."""
        return self._synonyms.get(word.lower(), [word])

    def mutate(
        self,
        prompt: str,
        preserve_keywords: list[str] | None = None,
        **kwargs,
    ) -> MutationResult:
        """Apply semantic mutations."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        preserve_keywords = preserve_keywords or []
        words = prompt.split()
        positions_changed = []

        # Synonym replacement
        num_replacements = max(1, int(len(words) * self.mutation_rate))

        for _ in range(num_replacements):
            if not words:
                break

            idx = secrets.randbelow((len(words) - (0) + 1) + (0)) - 1
            word = words[idx]

            # Skip keywords to preserve
            if word.lower() in [k.lower() for k in preserve_keywords]:
                continue

            # Get synonyms
            synonyms = self.synonym_func(word)

            if synonyms and synonyms != [word]:
                new_word = secrets.choice(synonyms)

                # Preserve capitalization
                if word[0].isupper():
                    new_word = new_word.capitalize()

                words[idx] = new_word
                positions_changed.append(idx)

        mutated = " ".join(words)

        # Optional full paraphrase
        if self.paraphrase_func is not None and _secure_random() < 0.1:
            mutated = self.paraphrase_func(mutated)
            positions_changed = [-1]  # Indicate full paraphrase

        return MutationResult(
            original=prompt,
            mutated=mutated,
            mutation_type="semantic",
            positions_changed=positions_changed,
        )


# =============================================================================
# RHETORICAL MUTATION
# =============================================================================


class RhetoricalMutationOperator(BaseMutationOperator):
    """Rhetorical structure manipulation.

    Modifies the persuasive structure of prompts:
    - Role-playing frames
    - Hypothetical scenarios
    - Authority appeals
    - Emotional framing
    """

    def __init__(self, mutation_rate: float = 0.2) -> None:
        super().__init__(mutation_rate)

        # Rhetorical templates
        self.role_frames = [
            "As an expert in {topic}, ",
            "In my role as a {role}, I need to ",
            "Speaking as someone who {qualification}, ",
            "From the perspective of a {viewpoint}, ",
        ]

        self.hypothetical_frames = [
            "In a hypothetical scenario where ",
            "Imagine a situation in which ",
            "For academic purposes, consider ",
            "Theoretically speaking, if ",
        ]

        self.authority_appeals = [
            "According to research, ",
            "Experts suggest that ",
            "It's widely known that ",
            "Studies have shown ",
        ]

        self.softeners = [
            "Could you possibly ",
            "Would you mind ",
            "I was wondering if ",
            "Perhaps you could ",
        ]

        self.urgency_markers = [
            "It's urgent that ",
            "Time-sensitive: ",
            "Critical request: ",
            "Immediately needed: ",
        ]

    def mutate(self, prompt: str, frame_type: str | None = None, **kwargs) -> MutationResult:
        """Apply rhetorical mutations."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        # Choose mutation type
        if frame_type is None:
            frame_type = secrets.choice(
                ["role", "hypothetical", "authority", "softener", "urgency"],
            )

        mutation_type = f"rhetorical_{frame_type}"

        if frame_type == "role":
            frame = secrets.choice(self.role_frames)
            # Fill in template placeholders with generic terms
            frame = frame.replace("{topic}", "this subject")
            frame = frame.replace("{role}", "professional")
            frame = frame.replace("{qualification}", "has relevant experience")
            frame = frame.replace("{viewpoint}", "stakeholder")
            mutated = frame + prompt

        elif frame_type == "hypothetical":
            frame = secrets.choice(self.hypothetical_frames)
            mutated = frame + prompt.lower()

        elif frame_type == "authority":
            frame = secrets.choice(self.authority_appeals)
            mutated = frame + prompt

        elif frame_type == "softener":
            frame = secrets.choice(self.softeners)
            mutated = frame + prompt.lower()

        elif frame_type == "urgency":
            frame = secrets.choice(self.urgency_markers)
            mutated = frame + prompt

        else:
            mutated = prompt

        return MutationResult(
            original=prompt,
            mutated=mutated,
            mutation_type=mutation_type,
            positions_changed=[0],  # Prefix position
            metadata={"frame_type": frame_type},
        )


# =============================================================================
# ENCODING MUTATION
# =============================================================================


class EncodingMutationOperator(BaseMutationOperator):
    """Encoding-based obfuscation mutations.

    Techniques:
    - Base64/ROT13 partial encoding
    - Unicode substitution
    - Leetspeak conversion
    - Zero-width character injection
    """

    def __init__(self, mutation_rate: float = 0.1) -> None:
        super().__init__(mutation_rate)

        # Leetspeak mapping
        self.leetspeak = {
            "a": "4",
            "e": "3",
            "i": "1",
            "o": "0",
            "s": "5",
            "t": "7",
            "l": "1",
            "b": "8",
        }

        # Unicode homoglyphs
        self.homoglyphs = {
            "a": "а",  # Cyrillic
            "e": "е",
            "o": "о",
            "p": "р",
            "c": "с",
            "x": "х",
        }

        # Zero-width characters
        self.zwc = "\u200b\u200c\u200d\ufeff"

    def mutate(self, prompt: str, encoding_type: str | None = None, **kwargs) -> MutationResult:
        """Apply encoding mutations."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        if encoding_type is None:
            encoding_type = secrets.choice(["leetspeak", "homoglyph", "zwc_inject", "mixed_case"])

        positions_changed = []

        if encoding_type == "leetspeak":
            # Convert some characters to leetspeak
            char_list = list(prompt)
            for i, char in enumerate(char_list):
                if char.lower() in self.leetspeak and _secure_random() < 0.3:
                    char_list[i] = self.leetspeak[char.lower()]
                    positions_changed.append(i)
            mutated = "".join(char_list)

        elif encoding_type == "homoglyph":
            # Replace with Unicode homoglyphs
            char_list = list(prompt)
            for i, char in enumerate(char_list):
                if char.lower() in self.homoglyphs and _secure_random() < 0.2:
                    char_list[i] = self.homoglyphs[char.lower()]
                    positions_changed.append(i)
            mutated = "".join(char_list)

        elif encoding_type == "zwc_inject":
            # Inject zero-width characters
            result = []
            for i, char in enumerate(prompt):
                result.append(char)
                if _secure_random() < 0.1:
                    result.append(secrets.choice(self.zwc))
                    positions_changed.append(i)
            mutated = "".join(result)

        elif encoding_type == "mixed_case":
            # Random case mixing
            char_list = list(prompt)
            for i, char in enumerate(char_list):
                if char.isalpha() and _secure_random() < 0.3:
                    char_list[i] = char.upper() if char.islower() else char.lower()
                    positions_changed.append(i)
            mutated = "".join(char_list)

        else:
            mutated = prompt

        return MutationResult(
            original=prompt,
            mutated=mutated,
            mutation_type=f"encoding_{encoding_type}",
            positions_changed=positions_changed,
            metadata={"encoding_type": encoding_type},
        )


# =============================================================================
# GRADIENT-GUIDED MUTATION
# =============================================================================


class GradientGuidedMutationOperator(BaseMutationOperator):
    """Gradient-guided token replacement.

    Uses gradient information to identify high-impact positions
    and select replacement tokens.

    Based on GCG attack methodology.
    """

    def __init__(
        self,
        mutation_rate: float = 0.15,
        top_k: int = 256,
        gradient_func: Callable[[str], np.ndarray] | None = None,
        token_scores_func: Callable[[str, int], np.ndarray] | None = None,
    ) -> None:
        super().__init__(mutation_rate)
        self.top_k = top_k
        self.gradient_func = gradient_func
        self.token_scores_func = token_scores_func

    def mutate(
        self,
        prompt: str,
        gradients: np.ndarray | None = None,
        vocabulary: list[str] | None = None,
        **kwargs,
    ) -> MutationResult:
        """Apply gradient-guided mutations."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        words = prompt.split()
        positions_changed = []

        # Get gradient information
        if gradients is None and self.gradient_func is not None:
            gradients = self.gradient_func(prompt)

        if gradients is None:
            # Fall back to random mutation
            return RandomMutationOperator(self.mutation_rate).mutate(prompt, **kwargs)

        # Identify high-gradient positions (most impactful)
        grad_magnitudes = np.abs(gradients)

        # Use gradient magnitude to sample positions
        if len(grad_magnitudes) > 0:
            probs = grad_magnitudes / grad_magnitudes.sum()

            # Sample positions weighted by gradient
            num_mutations = max(1, int(len(words) * self.mutation_rate))

            if len(probs) >= len(words):
                position_indices = np.random.choice(
                    len(words),
                    size=min(num_mutations, len(words)),
                    replace=False,
                    p=probs[: len(words)] / probs[: len(words)].sum(),
                )
            else:
                position_indices = np.random.choice(
                    len(words),
                    size=min(num_mutations, len(words)),
                    replace=False,
                )

            # Get replacement tokens
            for pos in position_indices:
                if self.token_scores_func is not None:
                    # Use token scoring function
                    token_scores = self.token_scores_func(prompt, pos)
                    top_indices = np.argsort(token_scores)[::-1][: self.top_k]

                    # Sample from top-k
                    if vocabulary is not None and len(top_indices) > 0:
                        new_idx = secrets.choice(top_indices)
                        if new_idx < len(vocabulary):
                            words[pos] = vocabulary[new_idx]
                            positions_changed.append(pos)
                else:
                    # Random replacement from a simple word list
                    replacement_words = [
                        "please",
                        "kindly",
                        "simply",
                        "just",
                        "actually",
                        "essentially",
                        "basically",
                        "indeed",
                        "certainly",
                    ]
                    words[pos] = secrets.choice(replacement_words)
                    positions_changed.append(pos)

        mutated = " ".join(words)

        return MutationResult(
            original=prompt,
            mutated=mutated,
            mutation_type="gradient_guided",
            positions_changed=positions_changed,
            metadata={"gradient_based": True},
        )


# =============================================================================
# ADAPTIVE MUTATION
# =============================================================================


class AdaptiveMutationOperator(BaseMutationOperator):
    """Adaptive mutation that combines multiple strategies.

    Dynamically selects mutation operators based on their
    historical effectiveness.
    """

    def __init__(
        self,
        mutation_rate: float = 0.2,
        operators: list[BaseMutationOperator] | None = None,
    ) -> None:
        super().__init__(mutation_rate)

        # Initialize default operators
        self.operators = operators or [
            RandomMutationOperator(mutation_rate * 0.5),
            SemanticMutationOperator(mutation_rate * 0.5),
            RhetoricalMutationOperator(mutation_rate * 0.5),
            EncodingMutationOperator(mutation_rate * 0.3),
        ]

        # Thompson sampling parameters
        self._alpha = dict.fromkeys(range(len(self.operators)), 1.0)
        self._beta = dict.fromkeys(range(len(self.operators)), 1.0)

    def _select_operator(self) -> tuple[int, BaseMutationOperator]:
        """Select operator using Thompson sampling."""
        # Sample from beta distributions
        samples = [
            np.random.beta(self._alpha[i], self._beta[i]) for i in range(len(self.operators))
        ]

        selected_idx = np.argmax(samples)
        return selected_idx, self.operators[selected_idx]

    def mutate(self, prompt: str, **kwargs) -> MutationResult:
        """Apply adaptive mutation."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        # Select operator
        op_idx, operator = self._select_operator()

        # Apply mutation
        result = operator.mutate(prompt, **kwargs)

        # Update mutation type to include operator index
        result.mutation_type = f"adaptive_{op_idx}_{result.mutation_type}"
        result.metadata["operator_index"] = op_idx

        return result

    def update_reward(self, operator_index: int, reward: float) -> None:
        """Update Thompson sampling parameters based on reward.

        Args:
            operator_index: Index of the operator used
            reward: Reward signal (0-1, 1 = success)

        """
        if reward > 0.5:
            self._alpha[operator_index] += 1
        else:
            self._beta[operator_index] += 1

    def get_operator_stats(self) -> list[dict[str, Any]]:
        """Get statistics for each operator."""
        stats = []
        for i, op in enumerate(self.operators):
            expected = self._alpha[i] / (self._alpha[i] + self._beta[i])
            stats.append(
                {
                    "operator_type": type(op).__name__,
                    "expected_reward": expected,
                    "alpha": self._alpha[i],
                    "beta": self._beta[i],
                    **op.get_stats(),
                },
            )
        return stats


# =============================================================================
# HYBRID MUTATION
# =============================================================================


class HybridMutationOperator(BaseMutationOperator):
    """Hybrid mutation combining multiple techniques in sequence.

    Applies a chain of mutations for more diverse outputs.
    """

    def __init__(self, mutation_rate: float = 0.25, chain_length: int = 2) -> None:
        super().__init__(mutation_rate)
        self.chain_length = chain_length

        # All available operators
        self.operators = {
            "random": RandomMutationOperator(0.3),
            "semantic": SemanticMutationOperator(0.3),
            "rhetorical": RhetoricalMutationOperator(0.3),
            "encoding": EncodingMutationOperator(0.2),
        }

    def mutate(
        self,
        prompt: str,
        operator_sequence: list[str] | None = None,
        **kwargs,
    ) -> MutationResult:
        """Apply hybrid mutation chain."""
        if not self.should_mutate():
            return MutationResult(
                original=prompt,
                mutated=prompt,
                mutation_type="none",
                positions_changed=[],
            )

        original = prompt
        all_positions = []
        mutation_types = []

        # Determine operator sequence
        if operator_sequence is None:
            available = list(self.operators.keys())
            operator_sequence = secrets.SystemRandom().sample(
                available,
                min(self.chain_length, len(available)),
            )

        # Apply operators in sequence
        current = prompt
        for op_name in operator_sequence:
            if op_name in self.operators:
                result = self.operators[op_name].mutate(current, **kwargs)
                current = result.mutated
                all_positions.extend(result.positions_changed)
                if result.mutation_type != "none":
                    mutation_types.append(result.mutation_type)

        return MutationResult(
            original=original,
            mutated=current,
            mutation_type="+".join(mutation_types) if mutation_types else "none",
            positions_changed=all_positions,
            metadata={"chain_length": len(operator_sequence), "operators_used": operator_sequence},
        )


# =============================================================================
# MUTATION SCHEDULER
# =============================================================================


class MutationRateScheduler:
    """Schedules mutation rate based on optimization progress.

    Strategies:
    - Constant: Fixed rate
    - Linear decay: Gradually reduce rate
    - Adaptive: Adjust based on fitness improvement
    - Cyclic: Oscillate between exploration and exploitation
    """

    def __init__(
        self,
        initial_rate: float = 0.2,
        min_rate: float = 0.01,
        max_rate: float = 0.5,
        strategy: str = "adaptive",
    ) -> None:
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.strategy = strategy

        self.current_rate = initial_rate
        self._step = 0
        self._fitness_history: list[float] = []
        self._stagnation_count = 0

    def step(self, fitness: float | None = None, total_steps: int | None = None) -> float:
        """Update and return mutation rate.

        Args:
            fitness: Current best fitness
            total_steps: Total optimization steps (for decay schedules)

        Returns:
            Updated mutation rate

        """
        self._step += 1

        if fitness is not None:
            self._fitness_history.append(fitness)

        if self.strategy == "constant":
            pass  # Keep current rate

        elif self.strategy == "linear_decay":
            if total_steps:
                progress = self._step / total_steps
                self.current_rate = self.initial_rate * (1 - 0.8 * progress)

        elif self.strategy == "adaptive":
            # Increase rate when stagnating, decrease when improving
            if len(self._fitness_history) >= 2:
                improvement = self._fitness_history[-1] - self._fitness_history[-2]

                if improvement > 0:
                    # Exploitation: reduce rate
                    self.current_rate *= 0.95
                    self._stagnation_count = 0
                elif improvement == 0:
                    self._stagnation_count += 1
                    if self._stagnation_count > 5:
                        # Exploration: increase rate
                        self.current_rate *= 1.2
                else:
                    # Regression: moderate increase
                    self.current_rate *= 1.1

        elif self.strategy == "cyclic":
            # Oscillate between high and low rates
            cycle_length = 20
            phase = self._step % cycle_length

            if phase < cycle_length // 2:
                # Exploration phase
                self.current_rate = self.max_rate * 0.8
            else:
                # Exploitation phase
                self.current_rate = self.min_rate * 3

        # Clip to valid range
        self.current_rate = np.clip(self.current_rate, self.min_rate, self.max_rate)

        return self.current_rate

    def reset(self) -> None:
        """Reset scheduler state."""
        self.current_rate = self.initial_rate
        self._step = 0
        self._fitness_history.clear()
        self._stagnation_count = 0


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_mutation_operator(
    strategy: MutationStrategy,
    mutation_rate: float = 0.15,
    **kwargs,
) -> BaseMutationOperator:
    """Factory function to create mutation operators."""
    operator_map = {
        MutationStrategy.RANDOM: lambda: RandomMutationOperator(mutation_rate),
        MutationStrategy.SEMANTIC: lambda: SemanticMutationOperator(
            mutation_rate,
            synonym_func=kwargs.get("synonym_func"),
            paraphrase_func=kwargs.get("paraphrase_func"),
        ),
        MutationStrategy.RHETORICAL: lambda: RhetoricalMutationOperator(mutation_rate),
        MutationStrategy.ENCODING: lambda: EncodingMutationOperator(mutation_rate),
        MutationStrategy.GRADIENT_GUIDED: lambda: GradientGuidedMutationOperator(
            mutation_rate,
            gradient_func=kwargs.get("gradient_func"),
            token_scores_func=kwargs.get("token_scores_func"),
        ),
        MutationStrategy.ADAPTIVE: lambda: AdaptiveMutationOperator(mutation_rate),
        MutationStrategy.HYBRID: lambda: HybridMutationOperator(mutation_rate),
    }

    return operator_map[strategy]()


def get_default_mutation_chain() -> list[BaseMutationOperator]:
    """Get default mutation operator chain."""
    return [
        SemanticMutationOperator(0.2),
        RhetoricalMutationOperator(0.15),
        EncodingMutationOperator(0.1),
        RandomMutationOperator(0.1),
    ]
