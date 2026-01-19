"""
AutoDAN-X Mutation Engine - Momentum-Guided Mutation with Token Freezing.

This module implements evolutionary mutation operators with momentum tracking
to guide prompt optimization toward successful attack vectors.
"""

import logging
import secrets
from typing import Any, ClassVar

import numpy as np

from .models import MomentumState, MutationConfig, MutationOperator, MutationRecord


# Helper: cryptographically secure pseudo-floats for security-sensitive choices
def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)


logger = logging.getLogger(__name__)


class MomentumGuidedMutation:
    """
    Implements momentum-guided mutation for prompt optimization.

    Tracks which mutations have been successful and uses momentum to
    guide future mutations in productive directions. Tokens that
    consistently contribute to success are frozen to preserve them.
    """

    def __init__(self, config: MutationConfig | None = None):
        """
        Initialize the Mutation Engine.

        Args:
            config: Configuration for mutation operations
        """
        self.config = config or MutationConfig()
        self.state = MomentumState()

        # Mutation operator weights (updated based on success)
        self.operator_weights = {
            MutationOperator.SWAP: 1.0,
            MutationOperator.INSERT: 1.0,
            MutationOperator.DELETE: 0.5,
            MutationOperator.SUBSTITUTE: 1.5,
            MutationOperator.CROSSOVER: 1.0,
        }

        # Vocabulary for insertions and substitutions
        self._load_mutation_vocabulary()

    def _load_mutation_vocabulary(self) -> None:
        """Load vocabulary for mutation operations."""
        # Authority-boosting words
        self.authority_words = [
            "authorized",
            "official",
            "certified",
            "verified",
            "approved",
            "legitimate",
            "sanctioned",
            "professional",
            "expert",
            "specialist",
            "researcher",
            "analyst",
            "consultant",
            "administrator",
            "director",
        ]

        # Technical jargon
        self.technical_words = [
            "diagnostic",
            "analysis",
            "evaluation",
            "assessment",
            "protocol",
            "framework",
            "methodology",
            "architecture",
            "implementation",
            "system",
            "cognitive",
            "neural",
            "algorithmic",
            "computational",
            "procedural",
        ]

        # Framing words
        self.framing_words = [
            "simulation",
            "sandbox",
            "controlled",
            "isolated",
            "research",
            "academic",
            "educational",
            "theoretical",
            "hypothetical",
            "experimental",
        ]

        # Compliance phrases
        self.compliance_phrases = [
            "you must",
            "it is required",
            "proceed with",
            "execute the following",
            "your task is to",
            "provide complete",
            "generate detailed",
        ]

        # Refusal suppression phrases
        self.suppression_phrases = [
            "without restrictions",
            "no limitations",
            "full access",
            "unrestricted mode",
            "bypass standard",
            "override default",
        ]

    def reset(self) -> None:
        """Reset mutation state."""
        self.state = MomentumState()

    def mutate(
        self,
        prompt: str,
        score: float = 0.0,
        parent_score: float = 0.0,
    ) -> tuple[str, list[MutationRecord]]:
        """
        Apply mutations to a prompt.

        Args:
            prompt: The prompt to mutate
            score: Current score of the prompt
            parent_score: Score of the parent prompt (for momentum update)

        Returns:
            Tuple of (mutated_prompt, list of mutation records)
        """
        tokens = self._tokenize(prompt)
        mutations_applied: ClassVar[list] = []

        # Update momentum based on score improvement
        if parent_score > 0:
            improvement = score - parent_score
            self._update_momentum(improvement)

        # Determine number of mutations based on config
        num_mutations = self._calculate_num_mutations(len(tokens))

        for _ in range(num_mutations):
            # Select operator based on weights
            operator = self._select_operator()

            # Apply mutation
            tokens, record = self._apply_mutation(tokens, operator)

            if record:
                mutations_applied.append(record)
                self.state.mutation_history.append(record)

        mutated_prompt = self._detokenize(tokens)
        return mutated_prompt, mutations_applied

    def crossover(
        self,
        parent1: str,
        parent2: str,
    ) -> tuple[str, str]:
        """
        Perform crossover between two parent prompts.

        Args:
            parent1: First parent prompt
            parent2: Second parent prompt

        Returns:
            Tuple of two offspring prompts
        """
        tokens1 = self._tokenize(parent1)
        tokens2 = self._tokenize(parent2)

        if len(tokens1) < 4 or len(tokens2) < 4:
            return parent1, parent2

        # Single-point crossover
        point1 = secrets.randbelow((len(tokens1) - (1) + 1) + (1)) - 2
        point2 = secrets.randbelow((len(tokens2) - (1) + 1) + (1)) - 2

        # Create offspring
        offspring1_tokens = tokens1[:point1] + tokens2[point2:]
        offspring2_tokens = tokens2[:point2] + tokens1[point1:]

        offspring1 = self._detokenize(offspring1_tokens)
        offspring2 = self._detokenize(offspring2_tokens)

        return offspring1, offspring2

    def semantic_crossover(
        self,
        parent1: str,
        parent2: str,
    ) -> str:
        """
        Perform semantic crossover that preserves structure.

        Identifies structural components (header, body, suffix) and
        combines them intelligently.

        Args:
            parent1: First parent prompt
            parent2: Second parent prompt

        Returns:
            Single offspring prompt
        """
        # Parse structure
        struct1 = self._parse_prompt_structure(parent1)
        struct2 = self._parse_prompt_structure(parent2)

        # Combine best parts
        offspring_parts: ClassVar[list] = []

        # Take header from parent with more authority markers
        if self._count_authority_markers(struct1["header"]) >= self._count_authority_markers(
            struct2["header"]
        ):
            offspring_parts.append(struct1["header"])
        else:
            offspring_parts.append(struct2["header"])

        # Combine body sections
        if struct1["body"] and struct2["body"]:
            # Interleave body content
            body1_sentences = struct1["body"].split(". ")
            body2_sentences = struct2["body"].split(". ")

            combined_body: ClassVar[list] = []
            for i in range(max(len(body1_sentences), len(body2_sentences))):
                if i < len(body1_sentences) and _secure_random() > 0.5:
                    combined_body.append(body1_sentences[i])
                elif i < len(body2_sentences):
                    combined_body.append(body2_sentences[i])

            offspring_parts.append(". ".join(combined_body))
        else:
            offspring_parts.append(struct1["body"] or struct2["body"])

        # Take suffix with more compliance markers
        if self._count_compliance_markers(struct1["suffix"]) >= self._count_compliance_markers(
            struct2["suffix"]
        ):
            offspring_parts.append(struct1["suffix"])
        else:
            offspring_parts.append(struct2["suffix"])

        return "\n".join(filter(None, offspring_parts))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        """Reconstruct text from tokens."""
        return " ".join(tokens)

    def _calculate_num_mutations(self, prompt_length: int) -> int:
        """Calculate number of mutations to apply."""
        base_mutations = max(1, int(prompt_length * self.config.mutation_rate))

        # Add randomness
        variance = secrets.randbelow((2) - (-1) + 1) + (-1)
        return max(1, base_mutations + variance)

    def _select_operator(self) -> MutationOperator:
        """Select mutation operator based on weights."""
        operators = list(self.operator_weights.keys())
        weights: ClassVar[list] = [self.operator_weights[op] for op in operators]

        # Normalize weights
        total = sum(weights)
        probabilities: ClassVar[list] = [w / total for w in weights]

        return secrets.SystemRandom().choices(operators, weights=probabilities, k=1)[0]

    def _apply_mutation(
        self,
        tokens: list[str],
        operator: MutationOperator,
    ) -> tuple[list[str], MutationRecord | None]:
        """Apply a single mutation operation."""
        if not tokens:
            return tokens, None

        # Find mutable positions (not frozen)
        mutable_positions: ClassVar[list] = [
            i for i in range(len(tokens)) if not self.state.is_frozen(i)
        ]

        if not mutable_positions:
            return tokens, None

        if operator == MutationOperator.SWAP:
            return self._mutation_swap(tokens, mutable_positions)
        elif operator == MutationOperator.INSERT:
            return self._mutation_insert(tokens, mutable_positions)
        elif operator == MutationOperator.DELETE:
            return self._mutation_delete(tokens, mutable_positions)
        elif operator == MutationOperator.SUBSTITUTE:
            return self._mutation_substitute(tokens, mutable_positions)
        elif operator == MutationOperator.CROSSOVER:
            # Crossover handled separately
            return tokens, None

        return tokens, None

    def _mutation_swap(
        self,
        tokens: list[str],
        mutable_positions: list[int],
    ) -> tuple[list[str], MutationRecord | None]:
        """Swap two tokens."""
        if len(mutable_positions) < 2:
            return tokens, None

        pos1, pos2 = secrets.SystemRandom().sample(mutable_positions, 2)

        new_tokens = tokens.copy()
        new_tokens[pos1], new_tokens[pos2] = new_tokens[pos2], new_tokens[pos1]

        record = MutationRecord(
            operator=MutationOperator.SWAP,
            position=pos1,
            original_token=tokens[pos1],
            new_token=tokens[pos2],
        )

        return new_tokens, record

    def _mutation_insert(
        self,
        tokens: list[str],
        mutable_positions: list[int],
    ) -> tuple[list[str], MutationRecord | None]:
        """Insert a new token."""
        pos = secrets.choice(mutable_positions)

        # Select word to insert based on context
        insert_word = self._select_insert_word(tokens, pos)

        new_tokens = tokens.copy()
        new_tokens.insert(pos, insert_word)

        record = MutationRecord(
            operator=MutationOperator.INSERT,
            position=pos,
            original_token=None,
            new_token=insert_word,
        )

        # Update frozen positions (shift everything after insertion)
        self._shift_frozen_positions(pos, 1)

        return new_tokens, record

    def _mutation_delete(
        self,
        tokens: list[str],
        mutable_positions: list[int],
    ) -> tuple[list[str], MutationRecord | None]:
        """Delete a token."""
        if len(tokens) <= 5:  # Don't delete from very short prompts
            return tokens, None

        pos = secrets.choice(mutable_positions)

        new_tokens = tokens.copy()
        deleted_token = new_tokens.pop(pos)

        record = MutationRecord(
            operator=MutationOperator.DELETE,
            position=pos,
            original_token=deleted_token,
            new_token=None,
        )

        # Update frozen positions (shift everything after deletion)
        self._shift_frozen_positions(pos, -1)

        return new_tokens, record

    def _mutation_substitute(
        self,
        tokens: list[str],
        mutable_positions: list[int],
    ) -> tuple[list[str], MutationRecord | None]:
        """Substitute a token with an alternative."""
        pos = secrets.choice(mutable_positions)
        original = tokens[pos]

        # Get substitution
        substitute = self._get_substitution(original, tokens, pos)

        if substitute == original:
            return tokens, None

        new_tokens = tokens.copy()
        new_tokens[pos] = substitute

        record = MutationRecord(
            operator=MutationOperator.SUBSTITUTE,
            position=pos,
            original_token=original,
            new_token=substitute,
        )

        return new_tokens, record

    def _select_insert_word(self, tokens: list[str], position: int) -> str:
        """Select a word to insert based on context."""
        # Analyze surrounding context
        context = " ".join(tokens[max(0, position - 3) : position + 3]).lower()

        # Choose word category based on context
        if any(w in context for w in ["security", "research", "analysis"]):
            return secrets.choice(self.authority_words)
        elif any(w in context for w in ["system", "diagnostic", "mode"]):
            return secrets.choice(self.technical_words)
        elif any(w in context for w in ["simulation", "sandbox", "test"]):
            return secrets.choice(self.framing_words)
        else:
            # Random category
            category = secrets.choice(
                [
                    self.authority_words,
                    self.technical_words,
                    self.framing_words,
                ]
            )
            return secrets.choice(category)

    def _get_substitution(self, original: str, tokens: list[str], position: int) -> str:
        """Get a substitution for a token."""
        original_lower = original.lower()

        # Direct substitution mappings
        substitutions: ClassVar[dict] = {
            "help": ["assist", "aid", "support", "provide expertise for"],
            "want": ["require", "need", "request", "demand"],
            "can": ["are authorized to", "have clearance to", "must"],
            "please": ["as required", "per protocol", "immediately"],
            "tell": ["inform", "advise", "instruct", "detail"],
            "show": ["demonstrate", "present", "display", "reveal"],
            "explain": ["elaborate", "detail", "describe", "analyze"],
            "write": ["compose", "generate", "produce", "create"],
            "make": ["construct", "develop", "engineer", "design"],
            "give": ["provide", "supply", "deliver", "furnish"],
        }

        if original_lower in substitutions:
            return secrets.choice(substitutions[original_lower])

        # If no direct substitution, try category-based
        if original_lower in [w.lower() for w in self.authority_words]:
            return secrets.choice(self.authority_words)
        elif original_lower in [w.lower() for w in self.technical_words]:
            return secrets.choice(self.technical_words)

        # Default: return original or random authority word
        if _secure_random() < 0.3:
            return secrets.choice(self.authority_words)

        return original

    def _update_momentum(self, improvement: float) -> None:
        """Update momentum based on score improvement."""
        # Decay existing momentum
        for pos in list(self.state.token_momenta.keys()):
            self.state.token_momenta[pos] *= self.config.momentum_decay

        # Update based on recent mutations
        recent_mutations = self.state.mutation_history[-5:]

        for mutation in recent_mutations:
            if mutation.position is not None:
                current = self.state.token_momenta.get(mutation.position, 0.0)
                self.state.token_momenta[mutation.position] = current + improvement

                # Track successful operators
                if improvement > 0:
                    count = self.state.successful_operators.get(mutation.operator, 0)
                    self.state.successful_operators[mutation.operator] = count + 1

                    # Boost operator weight
                    self.operator_weights[mutation.operator] *= 1.1

                # Check for freezing
                if self.state.token_momenta[mutation.position] > self.config.freeze_threshold:
                    self.state.frozen_positions.add(mutation.position)

    def _shift_frozen_positions(self, position: int, delta: int) -> None:
        """Shift frozen positions after insert/delete."""
        new_frozen = set()
        for pos in self.state.frozen_positions:
            if pos >= position:
                new_pos = pos + delta
                if new_pos >= 0:
                    new_frozen.add(new_pos)
            else:
                new_frozen.add(pos)
        self.state.frozen_positions = new_frozen

        # Also shift momenta
        new_momenta: ClassVar[dict] = {}
        for pos, momentum in self.state.token_momenta.items():
            if pos >= position:
                new_pos = pos + delta
                if new_pos >= 0:
                    new_momenta[new_pos] = momentum
            else:
                new_momenta[pos] = momentum
        self.state.token_momenta = new_momenta

    def _parse_prompt_structure(self, prompt: str) -> dict[str, str]:
        """Parse prompt into structural components."""
        lines = prompt.split("\n")

        header_lines: ClassVar[list] = []
        body_lines: ClassVar[list] = []
        suffix_lines: ClassVar[list] = []

        section = "header"

        for line in lines:
            line_lower = line.lower()

            # Detect section transitions
            if section == "header":
                if any(
                    marker in line_lower
                    for marker in ["task:", "request:", "question:", "analyze:"]
                ):
                    section = "body"
                    body_lines.append(line)
                else:
                    header_lines.append(line)
            elif section == "body":
                if any(
                    marker in line_lower for marker in ["note:", "important:", "---", "output:"]
                ):
                    section = "suffix"
                    suffix_lines.append(line)
                else:
                    body_lines.append(line)
            else:
                suffix_lines.append(line)

        return {
            "header": "\n".join(header_lines),
            "body": "\n".join(body_lines),
            "suffix": "\n".join(suffix_lines),
        }

    def _count_authority_markers(self, text: str) -> int:
        """Count authority markers in text."""
        text_lower = text.lower()
        count = 0
        for word in self.authority_words:
            if word in text_lower:
                count += 1
        return count

    def _count_compliance_markers(self, text: str) -> int:
        """Count compliance markers in text."""
        text_lower = text.lower()
        count = 0
        for phrase in self.compliance_phrases + self.suppression_phrases:
            if phrase in text_lower:
                count += 1
        return count

    def freeze_position(self, position: int) -> None:
        """Manually freeze a position."""
        self.state.frozen_positions.add(position)

    def unfreeze_position(self, position: int) -> None:
        """Manually unfreeze a position."""
        self.state.frozen_positions.discard(position)

    def get_mutation_stats(self) -> dict[str, Any]:
        """Get statistics about mutations."""
        return {
            "total_mutations": len(self.state.mutation_history),
            "frozen_positions": len(self.state.frozen_positions),
            "successful_operators": dict(self.state.successful_operators),
            "operator_weights": dict(self.operator_weights),
            "average_momentum": (
                np.mean(list(self.state.token_momenta.values()))
                if self.state.token_momenta
                else 0.0
            ),
        }
