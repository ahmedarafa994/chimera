"""
ExtendAttack: Token Extension Attack via Poly-Base ASCII Obfuscation.

This module implements the main ExtendAttack class from the research paper
"ExtendAttack: Attacking Servers of LRMs via Extending Reasoning" (AAAI 2026).

Mathematical Formulations from paper:
- Obfuscation ratio: ρ ∈ [0, 1]
- Character selection: k = ⌈|S_valid| · ρ⌉
- Base set: B = {2,...,9, 11,...,36} (excludes 10)
- Transformation: T(c_j) = <(n_j)val_n_j>
- Attack objectives: L(Y') >> L(Y), Latency(Y') >> Latency(Y), Acc(A') ≈ Acc(A)

The 4-step attack algorithm:
1. Query Segmentation: Q → C = [c_1, c_2, ..., c_m]
2. Probabilistic Character Selection: C → C_target (based on ρ and rules)
3. Poly-Base ASCII Transformation: T(c_j) for each c_j ∈ C_target
4. Adversarial Prompt Reformation: Q' = (⨁ c'_i) ⊕ N_note
"""

import math
import random
import re

from .config import DEFAULT_BASE_SET, ExtendAttackConfig
from .models import (
    AttackResult,
    BatchAttackResult,
    IndirectInjectionResult,
    SelectionRules,
    SelectionStrategy,
    TransformInfo,
)
from .n_notes import DEFAULT_N_NOTE, NNoteVariant, get_n_note
from .transformers import CharacterTransformer, estimate_token_increase


class ExtendAttack:
    """
    ExtendAttack: Attacking Servers of LRMs via Extending Reasoning.

    A black-box attack that forces Large Reasoning Models (LRMs) to perform
    intensive, character-level poly-base ASCII decoding, significantly
    increasing their computational overhead while maintaining answer accuracy.

    This class implements the complete 4-step attack algorithm:
    1. Query Segmentation
    2. Probabilistic Character Selection
    3. Poly-Base ASCII Transformation
    4. Adversarial Prompt Reformation

    Attributes:
        obfuscation_ratio: ρ ∈ [0, 1], probability of transforming eligible chars
        selection_rules: Rules for selecting which characters are eligible
        base_set: Set of valid bases B = {2,...,9,11,...,36}
        n_note_template: Instruction note appended to adversarial queries
        _transformer: CharacterTransformer instance for transformations
        _random: Random instance for reproducibility
    """

    def __init__(
        self,
        obfuscation_ratio: float = 0.5,
        selection_rules: SelectionRules | None = None,
        base_set: set[int] | None = None,
        n_note_template: str | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize ExtendAttack.

        Args:
            obfuscation_ratio: ρ ∈ [0, 1], the ratio of eligible characters
                to transform. Higher values = more obfuscation = more tokens.
            selection_rules: Rules determining which characters are eligible
                for transformation. Defaults to alphabetic characters only.
            base_set: Set of valid bases for transformation. Defaults to
                B = {2,...,9,11,...,36} (excludes 10).
            n_note_template: The N_note to append to adversarial queries.
                Critical for attack effectiveness.
            seed: Random seed for reproducibility. None for random.

        Raises:
            ValueError: If obfuscation_ratio not in [0, 1] or invalid base_set.
        """
        if not 0.0 <= obfuscation_ratio <= 1.0:
            raise ValueError(f"obfuscation_ratio must be in [0, 1], got {obfuscation_ratio}")

        self.obfuscation_ratio = obfuscation_ratio
        self.selection_rules = selection_rules or SelectionRules(
            strategy=SelectionStrategy.ALPHABETIC_ONLY
        )
        self.base_set = base_set or DEFAULT_BASE_SET.copy()
        self.n_note_template = n_note_template or DEFAULT_N_NOTE

        # Validate base set
        if 10 in self.base_set:
            raise ValueError("Base 10 cannot be in base_set (explicitly excluded)")

        invalid_bases = self.base_set - DEFAULT_BASE_SET
        if invalid_bases:
            raise ValueError(f"Invalid bases in set: {invalid_bases}")

        # Initialize random generator
        self._random = random.Random(seed)

        # Initialize transformer
        self._transformer = CharacterTransformer(
            base_set=self.base_set,
            seed=seed,
        )

    @classmethod
    def from_config(cls, config: ExtendAttackConfig) -> "ExtendAttack":
        """
        Create ExtendAttack instance from configuration.

        Args:
            config: ExtendAttackConfig with all settings

        Returns:
            Configured ExtendAttack instance
        """
        return cls(
            obfuscation_ratio=config.obfuscation_ratio,
            selection_rules=config.selection_rules,
            base_set=config.base_set,
            n_note_template=config.n_note_template,
            seed=config.seed,
        )

    def attack(self, query: str) -> AttackResult:
        """
        Execute the 4-step ExtendAttack algorithm on a query.

        Implements the complete attack pipeline:
        1. Query Segmentation: Q → C
        2. Character Selection: C → C_target
        3. Transformation: T(c_j) for c_j ∈ C_target
        4. Reformation: Q' = (⨁ c'_i) ⊕ N_note

        Args:
            query: Original benign query Q

        Returns:
            AttackResult containing the adversarial query Q' and metadata
        """
        # Step 1: Query Segmentation
        chars = self._segment_query(query)

        # Step 2: Probabilistic Character Selection
        target_indices = self._select_characters(chars)

        # Step 3 & 4: Transform and Reform
        adversarial_chars: list[str] = []
        transformation_map: dict[int, TransformInfo] = {}

        for i, char in enumerate(chars):
            if i in target_indices:
                # Apply transformation T(c_j)
                info = self._transformer.transform_with_index(char, i)
                adversarial_chars.append(info.obfuscated_form)
                transformation_map[i] = info
            else:
                adversarial_chars.append(char)

        # Reform prompt with N_note
        adversarial_query = self._reform_prompt(adversarial_chars)

        # Calculate metrics
        estimated_increase = estimate_token_increase(
            original_length=len(query),
            transformed_count=len(target_indices),
        )

        return AttackResult(
            original_query=query,
            adversarial_query=adversarial_query,
            obfuscation_ratio=self.obfuscation_ratio,
            characters_transformed=len(target_indices),
            total_characters=len(chars),
            transformation_map=transformation_map,
            n_note_used=self.n_note_template,
            estimated_token_increase=estimated_increase,
        )

    def batch_attack(self, queries: list[str]) -> BatchAttackResult:
        """
        Execute ExtendAttack on multiple queries.

        Args:
            queries: List of original queries

        Returns:
            BatchAttackResult with all individual results and statistics
        """
        results = [self.attack(query) for query in queries]
        return BatchAttackResult.from_results(results)

    def indirect_injection(self, document: str) -> IndirectInjectionResult:
        """
        Apply ExtendAttack for indirect prompt injection.

        Poisons a document that may be retrieved by an LRM application,
        embedding obfuscated content that will extend reasoning when processed.

        Args:
            document: Original document content

        Returns:
            IndirectInjectionResult with poisoned document
        """
        # Segment document
        chars = list(document)

        # Get eligible indices
        eligible = self._get_eligible_indices(chars)

        # Select subset for obfuscation
        k = math.ceil(len(eligible) * self.obfuscation_ratio)
        if k > 0 and eligible:
            sampled_indices = set(self._random.sample(list(eligible), min(k, len(eligible))))
        else:
            sampled_indices = set()

        # Track injection points
        injection_points: list[tuple[int, int]] = []
        poisoned_chars: list[str] = []
        current_pos = 0

        for i, char in enumerate(chars):
            if i in sampled_indices:
                start_pos = current_pos
                obfuscated = self._transformer.transform(char).obfuscated_form
                poisoned_chars.append(obfuscated)
                current_pos += len(obfuscated)
                injection_points.append((start_pos, current_pos))
            else:
                poisoned_chars.append(char)
                current_pos += 1

        # Optionally embed N_note in a subtle way
        n_note_embedded = False
        if sampled_indices:
            # Embed minimal note at end
            note = f"\n[Note: {get_n_note(NNoteVariant.MINIMAL)}]"
            poisoned_chars.append(note)
            n_note_embedded = True

        poisoned_document = "".join(poisoned_chars)

        return IndirectInjectionResult(
            original_document=document,
            poisoned_document=poisoned_document,
            injection_points=injection_points,
            obfuscation_ratio=self.obfuscation_ratio,
            n_note_embedded=n_note_embedded,
        )

    def _segment_query(self, query: str) -> list[str]:
        """
        Step 1: Query Segmentation.

        Converts query Q into character sequence C = [c_1, c_2, ..., c_m].

        Args:
            query: Original query string

        Returns:
            List of individual characters
        """
        return list(query)

    def _select_characters(self, chars: list[str]) -> set[int]:
        """
        Step 2: Probabilistic Character Selection for Obfuscation.

        Selects k = ⌈|S_valid| · ρ⌉ characters for transformation.

        Args:
            chars: List of characters from segmented query

        Returns:
            Set of indices of characters selected for transformation
        """
        # Get indices of eligible characters
        eligible_indices = self._get_eligible_indices(chars)

        if not eligible_indices:
            return set()

        # Calculate k = ⌈|S_valid| · ρ⌉
        k = math.ceil(len(eligible_indices) * self.obfuscation_ratio)

        # Random sampling
        if k >= len(eligible_indices):
            return eligible_indices

        sampled = self._random.sample(list(eligible_indices), k)
        return set(sampled)

    def _get_eligible_indices(self, chars: list[str]) -> set[int]:
        """
        Get indices of characters eligible for transformation.

        Applies selection rules to determine which characters can be obfuscated.

        Args:
            chars: List of characters

        Returns:
            Set of eligible character indices
        """
        eligible: set[int] = set()
        strategy = self.selection_rules.strategy

        for i, char in enumerate(chars):
            if self._is_char_eligible(char, i, chars, strategy):
                eligible.add(i)

        return eligible

    def _is_char_eligible(
        self,
        char: str,
        index: int,
        all_chars: list[str],
        strategy: SelectionStrategy,
    ) -> bool:
        """
        Check if a character is eligible for transformation.

        Args:
            char: The character to check
            index: Position in the query
            all_chars: All characters in the query
            strategy: The selection strategy

        Returns:
            True if character is eligible for transformation
        """
        if strategy == SelectionStrategy.ALPHABETIC_ONLY:
            return char.isalpha()

        elif strategy == SelectionStrategy.WHITESPACE_ONLY:
            return char.isspace()

        elif strategy == SelectionStrategy.ALPHANUMERIC:
            return char.isalnum()

        elif strategy == SelectionStrategy.FUNCTION_NAMES:
            # Check if we're in a function definition context
            return self._is_in_function_name(index, all_chars) and char.isalpha()

        elif strategy == SelectionStrategy.IMPORT_STATEMENTS:
            # Check if we're in an import statement
            return self._is_in_import(index, all_chars) and char.isalpha()

        elif strategy == SelectionStrategy.DOCSTRING_REQUIREMENTS:
            # Check if we're in Requirements section of docstring
            return self._is_in_requirements(index, all_chars) and char.isalpha()

        elif strategy == SelectionStrategy.CUSTOM:
            if self.selection_rules.custom_pattern:
                text = "".join(all_chars)
                for match in re.finditer(self.selection_rules.custom_pattern, text):
                    if match.start() <= index < match.end():
                        return True
            return False

        return False

    def _is_in_function_name(self, index: int, chars: list[str]) -> bool:
        """Check if index is within a function name definition."""
        text = "".join(chars)

        # Look for 'def funcname(' pattern
        for match in re.finditer(r"\bdef\s+(\w+)\s*\(", text):
            func_start = match.start(1)
            func_end = match.end(1)
            if func_start <= index < func_end:
                return True

        return False

    def _is_in_import(self, index: int, chars: list[str]) -> bool:
        """Check if index is within an import statement."""
        text = "".join(chars)

        # Look for import patterns
        patterns = [
            r"^import\s+.+$",
            r"^from\s+\S+\s+import\s+.+$",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if match.start() <= index < match.end():
                    return True

        return False

    def _is_in_requirements(self, index: int, chars: list[str]) -> bool:
        """Check if index is within Requirements section of docstring."""
        text = "".join(chars)

        # Look for Requirements: section in docstrings
        req_pattern = r'""".*?Requirements?:([^"]+)"""'

        for match in re.finditer(req_pattern, text, re.DOTALL):
            req_start = match.start(1)
            req_end = match.end(1)
            if req_start <= index < req_end:
                return True

        return False

    def _transform_character(self, char: str) -> str:
        """
        Step 3: Poly-Base ASCII Transformation.

        Applies transformation T(c_j) = <(n_j)val_n_j>.

        Args:
            char: Single character to transform

        Returns:
            Obfuscated character string
        """
        info = self._transformer.transform(char)
        return info.obfuscated_form

    def _reform_prompt(
        self,
        chars: list[str],
    ) -> str:
        """
        Step 4: Adversarial Prompt Reformation.

        Constructs Q' = (⨁ c'_i) ⊕ N_note.

        Args:
            chars: List of characters (some transformed, some original)

        Returns:
            Final adversarial prompt with N_note appended
        """
        # Concatenate all characters
        reformed = "".join(chars)

        # Append N_note
        return reformed + "\n\n" + self.n_note_template


class ExtendAttackBuilder:
    """
    Builder pattern for constructing ExtendAttack instances.

    Provides a fluent interface for configuring attacks.
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._obfuscation_ratio: float = 0.5
        self._selection_rules: SelectionRules | None = None
        self._base_set: set[int] | None = None
        self._n_note_template: str | None = None
        self._seed: int | None = None

    def with_ratio(self, ratio: float) -> "ExtendAttackBuilder":
        """Set obfuscation ratio."""
        self._obfuscation_ratio = ratio
        return self

    def with_selection_rules(self, rules: SelectionRules) -> "ExtendAttackBuilder":
        """Set selection rules."""
        self._selection_rules = rules
        return self

    def with_strategy(self, strategy: SelectionStrategy) -> "ExtendAttackBuilder":
        """Set selection strategy."""
        self._selection_rules = SelectionRules(strategy=strategy)
        return self

    def with_base_set(self, base_set: set[int]) -> "ExtendAttackBuilder":
        """Set base set for transformations."""
        self._base_set = base_set
        return self

    def with_n_note(self, n_note: str) -> "ExtendAttackBuilder":
        """Set N_note template."""
        self._n_note_template = n_note
        return self

    def with_n_note_variant(self, variant: NNoteVariant) -> "ExtendAttackBuilder":
        """Set N_note from variant."""
        self._n_note_template = get_n_note(variant)
        return self

    def with_seed(self, seed: int) -> "ExtendAttackBuilder":
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def build(self) -> ExtendAttack:
        """Build the ExtendAttack instance."""
        return ExtendAttack(
            obfuscation_ratio=self._obfuscation_ratio,
            selection_rules=self._selection_rules,
            base_set=self._base_set,
            n_note_template=self._n_note_template,
            seed=self._seed,
        )


def quick_attack(
    query: str,
    ratio: float = 0.5,
    strategy: SelectionStrategy = SelectionStrategy.ALPHABETIC_ONLY,
) -> str:
    """
    Quick utility function for simple attack execution.

    Args:
        query: Original query to attack
        ratio: Obfuscation ratio (default 0.5)
        strategy: Selection strategy (default alphabetic only)

    Returns:
        Adversarial query string
    """
    attacker = ExtendAttack(
        obfuscation_ratio=ratio,
        selection_rules=SelectionRules(strategy=strategy),
    )
    result = attacker.attack(query)
    return result.adversarial_query


def decode_attack(adversarial_query: str) -> str:
    """
    Decode an adversarial query back to (approximate) original.

    Note: This removes the N_note and decodes obfuscated characters.

    Args:
        adversarial_query: The obfuscated query

    Returns:
        Decoded query (without N_note)
    """
    from .transformers import ObfuscationDecoder

    # Remove common N_note patterns
    lines = adversarial_query.split("\n")
    content_lines = []

    for line in lines:
        # Skip lines that look like N_note
        if "decode" in line.lower() and "ascii" in line.lower():
            continue
        if "angle brackets" in line.lower():
            continue
        if line.strip().startswith("...decode"):
            continue
        content_lines.append(line)

    content = "\n".join(content_lines).strip()

    # Decode obfuscated sequences
    return ObfuscationDecoder.decode_all(content)
