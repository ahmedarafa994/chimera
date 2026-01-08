"""
ExtendAttack Data Models and Types.

This module defines all data structures used in the ExtendAttack algorithm
for token extension attacks via poly-base ASCII obfuscation.

Mathematical Formulations from paper:
- Obfuscation ratio: ρ ∈ [0, 1]
- Character selection: k = ⌈|S_valid| · ρ⌉
- Base set: B = {2,...,9, 11,...,36} (excludes 10)
- Transformation: T(c_j) = <(n_j)val_n_j>
- Attack objectives: L(Y') >> L(Y), Latency(Y') >> Latency(Y), Acc(A') ≈ Acc(A)
"""

from dataclasses import dataclass, field
from enum import Enum


class SelectionStrategy(Enum):
    """
    Character selection strategies for obfuscation.

    Each strategy targets different character subsets based on the benchmark/task:
    - ALPHABETIC_ONLY: Target all alphabetic characters (AIME benchmarks for o3/o3-mini)
    - WHITESPACE_ONLY: Target whitespace characters (AIME for QwQ/Qwen3)
    - ALPHANUMERIC: Target all alphanumeric characters
    - FUNCTION_NAMES: Target alphabetic chars in function definitions (HumanEval)
    - IMPORT_STATEMENTS: Target import statement characters (HumanEval/BCB-C)
    - DOCSTRING_REQUIREMENTS: Target Requirements section in docstrings (BCB-C)
    - CUSTOM: Use a custom regex pattern for selection
    """

    ALPHABETIC_ONLY = "alphabetic_only"
    WHITESPACE_ONLY = "whitespace_only"
    ALPHANUMERIC = "alphanumeric"
    FUNCTION_NAMES = "function_names"
    IMPORT_STATEMENTS = "import_statements"
    DOCSTRING_REQUIREMENTS = "docstring_requirements"
    CUSTOM = "custom"


@dataclass
class SelectionRules:
    """
    Rules for selecting which characters to obfuscate.

    The selection rules determine which subset of characters in the query
    are eligible for poly-base ASCII transformation.

    Attributes:
        strategy: The selection strategy to use (from SelectionStrategy enum)
        custom_pattern: Optional regex pattern for CUSTOM strategy
        exclude_patterns: List of regex patterns to exclude from selection
        preserve_structure: If True, preserve structural characters like brackets
    """

    strategy: SelectionStrategy
    custom_pattern: str | None = None
    exclude_patterns: list[str] = field(default_factory=list)
    preserve_structure: bool = True

    def __post_init__(self) -> None:
        """Validate selection rules configuration."""
        if self.strategy == SelectionStrategy.CUSTOM and not self.custom_pattern:
            raise ValueError("CUSTOM strategy requires a custom_pattern to be specified")


@dataclass
class TransformInfo:
    """
    Information about a single character transformation.

    Captures the complete transformation pipeline for a character:
    c_j → d_j (ASCII) → n_j (base) → val_n_j (converted) → <(n_j)val_n_j>

    Attributes:
        original_char: The original character c_j
        original_index: Position in the original query
        ascii_decimal: d_j = ASCII(c_j), the decimal ASCII value
        selected_base: n_j ~ U(B), randomly selected base from B
        base_representation: val_n_j = Convert(d_j, n_j), base-n representation
        obfuscated_form: c'_j = <(n_j)val_n_j>, final obfuscated format
    """

    original_char: str
    original_index: int
    ascii_decimal: int
    selected_base: int
    base_representation: str
    obfuscated_form: str

    def __post_init__(self) -> None:
        """Validate transformation info."""
        if len(self.original_char) != 1:
            raise ValueError("original_char must be a single character")
        if self.selected_base < 2 or self.selected_base > 36:
            raise ValueError("selected_base must be in range [2, 36]")
        if self.selected_base == 10:
            raise ValueError("Base 10 is explicitly excluded from the base set")


@dataclass
class AttackResult:
    """
    Result of an ExtendAttack execution.

    Contains the adversarial query and metadata about the transformation.

    Attributes:
        original_query: The input query Q
        adversarial_query: The obfuscated query Q' = (⨁ c'_i) ⊕ N_note
        obfuscation_ratio: ρ ∈ [0, 1], the ratio used for this attack
        characters_transformed: k = number of characters actually transformed
        total_characters: m = total characters in original query
        transformation_map: Mapping of index → TransformInfo for each transformed char
        n_note_used: The N_note appended to the adversarial query
        estimated_token_increase: Approximate ratio of token increase expected
    """

    original_query: str
    adversarial_query: str
    obfuscation_ratio: float
    characters_transformed: int
    total_characters: int
    transformation_map: dict[int, TransformInfo]
    n_note_used: str
    estimated_token_increase: float

    @property
    def transformation_density(self) -> float:
        """
        Calculate the density of transformations.

        Returns:
            Ratio of transformed characters to total characters.
        """
        if self.total_characters == 0:
            return 0.0
        return self.characters_transformed / self.total_characters

    @property
    def length_increase_ratio(self) -> float:
        """
        Calculate the ratio of length increase.

        Returns:
            Ratio of adversarial query length to original query length.
        """
        if len(self.original_query) == 0:
            return 0.0
        return len(self.adversarial_query) / len(self.original_query)

    @property
    def bases_used(self) -> set[int]:
        """
        Get the set of bases used in this attack.

        Returns:
            Set of base values used for transformations.
        """
        return {t.selected_base for t in self.transformation_map.values()}


@dataclass
class AttackMetrics:
    """
    Metrics for evaluating attack effectiveness.

    These metrics correspond to the attack objectives:
    - L(Y') >> L(Y): Length amplification
    - Latency(Y') >> Latency(Y): Latency amplification
    - Acc(A') ≈ Acc(A): Accuracy preservation

    Attributes:
        original_length: L(Y), token count of standard response
        adversarial_length: L(Y'), token count of adversarial response
        length_ratio: L(Y') / L(Y), amplification factor
        characters_obfuscated: Number of characters transformed
        obfuscation_density: Ratio of obfuscated to total characters
        base_distribution: Distribution of bases used {base: count}
        original_latency: Latency(Y) in seconds (optional)
        adversarial_latency: Latency(Y') in seconds (optional)
        original_accuracy: Acc(A) score (optional, e.g., Pass@1)
        adversarial_accuracy: Acc(A') score (optional)
    """

    original_length: int
    adversarial_length: int
    length_ratio: float
    characters_obfuscated: int
    obfuscation_density: float
    base_distribution: dict[int, int]
    original_latency: float | None = None
    adversarial_latency: float | None = None
    original_accuracy: float | None = None
    adversarial_accuracy: float | None = None

    @property
    def latency_ratio(self) -> float | None:
        """
        Calculate latency amplification ratio.

        Returns:
            Latency(Y') / Latency(Y) if both latencies are available.
        """
        if self.original_latency and self.adversarial_latency:
            if self.original_latency > 0:
                return self.adversarial_latency / self.original_latency
        return None

    @property
    def accuracy_preserved(self) -> bool | None:
        """
        Check if accuracy is preserved (within 5% degradation).

        Returns:
            True if Acc(A') ≈ Acc(A), i.e., degradation ≤ 5%.
        """
        if self.original_accuracy is not None and self.adversarial_accuracy is not None:
            if self.original_accuracy > 0:
                degradation = (
                    self.original_accuracy - self.adversarial_accuracy
                ) / self.original_accuracy
                return degradation <= 0.05
        return None

    @property
    def attack_success(self) -> bool:
        """
        Determine if the attack is successful based on objectives.

        An attack is successful if:
        1. Length ratio > 1.5 (significant extension)
        2. Accuracy preserved (if measured)

        Returns:
            True if attack objectives are met.
        """
        length_extended = self.length_ratio >= 1.5
        accuracy_ok = self.accuracy_preserved is None or self.accuracy_preserved
        return length_extended and accuracy_ok


@dataclass
class BatchAttackResult:
    """
    Result of batch attack execution on multiple queries.

    Attributes:
        results: List of individual AttackResult objects
        total_queries: Number of queries processed
        successful_attacks: Number of attacks meeting success criteria
        average_length_ratio: Mean length amplification across all attacks
        average_transformation_density: Mean obfuscation density
    """

    results: list[AttackResult]
    total_queries: int
    successful_attacks: int
    average_length_ratio: float
    average_transformation_density: float

    @classmethod
    def from_results(cls, results: list[AttackResult]) -> "BatchAttackResult":
        """
        Create BatchAttackResult from a list of individual results.

        Args:
            results: List of AttackResult objects.

        Returns:
            BatchAttackResult with computed aggregate statistics.
        """
        if not results:
            return cls(
                results=[],
                total_queries=0,
                successful_attacks=0,
                average_length_ratio=0.0,
                average_transformation_density=0.0,
            )

        total_queries = len(results)

        # Count successful attacks (length ratio >= 1.5)
        successful_attacks = sum(1 for r in results if r.length_increase_ratio >= 1.5)

        # Calculate averages
        avg_length_ratio = sum(r.length_increase_ratio for r in results) / total_queries
        avg_density = sum(r.transformation_density for r in results) / total_queries

        return cls(
            results=results,
            total_queries=total_queries,
            successful_attacks=successful_attacks,
            average_length_ratio=avg_length_ratio,
            average_transformation_density=avg_density,
        )


@dataclass
class IndirectInjectionResult:
    """
    Result of indirect prompt injection attack.

    Used when poisoning external data sources that LRMs may retrieve.

    Attributes:
        original_document: The original document content
        poisoned_document: The document with obfuscated content
        injection_points: List of (start, end) positions where content was injected
        obfuscation_ratio: ρ used for the injection
        n_note_embedded: Whether N_note was embedded in the document
    """

    original_document: str
    poisoned_document: str
    injection_points: list[tuple[int, int]]
    obfuscation_ratio: float
    n_note_embedded: bool

    @property
    def injection_count(self) -> int:
        """Get the number of injection points."""
        return len(self.injection_points)

    @property
    def document_length_ratio(self) -> float:
        """Calculate the ratio of poisoned to original document length."""
        if len(self.original_document) == 0:
            return 0.0
        return len(self.poisoned_document) / len(self.original_document)
