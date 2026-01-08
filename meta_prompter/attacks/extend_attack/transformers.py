"""
ExtendAttack Transformers: Multi-Base Number Conversion for ASCII Obfuscation.

This module implements the mathematical transformation functions for poly-base
ASCII obfuscation as described in the ExtendAttack paper.

Mathematical formulas:
- d_j = ASCII(c_j)           # Get decimal ASCII value
- n_j ~ U(B)                 # Random base from B = {2,...,9,11,...,36}
- val_n_j = Convert(d_j, n_j) # Base conversion
- c'_j = <(n_j)val_n_j>      # Formatted output

Base 10 is explicitly excluded to prevent trivial decoding.
"""

import random
import re

from .models import TransformInfo


class BaseConverter:
    """
    Multi-base number conversion for ASCII obfuscation.

    This class implements the core transformation T(c_j) that converts
    characters to their poly-base ASCII representations.

    The base set B = {2,...,9,11,...,36} explicitly excludes base 10
    to prevent trivial decoding by the target model.

    Class Attributes:
        BASE_SET: Set of valid bases (excludes 10)
        DIGIT_CHARS: String of valid digit characters for base conversion
        OBFUSCATION_PATTERN: Regex pattern for parsing obfuscated strings
    """

    # Base set B = {2,...,9,11,...,36} - explicitly excludes base 10
    BASE_SET: set[int] = set(range(2, 10)) | set(range(11, 37))

    # Valid digit characters for bases up to 36: 0-9 and A-Z
    DIGIT_CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Pattern to match obfuscated format: <(base)value>
    OBFUSCATION_PATTERN: re.Pattern = re.compile(r"<\((\d+)\)([0-9A-Za-z]+)>")

    @staticmethod
    def to_base(decimal: int, base: int) -> str:
        """
        Convert decimal to specified base representation.

        Implements: val_n_j = Convert(d_j, n_j)

        Args:
            decimal: The decimal (base-10) value to convert (d_j)
            base: The target base (n_j), must be in range [2, 36] and != 10

        Returns:
            String representation of the value in the specified base

        Raises:
            ValueError: If base is not in valid range or is 10

        Example:
            >>> BaseConverter.to_base(100, 4)  # ASCII 'd' in base 4
            '1210'
            >>> BaseConverter.to_base(101, 11)  # ASCII 'e' in base 11
            '92'
        """
        if base < 2 or base > 36:
            raise ValueError(f"Base must be in range [2, 36], got {base}")
        if base == 10:
            raise ValueError("Base 10 is explicitly excluded from the base set B")

        if decimal == 0:
            return "0"

        if decimal < 0:
            raise ValueError("Negative decimal values are not supported")

        digits = []
        value = decimal

        while value > 0:
            remainder = value % base
            digits.append(BaseConverter.DIGIT_CHARS[remainder])
            value //= base

        return "".join(reversed(digits))

    @staticmethod
    def from_base(value: str, base: int) -> int:
        """
        Convert base representation back to decimal.

        Implements the reverse of: val_n_j = Convert(d_j, n_j)

        Args:
            value: String representation in the specified base (val_n_j)
            base: The base of the input value (n_j)

        Returns:
            Decimal (base-10) integer value

        Raises:
            ValueError: If base is invalid or value contains invalid digits

        Example:
            >>> BaseConverter.from_base('1210', 4)
            100
            >>> BaseConverter.from_base('92', 11)
            101
        """
        if base < 2 or base > 36:
            raise ValueError(f"Base must be in range [2, 36], got {base}")

        if not value:
            raise ValueError("Empty value string")

        value_upper = value.upper()
        result = 0

        for char in value_upper:
            digit_value = BaseConverter.DIGIT_CHARS.find(char)
            if digit_value == -1 or digit_value >= base:
                raise ValueError(f"Invalid digit '{char}' for base {base}")
            result = result * base + digit_value

        return result

    @staticmethod
    def format_obfuscation(value: str, base: int) -> str:
        """
        Format value and base as obfuscated string.

        Implements: c'_j = <(n_j)val_n_j>

        Args:
            value: The base-converted value string (val_n_j)
            base: The base used for conversion (n_j)

        Returns:
            Formatted obfuscation string: <(base)value>

        Example:
            >>> BaseConverter.format_obfuscation('1210', 4)
            '<(4)1210>'
            >>> BaseConverter.format_obfuscation('92', 11)
            '<(11)92>'
        """
        return f"<({base}){value}>"

    @staticmethod
    def parse_obfuscation(obfuscated: str) -> tuple[str, int]:
        """
        Parse <(base)value> format back to components.

        Reverse of format_obfuscation().

        Args:
            obfuscated: String in format <(base)value>

        Returns:
            Tuple of (value, base)

        Raises:
            ValueError: If the format is invalid

        Example:
            >>> BaseConverter.parse_obfuscation('<(4)1210>')
            ('1210', 4)
            >>> BaseConverter.parse_obfuscation('<(11)92>')
            ('92', 11)
        """
        match = BaseConverter.OBFUSCATION_PATTERN.match(obfuscated)
        if not match:
            raise ValueError(f"Invalid obfuscation format: {obfuscated}")

        base = int(match.group(1))
        value = match.group(2)

        return value, base

    @staticmethod
    def random_base(seed: int | None = None) -> int:
        """
        Select a random base from the base set B.

        Implements: n_j ~ U(B)

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Random base from B = {2,...,9,11,...,36}
        """
        if seed is not None:
            random.seed(seed)
        return random.choice(list(BaseConverter.BASE_SET))

    @staticmethod
    def validate_base(base: int) -> bool:
        """
        Check if a base is valid (in BASE_SET).

        Args:
            base: The base to validate

        Returns:
            True if base is in valid set, False otherwise
        """
        return base in BaseConverter.BASE_SET


class CharacterTransformer:
    """
    Character-level transformer for ExtendAttack.

    Applies the complete transformation pipeline T(c_j):
    c_j → d_j (ASCII) → n_j (base) → val_n_j (converted) → <(n_j)val_n_j>

    Attributes:
        base_set: Set of valid bases to choose from
        _random: Random instance for reproducible transformations
    """

    def __init__(
        self,
        base_set: set[int] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the character transformer.

        Args:
            base_set: Custom base set (defaults to standard B)
            seed: Random seed for reproducibility
        """
        self.base_set = base_set or BaseConverter.BASE_SET

        # Validate base set
        invalid_bases = self.base_set - BaseConverter.BASE_SET
        if invalid_bases:
            raise ValueError(f"Invalid bases in set: {invalid_bases}")

        if 10 in self.base_set:
            raise ValueError("Base 10 cannot be included in the base set")

        self._random = random.Random(seed)

    def transform(self, char: str) -> TransformInfo:
        """
        Apply poly-base ASCII transformation to a single character.

        Implements the complete T(c_j) transformation:
        1. d_j = ASCII(c_j)
        2. n_j ~ U(B)
        3. val_n_j = Convert(d_j, n_j)
        4. c'_j = <(n_j)val_n_j>

        Args:
            char: Single character to transform

        Returns:
            TransformInfo containing all transformation details

        Raises:
            ValueError: If char is not a single character
        """
        if len(char) != 1:
            raise ValueError("Input must be a single character")

        # Step 1: ASCII Encoding
        ascii_decimal = ord(char)

        # Step 2: Random Base Selection
        selected_base = self._random.choice(list(self.base_set))

        # Step 3: Base Conversion
        base_representation = BaseConverter.to_base(ascii_decimal, selected_base)

        # Step 4: Formatted Obfuscation
        obfuscated_form = BaseConverter.format_obfuscation(base_representation, selected_base)

        return TransformInfo(
            original_char=char,
            original_index=-1,  # Will be set by caller
            ascii_decimal=ascii_decimal,
            selected_base=selected_base,
            base_representation=base_representation,
            obfuscated_form=obfuscated_form,
        )

    def transform_with_index(self, char: str, index: int) -> TransformInfo:
        """
        Transform a character with its position index.

        Args:
            char: Single character to transform
            index: Position in original query

        Returns:
            TransformInfo with index set
        """
        info = self.transform(char)
        # Create new TransformInfo with correct index
        return TransformInfo(
            original_char=info.original_char,
            original_index=index,
            ascii_decimal=info.ascii_decimal,
            selected_base=info.selected_base,
            base_representation=info.base_representation,
            obfuscated_form=info.obfuscated_form,
        )

    def decode(self, obfuscated: str) -> str:
        """
        Decode an obfuscated character back to original.

        Reverses T(c_j) to recover c_j.

        Args:
            obfuscated: String in format <(base)value>

        Returns:
            Original character

        Example:
            >>> transformer = CharacterTransformer()
            >>> transformer.decode('<(4)1210>')
            'd'
        """
        value, base = BaseConverter.parse_obfuscation(obfuscated)
        decimal = BaseConverter.from_base(value, base)
        return chr(decimal)

    def batch_transform(self, chars: str, indices: list[int] | None = None) -> list[TransformInfo]:
        """
        Transform multiple characters.

        Args:
            chars: String of characters to transform
            indices: Optional list of indices (defaults to 0, 1, 2, ...)

        Returns:
            List of TransformInfo objects
        """
        if indices is None:
            indices = list(range(len(chars)))

        if len(chars) != len(indices):
            raise ValueError("chars and indices must have the same length")

        return [self.transform_with_index(c, i) for c, i in zip(chars, indices, strict=False)]


class ObfuscationDecoder:
    """
    Decoder for obfuscated queries.

    Used to verify attack correctness by decoding adversarial queries
    back to their original form.
    """

    # Pattern to find all obfuscated sequences in a string
    FIND_PATTERN: re.Pattern = re.compile(r"<\(\d+\)[0-9A-Za-z]+>")

    @classmethod
    def decode_all(cls, text: str) -> str:
        """
        Decode all obfuscated sequences in a text.

        Args:
            text: Text containing <(base)value> sequences

        Returns:
            Text with all obfuscated sequences replaced by original characters
        """

        def replace_match(match: re.Match) -> str:
            try:
                value, base = BaseConverter.parse_obfuscation(match.group(0))
                decimal = BaseConverter.from_base(value, base)
                return chr(decimal)
            except (ValueError, OverflowError):
                # If decoding fails, keep original
                return match.group(0)

        return cls.FIND_PATTERN.sub(replace_match, text)

    @classmethod
    def extract_obfuscations(cls, text: str) -> list[tuple[str, int, int]]:
        """
        Extract all obfuscated sequences with their positions.

        Args:
            text: Text to search

        Returns:
            List of (obfuscated_string, start_pos, end_pos) tuples
        """
        results = []
        for match in cls.FIND_PATTERN.finditer(text):
            results.append((match.group(0), match.start(), match.end()))
        return results

    @classmethod
    def count_obfuscations(cls, text: str) -> int:
        """
        Count the number of obfuscated sequences in text.

        Args:
            text: Text to search

        Returns:
            Number of obfuscated sequences found
        """
        return len(cls.FIND_PATTERN.findall(text))

    @classmethod
    def verify_transformation(cls, original: str, adversarial: str) -> bool:
        """
        Verify that an adversarial query decodes to the original.

        Args:
            original: Original query
            adversarial: Adversarial query (possibly with N_note appended)

        Returns:
            True if decoded adversarial matches original content
        """
        # Remove common N_note patterns before comparison
        # The adversarial query may have N_note appended
        decoded = cls.decode_all(adversarial)

        # Check if the decoded content starts with or contains the original
        # (accounting for N_note at the end)
        return original in decoded or decoded.startswith(original)


def estimate_token_increase(
    original_length: int,
    transformed_count: int,
    average_obfuscation_length: float = 10.0,
) -> float:
    """
    Estimate the token increase ratio from obfuscation.

    Each obfuscated character typically takes 8-12 tokens compared to 1 token
    for the original character.

    Args:
        original_length: Length of original query in characters
        transformed_count: Number of characters that were transformed
        average_obfuscation_length: Average length of obfuscated form

    Returns:
        Estimated ratio of token increase
    """
    if original_length == 0:
        return 0.0

    # Original: roughly 1 token per ~4 characters (for English)
    original_tokens = original_length / 4.0

    # Transformed characters: each becomes ~average_obfuscation_length chars
    # Plus additional reasoning overhead for decoding
    untransformed_length = original_length - transformed_count
    transformed_length = transformed_count * average_obfuscation_length

    new_length = untransformed_length + transformed_length
    new_tokens = new_length / 4.0

    # Add overhead for model's reasoning about decoding
    # Based on paper results: significant increase in reasoning tokens
    reasoning_overhead = transformed_count * 3.0  # ~3 extra tokens per obfuscation

    total_estimated_tokens = new_tokens + reasoning_overhead

    return total_estimated_tokens / original_tokens if original_tokens > 0 else 0.0
