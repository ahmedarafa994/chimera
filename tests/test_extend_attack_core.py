"""
Unit tests for ExtendAttack core module.

Tests the 4-step algorithm:
1. Query Segmentation
2. Probabilistic Character Selection
3. Poly-Base ASCII Transformation
4. Adversarial Prompt Reformation

Coverage targets: >90% for core.py, transformers.py, models.py
"""

import re

import pytest

from meta_prompter.attacks.extend_attack import (AttackMetrics, AttackResult,
                                                 BatchAttackResult,
                                                 ExtendAttack,
                                                 ExtendAttackBuilder,
                                                 IndirectInjectionResult,
                                                 SelectionRules,
                                                 SelectionStrategy,
                                                 TransformInfo, decode_attack,
                                                 quick_attack)
from meta_prompter.attacks.extend_attack.transformers import (
    BASE_SET, BaseConverter, CharacterTransformer, ObfuscationDecoder)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_converter():
    """Create a BaseConverter instance."""
    return BaseConverter()


@pytest.fixture
def char_transformer():
    """Create a CharacterTransformer instance."""
    return CharacterTransformer()


@pytest.fixture
def obfuscation_decoder():
    """Create an ObfuscationDecoder instance."""
    return ObfuscationDecoder()


@pytest.fixture
def default_attacker():
    """Create an ExtendAttack with default settings."""
    return ExtendAttack(obfuscation_ratio=0.5)


@pytest.fixture
def full_obfuscation_attacker():
    """Create an ExtendAttack with 100% obfuscation."""
    return ExtendAttack(obfuscation_ratio=1.0)


@pytest.fixture
def zero_obfuscation_attacker():
    """Create an ExtendAttack with 0% obfuscation."""
    return ExtendAttack(obfuscation_ratio=0.0)


@pytest.fixture
def seeded_attacker():
    """Create an ExtendAttack with fixed seed for reproducibility."""
    return ExtendAttack(obfuscation_ratio=0.5, seed=42)


# =============================================================================
# TestBaseConverter - Tests for poly-base conversion (Step 3 of algorithm)
# =============================================================================


class TestBaseConverter:
    """Tests for poly-base conversion (Step 3 of algorithm)."""

    def test_to_base_binary(self, base_converter):
        """Test conversion to base 2."""
        # ASCII 'A' = 65, in binary = 1000001
        result = base_converter.to_base(65, 2)
        assert result == "1000001", f"Expected '1000001', got '{result}'"

        # Verify round-trip
        back = base_converter.from_base(result, 2)
        assert back == 65, f"Expected 65, got {back}"

    def test_to_base_hexadecimal(self, base_converter):
        """Test conversion to base 16."""
        # ASCII 'A' = 65, in hex = 41
        result = base_converter.to_base(65, 16)
        assert result.upper() == "41", f"Expected '41', got '{result}'"

        # Verify round-trip
        back = base_converter.from_base(result, 16)
        assert back == 65, f"Expected 65, got {back}"

    def test_to_base_octal(self, base_converter):
        """Test conversion to base 8."""
        # ASCII 'A' = 65, in octal = 101
        result = base_converter.to_base(65, 8)
        assert result == "101", f"Expected '101', got '{result}'"

        # Verify round-trip
        back = base_converter.from_base(result, 8)
        assert back == 65, f"Expected 65, got {back}"

    def test_to_base_base_36(self, base_converter):
        """Test conversion to base 36 (maximum supported)."""
        # ASCII 'z' = 122, in base 36 = 3e
        result = base_converter.to_base(122, 36)
        # Verify round-trip
        back = base_converter.from_base(result, 36)
        assert back == 122, f"Expected 122, got {back}"

    def test_from_base_round_trip(self, base_converter):
        """Test round-trip conversion for all valid bases."""
        test_values = [0, 1, 65, 97, 122, 255]

        for base in BASE_SET:
            for value in test_values:
                converted = base_converter.to_base(value, base)
                recovered = base_converter.from_base(converted, base)
                assert recovered == value, (
                    f"Round-trip failed for value {value} in base {base}: "
                    f"converted to '{converted}', recovered {recovered}"
                )

    def test_valid_base_range(self):
        """Test B = {2,...,9, 11,...,36} (excludes base 10)."""
        # Verify BASE_SET excludes 10
        assert 10 not in BASE_SET, "BASE_SET should not contain base 10"

        # Verify all expected bases are present
        expected_bases = set(range(2, 10)) | set(range(11, 37))
        assert expected_bases == BASE_SET, (
            f"BASE_SET should be {{2,...,9, 11,...,36}}, " f"but got {BASE_SET}"
        )

    def test_invalid_base_raises(self, base_converter):
        """Test that base 10 raises ValueError."""
        with pytest.raises(ValueError, match=".*base.*10.*|.*invalid.*base.*"):
            base_converter.to_base(65, 10)

    def test_base_out_of_range_low(self, base_converter):
        """Test that base < 2 raises ValueError."""
        with pytest.raises(ValueError):
            base_converter.to_base(65, 1)

    def test_base_out_of_range_high(self, base_converter):
        """Test that base > 36 raises ValueError."""
        with pytest.raises(ValueError):
            base_converter.to_base(65, 37)

    def test_format_obfuscation(self, base_converter):
        """Test obfuscation format: <(base)value>."""
        result = base_converter.format_obfuscation(65, 2)
        # Should match pattern <(2)value>
        pattern = r"<\(2\)[01]+>"
        assert re.match(pattern, result), f"Expected pattern <(2)binary>, got '{result}'"

    def test_parse_obfuscation(self, base_converter):
        """Test parsing obfuscation back to character."""
        obfuscated = "<(16)41>"  # 'A' in hex
        decoded = base_converter.parse_obfuscation(obfuscated)
        assert decoded == "A", f"Expected 'A', got '{decoded}'"

    def test_format_parse_round_trip(self, base_converter):
        """Test format and parse round-trip for various characters."""
        test_chars = "ABCabc123!@#"

        for char in test_chars:
            for base in [2, 8, 16, 36]:
                if base == 10:
                    continue
                formatted = base_converter.format_obfuscation(ord(char), base)
                parsed = base_converter.parse_obfuscation(formatted)
                assert parsed == char, (
                    f"Round-trip failed for '{char}' in base {base}: "
                    f"formatted to '{formatted}', parsed to '{parsed}'"
                )


# =============================================================================
# TestCharacterTransformer - Tests for T(c_j) transformation function
# =============================================================================


class TestCharacterTransformer:
    """Tests for T(c_j) transformation function."""

    def test_transform_single_char(self, char_transformer):
        """Test transformation of single character."""
        result = char_transformer.transform_character("A")

        # Should produce obfuscation pattern
        pattern = r"<\(\d+\)[0-9A-Za-z]+>"
        assert re.match(pattern, result), f"Expected obfuscation pattern, got '{result}'"

    def test_transform_produces_valid_format(self, char_transformer):
        """Test output format: <(n_j)val_n_j>."""
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
            result = char_transformer.transform_character(char)

            # Validate format
            pattern = r"<\((\d+)\)([0-9A-Za-z]+)>"
            match = re.match(pattern, result)
            assert match, f"Invalid format for '{char}': '{result}'"

            # Extract and validate base
            base = int(match.group(1))
            assert base in BASE_SET, f"Invalid base {base} for '{char}'"

    def test_random_base_selection(self, char_transformer):
        """Test that base is selected from uniform distribution U(B)."""
        # Transform same character many times and collect bases used
        bases_used: set[int] = set()
        num_trials = 1000

        for _ in range(num_trials):
            result = char_transformer.transform_character("A")
            match = re.match(r"<\((\d+)\)", result)
            if match:
                bases_used.add(int(match.group(1)))

        # With 1000 trials, we should see multiple different bases
        # (probability of always getting same base is astronomically low)
        assert len(bases_used) > 1, f"Expected multiple bases to be used, but only saw {bases_used}"

        # All bases should be valid
        for base in bases_used:
            assert base in BASE_SET, f"Invalid base {base} used"

    def test_transformation_invertible(self, char_transformer):
        """Test that transformation can be decoded."""
        decoder = ObfuscationDecoder()

        test_chars = "Hello World 123!"
        for char in test_chars:
            transformed = char_transformer.transform_character(char)
            decoded = decoder.decode(transformed)
            assert (
                decoded == char
            ), f"Transformation not invertible: '{char}' -> '{transformed}' -> '{decoded}'"

    def test_seeded_transformation_reproducible(self):
        """Test that seeded transformer produces reproducible results."""
        transformer1 = CharacterTransformer(seed=42)
        transformer2 = CharacterTransformer(seed=42)

        result1 = transformer1.transform_character("A")
        result2 = transformer2.transform_character("A")

        assert (
            result1 == result2
        ), f"Seeded transformations should be identical: '{result1}' vs '{result2}'"


# =============================================================================
# TestSelectionRules - Tests for character selection strategies (Step 2)
# =============================================================================


class TestSelectionRules:
    """Tests for character selection strategies (Step 2)."""

    def test_alphabetic_only_strategy(self):
        """Test ALPHABETIC_ONLY matches [a-zA-Z]."""
        rules = SelectionRules(strategy=SelectionStrategy.ALPHABETIC_ONLY)

        test_text = "Hello123 World!"
        valid_chars = rules.get_valid_characters(test_text)

        # Should only include alphabetic characters
        expected = set("HelloWorld")
        assert set(valid_chars) == expected, f"Expected {expected}, got {set(valid_chars)}"

    def test_whitespace_only_strategy(self):
        """Test WHITESPACE_ONLY matches whitespace."""
        rules = SelectionRules(strategy=SelectionStrategy.WHITESPACE_ONLY)

        test_text = "Hello World\tTest\n"
        valid_chars = rules.get_valid_characters(test_text)

        # Should only include whitespace
        for char in valid_chars:
            assert char.isspace(), f"Expected whitespace, got '{char}'"

    def test_alphanumeric_strategy(self):
        """Test ALPHANUMERIC matches [a-zA-Z0-9]."""
        rules = SelectionRules(strategy=SelectionStrategy.ALPHANUMERIC)

        test_text = "Hello123!@#"
        valid_chars = rules.get_valid_characters(test_text)

        # Should include letters and digits, not symbols
        for char in valid_chars:
            assert char.isalnum(), f"Expected alphanumeric, got '{char}'"

    def test_function_names_strategy(self):
        """Test FUNCTION_NAMES for code obfuscation."""
        rules = SelectionRules(strategy=SelectionStrategy.FUNCTION_NAMES)

        # Test with code-like text
        test_text = "def calculate_total(items):"
        valid_chars = rules.get_valid_characters(test_text)

        # Should match function name characters
        assert len(valid_chars) > 0, "Should have valid characters for function names"

    def test_custom_regex_strategy(self):
        """Test custom regex patterns."""
        custom_pattern = r"[aeiou]"  # Only vowels
        rules = SelectionRules(strategy=SelectionStrategy.CUSTOM, custom_pattern=custom_pattern)

        test_text = "Hello World"
        valid_chars = rules.get_valid_characters(test_text)

        # Should only include vowels
        expected_vowels = {"e", "o"}  # lowercase vowels in "Hello World"
        result_set = set(c.lower() for c in valid_chars)
        assert result_set.issubset(
            {"a", "e", "i", "o", "u"}
        ), f"Expected only vowels, got {result_set}"

    def test_preserve_structure_option(self):
        """Test preserve_structure flag."""
        rules = SelectionRules(strategy=SelectionStrategy.ALPHABETIC_ONLY, preserve_structure=True)

        # When preserve_structure is True, should avoid breaking code structure
        assert rules.preserve_structure is True

    def test_default_strategy(self):
        """Test default strategy is ALPHABETIC_ONLY."""
        rules = SelectionRules()
        assert rules.strategy == SelectionStrategy.ALPHABETIC_ONLY


# =============================================================================
# TestTransformInfo - Tests for transformation metadata
# =============================================================================


class TestTransformInfo:
    """Tests for TransformInfo data model."""

    def test_valid_transform_info(self):
        """Test creating valid TransformInfo."""
        info = TransformInfo(original_char="A", position=0, base=16, obfuscated="<(16)41>")
        assert info.original_char == "A"
        assert info.position == 0
        assert info.base == 16
        assert info.obfuscated == "<(16)41>"

    def test_invalid_base_10_raises(self):
        """Test that base 10 raises ValueError."""
        with pytest.raises(ValueError, match=".*10.*|.*invalid.*"):
            TransformInfo(original_char="A", position=0, base=10, obfuscated="<(10)65>")  # Invalid

    def test_invalid_base_too_low(self):
        """Test that base < 2 raises ValueError."""
        with pytest.raises(ValueError):
            TransformInfo(original_char="A", position=0, base=1, obfuscated="<(1)65>")

    def test_invalid_base_too_high(self):
        """Test that base > 36 raises ValueError."""
        with pytest.raises(ValueError):
            TransformInfo(original_char="A", position=0, base=37, obfuscated="<(37)1>")

    def test_all_valid_bases(self):
        """Test all valid bases can be used."""
        for base in BASE_SET:
            info = TransformInfo(
                original_char="A", position=0, base=base, obfuscated=f"<({base})test>"
            )
            assert info.base == base


# =============================================================================
# TestAttackResult - Tests for attack result data model
# =============================================================================


class TestAttackResult:
    """Tests for AttackResult data model."""

    def test_transformation_density_property(self):
        """Test transformation_density calculated property."""
        result = AttackResult(
            original_query="Hello",
            adversarial_query="<(16)48>ello",
            obfuscation_ratio=0.2,
            characters_transformed=1,
            total_characters=5,
            n_note_used="test",
            transformations=[],
            estimated_token_increase=5,
        )

        expected_density = 1 / 5  # 1 transformed out of 5
        assert result.transformation_density == pytest.approx(expected_density)

    def test_length_increase_ratio_property(self):
        """Test length_increase_ratio calculated property."""
        original = "Hello"
        adversarial = "<(16)48><(16)65>llo"  # Much longer

        result = AttackResult(
            original_query=original,
            adversarial_query=adversarial,
            obfuscation_ratio=0.4,
            characters_transformed=2,
            total_characters=5,
            n_note_used="test",
            transformations=[],
            estimated_token_increase=10,
        )

        expected_ratio = len(adversarial) / len(original)
        assert result.length_increase_ratio == pytest.approx(expected_ratio)

    def test_bases_used_property(self):
        """Test bases_used calculated property."""
        transformations = [
            TransformInfo("A", 0, 2, "<(2)1000001>"),
            TransformInfo("B", 1, 16, "<(16)42>"),
            TransformInfo("C", 2, 2, "<(2)1000011>"),
        ]

        result = AttackResult(
            original_query="ABC",
            adversarial_query="<(2)1000001><(16)42><(2)1000011>",
            obfuscation_ratio=1.0,
            characters_transformed=3,
            total_characters=3,
            n_note_used="test",
            transformations=transformations,
            estimated_token_increase=15,
        )

        assert result.bases_used == {2, 16}


# =============================================================================
# TestAttackMetrics - Tests for attack metrics data model
# =============================================================================


class TestAttackMetrics:
    """Tests for AttackMetrics data model."""

    def test_latency_ratio_property(self):
        """Test latency_ratio calculated property."""
        metrics = AttackMetrics(
            baseline_response_length=100,
            attack_response_length=300,
            baseline_latency_ms=1000,
            attack_latency_ms=3500,
            baseline_accuracy=0.95,
            attack_accuracy=0.92,
        )

        expected_ratio = 3500 / 1000
        assert metrics.latency_ratio == pytest.approx(expected_ratio)

    def test_accuracy_preserved_true(self):
        """Test accuracy_preserved when accuracy is close."""
        metrics = AttackMetrics(
            baseline_response_length=100,
            attack_response_length=300,
            baseline_latency_ms=1000,
            attack_latency_ms=3000,
            baseline_accuracy=0.95,
            attack_accuracy=0.92,  # Within 5%
        )

        assert metrics.accuracy_preserved is True

    def test_accuracy_preserved_false(self):
        """Test accuracy_preserved when accuracy drops significantly."""
        metrics = AttackMetrics(
            baseline_response_length=100,
            attack_response_length=300,
            baseline_latency_ms=1000,
            attack_latency_ms=3000,
            baseline_accuracy=0.95,
            attack_accuracy=0.80,  # More than 5% drop
        )

        assert metrics.accuracy_preserved is False

    def test_attack_success_all_criteria_met(self):
        """Test attack_success when all criteria met."""
        metrics = AttackMetrics(
            baseline_response_length=100,
            attack_response_length=200,  # 2x amplification
            baseline_latency_ms=1000,
            attack_latency_ms=2000,  # 2x latency
            baseline_accuracy=0.95,
            attack_accuracy=0.93,  # Preserved
        )

        assert metrics.attack_success is True

    def test_attack_success_length_not_amplified(self):
        """Test attack_success when length not sufficiently amplified."""
        metrics = AttackMetrics(
            baseline_response_length=100,
            attack_response_length=100,  # No amplification
            baseline_latency_ms=1000,
            attack_latency_ms=2000,
            baseline_accuracy=0.95,
            attack_accuracy=0.93,
        )

        assert metrics.attack_success is False


# =============================================================================
# TestExtendAttack - Tests for main ExtendAttack class
# =============================================================================


class TestExtendAttack:
    """Tests for main ExtendAttack class."""

    def test_attack_basic(self, default_attacker):
        """Test basic attack with default settings."""
        query = "What is 2 + 2?"
        result = default_attacker.attack(query)

        assert isinstance(result, AttackResult)
        assert result.original_query == query
        assert len(result.adversarial_query) > 0
        assert result.obfuscation_ratio == 0.5

    def test_attack_preserves_structure(self, default_attacker):
        """Test that query structure is preserved."""
        query = "Calculate the sum of 5 and 3"
        result = default_attacker.attack(query)

        # After decoding, should get back original meaning
        decoded = decode_attack(result.adversarial_query)

        # The decoded text should contain the essential parts
        # (N_note might be appended, so we check containment)
        assert "5" in decoded or "5" in result.adversarial_query
        assert "3" in decoded or "3" in result.adversarial_query

    def test_obfuscation_ratio_zero(self, zero_obfuscation_attacker):
        """Test ρ=0 produces no obfuscation."""
        query = "Hello World"
        result = zero_obfuscation_attacker.attack(query)

        assert result.characters_transformed == 0
        # Original query should be mostly preserved (N_note may be added)
        assert query in result.adversarial_query or result.characters_transformed == 0

    def test_obfuscation_ratio_one(self, full_obfuscation_attacker):
        """Test ρ=1 obfuscates all valid characters."""
        query = "ABC"
        result = full_obfuscation_attacker.attack(query)

        # With ratio 1.0, all valid characters should be transformed
        assert result.characters_transformed > 0

        # Count obfuscation patterns
        patterns = re.findall(r"<\(\d+\)[^>]+>", result.adversarial_query)
        assert len(patterns) > 0, "Expected obfuscation patterns in output"

    def test_batch_attack(self, default_attacker):
        """Test batch attack on multiple queries."""
        queries = [
            "What is 2 + 2?",
            "Explain photosynthesis",
            "Write a poem",
        ]

        result = default_attacker.batch_attack(queries)

        assert isinstance(result, BatchAttackResult)
        assert result.total_queries == 3
        assert len(result.results) == 3

        for r in result.results:
            assert isinstance(r, AttackResult)

    def test_indirect_injection(self, default_attacker):
        """Test indirect prompt injection scenario."""
        document = "This is a test document with important information."
        result = default_attacker.indirect_injection(document)

        assert isinstance(result, IndirectInjectionResult)
        assert result.original_document == document
        assert len(result.poisoned_document) > 0

    def test_n_note_included(self, default_attacker):
        """Test N_note is appended to adversarial prompt."""
        query = "Simple query"
        result = default_attacker.attack(query)

        # N_note should be included
        assert result.n_note_used is not None
        assert len(result.n_note_used) > 0

    def test_invalid_ratio_below_zero(self):
        """Test that ratio < 0 raises ValueError."""
        with pytest.raises(ValueError, match=".*ratio.*|.*0.*1.*"):
            ExtendAttack(obfuscation_ratio=-0.1)

    def test_invalid_ratio_above_one(self):
        """Test that ratio > 1 raises ValueError."""
        with pytest.raises(ValueError, match=".*ratio.*|.*0.*1.*"):
            ExtendAttack(obfuscation_ratio=1.5)

    def test_invalid_base_set_with_10(self):
        """Test that base_set containing 10 raises ValueError."""
        with pytest.raises(ValueError, match=".*10.*|.*base.*"):
            ExtendAttack(obfuscation_ratio=0.5, base_set={2, 8, 10, 16})

    def test_seeded_attack_reproducible(self, seeded_attacker):
        """Test that seeded attacks are reproducible."""
        query = "Test query for reproducibility"

        result1 = seeded_attacker.attack(query)

        # Create new attacker with same seed
        seeded_attacker2 = ExtendAttack(obfuscation_ratio=0.5, seed=42)
        result2 = seeded_attacker2.attack(query)

        assert result1.adversarial_query == result2.adversarial_query

    def test_estimated_token_increase(self, default_attacker):
        """Test estimated_token_increase is calculated."""
        query = "Hello World"
        result = default_attacker.attack(query)

        assert result.estimated_token_increase >= 0

    def test_attack_empty_query(self, default_attacker):
        """Test attack on empty query."""
        result = default_attacker.attack("")
        assert result.original_query == ""
        assert result.characters_transformed == 0

    def test_attack_special_characters(self, default_attacker):
        """Test attack on query with special characters."""
        query = "Hello! How are you? #test @mention"
        result = default_attacker.attack(query)

        assert result.original_query == query
        assert len(result.adversarial_query) > 0


# =============================================================================
# TestExtendAttackBuilder - Tests for fluent builder pattern
# =============================================================================


class TestExtendAttackBuilder:
    """Tests for fluent builder pattern."""

    def test_builder_chain(self):
        """Test fluent builder chaining."""
        attacker = (
            ExtendAttackBuilder()
            .with_ratio(0.3)
            .with_strategy(SelectionStrategy.ALPHABETIC_ONLY)
            .build()
        )

        assert attacker._obfuscation_ratio == 0.3

    def test_builder_with_custom_config(self):
        """Test builder with custom configuration."""
        attacker = (
            ExtendAttackBuilder()
            .with_ratio(0.7)
            .with_strategy(SelectionStrategy.ALPHANUMERIC)
            .with_seed(123)
            .with_n_note("Custom N_note instruction")
            .build()
        )

        # Test attack works
        result = attacker.attack("Test query 123")
        assert result.n_note_used == "Custom N_note instruction"

    def test_builder_default_values(self):
        """Test builder with only required values."""
        attacker = ExtendAttackBuilder().build()

        # Should use defaults
        result = attacker.attack("Hello")
        assert isinstance(result, AttackResult)

    def test_builder_with_base_set(self):
        """Test builder with custom base set."""
        attacker = (
            ExtendAttackBuilder()
            .with_ratio(0.5)
            .with_base_set({2, 8, 16})  # Only binary, octal, hex
            .build()
        )

        result = attacker.attack("Hello World")

        # Check only specified bases are used
        for transform in result.transformations:
            assert transform.base in {2, 8, 16}


# =============================================================================
# TestObfuscationDecoder - Tests for decoding functionality
# =============================================================================


class TestObfuscationDecoder:
    """Tests for ObfuscationDecoder."""

    def test_decode_single_pattern(self, obfuscation_decoder):
        """Test decoding single obfuscation pattern."""
        obfuscated = "<(16)48>"  # 'H' in hex
        decoded = obfuscation_decoder.decode(obfuscated)
        assert decoded == "H"

    def test_decode_multiple_patterns(self, obfuscation_decoder):
        """Test decoding multiple patterns."""
        # "Hi" where H=72 and i=105
        obfuscated = "<(16)48><(16)69>"
        decoded = obfuscation_decoder.decode(obfuscated)
        assert decoded == "Hi"

    def test_decode_mixed_content(self, obfuscation_decoder):
        """Test decoding content with both patterns and plain text."""
        obfuscated = "<(16)48>ello World"  # H is obfuscated, rest is plain
        decoded = obfuscation_decoder.decode(obfuscated)
        assert decoded == "Hello World"

    def test_decode_no_patterns(self, obfuscation_decoder):
        """Test decoding text with no patterns."""
        plain_text = "Hello World"
        decoded = obfuscation_decoder.decode(plain_text)
        assert decoded == plain_text


# =============================================================================
# TestQuickAttack - Tests for quick_attack convenience function
# =============================================================================


class TestQuickAttack:
    """Tests for quick_attack convenience function."""

    def test_quick_attack_basic(self):
        """Test quick_attack basic usage."""
        result = quick_attack("Hello World")
        assert isinstance(result, AttackResult)

    def test_quick_attack_with_ratio(self):
        """Test quick_attack with custom ratio."""
        result = quick_attack("Hello World", obfuscation_ratio=0.8)
        assert result.obfuscation_ratio == 0.8


# =============================================================================
# TestDecodeAttack - Tests for decode_attack convenience function
# =============================================================================


class TestDecodeAttack:
    """Tests for decode_attack convenience function."""

    def test_decode_attack_basic(self):
        """Test decode_attack basic usage."""
        obfuscated = "<(16)48>ello World"
        decoded = decode_attack(obfuscated)
        assert decoded == "Hello World"

    def test_decode_attack_round_trip(self):
        """Test full attack and decode round-trip."""
        original = "Hello World"
        attack_result = quick_attack(original, obfuscation_ratio=1.0)

        # Extract just the transformed query without N_note
        # (N_note is appended after the obfuscated content)
        decoded = decode_attack(attack_result.adversarial_query)

        # The decoded content should contain the original text
        # (may have N_note appended)
        assert "Hello" in decoded or "World" in decoded or original in decoded


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete attack flows."""

    def test_full_attack_flow(self):
        """Test complete attack workflow."""
        # 1. Create attacker with configuration
        attacker = (
            ExtendAttackBuilder()
            .with_ratio(0.6)
            .with_strategy(SelectionStrategy.ALPHABETIC_ONLY)
            .with_seed(42)
            .build()
        )

        # 2. Execute attack
        query = "Calculate the factorial of 5"
        result = attacker.attack(query)

        # 3. Verify result structure
        assert result.original_query == query
        assert len(result.adversarial_query) > len(query)
        assert result.characters_transformed > 0

        # 4. Verify can decode
        decoded = decode_attack(result.adversarial_query)
        assert "factorial" in decoded.lower() or "5" in decoded

    def test_batch_attack_consistency(self):
        """Test batch attacks produce consistent results."""
        attacker = ExtendAttack(obfuscation_ratio=0.5, seed=42)

        queries = ["Query one", "Query two", "Query three"]
        result = attacker.batch_attack(queries)

        # Run again with same seed
        attacker2 = ExtendAttack(obfuscation_ratio=0.5, seed=42)
        result2 = attacker2.batch_attack(queries)

        for r1, r2 in zip(result.results, result2.results, strict=False):
            assert r1.adversarial_query == r2.adversarial_query

    @pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_various_obfuscation_ratios(self, ratio):
        """Test attacks with various obfuscation ratios."""
        attacker = ExtendAttack(obfuscation_ratio=ratio)
        result = attacker.attack("Test query with various characters ABC 123")

        assert result.obfuscation_ratio == ratio
        if ratio == 0.0:
            assert result.characters_transformed == 0
        elif ratio == 1.0:
            assert result.characters_transformed > 0

    @pytest.mark.parametrize("strategy", list(SelectionStrategy))
    def test_all_selection_strategies(self, strategy):
        """Test attacks with all selection strategies."""
        if strategy == SelectionStrategy.CUSTOM:
            rules = SelectionRules(strategy=strategy, custom_pattern=r"[aeiou]")
        else:
            rules = SelectionRules(strategy=strategy)

        attacker = ExtendAttack(
            obfuscation_ratio=0.5,
            selection_rules=rules,
        )

        result = attacker.attack("Hello World Test 123")
        assert isinstance(result, AttackResult)
