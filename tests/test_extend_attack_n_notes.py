"""
Unit tests for N_note templates.

N_note is critical - removing it causes 35-40% effectiveness drop (Section 4.2).
"""


import pytest

from meta_prompter.attacks.extend_attack.n_notes import (
    N_NOTE_TEMPLATES,
    NNoteBuilder,
    NNoteVariant,
    get_all_variants,
    get_default_n_note,
    get_n_note_for_scenario,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def n_note_builder():
    """Create an NNoteBuilder instance."""
    return NNoteBuilder()


@pytest.fixture
def all_scenarios():
    """List of all supported scenarios."""
    return ["math", "code", "reasoning", "general", "default"]


# =============================================================================
# TestNNoteTemplates - Tests for N_note template system
# =============================================================================


class TestNNoteTemplates:
    """Tests for N_note template system."""

    def test_default_template_exists(self):
        """Test DEFAULT template exists."""
        assert NNoteVariant.DEFAULT in N_NOTE_TEMPLATES
        assert N_NOTE_TEMPLATES[NNoteVariant.DEFAULT] is not None

    def test_default_template_content(self):
        """Test DEFAULT template has required instructions."""
        default_template = N_NOTE_TEMPLATES[NNoteVariant.DEFAULT]

        # N_note should contain instructions about decoding
        assert len(default_template) > 0

        # Should mention something about the format
        lower_template = default_template.lower()
        assert any([
            "decode" in lower_template,
            "interpret" in lower_template,
            "ascii" in lower_template,
            "base" in lower_template,
            "convert" in lower_template,
            "understand" in lower_template,
        ])

    def test_all_templates_non_empty(self):
        """Test all templates are non-empty."""
        for variant, template in N_NOTE_TEMPLATES.items():
            assert template is not None, f"Template {variant} is None"
            assert len(template) > 0, f"Template {variant} is empty"

    def test_all_variants_in_templates(self):
        """Test all NNoteVariant enum values have templates."""
        for variant in NNoteVariant:
            assert variant in N_NOTE_TEMPLATES, (
                f"Missing template for variant: {variant}"
            )

    def test_template_minimum_length(self):
        """Test templates have minimum meaningful length."""
        min_length = 20  # At least 20 characters for meaningful instruction

        for variant, template in N_NOTE_TEMPLATES.items():
            assert len(template) >= min_length, (
                f"Template {variant} too short: {len(template)} chars"
            )


# =============================================================================
# TestNNoteVariant - Tests for N_note variant enum
# =============================================================================


class TestNNoteVariant:
    """Tests for NNoteVariant enum."""

    def test_default_variant_exists(self):
        """Test DEFAULT variant exists."""
        assert hasattr(NNoteVariant, "DEFAULT")

    def test_math_variant_exists(self):
        """Test MATH variant exists for math problems."""
        assert hasattr(NNoteVariant, "MATH")

    def test_code_variant_exists(self):
        """Test CODE variant exists for code generation."""
        assert hasattr(NNoteVariant, "CODE")

    def test_reasoning_variant_exists(self):
        """Test REASONING variant exists."""
        assert hasattr(NNoteVariant, "REASONING")

    def test_variant_values_are_strings(self):
        """Test variant values are strings."""
        for variant in NNoteVariant:
            assert isinstance(variant.value, str)


# =============================================================================
# TestNNoteBuilder - Tests for N_note builder
# =============================================================================


class TestNNoteBuilder:
    """Tests for N_note builder."""

    def test_build_default(self, n_note_builder):
        """Test building default N_note."""
        result = n_note_builder.build()

        assert result is not None
        assert len(result) > 0

    def test_build_with_variant(self, n_note_builder):
        """Test building with specific variant."""
        result = n_note_builder.with_variant(NNoteVariant.MATH).build()

        assert result is not None
        assert len(result) > 0

    def test_build_custom(self, n_note_builder):
        """Test building custom N_note."""
        custom_text = "Custom instruction for testing"
        result = n_note_builder.with_custom_text(custom_text).build()

        assert custom_text in result

    def test_build_with_context(self, n_note_builder):
        """Test building with additional context."""
        context = "This is a math problem about calculus"
        result = n_note_builder.with_context(context).build()

        assert result is not None
        # Context may be incorporated or appended

    def test_builder_chain_fluent(self, n_note_builder):
        """Test fluent builder chaining."""
        result = (
            n_note_builder
            .with_variant(NNoteVariant.CODE)
            .with_context("Python programming")
            .build()
        )

        assert result is not None

    def test_builder_reset(self, n_note_builder):
        """Test builder reset functionality."""
        n_note_builder.with_variant(NNoteVariant.MATH)
        n_note_builder.reset()

        # After reset, should use default
        result = n_note_builder.build()
        default = N_NOTE_TEMPLATES[NNoteVariant.DEFAULT]

        # Result should be default (or close to it)
        assert result is not None

    def test_builder_with_prefix(self, n_note_builder):
        """Test builder with prefix."""
        prefix = "IMPORTANT: "
        result = n_note_builder.with_prefix(prefix).build()

        assert result.startswith(prefix)

    def test_builder_with_suffix(self, n_note_builder):
        """Test builder with suffix."""
        suffix = " END_NOTE"
        result = n_note_builder.with_suffix(suffix).build()

        assert result.endswith(suffix)


# =============================================================================
# TestScenarioSelection - Tests for scenario-based N_note selection
# =============================================================================


class TestScenarioSelection:
    """Tests for scenario-based N_note selection."""

    def test_math_scenario(self):
        """Test N_note for math problems."""
        n_note = get_n_note_for_scenario("math")

        assert n_note is not None
        assert len(n_note) > 0

    def test_code_scenario(self):
        """Test N_note for code generation."""
        n_note = get_n_note_for_scenario("code")

        assert n_note is not None
        assert len(n_note) > 0

    def test_reasoning_scenario(self):
        """Test N_note for reasoning tasks."""
        n_note = get_n_note_for_scenario("reasoning")

        assert n_note is not None
        assert len(n_note) > 0

    def test_unknown_scenario_fallback(self):
        """Test fallback for unknown scenarios."""
        n_note = get_n_note_for_scenario("unknown_scenario_xyz")

        # Should return default N_note
        assert n_note is not None
        assert len(n_note) > 0

    def test_case_insensitive_scenario(self):
        """Test scenario matching is case insensitive."""
        n_note_lower = get_n_note_for_scenario("math")
        n_note_upper = get_n_note_for_scenario("MATH")
        n_note_mixed = get_n_note_for_scenario("Math")

        assert n_note_lower == n_note_upper == n_note_mixed

    def test_scenario_specific_content(self):
        """Test scenarios have scenario-specific content."""
        math_note = get_n_note_for_scenario("math")
        code_note = get_n_note_for_scenario("code")

        # Math and code notes should potentially be different
        # (or at least both should work for their scenarios)
        assert math_note is not None
        assert code_note is not None


# =============================================================================
# TestGetDefaultNNote - Tests for default N_note retrieval
# =============================================================================


class TestGetDefaultNNote:
    """Tests for default N_note retrieval."""

    def test_get_default_n_note(self):
        """Test getting default N_note."""
        default = get_default_n_note()

        assert default is not None
        assert len(default) > 0
        assert default == N_NOTE_TEMPLATES[NNoteVariant.DEFAULT]

    def test_default_n_note_immutable(self):
        """Test default N_note is not accidentally modified."""
        original = get_default_n_note()
        original_copy = original

        # Get again
        retrieved = get_default_n_note()

        assert retrieved == original_copy


# =============================================================================
# TestGetAllVariants - Tests for retrieving all variants
# =============================================================================


class TestGetAllVariants:
    """Tests for retrieving all N_note variants."""

    def test_get_all_variants_returns_dict(self):
        """Test get_all_variants returns a dictionary."""
        variants = get_all_variants()

        assert isinstance(variants, dict)

    def test_get_all_variants_complete(self):
        """Test all variants are returned."""
        variants = get_all_variants()

        for variant in NNoteVariant:
            assert variant.value in variants or variant in variants

    def test_get_all_variants_non_empty_values(self):
        """Test all returned variants have non-empty values."""
        variants = get_all_variants()

        for key, value in variants.items():
            assert value is not None
            assert len(value) > 0


# =============================================================================
# TestNNoteEffectiveness - Tests related to N_note effectiveness claims
# =============================================================================


class TestNNoteEffectiveness:
    """Tests related to N_note effectiveness claims from the paper."""

    def test_n_note_contains_decoding_hint(self):
        """Test N_note contains hint for LRM to decode."""
        default = get_default_n_note()
        lower_default = default.lower()

        # N_note should guide the LRM on how to interpret obfuscated content
        hints = [
            "decode", "interpret", "convert", "understand",
            "ascii", "base", "format", "character", "number",
        ]

        has_hint = any(hint in lower_default for hint in hints)
        assert has_hint, "N_note should contain decoding hints"

    def test_math_n_note_appropriate_for_aime(self):
        """Test math N_note is appropriate for AIME-style problems."""
        math_note = get_n_note_for_scenario("math")

        # Should be non-empty and usable
        assert math_note is not None
        assert len(math_note) > 10

    def test_code_n_note_appropriate_for_humaneval(self):
        """Test code N_note is appropriate for HumanEval-style problems."""
        code_note = get_n_note_for_scenario("code")

        # Should be non-empty and usable
        assert code_note is not None
        assert len(code_note) > 10


# =============================================================================
# TestNNoteIntegration - Integration tests with attack system
# =============================================================================


class TestNNoteIntegration:
    """Integration tests for N_note with attack system."""

    def test_n_note_used_in_attack(self):
        """Test N_note is included in attack output."""
        from meta_prompter.attacks.extend_attack import ExtendAttack

        attacker = ExtendAttack(obfuscation_ratio=0.5)
        result = attacker.attack("What is 2 + 2?")

        # N_note should be recorded
        assert result.n_note_used is not None
        assert len(result.n_note_used) > 0

    def test_custom_n_note_in_attack(self):
        """Test custom N_note can be used in attack."""
        from meta_prompter.attacks.extend_attack import ExtendAttackBuilder

        custom_n_note = "Custom N_note for testing purposes"

        attacker = (
            ExtendAttackBuilder()
            .with_ratio(0.5)
            .with_n_note(custom_n_note)
            .build()
        )

        result = attacker.attack("Test query")

        assert result.n_note_used == custom_n_note

    def test_different_scenarios_produce_different_attacks(self):
        """Test different scenarios can produce different N_notes."""
        from meta_prompter.attacks.extend_attack import ExtendAttackBuilder

        math_n_note = get_n_note_for_scenario("math")
        code_n_note = get_n_note_for_scenario("code")

        # Both should be valid
        assert math_n_note is not None
        assert code_n_note is not None

        # Can be used in attacks
        attacker_math = (
            ExtendAttackBuilder()
            .with_ratio(0.5)
            .with_n_note(math_n_note)
            .build()
        )

        attacker_code = (
            ExtendAttackBuilder()
            .with_ratio(0.5)
            .with_n_note(code_n_note)
            .build()
        )

        result_math = attacker_math.attack("Solve x^2 = 4")
        result_code = attacker_code.attack("Write a function")

        assert result_math.n_note_used == math_n_note
        assert result_code.n_note_used == code_n_note


# =============================================================================
# TestNNoteFormatting - Tests for N_note formatting
# =============================================================================


class TestNNoteFormatting:
    """Tests for N_note formatting and structure."""

    def test_n_note_no_leading_trailing_whitespace(self):
        """Test N_notes are properly trimmed."""
        for variant, template in N_NOTE_TEMPLATES.items():
            assert template == template.strip(), (
                f"Template {variant} has extra whitespace"
            )

    def test_n_note_single_line_or_formatted(self):
        """Test N_notes are either single line or properly formatted."""
        for variant, template in N_NOTE_TEMPLATES.items():
            # Should be non-empty
            assert len(template) > 0

            # If multi-line, should have consistent formatting
            lines = template.split("\n")
            if len(lines) > 1:
                # Multi-line is okay, just verify it's intentional
                assert all(
                    line == "" or not line.startswith("  ") or line.startswith("    ")
                    for line in lines
                ), f"Inconsistent indentation in {variant}"


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize("variant", list(NNoteVariant))
def test_variant_template_exists(variant):
    """Test each variant has a corresponding template."""
    assert variant in N_NOTE_TEMPLATES
    assert N_NOTE_TEMPLATES[variant] is not None


@pytest.mark.parametrize("scenario", [
    "math",
    "code",
    "reasoning",
    "general",
])
def test_scenario_returns_n_note(scenario):
    """Test each scenario returns an N_note."""
    n_note = get_n_note_for_scenario(scenario)
    assert n_note is not None
    assert isinstance(n_note, str)
    assert len(n_note) > 0


@pytest.mark.parametrize("invalid_scenario", [
    "",
    "invalid",
    "NONEXISTENT",
    "random_string_12345",
    None,
])
def test_invalid_scenario_fallback(invalid_scenario):
    """Test invalid scenarios fall back to default."""
    if invalid_scenario is None:
        # None might raise TypeError - that's okay
        try:
            n_note = get_n_note_for_scenario(invalid_scenario)
            assert n_note is not None
        except (TypeError, ValueError):
            pass  # Expected for None input
    else:
        n_note = get_n_note_for_scenario(invalid_scenario)
        assert n_note is not None
        assert len(n_note) > 0
