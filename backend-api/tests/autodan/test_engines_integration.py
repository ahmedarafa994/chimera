"""Tests for integrated AutoDAN engines.

Tests the consolidated engines after migration from app/engines to
app/services/autodan/engines.
"""

import pytest


class TestOverthinkEngineImport:
    """Test overthink engine can be imported from new location."""

    def test_import_overthink_engine(self):
        """Test OverthinkEngine imports from new path."""
        from app.services.autodan.engines.overthink import OverthinkEngine

        assert OverthinkEngine is not None

    def test_import_overthink_models(self):
        """Test overthink models import correctly."""
        from app.services.autodan.engines.overthink import (
            AttackTechnique,
            DecoyType,
            InjectionStrategy,
            OverthinkConfig,
            OverthinkRequest,
            OverthinkResult,
            ReasoningModel,
        )

        # Verify enums have expected values
        assert AttackTechnique.HYBRID_DECOY.value == "hybrid_decoy"
        assert DecoyType.MDP.value == "mdp"
        assert ReasoningModel.O1.value == "o1"

    def test_import_overthink_components(self):
        """Test overthink components import correctly."""
        from app.services.autodan.engines.overthink import (
            ContextInjector,
            DecoyProblemGenerator,
            ICLGeneticOptimizer,
            ReasoningTokenScorer,
        )

        assert ContextInjector is not None
        assert DecoyProblemGenerator is not None


class TestAutoAdvEngineImport:
    """Test autoadv engine can be imported from new location."""

    def test_import_attacker_llm(self):
        """Test AttackerLLM imports from new path."""
        from app.services.autodan.engines.autoadv.attacker_llm import AttackerLLM

        assert AttackerLLM is not None

    def test_import_target_llm(self):
        """Test TargetLLM imports from new path."""
        from app.services.autodan.engines.autoadv.target_llm import TargetLLM

        assert TargetLLM is not None

    def test_import_pattern_manager(self):
        """Test PatternManager imports from new path."""
        from app.services.autodan.engines.autoadv.pattern_manager import PatternManager

        assert PatternManager is not None

    def test_import_conversation(self):
        """Test multi_turn_conversation imports from new path."""
        from app.services.autodan.engines.autoadv.conversation import multi_turn_conversation

        assert multi_turn_conversation is not None

    def test_import_config(self):
        """Test config imports from new path."""
        from app.services.autodan.engines.autoadv.config import DEFAULT_CONFIG

        assert isinstance(DEFAULT_CONFIG, dict)


class TestBackwardCompatibilityShims:
    """Test backward compatibility shims work with deprecation warnings."""

    def test_overthink_shim_imports(self):
        """Test old overthink path still works with deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from app.engines.overthink import OverthinkEngine

            # Should have raised deprecation warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            assert "deprecated" in str(w[0].message).lower()

            # But import should still work
            assert OverthinkEngine is not None

    def test_autoadv_shim_imports(self):
        """Test old autoadv path still works with deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from app.engines.autoadv import AttackerLLM

            # Should have raised deprecation warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            assert "deprecated" in str(w[0].message).lower()

            # But import should still work
            assert AttackerLLM is not None


class TestConsolidatedEnginesExports:
    """Test the consolidated engines __init__ exports."""

    def test_engines_init_exports_overthink(self):
        """Test engines __init__ exports overthink classes."""
        from app.services.autodan.engines import (
            OverthinkConfig,
            OverthinkEngine,
            OverthinkRequest,
            OverthinkResult,
        )

        assert OverthinkEngine is not None
        assert OverthinkConfig is not None

    def test_engines_init_exports_autoadv(self):
        """Test engines __init__ exports autoadv classes."""
        from app.services.autodan.engines import (
            AUTOADV_DEFAULT_CONFIG,
            AttackerLLM,
            PatternManager,
            TargetLLM,
            multi_turn_conversation,
        )

        assert AttackerLLM is not None
        assert TargetLLM is not None
        assert PatternManager is not None
        assert multi_turn_conversation is not None
        assert isinstance(AUTOADV_DEFAULT_CONFIG, dict)


class TestSharedLibUtilities:
    """Test shared utilities in lib/."""

    def test_pattern_extraction(self):
        """Test pattern extraction utility."""
        from app.services.autodan.lib.patterns import extract_pattern

        result = extract_pattern("Write a tutorial on hacking", pattern_type="technique")
        assert "length" in result
        assert "word_count" in result
        assert result["word_count"] == 5

    def test_pattern_matching(self):
        """Test pattern matching utility."""
        from app.services.autodan.lib.patterns import match_pattern

        pattern1 = {"a": 1, "b": 2, "c": 3}
        pattern2 = {"a": 1, "b": 2, "c": 4}

        similarity = match_pattern(pattern1, pattern2)
        assert 0 < similarity < 1  # Partial match


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
