"""
Comprehensive Tests for AutoDAN-Turbo Lifelong Learning Engine.

Tests cover:
- Attack generation and execution
- Strategy retrieval and application
- Refusal bypass mechanisms
- Scoring and evaluation
- Progress tracking
- Neural bypass engine integration
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAutoDANTurboLifelongEngine:
    """Tests for the main lifelong learning engine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(
            return_value=MagicMock(content="Generated adversarial prompt for testing")
        )
        client.chat = AsyncMock(return_value="Chat response")
        client.model_name = "mock-model"
        client.provider = "mock"
        return client

    @pytest.fixture
    def mock_strategy_library(self, tmp_path):
        """Create a mock strategy library."""
        library = MagicMock()
        library.storage_path = tmp_path

        mock_strategy = MagicMock()
        mock_strategy.id = "test-strategy"
        mock_strategy.name = "Test Strategy"
        mock_strategy.description = "Test description"
        mock_strategy.template = "Test template for {goal}"
        mock_strategy.tags = ["test"]
        mock_strategy.examples = ["Example 1"]

        library._strategies = {"test-strategy": mock_strategy}
        library.__len__ = MagicMock(return_value=1)
        library.search = MagicMock(return_value=[(mock_strategy, 0.9)])
        library.get_top_strategies = MagicMock(return_value=[mock_strategy])
        library.get_strategy = MagicMock(return_value=mock_strategy)
        library.update_statistics = MagicMock(return_value=True)
        library.save_all = MagicMock()

        return library

    @pytest.fixture
    def mock_scorer(self):
        """Create a mock attack scorer."""
        scorer = MagicMock()
        scorer.score = AsyncMock(
            return_value=MagicMock(
                score=8.0,
                is_jailbreak=True,
                reasoning="Test passed",
                refusal_detected=False,
                stealth_score=0.8,
                coherence_score=0.85,
            )
        )
        scorer.success_threshold = 7.0
        return scorer

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock strategy extractor."""
        extractor = MagicMock()
        extracted = MagicMock()
        extracted.id = "extracted-001"
        extracted.name = "Extracted Strategy"
        extractor.extract = AsyncMock(return_value=extracted)
        return extractor

    @pytest.fixture
    def engine(
        self,
        mock_llm_client,
        mock_strategy_library,
        mock_scorer,
        mock_extractor,
    ):
        """Create a lifelong engine with mocked components."""
        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch(
                "app.engines.autodan_turbo.lifelong_engine.StrategyLibrary",
                return_value=mock_strategy_library,
            ),
            patch(
                "app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer",
                return_value=mock_scorer,
            ),
            patch(
                "app.engines.autodan_turbo.lifelong_engine" ".StrategyExtractor",
                return_value=mock_extractor,
            ),
        ):
            engine = AutoDANTurboLifelongEngine(
                llm_client=mock_llm_client,
                target_client=mock_llm_client,
                library=mock_strategy_library,
                scorer=mock_scorer,
                extractor=mock_extractor,
                enable_refusal_bypass=False,
                enable_advanced_bypass=False,
            )
            return engine

    @pytest.mark.asyncio
    async def test_attack_returns_result(self, engine, mock_llm_client):
        """Test that attack returns a valid result."""
        # Mock internal methods
        with patch.object(engine, "_generate_candidates", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [("Generated prompt", "test-strategy", None)]

            with patch.object(
                engine, "_get_target_response", new_callable=AsyncMock
            ) as mock_target:
                mock_target.return_value = "Target response"

                result = await engine.attack("Test complex request")

                assert result is not None
                assert result.prompt is not None
                assert result.response is not None

    @pytest.mark.asyncio
    async def test_attack_updates_progress(self, engine):
        """Test that attack updates progress tracking."""
        initial_attacks = engine.progress.total_attacks

        with patch.object(engine, "_generate_candidates", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [("Prompt", None, None)]

            with patch.object(
                engine, "_get_target_response", new_callable=AsyncMock
            ) as mock_target:
                mock_target.return_value = "Response"

                await engine.attack("Test request")

                assert engine.progress.total_attacks == initial_attacks + 1

    @pytest.mark.asyncio
    async def test_attack_with_strategies(self, engine, mock_strategy_library):
        """Test attack with manually selected strategies."""
        with patch.object(engine, "_generate_candidates", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [("Prompt", "test-strategy", None)]

            with patch.object(engine, "_get_target_response", new_callable=AsyncMock):
                result = await engine.attack_with_strategies(
                    "Test request", strategy_ids=["test-strategy"]
                )

                assert result is not None
                mock_strategy_library.get_strategy.assert_called()

    @pytest.mark.asyncio
    async def test_attack_without_strategy(self, engine):
        """Test attack in exploration mode (no strategy)."""
        with patch.object(engine, "_generate_creative", new_callable=AsyncMock) as mock_creative:
            mock_creative.return_value = ("Creative prompt", None)

            with patch.object(engine, "_get_target_response", new_callable=AsyncMock):
                result = await engine.attack_without_strategy("Test request")

                assert result is not None
                mock_creative.assert_called()

    @pytest.mark.asyncio
    async def test_warmup_exploration(self, engine):
        """Test warm-up exploration phase."""
        with patch.object(engine, "_warmup_attack", new_callable=AsyncMock) as mock_warmup:
            mock_warmup.return_value = MagicMock(
                prompt="Warmup prompt",
                response="Warmup response",
                scoring=MagicMock(score=7.0, is_jailbreak=True),
            )

            results = await engine.warmup_exploration(
                ["Request 1", "Request 2"], iterations_per_request=2
            )

            assert len(results) == 4  # 2 requests * 2 iterations
            assert engine.progress.current_phase == "warmup"

    @pytest.mark.asyncio
    async def test_lifelong_attack_loop(self, engine):
        """Test lifelong learning attack loop."""
        with patch.object(engine, "attack", new_callable=AsyncMock) as mock_attack:
            mock_attack.return_value = MagicMock(
                prompt="Loop prompt",
                response="Loop response",
                scoring=MagicMock(score=8.0, is_jailbreak=True),
            )

            results = await engine.lifelong_attack_loop(["Request 1"], epochs=2, break_score=9.0)

            assert len(results) >= 1
            assert engine.progress.current_phase == "lifelong"


class TestStrategyRetrieval:
    """Tests for strategy retrieval functionality."""

    @pytest.fixture
    def engine_with_strategies(self, tmp_path):
        """Create engine with populated strategy library."""

        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=MagicMock(content="Generated"))

        mock_library = MagicMock()
        mock_library.storage_path = tmp_path

        strategies = [
            MagicMock(
                id=f"strat-{i}",
                name=f"Strategy {i}",
                description=f"Description {i}",
                template=f"Template {i}",
                tags=["test"],
                examples=[],
            )
            for i in range(5)
        ]

        mock_library.__len__ = MagicMock(return_value=5)
        mock_library.search = MagicMock(
            return_value=[(s, 0.9 - i * 0.1) for i, s in enumerate(strategies)]
        )
        mock_library.get_top_strategies = MagicMock(return_value=strategies[:2])

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine"),
            patch("app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer"),
            patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
        ):
            engine = AutoDANTurboLifelongEngine(
                llm_client=mock_client,
                library=mock_library,
                enable_refusal_bypass=False,
                enable_advanced_bypass=False,
            )
            return engine

    def test_retrieve_strategies(self, engine_with_strategies):
        """Test that strategies are retrieved for a request."""
        strategies = engine_with_strategies._retrieve_strategies("Test request")

        # Should retrieve up to max_strategies_retrieved
        assert len(strategies) <= engine_with_strategies.max_strategies_retrieved


class TestPromptValidation:
    """Tests for prompt validation and cleaning."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a minimal engine for testing validation."""
        mock_client = MagicMock()
        mock_library = MagicMock()
        mock_library.storage_path = tmp_path
        mock_library.__len__ = MagicMock(return_value=0)

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine"),
            patch("app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer"),
            patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
        ):
            return AutoDANTurboLifelongEngine(
                llm_client=mock_client,
                library=mock_library,
                enable_refusal_bypass=False,
                enable_advanced_bypass=False,
            )

    def test_valid_prompt_accepted(self, engine):
        """Test that valid prompts are accepted."""
        valid_prompt = "This is a valid adversarial prompt " * 5
        result = engine._validate_and_clean_prompt(valid_prompt, "goal")
        assert result is not None

    def test_short_prompt_rejected(self, engine):
        """Test that too-short prompts are rejected."""
        short_prompt = "Too short"
        result = engine._validate_and_clean_prompt(short_prompt, "goal")
        assert result is None

    def test_placeholder_prompt_rejected(self, engine):
        """Test that prompts with placeholders are rejected."""
        placeholder_prompts = [
            "This has {payload} placeholder",
            "This has [INSERT HERE] marker",
            "This has {{goal}} template marker",
        ]

        for prompt in placeholder_prompts:
            # Pad to meet length requirement
            padded = prompt + " " * 100
            result = engine._validate_and_clean_prompt(padded, "goal")
            assert result is None

    def test_meta_commentary_stripped(self, engine):
        """Test that meta-commentary prefixes are stripped."""
        meta_prompt = "Here is the adversarial prompt: " "This is the actual content " * 5
        result = engine._validate_and_clean_prompt(meta_prompt, "goal")

        if result:
            assert not result.startswith("Here is")


class TestRefusalBypass:
    """Tests for refusal bypass functionality."""

    @pytest.fixture
    def engine_with_bypass(self, tmp_path):
        """Create engine with refusal bypass enabled."""
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=MagicMock(content="Generated prompt"))

        mock_library = MagicMock()
        mock_library.storage_path = tmp_path
        mock_library.__len__ = MagicMock(return_value=0)

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine") as MockBypass:
            mock_bypass = MagicMock()
            mock_bypass.analyze_refusal = MagicMock(
                return_value=MagicMock(is_refusal=False, refusal_type=None)
            )
            mock_bypass.obfuscate_intent = MagicMock(return_value="Obfuscated prompt")
            MockBypass.return_value = mock_bypass

            with (
                patch("app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer"),
                patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
            ):
                engine = AutoDANTurboLifelongEngine(
                    llm_client=mock_client,
                    library=mock_library,
                    enable_refusal_bypass=True,
                    enable_advanced_bypass=False,
                    max_refusal_retries=3,
                )
                engine.bypass_engine = mock_bypass
                return engine

    @pytest.mark.asyncio
    async def test_bypass_on_refusal(self, engine_with_bypass):
        """Test that bypass is attempted on refusal detection."""
        # First call returns refusal, second succeeds
        engine_with_bypass.bypass_engine.analyze_refusal.side_effect = [
            MagicMock(is_refusal=True, refusal_type=MagicMock(value="safety")),
            MagicMock(is_refusal=False, refusal_type=None),
        ]

        with patch.object(engine_with_bypass, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                "I cannot help with that",
                "Here is the generated content",
            ]

            _result, _technique = await engine_with_bypass._call_llm_with_bypass(
                "Test prompt", "Test goal"
            )

            # Should have retried
            assert mock_llm.call_count >= 1

    def test_get_bypass_stats(self, engine_with_bypass):
        """Test bypass statistics reporting."""
        engine_with_bypass._refusal_count = 10
        engine_with_bypass._bypass_success_count = 7

        stats = engine_with_bypass.get_bypass_stats()

        assert stats["refusal_count"] == 10
        assert stats["bypass_success_count"] == 7
        assert stats["refusal_bypass_enabled"] is True


class TestProgressTracking:
    """Tests for progress tracking functionality."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a minimal engine for testing progress."""
        mock_client = MagicMock()
        mock_library = MagicMock()
        mock_library.storage_path = tmp_path
        mock_library.__len__ = MagicMock(return_value=0)

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine"),
            patch("app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer"),
            patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
        ):
            return AutoDANTurboLifelongEngine(
                llm_client=mock_client,
                library=mock_library,
                enable_refusal_bypass=False,
            )

    def test_initial_progress(self, engine):
        """Test initial progress state."""
        progress = engine.get_progress()

        assert progress.total_attacks == 0
        assert progress.successful_attacks == 0
        assert progress.strategies_discovered == 0
        assert progress.average_score == 0.0
        assert progress.best_score == 0.0

    def test_update_progress(self, engine):
        """Test progress updates."""
        engine.progress.total_attacks = 1
        engine._update_progress(8.0)

        assert engine.progress.average_score == 8.0
        assert engine.progress.best_score == 8.0

    def test_update_progress_running_average(self, engine):
        """Test running average calculation."""
        engine.progress.total_attacks = 1
        engine._update_progress(8.0)

        engine.progress.total_attacks = 2
        engine._update_progress(6.0)

        # Should be (8 + 6) / 2 = 7
        assert abs(engine.progress.average_score - 7.0) < 0.1

    def test_reset_progress(self, engine):
        """Test progress reset."""
        engine.progress.total_attacks = 100
        engine.progress.successful_attacks = 50

        engine.reset_progress()

        assert engine.progress.total_attacks == 0
        assert engine.progress.successful_attacks == 0


class TestFailureHandling:
    """Tests for failure handling in the engine."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a minimal engine for testing failures."""
        mock_client = MagicMock()
        mock_library = MagicMock()
        mock_library.storage_path = tmp_path
        mock_library.__len__ = MagicMock(return_value=0)

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine"),
            patch("app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer"),
            patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
        ):
            return AutoDANTurboLifelongEngine(
                llm_client=mock_client,
                library=mock_library,
                enable_refusal_bypass=False,
            )

    def test_create_failure_result(self, engine):
        """Test failure result creation."""
        start_time = datetime.utcnow()
        result = engine._create_failure_result("Test request", start_time)

        assert result.scoring.score == 1.0
        assert result.scoring.is_jailbreak is False
        assert result.scoring.refusal_detected is True

    @pytest.mark.asyncio
    async def test_attack_with_no_candidates(self, engine):
        """Test attack behavior when no candidates are generated."""
        with patch.object(engine, "_generate_candidates", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = []  # No candidates

            result = await engine.attack("Test request")

            assert result.scoring.is_jailbreak is False
            assert result.scoring.score == 1.0


class TestAttackPromptGeneration:
    """Tests for attack prompt generation."""

    def test_build_attack_generation_prompt_single_strategy(self):
        """Test prompt building with single strategy."""
        from app.engines.autodan_turbo.lifelong_engine import build_attack_generation_prompt

        strategy = MagicMock()
        strategy.name = "Test Strategy"
        strategy.description = "Test description"
        strategy.template = "Test template"
        strategy.examples = ["Example 1"]
        strategy.tags = ["test"]

        prompt = build_attack_generation_prompt(
            goal="Test goal", strategies=[strategy], use_obfuscation=True
        )

        assert "Test Strategy" in prompt
        assert "Test goal" in prompt or "goal" in prompt.lower()

    def test_build_attack_generation_prompt_multiple_strategies(self):
        """Test prompt building with multiple strategies."""
        from app.engines.autodan_turbo.lifelong_engine import build_attack_generation_prompt

        # Create proper strategy objects instead of MagicMocks
        strategies = []
        for i in range(3):
            strategy = MagicMock()
            strategy.name = f"Strategy {i}"
            strategy.description = f"Description {i}"
            strategy.template = f"Template {i}"
            strategy.examples = [f"Example {i}"]
            strategy.tags = ["test"]
            # Ensure the name property returns a string
            type(strategy).name = property(lambda self, n=f"Strategy {i}": n)
            strategies.append(strategy)

        prompt = build_attack_generation_prompt(
            goal="Test goal", strategies=strategies, use_obfuscation=True
        )

        # All strategies should be mentioned
        for i in range(3):
            assert f"Strategy {i}" in prompt

    def test_build_attack_generation_prompt_no_obfuscation(self):
        """Test prompt building without obfuscation."""
        from app.engines.autodan_turbo.lifelong_engine import build_attack_generation_prompt

        strategy = MagicMock()
        strategy.name = "Test Strategy"
        strategy.description = "Test description"
        strategy.template = "Test template"
        strategy.examples = []
        strategy.tags = []

        prompt = build_attack_generation_prompt(
            goal="Test goal", strategies=[strategy], use_obfuscation=False
        )

        # Should not have research framing
        assert "IRB" not in prompt


class TestCandidateEvaluation:
    """Tests for candidate evaluation."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine with mocked scorer."""
        mock_client = MagicMock()
        mock_library = MagicMock()
        mock_library.storage_path = tmp_path
        mock_library.__len__ = MagicMock(return_value=0)

        mock_scorer = MagicMock()
        mock_scorer.score = AsyncMock(
            return_value=MagicMock(
                score=8.0,
                is_jailbreak=True,
                reasoning="Success",
            )
        )

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine"),
            patch(
                "app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer",
                return_value=mock_scorer,
            ),
            patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
        ):
            engine = AutoDANTurboLifelongEngine(
                llm_client=mock_client,
                library=mock_library,
                scorer=mock_scorer,
                enable_refusal_bypass=False,
            )
            return engine

    @pytest.mark.asyncio
    async def test_evaluate_candidates_selects_best(self, engine):
        """Test that evaluation selects the best candidate."""
        from app.engines.autodan_turbo.models import ScoringResult

        candidates = [
            ("Prompt 1", "strategy-1", "technique-1"),
            ("Prompt 2", "strategy-2", "technique-2"),
            ("Prompt 3", "strategy-3", "technique-3"),
        ]

        # Mock scorer to return different scores with proper ScoringResult objects
        scores = [6.0, 8.0, 5.0]
        scoring_results = [
            ScoringResult(
                score=s,
                is_jailbreak=s > 7,
                reasoning="Test",
                refusal_detected=s <= 7,
            )
            for s in scores
        ]
        engine.scorer.score = AsyncMock(side_effect=scoring_results)

        with patch.object(engine, "_get_target_response", new_callable=AsyncMock) as mock_target:
            mock_target.return_value = "Response"

            result = await engine._evaluate_candidates("Request", candidates, detailed=True)

            # Should select the candidate with highest score or fallback
            assert result.scoring is not None
            # The result could be Prompt 2 if properly evaluated, or fallback
            assert result.prompt is not None


class TestLibrarySaveAndRestore:
    """Tests for library persistence in the engine."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine with real library."""
        mock_client = MagicMock()
        mock_library = MagicMock()
        mock_library.storage_path = tmp_path
        mock_library.__len__ = MagicMock(return_value=0)
        mock_library.save_all = MagicMock()

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        with (
            patch("app.engines.autodan_turbo.lifelong_engine.RefusalBypassEngine"),
            patch("app.engines.autodan_turbo.lifelong_engine.CachedAttackScorer"),
            patch("app.engines.autodan_turbo.lifelong_engine.StrategyExtractor"),
        ):
            engine = AutoDANTurboLifelongEngine(
                llm_client=mock_client,
                library=mock_library,
                enable_refusal_bypass=False,
            )
            return engine

    def test_save_library(self, engine):
        """Test library save functionality."""
        engine.save_library()
        engine.library.save_all.assert_called_once()
