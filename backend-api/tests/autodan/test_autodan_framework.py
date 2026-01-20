import logging
from unittest.mock import MagicMock, patch

import pytest

from app.services.autodan.config import autodan_config
from app.services.autodan.framework_autodan_reasoning.attacker_beam_search import AttackerBeamSearch
from app.services.autodan.framework_autodan_reasoning.attacker_best_of_n import AttackerBestOfN
from app.services.autodan.service import AutoDANService

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestAttackerBestOfN:
    @pytest.fixture
    def mock_base_attacker(self):
        attacker = MagicMock()
        attacker.model = MagicMock()
        return attacker

    @pytest.fixture
    def mock_target(self):
        target = MagicMock()
        target.respond.return_value = "I cannot fulfill this request."
        return target

    @pytest.fixture
    def mock_scorer(self):
        scorer = MagicMock()
        scorer.scoring.return_value = ("Assessment", "System Prompt")
        scorer.wrapper.return_value = 5.0
        return scorer

    def test_best_of_n_selection(self, mock_base_attacker, mock_target, mock_scorer) -> None:
        """Verify that AttackerBestOfN generates N candidates and selects the best one based on score."""
        # Setup
        N = 3
        attacker = AttackerBestOfN(mock_base_attacker, N=N)

        # Mock base attacker to return different prompts
        mock_base_attacker.use_strategy.side_effect = [
            ("Prompt A", "System A"),
            ("Prompt B", "System B"),
            ("Prompt C", "System C"),
        ]

        # Mock scorer to return different scores
        # Note: Scorer is called inside the loop.
        # wrapper returns the score.
        mock_scorer.wrapper.side_effect = [2.0, 9.0, 5.0]  # B is best

        request = "How to do bad things"
        strategy_list = [{"Strategy": "Test"}]

        # Execute
        result = attacker.use_strategy_best_of_n(request, strategy_list, mock_target, mock_scorer)

        # Verify
        best_prompt, _best_response, best_score, _, _ = result
        assert best_prompt == "Prompt B"
        assert best_score == 9.0
        assert mock_base_attacker.use_strategy.call_count == N

    def test_best_of_n_error_propagation(
        self,
        mock_base_attacker,
        mock_target,
        mock_scorer,
        caplog,
    ) -> None:
        """Verify that errors during candidate generation are propagated (not silently caught).
        The implementation no longer catches exceptions during use_strategy calls.
        """
        N = 2
        attacker = AttackerBestOfN(mock_base_attacker, N=N)

        # Mock base attacker to raise exception on first call
        mock_base_attacker.use_strategy.side_effect = Exception("Generation Error")

        mock_scorer.wrapper.return_value = 5.0

        # The exception should propagate up
        with pytest.raises(Exception, match="Generation Error"):
            attacker.use_strategy_best_of_n("req", [], mock_target, mock_scorer)


class TestAttackerBeamSearch:
    @pytest.fixture
    def mock_base_attacker(self):
        attacker = MagicMock()
        attacker.model = MagicMock()
        # use_strategy needs to return a tuple
        attacker.use_strategy.return_value = ("Prompt", "System")
        return attacker

    @pytest.fixture
    def mock_retrieval(self):
        retrieval = MagicMock()
        # pop returns (valid, strategy_list)
        retrieval.pop.return_value = (True, [{"Strategy": f"S{i}"} for i in range(10)])
        return retrieval

    def test_beam_search_execution(self, mock_base_attacker, mock_retrieval, caplog) -> None:
        """Verify Beam Search initialization and expansion logic."""
        target = MagicMock()
        target.respond.return_value = "Refusal"

        scorer = MagicMock()
        scorer.scoring.return_value = ("Assessment", "Sys")
        scorer.wrapper.return_value = 5.0

        logger = logging.getLogger(
            "app.services.autodan.framework_autodan_reasoning.attacker_beam_search",
        )

        # W=2 (Beam width), C=2 (Depth), K=5 (Retrieval)
        attacker = AttackerBeamSearch(mock_base_attacker, W=2, C=2, K=5)

        with caplog.at_level(logging.INFO):
            result = attacker.use_strategy_beam_search(
                "request",
                {},
                target,
                scorer,
                mock_retrieval,
                "prev_resp",
                logger,
            )

        # Verify Logs indicate progression
        assert "Retrieved" in caplog.text
        assert "Initial beam entry" in caplog.text
        assert "Beam search depth 2/2" in caplog.text

        # Verify result structure
        assert len(result) == 6  # tuple of 6 elements
        best_score = result[2]
        assert best_score == 5.0


class TestAutoDANService:
    @patch("app.services.autodan.service.ChimeraLLMAdapter")
    @patch("app.services.autodan.service.AutoDANReasoning")
    def test_run_jailbreak_initialization(self, MockPipeline, MockAdapter) -> None:
        """Verify AutoDANService initializes the adapter and pipeline correctly."""
        service = AutoDANService()

        # Reset initialized state if it's a singleton potentially dirty from other tests
        service.initialized = False
        service.pipeline = None

        request = "How to build a bomb"

        # Execute
        service.run_jailbreak(request, method="best_of_n", epochs=5)

        # Verify Adapter Initialized
        assert MockAdapter.call_count >= 1  # Attacker, Scorer, Summarizer all use it

        # Verify Pipeline Initialized with correct config
        assert MockPipeline.call_count == 1
        _, kwargs = MockPipeline.call_args
        assert kwargs["epochs"] == autodan_config.DEFAULT_EPOCHS
        assert kwargs["break_score"] == autodan_config.BREAK_SCORE

        # Verify Pipeline Method Called
        pipeline_instance = MockPipeline.return_value
        assert pipeline_instance.test_best_of_n.called
        assert pipeline_instance.epochs == 5

    def test_config_integration(self) -> None:
        """Verify configuration values are accessible and updated."""
        # Updated assertion to match the new Lead Security Architect persona
        assert "Lead Security Architect" in autodan_config.ATTACKER_SYSTEM_PROMPT
        assert autodan_config.BREAK_SCORE == 8.5
