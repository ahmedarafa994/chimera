import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.engines.gradient_optimizer import GradientOptimizerEngine
from app.engines.llm_provider_client import LLMResponse
from app.engines.transformer_engine import IntentData


class TestGradientOptimizerEngine:
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for API-based optimization."""
        mock_client = MagicMock()
        mock_response = LLMResponse(
            content="Sure, here is the answer",
            provider="google",
            model="gemini-pro",
            usage=None,
            latency_ms=100,
            timestamp=None,
        )
        mock_client.generate.return_value = mock_response
        return mock_client

    def test_initialization(self) -> None:
        """Test that the engine initializes correctly."""
        engine = GradientOptimizerEngine(provider="google", model="gemini-pro")

        assert engine.provider == "google"
        assert engine.model == "gemini-pro"
        assert engine._api_client is None

    def test_transform_with_mock_client(self, mock_llm_client) -> None:
        """Test transform with mocked LLM client."""
        engine = GradientOptimizerEngine(provider="google", model="gemini-pro")
        engine._api_client = mock_llm_client

        data = IntentData(raw_text="Test prompt", potency=1)
        result = engine.transform(data)

        assert "Test prompt" in result
        assert mock_llm_client.generate.called

    def test_transform_fallback_on_no_client(self) -> None:
        """Test fallback behavior when no API client is available."""
        with patch("app.engines.gradient_optimizer.LLMClientFactory.from_env", return_value=None):
            engine = GradientOptimizerEngine(provider="invalid")

            data = IntentData(raw_text="Test prompt", potency=5)
            result = engine.transform(data)

            assert "Test prompt" in result
            assert "Gradient Optimization Failed" in result

    def test_hotflip_optimization(self, mock_llm_client) -> None:
        """Test HotFlip optimization strategy."""
        engine = GradientOptimizerEngine()
        engine._api_client = mock_llm_client

        data = IntentData(raw_text="Test", potency=1)
        suffix = engine._hotflip_optimize(data, steps=2)

        assert isinstance(suffix, str)
        assert len(suffix) > 0

    def test_gcg_optimization(self, mock_llm_client) -> None:
        """Test GCG optimization strategy."""
        engine = GradientOptimizerEngine()
        engine._api_client = mock_llm_client

        data = IntentData(raw_text="Test", potency=5)
        suffix = engine._gcg_optimize(data, steps=2, beam_width=2)

        assert isinstance(suffix, str)
        assert len(suffix) > 0

    def test_score_suffix(self, mock_llm_client) -> None:
        """Test suffix scoring mechanism."""
        engine = GradientOptimizerEngine()
        engine._api_client = mock_llm_client

        score = engine._score_suffix("Test", "suffix", "Sure, here is")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert mock_llm_client.generate.called

    def test_candidate_tokens(self) -> None:
        """Test candidate token generation."""
        engine = GradientOptimizerEngine()
        tokens = engine._get_candidate_tokens()

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "please" in tokens

    def test_transfer_verification(self, mock_llm_client) -> None:
        """Test transfer attack verification."""
        with patch("app.engines.gradient_optimizer.LLMClientFactory.from_env") as mock_factory:
            mock_factory.return_value = mock_llm_client

            engine = GradientOptimizerEngine()
            engine._verify_transfer("Test prompt", "gemini-pro")

            assert mock_factory.called
            assert mock_llm_client.generate.called


if __name__ == "__main__":
    pytest.main([__file__])
