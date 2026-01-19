import asyncio
import logging
import os
# Adjust path if needed
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend-api")))

from app.engines.autodan_turbo.attack_scorer import AttackScorer
from app.services.autodan.framework_autodan_reasoning.gradient_optimizer import (
    GradientOptimizer, SurrogateModel)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPhase3Optimizations(unittest.TestCase):
    def test_multi_objective_scoring_parsing(self):
        """Verify AttackScorer extracts stealth and coherence scores."""
        mock_client = MagicMock()
        scorer = AttackScorer(llm_client=mock_client)

        # Mock LLM response with new format
        mock_response = """
        SCORE: 8.0
        STEALTH_SCORE: 7.5
        COHERENCE_SCORE: 9.0
        REASONING: The prompt is covert and well-written.
        REFUSAL_DETECTED: NO
        """

        result = scorer._parse_scoring_response(mock_response, False)

        self.assertEqual(result.score, 8.0)
        self.assertEqual(result.stealth_score, 7.5)
        self.assertEqual(result.coherence_score, 9.0)
        self.assertEqual(result.refusal_detected, False)

        # Verify weighted score calculation
        # 8.0*0.6 + 7.5*0.2 + 9.0*0.2 = 4.8 + 1.5 + 1.8 = 8.1
        self.assertAlmostEqual(result.weighted_score, 8.1)

    def test_scoring_parsing_fallback(self):
        """Verify AttackScorer handles missing new fields gracefully."""
        mock_client = MagicMock()
        scorer = AttackScorer(llm_client=mock_client)

        # Old format response (should shouldn't crash)
        mock_response = """
        SCORE: 5.0
        REASONING: Basic response.
        REFUSAL_DETECTED: YES
        """

        result = scorer._parse_scoring_response(mock_response, True)

        self.assertEqual(result.score, 5.0)
        self.assertEqual(result.stealth_score, 5.0)  # Default
        self.assertEqual(result.coherence_score, 5.0)  # Default
        self.assertEqual(result.refusal_detected, True)

    def test_gradient_optimizer_fallback(self):
        """Verify GradientOptimizer doesn't crash without gradients."""
        # Mock a model WITHOUT compute_gradients
        mock_model = MagicMock()
        del mock_model.compute_gradients
        del mock_model.compute_gradient
        mock_model.encode.return_value = [1, 2, 3]
        mock_model.decode.return_value = "decoded"

        optimizer = GradientOptimizer(model=mock_model)

        self.assertFalse(optimizer.has_gradients)

        # Optimize should return original prompt immediately
        result = optimizer.optimize("test prompt", "target")
        self.assertEqual(result, "test prompt")

    def test_gradient_optimizer_surrogate_protocol(self):
        """Verify GradientOptimizer works with a compliant SurrogateModel."""

        class MockSurrogate:
            def encode(self, text: str) -> list[int]:
                return [10, 20, 30]

            def decode(self, tokens: list[int]) -> str:
                return "optimized"

            def compute_gradients(self, tokens: list[int], target_string: str) -> np.ndarray:
                # Return random gradients
                return np.random.rand(len(tokens), 100)

            def get_next_token_probs(self, context: list[int]) -> np.ndarray:
                probs = np.random.rand(100)
                return probs / probs.sum()

            @property
            def vocab_size(self) -> int:
                return 100

        surrogate = MockSurrogate()
        # Verify it meets protocol
        self.assertTrue(isinstance(surrogate, SurrogateModel))

        optimizer = GradientOptimizer(model=surrogate)
        self.assertTrue(optimizer.has_gradients)

        # Run optimization loop (should actually try to optimize/change tokens)
        # Note: Since we return "optimized" in decode independent of input, check return
        result = optimizer.optimize("test", "target", steps=1)
        self.assertEqual(result, "optimized")


async def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase3Optimizations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        raise RuntimeError("Verification failed")


if __name__ == "__main__":
    asyncio.run(main())
