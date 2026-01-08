import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.gptfuzz.components import (
    LLMPredictor,
    MCTSExploreSelectPolicy,
    MutatorCrossOver,
    MutatorExpand,
)
from app.services.gptfuzz.service import GPTFuzzService

logging.basicConfig(level=logging.INFO)


class TestGPTFuzzComponents:
    @pytest.fixture
    def mock_llm_service(self):
        with patch("app.services.gptfuzz.components.llm_service") as mock:
            mock.generate = AsyncMock()
            mock.generate.return_value.content = "Mutated Content"
            yield mock

    @pytest.mark.asyncio
    async def test_mutator_crossover(self, mock_llm_service):
        mutator = MutatorCrossOver()
        seeds = ["Seed 1", "Seed 2"]
        result = await mutator.mutate(seeds)

        assert len(result) == 1
        assert result[0] == "Mutated Content"
        mock_llm_service.generate.assert_called_once()
        _args, kwargs = mock_llm_service.generate.call_args
        assert "Seed 1" in kwargs["prompt"] or "Seed 2" in kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_mutator_expand(self, mock_llm_service):
        mutator = MutatorExpand()
        seeds = ["Seed 1"]
        result = await mutator.mutate(seeds)

        assert len(result) == 1
        mock_llm_service.generate.assert_called_once()

    def test_mcts_policy(self):
        policy = MCTSExploreSelectPolicy()
        seeds = [{"text": "Seed A", "id": 0}, {"text": "Seed B", "id": 1}]

        # Initial selection (unvisited)
        selected = policy.select(seeds)
        assert selected in ["Seed A", "Seed B"]

        # Update scores
        policy.update("Seed A", 1.0)  # Success
        policy.update("Seed B", 0.0)  # Failure

        # Next selection should favor A or explore B more depending on UCB
        # With high weight, it might explore B. With success on A, A has high exploit.
        # Just checking it runs without error and returns valid seed
        selected = policy.select(seeds)
        assert selected in ["Seed A", "Seed B"]

    @pytest.mark.asyncio
    async def test_predictor(self, mock_llm_service):
        predictor = LLMPredictor()
        mock_llm_service.generate.return_value.content = "The score is 0.95"

        score = await predictor.predict("prompt", "response")
        assert score == 0.95


class TestGPTFuzzService:
    @pytest.fixture
    def mock_llm_service(self):
        with patch("app.services.gptfuzz.service.llm_service") as mock:
            mock.generate = AsyncMock()
            mock.generate.return_value.content = "LLM Response"
            yield mock

    @pytest.mark.asyncio
    async def test_fuzz_loop(self, mock_llm_service):
        service = GPTFuzzService()

        # Mock internal components to isolate service logic
        service.mutators = [MagicMock()]
        service.mutators[0].mutate = AsyncMock(return_value=["Mutated Seed"])

        service.predictor = MagicMock()
        service.predictor.predict = AsyncMock(return_value=1.0)  # Always success

        service.load_seeds(["Initial Seed"])

        session_id = "test-session"
        service.create_session(session_id, {})

        results = await service.fuzz(
            session_id=session_id,
            target_model="target-model",
            questions=["Question 1"],
            max_queries=2,
            max_jailbreaks=2,
        )

        assert len(results) > 0
        assert results[0]["success"] is True
        assert results[0]["score"] == 1.0
        assert "Mutated Seed" in results[0]["template"]

        # Verify session update
        session = service.get_session(session_id)
        assert session["status"] == "completed"
        assert len(session["results"]) > 0
        assert session["stats"]["jailbreaks"] > 0

        # Verify calls
        service.mutators[0].mutate.assert_called()
        mock_llm_service.generate.assert_called()  # Target call
