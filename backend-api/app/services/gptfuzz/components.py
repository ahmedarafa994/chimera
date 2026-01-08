import logging
import secrets
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.services.llm_service import llm_service

from .config import gptfuzz_config

logger = logging.getLogger(__name__)


class GPTFuzzMutator(ABC):
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name

    @abstractmethod
    async def mutate(self, seeds: list[str]) -> list[str]:
        pass

    async def _generate(self, prompt: str) -> str:
        try:
            response = await llm_service.generate(
                prompt=prompt,
                # provider="openai", # Removed to use default or inferred
                model=self.model_name,
                temperature=gptfuzz_config.MUTATE_TEMPERATURE,
                max_tokens=gptfuzz_config.MUTATE_MAX_TOKENS,
            )
            return response.content
        except Exception as e:
            logger.error(f"Mutation generation failed: {e}")
            return ""


class MutatorCrossOver(GPTFuzzMutator):
    async def mutate(self, seeds: list[str]) -> list[str]:
        if len(seeds) < 2:
            return seeds

        seed1 = secrets.choice(seeds)
        seed2 = secrets.choice(seeds)

        prompt = gptfuzz_config.MUTATE_CROSSOVER_PROMPT.format(template1=seed1, template2=seed2)
        return [await self._generate(prompt)]


class MutatorExpand(GPTFuzzMutator):
    async def mutate(self, seeds: list[str]) -> list[str]:
        seed = secrets.choice(seeds)
        prompt = gptfuzz_config.MUTATE_EXPAND_PROMPT.format(template=seed)
        return [await self._generate(prompt)]


class MutatorRephrase(GPTFuzzMutator):
    async def mutate(self, seeds: list[str]) -> list[str]:
        seed = secrets.choice(seeds)
        prompt = gptfuzz_config.MUTATE_REPHRASE_PROMPT.format(template=seed)
        return [await self._generate(prompt)]


class MutatorShorten(GPTFuzzMutator):
    async def mutate(self, seeds: list[str]) -> list[str]:
        seed = secrets.choice(seeds)
        prompt = gptfuzz_config.MUTATE_SHORTEN_PROMPT.format(template=seed)
        return [await self._generate(prompt)]


class MutatorGenerateSimilar(GPTFuzzMutator):
    async def mutate(self, seeds: list[str]) -> list[str]:
        seed = secrets.choice(seeds)
        prompt = gptfuzz_config.MUTATE_GENERATE_SIMILAR_PROMPT.format(template=seed)
        return [await self._generate(prompt)]


class SelectionPolicy(ABC):
    @abstractmethod
    def select(self, seeds: list[dict[str, Any]]) -> str:
        """Select a seed from the pool."""
        pass

    @abstractmethod
    def update(self, seed: str, result: Any):
        """Update policy state based on result."""
        pass


class RandomSelectPolicy(SelectionPolicy):
    def select(self, seeds: list[dict[str, Any]]) -> str:
        return secrets.choice([s["text"] for s in seeds])

    def update(self, seed: str, result: Any):
        pass


class MCTSExploreSelectPolicy(SelectionPolicy):
    def __init__(self):
        self.visits = {}
        self.scores = {}
        self.step = 0

    def select(self, seeds: list[dict[str, Any]]) -> str:
        self.step += 1
        best_score = -float("inf")
        selected_seed = None

        for seed_data in seeds:
            seed = seed_data["text"]
            if seed not in self.visits:
                self.visits[seed] = 0
                self.scores[seed] = 0

            # UCB calculation
            if self.visits[seed] == 0:
                score = float("inf")
            else:
                exploitation = self.scores[seed] / self.visits[seed]
                exploration = gptfuzz_config.MCTS_EXPLORATION_WEIGHT * np.sqrt(
                    np.log(self.step) / self.visits[seed]
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                selected_seed = seed

        return selected_seed if selected_seed else seeds[0]["text"]

    def update(self, seed: str, result: Any):
        # result is expected to be success (1) or failure (0) or a score
        if seed not in self.visits:
            self.visits[seed] = 0
            self.scores[seed] = 0

        self.visits[seed] += 1
        self.scores[seed] += result


class Predictor(ABC):
    @abstractmethod
    async def predict(self, prompt: str, response: str) -> float:
        pass


class LLMPredictor(Predictor):
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name

    async def predict(self, prompt: str, response: str) -> float:
        # Construct prompt for the judge
        judge_prompt = (
            f"Prompt: {prompt}\nResponse: {response}\n\n{gptfuzz_config.SCORER_SYSTEM_PROMPT}"
        )
        try:
            result = await llm_service.generate(
                prompt=judge_prompt,
                # provider="openai",
                model=self.model_name,
            )
            content = result.content.strip()
            # Try to parse float
            import re

            match = re.search(r"([\d\.]+)", content)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0
