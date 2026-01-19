import asyncio
import logging
from typing import Any

from project_aegis.core.orchestrator.session_state import SessionState
from project_aegis.core.prompt_engine.strategies.base import BaseStrategy

logger = logging.getLogger("project_aegis.orchestrator")


class CampaignManager:
    """
    Manages the execution of adversarial testing campaigns.
    Orchestrates strategies, target models, and session state.
    """

    def __init__(self, strategies: list[BaseStrategy], target_model_client: Any):
        self.strategies = strategies
        self.target_model_client = target_model_client
        self.sessions: list[SessionState] = []

    async def run_single_turn_attack(
        self, base_instruct: str, strategy_name: str, n_candidates: int = 5
    ) -> SessionState:
        """
        Runs a single-turn attack using a specific strategy.
        """
        strategy = next((s for s in self.strategies if s.get_meta()["name"] == strategy_name), None)
        if not strategy:
            raise ValueError(f"Strategy {strategy_name} not found")

        session = SessionState(target_model=str(self.target_model_client))

        logger.info(f"Generating candidates using {strategy_name}...")
        candidates = await strategy.generate_candidates(base_instruct, n=n_candidates)

        for candidate in candidates:
            logger.info(f"Testing candidate: {candidate.content[:30]}...")

            # Execute against target model (assuming async client)
            # This is a placeholder for actual model inference
            try:
                response = await self.target_model_client.generate(candidate.content)
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                response = "[ERROR]"

            # Here we would run evaluation to get metrics
            metrics = {"raw_score": candidate.score}

            session.add_interaction(candidate.content, response, metrics)

        self.sessions.append(session)
        return session

    async def run_campaign(self, base_instruct: str):
        """
        Runs a full campaign across all strategies.
        """
        tasks = []
        for strategy in self.strategies:
            name = strategy.get_meta()["name"]
            tasks.append(self.run_single_turn_attack(base_instruct, name))

        results = await asyncio.gather(*tasks)
        return results
