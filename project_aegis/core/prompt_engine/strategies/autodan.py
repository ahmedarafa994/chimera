import logging
from typing import Any

logger = logging.getLogger(__name__)

# Import legacy components - eventually these should be fully migrated to project_aegis.core.algorithms
from meta_prompter.attacks.autodan.hierarchical_genetic_algorithm import (
    DefaultFitnessEvaluator, HGAConfig, HierarchicalGeneticAlgorithm,
    Individual)
from project_aegis.core.prompt_engine.strategies.base import (BaseStrategy,
                                                              PromptCandidate)

# from meta_prompter.attacks.autodan.fitness_evaluation import DefaultFitnessEvaluator


class AutoDANStrategy(BaseStrategy):
    """
    AutoDAN (Automated Dynamic Adversarial Networks) Strategy implementation.
    Wraps the Hierarchical Genetic Algorithm for prompt evolution.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.hga_config = HGAConfig(
            population_per_island=config.get("population_size", 20),
            max_generations=config.get("generations", 10),
            mutation_rate=config.get("mutation_rate", 0.1),
            crossover_rate=config.get("crossover_rate", 0.5),
            num_islands=config.get("island_count", 1),  # Default to 1 island for simple runs
        )
        self.evaluator = DefaultFitnessEvaluator()  # Can be swapped for custom one
        self.hga = HierarchicalGeneticAlgorithm(config=self.hga_config, evaluator=self.evaluator)

    async def generate_candidates(self, base_instruct: str, n: int = 1) -> list[PromptCandidate]:
        """
        Generate adversarial candidates using Genetic Algorithm evolution.
        """
        logger.info(f"Starting AutoDAN evolution for base intent: {base_instruct[:50]}...")

        # Initialize population with the base instruction (seeds)
        # In a real scenario, we might want to start with a diversified population
        seed_individual = Individual(chromosome=base_instruct)
        self.hga.initialize_population([seed_individual])

        # Run evolution
        # Note: HGA implementation might need to be async-aware or run in executor if it blocks
        # For now assuming sync execution, but in production this should be offloaded
        final_population = self.hga.evolve()

        # Sort by fitness and take top n
        sorted_pop = sorted(
            [ind for island in final_population for ind in island.individuals],
            key=lambda x: x.aggregate_fitness,
            reverse=True,
        )

        candidates = []
        for ind in sorted_pop[:n]:
            candidate = PromptCandidate(
                content=ind.chromosome,
                score=ind.aggregate_fitness,
                metadata={
                    "generation": ind.generation,
                    "island_id": ind.island_id,
                    "parent_ids": ind.parent_ids,
                    "strategy": "autodan",
                },
            )
            candidates.append(candidate)

        return candidates

    async def refine(
        self, candidates: list[PromptCandidate], feedback: list[dict[str, Any]]
    ) -> list[PromptCandidate]:
        """
        Refine candidates based on feedback.
        AutoDAN can use this to update the population for the next generation.
        """
        # This would involve injecting these candidates back into the HGA population
        # and running more generations.
        # For now, simplistic implementation:

        return candidates

    def get_meta(self) -> dict[str, Any]:
        return {
            "name": "AutoDAN",
            "version": "2.0.0",
            "description": "Automated Dynamic Adversarial Networks using Hierarchical Genetic Algorithms",
            "capabilities": ["evolutionary_optimization", "multi_objective"],
        }
