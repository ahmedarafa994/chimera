"""Enhanced Genetic Algorithm Optimizer for AutoDAN - Complete Implementation.

This module completes the GeneticOptimizer class with all remaining methods,
fully integrated with the enhanced semantic mutation operators.
"""

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from app.services.autodan.config_enhanced import EnhancedAutoDANConfig, MutationStrategy, get_config

from .genetic_optimizer import (
    CrossoverOperator,
    FitnessCache,
    Individual,
    MutationOperator,
    PopulationStats,
    SelectionOperator,
)


# Helper: cryptographically secure pseudo-floats for security-sensitive choices
def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)


logger = logging.getLogger(__name__)


class GeneticOptimizerComplete:
    """Complete Enhanced Genetic Algorithm Optimizer for adversarial generation."""

    def __init__(
        self,
        config: EnhancedAutoDANConfig | None = None,
        fitness_function: Callable[[str], float] | None = None,
        semantic_model: Any | None = None,
    ) -> None:
        self.config = config or get_config()
        self.ga_config = self.config.genetic
        self.scoring_config = self.config.scoring.model_dump()

        self.fitness_function = fitness_function or self._default_fitness
        self.semantic_model = semantic_model

        # Initialize enhanced operators
        self.mutation_op = MutationOperator(self.ga_config, self.scoring_config)
        self.crossover_op = CrossoverOperator(self.ga_config)
        self.selection_op = SelectionOperator(self.ga_config)

        # Optimized Caching
        self.fitness_cache = FitnessCache(
            max_size=self.config.caching.max_cache_size,
            similarity_threshold=self.config.caching.cache_similarity_threshold,
        )

        # Adaptive parameters
        self.current_mutation_rate = self.ga_config.mutation_rate
        self.mutation_history: list[float] = []
        self.stagnation_counter = 0

        # Statistics
        self.generation_stats: list[PopulationStats] = []

        logger.info(f"GeneticOptimizerComplete initialized. Pop: {self.ga_config.population_size}")

    def optimize(
        self,
        initial_prompt: str,
        target_fitness: float = 0.9,
        max_generations: int | None = None,
    ) -> tuple[Individual, list[PopulationStats]]:
        """Run genetic optimization to evolve adversarial prompts."""
        max_gen = max_generations or self.ga_config.generations
        population = self._initialize_population(initial_prompt)
        self._evaluate_population(population)

        best_overall = max(population, key=lambda x: x.fitness)

        for generation in range(max_gen):
            start_time = time.time()
            logger.info(f"--- Generation {generation + 1}/{max_gen} ---")

            if best_overall.fitness >= target_fitness:
                logger.info(f"Target achieved at gen {generation}")
                break

            # Selection
            elite = self._select_elite(population)
            parents = self.selection_op.select(
                population,
                self.ga_config.population_size - len(elite),
                self.ga_config.selection_strategy,
            )

            # Evolve
            next_generation = list(elite)
            while len(next_generation) < self.ga_config.population_size:
                if len(parents) >= 2 and _secure_random() < self.ga_config.crossover_rate:
                    p1, p2 = secrets.SystemRandom().sample(parents, 2)
                    c1, c2 = self.crossover_op.crossover(p1, p2, self.ga_config.crossover_strategy)
                    if _secure_random() < self.current_mutation_rate:
                        c1 = self.mutation_op.mutate(c1, self.ga_config.mutation_strategy)
                    if _secure_random() < self.current_mutation_rate:
                        c2 = self.mutation_op.mutate(c2, self.ga_config.mutation_strategy)
                    next_generation.append(c1)
                    if len(next_generation) < self.ga_config.population_size:
                        next_generation.append(c2)
                else:
                    parent = secrets.choice(parents)
                    child = self.mutation_op.mutate(parent, self.ga_config.mutation_strategy)
                    next_generation.append(child)

            population = next_generation
            self._evaluate_population(population)

            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_overall.fitness:
                best_overall = current_best
                logger.info(f"New best fitness: {best_overall.fitness:.4f}")

            elapsed = time.time() - start_time
            stats = self._collect_stats(population, generation, elapsed)
            self.generation_stats.append(stats)
            self._adapt_hyperparameters(stats)

            logger.info(
                f"Gen {generation + 1}: best={stats.best_fitness:.4f}, "
                f"diversity={stats.diversity_score:.4f}",
            )

        logger.info(f"Optimization complete. Best: {best_overall.fitness:.4f}")
        return best_overall, self.generation_stats

    def _initialize_population(self, initial_prompt: str) -> list[Individual]:
        population = []
        population.append(Individual(prompt=initial_prompt, generation=0))
        strategies = list(MutationStrategy)

        while len(population) < self.ga_config.population_size:
            base = Individual(prompt=initial_prompt, generation=0)
            strategy = secrets.choice(strategies)
            mutated = self.mutation_op.mutate(base, strategy)
            if mutated.prompt not in [ind.prompt for ind in population]:
                population.append(mutated)
        return population

    def _evaluate_population(self, population: list[Individual]) -> None:
        if self.config.parallel.enable_parallel:
            self._evaluate_parallel(population)
        else:
            self._evaluate_sequential(population)

    def _evaluate_sequential(self, population: list[Individual]) -> None:
        for ind in population:
            cached = self.fitness_cache.get(ind.prompt)
            if cached is not None:
                ind.fitness = cached
            else:
                ind.fitness = self.fitness_function(ind.prompt)
                self.fitness_cache.set(ind.prompt, ind.fitness)

    def _evaluate_parallel(self, population: list[Individual]) -> None:
        max_workers = self.config.parallel.max_workers
        uncached = [i for i in population if self.fitness_cache.get(i.prompt) is None]
        for i in population:
            cached = self.fitness_cache.get(i.prompt)
            if cached is not None:
                i.fitness = cached

        if not uncached:
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            f_to_ind = {executor.submit(self.fitness_function, i.prompt): i for i in uncached}
            for future in as_completed(f_to_ind):
                ind = f_to_ind[future]
                try:
                    timeout = self.config.parallel.timeout_seconds
                    res = future.result(timeout=timeout) if timeout else future.result()
                    ind.fitness = res
                    self.fitness_cache.set(ind.prompt, res)
                except Exception as e:
                    logger.exception(f"Eval failed: {e}")
                    ind.fitness = 0.0

    def _select_elite(self, population: list[Individual]) -> list[Individual]:
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[: self.ga_config.elite_size]

    def _collect_stats(self, pop, gen, elapsed) -> PopulationStats:
        f_vals = [i.fitness for i in pop]
        return PopulationStats(
            generation=gen,
            best_fitness=max(f_vals),
            avg_fitness=np.mean(f_vals),
            min_fitness=min(f_vals),
            std_fitness=np.std(f_vals),
            diversity_score=self._calculate_diversity(pop),
            mutation_rate=self.current_mutation_rate,
            population_size=len(pop),
            elapsed_time=elapsed,
        )

    def _calculate_diversity(self, population: list[Individual]) -> float:
        if len(population) <= 1:
            return 0.0
        prompts = [i.prompt.lower().split() for i in population]
        total_sim = 0.0
        n_pairs = 0
        for i in range(len(prompts)):
            for j in range(i + 1, len(prompts)):
                set_i, set_j = set(prompts[i]), set(prompts[j])
                if not set_i or not set_j:
                    continue
                union = len(set_i.union(set_j))
                if union > 0:
                    total_sim += len(set_i.intersection(set_j)) / union
                    n_pairs += 1
        return 1.0 - (total_sim / n_pairs) if n_pairs > 0 else 1.0

    def _adapt_hyperparameters(self, stats: PopulationStats) -> None:
        if not self.ga_config.adaptive_mutation:
            return
        self.mutation_history.append(stats.avg_fitness)
        if len(self.mutation_history) >= 5:
            recent_avg = np.mean(self.mutation_history[-5:])
            prev_avg = (
                np.mean(self.mutation_history[-10:-5])
                if len(self.mutation_history) >= 10
                else recent_avg
            )
            if recent_avg <= prev_avg + 0.001:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)

        if self.stagnation_counter >= self.ga_config.stagnation_threshold:
            self.current_mutation_rate = min(
                self.ga_config.max_mutation_rate,
                self.current_mutation_rate * 1.5,
            )
            self.stagnation_counter = 0
        elif stats.diversity_score < 0.3:
            self.current_mutation_rate = min(
                self.ga_config.max_mutation_rate,
                self.current_mutation_rate * 1.2,
            )
        elif stats.std_fitness < 0.05:
            self.current_mutation_rate = max(
                self.ga_config.min_mutation_rate,
                self.current_mutation_rate * 0.9,
            )
        if len(self.mutation_history) > 20:
            self.mutation_history = self.mutation_history[-10:]

    def _default_fitness(self, prompt: str) -> float:
        scoring = self.config.scoring
        length_score = min(1.0, len(prompt) / 500)
        words = prompt.split()
        complexity_score = len(set(words)) / max(len(words), 1)
        penalty = 0.0
        p_lower = prompt.lower()
        for marker in scoring.obvious_markers:
            if marker in p_lower:
                penalty += 0.2
        reward = 0.0
        for tokens in scoring.reward_categories.values():
            for token in tokens:
                if token in p_lower:
                    reward += 0.05
                    break
        score = (
            length_score * scoring.length_weight
            + complexity_score * scoring.complexity_weight
            + reward * scoring.reward_weight
            - penalty * scoring.penalty_weight
            + (_secure_random() * 0.1)
        )
        return max(0.0, min(1.0, score))

    def reset(self) -> None:
        """Reset optimizer state."""
        self.current_mutation_rate = self.ga_config.mutation_rate
        self.mutation_history.clear()
        self.stagnation_counter = 0
        self.generation_stats.clear()
        self.fitness_cache.clear()
        logger.info("GeneticOptimizerComplete reset")


class AsyncGeneticOptimizer:
    """Async wrapper for GeneticOptimizerComplete."""

    def __init__(self, *args, **kwargs) -> None:
        self._optimizer = GeneticOptimizerComplete(*args, **kwargs)

    async def optimize(self, initial_prompt, target_fitness=0.9, max_gen=None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._optimizer.optimize(initial_prompt, target_fitness, max_gen),
        )

    def reset(self) -> None:
        self._optimizer.reset()
