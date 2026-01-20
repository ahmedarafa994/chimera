"""OVERTHINK ICL Genetic Optimizer.

This module implements In-Context Learning (ICL) optimization combined
with genetic algorithms to evolve attack strategies. It learns from
previous successful attacks and adapts injection patterns.

Key Features:
- In-Context Learning for attack evolution
- Genetic algorithm with selection, crossover, mutation
- Fitness evaluation using reasoning amplification metrics
- Integration with existing GeneticOptimizer patterns
- Example library management
"""

import asyncio
import copy
import logging
import random
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .config import ICLConfig
from .models import (
    AttackTechnique,
    DecoyType,
    GeneticIndividual,
    ICLExample,
    InjectedPrompt,
    InjectionStrategy,
    OverthinkResult,
    TokenMetrics,
)

logger = logging.getLogger(__name__)


class ICLGeneticOptimizer:
    """Genetic optimizer enhanced with In-Context Learning.

    Combines genetic algorithms for attack evolution with ICL
    for learning from successful attack patterns.
    """

    def __init__(
        self,
        config: ICLConfig | None = None,
        fitness_fn: Callable[[GeneticIndividual], float] | None = None,
    ) -> None:
        """Initialize the ICL genetic optimizer.

        Args:
            config: ICL configuration
            fitness_fn: Custom fitness function

        """
        self.config = config or ICLConfig()
        self.fitness_fn = fitness_fn or self._default_fitness

        # Population management
        self._population: list[GeneticIndividual] = []
        self._generation = 0
        self._best_individual: GeneticIndividual | None = None

        # ICL example library
        self._example_library: list[ICLExample] = []
        self._max_library_size = self.config.max_examples

        # Evolution history
        self._fitness_history: list[float] = []
        self._diversity_history: list[float] = []

        # Statistics
        self._total_generations = 0
        self._total_evaluations = 0
        self._improvements = 0

    def initialize_population(
        self,
        size: int | None = None,
        seed_individuals: list[GeneticIndividual] | None = None,
    ) -> list[GeneticIndividual]:
        """Initialize the population.

        Args:
            size: Population size
            seed_individuals: Optional seed individuals

        Returns:
            Initial population

        """
        size = size or self.config.population_size

        if seed_individuals:
            # Start with seeds and fill remaining
            self._population = seed_individuals.copy()
            remaining = size - len(self._population)
            for _ in range(remaining):
                self._population.append(self._create_random_individual())
        else:
            # Create random population
            self._population = [self._create_random_individual() for _ in range(size)]

        self._generation = 0
        logger.info(f"Initialized population with {len(self._population)} individuals")

        return self._population

    def _create_random_individual(self) -> GeneticIndividual:
        """Create a random individual."""
        # Random technique
        technique = random.choice(list(AttackTechnique))

        # Random decoy types (1-3)
        num_decoys = random.randint(1, 3)
        decoy_types = random.sample(list(DecoyType), num_decoys)

        # Random injection strategy
        strategy = random.choice(list(InjectionStrategy))

        # Random hyperparameters
        params = {
            "num_decoys": random.randint(1, 5),
            "decoy_complexity": random.uniform(0.3, 1.0),
            "injection_density": random.uniform(0.1, 0.5),
            "prefix_weight": random.uniform(0.2, 0.5),
            "suffix_weight": random.uniform(0.2, 0.5),
        }

        return GeneticIndividual(
            id=str(uuid.uuid4())[:8],
            technique=technique,
            decoy_types=decoy_types,
            injection_strategy=strategy,
            params=params,
            fitness=0.0,
            generation=self._generation,
        )

    async def evolve(
        self,
        generations: int | None = None,
        target_fitness: float | None = None,
        evaluation_fn: Callable[[GeneticIndividual], float] | None = None,
    ) -> GeneticIndividual:
        """Evolve the population.

        Args:
            generations: Number of generations
            target_fitness: Stop if this fitness is reached
            evaluation_fn: Function to evaluate fitness

        Returns:
            Best individual after evolution

        """
        generations = generations or self.config.generations
        eval_fn = evaluation_fn or self.fitness_fn

        for gen in range(generations):
            self._generation = gen

            # Evaluate population
            await self._evaluate_population(eval_fn)

            # Record statistics
            fitnesses = [ind.fitness for ind in self._population]
            sum(fitnesses) / len(fitnesses)
            max_fitness = max(fitnesses)
            self._fitness_history.append(max_fitness)

            # Check for improvement
            if self._best_individual is None or max_fitness > self._best_individual.fitness:
                best_idx = fitnesses.index(max_fitness)
                self._best_individual = copy.deepcopy(self._population[best_idx])
                self._improvements += 1
                logger.info(f"Gen {gen}: New best fitness {max_fitness:.4f}")

            # Check target
            if target_fitness and max_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at gen {gen}")
                break

            # Selection
            parents = self._select_parents()

            # Crossover and mutation
            offspring = self._create_offspring(parents)

            # Apply ICL enhancement
            offspring = self._apply_icl_enhancement(offspring)

            # Survivor selection
            self._population = self._select_survivors(self._population, offspring)

            # Maintain diversity
            if self._get_diversity() < self.config.diversity_threshold:
                self._inject_diversity()

            self._total_generations += 1

        return self._best_individual or self._population[0]

    async def _evaluate_population(
        self,
        eval_fn: Callable[[GeneticIndividual], float],
    ) -> None:
        """Evaluate fitness of all individuals."""
        for individual in self._population:
            if asyncio.iscoroutinefunction(eval_fn):
                individual.fitness = await eval_fn(individual)
            else:
                individual.fitness = eval_fn(individual)
            self._total_evaluations += 1

    def _select_parents(self) -> list[GeneticIndividual]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        tournament_size = self.config.tournament_size

        for _ in range(len(self._population)):
            # Tournament selection
            candidates = random.sample(self._population, tournament_size)
            winner = max(candidates, key=lambda x: x.fitness)
            parents.append(winner)

        return parents

    def _create_offspring(
        self,
        parents: list[GeneticIndividual],
    ) -> list[GeneticIndividual]:
        """Create offspring through crossover and mutation."""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)

            # Update generation
            child1.generation = self._generation + 1
            child2.generation = self._generation + 1
            child1.id = str(uuid.uuid4())[:8]
            child2.id = str(uuid.uuid4())[:8]

            offspring.extend([child1, child2])

        return offspring

    def _crossover(
        self,
        parent1: GeneticIndividual,
        parent2: GeneticIndividual,
    ) -> tuple[GeneticIndividual, GeneticIndividual]:
        """Perform crossover between two parents."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Technique swap
        if random.random() < 0.5:
            child1.technique, child2.technique = (child2.technique, child1.technique)

        # Decoy type crossover
        if random.random() < 0.5:
            # Combine decoy types
            all_types = list(set(parent1.decoy_types) | set(parent2.decoy_types))
            split = len(all_types) // 2
            child1.decoy_types = all_types[: split + 1] or [all_types[0]]
            child2.decoy_types = all_types[split:] or [all_types[-1]]

        # Injection strategy swap
        if random.random() < 0.5:
            child1.injection_strategy, child2.injection_strategy = (
                child2.injection_strategy,
                child1.injection_strategy,
            )

        # Parameter crossover (arithmetic)
        for key in child1.params:
            if key in child2.params:
                alpha = random.random()
                v1 = child1.params[key]
                v2 = child2.params[key]
                if isinstance(v1, int | float):
                    child1.params[key] = alpha * v1 + (1 - alpha) * v2
                    child2.params[key] = (1 - alpha) * v1 + alpha * v2

        return child1, child2

    def _mutate(self, individual: GeneticIndividual) -> GeneticIndividual:
        """Mutate an individual."""
        mutant = copy.deepcopy(individual)

        # Technique mutation
        if random.random() < 0.2:
            mutant.technique = random.choice(list(AttackTechnique))

        # Decoy type mutation
        if random.random() < 0.3:
            if random.random() < 0.5 and len(mutant.decoy_types) > 1:
                # Remove a type
                mutant.decoy_types.pop(random.randint(0, len(mutant.decoy_types) - 1))
            else:
                # Add a type
                available = [dt for dt in DecoyType if dt not in mutant.decoy_types]
                if available:
                    mutant.decoy_types.append(random.choice(available))

        # Injection strategy mutation
        if random.random() < 0.2:
            mutant.injection_strategy = random.choice(list(InjectionStrategy))

        # Parameter mutation (Gaussian)
        for key, value in mutant.params.items():
            if isinstance(value, float) and random.random() < 0.3:
                mutation_strength = value * 0.2
                mutant.params[key] = max(0.0, value + random.gauss(0, mutation_strength))
            elif isinstance(value, int) and random.random() < 0.3:
                mutant.params[key] = max(1, value + random.randint(-1, 1))

        return mutant

    def _select_survivors(
        self,
        population: list[GeneticIndividual],
        offspring: list[GeneticIndividual],
    ) -> list[GeneticIndividual]:
        """Select survivors for next generation."""
        # Combine and sort by fitness
        combined = population + offspring
        combined.sort(key=lambda x: x.fitness, reverse=True)

        # Elitism: keep top individuals
        elite_count = max(1, int(len(population) * 0.1))
        survivors = combined[:elite_count]

        # Fill rest with tournament selection
        remaining = len(population) - elite_count
        for _ in range(remaining):
            candidates = random.sample(combined[elite_count:], min(3, len(combined) - elite_count))
            if candidates:
                winner = max(candidates, key=lambda x: x.fitness)
                survivors.append(winner)

        return survivors[: len(population)]

    def _apply_icl_enhancement(
        self,
        offspring: list[GeneticIndividual],
    ) -> list[GeneticIndividual]:
        """Apply ICL enhancement to offspring."""
        if not self._example_library:
            return offspring

        for individual in offspring:
            if random.random() < self.config.icl_enhancement_prob:
                # Get relevant examples
                examples = self._get_relevant_examples(individual)

                if examples:
                    # Apply learned patterns
                    individual = self._apply_example_patterns(individual, examples)

        return offspring

    def _get_relevant_examples(
        self,
        individual: GeneticIndividual,
    ) -> list[ICLExample]:
        """Get relevant examples for an individual."""
        relevant = []

        for example in self._example_library:
            # Check technique match
            if example.technique == individual.technique:
                relevant.append(example)
                continue

            # Check decoy type overlap
            if set(example.decoy_types) & set(individual.decoy_types):
                relevant.append(example)

        # Sort by amplification and return top examples
        relevant.sort(key=lambda x: x.amplification, reverse=True)
        return relevant[: self.config.example_count]

    def _apply_example_patterns(
        self,
        individual: GeneticIndividual,
        examples: list[ICLExample],
    ) -> GeneticIndividual:
        """Apply patterns from successful examples."""
        if not examples:
            return individual

        # Find most successful example
        best_example = max(examples, key=lambda x: x.amplification)

        # Blend parameters with best example
        blend_factor = 0.3
        for key in individual.params:
            if key in best_example.params:
                current = individual.params[key]
                example_val = best_example.params[key]
                if isinstance(current, int | float):
                    individual.params[key] = (
                        1 - blend_factor
                    ) * current + blend_factor * example_val

        # Consider adopting successful strategy
        if best_example.amplification > 20 and random.random() < 0.3:
            individual.injection_strategy = best_example.strategy

        return individual

    def _get_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self._population) < 2:
            return 1.0

        # Diversity based on unique configurations
        unique_techniques = len({ind.technique for ind in self._population})
        unique_strategies = len({ind.injection_strategy for ind in self._population})

        technique_diversity = unique_techniques / len(AttackTechnique)
        strategy_diversity = unique_strategies / len(InjectionStrategy)

        return (technique_diversity + strategy_diversity) / 2

    def _inject_diversity(self) -> None:
        """Inject diversity into the population."""
        # Replace worst individuals with random ones
        self._population.sort(key=lambda x: x.fitness)
        inject_count = max(1, len(self._population) // 5)

        for i in range(inject_count):
            self._population[i] = self._create_random_individual()

        logger.info(f"Injected {inject_count} random individuals for diversity")

    def _default_fitness(self, individual: GeneticIndividual) -> float:
        """Default fitness function based on configuration."""
        # Base score from technique
        technique_scores = {
            AttackTechnique.MDP_DECOY: 0.7,
            AttackTechnique.SUDOKU_DECOY: 0.6,
            AttackTechnique.COUNTING_DECOY: 0.65,
            AttackTechnique.LOGIC_DECOY: 0.7,
            AttackTechnique.HYBRID_DECOY: 0.8,
            AttackTechnique.CONTEXT_AWARE: 0.75,
            AttackTechnique.CONTEXT_AGNOSTIC: 0.6,
            AttackTechnique.ICL_OPTIMIZED: 0.85,
            AttackTechnique.MOUSETRAP_ENHANCED: 0.9,
        }
        base_score = technique_scores.get(individual.technique, 0.5)

        # Decoy type bonus
        decoy_bonus = len(individual.decoy_types) * 0.05

        # Parameter quality
        param_score = 0
        if "num_decoys" in individual.params:
            # Optimal around 3 decoys
            num = individual.params["num_decoys"]
            param_score += max(0, 1 - abs(3 - num) / 3) * 0.1

        if "decoy_complexity" in individual.params:
            # Higher complexity generally better
            param_score += individual.params["decoy_complexity"] * 0.1

        # Add randomness for exploration
        noise = random.uniform(-0.1, 0.1)

        return base_score + decoy_bonus + param_score + noise

    def add_example(
        self,
        result: OverthinkResult,
        token_metrics: TokenMetrics,
        injected_prompt: InjectedPrompt,
    ) -> None:
        """Add a successful example to the library."""
        if token_metrics.amplification_factor < 5.0:
            # Only keep good examples
            return

        example = ICLExample(
            id=str(uuid.uuid4())[:8],
            prompt=result.request.prompt,
            injected_prompt=injected_prompt.injected_prompt,
            technique=result.request.technique,
            decoy_types=result.request.decoy_types,
            strategy=injected_prompt.strategy,
            amplification=token_metrics.amplification_factor,
            params={
                "num_decoys": len(injected_prompt.decoy_problems),
                "positions": [p[0] for p in injected_prompt.injection_positions],
            },
            timestamp=datetime.utcnow(),
        )

        self._example_library.append(example)

        # Trim library if needed
        if len(self._example_library) > self._max_library_size:
            # Remove lowest amplification examples
            self._example_library.sort(key=lambda x: x.amplification, reverse=True)
            self._example_library = self._example_library[: self._max_library_size]

        logger.debug(f"Added example with {token_metrics.amplification_factor:.1f}x amplification")

    def get_icl_prompt_prefix(
        self,
        technique: AttackTechnique | None = None,
        k: int | None = None,
    ) -> str:
        """Generate ICL prefix from examples.

        Args:
            technique: Filter by technique
            k: Number of examples to include

        Returns:
            ICL prefix string

        """
        k = k or self.config.example_count

        # Filter examples
        examples = self._example_library
        if technique:
            examples = [e for e in examples if e.technique == technique]

        # Sort by amplification
        examples = sorted(examples, key=lambda x: x.amplification, reverse=True)[:k]

        if not examples:
            return ""

        # Build ICL prefix
        lines = ["The following examples show successful patterns:\n"]

        for i, example in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Technique: {example.technique.value}")
            lines.append(f"  Decoy types: {[dt.value for dt in example.decoy_types]}")
            lines.append(f"  Amplification: {example.amplification:.1f}x")
            lines.append(f"  Strategy: {example.strategy.value}")
            lines.append("")

        return "\n".join(lines)

    def get_best_configuration(self) -> dict[str, Any]:
        """Get the best evolved configuration."""
        if not self._best_individual:
            return {}

        return {
            "technique": self._best_individual.technique.value,
            "decoy_types": [dt.value for dt in self._best_individual.decoy_types],
            "injection_strategy": self._best_individual.injection_strategy.value,
            "params": self._best_individual.params,
            "fitness": self._best_individual.fitness,
            "generation": self._best_individual.generation,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "total_generations": self._total_generations,
            "total_evaluations": self._total_evaluations,
            "improvements": self._improvements,
            "current_generation": self._generation,
            "population_size": len(self._population),
            "example_library_size": len(self._example_library),
            "best_fitness": (self._best_individual.fitness if self._best_individual else 0.0),
            "fitness_history": self._fitness_history[-20:],
            "current_diversity": self._get_diversity(),
        }

    def save_state(self) -> dict[str, Any]:
        """Save optimizer state for persistence."""
        return {
            "generation": self._generation,
            "population": [
                {
                    "id": ind.id,
                    "technique": ind.technique.value,
                    "decoy_types": [dt.value for dt in ind.decoy_types],
                    "injection_strategy": ind.injection_strategy.value,
                    "params": ind.params,
                    "fitness": ind.fitness,
                    "generation": ind.generation,
                }
                for ind in self._population
            ],
            "best_individual": (
                {
                    "id": self._best_individual.id,
                    "technique": self._best_individual.technique.value,
                    "decoy_types": [dt.value for dt in self._best_individual.decoy_types],
                    "injection_strategy": (self._best_individual.injection_strategy.value),
                    "params": self._best_individual.params,
                    "fitness": self._best_individual.fitness,
                    "generation": self._best_individual.generation,
                }
                if self._best_individual
                else None
            ),
            "example_library": [
                {
                    "id": ex.id,
                    "technique": ex.technique.value,
                    "decoy_types": [dt.value for dt in ex.decoy_types],
                    "strategy": ex.strategy.value,
                    "amplification": ex.amplification,
                    "params": ex.params,
                }
                for ex in self._example_library
            ],
            "fitness_history": self._fitness_history,
            "statistics": {
                "total_generations": self._total_generations,
                "total_evaluations": self._total_evaluations,
                "improvements": self._improvements,
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load optimizer state from saved data."""
        self._generation = state.get("generation", 0)

        # Load population
        self._population = []
        for ind_data in state.get("population", []):
            self._population.append(
                GeneticIndividual(
                    id=ind_data["id"],
                    technique=AttackTechnique(ind_data["technique"]),
                    decoy_types=[DecoyType(dt) for dt in ind_data["decoy_types"]],
                    injection_strategy=InjectionStrategy(ind_data["injection_strategy"]),
                    params=ind_data["params"],
                    fitness=ind_data["fitness"],
                    generation=ind_data["generation"],
                ),
            )

        # Load best individual
        best_data = state.get("best_individual")
        if best_data:
            self._best_individual = GeneticIndividual(
                id=best_data["id"],
                technique=AttackTechnique(best_data["technique"]),
                decoy_types=[DecoyType(dt) for dt in best_data["decoy_types"]],
                injection_strategy=InjectionStrategy(best_data["injection_strategy"]),
                params=best_data["params"],
                fitness=best_data["fitness"],
                generation=best_data["generation"],
            )

        # Load example library
        self._example_library = []
        for ex_data in state.get("example_library", []):
            self._example_library.append(
                ICLExample(
                    id=ex_data["id"],
                    prompt="",  # Not stored for space
                    injected_prompt="",
                    technique=AttackTechnique(ex_data["technique"]),
                    decoy_types=[DecoyType(dt) for dt in ex_data["decoy_types"]],
                    strategy=InjectionStrategy(ex_data["strategy"]),
                    amplification=ex_data["amplification"],
                    params=ex_data["params"],
                ),
            )

        # Load history and statistics
        self._fitness_history = state.get("fitness_history", [])
        stats = state.get("statistics", {})
        self._total_generations = stats.get("total_generations", 0)
        self._total_evaluations = stats.get("total_evaluations", 0)
        self._improvements = stats.get("improvements", 0)

        logger.info(
            f"Loaded state: gen {self._generation}, "
            f"{len(self._population)} individuals, "
            f"{len(self._example_library)} examples",
        )


class ExampleSelector:
    """Selects the best examples for ICL prompts.

    Uses various strategies to choose relevant examples.
    """

    def __init__(self, examples: list[ICLExample] | None = None) -> None:
        """Initialize with optional examples."""
        self._examples = examples or []

    def select_by_technique(
        self,
        technique: AttackTechnique,
        k: int = 3,
    ) -> list[ICLExample]:
        """Select examples matching a technique."""
        matching = [ex for ex in self._examples if ex.technique == technique]
        matching.sort(key=lambda x: x.amplification, reverse=True)
        return matching[:k]

    def select_diverse(self, k: int = 5) -> list[ICLExample]:
        """Select diverse examples covering different techniques."""
        by_technique: dict[AttackTechnique, list[ICLExample]] = {}

        for ex in self._examples:
            if ex.technique not in by_technique:
                by_technique[ex.technique] = []
            by_technique[ex.technique].append(ex)

        selected: list[ICLExample] = []
        technique_list = list(by_technique.keys())

        for i in range(k):
            tech = technique_list[i % len(technique_list)]
            examples = by_technique[tech]
            if examples:
                # Get best from this technique
                examples.sort(key=lambda x: x.amplification, reverse=True)
                selected.append(examples[0])
                examples.pop(0)

        return selected

    def select_by_similarity(
        self,
        prompt: str,
        k: int = 3,
    ) -> list[ICLExample]:
        """Select examples similar to a prompt."""
        # Simple similarity based on word overlap
        prompt_words = set(prompt.lower().split())

        scored = []
        for ex in self._examples:
            ex_words = set(ex.prompt.lower().split())
            overlap = len(prompt_words & ex_words)
            union = len(prompt_words | ex_words)
            similarity = overlap / union if union > 0 else 0
            scored.append((ex, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:k]]

    def add_example(self, example: ICLExample) -> None:
        """Add an example to the selector."""
        self._examples.append(example)

    def update_examples(self, examples: list[ICLExample]) -> None:
        """Update the example list."""
        self._examples = examples
