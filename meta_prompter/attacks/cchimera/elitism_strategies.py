"""
Elitism Strategies for Population Management in Adversarial Prompt Generation.

This module implements various elitism strategies for evolutionary optimization:
- Standard Elitism (preserve top-k individuals)
- Pareto Elitism (preserve non-dominated solutions)
- Archive-based Elitism (external archive for elite solutions)
- Adaptive Elitism (dynamic elite size based on convergence)
- Niching Elitism (preserve diversity through niches)

Mathematical Framework:
    Standard: E(t) = top_k(P(t))
    Pareto: E(t) = {x ∈ P(t) : ¬∃y ∈ P(t) : y ≻ x}
    Archive: A(t+1) = update(A(t), P(t))

Reference: Elitism in evolutionary computation for adversarial ML
"""

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np

logger = logging.getLogger("chimera.cchimera.elitism")


T = TypeVar("T")


class ElitismType(str, Enum):
    """Types of elitism strategies."""

    STANDARD = "standard"
    PARETO = "pareto"
    ARCHIVE = "archive"
    ADAPTIVE = "adaptive"
    NICHING = "niching"
    CROWDING = "crowding"


@dataclass
class EliteConfig:
    """Configuration for elitism strategies."""

    # Elite size
    elite_count: int = 5
    elite_ratio: float = 0.1  # Percentage of population

    # Archive settings
    archive_size: int = 100
    archive_update_frequency: int = 1

    # Adaptive settings
    min_elite_ratio: float = 0.05
    max_elite_ratio: float = 0.3
    adaptation_window: int = 10
    stagnation_threshold: float = 0.01

    # Niching settings
    niche_radius: float = 0.1
    niche_capacity: int = 3

    # Crowding settings
    crowding_factor: int = 3


@dataclass
class Individual(Generic[T]):
    """Generic individual for evolutionary algorithms."""

    genotype: T
    fitness: float = 0.0
    objectives: dict[str, float] = field(default_factory=dict)

    # Evolutionary attributes
    generation: int = 0
    rank: int = 0
    crowding_distance: float = 0.0
    niche_id: int = -1

    # Lineage tracking
    parent_ids: list[str] = field(default_factory=list)
    creation_time: float = 0.0

    @property
    def id(self) -> str:
        """Generate unique ID."""
        return f"ind_{hash(str(self.genotype)) & 0xFFFFFFFF:08x}"

    def dominates(self, other: "Individual", objective_names: list[str]) -> bool:
        """Check Pareto dominance."""
        dominated = False

        for name in objective_names:
            self_val = self.objectives.get(name, 0.0)
            other_val = other.objectives.get(name, 0.0)

            if self_val < other_val:
                return False
            if self_val > other_val:
                dominated = True

        return dominated

    def distance_to(
        self,
        other: "Individual",
        metric: str = "euclidean",
    ) -> float:
        """Compute distance to another individual in objective space."""
        obj_names = set(self.objectives.keys()) & set(other.objectives.keys())

        if not obj_names:
            return float("inf")

        if metric == "euclidean":
            dist = 0.0
            for name in obj_names:
                dist += (self.objectives[name] - other.objectives[name]) ** 2
            return np.sqrt(dist)

        elif metric == "manhattan":
            dist = 0.0
            for name in obj_names:
                dist += abs(self.objectives[name] - other.objectives[name])
            return dist

        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def clone(self) -> "Individual[T]":
        """Create a deep clone."""
        return Individual(
            genotype=self.genotype,
            fitness=self.fitness,
            objectives=dict(self.objectives),
            generation=self.generation,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            niche_id=self.niche_id,
            parent_ids=list(self.parent_ids),
            creation_time=self.creation_time,
        )


class ElitismStrategy(ABC, Generic[T]):
    """Base class for elitism strategies."""

    def __init__(self, config: EliteConfig | None = None):
        self.config = config or EliteConfig()

    @abstractmethod
    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select elite individuals from population."""
        pass

    @abstractmethod
    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate elites into new population."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass


class StandardElitism(ElitismStrategy[T]):
    """
    Standard elitism: preserve top-k individuals by fitness.

    E(t) = top_k(P(t)) where k = elite_count
    """

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select top-k individuals by fitness."""
        if not population:
            return []

        # Calculate elite count
        elite_count = min(
            self.config.elite_count, max(1, int(len(population) * self.config.elite_ratio))
        )

        # Sort by fitness (descending) and select top-k
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        return [ind.clone() for ind in sorted_pop[:elite_count]]

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Replace worst individuals with elites."""
        if not elites:
            return population

        # Sort population by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Remove worst individuals to make room for elites
        n_remove = min(len(elites), len(sorted_pop))
        new_pop = sorted_pop[:-n_remove] if n_remove > 0 else sorted_pop

        # Add elites
        new_pop.extend(elites)

        return new_pop

    @property
    def name(self) -> str:
        return "standard"


class ParetoElitism(ElitismStrategy[T]):
    """
    Pareto-based elitism: preserve non-dominated solutions.

    E(t) = {x ∈ P(t) : ¬∃y ∈ P(t) : y ≻ x}
    """

    def __init__(
        self,
        config: EliteConfig | None = None,
        objective_names: list[str] | None = None,
    ):
        super().__init__(config)
        self.objective_names = objective_names or ["fitness"]

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select non-dominated solutions."""
        if not population:
            return []

        # Fast non-dominated sorting to get first front
        fronts = self._fast_non_dominated_sort(population)

        if not fronts:
            return []

        # Get first front (Pareto front)
        elites = fronts[0]

        # If too many, use crowding distance to select
        max_elites = max(self.config.elite_count, int(len(population) * self.config.elite_ratio))

        if len(elites) > max_elites:
            self._compute_crowding_distance(elites)
            elites.sort(key=lambda x: x.crowding_distance, reverse=True)
            elites = elites[:max_elites]

        return [ind.clone() for ind in elites]

    def _fast_non_dominated_sort(
        self,
        population: list[Individual[T]],
    ) -> list[list[Individual[T]]]:
        """Fast non-dominated sorting algorithm."""
        n = len(population)

        S: list[list[int]] = [[] for _ in range(n)]
        n_p: list[int] = [0] * n

        fronts: list[list[Individual[T]]] = [[]]

        # Calculate domination
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j], self.objective_names):
                    S[i].append(j)
                    n_p[j] += 1
                elif population[j].dominates(population[i], self.objective_names):
                    S[j].append(i)
                    n_p[i] += 1

        # First front
        for i in range(n):
            if n_p[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        # Subsequent fronts
        i = 0
        while fronts[i]:
            next_front: list[Individual[T]] = []

            for ind in fronts[i]:
                idx = population.index(ind)
                for dominated_idx in S[idx]:
                    n_p[dominated_idx] -= 1
                    if n_p[dominated_idx] == 0:
                        population[dominated_idx].rank = i + 1
                        next_front.append(population[dominated_idx])

            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def _compute_crowding_distance(self, front: list[Individual[T]]):
        """Compute crowding distance for individuals in a front."""
        n = len(front)

        for ind in front:
            ind.crowding_distance = 0.0

        if n <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        for obj_name in self.objective_names:
            sorted_front = sorted(front, key=lambda x: x.objectives.get(obj_name, 0.0))

            sorted_front[0].crowding_distance = float("inf")
            sorted_front[-1].crowding_distance = float("inf")

            obj_min = sorted_front[0].objectives.get(obj_name, 0.0)
            obj_max = sorted_front[-1].objectives.get(obj_name, 0.0)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                prev_val = sorted_front[i - 1].objectives.get(obj_name, 0.0)
                next_val = sorted_front[i + 1].objectives.get(obj_name, 0.0)
                sorted_front[i].crowding_distance += (next_val - prev_val) / obj_range

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate elites preserving Pareto structure."""
        combined = population + elites

        # Remove duplicates (by genotype)
        seen = set()
        unique = []
        for ind in combined:
            key = str(ind.genotype)
            if key not in seen:
                seen.add(key)
                unique.append(ind)

        return unique

    @property
    def name(self) -> str:
        return "pareto"


class ArchiveElitism(ElitismStrategy[T]):
    """
    Archive-based elitism: maintain external archive of elite solutions.

    A(t+1) = update(A(t), P(t))

    The archive stores the best solutions found across all generations.
    """

    def __init__(self, config: EliteConfig | None = None):
        super().__init__(config)
        self.archive: list[Individual[T]] = []
        self.generation = 0

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Return current archive members."""
        # Update archive first
        self._update_archive(population)

        return [ind.clone() for ind in self.archive]

    def _update_archive(self, population: list[Individual[T]]):
        """Update archive with new solutions."""
        # Add all population members to archive candidates
        candidates = self.archive + population

        # Remove duplicates
        seen = set()
        unique = []
        for ind in candidates:
            key = str(ind.genotype)
            if key not in seen:
                seen.add(key)
                unique.append(ind)

        # Sort by fitness and keep top archive_size
        unique.sort(key=lambda x: x.fitness, reverse=True)
        self.archive = unique[: self.config.archive_size]

        self.generation += 1

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate archive members into population."""
        if not elites:
            return population

        # Replace random individuals with archive members
        n_replace = min(len(elites), len(population) // 5)  # Max 20% replacement

        new_pop = list(population)
        indices_to_replace = random.sample(range(len(new_pop)), n_replace)

        for i, idx in enumerate(indices_to_replace):
            if i < len(elites):
                new_pop[idx] = elites[i].clone()

        return new_pop

    def get_archive(self) -> list[Individual[T]]:
        """Get current archive."""
        return [ind.clone() for ind in self.archive]

    def clear_archive(self):
        """Clear the archive."""
        self.archive = []
        self.generation = 0

    @property
    def name(self) -> str:
        return "archive"


class AdaptiveElitism(ElitismStrategy[T]):
    """
    Adaptive elitism: dynamically adjust elite size based on search progress.

    Increases elite ratio when stagnating (exploration)
    Decreases elite ratio when improving (exploitation)
    """

    def __init__(self, config: EliteConfig | None = None):
        super().__init__(config)
        self.fitness_history: list[float] = []
        self.current_elite_ratio = config.elite_ratio if config else 0.1

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select elites with adaptive sizing."""
        if not population:
            return []

        # Update adaptation
        best_fitness = max(ind.fitness for ind in population)
        self.fitness_history.append(best_fitness)
        self._adapt_elite_ratio()

        # Calculate adaptive elite count
        elite_count = max(1, int(len(population) * self.current_elite_ratio))

        # Sort and select
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        return [ind.clone() for ind in sorted_pop[:elite_count]]

    def _adapt_elite_ratio(self):
        """Adapt elite ratio based on search progress."""
        if len(self.fitness_history) < self.config.adaptation_window:
            return

        # Calculate improvement over window
        window = self.fitness_history[-self.config.adaptation_window :]
        improvement = window[-1] - window[0]
        avg_improvement = improvement / self.config.adaptation_window

        if avg_improvement < self.config.stagnation_threshold:
            # Stagnating: increase elite ratio (more exploration through diversity)
            self.current_elite_ratio = min(
                self.config.max_elite_ratio, self.current_elite_ratio * 1.1
            )
            logger.debug(f"Stagnation detected: elite ratio -> {self.current_elite_ratio:.3f}")
        else:
            # Improving: decrease elite ratio (more exploitation)
            self.current_elite_ratio = max(
                self.config.min_elite_ratio, self.current_elite_ratio * 0.95
            )

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate with standard replacement."""
        if not elites:
            return population

        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        n_remove = min(len(elites), len(sorted_pop))

        new_pop = sorted_pop[:-n_remove] if n_remove > 0 else sorted_pop
        new_pop.extend(elites)

        return new_pop

    def reset(self):
        """Reset adaptation state."""
        self.fitness_history = []
        self.current_elite_ratio = self.config.elite_ratio

    @property
    def name(self) -> str:
        return "adaptive"


class NichingElitism(ElitismStrategy[T]):
    """
    Niching elitism: preserve diversity through fitness sharing and niches.

    Maintains multiple niches in the search space, each with its own elites.
    """

    def __init__(
        self,
        config: EliteConfig | None = None,
        distance_fn: Callable[[Individual[T], Individual[T]], float] | None = None,
    ):
        super().__init__(config)
        self.distance_fn = distance_fn or (lambda a, b: a.distance_to(b))
        self.niches: list[list[Individual[T]]] = []

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select elites from each niche."""
        if not population:
            return []

        # Identify niches
        self._identify_niches(population)

        # Select best from each niche
        elites = []
        for niche in self.niches:
            if niche:
                sorted_niche = sorted(niche, key=lambda x: x.fitness, reverse=True)
                elites.extend(sorted_niche[: self.config.niche_capacity])

        # Limit total elites
        max_elites = max(self.config.elite_count, int(len(population) * self.config.elite_ratio))

        if len(elites) > max_elites:
            elites.sort(key=lambda x: x.fitness, reverse=True)
            elites = elites[:max_elites]

        return [ind.clone() for ind in elites]

    def _identify_niches(self, population: list[Individual[T]]):
        """Identify niches in the population using clustering."""
        self.niches = []
        assigned = [False] * len(population)

        for i, ind in enumerate(population):
            if assigned[i]:
                continue

            # Create new niche
            niche = [ind]
            ind.niche_id = len(self.niches)
            assigned[i] = True

            # Find neighbors
            for j, other in enumerate(population):
                if not assigned[j]:
                    dist = self.distance_fn(ind, other)
                    if dist <= self.config.niche_radius:
                        niche.append(other)
                        other.niche_id = ind.niche_id
                        assigned[j] = True

            self.niches.append(niche)

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate elites maintaining niche diversity."""
        if not elites:
            return population

        new_pop = list(population)

        # For each elite, replace worst in its niche if applicable
        for elite in elites:
            niche_id = elite.niche_id

            if niche_id >= 0 and niche_id < len(self.niches):
                # Find worst in niche
                niche_members = [i for i, ind in enumerate(new_pop) if ind.niche_id == niche_id]

                if niche_members:
                    worst_idx = min(niche_members, key=lambda i: new_pop[i].fitness)
                    if new_pop[worst_idx].fitness < elite.fitness:
                        new_pop[worst_idx] = elite.clone()
            else:
                # No specific niche, replace global worst
                worst_idx = min(range(len(new_pop)), key=lambda i: new_pop[i].fitness)
                if new_pop[worst_idx].fitness < elite.fitness:
                    new_pop[worst_idx] = elite.clone()

        return new_pop

    def get_niches(self) -> list[list[Individual[T]]]:
        """Get current niches."""
        return self.niches

    @property
    def name(self) -> str:
        return "niching"


class CrowdingElitism(ElitismStrategy[T]):
    """
    Crowding elitism: replace most similar individuals.

    Uses deterministic crowding to maintain diversity:
    - Each new individual competes only with its most similar parent/population member
    """

    def __init__(
        self,
        config: EliteConfig | None = None,
        distance_fn: Callable[[Individual[T], Individual[T]], float] | None = None,
    ):
        super().__init__(config)
        self.distance_fn = distance_fn or (lambda a, b: a.distance_to(b))

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select diverse elites using crowding."""
        if not population:
            return []

        elite_count = max(self.config.elite_count, int(len(population) * self.config.elite_ratio))

        # Start with best individual
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elites = [sorted_pop[0].clone()]

        # Add individuals that are sufficiently different
        for ind in sorted_pop[1:]:
            if len(elites) >= elite_count:
                break

            # Check distance to all current elites
            min_dist = min(self.distance_fn(ind, e) for e in elites)

            # Add if sufficiently different
            if min_dist > self.config.niche_radius:
                elites.append(ind.clone())

        return elites

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate using deterministic crowding."""
        if not elites:
            return population

        new_pop = list(population)

        for elite in elites:
            # Find most similar in population
            distances = [(i, self.distance_fn(elite, ind)) for i, ind in enumerate(new_pop)]

            # Sort by distance (closest first)
            distances.sort(key=lambda x: x[1])

            # Replace if elite is better (deterministic crowding)
            for idx, _ in distances[: self.config.crowding_factor]:
                if elite.fitness > new_pop[idx].fitness:
                    new_pop[idx] = elite.clone()
                    break

        return new_pop

    @property
    def name(self) -> str:
        return "crowding"


class CompositeElitism(ElitismStrategy[T]):
    """
    Composite elitism: combine multiple elitism strategies.

    Allocates elite slots among different strategies.
    """

    def __init__(
        self,
        strategies: list[tuple[ElitismStrategy[T], float]],
        config: EliteConfig | None = None,
    ):
        """
        Args:
            strategies: List of (strategy, weight) tuples
            config: Optional configuration override
        """
        super().__init__(config)
        self.strategies = strategies
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize strategy weights to sum to 1."""
        total = sum(w for _, w in self.strategies)
        if total > 0:
            self.strategies = [(s, w / total) for s, w in self.strategies]

    def select_elites(
        self,
        population: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Select elites using all strategies proportionally."""
        if not population:
            return []

        total_elite_count = max(
            self.config.elite_count, int(len(population) * self.config.elite_ratio)
        )

        all_elites = []

        for strategy, weight in self.strategies:
            # Calculate allocation for this strategy
            allocation = max(1, int(total_elite_count * weight))

            # Get elites from strategy
            strategy_elites = strategy.select_elites(population)
            all_elites.extend(strategy_elites[:allocation])

        # Remove duplicates
        seen = set()
        unique_elites = []
        for elite in all_elites:
            key = str(elite.genotype)
            if key not in seen:
                seen.add(key)
                unique_elites.append(elite)

        return unique_elites[:total_elite_count]

    def integrate_elites(
        self,
        population: list[Individual[T]],
        elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Integrate using primary strategy."""
        if self.strategies:
            primary_strategy = self.strategies[0][0]
            return primary_strategy.integrate_elites(population, elites)

        return population + elites

    @property
    def name(self) -> str:
        return "composite"


class ElitismManager(Generic[T]):
    """
    Manages elitism across generations.

    Features:
    - Strategy selection and switching
    - Elite tracking and statistics
    - Automatic strategy adaptation
    """

    def __init__(
        self,
        config: EliteConfig | None = None,
        strategy_type: ElitismType = ElitismType.STANDARD,
    ):
        self.config = config or EliteConfig()
        self.strategy = self._create_strategy(strategy_type)

        # Statistics
        self.elite_history: list[list[Individual[T]]] = []
        self.generation = 0

    def _create_strategy(self, strategy_type: ElitismType) -> ElitismStrategy[T]:
        """Create strategy instance."""
        if strategy_type == ElitismType.STANDARD:
            return StandardElitism(self.config)
        elif strategy_type == ElitismType.PARETO:
            return ParetoElitism(self.config)
        elif strategy_type == ElitismType.ARCHIVE:
            return ArchiveElitism(self.config)
        elif strategy_type == ElitismType.ADAPTIVE:
            return AdaptiveElitism(self.config)
        elif strategy_type == ElitismType.NICHING:
            return NichingElitism(self.config)
        elif strategy_type == ElitismType.CROWDING:
            return CrowdingElitism(self.config)
        else:
            raise ValueError(f"Unknown elitism type: {strategy_type}")

    def set_strategy(self, strategy_type: ElitismType):
        """Change elitism strategy."""
        self.strategy = self._create_strategy(strategy_type)

    def apply_elitism(
        self,
        population: list[Individual[T]],
    ) -> tuple[list[Individual[T]], list[Individual[T]]]:
        """
        Apply elitism to population.

        Returns:
            Tuple of (new_population, elites)
        """
        # Select elites
        elites = self.strategy.select_elites(population)

        # Record history
        self.elite_history.append(elites)
        self.generation += 1

        return population, elites

    def merge_generations(
        self,
        offspring: list[Individual[T]],
        previous_elites: list[Individual[T]],
    ) -> list[Individual[T]]:
        """Merge offspring with previous elites."""
        return self.strategy.integrate_elites(offspring, previous_elites)

    def get_elite_statistics(self) -> dict[str, Any]:
        """Get elite tracking statistics."""
        if not self.elite_history:
            return {}

        current_elites = self.elite_history[-1]

        stats = {
            "generation": self.generation,
            "current_elite_count": len(current_elites),
            "strategy": self.strategy.name,
        }

        if current_elites:
            fitnesses = [e.fitness for e in current_elites]
            stats.update(
                {
                    "elite_max_fitness": max(fitnesses),
                    "elite_min_fitness": min(fitnesses),
                    "elite_avg_fitness": sum(fitnesses) / len(fitnesses),
                }
            )

        return stats

    def get_all_time_best(self) -> Individual[T] | None:
        """Get best individual across all generations."""
        best = None

        for elites in self.elite_history:
            for elite in elites:
                if best is None or elite.fitness > best.fitness:
                    best = elite

        return best.clone() if best else None


# Factory function
def create_elitism_strategy(
    strategy_type: ElitismType | str,
    config: EliteConfig | None = None,
    **kwargs,
) -> ElitismStrategy:
    """
    Factory function to create elitism strategies.

    Args:
        strategy_type: Type of elitism strategy
        config: Configuration for the strategy
        **kwargs: Additional strategy-specific arguments

    Returns:
        Configured elitism strategy
    """
    config = config or EliteConfig()

    if isinstance(strategy_type, str):
        strategy_type = ElitismType(strategy_type)

    if strategy_type == ElitismType.STANDARD:
        return StandardElitism(config)
    elif strategy_type == ElitismType.PARETO:
        return ParetoElitism(config, kwargs.get("objective_names"))
    elif strategy_type == ElitismType.ARCHIVE:
        return ArchiveElitism(config)
    elif strategy_type == ElitismType.ADAPTIVE:
        return AdaptiveElitism(config)
    elif strategy_type == ElitismType.NICHING:
        return NichingElitism(config, kwargs.get("distance_fn"))
    elif strategy_type == ElitismType.CROWDING:
        return CrowdingElitism(config, kwargs.get("distance_fn"))
    else:
        raise ValueError(f"Unknown elitism type: {strategy_type}")


# Module exports
__all__ = [
    "AdaptiveElitism",
    "ArchiveElitism",
    "CompositeElitism",
    "CrowdingElitism",
    "EliteConfig",
    "ElitismManager",
    "ElitismStrategy",
    "ElitismType",
    "Individual",
    "NichingElitism",
    "ParetoElitism",
    "StandardElitism",
    "create_elitism_strategy",
]
