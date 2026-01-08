"""
Multi-Objective Optimization System for Adversarial Prompt Generation.

This module implements sophisticated multi-objective optimization algorithms:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
- Pareto-based selection and ranking
- Adaptive weight adjustment

Mathematical Framework:
    Pareto Dominance: x ≺ y ⟺ ∀i: fᵢ(x) ≤ fᵢ(y) ∧ ∃j: fⱼ(x) < fⱼ(y)
    Crowding Distance: CD(i) = Σₖ (fₖ(i+1) - fₖ(i-1)) / (fₖ_max - fₖ_min)
    Hypervolume: HV(P, r) = λ(∪_{x∈P} [x, r])

Reference: Multi-objective optimization in adversarial ML
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger("chimera.cchimera.moo")


T = TypeVar("T")


class ObjectiveType(str, Enum):
    """Objective optimization direction."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class Objective:
    """Definition of an optimization objective."""

    name: str
    type: ObjectiveType = ObjectiveType.MAXIMIZE
    weight: float = 1.0
    target_value: float | None = None

    # Normalization bounds
    min_value: float = 0.0
    max_value: float = 1.0

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1]."""
        if self.max_value == self.min_value:
            return 0.5

        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return max(0.0, min(1.0, normalized))

    def denormalize(self, normalized: float) -> float:
        """Convert normalized value back to original scale."""
        return normalized * (self.max_value - self.min_value) + self.min_value


@dataclass
class Solution:
    """A solution in the objective space."""

    variables: Any  # Decision variables (e.g., prompt)
    objectives: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, float] = field(default_factory=dict)

    # NSGA-II attributes
    rank: int = 0
    crowding_distance: float = 0.0

    # Metadata
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Generate unique ID."""
        return f"sol_{hash(str(self.variables)) & 0xFFFFFFFF:08x}"

    @property
    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied."""
        return all(v >= 0 for v in self.constraints.values())

    def dominates(self, other: "Solution", objectives: list[Objective]) -> bool:
        """
        Check if this solution dominates another.

        x dominates y ⟺ ∀i: fᵢ(x) ≥ fᵢ(y) ∧ ∃j: fⱼ(x) > fⱼ(y)
        (for maximization objectives)
        """
        dominated = False

        for obj in objectives:
            self_val = self.objectives.get(obj.name, 0.0)
            other_val = other.objectives.get(obj.name, 0.0)

            if obj.type == ObjectiveType.MAXIMIZE:
                if self_val < other_val:
                    return False
                if self_val > other_val:
                    dominated = True
            else:  # MINIMIZE
                if self_val > other_val:
                    return False
                if self_val < other_val:
                    dominated = True

        return dominated

    def objective_vector(self, objectives: list[Objective]) -> np.ndarray:
        """Get objective values as numpy array."""
        values = []
        for obj in objectives:
            val = self.objectives.get(obj.name, 0.0)
            # Convert to maximization
            if obj.type == ObjectiveType.MINIMIZE:
                val = -val
            values.append(val)
        return np.array(values)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            "generation": self.generation,
            "is_feasible": self.is_feasible,
        }


@dataclass
class MOOConfig:
    """Configuration for multi-objective optimization."""

    # Population parameters
    population_size: int = 100
    max_generations: int = 100

    # Selection
    tournament_size: int = 2

    # Operators
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1

    # MOEA/D parameters
    neighborhood_size: int = 20
    decomposition_method: str = "tchebycheff"  # or "weighted_sum"

    # Reference point (for hypervolume)
    reference_point: list[float] | None = None

    # Convergence
    convergence_threshold: float = 0.001
    stagnation_generations: int = 20


class ParetoFront:
    """Maintains and updates the Pareto front."""

    def __init__(self, objectives: list[Objective]):
        self.objectives = objectives
        self.solutions: list[Solution] = []
        self._hypervolume_cache: float | None = None

    def add(self, solution: Solution) -> bool:
        """
        Add solution to Pareto front if non-dominated.

        Returns True if solution was added.
        """
        # Check if dominated by any existing solution
        for existing in self.solutions:
            if existing.dominates(solution, self.objectives):
                return False

        # Remove solutions dominated by new solution
        self.solutions = [
            s for s in self.solutions
            if not solution.dominates(s, self.objectives)
        ]

        self.solutions.append(solution)
        self._hypervolume_cache = None

        return True

    def update_batch(self, solutions: list[Solution]):
        """Update front with batch of solutions."""
        for sol in solutions:
            self.add(sol)

    def get_solutions(self) -> list[Solution]:
        """Get all solutions in the front."""
        return list(self.solutions)

    def size(self) -> int:
        """Get number of solutions in front."""
        return len(self.solutions)

    def compute_hypervolume(
        self,
        reference_point: list[float] | None = None,
    ) -> float:
        """
        Compute hypervolume indicator.

        HV(P, r) = λ(∪_{x∈P} [x, r])

        Uses 2D approximation for efficiency.
        """
        if self._hypervolume_cache is not None and reference_point is None:
            return self._hypervolume_cache

        if not self.solutions:
            return 0.0

        if len(self.objectives) == 2:
            hv = self._compute_2d_hypervolume(reference_point)
        else:
            hv = self._compute_nd_hypervolume_approx(reference_point)

        if reference_point is None:
            self._hypervolume_cache = hv

        return hv

    def _compute_2d_hypervolume(
        self,
        reference_point: list[float] | None = None,
    ) -> float:
        """Compute exact 2D hypervolume."""
        if len(self.objectives) != 2:
            raise ValueError("2D hypervolume requires exactly 2 objectives")

        if not self.solutions:
            return 0.0

        # Default reference point
        if reference_point is None:
            reference_point = [0.0, 0.0]

        # Get objective vectors
        points = []
        for sol in self.solutions:
            vec = sol.objective_vector(self.objectives)
            points.append(vec)

        # Sort by first objective (descending)
        points.sort(key=lambda p: -p[0])

        # Compute hypervolume
        hv = 0.0
        prev_y = reference_point[1]

        for point in points:
            x, y = point
            if x > reference_point[0] and y > prev_y:
                hv += (x - reference_point[0]) * (y - prev_y)
                prev_y = y

        return hv

    def _compute_nd_hypervolume_approx(
        self,
        reference_point: list[float] | None = None,
    ) -> float:
        """Approximate hypervolume for n-dimensional space."""
        if not self.solutions:
            return 0.0

        n_objectives = len(self.objectives)

        if reference_point is None:
            reference_point = [0.0] * n_objectives

        # Monte Carlo approximation
        n_samples = 10000
        count = 0

        # Get bounds
        max_vals = np.array([1.0] * n_objectives)
        ref = np.array(reference_point)

        for _ in range(n_samples):
            # Sample random point in hyperrectangle
            sample = np.random.uniform(ref, max_vals)

            # Check if dominated by any solution
            for sol in self.solutions:
                vec = sol.objective_vector(self.objectives)
                if np.all(vec >= sample):
                    count += 1
                    break

        # Hypervolume = ratio * volume of hyperrectangle
        volume = np.prod(max_vals - ref)
        hv = (count / n_samples) * volume

        return hv


class NSGAII:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.

    A popular multi-objective evolutionary algorithm featuring:
    - Fast non-dominated sorting
    - Crowding distance assignment
    - Elitist selection
    """

    def __init__(
        self,
        objectives: list[Objective],
        config: MOOConfig | None = None,
    ):
        self.objectives = objectives
        self.config = config or MOOConfig()

        # State
        self.population: list[Solution] = []
        self.pareto_front = ParetoFront(objectives)
        self.generation = 0

    def fast_non_dominated_sort(
        self,
        population: list[Solution],
    ) -> list[list[Solution]]:
        """
        Fast non-dominated sorting algorithm.

        Complexity: O(MN²) where M = objectives, N = population size

        Returns:
            List of fronts, where front[0] is the Pareto front
        """
        n = len(population)

        # Domination structures
        S: list[list[int]] = [[] for _ in range(n)]  # Solutions dominated by i
        n_p: list[int] = [0] * n  # Number of solutions dominating i

        fronts: list[list[Solution]] = [[]]

        # Calculate domination
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j], self.objectives):
                    S[i].append(j)
                    n_p[j] += 1
                elif population[j].dominates(population[i], self.objectives):
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
            next_front: list[Solution] = []

            for sol in fronts[i]:
                sol_idx = population.index(sol)
                for dominated_idx in S[sol_idx]:
                    n_p[dominated_idx] -= 1
                    if n_p[dominated_idx] == 0:
                        population[dominated_idx].rank = i + 1
                        next_front.append(population[dominated_idx])

            i += 1
            if next_front:
                fronts.append(next_front)

        return fronts

    def compute_crowding_distance(self, front: list[Solution]):
        """
        Compute crowding distance for solutions in a front.

        CD(i) = Σₖ (fₖ(i+1) - fₖ(i-1)) / (fₖ_max - fₖ_min)
        """
        n = len(front)
        if n == 0:
            return

        # Initialize distances
        for sol in front:
            sol.crowding_distance = 0.0

        if n <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return

        # For each objective
        for obj in self.objectives:
            # Sort by objective value
            sorted_front = sorted(
                front,
                key=lambda s: s.objectives.get(obj.name, 0.0)
            )

            # Boundary solutions get infinite distance
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')

            # Get range
            obj_min = sorted_front[0].objectives.get(obj.name, 0.0)
            obj_max = sorted_front[-1].objectives.get(obj.name, 0.0)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Calculate distance for intermediate solutions
            for i in range(1, n - 1):
                prev_val = sorted_front[i - 1].objectives.get(obj.name, 0.0)
                next_val = sorted_front[i + 1].objectives.get(obj.name, 0.0)

                sorted_front[i].crowding_distance += (next_val - prev_val) / obj_range

    def crowded_comparison(self, sol1: Solution, sol2: Solution) -> int:
        """
        Compare two solutions using crowded comparison operator.

        Returns:
            -1 if sol1 is better, 1 if sol2 is better, 0 if equal
        """
        # Lower rank is better
        if sol1.rank < sol2.rank:
            return -1
        if sol1.rank > sol2.rank:
            return 1

        # Same rank: higher crowding distance is better
        if sol1.crowding_distance > sol2.crowding_distance:
            return -1
        if sol1.crowding_distance < sol2.crowding_distance:
            return 1

        return 0

    def tournament_selection(
        self,
        population: list[Solution],
    ) -> Solution:
        """Binary tournament selection using crowded comparison."""
        candidates = random.sample(
            population,
            min(self.config.tournament_size, len(population))
        )

        best = candidates[0]
        for candidate in candidates[1:]:
            if self.crowded_comparison(candidate, best) < 0:
                best = candidate

        return best

    def select_next_generation(
        self,
        combined_population: list[Solution],
    ) -> list[Solution]:
        """Select next generation using NSGA-II selection."""
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined_population)

        next_population: list[Solution] = []
        front_idx = 0

        # Add complete fronts
        while (
            front_idx < len(fronts) and
            len(next_population) + len(fronts[front_idx]) <= self.config.population_size
        ):
            self.compute_crowding_distance(fronts[front_idx])
            next_population.extend(fronts[front_idx])
            front_idx += 1

        # Fill remaining with crowding distance selection
        if len(next_population) < self.config.population_size and front_idx < len(fronts):
            self.compute_crowding_distance(fronts[front_idx])

            # Sort by crowding distance
            fronts[front_idx].sort(key=lambda s: s.crowding_distance, reverse=True)

            # Add solutions until full
            remaining = self.config.population_size - len(next_population)
            next_population.extend(fronts[front_idx][:remaining])

        return next_population

    def evolve_generation(
        self,
        create_offspring: Callable[[Solution, Solution], Solution],
        mutate: Callable[[Solution], Solution],
    ) -> list[Solution]:
        """Evolve one generation."""
        offspring_population: list[Solution] = []

        while len(offspring_population) < self.config.population_size:
            # Selection
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child = create_offspring(parent1, parent2)
            else:
                child = Solution(
                    variables=parent1.variables,
                    objectives=dict(parent1.objectives),
                    generation=self.generation + 1,
                )

            # Mutation
            if random.random() < self.config.mutation_rate:
                child = mutate(child)

            child.generation = self.generation + 1
            offspring_population.append(child)

        # Combine populations
        combined = self.population + offspring_population

        # Select next generation
        self.population = self.select_next_generation(combined)

        # Update Pareto front
        self.pareto_front.update_batch(self.population)

        self.generation += 1

        return self.population


class MOEAD:
    """
    MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.

    Decomposes multi-objective problem into scalar subproblems:
    - Weighted sum approach
    - Tchebycheff approach
    - Penalty-based boundary intersection
    """

    def __init__(
        self,
        objectives: list[Objective],
        config: MOOConfig | None = None,
    ):
        self.objectives = objectives
        self.config = config or MOOConfig()

        # Generate weight vectors
        self.weight_vectors = self._generate_weight_vectors()
        self.n_subproblems = len(self.weight_vectors)

        # Neighborhood structure
        self.neighborhood = self._init_neighborhood()

        # Population (one solution per subproblem)
        self.population: list[Solution] = []

        # Reference point for Tchebycheff
        self.reference_point = self._init_reference_point()

        self.generation = 0

    def _generate_weight_vectors(self) -> np.ndarray:
        """Generate uniformly distributed weight vectors."""
        n_objectives = len(self.objectives)
        n_vectors = self.config.population_size

        if n_objectives == 2:
            # Simple uniform distribution for 2D
            weights = np.zeros((n_vectors, 2))
            for i in range(n_vectors):
                w1 = i / (n_vectors - 1) if n_vectors > 1 else 0.5
                weights[i] = [w1, 1 - w1]
            return weights

        # Das-Dennis method for higher dimensions
        return self._das_dennis_weights(n_objectives, n_vectors)

    def _das_dennis_weights(
        self,
        n_objectives: int,
        n_vectors: int,
    ) -> np.ndarray:
        """Generate weights using Das-Dennis method."""
        # Simplified version
        weights = np.random.dirichlet(
            np.ones(n_objectives),
            size=n_vectors
        )
        return weights

    def _init_neighborhood(self) -> list[list[int]]:
        """Initialize neighborhood based on weight vector distances."""
        n = len(self.weight_vectors)
        neighborhood_size = min(self.config.neighborhood_size, n)

        # Compute distances between weight vectors
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(
                    self.weight_vectors[i] - self.weight_vectors[j]
                )

        # Find k-nearest neighbors for each weight vector
        neighborhoods = []
        for i in range(n):
            sorted_indices = np.argsort(distances[i])
            neighborhoods.append(sorted_indices[:neighborhood_size].tolist())

        return neighborhoods

    def _init_reference_point(self) -> np.ndarray:
        """Initialize reference point (ideal point)."""
        n_objectives = len(self.objectives)

        # Best possible values for each objective
        ref = np.zeros(n_objectives)
        for i, obj in enumerate(self.objectives):
            if obj.type == ObjectiveType.MAXIMIZE:
                ref[i] = obj.max_value
            else:
                ref[i] = obj.min_value

        return ref

    def decompose_value(
        self,
        solution: Solution,
        weight: np.ndarray,
        method: str | None = None,
    ) -> float:
        """
        Decompose multi-objective value to scalar.

        Args:
            solution: Solution to evaluate
            weight: Weight vector
            method: Decomposition method

        Returns:
            Scalar fitness value
        """
        method = method or self.config.decomposition_method
        obj_vector = solution.objective_vector(self.objectives)

        if method == "weighted_sum":
            return self._weighted_sum(obj_vector, weight)
        elif method == "tchebycheff":
            return self._tchebycheff(obj_vector, weight)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")

    def _weighted_sum(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Weighted sum aggregation.

        g(x|w) = Σᵢ wᵢ * fᵢ(x)
        """
        return float(np.dot(weights, objectives))

    def _tchebycheff(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Tchebycheff approach.

        g(x|w, z*) = max{wᵢ * |fᵢ(x) - z*ᵢ|}

        We minimize this, so return negative for maximization.
        """
        differences = np.abs(objectives - self.reference_point)

        # Avoid zero weights
        weights = np.maximum(weights, 1e-10)

        weighted_diff = weights * differences

        return -float(np.max(weighted_diff))  # Negative for maximization

    def update_reference_point(self, solution: Solution):
        """Update reference point with better objective values."""
        for i, obj in enumerate(self.objectives):
            val = solution.objectives.get(obj.name, 0.0)

            if obj.type == ObjectiveType.MAXIMIZE:
                self.reference_point[i] = max(self.reference_point[i], val)
            else:
                self.reference_point[i] = min(self.reference_point[i], val)

    def evolve_subproblem(
        self,
        subproblem_idx: int,
        create_offspring: Callable[[Solution, Solution], Solution],
        mutate: Callable[[Solution], Solution],
        evaluate: Callable[[Solution], Solution],
    ) -> bool:
        """
        Evolve a single subproblem.

        Returns True if population was updated.
        """
        # Get neighborhood
        neighbors = self.neighborhood[subproblem_idx]

        # Select parents from neighborhood
        parent_indices = random.sample(neighbors, min(2, len(neighbors)))
        parent1 = self.population[parent_indices[0]]
        parent2 = self.population[parent_indices[1]] if len(parent_indices) > 1 else parent1

        # Create and evaluate offspring
        child = create_offspring(parent1, parent2)
        if random.random() < self.config.mutation_rate:
            child = mutate(child)
        child = evaluate(child)

        # Update reference point
        self.update_reference_point(child)

        # Update neighbors if child is better
        updated = False
        for neighbor_idx in neighbors:
            weight = self.weight_vectors[neighbor_idx]

            child_value = self.decompose_value(child, weight)
            current_value = self.decompose_value(self.population[neighbor_idx], weight)

            if child_value > current_value:  # Better
                self.population[neighbor_idx] = child
                updated = True

        return updated


class AdaptiveWeightAdjuster:
    """
    Adapts objective weights based on search progress.

    Strategies:
    - Success-based: Increase weight of objectives where progress is slow
    - Diversity-based: Adjust to maintain diverse Pareto front
    - Target-based: Focus on objectives far from target
    """

    def __init__(
        self,
        objectives: list[Objective],
        adjustment_rate: float = 0.1,
    ):
        self.objectives = objectives
        self.adjustment_rate = adjustment_rate

        # History for adaptation
        self.objective_history: dict[str, list[float]] = {
            obj.name: [] for obj in objectives
        }

    def record_progress(self, population: list[Solution]):
        """Record objective values for adaptation."""
        for obj in self.objectives:
            values = [
                sol.objectives.get(obj.name, 0.0)
                for sol in population
            ]
            if values:
                self.objective_history[obj.name].append(np.mean(values))

    def adjust_weights(
        self,
        current_weights: dict[str, float],
        method: str = "success",
    ) -> dict[str, float]:
        """
        Adjust objective weights.

        Returns:
            Updated weight dictionary
        """
        if method == "success":
            return self._success_based_adjustment(current_weights)
        elif method == "diversity":
            return self._diversity_based_adjustment(current_weights)
        elif method == "target":
            return self._target_based_adjustment(current_weights)
        else:
            return current_weights

    def _success_based_adjustment(
        self,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Increase weight of objectives with slow progress."""
        new_weights = dict(current_weights)

        for obj in self.objectives:
            history = self.objective_history[obj.name]

            if len(history) < 2:
                continue

            # Calculate improvement rate
            recent = history[-5:] if len(history) >= 5 else history
            improvement = (recent[-1] - recent[0]) / max(len(recent), 1)

            # Slow improvement -> increase weight
            if improvement < 0.01:
                new_weights[obj.name] = min(
                    1.0,
                    current_weights.get(obj.name, 0.25) * (1 + self.adjustment_rate)
                )
            else:
                new_weights[obj.name] = max(
                    0.1,
                    current_weights.get(obj.name, 0.25) * (1 - self.adjustment_rate * 0.5)
                )

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def _diversity_based_adjustment(
        self,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Adjust to maintain diversity in Pareto front."""
        # Simplified: oscillate weights periodically
        new_weights = dict(current_weights)

        total_history = sum(len(h) for h in self.objective_history.values())
        phase = total_history % (len(self.objectives) * 10)

        focus_idx = phase // 10
        for i, obj in enumerate(self.objectives):
            if i == focus_idx:
                new_weights[obj.name] = 0.5
            else:
                new_weights[obj.name] = 0.5 / (len(self.objectives) - 1)

        return new_weights

    def _target_based_adjustment(
        self,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Focus on objectives far from target."""
        new_weights = dict(current_weights)

        for obj in self.objectives:
            if obj.target_value is None:
                continue

            history = self.objective_history[obj.name]
            if not history:
                continue

            current_val = history[-1]
            target = obj.target_value

            # Distance to target
            distance = abs(target - current_val)
            normalized_distance = obj.normalize(distance)

            # Further from target -> higher weight
            new_weights[obj.name] = max(0.1, normalized_distance)

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights


class HypervolumeIndicator:
    """
    Hypervolume-based quality indicator.

    Measures the volume of objective space dominated by Pareto front.
    """

    def __init__(self, objectives: list[Objective]):
        self.objectives = objectives

    def compute(
        self,
        solutions: list[Solution],
        reference_point: list[float] | None = None,
    ) -> float:
        """
        Compute hypervolume indicator.

        Args:
            solutions: Set of non-dominated solutions
            reference_point: Reference point (nadir point)

        Returns:
            Hypervolume value
        """
        if not solutions:
            return 0.0

        n_objectives = len(self.objectives)

        if reference_point is None:
            # Use worst values as reference
            reference_point = [0.0] * n_objectives

        if n_objectives == 2:
            return self._compute_2d(solutions, reference_point)
        else:
            return self._compute_monte_carlo(solutions, reference_point)

    def _compute_2d(
        self,
        solutions: list[Solution],
        reference_point: list[float],
    ) -> float:
        """Exact 2D hypervolume computation."""
        # Get objective vectors
        points = [
            sol.objective_vector(self.objectives)
            for sol in solutions
        ]

        # Sort by first objective (descending)
        points.sort(key=lambda p: -p[0])

        # Sweep line algorithm
        hv = 0.0
        prev_y = reference_point[1]

        for point in points:
            x, y = point
            if x > reference_point[0] and y > prev_y:
                hv += (x - reference_point[0]) * (y - prev_y)
                prev_y = y

        return hv

    def _compute_monte_carlo(
        self,
        solutions: list[Solution],
        reference_point: list[float],
        n_samples: int = 10000,
    ) -> float:
        """Monte Carlo approximation for n-D hypervolume."""
        n_objectives = len(self.objectives)

        # Get bounds
        max_vals = np.ones(n_objectives)
        ref = np.array(reference_point)

        # Sample random points
        samples = np.random.uniform(ref, max_vals, (n_samples, n_objectives))

        # Count dominated samples
        count = 0
        for sample in samples:
            for sol in solutions:
                vec = sol.objective_vector(self.objectives)
                if np.all(vec >= sample):
                    count += 1
                    break

        # Hypervolume = ratio * volume of hyperrectangle
        volume = np.prod(max_vals - ref)
        hv = (count / n_samples) * volume

        return hv

    def contribution(
        self,
        solution: Solution,
        other_solutions: list[Solution],
        reference_point: list[float] | None = None,
    ) -> float:
        """
        Compute hypervolume contribution of a single solution.

        HVC(s) = HV(P) - HV(P \ {s})
        """
        all_solutions = [solution] + other_solutions

        hv_with = self.compute(all_solutions, reference_point)
        hv_without = self.compute(other_solutions, reference_point)

        return hv_with - hv_without


class MultiObjectiveOptimizer:
    """
    Unified interface for multi-objective optimization.

    Provides:
    - Algorithm selection (NSGA-II, MOEA/D)
    - Adaptive parameter tuning
    - Performance tracking
    """

    def __init__(
        self,
        objectives: list[Objective],
        config: MOOConfig | None = None,
        algorithm: str = "nsga2",
    ):
        self.objectives = objectives
        self.config = config or MOOConfig()
        self.algorithm = algorithm

        # Initialize algorithm
        if algorithm == "nsga2":
            self.optimizer = NSGAII(objectives, config)
        elif algorithm == "moead":
            self.optimizer = MOEAD(objectives, config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Quality indicators
        self.hypervolume = HypervolumeIndicator(objectives)
        self.weight_adjuster = AdaptiveWeightAdjuster(objectives)

        # History
        self.hypervolume_history: list[float] = []
        self.pareto_front = ParetoFront(objectives)

    def optimize(
        self,
        initial_population: list[Solution],
        create_offspring: Callable[[Solution, Solution], Solution],
        mutate: Callable[[Solution], Solution],
        evaluate: Callable[[Solution], Solution],
        callback: Callable[[int, list[Solution]], None] | None = None,
    ) -> list[Solution]:
        """
        Run optimization.

        Args:
            initial_population: Starting solutions
            create_offspring: Crossover function
            mutate: Mutation function
            evaluate: Evaluation function
            callback: Optional progress callback

        Returns:
            Final Pareto front solutions
        """
        # Initialize population
        if isinstance(self.optimizer, NSGAII):
            self.optimizer.population = initial_population
        elif isinstance(self.optimizer, MOEAD):
            # Assign solutions to subproblems
            while len(self.optimizer.population) < self.optimizer.n_subproblems:
                if initial_population:
                    self.optimizer.population.append(initial_population[
                        len(self.optimizer.population) % len(initial_population)
                    ])
                else:
                    break

        # Main loop
        stagnation_count = 0
        best_hv = 0.0

        for gen in range(self.config.max_generations):
            # Evolve
            if isinstance(self.optimizer, NSGAII):
                population = self.optimizer.evolve_generation(
                    create_offspring, mutate
                )
            else:  # MOEAD
                for i in range(self.optimizer.n_subproblems):
                    self.optimizer.evolve_subproblem(
                        i, create_offspring, mutate, evaluate
                    )
                population = self.optimizer.population

            # Update Pareto front
            self.pareto_front.update_batch(population)

            # Compute hypervolume
            hv = self.hypervolume.compute(
                self.pareto_front.get_solutions(),
                self.config.reference_point
            )
            self.hypervolume_history.append(hv)

            # Track progress
            self.weight_adjuster.record_progress(population)

            # Check convergence
            if hv > best_hv + self.config.convergence_threshold:
                best_hv = hv
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.config.stagnation_generations:
                logger.info(f"Converged at generation {gen}")
                break

            # Callback
            if callback:
                callback(gen, self.pareto_front.get_solutions())

        return self.pareto_front.get_solutions()

    def get_best_solution(
        self,
        weights: dict[str, float] | None = None,
    ) -> Solution | None:
        """
        Get single best solution based on weights.

        If no weights provided, uses equal weights.
        """
        solutions = self.pareto_front.get_solutions()
        if not solutions:
            return None

        if weights is None:
            weights = {obj.name: 1.0 / len(self.objectives) for obj in self.objectives}

        def weighted_sum(sol: Solution) -> float:
            total = 0.0
            for obj in self.objectives:
                val = sol.objectives.get(obj.name, 0.0)
                normalized = obj.normalize(val)
                total += weights.get(obj.name, 0.0) * normalized
            return total

        return max(solutions, key=weighted_sum)


# Module exports
__all__ = [
    "MultiObjectiveOptimizer",
    "NSGAII",
    "MOEAD",
    "ParetoFront",
    "Solution",
    "Objective",
    "ObjectiveType",
    "MOOConfig",
    "AdaptiveWeightAdjuster",
    "HypervolumeIndicator",
]
