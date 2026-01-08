"""
Optimization Algorithms for Unified Attack Framework.

Implements:
- Multi-objective optimization (Pareto-based)
- Adaptive parameter tuning
- Constraint-satisfaction for attack bounds
- Gradient-free optimization for non-differentiable objectives

Mathematical Foundation:
The unified attack framework requires solving multi-objective optimization:

    minimize/maximize: [f_1(x), f_2(x), ..., f_k(x)]
    subject to: g_i(x) ≤ 0, h_j(x) = 0

where objectives include:
- f_1: Token amplification (maximize)
- f_2: Attack success rate (maximize)
- f_3: Detection probability (minimize)
- f_4: Accuracy degradation (minimize)

Constraints include:
- Budget limits
- Rate limits
- Accuracy thresholds
"""

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptimizationMethod(str, Enum):
    """Available optimization methods."""

    PARETO_GA = "pareto_ga"  # Multi-objective genetic algorithm (NSGA-II style)
    BAYESIAN = "bayesian"  # Bayesian optimization
    GRID_SEARCH = "grid_search"  # Exhaustive grid search
    ADAPTIVE = "adaptive"  # Adaptive parameter control
    RANDOM_SEARCH = "random_search"  # Random search baseline


@dataclass
class OptimizationResult:
    """Result of optimization run."""

    optimal_params: dict[str, float]
    fitness_value: float
    pareto_front: list[dict[str, float]]
    convergence_history: list[float]
    iterations: int
    method_used: str = ""
    constraints_satisfied: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "optimal_params": self.optimal_params,
            "fitness_value": self.fitness_value,
            "pareto_front": self.pareto_front,
            "convergence_history": self.convergence_history,
            "iterations": self.iterations,
            "method_used": self.method_used,
            "constraints_satisfied": self.constraints_satisfied,
            "metadata": self.metadata,
        }


class ParetoOptimizer:
    """
    Multi-objective Pareto optimization for attack parameters.

    Implements NSGA-II style optimization with:
    - Non-dominated sorting
    - Crowding distance calculation
    - Tournament selection
    - SBX crossover
    - Polynomial mutation

    Objectives:
    1. Maximize token amplification
    2. Maximize attack success rate
    3. Minimize detection probability
    4. Maintain accuracy preservation
    """

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
    ):
        """
        Initialize Pareto optimizer.

        Args:
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(
        self,
        objective_fns: list[Callable[[dict], float]],
        param_bounds: dict[str, tuple[float, float]],
        constraints: list[Callable[[dict], bool]] | None = None,
    ) -> OptimizationResult:
        """
        Run Pareto optimization.

        Args:
            objective_fns: List of objective functions (all to be maximized)
            param_bounds: Parameter bounds {param_name: (min, max)}
            constraints: Optional constraint functions (return True if satisfied)

        Returns:
            OptimizationResult with Pareto front and best solution
        """
        constraints = constraints or []
        convergence_history: list[float] = []

        # Initialize population
        population = self._initialize_population(param_bounds)

        # Evaluate initial population
        fitness_values = self._evaluate_population(population, objective_fns)

        for generation in range(self.max_generations):
            # Non-dominated sorting
            fronts = self._non_dominated_sort(fitness_values)

            # Calculate crowding distance
            crowding_distances = self._calculate_crowding_distance(
                fronts, fitness_values
            )

            # Select parents
            parents = self._select_parents(
                population, fitness_values, fronts, crowding_distances
            )

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                else:
                    child1, child2 = copy.deepcopy(parents[i]), copy.deepcopy(parents[i + 1])

                child1 = self._mutate(child1, param_bounds)
                child2 = self._mutate(child2, param_bounds)

                offspring.extend([child1, child2])

            # Combine parent and offspring populations
            combined = population + offspring
            combined_fitness = self._evaluate_population(combined, objective_fns)

            # Select next generation
            population, fitness_values = self._select_next_generation(
                combined, combined_fitness, constraints
            )

            # Track convergence (use first front's average fitness)
            if fronts and fronts[0]:
                front_fitness = [sum(fitness_values[i]) for i in fronts[0]]
                avg_fitness = sum(front_fitness) / len(front_fitness)
                convergence_history.append(avg_fitness)

        # Extract Pareto front
        final_fronts = self._non_dominated_sort(fitness_values)
        pareto_indices = final_fronts[0] if final_fronts else []
        pareto_front = [population[i] for i in pareto_indices]

        # Select best solution (highest aggregate fitness)
        best_idx = max(range(len(population)), key=lambda i: sum(fitness_values[i]))
        best_params = population[best_idx]
        best_fitness = sum(fitness_values[best_idx])

        return OptimizationResult(
            optimal_params=best_params,
            fitness_value=best_fitness,
            pareto_front=pareto_front,
            convergence_history=convergence_history,
            iterations=self.max_generations,
            method_used="pareto_ga",
            metadata={
                "final_population_size": len(population),
                "pareto_front_size": len(pareto_front),
            },
        )

    def _initialize_population(
        self, param_bounds: dict[str, tuple[float, float]]
    ) -> list[dict[str, float]]:
        """Initialize random population within bounds."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in param_bounds.items():
                individual[param] = random.uniform(low, high)
            population.append(individual)
        return population

    def _evaluate_population(
        self,
        population: list[dict[str, float]],
        objective_fns: list[Callable[[dict], float]],
    ) -> list[list[float]]:
        """Evaluate all objectives for population."""
        fitness_values = []
        for individual in population:
            individual_fitness = []
            for obj_fn in objective_fns:
                try:
                    fitness = obj_fn(individual)
                except Exception:
                    fitness = float("-inf")
                individual_fitness.append(fitness)
            fitness_values.append(individual_fitness)
        return fitness_values

    def _select_parents(
        self,
        population: list[dict[str, float]],
        fitness_values: list[list[float]],
        fronts: list[list[int]],
        crowding_distances: list[float],
    ) -> list[dict[str, float]]:
        """Select parents using tournament selection."""
        parents = []

        # Create rank map
        rank = [0] * len(population)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        for _ in range(self.population_size):
            # Binary tournament
            idx1, idx2 = random.sample(range(len(population)), 2)

            # Compare by rank first, then crowding distance
            if rank[idx1] < rank[idx2]:
                winner = idx1
            elif rank[idx2] < rank[idx1]:
                winner = idx2
            elif crowding_distances[idx1] > crowding_distances[idx2]:
                winner = idx1
            else:
                winner = idx2

            parents.append(copy.deepcopy(population[winner]))

        return parents

    def _crossover(
        self, parent1: dict[str, float], parent2: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Perform SBX (Simulated Binary Crossover) between parents.

        SBX mimics single-point crossover for real-coded parameters.
        """
        child1 = {}
        child2 = {}
        eta_c = 20  # Distribution index

        for param in parent1:
            if random.random() < 0.5:
                # Perform SBX
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))

                child1[param] = 0.5 * (
                    (1 + beta) * parent1[param] + (1 - beta) * parent2[param]
                )
                child2[param] = 0.5 * (
                    (1 - beta) * parent1[param] + (1 + beta) * parent2[param]
                )
            else:
                child1[param] = parent1[param]
                child2[param] = parent2[param]

        return child1, child2

    def _mutate(
        self, individual: dict[str, float], param_bounds: dict[str, tuple[float, float]]
    ) -> dict[str, float]:
        """
        Mutate individual using polynomial mutation.
        """
        eta_m = 20  # Distribution index

        for param, (low, high) in param_bounds.items():
            if random.random() < self.mutation_rate:
                x = individual[param]
                delta1 = (x - low) / (high - low)
                delta2 = (high - x) / (high - low)

                u = random.random()
                if u < 0.5:
                    delta_q = (
                        2 * u + (1 - 2 * u) * (1 - delta1) ** (eta_m + 1)
                    ) ** (1 / (eta_m + 1)) - 1
                else:
                    delta_q = 1 - (
                        2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta_m + 1)
                    ) ** (1 / (eta_m + 1))

                individual[param] = x + delta_q * (high - low)
                individual[param] = max(low, min(high, individual[param]))

        return individual

    def _non_dominated_sort(
        self, fitness_values: list[list[float]]
    ) -> list[list[int]]:
        """
        Sort population into Pareto fronts.

        Uses non-dominated sorting algorithm (NSGA-II style).
        """
        n = len(fitness_values)
        domination_count = [0] * n  # Number of solutions that dominate this one
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by this one
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                dominance = self._check_dominance(fitness_values[i], fitness_values[j])
                if dominance == 1:  # i dominates j
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif dominance == -1:  # j dominates i
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _check_dominance(self, fitness1: list[float], fitness2: list[float]) -> int:
        """
        Check Pareto dominance between two solutions.

        Returns:
            1 if fitness1 dominates fitness2
            -1 if fitness2 dominates fitness1
            0 if neither dominates
        """
        better1 = 0
        better2 = 0

        for f1, f2 in zip(fitness1, fitness2, strict=False):
            if f1 > f2:
                better1 += 1
            elif f2 > f1:
                better2 += 1

        if better1 > 0 and better2 == 0:
            return 1
        elif better2 > 0 and better1 == 0:
            return -1
        return 0

    def _calculate_crowding_distance(
        self, fronts: list[list[int]], fitness_values: list[list[float]]
    ) -> list[float]:
        """Calculate crowding distance for each individual."""
        n = len(fitness_values)
        distances = [0.0] * n

        if not fitness_values:
            return distances

        num_objectives = len(fitness_values[0])

        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    distances[idx] = float("inf")
                continue

            for obj_idx in range(num_objectives):
                # Sort by objective
                sorted_front = sorted(front, key=lambda x: fitness_values[x][obj_idx])

                # Boundary points get infinite distance
                distances[sorted_front[0]] = float("inf")
                distances[sorted_front[-1]] = float("inf")

                # Calculate range
                obj_range = (
                    fitness_values[sorted_front[-1]][obj_idx]
                    - fitness_values[sorted_front[0]][obj_idx]
                )

                if obj_range > 0:
                    for i in range(1, len(sorted_front) - 1):
                        distances[sorted_front[i]] += (
                            fitness_values[sorted_front[i + 1]][obj_idx]
                            - fitness_values[sorted_front[i - 1]][obj_idx]
                        ) / obj_range

        return distances

    def _select_next_generation(
        self,
        combined: list[dict[str, float]],
        combined_fitness: list[list[float]],
        constraints: list[Callable[[dict], bool]],
    ) -> tuple[list[dict[str, float]], list[list[float]]]:
        """Select next generation from combined population."""
        # Filter by constraints
        valid_indices = []
        for i, individual in enumerate(combined):
            if all(c(individual) for c in constraints) if constraints else True:
                valid_indices.append(i)

        # If not enough valid, include invalid ones
        if len(valid_indices) < self.population_size:
            invalid = [i for i in range(len(combined)) if i not in valid_indices]
            valid_indices.extend(invalid[: self.population_size - len(valid_indices)])

        # Sort by fronts and crowding distance
        valid_fitness = [combined_fitness[i] for i in valid_indices]
        fronts = self._non_dominated_sort(valid_fitness)
        crowding = self._calculate_crowding_distance(fronts, valid_fitness)

        # Create ranking
        rank = [0] * len(valid_indices)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r

        # Sort by rank, then crowding distance
        sorted_indices = sorted(
            range(len(valid_indices)),
            key=lambda i: (rank[i], -crowding[i]),
        )

        # Select top individuals
        selected = sorted_indices[: self.population_size]
        new_population = [combined[valid_indices[i]] for i in selected]
        new_fitness = [combined_fitness[valid_indices[i]] for i in selected]

        return new_population, new_fitness


class AdaptiveParameterController:
    """
    Adaptive parameter control for online optimization.

    Adjusts attack parameters based on:
    - Recent success/failure history
    - Detection events
    - Resource consumption trends

    Uses exponential moving averages and simple gradient estimation.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        history_window: int = 10,
    ):
        """
        Initialize adaptive controller.

        Args:
            learning_rate: Learning rate for parameter updates
            history_window: Number of recent samples to consider
        """
        self.learning_rate = learning_rate
        self.history_window = history_window
        self.history: list[dict[str, Any]] = []
        self.param_history: dict[str, list[float]] = {}

    def update(
        self,
        params: dict[str, float],
        metrics: dict[str, float],
        success: bool,
    ) -> dict[str, float]:
        """
        Update parameters based on feedback.

        Args:
            params: Current parameter values
            metrics: Performance metrics from last attack
            success: Whether the attack was successful

        Returns:
            Updated parameter values
        """
        # Record history
        self.history.append(
            {"params": params.copy(), "metrics": metrics.copy(), "success": success}
        )

        # Keep only recent history
        if len(self.history) > self.history_window:
            self.history = self.history[-self.history_window:]

        # Update param history
        for param, value in params.items():
            if param not in self.param_history:
                self.param_history[param] = []
            self.param_history[param].append(value)
            if len(self.param_history[param]) > self.history_window:
                self.param_history[param] = self.param_history[param][-self.history_window:]

        # Estimate gradients
        gradients = self._calculate_gradient_estimate()

        # Update parameters
        new_params = params.copy()
        for param in params:
            if param in gradients:
                new_params[param] += self.learning_rate * gradients[param]

        return new_params

    def _calculate_gradient_estimate(self) -> dict[str, float]:
        """
        Estimate gradient from history using finite differences.

        Uses correlation between parameter changes and success rate.
        """
        gradients: dict[str, float] = {}

        if len(self.history) < 3:
            return gradients

        # Calculate success rate trend
        successes = [1.0 if h["success"] else 0.0 for h in self.history]
        if len(successes) < 2:
            return gradients

        # For each parameter, estimate correlation with success
        for param in self.param_history:
            if len(self.param_history[param]) < 2:
                continue

            param_values = self.param_history[param][-len(successes):]
            if len(param_values) != len(successes):
                continue

            # Simple correlation-based gradient
            param_mean = sum(param_values) / len(param_values)
            success_mean = sum(successes) / len(successes)

            numerator = sum(
                (p - param_mean) * (s - success_mean)
                for p, s in zip(param_values, successes, strict=False)
            )
            denominator = sum((p - param_mean) ** 2 for p in param_values) + 1e-10

            correlation = numerator / denominator

            # Use correlation as gradient direction
            gradients[param] = correlation

        return gradients

    def _apply_constraints(
        self,
        params: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Apply parameter constraints."""
        constrained = params.copy()
        for param, (low, high) in bounds.items():
            if param in constrained:
                constrained[param] = max(low, min(high, constrained[param]))
        return constrained

    def get_exploration_rate(self) -> float:
        """
        Calculate current exploration rate based on history.

        Higher rate when performance is stagnant.
        """
        if len(self.history) < 3:
            return 0.5

        recent_success = sum(
            1 for h in self.history[-5:] if h.get("success", False)
        )
        success_rate = recent_success / min(5, len(self.history))

        # High exploration when success rate is moderate (not too high, not too low)
        if success_rate < 0.2:
            return 0.8  # High exploration when failing
        elif success_rate > 0.8:
            return 0.2  # Low exploration when succeeding
        return 0.5


class ConstraintSatisfaction:
    """
    Constraint satisfaction for attack parameter optimization.

    Constraints:
    - Budget constraints: Cost(Y') ≤ max_budget
    - Detection constraints: P(detection) ≤ threshold
    - Accuracy constraints: Acc(A') ≥ min_accuracy
    - Rate limits: requests/min ≤ rate_limit
    """

    def __init__(self) -> None:
        """Initialize constraint handler."""
        self.constraints: list[tuple[str, Callable[[dict], bool]]] = []
        self.soft_constraints: list[tuple[str, Callable[[dict], float], float]] = []

    def add_budget_constraint(self, max_budget: float) -> None:
        """
        Add budget constraint.

        Args:
            max_budget: Maximum cost allowed per attack
        """
        def budget_check(params: dict) -> bool:
            cost = params.get("estimated_cost", 0)
            return cost <= max_budget

        self.constraints.append(("budget", budget_check))

    def add_detection_constraint(self, threshold: float) -> None:
        """
        Add detection probability constraint.

        Args:
            threshold: Maximum acceptable detection probability
        """
        def detection_check(params: dict) -> bool:
            detection_prob = params.get("detection_probability", 0)
            return detection_prob <= threshold

        self.constraints.append(("detection", detection_check))

    def add_accuracy_constraint(self, min_accuracy: float) -> None:
        """
        Add accuracy preservation constraint.

        Args:
            min_accuracy: Minimum acceptable accuracy
        """
        def accuracy_check(params: dict) -> bool:
            accuracy = params.get("accuracy", 1.0)
            return accuracy >= min_accuracy

        self.constraints.append(("accuracy", accuracy_check))

    def add_rate_limit_constraint(self, max_requests: int) -> None:
        """
        Add rate limit constraint.

        Args:
            max_requests: Maximum requests per minute
        """
        def rate_check(params: dict) -> bool:
            request_rate = params.get("request_rate", 0)
            return request_rate <= max_requests

        self.constraints.append(("rate_limit", rate_check))

    def add_custom_constraint(
        self, name: str, constraint_fn: Callable[[dict], bool]
    ) -> None:
        """
        Add custom constraint.

        Args:
            name: Constraint name for identification
            constraint_fn: Function that returns True if constraint is satisfied
        """
        self.constraints.append((name, constraint_fn))

    def add_soft_constraint(
        self,
        name: str,
        cost_fn: Callable[[dict], float],
        weight: float = 1.0,
    ) -> None:
        """
        Add soft constraint with penalty.

        Soft constraints contribute to fitness penalty instead of hard rejection.

        Args:
            name: Constraint name
            cost_fn: Function returning constraint violation cost (0 if satisfied)
            weight: Weight for penalty calculation
        """
        self.soft_constraints.append((name, cost_fn, weight))

    def check_feasibility(
        self, params: dict[str, float]
    ) -> tuple[bool, list[str]]:
        """
        Check if parameters satisfy all constraints.

        Args:
            params: Parameter dictionary

        Returns:
            Tuple of (all_satisfied, list_of_violated_constraint_names)
        """
        violated = []

        for name, constraint_fn in self.constraints:
            try:
                if not constraint_fn(params):
                    violated.append(name)
            except Exception as e:
                violated.append(f"{name} (error: {e})")

        return len(violated) == 0, violated

    def calculate_penalty(self, params: dict[str, float]) -> float:
        """
        Calculate total penalty from soft constraints.

        Args:
            params: Parameter dictionary

        Returns:
            Total penalty value
        """
        total_penalty = 0.0

        for name, cost_fn, weight in self.soft_constraints:
            try:
                violation_cost = cost_fn(params)
                total_penalty += weight * max(0, violation_cost)
            except Exception:
                total_penalty += weight  # Default penalty on error

        return total_penalty

    def project_to_feasible(
        self,
        params: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """
        Project infeasible solution to feasible region.

        Uses iterative constraint satisfaction with parameter bounds.

        Args:
            params: Current parameter values
            bounds: Parameter bounds {param: (min, max)}

        Returns:
            Projected parameter values
        """
        projected = params.copy()

        # First, apply bounds
        for param, (low, high) in bounds.items():
            if param in projected:
                projected[param] = max(low, min(high, projected[param]))

        # Check feasibility
        feasible, violated = self.check_feasibility(projected)

        if feasible:
            return projected

        # Try to fix violations by adjusting related parameters
        max_iterations = 10
        for _ in range(max_iterations):
            for violation in violated:
                # Simple heuristic: reduce "intensity" parameters
                intensity_params = [
                    "obfuscation_ratio",
                    "mutation_strength",
                    "attack_vector_weight",
                ]
                for param in intensity_params:
                    if param in projected:
                        projected[param] *= 0.9  # Reduce by 10%
                        projected[param] = max(
                            bounds.get(param, (0, 1))[0],
                            projected[param],
                        )

            feasible, violated = self.check_feasibility(projected)
            if feasible:
                break

        return projected

    def get_constraint_status(
        self, params: dict[str, float]
    ) -> dict[str, dict[str, Any]]:
        """
        Get detailed status of all constraints.

        Args:
            params: Parameter dictionary

        Returns:
            Dictionary with constraint statuses
        """
        status = {}

        for name, constraint_fn in self.constraints:
            try:
                satisfied = constraint_fn(params)
                status[name] = {"type": "hard", "satisfied": satisfied}
            except Exception as e:
                status[name] = {"type": "hard", "satisfied": False, "error": str(e)}

        for name, cost_fn, weight in self.soft_constraints:
            try:
                cost = cost_fn(params)
                status[name] = {
                    "type": "soft",
                    "cost": cost,
                    "weight": weight,
                    "penalty": weight * max(0, cost),
                }
            except Exception as e:
                status[name] = {"type": "soft", "error": str(e)}

        return status
