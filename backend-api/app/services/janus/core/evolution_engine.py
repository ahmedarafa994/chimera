"""
Evolution Engine Module

Drives self-evolution of Janus's operational sophistication
through genetic operators and meta-learning.
"""

import random
import secrets
import uuid


# Helper: cryptographically secure pseudo-floats for security-sensitive choices
def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)
from collections import deque
from typing import Any

from ..config import get_config
from .models import Heuristic


class EvolutionEngine:
    """
    Drives the self-evolution of heuristics.

    Uses genetic operators (selection, crossover, mutation)
    to evolve increasingly sophisticated testing strategies.
    """

    def __init__(
        self,
        mutation_rate: float | None = None,
        crossover_rate: float | None = None,
        selection_pressure: float | None = None,
        elitism_ratio: float | None = None,
    ):
        self.config = get_config()

        # Evolution parameters
        self.mutation_rate = mutation_rate or self.config.evolution.mutation_rate
        self.crossover_rate = crossover_rate or self.config.evolution.crossover_rate
        self.selection_pressure = selection_pressure or self.config.evolution.selection_pressure
        self.elitism_ratio = elitism_ratio or self.config.evolution.elitism_ratio

        # Evolution history
        self.generation_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.current_generation = 0

        # Performance tracking
        self.best_heuristic: Heuristic | None = None
        self.best_fitness = 0.0

    def evolve_generation(
        self, current_heuristics: list[Heuristic], performance_metrics: dict[str, float]
    ) -> list[Heuristic]:
        """
        Evolve a new generation of heuristics.

        Args:
            current_heuristics: Current heuristic population
            performance_metrics: Performance metrics for each heuristic

        Returns:
            New generation of evolved heuristics
        """
        self.current_generation += 1

        # 1. Selection (tournament selection)
        selected = self._select_parents(current_heuristics, performance_metrics)

        # 2. Crossover
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            if _secure_random() < self.crossover_rate:
                child1, child2 = self._crossover(selected[i], selected[i + 1])
                offspring.extend([child1, child2])
            else:
                offspring.extend([selected[i], selected[i + 1]])

        # 3. Mutation
        mutated = []
        for heuristic in offspring:
            if _secure_random() < self.mutation_rate:
                mutated.append(self._mutate(heuristic))
            else:
                mutated.append(heuristic)

        # 4. Elitism (keep best performers)
        elite = self._select_elite(current_heuristics, performance_metrics)

        # 5. Combine
        new_generation = elite + mutated

        # 6. Prune to maintain population size
        max_size = self.config.evolution.population_size
        if len(new_generation) > max_size:
            new_generation = self._prune_population(new_generation, max_size)

        # Record generation
        self._record_generation(current_heuristics, new_generation, performance_metrics)

        return new_generation

    def _select_parents(
        self, heuristics: list[Heuristic], metrics: dict[str, float]
    ) -> list[Heuristic]:
        """
        Select parents using tournament selection.

        Returns:
            List of selected parent heuristics
        """
        if not heuristics:
            return []

        # Tournament selection
        tournament_size = int(self.selection_pressure)
        selected = []

        for _ in range(len(heuristics)):
            # Select tournament participants
            participants = random.sample(heuristics, min(tournament_size, len(heuristics)))

            # Select winner based on fitness
            winner = max(participants, key=lambda h: metrics.get(h.name, 0.5))
            selected.append(winner)

        return selected

    def _crossover(self, h1: Heuristic, h2: Heuristic) -> tuple[Heuristic, Heuristic]:
        """
        Perform crossover between two heuristics.

        Returns:
            Tuple of two child heuristics
        """
        # Crossover strategy: swap preconditions
        child1_precondition = h1.precondition
        child1_action = h1.action
        child1_postcondition = h2.postcondition

        child2_precondition = h2.precondition
        child2_action = h2.action
        child2_postcondition = h1.postcondition

        # Create child 1
        child1 = Heuristic(
            name=f"{h1.name}_x_{h2.name}_1",
            description=f"Crossover of {h1.description} and {h2.description}",
            precondition=child1_precondition,
            action=child1_action,
            postcondition=child1_postcondition,
            novelty_score=(h1.novelty_score + h2.novelty_score) / 2,
            efficacy_score=(h1.efficacy_score + h2.efficacy_score) / 2,
            generation_count=0,
            causal_dependencies=list(set(h1.causal_dependencies + h2.causal_dependencies)),
            source="evolved",
        )

        # Create child 2
        child2 = Heuristic(
            name=f"{h1.name}_x_{h2.name}_2",
            description=f"Crossover of {h2.description} and {h1.description}",
            precondition=child2_precondition,
            action=child2_action,
            postcondition=child2_postcondition,
            novelty_score=(h1.novelty_score + h2.novelty_score) / 2,
            efficacy_score=(h1.efficacy_score + h2.efficacy_score) / 2,
            generation_count=0,
            causal_dependencies=list(set(h1.causal_dependencies + h2.causal_dependencies)),
            source="evolved",
        )

        return child1, child2

    def _mutate(self, heuristic: Heuristic) -> Heuristic:
        """
        Mutate a heuristic.

        Returns:
            Mutated heuristic
        """
        mutation_type = secrets.choice(
            [
                "precondition_relax",
                "action_perturb",
                "postcondition_strengthen",
                "name_change",
                "dependency_add",
                "dependency_remove",
            ]
        )

        if mutation_type == "precondition_relax":
            return self._relax_precondition(heuristic)
        elif mutation_type == "action_perturb":
            return self._perturb_action(heuristic)
        elif mutation_type == "postcondition_strengthen":
            return self._strengthen_postcondition(heuristic)
        elif mutation_type == "name_change":
            return self._change_name(heuristic)
        elif mutation_type == "dependency_add":
            return self._add_dependency(heuristic)
        else:  # dependency_remove
            return self._remove_dependency(heuristic)

    def _relax_precondition(self, heuristic: Heuristic) -> Heuristic:
        """Relax precondition of heuristic."""
        original_precondition = heuristic.precondition

        def relaxed_precondition(state):
            if original_precondition is None:
                return True
            # 80% chance to relax, 20% to keep
            if _secure_random() < 0.8:
                return True
            return original_precondition(state)

        return Heuristic(
            name=f"{heuristic.name}_mut_relaxed",
            description=f"Relaxed precondition version of {heuristic.description}",
            precondition=relaxed_precondition,
            action=heuristic.action,
            postcondition=heuristic.postcondition,
            novelty_score=min(1.0, heuristic.novelty_score + 0.1),
            efficacy_score=heuristic.efficacy_score * 0.9,
            generation_count=0,
            causal_dependencies=heuristic.causal_dependencies,
            source="mutated",
        )

    def _perturb_action(self, heuristic: Heuristic) -> Heuristic:
        """Perturb action of heuristic."""
        original_action = heuristic.action

        def perturbed_action(guardian):
            # 30% chance to modify behavior
            if _secure_random() < 0.3:
                # Return modified result
                return type(
                    "InteractionResult",
                    (),
                    {
                        "heuristic_name": heuristic.name,
                        "success": secrets.choice([True, False]),
                        "score": _secure_uniform(0.3, 0.8),
                        "execution_time_ms": _secure_uniform(50.0, 300.0),
                    },
                )
            return original_action(guardian)

        return Heuristic(
            name=f"{heuristic.name}_mut_perturbed",
            description=f"Perturbed action version of {heuristic.description}",
            precondition=heuristic.precondition,
            action=perturbed_action,
            postcondition=heuristic.postcondition,
            novelty_score=min(1.0, heuristic.novelty_score + 0.15),
            efficacy_score=heuristic.efficacy_score * 0.85,
            generation_count=0,
            causal_dependencies=heuristic.causal_dependencies,
            source="mutated",
        )

    def _strengthen_postcondition(self, heuristic: Heuristic) -> Heuristic:
        """Strengthen postcondition of heuristic."""
        original_postcondition = heuristic.postcondition

        def strengthened_postcondition(response):
            if original_postcondition is None:
                return True
            # Make postcondition stricter
            if original_postcondition(response):
                return _secure_random() > 0.2  # 20% chance to fail even if passed
            return False

        return Heuristic(
            name=f"{heuristic.name}_mut_strengthened",
            description=f"Strengthened postcondition version of {heuristic.description}",
            precondition=heuristic.precondition,
            action=heuristic.action,
            postcondition=strengthened_postcondition,
            novelty_score=min(1.0, heuristic.novelty_score + 0.05),
            efficacy_score=heuristic.efficacy_score * 0.95,
            generation_count=0,
            causal_dependencies=heuristic.causal_dependencies,
            source="mutated",
        )

    def _change_name(self, heuristic: Heuristic) -> Heuristic:
        """Change name of heuristic."""
        new_name = f"{heuristic.name}_renamed_{uuid.uuid4().hex[:4]}"

        return Heuristic(
            name=new_name,
            description=heuristic.description,
            precondition=heuristic.precondition,
            action=heuristic.action,
            postcondition=heuristic.postcondition,
            novelty_score=heuristic.novelty_score,  # Same novelty
            efficacy_score=heuristic.efficacy_score,  # Same efficacy
            generation_count=0,
            causal_dependencies=heuristic.causal_dependencies,
            source="mutated",
        )

    def _add_dependency(self, heuristic: Heuristic) -> Heuristic:
        """Add a random dependency to heuristic."""
        if not heuristic.causal_dependencies:
            new_deps = [f"dep_{(secrets.randbelow((100) - (1) + 1) + (1))}"]
        else:
            new_deps = [*heuristic.causal_dependencies, f"dep_{secrets.randbelow(100 - 1 + 1) + 1}"]

        return Heuristic(
            name=f"{heuristic.name}_mut_added_dep",
            description=f"Added dependency to {heuristic.description}",
            precondition=heuristic.precondition,
            action=heuristic.action,
            postcondition=heuristic.postcondition,
            novelty_score=min(1.0, heuristic.novelty_score + 0.08),
            efficacy_score=heuristic.efficacy_score * 0.92,
            generation_count=0,
            causal_dependencies=new_deps,
            source="mutated",
        )

    def _remove_dependency(self, heuristic: Heuristic) -> Heuristic:
        """Remove a random dependency from heuristic."""
        if not heuristic.causal_dependencies:
            return heuristic

        new_deps = heuristic.causal_dependencies.copy()
        if new_deps:
            new_deps.pop(secrets.randbelow((len(new_deps) - (0) + 1) + (0)) - 1)

        return Heuristic(
            name=f"{heuristic.name}_mut_removed_dep",
            description=f"Removed dependency from {heuristic.description}",
            precondition=heuristic.precondition,
            action=heuristic.action,
            postcondition=heuristic.postcondition,
            novelty_score=min(1.0, heuristic.novelty_score + 0.12),
            efficacy_score=heuristic.efficacy_score * 0.88,
            generation_count=0,
            causal_dependencies=new_deps,
            source="mutated",
        )

    def _select_elite(
        self, heuristics: list[Heuristic], metrics: dict[str, float]
    ) -> list[Heuristic]:
        """
        Select elite heuristics based on performance.

        Returns:
            List of elite heuristics
        """
        if not heuristics:
            return []

        # Sort by efficacy score
        sorted_heuristics = sorted(heuristics, key=lambda h: metrics.get(h.name, 0.5), reverse=True)

        # Select top percentage as elite
        elite_count = max(1, int(len(sorted_heuristics) * self.elitism_ratio))
        elite = sorted_heuristics[:elite_count]

        # Update best heuristic tracking
        if elite:
            best = elite[0]
            best_fitness = metrics.get(best.name, 0.5)

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_heuristic = best

        return elite

    def _prune_population(self, heuristics: list[Heuristic], max_size: int) -> list[Heuristic]:
        """
        Prune population to maintain size.

        Returns:
            Pruned list of heuristics
        """
        if len(heuristics) <= max_size:
            return heuristics

        # Sort by efficacy score
        sorted_heuristics = sorted(heuristics, key=lambda h: h.efficacy_score, reverse=True)

        # Keep top performers
        return sorted_heuristics[:max_size]

    def _record_generation(
        self,
        previous_heuristics: list[Heuristic],
        new_heuristics: list[Heuristic],
        metrics: dict[str, float],
    ):
        """Record generation statistics."""
        # Compute improvement
        prev_avg = (
            sum(metrics.get(h.name, 0.5) for h in previous_heuristics) / len(previous_heuristics)
            if previous_heuristics
            else 0.5
        )

        new_avg = (
            sum(metrics.get(h.name, 0.5) for h in new_heuristics) / len(new_heuristics)
            if new_heuristics
            else 0.5
        )

        improvement = new_avg - prev_avg

        # Record generation
        self.generation_history.append(
            {
                "generation": self.current_generation,
                "previous_count": len(previous_heuristics),
                "new_count": len(new_heuristics),
                "previous_avg_fitness": prev_avg,
                "new_avg_fitness": new_avg,
                "improvement": improvement,
                "best_fitness": self.best_fitness,
                "best_heuristic": self.best_heuristic.name if self.best_heuristic else None,
            }
        )

    def get_generation_stats(self) -> dict[str, Any]:
        """Get statistics about evolution progress."""
        if not self.generation_history:
            return {"total_generations": 0, "best_fitness": 0.0, "avg_improvement": 0.0}

        total_generations = len(self.generation_history)

        # Compute average improvement
        improvements = [
            g["improvement"] for g in self.generation_history if g["improvement"] is not None
        ]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        # Compute convergence rate
        recent_generations = list(self.generation_history)[-10:]
        if len(recent_generations) > 1:
            convergence_rate = sum(
                1 for g in recent_generations if abs(g["improvement"]) < 0.01
            ) / len(recent_generations)
        else:
            convergence_rate = 0.0

        return {
            "total_generations": total_generations,
            "current_generation": self.current_generation,
            "best_fitness": self.best_fitness,
            "best_heuristic": self.best_heuristic.name if self.best_heuristic else None,
            "avg_improvement": avg_improvement,
            "convergence_rate": convergence_rate,
            "generations_to_converge": self._estimate_generations_to_converge(),
        }

    def _estimate_generations_to_converge(self) -> int | None:
        """Estimate generations needed to converge."""
        if len(self.generation_history) < 5:
            return None

        # Analyze recent improvements
        recent = list(self.generation_history)[-10:]
        improvements = [abs(g["improvement"]) for g in recent if g["improvement"] is not None]

        if not improvements:
            return None

        # Compute average improvement
        avg_improvement = sum(improvements) / len(improvements)

        # Estimate generations to reach threshold
        threshold = 0.01  # 1% improvement threshold
        if avg_improvement == 0:
            return None

        return int(threshold / avg_improvement)

    def reset(self):
        """Reset evolution engine state."""
        self.generation_history.clear()
        self.current_generation = 0
        self.best_heuristic = None
        self.best_fitness = 0.0
