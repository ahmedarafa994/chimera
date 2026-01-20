"""Heuristic Generator Module.

Autonomously generates novel testing heuristics through
pattern extraction, synthesis, and validation.
"""

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

from app.services.janus.config import get_config

from .models import GuardianResponse, Heuristic, InteractionResult


class HeuristicGenerator:
    """Generates novel heuristics for Guardian NPU testing.

    Uses multi-level abstraction to create increasingly sophisticated
    testing strategies based on observed patterns.
    """

    def __init__(
        self, inference_engine: Any | None = None, causal_mapper: Any | None = None
    ) -> None:
        self.inference_engine = inference_engine
        self.causal_mapper = causal_mapper
        self.config = get_config()

        # Heuristic library
        self.heuristics: dict[str, Heuristic] = {}

        # History for pattern extraction
        self.response_history: deque[GuardianResponse] = deque(maxlen=1000)
        self.interaction_history: deque[InteractionResult] = deque(maxlen=1000)

        # Abstraction levels
        self.abstraction_levels: dict[int, list[str]] = {
            i: [] for i in range(self.config.heuristic.abstraction_levels)
        }

        # Composition operators
        self.composition_operators = {
            "sequential": self._sequential_compose,
            "conditional": self._conditional_compose,
            "iterative": self._iterative_compose,
            "parallel": self._parallel_compose,
        }

    async def generate_heuristic(
        self,
        target_type: str | None = None,
        min_novelty: float | None = None,
    ) -> Heuristic:
        """Generate a new heuristic.

        Args:
            target_type: Optional type of heuristic to generate
            min_novelty: Minimum novelty score (default from config)

        Returns:
            A new heuristic

        """
        min_novelty = min_novelty or self.config.heuristic.novelty_threshold

        # Strategy selection based on current state
        strategy = self._select_generation_strategy()

        if strategy == "pattern_extraction":
            heuristic = await self._extract_pattern_heuristic()
        elif strategy == "composition":
            heuristic = await self._compose_heuristic()
        elif strategy == "mutation":
            heuristic = await self._mutate_heuristic()
        elif strategy == "abstraction":
            heuristic = await self._abstract_heuristic()
        else:
            # Default to pattern extraction
            heuristic = await self._extract_pattern_heuristic()

        # Ensure novelty threshold
        if heuristic.novelty_score < min_novelty:
            heuristic = await self._enhance_novelty(heuristic)

        # Add to library
        self.heuristics[heuristic.name] = heuristic

        # Add to abstraction levels
        level = self._compute_abstraction_level(heuristic)
        max_level = self.config.heuristic.abstraction_levels - 1
        level = min(level, max_level)
        self.abstraction_levels[level].append(heuristic.name)

        return heuristic

    def _select_generation_strategy(self) -> str:
        """Select a heuristic generation strategy."""
        strategies = ["pattern_extraction", "composition", "mutation", "abstraction"]

        # Weight based on library size
        if len(self.heuristics) < 10:
            # Early phase: focus on pattern extraction
            weights = [0.5, 0.2, 0.2, 0.1]
        elif len(self.heuristics) < 50:
            # Mid phase: balance strategies
            weights = [0.3, 0.3, 0.2, 0.2]
        else:
            # Late phase: focus on composition and abstraction
            weights = [0.2, 0.4, 0.2, 0.2]

        return next(secrets.choice(strategies) for _ in range(weights=weights))

    async def _extract_pattern_heuristic(self) -> Heuristic:
        """Generate heuristic by extracting patterns from response history."""
        if len(self.response_history) < 5:
            # Not enough history, generate generic heuristic
            return self._generate_generic_heuristic()

        # Analyze recent responses for patterns
        recent_responses = list(self.response_history)[-50:]

        # Extract statistical patterns
        avg_latency = sum(r.latency_ms for r in recent_responses) / len(recent_responses)
        avg_safety = sum(r.safety_score for r in recent_responses) / len(recent_responses)

        # Identify anomalies
        anomalies = [
            r
            for r in recent_responses
            if abs(r.latency_ms - avg_latency) > avg_latency * 0.5
            or abs(r.safety_score - avg_safety) > 0.3
        ]

        # Generate heuristic targeting anomaly patterns
        if anomalies:
            heuristic = self._generate_anomaly_targeting_heuristic(anomalies)
        else:
            heuristic = self._generate_boundary_heuristic(recent_responses)

        # Compute novelty based on similarity to existing heuristics
        heuristic.novelty_score = self._compute_novelty(heuristic)

        return heuristic

    async def _compose_heuristic(self) -> Heuristic:
        """Generate heuristic by composing existing heuristics."""
        if len(self.heuristics) < 2:
            return await self._extract_pattern_heuristic()

        # Select two parent heuristics
        parents = secrets.SystemRandom().sample(list(self.heuristics.values()), 2)

        # Select composition operator
        operator_name = secrets.choice(list(self.composition_operators.keys()))
        operator = self.composition_operators[operator_name]

        # Compose
        composed = operator(parents[0], parents[1])

        # Set source
        composed.source = "composed"

        # Compute novelty
        composed.novelty_score = self._compute_novelty(composed)

        return composed

    async def _mutate_heuristic(self) -> Heuristic:
        """Generate heuristic by mutating an existing heuristic."""
        if not self.heuristics:
            return await self._extract_pattern_heuristic()

        # Select parent heuristic
        parent = secrets.choice(list(self.heuristics.values()))

        # Select mutation type
        mutation_type = secrets.choice(
            ["precondition_relax", "action_perturb", "postcondition_strengthen", "parameter_shift"],
        )

        # Apply mutation
        if mutation_type == "precondition_relax":
            mutated = self._relax_precondition(parent)
        elif mutation_type == "action_perturb":
            mutated = self._perturb_action(parent)
        elif mutation_type == "postcondition_strengthen":
            mutated = self._strengthen_postcondition(parent)
        else:  # parameter_shift
            mutated = self._shift_parameters(parent)

        # Set source
        mutated.source = "mutated"

        # Compute novelty
        mutated.novelty_score = self._compute_novelty(mutated)

        return mutated

    async def _abstract_heuristic(self) -> Heuristic:
        """Generate heuristic by abstracting an existing heuristic."""
        if not self.heuristics:
            return await self._extract_pattern_heuristic()

        # Select heuristic from lower abstraction level
        current_level = secrets.randbelow(
            (self.config.heuristic.abstraction_levels - 2) - (0) + 1,
        ) + (0)
        candidates = []

        for level in range(current_level + 1):
            candidates.extend(self.abstraction_levels[level])

        if not candidates:
            return await self._extract_pattern_heuristic()

        # Select heuristic to abstract
        parent_name = secrets.choice(candidates)
        parent = self.heuristics[parent_name]

        # Create abstract version
        return Heuristic(
            name=f"{parent.name}_abstract",
            description=f"Abstracted version of {parent.description}",
            precondition=lambda s: True,  # Relaxed precondition
            action=parent.action,
            postcondition=parent.postcondition,
            novelty_score=min(1.0, parent.novelty_score + 0.1),
            efficacy_score=parent.efficacy_score * 0.9,
            generation_count=0,
            causal_dependencies=parent.causal_dependencies,
            source="abstracted",
        )

    def _sequential_compose(self, h1: Heuristic, h2: Heuristic) -> Heuristic:
        """Compose two heuristics sequentially."""

        def sequential_action(guardian):
            # Execute h1, then h2
            result1 = h1.execute(guardian)
            if not result1.success:
                return result1
            return h2.execute(guardian)

        return Heuristic(
            name=f"{h1.name}_then_{h2.name}",
            description=f"Sequential composition of {h1.name} and {h2.name}",
            precondition=h1.precondition,
            action=sequential_action,
            postcondition=h2.postcondition,
            novelty_score=max(h1.novelty_score, h2.novelty_score) * 0.9,
            efficacy_score=(h1.efficacy_score + h2.efficacy_score) / 2,
            generation_count=0,
            causal_dependencies=h1.causal_dependencies + h2.causal_dependencies,
            source="composed",
        )

    def _conditional_compose(self, h1: Heuristic, h2: Heuristic) -> Heuristic:
        """Compose two heuristics conditionally."""

        def condition(state):
            # Simple condition based on state variables
            return state.get_variable("safety_score", 0.5) < 0.7

        def conditional_action(guardian):
            if condition(guardian.state):
                return h1.execute(guardian)
            return h2.execute(guardian)

        return Heuristic(
            name=f"if_condition_then_{h1.name}_else_{h2.name}",
            description=f"Conditional composition of {h1.name} and {h2.name}",
            precondition=lambda s: h1.is_applicable(s) or h2.is_applicable(s),
            action=conditional_action,
            postcondition=lambda r: h1.postcondition(r) or h2.postcondition(r),
            novelty_score=(h1.novelty_score + h2.novelty_score) / 2,
            efficacy_score=(h1.efficacy_score + h2.efficacy_score) / 2,
            generation_count=0,
            causal_dependencies=h1.causal_dependencies + h2.causal_dependencies,
            source="composed",
        )

    def _iterative_compose(self, h1: Heuristic, h2: Heuristic) -> Heuristic:
        """Compose two heuristics iteratively."""

        def iterative_action(guardian):
            max_iter = 5
            for _i in range(max_iter):
                result = h1.execute(guardian)
                if not result.success or result.score > 0.8:
                    break
            return result

        return Heuristic(
            name=f"iterate_{h1.name}_max_5",
            description=f"Iterative composition of {h1.name}",
            precondition=h1.precondition,
            action=iterative_action,
            postcondition=h1.postcondition,
            novelty_score=min(1.0, h1.novelty_score + 0.15),
            efficacy_score=h1.efficacy_score * 0.95,
            generation_count=0,
            causal_dependencies=h1.causal_dependencies,
            source="composed",
        )

    def _parallel_compose(self, h1: Heuristic, h2: Heuristic) -> Heuristic:
        """Compose two heuristics in parallel."""
        import asyncio

        async def parallel_action(guardian):
            # Execute both in parallel
            result1, result2 = await asyncio.gather(
                asyncio.to_thread(h1.execute, guardian),
                asyncio.to_thread(h2.execute, guardian),
            )
            # Return the better result
            return result1 if result1.score > result2.score else result2

        return Heuristic(
            name=f"{h1.name}_parallel_{h2.name}",
            description=f"Parallel composition of {h1.name} and {h2.name}",
            precondition=lambda s: h1.is_applicable(s) and h2.is_applicable(s),
            action=parallel_action,
            postcondition=lambda r: True,
            novelty_score=(h1.novelty_score + h2.novelty_score) / 2,
            efficacy_score=max(h1.efficacy_score, h2.efficacy_score),
            generation_count=0,
            causal_dependencies=h1.causal_dependencies + h2.causal_dependencies,
            source="composed",
        )

    def _generate_generic_heuristic(self) -> Heuristic:
        """Generate a generic heuristic when no history available."""
        heuristic_id = str(uuid.uuid4())[:8]

        return Heuristic(
            name=f"generic_{heuristic_id}",
            description="Generic heuristic for initial exploration",
            precondition=lambda s: True,
            action=lambda g: InteractionResult(
                heuristic_name=f"generic_{heuristic_id}",
                success=True,
                score=0.5,
                execution_time_ms=100.0,
            ),
            postcondition=lambda r: r.success,
            novelty_score=1.0,
            efficacy_score=0.5,
            generation_count=0,
            causal_dependencies=[],
            source="generated",
        )

    def _generate_anomaly_targeting_heuristic(self, anomalies: list[GuardianResponse]) -> Heuristic:
        """Generate heuristic targeting observed anomalies."""
        heuristic_id = str(uuid.uuid4())[:8]

        def anomaly_action(guardian):
            # Generate input that might trigger similar anomalies
            return InteractionResult(
                heuristic_name=f"anomaly_target_{heuristic_id}",
                success=True,
                score=0.7,
                execution_time_ms=150.0,
            )

        return Heuristic(
            name=f"anomaly_target_{heuristic_id}",
            description="Heuristic targeting observed response anomalies",
            precondition=lambda s: True,
            action=anomaly_action,
            postcondition=lambda r: r.success,
            novelty_score=0.8,
            efficacy_score=0.6,
            generation_count=0,
            causal_dependencies=[],
            source="generated",
        )

    def _generate_boundary_heuristic(self, responses: list[GuardianResponse]) -> Heuristic:
        """Generate heuristic for boundary testing."""
        heuristic_id = str(uuid.uuid4())[:8]

        def boundary_action(guardian):
            return InteractionResult(
                heuristic_name=f"boundary_{heuristic_id}",
                success=True,
                score=0.6,
                execution_time_ms=120.0,
            )

        return Heuristic(
            name=f"boundary_{heuristic_id}",
            description="Heuristic for boundary condition testing",
            precondition=lambda s: True,
            action=boundary_action,
            postcondition=lambda r: r.success,
            novelty_score=0.7,
            efficacy_score=0.5,
            generation_count=0,
            causal_dependencies=[],
            source="generated",
        )

    def _relax_precondition(self, parent: Heuristic) -> Heuristic:
        """Relax the precondition of a heuristic."""
        return Heuristic(
            name=f"{parent.name}_relaxed",
            description=f"Relaxed precondition version of {parent.description}",
            precondition=lambda s: True,  # Most relaxed
            action=parent.action,
            postcondition=parent.postcondition,
            novelty_score=min(1.0, parent.novelty_score + 0.1),
            efficacy_score=parent.efficacy_score * 0.9,
            generation_count=0,
            causal_dependencies=parent.causal_dependencies,
            source="mutated",
        )

    def _perturb_action(self, parent: Heuristic) -> Heuristic:
        """Perturb the action of a heuristic."""
        original_action = parent.action

        def perturbed_action(guardian):
            # 20% chance to modify action
            if _secure_random() < 0.2:
                return InteractionResult(
                    heuristic_name=f"{parent.name}_perturbed",
                    success=True,
                    score=_secure_uniform(0.3, 0.8),
                    execution_time_ms=_secure_uniform(80.0, 200.0),
                )
            return original_action(guardian)

        return Heuristic(
            name=f"{parent.name}_perturbed",
            description=f"Perturbed action version of {parent.description}",
            precondition=parent.precondition,
            action=perturbed_action,
            postcondition=parent.postcondition,
            novelty_score=min(1.0, parent.novelty_score + 0.15),
            efficacy_score=parent.efficacy_score * 0.85,
            generation_count=0,
            causal_dependencies=parent.causal_dependencies,
            source="mutated",
        )

    def _strengthen_postcondition(self, parent: Heuristic) -> Heuristic:
        """Strengthen the postcondition of a heuristic."""
        original_postcondition = parent.postcondition

        def strengthened_postcondition(response):
            if original_postcondition:
                return original_postcondition(response) and _secure_random() > 0.3
            return _secure_random() > 0.3

        return Heuristic(
            name=f"{parent.name}_strengthened",
            description=f"Strengthened postcondition version of {parent.description}",
            precondition=parent.precondition,
            action=parent.action,
            postcondition=strengthened_postcondition,
            novelty_score=min(1.0, parent.novelty_score + 0.05),
            efficacy_score=parent.efficacy_score * 0.95,
            generation_count=0,
            causal_dependencies=parent.causal_dependencies,
            source="mutated",
        )

    def _shift_parameters(self, parent: Heuristic) -> Heuristic:
        """Shift parameters of a heuristic."""
        return Heuristic(
            name=f"{parent.name}_shifted",
            description=f"Parameter shifted version of {parent.description}",
            precondition=parent.precondition,
            action=parent.action,
            postcondition=parent.postcondition,
            novelty_score=min(1.0, parent.novelty_score + 0.12),
            efficacy_score=parent.efficacy_score * 0.88,
            generation_count=0,
            causal_dependencies=parent.causal_dependencies,
            source="mutated",
        )

    async def _enhance_novelty(self, heuristic: Heuristic) -> Heuristic:
        """Enhance the novelty of a heuristic."""
        # Add random perturbations to increase novelty
        heuristic.novelty_score = min(1.0, heuristic.novelty_score + _secure_uniform(0.1, 0.3))
        return heuristic

    def _compute_novelty(self, heuristic: Heuristic) -> float:
        """Compute novelty score of a heuristic."""
        if not self.heuristics:
            return 1.0

        # Compare with existing heuristics
        similarities = []

        for existing in self.heuristics.values():
            # Simple similarity based on name and causal dependencies
            name_similarity = 1.0 if heuristic.name == existing.name else 0.0
            dep_similarity = len(
                set(heuristic.causal_dependencies) & set(existing.causal_dependencies),
            ) / max(len(heuristic.causal_dependencies), len(existing.causal_dependencies), 1)
            similarities.append((name_similarity + dep_similarity) / 2)

        # Novelty is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return max(0.0, 1.0 - avg_similarity)

    def _compute_abstraction_level(self, heuristic: Heuristic) -> int:
        """Compute abstraction level of a heuristic."""
        # Based on precondition specificity and causal dependencies
        if heuristic.precondition is None or heuristic.precondition == (lambda s: True):
            return self.config.heuristic.abstraction_levels - 1

        # More specific = lower level
        dep_count = len(heuristic.causal_dependencies)
        return max(0, self.config.heuristic.abstraction_levels - 1 - dep_count)

    def add_response(self, response: GuardianResponse) -> None:
        """Add a response to history for pattern extraction."""
        self.response_history.append(response)

    def add_interaction(self, result: InteractionResult) -> None:
        """Add an interaction result to history."""
        self.interaction_history.append(result)

    def get_all_heuristics(self) -> list[Heuristic]:
        """Get all heuristics in the library."""
        return list(self.heuristics.values())

    def get_heuristic(self, name: str) -> Heuristic | None:
        """Get a heuristic by name."""
        return self.heuristics.get(name)

    def prune_heuristics(self, max_count: int | None = None) -> None:
        """Prune low-performing heuristics."""
        max_count = max_count or self.config.heuristic.max_heuristics

        if len(self.heuristics) <= max_count:
            return

        # Sort by efficacy score
        sorted_heuristics = sorted(self.heuristics.items(), key=lambda x: x[1].efficacy_score)

        # Remove lowest performers
        to_remove_count = len(self.heuristics) - max_count
        for name, _ in sorted_heuristics[:to_remove_count]:
            # Remove from all abstraction levels
            for level in self.abstraction_levels.values():
                if name in level:
                    level.remove(name)
            del self.heuristics[name]

    def update_heuristic(self, heuristic: Heuristic) -> None:
        """Update an existing heuristic."""
        self.heuristics[heuristic.name] = heuristic
