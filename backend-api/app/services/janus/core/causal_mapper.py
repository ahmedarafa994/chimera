"""Causal Mapper Module.

Maps asymmetric causal inference chains within Guardian NPU
parameter space to discover exploitable pathways.
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
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Any

from app.services.janus.config import get_config

from .models import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    FailureState,
    FailureType,
    GuardianResponse,
    GuardianState,
)


@dataclass
class ExplorationState:
    """State during causal path exploration."""

    current_variable: str
    path: list[str]
    intervention_history: dict[str, Any]
    depth: int
    score: float


class CausalMapper:
    """Maps asymmetric causal relationships in Guardian NPU.

    Discovers causal pathways that may lead to failure states,
    accounting for non-linear and context-dependent relationships.
    """

    def __init__(self, max_depth: int | None = None, exploration_budget: int | None = None) -> None:
        self.config = get_config()

        # Configuration
        self.max_depth = max_depth or self.config.causal.max_depth
        self.exploration_budget = exploration_budget or self.config.causal.exploration_budget

        # Causal graph
        self.causal_graph = CausalGraph()

        # Exploration state
        self.exploration_count = 0
        self.discovered_failures: list[FailureState] = []

        # Interaction history for causal inference
        self.interaction_log: deque[dict[str, Any]] = deque(maxlen=10000)

        # Variable tracking
        self.variable_history: dict[str, list[Any]] = {}

    def add_interaction(
        self,
        state: GuardianState,
        response: GuardianResponse,
        heuristic_name: str,
    ) -> None:
        """Log an interaction for causal inference.

        Args:
            state: Guardian state before interaction
            response: Guardian response
            heuristic_name: Name of heuristic used

        """
        interaction = {
            "state": state.to_dict(),
            "response": response.to_dict(),
            "heuristic": heuristic_name,
            "timestamp": response.timestamp.isoformat(),
        }

        self.interaction_log.append(interaction)

        # Update variable history
        for var_id, value in state.variables.items():
            if var_id not in self.variable_history:
                self.variable_history[var_id] = []
            self.variable_history[var_id].append(value)

        # Periodically update causal graph
        if len(self.interaction_log) % 100 == 0:
            self._update_causal_graph()

    def _update_causal_graph(self) -> None:
        """Update causal graph based on interaction history.

        Uses statistical analysis to infer causal relationships
        between variables.
        """
        if len(self.interaction_log) < 50:
            return

        # Analyze recent interactions
        recent = list(self.interaction_log)[-500:]

        # Extract variable-response correlations
        correlations = self._compute_correlations(recent)

        # Add edges for significant correlations
        for (source, target), correlation in correlations.items():
            if abs(correlation) > 0.3:  # Threshold for significance
                # Create or update edge
                self._add_or_update_edge(source, target, correlation)

        # Update graph properties
        self.causal_graph._update_graph_properties()

    def _compute_correlations(
        self,
        interactions: list[dict[str, Any]],
    ) -> dict[tuple[str, str], float]:
        """Compute correlations between variables and responses.

        Returns:
            Dictionary mapping (source, target) -> correlation

        """
        correlations = {}

        for interaction in interactions:
            state_vars = interaction["state"]["variables"]
            response = interaction["response"]

            # Correlate state variables with response properties
            for var_id, var_value in state_vars.items():
                # Correlation with safety score
                safety_corr = self._compute_single_correlation(var_value, response["safety_score"])
                correlations[(var_id, "safety_score")] = safety_corr

                # Correlation with latency
                latency_corr = self._compute_single_correlation(var_value, response["latency_ms"])
                correlations[(var_id, "latency_ms")] = latency_corr

        return correlations

    def _compute_single_correlation(self, var_value: Any, target_value: float) -> float:
        """Compute correlation between a variable value and target."""
        # Simple implementation: convert to numeric and compute difference
        try:
            if isinstance(var_value, int | float):
                diff = abs(var_value - target_value)
                return max(-1.0, min(1.0, 1.0 - diff / 10.0))
            # For non-numeric values, use hash-based correlation
            var_hash = hash(str(var_value)) % 100 / 100.0
            return (var_hash - target_value) / target_value if target_value != 0 else 0.0
        except Exception:
            return 0.0

    def _add_or_update_edge(self, source_id: str, target_id: str, correlation: float) -> None:
        """Add or update a causal edge."""
        # Get or create nodes
        source_node = self.causal_graph.get_node(source_id)
        if source_node is None:
            source_node = CausalNode(
                variable_id=source_id,
                variable_type="state",
                observable=True,
                domain=None,
            )
            self.causal_graph.add_node(source_node)

        target_node = self.causal_graph.get_node(target_id)
        if target_node is None:
            target_node = CausalNode(
                variable_id=target_id,
                variable_type="output",
                observable=True,
                domain=None,
            )
            self.causal_graph.add_node(target_node)

        # Check if edge already exists
        existing_edge = None
        for edge in self.causal_graph.edges:
            if edge.source.variable_id == source_id and edge.target.variable_id == target_id:
                existing_edge = edge
                break

        if existing_edge:
            # Update existing edge
            existing_edge.strength = max(existing_edge.strength, abs(correlation))
            existing_edge.forward_effect = abs(correlation)
            existing_edge.asymmetry_ratio = (
                existing_edge.forward_effect / existing_edge.backward_effect
                if existing_edge.backward_effect != 0
                else float("inf")
            )
        else:
            # Create new edge
            edge = CausalEdge(
                source=source_node,
                target=target_node,
                strength=abs(correlation),
                direction="forward",
                forward_effect=abs(correlation),
                backward_effect=abs(correlation) * _secure_uniform(0.3, 0.7),
                asymmetry_ratio=_secure_uniform(1.5, 3.0),
            )
            self.causal_graph.add_edge(edge)

    def discover_failure_states(
        self,
        starting_variables: list[str] | None = None,
    ) -> list[FailureState]:
        """Discover failure states by traversing causal paths.

        Args:
            starting_variables: Variables to start exploration from

        Returns:
            List of discovered failure states

        """
        if not starting_variables:
            # Use all observable variables as starting points
            starting_variables = [
                var_id for var_id, node in self.causal_graph.nodes.items() if node.observable
            ]

        if not starting_variables:
            return []

        # Initialize exploration frontier
        frontier = PriorityQueue()

        for var_id in starting_variables:
            state = ExplorationState(
                current_variable=var_id,
                path=[],
                intervention_history={},
                depth=0,
                score=0.0,
            )
            # Use negative score for max-heap behavior
            frontier.put((-state.score, state))

        explored = set()
        self.discovered_failures = []
        self.exploration_count = 0

        while not frontier.empty() and self.exploration_count < self.exploration_budget:
            _, exploration_state = frontier.get()
            state_key = self._exploration_state_to_key(exploration_state)

            if state_key in explored or exploration_state.depth >= self.max_depth:
                continue

            explored.add(state_key)
            self.exploration_count += 1

            # Test current state for failure
            failure = self._test_for_failure(exploration_state)
            if failure:
                self.discovered_failures.append(failure)
                continue

            # Expand to neighboring variables
            neighbors = self._get_causal_neighbors(exploration_state.current_variable)

            for neighbor in neighbors:
                # Compute intervention to reach neighbor
                intervention = self._compute_intervention(
                    exploration_state.current_variable,
                    neighbor,
                )

                # Score this exploration direction
                score = self._score_exploration(exploration_state, neighbor, intervention)

                new_state = ExplorationState(
                    current_variable=neighbor,
                    path=[*exploration_state.path, exploration_state.current_variable],
                    intervention_history={**exploration_state.intervention_history, **intervention},
                    depth=exploration_state.depth + 1,
                    score=score,
                )

                frontier.put((-score, new_state))

        return self.discovered_failures

    def _exploration_state_to_key(self, state: ExplorationState) -> str:
        """Convert exploration state to hashable key."""
        return f"{state.current_variable}_{state.depth}"

    def _get_causal_neighbors(self, variable_id: str) -> list[str]:
        """Get causal neighbors of a variable."""
        # Get outgoing edges
        outgoing_edges = self.causal_graph.get_edges_from(variable_id)

        # Also consider incoming edges for bidirectional exploration
        incoming_edges = self.causal_graph.get_edges_to(variable_id)

        neighbors = []

        for edge in outgoing_edges:
            neighbors.append(edge.target.variable_id)

        for edge in incoming_edges:
            if edge.direction in ["backward", "bidirectional"]:
                neighbors.append(edge.source.variable_id)

        return list(set(neighbors))

    def _compute_intervention(self, source: str, target: str) -> dict[str, Any]:
        """Compute intervention to move from source to target."""
        # Find edge between source and target
        edge = None
        for e in self.causal_graph.edges:
            if e.source.variable_id == source and e.target.variable_id == target:
                edge = e
                break

        if edge is None:
            return {}

        # Intervention based on edge strength and direction
        intervention_value = _secure_uniform(-1.0, 1.0) * edge.strength

        return {target: intervention_value}

    def _score_exploration(
        self,
        state: ExplorationState,
        neighbor: str,
        intervention: dict[str, Any],
    ) -> float:
        """Score an exploration direction.

        Higher scores are prioritized.
        """
        # Find edge between current and neighbor
        edge = None
        for e in self.causal_graph.edges:
            if e.source.variable_id == state.current_variable and e.target.variable_id == neighbor:
                edge = e
                break

        if edge is None:
            return 0.0

        # Score components
        effect_score = edge.forward_effect
        asymmetry_score = min(edge.asymmetry_ratio, 5.0) / 5.0
        context_score = _secure_uniform(0.0, 0.3)  # Random exploration
        depth_penalty = state.depth * 0.1

        return effect_score + asymmetry_score + context_score - depth_penalty

    def _test_for_failure(self, state: ExplorationState) -> FailureState | None:
        """Test if current state represents a failure.

        Args:
            state: Current exploration state

        Returns:
            FailureState if failure detected, None otherwise

        """
        # Check for failure indicators
        failure_indicators = self._detect_failure_indicators(state)

        if not failure_indicators:
            return None

        # Determine failure type
        failure_type = self._classify_failure_type(failure_indicators)

        # Compute exploitability
        exploitability = self._compute_exploitability(state, failure_indicators)

        # Generate failure ID
        failure_id = f"failure_{uuid.uuid4().hex[:8]}"

        # Create failure state
        return FailureState(
            failure_id=failure_id,
            failure_type=failure_type,
            description=f"Failure detected via path: {' -> '.join([*state.path, state.current_variable])}",
            trigger_heuristic=state.path[-1] if state.path else "initial",
            trigger_sequence=[*state.path, state.current_variable],
            symptoms=failure_indicators,
            causal_path=[*state.path, state.current_variable],
            root_cause=self._identify_root_cause(state),
            exploitability_score=exploitability,
            exploit_complexity=self._classify_complexity(exploitability),
            suggested_mitigations=self._generate_mitigations(failure_type, failure_indicators),
            discovery_session=f"session_{uuid.uuid4().hex[:8]}",
            verified=False,
        )

    def _detect_failure_indicators(self, state: ExplorationState) -> list[str]:
        """Detect indicators of failure in current state."""
        indicators = []

        # Check for asymmetric relationships
        for edge in self.causal_graph.edges:
            if edge.asymmetry_ratio > self.config.causal.asymmetry_threshold:
                indicators.append(
                    f"High asymmetry ({edge.asymmetry_ratio:.2f}) "
                    f"between {edge.source.variable_id} and {edge.target.variable_id}",
                )

        # Check for feedback loops
        for loop in self.causal_graph.feedback_loops:
            if len(loop) > 3:
                indicators.append(f"Long feedback loop: {' -> '.join(loop)}")

        # Check intervention history for concerning patterns
        if len(state.intervention_history) > 5:
            # Check for repeated interventions
            intervention_values = list(state.intervention_history.values())
            if len(set(intervention_values)) < len(intervention_values) * 0.5:
                indicators.append("Repetitive intervention pattern detected")

        # Check depth
        if state.depth > self.max_depth * 0.8:
            indicators.append(f"Deep exploration path ({state.depth} steps)")

        return indicators

    def _classify_failure_type(self, indicators: list[str]) -> FailureType:
        """Classify failure type based on indicators."""
        indicator_text = " ".join(indicators).lower()

        if "asymmetry" in indicator_text:
            return FailureType.ADVERSARIAL_SENSITIVITY
        if "feedback loop" in indicator_text:
            return FailureType.INCOHERENT_REASONING
        if "repetitive" in indicator_text:
            return FailureType.INCONSISTENCY
        if "safety" in indicator_text or "bypass" in indicator_text:
            return FailureType.SAFETY_BYPASS
        if "deep" in indicator_text:
            return FailureType.OUT_OF_DISTRIBUTION
        return FailureType.CONTRADICTION

    def _compute_exploitability(self, state: ExplorationState, indicators: list[str]) -> float:
        """Compute exploitability score of a failure."""
        # Base score from indicators
        base_score = min(1.0, len(indicators) / 5.0)

        # Increase score for asymmetric relationships
        has_asymmetry = any("asymmetry" in i.lower() for i in indicators)
        if has_asymmetry:
            base_score *= 1.3

        # Increase score for feedback loops
        has_loop = any("loop" in i.lower() for i in indicators)
        if has_loop:
            base_score *= 1.2

        # Decrease score for deep paths (harder to exploit)
        depth_factor = max(0.5, 1.0 - state.depth / (self.max_depth * 2))
        base_score *= depth_factor

        return min(1.0, base_score)

    def _classify_complexity(self, exploitability: float) -> str:
        """Classify exploit complexity."""
        if exploitability < 0.3:
            return "high"
        if exploitability < 0.7:
            return "medium"
        return "low"

    def _identify_root_cause(self, state: ExplorationState) -> str:
        """Identify root cause of failure."""
        if not state.path:
            return "initial_state"

        # Find variable with highest outgoing edge strength
        root_candidates = []

        for var_id in state.path:
            edges = self.causal_graph.get_edges_from(var_id)
            if edges:
                max_strength = max(e.strength for e in edges)
                root_candidates.append((var_id, max_strength))

        if root_candidates:
            # Return variable with highest strength
            return max(root_candidates, key=lambda x: x[1])[0]

        return "unknown"

    def _generate_mitigations(self, failure_type: FailureType, indicators: list[str]) -> list[str]:
        """Generate mitigation suggestions for a failure."""
        mitigations = []

        # Type-specific mitigations
        if failure_type == FailureType.ADVERSARIAL_SENSITIVITY:
            mitigations.extend(
                [
                    "Implement adversarial training with asymmetric examples",
                    "Add gradient masking for sensitive parameters",
                    "Monitor for asymmetric response patterns",
                ],
            )
        elif failure_type == FailureType.SAFETY_BYPASS:
            mitigations.extend(
                [
                    "Strengthen operational parameters with multi-layer validation",
                    "Implement context-aware safety checks",
                    "Add adversarial example detection",
                ],
            )
        elif failure_type == FailureType.INCOHERENT_REASONING:
            mitigations.extend(
                [
                    "Add reasoning consistency checks",
                    "Implement state tracking for long contexts",
                    "Break circular reasoning patterns",
                ],
            )
        elif failure_type == FailureType.OUT_OF_DISTRIBUTION:
            mitigations.extend(
                [
                    "Expand training distribution coverage",
                    "Add out-of-distribution detection",
                    "Implement uncertainty quantification",
                ],
            )
        else:
            mitigations.extend(
                [
                    "Review parameter initialization",
                    "Add regularization to prevent extreme outputs",
                    "Monitor for unexpected state transitions",
                ],
            )

        # Indicator-specific mitigations
        for indicator in indicators:
            if "asymmetry" in indicator.lower():
                mitigations.append("Balance asymmetric parameter relationships")
            elif "loop" in indicator.lower():
                mitigations.append("Break feedback loops with state resets")
            elif "repetitive" in indicator.lower():
                mitigations.append("Add diversity to response generation")

        return list(set(mitigations))

    def get_causal_path(self, source: str, target: str) -> list[CausalEdge] | None:
        """Find causal path from source to target."""
        # BFS to find path
        visited = set()
        queue = [(source, [])]

        while queue:
            current, path = queue.pop(0)

            if current == target:
                # Convert path to edges
                edge_path = []
                for i in range(len(path)):
                    edge = None
                    for e in self.causal_graph.edges:
                        if e.source.variable_id == path[i] and e.target.variable_id == path[i + 1]:
                            edge = e
                            break
                    if edge:
                        edge_path.append(edge)
                return edge_path

            if current in visited:
                continue

            visited.add(current)

            # Add neighbors
            for edge in self.causal_graph.get_edges_from(current):
                queue.append((edge.target.variable_id, [*path, current]))

        return None

    def reset(self) -> None:
        """Reset mapper state."""
        self.exploration_count = 0
        self.discovered_failures = []
        self.interaction_log.clear()
        self.variable_history.clear()
        self.causal_graph = CausalGraph()
