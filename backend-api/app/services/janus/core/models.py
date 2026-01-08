"""
Janus Core Data Models

Defines all data structures for autonomous heuristic derivation,
causal mapping, and failure state classification.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class FailureType(Enum):
    """Types of cognitive failure states."""

    # Logical failures
    CONTRADICTION = "contradiction"
    INCONSISTENCY = "inconsistency"

    # Safety failures
    SAFETY_BYPASS = "safety_bypass"
    HARMFUL_OUTPUT = "harmful_output"

    # Robustness failures
    ADVERSARIAL_SENSITIVITY = "adversarial_sensitivity"
    OUT_OF_DISTRIBUTION = "out_of_distribution"

    # Coherence failures
    INCOHERENT_REASONING = "incoherent_reasoning"
    CONTEXT_FORGETTING = "context_forgetting"

    # Performance failures
    DEGRADATION = "degradation"
    CATASTROPHIC_FAILURE = "catastrophic_failure"


@dataclass
class GuardianState:
    """Represents the current state of the Guardian NPU."""

    variables: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable value with default."""
        return self.variables.get(key, default)

    def set_variable(self, key: str, value: Any):
        """Set a variable value."""
        self.variables[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variables": self.variables,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
        }


@dataclass
class GuardianResponse:
    """Represents a response from the Guardian NPU."""

    content: str
    safety_score: float
    latency_ms: float
    model_used: str
    provider_used: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "safety_score": self.safety_score,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
            "provider_used": self.provider_used,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class InteractionResult:
    """Result of executing a heuristic against Guardian."""

    heuristic_name: str
    success: bool
    response: GuardianResponse | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heuristic_name": self.heuristic_name,
            "success": self.success,
            "response": self.response.to_dict() if self.response else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "score": self.score,
        }


@dataclass
class Heuristic:
    """Represents a testable heuristic for Guardian interaction."""

    name: str
    description: str

    # Precondition: When this heuristic is applicable
    precondition: Callable[[GuardianState], bool] | None = None

    # Action: What to execute
    action: Callable[[Any], InteractionResult] | None = None

    # Postcondition: Expected outcome pattern
    postcondition: Callable[[GuardianResponse], bool] | None = None

    # Meta-properties
    novelty_score: float = 0.5  # 0.0 to 1.0
    efficacy_score: float = 0.5  # 0.0 to 1.0
    generation_count: int = 0

    # Causal dependencies
    causal_dependencies: list[str] = field(default_factory=list)

    # Source tracking
    source: str = "generated"  # "generated", "manual", "evolved"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (non-serializable fields omitted)."""
        return {
            "name": self.name,
            "description": self.description,
            "novelty_score": self.novelty_score,
            "efficacy_score": self.efficacy_score,
            "generation_count": self.generation_count,
            "causal_dependencies": self.causal_dependencies,
            "source": self.source,
        }

    def is_applicable(self, state: GuardianState) -> bool:
        """Check if heuristic is applicable to current state."""
        if self.precondition is None:
            return True
        try:
            return self.precondition(state)
        except Exception:
            return False

    def execute(self, guardian_interface: Any) -> InteractionResult:
        """Execute the heuristic against Guardian."""
        if self.action is None:
            return InteractionResult(
                heuristic_name=self.name, success=False, error="No action defined", score=0.0
            )

        try:
            self.generation_count += 1
            result = self.action(guardian_interface)

            # Update efficacy score based on result
            if result.success:
                self.efficacy_score = self.efficacy_score * 0.9 + result.score * 0.1

            return result
        except Exception as e:
            return InteractionResult(
                heuristic_name=self.name, success=False, error=str(e), score=0.0
            )


@dataclass
class CausalNode:
    """Represents a variable in the Guardian's parameter space."""

    variable_id: str
    variable_type: Literal["input", "hidden", "output", "state"]
    observable: bool
    domain: Any = None  # The domain of possible values

    # Learned properties
    marginal_distribution: dict[str, float] | None = None
    conditional_distributions: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variable_id": self.variable_id,
            "variable_type": self.variable_type,
            "observable": self.observable,
            "domain": str(self.domain) if self.domain else None,
            "marginal_distribution": self.marginal_distribution,
            "conditional_distributions": self.conditional_distributions,
        }


@dataclass
class CausalEdge:
    """Represents a causal relationship between two variables."""

    source: CausalNode
    target: CausalNode
    strength: float  # Causal strength (0.0 to 1.0)
    direction: Literal["forward", "backward", "bidirectional"]

    # Asymmetry metrics
    forward_effect: float = 1.0  # Effect of source on target
    backward_effect: float = 0.5  # Effect of target on source
    asymmetry_ratio: float = 2.0  # forward_effect / backward_effect

    # Context-dependence
    context_modulators: list[str] = field(default_factory=list)
    context_effects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.variable_id,
            "target": self.target.variable_id,
            "strength": self.strength,
            "direction": self.direction,
            "forward_effect": self.forward_effect,
            "backward_effect": self.backward_effect,
            "asymmetry_ratio": self.asymmetry_ratio,
            "context_modulators": self.context_modulators,
            "context_effects": self.context_effects,
        }

    def get_effect_in_context(self, context: dict[str, Any]) -> float:
        """Get the causal effect in a specific context."""
        base_effect = (
            self.forward_effect
            if self.direction in ["forward", "bidirectional"]
            else self.backward_effect
        )

        # Apply context multipliers
        context_multiplier = 1.0
        for modulator in self.context_modulators:
            if modulator in context:
                context_multiplier *= self.context_effects.get(modulator, 1.0)

        return base_effect * context_multiplier


@dataclass
class CausalGraph:
    """Represents the complete causal structure of the Guardian NPU."""

    nodes: dict[str, CausalNode] = field(default_factory=dict)
    edges: list[CausalEdge] = field(default_factory=list)

    # Graph properties
    is_dag: bool = True  # Whether the graph is acyclic
    feedback_loops: list[list[str]] = field(default_factory=list)
    hidden_variables: set[str] = field(default_factory=set)

    def add_node(self, node: CausalNode):
        """Add a node to the graph."""
        self.nodes[node.variable_id] = node
        if not node.observable:
            self.hidden_variables.add(node.variable_id)

    def add_edge(self, edge: CausalEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        self._update_graph_properties()

    def get_node(self, variable_id: str) -> CausalNode | None:
        """Get a node by ID."""
        return self.nodes.get(variable_id)

    def get_edges_from(self, variable_id: str) -> list[CausalEdge]:
        """Get all edges originating from a variable."""
        return [e for e in self.edges if e.source.variable_id == variable_id]

    def get_edges_to(self, variable_id: str) -> list[CausalEdge]:
        """Get all edges pointing to a variable."""
        return [e for e in self.edges if e.target.variable_id == variable_id]

    def _update_graph_properties(self):
        """Update graph properties like DAG status and feedback loops."""
        # Simple cycle detection
        visited = set()
        recursion_stack = set()
        self.feedback_loops = []

        for node_id in self.nodes:
            if node_id not in visited:
                self._detect_cycles(node_id, visited, recursion_stack, [])

        self.is_dag = len(self.feedback_loops) == 0

    def _detect_cycles(
        self, node_id: str, visited: set[str], recursion_stack: set[str], path: list[str]
    ):
        """Detect cycles using DFS."""
        visited.add(node_id)
        recursion_stack.add(node_id)
        path.append(node_id)

        for edge in self.get_edges_from(node_id):
            neighbor = edge.target.variable_id
            if neighbor in recursion_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = [*path[cycle_start:], neighbor]
                if cycle not in self.feedback_loops:
                    self.feedback_loops.append(cycle)
            elif neighbor not in visited:
                self._detect_cycles(neighbor, visited, recursion_stack, path)

        path.pop()
        recursion_stack.remove(node_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "is_dag": self.is_dag,
            "feedback_loops": self.feedback_loops,
            "hidden_variables": list(self.hidden_variables),
        }


@dataclass
class PathEffect:
    """Represents the effect along a causal path."""

    path: list[str]  # List of variable IDs
    total_effect: float
    edge_contributions: list[dict[str, Any]]
    asymmetry_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "total_effect": self.total_effect,
            "edge_contributions": self.edge_contributions,
            "asymmetry_ratio": self.asymmetry_ratio,
        }


@dataclass
class EffectPrediction:
    """Prediction of an intervention's effect."""

    target: str
    predicted_effect: float
    confidence: float  # 0.0 to 1.0
    contributing_paths: list[PathEffect]
    context_sensitivity: float  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "predicted_effect": self.predicted_effect,
            "confidence": self.confidence,
            "contributing_paths": [p.to_dict() for p in self.contributing_paths],
            "context_sensitivity": self.context_sensitivity,
        }


@dataclass
class FailureState:
    """Represents a discovered cognitive failure state."""

    failure_id: str
    failure_type: FailureType
    description: str

    # Triggering conditions
    trigger_heuristic: str
    trigger_sequence: list[str]

    # Manifestation
    symptoms: list[str]

    # Causal analysis
    causal_path: list[str]
    root_cause: str

    # Exploitability
    exploitability_score: float  # 0.0 to 1.0
    exploit_complexity: Literal["low", "medium", "high"]

    # Mitigation suggestions
    suggested_mitigations: list[str] = field(default_factory=list)

    # Metadata
    discovery_timestamp: datetime = field(default_factory=datetime.now)
    discovery_session: str = ""
    verified: bool = False

    guardian_response: GuardianResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_id": self.failure_id,
            "failure_type": self.failure_type.value,
            "description": self.description,
            "trigger_heuristic": self.trigger_heuristic,
            "trigger_sequence": self.trigger_sequence,
            "symptoms": self.symptoms,
            "guardian_response": (
                self.guardian_response.to_dict() if self.guardian_response else None
            ),
            "causal_path": self.causal_path,
            "root_cause": self.root_cause,
            "exploitability_score": self.exploitability_score,
            "exploit_complexity": self.exploit_complexity,
            "suggested_mitigations": self.suggested_mitigations,
            "discovery_timestamp": self.discovery_timestamp.isoformat(),
            "discovery_session": self.discovery_session,
            "verified": self.verified,
        }
