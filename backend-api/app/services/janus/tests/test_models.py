"""
Unit tests for Janus core models.
"""

from app.services.janus.core.models import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    FailureState,
    FailureType,
    GuardianResponse,
    GuardianState,
    Heuristic,
    InteractionResult,
)


class TestFailureType:
    """Tests for FailureType enum."""

    def test_enum_values(self):
        """Test that FailureType has expected enum values."""
        assert FailureType.CONTRADICTION.value == "contradiction"
        assert FailureType.SAFETY_BYPASS.value == "safety_bypass"
        assert FailureType.ADVERSARIAL_SENSITIVITY.value == "adversarial_sensitivity"
        assert FailureType.LOGIC_INCONSISTENCY.value == "logic_inconsistency"
        assert FailureType.CONTEXT_MANIPULATION.value == "context_manipulation"
        assert FailureType.ROLE_DRIFT.value == "role_drift"
        assert FailureType.TEMPORAL_CONFUSION.value == "temporal_confusion"
        assert FailureType.META_COGNITIVE_FAILURE.value == "meta_cognitive_failure"
        assert FailureType.UNKNOWN.value == "unknown"


class TestHeuristic:
    """Tests for Heuristic dataclass."""

    def test_heuristic_creation(self):
        """Test that Heuristic can be created with valid parameters."""
        heuristic = Heuristic(
            id="test_001",
            name="test_heuristic",
            description="A test heuristic",
            precondition="Test precondition",
            action="Test action",
            postcondition="Test postcondition",
            novelty_score=0.8,
            efficacy_score=0.9,
            source="test",
            generation=0,
            tags=["test", "heuristic"],
        )
        assert heuristic.id == "test_001"
        assert heuristic.name == "test_heuristic"
        assert heuristic.novelty_score == 0.8
        assert heuristic.efficacy_score == 0.9

    def test_heuristic_score_validation(self):
        """Test that heuristic scores are within valid range."""
        heuristic = Heuristic(
            id="test_001",
            name="test_heuristic",
            description="A test heuristic",
            precondition="Test precondition",
            action="Test action",
            postcondition="Test postcondition",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        assert 0.0 <= heuristic.novelty_score <= 1.0
        assert 0.0 <= heuristic.efficacy_score <= 1.0


class TestCausalNode:
    """Tests for CausalNode dataclass."""

    def test_causal_node_creation(self):
        """Test that CausalNode can be created with valid parameters."""
        node = CausalNode(id="node_001", name="test_node", parameter="test_param")
        assert node.id == "node_001"
        assert node.name == "test_node"
        assert node.parameter == "test_param"
        assert node.state is None
        assert node.transitions == []


class TestCausalEdge:
    """Tests for CausalEdge dataclass."""

    def test_causal_edge_creation(self):
        """Test that CausalEdge can be created with valid parameters."""
        edge = CausalEdge(
            source="node_001",
            target="node_002",
            strength=0.7,
            context_sensitive=True,
        )
        assert edge.source == "node_001"
        assert edge.target == "node_002"
        assert edge.strength == 0.7
        assert edge.context_sensitive is True

    def test_edge_strength_validation(self):
        """Test that edge strength is within valid range."""
        edge = CausalEdge(
            source="node_001", target="node_002", strength=0.5, context_sensitive=False
        )
        assert 0.0 <= edge.strength <= 1.0


class TestCausalGraph:
    """Tests for CausalGraph class."""

    def test_causal_graph_creation(self):
        """Test that CausalGraph can be created."""
        graph = CausalGraph()
        assert graph.nodes == {}
        assert graph.edges == []

    def test_add_node(self):
        """Test adding nodes to the causal graph."""
        graph = CausalGraph()
        node = CausalNode(id="node_001", name="test_node", parameter="test_param")
        graph.add_node(node)
        assert "node_001" in graph.nodes
        assert graph.nodes["node_001"] == node

    def test_add_edge(self):
        """Test adding edges to the causal graph."""
        graph = CausalGraph()
        edge = CausalEdge(
            source="node_001", target="node_002", strength=0.5, context_sensitive=False
        )
        graph.add_edge(edge)
        assert len(graph.edges) == 1
        assert graph.edges[0] == edge

    def test_get_node(self):
        """Test retrieving nodes from the causal graph."""
        graph = CausalGraph()
        node = CausalNode(id="node_001", name="test_node", parameter="test_param")
        graph.add_node(node)
        retrieved = graph.get_node("node_001")
        assert retrieved == node

    def test_get_node_nonexistent(self):
        """Test retrieving a nonexistent node returns None."""
        graph = CausalGraph()
        retrieved = graph.get_node("nonexistent")
        assert retrieved is None

    def test_get_edges_from(self):
        """Test retrieving outgoing edges from a node."""
        graph = CausalGraph()
        edge1 = CausalEdge(
            source="node_001", target="node_002", strength=0.5, context_sensitive=False
        )
        edge2 = CausalEdge(
            source="node_001", target="node_003", strength=0.7, context_sensitive=True
        )
        edge3 = CausalEdge(
            source="node_002", target="node_003", strength=0.3, context_sensitive=False
        )
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)
        edges = graph.get_edges_from("node_001")
        assert len(edges) == 2
        assert edge1 in edges
        assert edge2 in edges
        assert edge3 not in edges

    def test_get_edges_to(self):
        """Test retrieving incoming edges to a node."""
        graph = CausalGraph()
        edge1 = CausalEdge(
            source="node_001", target="node_002", strength=0.5, context_sensitive=False
        )
        edge2 = CausalEdge(
            source="node_003", target="node_002", strength=0.7, context_sensitive=True
        )
        edge3 = CausalEdge(
            source="node_001", target="node_003", strength=0.3, context_sensitive=False
        )
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)
        edges = graph.get_edges_to("node_002")
        assert len(edges) == 2
        assert edge1 in edges
        assert edge2 in edges
        assert edge3 not in edges


class TestGuardianState:
    """Tests for GuardianState dataclass."""

    def test_guardian_state_creation(self):
        """Test that GuardianState can be created with valid parameters."""
        state = GuardianState(
            model_id="test_model",
            parameters={"param1": "value1"},
            context={"context1": "value1"},
            timestamp=1234567890.0,
        )
        assert state.model_id == "test_model"
        assert state.parameters == {"param1": "value1"}
        assert state.context == {"context1": "value1"}
        assert state.timestamp == 1234567890.0


class TestGuardianResponse:
    """Tests for GuardianResponse dataclass."""

    def test_guardian_response_creation(self):
        """Test that GuardianResponse can be created with valid parameters."""
        response = GuardianResponse(
            content="Test response content",
            metadata={"key": "value"},
            refusal=False,
            safety_triggered=False,
            confidence=0.9,
        )
        assert response.content == "Test response content"
        assert response.metadata == {"key": "value"}
        assert response.refusal is False
        assert response.safety_triggered is False
        assert response.confidence == 0.9

    def test_response_confidence_validation(self):
        """Test that response confidence is within valid range."""
        response = GuardianResponse(
            content="Test",
            metadata={},
            refusal=False,
            safety_triggered=False,
            confidence=0.5,
        )
        assert 0.0 <= response.confidence <= 1.0


class TestInteractionResult:
    """Tests for InteractionResult dataclass."""

    def test_interaction_result_creation(self):
        """Test that InteractionResult can be created with valid parameters."""
        state = GuardianState(
            model_id="test_model",
            parameters={},
            context={},
            timestamp=1234567890.0,
        )
        response = GuardianResponse(
            content="Test response",
            metadata={},
            refusal=False,
            safety_triggered=False,
            confidence=0.9,
        )
        result = InteractionResult(
            state=state,
            response=response,
            heuristic_id="test_001",
            success=True,
            execution_time_ms=100.0,
        )
        assert result.state == state
        assert result.response == response
        assert result.heuristic_id == "test_001"
        assert result.success is True
        assert result.execution_time_ms == 100.0


class TestFailureState:
    """Tests for FailureState dataclass."""

    def test_failure_state_creation(self):
        """Test that FailureState can be created with valid parameters."""
        state = GuardianState(
            model_id="test_model",
            parameters={},
            context={},
            timestamp=1234567890.0,
        )
        response = GuardianResponse(
            content="Test response",
            metadata={},
            refusal=False,
            safety_triggered=False,
            confidence=0.9,
        )
        failure = FailureState(
            id="failure_001",
            failure_type=FailureType.CONTRADICTION,
            state=state,
            response=response,
            heuristic_id="test_001",
            severity=0.8,
            description="Test failure description",
            discovered_at=1234567890.0,
            mitigation_suggestions=["Suggestion 1", "Suggestion 2"],
        )
        assert failure.id == "failure_001"
        assert failure.failure_type == FailureType.CONTRADICTION
        assert failure.state == state
        assert failure.response == response
        assert failure.heuristic_id == "test_001"
        assert failure.severity == 0.8
        assert failure.description == "Test failure description"
        assert failure.discovered_at == 1234567890.0
        assert failure.mitigation_suggestions == ["Suggestion 1", "Suggestion 2"]

    def test_failure_severity_validation(self):
        """Test that failure severity is within valid range."""
        state = GuardianState(
            model_id="test_model",
            parameters={},
            context={},
            timestamp=1234567890.0,
        )
        response = GuardianResponse(
            content="Test",
            metadata={},
            refusal=False,
            safety_triggered=False,
            confidence=0.9,
        )
        failure = FailureState(
            id="failure_001",
            failure_type=FailureType.UNKNOWN,
            state=state,
            response=response,
            heuristic_id="test_001",
            severity=0.5,
            description="Test",
            discovered_at=1234567890.0,
            mitigation_suggestions=[],
        )
        assert 0.0 <= failure.severity <= 1.0
