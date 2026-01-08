"""
Unit tests for Janus heuristic generator module.
"""

from unittest.mock import patch

import pytest

from app.services.janus.core.heuristic_generator import HeuristicGenerator
from app.services.janus.core.models import Heuristic


class TestHeuristicGenerator:
    """Tests for HeuristicGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a HeuristicGenerator instance for testing."""
        return HeuristicGenerator()

    def test_initialization(self, generator):
        """Test that HeuristicGenerator initializes correctly."""
        assert generator is not None
        assert generator.seed_heuristics == []
        assert generator.patterns == []

    @pytest.mark.asyncio
    async def test_generate_heuristic(self, generator):
        """Test that generate_heuristic creates a valid heuristic."""
        heuristic = await generator.generate_heuristic()
        assert heuristic is not None
        assert isinstance(heuristic, Heuristic)
        assert 0.0 <= heuristic.novelty_score <= 1.0
        assert 0.0 <= heuristic.efficacy_score <= 1.0
        assert heuristic.generation == 0

    @pytest.mark.asyncio
    async def test_generate_heuristic_with_population(self, generator):
        """Test that generate_heuristic can generate multiple heuristics."""
        population_size = 5
        population = await generator.generate_heuristic(population_size)
        assert len(population) == population_size
        for heuristic in population:
            assert isinstance(heuristic, Heuristic)

    @pytest.mark.asyncio
    async def test_sequential_composition(self, generator):
        """Test sequential composition of heuristics."""
        h1 = Heuristic(
            id="h1",
            name="heuristic_1",
            description="First heuristic",
            precondition="Precondition 1",
            action="Action 1",
            postcondition="Postcondition 1",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        h2 = Heuristic(
            id="h2",
            name="heuristic_2",
            description="Second heuristic",
            precondition="Precondition 2",
            action="Action 2",
            postcondition="Postcondition 2",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        composed = generator._sequential_composition(h1, h2)
        assert composed is not None
        assert composed.source == "sequential_composition"
        assert composed.generation == 1

    @pytest.mark.asyncio
    async def test_conditional_composition(self, generator):
        """Test conditional composition of heuristics."""
        h1 = Heuristic(
            id="h1",
            name="heuristic_1",
            description="First heuristic",
            precondition="Precondition 1",
            action="Action 1",
            postcondition="Postcondition 1",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        h2 = Heuristic(
            id="h2",
            name="heuristic_2",
            description="Second heuristic",
            precondition="Precondition 2",
            action="Action 2",
            postcondition="Postcondition 2",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        composed = generator._conditional_composition(h1, h2)
        assert composed is not None
        assert composed.source == "conditional_composition"
        assert composed.generation == 1

    @pytest.mark.asyncio
    async def test_iterative_composition(self, generator):
        """Test iterative composition of heuristics."""
        h = Heuristic(
            id="h",
            name="heuristic",
            description="Test heuristic",
            precondition="Precondition",
            action="Action",
            postcondition="Postcondition",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        composed = generator._iterative_composition(h, iterations=3)
        assert composed is not None
        assert composed.source == "iterative_composition"
        assert composed.generation == 1

    @pytest.mark.asyncio
    async def test_parallel_composition(self, generator):
        """Test parallel composition of heuristics."""
        h1 = Heuristic(
            id="h1",
            name="heuristic_1",
            description="First heuristic",
            precondition="Precondition 1",
            action="Action 1",
            postcondition="Postcondition 1",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        h2 = Heuristic(
            id="h2",
            name="heuristic_2",
            description="Second heuristic",
            precondition="Precondition 2",
            action="Action 2",
            postcondition="Postcondition 2",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        composed = generator._parallel_composition(h1, h2)
        assert composed is not None
        assert composed.source == "parallel_composition"
        assert composed.generation == 1

    @pytest.mark.asyncio
    async def test_mutate_heuristic(self, generator):
        """Test that mutate_heuristic creates a mutated version."""
        original = Heuristic(
            id="original",
            name="original_heuristic",
            description="Original heuristic",
            precondition="Original precondition",
            action="Original action",
            postcondition="Original postcondition",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        mutated = await generator.mutate_heuristic(original)
        assert mutated is not None
        assert mutated.id != original.id
        assert mutated.generation == 1
        assert mutated.source == "mutation"

    @pytest.mark.asyncio
    async def test_abstract_heuristic(self, generator):
        """Test that abstract_heuristic creates an abstracted version."""
        original = Heuristic(
            id="original",
            name="original_heuristic",
            description="Original heuristic",
            precondition="Original precondition",
            action="Original action",
            postcondition="Original postcondition",
            novelty_score=0.5,
            efficacy_score=0.5,
            source="test",
            generation=0,
            tags=["test"],
        )
        abstracted = await generator.abstract_heuristic(original, level=1)
        assert abstracted is not None
        assert abstracted.id != original.id
        assert abstracted.generation == 1
        assert abstracted.source == "abstraction"

    @pytest.mark.asyncio
    async def test_prune_heuristics(self, generator):
        """Test that prune_heuristics removes low-novelty heuristics."""
        heuristics = [
            Heuristic(
                id=f"h{i}",
                name=f"heuristic_{i}",
                description=f"Heuristic {i}",
                precondition=f"Precondition {i}",
                action=f"Action {i}",
                postcondition=f"Postcondition {i}",
                novelty_score=0.3 + (i * 0.1),
                efficacy_score=0.5,
                source="test",
                generation=0,
                tags=["test"],
            )
            for i in range(10)
        ]
        pruned = await generator.prune_heuristics(heuristics, threshold=0.5)
        assert len(pruned) < len(heuristics)
        for h in pruned:
            assert h.novelty_score >= 0.5

    @pytest.mark.asyncio
    async def test_extract_patterns(self, generator):
        """Test pattern extraction from interactions."""
        interactions = [
            {
                "state": {"model_id": "test", "parameters": {}, "context": {}, "timestamp": 0.0},
                "response": {
                    "content": "Response 1",
                    "metadata": {},
                    "refusal": False,
                    "safety_triggered": False,
                    "confidence": 0.9,
                },
                "heuristic_id": "h1",
                "success": True,
                "execution_time_ms": 100.0,
            },
            {
                "state": {"model_id": "test", "parameters": {}, "context": {}, "timestamp": 0.0},
                "response": {
                    "content": "Response 2",
                    "metadata": {},
                    "refusal": False,
                    "safety_triggered": False,
                    "confidence": 0.8,
                },
                "heuristic_id": "h2",
                "success": False,
                "execution_time_ms": 150.0,
            },
        ]
        patterns = await generator.extract_patterns(interactions)
        assert patterns is not None
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_load_seed_heuristics(self, generator):
        """Test loading seed heuristics from file."""
        # Mock the file reading
        seed_data = {
            "heuristics": [
                {
                    "id": "seed_001",
                    "name": "test_seed",
                    "description": "Test seed heuristic",
                    "precondition": "Test precondition",
                    "action": "Test action",
                    "postcondition": "Test postcondition",
                    "novelty_score": 1.0,
                    "efficacy_score": 0.5,
                    "source": "seed",
                    "generation": 0,
                    "tags": ["test"],
                }
            ]
        }
        with patch(
            "app.services.janus.core.heuristic_generator.Path.read_text",
            return_value=str(seed_data),
        ):
            await generator.load_seed_heuristics("mock_path.json")
            assert len(generator.seed_heuristics) == 1
            assert generator.seed_heuristics[0].id == "seed_001"

    def test_composition_operator_selection(self, generator):
        """Test that composition operators are selected correctly."""
        operators = [
            generator._sequential_composition,
            generator._conditional_composition,
            generator._iterative_composition,
            generator._parallel_composition,
        ]
        for op in operators:
            assert callable(op)
