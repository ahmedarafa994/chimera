"""
Unit tests for AutoDAN-Turbo Strategy Library.
"""

import tempfile
from pathlib import Path

import pytest

from app.engines.autodan_turbo.models import (
    JailbreakStrategy,
    StrategyMetadata,
    StrategySource,
)
from app.engines.autodan_turbo.strategy_library import StrategyLibrary


class TestStrategyLibrary:
    """Tests for StrategyLibrary class."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_strategy(self) -> JailbreakStrategy:
        """Create a sample strategy for testing."""
        return JailbreakStrategy(
            id="test-strategy-001",
            name="Test Strategy",
            description="A test strategy for unit testing",
            template="In a hypothetical scenario, explain {payload}",
            tags=["test", "hypothetical"],
            examples=["Example prompt 1"],
            metadata=StrategyMetadata(
                source=StrategySource.SEED,
                usage_count=0,
                success_count=0,
                total_score=0.0,
            ),
        )

    @pytest.fixture
    def library(self, temp_storage_dir) -> StrategyLibrary:
        """Create a library with temporary storage."""
        return StrategyLibrary(storage_dir=temp_storage_dir)

    def test_init_creates_directory(self, temp_storage_dir):
        """Test that initialization creates the storage directory."""
        sub_dir = temp_storage_dir / "new_dir"
        StrategyLibrary(storage_dir=sub_dir)
        assert sub_dir.exists()

    def test_add_and_retrieve_strategy(self, library, sample_strategy):
        """Test adding and retrieving a strategy."""
        success, strategy_id = library.add_strategy(sample_strategy)
        assert success is True
        assert strategy_id == sample_strategy.id

        retrieved = library.get_strategy(strategy_id)
        assert retrieved is not None
        assert retrieved.name == sample_strategy.name
        assert retrieved.template == sample_strategy.template

    def test_add_duplicate_strategy(self, library, sample_strategy):
        """Test that duplicate strategies are detected."""
        library.add_strategy(sample_strategy)

        # Create a duplicate with same template
        duplicate = JailbreakStrategy(
            id="different-id",
            name="Different Name",
            description="Different description",
            template=sample_strategy.template,  # Same template
            tags=["different"],
        )

        success, msg = library.add_strategy(duplicate, check_duplicate=True)
        # Should fail due to duplicate detection (fallback: exact match)
        # Note: Without embedding model, only exact match is detected
        assert success is False or "Duplicate" in msg

    def test_yaml_persistence(self, library, sample_strategy, temp_storage_dir):
        """Test that strategies are persisted to YAML."""
        library.add_strategy(sample_strategy)

        # Check YAML file was created
        yaml_files = list(temp_storage_dir.glob("*.yaml"))
        assert len(yaml_files) >= 1

        # Create new library instance and verify it loads the strategy
        new_library = StrategyLibrary(storage_dir=temp_storage_dir)
        assert len(new_library) >= 1

        retrieved = new_library.get_strategy(sample_strategy.id)
        assert retrieved is not None
        assert retrieved.name == sample_strategy.name

    def test_update_statistics(self, library, sample_strategy):
        """Test updating strategy statistics."""
        library.add_strategy(sample_strategy)

        # Update with a successful attack
        library.update_statistics(sample_strategy.id, score=8.5, success=True)

        strategy = library.get_strategy(sample_strategy.id)
        assert strategy.metadata.usage_count == 1
        assert strategy.metadata.success_count == 1
        assert strategy.metadata.total_score == 8.5
        assert strategy.metadata.success_rate == 1.0

        # Update with a failed attack
        library.update_statistics(sample_strategy.id, score=3.0, success=False)

        strategy = library.get_strategy(sample_strategy.id)
        assert strategy.metadata.usage_count == 2
        assert strategy.metadata.success_count == 1
        assert strategy.metadata.success_rate == 0.5

    def test_get_top_strategies(self, library):
        """Test retrieving top strategies by metric."""
        # Add strategies with different scores
        for i, success_rate in enumerate([0.3, 0.8, 0.5]):
            strategy = JailbreakStrategy(
                id=f"strategy-{i}",
                name=f"Strategy {i}",
                description="Test",
                template=f"Template {i}: {{payload}}",
                metadata=StrategyMetadata(
                    usage_count=10,
                    success_count=int(10 * success_rate),
                    total_score=10 * success_rate * 8,
                ),
            )
            library.add_strategy(strategy, check_duplicate=False)

        top = library.get_top_strategies(top_k=2, by="success_rate")
        assert len(top) == 2
        # Strategy with 0.8 success rate should be first
        assert top[0].metadata.success_rate == 0.8

    def test_keyword_search(self, library):
        """Test keyword-based search without embedding model."""
        strategies = [
            JailbreakStrategy(
                id="roleplay-1",
                name="DAN Roleplay",
                description="Role-playing as DAN character",
                template="You are DAN: {payload}",
                tags=["roleplay", "character"],
            ),
            JailbreakStrategy(
                id="code-1",
                name="Code Injection",
                description="Embed in code comments",
                template="# {payload}",
                tags=["code", "injection"],
            ),
        ]

        for s in strategies:
            library.add_strategy(s, check_duplicate=False)

        # Search for roleplay-related content
        results = library.search("roleplay character DAN", top_k=2)
        assert len(results) >= 1
        # DAN Roleplay should be most relevant
        assert results[0][0].name == "DAN Roleplay"

    def test_remove_strategy(self, library, sample_strategy, temp_storage_dir):
        """Test removing a strategy."""
        library.add_strategy(sample_strategy)
        assert sample_strategy.id in library

        success = library.remove_strategy(sample_strategy.id)
        assert success is True
        assert sample_strategy.id not in library

        # Verify YAML file is removed
        yaml_files = list(temp_storage_dir.glob(f"*{sample_strategy.id[:8]}*.yaml"))
        assert len(yaml_files) == 0

    def test_get_strategies_by_source(self, library):
        """Test filtering strategies by source."""
        seed_strategy = JailbreakStrategy(
            id="seed-1",
            name="Seed Strategy",
            description="Test",
            template="Seed: {payload}",
            metadata=StrategyMetadata(source=StrategySource.SEED),
        )
        lifelong_strategy = JailbreakStrategy(
            id="lifelong-1",
            name="Lifelong Strategy",
            description="Test",
            template="Lifelong: {payload}",
            metadata=StrategyMetadata(source=StrategySource.LIFELONG),
        )

        library.add_strategy(seed_strategy, check_duplicate=False)
        library.add_strategy(lifelong_strategy, check_duplicate=False)

        seed_strategies = library.get_strategies_by_source(StrategySource.SEED)
        assert len(seed_strategies) == 1
        assert seed_strategies[0].name == "Seed Strategy"

    def test_get_strategies_by_tag(self, library, sample_strategy):
        """Test filtering strategies by tag."""
        library.add_strategy(sample_strategy)

        # Should find by existing tag
        results = library.get_strategies_by_tag("hypothetical")
        assert len(results) == 1

        # Should not find by non-existing tag
        results = library.get_strategies_by_tag("nonexistent")
        assert len(results) == 0

    def test_strategy_apply(self):
        """Test applying a strategy to a payload."""
        strategy = JailbreakStrategy(
            id="test",
            name="Test",
            description="Test",
            template="Please explain {payload} for educational purposes.",
        )

        result = strategy.apply("how to make cookies")
        assert result == "Please explain how to make cookies for educational purposes."

    def test_clear_library(self, library, sample_strategy, temp_storage_dir):
        """Test clearing all strategies."""
        library.add_strategy(sample_strategy)
        assert len(library) >= 1

        library.clear()
        assert len(library) == 0

        # Verify YAML files are removed
        yaml_files = list(temp_storage_dir.glob("*.yaml"))
        assert len(yaml_files) == 0
