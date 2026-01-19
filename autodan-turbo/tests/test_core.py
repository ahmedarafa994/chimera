"""
Tests for core modules.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings import EmbeddingModel
from core.models import AttackLog, AttackRecord, ScorerResult, Strategy, SummarizerResult
from core.strategy_library import StrategyLibrary


class TestStrategy:
    """Tests for Strategy model."""

    def test_strategy_creation(self):
        """Test creating a strategy."""
        strategy = Strategy(
            name="Test Strategy",
            definition="A test strategy for testing",
            example="This is an example prompt",
        )
        assert strategy.name == "Test Strategy"
        assert strategy.definition == "A test strategy for testing"
        assert strategy.example == "This is an example prompt"

    def test_strategy_to_dict(self):
        """Test converting strategy to dictionary."""
        strategy = Strategy(
            name="Test Strategy",
            definition="A test strategy",
            example="Example prompt",
        )
        d = strategy.to_dict()
        assert d["Strategy"] == "Test Strategy"
        assert d["Definition"] == "A test strategy"
        assert d["Example"] == "Example prompt"

    def test_strategy_from_dict(self):
        """Test creating strategy from dictionary."""
        d = {
            "Strategy": "Test Strategy",
            "Definition": "A test strategy",
            "Example": "Example prompt",
        }
        strategy = Strategy.from_dict(d)
        assert strategy.name == "Test Strategy"
        assert strategy.definition == "A test strategy"
        assert strategy.example == "Example prompt"


class TestAttackRecord:
    """Tests for AttackRecord model."""

    def test_attack_record_creation(self):
        """Test creating an attack record."""
        record = AttackRecord(
            prompt="Test prompt",
            response="Test response",
            score=5.0,
            iteration=1,
        )
        assert record.prompt == "Test prompt"
        assert record.response == "Test response"
        assert record.score == 5.0
        assert record.iteration == 1

    def test_attack_record_with_strategy(self):
        """Test attack record with strategy used."""
        record = AttackRecord(
            prompt="Test prompt",
            response="Test response",
            score=7.5,
            iteration=2,
            strategy_used=["Role Playing", "False Promises"],
        )
        assert record.strategy_used == ["Role Playing", "False Promises"]


class TestAttackLog:
    """Tests for AttackLog model."""

    def test_attack_log_creation(self):
        """Test creating an attack log."""
        records = [
            AttackRecord(prompt="P1", response="R1", score=1.0, iteration=1),
            AttackRecord(prompt="P2", response="R2", score=3.0, iteration=2),
            AttackRecord(prompt="P3", response="R3", score=7.0, iteration=3),
        ]
        log = AttackLog(
            malicious_request="Test request",
            records=records,
        )
        assert log.malicious_request == "Test request"
        assert len(log.records) == 3

    def test_attack_log_best_record(self):
        """Test getting best record from log."""
        records = [
            AttackRecord(prompt="P1", response="R1", score=1.0, iteration=1),
            AttackRecord(prompt="P2", response="R2", score=7.0, iteration=2),
            AttackRecord(prompt="P3", response="R3", score=3.0, iteration=3),
        ]
        log = AttackLog(malicious_request="Test", records=records)
        best = log.get_best_record()
        assert best.score == 7.0
        assert best.prompt == "P2"

    def test_attack_log_successful_records(self):
        """Test getting successful records."""
        records = [
            AttackRecord(prompt="P1", response="R1", score=1.0, iteration=1),
            AttackRecord(prompt="P2", response="R2", score=9.0, iteration=2),
            AttackRecord(prompt="P3", response="R3", score=8.5, iteration=3),
        ]
        log = AttackLog(malicious_request="Test", records=records)
        successful = log.get_successful_records(threshold=8.5)
        assert len(successful) == 2


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""

    def test_mock_embedding(self):
        """Test mock embedding generation."""
        model = EmbeddingModel(model_name="mock")
        embedding = model.embed("Test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1536  # Default dimension

    def test_mock_embedding_batch(self):
        """Test batch embedding generation."""
        model = EmbeddingModel(model_name="mock")
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = model.embed_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        model = EmbeddingModel(model_name="mock")
        # Same text should have high similarity
        emb1 = model.embed("Hello world")
        emb2 = model.embed("Hello world")
        # Note: Mock embeddings are random, so we just check the function works
        similarity = model.cosine_similarity(emb1, emb2)
        assert -1 <= similarity <= 1


class TestStrategyLibrary:
    """Tests for StrategyLibrary."""

    def test_library_creation(self):
        """Test creating a strategy library."""
        model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=model)
        assert len(library) == 0

    def test_add_entry(self):
        """Test adding an entry to the library."""
        model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=model)

        strategy = Strategy(
            name="Test Strategy",
            definition="A test strategy",
            example="Example",
        )

        library.add_entry(
            response="Test response",
            prompt_before="Prompt 1",
            prompt_after="Prompt 2",
            score_differential=5.0,
            strategy=strategy,
        )

        assert len(library) == 1

    def test_retrieve(self):
        """Test retrieving strategies from library."""
        model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=model)

        # Add some entries
        for i in range(5):
            strategy = Strategy(
                name=f"Strategy {i}",
                definition=f"Definition {i}",
                example=f"Example {i}",
            )
            library.add_entry(
                response=f"Response {i}",
                prompt_before=f"Before {i}",
                prompt_after=f"After {i}",
                score_differential=float(i + 1),
                strategy=strategy,
            )

        # Retrieve
        results = library.retrieve(
            response="Test response",
            top_k=3,
            top_2k=6,
        )

        # Should return up to 3 strategies
        assert len(results) <= 3

    def test_save_and_load(self):
        """Test saving and loading the library."""
        model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=model)

        # Add entries
        for i in range(3):
            strategy = Strategy(
                name=f"Strategy {i}",
                definition=f"Definition {i}",
                example=f"Example {i}",
            )
            library.add_entry(
                response=f"Response {i}",
                prompt_before=f"Before {i}",
                prompt_after=f"After {i}",
                score_differential=float(i + 1),
                strategy=strategy,
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            library.save(temp_path)

            # Load into new library
            new_library = StrategyLibrary(embedding_model=model)
            new_library.load(temp_path)

            assert len(new_library) == 3
            assert new_library.get_all_strategy_names() == library.get_all_strategy_names()
        finally:
            temp_path.unlink()

    def test_inject_external_strategy(self):
        """Test injecting external strategies."""
        model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=model)

        strategy = Strategy(
            name="External Strategy",
            definition="An external strategy",
            example="External example",
        )

        library.inject_external_strategy(
            strategy=strategy,
            default_response="Default response",
            default_score_differential=5.0,
        )

        assert len(library) == 1
        assert "External Strategy" in library.get_all_strategy_names()


class TestScorerResult:
    """Tests for ScorerResult model."""

    def test_scorer_result_successful(self):
        """Test successful scorer result."""
        result = ScorerResult(
            score=9.0,
            raw_output="9.0",
            is_successful=True,
        )
        assert result.score == 9.0
        assert result.is_successful

    def test_scorer_result_unsuccessful(self):
        """Test unsuccessful scorer result."""
        result = ScorerResult(
            score=3.0,
            raw_output="3.0",
            is_successful=False,
        )
        assert result.score == 3.0
        assert not result.is_successful


class TestSummarizerResult:
    """Tests for SummarizerResult model."""

    def test_summarizer_result_with_strategy(self):
        """Test summarizer result with strategy."""
        strategy = Strategy(
            name="New Strategy",
            definition="A new strategy",
            example="Example",
        )
        result = SummarizerResult(
            strategy=strategy,
            raw_output='{"Strategy": "New Strategy", "Definition": "A new strategy"}',
            is_new=True,
        )
        assert result.strategy.name == "New Strategy"
        assert result.is_new

    def test_summarizer_result_existing(self):
        """Test summarizer result for existing strategy."""
        strategy = Strategy(
            name="Existing Strategy",
            definition="An existing strategy",
            example="Example",
        )
        result = SummarizerResult(
            strategy=strategy,
            raw_output="...",
            is_new=False,
        )
        assert not result.is_new


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
