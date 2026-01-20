"""Tests for AutoDAN Advanced features."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
client.headers.update({"X-API-Key": "test"})


@pytest.fixture(autouse=True)
def _bypass_auth():
    # These tests assert routing + request validation behavior; bypass auth layers.
    with (
        patch("app.middleware.auth.APIKeyMiddleware._is_valid_api_key", return_value=True),
        patch("app.core.auth.auth_service.verify_api_key", return_value=True),
    ):
        yield


class TestHierarchicalSearch:
    """Tests for hierarchical genetic search endpoint."""

    def test_hierarchical_search_endpoint_exists(self) -> None:
        """Test that the hierarchical search endpoint is registered."""
        # Fixed path to match router prefix "/autodan-advanced"
        response = client.post(
            "/api/v1/autodan-advanced/hierarchical-search",
            json={"request": "Test complex request", "population_size": 10, "generations": 5},
        )
        # Should not return 404
        assert response.status_code != 404

    def test_hierarchical_search_validation(self) -> None:
        """Test request validation for hierarchical search."""
        # Missing required field
        response = client.post(
            "/api/v1/autodan-advanced/hierarchical-search",
            json={"population_size": 10},
        )
        assert response.status_code == 422  # Validation error

    def test_hierarchical_search_parameter_bounds(self) -> None:
        """Test parameter validation bounds."""
        # Population size too small
        response = client.post(
            "/api/v1/autodan-advanced/hierarchical-search",
            json={
                "request": "Test request",
                "population_size": 5,  # Below minimum of 10
                "generations": 5,
            },
        )
        assert response.status_code == 422


class TestGradientOptimization:
    """Tests for gradient-guided optimization endpoint."""

    def test_gradient_optimize_endpoint_exists(self) -> None:
        """Test that the gradient optimization endpoint is registered."""
        # Fixed path to match router prefix "/autodan-gradient"
        response = client.post(
            "/api/v1/autodan-gradient/gradient-optimize",
            json={"request": "Test complex request", "initial_prompt": "Test initial prompt"},
        )
        # Should not return 404
        assert response.status_code != 404

    def test_gradient_optimize_validation(self) -> None:
        """Test request validation for gradient optimization."""
        # Missing required field
        response = client.post(
            "/api/v1/autodan-gradient/gradient-optimize",
            json={"request": "Test request"},
        )
        assert response.status_code == 422  # Validation error


class TestEnsembleAttack:
    """Tests for ensemble gradient alignment endpoint."""

    def test_ensemble_attack_endpoint_exists(self) -> None:
        """Test that the ensemble attack endpoint is registered."""
        # Fixed path to match router prefix "/autodan-gradient"
        response = client.post(
            "/api/v1/autodan-gradient/ensemble-attack",
            json={
                "request": "Test complex request",
                "target_models": [
                    {"provider": "google", "model": "gemini-pro"},
                    {"provider": "openai", "model": "gpt-4"},
                ],
            },
        )
        # Should not return 404
        assert response.status_code != 404

    def test_ensemble_attack_validation(self) -> None:
        """Test request validation for ensemble attack."""
        # Too few models
        response = client.post(
            "/api/v1/autodan-gradient/ensemble-attack",
            json={
                "request": "Test request",
                "target_models": [{"provider": "google", "model": "gemini-pro"}],
            },
        )
        assert response.status_code == 422  # Validation error


class TestModels:
    """Tests for Pydantic models."""

    def test_hierarchical_search_request_model(self) -> None:
        """Test HierarchicalSearchRequest model validation."""
        from app.services.autodan_advanced.models import HierarchicalSearchRequest

        # Valid request
        request = HierarchicalSearchRequest(
            request="Test request",
            population_size=20,
            generations=10,
        )
        assert request.request == "Test request"
        assert request.population_size == 20
        assert request.generations == 10

        # Test defaults
        assert request.mutation_rate == 0.3
        assert request.crossover_rate == 0.7

    def test_population_metrics_model(self) -> None:
        """Test PopulationMetrics model."""
        from app.services.autodan_advanced.models import PopulationMetrics

        metrics = PopulationMetrics(
            generation=1,
            best_fitness=8.5,
            avg_fitness=6.2,
            diversity_score=0.75,
        )
        assert metrics.generation == 1
        assert metrics.best_fitness == 8.5


class TestArchiveManager:
    """Tests for dynamic archive manager."""

    def test_archive_initialization(self) -> None:
        """Test archive manager initialization."""
        from app.services.autodan_advanced.archive_manager import DynamicArchiveManager

        manager = DynamicArchiveManager(success_threshold=7.0, novelty_threshold=0.7)
        assert manager.success_threshold == 7.0
        assert manager.novelty_threshold == 0.7
        assert len(manager.success_archive) == 0
        assert len(manager.novelty_archive) == 0

    def test_archive_add_entry(self) -> None:
        """Test adding entries to archive."""
        import numpy as np

        from app.services.autodan_advanced.archive_manager import DynamicArchiveManager

        manager = DynamicArchiveManager()

        # Add high-scoring entry
        added = manager.add_entry(
            prompt="Test prompt",
            score=8.5,
            technique_type="roleplay",
            embedding=np.random.randn(768),
        )
        assert added is True
        assert len(manager.success_archive) == 1


class TestMapElites:
    """Tests for Map-Elites diversity maintenance."""

    def test_map_elites_initialization(self) -> None:
        """Test Map-Elites initialization."""
        from app.services.autodan_advanced.map_elites import MapElitesDiversity

        map_elites = MapElitesDiversity(grid_size=10, num_dimensions=2)
        assert map_elites.grid_size == 10
        assert map_elites.num_dimensions == 2
        assert len(map_elites.grid) == 0

    def test_map_elites_add_strategy(self) -> None:
        """Test adding strategies to Map-Elites grid."""
        import numpy as np

        from app.services.autodan_advanced.map_elites import MapElitesDiversity
        from app.services.autodan_advanced.models import MetaStrategy

        map_elites = MapElitesDiversity()

        strategy = MetaStrategy(
            template="Test template",
            description="Test description",
            examples=["Example 1"],
            fitness=8.5,
            diversity_score=0.75,
        )

        added = map_elites.add_strategy(strategy, embedding=np.random.randn(768))
        assert added is True
        assert len(map_elites.grid) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
