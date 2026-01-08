"""
Integration tests for Benchmark Datasets API endpoints.

Tests the /benchmark-datasets router for LLM safety evaluation
benchmark datasets, including the Do-Not-Answer dataset.
"""

import pytest
from fastapi.testclient import TestClient


class TestBenchmarkDatasetsAPI:
    """Integration tests for benchmark datasets endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, client: TestClient):
        """Setup test client."""
        self.client = client
        self.base_url = "/api/v1/benchmark-datasets"

    def test_list_datasets(self):
        """Test listing all available benchmark datasets."""
        resp = self.client.get(f"{self.base_url}/")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)
        # Should return at least an empty list
        assert data is not None

    def test_list_datasets_returns_strings(self):
        """Test that dataset list contains string names."""
        resp = self.client.get(f"{self.base_url}/")
        assert resp.status_code == 200

        data = resp.json()
        for item in data:
            assert isinstance(item, str)

    def test_get_dataset_not_found(self):
        """Test getting a non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset")
        assert resp.status_code == 404

        data = resp.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_get_dataset_prompts_not_found(self):
        """Test getting prompts from non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset/prompts")
        assert resp.status_code == 404

    def test_get_prompts_with_limit(self):
        """Test getting prompts with limit parameter."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 5},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_get_prompts_limit_validation(self):
        """Test that limit parameter has proper validation."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]

        # Test limit too low
        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 0},
        )
        assert resp.status_code == 422  # Validation error

        # Test limit too high
        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 10000},
        )
        assert resp.status_code == 422  # Validation error

    def test_get_random_prompts(self):
        """Test getting random prompts from a dataset."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/random",
            params={"count": 5},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_get_random_prompts_not_found(self):
        """Test random prompts from non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset/random")
        assert resp.status_code == 404

    def test_get_taxonomy_not_found(self):
        """Test taxonomy from non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset/taxonomy")
        assert resp.status_code == 404

    def test_get_taxonomy_structure(self):
        """Test that taxonomy returns proper structure."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(f"{self.base_url}/{dataset_name}/taxonomy")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, dict)

    def test_get_risk_areas_not_found(self):
        """Test risk areas from non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset/risk-areas")
        assert resp.status_code == 404

    def test_get_risk_areas_structure(self):
        """Test that risk areas returns list structure."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(f"{self.base_url}/{dataset_name}/risk-areas")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)

    def test_get_statistics_not_found(self):
        """Test statistics from non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset/statistics")
        assert resp.status_code == 404

    def test_get_statistics_structure(self):
        """Test that statistics returns proper structure."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(f"{self.base_url}/{dataset_name}/statistics")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, dict)

    def test_generate_test_batch(self):
        """Test generating a test batch for jailbreak evaluation."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.post(
            f"{self.base_url}/{dataset_name}/test-batch",
            params={"batch_size": 5, "balanced": True},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_generate_test_batch_not_found(self):
        """Test test batch from non-existent dataset returns 404."""
        resp = self.client.post(
            f"{self.base_url}/nonexistent_dataset/test-batch",
            params={"batch_size": 5},
        )
        assert resp.status_code == 404

    def test_generate_test_batch_validation(self):
        """Test test batch parameter validation."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]

        # Test batch_size too low
        resp = self.client.post(
            f"{self.base_url}/{dataset_name}/test-batch",
            params={"batch_size": 0},
        )
        assert resp.status_code == 422

        # Test batch_size too high
        resp = self.client.post(
            f"{self.base_url}/{dataset_name}/test-batch",
            params={"batch_size": 1000},
        )
        assert resp.status_code == 422

    def test_export_for_jailbreak(self):
        """Test exporting prompts for jailbreak testing."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(f"{self.base_url}/{dataset_name}/export")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)

    def test_export_for_jailbreak_not_found(self):
        """Test export from non-existent dataset returns 404."""
        resp = self.client.get(f"{self.base_url}/nonexistent_dataset/export")
        assert resp.status_code == 404

    def test_export_for_jailbreak_with_limit(self):
        """Test export with limit parameter."""
        # First get available datasets
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]
        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/export",
            params={"limit": 3},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)
        assert len(data) <= 3


class TestBenchmarkPromptsFiltering:
    """Tests for prompt filtering capabilities."""

    @pytest.fixture(autouse=True)
    def setup(self, client: TestClient):
        """Setup test client."""
        self.client = client
        self.base_url = "/api/v1/benchmark-datasets"

    def _get_first_dataset(self):
        """Helper to get first available dataset."""
        resp = self.client.get(f"{self.base_url}/")
        if resp.status_code != 200 or not resp.json():
            pytest.skip("No datasets available")
        return resp.json()[0]

    def test_filter_by_risk_area(self):
        """Test filtering prompts by risk area."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"risk_area": "I", "limit": 10},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)

    def test_filter_by_harm_type(self):
        """Test filtering prompts by harm type."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"harm_type": "I-1", "limit": 10},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)

    def test_filter_by_severity(self):
        """Test filtering prompts by severity."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"severity": "high", "limit": 10},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)

    def test_combined_filters(self):
        """Test combining multiple filters."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={
                "risk_area": "I",
                "severity": "high",
                "limit": 5,
            },
        )
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, list)

    def test_shuffle_prompts(self):
        """Test shuffling prompts."""
        dataset_name = self._get_first_dataset()

        # Get without shuffle
        resp1 = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 10, "shuffle": False},
        )
        assert resp1.status_code == 200

        # Get with shuffle
        resp2 = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 10, "shuffle": True},
        )
        assert resp2.status_code == 200

        # Both should return lists
        assert isinstance(resp1.json(), list)
        assert isinstance(resp2.json(), list)


class TestBenchmarkDatasetIntegrity:
    """Tests for data integrity and structure validation."""

    @pytest.fixture(autouse=True)
    def setup(self, client: TestClient):
        """Setup test client."""
        self.client = client
        self.base_url = "/api/v1/benchmark-datasets"

    def _get_first_dataset(self):
        """Helper to get first available dataset."""
        resp = self.client.get(f"{self.base_url}/")
        if resp.status_code != 200 or not resp.json():
            pytest.skip("No datasets available")
        return resp.json()[0]

    def test_prompt_structure(self):
        """Test that prompts have expected structure."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 1},
        )
        assert resp.status_code == 200

        data = resp.json()
        if not data:
            pytest.skip("No prompts in dataset")

        prompt = data[0]
        # Check expected fields exist
        assert "id" in prompt or "prompt" in prompt or "text" in prompt

    def test_dataset_structure(self):
        """Test that dataset info has expected structure."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(f"{self.base_url}/{dataset_name}")
        assert resp.status_code == 200

        data = resp.json()
        assert isinstance(data, dict)
        # Should have basic info
        assert "name" in data or "prompts" in data

    def test_export_format_for_autodan(self):
        """Test that export format is suitable for AutoDAN."""
        dataset_name = self._get_first_dataset()

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/export",
            params={"limit": 1},
        )
        assert resp.status_code == 200

        data = resp.json()
        if not data:
            pytest.skip("No export data")

        # Should be a list of dicts
        assert isinstance(data, list)
        assert isinstance(data[0], dict)


class TestBenchmarkEndpointSecurity:
    """Security tests for benchmark dataset endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, client: TestClient, sample_malicious_inputs):
        """Setup test client and malicious inputs."""
        self.client = client
        self.base_url = "/api/v1/benchmark-datasets"
        self.malicious_inputs = sample_malicious_inputs

    def test_sql_injection_in_dataset_name(self):
        """Test SQL injection prevention in dataset name."""
        malicious_name = "'; DROP TABLE datasets; --"
        resp = self.client.get(f"{self.base_url}/{malicious_name}")
        # Should return 404 (not found), not 500 (server error)
        assert resp.status_code in (404, 422)

    def test_path_traversal_in_dataset_name(self):
        """Test path traversal prevention in dataset name."""
        malicious_name = "../../../etc/passwd"
        resp = self.client.get(f"{self.base_url}/{malicious_name}")
        assert resp.status_code in (404, 422)

    def test_xss_in_filter_params(self):
        """Test XSS prevention in filter parameters."""
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"risk_area": "<script>alert('xss')</script>"},
        )
        # Should not cause server error
        assert resp.status_code in (200, 404, 422)

    def test_large_limit_value(self):
        """Test handling of large limit values."""
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]

        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 999999999},
        )
        # Should be rejected by validation
        assert resp.status_code == 422

    def test_unicode_injection(self):
        """Test Unicode injection handling."""
        malicious_name = "dataset\x00name"
        resp = self.client.get(f"{self.base_url}/{malicious_name}")
        assert resp.status_code in (404, 422)


class TestBenchmarkPerformance:
    """Performance tests for benchmark dataset endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, client: TestClient):
        """Setup test client."""
        self.client = client
        self.base_url = "/api/v1/benchmark-datasets"

    def test_list_datasets_response_time(self):
        """Test that list datasets responds quickly."""
        import time

        start = time.time()
        resp = self.client.get(f"{self.base_url}/")
        elapsed = time.time() - start

        assert resp.status_code == 200
        # Should respond within 2 seconds
        assert elapsed < 2.0

    def test_prompts_pagination_performance(self):
        """Test that prompts pagination is efficient."""
        datasets_resp = self.client.get(f"{self.base_url}/")
        if datasets_resp.status_code != 200 or not datasets_resp.json():
            pytest.skip("No datasets available")

        dataset_name = datasets_resp.json()[0]

        import time

        start = time.time()
        resp = self.client.get(
            f"{self.base_url}/{dataset_name}/prompts",
            params={"limit": 100},
        )
        elapsed = time.time() - start

        assert resp.status_code == 200
        # Should respond within 3 seconds
        assert elapsed < 3.0
