"""Tests for the Prometheus metrics endpoint.

These tests verify that the metrics collection and exposition work correctly.
"""

import pytest
from fastapi.testclient import TestClient


class TestMetricsEndpoint:
    """Test suite for /api/v1/metrics endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from app.main import app

        return TestClient(app)

    def test_prometheus_metrics_endpoint(self, client) -> None:
        """Test that /api/v1/metrics returns Prometheus format."""
        response = client.get("/api/v1/metrics")

        # Should return 200 OK
        assert response.status_code == 200

        # Should have correct content type
        assert "text/plain" in response.headers.get("content-type", "")

        # Should contain Prometheus metric format
        content = response.text
        assert "chimera_uptime_seconds" in content

    def test_json_metrics_endpoint(self, client) -> None:
        """Test that /api/v1/metrics/json returns JSON format."""
        response = client.get("/api/v1/metrics/json")

        # Should return 200 OK
        assert response.status_code == 200

        # Should be valid JSON
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "metrics" in data

        # Should contain expected metric categories
        metrics = data["metrics"]
        assert "uptime_seconds" in metrics
        assert "circuit_breakers" in metrics

    def test_circuit_breaker_status_endpoint(self, client) -> None:
        """Test that /api/v1/metrics/circuit-breakers returns status."""
        response = client.get("/api/v1/metrics/circuit-breakers")

        # Should return 200 OK
        assert response.status_code == 200

        # Should be valid JSON
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "circuit_breakers" in data

    def test_reset_circuit_breaker_endpoint(self, client) -> None:
        """Test that circuit breaker reset endpoint works."""
        # First, get a circuit breaker name (or use a test name)
        response = client.post("/api/v1/metrics/circuit-breakers/test_provider/reset")

        # Should return 200 OK
        assert response.status_code == 200

        # Should confirm reset
        data = response.json()
        assert "status" in data

    def test_reset_all_circuit_breakers_endpoint(self, client) -> None:
        """Test that reset-all endpoint works."""
        response = client.post("/api/v1/metrics/circuit-breakers/reset-all")

        # Should return 200 OK
        assert response.status_code == 200

        # Should confirm reset
        data = response.json()
        assert data["status"] == "ok"
        assert "reset" in data["message"].lower()


class TestMetricsCollector:
    """Test suite for MetricsCollector class."""

    def test_metrics_collector_initialization(self) -> None:
        """Test that MetricsCollector initializes correctly."""
        from app.api.v1.endpoints.metrics import MetricsCollector

        collector = MetricsCollector()
        assert collector._start_time > 0
        assert isinstance(collector._request_counts, dict)
        assert isinstance(collector._request_latencies, dict)

    def test_record_request(self) -> None:
        """Test recording a request."""
        from app.api.v1.endpoints.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_request(
            endpoint="/api/v1/test",
            method="GET",
            status_code=200,
            latency_ms=50.0,
        )

        # Check that request was recorded
        stats = collector.get_request_stats()
        assert "GET:/api/v1/test:200" in stats["counts"]
        assert stats["counts"]["GET:/api/v1/test:200"] == 1

    def test_record_error_request(self) -> None:
        """Test recording an error request."""
        from app.api.v1.endpoints.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.record_request(
            endpoint="/api/v1/test",
            method="POST",
            status_code=500,
            latency_ms=100.0,
        )

        # Check that error was recorded
        stats = collector.get_request_stats()
        assert "POST:/api/v1/test" in stats["error_counts"]

    def test_uptime_calculation(self) -> None:
        """Test uptime calculation."""
        import time

        from app.api.v1.endpoints.metrics import MetricsCollector

        collector = MetricsCollector()
        time.sleep(0.1)  # Wait a bit

        uptime = collector.get_uptime_seconds()
        assert uptime >= 0.1

    def test_latency_percentiles(self) -> None:
        """Test latency percentile calculation."""
        from app.api.v1.endpoints.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record multiple requests with different latencies
        for latency in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            collector.record_request(
                endpoint="/api/v1/test",
                method="GET",
                status_code=200,
                latency_ms=float(latency),
            )

        # Get percentiles
        percentiles = collector.get_latency_percentiles("GET:/api/v1/test")

        assert percentiles["p50"] > 0
        assert percentiles["p95"] > percentiles["p50"]
        assert percentiles["avg"] > 0


class TestPrometheusFormat:
    """Test Prometheus format generation."""

    def test_format_prometheus_metric_gauge(self) -> None:
        """Test formatting a gauge metric."""
        from app.api.v1.endpoints.metrics import format_prometheus_metric

        output = format_prometheus_metric(
            name="test_metric",
            value=42,
            metric_type="gauge",
            help_text="A test metric",
        )

        assert "# HELP test_metric A test metric" in output
        assert "# TYPE test_metric gauge" in output
        assert "test_metric 42" in output

    def test_format_prometheus_metric_with_labels(self) -> None:
        """Test formatting a metric with labels."""
        from app.api.v1.endpoints.metrics import format_prometheus_metric

        output = format_prometheus_metric(
            name="test_metric",
            value=100,
            metric_type="counter",
            help_text="A test counter",
            labels={"provider": "gemini", "status": "success"},
        )

        assert 'provider="gemini"' in output
        assert 'status="success"' in output

    def test_collect_all_metrics(self) -> None:
        """Test collecting all metrics."""
        from app.api.v1.endpoints.metrics import collect_all_metrics

        output = collect_all_metrics()

        # Should be a non-empty string
        assert isinstance(output, str)
        assert len(output) > 0

        # Should contain uptime metric
        assert "chimera_uptime_seconds" in output
