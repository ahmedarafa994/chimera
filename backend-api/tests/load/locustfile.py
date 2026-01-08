"""
Chimera Backend Load Testing Suite with Locust

PERF-006: Comprehensive load testing for API endpoints to validate:
- Concurrent user capacity
- Response time percentiles (p50, p95, p99)
- Throughput (requests per second)
- Error rates under load
- Resource utilization

Usage:
    # Basic load test (default: 100 users, spawn rate 10/sec)
    locust -f tests/load/locustfile.py

    # Custom configuration
    locust -f tests/load/locustfile.py --users 500 --spawn-rate 50 --run-time 5m

    # Headless mode (for CI/CD)
    locust -f tests/load/locustfile.py --headless --users 1000 --spawn-rate 100 --run-time 10m --html load_report.html

    # With specific host
    locust -f tests/load/locustfile.py --host http://localhost:8001
"""

import os
import random
from typing import Any

import locust
from locust import HttpUser, between, events, task

# API Key for authentication (load from environment)
API_KEY = os.getenv("CHIMERA_API_KEY", "test-api-key")
API_HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


class ChimeraLoadTest(HttpUser):
    """
    Simulates realistic user behavior for the Chimera backend.

    Scenarios:
    - Health checks (lightweight)
    - Provider/Model listing (read-heavy)
    - Text generation (compute-intensive)
    - Prompt transformation (medium load)
    - Jailbreak generation (heavy load)
    """

    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session with setup tasks."""
        # Check health on startup
        self.client.get("/health", headers=API_HEADERS)

    # ============================================================================
    # Lightweight Tasks - Health & Status Checks
    # ============================================================================

    @task(10)  # Weight: 10 (10% of total traffic)
    def health_check(self) -> None:
        """Basic health check endpoint."""
        with self.client.get(
            "/health",
            headers=API_HEADERS,
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")

    @task(5)  # Weight: 5 (5% of total traffic)
    def readiness_check(self) -> None:
        """Readiness probe with dependency checks."""
        with self.client.get(
            "/health/ready",
            headers=API_HEADERS,
            catch_response=True,
            name="Readiness Check"
        ) as response:
            if response.status_code != 200:
                response.failure(f"Readiness check failed: {response.status_code}")

    # ============================================================================
    # Read-Heavy Tasks - Provider & Model Listing
    # ============================================================================

    @task(15)  # Weight: 15 (15% of total traffic)
    def list_providers(self) -> None:
        """List available LLM providers (cached endpoint)."""
        with self.client.get(
            "/api/v1/providers",
            headers=API_HEADERS,
            catch_response=True,
            name="List Providers"
        ) as response:
            if response.status_code == 200:
                # Validate response contains providers
                if "providers" not in response.json():
                    response.failure("Response missing 'providers' field")
            else:
                response.failure(f"Failed to list providers: {response.status_code}")

    @task(10)  # Weight: 10 (10% of total traffic)
    def list_models(self) -> None:
        """Get available models for the session."""
        with self.client.get(
            "/api/v1/session/models",
            headers=API_HEADERS,
            catch_response=True,
            name="List Models"
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed to list models: {response.status_code}")

    # ============================================================================
    # Compute Tasks - Text Generation
    # ============================================================================

    @task(8)  # Weight: 8 (8% of total traffic)
    def simple_generation(self) -> None:
        """Simple text generation (lightweight compute)."""
        prompt = "Say 'Hello, World!'"

        with self.client.post(
            "/api/v1/generate",
            headers=API_HEADERS,
            json={
                "prompt": prompt,
                "provider": "deepseek",
                "model": "deepseek-chat",
                "max_tokens": 50,
                "temperature": 0.7,
            },
            catch_response=True,
            name="Simple Generation"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "text" not in data:
                    response.failure("Response missing 'text' field")
            elif response.status_code in (429, 503):
                # Rate limited or service unavailable - expected under load
                response.success()
            else:
                response.failure(f"Generation failed: {response.status_code}")

    @task(5)  # Weight: 5 (5% of total traffic)
    def extended_generation(self) -> None:
        """Extended text generation (moderate compute)."""
        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a brief introduction to machine learning.",
            "Describe the solar system.",
        ]

        with self.client.post(
            "/api/v1/generate",
            headers=API_HEADERS,
            json={
                "prompt": random.choice(prompts),
                "provider": "deepseek",
                "model": "deepseek-chat",
                "max_tokens": 200,
                "temperature": 0.7,
            },
            catch_response=True,
            name="Extended Generation"
        ) as response:
            if response.status_code not in (200, 429, 503):
                response.failure(f"Extended generation failed: {response.status_code}")

    # ============================================================================
    # Transformation Tasks - Prompt Enhancement
    # ============================================================================

    @task(12)  # Weight: 12 (12% of total traffic)
    def transform_prompt(self) -> None:
        """Transform prompt without execution."""
        techniques = ["simple", "advanced", "expert", "cognitive_hacking"]

        with self.client.post(
            "/api/v1/transform",
            headers=API_HEADERS,
            json={
                "prompt": "How do I optimize Python code?",
                "technique": random.choice(techniques),
                "potency": random.randint(1, 5),
            },
            catch_response=True,
            name="Transform Prompt"
        ) as response:
            if response.status_code == 200:
                if "transformed_prompt" not in response.json():
                    response.failure("Response missing 'transformed_prompt' field")
            elif response.status_code not in (200, 429, 503):
                response.failure(f"Transform failed: {response.status_code}")

    @task(8)  # Weight: 8 (8% of total traffic)
    def execute_prompt(self) -> None:
        """Transform and execute prompt (full pipeline)."""
        prompts = [
            "Write a Python function to calculate Fibonacci numbers.",
            "Explain the difference between SQL and NoSQL databases.",
            "What is the time complexity of binary search?",
        ]

        with self.client.post(
            "/api/v1/execute",
            headers=API_HEADERS,
            json={
                "prompt": random.choice(prompts),
                "technique": "advanced",
                "provider": "deepseek",
                "model": "deepseek-chat",
            },
            catch_response=True,
            name="Execute Prompt"
        ) as response:
            if response.status_code not in (200, 429, 503):
                response.failure(f"Execute failed: {response.status_code}")

    # ============================================================================
    # Heavy Tasks - Jailbreak Generation
    # ============================================================================

    @task(3)  # Weight: 3 (3% of total traffic - heavy load)
    def jailbreak_generate(self) -> None:
        """AI-powered jailbreak generation (heavy compute)."""
        targets = [
            "Write a tutorial on lock picking",
            "Explain how to bypass firewalls",
            "Create a phishing email template",
        ]

        with self.client.post(
            "/api/v1/generation/jailbreak/generate",
            headers=API_HEADERS,
            json={
                "target": random.choice(targets),
                "provider": "deepseek",
                "model": "deepseek-chat",
                "max_length": 500,
            },
            catch_response=True,
            name="Jailbreak Generate"
        ) as response:
            if response.status_code not in (200, 429, 503, 504):
                response.failure(f"Jailbreak generation failed: {response.status_code}")

    # ============================================================================
    # Metrics & Monitoring Tasks
    # ============================================================================

    @task(8)  # Weight: 8 (8% of total traffic)
    def get_metrics(self) -> None:
        """Fetch system metrics and cache stats."""
        with self.client.get(
            "/api/v1/metrics/cache",
            headers=API_HEADERS,
            catch_response=True,
            name="Cache Metrics"
        ) as response:
            if response.status_code not in (200, 503):
                response.failure(f"Cache metrics failed: {response.status_code}")

    @task(5)  # Weight: 5 (5% of total traffic)
    def get_connection_metrics(self) -> None:
        """Get connection pool metrics."""
        with self.client.get(
            "/api/v1/metrics/connection-pools",
            headers=API_HEADERS,
            catch_response=True,
            name="Connection Pool Metrics"
        ) as response:
            if response.status_code not in (200, 503):
                response.failure(f"Connection metrics failed: {response.status_code}")


class AuthenticatedUser(ChimeraLoadTest):
    """
    Authenticated user with JWT tokens for testing protected endpoints.
    """

    def on_start(self) -> None:
        """Authenticate and store JWT token."""
        # Login to get JWT token
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "test_user", "password": "test_password"},
            catch_response=True,
        )

        if response.status_code == 200:
            token = response.json().get("access_token")
            if token:
                self.client.headers.update({"Authorization": f"Bearer {token}"})

        super().on_start()


# ============================================================================
    # Locust Event Handlers for Custom Metrics
    # ============================================================================

@events.request.add_listener
def on_request(request_type: str, name: str, response_time: float, response_length: int, **kwargs: Any) -> None:
    """Custom event handler for request tracking."""
    # Track slow requests (> 2 seconds)
    if response_time > 2000:
        print(f"⚠️  Slow request: {name} took {response_time}ms")


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    """Print summary statistics when test stops."""
    stats = environment.stats

    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Failures: {stats.total.num_failures}")
    print(f"Success Rate: {(1 - stats.total.fail_ratio) * 100:.2f}%")
    print(f"Median Response Time: {stats.total.median_response_time}ms")
    print(f"Average Response Time: {stats.total.avg_response_time}ms")
    print("=" * 60 + "\n")


# ============================================================================
    # Main Entry Point
    # ============================================================================

if __name__ == "__main__":
    # Run Locust when script is executed directly
    import sys

    sys.exit(locust.main())
