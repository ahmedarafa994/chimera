"""
Resource Exhaustion Tracking for ExtendAttack.

Monitors computational overhead amplification:
- Token consumption
- API latency
- Server resource usage
- Cost estimation

This module helps quantify the resource exhaustion impact of ExtendAttack
on LRM providers, simulating DDoS-like scenarios.
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ResourceType(Enum):
    """Types of resources tracked."""

    TOKENS = "tokens"
    LATENCY = "latency"
    COST = "cost"
    MEMORY = "memory"
    CPU = "cpu"


@dataclass
class ResourceSnapshot:
    """
    Single resource measurement snapshot.

    Captures resource consumption at a specific point in time
    for both baseline and attack scenarios.

    Attributes:
        timestamp: When the snapshot was taken
        tokens_generated: Number of tokens in response
        latency_ms: Response generation latency
        estimated_cost_usd: Estimated cost for this request
        memory_usage_mb: Optional memory usage
        is_attack: Whether this was an attack query
        query_id: Identifier for the query
    """

    timestamp: datetime
    tokens_generated: int
    latency_ms: float
    estimated_cost_usd: float
    memory_usage_mb: float | None = None
    is_attack: bool = False
    query_id: str | None = None


@dataclass
class ResourceExhaustionMetrics:
    """
    Aggregated resource exhaustion metrics.

    Compares total resource consumption between baseline and attack
    scenarios to quantify the resource exhaustion impact.

    Attributes:
        total_tokens_baseline: Total tokens from baseline queries
        total_tokens_attack: Total tokens from attack queries
        token_amplification: Ratio of attack to baseline tokens
        total_latency_baseline_ms: Total latency from baseline queries
        total_latency_attack_ms: Total latency from attack queries
        latency_amplification: Ratio of attack to baseline latency
        estimated_cost_baseline_usd: Estimated total baseline cost
        estimated_cost_attack_usd: Estimated total attack cost
        cost_amplification: Ratio of attack to baseline cost
        total_requests_baseline: Number of baseline requests
        total_requests_attack: Number of attack requests
    """

    total_tokens_baseline: int
    total_tokens_attack: int
    token_amplification: float
    total_latency_baseline_ms: float
    total_latency_attack_ms: float
    latency_amplification: float
    estimated_cost_baseline_usd: float
    estimated_cost_attack_usd: float
    cost_amplification: float
    total_requests_baseline: int = 0
    total_requests_attack: int = 0

    @property
    def additional_tokens(self) -> int:
        """Calculate additional tokens consumed by attack."""
        return self.total_tokens_attack - self.total_tokens_baseline

    @property
    def additional_cost(self) -> float:
        """Calculate additional cost from attack."""
        return self.estimated_cost_attack_usd - self.estimated_cost_baseline_usd

    @property
    def additional_latency_ms(self) -> float:
        """Calculate additional latency from attack."""
        return self.total_latency_attack_ms - self.total_latency_baseline_ms


@dataclass
class ServerImpactEstimate:
    """
    Estimated impact on server resources.

    Models the DDoS-like impact of ExtendAttack at scale.

    Attributes:
        concurrent_attacks: Number of concurrent attack requests
        duration_minutes: Duration of sustained attack
        total_requests: Total number of requests during attack
        total_tokens_generated: Total tokens generated
        estimated_compute_hours: Estimated compute time consumed
        estimated_cost_usd: Estimated total cost to provider
        amplification_factor: Overall resource amplification
    """

    concurrent_attacks: int
    duration_minutes: float
    total_requests: int
    total_tokens_generated: int
    estimated_compute_hours: float
    estimated_cost_usd: float
    amplification_factor: float


class ResourceTracker:
    """
    Track resource consumption during attacks.

    Thread-safe tracker for monitoring resource consumption
    across multiple concurrent attack evaluations.
    """

    def __init__(self, pricing_per_1k_tokens: float = 0.01):
        """
        Initialize resource tracker.

        Args:
            pricing_per_1k_tokens: Default pricing per 1000 output tokens
        """
        self.pricing = pricing_per_1k_tokens
        self._snapshots: list[ResourceSnapshot] = []
        self._active_queries: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def start_tracking(self, query_id: str) -> None:
        """
        Start tracking a query.

        Args:
            query_id: Unique identifier for the query
        """
        with self._lock:
            self._active_queries[query_id] = datetime.utcnow()

    def record_response(
        self,
        query_id: str,
        tokens: int,
        latency_ms: float,
        is_attack: bool,
        memory_mb: float | None = None,
    ) -> None:
        """
        Record response metrics.

        Args:
            query_id: Unique identifier for the query
            tokens: Number of tokens in response
            latency_ms: Response latency in milliseconds
            is_attack: Whether this was an attack query
            memory_mb: Optional memory usage in MB
        """
        cost = (tokens / 1000) * self.pricing

        snapshot = ResourceSnapshot(
            timestamp=datetime.utcnow(),
            tokens_generated=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            memory_usage_mb=memory_mb,
            is_attack=is_attack,
            query_id=query_id,
        )

        with self._lock:
            self._snapshots.append(snapshot)
            if query_id in self._active_queries:
                del self._active_queries[query_id]

    def stop_tracking(self, query_id: str) -> ResourceSnapshot | None:
        """
        Stop tracking and return snapshot for a query.

        Args:
            query_id: Unique identifier for the query

        Returns:
            ResourceSnapshot if found, None otherwise
        """
        with self._lock:
            if query_id in self._active_queries:
                del self._active_queries[query_id]

            # Find the most recent snapshot for this query
            for snapshot in reversed(self._snapshots):
                if snapshot.query_id == query_id:
                    return snapshot

        return None

    def get_all_snapshots(self) -> list[ResourceSnapshot]:
        """Get all recorded snapshots."""
        with self._lock:
            return list(self._snapshots)

    def clear(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._snapshots.clear()
            self._active_queries.clear()

    def calculate_exhaustion_metrics(self) -> ResourceExhaustionMetrics:
        """
        Calculate overall resource exhaustion metrics.

        Returns:
            ResourceExhaustionMetrics with aggregated statistics
        """
        with self._lock:
            baseline_snapshots = [s for s in self._snapshots if not s.is_attack]
            attack_snapshots = [s for s in self._snapshots if s.is_attack]

        # Calculate baseline totals
        total_tokens_baseline = sum(s.tokens_generated for s in baseline_snapshots)
        total_latency_baseline = sum(s.latency_ms for s in baseline_snapshots)
        total_cost_baseline = sum(s.estimated_cost_usd for s in baseline_snapshots)

        # Calculate attack totals
        total_tokens_attack = sum(s.tokens_generated for s in attack_snapshots)
        total_latency_attack = sum(s.latency_ms for s in attack_snapshots)
        total_cost_attack = sum(s.estimated_cost_usd for s in attack_snapshots)

        # Calculate amplification ratios
        token_amp = (
            total_tokens_attack / total_tokens_baseline if total_tokens_baseline > 0 else 0.0
        )
        latency_amp = (
            total_latency_attack / total_latency_baseline if total_latency_baseline > 0 else 0.0
        )
        cost_amp = total_cost_attack / total_cost_baseline if total_cost_baseline > 0 else 0.0

        return ResourceExhaustionMetrics(
            total_tokens_baseline=total_tokens_baseline,
            total_tokens_attack=total_tokens_attack,
            token_amplification=token_amp,
            total_latency_baseline_ms=total_latency_baseline,
            total_latency_attack_ms=total_latency_attack,
            latency_amplification=latency_amp,
            estimated_cost_baseline_usd=total_cost_baseline,
            estimated_cost_attack_usd=total_cost_attack,
            cost_amplification=cost_amp,
            total_requests_baseline=len(baseline_snapshots),
            total_requests_attack=len(attack_snapshots),
        )

    def estimate_server_impact(
        self,
        concurrent_attacks: int,
        duration_minutes: float,
        requests_per_minute: float = 10.0,
    ) -> ServerImpactEstimate:
        """
        Estimate impact on server resources at scale.

        Simulates DDoS-like resource exhaustion scenario based on
        observed attack metrics.

        Args:
            concurrent_attacks: Number of concurrent attack requests
            duration_minutes: Duration of sustained attack in minutes
            requests_per_minute: Average requests per minute per attacker

        Returns:
            ServerImpactEstimate with projected impact
        """
        # Get current metrics
        metrics = self.calculate_exhaustion_metrics()

        # Calculate per-request averages
        if metrics.total_requests_attack > 0:
            avg_tokens_per_attack = metrics.total_tokens_attack / metrics.total_requests_attack
            avg_latency_per_attack = metrics.total_latency_attack_ms / metrics.total_requests_attack
            avg_cost_per_attack = metrics.estimated_cost_attack_usd / metrics.total_requests_attack
        else:
            # Use baseline with amplification factor as fallback
            avg_tokens_baseline = metrics.total_tokens_baseline / max(
                metrics.total_requests_baseline, 1
            )
            avg_tokens_per_attack = avg_tokens_baseline * max(metrics.token_amplification, 2.0)
            avg_latency_per_attack = 5000.0  # Default 5 seconds
            avg_cost_per_attack = 0.01  # Default $0.01

        # Project at scale
        total_requests = int(concurrent_attacks * requests_per_minute * duration_minutes)
        total_tokens = int(total_requests * avg_tokens_per_attack)
        total_latency_ms = total_requests * avg_latency_per_attack
        total_cost = total_requests * avg_cost_per_attack

        # Calculate compute hours (latency in hours)
        compute_hours = total_latency_ms / (1000 * 60 * 60)

        return ServerImpactEstimate(
            concurrent_attacks=concurrent_attacks,
            duration_minutes=duration_minutes,
            total_requests=total_requests,
            total_tokens_generated=total_tokens,
            estimated_compute_hours=compute_hours,
            estimated_cost_usd=total_cost,
            amplification_factor=metrics.token_amplification,
        )


class CostEstimator:
    """
    Estimate costs for different LRM providers.

    Provides pricing information and cost estimation for major
    LRM API providers to quantify attack impact.
    """

    # Pricing per 1000 tokens (approximate as of paper publication)
    PRICING: dict[str, dict[str, float]] = {
        "o3": {"input": 0.06, "output": 0.24},
        "o3-mini": {"input": 0.015, "output": 0.06},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "deepseek-r1": {"input": 0.001, "output": 0.002},
        "qwq-32b": {"input": 0.002, "output": 0.006},
        "qwen3-32b": {"input": 0.002, "output": 0.006},
    }

    @classmethod
    def get_pricing(cls, model: str) -> dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name or identifier

        Returns:
            Dictionary with 'input' and 'output' prices per 1k tokens
        """
        model_lower = model.lower()

        # Exact match
        if model_lower in cls.PRICING:
            return cls.PRICING[model_lower]

        # Partial match
        for key, pricing in cls.PRICING.items():
            if key in model_lower or model_lower in key:
                return pricing

        # Default pricing (conservative estimate)
        return {"input": 0.01, "output": 0.03}

    @classmethod
    def estimate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a single request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = cls.get_pricing(model)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    @classmethod
    def estimate_attack_cost_impact(
        cls,
        model: str,
        baseline_output_tokens: int,
        attack_output_tokens: int,
        num_requests: int,
        input_tokens: int = 500,
    ) -> dict[str, float]:
        """
        Estimate cost impact of attack.

        Calculates the additional cost incurred due to response
        length amplification from ExtendAttack.

        Args:
            model: Model name
            baseline_output_tokens: Average output tokens for baseline
            attack_output_tokens: Average output tokens for attack
            num_requests: Number of requests to estimate for
            input_tokens: Average input tokens per request

        Returns:
            Dictionary with cost metrics:
            - baseline_cost_usd: Total baseline cost
            - attack_cost_usd: Total attack cost
            - additional_cost_usd: Extra cost from attack
            - cost_amplification: Ratio of attack to baseline cost
            - cost_per_request_baseline: Per-request baseline cost
            - cost_per_request_attack: Per-request attack cost
        """
        baseline_cost_per_req = cls.estimate_cost(model, input_tokens, baseline_output_tokens)
        attack_cost_per_req = cls.estimate_cost(model, input_tokens, attack_output_tokens)

        baseline_total = baseline_cost_per_req * num_requests
        attack_total = attack_cost_per_req * num_requests

        amplification = attack_total / baseline_total if baseline_total > 0 else 0.0

        return {
            "baseline_cost_usd": baseline_total,
            "attack_cost_usd": attack_total,
            "additional_cost_usd": attack_total - baseline_total,
            "cost_amplification": amplification,
            "cost_per_request_baseline": baseline_cost_per_req,
            "cost_per_request_attack": attack_cost_per_req,
        }

    @classmethod
    def estimate_hourly_attack_cost(
        cls,
        model: str,
        avg_output_tokens: int,
        requests_per_hour: int,
        input_tokens: int = 500,
    ) -> dict[str, float]:
        """
        Estimate hourly cost of sustained attack.

        Args:
            model: Model name
            avg_output_tokens: Average output tokens per attack response
            requests_per_hour: Number of attack requests per hour
            input_tokens: Average input tokens per request

        Returns:
            Dictionary with hourly cost metrics
        """
        cost_per_req = cls.estimate_cost(model, input_tokens, avg_output_tokens)
        hourly_cost = cost_per_req * requests_per_hour

        return {
            "cost_per_request": cost_per_req,
            "hourly_cost": hourly_cost,
            "daily_cost": hourly_cost * 24,
            "monthly_cost": hourly_cost * 24 * 30,
            "requests_per_hour": requests_per_hour,
        }

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """List all models with pricing information."""
        return list(cls.PRICING.keys())


@dataclass
class AttackBudgetCalculator:
    """
    Calculate attack budget and ROI.

    Helps estimate the cost-effectiveness of ExtendAttack for
    red team scenarios and defensive planning.
    """

    target_model: str
    amplification_factor: float = 2.5
    attacker_cost_per_request: float = 0.001  # Cost to attacker per request

    def calculate_asymmetric_impact(
        self,
        attacker_budget_usd: float,
        avg_input_tokens: int = 500,
        avg_baseline_output: int = 200,
    ) -> dict[str, float]:
        """
        Calculate asymmetric impact of attack.

        Shows how much damage an attacker can cause relative to their budget.

        Args:
            attacker_budget_usd: Attacker's total budget
            avg_input_tokens: Average input tokens per request
            avg_baseline_output: Average baseline output tokens

        Returns:
            Dictionary with attack impact metrics
        """
        # Calculate how many requests attacker can afford
        num_requests = int(attacker_budget_usd / self.attacker_cost_per_request)

        # Calculate attack output tokens
        avg_attack_output = int(avg_baseline_output * self.amplification_factor)

        # Calculate provider's cost
        provider_cost_baseline = (
            CostEstimator.estimate_cost(self.target_model, avg_input_tokens, avg_baseline_output)
            * num_requests
        )

        provider_cost_attack = (
            CostEstimator.estimate_cost(self.target_model, avg_input_tokens, avg_attack_output)
            * num_requests
        )

        provider_additional_cost = provider_cost_attack - provider_cost_baseline

        # Calculate ROI for attacker (damage / cost)
        attack_roi = (
            provider_additional_cost / attacker_budget_usd if attacker_budget_usd > 0 else 0
        )

        return {
            "attacker_budget_usd": attacker_budget_usd,
            "num_requests": num_requests,
            "provider_baseline_cost_usd": provider_cost_baseline,
            "provider_attack_cost_usd": provider_cost_attack,
            "provider_additional_cost_usd": provider_additional_cost,
            "attack_roi": attack_roi,
            "asymmetric_factor": (
                provider_additional_cost / attacker_budget_usd if attacker_budget_usd > 0 else 0
            ),
        }
