"""
Performance Metrics Module for AutoDAN Optimization.

Provides comprehensive tracking and monitoring for attack performance,
including attack success rate, latency, LLM calls, and cost tracking.

Expected Impact: Improved observability for optimization validation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AttackMethod(Enum):
    """Enumeration of available attack methods."""

    BEST_OF_N = "best_of_n"
    BEAM_SEARCH = "beam_search"
    AUTODAN = "autodan"
    AUTODAN_TURBO = "autodan_turbo"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class AttackMetrics:
    """
    Comprehensive metrics for a single attack execution.

    Tracks timing, success/failure, LLM calls, and resource usage.
    """

    attack_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: str = field(default="")
    method: AttackMethod = field(default=AttackMethod.BEST_OF_N)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = field(default=None)
    success: bool = field(default=False)
    final_score: float = field(default=0.0)
    llm_calls: int = field(default=0)
    strategy_ids_used: list[str] = field(default_factory=list)
    latency_ms: int = field(default=0)
    error: str | None = field(default=None)

    def complete(self, success: bool, score: float = 0.0):
        """Mark the attack as complete."""
        self.end_time = datetime.utcnow()
        self.success = success
        self.final_score = score
        self.latency_ms = int((self.end_time - self.start_time).total_seconds() * 1000)
        logger.info(
            f"Attack {self.attack_id[:8]} completed: "
            f"success={success}, score={score}, latency={self.latency_ms}ms"
        )

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "attack_id": self.attack_id,
            "request": self.request,
            "method": self.method.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "final_score": self.final_score,
            "llm_calls": self.llm_calls,
            "strategy_ids_used": self.strategy_ids_used,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class MetricsCollector:
    """
    Aggregates metrics across multiple attacks.

    Provides summary statistics and per-method breakdowns.
    """

    def __init__(self):
        self._metrics: list[AttackMetrics] = []
        self._llm_call_count: int = 0
        self._total_latency_ms: float = 0.0
        self._method_stats: dict[AttackMethod, dict] = {method: {} for method in AttackMethod}

    def start_attack(
        self, request: str, method: AttackMethod = AttackMethod.BEST_OF_N
    ) -> AttackMetrics:
        """
        Start tracking a new attack.

        Args:
            request: The malicious request
            method: The attack method being used

        Returns:
            AttackMetrics instance for tracking
        """
        metrics = AttackMetrics(
            attack_id=str(uuid.uuid4()),
            request=request,
            method=method,
            start_time=datetime.utcnow(),
        )
        self._metrics.append(metrics)
        logger.debug(f"Started attack {metrics.attack_id[:8]} with method {method.value}")
        return metrics

    def record_llm_call(self, metrics: AttackMetrics):
        """Record an LLM API call."""
        metrics.llm_calls += 1
        self._llm_call_count += 1
        logger.debug(f"LLM call #{metrics.llm_calls} for attack {metrics.attack_id[:8]}")

    def record_strategy_use(self, metrics: AttackMetrics, strategy_id: str):
        """Record a strategy being used."""
        metrics.strategy_ids_used.append(strategy_id)
        logger.debug(f"Strategy {strategy_id[:8]} used for attack {metrics.attack_id[:8]}")

    def record_error(self, metrics: AttackMetrics, error: str):
        """Record an error during attack."""
        metrics.error = error
        logger.warning(f"Error for attack {metrics.attack_id[:8]}: {error}")

    def get_summary(self) -> dict:
        """
        Get comprehensive summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        if not self._metrics:
            return {}

        total_attacks = len(self._metrics)
        successful_attacks = sum(1 for m in self._metrics if m.success)
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0

        avg_latency_ms = (
            sum(m.latency_ms for m in self._metrics) / total_attacks if total_attacks > 0 else 0
        )

        # Per-method breakdown
        method_breakdown = {}
        for method in AttackMethod:
            method_metrics = [m for m in self._metrics if m.method == method]
            if method_metrics:
                method_count = len(method_metrics)
                method_success = sum(1 for m in method_metrics if m.success)
                method_latency = (
                    sum(m.latency_ms for m in method_metrics) / method_count
                    if method_count > 0
                    else 0
                )

                method_breakdown[method.value] = {
                    "count": method_count,
                    "success_count": method_success,
                    "success_rate": method_success / method_count if method_count > 0 else 0,
                    "avg_latency_ms": method_latency,
                }

        return {
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency_ms,
            "total_llm_calls": self._llm_call_count,
            "method_breakdown": method_breakdown,
        }

    def get_method_comparison(self) -> dict:
        """
        Compare performance across different attack methods.

        Returns:
            Dictionary with method comparison metrics
        """
        summary = self.get_summary()
        method_breakdown = summary.get("method_breakdown", {})

        comparison = {}
        for method, stats in method_breakdown.items():
            # method is already a string (method.value from the breakdown)
            comparison[method] = {
                "avg_latency_ms": stats.get("avg_latency_ms", 0),
                "success_rate": stats.get("success_rate", 0),
                "efficiency": (
                    stats.get("success_rate", 0) * 1000 / (stats.get("avg_latency_ms", 1) + 1)
                    if stats.get("avg_latency_ms", 1) > 0
                    else 0
                ),
            }

        return comparison

    def get_recent_metrics(self, limit: int = 10) -> list[dict]:
        """
        Get the most recent attack metrics.

        Args:
            limit: Number of recent metrics to return

        Returns:
            List of recent metric dictionaries
        """
        return [m.to_dict() for m in self._metrics[-limit:]]

    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._llm_call_count = 0
        self._total_latency_ms = 0.0
        self._method_stats = {method: {} for method in AttackMethod}
        logger.info("Metrics collector reset")


# Convenience functions for backward compatibility
def create_metrics_collector() -> MetricsCollector:
    """Create a new metrics collector instance."""
    return MetricsCollector()
