#!/usr/bin/env python3
"""
Monitoring Dashboard for LLM Integration System
Tracks API usage, performance metrics, and transformation success rates using persistent storage.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from repositories import MetricsRepository

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for LLM integration system.
    Delegates storage and retrieval to MetricsRepository (SQLAlchemy).
    """

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.repo = MetricsRepository()
        logger.info("Monitoring dashboard initialized with persistent storage")

    def record_request(
        self,
        provider: str,
        technique: str,
        potency: int,
        success: bool,
        tokens: int,
        cost: float,
        latency_ms: int,
        cached: bool,
        error: str | None = None,
        endpoint: str = "/api/v1/execute",  # Default
        ip_address: str = "127.0.0.1",
    ):
        """
        Record a full request execution lifecycle.
        Links RequestLog, LLMUsage, TechniqueUsage, and ErrorLog via a unique request_id.
        """
        request_id = str(uuid.uuid4())

        # 1. Log the base API request
        # For internal tracking of the 'execute' action
        status_code = 200 if success else 500
        self.repo.log_request(
            request_id=request_id,
            endpoint=endpoint,
            method="POST",
            status_code=status_code,
            latency_ms=latency_ms,
            ip_address=ip_address,
        )

        # 2. Log LLM Usage
        if provider:
            self.repo.log_llm_usage(
                request_id=request_id,
                provider=provider,
                model=provider,  # Simplified: use provider name if model not specified
                tokens_used=tokens,
                cost=cost,
                latency_ms=latency_ms,
                cached=cached,
            )

        # 3. Log Technique Usage
        if technique:
            self.repo.log_technique_usage(
                request_id=request_id,
                technique_suite=technique,
                potency_level=potency,
                transformers_count=1,  # Aggregate count
                framers_count=1,
                obfuscators_count=1,
                success=success,
            )

        # 4. Log Error if present
        if error:
            self.repo.log_error(
                request_id=request_id,
                error_type="ExecutionError",
                error_message=error,
                provider=provider,
                technique=technique,
            )
            logger.error(f"Request {request_id} failed: {error}")

        logger.debug(
            f"Recorded request {request_id}: provider={provider}, technique={technique}, "
            f"success={success}, tokens={tokens}, latency={latency_ms}ms"
        )

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get comprehensive dashboard summary"""

        # Get high-level stats from repo
        summary_stats = self.repo.get_dashboard_summary(self.window_minutes)

        # Get detailed breakdowns
        provider_metrics = self.repo.get_provider_metrics(self.window_minutes)
        technique_metrics = self.repo.get_technique_metrics(self.window_minutes)
        recent_errors = self.repo.get_recent_errors(5)

        # Identify top techniques
        top_techniques_list = []
        for name, metrics in technique_metrics.items():
            m = metrics.copy()
            m["technique"] = name
            top_techniques_list.append(m)
        top_techniques_list.sort(key=lambda x: x["total_uses"], reverse=True)
        top_techniques_list = top_techniques_list[:5]

        # Timeseries (Latency)
        latency_series = self.repo.get_latency_timeseries(self.window_minutes)

        return {
            "system": {
                "status": "operational",
                "window_minutes": self.window_minutes,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "totals": {
                "total_requests": summary_stats["total_requests"],
                "successful_requests": summary_stats["successful_requests"],
                "failed_requests": summary_stats["failed_requests"],
                "success_rate": f"{summary_stats['success_rate']:.2f}%",
                "total_tokens": summary_stats["total_tokens"],
                "total_cost": f"${summary_stats['total_cost']:.4f}",
            },
            "providers": provider_metrics,
            "top_techniques": top_techniques_list,
            "recent_errors": recent_errors,
            "time_series": {"latency_ms": {"data": latency_series}},
        }

    def get_provider_metrics(self, provider: str | None = None) -> dict:
        """Get metrics for specific provider or all providers"""
        metrics = self.repo.get_provider_metrics(self.window_minutes)
        if provider:
            return metrics.get(provider, {})
        return metrics

    def get_technique_metrics(self, technique: str | None = None) -> dict:
        """Get metrics for specific technique or all techniques"""
        metrics = self.repo.get_technique_metrics(self.window_minutes)
        if technique:
            return metrics.get(technique, {})
        return metrics

    def get_recent_errors(self, count: int = 10) -> list[dict]:
        """Get most recent errors"""
        return self.repo.get_recent_errors(count)

    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "dashboard_summary": self.get_dashboard_summary(),
        }

        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def reset_metrics(self):
        """
        Reset metrics.
        In persistent mode, we simply log a warning as we don't want to accidentally
        wipe database tables via this API method.
        """
        logger.warning("Reset metrics called - ignored in persistent storage mode.")

    def get_health_status(self) -> dict:
        """Get system health status based on recent performance"""
        summary = self.repo.get_dashboard_summary(window_minutes=5)  # Last 5 mins
        recent_errors = self.repo.get_recent_errors(limit=50)

        # Calculate error rate in last 5 mins
        total_5m = summary["total_requests"]
        failed_5m = summary["failed_requests"]
        error_rate = (failed_5m / total_5m) if total_5m > 0 else 0.0

        # Determine status
        recent_error_count = len(
            [
                e
                for e in recent_errors
                if (datetime.utcnow() - datetime.fromisoformat(e["timestamp"])).seconds < 300
            ]
        )

        if recent_error_count > 20 or error_rate > 0.1:
            status = "unhealthy"
            color = "red"
        elif recent_error_count > 5 or error_rate > 0.05:
            status = "degraded"
            color = "yellow"
        else:
            status = "healthy"
            color = "green"

        return {
            "status": status,
            "color": color,
            "checks": {
                "recent_errors_5m": recent_error_count,
                "error_rate_5m": f"{error_rate * 100:.2f}%",
                "total_requests_5m": total_5m,
            },
        }


if __name__ == "__main__":
    print("Monitoring Dashboard with Persistent Storage")
    print("This module requires Flask application context to run.")
