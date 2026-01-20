"""Cost Tracking Middleware.

Middleware to track costs of AI operations:
1. Records request start time
2. Processes request
3. Extracts token usage from response
4. Calculates cost using config pricing
5. Tracks cumulative costs and alerts
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Individual cost entry for tracking."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime
    request_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


@dataclass
class CostAlert:
    """Cost alert when threshold is exceeded."""

    alert_type: str
    provider: str | None
    threshold: float
    current_value: float
    timestamp: datetime
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "provider": self.provider,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }


class CostTracker:
    """Track and aggregate costs for AI operations.

    Features:
    - Record costs per provider/model
    - Aggregate costs over time periods
    - Track cumulative costs
    - Generate alerts when thresholds exceeded
    """

    def __init__(
        self,
        daily_budget: float | None = None,
        provider_budgets: dict[str, float] | None = None,
        alert_threshold_percent: float = 80.0,
        max_entries: int = 10000,
    ) -> None:
        """Initialize cost tracker.

        Args:
            daily_budget: Daily budget limit (USD)
            provider_budgets: Per-provider budget limits
            alert_threshold_percent: Percentage of budget to trigger alert
            max_entries: Maximum cost entries to keep in memory

        """
        self._entries: list[CostEntry] = []
        self._cumulative_by_provider: dict[str, float] = defaultdict(float)
        self._cumulative_by_model: dict[str, float] = defaultdict(float)
        self._daily_costs: dict[str, float] = defaultdict(float)
        self._lock = Lock()

        self._daily_budget = daily_budget
        self._provider_budgets = provider_budgets or {}
        self._alert_threshold = alert_threshold_percent / 100.0
        self._max_entries = max_entries
        self._alerts: list[CostAlert] = []
        self._config_manager = None

    def _get_config_manager(self):
        """Lazily get the AI config manager."""
        if self._config_manager is None:
            try:
                from app.core.ai_config_manager import get_ai_config_manager

                self._config_manager = get_ai_config_manager()
            except ImportError:
                logger.warning("AIConfigManager not available for cost tracking")
        return self._config_manager

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for a request using config pricing.

        Args:
            provider: Provider ID
            model: Model ID
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            reasoning_tokens: Number of reasoning tokens (optional)
            cached_tokens: Number of cached tokens (optional)

        Returns:
            Calculated cost in USD

        """
        config_manager = self._get_config_manager()
        if config_manager and config_manager.is_loaded():
            try:
                return config_manager.calculate_cost(
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    reasoning_tokens,
                    cached_tokens,
                )
            except Exception as e:
                logger.warning(f"Cost calculation error: {e}")

        # Default pricing if config not available
        # Approximate pricing per 1K tokens
        default_input_cost = 0.001  # $0.001 per 1K input tokens
        default_output_cost = 0.002  # $0.002 per 1K output tokens

        cost = (input_tokens / 1000) * default_input_cost + (
            output_tokens / 1000
        ) * default_output_cost

        return round(cost, 6)

    def record_cost(
        self,
        provider: str,
        model: str,
        cost: float,
        metadata: dict | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        request_id: str | None = None,
    ) -> None:
        """Record a cost entry.

        Args:
            provider: Provider ID
            model: Model ID
            cost: Cost in USD
            metadata: Optional metadata
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_id: Optional request ID for correlation

        """
        timestamp = datetime.utcnow()
        day_key = timestamp.strftime("%Y-%m-%d")

        entry = CostEntry(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=timestamp,
            request_id=request_id,
            metadata=metadata or {},
        )

        with self._lock:
            # Add entry
            self._entries.append(entry)

            # Update cumulative costs
            self._cumulative_by_provider[provider] += cost
            self._cumulative_by_model[f"{provider}:{model}"] += cost
            self._daily_costs[day_key] += cost

            # Enforce max entries limit
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

            # Check for alerts
            self._check_alerts(provider, day_key)

        logger.debug(
            f"Recorded cost: provider={provider}, model={model}, "
            f"cost=${cost:.6f}, tokens={input_tokens}+{output_tokens}",
        )

    def _check_alerts(self, provider: str, day_key: str) -> None:
        """Check and generate cost alerts."""
        current_time = datetime.utcnow()

        # Check daily budget
        if self._daily_budget:
            daily_cost = self._daily_costs.get(day_key, 0)
            threshold_value = self._daily_budget * self._alert_threshold

            if daily_cost >= threshold_value:
                alert = CostAlert(
                    alert_type="daily_budget_warning",
                    provider=None,
                    threshold=self._daily_budget,
                    current_value=daily_cost,
                    timestamp=current_time,
                    message=(
                        f"Daily cost ${daily_cost:.2f} has reached "
                        f"{(daily_cost / self._daily_budget) * 100:.1f}% "
                        f"of budget ${self._daily_budget:.2f}"
                    ),
                )
                self._add_alert(alert)

            if daily_cost >= self._daily_budget:
                alert = CostAlert(
                    alert_type="daily_budget_exceeded",
                    provider=None,
                    threshold=self._daily_budget,
                    current_value=daily_cost,
                    timestamp=current_time,
                    message=(
                        f"Daily budget EXCEEDED: ${daily_cost:.2f} / ${self._daily_budget:.2f}"
                    ),
                )
                self._add_alert(alert)

        # Check provider budget
        if provider in self._provider_budgets:
            budget = self._provider_budgets[provider]
            provider_cost = self._cumulative_by_provider.get(provider, 0)
            threshold_value = budget * self._alert_threshold

            if provider_cost >= threshold_value:
                alert = CostAlert(
                    alert_type="provider_budget_warning",
                    provider=provider,
                    threshold=budget,
                    current_value=provider_cost,
                    timestamp=current_time,
                    message=(
                        f"Provider '{provider}' cost ${provider_cost:.2f} "
                        f"has reached {(provider_cost / budget) * 100:.1f}% "
                        f"of budget ${budget:.2f}"
                    ),
                )
                self._add_alert(alert)

    def _add_alert(self, alert: CostAlert) -> None:
        """Add alert if not duplicate."""
        # Avoid duplicate alerts within 5 minutes
        for existing in self._alerts[-10:]:
            if (
                existing.alert_type == alert.alert_type
                and existing.provider == alert.provider
                and (alert.timestamp - existing.timestamp) < timedelta(minutes=5)
            ):
                return

        self._alerts.append(alert)
        logger.warning(f"Cost alert: {alert.message}")

    def get_costs_by_provider(
        self,
        since: datetime | None = None,
    ) -> dict[str, float]:
        """Get aggregated costs by provider.

        Args:
            since: Optional start datetime to filter

        Returns:
            Dict mapping provider to total cost

        """
        with self._lock:
            if not since:
                return dict(self._cumulative_by_provider)

            # Filter entries by time
            costs: dict[str, float] = defaultdict(float)
            for entry in self._entries:
                if entry.timestamp >= since:
                    costs[entry.provider] += entry.cost

            return dict(costs)

    def get_costs_by_model(
        self,
        since: datetime | None = None,
    ) -> dict[str, float]:
        """Get aggregated costs by model.

        Args:
            since: Optional start datetime to filter

        Returns:
            Dict mapping provider:model to total cost

        """
        with self._lock:
            if not since:
                return dict(self._cumulative_by_model)

            # Filter entries by time
            costs: dict[str, float] = defaultdict(float)
            for entry in self._entries:
                if entry.timestamp >= since:
                    key = f"{entry.provider}:{entry.model}"
                    costs[key] += entry.cost

            return dict(costs)

    def get_daily_costs(
        self,
        days: int = 7,
    ) -> dict[str, float]:
        """Get daily costs for the last N days.

        Args:
            days: Number of days to retrieve

        Returns:
            Dict mapping date string to total cost

        """
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(days=days)
            cutoff_key = cutoff.strftime("%Y-%m-%d")

            return {k: v for k, v in self._daily_costs.items() if k >= cutoff_key}

    def get_cost_alerts(self) -> list[dict[str, Any]]:
        """Check for cost threshold alerts.

        Returns:
            List of alert dictionaries

        """
        with self._lock:
            return [alert.to_dict() for alert in self._alerts[-20:]]

    def get_recent_entries(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent cost entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of cost entry dictionaries

        """
        with self._lock:
            return [entry.to_dict() for entry in self._entries[-limit:]]

    def get_total_cost(
        self,
        since: datetime | None = None,
    ) -> float:
        """Get total cost.

        Args:
            since: Optional start datetime

        Returns:
            Total cost in USD

        """
        with self._lock:
            if not since:
                return sum(self._cumulative_by_provider.values())

            return sum(entry.cost for entry in self._entries if entry.timestamp >= since)

    def get_stats(self) -> dict[str, Any]:
        """Get cost tracker statistics."""
        with self._lock:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            return {
                "total_entries": len(self._entries),
                "total_cost": sum(self._cumulative_by_provider.values()),
                "cost_today": self._daily_costs.get(today, 0),
                "providers_tracked": len(self._cumulative_by_provider),
                "models_tracked": len(self._cumulative_by_model),
                "active_alerts": len(self._alerts),
                "daily_budget": self._daily_budget,
            }

    def reset(self) -> None:
        """Reset all cost tracking data."""
        with self._lock:
            self._entries.clear()
            self._cumulative_by_provider.clear()
            self._cumulative_by_model.clear()
            self._daily_costs.clear()
            self._alerts.clear()
            logger.info("Cost tracker reset")


# Global cost tracker instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track costs of AI operations.

    This middleware:
    1. Records request start time
    2. Processes request
    3. Extracts token usage from response body (if available)
    4. Calculates cost using config pricing
    5. Tracks cumulative costs

    Token usage is extracted from response JSON when available in the format:
    - usage.prompt_tokens / usage.completion_tokens (OpenAI style)
    - usage.input_tokens / usage.output_tokens (Anthropic style)
    """

    # Paths to track costs for
    COST_PATHS = [
        "/api/v1/generate",
        "/api/v1/chat",
        "/api/v1/completion",
        "/api/v1/llm",
        "/api/v1/jailbreak",
        "/api/v1/adversarial",
    ]

    # Paths to skip cost tracking
    SKIP_PATHS = [
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/auth",  # Exclude cost tracking for auth
    ]

    def __init__(
        self,
        app,
        *,
        enabled: bool = True,
        tracker: CostTracker | None = None,
        track_all_paths: bool = False,
    ) -> None:
        """Initialize cost tracking middleware.

        Args:
            app: ASGI application
            enabled: Whether cost tracking is enabled
            tracker: Optional custom cost tracker
            track_all_paths: Track all paths, not just COST_PATHS

        """
        super().__init__(app)
        self._enabled = enabled
        self._tracker = tracker or get_cost_tracker()
        self._track_all_paths = track_all_paths

    async def dispatch(
        self,
        request: Request,
        call_next,
    ) -> Response:
        """Process request with cost tracking.

        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain

        Returns:
            Response from next handler

        """
        if not self._enabled:
            return await call_next(request)

        path = request.url.path

        # Skip cost tracking for certain paths
        if self._should_skip(path):
            return await call_next(request)

        # Only track specific paths unless track_all_paths is enabled
        if not self._track_all_paths and not self._should_track(path):
            return await call_next(request)

        # Record start time
        start_time = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", str(int(time.time() * 1000)))

        try:
            # Process request
            response = await call_next(request)

            # Calculate elapsed time
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Try to extract and track costs
            await self._track_response_cost(
                request,
                response,
                request_id,
                elapsed_ms,
            )

            return response

        except Exception as e:
            # Track error costs too
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Cost tracking: request failed after {elapsed_ms:.2f}ms: {e}")
            raise

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip cost tracking."""
        return any(path.startswith(skip_path) for skip_path in self.SKIP_PATHS)

    def _should_track(self, path: str) -> bool:
        """Check if path should be tracked for costs."""
        return any(path.startswith(cost_path) for cost_path in self.COST_PATHS)

    async def _track_response_cost(
        self,
        request: Request,
        response: Response,
        request_id: str,
        elapsed_ms: float,
    ) -> None:
        """Extract token usage from response and track cost.

        Args:
            request: Original request
            response: Response object
            request_id: Request ID for correlation
            elapsed_ms: Request duration in milliseconds

        """
        try:
            # Get provider context
            context = getattr(request.state, "provider_context", None)
            if not context:
                return

            provider = context.provider
            model = context.model

            # Try to extract token usage from response
            usage = await self._extract_usage_from_response(response)

            if not usage:
                # No usage info - skip tracking
                return

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            reasoning_tokens = usage.get("reasoning_tokens", 0)
            cached_tokens = usage.get("cached_tokens", 0)

            # Calculate cost
            cost = self._tracker.calculate_cost(
                provider,
                model,
                input_tokens,
                output_tokens,
                reasoning_tokens,
                cached_tokens,
            )

            # Record the cost
            self._tracker.record_cost(
                provider=provider,
                model=model,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_id=request_id,
                metadata={
                    "path": request.url.path,
                    "elapsed_ms": elapsed_ms,
                    "reasoning_tokens": reasoning_tokens,
                    "cached_tokens": cached_tokens,
                },
            )

            # Add cost header to response
            response.headers["X-Request-Cost"] = f"${cost:.6f}"

        except Exception as e:
            logger.debug(f"Failed to track response cost: {e}")

    async def _extract_usage_from_response(
        self,
        response: Response,
    ) -> dict[str, int] | None:
        """Extract token usage from response body.

        Supports both OpenAI and Anthropic response formats.

        Args:
            response: Response object

        Returns:
            Dict with token counts or None

        """
        # Check content type
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return None

        # Try to get response body
        # Note: This only works with StreamingResponse that's been consumed
        # For streaming responses, usage tracking needs different approach
        body = getattr(response, "body", None)
        if not body:
            return None

        try:
            data = json.loads(body.decode("utf-8")) if isinstance(body, bytes) else json.loads(body)

            usage = data.get("usage", {})

            if not usage:
                return None

            # OpenAI format
            if "prompt_tokens" in usage:
                return {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                    "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                }

            # Anthropic format
            if "input_tokens" in usage:
                return {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "reasoning_tokens": 0,
                    "cached_tokens": usage.get("cache_read_input_tokens", 0),
                }

            return None

        except (json.JSONDecodeError, AttributeError):
            return None


def create_cost_tracking_middleware(
    enabled: bool = True,
    tracker: CostTracker | None = None,
    track_all_paths: bool = False,
) -> type:
    """Factory function to create cost tracking middleware.

    Args:
        enabled: Whether cost tracking is enabled
        tracker: Optional custom cost tracker
        track_all_paths: Track all paths, not just API paths

    Returns:
        Configured CostTrackingMiddleware class

    Example:
        from fastapi import FastAPI
        from app.middleware.cost_tracking_middleware import (
            create_cost_tracking_middleware
        )

        app = FastAPI()
        app.add_middleware(
            create_cost_tracking_middleware(enabled=True)
        )

    """

    class ConfiguredCostTrackingMiddleware(CostTrackingMiddleware):
        def __init__(self, app) -> None:
            super().__init__(
                app,
                enabled=enabled,
                tracker=tracker,
                track_all_paths=track_all_paths,
            )

    return ConfiguredCostTrackingMiddleware
