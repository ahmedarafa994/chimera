"""
Usage Tracking Service

Tracks API usage per tenant for billing, analytics, and quota enforcement.
Supports in-memory storage with optional Redis persistence.

Part of Phase 3: Enterprise Readiness implementation.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UsageEventType(str, Enum):
    """Types of usage events to track."""

    API_REQUEST = "api_request"
    TECHNIQUE_USED = "technique_used"
    PLUGIN_USED = "plugin_used"
    LLM_CALL = "llm_call"
    CACHE_HIT = "cache_hit"
    ERROR = "error"


@dataclass
class UsageEvent:
    """
    Single usage event record.

    Attributes:
        tenant_id: Tenant that generated the event
        event_type: Type of event
        endpoint: API endpoint called
        technique: Technique used (if applicable)
        metadata: Additional event data
        timestamp: When the event occurred
        duration_ms: Request duration in milliseconds
        tokens_used: Token count (for LLM calls)
    """

    tenant_id: str
    event_type: UsageEventType
    endpoint: str = ""
    technique: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: int = 0
    tokens_used: int = 0


@dataclass
class TenantUsageSummary:
    """Usage summary for a tenant."""

    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    requests_by_endpoint: dict[str, int] = field(default_factory=dict)
    requests_by_technique: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    total_errors: int = 0
    cache_hit_rate: float = 0.0
    avg_duration_ms: float = 0.0


class UsageTracker:
    """
    Tracks and aggregates API usage per tenant.

    Provides:
    - Real-time usage recording
    - Quota enforcement
    - Usage analytics
    - Billing data export
    """

    _instance: Optional["UsageTracker"] = None
    _events: list[UsageEvent]
    _hourly_counts: dict[str, dict[str, int]]  # tenant_id -> {hour: count}
    _monthly_counts: dict[str, int]  # tenant_id -> count

    def __new__(cls) -> "UsageTracker":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._events = []
            cls._hourly_counts = defaultdict(lambda: defaultdict(int))
            cls._monthly_counts = defaultdict(int)
            cls._cache_hits = defaultdict(int)
            cls._cache_requests = defaultdict(int)
            cls._durations = defaultdict(list)
        return cls._instance

    def record_event(
        self,
        tenant_id: str,
        event_type: UsageEventType,
        endpoint: str = "",
        technique: str = "",
        duration_ms: int = 0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> UsageEvent:
        """
        Record a usage event.

        Args:
            tenant_id: Tenant identifier
            event_type: Type of event
            endpoint: API endpoint
            technique: Technique used
            duration_ms: Request duration
            tokens_used: Token count
            metadata: Additional data

        Returns:
            The recorded event
        """
        event = UsageEvent(
            tenant_id=tenant_id,
            event_type=event_type,
            endpoint=endpoint,
            technique=technique,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            metadata=metadata or {},
        )

        # Store event (in production, would use Redis/DB)
        self._events.append(event)

        # Update counters
        hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
        self._hourly_counts[tenant_id][hour_key] += 1

        month_key = event.timestamp.strftime("%Y-%m")
        self._monthly_counts[f"{tenant_id}:{month_key}"] = (
            self._monthly_counts.get(f"{tenant_id}:{month_key}", 0) + 1
        )

        # Track cache stats
        if event_type == UsageEventType.CACHE_HIT:
            self._cache_hits[tenant_id] += 1
        if event_type in [UsageEventType.API_REQUEST, UsageEventType.CACHE_HIT]:
            self._cache_requests[tenant_id] += 1

        # Track durations
        if duration_ms > 0:
            durations = self._durations[tenant_id]
            durations.append(duration_ms)
            # Keep only last 1000 durations
            if len(durations) > 1000:
                self._durations[tenant_id] = durations[-1000:]

        # Trim old events (keep last 10000)
        if len(self._events) > 10000:
            self._events = self._events[-10000:]

        return event

    def record_api_request(
        self,
        tenant_id: str,
        endpoint: str,
        duration_ms: int = 0,
        technique: str = "",
        tokens_used: int = 0,
    ) -> UsageEvent:
        """Convenience method for recording API requests."""
        return self.record_event(
            tenant_id=tenant_id,
            event_type=UsageEventType.API_REQUEST,
            endpoint=endpoint,
            technique=technique,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
        )

    def record_technique_usage(
        self, tenant_id: str, technique: str, potency: int = 5
    ) -> UsageEvent:
        """Record technique usage."""
        return self.record_event(
            tenant_id=tenant_id,
            event_type=UsageEventType.TECHNIQUE_USED,
            technique=technique,
            metadata={"potency": potency},
        )

    def record_error(
        self, tenant_id: str, endpoint: str, error_type: str, error_message: str
    ) -> UsageEvent:
        """Record an error event."""
        return self.record_event(
            tenant_id=tenant_id,
            event_type=UsageEventType.ERROR,
            endpoint=endpoint,
            metadata={"error_type": error_type, "error_message": error_message},
        )

    def get_monthly_usage(self, tenant_id: str) -> int:
        """Get current month's usage count for a tenant."""
        month_key = datetime.utcnow().strftime("%Y-%m")
        return self._monthly_counts.get(f"{tenant_id}:{month_key}", 0)

    def check_quota(self, tenant_id: str, monthly_quota: int) -> tuple[bool, int]:
        """
        Check if tenant is within quota.

        Args:
            tenant_id: Tenant identifier
            monthly_quota: Tenant's monthly quota (-1 for unlimited)

        Returns:
            Tuple of (within_quota: bool, remaining: int)
        """
        if monthly_quota == -1:
            return True, -1  # Unlimited

        current_usage = self.get_monthly_usage(tenant_id)
        remaining = monthly_quota - current_usage

        return remaining > 0, remaining

    def get_tenant_summary(self, tenant_id: str, period_hours: int = 24) -> TenantUsageSummary:
        """
        Get usage summary for a tenant.

        Args:
            tenant_id: Tenant identifier
            period_hours: Hours to include in summary

        Returns:
            Usage summary
        """
        now = datetime.utcnow()
        period_start = now - timedelta(hours=period_hours)

        # Filter events for this tenant and period
        relevant_events = [
            e for e in self._events if e.tenant_id == tenant_id and e.timestamp >= period_start
        ]

        # Aggregate
        requests_by_endpoint: dict[str, int] = defaultdict(int)
        requests_by_technique: dict[str, int] = defaultdict(int)
        total_tokens = 0
        total_errors = 0

        for event in relevant_events:
            if event.endpoint:
                requests_by_endpoint[event.endpoint] += 1
            if event.technique:
                requests_by_technique[event.technique] += 1
            total_tokens += event.tokens_used
            if event.event_type == UsageEventType.ERROR:
                total_errors += 1

        # Calculate cache hit rate
        cache_reqs = self._cache_requests.get(tenant_id, 0)
        cache_hits = self._cache_hits.get(tenant_id, 0)
        cache_hit_rate = (cache_hits / cache_reqs * 100) if cache_reqs > 0 else 0.0

        # Calculate average duration
        durations = self._durations.get(tenant_id, [])
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return TenantUsageSummary(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=now,
            total_requests=len(relevant_events),
            requests_by_endpoint=dict(requests_by_endpoint),
            requests_by_technique=dict(requests_by_technique),
            total_tokens=total_tokens,
            total_errors=total_errors,
            cache_hit_rate=cache_hit_rate,
            avg_duration_ms=avg_duration,
        )

    def get_top_techniques(
        self, tenant_id: str | None = None, limit: int = 10
    ) -> list[tuple[str, int]]:
        """Get most used techniques."""
        technique_counts: dict[str, int] = defaultdict(int)

        for event in self._events:
            if tenant_id and event.tenant_id != tenant_id:
                continue
            if event.technique:
                technique_counts[event.technique] += 1

        sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_techniques[:limit]

    def get_global_statistics(self) -> dict[str, Any]:
        """Get global usage statistics."""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        today_events = [e for e in self._events if e.timestamp >= today_start]

        unique_tenants = {e.tenant_id for e in self._events}
        active_today = {e.tenant_id for e in today_events}

        return {
            "total_events_tracked": len(self._events),
            "unique_tenants": len(unique_tenants),
            "active_tenants_today": len(active_today),
            "requests_today": len(today_events),
            "top_techniques": self.get_top_techniques(limit=5),
        }

    def clear_tenant_data(self, tenant_id: str) -> int:
        """Clear all usage data for a tenant. Returns count removed."""
        original_count = len(self._events)
        self._events = [e for e in self._events if e.tenant_id != tenant_id]

        # Clear counters
        self._hourly_counts.pop(tenant_id, None)
        self._cache_hits.pop(tenant_id, None)
        self._cache_requests.pop(tenant_id, None)
        self._durations.pop(tenant_id, None)

        # Clear monthly counts
        keys_to_remove = [k for k in self._monthly_counts if k.startswith(f"{tenant_id}:")]
        for key in keys_to_remove:
            del self._monthly_counts[key]

        removed = original_count - len(self._events)
        logger.info(f"Cleared {removed} events for tenant: {tenant_id}")
        return removed


# Global instance
_usage_tracker = UsageTracker()


def get_usage_tracker() -> UsageTracker:
    """Get the global usage tracker."""
    return _usage_tracker
