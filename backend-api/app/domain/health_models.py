# =============================================================================
# Chimera - Provider Health Monitoring Domain Models
# =============================================================================
# Pydantic models for provider health metrics, historical data, uptime tracking,
# quota management, and rate limit monitoring.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# =============================================================================

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderHealthStatus(str, Enum):
    """Health status levels for LLM providers."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class AlertSeverity(str, Enum):
    """Severity levels for health alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class QuotaPeriod(str, Enum):
    """Quota period types for usage tracking."""

    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


# =============================================================================
# Provider Health Metrics Models
# =============================================================================


class LatencyMetrics(BaseModel):
    """Latency metrics for provider health monitoring."""

    avg_ms: float = Field(default=0.0, ge=0.0, description="Average latency in milliseconds")
    p50_ms: float = Field(default=0.0, ge=0.0, description="50th percentile latency (median)")
    p95_ms: float = Field(default=0.0, ge=0.0, description="95th percentile latency")
    p99_ms: float = Field(default=0.0, ge=0.0, description="99th percentile latency")
    min_ms: float = Field(default=0.0, ge=0.0, description="Minimum observed latency")
    max_ms: float = Field(default=0.0, ge=0.0, description="Maximum observed latency")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "avg_ms": 450.5,
                "p50_ms": 380.0,
                "p95_ms": 850.0,
                "p99_ms": 1200.0,
                "min_ms": 120.0,
                "max_ms": 2500.0,
            },
        },
    )


class RequestMetrics(BaseModel):
    """Request metrics for provider health monitoring."""

    total_requests: int = Field(default=0, ge=0, description="Total number of requests")
    successful_requests: int = Field(default=0, ge=0, description="Number of successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Number of failed requests")
    timeout_requests: int = Field(default=0, ge=0, description="Number of timed out requests")
    rate_limited_requests: int = Field(
        default=0,
        ge=0,
        description="Number of rate limited requests",
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_requests": 10000,
                "successful_requests": 9850,
                "failed_requests": 100,
                "timeout_requests": 30,
                "rate_limited_requests": 20,
            },
        },
    )


class ProviderHealthMetrics(BaseModel):
    """Comprehensive health metrics for a single LLM provider.

    Tracks latency, error rates, uptime, and availability metrics.
    """

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    provider_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Provider display name",
    )
    status: ProviderHealthStatus = Field(
        default=ProviderHealthStatus.UNKNOWN,
        description="Current health status (operational/degraded/down)",
    )

    # Latency metrics
    latency_ms: float = Field(default=0.0, ge=0.0, description="Current average latency in ms")
    latency_metrics: LatencyMetrics = Field(
        default_factory=LatencyMetrics,
        description="Detailed latency statistics",
    )

    # Error rate metrics
    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Error rate percentage (0-100)",
    )
    request_metrics: RequestMetrics = Field(
        default_factory=RequestMetrics,
        description="Detailed request statistics",
    )

    # Uptime metrics
    uptime_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Uptime percentage (0-100)",
    )
    availability: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Availability percentage over measurement window",
    )

    # Consecutive failure/success tracking
    consecutive_failures: int = Field(default=0, ge=0, description="Current consecutive failures")
    consecutive_successes: int = Field(default=0, ge=0, description="Current consecutive successes")

    # Timestamps
    last_check_at: datetime | None = Field(
        default=None,
        description="When the last health check was performed",
    )
    last_success_at: datetime | None = Field(
        default=None,
        description="When the last successful request occurred",
    )
    last_failure_at: datetime | None = Field(
        default=None,
        description="When the last failure occurred",
    )
    last_status_change_at: datetime | None = Field(
        default=None,
        description="When the status last changed",
    )

    # Circuit breaker state
    circuit_breaker_state: str = Field(
        default="closed",
        description="Circuit breaker state (closed/open/half_open)",
    )

    # Measurement metadata
    measurement_window_seconds: int = Field(
        default=3600,
        ge=0,
        description="Measurement window in seconds",
    )
    check_interval_seconds: int = Field(
        default=30,
        ge=0,
        description="Health check interval in seconds",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last metrics update time",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "status": "operational",
                "latency_ms": 450.5,
                "error_rate": 1.5,
                "uptime_percent": 99.95,
                "availability": 99.8,
                "consecutive_failures": 0,
                "consecutive_successes": 15,
                "circuit_breaker_state": "closed",
                "measurement_window_seconds": 3600,
                "check_interval_seconds": 30,
            },
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format for API responses."""
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "status": self.status.value,
            "latency": {
                "current_ms": round(self.latency_ms, 2),
                "avg_ms": round(self.latency_metrics.avg_ms, 2),
                "p50_ms": round(self.latency_metrics.p50_ms, 2),
                "p95_ms": round(self.latency_metrics.p95_ms, 2),
                "p99_ms": round(self.latency_metrics.p99_ms, 2),
            },
            "requests": {
                "total": self.request_metrics.total_requests,
                "successful": self.request_metrics.successful_requests,
                "failed": self.request_metrics.failed_requests,
                "error_rate_percent": round(self.error_rate, 2),
                "success_rate_percent": round(100.0 - self.error_rate, 2),
            },
            "availability": {
                "uptime_percent": round(self.uptime_percent, 2),
                "availability_percent": round(self.availability, 2),
                "consecutive_failures": self.consecutive_failures,
                "consecutive_successes": self.consecutive_successes,
            },
            "circuit_breaker": {
                "state": self.circuit_breaker_state,
            },
            "timestamps": {
                "last_check": self.last_check_at.isoformat() if self.last_check_at else None,
                "last_success": self.last_success_at.isoformat() if self.last_success_at else None,
                "last_failure": self.last_failure_at.isoformat() if self.last_failure_at else None,
                "last_status_change": (
                    self.last_status_change_at.isoformat() if self.last_status_change_at else None
                ),
                "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            },
        }


# =============================================================================
# Health History Models
# =============================================================================


class HealthHistoryEntry(BaseModel):
    """Single health history entry for time-series data.

    Records a snapshot of provider health at a specific point in time.
    """

    id: str | None = Field(default=None, max_length=100, description="Unique entry identifier")
    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this entry was recorded",
    )

    # Health snapshot
    status: ProviderHealthStatus = Field(..., description="Health status at this time")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Latency at this time")
    error_rate: float = Field(default=0.0, ge=0.0, le=100.0, description="Error rate at this time")
    uptime_percent: float = Field(default=100.0, ge=0.0, le=100.0, description="Uptime percentage")

    # Check result
    success: bool = Field(..., description="Whether the health check succeeded")
    response_time_ms: float = Field(default=0.0, ge=0.0, description="Response time for this check")
    error_message: str | None = Field(
        default=None,
        max_length=500,
        description="Error message if check failed",
    )

    # Additional metadata
    check_type: str = Field(
        default="automatic",
        description="Type of check (automatic/manual/triggered)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "entry_001",
                "provider_id": "openai",
                "timestamp": "2025-01-11T15:30:00Z",
                "status": "operational",
                "latency_ms": 450.0,
                "error_rate": 1.2,
                "uptime_percent": 99.95,
                "success": True,
                "response_time_ms": 230.5,
                "check_type": "automatic",
            },
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert history entry to dictionary format."""
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "error_rate": round(self.error_rate, 2),
            "uptime_percent": round(self.uptime_percent, 2),
            "success": self.success,
            "response_time_ms": round(self.response_time_ms, 2),
            "error_message": self.error_message,
            "check_type": self.check_type,
            "metadata": self.metadata,
        }


class HealthHistoryResponse(BaseModel):
    """Response model for health history queries."""

    provider_id: str | None = Field(default=None, description="Provider filter (if specified)")
    entries: list[HealthHistoryEntry] = Field(default_factory=list, description="History entries")
    total_count: int = Field(default=0, ge=0, description="Total number of entries matching filter")
    start_time: datetime | None = Field(default=None, description="Start of query range")
    end_time: datetime | None = Field(default=None, description="End of query range")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "entries": [],
                "total_count": 100,
                "start_time": "2025-01-10T00:00:00Z",
                "end_time": "2025-01-11T00:00:00Z",
            },
        },
    )


# =============================================================================
# Uptime Tracking Models
# =============================================================================


class UptimeWindow(BaseModel):
    """Uptime data for a specific time window."""

    window_start: datetime = Field(..., description="Start of the time window")
    window_end: datetime = Field(..., description="End of the time window")
    window_seconds: int = Field(..., ge=0, description="Window duration in seconds")

    uptime_seconds: float = Field(default=0.0, ge=0.0, description="Seconds the provider was up")
    downtime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds the provider was down",
    )
    degraded_seconds: float = Field(default=0.0, ge=0.0, description="Seconds in degraded state")

    uptime_percent: float = Field(default=100.0, ge=0.0, le=100.0, description="Uptime percentage")

    # Incident tracking
    incident_count: int = Field(default=0, ge=0, description="Number of incidents in window")
    longest_outage_seconds: float = Field(default=0.0, ge=0.0, description="Longest single outage")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "window_start": "2025-01-01T00:00:00Z",
                "window_end": "2025-01-11T00:00:00Z",
                "window_seconds": 864000,
                "uptime_seconds": 863136,
                "downtime_seconds": 864,
                "degraded_seconds": 0,
                "uptime_percent": 99.9,
                "incident_count": 2,
                "longest_outage_seconds": 600,
            },
        },
    )


class ProviderUptimeMetrics(BaseModel):
    """Comprehensive uptime metrics for a provider across multiple time windows."""

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    provider_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Provider display name",
    )

    # Current status
    current_status: ProviderHealthStatus = Field(
        default=ProviderHealthStatus.UNKNOWN,
        description="Current health status",
    )
    status_since: datetime | None = Field(default=None, description="When current status started")

    # Uptime by window
    last_hour: UptimeWindow | None = Field(default=None, description="Last hour uptime")
    last_24_hours: UptimeWindow | None = Field(default=None, description="Last 24 hours uptime")
    last_7_days: UptimeWindow | None = Field(default=None, description="Last 7 days uptime")
    last_30_days: UptimeWindow | None = Field(default=None, description="Last 30 days uptime")

    # Overall stats
    all_time_uptime_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="All-time uptime percentage",
    )
    total_incidents: int = Field(default=0, ge=0, description="Total incident count")
    mean_time_between_failures_hours: float | None = Field(
        default=None,
        ge=0.0,
        description="Mean time between failures in hours",
    )
    mean_time_to_recovery_minutes: float | None = Field(
        default=None,
        ge=0.0,
        description="Mean time to recovery in minutes",
    )

    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "current_status": "operational",
                "all_time_uptime_percent": 99.95,
                "total_incidents": 5,
                "mean_time_between_failures_hours": 720.0,
                "mean_time_to_recovery_minutes": 15.0,
            },
        },
    )


# =============================================================================
# Quota Status Models
# =============================================================================


class ProviderQuotaStatus(BaseModel):
    """Quota status for an LLM provider.

    Tracks usage against configured limits with reset timing.
    """

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    provider_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Provider display name",
    )

    # Usage tracking
    usage: int = Field(default=0, ge=0, description="Current usage count (requests or tokens)")
    limit: int = Field(default=0, ge=0, description="Configured limit")
    usage_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of quota used",
    )

    # Token-based usage (if applicable)
    tokens_used: int = Field(default=0, ge=0, description="Tokens used in current period")
    tokens_limit: int | None = Field(default=None, ge=0, description="Token limit (if applicable)")
    tokens_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Token usage percentage",
    )

    # Request-based usage
    requests_used: int = Field(default=0, ge=0, description="Requests made in current period")
    requests_limit: int | None = Field(
        default=None,
        ge=0,
        description="Request limit (if applicable)",
    )
    requests_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Request usage percentage",
    )

    # Quota period and reset
    period: QuotaPeriod = Field(default=QuotaPeriod.DAILY, description="Quota period type")
    period_start_at: datetime | None = Field(
        default=None,
        description="When current period started",
    )
    reset_at: datetime | None = Field(default=None, description="When quota resets")

    # Cost tracking (if applicable)
    cost_used: float = Field(default=0.0, ge=0.0, description="Cost incurred in current period")
    cost_limit: float | None = Field(default=None, ge=0.0, description="Cost limit (if applicable)")
    cost_currency: str = Field(
        default="USD",
        max_length=3,
        description="Currency for cost tracking",
    )

    # Alert thresholds
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Warning alert threshold",
    )
    critical_threshold_percent: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Critical alert threshold",
    )
    is_warning: bool = Field(default=False, description="Whether warning threshold exceeded")
    is_critical: bool = Field(default=False, description="Whether critical threshold exceeded")
    is_exceeded: bool = Field(default=False, description="Whether quota is exceeded")

    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "usage": 8500,
                "limit": 10000,
                "usage_percent": 85.0,
                "tokens_used": 1500000,
                "tokens_limit": 2000000,
                "tokens_percent": 75.0,
                "period": "daily",
                "reset_at": "2025-01-12T00:00:00Z",
                "warning_threshold_percent": 80.0,
                "critical_threshold_percent": 95.0,
                "is_warning": True,
                "is_critical": False,
                "is_exceeded": False,
            },
        },
    )

    @field_validator("usage_percent", "tokens_percent", "requests_percent", mode="before")
    @classmethod
    def calculate_percentages(cls, v, info):
        """Ensure percentages are calculated correctly."""
        if v is not None:
            return min(100.0, max(0.0, v))
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert quota status to dictionary format."""
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "usage": {
                "current": self.usage,
                "limit": self.limit,
                "percent": round(self.usage_percent, 2),
            },
            "tokens": {
                "used": self.tokens_used,
                "limit": self.tokens_limit,
                "percent": round(self.tokens_percent, 2) if self.tokens_percent else None,
            },
            "requests": {
                "used": self.requests_used,
                "limit": self.requests_limit,
                "percent": round(self.requests_percent, 2) if self.requests_percent else None,
            },
            "cost": {
                "used": round(self.cost_used, 4),
                "limit": self.cost_limit,
                "currency": self.cost_currency,
            },
            "period": {
                "type": self.period.value,
                "started_at": self.period_start_at.isoformat() if self.period_start_at else None,
                "resets_at": self.reset_at.isoformat() if self.reset_at else None,
            },
            "alerts": {
                "warning_threshold": self.warning_threshold_percent,
                "critical_threshold": self.critical_threshold_percent,
                "is_warning": self.is_warning,
                "is_critical": self.is_critical,
                "is_exceeded": self.is_exceeded,
            },
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# Rate Limit Models
# =============================================================================


class RateLimitMetrics(BaseModel):
    """Rate limit metrics for an LLM provider.

    Tracks requests per minute against provider caps.
    """

    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    provider_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Provider display name",
    )

    # Current rate
    requests_per_minute: float = Field(
        default=0.0,
        ge=0.0,
        description="Current requests per minute",
    )
    tokens_per_minute: float = Field(default=0.0, ge=0.0, description="Current tokens per minute")

    # Provider caps
    provider_rpm_cap: int = Field(default=0, ge=0, description="Provider requests per minute cap")
    provider_tpm_cap: int = Field(default=0, ge=0, description="Provider tokens per minute cap")

    # Custom limits (user-configured)
    custom_rpm_limit: int | None = Field(default=None, ge=0, description="Custom RPM limit")
    custom_tpm_limit: int | None = Field(default=None, ge=0, description="Custom TPM limit")

    # Effective limits (lower of provider cap and custom limit)
    effective_rpm_limit: int = Field(default=0, ge=0, description="Effective RPM limit")
    effective_tpm_limit: int = Field(default=0, ge=0, description="Effective TPM limit")

    # Usage percentages
    rpm_usage_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="RPM usage percentage",
    )
    tpm_usage_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="TPM usage percentage",
    )

    # Rate limit status
    is_rate_limited: bool = Field(default=False, description="Currently rate limited")
    rate_limit_reset_at: datetime | None = Field(default=None, description="When rate limit resets")
    rate_limit_retry_after_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Seconds until retry",
    )

    # Historical rate limit tracking
    rate_limit_hits_last_hour: int = Field(
        default=0,
        ge=0,
        description="Rate limit hits in last hour",
    )
    rate_limit_hits_last_24h: int = Field(
        default=0,
        ge=0,
        description="Rate limit hits in last 24 hours",
    )

    # Burst tracking
    burst_capacity: int = Field(default=0, ge=0, description="Burst capacity (if applicable)")
    burst_remaining: int = Field(default=0, ge=0, description="Remaining burst capacity")

    # Timestamps
    window_start_at: datetime | None = Field(default=None, description="Current rate window start")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "requests_per_minute": 45.0,
                "tokens_per_minute": 80000.0,
                "provider_rpm_cap": 60,
                "provider_tpm_cap": 100000,
                "effective_rpm_limit": 60,
                "effective_tpm_limit": 100000,
                "rpm_usage_percent": 75.0,
                "tpm_usage_percent": 80.0,
                "is_rate_limited": False,
                "rate_limit_hits_last_hour": 2,
                "rate_limit_hits_last_24h": 5,
            },
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert rate limit metrics to dictionary format."""
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "current": {
                "requests_per_minute": round(self.requests_per_minute, 2),
                "tokens_per_minute": round(self.tokens_per_minute, 2),
            },
            "limits": {
                "provider_rpm_cap": self.provider_rpm_cap,
                "provider_tpm_cap": self.provider_tpm_cap,
                "custom_rpm_limit": self.custom_rpm_limit,
                "custom_tpm_limit": self.custom_tpm_limit,
                "effective_rpm": self.effective_rpm_limit,
                "effective_tpm": self.effective_tpm_limit,
            },
            "usage": {
                "rpm_percent": round(self.rpm_usage_percent, 2),
                "tpm_percent": round(self.tpm_usage_percent, 2),
            },
            "status": {
                "is_rate_limited": self.is_rate_limited,
                "reset_at": (
                    self.rate_limit_reset_at.isoformat() if self.rate_limit_reset_at else None
                ),
                "retry_after_seconds": self.rate_limit_retry_after_seconds,
            },
            "history": {
                "hits_last_hour": self.rate_limit_hits_last_hour,
                "hits_last_24h": self.rate_limit_hits_last_24h,
            },
            "burst": {
                "capacity": self.burst_capacity,
                "remaining": self.burst_remaining,
            },
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# Health Alert Models
# =============================================================================


class HealthAlert(BaseModel):
    """Health alert for provider degradation or issues."""

    alert_id: str = Field(..., min_length=1, max_length=100, description="Unique alert identifier")
    provider_id: str = Field(..., min_length=1, max_length=50, description="Provider identifier")
    severity: AlertSeverity = Field(..., description="Alert severity level")

    # Alert details
    title: str = Field(..., min_length=1, max_length=200, description="Alert title")
    message: str = Field(..., min_length=1, max_length=1000, description="Alert message")

    # Associated metrics
    error_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Error rate at time of alert",
    )
    latency_ms: float | None = Field(default=None, ge=0.0, description="Latency at time of alert")
    uptime_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Uptime at time of alert",
    )

    # Status transition (if applicable)
    previous_status: ProviderHealthStatus | None = Field(
        default=None,
        description="Status before change",
    )
    current_status: ProviderHealthStatus | None = Field(
        default=None,
        description="Status after change",
    )

    # Timestamps
    triggered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When alert was triggered",
    )
    resolved_at: datetime | None = Field(default=None, description="When alert was resolved")
    acknowledged_at: datetime | None = Field(
        default=None,
        description="When alert was acknowledged",
    )
    acknowledged_by: str | None = Field(
        default=None,
        max_length=100,
        description="Who acknowledged the alert",
    )

    # Alert state
    is_active: bool = Field(default=True, description="Whether alert is still active")
    is_acknowledged: bool = Field(default=False, description="Whether alert has been acknowledged")

    # Additional context
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional alert metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alert_id": "alert_openai_20250111_001",
                "provider_id": "openai",
                "severity": "warning",
                "title": "High Error Rate Detected",
                "message": "OpenAI error rate exceeded warning threshold (15.5%)",
                "error_rate": 15.5,
                "latency_ms": 1200.0,
                "previous_status": "operational",
                "current_status": "degraded",
                "triggered_at": "2025-01-11T15:30:00Z",
                "is_active": True,
                "is_acknowledged": False,
            },
        },
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "alert_id": self.alert_id,
            "provider_id": self.provider_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "metrics": {
                "error_rate_percent": (
                    round(self.error_rate, 2) if self.error_rate is not None else None
                ),
                "latency_ms": round(self.latency_ms, 2) if self.latency_ms is not None else None,
                "uptime_percent": (
                    round(self.uptime_percent, 2) if self.uptime_percent is not None else None
                ),
            },
            "status_change": {
                "previous": self.previous_status.value if self.previous_status else None,
                "current": self.current_status.value if self.current_status else None,
            },
            "timestamps": {
                "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
                "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
                "acknowledged_at": (
                    self.acknowledged_at.isoformat() if self.acknowledged_at else None
                ),
            },
            "state": {
                "is_active": self.is_active,
                "is_acknowledged": self.is_acknowledged,
                "acknowledged_by": self.acknowledged_by,
            },
            "metadata": self.metadata,
        }


# =============================================================================
# Dashboard Response Models
# =============================================================================


class ProviderHealthDashboardResponse(BaseModel):
    """Complete health dashboard response with all provider metrics."""

    status: str = Field(..., description="Overall system health status")
    providers: dict[str, ProviderHealthMetrics] = Field(
        default_factory=dict,
        description="Health metrics per provider",
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Health summary statistics")
    alerts: list[HealthAlert] = Field(default_factory=list, description="Active alerts")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last dashboard update",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "providers": {},
                "summary": {
                    "total_providers": 4,
                    "operational": 3,
                    "degraded": 1,
                    "down": 0,
                },
                "alerts": [],
                "updated_at": "2025-01-11T15:30:00Z",
            },
        },
    )


class QuotaDashboardResponse(BaseModel):
    """Quota dashboard response with all provider quotas."""

    providers: dict[str, ProviderQuotaStatus] = Field(
        default_factory=dict,
        description="Quota status per provider",
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Quota summary statistics")
    alerts: list[HealthAlert] = Field(default_factory=list, description="Quota-related alerts")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "providers": {},
                "summary": {
                    "total_providers": 4,
                    "quota_warning": 1,
                    "quota_critical": 0,
                    "quota_exceeded": 0,
                },
                "alerts": [],
                "updated_at": "2025-01-11T15:30:00Z",
            },
        },
    )


class RateLimitDashboardResponse(BaseModel):
    """Rate limit dashboard response with all provider rate limits."""

    providers: dict[str, RateLimitMetrics] = Field(
        default_factory=dict,
        description="Rate limit metrics per provider",
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Rate limit summary")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "providers": {},
                "summary": {
                    "total_providers": 4,
                    "currently_rate_limited": 0,
                    "approaching_limit": 1,
                },
                "updated_at": "2025-01-11T15:30:00Z",
            },
        },
    )
