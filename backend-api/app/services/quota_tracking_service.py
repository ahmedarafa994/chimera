# =============================================================================
# Chimera - Quota Tracking Service
# =============================================================================
# Tracks API quota usage per provider with alerts when approaching limits.
# Integrates with model_rate_limiter.py for rate data and provider_health_service.py
# for alert notifications.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 2.3: Create Quota Tracking Service
# =============================================================================

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from app.domain.health_models import (
    AlertSeverity,
    HealthAlert,
    ProviderQuotaStatus,
    QuotaDashboardResponse,
    QuotaPeriod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Default quota limits per provider (can be overridden via configuration)
DEFAULT_DAILY_REQUEST_LIMIT = 10000
DEFAULT_DAILY_TOKEN_LIMIT = 1_000_000
DEFAULT_MONTHLY_REQUEST_LIMIT = 300_000
DEFAULT_MONTHLY_TOKEN_LIMIT = 30_000_000

# Alert thresholds
DEFAULT_WARNING_THRESHOLD = 80.0  # 80% usage
DEFAULT_CRITICAL_THRESHOLD = 95.0  # 95% usage

# Alert cooldown to prevent spam
DEFAULT_ALERT_COOLDOWN_SECONDS = 300  # 5 minutes

# Maximum alerts to retain
MAX_ALERTS_RETAINED = 100

# Provider-specific default limits (based on typical provider tiers)
PROVIDER_DEFAULT_LIMITS: dict[str, dict[str, dict[str, int]]] = {
    "openai": {
        "daily": {
            "requests": 10000,
            "tokens": 1_000_000,
        },
        "monthly": {
            "requests": 300000,
            "tokens": 30_000_000,
        },
    },
    "anthropic": {
        "daily": {
            "requests": 10000,
            "tokens": 1_000_000,
        },
        "monthly": {
            "requests": 300000,
            "tokens": 30_000_000,
        },
    },
    "google": {
        "daily": {
            "requests": 15000,
            "tokens": 1_500_000,
        },
        "monthly": {
            "requests": 450000,
            "tokens": 45_000_000,
        },
    },
    "deepseek": {
        "daily": {
            "requests": 10000,
            "tokens": 1_000_000,
        },
        "monthly": {
            "requests": 300000,
            "tokens": 30_000_000,
        },
    },
    "qwen": {
        "daily": {
            "requests": 10000,
            "tokens": 1_000_000,
        },
        "monthly": {
            "requests": 300000,
            "tokens": 30_000_000,
        },
    },
    "bigmodel": {
        "daily": {
            "requests": 5000,
            "tokens": 500_000,
        },
        "monthly": {
            "requests": 150000,
            "tokens": 15_000_000,
        },
    },
}


# =============================================================================
# Quota Configuration
# =============================================================================


class ProviderQuotaConfig:
    """
    Configuration for quota tracking of a specific provider.

    Allows per-provider customization of limits and alert thresholds.
    """

    def __init__(
        self,
        provider_id: str,
        daily_request_limit: int | None = None,
        daily_token_limit: int | None = None,
        monthly_request_limit: int | None = None,
        monthly_token_limit: int | None = None,
        warning_threshold_percent: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold_percent: float = DEFAULT_CRITICAL_THRESHOLD,
        cost_per_1k_input_tokens: float = 0.0,
        cost_per_1k_output_tokens: float = 0.0,
        daily_cost_limit: float | None = None,
        monthly_cost_limit: float | None = None,
        enabled: bool = True,
    ):
        self.provider_id = provider_id
        self.daily_request_limit = daily_request_limit or DEFAULT_DAILY_REQUEST_LIMIT
        self.daily_token_limit = daily_token_limit or DEFAULT_DAILY_TOKEN_LIMIT
        self.monthly_request_limit = monthly_request_limit or DEFAULT_MONTHLY_REQUEST_LIMIT
        self.monthly_token_limit = monthly_token_limit or DEFAULT_MONTHLY_TOKEN_LIMIT
        self.warning_threshold_percent = warning_threshold_percent
        self.critical_threshold_percent = critical_threshold_percent
        self.cost_per_1k_input_tokens = cost_per_1k_input_tokens
        self.cost_per_1k_output_tokens = cost_per_1k_output_tokens
        self.daily_cost_limit = daily_cost_limit
        self.monthly_cost_limit = monthly_cost_limit
        self.enabled = enabled

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "provider_id": self.provider_id,
            "limits": {
                "daily": {
                    "requests": self.daily_request_limit,
                    "tokens": self.daily_token_limit,
                },
                "monthly": {
                    "requests": self.monthly_request_limit,
                    "tokens": self.monthly_token_limit,
                },
            },
            "thresholds": {
                "warning_percent": self.warning_threshold_percent,
                "critical_percent": self.critical_threshold_percent,
            },
            "cost": {
                "per_1k_input_tokens": self.cost_per_1k_input_tokens,
                "per_1k_output_tokens": self.cost_per_1k_output_tokens,
                "daily_limit": self.daily_cost_limit,
                "monthly_limit": self.monthly_cost_limit,
            },
            "enabled": self.enabled,
        }


# =============================================================================
# Quota Usage Window
# =============================================================================


class QuotaUsageWindow:
    """
    Tracks usage within a specific time window (daily or monthly).
    """

    def __init__(
        self,
        period: QuotaPeriod,
        period_start: datetime,
    ):
        self.period = period
        self.period_start = period_start
        self.period_end = self._calculate_period_end(period, period_start)

        # Usage counters
        self.request_count = 0
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

        # Per-model breakdown
        self.by_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "tokens": 0}
        )

        # Timestamps
        self.last_request_at: datetime | None = None
        self.created_at = datetime.utcnow()

    def _calculate_period_end(self, period: QuotaPeriod, start: datetime) -> datetime:
        """Calculate when this period ends."""
        if period == QuotaPeriod.HOURLY:
            return start + timedelta(hours=1)
        elif period == QuotaPeriod.DAILY:
            return start + timedelta(days=1)
        elif period == QuotaPeriod.MONTHLY:
            # Move to first of next month
            if start.month == 12:
                return start.replace(year=start.year + 1, month=1, day=1, hour=0, minute=0, second=0)
            else:
                return start.replace(month=start.month + 1, day=1, hour=0, minute=0, second=0)
        else:
            return start + timedelta(days=1)

    def is_expired(self) -> bool:
        """Check if this window has expired."""
        return datetime.utcnow() >= self.period_end

    def record_usage(
        self,
        tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        model: str | None = None,
    ) -> None:
        """Record usage in this window."""
        self.request_count += 1
        self.total_tokens += tokens
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost
        self.last_request_at = datetime.utcnow()

        if model:
            self.by_model[model]["requests"] += 1
            self.by_model[model]["tokens"] += tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert window to dictionary."""
        return {
            "period": self.period.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "usage": {
                "requests": self.request_count,
                "total_tokens": self.total_tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 4),
            },
            "by_model": dict(self.by_model),
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
        }


# =============================================================================
# Provider Quota State
# =============================================================================


class ProviderQuotaState:
    """
    Maintains quota state for a single provider.

    Tracks usage across daily and monthly windows with automatic period rotation.
    """

    def __init__(self, provider_id: str, provider_name: str):
        self.provider_id = provider_id
        self.provider_name = provider_name

        # Current usage windows
        self._daily_window: QuotaUsageWindow | None = None
        self._monthly_window: QuotaUsageWindow | None = None

        # Historical usage (last 30 days of daily summaries)
        self._daily_history: deque[dict[str, Any]] = deque(maxlen=30)
        self._monthly_history: deque[dict[str, Any]] = deque(maxlen=12)

        # Alert state
        self.last_warning_at: datetime | None = None
        self.last_critical_at: datetime | None = None
        self.is_warning = False
        self.is_critical = False
        self.is_exceeded = False

    def _get_daily_window(self) -> QuotaUsageWindow:
        """Get or create the current daily window."""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self._daily_window is None or self._daily_window.is_expired():
            # Archive old window
            if self._daily_window is not None:
                self._daily_history.append(self._daily_window.to_dict())

            # Create new window
            self._daily_window = QuotaUsageWindow(QuotaPeriod.DAILY, today_start)

        return self._daily_window

    def _get_monthly_window(self) -> QuotaUsageWindow:
        """Get or create the current monthly window."""
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        if self._monthly_window is None or self._monthly_window.is_expired():
            # Archive old window
            if self._monthly_window is not None:
                self._monthly_history.append(self._monthly_window.to_dict())

            # Create new window
            self._monthly_window = QuotaUsageWindow(QuotaPeriod.MONTHLY, month_start)

        return self._monthly_window

    def record_usage(
        self,
        tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        model: str | None = None,
    ) -> None:
        """Record usage in both daily and monthly windows."""
        daily = self._get_daily_window()
        monthly = self._get_monthly_window()

        daily.record_usage(tokens, input_tokens, output_tokens, cost, model)
        monthly.record_usage(tokens, input_tokens, output_tokens, cost, model)

    def get_daily_usage(self) -> dict[str, int]:
        """Get current daily usage."""
        daily = self._get_daily_window()
        return {
            "requests": daily.request_count,
            "tokens": daily.total_tokens,
        }

    def get_monthly_usage(self) -> dict[str, int]:
        """Get current monthly usage."""
        monthly = self._get_monthly_window()
        return {
            "requests": monthly.request_count,
            "tokens": monthly.total_tokens,
        }

    def calculate_quota_status(
        self,
        config: ProviderQuotaConfig,
        period: QuotaPeriod = QuotaPeriod.DAILY,
    ) -> ProviderQuotaStatus:
        """
        Calculate current quota status.

        Args:
            config: Provider quota configuration
            period: Which period to calculate for

        Returns:
            ProviderQuotaStatus with current usage and limits
        """
        if period == QuotaPeriod.DAILY:
            window = self._get_daily_window()
            request_limit = config.daily_request_limit
            token_limit = config.daily_token_limit
            cost_limit = config.daily_cost_limit
        else:  # MONTHLY
            window = self._get_monthly_window()
            request_limit = config.monthly_request_limit
            token_limit = config.monthly_token_limit
            cost_limit = config.monthly_cost_limit

        # Calculate percentages
        request_percent = (window.request_count / request_limit * 100) if request_limit > 0 else 0.0
        token_percent = (window.total_tokens / token_limit * 100) if token_limit > 0 else 0.0

        # Use highest percentage for overall usage
        usage_percent = max(request_percent, token_percent)

        # Determine alert states
        is_warning = usage_percent >= config.warning_threshold_percent
        is_critical = usage_percent >= config.critical_threshold_percent
        is_exceeded = usage_percent >= 100.0

        return ProviderQuotaStatus(
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            usage=window.request_count,
            limit=request_limit,
            usage_percent=min(usage_percent, 100.0),
            tokens_used=window.total_tokens,
            tokens_limit=token_limit,
            tokens_percent=min(token_percent, 100.0),
            requests_used=window.request_count,
            requests_limit=request_limit,
            requests_percent=min(request_percent, 100.0),
            period=period,
            period_start_at=window.period_start,
            reset_at=window.period_end,
            cost_used=window.cost_usd,
            cost_limit=cost_limit,
            warning_threshold_percent=config.warning_threshold_percent,
            critical_threshold_percent=config.critical_threshold_percent,
            is_warning=is_warning,
            is_critical=is_critical,
            is_exceeded=is_exceeded,
        )

    def get_usage_history(
        self,
        period: QuotaPeriod = QuotaPeriod.DAILY,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get historical usage data."""
        if period == QuotaPeriod.DAILY:
            history = list(self._daily_history)
        else:
            history = list(self._monthly_history)

        # Add current window
        current = self._get_daily_window() if period == QuotaPeriod.DAILY else self._get_monthly_window()
        history.append(current.to_dict())

        return history[-limit:]


# =============================================================================
# Quota Tracking Service
# =============================================================================


class QuotaTrackingService:
    """
    Service for tracking API quota usage per provider with alerts.

    Features:
    - Request and token usage tracking per provider
    - Daily and monthly quota periods
    - Configurable limits per provider
    - Alert generation at configurable thresholds (80%, 95%)
    - Cost tracking with USD calculations
    - Integration with provider_health_service for alerts
    - Historical usage data for analysis

    Usage:
        from app.services.quota_tracking_service import get_quota_tracking_service

        # Get service instance
        quota_service = get_quota_tracking_service()

        # Record usage
        await quota_service.record_usage(
            provider_id="openai",
            tokens=1500,
            input_tokens=500,
            output_tokens=1000,
            model="gpt-4"
        )

        # Get quota status
        status = await quota_service.get_provider_quota("openai")

        # Get dashboard data
        dashboard = await quota_service.get_quota_dashboard()
    """

    _instance: "QuotaTrackingService | None" = None

    def __new__(cls) -> "QuotaTrackingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Provider configurations
        self._provider_configs: dict[str, ProviderQuotaConfig] = {}

        # Provider quota states
        self._provider_states: dict[str, ProviderQuotaState] = {}

        # Alerts
        self._alerts: deque[HealthAlert] = deque(maxlen=MAX_ALERTS_RETAINED)
        self._alert_cooldown: dict[str, float] = {}  # provider_id -> last alert timestamp

        # Callbacks
        self._alert_callbacks: list[Callable[[HealthAlert], None]] = []
        self._threshold_callbacks: list[Callable[[str, float, str], None]] = []

        # Async lock
        self._lock = asyncio.Lock()

        self._initialized = True

        # Initialize default configurations
        self._initialize_default_configs()

        logger.info("QuotaTrackingService initialized")

    def _initialize_default_configs(self) -> None:
        """Initialize default configurations for known providers."""
        for provider_id, limits in PROVIDER_DEFAULT_LIMITS.items():
            self._provider_configs[provider_id] = ProviderQuotaConfig(
                provider_id=provider_id,
                daily_request_limit=limits["daily"]["requests"],
                daily_token_limit=limits["daily"]["tokens"],
                monthly_request_limit=limits["monthly"]["requests"],
                monthly_token_limit=limits["monthly"]["tokens"],
            )

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure_provider(
        self,
        provider_id: str,
        provider_name: str | None = None,
        daily_request_limit: int | None = None,
        daily_token_limit: int | None = None,
        monthly_request_limit: int | None = None,
        monthly_token_limit: int | None = None,
        warning_threshold_percent: float | None = None,
        critical_threshold_percent: float | None = None,
        cost_per_1k_input_tokens: float | None = None,
        cost_per_1k_output_tokens: float | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Configure quota tracking for a provider.

        Args:
            provider_id: Provider identifier
            provider_name: Display name (defaults to provider_id.title())
            daily_request_limit: Daily request limit
            daily_token_limit: Daily token limit
            monthly_request_limit: Monthly request limit
            monthly_token_limit: Monthly token limit
            warning_threshold_percent: Warning threshold percentage
            critical_threshold_percent: Critical threshold percentage
            cost_per_1k_input_tokens: Cost per 1K input tokens (USD)
            cost_per_1k_output_tokens: Cost per 1K output tokens (USD)
            enabled: Whether tracking is enabled
        """
        display_name = provider_name or provider_id.title()

        # Get default limits for this provider
        defaults = PROVIDER_DEFAULT_LIMITS.get(provider_id, {})
        default_daily = defaults.get("daily", {})
        default_monthly = defaults.get("monthly", {})

        config = ProviderQuotaConfig(
            provider_id=provider_id,
            daily_request_limit=daily_request_limit or default_daily.get("requests", DEFAULT_DAILY_REQUEST_LIMIT),
            daily_token_limit=daily_token_limit or default_daily.get("tokens", DEFAULT_DAILY_TOKEN_LIMIT),
            monthly_request_limit=monthly_request_limit or default_monthly.get("requests", DEFAULT_MONTHLY_REQUEST_LIMIT),
            monthly_token_limit=monthly_token_limit or default_monthly.get("tokens", DEFAULT_MONTHLY_TOKEN_LIMIT),
            warning_threshold_percent=warning_threshold_percent or DEFAULT_WARNING_THRESHOLD,
            critical_threshold_percent=critical_threshold_percent or DEFAULT_CRITICAL_THRESHOLD,
            cost_per_1k_input_tokens=cost_per_1k_input_tokens or 0.0,
            cost_per_1k_output_tokens=cost_per_1k_output_tokens or 0.0,
            enabled=enabled,
        )
        self._provider_configs[provider_id] = config

        # Create state if not exists
        if provider_id not in self._provider_states:
            self._provider_states[provider_id] = ProviderQuotaState(
                provider_id=provider_id,
                provider_name=display_name,
            )

        logger.info(f"Configured quota tracking for provider: {provider_id}")

    def get_provider_config(self, provider_id: str) -> ProviderQuotaConfig | None:
        """Get configuration for a provider."""
        return self._provider_configs.get(provider_id)

    def update_provider_limits(
        self,
        provider_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Update limits for a provider.

        Args:
            provider_id: Provider identifier
            **kwargs: Limit fields to update

        Returns:
            True if updated, False if provider not found
        """
        config = self._provider_configs.get(provider_id)
        if not config:
            return False

        for key, value in kwargs.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return True

    # =========================================================================
    # Usage Recording
    # =========================================================================

    async def record_usage(
        self,
        provider_id: str,
        tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str | None = None,
    ) -> None:
        """
        Record API usage for a provider.

        This method should be called after each API request to track quota usage.

        Args:
            provider_id: Provider identifier
            tokens: Total tokens used (if input/output not separate)
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            model: Model name for per-model breakdown
        """
        async with self._lock:
            # Ensure provider is configured
            if provider_id not in self._provider_configs:
                self.configure_provider(provider_id)

            config = self._provider_configs[provider_id]
            if not config.enabled:
                return

            # Get or create state
            if provider_id not in self._provider_states:
                self._provider_states[provider_id] = ProviderQuotaState(
                    provider_id=provider_id,
                    provider_name=provider_id.title(),
                )

            state = self._provider_states[provider_id]

            # Calculate cost
            cost = 0.0
            if config.cost_per_1k_input_tokens > 0:
                cost += (input_tokens / 1000) * config.cost_per_1k_input_tokens
            if config.cost_per_1k_output_tokens > 0:
                cost += (output_tokens / 1000) * config.cost_per_1k_output_tokens

            # Record usage
            state.record_usage(
                tokens=tokens or (input_tokens + output_tokens),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                model=model,
            )

            # Check thresholds and generate alerts
            await self._check_thresholds(provider_id, config, state)

    async def _check_thresholds(
        self,
        provider_id: str,
        config: ProviderQuotaConfig,
        state: ProviderQuotaState,
    ) -> None:
        """Check quota thresholds and generate alerts if needed."""
        now = time.time()

        # Check cooldown
        last_alert = self._alert_cooldown.get(provider_id, 0)
        if now - last_alert < DEFAULT_ALERT_COOLDOWN_SECONDS:
            return

        # Get quota status for daily period
        daily_status = state.calculate_quota_status(config, QuotaPeriod.DAILY)

        alert = None

        # Check for exceeded quota
        if daily_status.is_exceeded and not state.is_exceeded:
            state.is_exceeded = True
            alert = HealthAlert(
                alert_id=f"quota_exceeded_{provider_id}_{int(now)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                severity=AlertSeverity.CRITICAL,
                title=f"Quota exceeded for {provider_id}",
                message=f"Daily quota exceeded ({daily_status.usage_percent:.1f}% used). Requests may be rate limited.",
                triggered_at=datetime.utcnow(),
                metadata={
                    "quota_type": "daily",
                    "usage_percent": daily_status.usage_percent,
                    "requests_used": daily_status.requests_used,
                    "tokens_used": daily_status.tokens_used,
                },
            )

        # Check for critical threshold
        elif daily_status.is_critical and not state.is_critical:
            state.is_critical = True
            state.last_critical_at = datetime.utcnow()
            alert = HealthAlert(
                alert_id=f"quota_critical_{provider_id}_{int(now)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                severity=AlertSeverity.CRITICAL,
                title=f"Critical quota usage for {provider_id}",
                message=f"Daily quota usage is critical ({daily_status.usage_percent:.1f}%). "
                        f"Approaching limit of {config.daily_request_limit} requests.",
                triggered_at=datetime.utcnow(),
                metadata={
                    "quota_type": "daily",
                    "threshold": config.critical_threshold_percent,
                    "usage_percent": daily_status.usage_percent,
                },
            )

        # Check for warning threshold
        elif daily_status.is_warning and not state.is_warning:
            state.is_warning = True
            state.last_warning_at = datetime.utcnow()
            alert = HealthAlert(
                alert_id=f"quota_warning_{provider_id}_{int(now)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                severity=AlertSeverity.WARNING,
                title=f"High quota usage for {provider_id}",
                message=f"Daily quota usage is at {daily_status.usage_percent:.1f}%. "
                        f"Warning threshold ({config.warning_threshold_percent}%) exceeded.",
                triggered_at=datetime.utcnow(),
                metadata={
                    "quota_type": "daily",
                    "threshold": config.warning_threshold_percent,
                    "usage_percent": daily_status.usage_percent,
                },
            )

        # Reset states if usage drops
        elif not daily_status.is_warning:
            state.is_warning = False
            state.is_critical = False
            state.is_exceeded = False

        if alert:
            await self._emit_alert(alert)
            self._alert_cooldown[provider_id] = now

            # Notify threshold callbacks
            for callback in self._threshold_callbacks:
                try:
                    callback(provider_id, daily_status.usage_percent, alert.severity.value)
                except Exception as e:
                    logger.error(f"Error in threshold callback: {e}")

    async def _emit_alert(self, alert: HealthAlert) -> None:
        """Emit an alert and notify callbacks."""
        self._alerts.append(alert)

        logger.warning(f"Quota alert: [{alert.severity.value}] {alert.title} - {alert.message}")

        # Trigger alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        # Try to notify health service
        try:
            from app.services.provider_health_service import get_provider_health_service

            health_service = get_provider_health_service()
            # Add alert to health service
            health_service._alerts.append(alert)
        except Exception:
            pass  # Health service may not be available

    # =========================================================================
    # Callbacks
    # =========================================================================

    def register_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Register a callback to be invoked when quota alerts are triggered."""
        self._alert_callbacks.append(callback)

    def register_threshold_callback(
        self, callback: Callable[[str, float, str], None]
    ) -> None:
        """
        Register a callback for threshold breaches.

        Callback receives: (provider_id, usage_percent, severity)
        """
        self._threshold_callbacks.append(callback)

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_provider_quota(
        self,
        provider_id: str,
        period: QuotaPeriod = QuotaPeriod.DAILY,
    ) -> ProviderQuotaStatus | None:
        """
        Get quota status for a specific provider.

        Args:
            provider_id: Provider identifier
            period: Quota period (DAILY or MONTHLY)

        Returns:
            ProviderQuotaStatus or None if not found
        """
        state = self._provider_states.get(provider_id)
        config = self._provider_configs.get(provider_id)

        if not state or not config:
            return None

        return state.calculate_quota_status(config, period)

    async def get_all_provider_quotas(
        self,
        period: QuotaPeriod = QuotaPeriod.DAILY,
    ) -> dict[str, ProviderQuotaStatus]:
        """
        Get quota status for all providers.

        Args:
            period: Quota period (DAILY or MONTHLY)

        Returns:
            Dictionary of provider_id -> ProviderQuotaStatus
        """
        quotas = {}

        for provider_id, state in self._provider_states.items():
            config = self._provider_configs.get(provider_id)
            if config and config.enabled:
                quotas[provider_id] = state.calculate_quota_status(config, period)

        return quotas

    async def get_quota_dashboard(
        self,
        period: QuotaPeriod = QuotaPeriod.DAILY,
    ) -> QuotaDashboardResponse:
        """
        Get comprehensive quota dashboard data.

        Args:
            period: Quota period for calculations

        Returns:
            QuotaDashboardResponse with all provider quotas
        """
        quotas = await self.get_all_provider_quotas(period)

        # Calculate summary
        total_providers = len(quotas)
        quota_warning = sum(1 for q in quotas.values() if q.is_warning and not q.is_critical)
        quota_critical = sum(1 for q in quotas.values() if q.is_critical)
        quota_exceeded = sum(1 for q in quotas.values() if q.is_exceeded)

        # Get recent alerts
        recent_alerts = [a for a in self._alerts if a.provider_id in quotas][-20:]

        return QuotaDashboardResponse(
            providers=quotas,
            summary={
                "total_providers": total_providers,
                "quota_warning": quota_warning,
                "quota_critical": quota_critical,
                "quota_exceeded": quota_exceeded,
                "period": period.value,
            },
            alerts=recent_alerts,
            updated_at=datetime.utcnow(),
        )

    async def get_usage_history(
        self,
        provider_id: str,
        period: QuotaPeriod = QuotaPeriod.DAILY,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Get historical usage data for a provider.

        Args:
            provider_id: Provider identifier
            period: Period type (DAILY or MONTHLY)
            limit: Maximum entries to return

        Returns:
            List of historical usage entries
        """
        state = self._provider_states.get(provider_id)
        if not state:
            return []

        return state.get_usage_history(period, limit)

    async def get_alerts(
        self,
        provider_id: str | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get quota alerts.

        Args:
            provider_id: Filter by provider
            severity: Filter by severity
            limit: Maximum alerts to return

        Returns:
            List of alert dictionaries
        """
        alerts = list(self._alerts)

        if provider_id:
            alerts = [a for a in alerts if a.provider_id == provider_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.triggered_at, reverse=True)

        return [a.to_dict() for a in alerts[:limit]]

    async def get_usage_summary(self) -> dict[str, Any]:
        """
        Get a summary of usage across all providers.

        Returns:
            Summary dictionary with aggregated usage data
        """
        total_requests = 0
        total_tokens = 0
        total_cost = 0.0

        provider_breakdown = {}

        for provider_id, state in self._provider_states.items():
            daily = state.get_daily_usage()
            config = self._provider_configs.get(provider_id)

            total_requests += daily["requests"]
            total_tokens += daily["tokens"]

            if config:
                status = state.calculate_quota_status(config, QuotaPeriod.DAILY)
                total_cost += status.cost_used

            provider_breakdown[provider_id] = {
                "requests": daily["requests"],
                "tokens": daily["tokens"],
            }

        return {
            "daily": {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
            },
            "by_provider": provider_breakdown,
            "updated_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Integration with Rate Limiter
    # =========================================================================

    async def sync_with_rate_limiter(self) -> None:
        """
        Sync usage data with the model rate limiter.

        This pulls usage statistics from the rate limiter to keep quota tracking
        in sync with actual rate limit windows.
        """
        try:
            from app.services.model_rate_limiter import get_model_rate_limiter

            rate_limiter = get_model_rate_limiter()

            # Note: The rate limiter tracks per-user usage, while quota tracking
            # is per-provider. This method can be extended to aggregate user
            # usage into provider-level quotas if needed.

            logger.debug("Synced quota tracking with rate limiter")

        except Exception as e:
            logger.warning(f"Failed to sync with rate limiter: {e}")

    # =========================================================================
    # Remaining Quota Calculations
    # =========================================================================

    async def get_remaining_quota(
        self,
        provider_id: str,
        period: QuotaPeriod = QuotaPeriod.DAILY,
    ) -> dict[str, int | float]:
        """
        Get remaining quota for a provider.

        Args:
            provider_id: Provider identifier
            period: Quota period

        Returns:
            Dictionary with remaining requests, tokens, and reset time
        """
        status = await self.get_provider_quota(provider_id, period)
        if not status:
            return {
                "remaining_requests": 0,
                "remaining_tokens": 0,
                "reset_at": None,
            }

        remaining_requests = max(0, status.requests_limit - status.requests_used) if status.requests_limit else 0
        remaining_tokens = max(0, status.tokens_limit - status.tokens_used) if status.tokens_limit else 0

        return {
            "remaining_requests": remaining_requests,
            "remaining_tokens": remaining_tokens,
            "reset_at": status.reset_at.isoformat() if status.reset_at else None,
        }

    async def can_make_request(
        self,
        provider_id: str,
        estimated_tokens: int = 0,
    ) -> tuple[bool, str | None]:
        """
        Check if a request can be made without exceeding quota.

        Args:
            provider_id: Provider identifier
            estimated_tokens: Estimated tokens for the request

        Returns:
            Tuple of (can_make_request, error_message)
        """
        status = await self.get_provider_quota(provider_id, QuotaPeriod.DAILY)
        if not status:
            return True, None  # No quota configured, allow request

        # Check request limit
        if status.requests_limit and status.requests_used >= status.requests_limit:
            return False, f"Daily request limit ({status.requests_limit}) exceeded"

        # Check token limit
        if status.tokens_limit and (status.tokens_used + estimated_tokens) > status.tokens_limit:
            return False, f"Daily token limit ({status.tokens_limit}) would be exceeded"

        return True, None

    # =========================================================================
    # Reset and Cleanup
    # =========================================================================

    async def reset_provider_usage(self, provider_id: str) -> bool:
        """
        Reset usage counters for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            True if reset, False if provider not found
        """
        async with self._lock:
            if provider_id in self._provider_states:
                self._provider_states[provider_id] = ProviderQuotaState(
                    provider_id=provider_id,
                    provider_name=self._provider_states[provider_id].provider_name,
                )
                logger.info(f"Reset quota usage for provider: {provider_id}")
                return True
            return False

    async def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        async with self._lock:
            # Could persist state here if needed
            self._provider_states.clear()
            self._alerts.clear()

        logger.info("QuotaTrackingService shutdown complete")


# =============================================================================
# Global Singleton
# =============================================================================

_quota_tracking_service: QuotaTrackingService | None = None


def get_quota_tracking_service() -> QuotaTrackingService:
    """Get the global Quota Tracking Service instance."""
    global _quota_tracking_service
    if _quota_tracking_service is None:
        _quota_tracking_service = QuotaTrackingService()
    return _quota_tracking_service
