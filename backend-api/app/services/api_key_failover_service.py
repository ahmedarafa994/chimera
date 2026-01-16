# =============================================================================
# Chimera - API Key Failover Manager Service
# =============================================================================
# Intelligent failover logic for automatic API key rotation when primary keys
# hit rate limits or experience failures. Supports multiple failover strategies
# and maintains audit logs for all failover events.
#
# Part of Feature: API Key Management & Provider Health Dashboard
# Subtask 3.1: Create Failover Manager Service
# =============================================================================

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Default cooldown periods in seconds
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 60  # 1 minute for rate limit
DEFAULT_ERROR_COOLDOWN_SECONDS = 30  # 30 seconds for transient errors
DEFAULT_MAX_COOLDOWN_SECONDS = 600  # 10 minutes max cooldown

# Cooldown multiplier for exponential backoff
COOLDOWN_BACKOFF_MULTIPLIER = 2.0

# Maximum failover events to retain for audit
MAX_FAILOVER_EVENTS_RETAINED = 500

# Rate limit detection patterns in error messages
RATE_LIMIT_ERROR_PATTERNS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "quota exceeded",
    "quota_exceeded",
    "too many requests",
    "429",
    "resource exhausted",
    "capacity exceeded",
    "requests per minute",
    "tokens per minute",
    "rpm limit",
    "tpm limit",
]


# =============================================================================
# Enums
# =============================================================================


class FailoverReason(str, Enum):
    """Reason for failover event."""

    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    KEY_INACTIVE = "key_inactive"
    KEY_EXPIRED = "key_expired"
    MANUAL = "manual"


class FailoverStrategy(str, Enum):
    """Strategy for selecting backup keys during failover."""

    PRIORITY = "priority"  # Use key with lowest priority number
    ROUND_ROBIN = "round_robin"  # Rotate through keys in order
    LEAST_USED = "least_used"  # Use key with fewest recent requests
    RANDOM = "random"  # Random selection from available keys


class KeyCooldownState(str, Enum):
    """State of a key's cooldown."""

    AVAILABLE = "available"
    COOLING_DOWN = "cooling_down"
    PERMANENTLY_BLOCKED = "permanently_blocked"


# =============================================================================
# Models
# =============================================================================


class FailoverEvent(BaseModel):
    """Record of a failover event for auditing."""

    event_id: str = Field(..., description="Unique event identifier")
    provider_id: str = Field(..., description="Provider identifier")
    from_key_id: str = Field(..., description="Key being failed over from")
    to_key_id: str | None = Field(default=None, description="Key being failed over to")
    reason: FailoverReason = Field(..., description="Reason for failover")
    error_message: str | None = Field(default=None, description="Error message that triggered failover")
    triggered_at: datetime = Field(default_factory=datetime.utcnow, description="When failover occurred")
    cooldown_until: datetime | None = Field(default=None, description="When the from_key will be available again")
    success: bool = Field(default=False, description="Whether failover succeeded")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_id": "fo_1705234567890_abc123",
                "provider_id": "openai",
                "from_key_id": "key_openai_primary_001",
                "to_key_id": "key_openai_backup_001",
                "reason": "rate_limited",
                "error_message": "Rate limit exceeded: 429 Too Many Requests",
                "triggered_at": "2025-01-11T15:30:00Z",
                "cooldown_until": "2025-01-11T15:31:00Z",
                "success": True,
            }
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "provider_id": self.provider_id,
            "from_key_id": self.from_key_id,
            "to_key_id": self.to_key_id,
            "reason": self.reason.value,
            "error_message": self.error_message,
            "triggered_at": self.triggered_at.isoformat(),
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "success": self.success,
            "metadata": self.metadata,
        }


class KeyCooldownInfo(BaseModel):
    """Information about a key's cooldown state."""

    key_id: str = Field(..., description="Key identifier")
    provider_id: str = Field(..., description="Provider identifier")
    state: KeyCooldownState = Field(default=KeyCooldownState.AVAILABLE, description="Current cooldown state")
    cooldown_until: datetime | None = Field(default=None, description="When cooldown ends")
    reason: FailoverReason | None = Field(default=None, description="Reason for cooldown")
    consecutive_failures: int = Field(default=0, description="Number of consecutive failures")
    last_failure_at: datetime | None = Field(default=None, description="When last failure occurred")
    total_failures: int = Field(default=0, description="Total failures for this key")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "key_id": "key_openai_primary_001",
                "provider_id": "openai",
                "state": "cooling_down",
                "cooldown_until": "2025-01-11T15:31:00Z",
                "reason": "rate_limited",
                "consecutive_failures": 3,
                "total_failures": 15,
            }
        }
    )

    def is_available(self) -> bool:
        """Check if key is available for use."""
        if self.state == KeyCooldownState.PERMANENTLY_BLOCKED:
            return False
        if self.state == KeyCooldownState.COOLING_DOWN:
            if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
                return False
            # Cooldown expired, reset state
            self.state = KeyCooldownState.AVAILABLE
            self.cooldown_until = None
        return True

    def remaining_cooldown_seconds(self) -> float:
        """Get remaining cooldown time in seconds."""
        if not self.cooldown_until or self.state != KeyCooldownState.COOLING_DOWN:
            return 0
        remaining = (self.cooldown_until - datetime.utcnow()).total_seconds()
        return max(0, remaining)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "provider_id": self.provider_id,
            "state": self.state.value,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "reason": self.reason.value if self.reason else None,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "total_failures": self.total_failures,
            "remaining_cooldown_seconds": self.remaining_cooldown_seconds(),
            "is_available": self.is_available(),
        }


class ProviderFailoverState(BaseModel):
    """Failover state for a provider."""

    provider_id: str = Field(..., description="Provider identifier")
    current_key_id: str | None = Field(default=None, description="Currently active key ID")
    primary_key_id: str | None = Field(default=None, description="Primary key ID")
    is_using_backup: bool = Field(default=False, description="Whether currently using a backup key")
    last_failover_at: datetime | None = Field(default=None, description="When last failover occurred")
    failover_count: int = Field(default=0, description="Total failovers for this provider")
    strategy: FailoverStrategy = Field(default=FailoverStrategy.PRIORITY, description="Failover strategy")
    round_robin_index: int = Field(default=0, description="Current index for round-robin strategy")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider_id": "openai",
                "current_key_id": "key_openai_backup_001",
                "primary_key_id": "key_openai_primary_001",
                "is_using_backup": True,
                "last_failover_at": "2025-01-11T15:30:00Z",
                "failover_count": 5,
                "strategy": "priority",
            }
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "current_key_id": self.current_key_id,
            "primary_key_id": self.primary_key_id,
            "is_using_backup": self.is_using_backup,
            "last_failover_at": self.last_failover_at.isoformat() if self.last_failover_at else None,
            "failover_count": self.failover_count,
            "strategy": self.strategy.value,
        }


class FailoverConfig(BaseModel):
    """Configuration for failover behavior per provider."""

    provider_id: str = Field(..., description="Provider identifier")
    enabled: bool = Field(default=True, description="Whether failover is enabled")
    strategy: FailoverStrategy = Field(default=FailoverStrategy.PRIORITY, description="Failover strategy")
    rate_limit_cooldown_seconds: int = Field(
        default=DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
        ge=5,
        le=3600,
        description="Cooldown for rate limit errors",
    )
    error_cooldown_seconds: int = Field(
        default=DEFAULT_ERROR_COOLDOWN_SECONDS,
        ge=5,
        le=3600,
        description="Cooldown for transient errors",
    )
    max_cooldown_seconds: int = Field(
        default=DEFAULT_MAX_COOLDOWN_SECONDS,
        ge=60,
        le=7200,
        description="Maximum cooldown duration",
    )
    use_exponential_backoff: bool = Field(
        default=True,
        description="Use exponential backoff for cooldown",
    )
    max_consecutive_failures: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max consecutive failures before permanent block",
    )
    auto_recover: bool = Field(
        default=True,
        description="Automatically try to recover to primary key",
    )
    recovery_check_interval_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Interval for checking if primary can be recovered",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "enabled": self.enabled,
            "strategy": self.strategy.value,
            "rate_limit_cooldown_seconds": self.rate_limit_cooldown_seconds,
            "error_cooldown_seconds": self.error_cooldown_seconds,
            "max_cooldown_seconds": self.max_cooldown_seconds,
            "use_exponential_backoff": self.use_exponential_backoff,
            "max_consecutive_failures": self.max_consecutive_failures,
            "auto_recover": self.auto_recover,
            "recovery_check_interval_seconds": self.recovery_check_interval_seconds,
        }


# =============================================================================
# API Key Failover Service
# =============================================================================


class ApiKeyFailoverService:
    """
    Intelligent failover service for API key rotation.

    Features:
    - Detect rate limit errors from provider responses
    - Automatically switch to backup API keys when primary is rate limited
    - Implement cooldown periods with exponential backoff
    - Log all failover events for auditing
    - Support configurable failover strategies (priority, round-robin, least-used)

    Usage:
        from app.services.api_key_failover_service import get_api_key_failover_service

        failover_service = get_api_key_failover_service()

        # Get best available key for a request
        key_id, api_key = await failover_service.get_available_key("openai")

        # Report error (will trigger failover if needed)
        await failover_service.handle_error(
            provider_id="openai",
            key_id="key_openai_primary_001",
            error="Rate limit exceeded",
            is_rate_limit=True,
        )

        # Check failover status
        status = await failover_service.get_failover_status("openai")
    """

    _instance: "ApiKeyFailoverService | None" = None

    def __new__(cls) -> "ApiKeyFailoverService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Provider failover states
        self._provider_states: dict[str, ProviderFailoverState] = {}

        # Key cooldown tracking
        self._key_cooldowns: dict[str, KeyCooldownInfo] = {}

        # Failover events for audit
        self._failover_events: deque[FailoverEvent] = deque(maxlen=MAX_FAILOVER_EVENTS_RETAINED)

        # Per-provider configurations
        self._configs: dict[str, FailoverConfig] = {}

        # Callbacks
        self._failover_callbacks: list[Callable[[FailoverEvent], None]] = []
        self._recovery_callbacks: list[Callable[[str, str], None]] = []

        # Async lock
        self._lock = asyncio.Lock()

        # Recovery check task
        self._recovery_task: asyncio.Task | None = None
        self._running = False

        self._initialized = True

        logger.info("ApiKeyFailoverService initialized")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the failover service with background recovery checks."""
        if self._running:
            return

        self._running = True
        self._recovery_task = asyncio.create_task(self._recovery_check_loop())
        logger.info("ApiKeyFailoverService started")

    async def stop(self) -> None:
        """Stop the failover service."""
        if not self._running:
            return

        self._running = False
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        logger.info("ApiKeyFailoverService stopped")

    async def _recovery_check_loop(self) -> None:
        """Background loop to check for key recovery opportunities."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_for_recovery()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery check loop: {e}")
                await asyncio.sleep(30)

    async def _check_for_recovery(self) -> None:
        """Check if any primary keys can be recovered."""
        async with self._lock:
            for provider_id, state in self._provider_states.items():
                config = self._get_config(provider_id)
                if not config.auto_recover or not state.is_using_backup:
                    continue

                # Check if primary key's cooldown has expired
                primary_cooldown = self._key_cooldowns.get(state.primary_key_id or "")
                if primary_cooldown and primary_cooldown.is_available():
                    await self._recover_to_primary(provider_id)

    async def _recover_to_primary(self, provider_id: str) -> bool:
        """
        Attempt to recover to primary key.

        Args:
            provider_id: Provider identifier

        Returns:
            True if recovery successful
        """
        state = self._provider_states.get(provider_id)
        if not state or not state.primary_key_id:
            return False

        # Check if primary is really available
        primary_cooldown = self._key_cooldowns.get(state.primary_key_id)
        if primary_cooldown and not primary_cooldown.is_available():
            return False

        # Recover to primary
        old_key_id = state.current_key_id
        state.current_key_id = state.primary_key_id
        state.is_using_backup = False

        logger.info(
            f"Recovered to primary key for {provider_id}: "
            f"{state.primary_key_id} (was: {old_key_id})"
        )

        # Trigger callbacks
        for callback in self._recovery_callbacks:
            try:
                callback(provider_id, state.primary_key_id)
            except Exception as e:
                logger.error(f"Error in recovery callback: {e}")

        return True

    # =========================================================================
    # Configuration
    # =========================================================================

    def configure_provider(
        self,
        provider_id: str,
        enabled: bool = True,
        strategy: FailoverStrategy = FailoverStrategy.PRIORITY,
        rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
        error_cooldown_seconds: int = DEFAULT_ERROR_COOLDOWN_SECONDS,
        max_cooldown_seconds: int = DEFAULT_MAX_COOLDOWN_SECONDS,
        use_exponential_backoff: bool = True,
        max_consecutive_failures: int = 5,
        auto_recover: bool = True,
    ) -> None:
        """
        Configure failover behavior for a provider.

        Args:
            provider_id: Provider identifier
            enabled: Whether failover is enabled
            strategy: Failover strategy
            rate_limit_cooldown_seconds: Cooldown for rate limits
            error_cooldown_seconds: Cooldown for transient errors
            max_cooldown_seconds: Maximum cooldown
            use_exponential_backoff: Use exponential backoff
            max_consecutive_failures: Max failures before permanent block
            auto_recover: Auto-recover to primary
        """
        self._configs[provider_id] = FailoverConfig(
            provider_id=provider_id,
            enabled=enabled,
            strategy=strategy,
            rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
            error_cooldown_seconds=error_cooldown_seconds,
            max_cooldown_seconds=max_cooldown_seconds,
            use_exponential_backoff=use_exponential_backoff,
            max_consecutive_failures=max_consecutive_failures,
            auto_recover=auto_recover,
        )

        # Initialize provider state if not exists
        if provider_id not in self._provider_states:
            self._provider_states[provider_id] = ProviderFailoverState(
                provider_id=provider_id,
                strategy=strategy,
            )

        logger.info(f"Configured failover for provider: {provider_id} (strategy={strategy.value})")

    def _get_config(self, provider_id: str) -> FailoverConfig:
        """Get configuration for a provider, with defaults."""
        if provider_id in self._configs:
            return self._configs[provider_id]

        # Return default config
        return FailoverConfig(provider_id=provider_id)

    def set_strategy(self, provider_id: str, strategy: FailoverStrategy) -> None:
        """Set failover strategy for a provider."""
        config = self._get_config(provider_id)
        config.strategy = strategy

        state = self._provider_states.get(provider_id)
        if state:
            state.strategy = strategy

    # =========================================================================
    # Error Detection
    # =========================================================================

    @staticmethod
    def is_rate_limit_error(error_message: str | None) -> bool:
        """
        Detect if an error message indicates a rate limit.

        Args:
            error_message: Error message from provider

        Returns:
            True if this is a rate limit error
        """
        if not error_message:
            return False

        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in RATE_LIMIT_ERROR_PATTERNS)

    @staticmethod
    def detect_failover_reason(
        error_message: str | None,
        is_timeout: bool = False,
        is_circuit_breaker: bool = False,
    ) -> FailoverReason:
        """
        Detect the reason for failover from error context.

        Args:
            error_message: Error message
            is_timeout: Whether this was a timeout
            is_circuit_breaker: Whether circuit breaker is open

        Returns:
            FailoverReason enum
        """
        if is_circuit_breaker:
            return FailoverReason.CIRCUIT_BREAKER_OPEN

        if is_timeout:
            return FailoverReason.TIMEOUT

        if ApiKeyFailoverService.is_rate_limit_error(error_message):
            return FailoverReason.RATE_LIMITED

        return FailoverReason.ERROR

    # =========================================================================
    # Key Selection
    # =========================================================================

    async def get_available_key(
        self,
        provider_id: str,
        exclude_key_ids: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Get the best available API key for a provider.

        Integrates with ApiKeyStorageService to get keys and applies
        failover logic based on cooldowns and strategy.

        Args:
            provider_id: Provider identifier
            exclude_key_ids: Key IDs to exclude from selection

        Returns:
            Tuple of (key_id, decrypted_api_key) or (None, None) if no key available
        """
        try:
            from app.services.api_key_service import api_key_service
        except ImportError:
            logger.warning("ApiKeyStorageService not available")
            return None, None

        config = self._get_config(provider_id)
        exclude_ids = exclude_key_ids or []

        async with self._lock:
            # Get all active keys for this provider
            key_list = await api_key_service.list_keys(provider_id=provider_id)

            # Filter available keys
            available_keys = []
            for key_response in key_list.keys:
                if key_response.id in exclude_ids:
                    continue

                # Check if key is on cooldown
                cooldown_info = self._key_cooldowns.get(key_response.id)
                if cooldown_info and not cooldown_info.is_available():
                    continue

                # Check if key is rate limited or inactive
                if key_response.is_rate_limited or key_response.status.value not in ("active",):
                    continue

                available_keys.append(key_response)

            if not available_keys:
                logger.warning(f"No available keys for provider: {provider_id}")
                return None, None

            # Select key based on strategy
            selected_key = self._select_key_by_strategy(
                provider_id, available_keys, config.strategy
            )

            if not selected_key:
                return None, None

            # Get decrypted key
            decrypted = await api_key_service.get_decrypted_key(selected_key.id)

            # Update provider state
            state = self._provider_states.setdefault(
                provider_id,
                ProviderFailoverState(provider_id=provider_id, strategy=config.strategy),
            )
            state.current_key_id = selected_key.id

            # Determine if this is a backup key
            if state.primary_key_id and selected_key.id != state.primary_key_id:
                state.is_using_backup = True
            elif selected_key.role.value == "primary":
                state.primary_key_id = selected_key.id
                state.is_using_backup = False

            return selected_key.id, decrypted

    def _select_key_by_strategy(
        self,
        provider_id: str,
        available_keys: list,
        strategy: FailoverStrategy,
    ):
        """
        Select a key based on the configured strategy.

        Args:
            provider_id: Provider identifier
            available_keys: List of available ApiKeyResponse objects
            strategy: Selection strategy

        Returns:
            Selected key or None
        """
        if not available_keys:
            return None

        if strategy == FailoverStrategy.PRIORITY:
            # Sort by role (primary first) and priority
            role_order = {"primary": 0, "backup": 1, "fallback": 2}
            sorted_keys = sorted(
                available_keys,
                key=lambda k: (role_order.get(k.role.value, 99), k.priority),
            )
            return sorted_keys[0]

        elif strategy == FailoverStrategy.ROUND_ROBIN:
            state = self._provider_states.get(provider_id)
            if state:
                index = state.round_robin_index % len(available_keys)
                state.round_robin_index = (state.round_robin_index + 1) % len(available_keys)
                return available_keys[index]
            return available_keys[0]

        elif strategy == FailoverStrategy.LEAST_USED:
            # Sort by request count (ascending)
            sorted_keys = sorted(available_keys, key=lambda k: k.request_count)
            return sorted_keys[0]

        elif strategy == FailoverStrategy.RANDOM:
            import random
            return random.choice(available_keys)

        # Default to first available
        return available_keys[0]

    # =========================================================================
    # Error Handling and Failover
    # =========================================================================

    async def handle_error(
        self,
        provider_id: str,
        key_id: str,
        error: str | None = None,
        is_rate_limit: bool = False,
        is_timeout: bool = False,
        is_circuit_breaker: bool = False,
    ) -> tuple[str | None, str | None]:
        """
        Handle an error for a key and trigger failover if needed.

        Args:
            provider_id: Provider identifier
            key_id: Key that experienced the error
            error: Error message
            is_rate_limit: Whether this was explicitly a rate limit
            is_timeout: Whether this was a timeout
            is_circuit_breaker: Whether circuit breaker is open

        Returns:
            Tuple of (new_key_id, new_api_key) or (None, None) if no backup available
        """
        # Detect reason
        if is_rate_limit or self.is_rate_limit_error(error):
            reason = FailoverReason.RATE_LIMITED
        else:
            reason = self.detect_failover_reason(error, is_timeout, is_circuit_breaker)

        config = self._get_config(provider_id)

        async with self._lock:
            # Update cooldown info
            cooldown_info = self._key_cooldowns.setdefault(
                key_id,
                KeyCooldownInfo(key_id=key_id, provider_id=provider_id),
            )

            cooldown_info.consecutive_failures += 1
            cooldown_info.total_failures += 1
            cooldown_info.last_failure_at = datetime.utcnow()
            cooldown_info.reason = reason

            # Calculate cooldown duration
            base_cooldown = (
                config.rate_limit_cooldown_seconds
                if reason == FailoverReason.RATE_LIMITED
                else config.error_cooldown_seconds
            )

            if config.use_exponential_backoff:
                # Exponential backoff based on consecutive failures
                cooldown_seconds = min(
                    base_cooldown * (COOLDOWN_BACKOFF_MULTIPLIER ** (cooldown_info.consecutive_failures - 1)),
                    config.max_cooldown_seconds,
                )
            else:
                cooldown_seconds = base_cooldown

            cooldown_info.cooldown_until = datetime.utcnow() + timedelta(seconds=cooldown_seconds)
            cooldown_info.state = KeyCooldownState.COOLING_DOWN

            # Check for permanent block
            if cooldown_info.consecutive_failures >= config.max_consecutive_failures:
                cooldown_info.state = KeyCooldownState.PERMANENTLY_BLOCKED
                logger.warning(
                    f"Key {key_id} permanently blocked after "
                    f"{cooldown_info.consecutive_failures} consecutive failures"
                )

            # Record in api_key_service
            try:
                from app.services.api_key_service import api_key_service
                await api_key_service.record_rate_limit_hit(key_id)
            except Exception as e:
                logger.warning(f"Failed to record rate limit hit: {e}")

        # Attempt failover
        if config.enabled:
            new_key_id, new_api_key = await self.get_available_key(
                provider_id, exclude_key_ids=[key_id]
            )

            # Create failover event
            event = FailoverEvent(
                event_id=f"fo_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
                provider_id=provider_id,
                from_key_id=key_id,
                to_key_id=new_key_id,
                reason=reason,
                error_message=error[:500] if error else None,
                triggered_at=datetime.utcnow(),
                cooldown_until=cooldown_info.cooldown_until,
                success=new_key_id is not None,
                metadata={
                    "consecutive_failures": cooldown_info.consecutive_failures,
                    "cooldown_seconds": cooldown_seconds if 'cooldown_seconds' in dir() else 0,
                },
            )
            await self._record_failover_event(event)

            if new_key_id:
                # Update provider state
                state = self._provider_states.get(provider_id)
                if state:
                    state.current_key_id = new_key_id
                    state.is_using_backup = True
                    state.last_failover_at = datetime.utcnow()
                    state.failover_count += 1

                logger.info(
                    f"Failover successful for {provider_id}: "
                    f"{key_id} -> {new_key_id} (reason: {reason.value})"
                )
                return new_key_id, new_api_key

            logger.warning(f"Failover failed for {provider_id}: no backup keys available")

        return None, None

    async def _record_failover_event(self, event: FailoverEvent) -> None:
        """Record a failover event and trigger callbacks."""
        self._failover_events.append(event)

        logger.info(
            f"Failover event recorded: [{event.reason.value}] "
            f"{event.provider_id}: {event.from_key_id} -> {event.to_key_id or 'NONE'}"
        )

        # Trigger callbacks
        for callback in self._failover_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in failover callback: {e}")

    async def record_success(self, provider_id: str, key_id: str) -> None:
        """
        Record a successful request for a key.

        Resets consecutive failure counter.

        Args:
            provider_id: Provider identifier
            key_id: Key that succeeded
        """
        async with self._lock:
            cooldown_info = self._key_cooldowns.get(key_id)
            if cooldown_info:
                cooldown_info.consecutive_failures = 0

    # =========================================================================
    # Reset and Recovery
    # =========================================================================

    async def reset_provider(self, provider_id: str) -> bool:
        """
        Reset a provider to its primary key.

        Clears cooldowns and resets failover state.

        Args:
            provider_id: Provider identifier

        Returns:
            True if reset successful
        """
        async with self._lock:
            state = self._provider_states.get(provider_id)
            if not state:
                return False

            # Clear cooldowns for all keys of this provider
            keys_to_clear = [
                key_id for key_id, info in self._key_cooldowns.items()
                if info.provider_id == provider_id
            ]
            for key_id in keys_to_clear:
                cooldown = self._key_cooldowns[key_id]
                cooldown.state = KeyCooldownState.AVAILABLE
                cooldown.cooldown_until = None
                cooldown.consecutive_failures = 0

                # Clear rate limit in api_key_service
                try:
                    from app.services.api_key_service import api_key_service
                    await api_key_service.clear_rate_limit(key_id)
                except Exception as e:
                    logger.warning(f"Failed to clear rate limit: {e}")

            # Reset provider state
            if state.primary_key_id:
                state.current_key_id = state.primary_key_id
                state.is_using_backup = False

            logger.info(f"Reset failover state for provider: {provider_id}")
            return True

    async def clear_cooldown(self, key_id: str) -> bool:
        """
        Clear cooldown for a specific key.

        Args:
            key_id: Key identifier

        Returns:
            True if cooldown cleared
        """
        async with self._lock:
            cooldown = self._key_cooldowns.get(key_id)
            if not cooldown:
                return False

            cooldown.state = KeyCooldownState.AVAILABLE
            cooldown.cooldown_until = None
            cooldown.consecutive_failures = 0

            # Clear in api_key_service
            try:
                from app.services.api_key_service import api_key_service
                await api_key_service.clear_rate_limit(key_id)
            except Exception as e:
                logger.warning(f"Failed to clear rate limit: {e}")

            logger.info(f"Cleared cooldown for key: {key_id}")
            return True

    # =========================================================================
    # Callbacks
    # =========================================================================

    def register_failover_callback(self, callback: Callable[[FailoverEvent], None]) -> None:
        """Register a callback for failover events."""
        self._failover_callbacks.append(callback)

    def register_recovery_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for recovery events (provider_id, key_id)."""
        self._recovery_callbacks.append(callback)

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_failover_status(self, provider_id: str) -> dict[str, Any]:
        """
        Get current failover status for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Status dictionary
        """
        state = self._provider_states.get(provider_id)
        if not state:
            return {
                "provider_id": provider_id,
                "configured": False,
                "current_key_id": None,
                "is_using_backup": False,
            }

        # Get cooldowns for this provider's keys
        key_cooldowns = [
            info.to_dict()
            for info in self._key_cooldowns.values()
            if info.provider_id == provider_id
        ]

        return {
            "provider_id": provider_id,
            "configured": True,
            **state.to_dict(),
            "config": self._get_config(provider_id).to_dict(),
            "key_cooldowns": key_cooldowns,
        }

    async def get_all_failover_status(self) -> dict[str, Any]:
        """Get failover status for all providers."""
        statuses = {}
        for provider_id in self._provider_states:
            statuses[provider_id] = await self.get_failover_status(provider_id)

        return {
            "providers": statuses,
            "total_failovers": sum(s.failover_count for s in self._provider_states.values()),
            "using_backup_count": sum(
                1 for s in self._provider_states.values() if s.is_using_backup
            ),
        }

    async def get_failover_history(
        self,
        provider_id: str | None = None,
        reason: FailoverReason | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get failover event history.

        Args:
            provider_id: Filter by provider
            reason: Filter by reason
            limit: Maximum events to return

        Returns:
            List of failover event dictionaries
        """
        events = list(self._failover_events)

        if provider_id:
            events = [e for e in events if e.provider_id == provider_id]

        if reason:
            events = [e for e in events if e.reason == reason]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.triggered_at, reverse=True)

        return [e.to_dict() for e in events[:limit]]

    async def get_cooldown_info(self, key_id: str) -> dict[str, Any] | None:
        """Get cooldown info for a specific key."""
        info = self._key_cooldowns.get(key_id)
        if not info:
            return None
        return info.to_dict()


# =============================================================================
# Global Singleton
# =============================================================================

_api_key_failover_service: ApiKeyFailoverService | None = None


def get_api_key_failover_service() -> ApiKeyFailoverService:
    """Get the global API Key Failover Service instance."""
    global _api_key_failover_service
    if _api_key_failover_service is None:
        _api_key_failover_service = ApiKeyFailoverService()
    return _api_key_failover_service
