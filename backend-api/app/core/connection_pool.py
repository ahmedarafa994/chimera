"""HTTP Connection Pool Management for LLM Providers.

PERF-001 FIX: Implements efficient connection pooling for HTTP/HTTPS connections
to LLM providers. This reduces connection overhead and improves throughput.

Features:
- Configurable pool size for concurrent connections
- Connection reuse and keep-alive
- Pool statistics for monitoring
- Provider-specific pool configuration
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class PoolStats:
    """Statistics for a connection pool."""

    pool_name: str
    created_at: float = field(default_factory=time.time)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    last_request_time: float | None = None
    avg_response_time_ms: float = 0.0
    active_connections: int = 0
    idle_connections: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pool_name": self.pool_name,
            "created_at": self.created_at,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
            ),
            "total_response_time_ms": self.total_response_time_ms,
            "avg_response_time_ms": self.avg_response_time_ms,
            "last_request_time": self.last_request_time,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
        }


class PooledHTTPAdapter(HTTPAdapter):
    """Custom HTTPAdapter with connection pooling and metrics tracking.

    PERF-001 FIX: Enhances standard HTTPAdapter with:
    - Configurable pool size
    - Connection keep-alive
    - Request metrics tracking
    """

    def __init__(
        self,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
        max_retries: int = 3,
        pool_block: bool = False,
        stats_name: str = "default",
    ) -> None:
        """Initialize pooled HTTP adapter.

        Args:
            pool_connections: Number of connection pools to cache
            pool_maxsize: Maximum number of connections in pool
            max_retries: Maximum number of retries
            pool_block: Block the pool when full
            stats_name: Name for metrics tracking

        """
        self.stats_name = stats_name
        self.stats = PoolStats(pool_name=stats_name)
        self._lock = threading.Lock()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Conservative backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )

        super().__init__(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy,
            pool_block=pool_block,
        )

        logger.info(
            f"Created PooledHTTPAdapter '{stats_name}': "
            f"pool_connections={pool_connections}, pool_maxsize={pool_maxsize}",
        )

    def send(self, request, **kwargs):
        """Send request with metrics tracking."""
        start_time = time.time()
        self.stats.total_requests += 1
        self.stats.last_request_time = start_time

        try:
            response = super().send(request, **kwargs)

            # Update stats
            response_time_ms = (time.time() - start_time) * 1000
            self.stats.successful_requests += 1
            self.stats.total_response_time_ms += response_time_ms
            self.stats.avg_response_time_ms = (
                self.stats.total_response_time_ms / self.stats.successful_requests
            )

            # Update connection counts from poolmanager
            if hasattr(self, "poolmanager"):
                self.stats.active_connections = getattr(self.poolmanager, "num_connections", 0)
                self.stats.idle_connections = getattr(self.poolmanager, "num_idle_connections", 0)

            logger.debug(
                f"{self.stats_name}: {request.method} {response.status_code} "
                f"in {response_time_ms:.2f}ms",
            )

            return response

        except Exception as e:
            self.stats.failed_requests += 1
            logger.exception(f"{self.stats_name}: Request failed - {e}")
            raise


class ConnectionPoolManager:
    """Manages connection pools for LLM providers.

    PERF-001 FIX: Centralized connection pool management with:
    - Provider-specific pool configuration
    - Automatic session creation with pooling
    - Pool statistics and monitoring
    - Graceful shutdown
    """

    _instance: "ConnectionPoolManager | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global pool manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize connection pool manager."""
        if hasattr(self, "_initialized"):
            return

        self._pools: dict[str, PooledHTTPAdapter] = {}
        self._sessions: dict[str, requests.Session] = {}
        self._config: dict[str, dict[str, Any]] = {
            "openai": {
                "pool_connections": 10,
                "pool_maxsize": 20,
                "max_retries": 3,
            },
            "anthropic": {
                "pool_connections": 10,
                "pool_maxsize": 20,
                "max_retries": 3,
            },
            "google": {
                "pool_connections": 10,
                "pool_maxsize": 20,
                "max_retries": 3,
            },
            "deepseek": {
                "pool_connections": 5,
                "pool_maxsize": 10,
                "max_retries": 2,
            },
            "default": {
                "pool_connections": 5,
                "pool_maxsize": 10,
                "max_retries": 2,
            },
        }
        self._initialized = True
        logger.info("ConnectionPoolManager initialized")

    def get_adapter(self, provider: str) -> PooledHTTPAdapter:
        """Get or create a pooled HTTP adapter for a provider.

        Args:
            provider: Provider name (e.g., "openai", "anthropic", "google")

        Returns:
            PooledHTTPAdapter with provider-specific configuration

        """
        provider_key = provider.lower()

        if provider_key not in self._pools:
            with threading.Lock():
                # Double-check after acquiring lock
                if provider_key not in self._pools:
                    config = self._config.get(provider_key, self._config["default"])
                    adapter = PooledHTTPAdapter(
                        stats_name=provider_key,
                        **config,
                    )
                    self._pools[provider_key] = adapter
                    logger.info(
                        f"Created connection pool for '{provider_key}': "
                        f"pool_maxsize={config['pool_maxsize']}",
                    )

        return self._pools[provider_key]

    def create_session(self, provider: str, timeout: float | None = None) -> requests.Session:
        """Create a requests session with connection pooling.

        Args:
            provider: Provider name for pool configuration
            timeout: Request timeout in seconds

        Returns:
            Configured requests.Session with pooled adapter

        """
        provider_key = provider.lower()

        # Check for existing session
        if provider_key in self._sessions:
            return self._sessions[provider_key]

        with threading.Lock():
            # Double-check after acquiring lock
            if provider_key not in self._sessions:
                session = requests.Session()

                # Add pooled adapter for both http and https
                adapter = self.get_adapter(provider)
                session.mount("http://", adapter)
                session.mount("https://", adapter)

                # Set default timeout if provided
                if timeout:
                    session.request = lambda *args, **kwargs: super(
                        requests.Session,
                        session,
                    ).request(*args, timeout=timeout, **kwargs)

                self._sessions[provider_key] = session
                logger.info(
                    f"Created pooled session for '{provider_key}' "
                    f"(timeout: {timeout or 'default'})",
                )

        return self._sessions[provider_key]

    def get_stats(self, provider: str | None = None) -> dict[str, Any]:
        """Get connection pool statistics.

        Args:
            provider: Specific provider stats, or all if None

        Returns:
            Dictionary of pool statistics

        """
        if provider:
            provider_key = provider.lower()
            if provider_key in self._pools:
                return self._pools[provider_key].stats.to_dict()
            return {}

        # Return all pool stats
        return {name: adapter.stats.to_dict() for name, adapter in self._pools.items()}

    def reset_stats(self, provider: str | None = None) -> None:
        """Reset statistics for a provider or all providers."""
        if provider:
            provider_key = provider.lower()
            if provider_key in self._pools:
                adapter = self._pools[provider_key]
                adapter.stats = PoolStats(pool_name=provider_key)
                logger.info(f"Reset stats for '{provider_key}'")
        else:
            # Reset all
            for name, adapter in self._pools.items():
                adapter.stats = PoolStats(pool_name=name)
            logger.info("Reset all pool stats")

    def close(self) -> None:
        """Close all sessions and cleanup pools."""
        logger.info("Closing connection pools...")
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._pools.clear()
        logger.info("All connection pools closed")


# Global instance
connection_pool_manager = ConnectionPoolManager()


def get_pooled_session(provider: str, timeout: float | None = None) -> requests.Session:
    """Convenience function to get a pooled session for a provider.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        timeout: Optional timeout in seconds

    Returns:
        requests.Session with connection pooling configured

    Example:
        session = get_pooled_session("openai", timeout=30)
        response = session.post("https://api.openai.com/v1/chat/completions", ...)

    """
    return connection_pool_manager.create_session(provider, timeout)


def get_pool_stats(provider: str | None = None) -> dict[str, Any]:
    """Get connection pool statistics.

    Args:
        provider: Specific provider or all if None

    Returns:
        Dictionary of pool statistics

    """
    return connection_pool_manager.get_stats(provider)


__all__ = [
    "ConnectionPoolManager",
    "PoolStats",
    "PooledHTTPAdapter",
    "connection_pool_manager",
    "get_pool_stats",
    "get_pooled_session",
]
