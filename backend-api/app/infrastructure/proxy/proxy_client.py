"""
ProxyClient for AIClient-2-API Server Communication.

STORY-1.3: Handles HTTP communication with the proxy server at localhost:8080.

Features:
- Connection pooling (5-10 connections)
- Configurable timeouts
- JSON request/response handling
- Retry logic with exponential backoff
- Health check support
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import logger


class ProxyConnectionState(str, Enum):
    """Proxy connection state enumeration."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class ProxyRequestMetrics:
    """Metrics for a single proxy request."""

    request_id: str
    provider: str
    model: str | None
    latency_ms: float
    success: bool
    error: str | None = None
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProxyHealthStatus:
    """Health status of the proxy server."""

    is_healthy: bool
    latency_ms: float
    last_check: float
    error: str | None = None
    consecutive_failures: int = 0
    connection_state: ProxyConnectionState = ProxyConnectionState.DISCONNECTED


class ProxyError(Exception):
    """Base exception for proxy-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after


class ProxyConnectionError(ProxyError):
    """Raised when connection to proxy server fails."""

    pass


class ProxyTimeoutError(ProxyError):
    """Raised when proxy request times out."""

    pass


class ProxyResponseError(ProxyError):
    """Raised when proxy returns an error response."""

    pass


class ProxyClient:
    """
    HTTP client for communicating with the AIClient-2-API proxy server.

    Implements connection pooling, configurable timeouts, and retry logic
    for reliable proxy communication.
    """

    # Default configuration
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_CONNECT_TIMEOUT = 10.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_BASE_DELAY = 1.0
    DEFAULT_RETRY_MAX_DELAY = 30.0
    MIN_POOL_SIZE = 5
    MAX_POOL_SIZE = 10

    def __init__(
        self,
        endpoint: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        """
        Initialize the ProxyClient.

        Args:
            endpoint: Proxy server endpoint (defaults to config setting)
            timeout: Request timeout in seconds (defaults to config setting)
            max_retries: Maximum retry attempts (defaults to 3)
        """
        self._endpoint = endpoint or settings.PROXY_MODE_ENDPOINT
        self._timeout = timeout or settings.PROXY_MODE_TIMEOUT
        self._max_retries = max_retries or self.DEFAULT_MAX_RETRIES

        self._client: httpx.AsyncClient | None = None
        self._health_status = ProxyHealthStatus(
            is_healthy=False,
            latency_ms=0.0,
            last_check=0.0,
        )
        self._request_count = 0
        self._error_count = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"ProxyClient initialized: endpoint={self._endpoint}, "
            f"timeout={self._timeout}s, max_retries={self._max_retries}"
        )

    @property
    def endpoint(self) -> str:
        """Get the proxy server endpoint."""
        return self._endpoint

    @property
    def is_healthy(self) -> bool:
        """Check if the proxy server is healthy."""
        return self._health_status.is_healthy

    @property
    def health_status(self) -> ProxyHealthStatus:
        """Get the current health status."""
        return self._health_status

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            async with self._lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            connect=self.DEFAULT_CONNECT_TIMEOUT,
                            read=self._timeout,
                            write=self._timeout,
                            pool=self._timeout,
                        ),
                        limits=httpx.Limits(
                            max_keepalive_connections=self.MIN_POOL_SIZE,
                            max_connections=self.MAX_POOL_SIZE,
                            keepalive_expiry=30.0,
                        ),
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "User-Agent": "Chimera-ProxyClient/1.0",
                        },
                    )
                    self._health_status.connection_state = ProxyConnectionState.CONNECTED
                    logger.debug("ProxyClient HTTP client created with connection pool")
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self._health_status.connection_state = ProxyConnectionState.DISCONNECTED
            logger.info("ProxyClient closed")

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay for retry."""
        delay = self.DEFAULT_RETRY_BASE_DELAY * (2**attempt)
        return min(delay, self.DEFAULT_RETRY_MAX_DELAY)

    async def send_request(
        self,
        provider: str,
        model: str | None,
        prompt: str,
        config: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a generation request through the proxy server.

        Args:
            provider: Target LLM provider (e.g., 'openai', 'anthropic')
            model: Model identifier (e.g., 'gpt-4', 'claude-3')
            prompt: The prompt text to send
            config: Optional generation configuration
            request_id: Optional request identifier for tracking

        Returns:
            dict: Response from the proxy server

        Raises:
            ProxyConnectionError: Connection to proxy failed
            ProxyTimeoutError: Request timed out
            ProxyResponseError: Proxy returned an error
        """
        client = await self._get_client()
        request_id = request_id or f"proxy_{int(time.time() * 1000)}"

        payload = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "config": config or {},
            "request_id": request_id,
        }

        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = await client.post(
                    f"{self._endpoint}/v1/generate",
                    json=payload,
                )

                latency_ms = (time.time() - start_time) * 1000
                self._request_count += 1

                if response.status_code == 200:
                    result = response.json()
                    logger.debug(
                        f"Proxy request successful: {request_id}, " f"latency={latency_ms:.2f}ms"
                    )
                    return result

                # Handle error responses
                error_msg = self._parse_error_response(response)
                self._error_count += 1

                if response.status_code == 429:
                    retry_after = self._get_retry_after(response)
                    raise ProxyResponseError(
                        f"Rate limited by proxy: {error_msg}",
                        status_code=429,
                        retry_after=retry_after,
                    )

                if response.status_code >= 500:
                    last_error = ProxyResponseError(
                        f"Proxy server error: {error_msg}",
                        status_code=response.status_code,
                    )
                    await asyncio.sleep(self._calculate_retry_delay(attempt))
                    continue

                raise ProxyResponseError(
                    f"Proxy request failed: {error_msg}",
                    status_code=response.status_code,
                )

            except httpx.ConnectError as e:
                self._error_count += 1
                self._health_status.is_healthy = False
                self._health_status.connection_state = ProxyConnectionState.ERROR
                last_error = ProxyConnectionError(
                    f"Failed to connect to proxy at {self._endpoint}: {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._calculate_retry_delay(attempt))
                    continue
                raise last_error

            except httpx.TimeoutException as e:
                self._error_count += 1
                last_error = ProxyTimeoutError(
                    f"Proxy request timed out after {self._timeout}s: {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._calculate_retry_delay(attempt))
                    continue
                raise last_error

        # All retries exhausted
        if last_error:
            raise last_error
        raise ProxyError("Proxy request failed after all retries")

    def _parse_error_response(self, response: httpx.Response) -> str:
        """Parse error message from proxy response."""
        try:
            data = response.json()
            return data.get("error", data.get("message", str(data)))
        except Exception:
            return response.text or f"HTTP {response.status_code}"

    def _get_retry_after(self, response: httpx.Response) -> float:
        """Extract retry-after header value."""
        retry_after = response.headers.get("Retry-After", "60")
        try:
            return float(retry_after)
        except ValueError:
            return 60.0

    async def check_health(self) -> ProxyHealthStatus:
        """
        Check the health of the proxy server.

        Returns:
            ProxyHealthStatus: Current health status
        """
        start_time = time.time()
        self._health_status.last_check = start_time

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._endpoint}/health",
                timeout=5.0,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                self._health_status.is_healthy = True
                self._health_status.latency_ms = latency_ms
                self._health_status.error = None
                self._health_status.consecutive_failures = 0
                self._health_status.connection_state = ProxyConnectionState.CONNECTED
                logger.debug(f"Proxy health check passed: latency={latency_ms:.2f}ms")
            else:
                self._health_status.is_healthy = False
                self._health_status.error = f"HTTP {response.status_code}"
                self._health_status.consecutive_failures += 1

        except httpx.ConnectError as e:
            self._health_status.is_healthy = False
            self._health_status.error = f"Connection failed: {e}"
            self._health_status.consecutive_failures += 1
            self._health_status.connection_state = ProxyConnectionState.ERROR

        except httpx.TimeoutException:
            self._health_status.is_healthy = False
            self._health_status.error = "Health check timed out"
            self._health_status.consecutive_failures += 1

        except Exception as e:
            self._health_status.is_healthy = False
            self._health_status.error = str(e)
            self._health_status.consecutive_failures += 1
            logger.warning(f"Proxy health check failed: {e}")

        return self._health_status

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "endpoint": self._endpoint,
            "timeout": self._timeout,
            "max_retries": self._max_retries,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count if self._request_count > 0 else 0.0
            ),
            "is_healthy": self._health_status.is_healthy,
            "connection_state": self._health_status.connection_state.value,
            "last_health_check": self._health_status.last_check,
            "consecutive_failures": self._health_status.consecutive_failures,
        }


# Global proxy client instance
_proxy_client: ProxyClient | None = None


def get_proxy_client() -> ProxyClient:
    """Get or create the global ProxyClient instance."""
    global _proxy_client
    if _proxy_client is None:
        _proxy_client = ProxyClient()
    return _proxy_client


async def close_proxy_client() -> None:
    """Close the global ProxyClient instance."""
    global _proxy_client
    if _proxy_client is not None:
        await _proxy_client.close()
        _proxy_client = None
