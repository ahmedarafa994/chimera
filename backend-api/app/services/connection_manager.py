"""
API Connection Manager Service
Direct-only API connection management.
Uses centralized settings for provider endpoint resolution.
"""

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import APIConnectionMode, settings


@dataclass
class ConnectionStatus:
    """Connection status information."""

    mode: str
    is_connected: bool
    base_url: str
    error_message: str | None = None
    latency_ms: float | None = None
    available_models: list | None = None


class ConnectionManager:
    """
    Manages direct API connections using centralized settings.
    """

    def __init__(self):
        self._mode: APIConnectionMode = APIConnectionMode.DIRECT

    @property
    def mode(self) -> APIConnectionMode:
        """Get current connection mode."""
        return self._mode

    @property
    def base_url(self) -> str:
        """Get the base URL for direct mode using centralized settings."""
        return settings.get_provider_endpoint("google")

    @property
    def api_key(self) -> str | None:
        """Get the API key for direct mode."""
        return settings.GOOGLE_API_KEY

    async def check_connection(self) -> ConnectionStatus:
        """
        Check connection status for direct mode.
        Returns detailed status information.
        """
        import time

        start_time = time.time()
        try:
            return await self._check_direct_connection(start_time)
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ConnectionStatus(
                mode=self._mode.value,
                is_connected=False,
                base_url=self.base_url,
                error_message=str(e),
                latency_ms=latency,
            )

    async def _check_direct_connection(self, start_time: float) -> ConnectionStatus:
        """Check direct Google API connection using centralized settings."""
        import time

        direct_base_url = settings.get_provider_endpoint("google")
        api_key = settings.GOOGLE_API_KEY

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                url = f"{direct_base_url}/models?key={api_key}"
                response = await client.get(url)
                latency = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()
                    models = [
                        m.get("name", "").replace("models/", "") for m in data.get("models", [])
                    ]
                    return ConnectionStatus(
                        mode="direct",
                        is_connected=True,
                        base_url=direct_base_url,
                        latency_ms=latency,
                        available_models=models[:20],
                    )
                else:
                    content_type = response.headers.get("content-type", "")
                    if content_type.startswith("application/json"):
                        error_data = response.json()
                    else:
                        error_data = {}
                    error_msg = error_data.get("error", {}).get("message", response.text[:200])
                    return ConnectionStatus(
                        mode="direct",
                        is_connected=False,
                        base_url=direct_base_url,
                        error_message=(f"HTTP {response.status_code}: {error_msg}"),
                        latency_ms=latency,
                    )
            except httpx.ConnectError as e:
                latency = (time.time() - start_time) * 1000
                return ConnectionStatus(
                    mode="direct",
                    is_connected=False,
                    base_url=direct_base_url,
                    error_message=f"Connection failed: {e!s}",
                    latency_ms=latency,
                )

    def get_config_summary(self) -> dict[str, Any]:
        """Get current configuration summary with direct provider endpoints."""
        direct_url = settings.get_provider_endpoint("google")
        providers = settings.get_all_provider_endpoints()

        return {
            "current_mode": self._mode.value,
            "direct": {
                "url": direct_url,
                "api_key_configured": bool(settings.GOOGLE_API_KEY),
                "api_key_preview": (
                    f"{settings.GOOGLE_API_KEY[:10]}..." if settings.GOOGLE_API_KEY else None
                ),
            },
            "providers": providers,
        }


# Singleton instance
connection_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the connection manager singleton."""
    return connection_manager
