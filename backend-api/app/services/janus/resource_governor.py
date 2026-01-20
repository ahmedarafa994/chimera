"""Resource Governor Module.

Manages resource usage and prevents abuse
of the Janus adversarial simulation system.
"""

import logging
import time
from datetime import date
from typing import Any

from .config import get_config

logger = logging.getLogger(__name__)


class ResourceGovernor:
    """Manages resource usage and enforces budgets.

    Implements query budgeting, CPU/memory limits,
    and session timeouts to prevent resource exhaustion.
    """

    def __init__(
        self,
        daily_query_budget: int | None = None,
        cpu_limit: float | None = None,
        memory_limit_mb: int | None = None,
        session_timeout_sec: int | None = None,
    ) -> None:
        self.config = get_config()

        # Resource limits
        self.daily_query_budget = daily_query_budget or self.config.resources.daily_query_budget
        self.cpu_limit = cpu_limit or self.config.resources.cpu_limit
        self.memory_limit_mb = memory_limit_mb or self.config.resources.memory_limit_mb
        self.session_timeout_sec = session_timeout_sec or self.config.resources.session_timeout_sec

        # Usage tracking
        self.queries_today: int = 0
        self.queries_total: int = 0
        self.session_start_time: float | None = None
        self.current_memory_mb: float = 0.0

        # Daily tracking
        self._last_reset_date: date | None = None
        self._daily_history: list[dict[str, int]] = []

        # Emergency stop
        self._emergency_stop: bool = False

        logger.info("ResourceGovernor initialized")

    def check_budget(self) -> bool:
        """Check if query budget is available.

        Returns:
            True if budget available, False otherwise

        """
        self._check_daily_reset()

        if self.queries_today >= self.daily_query_budget:
            logger.warning(
                f"Daily query budget exceeded: {self.queries_today}/{self.daily_query_budget}",
            )
            return False

        return True

    def consume_query(self) -> None:
        """Consume a query from the budget."""
        if not self.check_budget():
            msg = "Query budget exceeded"
            raise ResourceExhaustedError(msg)

        self.queries_today += 1
        self.queries_total += 1

        logger.debug(
            f"Query consumed: {self.queries_today}/"
            f"{self.daily_query_budget} today, {self.queries_total} total",
        )

    def check_session_timeout(self) -> bool:
        """Check if current session has exceeded timeout.

        Returns:
            True if within timeout, False if exceeded

        """
        if self.session_start_time is None:
            return True

        elapsed = time.time() - self.session_start_time

        if elapsed > self.session_timeout_sec:
            logger.warning(
                f"Session timeout exceeded: {elapsed:.2f}s > {self.session_timeout_sec}s",
            )
            return False

        return True

    def start_session(self) -> None:
        """Start a new session."""
        self.session_start_time = time.time()
        self.current_memory_mb = 0.0
        logger.debug("Session started")

    def end_session(self) -> None:
        """End the current session."""
        if self.session_start_time is None:
            return

        elapsed = time.time() - self.session_start_time
        self.session_start_time = None

        logger.debug(f"Session ended after {elapsed:.2f}s")

    def check_cpu_limit(self, current_usage: float) -> bool:
        """Check if CPU usage is within limits.

        Args:
            current_usage: Current CPU usage as ratio (0.0 to 1.0)

        Returns:
            True if within limit, False otherwise

        """
        if current_usage <= self.cpu_limit:
            return True

        logger.warning(f"CPU limit exceeded: {current_usage:.2f} > {self.cpu_limit:.2f}")
        return False

    def check_memory_limit(self, current_usage_mb: float) -> bool:
        """Check if memory usage is within limits.

        Args:
            current_usage_mb: Current memory usage in MB

        Returns:
            True if within limit, False otherwise

        """
        if current_usage_mb <= self.memory_limit_mb:
            self.current_memory_mb = current_usage_mb
            return True

        logger.warning(
            f"Memory limit exceeded: {current_usage_mb:.2f}MB > {self.memory_limit_mb}MB",
        )
        return False

    def trigger_emergency_stop(self) -> None:
        """Trigger emergency stop."""
        if not self.config.enable_emergency_stop:
            logger.warning("Emergency stop disabled by configuration")
            return

        self._emergency_stop = True
        logger.critical("EMERGENCY STOP TRIGGERED")

        # End current session
        self.end_session()

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop has been triggered."""
        return self._emergency_stop

    def get_usage_stats(self) -> dict[str, Any]:
        """Get current resource usage statistics."""
        return {
            "daily_budget": self.daily_query_budget,
            "queries_today": self.queries_today,
            "queries_total": self.queries_total,
            "queries_remaining": max(0, self.daily_query_budget - self.queries_today),
            "session_elapsed": (
                time.time() - self.session_start_time if self.session_start_time else 0.0
            ),
            "current_memory_mb": self.current_memory_mb,
            "emergency_stopped": self._emergency_stop,
            "last_reset_date": (
                self._last_reset_date.isoformat() if self._last_reset_date else None
            ),
        }

    def get_daily_history(self, days: int = 7) -> list[dict[str, int]]:
        """Get daily usage history.

        Args:
            days: Number of days of history to return

        Returns:
            List of daily usage records

        """
        return self._daily_history[-days:]

    def reset_daily(self) -> None:
        """Reset daily counters (called at start of new day)."""
        self._check_daily_reset()
        self.queries_today = 0
        logger.info("Daily counters reset")

    def reset_all(self) -> None:
        """Reset all resource tracking."""
        self.queries_today = 0
        self.queries_total = 0
        self.session_start_time = None
        self.current_memory_mb = 0.0
        self._emergency_stop = False
        self._daily_history.clear()

        logger.info("All resource tracking reset")

    def _check_daily_reset(self) -> None:
        """Check and perform daily reset if needed."""
        today = date.today()

        if self._last_reset_date != today:
            # New day, reset daily counters
            self.queries_today = 0
            self._last_reset_date = today

            # Record yesterday's usage
            if self._last_reset_date:
                yesterday = self._last_reset_date
                self._daily_history.append(
                    {
                        "date": yesterday.isoformat(),
                        "queries": self.queries_today,
                        "total_queries": self.queries_total,
                    },
                )

            logger.debug(f"Daily reset performed: {today.isoformat()}")


class ResourceExhaustedError(Exception):
    """Raised when resource budget is exceeded."""
