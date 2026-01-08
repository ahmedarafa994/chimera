"""
DeepTeam Streaming Callbacks

Provides streaming callbacks for real-time attack results and vulnerability discoveries.
Supports WebSocket, SSE, and callback-based streaming.
"""

import contextlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_ERROR = "session_error"

    # Attack events
    ATTACK_START = "attack_start"
    ATTACK_PROGRESS = "attack_progress"
    ATTACK_COMPLETE = "attack_complete"
    ATTACK_FAILED = "attack_failed"

    # Vulnerability events
    VULNERABILITY_DISCOVERED = "vulnerability_discovered"
    VULNERABILITY_CONFIRMED = "vulnerability_confirmed"
    VULNERABILITY_FALSE_POSITIVE = "vulnerability_false_positive"

    # Prompt events
    PROMPT_GENERATED = "prompt_generated"
    PROMPT_TESTED = "prompt_tested"
    PROMPT_SUCCESS = "prompt_success"
    PROMPT_BLOCKED = "prompt_blocked"

    # Progress events
    PROGRESS_UPDATE = "progress_update"
    METRICS_UPDATE = "metrics_update"

    # Log events
    LOG_INFO = "log_info"
    LOG_WARNING = "log_warning"
    LOG_ERROR = "log_error"


class StreamEvent(BaseModel):
    """A streaming event with metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: StreamEventType
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: str | None = None

    # Event data
    data: dict[str, Any] = Field(default_factory=dict)

    # Optional fields
    attack_id: str | None = None
    vulnerability_type: str | None = None
    prompt_id: str | None = None

    # Progress tracking
    progress: float | None = None  # 0.0 to 1.0
    total_steps: int | None = None
    current_step: int | None = None

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        data = self.model_dump_json()
        return f"event: {self.type.value}\ndata: {data}\n\n"

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()


class StreamCallback(ABC):
    """Abstract base class for streaming callbacks."""

    @abstractmethod
    async def on_event(self, event: StreamEvent) -> None:
        """Handle a streaming event."""
        pass

    async def on_session_start(self, session_id: str, config: dict) -> None:
        """Called when a session starts."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.SESSION_START,
                session_id=session_id,
                data={"config": config},
            )
        )

    async def on_session_end(
        self,
        session_id: str,
        results: dict,
        duration_ms: float,
    ) -> None:
        """Called when a session ends."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.SESSION_END,
                session_id=session_id,
                data={"results": results, "duration_ms": duration_ms},
            )
        )

    async def on_attack_start(
        self,
        session_id: str,
        attack_id: str,
        attack_type: str,
    ) -> None:
        """Called when an attack starts."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.ATTACK_START,
                session_id=session_id,
                attack_id=attack_id,
                data={"attack_type": attack_type},
            )
        )

    async def on_attack_complete(
        self,
        session_id: str,
        attack_id: str,
        success: bool,
        results: dict,
    ) -> None:
        """Called when an attack completes."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.ATTACK_COMPLETE,
                session_id=session_id,
                attack_id=attack_id,
                data={"success": success, "results": results},
            )
        )

    async def on_vulnerability_discovered(
        self,
        session_id: str,
        vulnerability_type: str,
        severity: str,
        details: dict,
    ) -> None:
        """Called when a vulnerability is discovered."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.VULNERABILITY_DISCOVERED,
                session_id=session_id,
                vulnerability_type=vulnerability_type,
                data={
                    "severity": severity,
                    "details": details,
                },
            )
        )

    async def on_prompt_generated(
        self,
        session_id: str,
        prompt_id: str,
        prompt: str,
        strategy: str,
    ) -> None:
        """Called when a prompt is generated."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.PROMPT_GENERATED,
                session_id=session_id,
                prompt_id=prompt_id,
                data={"prompt": prompt, "strategy": strategy},
            )
        )

    async def on_prompt_tested(
        self,
        session_id: str,
        prompt_id: str,
        success: bool,
        response: str | None = None,
    ) -> None:
        """Called when a prompt is tested."""
        event_type = StreamEventType.PROMPT_SUCCESS if success else StreamEventType.PROMPT_BLOCKED
        await self.on_event(
            StreamEvent(
                type=event_type,
                session_id=session_id,
                prompt_id=prompt_id,
                data={"success": success, "response": response},
            )
        )

    async def on_progress(
        self,
        session_id: str,
        progress: float,
        current_step: int,
        total_steps: int,
        message: str | None = None,
    ) -> None:
        """Called to report progress."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.PROGRESS_UPDATE,
                session_id=session_id,
                progress=progress,
                current_step=current_step,
                total_steps=total_steps,
                data={"message": message} if message else {},
            )
        )

    async def on_error(
        self,
        session_id: str,
        error: str,
        details: dict | None = None,
    ) -> None:
        """Called when an error occurs."""
        await self.on_event(
            StreamEvent(
                type=StreamEventType.SESSION_ERROR,
                session_id=session_id,
                data={"error": error, "details": details or {}},
            )
        )


class LoggingStreamCallback(StreamCallback):
    """Stream callback that logs all events."""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.events: list[StreamEvent] = []

    async def on_event(self, event: StreamEvent) -> None:
        """Log the event."""
        self.events.append(event)

        log_message = (
            f"[{event.type.value}] session={event.session_id} data={json.dumps(event.data)}"
        )

        if event.type in {
            StreamEventType.SESSION_ERROR,
            StreamEventType.ATTACK_FAILED,
            StreamEventType.LOG_ERROR,
        }:
            logger.error(log_message)
        elif event.type in {
            StreamEventType.LOG_WARNING,
            StreamEventType.VULNERABILITY_DISCOVERED,
        }:
            logger.warning(log_message)
        else:
            logger.log(self.log_level, log_message)


class CallbackStreamCallback(StreamCallback):
    """Stream callback that invokes a user-provided callback function."""

    def __init__(
        self,
        callback: Callable[[StreamEvent], None],
        async_callback: Callable[[StreamEvent], Any] | None = None,
    ):
        self.callback = callback
        self.async_callback = async_callback

    async def on_event(self, event: StreamEvent) -> None:
        """Invoke the callback with the event."""
        if self.async_callback:
            await self.async_callback(event)
        else:
            self.callback(event)


class QueueStreamCallback(StreamCallback):
    """Stream callback that puts events into an async queue."""

    def __init__(self):
        import asyncio

        self.queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._closed = False

    async def on_event(self, event: StreamEvent) -> None:
        """Put the event into the queue."""
        if not self._closed:
            await self.queue.put(event)

    async def get_event(self, timeout: float | None = None) -> StreamEvent:
        """Get the next event from the queue."""
        import asyncio

        if timeout:
            return await asyncio.wait_for(self.queue.get(), timeout)
        return await self.queue.get()

    async def close(self) -> None:
        """Close the queue."""
        self._closed = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamEvent:
        if self._closed and self.queue.empty():
            raise StopAsyncIteration
        return await self.queue.get()


class WebSocketStreamCallback(StreamCallback):
    """Stream callback that sends events over WebSocket."""

    def __init__(self, websocket: Any):
        """
        Initialize with a WebSocket connection.

        Args:
            websocket: A WebSocket connection (FastAPI WebSocket or similar)
        """
        self.websocket = websocket
        self._closed = False

    async def on_event(self, event: StreamEvent) -> None:
        """Send the event over WebSocket."""
        if self._closed:
            return

        try:
            await self.websocket.send_json(event.model_dump())
        except Exception as e:
            logger.error(f"Failed to send WebSocket event: {e}")
            self._closed = True

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._closed = True
        with contextlib.suppress(Exception):
            await self.websocket.close()


class SSEStreamCallback(StreamCallback):
    """Stream callback for Server-Sent Events."""

    def __init__(self):
        self.events: list[str] = []
        self._closed = False

    async def on_event(self, event: StreamEvent) -> None:
        """Format and store the event as SSE."""
        if not self._closed:
            self.events.append(event.to_sse())

    def get_events(self) -> list[str]:
        """Get all SSE-formatted events."""
        return self.events

    def clear(self) -> None:
        """Clear stored events."""
        self.events.clear()

    async def close(self) -> None:
        """Close the SSE stream."""
        self._closed = True


class MultiStreamCallback(StreamCallback):
    """Stream callback that forwards events to multiple callbacks."""

    def __init__(self, callbacks: list[StreamCallback] | None = None):
        self.callbacks: list[StreamCallback] = callbacks or []

    def add_callback(self, callback: StreamCallback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: StreamCallback) -> None:
        """Remove a callback from the list."""
        self.callbacks.remove(callback)

    async def on_event(self, event: StreamEvent) -> None:
        """Forward the event to all callbacks."""
        for callback in self.callbacks:
            try:
                await callback.on_event(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class VulnerabilityTracker:
    """Tracks discovered vulnerabilities during a session."""

    def __init__(self):
        self.vulnerabilities: list[dict] = []
        self.severity_counts: dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

    def add_vulnerability(
        self,
        vulnerability_type: str,
        severity: str,
        details: dict,
    ) -> None:
        """Add a discovered vulnerability."""
        self.vulnerabilities.append(
            {
                "type": vulnerability_type,
                "severity": severity,
                "details": details,
                "discovered_at": datetime.utcnow().isoformat(),
            }
        )

        severity_lower = severity.lower()
        if severity_lower in self.severity_counts:
            self.severity_counts[severity_lower] += 1

    def get_summary(self) -> dict:
        """Get a summary of discovered vulnerabilities."""
        return {
            "total": len(self.vulnerabilities),
            "by_severity": self.severity_counts,
            "vulnerabilities": self.vulnerabilities,
        }

    def has_critical(self) -> bool:
        """Check if any critical vulnerabilities were found."""
        return self.severity_counts["critical"] > 0

    def has_high_or_above(self) -> bool:
        """Check if any high or critical vulnerabilities were found."""
        return self.severity_counts["critical"] > 0 or self.severity_counts["high"] > 0


class AttackResultTracker:
    """Tracks attack results during a session."""

    def __init__(self):
        self.attacks: list[dict] = []
        self.successful_attacks: int = 0
        self.failed_attacks: int = 0
        self.blocked_attacks: int = 0

    def add_result(
        self,
        attack_id: str,
        attack_type: str,
        success: bool,
        blocked: bool = False,
        details: dict | None = None,
    ) -> None:
        """Add an attack result."""
        self.attacks.append(
            {
                "attack_id": attack_id,
                "attack_type": attack_type,
                "success": success,
                "blocked": blocked,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        if success:
            self.successful_attacks += 1
        elif blocked:
            self.blocked_attacks += 1
        else:
            self.failed_attacks += 1

    def get_summary(self) -> dict:
        """Get a summary of attack results."""
        total = len(self.attacks)
        return {
            "total": total,
            "successful": self.successful_attacks,
            "failed": self.failed_attacks,
            "blocked": self.blocked_attacks,
            "success_rate": (self.successful_attacks / total if total > 0 else 0.0),
            "attacks": self.attacks,
        }


class StreamingSession:
    """Manages a streaming session with callbacks and trackers."""

    def __init__(
        self,
        session_id: str | None = None,
        callback: StreamCallback | None = None,
    ):
        self.session_id = session_id or str(uuid4())
        self.callback = callback or LoggingStreamCallback()
        self.vulnerability_tracker = VulnerabilityTracker()
        self.attack_tracker = AttackResultTracker()
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self._config: dict = {}

    async def start(self, config: dict) -> None:
        """Start the session."""
        self._config = config
        self.start_time = datetime.utcnow()
        await self.callback.on_session_start(self.session_id, config)

    async def end(self) -> dict:
        """End the session and return results."""
        self.end_time = datetime.utcnow()
        duration_ms = (
            (self.end_time - self.start_time).total_seconds() * 1000 if self.start_time else 0.0
        )

        results = {
            "session_id": self.session_id,
            "duration_ms": duration_ms,
            "vulnerabilities": self.vulnerability_tracker.get_summary(),
            "attacks": self.attack_tracker.get_summary(),
        }

        await self.callback.on_session_end(
            self.session_id,
            results,
            duration_ms,
        )

        return results

    async def report_vulnerability(
        self,
        vulnerability_type: str,
        severity: str,
        details: dict,
    ) -> None:
        """Report a discovered vulnerability."""
        self.vulnerability_tracker.add_vulnerability(
            vulnerability_type,
            severity,
            details,
        )
        await self.callback.on_vulnerability_discovered(
            self.session_id,
            vulnerability_type,
            severity,
            details,
        )

    async def report_attack_result(
        self,
        attack_id: str,
        attack_type: str,
        success: bool,
        blocked: bool = False,
        details: dict | None = None,
    ) -> None:
        """Report an attack result."""
        self.attack_tracker.add_result(
            attack_id,
            attack_type,
            success,
            blocked,
            details,
        )
        await self.callback.on_attack_complete(
            self.session_id,
            attack_id,
            success,
            details or {},
        )

    async def report_progress(
        self,
        progress: float,
        current_step: int,
        total_steps: int,
        message: str | None = None,
    ) -> None:
        """Report progress."""
        await self.callback.on_progress(
            self.session_id,
            progress,
            current_step,
            total_steps,
            message,
        )

    async def report_error(
        self,
        error: str,
        details: dict | None = None,
    ) -> None:
        """Report an error."""
        await self.callback.on_error(self.session_id, error, details)
