"""Service Event Bus - Async Pub/Sub pattern for service communication.

Provides:
- Event publishing with type safety
- Async event handlers with error isolation
- Event persistence for reliability
- Correlation ID tracking for distributed tracing
- Replay capability for recovery scenarios

Usage:
    from app.core.event_bus import event_bus, Event, EventType

    # Subscribe to events
    @event_bus.on(EventType.TRANSFORMATION_COMPLETED)
    async def handle_transformation(event: Event):
        print(f"Transformation complete: {event.payload}")

    # Publish events
    await event_bus.publish(Event(
        event_type=EventType.TRANSFORMATION_COMPLETED,
        payload={"prompt_id": "abc123", "technique": "quantum"}
    ))
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EventType(str, Enum):
    """System event types for service communication."""

    # Transformation events
    TRANSFORMATION_STARTED = "transformation.started"
    TRANSFORMATION_COMPLETED = "transformation.completed"
    TRANSFORMATION_FAILED = "transformation.failed"

    # Generation events
    GENERATION_STARTED = "generation.started"
    GENERATION_COMPLETED = "generation.completed"
    GENERATION_FAILED = "generation.failed"

    # Provider events
    PROVIDER_REGISTERED = "provider.registered"
    PROVIDER_STATUS_CHANGED = "provider.status_changed"
    PROVIDER_CIRCUIT_OPENED = "provider.circuit_opened"
    PROVIDER_CIRCUIT_CLOSED = "provider.circuit_closed"

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_EXPIRED = "session.expired"
    SESSION_TERMINATED = "session.terminated"

    # Cache events
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    CACHE_INVALIDATED = "cache.invalidated"

    # Health events
    HEALTH_STATUS_CHANGED = "health.status_changed"
    HEALTH_CHECK_FAILED = "health.check_failed"

    # Jailbreak events
    JAILBREAK_STARTED = "jailbreak.started"
    JAILBREAK_COMPLETED = "jailbreak.completed"
    JAILBREAK_FAILED = "jailbreak.failed"


@dataclass
class Event:
    """Base event structure for the event bus.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of event from EventType enum
        payload: Event data/payload
        timestamp: ISO format timestamp when event was created
        correlation_id: ID for tracing related events across services
        metadata: Additional metadata (source, version, etc.)

    """

    event_type: EventType
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EventHandlerStats:
    """Statistics for an event handler."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time_ms: float = 0.0
    last_call: str | None = None

    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.calls if self.calls > 0 else 0.0


class EventBus:
    """In-process event bus with async handlers and persistence.

    Features:
    - Async handler execution with error isolation
    - Event store for replay and auditing
    - Handler statistics for monitoring
    - Correlation ID propagation
    """

    def __init__(self, max_store_size: int = 1000) -> None:
        self._handlers: dict[EventType, list[Callable]] = {}
        self._handler_names: dict[EventType, list[str]] = {}
        self._handler_stats: dict[str, EventHandlerStats] = {}
        self._event_store: list[Event] = []
        self._max_store_size = max_store_size
        self._running = True

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Awaitable[None] | None],
        handler_name: str | None = None,
    ) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: The type of event to subscribe to
            handler: Async or sync callable that receives Event
            handler_name: Optional name for handler (for stats/logging)

        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
            self._handler_names[event_type] = []

        name = handler_name or handler.__name__
        self._handlers[event_type].append(handler)
        self._handler_names[event_type].append(name)
        self._handler_stats[name] = EventHandlerStats()

        logger.info(f"Subscribed handler '{name}' to {event_type.value}")

    def on(self, event_type: EventType, handler_name: str | None = None):
        """Decorator for subscribing handlers.

        Usage:
            @event_bus.on(EventType.TRANSFORMATION_COMPLETED)
            async def handle_transformation(event: Event):
                ...
        """

        def decorator(func: Callable[[Event], Awaitable[None] | None]):
            self.subscribe(event_type, func, handler_name or func.__name__)
            return func

        return decorator

    def unsubscribe(self, event_type: EventType, handler: Callable) -> bool:
        """Unsubscribe a handler from an event type.

        Returns True if handler was found and removed.
        """
        if event_type in self._handlers:
            try:
                idx = self._handlers[event_type].index(handler)
                self._handlers[event_type].pop(idx)
                self._handler_names[event_type].pop(idx)
                return True
            except ValueError:
                pass
        return False

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Events are stored for replay and dispatched to handlers async.
        Errors in handlers are isolated and logged.
        """
        if not self._running:
            logger.warning("Event bus not running, event dropped")
            return

        # Store event for replay/auditing
        self._store_event(event)

        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])
        handler_names = self._handler_names.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No handlers for event type: {event.event_type.value}")
            return

        logger.debug(f"Publishing {event.event_type.value} to {len(handlers)} handlers")

        # Dispatch to all handlers
        for handler, name in zip(handlers, handler_names, strict=False):
            await self._execute_handler(handler, name, event)

    async def publish_async(self, event: Event) -> asyncio.Task:
        """Publish an event asynchronously (fire-and-forget).

        Returns the task for optional awaiting.
        """
        return asyncio.create_task(self.publish(event))

    async def _execute_handler(self, handler: Callable, name: str, event: Event) -> None:
        """Execute a handler with error isolation and timing."""
        stats = self._handler_stats.get(name)
        if stats:
            stats.calls += 1
            stats.last_call = datetime.utcnow().isoformat() + "Z"

        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

            if stats:
                stats.successes += 1

        except Exception as e:
            logger.error(
                f"Handler '{name}' failed for {event.event_type.value}: {e}",
                exc_info=True,
            )
            if stats:
                stats.failures += 1

        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            if stats:
                stats.total_time_ms += elapsed_ms

    def _store_event(self, event: Event) -> None:
        """Store event with size limit enforcement."""
        self._event_store.append(event)

        # Enforce max size with FIFO eviction
        if len(self._event_store) > self._max_store_size:
            excess = len(self._event_store) - self._max_store_size
            self._event_store = self._event_store[excess:]

    def get_events(
        self,
        event_type: EventType | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get stored events with optional filtering.

        Args:
            event_type: Filter by event type
            since: Get events after this ISO timestamp
            limit: Maximum number of events to return

        """
        events = self._event_store

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    async def replay_events(
        self,
        event_type: EventType | None = None,
        since: str | None = None,
    ) -> int:
        """Replay stored events to handlers.

        Returns the number of events replayed.
        """
        events = self.get_events(event_type, since, limit=self._max_store_size)

        for event in events:
            await self.publish(event)

        return len(events)

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""
        return {
            "running": self._running,
            "event_types_subscribed": len(self._handlers),
            "total_handlers": sum(len(h) for h in self._handlers.values()),
            "events_stored": len(self._event_store),
            "max_store_size": self._max_store_size,
            "handlers": {
                name: {
                    "calls": s.calls,
                    "successes": s.successes,
                    "failures": s.failures,
                    "avg_time_ms": round(s.avg_time_ms, 2),
                    "last_call": s.last_call,
                }
                for name, s in self._handler_stats.items()
            },
        }

    def shutdown(self) -> None:
        """Shutdown the event bus."""
        self._running = False
        logger.info("Event bus shutdown")

    def start(self) -> None:
        """Start/restart the event bus."""
        self._running = True
        logger.info("Event bus started")

    def clear_events(self) -> None:
        """Clear the event store."""
        self._event_store.clear()
        logger.info("Event store cleared")


# Global event bus instance
event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return event_bus
