"""
Event Bus for Inter-Agent Communication
Provides pub/sub messaging, event streaming, and distributed coordination
"""

import asyncio
import contextlib
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable

from .enhanced_models import EventType, SystemEvent

logger = logging.getLogger(__name__)


class EventHandler:
    """Wrapper for event handler callbacks."""

    def __init__(
        self,
        callback: Callable,
        event_types: list[EventType] | None = None,
        filter_func: Callable[[SystemEvent], bool] | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.callback = callback
        self.event_types = event_types or []
        self.filter_func = filter_func
        self.is_async = asyncio.iscoroutinefunction(callback)

    async def handle(self, event: SystemEvent) -> bool:
        """Handle an event if it matches the filter criteria."""
        # Check event type filter
        if self.event_types and event.type not in self.event_types:
            return False

        # Check custom filter
        if self.filter_func and not self.filter_func(event):
            return False

        try:
            if self.is_async:
                await self.callback(event)
            else:
                self.callback(event)
            return True
        except Exception as e:
            logger.error(f"Event handler error: {e}")
            return False


class EventBus(ABC):
    """Abstract base class for event bus implementations."""

    @abstractmethod
    async def publish(self, event: SystemEvent) -> bool:
        """Publish an event to the bus."""
        pass

    @abstractmethod
    async def subscribe(self, handler: EventHandler, channels: list[str] | None = None) -> str:
        """Subscribe to events. Returns subscription ID."""
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        pass

    @abstractmethod
    async def start(self):
        """Start the event bus."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the event bus."""
        pass


class InMemoryEventBus(EventBus):
    """In-memory event bus for single-process deployments."""

    def __init__(self, max_history: int = 1000):
        self._handlers: dict[str, EventHandler] = {}
        self._channel_handlers: dict[str, set[str]] = defaultdict(set)
        self._event_history: list[SystemEvent] = []
        self._max_history = max_history
        self._running = False
        self._lock = asyncio.Lock()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: asyncio.Task | None = None

    async def start(self):
        """Start the event processor."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("In-memory event bus started")

    async def stop(self):
        """Stop the event processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processor_task
        logger.info("In-memory event bus stopped")

    async def publish(self, event: SystemEvent) -> bool:
        """Publish an event to all subscribers."""
        if not self._running:
            logger.warning("Event bus not running, cannot publish")
            return False

        await self._event_queue.put(event)
        return True

    async def subscribe(self, handler: EventHandler, channels: list[str] | None = None) -> str:
        """Subscribe to events on specific channels."""
        async with self._lock:
            self._handlers[handler.id] = handler

            # Subscribe to channels
            target_channels = channels or ["default"]
            for channel in target_channels:
                self._channel_handlers[channel].add(handler.id)

            logger.debug(f"Handler {handler.id} subscribed to channels: {target_channels}")
            return handler.id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe a handler."""
        async with self._lock:
            if subscription_id not in self._handlers:
                return False

            del self._handlers[subscription_id]

            # Remove from all channels
            for channel_handlers in self._channel_handlers.values():
                channel_handlers.discard(subscription_id)

            return True

    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Store in history
                async with self._lock:
                    self._event_history.append(event)
                    if len(self._event_history) > self._max_history:
                        self._event_history = self._event_history[-self._max_history :]

                # Dispatch to handlers
                await self._dispatch_event(event)

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    async def _dispatch_event(self, event: SystemEvent):
        """Dispatch an event to all matching handlers."""
        # Get handlers for the event's channel (use job_id or "default")
        channel = event.job_id or "default"
        handler_ids = self._channel_handlers.get(channel, set()) | self._channel_handlers.get(
            "default", set()
        )

        tasks = []
        for handler_id in handler_ids:
            handler = self._handlers.get(handler_id)
            if handler:
                tasks.append(handler.handle(event))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_event_history(
        self, limit: int = 100, event_type: EventType | None = None, job_id: str | None = None
    ) -> list[SystemEvent]:
        """Get event history with optional filtering."""
        events = self._event_history.copy()

        if event_type:
            events = [e for e in events if e.type == event_type]

        if job_id:
            events = [e for e in events if e.job_id == job_id]

        return events[-limit:]


class RedisEventBus(EventBus):
    """Redis-backed event bus for distributed deployments."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", channel_prefix: str = "chimera:events:"
    ):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self._redis = None
        self._pubsub = None
        self._handlers: dict[str, EventHandler] = {}
        self._subscriptions: dict[str, list[str]] = {}  # handler_id -> channels
        self._running = False
        self._listener_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the Redis event bus."""
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._pubsub = self._redis.pubsub()
            self._running = True
            self._listener_task = asyncio.create_task(self._listen_for_events())
            logger.info(f"Redis event bus connected to {self.redis_url}")
        except ImportError:
            logger.error("redis-py not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    async def stop(self):
        """Stop the Redis event bus."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("Redis event bus stopped")

    async def publish(self, event: SystemEvent) -> bool:
        """Publish an event to Redis."""
        if not self._redis:
            return False

        try:
            channel = f"{self.channel_prefix}{event.job_id or 'default'}"
            await self._redis.publish(channel, event.to_json())

            # Also store in a list for history
            await self._redis.lpush(f"{self.channel_prefix}history", event.to_json())
            await self._redis.ltrim(f"{self.channel_prefix}history", 0, 999)

            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False

    async def subscribe(self, handler: EventHandler, channels: list[str] | None = None) -> str:
        """Subscribe to Redis channels."""
        async with self._lock:
            self._handlers[handler.id] = handler

            target_channels = channels or ["default"]
            redis_channels = [f"{self.channel_prefix}{c}" for c in target_channels]

            self._subscriptions[handler.id] = redis_channels

            if self._pubsub:
                await self._pubsub.subscribe(*redis_channels)

            logger.debug(f"Handler {handler.id} subscribed to Redis channels: {redis_channels}")
            return handler.id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Redis channels."""
        async with self._lock:
            if subscription_id not in self._handlers:
                return False

            channels = self._subscriptions.get(subscription_id, [])
            if channels and self._pubsub:
                await self._pubsub.unsubscribe(*channels)

            del self._handlers[subscription_id]
            del self._subscriptions[subscription_id]

            return True

    async def _listen_for_events(self):
        """Listen for events from Redis."""
        while self._running:
            try:
                if not self._pubsub:
                    await asyncio.sleep(1)
                    continue

                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )

                if message and message["type"] == "message":
                    event = SystemEvent.from_json(message["data"])
                    await self._dispatch_event(event, message["channel"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis listener error: {e}")
                await asyncio.sleep(1)

    async def _dispatch_event(self, event: SystemEvent, channel: str):
        """Dispatch an event to matching handlers."""
        for handler_id, handler in self._handlers.items():
            subscribed_channels = self._subscriptions.get(handler_id, [])
            if (
                channel in subscribed_channels
                or f"{self.channel_prefix}default" in subscribed_channels
            ):
                await handler.handle(event)


class HybridEventBus(EventBus):
    """Hybrid event bus that uses Redis when available, falls back to in-memory."""

    def __init__(self, redis_url: str | None = None, channel_prefix: str = "chimera:events:"):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self._primary_bus: EventBus | None = None
        self._fallback_bus = InMemoryEventBus()
        self._using_redis = False

    async def start(self):
        """Start the hybrid event bus."""
        if self.redis_url:
            try:
                self._primary_bus = RedisEventBus(
                    redis_url=self.redis_url, channel_prefix=self.channel_prefix
                )
                await self._primary_bus.start()
                self._using_redis = True
                logger.info("Hybrid event bus using Redis")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory: {e}")
                self._primary_bus = None

        if not self._using_redis:
            await self._fallback_bus.start()
            logger.info("Hybrid event bus using in-memory")

    async def stop(self):
        """Stop the hybrid event bus."""
        if self._primary_bus:
            await self._primary_bus.stop()
        await self._fallback_bus.stop()

    @property
    def _active_bus(self) -> EventBus:
        """Get the currently active bus."""
        return self._primary_bus if self._using_redis else self._fallback_bus

    async def publish(self, event: SystemEvent) -> bool:
        """Publish to the active bus."""
        return await self._active_bus.publish(event)

    async def subscribe(self, handler: EventHandler, channels: list[str] | None = None) -> str:
        """Subscribe on the active bus."""
        return await self._active_bus.subscribe(handler, channels)

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from the active bus."""
        return await self._active_bus.unsubscribe(subscription_id)


def create_event_bus(bus_type: str = "memory", redis_url: str | None = None, **kwargs) -> EventBus:
    """
    Factory function to create an event bus.

    Args:
        bus_type: "memory", "redis", or "hybrid"
        redis_url: Redis connection URL
        **kwargs: Additional arguments

    Returns:
        EventBus instance
    """
    if bus_type == "redis":
        return RedisEventBus(redis_url=redis_url or "redis://localhost:6379", **kwargs)
    elif bus_type == "hybrid":
        return HybridEventBus(redis_url=redis_url, **kwargs)
    else:
        return InMemoryEventBus(**kwargs)
