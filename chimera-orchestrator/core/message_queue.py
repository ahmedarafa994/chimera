"""
Asynchronous Message Queue for Inter-Agent Communication
"""

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .models import AgentType, Message, MessageType

logger = logging.getLogger(__name__)


class MessageQueue:
    """
    In-memory async message queue for agent communication.
    Can be extended to use Redis, RabbitMQ, or other backends.
    """

    def __init__(self):
        self._queues: dict[AgentType, asyncio.Queue] = {}
        self._subscribers: dict[MessageType, list[Callable]] = defaultdict(list)
        self._message_history: list[Message] = []
        self._max_history = 1000
        self._running = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize queues for all agent types."""
        for agent_type in AgentType:
            self._queues[agent_type] = asyncio.Queue()
        self._running = True
        logger.info("Message queue initialized")

    async def shutdown(self):
        """Shutdown the message queue."""
        self._running = False
        # Clear all queues
        for queue in self._queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        logger.info("Message queue shutdown")

    async def publish(self, message: Message) -> bool:
        """
        Publish a message to the queue.

        Args:
            message: The message to publish

        Returns:
            True if published successfully
        """
        if not self._running:
            logger.warning("Cannot publish - queue not running")
            return False

        async with self._lock:
            # Add to history
            self._message_history.append(message)
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history :]

        # Route to target queue if specified
        if message.target:
            if message.target in self._queues:
                await self._queues[message.target].put(message)
                logger.debug(f"Message {message.id} routed to {message.target.value}")
            else:
                logger.warning(f"Unknown target agent: {message.target}")
                return False
        else:
            # Broadcast to all queues except source
            for agent_type, queue in self._queues.items():
                if agent_type != message.source:
                    await queue.put(message)
            logger.debug(f"Message {message.id} broadcast to all agents")

        # Notify subscribers
        await self._notify_subscribers(message)

        return True

    async def subscribe(self, message_type: MessageType, callback: Callable[[Message], Any]):
        """
        Subscribe to a specific message type.

        Args:
            message_type: Type of messages to subscribe to
            callback: Async callback function to handle messages
        """
        self._subscribers[message_type].append(callback)
        logger.debug(f"Subscribed to {message_type.value}")

    async def unsubscribe(self, message_type: MessageType, callback: Callable[[Message], Any]):
        """Unsubscribe from a message type."""
        if callback in self._subscribers[message_type]:
            self._subscribers[message_type].remove(callback)

    async def _notify_subscribers(self, message: Message):
        """Notify all subscribers of a message."""
        callbacks = self._subscribers.get(message.type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    async def consume(self, agent_type: AgentType, timeout: float | None = None) -> Message | None:
        """
        Consume a message from an agent's queue.

        Args:
            agent_type: The agent type to consume from
            timeout: Optional timeout in seconds

        Returns:
            The next message or None if timeout
        """
        if agent_type not in self._queues:
            logger.warning(f"No queue for agent type: {agent_type}")
            return None

        queue = self._queues[agent_type]

        try:
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()
            return message
        except TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error consuming message: {e}")
            return None

    async def consume_batch(
        self, agent_type: AgentType, max_messages: int = 10, timeout: float = 1.0
    ) -> list[Message]:
        """
        Consume multiple messages from an agent's queue.

        Args:
            agent_type: The agent type to consume from
            max_messages: Maximum number of messages to consume
            timeout: Timeout for the entire batch operation

        Returns:
            List of messages
        """
        messages = []
        deadline = asyncio.get_event_loop().time() + timeout

        while len(messages) < max_messages:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            message = await self.consume(agent_type, timeout=remaining)
            if message:
                messages.append(message)
            else:
                break

        return messages

    def get_queue_size(self, agent_type: AgentType) -> int:
        """Get the current size of an agent's queue."""
        if agent_type in self._queues:
            return self._queues[agent_type].qsize()
        return 0

    def get_all_queue_sizes(self) -> dict[str, int]:
        """Get sizes of all queues."""
        return {agent_type.value: queue.qsize() for agent_type, queue in self._queues.items()}

    def get_message_history(
        self,
        limit: int = 100,
        message_type: MessageType | None = None,
        job_id: str | None = None,
    ) -> list[Message]:
        """
        Get message history with optional filtering.

        Args:
            limit: Maximum number of messages to return
            message_type: Filter by message type
            job_id: Filter by job ID

        Returns:
            List of messages
        """
        messages = self._message_history.copy()

        if message_type:
            messages = [m for m in messages if m.type == message_type]

        if job_id:
            messages = [m for m in messages if m.job_id == job_id]

        return messages[-limit:]


class PriorityMessageQueue(MessageQueue):
    """
    Priority-based message queue that processes high-priority messages first.
    """

    def __init__(self):
        super().__init__()
        self._priority_queues: dict[AgentType, asyncio.PriorityQueue] = {}

    async def initialize(self):
        """Initialize priority queues for all agent types."""
        for agent_type in AgentType:
            self._priority_queues[agent_type] = asyncio.PriorityQueue()
        self._running = True
        logger.info("Priority message queue initialized")

    async def publish(self, message: Message) -> bool:
        """Publish a message with priority."""
        if not self._running:
            return False

        async with self._lock:
            self._message_history.append(message)
            if len(self._message_history) > self._max_history:
                self._message_history = self._message_history[-self._max_history :]

        # Priority is inverted (lower number = higher priority)
        priority = 10 - message.priority

        if message.target:
            if message.target in self._priority_queues:
                await self._priority_queues[message.target].put(
                    (priority, message.timestamp.timestamp(), message)
                )
        else:
            for agent_type, queue in self._priority_queues.items():
                if agent_type != message.source:
                    await queue.put((priority, message.timestamp.timestamp(), message))

        await self._notify_subscribers(message)
        return True

    async def consume(self, agent_type: AgentType, timeout: float | None = None) -> Message | None:
        """Consume the highest priority message."""
        if agent_type not in self._priority_queues:
            return None

        queue = self._priority_queues[agent_type]

        try:
            if timeout:
                _, _, message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                _, _, message = await queue.get()
            return message
        except TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error consuming priority message: {e}")
            return None

    def get_queue_size(self, agent_type: AgentType) -> int:
        """Get the current size of an agent's priority queue."""
        if agent_type in self._priority_queues:
            return self._priority_queues[agent_type].qsize()
        return 0


class RedisMessageQueue(MessageQueue):
    """
    Redis-backed message queue for distributed deployments.
    Requires redis-py async support.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__()
        self.redis_url = redis_url
        self._redis = None

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._running = True
            logger.info(f"Redis message queue connected to {self.redis_url}")
        except ImportError:
            logger.warning("redis-py not installed, falling back to in-memory queue")
            await super().initialize()
        except Exception as e:
            logger.error(f"Redis connection failed: {e}, falling back to in-memory")
            await super().initialize()

    async def shutdown(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        self._running = False

    async def publish(self, message: Message) -> bool:
        """Publish message to Redis."""
        if not self._redis:
            return await super().publish(message)

        try:
            channel = f"chimera:{message.target.value if message.target else 'broadcast'}"
            await self._redis.publish(channel, json.dumps(message.to_dict()))

            # Also store in list for history
            await self._redis.lpush("chimera:history", json.dumps(message.to_dict()))
            await self._redis.ltrim("chimera:history", 0, self._max_history - 1)

            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False

    async def consume(self, agent_type: AgentType, timeout: float | None = None) -> Message | None:
        """Consume message from Redis."""
        if not self._redis:
            return await super().consume(agent_type, timeout)

        try:
            channel = f"chimera:{agent_type.value}"
            result = await self._redis.blpop(channel, timeout=timeout or 0)
            if result:
                _, data = result
                return Message.from_dict(json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Redis consume error: {e}")
            return None


# Factory function to create appropriate queue
def create_message_queue(queue_type: str = "memory", **kwargs) -> MessageQueue:
    """
    Create a message queue of the specified type.

    Args:
        queue_type: "memory", "priority", or "redis"
        **kwargs: Additional arguments for the queue

    Returns:
        MessageQueue instance
    """
    if queue_type == "priority":
        return PriorityMessageQueue()
    elif queue_type == "redis":
        return RedisMessageQueue(kwargs.get("redis_url", "redis://localhost:6379"))
    else:
        return MessageQueue()
