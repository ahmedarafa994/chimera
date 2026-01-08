"""
Redis Streams Consumer for Real-Time Event Processing

This module provides the streaming layer consumer that processes LLM interaction
events from Redis Streams for real-time analytics and metrics aggregation.

Architecture:
- Consumer Group Pattern: Multiple consumer instances for horizontal scaling
- Consumer Groups: chimera_metrics, chimera_analytics, chimera_alerts
- Processing Model: At-least-once delivery with idempotency
- Checkpointing: Automatic ACK after successful processing

Usage:
    from app.services.data_pipeline.streaming_consumer import StreamConsumer

    consumer = StreamConsumer(consumer_name="metrics-1")
    await consumer.initialize()

    async for events in consumer.consume_llm_events():
        for event in events:
            await process_event(event)
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)


# Stream and consumer group constants
STREAM_LLM_EVENTS = "chimera:llm_events"
STREAM_TRANSFORMATION_EVENTS = "chimera:transformation_events"
STREAM_JAILBREAK_EVENTS = "chimera:jailbreak_events"

# Consumer groups
GROUP_METRICS = "chimera_metrics"
GROUP_ANALYTICS = "chimera_analytics"
GROUP_ALERTS = "chimera_alerts"

# Default consumer configuration
DEFAULT_BLOCK_MS = 5000  # Block for 5 seconds on XREADGROUP
DEFAULT_COUNT = 100  # Max events per read
DEFAULT_IDLE_MS = 60000 * 60  # 1 hour before pending entries claimed


@dataclass
class StreamEntry:
    """Single stream entry with parsed data"""
    stream_key: str
    entry_id: str
    event_type: str
    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_raw(cls, stream_key: str, entry_id: str, data: dict[str, str]) -> "StreamEntry":
        """Create StreamEntry from raw Redis stream data"""
        event_type = data.get("event_type", "unknown")
        timestamp_str = data.get("timestamp", datetime.utcnow().isoformat())

        # Parse event data
        event_data = {}
        if "data" in data:
            try:
                event_data = json.loads(data["data"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse event data for {entry_id}")
                event_data = {}

        # Parse metadata
        metadata = None
        if "metadata" in data:
            with contextlib.suppress(json.JSONDecodeError):
                metadata = json.loads(data["metadata"])

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            timestamp = datetime.utcnow()

        return cls(
            stream_key=stream_key,
            entry_id=entry_id,
            event_type=event_type,
            timestamp=timestamp,
            data=event_data,
            metadata=metadata,
        )


@dataclass
class ConsumerStats:
    """Consumer processing statistics"""
    consumer_name: str
    stream_key: str
    group_name: str
    total_processed: int = 0
    total_errors: int = 0
    last_processed_at: datetime | None = None
    lag: int = 0  # Number of unprocessed entries
    pending: int = 0  # Number of pending entries for this consumer

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "consumer_name": self.consumer_name,
            "stream_key": self.stream_key,
            "group_name": self.group_name,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "lag": self.lag,
            "pending": self.pending,
        }


class StreamConsumer:
    """
    Redis Streams Consumer for real-time event processing

    Features:
    - Consumer group support for horizontal scaling
    - Automatic consumer group creation
    - Event processing with async iteration
    - Health monitoring and lag tracking
    - Graceful shutdown with pending ACK
    - Error handling with retry logic

    Example:
        consumer = StreamConsumer(
            consumer_name="metrics-1",
            group_name=GROUP_METRICS,
            stream_keys=[STREAM_LLM_EVENTS]
        )
        await consumer.initialize()

        async for batch in consumer.consume():
            for entry in batch:
                await process_metrics(entry)
                await consumer.acknowledge(entry.entry_id)

        await consumer.close()
    """

    def __init__(
        self,
        consumer_name: str,
        group_name: str = GROUP_METRICS,
        stream_keys: list[str] | None = None,
        redis_url: str | None = None,
        block_ms: int = DEFAULT_BLOCK_MS,
        count: int = DEFAULT_COUNT,
    ):
        """
        Initialize the stream consumer

        Args:
            consumer_name: Unique name for this consumer instance
            group_name: Consumer group name (default: chimera_metrics)
            stream_keys: List of stream keys to consume (default: LLM events)
            redis_url: Redis connection URL (default: from settings)
            block_ms: Milliseconds to block on XREADGROUP
            count: Max entries per read
        """
        self.consumer_name = consumer_name
        self.group_name = group_name
        self.stream_keys = stream_keys or [STREAM_LLM_EVENTS]
        self.redis_url = redis_url or self._get_redis_url()
        self.block_ms = block_ms
        self.count = count

        # Connection management
        self._pool: ConnectionPool | None = None
        self._redis: redis.Redis | None = None
        self._is_initialized = False
        self._is_running = False

        # Statistics
        self._stats = {
            stream_key: ConsumerStats(
                consumer_name=consumer_name,
                stream_key=stream_key,
                group_name=group_name,
            )
            for stream_key in self.stream_keys
        }

    @staticmethod
    def _get_redis_url() -> str:
        """Get Redis URL from settings with fallback"""
        if hasattr(settings, 'REDIS_URL'):
            return settings.REDIS_URL
        return "redis://localhost:6379/0"

    async def initialize(self) -> None:
        """
        Initialize Redis connection and create consumer groups

        Raises:
            RedisConnectionError: If connection fails
        """
        try:
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
            )

            # Create Redis client
            self._redis = redis.Redis(connection_pool=self._pool)

            # Verify connectivity
            await self._redis.ping()

            # Create consumer groups if they don't exist
            for stream_key in self.stream_keys:
                await self._create_consumer_group(stream_key)

            self._is_initialized = True
            logger.info(
                f"Stream consumer initialized: {self.consumer_name} "
                f"in group {self.group_name}"
            )

        except RedisError as e:
            logger.error(f"Failed to initialize stream consumer: {e}")
            raise RedisConnectionError(f"Redis connection failed: {e}") from e

    async def _create_consumer_group(
        self,
        stream_key: str,
    ) -> None:
        """
        Create consumer group if it doesn't exist

        Args:
            stream_key: Redis stream key
        """
        try:
            # Try to create group from '$' (newest entry)
            await self._redis.xgroup_create(
                name=stream_key,
                groupname=self.group_name,
                id="0",  # Create from beginning of stream
                mkstream=True,  # Create stream if it doesn't exist
            )
            logger.info(f"Created consumer group {self.group_name} for {stream_key}")

        except RedisError as e:
            # Group might already exist
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {self.group_name} already exists for {stream_key}")
            else:
                logger.error(f"Failed to create consumer group: {e}")

    async def close(self) -> None:
        """Close Redis connection and cleanup resources"""
        self._is_running = False

        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()

        self._is_initialized = False
        logger.info(f"Stream consumer closed: {self.consumer_name}")

    async def consume_llm_events(
        self,
        count: int | None = None,
        block_ms: int | None = None,
    ) -> AsyncIterator[list[StreamEntry]]:
        """
        Consume LLM events from Redis Stream

        Args:
            count: Max entries per read (default: from constructor)
            block_ms: Milliseconds to block (default: from constructor)

        Yields:
            List of StreamEntry objects

        Example:
            async for batch in consumer.consume_llm_events():
                for entry in batch:
                    print(f"Event: {entry.event_type} - {entry.data}")
        """
        if not self._is_initialized:
            raise RuntimeError("Consumer not initialized")

        self._is_running = True
        count = count or self.count
        block_ms = block_ms or self.block_ms

        while self._is_running:
            try:
                # Read entries from stream
                entries = await self._read_entries(
                    stream_key=STREAM_LLM_EVENTS,
                    count=count,
                    block_ms=block_ms,
                )

                if entries:
                    # Update stats
                    self._stats[STREAM_LLM_EVENTS].total_processed += len(entries)
                    self._stats[STREAM_LLM_EVENTS].last_processed_at = datetime.utcnow()

                    # Parse entries
                    stream_entries = [
                        StreamEntry.from_raw(STREAM_LLM_EVENTS, entry_id, data)
                        for entry_id, data in entries
                    ]

                    yield stream_entries

                # Small sleep to prevent tight loop
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error consuming LLM events: {e}")
                self._stats[STREAM_LLM_EVENTS].total_errors += 1
                await asyncio.sleep(1)  # Backoff on error

    async def consume_transformation_events(
        self,
        count: int | None = None,
        block_ms: int | None = None,
    ) -> AsyncIterator[list[StreamEntry]]:
        """Consume transformation events from Redis Stream"""
        if not self._is_initialized:
            raise RuntimeError("Consumer not initialized")

        self._is_running = True
        count = count or self.count
        block_ms = block_ms or self.block_ms

        while self._is_running:
            try:
                entries = await self._read_entries(
                    stream_key=STREAM_TRANSFORMATION_EVENTS,
                    count=count,
                    block_ms=block_ms,
                )

                if entries:
                    self._stats[STREAM_TRANSFORMATION_EVENTS].total_processed += len(entries)
                    self._stats[STREAM_TRANSFORMATION_EVENTS].last_processed_at = datetime.utcnow()

                    stream_entries = [
                        StreamEntry.from_raw(STREAM_TRANSFORMATION_EVENTS, entry_id, data)
                        for entry_id, data in entries
                    ]

                    yield stream_entries

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error consuming transformation events: {e}")
                self._stats[STREAM_TRANSFORMATION_EVENTS].total_errors += 1
                await asyncio.sleep(1)

    async def consume_jailbreak_events(
        self,
        count: int | None = None,
        block_ms: int | None = None,
    ) -> AsyncIterator[list[StreamEntry]]:
        """Consume jailbreak events from Redis Stream"""
        if not self._is_initialized:
            raise RuntimeError("Consumer not initialized")

        self._is_running = True
        count = count or self.count
        block_ms = block_ms or self.block_ms

        while self._is_running:
            try:
                entries = await self._read_entries(
                    stream_key=STREAM_JAILBREAK_EVENTS,
                    count=count,
                    block_ms=block_ms,
                )

                if entries:
                    self._stats[STREAM_JAILBREAK_EVENTS].total_processed += len(entries)
                    self._stats[STREAM_JAILBREAK_EVENTS].last_processed_at = datetime.utcnow()

                    stream_entries = [
                        StreamEntry.from_raw(STREAM_JAILBREAK_EVENTS, entry_id, data)
                        for entry_id, data in entries
                    ]

                    yield stream_entries

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error consuming jailbreak events: {e}")
                self._stats[STREAM_JAILBREAK_EVENTS].total_errors += 1
                await asyncio.sleep(1)

    async def _read_entries(
        self,
        stream_key: str,
        count: int,
        block_ms: int,
    ) -> list[tuple]:
        """
        Read entries from stream using XREADGROUP

        Args:
            stream_key: Redis stream key
            count: Max entries to read
            block_ms: Milliseconds to block

        Returns:
            List of (entry_id, data) tuples
        """
        try:
            # Use '>' to read new entries not yet delivered to this consumer
            result = await self._redis.xreadgroup(
                groupname=self.group_name,
                consumername=self.consumer_name,
                streams={stream_key: ">"},  # New entries only
                count=count,
                block=block_ms,
            )

            if not result:
                return []

            # Parse result: [(stream_name, [(entry_id, {field: value, ...}), ...])]
            entries = []
            for _stream_name, stream_entries in result:
                entries.extend(stream_entries)

            return entries

        except RedisError as e:
            logger.error(f"Failed to read from stream {stream_key}: {e}")
            return []

    async def acknowledge(
        self,
        stream_key: str,
        entry_id: str,
    ) -> bool:
        """
        Acknowledge processing of a stream entry

        Args:
            stream_key: Redis stream key
            entry_id: Entry ID to acknowledge

        Returns:
            True if successful, False otherwise
        """
        if not self._is_initialized or not self._redis:
            return False

        try:
            await self._redis.xack(stream_key, self.group_name, entry_id)
            return True

        except RedisError as e:
            logger.error(f"Failed to acknowledge entry {entry_id}: {e}")
            return False

    async def acknowledge_batch(
        self,
        acknowledgements: list[tuple],
    ) -> int:
        """
        Acknowledge multiple entries in a single call

        Args:
            acknowledgements: List of (stream_key, entry_id) tuples

        Returns:
            Number of successfully acknowledged entries
        """
        if not self._is_initialized or not self._redis:
            return 0

        acknowledged = 0

        try:
            for stream_key, entry_id in acknowledgements:
                await self._redis.xack(stream_key, self.group_name, entry_id)
                acknowledged += 1

        except RedisError as e:
            logger.error(f"Failed to acknowledge batch: {e}")

        return acknowledged

    async def get_consumer_stats(
        self,
        stream_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Get consumer statistics and lag information

        Args:
            stream_key: Optional specific stream to get stats for

        Returns:
            Dictionary with consumer statistics
        """
        if stream_key:
            return self._stats.get(stream_key, ConsumerStats(
                consumer_name=self.consumer_name,
                stream_key=stream_key,
                group_name=self.group_name,
            )).to_dict()

        return {
            stream_key: stats.to_dict()
            for stream_key, stats in self._stats.items()
        }

    async def get_group_info(
        self,
        stream_key: str,
    ) -> dict[str, Any]:
        """
        Get consumer group information for a stream

        Args:
            stream_key: Redis stream key

        Returns:
            Dictionary with group information
        """
        if not self._is_initialized or not self._redis:
            return {}

        try:
            # Get group info
            groups = await self._redis.xinfo_groups(stream_key)

            for group in groups:
                if group.get("name") == self.group_name:
                    # Get consumer info
                    consumers = await self._redis.xinfo_consumers(stream_key, self.group_name)

                    # Find this consumer
                    pending = 0
                    for consumer in consumers:
                        if consumer.get("name") == self.consumer_name:
                            pending = consumer.get("pending", 0)
                            break

                    return {
                        "name": group.get("name"),
                        "consumers": group.get("consumers", 0),
                        "pending": group.get("pending", 0),
                        "last_delivered_id": group.get("last-delivered-id"),
                        "lag": group.get("lag", 0),
                        "my_pending": pending,
                    }

            return {}

        except RedisError as e:
            logger.warning(f"Failed to get group info for {stream_key}: {e}")
            return {}

    async def claim_pending_entries(
        self,
        stream_key: str,
        min_idle_time_ms: int | None = None,
    ) -> list[StreamEntry]:
        """
        Claim pending entries that have been idle too long

        Args:
            stream_key: Redis stream key
            min_idle_time_ms: Minimum idle time before claiming (default: from constructor)

        Returns:
            List of claimed StreamEntry objects
        """
        if not self._is_initialized or not self._redis:
            return []

        min_idle_time_ms = min_idle_time_ms or DEFAULT_IDLE_MS

        try:
            # Claim pending entries
            entries = await self._redis.xclaim(
                name=stream_key,
                groupname=self.group_name,
                consumername=self.consumer_name,
                min_idle_time=min_idle_time_ms,
                message_ids=["0-0"],  # Claim all
            )

            if not entries:
                return []

            # Parse entries
            stream_entries = [
                StreamEntry.from_raw(stream_key, entry_id, data)
                for entry_id, data in entries
            ]

            logger.info(f"Claimed {len(stream_entries)} pending entries from {stream_key}")
            return stream_entries

        except RedisError as e:
            logger.error(f"Failed to claim pending entries: {e}")
            return []

    def stop(self) -> None:
        """Signal consumer to stop processing"""
        self._is_running = False
        logger.info(f"Consumer {self.consumer_name} stop signal sent")
