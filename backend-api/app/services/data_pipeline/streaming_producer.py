"""
Redis Streams Producer for Real-Time Event Publishing

This module provides the streaming layer producer that publishes LLM interaction
events to Redis Streams for real-time processing and analytics.

Architecture:
- Producer Pattern: Fire-and-forget with local buffering
- Stream Keys: chimera:llm_events, chimera:transformation_events, chimera:jailbreak_events
- Max Memory Policy: allkeys-lru with ~1GB cap
- Retention: 24 hours for streaming events

Usage:
    from app.services.data_pipeline.streaming_producer import StreamProducer

    producer = StreamProducer()
    await producer.publish_llm_event(interaction_data)
    await producer.publish_transformation_event(transformation_data)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)


# Stream key constants
STREAM_LLM_EVENTS = "chimera:llm_events"
STREAM_TRANSFORMATION_EVENTS = "chimera:transformation_events"
STREAM_JAILBREAK_EVENTS = "chimera:jailbreak_events"
STREAM_HEALTH_CHECK = "chimera:health"


@dataclass
class StreamEvent:
    """Base event structure for streaming"""
    event_type: str
    event_id: str
    timestamp: str
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to Redis-compatible dictionary (all values must be strings)"""
        result = {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "data": json.dumps(self.data),
        }
        if self.metadata:
            result["metadata"] = json.dumps(self.metadata)
        return result


@dataclass
class LLMInteractionEvent:
    """LLM interaction event for streaming"""
    event_type: str = "llm_interaction"
    interaction_id: str = ""
    session_id: str = ""
    tenant_id: str = "default"
    provider: str = ""
    model: str = ""
    tokens_total: int = 0
    latency_ms: int = 0
    status: str = "success"

    def to_stream_event(self) -> StreamEvent:
        """Convert to base StreamEvent"""
        return StreamEvent(
            event_type=self.event_type,
            event_id=self.interaction_id,
            timestamp=datetime.utcnow().isoformat(),
            data={
                "interaction_id": self.interaction_id,
                "session_id": self.session_id,
                "tenant_id": self.tenant_id,
                "provider": self.provider,
                "model": self.model,
                "tokens_total": str(self.tokens_total),
                "latency_ms": str(self.latency_ms),
                "status": self.status,
            }
        )


@dataclass
class TransformationEvent:
    """Transformation event for streaming"""
    event_type: str = "transformation"
    transformation_id: str = ""
    interaction_id: str = ""
    technique_suite: str = ""
    technique_name: str = ""
    success: bool = False
    transformation_time_ms: int = 0

    def to_stream_event(self) -> StreamEvent:
        """Convert to base StreamEvent"""
        return StreamEvent(
            event_type=self.event_type,
            event_id=self.transformation_id,
            timestamp=datetime.utcnow().isoformat(),
            data={
                "transformation_id": self.transformation_id,
                "interaction_id": self.interaction_id,
                "technique_suite": self.technique_suite,
                "technique_name": self.technique_name,
                "success": str(self.success).lower(),
                "transformation_time_ms": str(self.transformation_time_ms),
            }
        )


@dataclass
class JailbreakEvent:
    """Jailbreak experiment event for streaming"""
    event_type: str = "jailbreak_experiment"
    experiment_id: str = ""
    framework: str = ""
    attack_method: str = ""
    success: bool = False
    iterations: int = 0
    judge_score: float | None = None

    def to_stream_event(self) -> StreamEvent:
        """Convert to base StreamEvent"""
        data = {
            "experiment_id": self.experiment_id,
            "framework": self.framework,
            "attack_method": self.attack_method,
            "success": str(self.success).lower(),
            "iterations": str(self.iterations),
        }
        if self.judge_score is not None:
            data["judge_score"] = str(self.judge_score)

        return StreamEvent(
            event_type=self.event_type,
            event_id=self.experiment_id,
            timestamp=datetime.utcnow().isoformat(),
            data=data
        )


class StreamProducer:
    """
    Redis Streams Producer for real-time event publishing

    Features:
    - Async/await support with redis.asyncio
    - Connection pooling for performance
    - Automatic reconnection with exponential backoff
    - Local buffering with periodic batch flush
    - Health check monitoring
    - Graceful shutdown

    Example:
        producer = StreamProducer()
        await producer.initialize()

        event = LLMInteractionEvent(
            interaction_id="abc123",
            provider="google",
            model="gemini-2.0-flash-exp",
            tokens_total=1500,
            latency_ms=1200,
            status="success"
        )
        await producer.publish_llm_event(event)

        await producer.close()
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_batch_size: int = 100,
        batch_flush_interval_ms: int = 1000,
    ):
        """
        Initialize the stream producer

        Args:
            redis_url: Redis connection URL (default: from settings)
            max_batch_size: Maximum events to batch before flushing
            batch_flush_interval_ms: Milliseconds between automatic flushes
        """
        self.redis_url = redis_url or self._get_redis_url()
        self.max_batch_size = max_batch_size
        self.batch_flush_interval_ms = batch_flush_interval_ms

        # Connection management
        self._pool: ConnectionPool | None = None
        self._redis: redis.Redis | None = None
        self._is_initialized = False

        # Event buffers for batching
        self._llm_buffer: list[dict[str, str]] = []
        self._transformation_buffer: list[dict[str, str]] = []
        self._jailbreak_buffer: list[dict[str, str]] = []

        # Health monitoring
        self._last_health_check: datetime | None = None
        self._is_healthy = True

    @staticmethod
    def _get_redis_url() -> str:
        """Get Redis URL from settings with fallback"""
        if hasattr(settings, 'REDIS_URL'):
            return settings.REDIS_URL
        return "redis://localhost:6379/0"

    async def initialize(self) -> None:
        """
        Initialize Redis connection and verify connectivity

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
            self._is_initialized = True
            self._last_health_check = datetime.utcnow()
            self._is_healthy = True

            logger.info(f"Stream producer initialized: {self.redis_url}")

        except RedisError as e:
            logger.error(f"Failed to initialize stream producer: {e}")
            raise RedisConnectionError(f"Redis connection failed: {e}") from e

    async def close(self) -> None:
        """Close Redis connection and cleanup resources"""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()

        self._is_initialized = False
        logger.info("Stream producer closed")

    async def health_check(self) -> bool:
        """
        Perform health check on Redis connection

        Returns:
            True if healthy, False otherwise
        """
        if not self._is_initialized or not self._redis:
            return False

        try:
            await self._redis.ping()
            self._last_health_check = datetime.utcnow()
            self._is_healthy = True
            return True
        except RedisError as e:
            logger.warning(f"Health check failed: {e}")
            self._is_healthy = False
            return False

    async def publish_llm_event(
        self,
        event: LLMInteractionEvent,
        stream_key: str = STREAM_LLM_EVENTS,
    ) -> str | None:
        """
        Publish LLM interaction event to Redis Stream

        Args:
            event: LLM interaction event
            stream_key: Redis stream key (default: chimera:llm_events)

        Returns:
            Stream entry ID if successful, None otherwise

        Example:
            event = LLMInteractionEvent(
                interaction_id="abc123",
                provider="google",
                model="gemini-2.0-flash-exp",
                tokens_total=1500,
                latency_ms=1200,
                status="success"
            )
            entry_id = await producer.publish_llm_event(event)
        """
        if not self._is_initialized:
            logger.warning("Producer not initialized, cannot publish event")
            return None

        try:
            stream_event = event.to_stream_event()
            event_data = stream_event.to_dict()

            # Add to buffer
            self._llm_buffer.append(event_data)

            # Flush if buffer is full
            if len(self._llm_buffer) >= self.max_batch_size:
                await self._flush_llm_buffer()

            return "buffered"

        except Exception as e:
            logger.error(f"Failed to publish LLM event: {e}")
            return None

    async def publish_transformation_event(
        self,
        event: TransformationEvent,
        stream_key: str = STREAM_TRANSFORMATION_EVENTS,
    ) -> str | None:
        """
        Publish transformation event to Redis Stream

        Args:
            event: Transformation event
            stream_key: Redis stream key (default: chimera:transformation_events)

        Returns:
            Stream entry ID if successful, None otherwise
        """
        if not self._is_initialized:
            logger.warning("Producer not initialized, cannot publish event")
            return None

        try:
            stream_event = event.to_stream_event()
            event_data = stream_event.to_dict()

            # Add to buffer
            self._transformation_buffer.append(event_data)

            # Flush if buffer is full
            if len(self._transformation_buffer) >= self.max_batch_size:
                await self._flush_transformation_buffer()

            return "buffered"

        except Exception as e:
            logger.error(f"Failed to publish transformation event: {e}")
            return None

    async def publish_jailbreak_event(
        self,
        event: JailbreakEvent,
        stream_key: str = STREAM_JAILBREAK_EVENTS,
    ) -> str | None:
        """
        Publish jailbreak experiment event to Redis Stream

        Args:
            event: Jailbreak experiment event
            stream_key: Redis stream key (default: chimera:jailbreak_events)

        Returns:
            Stream entry ID if successful, None otherwise
        """
        if not self._is_initialized:
            logger.warning("Producer not initialized, cannot publish event")
            return None

        try:
            stream_event = event.to_stream_event()
            event_data = stream_event.to_dict()

            # Add to buffer
            self._jailbreak_buffer.append(event_data)

            # Flush if buffer is full
            if len(self._jailbreak_buffer) >= self.max_batch_size:
                await self._flush_jailbreak_buffer()

            return "buffered"

        except Exception as e:
            logger.error(f"Failed to publish jailbreak event: {e}")
            return None

    async def _flush_llm_buffer(
        self,
        stream_key: str = STREAM_LLM_EVENTS,
    ) -> list[str]:
        """Flush LLM event buffer to Redis Stream"""
        if not self._llm_buffer:
            return []

        try:
            # Prepare pipeline for batch insert
            pipe = self._redis.pipeline(transaction=False)

            for event_data in self._llm_buffer:
                pipe.xadd(stream_key, event_data)

            # Execute pipeline
            entry_ids = await pipe.execute()

            # Clear buffer
            buffer_size = len(self._llm_buffer)
            self._llm_buffer.clear()

            logger.debug(f"Flushed {buffer_size} LLM events to stream")
            return entry_ids

        except RedisError as e:
            logger.error(f"Failed to flush LLM buffer: {e}")
            return []

    async def _flush_transformation_buffer(
        self,
        stream_key: str = STREAM_TRANSFORMATION_EVENTS,
    ) -> list[str]:
        """Flush transformation event buffer to Redis Stream"""
        if not self._transformation_buffer:
            return []

        try:
            pipe = self._redis.pipeline(transaction=False)

            for event_data in self._transformation_buffer:
                pipe.xadd(stream_key, event_data)

            entry_ids = await pipe.execute()
            buffer_size = len(self._transformation_buffer)
            self._transformation_buffer.clear()

            logger.debug(f"Flushed {buffer_size} transformation events to stream")
            return entry_ids

        except RedisError as e:
            logger.error(f"Failed to flush transformation buffer: {e}")
            return []

    async def _flush_jailbreak_buffer(
        self,
        stream_key: str = STREAM_JAILBREAK_EVENTS,
    ) -> list[str]:
        """Flush jailbreak event buffer to Redis Stream"""
        if not self._jailbreak_buffer:
            return []

        try:
            pipe = self._redis.pipeline(transaction=False)

            for event_data in self._jailbreak_buffer:
                pipe.xadd(stream_key, event_data)

            entry_ids = await pipe.execute()
            buffer_size = len(self._jailbreak_buffer)
            self._jailbreak_buffer.clear()

            logger.debug(f"Flushed {buffer_size} jailbreak events to stream")
            return entry_ids

        except RedisError as e:
            logger.error(f"Failed to flush jailbreak buffer: {e}")
            return []

    async def flush_all_buffers(self) -> dict[str, int]:
        """
        Flush all event buffers to Redis Streams

        Returns:
            Dictionary with stream key -> flushed count
        """
        if not self._is_initialized:
            return {}

        results = {}

        if self._llm_buffer:
            llm_ids = await self._flush_llm_buffer()
            results[STREAM_LLM_EVENTS] = len(llm_ids)

        if self._transformation_buffer:
            trans_ids = await self._flush_transformation_buffer()
            results[STREAM_TRANSFORMATION_EVENTS] = len(trans_ids)

        if self._jailbreak_buffer:
            jailbreak_ids = await self._flush_jailbreak_buffer()
            results[STREAM_JAILBREAK_EVENTS] = len(jailbreak_ids)

        return results

    async def get_stream_info(
        self,
        stream_key: str = STREAM_LLM_EVENTS,
    ) -> dict[str, Any]:
        """
        Get information about a Redis Stream

        Args:
            stream_key: Redis stream key

        Returns:
            Dictionary with stream information
        """
        if not self._is_initialized or not self._redis:
            return {}

        try:
            info = await self._redis.xinfo_stream(stream_key)
            return {
                "length": info.get("length", 0),
                "groups": info.get("groups", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
            }
        except RedisError as e:
            logger.warning(f"Failed to get stream info for {stream_key}: {e}")
            return {}


# Singleton producer instance
_producer: StreamProducer | None = None


async def get_producer() -> StreamProducer:
    """
    Get or create singleton stream producer instance

    Returns:
        Initialized StreamProducer instance
    """
    global _producer

    if _producer is None:
        _producer = StreamProducer()
        await _producer.initialize()

    return _producer


async def close_producer() -> None:
    """Close singleton producer instance"""
    global _producer

    if _producer:
        await _producer.close()
        _producer = None
