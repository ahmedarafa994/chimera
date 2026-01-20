"""Streaming Pipeline Service.

This module orchestrates the streaming data pipeline, coordinating the producer,
consumer, and real-time metrics components.

Architecture:
- Producer publishes events to Redis Streams
- Consumer groups process events in parallel (metrics, analytics, alerts)
- Real-time metrics aggregator records TimeSeries data
- Health monitoring and graceful shutdown

Usage:
    from app.services.data_pipeline.streaming_pipeline import StreamingPipeline

    pipeline = StreamingPipeline()
    await pipeline.initialize()

    # Publish an event
    await pipeline.publish_llm_event(interaction_data)

    # Background processing starts automatically

    await pipeline.close()
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from app.services.data_pipeline.realtime_metrics import AggregationType, MetricType, RealtimeMetrics
from app.services.data_pipeline.streaming_consumer import (
    GROUP_ALERTS,
    GROUP_ANALYTICS,
    GROUP_METRICS,
    StreamConsumer,
)
from app.services.data_pipeline.streaming_producer import (
    JailbreakEvent,
    LLMInteractionEvent,
    StreamProducer,
    TransformationEvent,
)

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """Orchestrates the streaming data pipeline.

    Features:
    - Automatic event publishing to Redis Streams
    - Parallel consumer groups for different processing needs
    - Real-time metrics aggregation
    - Health monitoring for all components
    - Graceful shutdown with pending message processing
    - Automatic reconnection on failures

    Consumer Groups:
    - chimera_metrics: Aggregates TimeSeries data for dashboards
    - chimera_analytics: Performs enrichment and joins
    - chimera_alerts: Monitors for anomalies and sends alerts

    Example:
        pipeline = StreamingPipeline()
        await pipeline.initialize()

        # Start background processing
        await pipeline.start()

        # Publish events
        await pipeline.publish_llm_event({
            "interaction_id": "abc123",
            "provider": "google",
            "model": "gemini-2.0-flash-exp",
            "tokens_total": 1500,
            "latency_ms": 1200,
            "status": "success"
        })

        # ... more events ...

        await pipeline.stop()
        await pipeline.close()

    """

    def __init__(
        self,
        redis_url: str | None = None,
    ) -> None:
        """Initialize the streaming pipeline.

        Args:
            redis_url: Redis connection URL

        """
        self.redis_url = redis_url

        # Components
        self._producer: StreamProducer | None = None
        self._metrics: RealtimeMetrics | None = None

        # Consumers
        self._metrics_consumer: StreamConsumer | None = None
        self._analytics_consumer: StreamConsumer | None = None
        self._alerts_consumer: StreamConsumer | None = None

        # Task management
        self._consumer_tasks: list[asyncio.Task] = []
        self._is_running = False
        self._is_initialized = False

        # Health monitoring
        self._last_health_check: datetime | None = None
        self._is_healthy = True

        # Custom handlers
        self._metrics_handler: Callable | None = None
        self._analytics_handler: Callable | None = None
        self._alerts_handler: Callable | None = None

    async def initialize(self) -> None:
        """Initialize all pipeline components.

        Raises:
            Exception: If initialization fails

        """
        try:
            logger.info("Initializing streaming pipeline...")

            # Initialize producer
            self._producer = StreamProducer(redis_url=self.redis_url)
            await self._producer.initialize()

            # Initialize metrics
            self._metrics = RealtimeMetrics(redis_url=self.redis_url)
            await self._metrics.initialize()

            # Initialize consumers
            self._metrics_consumer = StreamConsumer(
                consumer_name="metrics-consumer-1",
                group_name=GROUP_METRICS,
                redis_url=self.redis_url,
            )
            await self._metrics_consumer.initialize()

            self._analytics_consumer = StreamConsumer(
                consumer_name="analytics-consumer-1",
                group_name=GROUP_ANALYTICS,
                redis_url=self.redis_url,
            )
            await self._analytics_consumer.initialize()

            self._alerts_consumer = StreamConsumer(
                consumer_name="alerts-consumer-1",
                group_name=GROUP_ALERTS,
                redis_url=self.redis_url,
            )
            await self._alerts_consumer.initialize()

            self._is_initialized = True
            self._last_health_check = datetime.utcnow()
            self._is_healthy = True

            logger.info("Streaming pipeline initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize streaming pipeline: {e}")
            self._is_healthy = False
            raise

    async def start(
        self,
        process_metrics: bool = True,
        process_analytics: bool = False,
        process_alerts: bool = False,
    ) -> None:
        """Start background consumer tasks.

        Args:
            process_metrics: Start metrics consumer (default: True)
            process_analytics: Start analytics consumer (default: False)
            process_alerts: Start alerts consumer (default: False)

        """
        if not self._is_initialized:
            msg = "Pipeline not initialized"
            raise RuntimeError(msg)

        if self._is_running:
            logger.warning("Pipeline already running")
            return

        self._is_running = True
        logger.info("Starting streaming pipeline consumers...")

        # Start metrics consumer
        if process_metrics and self._metrics_consumer:
            task = asyncio.create_task(
                self._run_metrics_consumer(),
                name="metrics-consumer",
            )
            self._consumer_tasks.append(task)

        # Start analytics consumer
        if process_analytics and self._analytics_consumer:
            task = asyncio.create_task(
                self._run_analytics_consumer(),
                name="analytics-consumer",
            )
            self._consumer_tasks.append(task)

        # Start alerts consumer
        if process_alerts and self._alerts_consumer:
            task = asyncio.create_task(
                self._run_alerts_consumer(),
                name="alerts-consumer",
            )
            self._consumer_tasks.append(task)

        logger.info(f"Started {len(self._consumer_tasks)} consumer tasks")

    async def stop(self) -> None:
        """Stop background consumer tasks gracefully.

        Waits for pending messages to be processed
        """
        if not self._is_running:
            return

        logger.info("Stopping streaming pipeline...")

        # Signal consumers to stop
        if self._metrics_consumer:
            self._metrics_consumer.stop()
        if self._analytics_consumer:
            self._analytics_consumer.stop()
        if self._alerts_consumer:
            self._alerts_consumer.stop()

        # Wait for tasks to complete (with timeout)
        if self._consumer_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._consumer_tasks, return_exceptions=True),
                    timeout=30,
                )
            except TimeoutError:
                logger.warning("Consumer tasks did not complete within timeout")

        self._consumer_tasks.clear()
        self._is_running = False

        logger.info("Streaming pipeline stopped")

    async def close(self) -> None:
        """Close all pipeline components and cleanup resources."""
        await self.stop()

        if self._producer:
            await self._producer.close()
        if self._metrics:
            await self._metrics.close()

        if self._metrics_consumer:
            await self._metrics_consumer.close()
        if self._analytics_consumer:
            await self._analytics_consumer.close()
        if self._alerts_consumer:
            await self._alerts_consumer.close()

        self._is_initialized = False
        logger.info("Streaming pipeline closed")

    async def publish_llm_event(self, event_data: dict[str, Any]) -> bool:
        """Publish LLM interaction event to stream.

        Args:
            event_data: Event data dictionary

        Returns:
            True if published successfully

        """
        if not self._producer:
            return False

        event = LLMInteractionEvent(
            interaction_id=event_data.get("interaction_id", ""),
            session_id=event_data.get("session_id", ""),
            tenant_id=event_data.get("tenant_id", "default"),
            provider=event_data.get("provider", ""),
            model=event_data.get("model", ""),
            tokens_total=event_data.get("tokens_total", 0),
            latency_ms=event_data.get("latency_ms", 0),
            status=event_data.get("status", "success"),
        )

        result = await self._producer.publish_llm_event(event)
        return result is not None

    async def publish_transformation_event(self, event_data: dict[str, Any]) -> bool:
        """Publish transformation event to stream."""
        if not self._producer:
            return False

        event = TransformationEvent(
            transformation_id=event_data.get("transformation_id", ""),
            interaction_id=event_data.get("interaction_id", ""),
            technique_suite=event_data.get("technique_suite", ""),
            technique_name=event_data.get("technique_name", ""),
            success=event_data.get("success", False),
            transformation_time_ms=event_data.get("transformation_time_ms", 0),
        )

        result = await self._producer.publish_transformation_event(event)
        return result is not None

    async def publish_jailbreak_event(self, event_data: dict[str, Any]) -> bool:
        """Publish jailbreak experiment event to stream."""
        if not self._producer:
            return False

        event = JailbreakEvent(
            experiment_id=event_data.get("experiment_id", ""),
            framework=event_data.get("framework", ""),
            attack_method=event_data.get("attack_method", ""),
            success=event_data.get("success", False),
            iterations=event_data.get("iterations", 0),
            judge_score=event_data.get("judge_score"),
        )

        result = await self._producer.publish_jailbreak_event(event)
        return result is not None

    async def _run_metrics_consumer(self) -> None:
        """Run the metrics consumer (aggregates TimeSeries)."""
        logger.info("Metrics consumer started")

        try:
            async for batch in self._metrics_consumer.consume_llm_events():
                acknowledgements = []

                for entry in batch:
                    try:
                        # Extract data
                        data = entry.data

                        # Record metrics
                        await self._metrics.record_llm_request(
                            provider=data.get("provider", "unknown"),
                            model=data.get("model", "unknown"),
                            tokens=int(data.get("tokens_total", 0)),
                            latency_ms=int(data.get("latency_ms", 0)),
                            status=data.get("status", "unknown"),
                            tenant_id=data.get("tenant_id", "default"),
                            timestamp=entry.timestamp,
                        )

                        # Call custom handler if set
                        if self._metrics_handler:
                            await self._metrics_handler(entry)

                        acknowledgements.append((entry.stream_key, entry.entry_id))

                    except Exception as e:
                        logger.exception(f"Error processing metric entry: {e}")

                # Acknowledge batch
                if acknowledgements:
                    await self._metrics_consumer.acknowledge_batch(acknowledgements)

        except asyncio.CancelledError:
            logger.info("Metrics consumer cancelled")
        except Exception as e:
            logger.exception(f"Metrics consumer error: {e}")

    async def _run_analytics_consumer(self) -> None:
        """Run the analytics consumer (performs enrichment)."""
        logger.info("Analytics consumer started")

        try:
            async for batch in self._analytics_consumer.consume_llm_events():
                acknowledgements = []

                for entry in batch:
                    try:
                        # Perform enrichment or analytics
                        # This is where you'd add business logic for enrichment

                        # Call custom handler if set
                        if self._analytics_handler:
                            await self._analytics_handler(entry)

                        acknowledgements.append((entry.stream_key, entry.entry_id))

                    except Exception as e:
                        logger.exception(f"Error processing analytics entry: {e}")

                if acknowledgements:
                    await self._analytics_consumer.acknowledge_batch(acknowledgements)

        except asyncio.CancelledError:
            logger.info("Analytics consumer cancelled")
        except Exception as e:
            logger.exception(f"Analytics consumer error: {e}")

    async def _run_alerts_consumer(self) -> None:
        """Run the alerts consumer (monitors anomalies)."""
        logger.info("Alerts consumer started")

        try:
            async for batch in self._alerts_consumer.consume_llm_events():
                acknowledgements = []

                for entry in batch:
                    try:
                        # Check for anomalies
                        data = entry.data

                        # Example alert: high latency
                        latency_ms = int(data.get("latency_ms", 0))
                        if latency_ms > 10000:  # 10 second threshold
                            logger.warning(
                                f"High latency detected: {latency_ms}ms "
                                f"for provider={data.get('provider')}",
                            )

                        # Example alert: error spike
                        if data.get("status") == "error":
                            logger.warning(
                                f"Error detected for provider={data.get('provider')}, "
                                f"model={data.get('model')}",
                            )

                        # Call custom handler if set
                        if self._alerts_handler:
                            await self._alerts_handler(entry)

                        acknowledgements.append((entry.stream_key, entry.entry_id))

                    except Exception as e:
                        logger.exception(f"Error processing alerts entry: {e}")

                if acknowledgements:
                    await self._alerts_consumer.acknowledge_batch(acknowledgements)

        except asyncio.CancelledError:
            logger.info("Alerts consumer cancelled")
        except Exception as e:
            logger.exception(f"Alerts consumer error: {e}")

    def set_metrics_handler(self, handler: Callable) -> None:
        """Set custom handler for metrics processing."""
        self._metrics_handler = handler

    def set_analytics_handler(self, handler: Callable) -> None:
        """Set custom handler for analytics processing."""
        self._analytics_handler = handler

    def set_alerts_handler(self, handler: Callable) -> None:
        """Set custom handler for alerts processing."""
        self._alerts_handler = handler

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all components.

        Returns:
            Dictionary with health status

        """
        health = {
            "pipeline": "running" if self._is_running else "stopped",
            "initialized": self._is_initialized,
            "healthy": self._is_healthy,
            "components": {},
        }

        # Check producer
        if self._producer:
            health["components"]["producer"] = await self._producer.health_check()

        # Check metrics
        if self._metrics:
            # TimeSeries doesn't have health check, check connectivity
            try:
                await self._metrics._redis.ping()
                health["components"]["metrics"] = True
            except Exception:
                health["components"]["metrics"] = False

        # Check consumers
        for name, consumer in [
            ("metrics_consumer", self._metrics_consumer),
            ("analytics_consumer", self._analytics_consumer),
            ("alerts_consumer", self._alerts_consumer),
        ]:
            if consumer:
                try:
                    stats = await consumer.get_group_info(
                        consumer.stream_keys[0] if consumer.stream_keys else "chimera:llm_events",
                    )
                    health["components"][name] = bool(stats)
                except Exception:
                    health["components"][name] = False

        self._last_health_check = datetime.utcnow()
        return health

    async def get_metrics(
        self,
        provider: str,
        metric_type: MetricType = MetricType.LATENCY,
        start_ts: str = "-1h",
        aggregation_type: AggregationType = AggregationType.AVG,
        bucket_size_ms: int = 60000,
    ) -> dict[str, Any]:
        """Query real-time metrics.

        Args:
            provider: LLM provider
            metric_type: Type of metric
            start_ts: Start timestamp
            aggregation_type: Aggregation type
            bucket_size_ms: Bucket size

        Returns:
            Dictionary with metrics data

        """
        if not self._metrics:
            return {}

        result = await self._metrics.get_provider_metrics(
            provider=provider,
            metric_type=metric_type,
            start_ts=start_ts,
            aggregation_type=aggregation_type,
            bucket_size_ms=bucket_size_ms,
        )

        return result.to_dict()

    async def get_pipeline_stats(self) -> dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with pipeline stats

        """
        stats = {
            "running": self._is_running,
            "consumer_tasks": len(self._consumer_tasks),
            "producers": {},
            "consumers": {},
        }

        # Get producer stream info
        if self._producer:
            for stream_key in ["chimera:llm_events", "chimera:transformation_events"]:
                info = await self._producer.get_stream_info(stream_key)
                stats["producers"][stream_key] = info

        # Get consumer stats
        for name, consumer in [
            ("metrics", self._metrics_consumer),
            ("analytics", self._analytics_consumer),
            ("alerts", self._alerts_consumer),
        ]:
            if consumer:
                consumer_stats = await consumer.get_consumer_stats()
                stats["consumers"][name] = consumer_stats

        return stats


# Singleton pipeline instance
_pipeline: StreamingPipeline | None = None


async def get_pipeline() -> StreamingPipeline:
    """Get or create singleton pipeline instance."""
    global _pipeline

    if _pipeline is None:
        _pipeline = StreamingPipeline()
        await _pipeline.initialize()

    return _pipeline


async def close_pipeline() -> None:
    """Close singleton pipeline instance."""
    global _pipeline

    if _pipeline:
        await _pipeline.close()
        _pipeline = None
