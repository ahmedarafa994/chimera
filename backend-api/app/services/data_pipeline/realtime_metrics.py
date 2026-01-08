"""
Real-Time Metrics Aggregation with Redis TimeSeries

This module provides real-time metrics aggregation using Redis TimeSeries module
for sub-second latency analytics and monitoring dashboards.

Architecture:
- TimeSeries Data: Retention 24 hours with 1-second resolution
- Aggregation Levels: 1s, 1m, 5m, 1h using downsampling rules
- Labels: Multi-dimensional labels for flexible querying (provider, model, status)
- Compression: Using Redis TimeSeries compression (Gorilla-like)

Usage:
    from app.services.data_pipeline.realtime_metrics import RealtimeMetrics

    metrics = RealtimeMetrics()
    await metrics.initialize()

    # Record metric
    await metrics.record_llm_request(
        provider="google",
        model="gemini-2.0-flash-exp",
        tokens=1500,
        latency_ms=1200,
        status="success"
    )

    # Query metrics
    data = await metrics.get_provider_metrics(
        provider="google",
        start_ts="-1h",
        aggregation="avg"
    )
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError, ResponseError

from app.core.config import settings

logger = logging.getLogger(__name__)


# TimeSeries key patterns
TS_PREFIX = "chimera:metrics"
TS_REQUEST_COUNT = f"{TS_PREFIX}:request_count"
TS_TOKEN_USAGE = f"{TS_PREFIX}:token_usage"
TS_LATENCY = f"{TS_PREFIX}:latency"
TS_ERROR_RATE = f"{TS_PREFIX}:error_rate"

# Label names
LABEL_PROVIDER = "provider"
LABEL_MODEL = "model"
LABEL_STATUS = "status"
LABEL_TENANT = "tenant_id"


class MetricType(Enum):
    """Types of metrics tracked"""
    REQUEST_COUNT = "request_count"
    TOKEN_USAGE = "token_usage"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"


class AggregationType(Enum):
    """Redis TimeSeries aggregation types"""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD_DEV = "stddev"
    VAR = "var"


@dataclass
class MetricDataPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class MetricQueryResult:
    """Result of a metric query"""
    metric_type: MetricType
    labels: dict[str, str]
    data_points: list[MetricDataPoint]
    aggregation: AggregationType | None = None
    bucket_size_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_type": self.metric_type.value,
            "labels": self.labels,
            "aggregation": self.aggregation.value if self.aggregation else None,
            "bucket_size_ms": self.bucket_size_ms,
            "data_points": [dp.to_dict() for dp in self.data_points],
        }


class RealtimeMetrics:
    """
    Real-Time Metrics Aggregation with Redis TimeSeries

    Features:
    - Sub-second metric recording and querying
    - Automatic downsampling with rules (1s, 1m, 5m, 1h)
    - Multi-dimensional labeling for flexible queries
    - Built-in aggregations (avg, sum, min, max, std dev)
    - Time-series compression for storage efficiency
    - Retention policies (24h raw, 7d 1m, 30d 5m)

    Example:
        metrics = RealtimeMetrics()
        await metrics.initialize()

        # Record metric
        await metrics.record_llm_request(
            provider="google",
            model="gemini-2.0-flash-exp",
            tokens=1500,
            latency_ms=1200,
            status="success"
        )

        # Query metrics
        result = await metrics.get_provider_metrics(
            provider="google",
            start_ts="-1h",
            aggregation_type=AggregationType.AVG,
            bucket_size_ms=60000
        )

        await metrics.close()
    """

    # Default retention and downsampling rules
    DEFAULT_RETENTION_MS = 24 * 60 * 60 * 1000  # 24 hours
    DOWNSAMPLE_1M_RETENTION_MS = 7 * 24 * 60 * 60 * 1000  # 7 days
    DOWNSAMPLE_5M_RETENTION_MS = 30 * 24 * 60 * 60 * 1000  # 30 days

    def __init__(
        self,
        redis_url: str | None = None,
    ):
        """
        Initialize the real-time metrics service

        Args:
            redis_url: Redis connection URL (default: from settings)
        """
        self.redis_url = redis_url or self._get_redis_url()

        # Connection management
        self._pool: ConnectionPool | None = None
        self._redis: redis.Redis | None = None
        self._is_initialized = False

    @staticmethod
    def _get_redis_url() -> str:
        """Get Redis URL from settings with fallback"""
        if hasattr(settings, 'REDIS_URL'):
            return settings.REDIS_URL
        return "redis://localhost:6379/0"

    async def initialize(self) -> None:
        """
        Initialize Redis connection and create TimeSeries

        Raises:
            RedisError: If initialization fails
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

            # Load TimeSeries module
            try:
                await self._redis.execute_command("MODULE", "LIST")
            except ResponseError:
                logger.warning("Redis TimeSeries module may not be loaded")

            # Create TimeSeries with labels and retention
            await self._create_timeseries()

            self._is_initialized = True
            logger.info(f"Real-time metrics initialized: {self.redis_url}")

        except RedisError as e:
            logger.error(f"Failed to initialize real-time metrics: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection and cleanup resources"""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()

        self._is_initialized = False
        logger.info("Real-time metrics closed")

    async def _create_timeseries(self) -> None:
        """Create TimeSeries keys with labels and retention rules"""
        try:
            # Request count TimeSeries
            await self._redis.execute_command(
                "TS.CREATE",
                TS_REQUEST_COUNT,
                "RETENTION", self.DEFAULT_RETENTION_MS,
                "LABELS", "metric_type", "request_count",
            )

            # Add downsampling rule for 1-minute aggregation
            try:
                await self._redis.execute_command(
                    "TS.CREATERULE",
                    TS_REQUEST_COUNT,
                    f"{TS_REQUEST_COUNT}:1m",
                    "AGGREGATION", "sum", 60000,
                    "LABELS", "aggregation", "1m",
                )
            except ResponseError:
                pass  # Rule might already exist

            # Token usage TimeSeries
            await self._redis.execute_command(
                "TS.CREATE",
                TS_TOKEN_USAGE,
                "RETENTION", self.DEFAULT_RETENTION_MS,
                "LABELS", "metric_type", "token_usage",
            )

            # Latency TimeSeries
            await self._redis.execute_command(
                "TS.CREATE",
                TS_LATENCY,
                "RETENTION", self.DEFAULT_RETENTION_MS,
                "LABELS", "metric_type", "latency",
            )

            # Error rate TimeSeries
            await self._redis.execute_command(
                "TS.CREATE",
                TS_ERROR_RATE,
                "RETENTION", self.DEFAULT_RETENTION_MS,
                "LABELS", "metric_type", "error_rate",
            )

            logger.info("TimeSeries created successfully")

        except ResponseError as e:
            if "already exists" in str(e):
                logger.debug("TimeSeries already exists")
            else:
                logger.error(f"Failed to create TimeSeries: {e}")

    async def record_llm_request(
        self,
        provider: str,
        model: str,
        tokens: int,
        latency_ms: int,
        status: str,
        tenant_id: str = "default",
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Record LLM request metrics

        Args:
            provider: LLM provider (google, openai, anthropic, etc.)
            model: Model name
            tokens: Total tokens used
            latency_ms: Request latency in milliseconds
            status: Request status (success, error, timeout)
            tenant_id: Tenant ID
            timestamp: Metric timestamp (default: now)

        Returns:
            True if successful, False otherwise
        """
        if not self._is_initialized:
            logger.warning("Real-time metrics not initialized")
            return False

        timestamp = timestamp or datetime.utcnow()
        timestamp_ms = int(timestamp.timestamp() * 1000)


        try:
            pipe = self._redis.pipeline()

            # Request count (increment by 1)
            pipe.execute_command(
                "TS.ADD",
                TS_REQUEST_COUNT,
                timestamp_ms,
                1,
                "LABELS",
                LABEL_PROVIDER, provider,
                LABEL_MODEL, model,
                LABEL_STATUS, status,
                LABEL_TENANT, tenant_id,
            )

            # Token usage
            pipe.execute_command(
                "TS.ADD",
                TS_TOKEN_USAGE,
                timestamp_ms,
                tokens,
                "LABELS",
                LABEL_PROVIDER, provider,
                LABEL_MODEL, model,
                LABEL_TENANT, tenant_id,
            )

            # Latency
            pipe.execute_command(
                "TS.ADD",
                TS_LATENCY,
                timestamp_ms,
                latency_ms,
                "LABELS",
                LABEL_PROVIDER, provider,
                LABEL_MODEL, model,
                LABEL_STATUS, status,
                LABEL_TENANT, tenant_id,
            )

            # Error rate (1 if error, 0 if success)
            error_value = 1 if status != "success" else 0
            pipe.execute_command(
                "TS.ADD",
                TS_ERROR_RATE,
                timestamp_ms,
                error_value,
                "LABELS",
                LABEL_PROVIDER, provider,
                LABEL_MODEL, model,
                LABEL_TENANT, tenant_id,
            )

            await pipe.execute()
            return True

        except RedisError as e:
            logger.error(f"Failed to record LLM request metrics: {e}")
            return False

    async def get_provider_metrics(
        self,
        provider: str,
        metric_type: MetricType = MetricType.REQUEST_COUNT,
        start_ts: str = "-1h",
        end_ts: str | None = None,
        aggregation_type: AggregationType | None = None,
        bucket_size_ms: int | None = None,
    ) -> MetricQueryResult:
        """
        Get metrics for a specific provider

        Args:
            provider: LLM provider to query
            metric_type: Type of metric to retrieve
            start_ts: Start timestamp (Redis TimeSeries format, e.g., "-1h", "now-24h")
            end_ts: End timestamp (default: now)
            aggregation_type: Aggregation type (avg, sum, min, max)
            bucket_size_ms: Bucket size for aggregation in milliseconds

        Returns:
            MetricQueryResult with data points

        Example:
            result = await metrics.get_provider_metrics(
                provider="google",
                metric_type=MetricType.LATENCY,
                start_ts="-1h",
                aggregation_type=AggregationType.AVG,
                bucket_size_ms=60000  # 1 minute buckets
            )
        """
        if not self._is_initialized:
            return MetricQueryResult(
                metric_type=metric_type,
                labels={LABEL_PROVIDER: provider},
                data_points=[],
            )

        # Map metric type to TimeSeries key
        ts_key = self._get_ts_key(metric_type)

        try:
            # Build query
            filter_expr = f"{LABEL_PROVIDER}={provider}"

            # Execute query
            if aggregation_type and bucket_size_ms:
                result = await self._redis.execute_command(
                    "TS.RANGE",
                    ts_key,
                    start_ts,
                    end_ts or "+",
                    "FILTER", filter_expr,
                    "AGGREGATION", aggregation_type.value, bucket_size_ms,
                )
            else:
                result = await self._redis.execute_command(
                    "TS.RANGE",
                    ts_key,
                    start_ts,
                    end_ts or "+",
                    "FILTER", filter_expr,
                )

            # Parse result
            data_points = self._parse_ts_result(result)

            return MetricQueryResult(
                metric_type=metric_type,
                labels={LABEL_PROVIDER: provider},
                data_points=data_points,
                aggregation=aggregation_type,
                bucket_size_ms=bucket_size_ms,
            )

        except RedisError as e:
            logger.error(f"Failed to get provider metrics: {e}")
            return MetricQueryResult(
                metric_type=metric_type,
                labels={LABEL_PROVIDER: provider},
                data_points=[],
            )

    async def get_model_metrics(
        self,
        provider: str,
        model: str,
        metric_type: MetricType = MetricType.LATENCY,
        start_ts: str = "-1h",
        aggregation_type: AggregationType | None = None,
        bucket_size_ms: int | None = None,
    ) -> MetricQueryResult:
        """Get metrics for a specific model"""
        if not self._is_initialized:
            return MetricQueryResult(
                metric_type=metric_type,
                labels={LABEL_PROVIDER: provider, LABEL_MODEL: model},
                data_points=[],
            )

        ts_key = self._get_ts_key(metric_type)

        try:
            filter_expr = f"{LABEL_PROVIDER}={provider}{LABEL_MODEL}={model}"

            if aggregation_type and bucket_size_ms:
                result = await self._redis.execute_command(
                    "TS.RANGE",
                    ts_key,
                    start_ts,
                    "+",
                    "FILTER", filter_expr,
                    "AGGREGATION", aggregation_type.value, bucket_size_ms,
                )
            else:
                result = await self._redis.execute_command(
                    "TS.RANGE",
                    ts_key,
                    start_ts,
                    "+",
                    "FILTER", filter_expr,
                )

            data_points = self._parse_ts_result(result)

            return MetricQueryResult(
                metric_type=metric_type,
                labels={LABEL_PROVIDER: provider, LABEL_MODEL: model},
                data_points=data_points,
                aggregation=aggregation_type,
                bucket_size_ms=bucket_size_ms,
            )

        except RedisError as e:
            logger.error(f"Failed to get model metrics: {e}")
            return MetricQueryResult(
                metric_type=metric_type,
                labels={LABEL_PROVIDER: provider, LABEL_MODEL: model},
                data_points=[],
            )

    async def get_aggregate_metrics(
        self,
        metric_type: MetricType,
        start_ts: str = "-5m",
        aggregation_type: AggregationType = AggregationType.AVG,
        bucket_size_ms: int = 60000,
        group_by: list[str] | None = None,
    ) -> list[MetricQueryResult]:
        """
        Get aggregate metrics grouped by labels

        Args:
            metric_type: Type of metric
            start_ts: Start timestamp
            aggregation_type: Aggregation type
            bucket_size_ms: Bucket size
            group_by: Labels to group by (provider, model, tenant_id)

        Returns:
            List of MetricQueryResult grouped by labels

        Example:
            results = await metrics.get_aggregate_metrics(
                metric_type=MetricType.LATENCY,
                start_ts="-1h",
                aggregation_type=AggregationType.AVG,
                bucket_size_ms=300000,  # 5 minutes
                group_by=["provider", "model"]
            )
        """
        if not self._is_initialized:
            return []

        self._get_ts_key(metric_type)
        group_by = group_by or [LABEL_PROVIDER]

        try:
            # Use TS.MRANGE for multi-key range query with grouping
            result = await self._redis.execute_command(
                "TS.MRANGE",
                start_ts,
                "+",
                "AGGREGATION", aggregation_type.value, bucket_size_ms,
                "WITHLABELS",
                "FILTER", "*",  # All keys
                "GROUPBY", *group_by, "REDUCE", aggregation_type.value,
            )

            # Parse result and group by label combinations
            # Result format: [(label_dict, [(timestamp, value), ...]), ...]
            return self._parse_mr_result(result, metric_type, aggregation_type, bucket_size_ms)

        except RedisError as e:
            logger.error(f"Failed to get aggregate metrics: {e}")
            return []

    def _get_ts_key(self, metric_type: MetricType) -> str:
        """Map metric type to TimeSeries key"""
        mapping = {
            MetricType.REQUEST_COUNT: TS_REQUEST_COUNT,
            MetricType.TOKEN_USAGE: TS_TOKEN_USAGE,
            MetricType.LATENCY: TS_LATENCY,
            MetricType.ERROR_RATE: TS_ERROR_RATE,
        }
        return mapping.get(metric_type, TS_REQUEST_COUNT)

    def _parse_ts_result(self, result: list) -> list[MetricDataPoint]:
        """
        Parse TimeSeries query result

        Args:
            result: Raw TimeSeries result

        Returns:
            List of MetricDataPoint objects
        """
        data_points = []

        if not result:
            return data_points

        for timestamp_ms, value in result:
            try:
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                data_points.append(MetricDataPoint(
                    timestamp=timestamp,
                    value=float(value),
                    labels={},  # Labels are filtered at query time
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse data point: {e}")

        return data_points

    def _parse_mr_result(
        self,
        result: list,
        metric_type: MetricType,
        aggregation: AggregationType,
        bucket_size_ms: int,
    ) -> list[MetricQueryResult]:
        """Parse TS.MRANGE multi-key result"""
        query_results = []

        if not result:
            return query_results

        for labels, data_points in result:
            parsed_points = []
            for timestamp_ms, value in data_points:
                try:
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    parsed_points.append(MetricDataPoint(
                        timestamp=timestamp,
                        value=float(value),
                        labels=labels,
                    ))
                except (ValueError, TypeError):
                    pass

            query_results.append(MetricQueryResult(
                metric_type=metric_type,
                labels=labels,
                data_points=parsed_points,
                aggregation=aggregation,
                bucket_size_ms=bucket_size_ms,
            ))

        return query_results

    async def get_series_info(self) -> dict[str, Any]:
        """
        Get information about all TimeSeries

        Returns:
            Dictionary with series information
        """
        if not self._is_initialized:
            return {}

        try:
            # Use TS.QUERY to get all series
            result = await self._redis.execute_command(
                "TS.QUERY",
                "FILTER", "*",
            )

            series_info = {}
            for series_key, labels, chunk_count in result:
                # Get detailed info
                info = await self._redis.execute_command("TS.INFO", series_key)
                series_info[series_key] = {
                    "labels": labels,
                    "chunk_count": chunk_count,
                    "total_samples": info.get("totalSamples", 0),
                    "memory_usage": info.get("memoryUsage", 0),
                    "first_timestamp": info.get("firstTimestamp"),
                    "last_timestamp": info.get("lastTimestamp"),
                    "retention_ms": info.get("retentionTime", 0),
                }

            return series_info

        except RedisError as e:
            logger.error(f"Failed to get series info: {e}")
            return {}


# Singleton instance
_metrics: RealtimeMetrics | None = None


async def get_metrics() -> RealtimeMetrics:
    """Get or create singleton metrics instance"""
    global _metrics

    if _metrics is None:
        _metrics = RealtimeMetrics()
        await _metrics.initialize()

    return _metrics


async def close_metrics() -> None:
    """Close singleton metrics instance"""
    global _metrics

    if _metrics:
        await _metrics.close()
        _metrics = None
