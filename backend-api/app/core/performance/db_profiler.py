"""Advanced Database Performance Profiler for Chimera AI System.

Comprehensive profiling solution with:
- Real-time query performance monitoring
- N+1 query detection
- Connection pool analytics
- Slow query analysis and recommendations
- Cache hit rate optimization
- LLM provider-specific query patterns analysis
"""

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import psutil
from sqlalchemy import event
from sqlalchemy.pool import Pool

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a database query."""

    query_hash: str
    query_sql: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    rows_examined: int = 0
    rows_returned: int = 0
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    def update(
        self, execution_time_ms: float, rows_examined: int = 0, rows_returned: int = 0
    ) -> None:
        """Update metrics with new execution data."""
        self.execution_count += 1
        self.total_time_ms += execution_time_ms
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.avg_time_ms = self.total_time_ms / self.execution_count
        self.rows_examined += rows_examined
        self.rows_returned += rows_returned
        self.last_seen = datetime.utcnow()


@dataclass
class NPlusOnePattern:
    """Detected N+1 query pattern."""

    base_query_hash: str
    base_query: str
    child_query_hash: str
    child_query: str
    occurrences: int = 0
    first_detected: datetime = field(default_factory=datetime.utcnow)
    last_detected: datetime = field(default_factory=datetime.utcnow)
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics."""

    pool_name: str
    size: int
    checked_in: int
    checked_out: int
    overflow: int
    invalidated: int
    peak_usage: int = 0
    total_connections_created: int = 0
    total_connections_closed: int = 0
    wait_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class DatabaseProfiler:
    """Advanced database profiler for comprehensive performance monitoring.

    Features:
    - Real-time query execution tracking
    - N+1 query pattern detection
    - Connection pool monitoring
    - Performance recommendations
    - LLM provider query optimization
    """

    def __init__(self) -> None:
        self.enabled = settings.get("DB_PROFILER_ENABLED", True)
        self.slow_query_threshold_ms = settings.get("DB_SLOW_QUERY_THRESHOLD_MS", 100)
        self.n_plus_one_threshold = settings.get("DB_N_PLUS_ONE_THRESHOLD", 10)

        # Query tracking
        self.query_metrics: dict[str, QueryMetrics] = {}
        self.recent_queries: deque = deque(maxlen=1000)
        self.n_plus_one_patterns: dict[str, NPlusOnePattern] = {}

        # Connection pool tracking
        self.pool_metrics: dict[str, ConnectionPoolMetrics] = {}

        # Performance recommendations
        self.recommendations: list[dict[str, Any]] = []

        # Event tracking for N+1 detection
        self.query_stack: list[str] = []
        self.execution_contexts: dict[str, list[str]] = defaultdict(list)

        self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for comprehensive monitoring."""
        if not self.enabled:
            return

        @event.listens_for(Pool, "connect")
        def on_pool_connect(dbapi_connection, connection_record) -> None:
            """Track new connections."""
            pool_name = "default"  # Could be enhanced to track multiple pools
            if pool_name not in self.pool_metrics:
                self.pool_metrics[pool_name] = ConnectionPoolMetrics(
                    pool_name=pool_name,
                    size=0,
                    checked_in=0,
                    checked_out=0,
                    overflow=0,
                    invalidated=0,
                )
            self.pool_metrics[pool_name].total_connections_created += 1

        @event.listens_for(Pool, "checkout")
        def on_pool_checkout(dbapi_connection, connection_record, connection_proxy) -> None:
            """Track connection checkout with timing."""
            pool_name = "default"
            if pool_name in self.pool_metrics:
                self.pool_metrics[pool_name].checked_out += 1
                self.pool_metrics[pool_name].peak_usage = max(
                    self.pool_metrics[pool_name].peak_usage,
                    self.pool_metrics[pool_name].checked_out,
                )

        @event.listens_for(Pool, "checkin")
        def on_pool_checkin(dbapi_connection, connection_record) -> None:
            """Track connection checkin."""
            pool_name = "default"
            if pool_name in self.pool_metrics:
                self.pool_metrics[pool_name].checked_out = max(
                    0,
                    self.pool_metrics[pool_name].checked_out - 1,
                )
                self.pool_metrics[pool_name].checked_in += 1

    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        # Remove extra whitespace and normalize case
        normalized = re.sub(r"\s+", " ", query.strip().upper())

        # Replace parameters with placeholders
        normalized = re.sub(r"= \$?\d+", "= ?", normalized)
        normalized = re.sub(r"IN \([^)]+\)", "IN (?)", normalized)
        normalized = re.sub(r"'[^']+'", "'?'", normalized)
        return re.sub(r"\d+", "?", normalized)

    def _get_query_hash(self, query: str) -> str:
        """Generate consistent hash for query patterns."""
        normalized = self._normalize_query(query)
        return hashlib.md5(
            normalized.encode(),
        ).hexdigest()  # - cache key, not cryptographic

    @asynccontextmanager
    async def profile_query_execution(
        self,
        query: str,
        params: dict | None = None,
    ) -> AsyncGenerator[None, None]:
        """Context manager for profiling individual query execution."""
        if not self.enabled:
            yield
            return

        query_hash = self._get_query_hash(query)
        start_time = time.perf_counter()
        f"{query_hash}_{len(self.query_stack)}"

        # Track query stack for N+1 detection
        self.query_stack.append(query_hash)

        try:
            yield

            # Record successful execution
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._record_query_execution(query_hash, query, execution_time_ms, success=True)

        except Exception as e:
            # Record failed execution
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._record_query_execution(
                query_hash,
                query,
                execution_time_ms,
                success=False,
                error=str(e),
            )
            raise
        finally:
            # Clean up query stack
            if self.query_stack and self.query_stack[-1] == query_hash:
                self.query_stack.pop()

    def _record_query_execution(
        self,
        query_hash: str,
        query: str,
        execution_time_ms: float,
        success: bool = True,
        error: str | None = None,
        rows_examined: int = 0,
        rows_returned: int = 0,
    ) -> None:
        """Record query execution metrics and detect patterns."""
        # Update or create query metrics
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_sql=query[:500],  # Truncate for storage
            )

        self.query_metrics[query_hash].update(execution_time_ms, rows_examined, rows_returned)

        # Add to recent queries for N+1 detection
        query_info = {
            "hash": query_hash,
            "query": query[:200],
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow(),
            "success": success,
            "error": error,
            "stack_depth": len(self.query_stack),
        }
        self.recent_queries.append(query_info)

        # Detect N+1 patterns
        self._detect_n_plus_one_pattern(query_hash, query)

        # Generate recommendations for slow queries
        if execution_time_ms > self.slow_query_threshold_ms:
            self._analyze_slow_query(query_hash, query, execution_time_ms)

    def _detect_n_plus_one_pattern(self, query_hash: str, query: str) -> None:
        """Detect N+1 query patterns based on recent execution history."""
        if len(self.recent_queries) < 2:
            return

        # Look for repetitive patterns in recent queries
        recent_list = list(self.recent_queries)
        query_sequence = [q["hash"] for q in recent_list[-self.n_plus_one_threshold :]]

        # Count consecutive occurrences of the same query
        consecutive_count = 0
        for i in range(len(query_sequence) - 1, -1, -1):
            if query_sequence[i] == query_hash:
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 5:  # Threshold for N+1 detection
            pattern_key = f"n_plus_one_{query_hash}"

            if pattern_key not in self.n_plus_one_patterns:
                self.n_plus_one_patterns[pattern_key] = NPlusOnePattern(
                    base_query_hash="unknown",  # Would be enhanced with context analysis
                    base_query="unknown",
                    child_query_hash=query_hash,
                    child_query=query[:200],
                )

            pattern = self.n_plus_one_patterns[pattern_key]
            pattern.occurrences = consecutive_count
            pattern.last_detected = datetime.utcnow()
            pattern.severity = self._calculate_n_plus_one_severity(consecutive_count)

            logger.warning(
                f"N+1 query pattern detected: {consecutive_count} consecutive executions of query {query_hash}",
            )

    def _calculate_n_plus_one_severity(self, count: int) -> str:
        """Calculate severity of N+1 pattern based on occurrence count."""
        if count >= 100:
            return "CRITICAL"
        if count >= 50:
            return "HIGH"
        if count >= 20:
            return "MEDIUM"
        return "LOW"

    def _analyze_slow_query(self, query_hash: str, query: str, execution_time_ms: float) -> None:
        """Analyze slow query and generate optimization recommendations."""
        recommendations = []

        # Basic pattern analysis
        query_upper = query.upper()

        if "SELECT" in query_upper:
            if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
                recommendations.append(
                    {
                        "type": "missing_limit",
                        "message": "Consider adding LIMIT to ORDER BY queries to improve performance",
                        "severity": "MEDIUM",
                    },
                )

            if "WHERE" not in query_upper:
                recommendations.append(
                    {
                        "type": "missing_where",
                        "message": "Full table scan detected - consider adding WHERE clause",
                        "severity": "HIGH",
                    },
                )

            if "JOIN" in query_upper and "INDEX" not in query_upper:
                recommendations.append(
                    {
                        "type": "join_optimization",
                        "message": "JOIN operation without explicit index usage - verify indexes exist",
                        "severity": "MEDIUM",
                    },
                )

        # Store recommendations
        for rec in recommendations:
            rec.update(
                {
                    "query_hash": query_hash,
                    "execution_time_ms": execution_time_ms,
                    "timestamp": datetime.utcnow(),
                },
            )
            self.recommendations.append(rec)

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        now = datetime.utcnow()

        # Calculate summary metrics
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        avg_execution_time = (
            sum(m.avg_time_ms for m in self.query_metrics.values()) / len(self.query_metrics)
            if self.query_metrics
            else 0
        )

        # Find slowest queries
        slowest_queries = sorted(
            self.query_metrics.items(),
            key=lambda x: x[1].avg_time_ms,
            reverse=True,
        )[:10]

        # Get recent N+1 patterns
        recent_n_plus_one = [
            {
                "pattern_id": key,
                "child_query": pattern.child_query,
                "occurrences": pattern.occurrences,
                "severity": pattern.severity,
                "last_detected": pattern.last_detected.isoformat(),
            }
            for key, pattern in self.n_plus_one_patterns.items()
            if pattern.last_detected > now - timedelta(hours=1)
        ]

        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            "timestamp": now.isoformat(),
            "summary": {
                "total_queries": total_queries,
                "avg_execution_time_ms": round(avg_execution_time, 2),
                "slow_queries_count": len(
                    [
                        m
                        for m in self.query_metrics.values()
                        if m.avg_time_ms > self.slow_query_threshold_ms
                    ],
                ),
                "n_plus_one_patterns": len(recent_n_plus_one),
            },
            "slowest_queries": [
                {
                    "hash": hash_,
                    "sql": metrics.query_sql,
                    "execution_count": metrics.execution_count,
                    "avg_time_ms": round(metrics.avg_time_ms, 2),
                    "max_time_ms": round(metrics.max_time_ms, 2),
                    "last_seen": metrics.last_seen.isoformat(),
                }
                for hash_, metrics in slowest_queries
            ],
            "n_plus_one_patterns": recent_n_plus_one,
            "connection_pools": [
                {
                    "name": pool.pool_name,
                    "size": pool.size,
                    "checked_out": pool.checked_out,
                    "peak_usage": pool.peak_usage,
                    "total_created": pool.total_connections_created,
                }
                for pool in self.pool_metrics.values()
            ],
            "recommendations": self.recommendations[-20:],  # Latest 20 recommendations
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
            },
        }

    async def get_llm_provider_metrics(self) -> dict[str, Any]:
        """Get metrics specific to LLM provider operations."""
        provider_queries = {}

        for hash_, metrics in self.query_metrics.items():
            query = metrics.query_sql.upper()

            # Identify LLM-related queries
            if any(
                term in query for term in ["LLM_MODELS", "EVASION_TASKS", "JAILBREAK_", "PROVIDER"]
            ):
                provider_type = "unknown"

                if "LLM_MODELS" in query:
                    provider_type = "model_management"
                elif "EVASION_TASKS" in query:
                    provider_type = "jailbreak_research"
                elif "JAILBREAK_" in query:
                    provider_type = "jailbreak_dataset"

                if provider_type not in provider_queries:
                    provider_queries[provider_type] = []

                provider_queries[provider_type].append(
                    {
                        "hash": hash_,
                        "execution_count": metrics.execution_count,
                        "avg_time_ms": round(metrics.avg_time_ms, 2),
                        "total_time_ms": round(metrics.total_time_ms, 2),
                    },
                )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "provider_queries": provider_queries,
            "optimization_opportunities": self._identify_llm_optimizations(),
        }

    def _identify_llm_optimizations(self) -> list[dict[str, Any]]:
        """Identify specific optimization opportunities for LLM operations."""
        optimizations = []

        # Check for frequent provider config lookups
        config_queries = [
            metrics
            for metrics in self.query_metrics.values()
            if "LLM_MODELS" in metrics.query_sql.upper() and metrics.execution_count > 100
        ]

        if config_queries:
            optimizations.append(
                {
                    "type": "cache_provider_configs",
                    "description": "Frequent LLM provider config lookups detected - implement caching",
                    "priority": "HIGH",
                    "queries_affected": len(config_queries),
                },
            )

        # Check for repetitive jailbreak queries
        jailbreak_queries = [
            metrics
            for metrics in self.query_metrics.values()
            if any(term in metrics.query_sql.upper() for term in ["JAILBREAK_", "EVASION_"])
        ]

        if len(jailbreak_queries) > 50:
            optimizations.append(
                {
                    "type": "batch_jailbreak_operations",
                    "description": "High volume of individual jailbreak queries - consider batch operations",
                    "priority": "MEDIUM",
                    "queries_affected": len(jailbreak_queries),
                },
            )

        return optimizations

    def clear_metrics(self, older_than_hours: int = 24) -> None:
        """Clear old metrics to prevent memory bloat."""
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)

        # Clear old query metrics
        old_queries = [
            hash_ for hash_, metrics in self.query_metrics.items() if metrics.last_seen < cutoff
        ]

        for hash_ in old_queries:
            del self.query_metrics[hash_]

        # Clear old N+1 patterns
        old_patterns = [
            key
            for key, pattern in self.n_plus_one_patterns.items()
            if pattern.last_detected < cutoff
        ]

        for key in old_patterns:
            del self.n_plus_one_patterns[key]

        # Clear old recommendations
        self.recommendations = [
            rec for rec in self.recommendations if rec.get("timestamp", datetime.utcnow()) > cutoff
        ]

        logger.info(
            f"Cleared {len(old_queries)} old query metrics, {len(old_patterns)} old N+1 patterns",
        )


# Global profiler instance
db_profiler = DatabaseProfiler()


# Decorator for profiling async database operations
def profile_db_operation(operation_name: str):
    """Decorator to profile database operations."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            query_info = f"{operation_name}: {func.__name__}"
            async with db_profiler.profile_query_execution(query_info):
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            query_info = f"{operation_name}: {func.__name__}"
            # For sync operations, we'll use a simpler tracking approach
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.perf_counter() - start_time) * 1000
                hash_ = db_profiler._get_query_hash(query_info)
                db_profiler._record_query_execution(hash_, query_info, execution_time)
                return result
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                hash_ = db_profiler._get_query_hash(query_info)
                db_profiler._record_query_execution(
                    hash_,
                    query_info,
                    execution_time,
                    success=False,
                    error=str(e),
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
