"""
Database and Redis Performance Profiling
Monitors query performance, connection pooling, cache hit rates, and optimization opportunities
"""

# Database connectors
import importlib.util as _importlib
import json
import re
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from profiling_config import MetricType, config

# Availability checks for optional DB/Redis connectors
SQLITE_AVAILABLE = _importlib.find_spec("sqlite3") is not None
AIOSQLITE_AVAILABLE = _importlib.find_spec("aiosqlite") is not None

POSTGRES_AVAILABLE = (
    _importlib.find_spec("asyncpg") is not None or _importlib.find_spec("psycopg2") is not None
)

# Redis connector availability
REDIS_AVAILABLE = _importlib.find_spec("redis") is not None
AIREDIS_AVAILABLE = _importlib.find_spec("redis.asyncio") is not None


@dataclass
class QueryMetrics:
    """Database query performance metrics"""

    query_id: str
    query: str
    execution_time_ms: float
    rows_affected: int
    timestamp: datetime
    database_type: str
    connection_id: str | None = None
    plan: dict[str, Any] | None = None
    indexes_used: list[str] = None


@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics"""

    pool_name: str
    active_connections: int
    idle_connections: int
    total_connections: int
    max_connections: int
    queue_size: int
    checkout_time_ms: float
    timestamp: datetime


@dataclass
class CacheMetrics:
    """Redis cache performance metrics"""

    operation: str  # get, set, delete, etc.
    key_pattern: str
    hit: bool
    execution_time_ms: float
    data_size_bytes: int
    timestamp: datetime
    ttl_seconds: int | None = None


@dataclass
class DatabasePerformanceReport:
    """Comprehensive database performance report"""

    report_id: str
    timestamp: datetime
    query_metrics: list[QueryMetrics]
    connection_metrics: list[ConnectionPoolMetrics]
    cache_metrics: list[CacheMetrics]
    slow_queries: list[QueryMetrics]
    optimization_recommendations: list[str]
    performance_summary: dict[str, Any]


class DatabaseProfiler:
    """Database and cache performance profiler"""

    def __init__(self):
        self.query_metrics: deque = deque(maxlen=10000)
        self.connection_metrics: deque = deque(maxlen=1000)
        self.cache_metrics: deque = deque(maxlen=5000)

        # Query analysis patterns
        self.query_patterns = {
            "select": re.compile(r"^\s*select\b", re.IGNORECASE),
            "insert": re.compile(r"^\s*insert\b", re.IGNORECASE),
            "update": re.compile(r"^\s*update\b", re.IGNORECASE),
            "delete": re.compile(r"^\s*delete\b", re.IGNORECASE),
            "join": re.compile(r"\bjoin\b", re.IGNORECASE),
            "subquery": re.compile(r"\(\s*select\b", re.IGNORECASE),
            "index_scan": re.compile(r"index\s+scan", re.IGNORECASE),
        }

        # Connection pools tracking
        self.connection_pools: dict[str, Any] = {}

        # Redis client for monitoring
        self.redis_client: Any | None = None
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis client for monitoring"""
        if not REDIS_AVAILABLE:
            return

        try:
            import importlib

            redis = importlib.import_module("redis")

            self.redis_client = redis.from_url(
                config.redis_url,
                decode_responses=True,
                health_check_interval=30,
            )
            # Test connection
            self.redis_client.ping()
            print("Redis connection established for profiling")
        except Exception as e:
            print(f"Could not connect to Redis: {e}")
            self.redis_client = None

    def profile_query(
        self, query: str, database_type: str = "sqlite", connection_id: str | None = None
    ) -> QueryMetrics:
        """Profile a single database query"""
        query_id = f"query_{int(time.time())}_{hash(query) % 10000}"
        start_time = time.time()

        # Execute query based on database type
        rows_affected = 0
        execution_plan = None

        try:
            if database_type.lower() == "sqlite" and SQLITE_AVAILABLE:
                rows_affected, execution_plan = self._execute_sqlite_query(query)
            elif database_type.lower() == "postgres" and POSTGRES_AVAILABLE:
                rows_affected, execution_plan = self._execute_postgres_query(query)
            else:
                # Simulate query execution for profiling
                time.sleep(0.001)  # Minimal delay
                rows_affected = 1

        except Exception as e:
            print(f"Error executing query: {e}")

        execution_time_ms = (time.time() - start_time) * 1000

        # Analyze query for index usage
        indexes_used = self._analyze_query_indexes(query)

        metrics = QueryMetrics(
            query_id=query_id,
            query=query[:500],  # Truncate long queries
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            timestamp=datetime.now(UTC),
            database_type=database_type,
            connection_id=connection_id,
            plan=execution_plan,
            indexes_used=indexes_used,
        )

        self.query_metrics.append(metrics)

        # Log slow queries
        if execution_time_ms > 1000:  # Slower than 1 second
            print(f"Slow query detected: {execution_time_ms:.0f}ms - {query[:100]}...")

        return metrics

    def _execute_sqlite_query(self, query: str) -> tuple:
        """Execute SQLite query and get execution plan"""
        try:
            import sqlite3

            # Connect to Chimera SQLite database
            db_path = "D:/MUZIK/chimera/chimera.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get query plan
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            cursor.execute(explain_query)
            plan_rows = cursor.fetchall()

            execution_plan = {
                "plan_rows": [
                    {"id": row[0], "parent": row[1], "detail": row[3]} for row in plan_rows
                ]
            }

            # Execute actual query
            cursor.execute(query)
            rows_affected = cursor.rowcount

            conn.close()
            return rows_affected, execution_plan

        except Exception as e:
            print(f"SQLite query execution error: {e}")
            return 0, None

    def _execute_postgres_query(self, _query: str) -> tuple:
        """Execute PostgreSQL query and get execution plan"""
        # This would require PostgreSQL connection details
        # For now, simulate execution
        return 1, {"plan": "simulated"}

    def _analyze_query_indexes(self, query: str) -> list[str]:
        """Analyze query for potential index usage"""
        indexes = []

        # Look for WHERE clauses that might benefit from indexes
        where_match = re.search(
            r"\bwhere\s+(.+?)(?:\border\s+by|\bgroup\s+by|\blimit\s+|\bhaving\s+|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )

        if where_match:
            where_clause = where_match.group(1)

            # Extract column names from WHERE clause
            column_matches = re.findall(r"(\w+)\s*[=<>!]", where_clause)
            for column in column_matches:
                indexes.append(f"idx_{column}")

        return indexes

    async def monitor_connection_pool(self, pool_name: str, pool_obj: Any) -> ConnectionPoolMetrics:
        """Monitor database connection pool performance"""
        start_time = time.time()

        try:
            # Get pool statistics (this will vary by pool implementation)
            if hasattr(pool_obj, "get_stats"):
                stats = pool_obj.get_stats()
                active = stats.get("active", 0)
                idle = stats.get("idle", 0)
                total = stats.get("total", 0)
                max_size = stats.get("max_size", 0)
                queue_size = stats.get("queue_size", 0)
            else:
                # Simulate pool stats
                active = 2
                idle = 3
                total = 5
                max_size = 10
                queue_size = 0

        except Exception as e:
            print(f"Error monitoring connection pool: {e}")
            active = idle = total = max_size = queue_size = 0

        checkout_time_ms = (time.time() - start_time) * 1000

        metrics = ConnectionPoolMetrics(
            pool_name=pool_name,
            active_connections=active,
            idle_connections=idle,
            total_connections=total,
            max_connections=max_size,
            queue_size=queue_size,
            checkout_time_ms=checkout_time_ms,
            timestamp=datetime.now(UTC),
        )

        self.connection_metrics.append(metrics)
        return metrics

    def profile_cache_operation(
        self, operation: str, key: str, data: Any = None, ttl: int | None = None
    ) -> CacheMetrics:
        """Profile Redis cache operation"""
        if not self.redis_client:
            # Return dummy metrics if Redis not available
            return CacheMetrics(
                operation=operation,
                key_pattern=self._get_key_pattern(key),
                hit=False,
                execution_time_ms=0.0,
                data_size_bytes=0,
                timestamp=datetime.now(UTC),
                ttl_seconds=ttl,
            )

        start_time = time.time()
        hit = False
        data_size = 0

        try:
            if operation.lower() == "get":
                result = self.redis_client.get(key)
                hit = result is not None
                if result:
                    data_size = len(str(result).encode("utf-8"))

            elif operation.lower() == "set":
                if ttl:
                    self.redis_client.setex(key, ttl, str(data))
                else:
                    self.redis_client.set(key, str(data))
                hit = True
                data_size = len(str(data).encode("utf-8"))

            elif operation.lower() == "delete":
                result = self.redis_client.delete(key)
                hit = result > 0

            elif operation.lower() == "exists":
                result = self.redis_client.exists(key)
                hit = result > 0

        except Exception as e:
            print(f"Redis operation error: {e}")

        execution_time_ms = (time.time() - start_time) * 1000

        metrics = CacheMetrics(
            operation=operation,
            key_pattern=self._get_key_pattern(key),
            hit=hit,
            execution_time_ms=execution_time_ms,
            data_size_bytes=data_size,
            timestamp=datetime.now(UTC),
            ttl_seconds=ttl,
        )

        self.cache_metrics.append(metrics)

        # Log slow cache operations
        if execution_time_ms > 100:  # Slower than 100ms
            print(f"Slow cache operation: {operation} on {key} took {execution_time_ms:.0f}ms")

        return metrics

    def _get_key_pattern(self, key: str) -> str:
        """Extract pattern from Redis key"""
        # Replace IDs and timestamps with wildcards
        pattern = re.sub(r"\d+", "*", key)
        pattern = re.sub(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "UUID", pattern
        )
        return pattern

    def get_redis_info(self) -> dict[str, Any]:
        """Get Redis server information and statistics"""
        if not self.redis_client:
            return {}

        try:
            info = self.redis_client.info()
            memory_info = self.redis_client.info("memory")
            stats_info = self.redis_client.info("stats")

            return {
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": memory_info.get("used_memory", 0),
                "used_memory_human": memory_info.get("used_memory_human", "0B"),
                "keyspace_hits": stats_info.get("keyspace_hits", 0),
                "keyspace_misses": stats_info.get("keyspace_misses", 0),
                "total_commands_processed": stats_info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            }
        except Exception as e:
            print(f"Error getting Redis info: {e}")
            return {}

    def analyze_slow_queries(self, threshold_ms: float = 1000.0) -> list[QueryMetrics]:
        """Identify and analyze slow queries"""
        slow_queries = [q for q in self.query_metrics if q.execution_time_ms > threshold_ms]

        # Sort by execution time descending
        slow_queries.sort(key=lambda q: q.execution_time_ms, reverse=True)

        return slow_queries[:50]  # Top 50 slow queries

    def analyze_cache_performance(self) -> dict[str, Any]:
        """Analyze cache hit rates and performance"""
        if not self.cache_metrics:
            return {}

        # Calculate hit rates by operation and key pattern
        hit_rates = defaultdict(lambda: {"hits": 0, "total": 0})
        avg_times = defaultdict(list)

        for metric in self.cache_metrics:
            key = f"{metric.operation}:{metric.key_pattern}"
            hit_rates[key]["total"] += 1
            avg_times[key].append(metric.execution_time_ms)

            if metric.hit:
                hit_rates[key]["hits"] += 1

        # Calculate statistics
        cache_stats = {}
        for key, data in hit_rates.items():
            hit_rate = data["hits"] / data["total"] if data["total"] > 0 else 0
            avg_time = statistics.mean(avg_times[key]) if avg_times[key] else 0

            cache_stats[key] = {
                "hit_rate": hit_rate,
                "total_operations": data["total"],
                "avg_response_time_ms": avg_time,
            }

        # Overall cache statistics
        total_operations = sum(data["total"] for data in hit_rates.values())
        total_hits = sum(data["hits"] for data in hit_rates.values())
        overall_hit_rate = total_hits / total_operations if total_operations > 0 else 0

        redis_info = self.get_redis_info()

        return {
            "overall_hit_rate": overall_hit_rate,
            "total_operations": total_operations,
            "operation_stats": cache_stats,
            "redis_info": redis_info,
        }

    def generate_optimization_recommendations(self) -> list[str]:
        """Generate database and cache optimization recommendations"""
        recommendations = []

        # Analyze slow queries
        slow_queries = self.analyze_slow_queries()
        if slow_queries:
            recommendations.append(
                f"Found {len(slow_queries)} slow queries - consider query optimization"
            )

            # Check for common optimization opportunities
            for query in slow_queries[:10]:  # Top 10 slow queries
                if not query.indexes_used:
                    recommendations.append(
                        f"Consider adding indexes for query: {query.query[:100]}..."
                    )

                if "join" in query.query.lower() and query.execution_time_ms > 5000:
                    recommendations.append("Optimize JOIN operations with proper indexing")

                if "select *" in query.query.lower():
                    recommendations.append("Avoid SELECT * queries - specify needed columns")

        # Analyze cache performance
        cache_stats = self.analyze_cache_performance()
        if cache_stats:
            overall_hit_rate = cache_stats.get("overall_hit_rate", 0)

            if overall_hit_rate < 0.8:  # Less than 80% hit rate
                recommendations.append(
                    f"Cache hit rate is {overall_hit_rate:.1%} - consider optimizing cache strategy"
                )

            if overall_hit_rate < 0.5:  # Less than 50% hit rate
                recommendations.append(
                    "Very low cache hit rate - review caching patterns and TTL values"
                )

        # Connection pool analysis
        if self.connection_metrics:
            recent_pool_metrics = list(self.connection_metrics)[-10:]  # Last 10 measurements

            for metric in recent_pool_metrics:
                utilization = (
                    metric.active_connections / metric.max_connections
                    if metric.max_connections > 0
                    else 0
                )

                if utilization > 0.8:  # Over 80% pool utilization
                    recommendations.append(
                        f"High connection pool utilization ({utilization:.1%}) for {metric.pool_name}"
                    )

                if metric.queue_size > 0:
                    recommendations.append(
                        f"Connection pool queue detected for {metric.pool_name} - consider increasing pool size"
                    )

        # General recommendations
        recommendations.extend(
            [
                "Monitor query execution plans regularly",
                "Implement database query result caching",
                "Use connection pooling for better resource utilization",
                "Consider read replicas for read-heavy workloads",
                "Regular database maintenance (vacuum, analyze, reindex)",
            ]
        )

        return recommendations

    def generate_database_report(self) -> DatabasePerformanceReport:
        """Generate comprehensive database performance report"""
        report_id = f"db_report_{int(time.time())}"
        timestamp = datetime.now(UTC)

        # Get slow queries
        slow_queries = self.analyze_slow_queries()

        # Get cache performance
        cache_analysis = self.analyze_cache_performance()

        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()

        # Performance summary
        performance_summary = {
            "total_queries": len(self.query_metrics),
            "slow_queries_count": len(slow_queries),
            "avg_query_time_ms": (
                statistics.mean([q.execution_time_ms for q in self.query_metrics])
                if self.query_metrics
                else 0
            ),
            "cache_hit_rate": cache_analysis.get("overall_hit_rate", 0),
            "total_cache_operations": cache_analysis.get("total_operations", 0),
            "connection_pools": len(self.connection_pools),
        }

        if slow_queries:
            performance_summary["slowest_query_time_ms"] = max(
                q.execution_time_ms for q in slow_queries
            )

        report = DatabasePerformanceReport(
            report_id=report_id,
            timestamp=timestamp,
            query_metrics=list(self.query_metrics),
            connection_metrics=list(self.connection_metrics),
            cache_metrics=list(self.cache_metrics),
            slow_queries=slow_queries,
            optimization_recommendations=recommendations,
            performance_summary=performance_summary,
        )

        # Save report
        self._save_database_report(report)

        return report

    def _save_database_report(self, report: DatabasePerformanceReport) -> None:
        """Save database performance report to file"""
        output_path = config.get_output_path(MetricType.DATABASE, f"{report.report_id}.json")

        # Convert to serializable format
        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "performance_summary": report.performance_summary,
            "slow_queries_count": len(report.slow_queries),
            "optimization_recommendations": report.optimization_recommendations,
            "cache_analysis": self.analyze_cache_performance(),
            "metrics_counts": {
                "queries": len(report.query_metrics),
                "connections": len(report.connection_metrics),
                "cache_operations": len(report.cache_metrics),
            },
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"Database performance report saved: {output_path}")


# Global database profiler instance
db_profiler = DatabaseProfiler()


# Decorators for automatic profiling
def profile_query(database_type: str = "sqlite"):
    """Decorator to profile database queries"""

    def decorator(func):
        _database_type = database_type

        def wrapper(*args, **kwargs):
            # Extract query from arguments (assuming first arg or 'query' kwarg)
            query = ""
            if args and isinstance(args[0], str):
                query = args[0]
            elif "query" in kwargs:
                query = kwargs["query"]

            if query:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                except Exception:
                    raise
                finally:
                    execution_time = (time.time() - start_time) * 1000

                    if execution_time > 100:  # Log queries slower than 100ms
                        print(f"Query executed in {execution_time:.0f}ms: {query[:100]}...")

                return result
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def profile_cache(operation: str = "get"):
    """Decorator to profile cache operations"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract cache key from arguments
            cache_key = ""
            if args and isinstance(args[0], str):
                cache_key = args[0]
            elif "key" in kwargs:
                cache_key = kwargs["key"]

            if cache_key:
                db_profiler.profile_cache_operation(operation, cache_key)

            return func(*args, **kwargs)

        return wrapper

    return decorator
