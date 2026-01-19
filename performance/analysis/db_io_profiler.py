# Database and I/O Performance Analysis Script
# Analyzes query performance, connection pooling, and data pipeline efficiency

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any

import aioredis
import asyncpg
import matplotlib.pyplot as plt
import pandas as pd
import psutil


class DatabaseProfiler:
    """PostgreSQL database performance profiler"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.query_stats = []
        self.connection_pool = None

    async def initialize_pool(self):
        """Initialize connection pool for testing"""
        self.connection_pool = await asyncpg.create_pool(
            self.connection_string, min_size=5, max_size=20, command_timeout=30
        )

    async def analyze_query_performance(self, queries: list[str]) -> dict[str, Any]:
        """Analyze performance of specific queries"""
        results = []

        for query in queries:
            # Run query multiple times to get average
            times = []
            for _ in range(5):
                start_time = time.time()
                try:
                    async with self.connection_pool.acquire() as conn:
                        await conn.fetch(query)
                    times.append(time.time() - start_time)
                except Exception as e:
                    logging.error(f"Query failed: {e}")
                    times.append(None)

            valid_times = [t for t in times if t is not None]
            if valid_times:
                results.append(
                    {
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "avg_time": sum(valid_times) / len(valid_times),
                        "min_time": min(valid_times),
                        "max_time": max(valid_times),
                        "success_rate": len(valid_times) / len(times),
                    }
                )

        return results

    async def get_slow_queries(self) -> list[dict[str, Any]]:
        """Get slow queries from pg_stat_statements"""
        query = """
        SELECT
            query,
            calls,
            total_time,
            mean_time,
            stddev_time,
            rows
        FROM pg_stat_statements
        WHERE mean_time > 100
        ORDER BY mean_time DESC
        LIMIT 20;
        """

        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Failed to get slow queries: {e}")
            return []

    async def analyze_index_usage(self) -> list[dict[str, Any]]:
        """Analyze index usage efficiency"""
        query = """
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch,
            CASE
                WHEN idx_scan = 0 THEN 0
                ELSE idx_tup_fetch / idx_scan
            END as avg_fetch_per_scan
        FROM pg_stat_user_indexes
        ORDER BY idx_scan DESC;
        """

        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Failed to analyze index usage: {e}")
            return []

    async def check_table_stats(self) -> list[dict[str, Any]]:
        """Check table statistics for optimization opportunities"""
        query = """
        SELECT
            schemaname,
            tablename,
            seq_scan,
            seq_tup_read,
            idx_scan,
            idx_tup_fetch,
            n_tup_ins,
            n_tup_upd,
            n_tup_del,
            n_live_tup,
            n_dead_tup,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY seq_scan DESC;
        """

        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Failed to get table stats: {e}")
            return []

    async def connection_pool_analysis(self) -> dict[str, Any]:
        """Analyze connection pool performance"""
        pool_stats = {
            "pool_size": self.connection_pool.get_size(),
            "pool_max_size": self.connection_pool.get_max_size(),
            "pool_min_size": self.connection_pool.get_min_size(),
            "connections_in_use": self.connection_pool.get_size()
            - len(self.connection_pool._holders),
        }

        # Test connection acquisition times
        acquisition_times = []
        for _ in range(10):
            start_time = time.time()
            async with self.connection_pool.acquire():
                acquisition_times.append(time.time() - start_time)
                await asyncio.sleep(0.1)

        pool_stats.update(
            {
                "avg_acquisition_time": sum(acquisition_times) / len(acquisition_times),
                "max_acquisition_time": max(acquisition_times),
                "min_acquisition_time": min(acquisition_times),
            }
        )

        return pool_stats


class RedisProfiler:
    """Redis cache performance profiler"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(self.redis_url)

    async def benchmark_operations(self) -> dict[str, Any]:
        """Benchmark Redis operations"""
        operations = {"SET": [], "GET": [], "MSET": [], "MGET": [], "PIPELINE": []}

        # Test SET operations
        for i in range(100):
            start_time = time.time()
            await self.redis.set(f"test_key_{i}", f"test_value_{i}")
            operations["SET"].append(time.time() - start_time)

        # Test GET operations
        for i in range(100):
            start_time = time.time()
            await self.redis.get(f"test_key_{i}")
            operations["GET"].append(time.time() - start_time)

        # Test MSET operations
        test_data = {f"mtest_key_{i}": f"mtest_value_{i}" for i in range(10)}
        start_time = time.time()
        await self.redis.mset(test_data)
        operations["MSET"].append(time.time() - start_time)

        # Test MGET operations
        keys = list(test_data.keys())
        start_time = time.time()
        await self.redis.mget(keys)
        operations["MGET"].append(time.time() - start_time)

        # Test Pipeline operations
        pipe = self.redis.pipeline()
        for i in range(10):
            pipe.set(f"pipe_key_{i}", f"pipe_value_{i}")
        start_time = time.time()
        await pipe.execute()
        operations["PIPELINE"].append(time.time() - start_time)

        # Calculate statistics
        stats = {}
        for op, times in operations.items():
            if times:
                stats[op] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_ops": len(times),
                }

        return stats

    async def analyze_memory_usage(self) -> dict[str, Any]:
        """Analyze Redis memory usage"""
        info = await self.redis.info("memory")

        memory_stats = {
            "used_memory": info.get("used_memory", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "used_memory_rss": info.get("used_memory_rss", 0),
            "used_memory_peak": info.get("used_memory_peak", 0),
            "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
            "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
            "maxmemory": info.get("maxmemory", 0),
        }

        return memory_stats

    async def cleanup_test_data(self):
        """Clean up test data"""
        pipe = self.redis.pipeline()
        for i in range(100):
            pipe.delete(f"test_key_{i}")
            pipe.delete(f"mtest_key_{i}")
            pipe.delete(f"pipe_key_{i}")
        await pipe.execute()


class IOProfiler:
    """System I/O performance profiler"""

    def __init__(self):
        self.process = psutil.Process()

    def get_disk_usage(self) -> dict[str, Any]:
        """Get disk usage statistics"""
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        return {
            "total_space": disk_usage.total,
            "used_space": disk_usage.used,
            "free_space": disk_usage.free,
            "usage_percent": (disk_usage.used / disk_usage.total) * 100,
            "read_bytes": disk_io.read_bytes if disk_io else 0,
            "write_bytes": disk_io.write_bytes if disk_io else 0,
            "read_time": disk_io.read_time if disk_io else 0,
            "write_time": disk_io.write_time if disk_io else 0,
        }

    def benchmark_file_io(self, file_size_mb: int = 100) -> dict[str, Any]:
        """Benchmark file I/O operations"""
        import os
        import tempfile

        fd, test_file = tempfile.mkstemp(prefix="io_test_", suffix=".dat")
        os.close(fd)
        data = b"x" * (1024 * 1024)  # 1MB of data

        # Write test
        start_time = time.time()
        with open(test_file, "wb") as f:
            for _ in range(file_size_mb):
                f.write(data)
        write_time = time.time() - start_time

        # Read test
        start_time = time.time()
        with open(test_file, "rb") as f:
            while f.read(1024 * 1024):
                pass
        read_time = time.time() - start_time

        # Cleanup
        try:
            os.remove(test_file)
        except Exception as e:
            print(f"Failed to remove temp file {test_file}: {e}")

        return {
            "file_size_mb": file_size_mb,
            "write_time": write_time,
            "read_time": read_time,
            "write_throughput_mbps": file_size_mb / write_time,
            "read_throughput_mbps": file_size_mb / read_time,
        }


class DataPipelineProfiler:
    """Data pipeline performance profiler"""

    def __init__(self, db_profiler: DatabaseProfiler, redis_profiler: RedisProfiler):
        self.db_profiler = db_profiler
        self.redis_profiler = redis_profiler
        self.pipeline_metrics = []

    async def simulate_etl_process(self, num_records: int = 1000) -> dict[str, Any]:
        """Simulate and profile ETL process"""
        start_time = time.time()

        # Extract phase
        extract_start = time.time()
        num_records = int(num_records)
        # Safe: cast num_records to int before embedding
        extract_query = (
            "SELECT\n"
            "    id,\n"
            "    created_at,\n"
            "    prompt_text,\n"
            "    provider_used,\n"
            "    response_time,\n"
            "    token_count\n"
            f"FROM llm_requests\nLIMIT {num_records}\n"
        )

        try:
            async with self.db_profiler.connection_pool.acquire() as conn:
                rows = await conn.fetch(extract_query)
            extract_time = time.time() - extract_start
            extract_count = len(rows)
        except Exception as e:
            logging.error(f"Extract phase failed: {e}")
            return {"error": "Extract phase failed"}

        # Transform phase (simulate data transformation)
        transform_start = time.time()
        transformed_data = []
        for row in rows:
            transformed_data.append(
                {
                    "id": row["id"],
                    "date": row["created_at"].date(),
                    "hour": row["created_at"].hour,
                    "provider": row["provider_used"],
                    "response_time_bucket": self._bucket_response_time(row["response_time"]),
                    "token_efficiency": row["token_count"] / max(row["response_time"], 1),
                }
            )
        transform_time = time.time() - transform_start

        # Load phase (simulate loading to analytics tables)
        load_start = time.time()
        # Simulate batch insert
        await asyncio.sleep(0.1 * (num_records / 100))  # Simulate load time
        load_time = time.time() - load_start

        total_time = time.time() - start_time

        return {
            "total_records": num_records,
            "extracted_records": extract_count,
            "extract_time": extract_time,
            "transform_time": transform_time,
            "load_time": load_time,
            "total_time": total_time,
            "records_per_second": num_records / total_time,
            "extract_rate": extract_count / extract_time,
            "transform_rate": len(transformed_data) / transform_time,
        }

    def _bucket_response_time(self, response_time: float) -> str:
        """Bucket response times for analysis"""
        if response_time < 1:
            return "fast"
        elif response_time < 5:
            return "medium"
        elif response_time < 10:
            return "slow"
        else:
            return "very_slow"

    async def analyze_data_freshness(self) -> dict[str, Any]:
        """Analyze data freshness and staleness"""
        freshness_query = """
        SELECT
            table_name,
            MAX(created_at) as latest_record,
            MIN(created_at) as earliest_record,
            COUNT(*) as total_records,
            COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as records_last_hour,
            COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 day' THEN 1 END) as records_last_day
        FROM (
            SELECT 'llm_requests' as table_name, created_at FROM llm_requests
            UNION ALL
            SELECT 'transformations' as table_name, created_at FROM transformations
        ) combined
        GROUP BY table_name
        """

        try:
            async with self.db_profiler.connection_pool.acquire() as conn:
                rows = await conn.fetch(freshness_query)
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Failed to analyze data freshness: {e}")
            return []


async def run_comprehensive_analysis():
    """Run comprehensive database and I/O performance analysis"""

    # Configuration
    db_connection = "postgresql://username:password@localhost:5432/chimera"
    redis_url = "redis://localhost:6379/0"

    # Initialize profilers
    db_profiler = DatabaseProfiler(db_connection)
    redis_profiler = RedisProfiler(redis_url)
    io_profiler = IOProfiler()

    try:
        # Initialize connections
        await db_profiler.initialize_pool()
        await redis_profiler.connect()

        # Database analysis
        print("Analyzing database performance...")

        # Common queries to analyze
        test_queries = [
            "SELECT COUNT(*) FROM llm_requests;",
            "SELECT * FROM llm_requests ORDER BY created_at DESC LIMIT 10;",
            "SELECT provider_used, AVG(response_time) FROM llm_requests GROUP BY provider_used;",
            "SELECT DATE(created_at), COUNT(*) FROM llm_requests GROUP BY DATE(created_at);",
        ]

        query_performance = await db_profiler.analyze_query_performance(test_queries)
        slow_queries = await db_profiler.get_slow_queries()
        index_usage = await db_profiler.analyze_index_usage()
        table_stats = await db_profiler.check_table_stats()
        pool_stats = await db_profiler.connection_pool_analysis()

        # Redis analysis
        print("Analyzing Redis performance...")
        redis_ops = await redis_profiler.benchmark_operations()
        redis_memory = await redis_profiler.analyze_memory_usage()

        # I/O analysis
        print("Analyzing I/O performance...")
        disk_stats = io_profiler.get_disk_usage()
        io_benchmark = io_profiler.benchmark_file_io()

        # Data pipeline analysis
        print("Analyzing data pipeline...")
        pipeline_profiler = DataPipelineProfiler(db_profiler, redis_profiler)
        etl_performance = await pipeline_profiler.simulate_etl_process(1000)
        data_freshness = await pipeline_profiler.analyze_data_freshness()

        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "database": {
                "query_performance": query_performance,
                "slow_queries": slow_queries[:5],  # Top 5 slow queries
                "index_usage": index_usage[:10],  # Top 10 indices
                "table_stats": table_stats[:5],  # Top 5 tables
                "connection_pool": pool_stats,
            },
            "redis": {"operations": redis_ops, "memory_usage": redis_memory},
            "io": {"disk_stats": disk_stats, "benchmark": io_benchmark},
            "data_pipeline": {"etl_performance": etl_performance, "data_freshness": data_freshness},
        }

        # Save report
        with open(f"performance/analysis/db_io_analysis_{int(time.time())}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualizations
        generate_performance_charts(report)

        print("Analysis complete! Report saved to performance/analysis/")
        return report

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return None

    finally:
        # Cleanup
        if redis_profiler.redis:
            await redis_profiler.cleanup_test_data()
            await redis_profiler.redis.close()

        if db_profiler.connection_pool:
            await db_profiler.connection_pool.close()


def generate_performance_charts(report: dict[str, Any]):
    """Generate performance visualization charts"""
    try:
        # Create figure with subplots
        _fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Database query performance
        if report["database"]["query_performance"]:
            query_data = pd.DataFrame(report["database"]["query_performance"])
            query_data.plot(x="query", y="avg_time", kind="bar", ax=axes[0, 0])
            axes[0, 0].set_title("Database Query Performance")
            axes[0, 0].set_ylabel("Average Time (seconds)")
            axes[0, 0].tick_params(axis="x", rotation=45)

        # Redis operation performance
        if report["redis"]["operations"]:
            redis_data = pd.DataFrame(
                [
                    {"operation": k, "avg_time": v["avg_time"]}
                    for k, v in report["redis"]["operations"].items()
                ]
            )
            redis_data.plot(x="operation", y="avg_time", kind="bar", ax=axes[0, 1])
            axes[0, 1].set_title("Redis Operation Performance")
            axes[0, 1].set_ylabel("Average Time (seconds)")

        # I/O throughput
        io_data = pd.DataFrame(
            [
                {
                    "operation": "Read",
                    "throughput": report["io"]["benchmark"]["read_throughput_mbps"],
                },
                {
                    "operation": "Write",
                    "throughput": report["io"]["benchmark"]["write_throughput_mbps"],
                },
            ]
        )
        io_data.plot(x="operation", y="throughput", kind="bar", ax=axes[1, 0])
        axes[1, 0].set_title("I/O Throughput")
        axes[1, 0].set_ylabel("Throughput (MB/s)")

        # ETL performance metrics
        etl_times = [
            report["data_pipeline"]["etl_performance"]["extract_time"],
            report["data_pipeline"]["etl_performance"]["transform_time"],
            report["data_pipeline"]["etl_performance"]["load_time"],
        ]
        axes[1, 1].pie(etl_times, labels=["Extract", "Transform", "Load"], autopct="%1.1f%%")
        axes[1, 1].set_title("ETL Phase Distribution")

        plt.tight_layout()
        plt.savefig(
            f"performance/analysis/performance_charts_{int(time.time())}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as e:
        logging.error(f"Failed to generate charts: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_comprehensive_analysis())
