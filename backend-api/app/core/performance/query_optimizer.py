"""
Advanced Database Query Optimizer for Chimera AI System

Comprehensive query optimization with:
- Automatic index recommendations
- Query rewriting suggestions
- N+1 query resolution strategies
- Connection pool optimization
- Cache strategy recommendations
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import db_manager
from app.core.performance.db_profiler import db_profiler

logger = logging.getLogger(__name__)


@dataclass
class IndexRecommendation:
    """Database index recommendation."""

    table_name: str
    columns: list[str]
    index_type: str = "btree"  # btree, gin, gist, hash
    reason: str = ""
    estimated_improvement: str = ""  # LOW, MEDIUM, HIGH
    query_patterns: list[str] = field(default_factory=list)
    estimated_size_mb: float | None = None


@dataclass
class QueryOptimizationSuggestion:
    """Query optimization suggestion."""

    query_hash: str
    original_query: str
    suggested_query: str
    optimization_type: str
    description: str
    estimated_improvement: str  # LOW, MEDIUM, HIGH
    confidence: float  # 0.0 to 1.0


class DatabaseQueryOptimizer:
    """
    Advanced database query optimizer with intelligent recommendations.

    Features:
    - Automatic index analysis and recommendations
    - Query rewriting for performance
    - N+1 query resolution patterns
    - Connection pool optimization
    - LLM provider-specific optimizations
    """

    def __init__(self):
        self.index_recommendations: list[IndexRecommendation] = []
        self.query_optimizations: list[QueryOptimizationSuggestion] = []

        # Common patterns for different database systems
        self.db_patterns = {
            "postgresql": {
                "json_operations": r"JSON_EXTRACT|->|->>"
                # Add more PostgreSQL-specific patterns
            },
            "sqlite": {"like_operations": r'LIKE\s+[\'"][%_]', "json_operations": r"JSON_EXTRACT"},
        }

    async def analyze_and_optimize(self) -> dict[str, Any]:
        """
        Perform comprehensive database optimization analysis.

        Returns:
            Dictionary with optimization recommendations and metrics
        """
        logger.info("Starting comprehensive database optimization analysis")

        # Get performance data from profiler
        performance_report = await db_profiler.get_performance_report()

        # Analyze indexes
        await self._analyze_indexes()

        # Analyze slow queries
        await self._analyze_slow_queries(performance_report["slowest_queries"])

        # Analyze N+1 patterns
        n_plus_one_analysis = await self._analyze_n_plus_one_patterns(
            performance_report["n_plus_one_patterns"]
        )

        # Analyze connection pool
        pool_analysis = await self._analyze_connection_pools(performance_report["connection_pools"])

        # Generate LLM-specific optimizations
        llm_optimizations = await self._generate_llm_optimizations()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_summary": {
                "total_recommendations": len(self.index_recommendations)
                + len(self.query_optimizations),
                "high_priority_items": len(
                    [r for r in self.index_recommendations if r.estimated_improvement == "HIGH"]
                )
                + len([q for q in self.query_optimizations if q.estimated_improvement == "HIGH"]),
                "estimated_performance_gain": self._calculate_overall_improvement(),
            },
            "index_recommendations": [
                self._index_to_dict(idx) for idx in self.index_recommendations
            ],
            "query_optimizations": [
                self._query_opt_to_dict(opt) for opt in self.query_optimizations
            ],
            "n_plus_one_resolutions": n_plus_one_analysis,
            "connection_pool_optimizations": pool_analysis,
            "llm_specific_optimizations": llm_optimizations,
            "implementation_plan": await self._generate_implementation_plan(),
        }

    async def _analyze_indexes(self) -> dict[str, Any]:
        """Analyze current indexes and recommend optimizations."""
        try:
            async with db_manager.session() as session:
                # Get database schema information
                inspect(session.bind)

                # For each table, analyze query patterns and suggest indexes
                tables = await self._get_table_list(session)

                for table in tables:
                    await self._analyze_table_indexes(session, table)

                return {
                    "analyzed_tables": len(tables),
                    "new_recommendations": len(self.index_recommendations),
                }

        except Exception as e:
            logger.error(f"Index analysis failed: {e}")
            return {"error": str(e)}

    async def _get_table_list(self, session: AsyncSession) -> list[str]:
        """Get list of tables in the database."""
        try:
            # This is database-specific; adapt as needed
            result = await session.execute(
                text(
                    """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
                )
            )
            return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Could not get table list: {e}")
            return ["llm_models", "evasion_tasks", "jailbreak_datasets", "jailbreak_prompts"]

    async def _analyze_table_indexes(self, session: AsyncSession, table_name: str):
        """Analyze indexes needed for a specific table."""
        # Get query patterns that use this table
        table_queries = [
            (hash_, metrics)
            for hash_, metrics in db_profiler.query_metrics.items()
            if table_name.upper() in metrics.query_sql.upper()
        ]

        if not table_queries:
            return

        # Analyze WHERE clause patterns
        for _query_hash, metrics in table_queries:
            where_columns = self._extract_where_columns(metrics.query_sql)
            order_by_columns = self._extract_order_by_columns(metrics.query_sql)
            self._extract_join_columns(metrics.query_sql)

            # Recommend indexes for frequently used WHERE columns
            if where_columns and metrics.execution_count > 10:
                self.index_recommendations.append(
                    IndexRecommendation(
                        table_name=table_name,
                        columns=where_columns,
                        reason=f"Frequent WHERE clause usage ({metrics.execution_count} executions)",
                        estimated_improvement="HIGH" if metrics.avg_time_ms > 100 else "MEDIUM",
                        query_patterns=[metrics.query_sql[:100]],
                    )
                )

            # Recommend indexes for ORDER BY columns
            if order_by_columns and metrics.avg_time_ms > 50:
                self.index_recommendations.append(
                    IndexRecommendation(
                        table_name=table_name,
                        columns=order_by_columns,
                        reason="ORDER BY performance optimization",
                        estimated_improvement="MEDIUM",
                        query_patterns=[metrics.query_sql[:100]],
                    )
                )

            # Recommend covering indexes for frequent SELECT patterns
            if metrics.execution_count > 50:
                select_columns = self._extract_select_columns(metrics.query_sql)
                if select_columns and where_columns:
                    covering_columns = where_columns + [
                        col for col in select_columns if col not in where_columns
                    ]
                    self.index_recommendations.append(
                        IndexRecommendation(
                            table_name=table_name,
                            columns=covering_columns[:5],  # Limit to 5 columns
                            reason="Covering index for frequent query pattern",
                            estimated_improvement="HIGH",
                            query_patterns=[metrics.query_sql[:100]],
                        )
                    )

    def _extract_where_columns(self, query: str) -> list[str]:
        """Extract column names from WHERE clauses."""
        columns = []
        query_upper = query.upper()

        # Simple regex patterns for common WHERE patterns
        patterns = [r"WHERE\s+(\w+)\s*[=<>!]", r"AND\s+(\w+)\s*[=<>!]", r"OR\s+(\w+)\s*[=<>!]"]

        for pattern in patterns:
            matches = re.findall(pattern, query_upper)
            columns.extend(matches)

        return list(set(columns))  # Remove duplicates

    def _extract_order_by_columns(self, query: str) -> list[str]:
        """Extract column names from ORDER BY clauses."""
        query_upper = query.upper()
        match = re.search(r"ORDER\s+BY\s+([\w\s,]+)", query_upper)

        if not match:
            return []

        order_clause = match.group(1)
        # Extract column names (before ASC/DESC)
        columns = []
        for part in order_clause.split(","):
            col_match = re.match(r"\s*(\w+)", part.strip())
            if col_match:
                columns.append(col_match.group(1))

        return columns

    def _extract_join_columns(self, query: str) -> list[str]:
        """Extract column names from JOIN conditions."""
        query_upper = query.upper()
        join_patterns = [
            r"JOIN\s+\w+\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)",
            r"LEFT\s+JOIN\s+\w+\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)",
            r"INNER\s+JOIN\s+\w+\s+ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)",
        ]

        columns = []
        for pattern in join_patterns:
            matches = re.findall(pattern, query_upper)
            for match in matches:
                columns.extend(match)

        return list(set(columns))

    def _extract_select_columns(self, query: str) -> list[str]:
        """Extract column names from SELECT clauses."""
        query_upper = query.upper()
        match = re.search(r"SELECT\s+(.*?)\s+FROM", query_upper)

        if not match:
            return []

        select_clause = match.group(1)
        if "*" in select_clause:
            return []  # Can't optimize SELECT *

        columns = []
        for part in select_clause.split(","):
            # Handle aliases and functions
            col_match = re.search(r"(\w+)(?:\s+AS\s+\w+)?", part.strip())
            if col_match:
                columns.append(col_match.group(1))

        return columns

    async def _analyze_slow_queries(
        self, slow_queries: list[dict]
    ) -> list[QueryOptimizationSuggestion]:
        """Analyze slow queries and suggest optimizations."""
        suggestions = []

        for query_data in slow_queries:
            query_sql = query_data.get("sql", "")
            avg_time = query_data.get("avg_time_ms", 0)

            # Generate optimization suggestions
            query_suggestions = self._generate_query_optimizations(query_sql, avg_time)
            suggestions.extend(query_suggestions)

        self.query_optimizations.extend(suggestions)
        return suggestions

    def _generate_query_optimizations(
        self, query: str, avg_time_ms: float
    ) -> list[QueryOptimizationSuggestion]:
        """Generate specific optimization suggestions for a query."""
        suggestions = []
        query_upper = query.upper()
        query_hash = db_profiler._get_query_hash(query)

        # Suggest LIMIT for unbounded queries
        if "SELECT" in query_upper and "LIMIT" not in query_upper and avg_time_ms > 100:
            suggested_query = query.rstrip(";") + " LIMIT 100"
            suggestions.append(
                QueryOptimizationSuggestion(
                    query_hash=query_hash,
                    original_query=query,
                    suggested_query=suggested_query,
                    optimization_type="add_limit",
                    description="Add LIMIT clause to prevent large result sets",
                    estimated_improvement="HIGH",
                    confidence=0.9,
                )
            )

        # Suggest EXISTS instead of IN for subqueries
        if "IN (SELECT" in query_upper:
            suggested_query = re.sub(
                r"IN \(SELECT", "EXISTS (SELECT 1 FROM", query, flags=re.IGNORECASE
            )
            suggestions.append(
                QueryOptimizationSuggestion(
                    query_hash=query_hash,
                    original_query=query,
                    suggested_query=suggested_query,
                    optimization_type="exists_vs_in",
                    description="Use EXISTS instead of IN with subquery for better performance",
                    estimated_improvement="MEDIUM",
                    confidence=0.8,
                )
            )

        # Suggest JOIN instead of WHERE with multiple tables
        if query.count("FROM") == 1 and query.count(",") > 0 and "WHERE" in query_upper:
            suggestions.append(
                QueryOptimizationSuggestion(
                    query_hash=query_hash,
                    original_query=query,
                    suggested_query="-- Consider rewriting with explicit JOINs",
                    optimization_type="explicit_joins",
                    description="Replace implicit JOINs with explicit JOIN syntax",
                    estimated_improvement="MEDIUM",
                    confidence=0.7,
                )
            )

        return suggestions

    async def _analyze_n_plus_one_patterns(self, n_plus_one_patterns: list[dict]) -> dict[str, Any]:
        """Analyze N+1 patterns and suggest resolution strategies."""
        resolutions = []

        for pattern in n_plus_one_patterns:
            query = pattern.get("child_query", "")
            occurrences = pattern.get("occurrences", 0)

            resolution_strategies = self._generate_n_plus_one_resolutions(query, occurrences)
            resolutions.extend(resolution_strategies)

        return {
            "total_patterns": len(n_plus_one_patterns),
            "resolutions": resolutions,
            "estimated_savings_ms": sum(r.get("estimated_savings_ms", 0) for r in resolutions),
        }

    def _generate_n_plus_one_resolutions(
        self, query: str, occurrences: int
    ) -> list[dict[str, Any]]:
        """Generate resolution strategies for N+1 query patterns."""
        strategies = []

        # Strategy 1: Batch loading with IN clause
        if "WHERE" in query.upper() and "=" in query:
            strategies.append(
                {
                    "strategy": "batch_loading",
                    "description": f"Replace {occurrences} individual queries with single batch query using IN clause",
                    "implementation": "Use relationship.load() or selectinload() in SQLAlchemy",
                    "estimated_savings_ms": occurrences * 5,  # Rough estimate
                    "confidence": 0.9,
                }
            )

        # Strategy 2: Eager loading with JOIN
        strategies.append(
            {
                "strategy": "eager_loading",
                "description": "Use eager loading to fetch related data in parent query",
                "implementation": "Use joinedload() or selectinload() in SQLAlchemy ORM",
                "estimated_savings_ms": occurrences * 3,
                "confidence": 0.8,
            }
        )

        # Strategy 3: Caching
        if occurrences > 20:
            strategies.append(
                {
                    "strategy": "caching",
                    "description": "Implement query result caching for frequently accessed data",
                    "implementation": "Add Redis caching layer with appropriate TTL",
                    "estimated_savings_ms": occurrences * 8,  # Higher savings for cache hits
                    "confidence": 0.7,
                }
            )

        return strategies

    async def _analyze_connection_pools(self, pool_data: list[dict]) -> dict[str, Any]:
        """Analyze connection pool usage and suggest optimizations."""
        recommendations = []

        for pool in pool_data:
            pool_name = pool.get("name", "unknown")
            pool.get("checked_out", 0)
            peak_usage = pool.get("peak_usage", 0)
            size = pool.get("size", 0)

            # Check for pool exhaustion
            if peak_usage >= size * 0.9:
                recommendations.append(
                    {
                        "type": "increase_pool_size",
                        "pool": pool_name,
                        "current_size": size,
                        "suggested_size": int(size * 1.5),
                        "reason": f"Peak usage ({peak_usage}) near pool limit ({size})",
                    }
                )

            # Check for underutilization
            elif peak_usage < size * 0.3 and size > 5:
                recommendations.append(
                    {
                        "type": "decrease_pool_size",
                        "pool": pool_name,
                        "current_size": size,
                        "suggested_size": max(3, int(size * 0.7)),
                        "reason": f"Low peak usage ({peak_usage}) suggests overprovisioning",
                    }
                )

        return {"total_pools_analyzed": len(pool_data), "recommendations": recommendations}

    async def _generate_llm_optimizations(self) -> list[dict[str, Any]]:
        """Generate LLM provider-specific optimizations."""
        optimizations = []

        # Get LLM-specific metrics
        llm_metrics = await db_profiler.get_llm_provider_metrics()

        # Analyze provider configuration queries
        provider_queries = llm_metrics.get("provider_queries", {})

        if "model_management" in provider_queries:
            model_queries = provider_queries["model_management"]
            total_executions = sum(q["execution_count"] for q in model_queries)

            if total_executions > 1000:
                optimizations.append(
                    {
                        "type": "cache_llm_configurations",
                        "description": "Cache LLM provider configurations to reduce database lookups",
                        "implementation": "Implement Redis cache with 5-minute TTL for provider configs",
                        "estimated_impact": f"Reduce {total_executions} DB queries to ~{total_executions//50} per hour",
                        "priority": "HIGH",
                    }
                )

        # Analyze jailbreak research patterns
        if "jailbreak_research" in provider_queries:
            jailbreak_queries = provider_queries["jailbreak_research"]

            if len(jailbreak_queries) > 100:
                optimizations.append(
                    {
                        "type": "batch_jailbreak_operations",
                        "description": "Batch jailbreak experiment database operations",
                        "implementation": "Use bulk insert/update operations for experiment results",
                        "estimated_impact": "Reduce individual operations by 70%",
                        "priority": "MEDIUM",
                    }
                )

        return optimizations

    async def _generate_implementation_plan(self) -> dict[str, Any]:
        """Generate prioritized implementation plan for optimizations."""
        high_priority = []
        medium_priority = []
        low_priority = []

        # Categorize index recommendations
        for idx in self.index_recommendations:
            item = {
                "type": "index",
                "description": f"Create {idx.index_type} index on {idx.table_name}({', '.join(idx.columns)})",
                "reason": idx.reason,
                "sql": f"CREATE INDEX IF NOT EXISTS idx_{idx.table_name}_{'_'.join(idx.columns)} ON {idx.table_name} ({', '.join(idx.columns)});",
            }

            if idx.estimated_improvement == "HIGH":
                high_priority.append(item)
            elif idx.estimated_improvement == "MEDIUM":
                medium_priority.append(item)
            else:
                low_priority.append(item)

        # Categorize query optimizations
        for opt in self.query_optimizations:
            item = {
                "type": "query_optimization",
                "description": opt.description,
                "optimization_type": opt.optimization_type,
                "confidence": opt.confidence,
            }

            if opt.estimated_improvement == "HIGH":
                high_priority.append(item)
            elif opt.estimated_improvement == "MEDIUM":
                medium_priority.append(item)
            else:
                low_priority.append(item)

        return {
            "phase_1_critical": high_priority,
            "phase_2_important": medium_priority,
            "phase_3_nice_to_have": low_priority,
            "estimated_timeline": {
                "phase_1": "1-2 days",
                "phase_2": "3-5 days",
                "phase_3": "1-2 weeks",
            },
        }

    def _calculate_overall_improvement(self) -> str:
        """Calculate estimated overall performance improvement."""
        high_impact_count = len(
            [r for r in self.index_recommendations if r.estimated_improvement == "HIGH"]
        )
        high_impact_count += len(
            [q for q in self.query_optimizations if q.estimated_improvement == "HIGH"]
        )

        if high_impact_count >= 5:
            return "30-50% performance improvement expected"
        elif high_impact_count >= 3:
            return "20-30% performance improvement expected"
        elif high_impact_count >= 1:
            return "10-20% performance improvement expected"
        else:
            return "5-10% performance improvement expected"

    def _index_to_dict(self, idx: IndexRecommendation) -> dict[str, Any]:
        """Convert IndexRecommendation to dictionary."""
        return {
            "table_name": idx.table_name,
            "columns": idx.columns,
            "index_type": idx.index_type,
            "reason": idx.reason,
            "estimated_improvement": idx.estimated_improvement,
            "query_patterns": idx.query_patterns,
        }

    def _query_opt_to_dict(self, opt: QueryOptimizationSuggestion) -> dict[str, Any]:
        """Convert QueryOptimizationSuggestion to dictionary."""
        return {
            "query_hash": opt.query_hash,
            "original_query": opt.original_query[:200],  # Truncate for display
            "suggested_query": opt.suggested_query[:200],
            "optimization_type": opt.optimization_type,
            "description": opt.description,
            "estimated_improvement": opt.estimated_improvement,
            "confidence": opt.confidence,
        }


# Global optimizer instance
db_optimizer = DatabaseQueryOptimizer()
