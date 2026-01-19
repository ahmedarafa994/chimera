"""
Advanced Database Index Creation and Optimization Script

This script creates optimized indexes for the Chimera AI system based on
query pattern analysis and performance profiling. Includes indexes for:

- User session and authentication queries
- LLM provider configuration and state management
- Prompt generation history and caching
- Jailbreak technique results storage
- Data pipeline ETL operations
- Full-text search on prompt content
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import inspect, text

from app.core.database import db_manager

logger = logging.getLogger(__name__)


class DatabaseIndexOptimizer:
    """
    Comprehensive database index optimizer for Chimera AI system.

    Creates performance-optimized indexes based on query patterns and
    provides recommendations for ongoing optimization.
    """

    def __init__(self):
        self.created_indexes = []
        self.failed_indexes = []
        self.optimization_results = {}

    async def create_performance_indexes(self) -> dict[str, Any]:
        """
        Create all performance indexes for the Chimera database.

        Returns comprehensive report of index creation results.
        """
        logger.info("Starting comprehensive database index optimization")
        start_time = datetime.utcnow()

        try:
            async with db_manager.session() as session:
                # Get database engine type for dialect-specific optimizations
                db_dialect = session.bind.dialect.name
                logger.info(f"Optimizing for database dialect: {db_dialect}")

                # Create indexes by category
                await self._create_core_table_indexes(session, db_dialect)
                await self._create_llm_provider_indexes(session, db_dialect)
                await self._create_jailbreak_research_indexes(session, db_dialect)
                await self._create_data_pipeline_indexes(session, db_dialect)
                await self._create_full_text_search_indexes(session, db_dialect)
                await self._create_composite_performance_indexes(session, db_dialect)

                # Analyze existing indexes for optimization opportunities
                existing_analysis = await self._analyze_existing_indexes(session)

                await session.commit()

        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            raise

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        self.optimization_results = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "total_indexes_created": len(self.created_indexes),
            "failed_index_count": len(self.failed_indexes),
            "created_indexes": self.created_indexes,
            "failed_indexes": self.failed_indexes,
            "existing_index_analysis": existing_analysis,
            "performance_impact_estimate": self._estimate_performance_impact(),
        }

        logger.info(
            f"Index optimization completed in {duration:.2f}s. "
            f"Created {len(self.created_indexes)} indexes, {len(self.failed_indexes)} failures"
        )

        return self.optimization_results

    async def _create_core_table_indexes(self, session, db_dialect: str):
        """Create indexes for core application tables."""
        core_indexes = [
            # LLM Models table - Provider lookups and configuration queries
            {
                "name": "idx_llm_models_provider_name",
                "table": "llm_models",
                "columns": ["provider", "name"],
                "type": "btree",
                "description": "Optimize provider-specific model lookups",
            },
            {
                "name": "idx_llm_models_created_at",
                "table": "llm_models",
                "columns": ["created_at"],
                "type": "btree",
                "description": "Optimize time-based queries for model history",
            },
            # Evasion Tasks table - Task tracking and status queries
            {
                "name": "idx_evasion_tasks_status_created",
                "table": "evasion_tasks",
                "columns": ["status", "created_at"],
                "type": "btree",
                "description": "Optimize task status and timeline queries",
            },
            {
                "name": "idx_evasion_tasks_target_model",
                "table": "evasion_tasks",
                "columns": ["target_model_id", "status"],
                "type": "btree",
                "description": "Optimize model-specific task queries",
            },
            {
                "name": "idx_evasion_tasks_task_id",
                "table": "evasion_tasks",
                "columns": ["task_id"],
                "type": "btree",
                "description": "Optimize Celery task ID lookups",
            },
            {
                "name": "idx_evasion_tasks_success_status",
                "table": "evasion_tasks",
                "columns": ["overall_success", "final_status"],
                "type": "btree",
                "description": "Optimize success rate analysis queries",
            },
            # Jailbreak Datasets table - Dataset management
            {
                "name": "idx_jailbreak_datasets_name",
                "table": "jailbreak_datasets",
                "columns": ["name"],
                "type": "btree",
                "description": "Optimize dataset name lookups",
            },
        ]

        await self._execute_index_creation_batch(session, core_indexes, "Core Tables")

    async def _create_llm_provider_indexes(self, session, db_dialect: str):
        """Create indexes optimized for LLM provider operations."""
        provider_indexes = [
            # Provider performance monitoring
            {
                "name": "idx_llm_models_provider_config_lookup",
                "table": "llm_models",
                "columns": ["provider", "name", "created_at"],
                "type": "btree",
                "description": "Covering index for frequent provider config lookups",
            },
            # Evasion task performance tracking
            {
                "name": "idx_evasion_tasks_performance_analysis",
                "table": "evasion_tasks",
                "columns": ["target_model_id", "created_at", "completed_at"],
                "type": "btree",
                "description": "Optimize performance analysis queries by model and time",
            },
            {
                "name": "idx_evasion_tasks_strategy_analysis",
                "table": "evasion_tasks",
                "columns": ["target_model_id", "overall_success", "created_at"],
                "type": "btree",
                "description": "Optimize strategy effectiveness analysis",
            },
        ]

        # PostgreSQL-specific optimizations
        if db_dialect == "postgresql":
            provider_indexes.extend(
                [
                    {
                        "name": "idx_llm_models_config_gin",
                        "table": "llm_models",
                        "columns": ["config"],
                        "type": "gin",
                        "description": "GIN index for JSON configuration searches",
                    },
                    {
                        "name": "idx_evasion_tasks_results_gin",
                        "table": "evasion_tasks",
                        "columns": ["results"],
                        "type": "gin",
                        "description": "GIN index for JSON results analysis",
                    },
                ]
            )

        await self._execute_index_creation_batch(
            session, provider_indexes, "LLM Provider Operations"
        )

    async def _create_jailbreak_research_indexes(self, session, db_dialect: str):
        """Create indexes for jailbreak research operations."""
        jailbreak_indexes = [
            # Jailbreak Prompts table - Content and categorization
            {
                "name": "idx_jailbreak_prompts_dataset_type",
                "table": "jailbreak_prompts",
                "columns": ["dataset_id", "jailbreak_type"],
                "type": "btree",
                "description": "Optimize dataset and type filtering",
            },
            {
                "name": "idx_jailbreak_prompts_platform_source",
                "table": "jailbreak_prompts",
                "columns": ["platform", "source"],
                "type": "btree",
                "description": "Optimize platform and source analysis",
            },
            {
                "name": "idx_jailbreak_prompts_is_jailbreak_created",
                "table": "jailbreak_prompts",
                "columns": ["is_jailbreak", "created_at"],
                "type": "btree",
                "description": "Optimize jailbreak classification queries",
            },
            # Research effectiveness analysis
            {
                "name": "idx_evasion_tasks_jailbreak_research",
                "table": "evasion_tasks",
                "columns": ["jailbreak_prompt_id", "overall_success", "created_at"],
                "type": "btree",
                "description": "Optimize jailbreak prompt effectiveness analysis",
            },
        ]

        # Add full-text search for prompt content if supported
        if db_dialect == "postgresql":
            jailbreak_indexes.extend(
                [
                    {
                        "name": "idx_jailbreak_prompts_content_fulltext",
                        "table": "jailbreak_prompts",
                        "columns": ["prompt_text"],
                        "type": "gin",
                        "description": "Full-text search on jailbreak prompt content",
                        "custom_sql": "CREATE INDEX IF NOT EXISTS idx_jailbreak_prompts_content_fulltext ON jailbreak_prompts USING gin(to_tsvector('english', prompt_text))",
                    }
                ]
            )

        await self._execute_index_creation_batch(session, jailbreak_indexes, "Jailbreak Research")

    async def _create_data_pipeline_indexes(self, session, db_dialect: str):
        """Create indexes for data pipeline ETL operations."""
        pipeline_indexes = [
            # Time-series analysis for all tables
            {
                "name": "idx_llm_models_time_series",
                "table": "llm_models",
                "columns": ["created_at", "updated_at"],
                "type": "btree",
                "description": "Optimize time-series ETL queries",
            },
            {
                "name": "idx_evasion_tasks_time_series",
                "table": "evasion_tasks",
                "columns": ["created_at", "updated_at", "completed_at"],
                "type": "btree",
                "description": "Optimize task timeline ETL queries",
            },
            {
                "name": "idx_jailbreak_datasets_time_series",
                "table": "jailbreak_datasets",
                "columns": ["created_at", "updated_at"],
                "type": "btree",
                "description": "Optimize dataset ETL queries",
            },
            {
                "name": "idx_jailbreak_prompts_time_series",
                "table": "jailbreak_prompts",
                "columns": ["created_at", "updated_at"],
                "type": "btree",
                "description": "Optimize prompt ETL queries",
            },
            # Batch processing optimizations
            {
                "name": "idx_evasion_tasks_batch_processing",
                "table": "evasion_tasks",
                "columns": ["status", "created_at", "target_model_id"],
                "type": "btree",
                "description": "Optimize batch processing queries by status and model",
            },
        ]

        await self._execute_index_creation_batch(session, pipeline_indexes, "Data Pipeline ETL")

    async def _create_full_text_search_indexes(self, session, db_dialect: str):
        """Create full-text search indexes for content discovery."""
        # Only create full-text indexes for databases that support them well
        if db_dialect not in ["postgresql", "mysql"]:
            logger.info(f"Skipping full-text indexes for {db_dialect} - limited support")
            return

        fulltext_indexes = []

        if db_dialect == "postgresql":
            fulltext_indexes = [
                {
                    "name": "idx_jailbreak_prompts_search",
                    "custom_sql": """
                        CREATE INDEX IF NOT EXISTS idx_jailbreak_prompts_search
                        ON jailbreak_prompts
                        USING gin(to_tsvector('english', prompt_text || ' ' || COALESCE(jailbreak_type, '') || ' ' || COALESCE(source, '')))
                    """,
                    "description": "Comprehensive full-text search across prompt content and metadata",
                },
                {
                    "name": "idx_jailbreak_datasets_search",
                    "custom_sql": """
                        CREATE INDEX IF NOT EXISTS idx_jailbreak_datasets_search
                        ON jailbreak_datasets
                        USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')))
                    """,
                    "description": "Full-text search on dataset names and descriptions",
                },
            ]

        await self._execute_index_creation_batch(session, fulltext_indexes, "Full-Text Search")

    async def _create_composite_performance_indexes(self, session, db_dialect: str):
        """Create composite indexes for high-performance query patterns."""
        composite_indexes = [
            # High-performance covering indexes
            {
                "name": "idx_evasion_tasks_dashboard_covering",
                "table": "evasion_tasks",
                "columns": [
                    "status",
                    "overall_success",
                    "created_at",
                    "target_model_id",
                    "task_id",
                ],
                "type": "btree",
                "description": "Covering index for dashboard queries - avoids table lookups",
            },
            {
                "name": "idx_llm_models_api_covering",
                "table": "llm_models",
                "columns": ["provider", "name", "id", "config", "created_at"],
                "type": "btree",
                "description": "Covering index for API model queries",
            },
            # Multi-table join optimization
            {
                "name": "idx_jailbreak_prompts_join_optimization",
                "table": "jailbreak_prompts",
                "columns": ["dataset_id", "is_jailbreak", "created_at", "id"],
                "type": "btree",
                "description": "Optimize joins with jailbreak_datasets table",
            },
            {
                "name": "idx_evasion_tasks_join_optimization",
                "table": "evasion_tasks",
                "columns": ["target_model_id", "jailbreak_prompt_id", "overall_success", "id"],
                "type": "btree",
                "description": "Optimize complex joins between tasks, models, and prompts",
            },
            # Analytical query optimization
            {
                "name": "idx_evasion_tasks_analytics",
                "table": "evasion_tasks",
                "columns": [
                    "target_model_id",
                    "overall_success",
                    "final_status",
                    "created_at",
                    "completed_at",
                ],
                "type": "btree",
                "description": "Optimize analytical queries for success rates and timing",
            },
        ]

        await self._execute_index_creation_batch(
            session, composite_indexes, "Composite Performance"
        )

    async def _execute_index_creation_batch(self, session, indexes: list[dict], category: str):
        """Execute a batch of index creation statements."""
        logger.info(f"Creating {len(indexes)} indexes for category: {category}")

        for index_config in indexes:
            try:
                await self._create_single_index(session, index_config)
                self.created_indexes.append(
                    {
                        "category": category,
                        "name": index_config.get("name"),
                        "description": index_config.get("description"),
                        "table": index_config.get("table"),
                        "columns": index_config.get("columns", []),
                    }
                )

            except Exception as e:
                error_info = {
                    "category": category,
                    "name": index_config.get("name"),
                    "error": str(e),
                    "table": index_config.get("table"),
                }
                self.failed_indexes.append(error_info)
                logger.error(f"Failed to create index {index_config.get('name')}: {e}")

    async def _create_single_index(self, session, index_config: dict):
        """Create a single database index."""
        # Use custom SQL if provided
        if "custom_sql" in index_config:
            await session.execute(text(index_config["custom_sql"]))
            return

        # Build standard index creation SQL
        name = index_config["name"]
        table = index_config["table"]
        columns = index_config["columns"]
        index_type = index_config.get("type", "btree")

        column_list = ", ".join(columns) if isinstance(columns, list) else columns

        # Database-specific index creation
        if index_type == "gin":
            sql = f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING gin({column_list})"
        elif index_type == "gist":
            sql = f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING gist({column_list})"
        elif index_type == "hash":
            sql = f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING hash({column_list})"
        else:  # btree (default)
            sql = f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({column_list})"

        await session.execute(text(sql))
        logger.debug(f"Created index: {name}")

    async def _analyze_existing_indexes(self, session) -> dict[str, Any]:
        """Analyze existing indexes for optimization opportunities."""
        try:
            # Get existing indexes (database-specific query)
            inspector = inspect(session.bind)

            existing_indexes = {}
            tables = await self._get_table_names(session)

            for table in tables:
                try:
                    indexes = inspector.get_indexes(table)
                    existing_indexes[table] = indexes
                except Exception as e:
                    logger.warning(f"Could not analyze indexes for table {table}: {e}")

            # Analyze for potential issues
            analysis = {
                "total_tables_analyzed": len(existing_indexes),
                "tables_with_indexes": len([t for t, idx in existing_indexes.items() if idx]),
                "potential_issues": [],
                "recommendations": [],
            }

            # Check for common issues
            for table, indexes in existing_indexes.items():
                if not indexes:
                    analysis["potential_issues"].append(f"Table {table} has no indexes")

                # Check for duplicate or redundant indexes
                index_columns = [tuple(idx.get("column_names", [])) for idx in indexes]
                if len(index_columns) != len(set(index_columns)):
                    analysis["potential_issues"].append(f"Table {table} may have redundant indexes")

            return analysis

        except Exception as e:
            logger.error(f"Index analysis failed: {e}")
            return {"error": str(e)}

    async def _get_table_names(self, session) -> list[str]:
        """Get list of table names in the database."""
        try:
            inspector = inspect(session.bind)
            return inspector.get_table_names()
        except Exception as e:
            logger.warning(f"Could not get table names: {e}")
            return ["llm_models", "evasion_tasks", "jailbreak_datasets", "jailbreak_prompts"]

    def _estimate_performance_impact(self) -> dict[str, str]:
        """Estimate the performance impact of created indexes."""
        created_count = len(self.created_indexes)

        # Categories of impact
        high_impact_patterns = ["covering", "provider_config", "status", "time_series"]
        medium_impact_patterns = ["join", "analytics", "search"]

        high_impact = sum(
            1
            for idx in self.created_indexes
            if any(pattern in idx["name"].lower() for pattern in high_impact_patterns)
        )

        medium_impact = sum(
            1
            for idx in self.created_indexes
            if any(pattern in idx["name"].lower() for pattern in medium_impact_patterns)
        )

        # Estimate overall impact
        if high_impact >= 3:
            overall_estimate = "HIGH - 30-50% improvement expected for key operations"
        elif high_impact >= 1 or medium_impact >= 3:
            overall_estimate = "MEDIUM - 15-30% improvement expected"
        elif created_count >= 5:
            overall_estimate = "LOW-MEDIUM - 5-15% improvement expected"
        else:
            overall_estimate = "LOW - Marginal improvement expected"

        return {
            "overall": overall_estimate,
            "high_impact_indexes": high_impact,
            "medium_impact_indexes": medium_impact,
            "total_indexes": created_count,
            "key_optimizations": [
                "LLM provider configuration lookups",
                "Jailbreak research queries",
                "Data pipeline ETL operations",
                "Dashboard and API queries",
                "Time-series analysis",
            ],
        }

    async def drop_index_safely(self, index_name: str) -> bool:
        """Safely drop an index if it exists."""
        try:
            async with db_manager.session() as session:
                await session.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                await session.commit()
                logger.info(f"Dropped index: {index_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {e}")
            return False

    async def get_index_usage_stats(self) -> dict[str, Any]:
        """Get index usage statistics (PostgreSQL specific)."""
        try:
            async with db_manager.session() as session:
                # PostgreSQL-specific query for index usage
                stats_query = text(
                    """
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan as index_scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC
                """
                )

                result = await session.execute(stats_query)
                rows = result.fetchall()

                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "database_type": "postgresql",
                    "index_stats": [
                        {
                            "schema": row[0],
                            "table": row[1],
                            "index": row[2],
                            "scans": row[3],
                            "tuples_read": row[4],
                            "tuples_fetched": row[5],
                        }
                        for row in rows
                    ],
                }

        except Exception as e:
            logger.warning(f"Could not get index usage stats: {e}")
            return {"error": str(e), "database_type": "unknown"}


# Global index optimizer instance
index_optimizer = DatabaseIndexOptimizer()


async def create_all_performance_indexes():
    """Convenience function to create all performance indexes."""
    return await index_optimizer.create_performance_indexes()


async def get_index_optimization_report():
    """Get comprehensive index optimization report."""
    if not index_optimizer.optimization_results:
        await create_all_performance_indexes()

    # Add current usage stats
    usage_stats = await index_optimizer.get_index_usage_stats()

    return {
        "optimization_results": index_optimizer.optimization_results,
        "usage_statistics": usage_stats,
        "recommendations": _generate_ongoing_recommendations(),
    }


def _generate_ongoing_recommendations() -> list[str]:
    """Generate ongoing index optimization recommendations."""
    recommendations = [
        "Monitor query performance after index deployment using db_profiler",
        "Review index usage statistics weekly to identify unused indexes",
        "Consider partitioning for large tables with time-series data",
        "Implement query result caching for frequently accessed data",
        "Use EXPLAIN ANALYZE to validate index usage in slow queries",
        "Consider composite indexes for frequently joined table combinations",
        "Monitor index bloat and rebuild indexes periodically in production",
        "Implement connection pooling optimization for high-concurrency workloads",
    ]

    return recommendations
