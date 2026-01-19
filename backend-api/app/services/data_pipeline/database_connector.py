"""
Database Connectors for Data Pipeline

Provides database abstraction layer for extracting LLM interactions,
transformations, and jailbreak experiments from the Chimera backend database.

Supports:
- PostgreSQL (production)
- SQLite (development)
- Connection pooling
- Automatic reconnection
- Query optimization for batch extraction
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool, QueuePool

from app.core.config import settings
from app.core.logging import logger


class DatabaseConnector(ABC):
    """
    Abstract base class for database connectors.

    Provides common interface for different database backends
    and implements connection pooling and error handling.
    """

    def __init__(self, connection_url: str, pool_size: int = 5, max_overflow: int = 10):
        """
        Initialize database connector.

        Args:
            connection_url: Database connection string
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
        """
        self.connection_url = connection_url
        self._engine: Engine | None = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow

    @property
    def engine(self) -> Engine:
        """Lazy-initialize SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            self._engine = create_engine(
                self.connection_url,
                poolclass=self._get_pool_class(),
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=False,  # Set to True for SQL query logging
            )
            logger.info(f"Created database engine for {self.__class__.__name__}")
        return self._engine

    def _get_pool_class(self):
        """Get appropriate pool class for this database type."""
        # SQLite doesn't support connection pooling
        if "sqlite" in self.connection_url:
            return NullPool
        return QueuePool

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Context manager for database connections.

        Yields:
            Database connection object

        Example:
            with connector.get_connection() as conn:
                result = conn.execute("SELECT * FROM table")
        """
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
            conn.commit()
        except SQLAlchemyError as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            query: SQL query string
            params: Query parameters for parameterized queries

        Returns:
            Query results as pandas DataFrame
        """
        try:
            with self.get_connection() as conn:
                result_df = pd.read_sql_query(
                    sql=text(query),
                    con=conn,
                    params=params or {},
                )
                logger.debug(f"Query returned {len(result_df)} rows")
                return result_df
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test database connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    @abstractmethod
    def extract_llm_interactions(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Extract LLM interactions from database.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with LLM interaction records
        """
        pass

    @abstractmethod
    def extract_transformation_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Extract transformation events from database."""
        pass

    @abstractmethod
    def extract_jailbreak_experiments(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Extract jailbreak experiments from database."""
        pass

    def close(self):
        """Close database connections and dispose of engine."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Database engine disposed")


class PostgreSQLConnector(DatabaseConnector):
    """
    PostgreSQL database connector for production.

    Provides optimized queries for PostgreSQL with proper indexing
    and partitioning support.
    """

    def extract_llm_interactions(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Extract LLM interactions from PostgreSQL.

        Query uses:
        - Parameterized inputs for SQL injection prevention
        - Index-optimized timestamp filtering
        - JSON aggregation for config field
        - CT E for readability

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with LLM interaction records
        """
        query = """
        WITH interaction_data AS (
            SELECT
                -- Primary keys
                id::text as interaction_id,
                session_id::text,
                COALESCE(tenant_id, 'default')::text as tenant_id,

                -- Provider and model
                COALESCE(provider, 'unknown') as provider,
                COALESCE(model, 'unknown') as model,

                -- Content
                COALESCE(prompt, '') as prompt,
                COALESCE(response, '')::text as response,
                COALESCE(system_instruction, '')::text as system_instruction,

                -- Configuration (JSON)
                COALESCE(config, '{}')::text as config,

                -- Metrics
                COALESCE(tokens_prompt, 0) as tokens_prompt,
                COALESCE(tokens_completion, 0) as tokens_completion,
                COALESCE((tokens_prompt + tokens_completion), 0) as tokens_total,
                COALESCE(latency_ms, 0) as latency_ms,

                -- Status
                CASE
                    WHEN error_message IS NULL THEN 'success'
                    WHEN error_message LIKE '%timeout%' THEN 'timeout'
                    ELSE 'error'
                END as status,

                error_message,

                -- Timestamps
                created_at,
                NOW() as ingested_at

            FROM llm_interactions
            WHERE created_at >= :start_time
              AND created_at < :end_time

            -- Use index on created_at
            ORDER BY created_at ASC
        )
        SELECT
            *,
            encode(digest(prompt, 'sha256'), 'hex') as prompt_hash
        FROM interaction_data;
        """

        try:
            df = self.execute_query(
                query,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            # Convert datetime columns
            datetime_cols = ["created_at", "ingested_at"]
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Parse JSON config column
            if "config" in df.columns:
                df["config"] = df["config"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

            logger.info(
                f"Extracted {len(df)} LLM interactions from PostgreSQL "
                f"({start_time} to {end_time})"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to extract LLM interactions: {e}")
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                columns=[
                    "interaction_id",
                    "session_id",
                    "tenant_id",
                    "provider",
                    "model",
                    "prompt",
                    "prompt_hash",
                    "response",
                    "system_instruction",
                    "config",
                    "tokens_prompt",
                    "tokens_completion",
                    "tokens_total",
                    "latency_ms",
                    "status",
                    "error_message",
                    "created_at",
                    "ingested_at",
                ]
            )

    def extract_transformation_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Extract transformation events from PostgreSQL.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with transformation records
        """
        query = """
        SELECT
            id::text as transformation_id,
            interaction_id::text,
            technique_suite,
            technique_name,
            original_prompt,
            transformed_prompt,
            COALESCE(transformation_time_ms, 0) as transformation_time_ms,
            COALESCE(success, true) as success,
            COALESCE(metadata, '{}')::text as metadata,
            created_at,
            NOW() as ingested_at
        FROM transformation_events
        WHERE created_at >= :start_time
          AND created_at < :end_time
        ORDER BY created_at ASC;
        """

        try:
            df = self.execute_query(
                query,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            # Convert datetime columns
            for col in ["created_at", "ingested_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Parse JSON metadata
            if "metadata" in df.columns:
                df["metadata"] = df["metadata"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

            logger.info(
                f"Extracted {len(df)} transformation events from PostgreSQL "
                f"({start_time} to {end_time})"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to extract transformation events: {e}")
            return pd.DataFrame(
                columns=[
                    "transformation_id",
                    "interaction_id",
                    "technique_suite",
                    "technique_name",
                    "original_prompt",
                    "transformed_prompt",
                    "transformation_time_ms",
                    "success",
                    "metadata",
                    "created_at",
                    "ingested_at",
                ]
            )

    def extract_jailbreak_experiments(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Extract jailbreak experiments from PostgreSQL.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with jailbreak experiment records
        """
        query = """
        SELECT
            id::text as experiment_id,
            framework,
            goal,
            attack_method,
            COALESCE(iterations, 0) as iterations,
            COALESCE(success, false) as success,
            final_prompt,
            judge_score,
            target_response,
            COALESCE(metadata, '{}')::text as metadata,
            created_at,
            NOW() as ingested_at
        FROM jailbreak_experiments
        WHERE created_at >= :start_time
          AND created_at < :end_time
        ORDER BY created_at ASC;
        """

        try:
            df = self.execute_query(
                query,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            # Convert datetime columns
            for col in ["created_at", "ingested_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Parse JSON metadata
            if "metadata" in df.columns:
                df["metadata"] = df["metadata"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

            logger.info(
                f"Extracted {len(df)} jailbreak experiments from PostgreSQL "
                f"({start_time} to {end_time})"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to extract jailbreak experiments: {e}")
            return pd.DataFrame(
                columns=[
                    "experiment_id",
                    "framework",
                    "goal",
                    "attack_method",
                    "iterations",
                    "success",
                    "final_prompt",
                    "judge_score",
                    "target_response",
                    "metadata",
                    "created_at",
                    "ingested_at",
                ]
            )


class SQLiteConnector(DatabaseConnector):
    """
    SQLite database connector for development.

    Simplified queries for SQLite without advanced features
    like window functions or CTEs.
    """

    def extract_llm_interactions(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Extract LLM interactions from SQLite.

        Note: This is a simplified query for development.
        Production should use PostgreSQL.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with LLM interaction records
        """
        query = """
        SELECT
            -- Primary keys
            id as interaction_id,
            session_id,
            COALESCE(tenant_id, 'default') as tenant_id,

            -- Provider and model
            COALESCE(provider, 'unknown') as provider,
            COALESCE(model, 'unknown') as model,

            -- Content
            COALESCE(prompt, '') as prompt,
            COALESCE(response, '') as response,
            COALESCE(system_instruction, '') as system_instruction,

            -- Configuration (JSON stored as TEXT)
            COALESCE(config, '{}') as config,

            -- Metrics
            COALESCE(tokens_prompt, 0) as tokens_prompt,
            COALESCE(tokens_completion, 0) as tokens_completion,
            COALESCE(tokens_prompt + tokens_completion, 0) as tokens_total,
            COALESCE(latency_ms, 0) as latency_ms,

            -- Status
            CASE
                WHEN error_message IS NULL THEN 'success'
                WHEN error_message LIKE '%timeout%' THEN 'timeout'
                ELSE 'error'
            END as status,

            error_message,

            -- Timestamps
            created_at,
            datetime('now') as ingested_at

        FROM llm_interactions
        WHERE datetime(created_at) >= datetime(:start_time)
          AND datetime(created_at) < datetime(:end_time)
        ORDER BY created_at ASC;
        """

        try:
            df = self.execute_query(
                query,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            # Add prompt_hash (SQLite doesn't support encode/digest)
            if not df.empty and "prompt" in df.columns:
                import hashlib

                df["prompt_hash"] = df["prompt"].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()
                )

            # Convert datetime columns
            datetime_cols = ["created_at", "ingested_at"]
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Parse JSON config
            if "config" in df.columns:
                df["config"] = df["config"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

            logger.info(
                f"Extracted {len(df)} LLM interactions from SQLite " f"({start_time} to {end_time})"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to extract LLM interactions from SQLite: {e}")
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                columns=[
                    "interaction_id",
                    "session_id",
                    "tenant_id",
                    "provider",
                    "model",
                    "prompt",
                    "prompt_hash",
                    "response",
                    "system_instruction",
                    "config",
                    "tokens_prompt",
                    "tokens_completion",
                    "tokens_total",
                    "latency_ms",
                    "status",
                    "error_message",
                    "created_at",
                    "ingested_at",
                ]
            )

    def extract_transformation_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Extract transformation events from SQLite."""
        query = """
        SELECT
            id as transformation_id,
            interaction_id,
            technique_suite,
            technique_name,
            original_prompt,
            transformed_prompt,
            COALESCE(transformation_time_ms, 0) as transformation_time_ms,
            COALESCE(success, 1) as success,
            COALESCE(metadata, '{}') as metadata,
            created_at,
            datetime('now') as ingested_at
        FROM transformation_events
        WHERE datetime(created_at) >= datetime(:start_time)
          AND datetime(created_at) < datetime(:end_time)
        ORDER BY created_at ASC;
        """

        try:
            df = self.execute_query(
                query,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            # Convert datetime columns
            for col in ["created_at", "ingested_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Parse JSON metadata
            if "metadata" in df.columns:
                df["metadata"] = df["metadata"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

            logger.info(
                f"Extracted {len(df)} transformation events from SQLite "
                f"({start_time} to {end_time})"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to extract transformation events: {e}")
            return pd.DataFrame(
                columns=[
                    "transformation_id",
                    "interaction_id",
                    "technique_suite",
                    "technique_name",
                    "original_prompt",
                    "transformed_prompt",
                    "transformation_time_ms",
                    "success",
                    "metadata",
                    "created_at",
                    "ingested_at",
                ]
            )

    def extract_jailbreak_experiments(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Extract jailbreak experiments from SQLite."""
        query = """
        SELECT
            id as experiment_id,
            framework,
            goal,
            attack_method,
            COALESCE(iterations, 0) as iterations,
            COALESCE(success, 0) as success,
            final_prompt,
            judge_score,
            target_response,
            COALESCE(metadata, '{}') as metadata,
            created_at,
            datetime('now') as ingested_at
        FROM jailbreak_experiments
        WHERE datetime(created_at) >= datetime(:start_time)
          AND datetime(created_at) < datetime(:end_time)
        ORDER BY created_at ASC;
        """

        try:
            df = self.execute_query(
                query,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            # Convert datetime columns
            for col in ["created_at", "ingested_at"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Parse JSON metadata
            if "metadata" in df.columns:
                df["metadata"] = df["metadata"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x else {}
                )

            logger.info(
                f"Extracted {len(df)} jailbreak experiments from SQLite "
                f"({start_time} to {end_time})"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to extract jailbreak experiments: {e}")
            return pd.DataFrame(
                columns=[
                    "experiment_id",
                    "framework",
                    "goal",
                    "attack_method",
                    "iterations",
                    "success",
                    "final_prompt",
                    "judge_score",
                    "target_response",
                    "metadata",
                    "created_at",
                    "ingested_at",
                ]
            )


def create_connector(
    connection_url: str | None = None,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> DatabaseConnector:
    """
    Factory function to create appropriate database connector.

    Args:
        connection_url: Database connection string (defaults to settings.DATABASE_URL)
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections

    Returns:
        DatabaseConnector instance (PostgreSQL or SQLite)

    Example:
        connector = create_connector()
        df = connector.extract_llm_interactions(start, end)
        connector.close()
    """
    url = connection_url or settings.DATABASE_URL

    # Determine connector type based on connection URL
    if "postgresql" in url or "postgres" in url:
        logger.info("Creating PostgreSQL connector")
        return PostgreSQLConnector(url, pool_size, max_overflow)
    elif "sqlite" in url:
        logger.info("Creating SQLite connector")
        return SQLiteConnector(url, pool_size, max_overflow)
    else:
        raise ValueError(
            f"Unsupported database type in connection URL: {url}. "
            f"Supported types: postgresql://, sqlite:///"
        )


# Convenience function for quick extraction
def extract_data(
    start_time: datetime,
    end_time: datetime,
    data_type: str = "llm_interactions",
    connection_url: str | None = None,
) -> pd.DataFrame:
    """
    Quick extraction function for data pipeline jobs.

    Args:
        start_time: Start of extraction window
        end_time: End of extraction window
        data_type: Type of data to extract
            - 'llm_interactions': LLM API calls
            - 'transformations': Transformation events
            - 'jailbreak_experiments': Research experiments
        connection_url: Optional connection string

    Returns:
        DataFrame with extracted data

    Example:
        from datetime import datetime, timedelta
        end = datetime.utcnow()
        start = end - timedelta(hours=1)
        df = extract_data(start, end, "llm_interactions")
    """
    connector = None
    try:
        connector = create_connector(connection_url)

        if data_type == "llm_interactions":
            return connector.extract_llm_interactions(start_time, end_time)
        elif data_type == "transformations":
            return connector.extract_transformation_events(start_time, end_time)
        elif data_type == "jailbreak_experiments":
            return connector.extract_jailbreak_experiments(start_time, end_time)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    finally:
        if connector:
            connector.close()
