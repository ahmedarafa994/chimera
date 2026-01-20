"""Batch Data Ingestion Service.

Handles extraction of LLM interactions, transformations, and system metrics
from various sources and writes them to Parquet files in the data lake.

Features:
- Incremental loading with watermark tracking
- Schema validation and data cleaning
- Dead letter queue for failed records
- Automatic partitioning by date/hour
- Metadata enrichment
- Database connector integration for actual data extraction
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from app.core.logging import logger


class IngestionConfig(BaseModel):
    """Configuration for batch ingestion."""

    data_lake_path: str = Field(default="/data/chimera-lake", description="Path to data lake root")
    partition_by: list[str] = Field(default=["dt", "hour"], description="Partition columns")
    compression: Literal["snappy", "gzip", "zstd"] = Field(default="snappy")
    max_file_size_mb: int = Field(default=512, ge=100, le=2048)
    enable_dead_letter_queue: bool = Field(default=True)
    watermark_table: str = Field(default="ingestion_watermarks")


class LLMInteractionRecord(BaseModel):
    """Schema for LLM interaction events."""

    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    tenant_id: str = "default"
    provider: str
    model: str
    prompt: str
    prompt_hash: str = ""
    response: str | None = None
    system_instruction: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    latency_ms: int
    status: Literal["success", "error", "timeout"] = "success"
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not self.prompt_hash:
            self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()


class TransformationRecord(BaseModel):
    """Schema for transformation events."""

    transformation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interaction_id: str | None = None
    technique_suite: str
    technique_name: str
    original_prompt: str
    transformed_prompt: str
    transformation_time_ms: int
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JailbreakExperimentRecord(BaseModel):
    """Schema for jailbreak research experiments."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    framework: Literal["autodan", "gptfuzz", "janus", "deepteam"]
    goal: str
    attack_method: str
    iterations: int = 0
    success: bool = False
    final_prompt: str | None = None
    judge_score: float | None = None
    target_response: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchDataIngester:
    """Batch ingestion engine for Chimera data pipeline.

    Extracts data from application logs, databases, and in-memory stores,
    validates schemas, and writes to partitioned Parquet files.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        self.config = config or IngestionConfig()
        self.data_lake_path = Path(self.config.data_lake_path)
        self.dlq_path = self.data_lake_path / "dead_letter_queue"
        self._failed_records: list[dict] = []
        self._db_connector: Any | None = None

        # Create directories
        self.data_lake_path.mkdir(parents=True, exist_ok=True)
        if self.config.enable_dead_letter_queue:
            self.dlq_path.mkdir(parents=True, exist_ok=True)

    @property
    def db_connector(self) -> Any:
        """Lazy-initialize database connector.

        Returns:
            DatabaseConnector instance

        """
        if self._db_connector is None:
            try:
                from app.services.data_pipeline.database_connector import create_connector

                self._db_connector = create_connector()
                logger.info("Database connector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database connector: {e}")
                self._db_connector = False  # Mark as failed
        return self._db_connector if self._db_connector is not None else None

    def extract_llm_interactions(
        self,
        start_time: datetime,
        end_time: datetime,
        source_db: Any | None = None,
    ) -> pd.DataFrame:
        """Extract LLM interaction events from the database.

        Uses the database connector to query the backend database for
        LLM interactions within the specified time window.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window
            source_db: Optional database connection (deprecated, use db_connector)

        Returns:
            DataFrame with LLM interaction records

        """
        logger.info(f"Extracting LLM interactions from {start_time} to {end_time}")

        # Use provided database connection or initialize connector
        connector = source_db or self.db_connector

        if connector is None:
            logger.warning(
                "No database connector available. "
                "Ensure DATABASE_URL is configured and database is accessible.",
            )
            return pd.DataFrame(columns=list(LLMInteractionRecord.model_fields.keys()))

        try:
            # Extract using database connector
            df = connector.extract_llm_interactions(start_time, end_time)

            if df.empty:
                logger.info("No LLM interactions found in time window")
                return pd.DataFrame(columns=list(LLMInteractionRecord.model_fields.keys()))

            logger.info(f"Extracted {len(df)} LLM interactions from database")
            return df

        except Exception as e:
            logger.error(f"Failed to extract LLM interactions: {e}")
            # Return empty DataFrame with correct schema on error
            return pd.DataFrame(columns=list(LLMInteractionRecord.model_fields.keys()))

    def extract_transformation_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Extract transformation events from the database.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with transformation event records

        """
        logger.info(f"Extracting transformation events from {start_time} to {end_time}")

        connector = self.db_connector

        if connector is None:
            logger.warning(
                "No database connector available. "
                "Ensure DATABASE_URL is configured and database is accessible.",
            )
            return pd.DataFrame(columns=list(TransformationRecord.model_fields.keys()))

        try:
            df = connector.extract_transformation_events(start_time, end_time)

            if df.empty:
                logger.info("No transformation events found in time window")
                return pd.DataFrame(columns=list(TransformationRecord.model_fields.keys()))

            logger.info(f"Extracted {len(df)} transformation events from database")
            return df

        except Exception as e:
            logger.error(f"Failed to extract transformation events: {e}")
            return pd.DataFrame(columns=list(TransformationRecord.model_fields.keys()))

    def extract_jailbreak_experiments(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Extract jailbreak research experiment results from the database.

        Args:
            start_time: Start of extraction window
            end_time: End of extraction window

        Returns:
            DataFrame with jailbreak experiment records

        """
        logger.info(f"Extracting jailbreak experiments from {start_time} to {end_time}")

        connector = self.db_connector

        if connector is None:
            logger.warning(
                "No database connector available. "
                "Ensure DATABASE_URL is configured and database is accessible.",
            )
            return pd.DataFrame(columns=list(JailbreakExperimentRecord.model_fields.keys()))

        try:
            df = connector.extract_jailbreak_experiments(start_time, end_time)

            if df.empty:
                logger.info("No jailbreak experiments found in time window")
                return pd.DataFrame(columns=list(JailbreakExperimentRecord.model_fields.keys()))

            logger.info(f"Extracted {len(df)} jailbreak experiments from database")
            return df

        except Exception as e:
            logger.error(f"Failed to extract jailbreak experiments: {e}")
            return pd.DataFrame(columns=list(JailbreakExperimentRecord.model_fields.keys()))

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        schema: dict[str, Any],
    ) -> pd.DataFrame:
        """Validate DataFrame against schema and clean invalid records.

        Args:
            df: Input DataFrame
            schema: Schema definition with required fields and data types

        Returns:
            Cleaned DataFrame

        """
        if df.empty:
            return df

        initial_count = len(df)

        # Check required fields
        required_fields = schema.get("required_fields", [])
        missing_fields = set(required_fields) - set(df.columns)
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")

        # Validate data types
        expected_dtypes = schema.get("dtypes", {})
        for col, dtype in expected_dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.error(f"Failed to cast {col} to {dtype}: {e}")
                    self._failed_records.extend(df[[col]].to_dict("records"))
                    df = df.drop(columns=[col])

        # Remove duplicates
        if "interaction_id" in df.columns:
            df = df.drop_duplicates(subset=["interaction_id"])

        # Remove rows with critical nulls
        for field in required_fields:
            if field in df.columns:
                df = df[df[field].notna()]

        final_count = len(df)
        if final_count < initial_count:
            logger.info(f"Cleaned {initial_count - final_count} invalid records")

        return df

    def write_to_parquet(
        self,
        df: pd.DataFrame,
        table_name: str,
        partition_cols: list[str] | None = None,
        mode: Literal["append", "overwrite"] = "append",
    ) -> Path:
        """Write DataFrame to partitioned Parquet files.

        Args:
            df: DataFrame to write
            table_name: Name of the table (e.g., 'llm_interactions')
            partition_cols: Columns to partition by
            mode: Write mode (append or overwrite)

        Returns:
            Path to written files

        """
        if df.empty:
            logger.warning(f"No data to write for table {table_name}")
            return self.data_lake_path

        # Add partitioning columns if not present
        if "dt" not in df.columns:
            df["dt"] = df["created_at"].dt.strftime("%Y-%m-%d")
        if "hour" not in df.columns:
            df["hour"] = df["created_at"].dt.hour

        # Convert datetime columns to strings for Parquet compatibility
        for col in df.select_dtypes(include=["datetime64"]).columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Define output path
        table_path = self.data_lake_path / "raw" / table_name

        # Write to Parquet
        partition_cols = partition_cols or self.config.partition_by
        try:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=str(table_path),
                partition_cols=partition_cols,
                compression=self.config.compression,
                existing_data_behavior=(
                    "overwrite_or_ignore" if mode == "append" else "delete_matching"
                ),
            )
            logger.info(f"Wrote {len(df)} rows to {table_path}")
            return table_path
        except Exception as e:
            logger.error(f"Failed to write Parquet: {e}")
            raise

    def save_dead_letter_queue(self, path: str | None = None) -> Path:
        """Save failed records to dead letter queue for manual review.

        Args:
            path: Custom path for DLQ (default: data_lake/dead_letter_queue)

        Returns:
            Path to DLQ file

        """
        if not self._failed_records:
            logger.info("No failed records to save")
            return self.dlq_path

        dlq_path = Path(path) if path else self.dlq_path
        dlq_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dlq_file = dlq_path / f"failed_records_{timestamp}.json"

        with open(dlq_file, "w") as f:
            json.dump(self._failed_records, f, indent=2, default=str)

        logger.warning(f"Saved {len(self._failed_records)} failed records to {dlq_file}")
        self._failed_records = []  # Clear after saving
        return dlq_file

    def get_last_watermark(self, table_name: str) -> datetime:
        """Get the last successful ingestion timestamp for incremental loading.

        Args:
            table_name: Name of the table

        Returns:
            Last watermark timestamp

        """
        watermark_file = self.data_lake_path / ".watermarks" / f"{table_name}.json"

        if not watermark_file.exists():
            # First run - start from 24 hours ago
            return datetime.utcnow() - timedelta(days=1)

        with open(watermark_file) as f:
            watermark = json.load(f)
            return datetime.fromisoformat(watermark["last_timestamp"])

    def update_watermark(self, table_name: str, timestamp: datetime) -> None:
        """Update the watermark for a table after successful ingestion.

        Args:
            table_name: Name of the table
            timestamp: New watermark timestamp

        """
        watermark_dir = self.data_lake_path / ".watermarks"
        watermark_dir.mkdir(parents=True, exist_ok=True)

        watermark_file = watermark_dir / f"{table_name}.json"
        with open(watermark_file, "w") as f:
            json.dump(
                {
                    "table": table_name,
                    "last_timestamp": timestamp.isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"Updated watermark for {table_name} to {timestamp}")

    def close(self) -> None:
        """Close database connections and cleanup resources.

        Should be called when the ingester is no longer needed to properly
        release database connections.
        """
        if self._db_connector is not None and self._db_connector is not False:
            self._db_connector.close()
            self._db_connector = None
            logger.info("Database connector closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False


# Convenience functions for ETL jobs


def ingest_llm_interactions_batch(lookback_hours: int = 1) -> None:
    """Run batch ingestion for LLM interactions.

    Args:
        lookback_hours: Hours to look back for new data

    """
    with BatchDataIngester() as ingester:
        # Get time window
        end_time = datetime.utcnow()
        start_time = ingester.get_last_watermark("llm_interactions")

        # Extract
        df = ingester.extract_llm_interactions(start_time, end_time)

        # Validate
        schema = {
            "required_fields": ["interaction_id", "provider", "model", "prompt", "created_at"],
            "dtypes": {"interaction_id": "object", "latency_ms": "int64", "tokens_total": "int64"},
        }
        df = ingester.validate_and_clean(df, schema)

        # Load
        if not df.empty:
            ingester.write_to_parquet(df, "llm_interactions", mode="append")
            ingester.update_watermark("llm_interactions", end_time)

        # Save failed records
        ingester.save_dead_letter_queue()


def ingest_transformation_events_batch(lookback_hours: int = 1) -> None:
    """Run batch ingestion for transformation events."""
    with BatchDataIngester() as ingester:
        end_time = datetime.utcnow()
        start_time = ingester.get_last_watermark("transformations")

        df = ingester.extract_transformation_events(start_time, end_time)

        schema = {
            "required_fields": [
                "transformation_id",
                "technique_suite",
                "technique_name",
                "created_at",
            ],
            "dtypes": {"transformation_id": "object", "transformation_time_ms": "int64"},
        }
        df = ingester.validate_and_clean(df, schema)

        if not df.empty:
            ingester.write_to_parquet(df, "transformations", mode="append")
            ingester.update_watermark("transformations", end_time)

        ingester.save_dead_letter_queue()


def ingest_jailbreak_experiments_batch(lookback_hours: int = 1) -> None:
    """Run batch ingestion for jailbreak experiments."""
    with BatchDataIngester() as ingester:
        end_time = datetime.utcnow()
        start_time = ingester.get_last_watermark("jailbreak_experiments")

        df = ingester.extract_jailbreak_experiments(start_time, end_time)

        schema = {
            "required_fields": ["experiment_id", "framework", "goal", "created_at"],
            "dtypes": {"experiment_id": "object", "iterations": "int64", "success": "bool"},
        }
        df = ingester.validate_and_clean(df, schema)

        if not df.empty:
            ingester.write_to_parquet(df, "jailbreak_experiments", mode="append")
            ingester.update_watermark("jailbreak_experiments", end_time)

        ingester.save_dead_letter_queue()
