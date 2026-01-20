"""Delta Lake Storage Manager.

Provides ACID transaction support, time travel, and optimized storage
for the Chimera data pipeline using Delta Lake format.

Features:
- ACID transactions (atomicity, consistency, isolation, durability)
- Time travel queries for historical analysis
- Upsert/merge operations with predicate matching
- Automatic schema evolution
- File compaction and optimization
- Z-order clustering for query performance
- Vacuum for removing old files
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

try:
    from delta import DeltaTable, configure_spark_with_delta_pip
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F

    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False
    DeltaTable = None
    SparkSession = None

from app.core.logging import logger


class DeltaTableConfig(BaseModel):
    """Configuration for Delta Lake tables."""

    storage_path: str = Field(
        default="/data/chimera-lake",
        description="Root path for Delta tables",
    )
    enable_time_travel: bool = Field(default=True, description="Enable time travel queries")
    retention_days: int = Field(default=30, ge=1, description="Days to retain historical versions")
    enable_auto_optimize: bool = Field(default=True, description="Auto optimize on write")
    enable_auto_compact: bool = Field(default=True, description="Auto compact small files")
    target_file_size_mb: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Target file size for optimization",
    )
    z_order_columns: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Z-order columns per table for clustering",
    )


class DeltaLakeManager:
    """Manages Delta Lake tables for Chimera data pipeline.

    Provides:
    - Table creation and management
    - ACID transactions
    - Upsert/merge operations
    - Time travel queries
    - Optimization and compaction
    """

    def __init__(self, config: DeltaTableConfig | None = None) -> None:
        if not DELTA_AVAILABLE:
            msg = "Delta Lake is not available. Install with: pip install delta-spark pyspark"
            raise ImportError(
                msg,
            )

        self.config = config or DeltaTableConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize Spark session with Delta Lake
        self.spark = self._init_spark_session()

    def _init_spark_session(self) -> SparkSession:
        """Initialize Spark session with Delta Lake configuration."""
        builder = (
            SparkSession.builder.appName("ChimeraDataPipeline")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        )

        # Configure Delta with pip
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        logger.info("Initialized Spark session with Delta Lake support")
        return spark

    def create_or_update_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        partition_columns: list[str] | None = None,
        mode: Literal["append", "overwrite", "merge"] = "append",
        merge_keys: list[str] | None = None,
    ) -> str:
        """Create or update a Delta table.

        Args:
            df: Pandas DataFrame to write
            table_name: Name of the Delta table
            partition_columns: Columns to partition by
            mode: Write mode (append, overwrite, merge)
            merge_keys: Keys for merge operations (required if mode='merge')

        Returns:
            Path to the Delta table

        """
        table_path = str(self.storage_path / table_name)
        spark_df = self.spark.createDataFrame(df)

        # Add metadata columns
        spark_df = spark_df.withColumn("_ingested_at", F.current_timestamp())

        if mode == "merge" and merge_keys:
            return self._merge_into_table(spark_df, table_name, merge_keys)

        # Write with Delta
        writer = spark_df.write.format("delta")

        if partition_columns:
            writer = writer.partitionBy(*partition_columns)

        if self.config.enable_auto_optimize:
            writer = writer.option("delta.autoOptimize.optimizeWrite", "true")

        if self.config.enable_auto_compact:
            writer = writer.option("delta.autoOptimize.autoCompact", "true")

        writer.mode(mode).save(table_path)

        logger.info(f"Wrote {df.shape[0]} rows to Delta table {table_name} at {table_path}")

        # Run optimization if configured
        if table_name in self.config.z_order_columns:
            self.optimize_table(table_name, z_order_by=self.config.z_order_columns[table_name])

        return table_path

    def _merge_into_table(self, df: "DataFrame", table_name: str, merge_keys: list[str]) -> str:
        """Perform upsert operation using Delta merge.

        Args:
            df: Spark DataFrame with new data
            table_name: Target table name
            merge_keys: Keys to match on (e.g., ['interaction_id'])

        Returns:
            Path to the Delta table

        """
        table_path = str(self.storage_path / table_name)

        # Check if table exists
        if not DeltaTable.isDeltaTable(self.spark, table_path):
            # First write - just save
            df.write.format("delta").save(table_path)
            logger.info(f"Created new Delta table {table_name}")
            return table_path

        # Load existing table
        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Build merge condition
        merge_condition = " AND ".join([f"target.{key} = source.{key}" for key in merge_keys])

        # Perform merge
        (
            delta_table.alias("target")
            .merge(df.alias("source"), merge_condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

        logger.info(f"Merged data into Delta table {table_name}")
        return table_path

    def read_table(
        self,
        table_name: str,
        as_of_version: int | None = None,
        as_of_timestamp: datetime | None = None,
        filters: str | None = None,
    ) -> pd.DataFrame:
        """Read data from a Delta table with optional time travel.

        Args:
            table_name: Name of the Delta table
            as_of_version: Version number for time travel
            as_of_timestamp: Timestamp for time travel
            filters: SQL filter expression

        Returns:
            Pandas DataFrame with table data

        """
        table_path = str(self.storage_path / table_name)

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            msg = f"Table {table_name} does not exist at {table_path}"
            raise ValueError(msg)

        # Build read query
        reader = self.spark.read.format("delta")

        # Time travel
        if as_of_version is not None:
            reader = reader.option("versionAsOf", as_of_version)
            logger.info(f"Reading {table_name} as of version {as_of_version}")
        elif as_of_timestamp is not None:
            timestamp_str = as_of_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            reader = reader.option("timestampAsOf", timestamp_str)
            logger.info(f"Reading {table_name} as of timestamp {timestamp_str}")

        df = reader.load(table_path)

        # Apply filters
        if filters:
            df = df.filter(filters)

        # Convert to Pandas
        pandas_df = df.toPandas()
        logger.info(f"Read {len(pandas_df)} rows from {table_name}")
        return pandas_df

    def optimize_table(
        self,
        table_name: str,
        z_order_by: list[str] | None = None,
    ) -> dict[str, Any]:
        """Optimize Delta table by compacting small files and Z-ordering.

        Args:
            table_name: Name of the Delta table
            z_order_by: Columns to Z-order by for clustering

        Returns:
            Optimization statistics

        """
        table_path = str(self.storage_path / table_name)

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            msg = f"Table {table_name} does not exist"
            raise ValueError(msg)

        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Compact files
        logger.info(f"Optimizing table {table_name}...")
        if z_order_by:
            delta_table.optimize().executeZOrderBy(*z_order_by)
            logger.info(f"Optimized {table_name} with Z-ordering on {z_order_by}")
        else:
            delta_table.optimize().executeCompaction()
            logger.info(f"Optimized {table_name} with file compaction")

        # Get optimization stats
        history = delta_table.history(limit=1).select("metrics").collect()
        metrics = history[0]["metrics"] if history else {}

        return {
            "table": table_name,
            "optimized_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }

    def vacuum_table(self, table_name: str, retention_hours: int | None = None) -> None:
        """Remove old file versions to free up storage.

        Args:
            table_name: Name of the Delta table
            retention_hours: Hours to retain (default: from config)

        """
        table_path = str(self.storage_path / table_name)

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            msg = f"Table {table_name} does not exist"
            raise ValueError(msg)

        delta_table = DeltaTable.forPath(self.spark, table_path)

        retention_hours = retention_hours or (self.config.retention_days * 24)

        logger.info(f"Vacuuming {table_name} with retention of {retention_hours} hours")
        delta_table.vacuum(retention_hours)
        logger.info(f"Completed vacuum for {table_name}")

    def get_table_history(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Get Delta table commit history.

        Args:
            table_name: Name of the Delta table
            limit: Number of commits to retrieve

        Returns:
            DataFrame with commit history

        """
        table_path = str(self.storage_path / table_name)

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            msg = f"Table {table_name} does not exist"
            raise ValueError(msg)

        delta_table = DeltaTable.forPath(self.spark, table_path)
        return delta_table.history(limit).toPandas()

    def restore_table(self, table_name: str, version: int) -> None:
        """Restore table to a previous version.

        Args:
            table_name: Name of the Delta table
            version: Version number to restore to

        """
        table_path = str(self.storage_path / table_name)

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            msg = f"Table {table_name} does not exist"
            raise ValueError(msg)

        delta_table = DeltaTable.forPath(self.spark, table_path)

        logger.warning(f"Restoring {table_name} to version {version}")
        delta_table.restoreToVersion(version)
        logger.info(f"Restored {table_name} to version {version}")

    def get_table_details(self, table_name: str) -> dict[str, Any]:
        """Get detailed information about a Delta table.

        Args:
            table_name: Name of the Delta table

        Returns:
            Dictionary with table metadata

        """
        table_path = str(self.storage_path / table_name)

        if not DeltaTable.isDeltaTable(self.spark, table_path):
            msg = f"Table {table_name} does not exist"
            raise ValueError(msg)

        delta_table = DeltaTable.forPath(self.spark, table_path)

        # Get basic details
        detail_df = delta_table.detail().toPandas()
        details = detail_df.to_dict("records")[0] if not detail_df.empty else {}

        # Get latest history
        history_df = delta_table.history(limit=1).toPandas()
        latest_commit = history_df.to_dict("records")[0] if not history_df.empty else {}

        return {
            "table_name": table_name,
            "table_path": table_path,
            "format": details.get("format", "delta"),
            "partition_columns": details.get("partitionColumns", []),
            "num_files": details.get("numFiles", 0),
            "size_in_bytes": details.get("sizeInBytes", 0),
            "latest_version": latest_commit.get("version", 0),
            "latest_timestamp": latest_commit.get("timestamp"),
            "created_at": details.get("createdAt"),
        }

    def list_tables(self) -> list[str]:
        """List all Delta tables in the storage path.

        Returns:
            List of table names

        """
        tables = []
        for path in self.storage_path.iterdir():
            if path.is_dir() and DeltaTable.isDeltaTable(self.spark, str(path)):
                tables.append(path.name)

        logger.info(f"Found {len(tables)} Delta tables")
        return tables

    def close(self) -> None:
        """Close Spark session and cleanup resources."""
        if self.spark:
            self.spark.stop()
            logger.info("Closed Spark session")


# Convenience functions for common operations


def create_llm_interactions_table(df: pd.DataFrame) -> str:
    """Create or append to llm_interactions Delta table.

    Args:
        df: DataFrame with LLM interaction records

    Returns:
        Path to the Delta table

    """
    manager = DeltaLakeManager()

    # Ensure required columns
    required_cols = ["interaction_id", "provider", "model", "created_at"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Add partitioning columns
    df["dt"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d")
    df["hour"] = pd.to_datetime(df["created_at"]).dt.hour

    return manager.create_or_update_table(
        df=df,
        table_name="llm_interactions",
        partition_columns=["dt", "hour"],
        mode="append",
    )


def query_llm_interactions(
    start_date: str,
    end_date: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    """Query LLM interactions with filters.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to start_date
        provider: Filter by provider

    Returns:
        DataFrame with filtered interactions

    """
    manager = DeltaLakeManager()

    end_date = end_date or start_date
    filters = f"dt >= '{start_date}' AND dt <= '{end_date}'"

    if provider:
        filters += f" AND provider = '{provider}'"

    return manager.read_table("llm_interactions", filters=filters)
