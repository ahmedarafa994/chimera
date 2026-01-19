"""
Chimera Data Pipeline Package

Production-grade data pipeline for LLM analytics, jailbreak research tracking,
and system observability.

Components:
- batch_ingestion: ETL service for extracting and loading data
- delta_lake_manager: ACID storage with time travel capabilities
- data_quality: Automated validation with Great Expectations

Usage:
    from app.services.data_pipeline import BatchDataIngester, DeltaLakeManager

    # Ingest data
    ingester = BatchDataIngester()
    df = ingester.extract_llm_interactions(start_time, end_time)

    # Store with ACID guarantees
    manager = DeltaLakeManager()
    manager.create_or_update_table(df, 'llm_interactions', mode='append')
"""

from .batch_ingestion import (
                              BatchDataIngester,
                              IngestionConfig,
                              JailbreakExperimentRecord,
                              LLMInteractionRecord,
                              TransformationRecord,
                              ingest_jailbreak_experiments_batch,
                              ingest_llm_interactions_batch,
                              ingest_transformation_events_batch,
)
from .data_quality import (
                              DataQualityConfig,
                              DataQualityFramework,
                              QualityCheckResult,
                              validate_llm_interactions,
                              validate_transformations,
)
from .delta_lake_manager import (
                              DeltaLakeManager,
                              DeltaTableConfig,
                              create_llm_interactions_table,
                              query_llm_interactions,
)

__all__ = [
    # Batch Ingestion
    "BatchDataIngester",
    "DataQualityConfig",
    # Data Quality
    "DataQualityFramework",
    # Delta Lake
    "DeltaLakeManager",
    "DeltaTableConfig",
    "IngestionConfig",
    "JailbreakExperimentRecord",
    "LLMInteractionRecord",
    "QualityCheckResult",
    "TransformationRecord",
    "create_llm_interactions_table",
    "ingest_jailbreak_experiments_batch",
    "ingest_llm_interactions_batch",
    "ingest_transformation_events_batch",
    "query_llm_interactions",
    "validate_llm_interactions",
    "validate_transformations",
]

__version__ = "1.0.0"
