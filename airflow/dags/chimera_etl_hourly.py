"""
Airflow DAG: Chimera ETL Hourly

Orchestrates hourly ETL pipeline for Chimera LLM analytics:
1. Extract data from logs, databases, and event streams
2. Validate data quality with Great Expectations
3. Transform data with dbt models
4. Load into Delta Lake with optimization
5. Refresh analytics views and dashboards

Schedule: Every hour at :05 (to allow time for logs to flush)
Retries: 3 attempts with exponential backoff
SLA: 10 minutes
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator
from airflow.providers.dbt.operators.dbt import DbtRunOperator, DbtTestOperator
from airflow.utils.dates import days_ago

from airflow import DAG

# Add Chimera backend to path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.data_pipeline.batch_ingestion import (
    ingest_jailbreak_experiments_batch,
    ingest_llm_interactions_batch,
    ingest_transformation_events_batch,
)
from app.services.data_pipeline.data_quality import (
    validate_llm_interactions,
    validate_transformations,
)
from app.services.data_pipeline.delta_lake_manager import DeltaLakeManager

# =============================================================================
# DAG Configuration
# =============================================================================

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email": ["data-team@chimera.ai"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "sla": timedelta(minutes=10),
}

# =============================================================================
# Task Functions
# =============================================================================


def extract_llm_interactions(**_context):
    """Extract LLM interactions from the last hour."""
    ingest_llm_interactions_batch(lookback_hours=1)
    return {"status": "success", "task": "extract_llm_interactions"}


def extract_transformations(**_context):
    """Extract transformation events from the last hour."""
    ingest_transformation_events_batch(lookback_hours=1)
    return {"status": "success", "task": "extract_transformations"}


def extract_jailbreak_experiments(**_context):
    """Extract jailbreak experiment results from the last hour."""
    ingest_jailbreak_experiments_batch(lookback_hours=1)
    return {"status": "success", "task": "extract_jailbreak_experiments"}


def validate_quality(**_context):
    """
    Run data quality checks on extracted data.

    Validates:
    - LLM interactions schema and constraints
    - Transformation events completeness
    - Data freshness and volume
    """

    from app.services.data_pipeline.batch_ingestion import BatchDataIngester

    ingester = BatchDataIngester()

    # Validate LLM interactions (last hour)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    llm_df = ingester.extract_llm_interactions(start_time, end_time)
    if not llm_df.empty:
        llm_result = validate_llm_interactions(llm_df)
        if not llm_result.success:
            raise ValueError(
                f"LLM interactions validation failed: {llm_result.failed_expectations}"
            )

    # Validate transformations
    trans_df = ingester.extract_transformation_events(start_time, end_time)
    if not trans_df.empty:
        trans_result = validate_transformations(trans_df)
        if not trans_result.success:
            raise ValueError(
                f"Transformations validation failed: {trans_result.failed_expectations}"
            )

    return {"status": "success", "task": "validate_quality"}


def optimize_delta_tables(**_context):
    """
    Optimize Delta Lake tables with compaction and Z-ordering.

    Runs:
    - File compaction to merge small files
    - Z-order clustering for query performance
    - Statistics collection for query optimization
    """
    manager = DeltaLakeManager()

    tables_to_optimize = [
        ("llm_interactions", ["provider", "dt"]),
        ("transformations", ["technique_suite", "dt"]),
        ("jailbreak_experiments", ["framework", "dt"]),
    ]

    results = []
    for table_name, z_order_cols in tables_to_optimize:
        try:
            result = manager.optimize_table(table_name, z_order_by=z_order_cols)
            results.append(result)
        except Exception as e:
            # Non-critical - log and continue
            print(f"Optimization warning for {table_name}: {e}")

    return {"status": "success", "optimized_tables": len(results)}


def vacuum_old_versions(**_context):
    """
    Remove old Delta Lake file versions to free storage.

    Retains versions for 7 days (168 hours) for time travel.
    """
    manager = DeltaLakeManager()

    tables = ["llm_interactions", "transformations", "jailbreak_experiments"]

    for table_name in tables:
        try:
            manager.vacuum_table(table_name, retention_hours=168)  # 7 days
        except Exception as e:
            print(f"Vacuum warning for {table_name}: {e}")

    return {"status": "success", "task": "vacuum_old_versions"}


def refresh_analytics_views(**_context):
    """
    Refresh materialized views and aggregations for dashboards.

    Updates:
    - Hourly provider metrics
    - Daily usage statistics
    - Technique effectiveness scores
    - Cost estimations
    """
    # Placeholder for view refresh logic
    # In production, this would:
    # 1. Execute SQL to refresh materialized views
    # 2. Update pre-computed aggregations
    # 3. Invalidate query result caches

    return {"status": "success", "task": "refresh_analytics_views"}


def send_pipeline_metrics(**context):
    """
    Send pipeline execution metrics to monitoring system.

    Tracks:
    - Pipeline execution time
    - Data volume processed
    - Quality check results
    - Optimization statistics
    """
    execution_date = context["execution_date"]
    duration = datetime.utcnow() - execution_date

    metrics = {
        "pipeline": "chimera_etl_hourly",
        "execution_date": execution_date.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "status": "success",
    }

    # In production, send to Prometheus Pushgateway or CloudWatch
    print(f"Pipeline metrics: {metrics}")

    return metrics


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    "chimera_etl_hourly",
    default_args=default_args,
    description="Hourly ETL pipeline for Chimera LLM analytics",
    schedule_interval="5 * * * *",  # Every hour at :05
    start_date=days_ago(1),
    catchup=False,
    tags=["chimera", "production", "etl"],
    max_active_runs=1,  # Prevent overlapping runs
) as dag:
    # =============================================================================
    # Extract Tasks (Parallel)
    # =============================================================================

    extract_llm = PythonOperator(
        task_id="extract_llm_interactions",
        python_callable=extract_llm_interactions,
        provide_context=True,
    )

    extract_trans = PythonOperator(
        task_id="extract_transformations",
        python_callable=extract_transformations,
        provide_context=True,
    )

    extract_jailbreak = PythonOperator(
        task_id="extract_jailbreak_experiments",
        python_callable=extract_jailbreak_experiments,
        provide_context=True,
    )

    # =============================================================================
    # Validate Data Quality
    # =============================================================================

    validate = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_quality,
        provide_context=True,
    )

    # =============================================================================
    # Transform with dbt (Staging → Marts)
    # =============================================================================

    dbt_staging = DbtRunOperator(
        task_id="dbt_run_staging",
        models="staging",
        profiles_dir="/opt/dbt",
        project_dir="/opt/dbt/chimera",
        target="prod",
        dbt_bin="/usr/local/bin/dbt",
    )

    dbt_marts = DbtRunOperator(
        task_id="dbt_run_marts",
        models="marts",
        profiles_dir="/opt/dbt",
        project_dir="/opt/dbt/chimera",
        target="prod",
        dbt_bin="/usr/local/bin/dbt",
    )

    # =============================================================================
    # Test Data Quality with dbt
    # =============================================================================

    dbt_test = DbtTestOperator(
        task_id="dbt_test_all",
        profiles_dir="/opt/dbt",
        project_dir="/opt/dbt/chimera",
        target="prod",
        dbt_bin="/usr/local/bin/dbt",
    )

    # =============================================================================
    # Optimize Storage
    # =============================================================================

    optimize = PythonOperator(
        task_id="optimize_delta_tables",
        python_callable=optimize_delta_tables,
        provide_context=True,
    )

    vacuum = PythonOperator(
        task_id="vacuum_old_versions",
        python_callable=vacuum_old_versions,
        provide_context=True,
        trigger_rule="all_done",  # Run even if upstream fails
    )

    # =============================================================================
    # Refresh Analytics
    # =============================================================================

    refresh_views = PythonOperator(
        task_id="refresh_analytics_views",
        python_callable=refresh_analytics_views,
        provide_context=True,
    )

    # =============================================================================
    # Send Metrics
    # =============================================================================

    send_metrics = PythonOperator(
        task_id="send_pipeline_metrics",
        python_callable=send_pipeline_metrics,
        provide_context=True,
        trigger_rule="all_done",
    )

    # =============================================================================
    # Task Dependencies
    # =============================================================================

    # Extract (parallel) → Validate → dbt Staging → dbt Marts → dbt Test
    [extract_llm, extract_trans, extract_jailbreak] >> validate
    validate >> dbt_staging >> dbt_marts >> dbt_test

    # Optimize and refresh after dbt
    dbt_test >> optimize >> refresh_views

    # Vacuum and metrics run last
    refresh_views >> [vacuum, send_metrics]
