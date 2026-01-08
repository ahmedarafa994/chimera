# Chimera Data Pipeline Architecture

## Executive Summary

This document outlines a scalable, cost-effective data pipeline architecture for the Chimera AI prompt optimization system. The pipeline captures LLM interaction data, prompt transformations, jailbreak research metrics, and system telemetry for analytics, compliance, and research purposes.

## Architecture Overview

### Pipeline Pattern: Lambda Architecture (Batch + Speed Layers)

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                               │
├──────────────────────────────────────────────────────────────────┤
│  • LLM API Calls (Google, OpenAI, Anthropic, DeepSeek)          │
│  • Transformation Events (20+ technique suites)                   │
│  • Jailbreak Research Data (AutoDAN, GPTFuzz, Janus)            │
│  • System Metrics (Prometheus, Usage Tracker)                    │
│  • User Interactions (Sessions, Providers, Model Selection)      │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ├─────────────┬─────────────────┐
                          │             │                 │
                  ┌───────▼──────┐  ┌──▼──────┐  ┌──────▼──────┐
                  │ BATCH LAYER  │  │ SPEED   │  │   SERVING   │
                  │              │  │ LAYER   │  │    LAYER    │
                  │ • S3/Local   │  │ • Redis │  │ • Postgres  │
                  │ • Parquet    │  │ • Kafka │  │ • DuckDB    │
                  │ • Airflow    │  │ • Stream│  │ • FastAPI   │
                  │ • dbt        │  │ • Alerts│  │ • Grafana   │
                  └──────────────┘  └─────────┘  └─────────────┘
```

### Design Decisions

1. **Storage**: Parquet files on S3/local filesystem for cost efficiency
2. **Orchestration**: Airflow for batch ETL (hourly/daily)
3. **Transformation**: dbt for SQL-based analytics transformations
4. **Real-time**: Redis for streaming metrics and alerts
5. **Analytics**: DuckDB for OLAP queries on Parquet files
6. **Monitoring**: Grafana + Prometheus for observability

## Data Flow Architecture

### 1. Ingestion Layer

#### Batch Ingestion (Hourly)
- **Source**: Application logs, database exports, API metrics
- **Format**: JSON → Parquet with Snappy compression
- **Partitioning**: `dt=YYYY-MM-DD/hour=HH`
- **Metadata**: `_ingested_at`, `_source_system`, `_batch_id`

#### Streaming Ingestion (Real-time)
- **Source**: FastAPI middleware, WebSocket events
- **Destination**: Redis Streams
- **TTL**: 24 hours (then archived to batch)
- **Use Cases**: Live dashboards, real-time alerts

### 2. Storage Layer

#### Data Lake Structure
```
s3://chimera-data-lake/ (or local: /data/chimera/)
├── raw/                          # Raw ingested data
│   ├── llm_interactions/
│   │   └── dt=2026-01-01/
│   │       └── hour=12/
│   │           └── interactions.parquet
│   ├── transformations/
│   ├── jailbreak_experiments/
│   └── system_metrics/
│
├── staging/                      # Cleaned and validated
│   ├── llm_interactions/
│   ├── transformations/
│   └── metrics/
│
└── marts/                        # Analytics-ready tables
    ├── fact_llm_usage/
    ├── dim_providers/
    ├── dim_techniques/
    └── agg_daily_stats/
```

#### Schema Design

**fact_llm_interactions**
```python
{
    "interaction_id": "uuid",
    "session_id": "string",
    "tenant_id": "string",
    "provider": "string",           # google, openai, anthropic, etc.
    "model": "string",              # gemini-2.0-flash-exp, gpt-4, etc.
    "prompt": "string",             # Original prompt
    "prompt_hash": "string",        # SHA256 for deduplication
    "response": "string",           # LLM response
    "system_instruction": "string", # System prompt
    "config": "json",               # GenerationConfig
    "tokens_prompt": "int",
    "tokens_completion": "int",
    "tokens_total": "int",
    "latency_ms": "int",
    "status": "string",             # success, error, timeout
    "error_message": "string",
    "created_at": "timestamp",
    "ingested_at": "timestamp"
}
```

**fact_transformations**
```python
{
    "transformation_id": "uuid",
    "interaction_id": "uuid",       # FK to llm_interactions
    "technique_suite": "string",     # advanced, cognitive_hacking, etc.
    "technique_name": "string",
    "original_prompt": "string",
    "transformed_prompt": "string",
    "transformation_time_ms": "int",
    "success": "boolean",
    "created_at": "timestamp"
}
```

**fact_jailbreak_experiments**
```python
{
    "experiment_id": "uuid",
    "framework": "string",          # autodan, gptfuzz, janus
    "goal": "string",               # Target jailbreak objective
    "attack_method": "string",      # vanilla, best_of_n, beam_search
    "iterations": "int",
    "success": "boolean",
    "final_prompt": "string",
    "judge_score": "float",
    "target_response": "string",
    "metadata": "json",             # Framework-specific data
    "created_at": "timestamp"
}
```

**agg_provider_metrics_hourly**
```python
{
    "provider": "string",
    "model": "string",
    "hour_timestamp": "timestamp",  # Truncated to hour
    "total_requests": "int",
    "successful_requests": "int",
    "failed_requests": "int",
    "total_tokens": "int",
    "total_tokens_prompt": "int",
    "total_tokens_completion": "int",
    "avg_latency_ms": "float",
    "p95_latency_ms": "float",
    "p99_latency_ms": "float",
    "circuit_breaker_opens": "int"
}
```

### 3. Transformation Layer (dbt)

#### Staging Models (`staging/`)
- `stg_llm_interactions`: Deduplicated, validated interactions
- `stg_transformations`: Cleaned transformation events
- `stg_system_metrics`: Normalized Prometheus metrics

#### Intermediate Models (`intermediate/`)
- `int_session_enriched`: Sessions with user context
- `int_provider_stats`: Provider performance metrics
- `int_technique_effectiveness`: Transformation success rates

#### Mart Models (`marts/`)
- `fact_llm_usage`: Star schema for BI tools
- `dim_providers`: Provider dimension with SCD Type 2
- `dim_techniques`: Transformation technique catalog
- `agg_daily_stats`: Daily rollups for dashboards

### 4. Data Quality Framework

#### Great Expectations Suites

**llm_interactions_suite**
```python
expectations = [
    # Table-level
    expect_table_row_count_to_be_between(min_value=0, max_value=1000000),
    expect_table_column_count_to_equal(value=15),

    # Column-level
    expect_column_values_to_not_be_null(column="interaction_id"),
    expect_column_values_to_be_unique(column="interaction_id"),
    expect_column_values_to_be_in_set(
        column="provider",
        value_set=["google", "openai", "anthropic", "deepseek", "mock"]
    ),
    expect_column_values_to_be_between(
        column="latency_ms",
        min_value=0,
        max_value=300000  # 5 minutes max
    ),
    expect_column_values_to_match_regex(
        column="interaction_id",
        regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    ),
]
```

#### dbt Tests (`models/schema.yml`)
```yaml
version: 2

models:
  - name: stg_llm_interactions
    description: "Staged LLM interaction events"
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - interaction_id
            - created_at
    columns:
      - name: interaction_id
        tests:
          - unique
          - not_null
      - name: provider
        tests:
          - accepted_values:
              values: ['google', 'openai', 'anthropic', 'deepseek', 'mock']
      - name: tokens_total
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_between:
              min_value: 0
              max_value: 100000
      - name: created_at
        tests:
          - not_null
          - dbt_expectations.expect_column_values_to_be_of_type:
              column_type: timestamp
```

### 5. Orchestration (Airflow)

#### DAG: `chimera_etl_hourly`
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.dbt.operators.dbt import DbtRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

with DAG(
    'chimera_etl_hourly',
    default_args=default_args,
    description='Hourly ETL for Chimera LLM analytics',
    schedule_interval='@hourly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['chimera', 'production'],
) as dag:

    # Extract
    extract_llm_logs = PythonOperator(
        task_id='extract_llm_interactions',
        python_callable=extract_llm_interactions,
        op_kwargs={'lookback_hours': 1}
    )

    extract_transformations = PythonOperator(
        task_id='extract_transformations',
        python_callable=extract_transformation_events,
    )

    extract_metrics = PythonOperator(
        task_id='extract_system_metrics',
        python_callable=extract_prometheus_metrics,
    )

    # Validate
    validate_data = PythonOperator(
        task_id='validate_with_great_expectations',
        python_callable=run_data_quality_checks,
    )

    # Transform (dbt)
    dbt_run_staging = DbtRunOperator(
        task_id='dbt_run_staging',
        models='staging',
        profiles_dir='/opt/dbt',
        target='prod',
    )

    dbt_run_marts = DbtRunOperator(
        task_id='dbt_run_marts',
        models='marts',
        profiles_dir='/opt/dbt',
        target='prod',
    )

    dbt_test = DbtRunOperator(
        task_id='dbt_test',
        models='all',
        profiles_dir='/opt/dbt',
        target='prod',
        test=True,
    )

    # Load
    refresh_analytics_views = PythonOperator(
        task_id='refresh_analytics_views',
        python_callable=refresh_duckdb_views,
    )

    # Dependencies
    [extract_llm_logs, extract_transformations, extract_metrics] >> validate_data
    validate_data >> dbt_run_staging >> dbt_run_marts >> dbt_test
    dbt_test >> refresh_analytics_views
```

### 6. Monitoring & Observability

#### Metrics to Track

**Pipeline Health**
- `chimera_pipeline_execution_time_seconds{dag="chimera_etl_hourly"}`
- `chimera_pipeline_success_total{dag="chimera_etl_hourly"}`
- `chimera_pipeline_failure_total{dag="chimera_etl_hourly"}`
- `chimera_data_quality_tests_passed_total`
- `chimera_data_quality_tests_failed_total`

**Data Freshness**
- `chimera_latest_interaction_lag_seconds` (time since last interaction)
- `chimera_table_row_count{table="fact_llm_interactions"}`
- `chimera_table_update_timestamp{table="agg_provider_metrics_hourly"}`

**Data Quality**
- `chimera_duplicate_records_total{table="fact_llm_interactions"}`
- `chimera_null_values_total{table="fact_llm_interactions", column="provider"}`
- `chimera_schema_validation_failures_total`

#### Alerting Rules (Prometheus)
```yaml
groups:
  - name: chimera_data_pipeline
    interval: 60s
    rules:
      - alert: DataPipelineFailure
        expr: chimera_pipeline_failure_total > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Chimera data pipeline failed"
          description: "Pipeline {{ $labels.dag }} has failed"

      - alert: DataFreshnessViolation
        expr: chimera_latest_interaction_lag_seconds > 7200
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data freshness SLA violated"
          description: "Latest interaction is {{ $value }} seconds old"

      - alert: DataQualityFailure
        expr: rate(chimera_data_quality_tests_failed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data quality tests failing"
          description: "{{ $value }} quality tests failed in last 5 minutes"
```

#### Grafana Dashboards

**Dashboard: Chimera Data Pipeline Overview**
- Pipeline execution times (P50, P95, P99)
- Success/failure rates
- Data volume trends (rows ingested per hour)
- Data freshness gauges
- Quality test pass rates

**Dashboard: LLM Analytics**
- Requests per provider (time series)
- Token usage by model (stacked area)
- Latency percentiles by provider (heatmap)
- Error rates by provider (pie chart)
- Cost estimation by provider

**Dashboard: Jailbreak Research**
- AutoDAN success rates over time
- GPTFuzz mutation effectiveness
- Technique utilization heatmap
- Attack success correlation matrix

## Cost Optimization Strategy

### Storage Optimization
- **Partitioning**: Date-based partitioning reduces scan costs by 90%+
- **Compression**: Snappy compression achieves 4:1 ratio on JSON logs
- **File Sizing**: Target 512MB-1GB Parquet files for optimal query performance
- **Lifecycle**:
  - Hot (last 7 days): Local SSD
  - Warm (8-90 days): S3 Standard
  - Cold (90+ days): S3 Glacier Deep Archive

### Compute Optimization
- **Batch Windows**: Process hourly to balance freshness and efficiency
- **Spot Instances**: 70% cost savings for batch workloads
- **Query Optimization**: Pre-aggregate daily stats to avoid scanning raw data
- **Caching**: DuckDB result caching reduces repeat query costs

### Estimated Costs (AWS)
- **Storage**: ~$23/TB/month (S3 Standard) → $1/TB/month (Glacier)
- **Compute**: ~$100/month for t3.medium Airflow server (spot)
- **Data Transfer**: Minimal (most processing in same region)
- **Total**: ~$200/month for 10TB storage + hourly processing

## Deployment Guide

### Prerequisites
```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3.11 postgresql-client

# Python dependencies
pip install -r backend-api/requirements-pipeline.txt

# Install Airflow
pip install apache-airflow==2.8.1
pip install apache-airflow-providers-dbt-cloud==3.5.1

# Install dbt
pip install dbt-duckdb==1.7.0

# Install Great Expectations
pip install great-expectations==0.18.8
```

### Configuration

**Environment Variables** (`.env.pipeline`)
```bash
# Storage
DATA_LAKE_PATH=/data/chimera-lake  # or s3://bucket-name
WAREHOUSE_PATH=/data/chimera-warehouse

# Airflow
AIRFLOW_HOME=/opt/airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://user:pass@localhost/airflow

# dbt
DBT_PROFILES_DIR=/opt/dbt
DBT_PROJECT_DIR=/opt/dbt/chimera

# Redis (for streaming)
REDIS_URL=redis://localhost:6379/1

# Monitoring
PROMETHEUS_PUSHGATEWAY_URL=http://localhost:9091
```

### Initialize Airflow
```bash
# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@chimera.ai

# Start services
airflow webserver -p 8080 &
airflow scheduler &
```

### Deploy dbt Project
```bash
cd /opt/dbt/chimera

# Install dbt dependencies
dbt deps

# Run dbt models
dbt run --target prod

# Run data quality tests
dbt test --target prod
```

### Start Ingestion
```bash
# Enable Chimera ETL DAG
airflow dags unpause chimera_etl_hourly

# Trigger manual run
airflow dags trigger chimera_etl_hourly
```

## Success Metrics

### Performance SLAs
- **Data Freshness**: < 1 hour lag for batch, < 5 minutes for streaming
- **Pipeline Execution**: < 10 minutes for hourly ETL
- **Query Performance**: < 5 seconds for dashboard queries
- **Data Quality**: > 99% test pass rate

### Business Metrics
- **Cost Efficiency**: < $0.10 per million interactions stored
- **Availability**: 99.9% pipeline uptime
- **Scalability**: Support 10x traffic growth without architecture changes
- **Team Velocity**: Data analysts can create new metrics without engineering support

## Future Enhancements

### Phase 2: Advanced Analytics
- ML feature store for prompt similarity search
- Real-time anomaly detection for jailbreak attempts
- A/B testing framework for prompt techniques
- User segmentation and cohort analysis

### Phase 3: Data Science Integration
- Training data pipeline for fine-tuning models
- Reinforcement learning feedback loop for AutoDAN
- Multi-armed bandit for provider selection
- Causal inference for technique effectiveness

### Phase 4: Compliance & Governance
- Data lineage tracking (dbt docs + Marquez)
- GDPR compliance (right to deletion, data portability)
- Audit logs for sensitive prompt access
- PII detection and redaction pipeline
