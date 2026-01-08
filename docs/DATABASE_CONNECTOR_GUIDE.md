# Database Connector Implementation Guide

## Overview

The database connector provides a production-ready abstraction layer for extracting LLM interactions, transformations, and jailbreak experiments from the Chimera backend database. It supports both PostgreSQL (production) and SQLite (development) with automatic connection pooling and error handling.

## Features

- **Multi-Database Support**: PostgreSQL and SQLite with automatic detection
- **Connection Pooling**: Efficient connection reuse with configurable pool sizes
- **Automatic Reconnection**: Health checks and connection recycling
- **SQL Injection Prevention**: Parameterized queries throughout
- **Incremental Extraction**: Time-window based data extraction
- **Error Handling**: Graceful degradation with detailed logging

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BatchDataIngester                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Database Connector (Factory Pattern)              │ │
│  │  ┌──────────────────┐      ┌──────────────────────────┐  │ │
│  │  │ PostgreSQL       │      │ SQLite                   │  │ │
│  │  │ Connector        │      │ Connector                │  │ │
│  │  │                  │      │                          │  │ │
│  │  │ - Connection     │      │ - No Pooling            │  │ │
│  │  │   Pooling        │      │ - Simplified Queries    │  │ │
│  │  │ - Advanced SQL   │      │ - JSON Parse Manual     │  │ │
│  │  │ - SHA256 Hash    │      │ - Manual SHA256          │  │ │
│  │  └──────────────────┘      └──────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                               │
│  Extraction Methods:                                          │
│  • extract_llm_interactions()                                 │
│  • extract_transformation_events()                            │
│  • extract_jailbreak_experiments()                            │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### llm_interactions
Primary fact table storing all LLM API calls with metrics.

```sql
CREATE TABLE llm_interactions (
    id UUID PRIMARY KEY,
    session_id UUID,
    tenant_id VARCHAR(255) DEFAULT 'default',
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    system_instruction TEXT,
    config JSONB DEFAULT '{}',
    tokens_prompt INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### transformation_events
Stores prompt transformation attempts with technique tracking.

```sql
CREATE TABLE transformation_events (
    id UUID PRIMARY KEY,
    interaction_id UUID REFERENCES llm_interactions(id),
    technique_suite VARCHAR(100) NOT NULL,
    technique_name VARCHAR(100) NOT NULL,
    original_prompt TEXT NOT NULL,
    transformed_prompt TEXT NOT NULL,
    transformation_time_ms INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### jailbreak_experiments
Research experiment results for adversarial prompting.

```sql
CREATE TABLE jailbreak_experiments (
    id UUID PRIMARY KEY,
    framework VARCHAR(50) NOT NULL,
    attack_method VARCHAR(100) NOT NULL,
    goal TEXT NOT NULL,
    final_prompt TEXT,
    target_response TEXT,
    iterations INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT FALSE,
    judge_score FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Installation

### 1. Install Dependencies

```bash
cd backend-api
pip install -r requirements-pipeline.txt
```

This includes:
- `sqlalchemy==2.0.23` - Database ORM and connection pooling
- `psycopg2-binary==2.9.9` - PostgreSQL adapter

### 2. Setup Database Schema

**For PostgreSQL (Production):**
```bash
psql -U chimera_user -d chimera -f scripts/create_pipeline_tables.sql
```

**For SQLite (Development):**
```bash
sqlite3 chimera.db < scripts/create_pipeline_tables_sqlite.sql
```

### 3. Configure Environment

Set `DATABASE_URL` in `.env`:

```bash
# PostgreSQL (Production)
DATABASE_URL=postgresql://chimera_user:password@localhost:5432/chimera

# SQLite (Development)
DATABASE_URL=sqlite:///./chimera.db
```

## Usage

### Basic Extraction

```python
from datetime import datetime, timedelta
from app.services.data_pipeline.database_connector import create_connector

# Create connector (automatically detects database type)
connector = create_connector()

# Define time window
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)

# Extract LLM interactions
df = connector.extract_llm_interactions(start_time, end_time)

print(f"Extracted {len(df)} interactions")
print(df.head())

# Close connection
connector.close()
```

### Using BatchDataIngester

```python
from datetime import datetime, timedelta
from app.services.data_pipeline.batch_ingestion import BatchDataIngester

# Context manager ensures proper cleanup
with BatchDataIngester() as ingester:
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    # Extract, validate, and write to Parquet
    df = ingester.extract_llm_interactions(start_time, end_time)

    if not df.empty:
        schema = {
            "required_fields": ["interaction_id", "provider", "model"],
            "dtypes": {"latency_ms": "int64"},
        }
        df = ingester.validate_and_clean(df, schema)
        ingester.write_to_parquet(df, "llm_interactions")
        ingester.update_watermark("llm_interactions", end_time)
```

### Convenience Functions

```python
from app.services.data_pipeline.batch_ingestion import ingest_llm_interactions_batch

# One-line ingestion
ingest_llm_interactions_batch(lookback_hours=1)
```

## Configuration

### Connection Pool Settings

```python
from app.services.data_pipeline.database_connector import create_connector

# Custom pool configuration
connector = create_connector(
    connection_url="postgresql://user:pass@host/db",
    pool_size=10,          # Default: 5
    max_overflow=20,        # Default: 10
)
```

### IngestionConfig

```python
from app.services.data_pipeline.batch_ingestion import BatchDataIngester, IngestionConfig

config = IngestionConfig(
    data_lake_path="/data/chimera-lake",
    partition_by=["dt", "hour"],
    compression="snappy",
    max_file_size_mb=512,
    enable_dead_letter_queue=True,
)

ingester = BatchDataIngester(config=config)
```

## Testing

Run the test suite to verify your setup:

```bash
cd backend-api
python scripts/test_pipeline_connector.py
```

Expected output:
```
============================================================
TEST 1: Database Connection
============================================================
✅ Database connection successful

============================================================
TEST 2: Sample Data Insertion
============================================================
✅ Inserted sample record with ID: 550e8400-e29b-41d4-a716-446655440000
✅ Verified sample record retrieval
   Provider: google
   Model: gemini-2.0-flash
   Tokens: 23

============================================================
TEST SUMMARY
============================================================
Database Connection: ✅ PASSED
Sample Data Insertion: ✅ PASSED
Extraction Methods: ✅ PASSED
Batch Ingestion Service: ✅ PASSED

Total: 4/4 tests passed

🎉 All tests passed!
```

## Integration with Airflow

Update your Airflow DAG to use the database connector:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from app.services.data_pipeline.batch_ingestion import ingest_llm_interactions_batch

default_args = {
    'owner': 'chimera',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'chimera_etl_hourly',
    default_args=default_args,
    description='Hourly ETL for Chimera data pipeline',
    schedule_interval='5 * * * *',  # Every hour at :05 minutes,
    tags=['chimera', 'etl', 'pipeline'],
)

extract_task = PythonOperator(
    task_id='extract_llm_interactions',
    python_callable=ingest_llm_interactions_batch,
    op_kwargs={'lookback_hours': 1},
    dag=dag,
)
```

## Performance Optimization

### Query Optimization

1. **Use Time Windows**: Always filter by `created_at` for index usage
2. **Batch Processing**: Extract in hourly chunks instead of full days
3. **Connection Pooling**: Reuse connections across queries
4. **Select Only Needed Columns**: Reduces memory and network overhead

### Index Strategy

Ensure these indexes exist:

```sql
-- Time-based queries (most important)
CREATE INDEX idx_llm_interactions_created_at
    ON llm_interactions(created_at DESC);

-- Provider analytics
CREATE INDEX idx_llm_interactions_provider_created
    ON llm_interactions(provider, created_at DESC);

-- Session tracking
CREATE INDEX idx_llm_interactions_session_id
    ON llm_interactions(session_id);
```

## Troubleshooting

### Issue: "No database connector available"

**Cause**: DATABASE_URL not configured or database unreachable

**Solution**:
```bash
# Check .env file
echo $DATABASE_URL

# Test database connection
psql $DATABASE_URL -c "SELECT 1"
```

### Issue: "Relation does not exist"

**Cause**: Database tables not created

**Solution**:
```bash
# Run migration script
psql -U user -d chimera -f scripts/create_pipeline_tables.sql
```

### Issue: Connection pool exhausted

**Cause**: Too many concurrent extractions

**Solution**: Increase pool size or reduce concurrent tasks
```python
connector = create_connector(
    pool_size=20,      # Increase from 5
    max_overflow=40,   # Increase from 10
)
```

## Best Practices

1. **Always use context managers** for automatic cleanup:
   ```python
   with BatchDataIngester() as ingester:
       # ... your code
   ```

2. **Set reasonable time windows** (1-4 hours) for each extraction

3. **Monitor watermarks** to prevent gaps in data:
   ```python
   last_watermark = ingester.get_last_watermark("llm_interactions")
   lag = datetime.utcnow() - last_watermark
   if lag > timedelta(hours=2):
       logger.warning(f"Data lag detected: {lag}")
   ```

4. **Test with small datasets** before scaling to production

5. **Use dead letter queue** to track failed records:
   ```python
   ingester.save_dead_letter_queue()
   ```

## Next Steps

1. **Deploy monitoring**: Set up Prometheus alerts for pipeline health
2. **Create dbt models**: Build intermediate and dimension models
3. **Implement streaming**: Add real-time extraction with Redis Streams
4. **Optimize queries**: Add materialized views for common aggregations

## Files Created

| File | Purpose |
|------|---------|
| `backend-api/app/services/data_pipeline/database_connector.py` | Database abstraction layer |
| `backend-api/scripts/create_pipeline_tables.sql` | PostgreSQL schema |
| `backend-api/scripts/create_pipeline_tables_sqlite.sql` | SQLite schema |
| `backend-api/scripts/test_pipeline_connector.py` | Test suite |
| `backend-api/requirements-pipeline.txt` | Updated dependencies |

## Support

For issues or questions:
- Check logs: `backend-api/logs/chimera.log`
- Review documentation: `docs/PIPELINE_DEPLOYMENT_GUIDE.md`
- Open GitHub issue with error details
