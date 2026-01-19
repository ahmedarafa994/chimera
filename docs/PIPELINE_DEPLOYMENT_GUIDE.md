# Chimera Data Pipeline - Deployment & Operations Guide

## Quick Start

This guide provides step-by-step instructions for deploying and operating the Chimera data pipeline in production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment](#deployment)
5. [Operations](#operations)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Disaster Recovery](#disaster-recovery)

## Prerequisites

### System Requirements

**Minimum (Development)**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- OS: Ubuntu 22.04 LTS, macOS 12+, Windows 11 with WSL2

**Recommended (Production)**:
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 500 GB SSD (data lake) + 100 GB (system)
- OS: Ubuntu 22.04 LTS

### Software Dependencies

```bash
# Python 3.11+
python --version  # Should be >= 3.11

# PostgreSQL 14+ (for Airflow metadata)
psql --version

# Redis 7+ (for streaming and caching)
redis-cli --version

# Java 11+ (for Spark/Delta Lake)
java -version
```

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/chimera.git
cd chimera
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend-api
pip install -r requirements.txt

# Install data pipeline dependencies
pip install -r requirements-pipeline.txt

# Verify installations
pip list | grep -E "pandas|pyarrow|delta-spark|great-expectations|dbt-core|apache-airflow"
```

**requirements-pipeline.txt**:
```
# Data Processing
pandas==2.1.4
pyarrow==14.0.1
delta-spark==3.0.0
pyspark==3.5.0

# Data Quality
great-expectations==0.18.8

# Transformation
dbt-core==1.7.4
dbt-duckdb==1.7.1

# Orchestration
apache-airflow==2.8.1
apache-airflow-providers-dbt-cloud==3.5.1

# Monitoring
prometheus-client==0.19.0
```

### Step 3: Setup Airflow

```bash
# Set Airflow home
export AIRFLOW_HOME=/opt/airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@chimera.ai \
    --password <secure-password>

# Copy DAGs
cp -r airflow/dags/* $AIRFLOW_HOME/dags/
```

### Step 4: Setup dbt

```bash
# Install dbt with DuckDB adapter
pip install dbt-core dbt-duckdb

# Initialize dbt project
cd dbt/chimera
dbt deps  # Install dependencies

# Test connection
dbt debug --profiles-dir .
```

### Step 5: Setup Great Expectations

```bash
# Initialize GE project
cd /data/chimera-lake
great_expectations init

# Copy expectation suites
cp -r backend-api/app/services/data_pipeline/expectations/* \
    /data/chimera-lake/great_expectations/expectations/
```

## Configuration

### Environment Variables

Create `.env.pipeline` in project root:

```bash
# ============================================================================
# Data Lake Configuration
# ============================================================================
DATA_LAKE_PATH=/data/chimera-lake
WAREHOUSE_PATH=/data/chimera-warehouse

# Storage backend: local, s3, azure, gcs
STORAGE_BACKEND=local

# S3 Configuration (if using S3)
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_DEFAULT_REGION=us-west-2
S3_BUCKET=chimera-data-lake

# ============================================================================
# Airflow Configuration
# ============================================================================
AIRFLOW_HOME=/opt/airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@localhost:5432/airflow
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
AIRFLOW__WEBSERVER__SECRET_KEY=<generate-with-openssl-rand-base64-32>

# Email alerts (optional)
AIRFLOW__EMAIL__EMAIL_BACKEND=airflow.providers.smtp.utils.emailer.send_email_smtp
AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
AIRFLOW__SMTP__SMTP_PORT=587
AIRFLOW__SMTP__SMTP_USER=alerts@chimera.ai
AIRFLOW__SMTP__SMTP_PASSWORD=<app-password>
AIRFLOW__SMTP__SMTP_MAIL_FROM=alerts@chimera.ai

# ============================================================================
# dbt Configuration
# ============================================================================
DBT_PROFILES_DIR=/opt/dbt
DBT_PROJECT_DIR=/opt/dbt/chimera
DBT_TARGET=prod

# ============================================================================
# Great Expectations
# ============================================================================
GE_ROOT_DIRECTORY=/data/chimera-lake/great_expectations
GE_ENABLE_DATA_DOCS=true

# ============================================================================
# Redis (for streaming and caching)
# ============================================================================
REDIS_URL=redis://localhost:6379/1

# ============================================================================
# Monitoring
# ============================================================================
PROMETHEUS_PUSHGATEWAY_URL=http://localhost:9091
ENABLE_PROMETHEUS_METRICS=true

# Grafana
GRAFANA_URL=http://localhost:3001
GRAFANA_API_KEY=<your-api-key>

# ============================================================================
# Data Quality
# ============================================================================
MIN_DATA_QUALITY_PASS_RATE=0.95
FAIL_ON_QUALITY_ERROR=false
ALERT_ON_QUALITY_FAILURE=true

# ============================================================================
# Optimization
# ============================================================================
ENABLE_AUTO_OPTIMIZE=true
ENABLE_AUTO_COMPACT=true
TARGET_FILE_SIZE_MB=512
DELTA_RETENTION_DAYS=30
```

### Airflow Connection Setup

```bash
# Add database connection for Chimera backend
airflow connections add 'chimera_db' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-port '5432' \
    --conn-schema 'chimera' \
    --conn-login 'chimera_user' \
    --conn-password '<password>'

# Add S3 connection (if using S3)
airflow connections add 'aws_default' \
    --conn-type 'aws' \
    --conn-login '<AWS_ACCESS_KEY_ID>' \
    --conn-password '<AWS_SECRET_ACCESS_KEY>' \
    --conn-extra '{"region_name": "us-west-2"}'
```

## Deployment

### Development Deployment

```bash
# 1. Start PostgreSQL (Airflow metadata)
sudo systemctl start postgresql

# 2. Start Redis
sudo systemctl start redis

# 3. Start Airflow services
airflow webserver -p 8080 &
airflow scheduler &

# 4. Verify DAGs are loaded
airflow dags list | grep chimera

# 5. Trigger manual test run
airflow dags trigger chimera_etl_hourly
```

### Production Deployment (Docker Compose)

**docker-compose.pipeline.yml**:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 5

  airflow-init:
    image: apache/airflow:2.8.1
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --password admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@chimera.ai
    environment:
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql://airflow:airflow@postgres:5432/airflow
    depends_on:
      - postgres

  airflow-webserver:
    image: apache/airflow:2.8.1
    command: webserver
    ports:
      - "8080:8080"
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql://airflow:airflow@postgres:5432/airflow
      AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./backend-api:/opt/chimera/backend-api
      - /data/chimera-lake:/data/chimera-lake
    depends_on:
      - postgres
      - redis
      - airflow-init

  airflow-scheduler:
    image: apache/airflow:2.8.1
    command: scheduler
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql://airflow:airflow@postgres:5432/airflow
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./backend-api:/opt/chimera/backend-api
      - /data/chimera-lake:/data/chimera-lake
    depends_on:
      - postgres
      - redis
      - airflow-init

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
```

Start production stack:
```bash
docker-compose -f docker-compose.pipeline.yml up -d
```

## Operations

### Daily Operations

#### 1. Monitor Pipeline Health

```bash
# Check Airflow DAG status
airflow dags list-runs -d chimera_etl_hourly --state failed

# View recent logs
airflow tasks logs chimera_etl_hourly extract_llm_interactions -1

# Check data freshness
dbt run --select agg_provider_metrics_hourly --profiles-dir /opt/dbt
```

#### 2. Data Quality Monitoring

```bash
# Run Great Expectations checkpoint
cd /data/chimera-lake/great_expectations
great_expectations checkpoint run llm_interactions_checkpoint

# View data docs
great_expectations docs build
```

#### 3. Optimization Tasks

```bash
# Optimize Delta tables (weekly)
python -c "
from app.services.data_pipeline.delta_lake_manager import DeltaLakeManager
mgr = DeltaLakeManager()
for table in ['llm_interactions', 'transformations', 'jailbreak_experiments']:
    mgr.optimize_table(table)
"

# Vacuum old versions (weekly)
python -c "
from app.services.data_pipeline.delta_lake_manager import DeltaLakeManager
mgr = DeltaLakeManager()
for table in mgr.list_tables():
    mgr.vacuum_table(table, retention_hours=168)
"
```

### Weekly Operations

#### 1. Review Data Quality Trends

```bash
# Generate Great Expectations data docs
cd /data/chimera-lake/great_expectations
great_expectations docs build

# Open in browser: file:///data/chimera-lake/great_expectations/uncommitted/data_docs/local_site/index.html
```

#### 2. Cost Analysis

```bash
# Check storage usage
du -sh /data/chimera-lake/*

# Estimate query costs (if using cloud)
# Review Prometheus metrics: chimera_estimated_cost_usd
```

### Monthly Operations

#### 1. Archive Old Data

```bash
# Archive data older than 90 days to cold storage
python scripts/archive_old_data.py --days 90 --target glacier
```

#### 2. Review SLAs

- Data freshness: < 1 hour
- Pipeline execution: < 10 minutes
- Query performance: < 5 seconds
- Data quality: > 99% pass rate

#### 3. Update Dependencies

```bash
# Update dbt packages
cd dbt/chimera
dbt deps --upgrade

# Update Python packages (test in dev first)
pip list --outdated
```

## Monitoring

### Grafana Setup & Configuration

#### Installation

**Option 1: Docker Compose (Recommended for Production)**

The `docker-compose.pipeline.yml` file already includes Grafana configuration:

```yaml
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_INSTALL_PLUGINS: redis-datasource
      GF_SERVER_ROOT_URL: http://localhost:3001
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
```

**Option 2: Standalone Installation**

```bash
# Download Grafana
wget https://dl.grafana.com/oss/release/grafana_10.2.2_amd64.deb
sudo apt-get install -y adduser libfontconfig1
sudo dpkg -i grafana_10.2.2_amd64.deb

# Start Grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# Access Grafana
open http://localhost:3001
# Default credentials: admin / admin
```

#### Initial Configuration

1. **Login and Change Password**

   - Navigate to `http://localhost:3001`
   - Login with `admin` / `admin`
   - Change password on first login

2. **Add Prometheus Data Source**

   Go to **Configuration → Data Sources → Add data source**:

   ```
   Name: Prometheus
   Type: Prometheus
   URL: http://prometheus:9090
   Access: Server (default)
   ```

   Click **Save & Test** to verify connectivity.

3. **Add Redis Data Source (for Real-Time Metrics)**

   Install the Redis plugin first:
   ```bash
   grafana-cli plugins install redis-datasource
   sudo systemctl restart grafana-server
   ```

   Then go to **Configuration → Data Sources → Add data source**:

   ```
   Name: Redis
   Type: Redis
   URL: redis://redis:6379
   Database: 1
   ```

#### Dashboard Provisioning (Automated Setup)

Create provisioning configuration at `monitoring/grafana/provisioning/datasources/datasources.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false

  - name: Redis
    type: redis-datasource
    access: proxy
    url: redis://redis:6379
    database: 1
    editable: false
```

Create dashboard provisioning at `monitoring/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'Chimera Dashboards'
    orgId: 1
    folder: 'Chimera'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: true
```

Restart Grafana to apply provisioning:
```bash
docker-compose restart grafana
```

#### Manual Dashboard Import

If not using provisioning, import dashboards manually:

1. Navigate to **Dashboards → Import**
2. Upload JSON file or paste JSON content
3. Select data source: `Prometheus`
4. Click **Import**

**Available Dashboards**:
- `monitoring/grafana/dashboards/chimera_pipeline_overview.json`
- `monitoring/grafana/dashboards/chimera_llm_providers.json`
- `monitoring/grafana/dashboards/chimera_jailbreak_analytics.json`
- `monitoring/grafana/dashboards/chimera_realtime_streaming.json`

### Grafana Dashboards

#### 1. Chimera Pipeline Overview Dashboard

**File**: `monitoring/grafana/dashboards/chimera_pipeline_overview.json`
**UID**: `chimera_pipeline_overview`

**Purpose**: Comprehensive monitoring of the entire Chimera data pipeline infrastructure.

**Panels**:

**Pipeline Health Overview**:
- Pipeline health status (timeseries - last 6 hours)
- Current status indicator (green/red stat)
- Request rate (req/s with trend graph)
- Ingestion lag (seconds with thresholds: <60s green, 60-300s yellow, >300s red)
- Active WebSocket connections

**Data Quality Metrics**:
- Data quality score vs test pass rate (timeseries)
- Request status distribution (pie chart: success, error, timeout)
- Quality test failures by category

**Pipeline Performance**:
- ETL execution duration (timeseries with percentiles)
- Records processing rate (records/second)
- Last ETL completion time
- Data quality pass percentage (with threshold coloring)

**Storage & Cost**:
- Delta Lake table size (timeseries)
- Parquet files created per hour
- Budget utilization percentage
- Estimated monthly cost projection

**Refresh**: 30 seconds
**Time Range**: Last 6 hours (adjustable)

#### 2. LLM Provider Metrics Dashboard

**File**: `monitoring/grafana/dashboards/chimera_llm_providers.json`
**UID**: `chimera_llm_providers`

**Purpose**: Monitor performance and usage across all LLM providers.

**Panels**:

**Request Volume**:
- Requests per provider (timeseries, stacked)
- Total request rate (stat with trend)
- Request distribution by provider (pie chart)

**Performance Metrics**:
- Latency by provider (p50, p95, p99)
- Error rate by provider (percentage)
- Timeout rate by provider

**Token Usage**:
- Total tokens consumed (timeseries)
- Tokens per provider (grouped)
- Average tokens per request
- Token cost estimation

**Provider Health**:
- Provider availability status (stat panel)
- Circuit breaker state per provider
- Active connections per provider

**Refresh**: 30 seconds
**Variables**: Provider selection (All, google, openai, anthropic, deepseek, qwen)

#### 3. Jailbreak Research Analytics Dashboard

**File**: `monitoring/grafana/dashboards/chimera_jailbreak_analytics.json`
**UID**: `chimera_jailbreak_analytics`

**Purpose**: Track jailbreak research experiments and success rates.

**Panels**:

**Experiment Overview**:
- Total experiments conducted (stat)
- Success rate percentage (with color coding)
- Experiments per framework (AutoDAN, GPTFuzz)
- Active research sessions

**Success Metrics**:
- Success rate by framework (timeseries)
- Success rate by attack method (heatmap)
- Average iterations to success
- Judge score distribution (histogram)

**Framework Performance**:
- AutoDAN success rate trends
- GPTFuzz mutation effectiveness
- Mousetrap technique success rate

**Research Trends**:
- Experiments over time (timeseries)
- Successful techniques by day
- Technique effectiveness ranking

**Refresh**: 30 seconds
**Time Range**: Last 24 hours (for research patterns)

#### 4. Real-Time Streaming Metrics Dashboard

**File**: `monitoring/grafana/dashboards/chimera_realtime_streaming.json`
**UID**: `chimera_realtime_streaming`

**Purpose**: Monitor real-time streaming infrastructure and WebSocket connections.

**Panels**:

**Stream Health**:
- Redis Streams health status
- Active stream consumers
- Stream length by type (llm_events, transformation_events, jailbreak_events)
- Consumer lag per stream

**WebSocket Metrics**:
- Active WebSocket connections (stat)
- Connection rate (connections/second)
- Disconnection rate
- Average session duration

**Real-Time Throughput**:
- Events published per second (timeseries)
- Events consumed per second
- Processing lag (milliseconds)
- Buffer utilization percentage

**TimeSeries Metrics**:
- TimeSeries write rate (operations/second)
- TimeSeries query latency
- Active time series count
- Memory usage by time series

**Refresh**: 10 seconds (for real-time monitoring)
**Data Source**: Prometheus + Redis (for stream metrics)

### Dashboard Customization

#### Adding Custom Queries

To add custom Prometheus queries to any dashboard:

1. **Edit Panel** → Click panel title → **Edit**
2. **Query** tab → Enter PromQL expression

**Useful PromQL Examples**:

```promql
# Pipeline success rate (last 1 hour)
rate(chimera_pipeline_success_total[1h]) / rate(chimera_pipeline_execution_total[1h]) * 100

# P95 latency by provider
histogram_quantile(0.95, sum by(provider, le) (rate(chimera_llm_latency_ms_bucket[5m])))

# Data quality pass rate
sum(chimera_data_quality_tests_passed_total) / sum(chimera_data_quality_tests_total) * 100

# Token usage rate
sum by(provider) (rate(chimera_llm_tokens_total[5m]))

# Jailbreak success rate by framework
sum by(framework) (rate(chimera_jailbreak_success_total[1h])) / sum by(framework) (rate(chimera_jailbreak_experiments_total[1h])) * 100

# Real-time events per second
sum by(stream) (rate(chimera_stream_events_published_total[1m]))
```

#### Creating Alerts in Grafana

1. Navigate to **Alerting → New alert rule**
2. Select dashboard and panel
3. Configure alert conditions:

**Example Alerts**:

```yaml
# Pipeline Failure Alert
Name: Pipeline High Failure Rate
Query: rate(chimera_pipeline_success_total[5m]) / rate(chimera_pipeline_execution_total[5m]) < 0.95
Condition: Failure rate > 5% for 5 minutes
Severity: Warning

# Data Quality Alert
Name: Data Quality Below Threshold
Query: chimera_data_quality_pass_rate < 0.99
Condition: Pass rate < 99% for 10 minutes
Severity: Critical

# High Latency Alert
Name: LLM High P95 Latency
Query: histogram_quantile(0.95, rate(chimera_llm_latency_ms_bucket[5m])) > 5000
Condition: P95 latency > 5000ms for 5 minutes
Severity: Warning

# Stream Lag Alert
Name: Redis Stream Consumer Lag
Query: chimera_stream_consumer_lag > 1000
Condition: Consumer lag > 1000 messages
Severity: Critical
```

4. Configure notification channels (Email, Slack, PagerDuty)
5. Set evaluation interval (e.g., every 1 minute)

### Grafana CLI Management

```bash
# List installed dashboards
grafana-cli admin dashboards list

# Export dashboard as JSON
curl -u admin:admin http://localhost:3001/api/dashboards/uid/chimera_pipeline_overview

# Import dashboard via API
curl -X POST -H "Content-Type: application/json" \
  -u admin:admin \
  -d @dashboard.json \
  http://localhost:3001/api/dashboards/import

# Restart Grafana
sudo systemctl restart grafana-server
# OR Docker
docker-compose restart grafana
```

### Prometheus Queries

```promql
# Pipeline success rate (last 24h)
rate(chimera_pipeline_success_total[24h]) / rate(chimera_pipeline_execution_total[24h])

# Data freshness
chimera_latest_interaction_lag_seconds

# Quality test pass rate
sum(chimera_data_quality_tests_passed_total) / sum(chimera_data_quality_tests_total)

# P95 latency by provider
histogram_quantile(0.95, sum by(provider, le) (rate(chimera_llm_latency_ms_bucket[5m])))
```

## Troubleshooting

### Common Issues

#### Issue: Pipeline Fails with "Table Does Not Exist"

**Symptoms**: Airflow DAG fails on dbt step
**Cause**: Tables not initialized
**Solution**:
```bash
# Run initial data load
python -m app.services.data_pipeline.batch_ingestion

# Create staging tables with dbt
dbt run --select staging.* --full-refresh
```

#### Issue: High Memory Usage During Optimization

**Symptoms**: OOM errors during Delta optimize
**Cause**: Too many small files
**Solution**:
```bash
# Increase Spark memory
export SPARK_DRIVER_MEMORY=8g
export SPARK_EXECUTOR_MEMORY=8g

# Optimize in batches
for table in $(ls /data/chimera-lake/); do
    python -c "from app.services.data_pipeline.delta_lake_manager import DeltaLakeManager; DeltaLakeManager().optimize_table('$table')"
    sleep 300  # Wait 5 minutes between tables
done
```

#### Issue: Data Quality Tests Timing Out

**Symptoms**: Great Expectations validation takes > 5 minutes
**Cause**: Large dataset, unoptimized queries
**Solution**:
```python
# Sample data for validation
df_sample = df.sample(n=100000, random_state=42)
result = validate_llm_interactions(df_sample)
```

#### Issue: Airflow DAG Not Picking Up Changes

**Symptoms**: Code changes not reflected
**Cause**: DAG file caching
**Solution**:
```bash
# Restart Airflow scheduler
pkill -f "airflow scheduler"
airflow scheduler &

# Or in Docker
docker-compose restart airflow-scheduler
```

### Log Locations

- **Airflow logs**: `/opt/airflow/logs/`
- **dbt logs**: `/opt/dbt/chimera/logs/`
- **Great Expectations**: `/data/chimera-lake/great_expectations/uncommitted/validations/`
- **Pipeline logs**: `/var/log/chimera-pipeline/`

## Disaster Recovery

### Backup Strategy

**Daily Backups**:
- Airflow metadata (PostgreSQL dump)
- dbt project and profiles
- Great Expectations configuration

**Weekly Backups**:
- Delta Lake metadata
- Watermark files
- Aggregated marts

### Recovery Procedures

#### 1. Restore from Backup

```bash
# Restore Airflow metadata
psql -U airflow -d airflow < backups/airflow_metadata_YYYYMMDD.sql

# Restore dbt project
tar -xzf backups/dbt_project_YYYYMMDD.tar.gz -C /opt/dbt/

# Restore GE configuration
tar -xzf backups/great_expectations_YYYYMMDD.tar.gz -C /data/chimera-lake/
```

#### 2. Reprocess Historical Data

```bash
# Backfill last 7 days
airflow dags backfill chimera_etl_hourly \
    --start-date 2026-01-01 \
    --end-date 2026-01-08 \
    --reset-dagruns

# Full refresh dbt models
dbt run --full-refresh --select marts.*
```

#### 3. Validate Recovery

```bash
# Check data completeness
dbt test --select marts.*

# Verify record counts
python scripts/validate_recovery.py --start-date 2026-01-01 --end-date 2026-01-08
```

### Contact Information

**Data Engineering Team**: data-team@chimera.ai
**On-Call Rotation**: PagerDuty rotation
**Escalation**: Platform Lead

## Additional Resources

- [Architecture Documentation](./DATA_PIPELINE_ARCHITECTURE.md)
- [API Documentation](../backend-api/docs/)
- [dbt Documentation](https://docs.getdbt.com/)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Delta Lake Documentation](https://docs.delta.io/)
