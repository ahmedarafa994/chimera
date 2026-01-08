# Technical Specification: Analytics and Compliance

Date: 2026-01-02
Author: BMAD USER
Epic ID: Epic 4
Status: Draft

---

## Overview

Epic 4 implements a production-grade data pipeline with Airflow orchestration, Delta Lake storage, Great Expectations validation, and compliance reporting for research tracking and regulatory requirements. This epic enables enterprise-grade analytics, quality assurance, and regulatory compliance without impacting API performance.

## Objectives and Scope

**Objectives:**
- Implement Airflow DAG orchestration for hourly ETL pipelines
- Build batch ingestion service with watermark tracking
- Deploy Delta Lake manager with ACID transactions and time travel
- Integrate Great Expectations for data quality validation (99%+ pass rate)
- Create analytics dashboard with key metrics and visualizations
- Deliver automated compliance reporting with audit trails

**Scope:**
- 6 user stories covering Airflow orchestration, batch ingestion, Delta Lake management, Great Expectations validation, analytics dashboard, and compliance reporting
- Hourly ETL pipeline with 10-minute SLA
- Delta Lake on S3 for analytics storage
- Great Expectations validation suites
- Analytics dashboard with drill-down capability
- Compliance reports with data lineage

**Out of Scope:**
- Real-time metrics (use Redis + hourly batch hybrid approach)
- ML model training (analytics only, not predictive modeling)
- Real-time alerting (batch notifications only)

## System Architecture Alignment

Epic 4 implements the **Data Pipeline Layer** from the solution architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Pipeline Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Airflow Orchestration                          │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  chimera_etl_hourly DAG                              │  │ │
│  │  │  Schedule: Hourly (@hourly)                          │  │ │
│  │  │  SLA: 10 minutes                                     │  │ │
│  │  │  Tasks: Extract → Validate → Transform → Optimize   │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              ETL Pipeline Stages                           │ │
│  │                                                             │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │ │
│  │  │   Batch     │   │   Great     │   │    dbt      │      │ │
│  │  │ Ingestion   │──▶│ Expectations│──▶│ Transform   │      │ │
│  │  │             │   │ Validation  │   │             │      │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘      │ │
│  │       │                                     │               │ │
│  │       ▼                                     ▼               │ │
│  │  ┌─────────────┐                   ┌─────────────┐          │ │
│  │  │  Delta Lake │                   │  Analytics  │          │ │
│  │  │  Storage    │◀──────────────────│   Marts     │          │ │
│  │  │  (S3)       │                   │             │          │ │
│  │  └─────────────┘                   └─────────────┘          │ │
│  │       │                                     │               │ │
│  │       ▼                                     ▼               │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │ │
│  │  │   File      │   │ Prometheus  │   │ Compliance  │      │ │
│  │  │Optimization │   │  Metrics    │   │  Reports    │      │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Hybrid Real-Time + Batch                      │ │
│  │  ┌─────────────────┐         ┌─────────────────┐          │ │
│  │  │  Redis (Real-   │         │  Delta Lake     │          │ │
│  │  │   Time Metrics)│         │  (Hourly Batch) │          │ │
│  │  │  • Recent data  │         │  • Historical   │          │ │
│  │  │  • Sub-second   │         │  • Time travel  │          │ │
│  │  └─────────────────┘         └─────────────────┘          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions Referenced:**
- **ADR-001**: Monolithic Full-Stack Architecture
- **ADR-004**: ETL Data Pipeline with Separate Analytics Storage

## Detailed Design

### Services and Modules

**Backend Pipeline Services:**

1. **`app/services/data_pipeline/batch_ingestion.py`** - Batch Ingestion Service
   ```python
   class BatchIngestionService:
       watermark_manager: WatermarkManager
       schema_validator: SchemaValidator
       dead_letter_queue: DeadLetterQueue

       async def extract_batch(self, since_watermark: datetime) -> List[Record]: ...
       async def validate_batch(self, records: List[Record]) -> ValidationResult: ...
       async def write_parquet(self, records: List[Record], partition: Partition) -> str: ...
       async def update_watermark(self, new_watermark: datetime): ...
   ```

2. **`app/services/data_pipeline/delta_lake_manager.py`** - Delta Lake Manager
   ```python
   class DeltaLakeManager:
       table_path: str  # S3 path

       async def write(self, data: DataFrame, mode: str = "append"): ...
       async def time_travel(self, as_of: datetime) -> DataFrame: ...
       async def optimize_zorder(self, columns: List[str]): ...
       async def vacuum(self, retain_hours: int = 168): ...  # 7 days default
       async def get_history(self) -> List[DeltaTableVersion]: ...
   ```

3. **`app/services/data_pipeline/data_quality.py`** - Great Expectations Integration
   ```python
   class DataQualityService:
       ge_suite: ExpectationSuite

       async def validate_batch(self, data: DataFrame) -> ValidationResults: ...
       async def get_expectations(self) -> List[Expectation]: ...
       async def update_expectations(self, expectations: List[Expectation]): ...
       async def get_validation_results(self, run_id: str) -> ValidationResults: ...
   ```

4. **`airflow/dags/chimera_etl_hourly.py`** - Airflow DAG
   ```python
   @dag(
       dag_id="chimera_etl_hourly",
       schedule="@hourly",
       start_date=days_ago(1),
       sla_miss timedelta(hours=1),
       max_active_runs=1,
       catchup=False,
   )
   def chimera_etl_hourly():
       extract_task = extract_data()
       validate_task = validate_data(extract_task.output)
       transform_task = dbt_run(validate_task.output)
       optimize_task = optimize_files(transform_task.output)
       notify_task = send_notification([optimize_task.output])
   ```

5. **`dbt/chimera/models/`** - dbt Transformations
   - `staging/` - Deduplication and validation
     - `stg_generations.sql` - Generation events staging
     - `stg_providers.sql` - Provider metrics staging
     - `stg_transformations.sql` - Transformation usage staging
   - `marts/` - Analytics-ready dimensional models
     - `dim_providers.sql` - Provider dimension
     - `fact_generations.sql` - Generation fact table
     - `fact_jailbreak_tests.sql` - Jailbreak testing fact
   - `aggregations/` - Pre-computed metrics
     - `agg_provider_metrics.sql` - Hourly provider metrics
     - `agg_technique_effectiveness.sql` - Technique success rates

6. **`app/api/v1/endpoints/analytics.py`** - Analytics API Endpoints
   - `GET /api/v1/analytics/metrics` - Key metrics summary
   - `GET /api/v1/analytics/providers` - Provider analytics
   - `GET /api/v1/analytics/techniques` - Technique effectiveness
   - `GET /api/v1/analytics/timeseries` - Time series data
   - `POST /api/v1/analytics/reports` - Generate compliance report

7. **`app/services/compliance_report_service.py`** - Compliance Reporting
   ```python
   class ComplianceReportService:
       async def generate_usage_report(self, period: DateRange) -> ComplianceReport: ...
       async def generate_data_lineage(self, record_id: str) -> DataLineage: ...
       async def generate_access_logs(self, period: DateRange) -> AccessLog: ...
       async def export_report(self, report: ComplianceReport, format: ExportFormat): bytes: ...
   ```

**Frontend Components:**

1. **`src/app/dashboard/analytics/page.tsx`** - Analytics Dashboard
   - Key metrics cards (requests, success rate, provider usage)
   - Date range filter with presets
   - Charts and visualizations (Recharts)
   - Drill-down capability for detailed data

2. **`src/components/analytics/MetricsCards.tsx`** - Key Metrics Display
   - Request count and trend
   - Success rate percentage
   - Average latency
   - Provider usage distribution

3. **`src/components/analytics/ChartContainer.tsx`** - Chart Components
   - Line chart for time series
   - Bar chart for provider comparison
   - Heatmap for usage patterns
   - Pie chart for distribution

### Data Models and Contracts

**Pipeline Data Models (Pydantic):**

```python
class GenerationEvent(BaseModel):
    event_id: str
    timestamp: datetime
    provider: str
    model: str
    prompt_length: int
    response_length: int
    latency_ms: float
    success: bool
    error_code: Optional[str]
    techniques_applied: List[str]
    tokens_used: int
    cost_usd: float

class ProviderMetrics(BaseModel):
    provider_id: str
    hour_partition: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tokens: int
    total_cost_usd: float

class ValidationResult(BaseModel):
    run_id: str
    timestamp: datetime
    table_name: str
    row_count: int
    passed: int
    failed: int
    pass_rate: float
    expectation_results: List[ExpectationResult]

class ComplianceReport(BaseModel):
    report_id: str
    period_start: datetime
    period_end: datetime
    sections: List[ReportSection]
    generated_at: datetime
    data_lineage: List[LineageRecord]
    audit_trail: List[AuditRecord]

class AnalyticsMetric(BaseModel):
    metric_name: str
    value: float
    change_percent: Optional[float] = None
    period: str
```

**Delta Lake Schema:**

```sql
-- Generations Fact Table
CREATE TABLE fact_generations (
    event_id STRING PRIMARY KEY,
    timestamp TIMESTAMP,
    provider STRING,
    model STRING,
    prompt_length INT,
    response_length INT,
    latency_ms FLOAT,
    success BOOLEAN,
    error_code STRING,
    techniques_applied ARRAY<STRING>,
    tokens_used INT,
    cost_usd FLOAT,
    hour_partition DATE,
    hour INT
) PARTITIONED BY (hour_partition, hour);

-- Provider Dimension Table
CREATE TABLE dim_providers (
    provider_id STRING PRIMARY KEY,
    provider_name STRING,
    base_url STRING,
    models ARRAY<STRING>,
    enabled BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Jailbreak Testing Fact Table
CREATE TABLE fact_jailbreak_tests (
    test_id STRING PRIMARY KEY,
    timestamp TIMESTAMP,
    test_type STRING,  -- 'autodan' or 'gptfuzz'
    target_provider STRING,
    target_model STRING,
    iterations INT,
    successful_mutations INT,
    asr_rate FLOAT,  -- Attack Success Rate
    best_prompt STRING,
    hour_partition DATE,
    hour INT
) PARTITIONED BY (hour_partition, hour);
```

**Frontend TypeScript Types:**

```typescript
interface AnalyticsMetrics {
  totalRequests: number;
  successRate: number;
  avgLatency: number;
  providerUsage: ProviderUsage[];
}

interface TimeSeriesData {
  timestamp: Date;
  value: number;
  label?: string;
}

interface ProviderUsage {
  provider: string;
  requestCount: number;
  percentage: number;
  avgLatency: number;
}

interface ComplianceReport {
  reportId: string;
  periodStart: Date;
  periodEnd: Date;
  sections: ReportSection[];
  generatedAt: Date;
}
```

### APIs and Interfaces

**Analytics API Endpoints:**

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/analytics/metrics` | Key metrics summary (requests, success rate, latency) | API Key |
| GET | `/api/v1/analytics/providers` | Provider analytics with comparison | API Key |
| GET | `/api/v1/analytics/techniques` | Technique effectiveness metrics | API Key |
| GET | `/api/v1/analytics/timeseries` | Time series data for charts | API Key |
| GET | `/api/v1/analytics/reports` | List available compliance reports | API Key |
| POST | `/api/v1/analytics/reports` | Generate new compliance report | API Key |
| GET | `/api/v1/analytics/reports/{id}` | Get specific compliance report | API Key |
| GET | `/api/v1/analytics/data-quality` | Data quality validation results | API Key |

**Analytics Query Parameters:**

```
GET /api/v1/analytics/timeseries?
    metric=latency_ms&
    aggregation=avg&
    start_date=2026-01-01&
    end_date=2026-01-02&
    interval=hour&
    provider=google
```

**Compliance Report Sections:**

| Section | Description | Data Sources |
|---------|-------------|--------------|
| Usage Summary | Total requests, by provider, by model | fact_generations |
| Retention Status | Data retention compliance, archival status | Delta Lake metadata |
| Access Logs | API access by user, endpoint, timestamp | API access logs |
| Data Lineage | Record provenance and transformation history | Lineage tracking |
| Audit Trail | All changes to data and configurations | Audit log |

### Workflows and Sequencing

**Airflow DAG Execution Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  Hourly Trigger (e.g., 2026-01-02 14:00:00)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 1: Extract Batch                                      │
│  • Read last watermark from PostgreSQL                       │
│  • Extract records since watermark (e.g., 13:00-14:00)      │
│  • Pull from: generations, jailbreak_tests, provider_health │
│  • Output: List of records                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 2: Validate Batch                                     │
│  • Run Great Expectations validation suite                  │
│  • Check: nulls, ranges, types, distributions               │
│  • Route invalid records to dead letter queue                │
│  • Ensure 99%+ pass rate                                     │
│  • Alert if pass rate below threshold                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 3: Write to Delta Lake                                 │
│  • Write valid records to Parquet with partitioning          │
│  • Partition by date (hour_partition) and hour               │
│  • Use ACID transactions for consistency                     │
│  • Update watermark to latest timestamp                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 4: dbt Transform (parallel with Task 3)               │
│  • Run dbt models: staging → marts → aggregations           │
│  • Create optimized tables for analytics queries            │
│  • Pre-compute provider metrics and technique effectiveness  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 5: Optimize Delta Lake                                 │
│  • Run Z-order clustering on frequently queried columns      │
│  • Compact small files into larger files                     │
│  • Vacuum old files beyond retention period (7 days default) │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 6: Update Prometheus Metrics                          │
│  • Push latest metrics to Prometheus pushgateway             │
│  • Metrics: request counts, error rates, provider usage     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 7: Send Notifications                                 │
│  • Send success notification if SLA met (<10 min)            │
│  • Send alert if SLA missed or quality issues detected       │
└─────────────────────────────────────────────────────────────┘
```

**Data Quality Validation Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  Batch data extracted and ready for validation              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Load Great Expectations Expectation Suite                  │
│  Suite: chimera_batch_validation                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Run Expectations (parallel where possible)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Expectation 1: column 'provider_id' must not be null │   │
│  │  Expectation 2: 'latency_ms' must be between 0 and   │   │
│  │                   10000                                │   │
│  │  Expectation 3: 'success' must be boolean            │   │
│  │  Expectation 4: 'tokens_used' must be >= 0           │   │
│  │  Expectation 5: no duplicate 'event_id'              │   │
│  │  Expectation 6: 'timestamp' within expected range    │   │
│  │  ... (20+ expectations)                               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Calculate Validation Results                               │
│  • Total rows: N                                           │
│  • Passed expectations: P                                  │
│  • Failed expectations: F                                   │
│  • Pass rate: P / (P + F)                                   │
│  • Target: 99%+ pass rate                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                ┌────────┴────────┐
                │                 │
            Pass Rate          Pass Rate
            >= 99%             < 99%
                │                 │
                ▼                 ▼
        ┌───────────────┐   ┌───────────────┐
        │ Continue to   │   │ Send Alert    │
        │ Delta Lake    │   │ Route to DLQ  │
        └───────────────┘   └───────────────┘
```

**Compliance Report Generation Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  User or scheduled job requests compliance report            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Determine report scope and period                          │
│  • Period: daily, weekly, monthly, custom                   │
│  • Sections: usage, retention, access_logs, data_lineage    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Section 1: Usage Summary                                   │
│  • Query fact_generations for period                        │
│  • Aggregate by provider, model, technique                  │
│  • Calculate totals, averages, trends                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Section 2: Retention Status                                 │
│  • Query Delta Lake history for oldest records              │
│  • Check retention compliance (e.g., 90-day minimum)        │
│  • List records pending archival or deletion                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Section 3: Access Logs                                      │
│  • Query API access logs for period                         │
│  • Aggregate by user, endpoint, status                      │
│  • List failed authentication attempts                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Section 4: Data Lineage                                     │
│  • Query lineage tracking for sample records                │
│  • Trace data source, transformations, storage              │
│  • Document schema changes and migrations                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Section 5: Audit Trail                                      │
│  • Query audit log for configuration changes                 │
│  • Document user actions, system events                     │
│  • Include timestamps and user attribution                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Assemble Report and Export                                 │
│  • Combine all sections into single document                │
│  • Format according to template (HTML, PDF)                 │
│  • Apply access controls and permissions                     │
│  • Store in secure location with versioning                 │
└─────────────────────────────────────────────────────────────┘
```

## Non-Functional Requirements

### Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Batch Ingestion Time | <5 min | Per hour of data processing |
| Delta Lake Write Throughput | 10K rows/sec | Parquet write speed |
| dbt Transform Time | <3 min | Full model run |
| Analytics Query Response | <2s | Typical dashboard query |
| Report Generation Time | <30s | Compliance report generation |

**Performance Optimization:**
- Parquet file format for columnar storage
- Z-order clustering for query optimization
- Pre-computed aggregations in dbt models
- Query result caching in Redis
- Materialized views for frequently accessed metrics

### Security

| Aspect | Implementation |
|--------|----------------|
| Data Encryption | S3 encryption at rest (AES-256) |
| Access Control | Role-based access for analytics data |
| Audit Logging | All data access logged with user attribution |
| Data Retention | Configurable retention policies |
| PII Handling | Data masking for sensitive information |

**Security Considerations:**
- Analytics data isolated from production operational DB
- Compliance reports access-controlled
- S3 bucket policies restrict access
- Encryption in transit (TLS 1.3)

### Reliability/Availability

| Metric | Target | Mechanism |
|--------|--------|-----------|
| Pipeline Success Rate | 99.5% | DAG retries and alerting |
| Data Freshness | <1 hour lag | Hourly batch schedule |
| Data Quality Pass Rate | 99%+ | Great Expectations validation |
| Report Generation | 100% | Retry with backoff |

**Reliability Features:**
- Airflow automatic retries with exponential backoff
- Dead letter queue for invalid records
- Data quality alerts before SLA breach
- Pipeline health monitoring and alerting

### Observability

| Aspect | Implementation |
|--------|----------------|
| Pipeline Metrics | Airflow metrics, Prometheus integration |
| Data Quality Metrics | Great Expectations validation results |
| Performance Metrics | DAG duration, task timing |
| Business Metrics | Request counts, provider usage, ASR rates |

**Observable Metrics:**
- `airflow_dag_duration_seconds` - DAG execution time
- `airflow_task_success_count` - Task success rate
- `data_quality_pass_rate` - GE validation pass rate
- `delta_lake_row_count` - Total rows by table
- `analytics_query_duration_seconds` - Query response time

## Dependencies and Integrations

**Internal Dependencies:**
- Epic 1 (Multi-Provider Foundation) - Provider usage metrics
- Epic 2 (Advanced Transformation Engine) - Transformation effectiveness
- Epic 3 (Real-Time Research Platform) - User interaction metrics

**External Dependencies:**

| Dependency | Version | Purpose |
|------------|---------|---------|
| Apache Airflow | 2.7+ | Pipeline orchestration |
| Delta Lake | 0.10+ | ACID transactions on S3 |
| Great Expectations | 0.18+ | Data quality validation |
| dbt | 1.5+ | Data transformation |
| PostgreSQL | 14+ | Operational DB and Airflow metadata |
| AWS S3 | - | Analytics storage |
| Prometheus | 2.45+ | Metrics and alerting |

**Python Library Dependencies:**
- `pandas` - Data manipulation
- `pyarrow` - Parquet read/write
- `deltalake` - Delta Lake operations
- `great-expectations` - Data quality
- `sqlalchemy` - Database operations
- `psycopg2` - PostgreSQL adapter

## Acceptance Criteria (Authoritative)

### Story AC-001: Airflow DAG Orchestration
- [ ] DAG executes hourly with configurable schedule
- [ ] DAG includes tasks: extraction, validation, dbt transformation, optimization
- [ ] Tasks run in parallel where dependencies allow
- [ ] Failures trigger retries with exponential backoff
- [ ] SLA of 10 minutes for pipeline completion
- [ ] DAG includes success and failure notifications
- [ ] Task logs available for debugging
- [ ] DAG pausable and manually triggerable

### Story AC-002: Batch Ingestion Service
- [ ] Service extracts data since last watermark
- [ ] Data validated against schema definitions
- [ ] Invalid data routed to dead letter queue
- [ ] Valid data written to Parquet with date/hour partitioning
- [ ] Watermark updated after successful processing
- [ ] Processing handles late-arriving data
- [ ] Job logs metrics and errors
- [ ] Job completes within SLA (5 minutes target)

### Story AC-003: Delta Lake Manager
- [ ] Writes atomic with ACID guarantees
- [ ] Time travel queries access historical data
- [ ] Z-order clustering optimizes query performance
- [ ] File optimization runs automatically
- [ ] Vacuum operations clean up old files
- [ ] Schema evolution handles schema changes
- [ ] Operations maintain data consistency
- [ ] Performance meets query benchmarks

### Story AC-004: Great Expectations Validation
- [ ] Validation suites run automatically
- [ ] Expectations check: nulls, ranges, types, distributions
- [ ] Pass rate 99%+ for production data
- [ ] Failures trigger alerts and prevent bad data
- [ ] Validation results logged and tracked
- [ ] Expectations version controlled
- [ ] New expectations addable via configuration
- [ ] Validation runs within performance targets

### Story AC-005: Analytics Dashboard
- [ ] Dashboard shows key metrics: requests, success rates, provider usage
- [ ] Dashboard supports date range filtering
- [ ] Dashboard shows visualizations: charts, graphs, heatmaps
- [ ] Dashboard supports drill-down into detailed data
- [ ] Dashboard updates with near real-time data
- [ ] Dashboard supports export of reports
- [ ] Dashboard responsive and performant
- [ ] Dashboard shows compliance status and alerts

### Story AC-006: Compliance Reporting
- [ ] Reports include: data usage, retention, access logs
- [ ] Reports support configurable time periods
- [ ] Reports generated on schedule (daily, weekly, monthly)
- [ ] Reports exportable to standard formats (PDF, CSV)
- [ ] Reports include audit trail of changes
- [ ] Reports show data lineage and provenance
- [ ] Reports support custom sections and metrics
- [ ] Reports securely stored and access-controlled

## Traceability Mapping

**Requirements from PRD:**

| PRD Requirement | Epic 4 Story | Implementation |
|----------------|--------------|----------------|
| FR-10: Production-grade analytics pipeline | AC-001, AC-002, AC-003 | Airflow + Delta Lake |
| FR-11: Compliance and audit reporting | AC-006 | Compliance report service |
| NFR-01: <100ms API response | N/A | Analytics queries don't impact API |
| NFR-09: Data quality validation | AC-004 | Great Expectations |
| NFR-10: Audit trail for all transformations | AC-006 | Audit logging in reports |

**Epic-to-Architecture Mapping:**

| Architecture Component | Epic 4 Implementation |
|-----------------------|----------------------|
| Data Pipeline Layer | All stories (AC-001 through AC-006) |
| Airflow Orchestration | AC-001 |
| Batch Ingestion | AC-002 |
| Delta Lake Storage | AC-003 |
| Data Quality | AC-004 |
| Analytics Dashboard | AC-005 |
| Compliance Reporting | AC-006 |

## Risks, Assumptions, Open Questions

**Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Airflow infrastructure complexity | Medium | Use managed Airflow (MWAA, Cloud Composer) |
| Delta Lake compatibility issues | Medium | Thorough testing, version pinning |
| Great Expectations validation too strict | Medium | Configurable thresholds, tuning period |
| Pipeline SLA breaches due to data volume | High | Scalability testing, auto-scaling |

**Assumptions:**
- S3 storage cost acceptable for analytics data
- Airflow infrastructure can be deployed (self-managed or cloud)
- Delta Lake ecosystem stable and well-supported
- Great Expectations learning curve manageable for team

**Open Questions:**
- What is the data retention period for analytics? → **Decision: 90 days default, configurable per table**
- Should failed validation records be completely discarded? → **Decision: Route to DLQ for investigation, retain for 7 days**
- How often should compliance reports be generated? → **Decision: Weekly automatic, on-demand available**

## Test Strategy Summary

**Unit Tests:**
- Watermark manager logic
- Schema validation rules
- Data quality expectation logic
- Delta Lake write/read operations
- Compliance report section generation

**Integration Tests:**
- End-to-end Airflow DAG execution
- Great Expectations validation with real data
- Delta Lake time travel queries
- dbt model execution and results
- Analytics API endpoint responses

**End-to-End Tests:**
- Full pipeline execution (extract → validate → write → transform)
- Analytics dashboard queries against real data
- Compliance report generation and export
- Time travel query for historical data
- Dead letter queue handling

**Performance Tests:**
- Pipeline throughput (rows per second)
- Query performance for dashboard metrics
- Report generation time for large date ranges
- Delta Lake optimization performance
- Concurrent query handling

**Data Quality Tests:**
- Great Expectations suite validation
- Schema enforcement testing
- Data completeness checks
- Accuracy validation (spot checks)
- Consistency checks across tables

**Test Coverage Target:** 75%+ for pipeline services, 70%+ for dbt models

---

_This technical specification serves as the implementation guide for Epic 4: Analytics and Compliance. All development should reference this document for detailed design decisions, data pipeline architecture, and acceptance criteria._
