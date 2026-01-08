graph TD
    API[API Server] --> |Records Event| Dashboard[Monitoring Dashboard]
    Dashboard --> |Delegates| Repo[Metrics Repository]
    Repo --> |ORM Calls| Models[SQLAlchemy Models]
    Models --> |Persists| DB[(SQLite Database)]
```

### Key Components

1.  **`models.py`**: Defines the database schema using Flask-SQLAlchemy.
2.  **`repositories.py`**: Implements the `MetricsRepository` class, handling all CRUD operations and complex analytical queries (aggregations, time-series bucketing).
3.  **`monitoring_dashboard.py`**: Refactored to act as a high-level interface, delegating storage and retrieval to the repository.
4.  **`api_server.py`**: Configured to initialize the SQLite database (`chimera_logs.db`) on startup.

## 3. Database Schema

The new schema introduces four relational tables linked by a unique `request_id`:

| Table | Purpose | Key Fields |
|-------|---------|------------|
| **`request_logs`** | Audit trail for every API hit. | `request_id`, `endpoint`, `status_code`, `latency_ms`, `ip_address` |
| **`llm_usage`** | Tracks cost and performance of LLM calls. | `provider`, `model`, `tokens_used`, `cost`, `cached` |
| **`technique_usage`** | Analyzes prompt engineering effectiveness. | `technique_suite`, `potency_level`, `success`, `transformers_count` |
| **`error_logs`** | Centralized error tracking with stack traces. | `error_type`, `error_message`, `stack_trace` |

## 4. Operational Benefits

1.  **Data Persistence**: Metrics are now saved to `chimera_logs.db`. Restarting the server no longer wipes the dashboard history.
2.  **Advanced Analytics**: The Repository allows for complex SQL-based queries (e.g., "Average latency per minute over the last hour") that were difficult to implement with in-memory lists.
3.  **Audit Compliance**: Every request is logged with an IP address and timestamp, providing a full audit trail for security verification.
4.  **Scalability**: The system can easily be migrated to PostgreSQL or MySQL in production by changing the `SQLALCHEMY_DATABASE_URI` in `api_server.py`.

## 5. Verification

A comprehensive integration test suite (`verify_monitoring_db.py`) was executed to validate the upgrade.

**Test Coverage:**
*   **End-to-End Flow:** Verified that a `record_request` call cascades correctly to all four database tables.
*   **Aggregation Logic:** Verified that dashboard summaries (Success Rate, Total Cost) are correctly calculated from DB records.
*   **Health Checks:** Confirmed that error rate calculations trigger correct health status (Healthy/Degraded/Unhealthy).

**Result:** PASSED ✅

## 6. API Usage

The existing Monitoring API endpoints remain unchanged but now return persistent data:

*   `GET /api/v1/metrics` - System-wide summary.
*   `GET /api/v1/metrics/providers` - Provider performance stats.
*   `GET /api/v1/metrics/techniques` - Technique success rates.
*   `GET /health` - System health status based on recent DB error logs.

---
**System is fully operational and ready for deployment.**