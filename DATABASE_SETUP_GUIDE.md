# Chimera Database Setup Guide

This guide covers database setup, migrations, and schema management for the Chimera platform.

## Quick Start

### 1. Check Database Connection

```bash
cd backend-api
py scripts/setup_database.py --check
```

### 2. Run Migrations

```bash
cd backend-api
alembic upgrade head
```

### 3. Seed Development Data (Optional)

```bash
cd backend-api
py scripts/setup_database.py --seed
```

## Database Configuration

### Environment Variables

Configure your database connection in `backend-api/.env`:

```env
# SQLite (Development)
DATABASE_URL=sqlite:///./chimera.db

# PostgreSQL (Production - Recommended)
DATABASE_URL=postgresql://user:password@localhost:5432/chimera

# PostgreSQL with connection pooling
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/chimera?pool_size=10&max_overflow=20
```

### Production Requirements

⚠️ **SQLite is NOT supported in production** due to concurrency limitations.

For production deployments, use PostgreSQL 14+ with:

- Connection pooling
- Proper backup strategy
- Read replicas for analytics queries
- Partitioning for telemetry tables

## Database Schema Overview

### Core Tables

#### User Management

- **users** - User accounts with role-based access control
- **user_api_keys** - Per-user API keys for programmatic access
- **user_sessions** - JWT session management
- **user_model_preferences** - User's preferred AI models

#### Campaign Management

- **campaigns** - Research campaigns for adversarial testing
- **campaign_shares** - Campaign sharing and collaboration
- **campaign_executions** - Execution results and telemetry
- **campaign_analytics** - Pre-computed analytics aggregates

#### Prompt Library

- **prompt_templates** - Community-shared prompt templates
- **template_ratings** - User ratings and reviews
- **template_usage** - Template usage tracking

#### Provider Management

- **llm_providers** - LLM provider configurations
- **llm_models** - Available models per provider
- **user_provider_keys** - User's encrypted API keys
- **model_usage_records** - Usage tracking for billing

#### Telemetry & Analytics

- **telemetry_events** - Real-time event stream (partitioned)
- **campaign_analytics** - Aggregated campaign metrics
- **technique_performance** - Technique effectiveness stats

#### Audit & Compliance

- **audit_log** - Comprehensive audit trail with hash chain
- **config_audit** - Configuration change tracking
- **data_retention_policies** - Automated data lifecycle

### Key Features

#### 1. Multi-User Authentication

- Email/password authentication
- Email verification workflow
- Password reset tokens
- Role-based access control (Admin, Researcher, Viewer)
- Per-user API keys with scopes

#### 2. Campaign Collaboration

- Private, team, and public campaigns
- Explicit user sharing with permissions (view/edit)
- Real-time telemetry streaming
- Cost tracking and limits

#### 3. Prompt Library

- Community template sharing
- Version control for templates
- Effectiveness ratings and reviews
- Tag-based categorization
- Full-text search

#### 4. Performance Optimization

- Strategic indexes for common queries
- Partitioned telemetry tables (monthly)
- Pre-computed analytics aggregates
- Connection pooling support

#### 5. Security & Compliance

- Encrypted API key storage
- Tamper-evident audit log (hash chain)
- Row-level security policies
- Data retention automation

## Migration Management

### Create a New Migration

```bash
cd backend-api
alembic revision --autogenerate -m "Description of changes"
```

### Review Generated Migration

Always review auto-generated migrations before applying:

```bash
# Check the latest migration file in backend-api/alembic/versions/
```

### Apply Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to specific version
alembic downgrade <revision_id>
```

### Check Current Version

```bash
alembic current
```

### View Migration History

```bash
alembic history --verbose
```

## Common Tasks

### Reset Database (Development Only)

```bash
cd backend-api
py scripts/setup_database.py --reset --seed
```

### Create Admin User

```python
from app.db.models import User, UserRole
from app.core.security import get_password_hash
from sqlalchemy.orm import Session

user = User(
    email="admin@example.com",
    username="admin",
    hashed_password=get_password_hash("secure_password"),
    role=UserRole.ADMIN,
    is_active=True,
    is_verified=True
)
session.add(user)
session.commit()
```

### Backup Database

```bash
# SQLite
cp chimera.db chimera.db.backup

# PostgreSQL
pg_dump -U user -d chimera > chimera_backup.sql
```

### Restore Database

```bash
# SQLite
cp chimera.db.backup chimera.db

# PostgreSQL
psql -U user -d chimera < chimera_backup.sql
```

## Schema Validation

### Check for Missing Indexes

```sql
-- PostgreSQL: Find tables without indexes
SELECT tablename 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename NOT IN (
    SELECT tablename 
    FROM pg_indexes 
    WHERE schemaname = 'public'
);
```

### Check Foreign Key Constraints

```sql
-- PostgreSQL: List all foreign keys
SELECT
    tc.table_name, 
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name 
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY';
```

## Performance Tuning

### PostgreSQL Configuration

For production deployments, tune PostgreSQL settings:

```ini
# postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
```

### Index Maintenance

```sql
-- Rebuild indexes (PostgreSQL)
REINDEX DATABASE chimera;

-- Analyze tables for query planner
ANALYZE;

-- Vacuum to reclaim space
VACUUM ANALYZE;
```

## Monitoring

### Check Database Size

```sql
-- PostgreSQL
SELECT pg_size_pretty(pg_database_size('chimera'));

-- SQLite
SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();
```

### Monitor Active Connections

```sql
-- PostgreSQL
SELECT count(*) FROM pg_stat_activity WHERE datname = 'chimera';
```

### Slow Query Log

```sql
-- PostgreSQL: Enable slow query logging
ALTER DATABASE chimera SET log_min_duration_statement = 1000; -- 1 second
```

## Troubleshooting

### Migration Conflicts

If you encounter migration conflicts:

```bash
# Check for multiple heads
alembic heads

# Merge heads
alembic merge heads -m "Merge migration branches"

# Apply merged migration
alembic upgrade head
```

### Database Locked (SQLite)

SQLite locks during writes. For development:

- Use `--check-same-thread=False` in connection string
- Consider switching to PostgreSQL for multi-user testing

### Connection Pool Exhausted

Increase pool size in DATABASE_URL:

```env
DATABASE_URL=postgresql://user:pass@host/db?pool_size=20&max_overflow=40
```

## Security Best Practices

1. **Never commit database credentials** - Use environment variables
2. **Encrypt API keys at rest** - Use `ENCRYPT_API_KEYS_AT_REST=true`
3. **Enable audit logging** - Track all sensitive operations
4. **Regular backups** - Automate daily backups with retention
5. **Use read replicas** - Separate analytics from transactional queries
6. **Implement data retention** - Automatically purge old telemetry data

## Testing

### Run Database Tests

```bash
cd backend-api
pytest tests/test_database.py -v
```

### Test Migrations

```bash
# Test upgrade
alembic upgrade head

# Test downgrade
alembic downgrade -1

# Test upgrade again
alembic upgrade head
```

## Production Deployment Checklist

- [ ] PostgreSQL 14+ configured
- [ ] Connection pooling enabled
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting configured
- [ ] Read replicas for analytics (optional)
- [ ] Partitioning strategy for telemetry tables
- [ ] Data retention policies configured
- [ ] SSL/TLS enabled for database connections
- [ ] Database credentials in secure vault
- [ ] Migration rollback plan documented

## Additional Resources

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Database Design Best Practices](https://www.postgresql.org/docs/current/ddl.html)
