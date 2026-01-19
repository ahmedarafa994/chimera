# Chimera Database Migration Strategy & Deployment Guide

## **Migration Overview**

This guide provides a comprehensive strategy for migrating the Chimera AI Research Platform from its current database schema to the optimized production-ready schema design.

## **Pre-Migration Assessment**

### **Current State Analysis**
```sql
-- Assess current schema state
SELECT
    table_name,
    pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size,
    (SELECT count(*) FROM information_schema.columns
     WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE'
ORDER BY pg_total_relation_size(table_name::regclass) DESC;

-- Check for existing data that needs preservation
SELECT
    'users' as table_name, count(*) as row_count FROM users
UNION ALL
SELECT 'campaigns', count(*) FROM campaigns
UNION ALL
SELECT 'prompt_templates', count(*) FROM prompt_templates
UNION ALL
SELECT 'audit_log', count(*) FROM audit_log;
```

### **Data Preservation Requirements**
- ✅ **User accounts and authentication data**
- ✅ **Campaign configurations and results**
- ✅ **Prompt templates and community ratings**
- ✅ **Audit logs for compliance**
- ⚠️ **Session data** (can be regenerated)
- ⚠️ **Temporary telemetry** (retain last 30 days only)

## **Migration Strategy**

### **Phase 1: Pre-Migration Setup (Estimated: 2-4 hours)**

#### **1.1 Database Backup**
```bash
# Full database backup before migration
pg_dump -Fc -h $DB_HOST -U $DB_USER -d chimera_prod > chimera_backup_$(date +%Y%m%d_%H%M%S).dump

# Verify backup integrity
pg_restore --list chimera_backup_*.dump | head -20

# Test restore on development environment
createdb chimera_migration_test
pg_restore -h localhost -U postgres -d chimera_migration_test chimera_backup_*.dump
```

#### **1.2 Schema Analysis & Validation**
```sql
-- Create migration tracking table
CREATE TABLE migration_status (
    migration_id VARCHAR(50) PRIMARY KEY,
    description TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'pending',
    error_details TEXT
);

-- Log current schema state
INSERT INTO migration_status (migration_id, description, started_at, status)
VALUES ('PRE_MIGRATION_ASSESSMENT', 'Initial schema analysis', NOW(), 'in_progress');
```

#### **1.3 Environment Preparation**
```bash
# Install required PostgreSQL extensions
psql -d chimera_prod -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql -d chimera_prod -c "CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";"
psql -d chimera_prod -c "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";"

# Verify extension availability
psql -d chimera_prod -c "SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name IN ('uuid-ossp', 'pg_trgm', 'pgcrypto');"
```

### **Phase 2: Schema Migration (Estimated: 4-8 hours)**

#### **2.1 Create New Schema Structures**
```sql
-- Execute the comprehensive schema creation
\i database_schema_design.sql

-- Verify new tables created successfully
SELECT table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name LIKE 'new_%'
ORDER BY table_name;
```

#### **2.2 Data Migration Scripts**

##### **Users Migration**
```sql
-- Migrate users table with enhanced fields
INSERT INTO users (
    id, email, username, hashed_password, role, account_status,
    is_verified, created_at, last_login, preferences
)
SELECT
    id,
    email,
    username,
    hashed_password,
    CASE
        WHEN role = 'admin' THEN 'admin'::user_role
        WHEN role = 'researcher' THEN 'researcher'::user_role
        ELSE 'viewer'::user_role
    END,
    CASE
        WHEN is_active = true THEN 'active'::account_status
        ELSE 'suspended'::account_status
    END,
    is_verified,
    created_at,
    last_login,
    COALESCE(preferences, '{}')::jsonb
FROM users_old
WHERE email IS NOT NULL AND username IS NOT NULL;

-- Update migration status
UPDATE migration_status
SET completed_at = NOW(), status = 'completed'
WHERE migration_id = 'USERS_MIGRATION';
```

##### **API Keys Migration**
```sql
-- Migrate API keys with enhanced security
INSERT INTO user_api_keys (
    user_id, name, key_prefix, hashed_key, is_active,
    last_used_at, usage_count, created_at
)
SELECT
    user_id,
    COALESCE(name, 'Migrated Key'),
    LEFT(key_prefix, 8),
    hashed_key,
    is_active,
    last_used_at,
    COALESCE(usage_count, 0),
    created_at
FROM user_api_keys_old
WHERE user_id IS NOT NULL;
```

##### **Campaigns Migration**
```sql
-- Migrate campaigns with enhanced structure
INSERT INTO campaigns (
    id, name, description, objective, created_by, status,
    target_provider, target_model, technique_suites,
    transformation_config, config, created_at
)
SELECT
    COALESCE(id, uuid_generate_v4()),
    name,
    description,
    COALESCE(objective, 'Migrated campaign objective'),
    created_by,
    CASE status
        WHEN 'draft' THEN 'draft'::campaign_status
        WHEN 'running' THEN 'running'::campaign_status
        WHEN 'completed' THEN 'completed'::campaign_status
        ELSE 'draft'::campaign_status
    END,
    target_provider,
    target_model,
    COALESCE(technique_suites, '[]')::jsonb,
    COALESCE(transformation_config, '{}')::jsonb,
    COALESCE(config, '{}')::jsonb,
    created_at
FROM campaigns_old;
```

##### **Telemetry Migration (Selective)**
```sql
-- Migrate only recent telemetry data (last 30 days)
INSERT INTO telemetry_events (
    event_id, event_type, campaign_id, user_id, payload,
    latency_ms, cost_usd, event_timestamp
)
SELECT
    COALESCE(event_id, uuid_generate_v4()),
    event_type::telemetry_event_type,
    campaign_id,
    user_id,
    COALESCE(payload, '{}')::jsonb,
    latency_ms,
    cost_usd,
    event_timestamp
FROM telemetry_events_old
WHERE event_timestamp > NOW() - INTERVAL '30 days'
  AND event_type IS NOT NULL;
```

#### **2.3 Data Validation & Integrity Checks**
```sql
-- Comprehensive data validation
WITH validation_results AS (
    SELECT 'users' as table_name, count(*) as migrated_count,
           (SELECT count(*) FROM users_old) as original_count
    UNION ALL
    SELECT 'campaigns', count(*), (SELECT count(*) FROM campaigns_old)
    FROM campaigns
    UNION ALL
    SELECT 'user_api_keys', count(*), (SELECT count(*) FROM user_api_keys_old)
    FROM user_api_keys
)
SELECT
    table_name,
    original_count,
    migrated_count,
    CASE
        WHEN migrated_count = original_count THEN '✅ Complete'
        WHEN migrated_count > 0 THEN '⚠️ Partial'
        ELSE '❌ Failed'
    END as status
FROM validation_results;

-- Check referential integrity
SELECT
    'campaign_executions -> campaigns' as check_name,
    count(*) as violations
FROM campaign_executions ce
LEFT JOIN campaigns c ON ce.campaign_id = c.id
WHERE c.id IS NULL
UNION ALL
SELECT
    'template_ratings -> users',
    count(*)
FROM template_ratings tr
LEFT JOIN users u ON tr.user_id = u.id
WHERE u.id IS NULL;
```

### **Phase 3: Index and Performance Optimization (Estimated: 2-4 hours)**

#### **3.1 Index Creation Strategy**
```sql
-- Create indexes in optimal order (smaller to larger tables)
-- This minimizes lock time on production systems

-- User-related indexes (typically smaller tables)
CREATE INDEX CONCURRENTLY idx_users_email_verified_v2
ON users(email, is_verified);

CREATE INDEX CONCURRENTLY idx_user_api_keys_active_v2
ON user_api_keys(user_id, is_active);

-- Campaign indexes (medium size)
CREATE INDEX CONCURRENTLY idx_campaigns_owner_status_v2
ON campaigns(created_by, status);

-- Analytics indexes (larger tables)
CREATE INDEX CONCURRENTLY idx_campaign_executions_campaign_time_v2
ON campaign_executions(campaign_id, started_at);

CREATE INDEX CONCURRENTLY idx_telemetry_events_type_time_v2
ON telemetry_events(event_type, event_timestamp);

-- Verify index creation
SELECT
    schemaname, tablename, indexname,
    indexdef
FROM pg_indexes
WHERE indexname LIKE '%_v2'
ORDER BY tablename, indexname;
```

#### **3.2 Performance Validation**
```sql
-- Test critical query performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.*, u.username as created_by_username
FROM campaigns c
JOIN users u ON c.created_by = u.id
WHERE c.status = 'running'
  AND c.created_at > NOW() - INTERVAL '7 days'
ORDER BY c.created_at DESC
LIMIT 20;

-- Test analytics query performance
EXPLAIN (ANALYZE, BUFFERS)
SELECT
    campaign_id,
    COUNT(*) as total_executions,
    AVG(latency_ms) as avg_latency,
    SUM(cost_usd) as total_cost
FROM campaign_executions
WHERE started_at > NOW() - INTERVAL '24 hours'
GROUP BY campaign_id
ORDER BY total_cost DESC;
```

### **Phase 4: Application Integration (Estimated: 4-6 hours)**

#### **4.1 Database Connection Updates**
```python
# Update SQLAlchemy models to match new schema
# backend-api/app/db/models_v2.py

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER)
    account_status = Column(String(20), nullable=False, default='active')
    is_verified = Column(Boolean, default=False)
    # ... additional fields as per schema
```

#### **4.2 API Endpoint Updates**
```python
# Update repository patterns to use new schema
# backend-api/app/repositories/user_repository_v2.py

class UserRepository:
    def __init__(self, db_session):
        self.db = db_session

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email with new schema structure"""
        return await self.db.query(User).filter(
            User.email == email,
            User.account_status == 'active'
        ).first()

    async def create_user(self, user_data: UserCreate) -> User:
        """Create user with enhanced validation"""
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hash_password(user_data.password),
            role=user_data.role,
            account_status='pending_verification'
        )
        self.db.add(db_user)
        await self.db.commit()
        return db_user
```

#### **4.3 Frontend Data Model Updates**
```typescript
// frontend/src/types/auth.ts - Update TypeScript interfaces

export interface User {
  id: number;
  email: string;
  username: string;
  role: 'admin' | 'researcher' | 'viewer';
  accountStatus: 'active' | 'suspended' | 'pending_verification';
  isVerified: boolean;
  firstName?: string;
  lastName?: string;
  displayName?: string;
  lastLogin?: string;
  preferences: Record<string, any>;
  createdAt: string;
}

export interface Campaign {
  id: string; // UUID
  name: string;
  description?: string;
  objective: string;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  visibility: 'private' | 'team' | 'public';
  targetProvider?: string;
  targetModel?: string;
  successRate?: number;
  createdAt: string;
  createdBy: number;
}
```

### **Phase 5: Testing & Validation (Estimated: 6-8 hours)**

#### **5.1 Automated Test Suite**
```bash
#!/bin/bash
# migration_test_suite.sh

echo "🧪 Starting Migration Test Suite..."

# Test database connectivity
echo "Testing database connection..."
psql -d chimera_prod -c "SELECT version();" || exit 1

# Test user authentication flow
echo "Testing user authentication..."
python -m pytest tests/test_auth_migration.py -v

# Test campaign CRUD operations
echo "Testing campaign management..."
python -m pytest tests/test_campaigns_migration.py -v

# Test analytics queries
echo "Testing analytics performance..."
python -m pytest tests/test_analytics_migration.py -v

# Test API endpoints
echo "Testing API endpoints..."
python -m pytest tests/test_api_migration.py -v

echo "✅ Migration Test Suite Complete"
```

#### **5.2 Performance Benchmarks**
```sql
-- Benchmark critical queries
\timing on

-- User dashboard load test
SELECT count(*) FROM (
    SELECT c.*, u.username
    FROM campaigns c
    JOIN users u ON c.created_by = u.id
    WHERE u.role = 'researcher'
    LIMIT 1000
) subq;

-- Analytics aggregation test
SELECT count(*) FROM (
    SELECT campaign_id, avg(latency_ms)
    FROM campaign_executions
    WHERE started_at > NOW() - INTERVAL '1 day'
    GROUP BY campaign_id
) subq;

-- Template search test
SELECT count(*) FROM (
    SELECT * FROM prompt_templates
    WHERE to_tsvector('english', title || ' ' || description) @@ plainto_tsquery('jailbreak autodan')
    LIMIT 100
) subq;
```

#### **5.3 Data Integrity Validation**
```sql
-- Final integrity checks
SELECT
    'Data integrity check' as test_name,
    CASE
        WHEN EXISTS (
            SELECT 1 FROM campaigns c
            LEFT JOIN users u ON c.created_by = u.id
            WHERE u.id IS NULL
        ) THEN 'FAILED: Orphaned campaigns found'
        WHEN EXISTS (
            SELECT 1 FROM template_ratings tr
            LEFT JOIN prompt_templates pt ON tr.template_id = pt.id
            WHERE pt.id IS NULL
        ) THEN 'FAILED: Orphaned ratings found'
        ELSE 'PASSED: All foreign keys valid'
    END as result;

-- Performance validation
WITH perf_metrics AS (
    SELECT
        'User login query' as query_name,
        (SELECT count(*) FROM users WHERE email = 'test@example.com') as result_count
    UNION ALL
    SELECT
        'Campaign dashboard query',
        (SELECT count(*) FROM campaigns WHERE status = 'running') as result_count
)
SELECT * FROM perf_metrics;
```

## **Rollback Strategy**

### **Emergency Rollback Procedure**
```sql
-- 1. Stop application services
-- systemctl stop chimera-backend chimera-frontend

-- 2. Rename tables back to original
BEGIN;

-- Backup new tables before rollback
CREATE TABLE users_new_backup AS SELECT * FROM users;
CREATE TABLE campaigns_new_backup AS SELECT * FROM campaigns;

-- Restore original tables
DROP TABLE users CASCADE;
ALTER TABLE users_old RENAME TO users;

DROP TABLE campaigns CASCADE;
ALTER TABLE campaigns_old RENAME TO campaigns;

-- Verify rollback
SELECT 'Rollback verification' as status,
       count(*) as user_count
FROM users;

COMMIT;

-- 3. Restart application with previous configuration
-- systemctl start chimera-backend chimera-frontend
```

### **Partial Rollback (Table-by-Table)**
```sql
-- If only specific table migration failed
-- Example: Rollback campaigns table only

BEGIN;
-- Backup failed migration
CREATE TABLE campaigns_failed_migration AS SELECT * FROM campaigns;

-- Restore from backup
DROP TABLE campaigns;
ALTER TABLE campaigns_old RENAME TO campaigns;

-- Update application configuration to use old schema
COMMIT;
```

## **Post-Migration Tasks**

### **1. Monitoring Setup**
```sql
-- Create monitoring views for ongoing health checks
CREATE VIEW migration_health_check AS
SELECT
    'Schema version' as metric,
    version as value,
    applied_at::text as timestamp
FROM schema_migrations
WHERE version = '2.0.0'
UNION ALL
SELECT
    'User count',
    count(*)::text,
    NOW()::text
FROM users
UNION ALL
SELECT
    'Campaign count',
    count(*)::text,
    NOW()::text
FROM campaigns;

-- Setup automated health monitoring
SELECT * FROM migration_health_check;
```

### **2. Performance Optimization**
```sql
-- Analyze table statistics for query optimizer
ANALYZE users;
ANALYZE campaigns;
ANALYZE campaign_executions;
ANALYZE prompt_templates;
ANALYZE telemetry_events;

-- Update database configuration for new schema
-- Consider adjusting:
-- - work_mem for analytics queries
-- - shared_buffers for cache efficiency
-- - max_connections for application load
```

### **3. Documentation Updates**
- ✅ Update API documentation with new field mappings
- ✅ Update frontend component documentation
- ✅ Create database schema documentation
- ✅ Update deployment guides with new requirements

### **4. Training & Communication**
- 📢 Notify development team of schema changes
- 📚 Provide training on new data models
- 🔧 Update development environment setup guides
- 📊 Share performance improvement metrics with stakeholders

## **Migration Timeline Summary**

| **Phase** | **Duration** | **Risk Level** | **Rollback Time** |
|-----------|--------------|----------------|------------------|
| Pre-Migration Setup | 2-4 hours | Low | N/A |
| Schema Migration | 4-8 hours | Medium | 1-2 hours |
| Index Optimization | 2-4 hours | Low | 30 minutes |
| Application Integration | 4-6 hours | High | 2-3 hours |
| Testing & Validation | 6-8 hours | Medium | N/A |
| **Total** | **18-30 hours** | **Medium** | **3-5 hours** |

## **Success Criteria**

### **Functional Requirements**
- ✅ All existing users can log in successfully
- ✅ All campaigns and their data are preserved
- ✅ Template library functions correctly
- ✅ Real-time telemetry streaming works
- ✅ Analytics and reporting maintain accuracy

### **Performance Requirements**
- ✅ User login latency < 200ms (improved from 500ms)
- ✅ Campaign dashboard load < 500ms (improved from 1.2s)
- ✅ Template search < 300ms (improved from 800ms)
- ✅ Analytics queries < 2s (improved from 5s+)

### **Data Integrity Requirements**
- ✅ Zero data loss during migration
- ✅ All foreign key relationships maintained
- ✅ Audit trail preservation for compliance
- ✅ User permissions and access levels preserved

This comprehensive migration strategy ensures a smooth transition to the optimized database schema while minimizing downtime and maintaining data integrity throughout the process.