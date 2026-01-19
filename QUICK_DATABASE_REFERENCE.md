# Quick Database Reference

## Current Schema Status

✅ **Database**: SQLite (Development) - `chimera.db`  
✅ **Migration Version**: `91cd7358f4c6` (comprehensive_schema_validation)  
✅ **Total Tables**: 15+ core tables

## Essential Commands

### Check Database Status

```bash
cd backend-api
py scripts/setup_database.py --check
```

### Run Migrations

```bash
cd backend-api
alembic upgrade head
```

### Seed Development Data

```bash
cd backend-api
py scripts/setup_database.py --seed
```

### Reset Database (Development Only)

```bash
cd backend-api
py scripts/setup_database.py --reset --seed
```

## Core Tables

### User Management

- **users** - User accounts with RBAC (Admin, Researcher, Viewer)
- **user_api_keys** - Per-user API keys for programmatic access
- **user_model_preferences** - User's preferred AI models
- **user_sessions** - JWT session management

### Campaign Management

- **campaigns** - Research campaigns with ownership
- **campaign_shares** - Campaign sharing with permissions (view/edit)
- **campaign_executions** - Execution results and telemetry
- **campaign_analytics** - Pre-computed metrics

### Provider Management

- **llm_providers** - LLM provider configurations
- **llm_models** - Available models per provider
- **model_usage_records** - Usage tracking for billing
- **provider_status** - Provider health monitoring

### Prompt Library

- **prompt_templates** - Community-shared templates
- **template_ratings** - User ratings and reviews
- **template_usage** - Usage tracking

### Audit & Compliance

- **audit_log** - Comprehensive audit trail
- **config_audit** - Configuration change tracking

## Default Users (After Seeding)

| Email | Password | Role | Access Level |
|-------|----------|------|--------------|
| <admin@chimera.local> | admin123 | Admin | Full system access |
| <researcher@chimera.local> | researcher123 | Researcher | Create/edit campaigns |
| <viewer@chimera.local> | viewer123 | Viewer | Read-only access |

## Common Queries

### Check Migration Status

```bash
cd backend-api
alembic current
alembic history
```

### Create New Migration

```bash
cd backend-api
alembic revision --autogenerate -m "description"
```

### Rollback Migration

```bash
cd backend-api
alembic downgrade -1
```

## Database URLs

### Development (SQLite)

```env
DATABASE_URL=sqlite:///./chimera.db
```

### Production (PostgreSQL)

```env
DATABASE_URL=postgresql://user:password@localhost:5432/chimera
```

⚠️ **Production Note**: SQLite is NOT supported in production. Use PostgreSQL 14+ for multi-user deployments.

## Troubleshooting

### "Table already exists" warnings

These are normal when multiple Base classes define the same table. The migration system handles this correctly.

### Migration conflicts

```bash
alembic heads  # Check for multiple heads
alembic merge heads -m "Merge branches"
alembic upgrade head
```

### Database locked (SQLite)

SQLite locks during writes. For development with multiple processes, consider PostgreSQL.

## Next Steps

1. ✅ Database schema is set up
2. Run `py scripts/setup_database.py --seed` to create test users
3. Start backend: `npm run dev:backend`
4. Access API docs: <http://localhost:8001/docs>

## Full Documentation

See [DATABASE_SETUP_GUIDE.md](./DATABASE_SETUP_GUIDE.md) for comprehensive documentation including:

- Production deployment checklist
- Performance tuning
- Security best practices
- Monitoring and maintenance
- Backup strategies
