# Database Migrations Guide

This guide covers database migration management using Alembic for the Chimera FastAPI backend.

## Overview

Alembic is configured to work with both SQLite (development) and PostgreSQL (production) databases. All migrations are version-controlled and support both upgrade and rollback operations.

## Configuration

### Database URL

The database URL is automatically loaded from `app/core/config.py` settings:
- **Development**: `sqlite:///./chimera.db` (default)
- **Production**: PostgreSQL connection string from `DATABASE_URL` environment variable

### Alembic Files

- `alembic.ini` - Alembic configuration
- `alembic/env.py` - Migration environment setup (imports models and config)
- `alembic/versions/` - Migration scripts directory

## Common Commands

### Create a New Migration

After modifying SQLAlchemy models, generate a migration:

```bash
cd backend-api
poetry run alembic revision --autogenerate -m "Description of changes"
```

**Example:**
```bash
poetry run alembic revision --autogenerate -m "Add user preferences table"
```

### Apply Migrations (Upgrade)

Apply all pending migrations:

```bash
poetry run alembic upgrade head
```

Apply migrations up to a specific revision:

```bash
poetry run alembic upgrade <revision_id>
```

### Rollback Migrations (Downgrade)

Rollback one migration:

```bash
poetry run alembic downgrade -1
```

Rollback to a specific revision:

```bash
poetry run alembic downgrade <revision_id>
```

Rollback all migrations:

```bash
poetry run alembic downgrade base
```

### View Migration History

Show current database version:

```bash
poetry run alembic current
```

Show migration history:

```bash
poetry run alembic history
```

Show pending migrations:

```bash
poetry run alembic history --verbose
```

## Migration Workflow

### 1. Modify Models

Edit SQLAlchemy models in:
- `app/infrastructure/database/models.py`
- `app/db/models.py`

**Example:**
```python
from sqlalchemy import Column, String, Integer
from app.infrastructure.database.models import Base

class NewTable(Base):
    __tablename__ = "new_table"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
```

### 2. Generate Migration

```bash
cd backend-api
poetry run alembic revision --autogenerate -m "Add new_table"
```

### 3. Review Migration

Check the generated migration file in `alembic/versions/`:

```python
def upgrade() -> None:
    op.create_table('new_table',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('new_table')
```

### 4. Apply Migration

```bash
poetry run alembic upgrade head
```

### 5. Verify

Check database schema or run application tests.

## Best Practices

### 1. Always Review Generated Migrations

Autogenerate is smart but not perfect. Always review migrations before applying:
- Check for unintended changes
- Verify data migrations are safe
- Add custom data transformations if needed

### 2. Test Migrations

Test both upgrade and downgrade:

```bash
# Apply migration
poetry run alembic upgrade head

# Test rollback
poetry run alembic downgrade -1

# Re-apply
poetry run alembic upgrade head
```

### 3. Never Edit Applied Migrations

Once a migration is applied to production, never edit it. Create a new migration instead.

### 4. Use Descriptive Messages

```bash
# Good
poetry run alembic revision --autogenerate -m "Add user_preferences table with provider selection"

# Bad
poetry run alembic revision --autogenerate -m "Update db"
```

### 5. Handle Data Migrations Carefully

For complex data transformations, edit the generated migration:

```python
def upgrade() -> None:
    # Schema change
    op.add_column('users', sa.Column('full_name', sa.String(200)))

    # Data migration
    connection = op.get_bind()
    connection.execute(
        "UPDATE users SET full_name = first_name || ' ' || last_name"
    )

    # Cleanup
    op.drop_column('users', 'first_name')
    op.drop_column('users', 'last_name')
```

## Production Deployment

### Initial Setup

1. Set `DATABASE_URL` environment variable:
```bash
export DATABASE_URL="postgresql://user:password@host:5432/chimera"
```

2. Apply all migrations:
```bash
cd backend-api
poetry run alembic upgrade head
```

### Updating Production

1. Pull latest code with new migrations
2. Backup database (always!)
3. Apply migrations:
```bash
poetry run alembic upgrade head
```

### Rollback in Production

If issues occur after migration:

```bash
# Rollback last migration
poetry run alembic downgrade -1

# Or rollback to specific version
poetry run alembic downgrade <revision_id>
```

## Troubleshooting

### "Target database is not up to date"

Your database is behind. Apply pending migrations:
```bash
poetry run alembic upgrade head
```

### "Can't locate revision identified by"

Migration history is out of sync. Check:
```bash
poetry run alembic current
poetry run alembic history
```

### "No changes in schema detected"

Alembic didn't detect model changes. Ensure:
1. Models are imported in `alembic/env.py`
2. Models inherit from correct `Base`
3. Changes are actually schema changes (not just code refactoring)

### SQLite Limitations

SQLite has limited ALTER TABLE support. Some operations require table recreation:
- Renaming columns
- Changing column types
- Adding foreign keys to existing tables

Alembic handles this automatically with `batch_alter_table`.

## Advanced Usage

### Create Empty Migration

For data-only migrations:

```bash
poetry run alembic revision -m "Seed initial data"
```

Edit the generated file:

```python
def upgrade() -> None:
    connection = op.get_bind()
    connection.execute(
        "INSERT INTO users (id, email) VALUES ('1', 'admin@example.com')"
    )

def downgrade() -> None:
    connection = op.get_bind()
    connection.execute("DELETE FROM users WHERE id = '1'")
```

### Branching and Merging

For parallel development:

```bash
# Create branch
poetry run alembic revision -m "Feature A" --head=<base_revision>

# Merge branches
poetry run alembic merge -m "Merge feature branches" <rev1> <rev2>
```

### Offline SQL Generation

Generate SQL without applying:

```bash
poetry run alembic upgrade head --sql > migration.sql
```

## Integration with Application

### Startup Migration Check

Add to `app/main.py` or startup script:

```python
from alembic import command
from alembic.config import Config

def run_migrations():
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
```

### Health Check

Check if migrations are current:

```python
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine

def check_migrations_current():
    engine = create_engine(settings.DATABASE_URL)
    script = ScriptDirectory.from_config(alembic_cfg)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current = context.get_current_revision()
        head = script.get_current_head()

        return current == head
```

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [FastAPI with Alembic](https://fastapi.tiangolo.com/tutorial/sql-databases/#alembic-note)
