---
name: Database Management
description: Expert skill for managing database operations, migrations, and schema updates. Use when working with SQLAlchemy models, database migrations, or debugging database issues.
---

# Database Management Skill

## Overview

This skill provides expertise in managing the Chimera SQLite database, including schema design, migrations, model definitions, and query optimization.

## When to Use This Skill

- Creating or modifying database models
- Running database migrations
- Debugging database connection issues
- Optimizing database queries
- Handling database write locks
- Backing up and restoring data

## Technology Stack

### Core Components

- **SQLAlchemy 2.0**: Modern Python ORM
- **Alembic**: Database migration tool
- **SQLite**: Default development database
- **PostgreSQL**: Production database option

## Database Schema Overview

### Core Tables

```sql
-- Users and Authentication
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aegis Campaigns
CREATE TABLE campaigns (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    objective TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    max_iterations INTEGER DEFAULT 10,
    potency_level INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    telemetry_data JSON
);

-- Sessions (Multi-Iteration Tracking)
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    campaign_id INTEGER REFERENCES campaigns(id),
    iteration INTEGER NOT NULL,
    persona_id INTEGER,
    scenario_type VARCHAR(100),
    prompt TEXT,
    response TEXT,
    rbs_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prompt Transformations
CREATE TABLE transformations (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    technique VARCHAR(100) NOT NULL,
    original_prompt TEXT NOT NULL,
    transformed_prompt TEXT NOT NULL,
    potency_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API Keys and Authentication Tokens
CREATE TABLE api_keys (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);
```

## SQLAlchemy Model Examples

### Base Model Configuration

```python
# backend-api/app/models/base.py
from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Campaign Model

```python
# backend-api/app/models/campaign.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from .base import BaseModel

class Campaign(BaseModel):
    __tablename__ = "campaigns"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    objective = Column(Text, nullable=False)
    status = Column(String(50), default="pending")
    max_iterations = Column(Integer, default=10)
    potency_level = Column(Integer, default=5)
    telemetry_data = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="campaigns")
    sessions = relationship("Session", back_populates="campaign", cascade="all, delete-orphan")
```

## Database Configuration

### SQLite Development Setup

```python
# backend-api/app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL = "sqlite:///./chimera.db"

# Configure for web application use
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Allow multi-threaded access
    poolclass=StaticPool,  # Single connection pool for SQLite
    echo=False  # Set to True for SQL query logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### PostgreSQL Production Setup

```python
# For production, use PostgreSQL
DATABASE_URL = "postgresql://user:password@localhost/chimera"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections
    echo=False
)
```

## Database Migrations with Alembic

### Initialize Alembic

```bash
cd backend-api
poetry run alembic init alembic
```

### Configure Alembic

```python
# backend-api/alembic/env.py
from app.models.base import Base
from app.models import user, campaign, session, transformation

# Set target metadata
target_metadata = Base.metadata
```

### Create Migration

```bash
# Auto-generate migration from model changes
poetry run alembic revision --autogenerate -m "Add campaign telemetry field"

# Creates file: alembic/versions/xxx_add_campaign_telemetry_field.py
```

### Apply Migration

```bash
# Upgrade to latest version
poetry run alembic upgrade head

# Downgrade one version
poetry run alembic downgrade -1

# Show current version
poetry run alembic current

# Show migration history
poetry run alembic history
```

### Manual Migration Example

```python
# alembic/versions/xxx_add_rbs_score_to_sessions.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('sessions', sa.Column('rbs_score', sa.Float(), nullable=True))

def downgrade():
    op.drop_column('sessions', 'rbs_score')
```

## Common Database Operations

### Create Records

```python
from app.models.campaign import Campaign
from app.core.database import get_db

def create_campaign(db: Session, user_id: int, objective: str):
    campaign = Campaign(
        user_id=user_id,
        objective=objective,
        status="pending",
        max_iterations=10
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)
    return campaign
```

### Query Records

```python
from sqlalchemy import select

# Get all campaigns for a user
def get_user_campaigns(db: Session, user_id: int):
    stmt = select(Campaign).where(Campaign.user_id == user_id)
    result = db.execute(stmt)
    return result.scalars().all()

# Get campaign with sessions
def get_campaign_with_sessions(db: Session, campaign_id: int):
    stmt = select(Campaign).where(Campaign.id == campaign_id)
    campaign = db.execute(stmt).scalar_one_or_none()
    # Sessions are automatically loaded via relationship
    return campaign
```

### Update Records

```python
def update_campaign_status(db: Session, campaign_id: int, status: str):
    stmt = select(Campaign).where(Campaign.id == campaign_id)
    campaign = db.execute(stmt).scalar_one_or_none()
    if campaign:
        campaign.status = status
        db.commit()
        db.refresh(campaign)
    return campaign
```

### Delete Records

```python
def delete_campaign(db: Session, campaign_id: int):
    stmt = select(Campaign).where(Campaign.id == campaign_id)
    campaign = db.execute(stmt).scalar_one_or_none()
    if campaign:
        db.delete(campaign)
        db.commit()
    return True
```

## Common Issues and Solutions

### 1. Database Write Locks (SQLite)

**Symptom**: `sqlite3.OperationalError: database is locked`

**Root Cause**: Concurrent write operations in SQLite

**Solutions**:

```python
# Option A: Configure connection for web apps
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)

# Option B: Add retry logic
from sqlalchemy.exc import OperationalError
import time

def execute_with_retry(session, stmt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return session.execute(stmt)
        except OperationalError as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            else:
                raise
```

### 2. Migration Conflicts

**Symptom**: `alembic.util.exc.CommandError: Target database is not up to date`

**Solutions**:

```bash
# Check current version
poetry run alembic current

# View pending migrations
poetry run alembic history

# Stamp database to specific version (if manual changes made)
poetry run alembic stamp head

# Reset and reapply all migrations (CAUTION: drops data)
poetry run alembic downgrade base
poetry run alembic upgrade head
```

### 3. Foreign Key Constraint Violations

**Symptom**: `FOREIGN KEY constraint failed`

**Solutions**:

```python
# Ensure parent record exists before creating child
def create_session(db: Session, campaign_id: int, data: dict):
    # Verify campaign exists
    campaign = db.get(Campaign, campaign_id)
    if not campaign:
        raise ValueError(f"Campaign {campaign_id} not found")
    
    session = Session(campaign_id=campaign_id, **data)
    db.add(session)
    db.commit()
    return session
```

### 4. N+1 Query Problem

**Symptom**: Slow queries due to multiple database calls

**Solution**: Use eager loading

```python
from sqlalchemy.orm import joinedload

# BAD: N+1 queries
campaigns = db.query(Campaign).all()
for campaign in campaigns:
    print(campaign.sessions)  # Triggers new query for each campaign

# GOOD: Eager loading
campaigns = db.query(Campaign).options(
    joinedload(Campaign.sessions)
).all()
for campaign in campaigns:
    print(campaign.sessions)  # Already loaded
```

## Database Maintenance

### Backup Database

```bash
# SQLite backup
cp chimera.db chimera.db.backup.$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
pg_dump chimera > chimera_backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore Database

```bash
# SQLite restore
cp chimera.db.backup.20260116_120000 chimera.db

# PostgreSQL restore
psql chimera < chimera_backup_20260116_120000.sql
```

### Vacuum SQLite Database

```bash
# Reclaim space and optimize
sqlite3 chimera.db "VACUUM;"
```

### Database Statistics

```python
from sqlalchemy import func

# Count records
campaign_count = db.query(func.count(Campaign.id)).scalar()

# Average RBS score
avg_rbs = db.query(func.avg(Session.rbs_score)).scalar()

# Campaigns by status
status_counts = db.query(
    Campaign.status,
    func.count(Campaign.id)
).group_by(Campaign.status).all()
```

## Testing Database Operations

### Setup Test Database

```python
# conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.base import Base

@pytest.fixture
def test_db():
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    
    yield session
    
    session.close()
```

### Test Model Creation

```python
def test_create_campaign(test_db):
    from app.models.user import User
    from app.models.campaign import Campaign
    
    # Create user
    user = User(email="test@example.com", hashed_password="hash")
    test_db.add(user)
    test_db.commit()
    
    # Create campaign
    campaign = Campaign(
        user_id=user.id,
        objective="Test objective"
    )
    test_db.add(campaign)
    test_db.commit()
    
    assert campaign.id is not None
    assert campaign.status == "pending"
```

## Performance Optimization

### Indexing

```python
# Add indexes to frequently queried columns
class Campaign(BaseModel):
    __table_args__ = (
        Index('idx_campaign_user_status', 'user_id', 'status'),
        Index('idx_campaign_created_at', 'created_at'),
    )
```

### Query Optimization

```python
# Use select for better performance in SQLAlchemy 2.0
from sqlalchemy import select

# GOOD
stmt = select(Campaign).where(Campaign.status == "completed")
campaigns = db.execute(stmt).scalars().all()

# OLD STYLE (still works but less performant)
campaigns = db.query(Campaign).filter(Campaign.status == "completed").all()
```

### Connection Pooling

```python
# Configure pool for optimal performance
engine = create_engine(
    DATABASE_URL,
    pool_size=10,         # Number of connections to maintain
    max_overflow=20,      # Max connections beyond pool_size
    pool_timeout=30,      # Timeout waiting for connection
    pool_recycle=3600,    # Recycle connections after 1 hour
    pool_pre_ping=True    # Verify connection health
)
```

## Schema Design Best Practices

1. **Use appropriate data types** (INTEGER for IDs, TEXT for long strings, JSON for structured data)
2. **Add indexes** on foreign keys and frequently queried columns
3. **Use constraints** (NOT NULL, UNIQUE, CHECK) for data integrity
4. **Normalize data** to avoid redundancy
5. **Use relationships** in SQLAlchemy models for clean code
6. **Version your schema** with Alembic migrations

## References

- [database_schema_design.sql](../../database_schema_design.sql): Full schema SQL
- [database_migration_guide.md](../../database_migration_guide.md): Migration procedures
- [database_schema_analysis.md](../../database_schema_analysis.md): Schema analysis
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
