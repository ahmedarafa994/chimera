"""
SQLAlchemy models for model synchronization system.
"""

# Use appropriate JSON type based on database
import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON as PostgreSQLJSON
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.ext.declarative import declarative_base

IS_POSTGRES = "postgresql" in os.getenv("DATABASE_URL", "").lower()
JSONType = PostgreSQLJSON if IS_POSTGRES else SQLiteJSON

Base = declarative_base()


class Model(Base):
    """Model configuration table."""

    __tablename__ = "models"

    id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    provider = Column(String(50), nullable=False)
    capabilities = Column(JSONType, nullable=False, default=list)
    max_tokens = Column(Integer, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes for performance
    __table_args__ = (
        Index("idx_models_provider", "provider"),
        Index("idx_models_active", "is_active"),
        Index("idx_models_updated", "updated_at"),
    )


class UserSession(Base):
    """User session tracking table."""

    __tablename__ = "user_sessions"

    user_id = Column(String(255), nullable=False)
    session_id = Column(String(255), primary_key=True)
    selected_model_id = Column(String(100), nullable=True)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_expires", "expires_at"),
        Index("idx_sessions_updated", "last_updated"),
    )


class ModelChangeLog(Base):
    """Model change audit log table."""

    __tablename__ = "model_change_log"

    id = Column(
        String(100),
        primary_key=True,
        default=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"),
    )
    user_id = Column(String(255), nullable=False)
    session_id = Column(String(255), nullable=False)
    old_model_id = Column(String(100), nullable=True)
    new_model_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(Text, nullable=True)
    success = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_model_changes_user", "user_id"),
        Index("idx_model_changes_timestamp", "timestamp"),
        Index("idx_model_changes_session", "session_id"),
    )


class ModelAvailabilityCache(Base):
    """Model availability cache table."""

    __tablename__ = "model_availability_cache"

    model_id = Column(String(100), primary_key=True)
    is_available = Column(Boolean, nullable=False, default=True)
    last_checked = Column(DateTime, nullable=False, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)
    retry_after = Column(Integer, nullable=True)  # Seconds to wait before retry

    # Indexes for performance
    __table_args__ = (
        Index("idx_model_cache_checked", "last_checked"),
        Index("idx_model_cache_available", "is_available"),
    )


# Create all tables
def create_model_sync_tables(engine):
    """Create model synchronization tables."""
    Base.metadata.create_all(engine, checkfirst=True)


# Helper functions for database operations
async def get_active_models(db_session):
    """Get all active models from database."""
    from sqlalchemy import select

    from app.domain.model_sync import ModelInfo

    result = await db_session.execute(
        select(Model).where(Model.is_active).order_by(Model.provider, Model.name)
    )
    models = result.scalars().all()

    # Convert to ModelInfo objects
    return [
        ModelInfo(
            id=model.id,
            name=model.name,
            description=model.description,
            capabilities=model.capabilities or [],
            max_tokens=model.max_tokens,
            is_active=model.is_active,
            provider=model.provider,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
        for model in models
    ]


async def get_user_session(db_session, session_id: str):
    """Get user session by session ID."""
    from sqlalchemy import select

    result = await db_session.execute(
        select(UserSession).where(UserSession.session_id == session_id)
    )
    return result.scalar_one_or_none()


async def create_user_session(
    db_session, user_id: str, session_id: str, ip_address: str, user_agent: str | None = None
):
    """Create a new user session."""
    from datetime import timedelta

    session = UserSession(
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
        expires_at=datetime.utcnow() + timedelta(hours=24),  # 24 hour session
    )

    db_session.add(session)
    await db_session.commit()
    return session


async def update_user_session_model(db_session, session_id: str, model_id: str):
    """Update user session with selected model."""
    from sqlalchemy import select, update

    # Get current session
    result = await db_session.execute(
        select(UserSession).where(UserSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()

    if session:
        old_model_id = session.selected_model_id

        # Update session
        await db_session.execute(
            update(UserSession)
            .where(UserSession.session_id == session_id)
            .values(selected_model_id=model_id, last_updated=datetime.utcnow())
        )

        # Log the change
        change_log = ModelChangeLog(
            user_id=session.user_id,
            session_id=session_id,
            old_model_id=old_model_id,
            new_model_id=model_id,
            ip_address=session.ip_address,
            user_agent=session.user_agent,
            success=True,
        )

        db_session.add(change_log)
        await db_session.commit()

        return True
    return False


async def log_model_change_error(
    db_session,
    user_id: str,
    session_id: str,
    model_id: str,
    error_message: str,
    ip_address: str,
    user_agent: str | None = None,
):
    """Log failed model change attempt."""
    change_log = ModelChangeLog(
        user_id=user_id,
        session_id=session_id,
        new_model_id=model_id,
        error_message=error_message,
        ip_address=ip_address,
        user_agent=user_agent,
        success=False,
    )

    db_session.add(change_log)
    await db_session.commit()


async def cleanup_expired_sessions(db_session):
    """Clean up expired user sessions."""
    from sqlalchemy import delete

    await db_session.execute(delete(UserSession).where(UserSession.expires_at < datetime.utcnow()))
    await db_session.commit()
