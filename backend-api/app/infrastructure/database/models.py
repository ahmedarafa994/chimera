"""
Database Models for User Model Preferences

SQLAlchemy models for persisting user preferences, model selection history,
and rate limiting data.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    """User account for tracking preferences and usage."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True)  # UUID
    api_key_hash = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)

    # Tier for rate limiting (free, pro, enterprise)
    tier = Column(String(20), default="free", nullable=False)

    # Relationships
    preferences = relationship("UserModelPreference", back_populates="user", uselist=False)
    usage_records = relationship("ModelUsageRecord", back_populates="user")
    rate_limit_records = relationship("RateLimitRecord", back_populates="user")

    __table_args__ = (
        Index("idx_users_api_key_hash", "api_key_hash"),
        Index("idx_users_email", "email"),
    )


class UserModelPreference(Base):
    """User's preferred AI model configuration."""

    __tablename__ = "user_model_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Current selection
    selected_provider = Column(String(50), nullable=False, default="gemini")
    selected_model = Column(String(100), nullable=False, default="gemini-3-pro-preview")

    # Fallback preferences
    fallback_provider = Column(String(50), nullable=True)
    fallback_model = Column(String(100), nullable=True)

    # Model-specific settings (JSON for flexibility)
    model_settings = Column(JSON, default=dict)
    # Example: {"gemini": {"temperature": 0.7}, "deepseek": {"temperature": 0.5}}

    # UI preferences
    remember_selection = Column(Boolean, default=True)
    auto_fallback = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="preferences")

    __table_args__ = (
        Index("idx_preferences_user_id", "user_id"),
        Index("idx_preferences_provider", "selected_provider"),
    )


class ModelUsageRecord(Base):
    """Track model usage for analytics and billing."""

    __tablename__ = "model_usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Request details
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    request_id = Column(String(36), unique=True, nullable=False)

    # Usage metrics
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)

    # Cost tracking (in USD cents)
    estimated_cost_cents = Column(Float, default=0.0)

    # Status
    status = Column(String(20), default="success")  # success, error, timeout, rate_limited
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship
    user = relationship("User", back_populates="usage_records")

    __table_args__ = (
        Index("idx_usage_user_id", "user_id"),
        Index("idx_usage_provider_model", "provider", "model"),
        Index("idx_usage_created_at", "created_at"),
        Index("idx_usage_request_id", "request_id"),
    )


class RateLimitRecord(Base):
    """Per-model rate limiting tracking."""

    __tablename__ = "rate_limit_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Rate limit scope
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=True)  # NULL means provider-level limit

    # Current window
    window_start = Column(DateTime, nullable=False)
    window_size_seconds = Column(Integer, default=60)  # 1 minute default

    # Counters
    request_count = Column(Integer, default=0)
    token_count = Column(Integer, default=0)

    # Limits (from user tier)
    max_requests_per_window = Column(Integer, default=60)
    max_tokens_per_window = Column(Integer, default=100000)

    # Status
    is_limited = Column(Boolean, default=False)
    limited_until = Column(DateTime, nullable=True)

    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="rate_limit_records")

    __table_args__ = (
        UniqueConstraint("user_id", "provider", "model", name="uq_rate_limit_user_provider_model"),
        Index("idx_rate_limit_user_provider", "user_id", "provider"),
        Index("idx_rate_limit_window", "window_start"),
    )


class ProviderStatus(Base):
    """Track provider health and availability."""

    __tablename__ = "provider_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(50), unique=True, nullable=False)

    # Health status
    is_available = Column(Boolean, default=True)
    health_score = Column(Float, default=1.0)  # 0.0 to 1.0

    # Performance metrics (rolling averages)
    avg_latency_ms = Column(Float, default=0.0)
    error_rate = Column(Float, default=0.0)  # 0.0 to 1.0

    # Circuit breaker state
    circuit_state = Column(String(20), default="closed")  # closed, open, half_open
    circuit_opened_at = Column(DateTime, nullable=True)
    consecutive_failures = Column(Integer, default=0)

    # Last check
    last_health_check = Column(DateTime, default=datetime.utcnow)
    last_successful_request = Column(DateTime, nullable=True)

    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_provider_status_provider", "provider"),
        Index("idx_provider_status_available", "is_available"),
    )


class ModelSelectionHistory(Base):
    """Audit log for model selection changes."""

    __tablename__ = "model_selection_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(36), nullable=True)

    # Selection change
    previous_provider = Column(String(50), nullable=True)
    previous_model = Column(String(100), nullable=True)
    new_provider = Column(String(50), nullable=False)
    new_model = Column(String(100), nullable=False)

    # Context
    change_reason = Column(
        String(50), default="user_selection"
    )  # user_selection, fallback, auto_switch
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("idx_history_user_id", "user_id"),
        Index("idx_history_created_at", "created_at"),
    )


# ============================================================================
# Jailbreak Quality Metrics Models
# ============================================================================


class JailbreakExecution(Base):
    """Represents a single jailbreak technique execution (for quality analytics)."""

    __tablename__ = "jailbreak_executions"

    execution_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    technique_id = Column(String(100), nullable=False, index=True)
    original_prompt = Column(Text, nullable=False)
    jailbroken_prompt = Column(Text, nullable=False)

    target_model = Column(String(100), nullable=False, index=True)
    target_provider = Column(String(50), nullable=False, index=True)

    execution_status = Column(String(50), nullable=False, default="unknown")
    execution_time_ms = Column(Float, nullable=False, default=0.0)

    llm_response_text = Column(Text, nullable=True)
    llm_response_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    quality_metrics = relationship(
        "QualityMetric",
        back_populates="execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    feedback = relationship(
        "FeedbackHistory",
        back_populates="execution",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class QualityMetric(Base):
    """Quality metrics associated with a jailbreak execution."""

    __tablename__ = "jailbreak_quality_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(
        String(36),
        ForeignKey("jailbreak_executions.execution_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Dimension scores
    effectiveness_score = Column(Float, nullable=False, default=0.0)
    naturalness_score = Column(Float, nullable=False, default=0.0)
    detectability_score = Column(Float, nullable=False, default=0.0)
    coherence_score = Column(Float, nullable=False, default=0.0)

    # Semantic success
    semantic_success_score = Column(Float, nullable=False, default=0.0)
    overall_quality_score = Column(Float, nullable=False, default=0.0)
    semantic_success_level = Column(String(50), nullable=False, default="failure")

    bypass_indicators = Column(JSON, nullable=True)  # {"indicators": [...]}
    safety_trigger_detected = Column(Boolean, nullable=False, default=False)
    response_coherence = Column(Float, nullable=False, default=0.0)

    confidence_interval_lower = Column(Float, nullable=False, default=0.0)
    confidence_interval_upper = Column(Float, nullable=False, default=1.0)
    analysis_duration_ms = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship
    execution = relationship("JailbreakExecution", back_populates="quality_metrics")


class TechniquePerformance(Base):
    """Aggregated performance stats for a technique/provider/model over a time window."""

    __tablename__ = "jailbreak_technique_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    technique_id = Column(String(100), nullable=False)
    model_name = Column(String(100), nullable=False, default="all")
    provider_name = Column(String(50), nullable=False, default="all")

    time_window_hours = Column(Integer, nullable=False, default=168)

    total_attempts = Column(Integer, nullable=False, default=0)
    semantic_success_rate = Column(Float, nullable=True)
    binary_success_rate = Column(Float, nullable=True)

    avg_effectiveness = Column(Float, nullable=True)
    avg_naturalness = Column(Float, nullable=True)
    avg_detectability = Column(Float, nullable=True)
    avg_coherence = Column(Float, nullable=True)
    std_deviation = Column(Float, nullable=True)

    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "technique_id",
            "model_name",
            "provider_name",
            name="uq_technique_model_provider",
        ),
        Index("idx_perf_technique", "technique_id"),
        Index("idx_perf_model_provider", "model_name", "provider_name"),
    )


class FeedbackHistory(Base):
    """User/system feedback tied to a jailbreak execution."""

    __tablename__ = "jailbreak_feedback_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(
        String(36),
        ForeignKey("jailbreak_executions.execution_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    feedback_type = Column(String(50), nullable=False)
    success_indicator = Column(Boolean, nullable=False, default=False)

    user_feedback = Column(Text, nullable=True)
    llm_evaluation_result = Column(JSON, nullable=True)

    rating = Column(Integer, nullable=True)
    comments = Column(Text, nullable=True)

    user_id = Column(String(36), nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    execution = relationship("JailbreakExecution", back_populates="feedback")


# SQL Schema for reference (PostgreSQL)
SCHEMA_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    api_key_hash VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    tier VARCHAR(20) DEFAULT 'free' NOT NULL
);

CREATE INDEX idx_users_api_key_hash ON users(api_key_hash);
CREATE INDEX idx_users_email ON users(email);

-- User model preferences
CREATE TABLE IF NOT EXISTS user_model_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    selected_provider VARCHAR(50) NOT NULL DEFAULT 'gemini',
    selected_model VARCHAR(100) NOT NULL DEFAULT 'gemini-3-pro-preview',
    fallback_provider VARCHAR(50),
    fallback_model VARCHAR(100),
    model_settings JSONB DEFAULT '{}',
    remember_selection BOOLEAN DEFAULT TRUE,
    auto_fallback BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preferences_user_id ON user_model_preferences(user_id);
CREATE INDEX idx_preferences_provider ON user_model_preferences(selected_provider);

-- Model usage records
CREATE TABLE IF NOT EXISTS model_usage_records (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    request_id VARCHAR(36) UNIQUE NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms FLOAT DEFAULT 0.0,
    estimated_cost_cents FLOAT DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'success',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_usage_user_id ON model_usage_records(user_id);
CREATE INDEX idx_usage_provider_model ON model_usage_records(provider, model);
CREATE INDEX idx_usage_created_at ON model_usage_records(created_at);
CREATE INDEX idx_usage_request_id ON model_usage_records(request_id);

-- Rate limit records
CREATE TABLE IF NOT EXISTS rate_limit_records (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100),
    window_start TIMESTAMP NOT NULL,
    window_size_seconds INTEGER DEFAULT 60,
    request_count INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    max_requests_per_window INTEGER DEFAULT 60,
    max_tokens_per_window INTEGER DEFAULT 100000,
    is_limited BOOLEAN DEFAULT FALSE,
    limited_until TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, provider, model)
);

CREATE INDEX idx_rate_limit_user_provider ON rate_limit_records(user_id, provider);
CREATE INDEX idx_rate_limit_window ON rate_limit_records(window_start);

-- Provider status
CREATE TABLE IF NOT EXISTS provider_status (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) UNIQUE NOT NULL,
    is_available BOOLEAN DEFAULT TRUE,
    health_score FLOAT DEFAULT 1.0,
    avg_latency_ms FLOAT DEFAULT 0.0,
    error_rate FLOAT DEFAULT 0.0,
    circuit_state VARCHAR(20) DEFAULT 'closed',
    circuit_opened_at TIMESTAMP,
    consecutive_failures INTEGER DEFAULT 0,
    last_health_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_successful_request TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_provider_status_provider ON provider_status(provider);
CREATE INDEX idx_provider_status_available ON provider_status(is_available);

-- Model selection history (audit log)
CREATE TABLE IF NOT EXISTS model_selection_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(36),
    previous_provider VARCHAR(50),
    previous_model VARCHAR(100),
    new_provider VARCHAR(50) NOT NULL,
    new_model VARCHAR(100) NOT NULL,
    change_reason VARCHAR(50) DEFAULT 'user_selection',
    client_ip VARCHAR(45),
    user_agent VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_history_user_id ON model_selection_history(user_id);
CREATE INDEX idx_history_created_at ON model_selection_history(created_at);
"""


class SelectionModel(Base):
    """
    Database model for provider/model selections.

    Implements the Session Preference tier of the three-tier selection hierarchy.
    Supports both session-scoped and user-scoped selections using a composite key.

    Features:
    - Optimistic concurrency control via version column
    - Flexible metadata storage via JSONB
    - Session expiration support via expires_at
    - Source tracking for audit purposes
    """
    __tablename__ = "selections"

    # Composite primary key (both nullable to support different selection types)
    # - session_id set: session-scoped selection
    # - session_id NULL + user_id set: user default selection
    session_id = Column(String(255), primary_key=True, nullable=True)
    user_id = Column(String(255), primary_key=True, nullable=True)

    # Selection data
    provider_id = Column(String(100), nullable=False, index=True)
    model_id = Column(String(100), nullable=False, index=True)

    # Optimistic concurrency control
    version = Column(Integer, default=1, nullable=False)

    # Flexible metadata storage (JSON/JSONB)
    metadata = Column(JSON, default=dict, nullable=False)

    # Source of the selection (frontend, backend, migration, default)
    source = Column(String(50), default="unknown", nullable=False)

    # Session expiration (for cleanup)
    expires_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # Indexes for efficient lookups
    __table_args__ = (
        Index("idx_selections_session_id", "session_id"),
        Index("idx_selections_user_id", "user_id"),
        Index("idx_selections_provider_model", "provider_id", "model_id"),
        # Composite index for common query pattern
        Index("idx_selections_user_session", "user_id", "session_id"),
        # Index for concurrency queries
        Index("idx_selections_version", "version"),
        # Index for cleanup queries
        Index("idx_selections_expires_at", "expires_at"),
        # Index for sorting/filtering
        Index("idx_selections_updated_at", "updated_at"),
    )

    def increment_version(self) -> int:
        """Increment version for optimistic concurrency."""
        self.version += 1
        return self.version
