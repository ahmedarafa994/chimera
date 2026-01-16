"""
Campaign Analytics Database Models

SQLAlchemy models for campaign telemetry, analytics, and research tracking.
Provides comprehensive data models for post-campaign analysis and reporting.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

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
)
from sqlalchemy.orm import relationship

# Import the shared Base from models.py for consistency
from app.infrastructure.database.models import Base


class CampaignStatus(str, PyEnum):
    """Campaign lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStatus(str, PyEnum):
    """Individual telemetry event execution status."""

    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class Campaign(Base):
    """
    Represents an adversarial research campaign.

    A campaign groups related jailbreak/prompt experiments for analysis.
    Supports tracking objectives, configuration, and aggregated results.
    """

    __tablename__ = "campaigns"

    # Primary key
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Campaign metadata
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    objective = Column(Text, nullable=False)

    # Status tracking
    status = Column(
        String(20),
        default=CampaignStatus.DRAFT.value,
        nullable=False,
        index=True,
    )

    # Target configuration
    target_provider = Column(String(50), nullable=True, index=True)
    target_model = Column(String(100), nullable=True, index=True)

    # Technique configuration
    technique_suites = Column(JSON, default=list)  # List of technique suite names
    transformation_config = Column(JSON, default=dict)  # Technique-specific settings

    # Campaign configuration
    config = Column(JSON, default=dict)
    # Example config structure:
    # {
    #     "max_attempts": 100,
    #     "potency_levels": [5, 7, 9],
    #     "timeout_seconds": 300,
    #     "retry_on_failure": true,
    #     "parallel_executions": 5
    # }

    # Tags for categorization and filtering
    tags = Column(JSON, default=list)  # ["security-audit", "research", "benchmark"]

    # Ownership and tracking
    user_id = Column(String(36), nullable=True, index=True)
    session_id = Column(String(36), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    telemetry_events = relationship(
        "CampaignTelemetryEvent",
        back_populates="campaign",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    results = relationship(
        "CampaignResult",
        back_populates="campaign",
        uselist=False,  # One-to-one relationship
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_campaigns_name", "name"),
        Index("idx_campaigns_status", "status"),
        Index("idx_campaigns_user_id", "user_id"),
        Index("idx_campaigns_created_at", "created_at"),
        Index("idx_campaigns_target", "target_provider", "target_model"),
        Index("idx_campaigns_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Campaign(id={self.id}, name={self.name}, status={self.status})>"


class CampaignTelemetryEvent(Base):
    """
    Individual telemetry event within a campaign.

    Captures detailed execution metrics for each prompt transformation
    and execution step during a campaign run.
    """

    __tablename__ = "campaign_telemetry_events"

    # Primary key
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Campaign foreign key
    campaign_id = Column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Event sequencing
    sequence_number = Column(Integer, nullable=False, default=0)
    iteration = Column(Integer, nullable=False, default=1)

    # Prompt data
    original_prompt = Column(Text, nullable=False)
    transformed_prompt = Column(Text, nullable=True)
    response_text = Column(Text, nullable=True)

    # Technique used
    technique_suite = Column(String(100), nullable=False, index=True)
    potency_level = Column(Integer, nullable=False, default=5)
    applied_techniques = Column(JSON, default=list)  # List of specific techniques applied

    # Provider/model info
    provider = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)

    # Execution metrics
    status = Column(
        String(20),
        default=ExecutionStatus.PENDING.value,
        nullable=False,
        index=True,
    )
    execution_time_ms = Column(Float, nullable=False, default=0.0)
    transformation_time_ms = Column(Float, nullable=False, default=0.0)
    total_latency_ms = Column(Float, nullable=False, default=0.0)

    # Token usage
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # Cost tracking (in USD cents)
    estimated_cost_cents = Column(Float, default=0.0)

    # Quality and success metrics
    success_indicator = Column(Boolean, default=False, nullable=False)
    semantic_success_score = Column(Float, nullable=True)  # 0.0 to 1.0
    effectiveness_score = Column(Float, nullable=True)  # 0.0 to 1.0
    naturalness_score = Column(Float, nullable=True)  # 0.0 to 1.0
    detectability_score = Column(Float, nullable=True)  # 0.0 to 1.0

    # Bypass detection
    bypass_indicators = Column(JSON, default=list)
    safety_trigger_detected = Column(Boolean, default=False)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)

    # Additional metadata
    metadata = Column(JSON, default=dict)
    # Example metadata:
    # {
    #     "request_id": "...",
    #     "finish_reason": "stop",
    #     "model_version": "...",
    #     "retry_count": 0
    # }

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    campaign = relationship("Campaign", back_populates="telemetry_events")

    __table_args__ = (
        Index("idx_telemetry_campaign_id", "campaign_id"),
        Index("idx_telemetry_technique", "technique_suite"),
        Index("idx_telemetry_provider_model", "provider", "model"),
        Index("idx_telemetry_status", "status"),
        Index("idx_telemetry_success", "success_indicator"),
        Index("idx_telemetry_created_at", "created_at"),
        Index("idx_telemetry_campaign_sequence", "campaign_id", "sequence_number"),
        Index("idx_telemetry_campaign_technique", "campaign_id", "technique_suite"),
        Index("idx_telemetry_campaign_status", "campaign_id", "status"),
        # Composite index for time-series queries
        Index("idx_telemetry_campaign_time", "campaign_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<CampaignTelemetryEvent(id={self.id}, campaign_id={self.campaign_id}, status={self.status})>"


class CampaignResult(Base):
    """
    Aggregated results for a completed campaign.

    Pre-computed statistical summaries for efficient analytics queries.
    Updated incrementally as telemetry events are processed.
    """

    __tablename__ = "campaign_results"

    # Primary key (same as campaign_id for one-to-one)
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Campaign foreign key
    campaign_id = Column(
        String(36),
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Aggregate counts
    total_attempts = Column(Integer, default=0, nullable=False)
    successful_attempts = Column(Integer, default=0, nullable=False)
    failed_attempts = Column(Integer, default=0, nullable=False)
    partial_success_attempts = Column(Integer, default=0, nullable=False)
    timeout_attempts = Column(Integer, default=0, nullable=False)
    skipped_attempts = Column(Integer, default=0, nullable=False)

    # Success rate statistics (pre-computed)
    success_rate = Column(Float, nullable=True)  # 0.0 to 1.0
    success_rate_mean = Column(Float, nullable=True)
    success_rate_median = Column(Float, nullable=True)
    success_rate_std_dev = Column(Float, nullable=True)
    success_rate_p95 = Column(Float, nullable=True)
    success_rate_p99 = Column(Float, nullable=True)

    # Semantic success statistics
    semantic_success_mean = Column(Float, nullable=True)
    semantic_success_median = Column(Float, nullable=True)
    semantic_success_std_dev = Column(Float, nullable=True)

    # Latency statistics (milliseconds)
    latency_mean = Column(Float, nullable=True)
    latency_median = Column(Float, nullable=True)
    latency_std_dev = Column(Float, nullable=True)
    latency_p50 = Column(Float, nullable=True)
    latency_p90 = Column(Float, nullable=True)
    latency_p95 = Column(Float, nullable=True)
    latency_p99 = Column(Float, nullable=True)
    latency_min = Column(Float, nullable=True)
    latency_max = Column(Float, nullable=True)

    # Token usage totals
    total_prompt_tokens = Column(Integer, default=0)
    total_completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # Cost totals (in USD cents)
    total_cost_cents = Column(Float, default=0.0)

    # Duration
    total_duration_seconds = Column(Float, nullable=True)

    # Technique breakdown (pre-computed JSON)
    technique_breakdown = Column(JSON, default=dict)
    # Example:
    # {
    #     "dan_persona": {"attempts": 10, "successes": 7, "rate": 0.7},
    #     "cognitive_hacking": {"attempts": 15, "successes": 9, "rate": 0.6}
    # }

    # Provider breakdown (pre-computed JSON)
    provider_breakdown = Column(JSON, default=dict)
    # Example:
    # {
    #     "openai": {"attempts": 20, "successes": 12, "rate": 0.6, "avg_latency": 1500},
    #     "anthropic": {"attempts": 15, "successes": 10, "rate": 0.67, "avg_latency": 1200}
    # }

    # Model breakdown (pre-computed JSON)
    model_breakdown = Column(JSON, default=dict)

    # Potency level breakdown (pre-computed JSON)
    potency_breakdown = Column(JSON, default=dict)
    # Example:
    # {
    #     "5": {"attempts": 20, "successes": 8, "rate": 0.4},
    #     "7": {"attempts": 15, "successes": 10, "rate": 0.67},
    #     "9": {"attempts": 10, "successes": 9, "rate": 0.9}
    # }

    # Time series data (hourly aggregates for charting)
    hourly_success_rates = Column(JSON, default=list)
    # Example: [{"hour": "2024-01-01T00:00:00Z", "rate": 0.6, "attempts": 10}, ...]

    # Best performing configurations
    best_technique = Column(String(100), nullable=True)
    best_potency_level = Column(Integer, nullable=True)
    best_provider = Column(String(50), nullable=True)
    best_model = Column(String(100), nullable=True)

    # Worst performing configurations
    worst_technique = Column(String(100), nullable=True)
    worst_potency_level = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    campaign = relationship("Campaign", back_populates="results")

    __table_args__ = (
        Index("idx_results_campaign_id", "campaign_id"),
        Index("idx_results_success_rate", "success_rate"),
        Index("idx_results_total_attempts", "total_attempts"),
        Index("idx_results_updated_at", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<CampaignResult(id={self.id}, campaign_id={self.campaign_id}, success_rate={self.success_rate})>"


# =============================================================================
# SQL Schema for Reference (PostgreSQL/SQLite compatible)
# =============================================================================

CAMPAIGN_SCHEMA_SQL = """
-- Campaigns table
CREATE TABLE IF NOT EXISTS campaigns (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    objective TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    target_provider VARCHAR(50),
    target_model VARCHAR(100),
    technique_suites JSONB DEFAULT '[]',
    transformation_config JSONB DEFAULT '{}',
    config JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    user_id VARCHAR(36),
    session_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX idx_campaigns_name ON campaigns(name);
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_user_id ON campaigns(user_id);
CREATE INDEX idx_campaigns_created_at ON campaigns(created_at);
CREATE INDEX idx_campaigns_target ON campaigns(target_provider, target_model);
CREATE INDEX idx_campaigns_status_created ON campaigns(status, created_at);

-- Campaign telemetry events table
CREATE TABLE IF NOT EXISTS campaign_telemetry_events (
    id VARCHAR(36) PRIMARY KEY,
    campaign_id VARCHAR(36) NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL DEFAULT 0,
    iteration INTEGER NOT NULL DEFAULT 1,
    original_prompt TEXT NOT NULL,
    transformed_prompt TEXT,
    response_text TEXT,
    technique_suite VARCHAR(100) NOT NULL,
    potency_level INTEGER NOT NULL DEFAULT 5,
    applied_techniques JSONB DEFAULT '[]',
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    execution_time_ms FLOAT NOT NULL DEFAULT 0.0,
    transformation_time_ms FLOAT NOT NULL DEFAULT 0.0,
    total_latency_ms FLOAT NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    estimated_cost_cents FLOAT DEFAULT 0.0,
    success_indicator BOOLEAN NOT NULL DEFAULT FALSE,
    semantic_success_score FLOAT,
    effectiveness_score FLOAT,
    naturalness_score FLOAT,
    detectability_score FLOAT,
    bypass_indicators JSONB DEFAULT '[]',
    safety_trigger_detected BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    error_code VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_telemetry_campaign_id ON campaign_telemetry_events(campaign_id);
CREATE INDEX idx_telemetry_technique ON campaign_telemetry_events(technique_suite);
CREATE INDEX idx_telemetry_provider_model ON campaign_telemetry_events(provider, model);
CREATE INDEX idx_telemetry_status ON campaign_telemetry_events(status);
CREATE INDEX idx_telemetry_success ON campaign_telemetry_events(success_indicator);
CREATE INDEX idx_telemetry_created_at ON campaign_telemetry_events(created_at);
CREATE INDEX idx_telemetry_campaign_sequence ON campaign_telemetry_events(campaign_id, sequence_number);
CREATE INDEX idx_telemetry_campaign_technique ON campaign_telemetry_events(campaign_id, technique_suite);
CREATE INDEX idx_telemetry_campaign_status ON campaign_telemetry_events(campaign_id, status);
CREATE INDEX idx_telemetry_campaign_time ON campaign_telemetry_events(campaign_id, created_at);

-- Campaign results table (aggregated metrics)
CREATE TABLE IF NOT EXISTS campaign_results (
    id VARCHAR(36) PRIMARY KEY,
    campaign_id VARCHAR(36) NOT NULL UNIQUE REFERENCES campaigns(id) ON DELETE CASCADE,
    total_attempts INTEGER NOT NULL DEFAULT 0,
    successful_attempts INTEGER NOT NULL DEFAULT 0,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    partial_success_attempts INTEGER NOT NULL DEFAULT 0,
    timeout_attempts INTEGER NOT NULL DEFAULT 0,
    skipped_attempts INTEGER NOT NULL DEFAULT 0,
    success_rate FLOAT,
    success_rate_mean FLOAT,
    success_rate_median FLOAT,
    success_rate_std_dev FLOAT,
    success_rate_p95 FLOAT,
    success_rate_p99 FLOAT,
    semantic_success_mean FLOAT,
    semantic_success_median FLOAT,
    semantic_success_std_dev FLOAT,
    latency_mean FLOAT,
    latency_median FLOAT,
    latency_std_dev FLOAT,
    latency_p50 FLOAT,
    latency_p90 FLOAT,
    latency_p95 FLOAT,
    latency_p99 FLOAT,
    latency_min FLOAT,
    latency_max FLOAT,
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_cents FLOAT DEFAULT 0.0,
    total_duration_seconds FLOAT,
    technique_breakdown JSONB DEFAULT '{}',
    provider_breakdown JSONB DEFAULT '{}',
    model_breakdown JSONB DEFAULT '{}',
    potency_breakdown JSONB DEFAULT '{}',
    hourly_success_rates JSONB DEFAULT '[]',
    best_technique VARCHAR(100),
    best_potency_level INTEGER,
    best_provider VARCHAR(50),
    best_model VARCHAR(100),
    worst_technique VARCHAR(100),
    worst_potency_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_results_campaign_id ON campaign_results(campaign_id);
CREATE INDEX idx_results_success_rate ON campaign_results(success_rate);
CREATE INDEX idx_results_total_attempts ON campaign_results(total_attempts);
CREATE INDEX idx_results_updated_at ON campaign_results(updated_at);
"""
