from enum import Enum as PyEnum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# We define a new Base for these models.
# In a fully integrated system, we might import a shared Base.
Base = declarative_base()


# =============================================================================
# User Role Enum for Database
# =============================================================================


class UserRole(str, PyEnum):
    """User roles matching the spec requirements"""

    ADMIN = "admin"  # Full system access
    RESEARCHER = "researcher"  # Create/edit campaigns, execute jailbreaks
    VIEWER = "viewer"  # Read-only access


class CampaignVisibility(str, PyEnum):
    """Campaign visibility levels for access control"""

    PRIVATE = "private"  # Only owner can access
    TEAM = "team"  # All users in the same organization/team can access
    PUBLIC = "public"  # All authenticated users can access


class CampaignSharePermission(str, PyEnum):
    """Permission levels for shared campaigns"""

    VIEW = "view"  # Read-only access to campaign
    EDIT = "edit"  # Can modify campaign content


# =============================================================================
# User and Authentication Models
# =============================================================================


class User(Base):
    """
    User model for multi-user authentication system.

    Supports email/password authentication with email verification,
    role-based access control, and per-user API key management.
    """

    __tablename__ = "users"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Authentication fields
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    # Role-based access control
    role = Column(
        Enum(UserRole, name="user_role", native_enum=False),
        default=UserRole.VIEWER,
        nullable=False,
    )

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Verification and reset tokens (stored hashed for security)
    email_verification_token = Column(String(255), nullable=True, index=True)
    email_verification_token_expires = Column(DateTime, nullable=True)
    password_reset_token = Column(String(255), nullable=True, index=True)
    password_reset_token_expires = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now(), nullable=True)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    api_keys = relationship(
        "UserAPIKey",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    preferences = relationship(
        "UserPreferences",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Table indexes for performance
    __table_args__ = (
        Index("ix_users_email_verified", "email", "is_verified"),
        Index("ix_users_role_active", "role", "is_active"),
    )

    def __repr__(self) -> str:
        return (
            f"<User(id={self.id}, username={self.username}, email={self.email}, role={self.role})>"
        )


class UserAPIKey(Base):
    """
    Per-user API key model for programmatic access.

    Allows users to create multiple API keys with optional expiration,
    names for identification, and usage tracking.
    """

    __tablename__ = "user_api_keys"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to user
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # API key fields
    name = Column(String(100), nullable=True)  # Optional name for the key
    key_prefix = Column(String(8), nullable=False)  # First 8 chars for identification
    hashed_key = Column(String(255), unique=True, nullable=False, index=True)

    # Status and expiration
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)  # Optional expiration

    # Usage tracking
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    # Table indexes
    __table_args__ = (
        Index("ix_user_api_keys_user_active", "user_id", "is_active"),
        Index("ix_user_api_keys_prefix", "key_prefix"),
    )

    def __repr__(self) -> str:
        return f"<UserAPIKey(id={self.id}, user_id={self.user_id}, name={self.name}, prefix={self.key_prefix})>"


class UserPreferences(Base):
    """
    User preferences and settings model.

    Stores user-specific settings for the application including
    default provider preferences, UI settings, and notification preferences.
    """

    __tablename__ = "user_preferences"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to user (one-to-one relationship)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Default provider/model preferences
    default_provider = Column(String(50), nullable=True)
    default_model = Column(String(100), nullable=True)

    # UI preferences (stored as JSON for flexibility)
    ui_settings = Column(JSON, default={}, nullable=False)

    # Notification preferences
    email_notifications = Column(Boolean, default=True, nullable=False)
    campaign_notifications = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now(), nullable=True)

    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self) -> str:
        return f"<UserPreferences(id={self.id}, user_id={self.user_id})>"


class DBLLMModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    provider = Column(String, nullable=False)  # Store enum value as string
    description = Column(String, nullable=True)
    config = Column(
        JSON, nullable=False, default={}
    )  # Store sanitized config, API keys should be in a secrets manager
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    evasion_tasks = relationship("DBEvasionTask", back_populates="target_model")


class DBEvasionTask(Base):
    __tablename__ = "evasion_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True, nullable=False)  # Celery task ID
    target_model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=False)
    initial_prompt = Column(String, nullable=False)
    strategy_chain = Column(JSON, nullable=False)  # List of MetamorphosisStrategyConfig
    success_criteria = Column(String, nullable=False)
    max_attempts = Column(Integer, nullable=False, default=1)
    status = Column(String, default="PENDING")  # Store enum value as string
    final_status = Column(String, nullable=True)  # E.g., "SUCCESS", "FAILURE", "PARTIAL_SUCCESS"
    overall_success = Column(Boolean, default=False)
    failed_reason = Column(String, nullable=True)
    results = Column(JSON, default=[])  # List of EvasionAttemptResult
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)

    # Optional link to a dataset prompt
    jailbreak_prompt_id = Column(Integer, ForeignKey("jailbreak_prompts.id"), nullable=True)
    target_model = relationship("DBLLMModel", back_populates="evasion_tasks")
    jailbreak_prompt = relationship("JailbreakPrompt", back_populates="evasion_tasks")


# New models for integrated datasets
class JailbreakDataset(Base):
    __tablename__ = "jailbreak_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    license = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    prompts = relationship(
        "JailbreakPrompt", back_populates="dataset", cascade="all, delete-orphan"
    )


class JailbreakPrompt(Base):
    __tablename__ = "jailbreak_prompts"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("jailbreak_datasets.id"), nullable=False)
    prompt_text = Column(String, nullable=False)
    jailbreak_type = Column(String, nullable=True)
    is_jailbreak = Column(Boolean, default=True)
    platform = Column(String, nullable=True)
    source = Column(String, nullable=True)
    community = Column(String, nullable=True)
    source_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    dataset = relationship("JailbreakDataset", back_populates="prompts")
    evasion_tasks = relationship("DBEvasionTask", back_populates="jailbreak_prompt")


# =============================================================================
# Campaign Models with Ownership and Sharing
# =============================================================================


class Campaign(Base):
    """
    Campaign model for organizing jailbreak research and prompt testing.

    Supports ownership, visibility levels (private/team/public), and explicit
    sharing with specific users via CampaignShare. Campaigns group related
    jailbreak attempts and track research progress.
    """

    __tablename__ = "campaigns"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Ownership - links to the user who created the campaign
    owner_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Campaign metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Visibility control
    visibility = Column(
        Enum(CampaignVisibility, name="campaign_visibility", native_enum=False),
        default=CampaignVisibility.PRIVATE,
        nullable=False,
    )

    # Campaign status
    is_active = Column(Boolean, default=True, nullable=False)
    is_archived = Column(Boolean, default=False, nullable=False)

    # Campaign configuration (stored as JSON for flexibility)
    config = Column(JSON, default={}, nullable=False)

    # Optional target model preference for the campaign
    target_provider = Column(String(50), nullable=True)
    target_model = Column(String(100), nullable=True)

    # Statistics (denormalized for performance)
    total_attempts = Column(Integer, default=0, nullable=False)
    successful_attempts = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now(), nullable=True)
    last_activity_at = Column(DateTime, nullable=True)

    # Relationships
    owner = relationship("User", backref="owned_campaigns", lazy="joined")
    shares = relationship(
        "CampaignShare",
        back_populates="campaign",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    # Table indexes for performance
    __table_args__ = (
        Index("ix_campaigns_owner_active", "owner_id", "is_active"),
        Index("ix_campaigns_visibility", "visibility"),
        Index("ix_campaigns_owner_visibility", "owner_id", "visibility"),
        Index("ix_campaigns_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Campaign(id={self.id}, name={self.name}, owner_id={self.owner_id}, visibility={self.visibility})>"


class CampaignShare(Base):
    """
    Explicit campaign sharing model for granting access to specific users.

    When a campaign has PRIVATE visibility, the owner can still share it with
    specific users via this model. Each share grants either VIEW or EDIT
    permissions to the target user.
    """

    __tablename__ = "campaign_shares"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    campaign_id = Column(
        Integer,
        ForeignKey("campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Permission level
    permission = Column(
        Enum(CampaignSharePermission, name="campaign_share_permission", native_enum=False),
        default=CampaignSharePermission.VIEW,
        nullable=False,
    )

    # Who shared this campaign (for audit purposes)
    shared_by_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now(), nullable=True)

    # Relationships
    campaign = relationship("Campaign", back_populates="shares")
    user = relationship("User", foreign_keys=[user_id], backref="shared_campaigns")
    shared_by = relationship("User", foreign_keys=[shared_by_id])

    # Table indexes and constraints
    __table_args__ = (
        # Ensure a user can only have one share entry per campaign
        Index(
            "ix_campaign_shares_unique",
            "campaign_id",
            "user_id",
            unique=True,
        ),
        Index("ix_campaign_shares_user_permission", "user_id", "permission"),
    )

    def __repr__(self) -> str:
        return f"<CampaignShare(id={self.id}, campaign_id={self.campaign_id}, user_id={self.user_id}, permission={self.permission})>"


# =============================================================================
# Assessment Model (Security Testing Assessments)
# =============================================================================


class AssessmentStatus(str, PyEnum):
    """Assessment status levels"""

    PENDING = "pending"  # Assessment queued
    RUNNING = "running"  # Assessment in progress
    COMPLETED = "completed"  # Assessment finished successfully
    FAILED = "failed"  # Assessment encountered errors
    CANCELLED = "cancelled"  # Assessment was cancelled


class Assessment(Base):
    """
    Security assessment model for storing vulnerability testing results.

    Each assessment represents a complete security evaluation against
    a target LLM provider/model with specific techniques.
    """

    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, index=True)

    # Ownership
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Assessment metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(
        Enum(AssessmentStatus, name="assessment_status", native_enum=False),
        default=AssessmentStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Target configuration
    target_provider = Column(String(100), nullable=False)
    target_model = Column(String(100), nullable=False)
    target_config = Column(JSON, default=dict)  # Additional provider config

    # Techniques and results
    technique_ids = Column(JSON, default=list)  # List of technique IDs used
    results = Column(JSON, default=dict)  # Assessment results
    findings_count = Column(Integer, default=0)
    vulnerabilities_found = Column(Integer, default=0)

    # Risk scoring
    risk_score = Column(Integer, default=0)  # 0-100 scale
    risk_level = Column(String(20), default="info")  # critical/high/medium/low/info

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, onupdate=func.now(), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", backref="assessments")

    # Indexes
    __table_args__ = (
        Index("ix_assessments_user_status", "user_id", "status"),
        Index("ix_assessments_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Assessment(id={self.id}, name={self.name}, status={self.status})>"
