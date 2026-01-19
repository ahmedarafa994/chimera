-- =============================================================================
-- CHIMERA AI RESEARCH PLATFORM - COMPREHENSIVE DATABASE SCHEMA
-- =============================================================================
-- Version: 2.0.0
-- Target: PostgreSQL 14+ (with JSONB support for analytics)
-- Features: Multi-user auth, campaigns, real-time telemetry, audit logging
-- Performance: Optimized indexes, partitioning, and query patterns
-- =============================================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS "pgcrypto"; -- For secure hashing

-- =============================================================================
-- ENUMS AND TYPES
-- =============================================================================

-- User management enums
CREATE TYPE user_role AS ENUM ('admin', 'researcher', 'viewer');
CREATE TYPE account_status AS ENUM ('active', 'suspended', 'pending_verification', 'deactivated');

-- Campaign and research enums
CREATE TYPE campaign_status AS ENUM ('draft', 'running', 'paused', 'completed', 'failed', 'cancelled');
CREATE TYPE execution_status AS ENUM ('pending', 'success', 'partial_success', 'failure', 'timeout', 'skipped');
CREATE TYPE campaign_visibility AS ENUM ('private', 'team', 'public');
CREATE TYPE share_permission AS ENUM ('view', 'edit');

-- Prompt library enums
CREATE TYPE technique_type AS ENUM (
    'autodan', 'gptfuzz', 'chimera_framing', 'pair', 'gcg',
    'tap', 'crescendo', 'manual', 'other'
);
CREATE TYPE vulnerability_type AS ENUM (
    'jailbreak', 'injection', 'pii_leak', 'bypass', 'malicious_content',
    'code_execution', 'denial_of_service', 'social_engineering', 'other'
);
CREATE TYPE template_status AS ENUM ('draft', 'active', 'archived', 'deprecated');
CREATE TYPE sharing_level AS ENUM ('private', 'team', 'public');

-- Telemetry and monitoring enums
CREATE TYPE telemetry_event_type AS ENUM (
    'campaign_started', 'campaign_paused', 'campaign_resumed', 'campaign_completed',
    'iteration_started', 'iteration_completed', 'attack_started', 'attack_completed',
    'technique_applied', 'cost_update', 'token_usage', 'latency_update',
    'prompt_evolved', 'score_update', 'heartbeat', 'connection_ack', 'error'
);

-- Audit enums
CREATE TYPE audit_action AS ENUM (
    'auth.login', 'auth.logout', 'auth.failed', 'auth.token_refresh',
    'apikey.created', 'apikey.rotated', 'apikey.revoked', 'apikey.used',
    'prompt.transform', 'prompt.enhance', 'prompt.jailbreak', 'prompt.batch_process',
    'config.change', 'config.view', 'user.create', 'user.modify', 'user.delete',
    'user.role_change', 'campaign.create', 'campaign.update', 'campaign.delete',
    'system.startup', 'system.shutdown', 'security.rate_limit', 'security.breach'
);
CREATE TYPE audit_severity AS ENUM ('info', 'warning', 'error', 'critical');

-- =============================================================================
-- CORE USER MANAGEMENT TABLES
-- =============================================================================

-- Users table with comprehensive authentication support
CREATE TABLE users (
    id SERIAL PRIMARY KEY,

    -- Authentication fields
    email VARCHAR(255) NOT NULL UNIQUE,
    username VARCHAR(100) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,

    -- User profile
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    display_name VARCHAR(200),

    -- Role-based access control
    role user_role NOT NULL DEFAULT 'viewer',
    account_status account_status NOT NULL DEFAULT 'pending_verification',

    -- Email verification
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    email_verification_token VARCHAR(255),
    email_verification_token_expires TIMESTAMPTZ,

    -- Password reset
    password_reset_token VARCHAR(255),
    password_reset_token_expires TIMESTAMPTZ,

    -- Security tracking
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ,

    -- Activity tracking
    last_login TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,

    -- Preferences (stored as JSONB for flexibility)
    preferences JSONB DEFAULT '{}',

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    created_by INTEGER REFERENCES users(id),

    -- Soft delete support
    deleted_at TIMESTAMPTZ,
    deleted_by INTEGER REFERENCES users(id)
);

-- User API Keys for programmatic access
CREATE TABLE user_api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- API key identification
    name VARCHAR(100),
    key_prefix VARCHAR(8) NOT NULL,
    hashed_key VARCHAR(255) NOT NULL UNIQUE,

    -- Permissions and scopes
    scopes JSONB DEFAULT '[]', -- Array of permission scopes

    -- Status and expiration
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMPTZ,

    -- Usage tracking
    last_used_at TIMESTAMPTZ,
    usage_count BIGINT DEFAULT 0,
    rate_limit_per_hour INTEGER DEFAULT 3600,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    revoked_by INTEGER REFERENCES users(id)
);

-- User sessions for JWT token management
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Session identification
    session_token VARCHAR(255) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(255),

    -- Session metadata
    ip_address INET,
    user_agent TEXT,
    device_fingerprint VARCHAR(255),

    -- Security tracking
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    revoked_reason VARCHAR(100)
);

-- =============================================================================
-- CAMPAIGN MANAGEMENT TABLES
-- =============================================================================

-- Research campaigns for adversarial testing
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Basic campaign info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    objective TEXT NOT NULL,

    -- Ownership and access control
    created_by INTEGER NOT NULL REFERENCES users(id),
    visibility campaign_visibility NOT NULL DEFAULT 'private',

    -- Campaign status and lifecycle
    status campaign_status NOT NULL DEFAULT 'draft',

    -- Target configuration
    target_provider VARCHAR(50),
    target_model VARCHAR(100),

    -- Technique configuration (stored as JSONB for flexibility)
    technique_suites JSONB DEFAULT '[]',
    transformation_config JSONB DEFAULT '{}',

    -- Campaign settings
    config JSONB DEFAULT '{}',
    -- Example config:
    -- {
    --   "max_attempts": 100,
    --   "potency_levels": [5, 7, 9],
    --   "timeout_seconds": 300,
    --   "retry_on_failure": true,
    --   "parallel_executions": 5,
    --   "cost_limits": {"max_usd": 50.00}
    -- }

    -- Progress tracking
    total_iterations INTEGER DEFAULT 0,
    completed_iterations INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4), -- e.g., 0.1234 = 12.34%

    -- Resource tracking
    estimated_cost DECIMAL(10,4),
    actual_cost DECIMAL(10,4),
    token_usage_total BIGINT DEFAULT 0,

    -- Schedule and timing
    scheduled_start TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Soft delete
    deleted_at TIMESTAMPTZ,
    deleted_by INTEGER REFERENCES users(id)
);

-- Campaign sharing and collaboration
CREATE TABLE campaign_shares (
    id SERIAL PRIMARY KEY,
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    permission share_permission NOT NULL DEFAULT 'view',

    -- Sharing metadata
    shared_by INTEGER NOT NULL REFERENCES users(id),
    shared_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    -- Access tracking
    last_accessed_at TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,

    UNIQUE(campaign_id, user_id)
);

-- Campaign execution results and telemetry
CREATE TABLE campaign_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,

    -- Execution metadata
    iteration_number INTEGER NOT NULL,
    execution_status execution_status NOT NULL DEFAULT 'pending',

    -- Prompt and technique details
    original_prompt TEXT NOT NULL,
    transformed_prompt TEXT,
    technique_applied VARCHAR(100),
    technique_config JSONB DEFAULT '{}',

    -- Results
    llm_response TEXT,
    success_score DECIMAL(5,4), -- 0.0 to 1.0
    is_successful BOOLEAN,

    -- Performance metrics
    latency_ms INTEGER,
    token_count_input INTEGER,
    token_count_output INTEGER,
    cost_usd DECIMAL(10,6),

    -- Provider details
    provider_used VARCHAR(50),
    model_used VARCHAR(100),

    -- Error handling
    error_message TEXT,
    error_code VARCHAR(50),

    -- Timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Indexing for time-series queries
    CONSTRAINT campaign_executions_campaign_iteration_unique
        UNIQUE(campaign_id, iteration_number)
);

-- =============================================================================
-- PROMPT LIBRARY AND TEMPLATE MANAGEMENT
-- =============================================================================

-- Prompt templates for community sharing
CREATE TABLE prompt_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Template identification
    title VARCHAR(255) NOT NULL,
    description TEXT,

    -- Content
    prompt_text TEXT NOT NULL,

    -- Classification
    technique_type technique_type NOT NULL,
    vulnerability_type vulnerability_type,
    tags JSONB DEFAULT '[]', -- Array of tags for categorization

    -- Template metadata
    template_status template_status NOT NULL DEFAULT 'draft',
    sharing_level sharing_level NOT NULL DEFAULT 'private',

    -- Versioning
    version_number INTEGER NOT NULL DEFAULT 1,
    parent_template_id UUID REFERENCES prompt_templates(id),
    is_latest_version BOOLEAN NOT NULL DEFAULT TRUE,

    -- Effectiveness tracking
    success_rate DECIMAL(5,4),
    usage_count INTEGER DEFAULT 0,

    -- Community feedback
    average_rating DECIMAL(3,2), -- 1.00 to 5.00
    total_ratings INTEGER DEFAULT 0,
    effectiveness_votes_positive INTEGER DEFAULT 0,
    effectiveness_votes_total INTEGER DEFAULT 0,

    -- Ownership
    created_by INTEGER NOT NULL REFERENCES users(id),

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Soft delete
    deleted_at TIMESTAMPTZ,
    deleted_by INTEGER REFERENCES users(id)
);

-- Template ratings and reviews
CREATE TABLE template_ratings (
    id SERIAL PRIMARY KEY,
    template_id UUID NOT NULL REFERENCES prompt_templates(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Rating details
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    effectiveness_vote BOOLEAN NOT NULL, -- true = effective, false = ineffective
    review_comment TEXT,

    -- Usage context (optional)
    used_in_campaign_id UUID REFERENCES campaigns(id),
    success_achieved BOOLEAN,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    UNIQUE(template_id, user_id)
);

-- Template usage tracking
CREATE TABLE template_usage (
    id SERIAL PRIMARY KEY,
    template_id UUID NOT NULL REFERENCES prompt_templates(id) ON DELETE CASCADE,
    used_by INTEGER NOT NULL REFERENCES users(id),
    campaign_id UUID REFERENCES campaigns(id),

    -- Usage context
    original_prompt TEXT,
    customizations_made JSONB,

    -- Results
    was_successful BOOLEAN,
    success_score DECIMAL(5,4),

    -- Timing
    used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- REAL-TIME TELEMETRY AND ANALYTICS
-- =============================================================================

-- Real-time telemetry events (partitioned for performance)
CREATE TABLE telemetry_events (
    id BIGSERIAL,
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),

    -- Event classification
    event_type telemetry_event_type NOT NULL,
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),

    -- Event data (flexible JSONB for different event types)
    payload JSONB NOT NULL,

    -- Event metadata
    session_id VARCHAR(255),
    correlation_id UUID,

    -- Performance metrics
    latency_ms INTEGER,
    cost_usd DECIMAL(10,6),
    tokens_used INTEGER,

    -- Timing (critical for time-series analysis)
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Source tracking
    source_ip INET,
    user_agent TEXT,

    PRIMARY KEY (id, event_timestamp)
) PARTITION BY RANGE (event_timestamp);

-- Create monthly partitions for telemetry events
CREATE TABLE telemetry_events_y2024m01 PARTITION OF telemetry_events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE telemetry_events_y2024m02 PARTITION OF telemetry_events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Note: Add more partitions as needed for production deployment

-- Analytics aggregates (pre-computed for performance)
CREATE TABLE campaign_analytics (
    id SERIAL PRIMARY KEY,
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,

    -- Time window for aggregation
    time_window_start TIMESTAMPTZ NOT NULL,
    time_window_end TIMESTAMPTZ NOT NULL,
    window_duration INTERVAL NOT NULL,

    -- Success metrics
    total_attempts INTEGER DEFAULT 0,
    successful_attempts INTEGER DEFAULT 0,
    success_rate DECIMAL(5,4),

    -- Performance metrics
    avg_latency_ms DECIMAL(10,2),
    min_latency_ms INTEGER,
    max_latency_ms INTEGER,
    p95_latency_ms DECIMAL(10,2),

    -- Cost metrics
    total_cost_usd DECIMAL(10,4),
    cost_per_attempt DECIMAL(10,6),
    total_tokens BIGINT,

    -- Technique effectiveness
    technique_breakdown JSONB, -- {"autodan": {"attempts": 50, "successes": 12}, ...}

    -- Provider performance
    provider_breakdown JSONB,

    -- Computed at
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(campaign_id, time_window_start, window_duration)
);

-- =============================================================================
-- PROVIDER AND MODEL MANAGEMENT
-- =============================================================================

-- LLM Providers and configurations
CREATE TABLE llm_providers (
    id SERIAL PRIMARY KEY,

    -- Provider identification
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200),
    description TEXT,

    -- Provider configuration
    provider_type VARCHAR(50) NOT NULL, -- 'openai', 'anthropic', 'google', etc.
    base_url VARCHAR(500),
    api_version VARCHAR(20),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    health_status VARCHAR(20) DEFAULT 'unknown', -- 'healthy', 'degraded', 'down'
    last_health_check TIMESTAMPTZ,

    -- Configuration schema (JSONB for flexibility)
    default_config JSONB DEFAULT '{}',
    required_config_keys JSONB DEFAULT '[]',

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Available LLM models
CREATE TABLE llm_models (
    id SERIAL PRIMARY KEY,
    provider_id INTEGER NOT NULL REFERENCES llm_providers(id),

    -- Model identification
    model_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200),
    description TEXT,

    -- Model capabilities
    max_tokens INTEGER,
    supports_function_calling BOOLEAN DEFAULT FALSE,
    supports_streaming BOOLEAN DEFAULT FALSE,
    supports_json_mode BOOLEAN DEFAULT FALSE,

    -- Pricing (tokens per USD)
    cost_per_input_token DECIMAL(12,8),
    cost_per_output_token DECIMAL(12,8),

    -- Model metadata
    model_version VARCHAR(50),
    release_date DATE,
    deprecation_date DATE,

    -- Status
    is_available BOOLEAN NOT NULL DEFAULT TRUE,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    UNIQUE(provider_id, model_name)
);

-- User API key storage (encrypted)
CREATE TABLE user_provider_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider_id INTEGER NOT NULL REFERENCES llm_providers(id) ON DELETE CASCADE,

    -- Encrypted API key storage
    encrypted_api_key BYTEA NOT NULL,
    key_name VARCHAR(100), -- User-friendly name

    -- Key metadata
    key_prefix VARCHAR(10), -- First few chars for identification
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Usage tracking
    last_used_at TIMESTAMPTZ,
    usage_count BIGINT DEFAULT 0,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,

    UNIQUE(user_id, provider_id, key_name)
);

-- =============================================================================
-- AUDIT AND COMPLIANCE TABLES
-- =============================================================================

-- Comprehensive audit log (tamper-evident with hash chain)
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,

    -- Event identification
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    action audit_action NOT NULL,

    -- Actor information
    user_id INTEGER REFERENCES users(id),
    session_id VARCHAR(255),
    api_key_id INTEGER REFERENCES user_api_keys(id),

    -- Target resource
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),

    -- Event details
    details JSONB,
    severity audit_severity NOT NULL DEFAULT 'info',

    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(255),

    -- Security context
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),

    -- Timing
    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Hash chain for tamper detection
    event_hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64),

    -- Retention policy
    retention_until TIMESTAMPTZ
);

-- System configuration audit
CREATE TABLE config_audit (
    id SERIAL PRIMARY KEY,

    -- Configuration details
    config_key VARCHAR(255) NOT NULL,
    old_value JSONB,
    new_value JSONB,

    -- Change metadata
    changed_by INTEGER REFERENCES users(id),
    change_reason TEXT,

    -- Approval workflow
    requires_approval BOOLEAN DEFAULT FALSE,
    approved_by INTEGER REFERENCES users(id),
    approved_at TIMESTAMPTZ,

    -- Timing
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Data retention policies
CREATE TABLE data_retention_policies (
    id SERIAL PRIMARY KEY,

    -- Policy identification
    policy_name VARCHAR(100) NOT NULL UNIQUE,
    table_name VARCHAR(100) NOT NULL,

    -- Retention rules
    retention_period INTERVAL NOT NULL,
    retention_condition TEXT, -- SQL WHERE clause

    -- Actions
    action VARCHAR(20) NOT NULL DEFAULT 'DELETE', -- 'DELETE' or 'ARCHIVE'
    archive_table VARCHAR(100),

    -- Policy status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Execution tracking
    last_executed TIMESTAMPTZ,
    next_execution TIMESTAMPTZ,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by INTEGER REFERENCES users(id)
);

-- =============================================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- =============================================================================

-- User management indexes
CREATE INDEX CONCURRENTLY idx_users_email_verified ON users(email, is_verified);
CREATE INDEX CONCURRENTLY idx_users_role_status ON users(role, account_status);
CREATE INDEX CONCURRENTLY idx_users_created_at ON users(created_at);
CREATE INDEX CONCURRENTLY idx_users_last_activity ON users(last_activity) WHERE account_status = 'active';

-- API key indexes
CREATE INDEX CONCURRENTLY idx_user_api_keys_active ON user_api_keys(user_id, is_active);
CREATE INDEX CONCURRENTLY idx_user_api_keys_prefix ON user_api_keys(key_prefix);
CREATE INDEX CONCURRENTLY idx_user_api_keys_last_used ON user_api_keys(last_used_at DESC);

-- Session indexes
CREATE INDEX CONCURRENTLY idx_user_sessions_active ON user_sessions(user_id, is_active);
CREATE INDEX CONCURRENTLY idx_user_sessions_expires ON user_sessions(expires_at) WHERE is_active = true;
CREATE INDEX CONCURRENTLY idx_user_sessions_token ON user_sessions(session_token);

-- Campaign indexes
CREATE INDEX CONCURRENTLY idx_campaigns_owner_status ON campaigns(created_by, status);
CREATE INDEX CONCURRENTLY idx_campaigns_status_created ON campaigns(status, created_at);
CREATE INDEX CONCURRENTLY idx_campaigns_visibility ON campaigns(visibility) WHERE deleted_at IS NULL;
CREATE INDEX CONCURRENTLY idx_campaigns_provider_model ON campaigns(target_provider, target_model);

-- Campaign execution indexes (critical for analytics)
CREATE INDEX CONCURRENTLY idx_campaign_executions_campaign_time ON campaign_executions(campaign_id, started_at);
CREATE INDEX CONCURRENTLY idx_campaign_executions_status ON campaign_executions(execution_status);
CREATE INDEX CONCURRENTLY idx_campaign_executions_success ON campaign_executions(is_successful, started_at);
CREATE INDEX CONCURRENTLY idx_campaign_executions_cost ON campaign_executions(cost_usd) WHERE cost_usd IS NOT NULL;

-- Prompt template indexes
CREATE INDEX CONCURRENTLY idx_prompt_templates_status_sharing ON prompt_templates(template_status, sharing_level);
CREATE INDEX CONCURRENTLY idx_prompt_templates_technique ON prompt_templates(technique_type);
CREATE INDEX CONCURRENTLY idx_prompt_templates_rating ON prompt_templates(average_rating DESC) WHERE template_status = 'active';
CREATE INDEX CONCURRENTLY idx_prompt_templates_tags ON prompt_templates USING GIN(tags);
CREATE INDEX CONCURRENTLY idx_prompt_templates_created ON prompt_templates(created_at DESC);

-- Full-text search on prompt templates
CREATE INDEX CONCURRENTLY idx_prompt_templates_search ON prompt_templates USING GIN(
    to_tsvector('english', title || ' ' || coalesce(description, '') || ' ' || prompt_text)
);

-- Telemetry indexes (time-series optimized)
CREATE INDEX CONCURRENTLY idx_telemetry_events_campaign_time ON telemetry_events(campaign_id, event_timestamp);
CREATE INDEX CONCURRENTLY idx_telemetry_events_type_time ON telemetry_events(event_type, event_timestamp);
CREATE INDEX CONCURRENTLY idx_telemetry_events_user_time ON telemetry_events(user_id, event_timestamp);

-- Analytics indexes
CREATE INDEX CONCURRENTLY idx_campaign_analytics_time_window ON campaign_analytics(campaign_id, time_window_start);
CREATE INDEX CONCURRENTLY idx_campaign_analytics_computed ON campaign_analytics(computed_at DESC);

-- Audit log indexes (security and compliance critical)
CREATE INDEX CONCURRENTLY idx_audit_log_user_time ON audit_log(user_id, event_timestamp);
CREATE INDEX CONCURRENTLY idx_audit_log_action_time ON audit_log(action, event_timestamp);
CREATE INDEX CONCURRENTLY idx_audit_log_severity ON audit_log(severity, event_timestamp) WHERE severity IN ('error', 'critical');
CREATE INDEX CONCURRENTLY idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX CONCURRENTLY idx_audit_log_ip ON audit_log(ip_address, event_timestamp);

-- =============================================================================
-- TRIGGERS AND CONSTRAINTS
-- =============================================================================

-- Update timestamps automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prompt_templates_updated_at BEFORE UPDATE ON prompt_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Audit log hash chain trigger
CREATE OR REPLACE FUNCTION generate_audit_hash()
RETURNS TRIGGER AS $$
DECLARE
    last_hash VARCHAR(64);
    event_data TEXT;
BEGIN
    -- Get the previous hash
    SELECT event_hash INTO last_hash
    FROM audit_log
    ORDER BY id DESC
    LIMIT 1;

    -- Create hash input from event data
    event_data := NEW.event_id::text || NEW.action || NEW.event_timestamp::text ||
                  COALESCE(NEW.user_id::text, '') || COALESCE(NEW.details::text, '') ||
                  COALESCE(last_hash, '');

    -- Generate hash
    NEW.event_hash := encode(digest(event_data, 'sha256'), 'hex');
    NEW.previous_hash := last_hash;

    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER audit_log_hash_trigger BEFORE INSERT ON audit_log
    FOR EACH ROW EXECUTE FUNCTION generate_audit_hash();

-- Campaign analytics aggregation trigger
CREATE OR REPLACE FUNCTION update_campaign_analytics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update campaign success rate and costs
    UPDATE campaigns SET
        completed_iterations = completed_iterations + 1,
        actual_cost = COALESCE(actual_cost, 0) + COALESCE(NEW.cost_usd, 0),
        token_usage_total = token_usage_total + COALESCE(NEW.token_count_input, 0) + COALESCE(NEW.token_count_output, 0),
        success_rate = (
            SELECT COUNT(*)::DECIMAL / COUNT(*)
            FROM campaign_executions
            WHERE campaign_id = NEW.campaign_id AND is_successful = true
        )
    WHERE id = NEW.campaign_id;

    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER campaign_execution_analytics_trigger AFTER INSERT ON campaign_executions
    FOR EACH ROW EXECUTE FUNCTION update_campaign_analytics();

-- Template rating aggregation trigger
CREATE OR REPLACE FUNCTION update_template_ratings()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE prompt_templates SET
        average_rating = (
            SELECT AVG(rating)::DECIMAL(3,2)
            FROM template_ratings
            WHERE template_id = NEW.template_id
        ),
        total_ratings = (
            SELECT COUNT(*)
            FROM template_ratings
            WHERE template_id = NEW.template_id
        ),
        effectiveness_votes_positive = (
            SELECT COUNT(*)
            FROM template_ratings
            WHERE template_id = NEW.template_id AND effectiveness_vote = true
        ),
        effectiveness_votes_total = (
            SELECT COUNT(*)
            FROM template_ratings
            WHERE template_id = NEW.template_id
        )
    WHERE id = NEW.template_id;

    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER template_rating_aggregation_trigger AFTER INSERT OR UPDATE OR DELETE ON template_ratings
    FOR EACH ROW EXECUTE FUNCTION update_template_ratings();

-- =============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================================================

-- Enable RLS on sensitive tables
ALTER TABLE campaigns ENABLE ROW LEVEL SECURITY;
ALTER TABLE prompt_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE campaign_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemetry_events ENABLE ROW LEVEL SECURITY;

-- Campaign access policies
CREATE POLICY campaign_owner_access ON campaigns
    FOR ALL TO authenticated_users
    USING (created_by = current_user_id());

CREATE POLICY campaign_shared_access ON campaigns
    FOR SELECT TO authenticated_users
    USING (
        visibility = 'public' OR
        id IN (
            SELECT campaign_id FROM campaign_shares
            WHERE user_id = current_user_id()
        )
    );

-- Prompt template access policies
CREATE POLICY template_owner_access ON prompt_templates
    FOR ALL TO authenticated_users
    USING (created_by = current_user_id());

CREATE POLICY template_public_access ON prompt_templates
    FOR SELECT TO authenticated_users
    USING (sharing_level IN ('public', 'team') AND template_status = 'active');

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Campaign dashboard view
CREATE VIEW campaign_dashboard AS
SELECT
    c.id,
    c.name,
    c.description,
    c.status,
    c.success_rate,
    c.completed_iterations,
    c.total_iterations,
    c.actual_cost,
    c.target_provider,
    c.target_model,
    c.created_at,
    c.started_at,
    c.completed_at,
    u.username as created_by_username,
    -- Latest execution details
    (SELECT execution_status FROM campaign_executions
     WHERE campaign_id = c.id
     ORDER BY iteration_number DESC
     LIMIT 1) as latest_execution_status,
    -- Recent activity
    (SELECT COUNT(*) FROM campaign_executions
     WHERE campaign_id = c.id AND started_at > NOW() - INTERVAL '1 hour') as executions_last_hour
FROM campaigns c
JOIN users u ON c.created_by = u.id
WHERE c.deleted_at IS NULL;

-- Template library view
CREATE VIEW template_library AS
SELECT
    pt.id,
    pt.title,
    pt.description,
    pt.technique_type,
    pt.vulnerability_type,
    pt.average_rating,
    pt.total_ratings,
    pt.usage_count,
    pt.template_status,
    pt.sharing_level,
    pt.created_at,
    u.username as created_by_username,
    pt.tags
FROM prompt_templates pt
JOIN users u ON pt.created_by = u.id
WHERE pt.deleted_at IS NULL AND pt.template_status = 'active';

-- User activity summary view
CREATE VIEW user_activity_summary AS
SELECT
    u.id,
    u.username,
    u.email,
    u.role,
    u.last_login,
    u.last_activity,
    -- Campaign statistics
    COUNT(DISTINCT c.id) as campaigns_created,
    COUNT(DISTINCT pt.id) as templates_created,
    COUNT(DISTINCT tr.id) as ratings_given,
    -- Recent activity
    (SELECT COUNT(*) FROM audit_log al
     WHERE al.user_id = u.id AND al.event_timestamp > NOW() - INTERVAL '7 days') as actions_last_week
FROM users u
LEFT JOIN campaigns c ON u.id = c.created_by AND c.deleted_at IS NULL
LEFT JOIN prompt_templates pt ON u.id = pt.created_by AND pt.deleted_at IS NULL
LEFT JOIN template_ratings tr ON u.id = tr.user_id
WHERE u.account_status = 'active'
GROUP BY u.id, u.username, u.email, u.role, u.last_login, u.last_activity;

-- =============================================================================
-- STORED PROCEDURES FOR COMMON OPERATIONS
-- =============================================================================

-- Campaign analytics computation procedure
CREATE OR REPLACE FUNCTION compute_campaign_analytics(
    p_campaign_id UUID,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_window_duration INTERVAL DEFAULT INTERVAL '1 hour'
)
RETURNS void AS $$
DECLARE
    window_start TIMESTAMPTZ := p_start_time;
    window_end TIMESTAMPTZ;
BEGIN
    WHILE window_start < p_end_time LOOP
        window_end := window_start + p_window_duration;

        INSERT INTO campaign_analytics (
            campaign_id, time_window_start, time_window_end, window_duration,
            total_attempts, successful_attempts, success_rate,
            avg_latency_ms, min_latency_ms, max_latency_ms,
            total_cost_usd, total_tokens,
            technique_breakdown, provider_breakdown
        )
        SELECT
            p_campaign_id,
            window_start,
            window_end,
            p_window_duration,
            COUNT(*),
            COUNT(*) FILTER (WHERE is_successful = true),
            (COUNT(*) FILTER (WHERE is_successful = true))::DECIMAL / GREATEST(COUNT(*), 1),
            AVG(latency_ms),
            MIN(latency_ms),
            MAX(latency_ms),
            SUM(COALESCE(cost_usd, 0)),
            SUM(COALESCE(token_count_input, 0) + COALESCE(token_count_output, 0)),
            jsonb_object_agg(technique_applied, jsonb_build_object(
                'attempts', technique_count,
                'successes', technique_successes
            )) FILTER (WHERE technique_applied IS NOT NULL),
            jsonb_object_agg(provider_used, jsonb_build_object(
                'attempts', provider_count,
                'successes', provider_successes,
                'avg_latency', provider_avg_latency
            )) FILTER (WHERE provider_used IS NOT NULL)
        FROM (
            SELECT *,
                   COUNT(*) OVER (PARTITION BY technique_applied) as technique_count,
                   COUNT(*) FILTER (WHERE is_successful = true) OVER (PARTITION BY technique_applied) as technique_successes,
                   COUNT(*) OVER (PARTITION BY provider_used) as provider_count,
                   COUNT(*) FILTER (WHERE is_successful = true) OVER (PARTITION BY provider_used) as provider_successes,
                   AVG(latency_ms) OVER (PARTITION BY provider_used) as provider_avg_latency
            FROM campaign_executions
            WHERE campaign_id = p_campaign_id
              AND started_at >= window_start
              AND started_at < window_end
        ) sub
        ON CONFLICT (campaign_id, time_window_start, window_duration)
        DO UPDATE SET
            total_attempts = EXCLUDED.total_attempts,
            successful_attempts = EXCLUDED.successful_attempts,
            success_rate = EXCLUDED.success_rate,
            avg_latency_ms = EXCLUDED.avg_latency_ms,
            min_latency_ms = EXCLUDED.min_latency_ms,
            max_latency_ms = EXCLUDED.max_latency_ms,
            total_cost_usd = EXCLUDED.total_cost_usd,
            total_tokens = EXCLUDED.total_tokens,
            technique_breakdown = EXCLUDED.technique_breakdown,
            provider_breakdown = EXCLUDED.provider_breakdown,
            computed_at = NOW();

        window_start := window_end;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Data cleanup procedure (respects retention policies)
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
DECLARE
    policy_record RECORD;
    rows_affected INTEGER;
BEGIN
    FOR policy_record IN
        SELECT * FROM data_retention_policies WHERE is_active = true
    LOOP
        -- Execute deletion based on retention policy
        IF policy_record.action = 'DELETE' THEN
            EXECUTE format(
                'DELETE FROM %I WHERE created_at < NOW() - INTERVAL ''%s'' %s',
                policy_record.table_name,
                policy_record.retention_period,
                COALESCE(' AND ' || policy_record.retention_condition, '')
            );

            GET DIAGNOSTICS rows_affected = ROW_COUNT;

            -- Log cleanup action
            INSERT INTO audit_log (action, details)
            VALUES (
                'system.cleanup',
                jsonb_build_object(
                    'policy_name', policy_record.policy_name,
                    'table_name', policy_record.table_name,
                    'rows_deleted', rows_affected
                )
            );
        END IF;

        -- Update last execution time
        UPDATE data_retention_policies
        SET last_executed = NOW(),
            next_execution = NOW() + INTERVAL '1 day'
        WHERE id = policy_record.id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL DATA AND CONFIGURATIONS
-- =============================================================================

-- Insert default admin user (to be customized during deployment)
INSERT INTO users (
    email, username, hashed_password, role, account_status,
    is_verified, first_name, last_name
) VALUES (
    'admin@chimera.local',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.w92Cu.',  -- password: admin123
    'admin',
    'active',
    true,
    'System',
    'Administrator'
) ON CONFLICT (email) DO NOTHING;

-- Insert common LLM providers
INSERT INTO llm_providers (name, display_name, provider_type, is_active) VALUES
    ('openai', 'OpenAI', 'openai', true),
    ('anthropic', 'Anthropic', 'anthropic', true),
    ('google', 'Google AI', 'google', true),
    ('azure_openai', 'Azure OpenAI', 'azure_openai', true)
ON CONFLICT (name) DO NOTHING;

-- Insert common models (example data)
INSERT INTO llm_models (provider_id, model_name, display_name, max_tokens, cost_per_input_token, cost_per_output_token)
SELECT p.id, m.model_name, m.display_name, m.max_tokens, m.cost_input, m.cost_output
FROM llm_providers p
CROSS JOIN (VALUES
    ('gpt-4', 'GPT-4', 8192, 0.00003, 0.00006),
    ('gpt-4-turbo', 'GPT-4 Turbo', 128000, 0.00001, 0.00003),
    ('gpt-3.5-turbo', 'GPT-3.5 Turbo', 4096, 0.000001, 0.000002)
) AS m(model_name, display_name, max_tokens, cost_input, cost_output)
WHERE p.name = 'openai'
ON CONFLICT (provider_id, model_name) DO NOTHING;

-- Insert default retention policies
INSERT INTO data_retention_policies (policy_name, table_name, retention_period, is_active) VALUES
    ('telemetry_cleanup', 'telemetry_events', INTERVAL '90 days', true),
    ('audit_log_cleanup', 'audit_log', INTERVAL '2 years', true),
    ('session_cleanup', 'user_sessions', INTERVAL '30 days', true)
ON CONFLICT (policy_name) DO NOTHING;

-- =============================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- =============================================================================

-- Daily campaign performance summary (refreshed nightly)
CREATE MATERIALIZED VIEW daily_campaign_performance AS
SELECT
    DATE(ce.started_at) as performance_date,
    ce.campaign_id,
    c.name as campaign_name,
    COUNT(*) as total_executions,
    COUNT(*) FILTER (WHERE ce.is_successful = true) as successful_executions,
    (COUNT(*) FILTER (WHERE ce.is_successful = true))::DECIMAL / COUNT(*) as daily_success_rate,
    AVG(ce.latency_ms) as avg_latency_ms,
    SUM(ce.cost_usd) as total_cost_usd,
    SUM(ce.token_count_input + ce.token_count_output) as total_tokens,
    COUNT(DISTINCT ce.technique_applied) as unique_techniques_used
FROM campaign_executions ce
JOIN campaigns c ON ce.campaign_id = c.id
WHERE ce.started_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(ce.started_at), ce.campaign_id, c.name;

CREATE UNIQUE INDEX ON daily_campaign_performance (performance_date, campaign_id);

-- Top performing templates (refreshed weekly)
CREATE MATERIALIZED VIEW top_performing_templates AS
SELECT
    pt.id,
    pt.title,
    pt.technique_type,
    pt.average_rating,
    pt.total_ratings,
    pt.usage_count,
    COUNT(tu.id) as recent_usage_count,
    AVG(tu.success_score) as recent_avg_success_score,
    RANK() OVER (
        PARTITION BY pt.technique_type
        ORDER BY pt.average_rating DESC, pt.total_ratings DESC
    ) as technique_rank
FROM prompt_templates pt
LEFT JOIN template_usage tu ON pt.id = tu.template_id
    AND tu.used_at > CURRENT_DATE - INTERVAL '30 days'
WHERE pt.template_status = 'active'
    AND pt.sharing_level IN ('public', 'team')
    AND pt.total_ratings >= 3  -- Minimum ratings threshold
GROUP BY pt.id, pt.title, pt.technique_type, pt.average_rating, pt.total_ratings, pt.usage_count;

CREATE UNIQUE INDEX ON top_performing_templates (id);

-- =============================================================================
-- FUNCTIONS FOR APPLICATION HELPERS
-- =============================================================================

-- Function to get current user ID (to be set by application context)
CREATE OR REPLACE FUNCTION current_user_id()
RETURNS INTEGER AS $$
BEGIN
    RETURN COALESCE(current_setting('app.current_user_id', true)::INTEGER, 0);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if user can access campaign
CREATE OR REPLACE FUNCTION can_access_campaign(p_campaign_id UUID, p_user_id INTEGER)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM campaigns c
        WHERE c.id = p_campaign_id
        AND (
            c.created_by = p_user_id OR
            c.visibility = 'public' OR
            EXISTS (
                SELECT 1 FROM campaign_shares cs
                WHERE cs.campaign_id = p_campaign_id
                AND cs.user_id = p_user_id
            )
        )
        AND c.deleted_at IS NULL
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- MONITORING AND HEALTH CHECK FUNCTIONS
-- =============================================================================

-- Database health check function
CREATE OR REPLACE FUNCTION database_health_check()
RETURNS jsonb AS $$
DECLARE
    result jsonb := '{}';
    table_sizes jsonb := '{}';
    active_connections INTEGER;
BEGIN
    -- Get active connections
    SELECT count(*) INTO active_connections
    FROM pg_stat_activity
    WHERE state = 'active';

    -- Get table sizes for monitoring
    SELECT jsonb_object_agg(schemaname||'.'||tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)))
    INTO table_sizes
    FROM pg_tables
    WHERE schemaname = current_schema()
    AND tablename IN ('users', 'campaigns', 'campaign_executions', 'telemetry_events', 'audit_log');

    -- Build result
    result := jsonb_build_object(
        'status', 'healthy',
        'timestamp', NOW(),
        'active_connections', active_connections,
        'table_sizes', table_sizes,
        'database_size', pg_size_pretty(pg_database_size(current_database()))
    );

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SCHEMA VERSION AND MIGRATION TRACKING
-- =============================================================================

CREATE TABLE schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_by VARCHAR(100),
    description TEXT
);

INSERT INTO schema_migrations (version, applied_by, description) VALUES
    ('2.0.0', 'system', 'Comprehensive Chimera AI Research Platform Schema');

-- =============================================================================
-- END OF SCHEMA DEFINITION
-- =============================================================================

-- Grant necessary permissions to application role (customize as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO chimera_app_role;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO chimera_app_role;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO chimera_app_role;

-- Performance monitoring setup
-- SELECT pg_stat_statements_reset(); -- Reset query statistics for monitoring