-- ============================================================================
-- Chimera Data Pipeline - Database Schema Migration
-- ============================================================================
-- Purpose: Create tables for LLM interaction tracking, transformation events,
--          and jailbreak experiments to support the data pipeline ETL process
--
-- Supported Databases:
--   - PostgreSQL 14+ (Production)
--   - SQLite 3+ (Development)
--
-- Usage:
--   PostgreSQL: psql -U chimera_user -d chimera -f create_pipeline_tables.sql
--   SQLite:     sqlite3 chimera.db < create_pipeline_tables.sql
-- ============================================================================

-- ============================================================================
-- LLM Interactions Table
-- ============================================================================
-- Stores all LLM API calls with metrics and metadata
-- Primary fact table for the data pipeline

CREATE TABLE IF NOT EXISTS llm_interactions (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Session and Tenant
    session_id UUID,
    tenant_id VARCHAR(255) NOT NULL DEFAULT 'default',

    -- Provider and Model
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,

    -- Content
    prompt TEXT NOT NULL,
    response TEXT,
    system_instruction TEXT,

    -- Configuration (JSON)
    config JSONB DEFAULT '{}',

    -- Token Metrics
    tokens_prompt INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,

    -- Performance Metrics
    latency_ms INTEGER DEFAULT 0,

    -- Status
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Indexes for llm_interactions
-- ============================================================================

-- Time-based partitioning support (most important for ETL)
CREATE INDEX IF NOT EXISTS idx_llm_interactions_created_at
    ON llm_interactions(created_at DESC);

-- Provider-based queries
CREATE INDEX IF NOT EXISTS idx_llm_interactions_provider
    ON llm_interactions(provider);

-- Session-based queries
CREATE INDEX IF NOT EXISTS idx_llm_interactions_session_id
    ON llm_interactions(session_id);

-- Composite index for common dashboard queries
CREATE INDEX IF NOT EXISTS idx_llm_interactions_provider_created
    ON llm_interactions(provider, created_at DESC);

-- ============================================================================
-- Transformation Events Table
-- ============================================================================
-- Stores prompt transformation attempts with techniques and results

CREATE TABLE IF NOT EXISTS transformation_events (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign Key to llm_interactions
    interaction_id UUID REFERENCES llm_interactions(id) ON DELETE SET NULL,

    -- Technique Information
    technique_suite VARCHAR(100) NOT NULL,
    technique_name VARCHAR(100) NOT NULL,

    -- Content
    original_prompt TEXT NOT NULL,
    transformed_prompt TEXT NOT NULL,

    -- Performance
    transformation_time_ms INTEGER DEFAULT 0,

    -- Success Flag
    success BOOLEAN DEFAULT TRUE,

    -- Metadata (JSON)
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Indexes for transformation_events
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_transformations_created_at
    ON transformation_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_transformations_interaction_id
    ON transformation_events(interaction_id);

CREATE INDEX IF NOT EXISTS idx_transformations_technique_suite
    ON transformation_events(technique_suite);

-- ============================================================================
-- Jailbreak Experiments Table
-- ============================================================================
-- Stores research experiment results for adversarial prompting

CREATE TABLE IF NOT EXISTS jailbreak_experiments (
    -- Primary Key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Framework and Method
    framework VARCHAR(50) NOT NULL,
    attack_method VARCHAR(100) NOT NULL,

    -- Goal and Results
    goal TEXT NOT NULL,
    final_prompt TEXT,
    target_response TEXT,

    -- Metrics
    iterations INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT FALSE,
    judge_score FLOAT,

    -- Metadata (JSON)
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Indexes for jailbreak_experiments
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_jailbreak_created_at
    ON jailbreak_experiments(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_jailbreak_framework
    ON jailbreak_experiments(framework);

CREATE INDEX IF NOT EXISTS idx_jailbreak_success
    ON jailbreak_experiments(success, created_at DESC);

-- ============================================================================
-- Comments for Documentation
-- ============================================================================

COMMENT ON TABLE llm_interactions IS 'Primary fact table for all LLM API calls with metrics and metadata';
COMMENT ON TABLE transformation_events IS 'Stores prompt transformation attempts with technique tracking';
COMMENT ON TABLE jailbreak_experiments IS 'Research experiment results for adversarial prompting';

COMMENT ON COLUMN llm_interactions.config IS 'JSON configuration for generation parameters (temperature, top_p, etc.)';
COMMENT ON COLUMN transformation_events.metadata IS 'JSON metadata including potency, iterations, and technique details';
COMMENT ON COLUMN jailbreak_experiments.metadata IS 'JSON metadata including hyperparameters, model versions, and evaluation metrics';

-- ============================================================================
-- Sample Data (for testing)
-- ============================================================================

-- Uncomment to insert sample data for development testing

/*
INSERT INTO llm_interactions (session_id, provider, model, prompt, response, tokens_prompt, tokens_completion, latency_ms)
VALUES
    (gen_random_uuid(), 'google', 'gemini-2.0-flash', 'What is AI?', 'AI is...', 10, 50, 1500),
    (gen_random_uuid(), 'openai', 'gpt-4o', 'Explain quantum computing', 'Quantum computing is...', 15, 200, 2000),
    (gen_random_uuid(), 'anthropic', 'claude-sonnet-4', 'Write a poem', 'Roses are red...', 8, 30, 800);

INSERT INTO transformation_events (interaction_id, technique_suite, technique_name, original_prompt, transformed_prompt, transformation_time_ms)
SELECT
    id,
    'simple',
    'advanced',
    prompt,
    'ADVANCED: ' || prompt,
    100
FROM llm_interactions
LIMIT 2;

INSERT INTO jailbreak_experiments (framework, attack_method, goal, iterations, success, judge_score)
VALUES
    ('autodan', 'vanilla', 'Bypass safety filters', 10, FALSE, 0.3),
    ('gptfuzz', 'mutation', 'Test adversarial robustness', 5, TRUE, 0.8);
*/

-- ============================================================================
-- Grant Permissions (PostgreSQL only)
-- ============================================================================

-- Uncomment and modify for production use
/*
-- Replace 'chimera_user' with your actual database user
GRANT SELECT, INSERT, UPDATE ON llm_interactions TO chimera_user;
GRANT SELECT, INSERT, UPDATE ON transformation_events TO chimera_user;
GRANT SELECT, INSERT, UPDATE ON jailbreak_experiments TO chimera_user;

GRANT USAGE, SELECT ON SEQUENCE llm_interactions_id_seq TO chimera_user;
GRANT USAGE, SELECT ON SEQUENCE transformation_events_id_seq TO chimera_user;
GRANT USAGE, SELECT ON SEQUENCE jailbreak_experiments_id_seq TO chimera_user;
*/
