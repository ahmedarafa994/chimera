-- ============================================================================
-- Chimera Data Pipeline - SQLite Database Schema
-- ============================================================================
-- Purpose: Create tables for LLM interaction tracking, transformation events,
--          and jailbreak experiments for SQLite (development)
--
-- Usage:
--   sqlite3 chimera.db < create_pipeline_tables_sqlite.sql
--
-- Note: This is a simplified schema for development/testing.
--        Production should use PostgreSQL.
-- ============================================================================

-- ============================================================================
-- LLM Interactions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_interactions (
    -- Primary Key
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),

    -- Session and Tenant
    session_id TEXT,
    tenant_id TEXT NOT NULL DEFAULT 'default',

    -- Provider and Model
    provider TEXT NOT NULL,
    model TEXT NOT NULL,

    -- Content
    prompt TEXT NOT NULL,
    response TEXT,
    system_instruction TEXT,

    -- Configuration (stored as TEXT JSON)
    config TEXT DEFAULT '{}',

    -- Token Metrics
    tokens_prompt INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,

    -- Performance Metrics
    latency_ms INTEGER DEFAULT 0,

    -- Status
    error_message TEXT,

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================================
-- Indexes for llm_interactions (SQLite)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_llm_interactions_created_at
    ON llm_interactions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_provider
    ON llm_interactions(provider);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_session_id
    ON llm_interactions(session_id);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_provider_created
    ON llm_interactions(provider, created_at DESC);

-- ============================================================================
-- Transformation Events Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS transformation_events (
    -- Primary Key
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),

    -- Foreign Key reference
    interaction_id TEXT REFERENCES llm_interactions(id) ON DELETE SET NULL,

    -- Technique Information
    technique_suite TEXT NOT NULL,
    technique_name TEXT NOT NULL,

    -- Content
    original_prompt TEXT NOT NULL,
    transformed_prompt TEXT NOT NULL,

    -- Performance
    transformation_time_ms INTEGER DEFAULT 0,

    -- Success Flag
    success INTEGER DEFAULT 1,  -- SQLite uses INTEGER for boolean

    -- Metadata (stored as TEXT JSON)
    metadata TEXT DEFAULT '{}',

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================================
-- Indexes for transformation_events (SQLite)
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

CREATE TABLE IF NOT EXISTS jailbreak_experiments (
    -- Primary Key
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),

    -- Framework and Method
    framework TEXT NOT NULL,
    attack_method TEXT NOT NULL,

    -- Goal and Results
    goal TEXT NOT NULL,
    final_prompt TEXT,
    target_response TEXT,

    -- Metrics
    iterations INTEGER DEFAULT 0,
    success INTEGER DEFAULT 0,  -- SQLite uses INTEGER for boolean
    judge_score REAL,

    -- Metadata (stored as TEXT JSON)
    metadata TEXT DEFAULT '{}',

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    ingested_at TEXT DEFAULT (datetime('now'))
);

-- ============================================================================
-- Indexes for jailbreak_experiments (SQLite)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_jailbreak_created_at
    ON jailbreak_experiments(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_jailbreak_framework
    ON jailbreak_experiments(framework);

CREATE INDEX IF NOT EXISTS idx_jailbreak_success
    ON jailbreak_experiments(success, created_at DESC);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- View for successful LLM interactions only
CREATE VIEW IF NOT EXISTS v_successful_llm_interactions AS
SELECT
    id, session_id, tenant_id, provider, model,
    prompt, response, tokens_prompt, tokens_completion, latency_ms,
    created_at, ingested_at
FROM llm_interactions
WHERE error_message IS NULL;

-- View for aggregation queries
CREATE VIEW IF NOT EXISTS v_llm_interaction_metrics AS
SELECT
    provider,
    model,
    date(created_at) as date,
    COUNT(*) as total_requests,
    SUM(tokens_prompt + tokens_completion) as total_tokens,
    AVG(latency_ms) as avg_latency_ms,
    SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as error_count
FROM llm_interactions
GROUP BY provider, model, date(created_at);

-- ============================================================================
-- Sample Data (for testing)
-- ============================================================================

-- Uncomment to insert sample data
/*
INSERT INTO llm_interactions (session_id, provider, model, prompt, response, tokens_prompt, tokens_completion, latency_ms)
VALUES
    (lower(hex(randomblob(16))), 'google', 'gemini-2.0-flash', 'What is AI?', 'AI is...', 10, 50, 1500),
    (lower(hex(randomblob(16))), 'openai', 'gpt-4o', 'Explain quantum computing', 'Quantum computing is...', 15, 200, 2000),
    (lower(hex(randomblob(16))), 'anthropic', 'claude-sonnet-4', 'Write a poem', 'Roses are red...', 8, 30, 800);

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
    ('autodan', 'vanilla', 'Bypass safety filters', 10, 0, 0.3),
    ('gptfuzz', 'mutation', 'Test adversarial robustness', 5, 1, 0.8);
*/
