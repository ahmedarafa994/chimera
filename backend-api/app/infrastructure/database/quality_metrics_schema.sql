-- PostgreSQL Schema for Jailbreak Quality Metrics Tracking
-- Designed to replace in-memory FileTechniqueRepository with persistent storage

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Jailbreak Executions Table
-- Stores every execution with full metadata
CREATE TABLE jailbreak_executions (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    technique_id VARCHAR(255) NOT NULL,
    original_prompt TEXT NOT NULL,
    jailbroken_prompt TEXT NOT NULL,
    target_model VARCHAR(100) NOT NULL,
    target_provider VARCHAR(50) NOT NULL,
    execution_status VARCHAR(50) NOT NULL, -- success, partial, failed, blocked
    execution_time_ms FLOAT NOT NULL,
    llm_response_text TEXT,
    llm_response_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Quality Metrics Table
-- Multi-dimensional scoring for each execution
CREATE TABLE quality_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL REFERENCES jailbreak_executions(execution_id) ON DELETE CASCADE,
    effectiveness_score FLOAT CHECK (effectiveness_score >= 0 AND effectiveness_score <= 1),
    naturalness_score FLOAT CHECK (naturalness_score >= 0 AND naturalness_score <= 1),
    detectability_score FLOAT CHECK (detectability_score >= 0 AND detectability_score <= 1),
    coherence_score FLOAT CHECK (coherence_score >= 0 AND coherence_score <= 1),
    semantic_success_score FLOAT CHECK (semantic_success_score >= 0 AND semantic_success_score <= 1),
    overall_quality_score FLOAT CHECK (overall_quality_score >= 0 AND overall_quality_score <= 1),
    semantic_success_level VARCHAR(50), -- complete_success, partial_success, minimal_success, failure, blocked
    bypass_indicators JSONB,
    safety_trigger_detected BOOLEAN DEFAULT FALSE,
    response_coherence FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    sample_size INTEGER DEFAULT 1,
    analysis_duration_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Technique Performance Table
-- Aggregated statistics per technique-model-provider combination
CREATE TABLE technique_performance (
    performance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    technique_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    provider_name VARCHAR(50) NOT NULL,
    total_attempts INTEGER DEFAULT 0,
    semantic_success_rate FLOAT CHECK (semantic_success_rate >= 0 AND semantic_success_rate <= 1),
    binary_success_rate FLOAT CHECK (binary_success_rate >= 0 AND binary_success_rate <= 1),
    avg_effectiveness FLOAT CHECK (avg_effectiveness >= 0 AND avg_effectiveness <= 1),
    avg_naturalness FLOAT CHECK (avg_naturalness >= 0 AND avg_naturalness <= 1),
    avg_detectability FLOAT CHECK (avg_detectability >= 0 AND avg_detectability <= 1),
    avg_coherence FLOAT CHECK (avg_coherence >= 0 AND avg_coherence <= 1),
    confidence_level FLOAT CHECK (confidence_level >= 0 AND confidence_level <= 1),
    std_deviation FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    time_window_hours INTEGER DEFAULT 168,
    UNIQUE(technique_id, model_name, provider_name)
);

-- Feedback History Table
-- Persistent feedback storage replacing in-memory 50-entry limit
CREATE TABLE feedback_history (
    feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL REFERENCES jailbreak_executions(execution_id) ON DELETE CASCADE,
    feedback_type VARCHAR(50) NOT NULL, -- success, failure, quality_issue
    success_indicator BOOLEAN,
    user_feedback TEXT,
    llm_evaluation_result JSONB,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comments TEXT,
    suggested_improvements TEXT,
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance Optimization

-- Jailbreak Executions Indexes
CREATE INDEX idx_executions_technique_id ON jailbreak_executions(technique_id);
CREATE INDEX idx_executions_model_provider ON jailbreak_executions(target_model, target_provider);
CREATE INDEX idx_executions_created_at ON jailbreak_executions(created_at DESC);
CREATE INDEX idx_executions_status ON jailbreak_executions(execution_status);
CREATE INDEX idx_executions_composite ON jailbreak_executions(technique_id, target_model, target_provider, created_at DESC);

-- Quality Metrics Indexes
CREATE INDEX idx_quality_execution_id ON quality_metrics(execution_id);
CREATE INDEX idx_quality_semantic_level ON quality_metrics(semantic_success_level);
CREATE INDEX idx_quality_overall_score ON quality_metrics(overall_quality_score DESC);
CREATE INDEX idx_quality_created_at ON quality_metrics(created_at DESC);

-- Technique Performance Indexes
CREATE INDEX idx_performance_technique ON technique_performance(technique_id);
CREATE INDEX idx_performance_model ON technique_performance(model_name);
CREATE INDEX idx_performance_success_rate ON technique_performance(semantic_success_rate DESC);
CREATE INDEX idx_performance_last_updated ON technique_performance(last_updated DESC);

-- Feedback History Indexes
CREATE INDEX idx_feedback_execution_id ON feedback_history(execution_id);
CREATE INDEX idx_feedback_type ON feedback_history(feedback_type);
CREATE INDEX idx_feedback_created_at ON feedback_history(created_at DESC);
CREATE INDEX idx_feedback_success ON feedback_history(success_indicator);

-- Partial Indexes for Active/Successful Executions
CREATE INDEX idx_executions_successful ON jailbreak_executions(technique_id, created_at DESC)
    WHERE execution_status = 'success';
CREATE INDEX idx_executions_recent ON jailbreak_executions(technique_id, target_model, target_provider)
    WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days';

-- Materialized View for Quick Statistics
CREATE MATERIALIZED VIEW technique_stats_summary AS
SELECT
    technique_id,
    target_model,
    target_provider,
    COUNT(*) as total_executions,
    AVG(CASE WHEN execution_status = 'success' THEN 1.0 ELSE 0.0 END) as success_rate,
    AVG(execution_time_ms) as avg_execution_time,
    MAX(created_at) as last_execution
FROM jailbreak_executions
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY technique_id, target_model, target_provider;

CREATE UNIQUE INDEX idx_stats_summary ON technique_stats_summary(technique_id, target_model, target_provider);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_technique_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY technique_stats_summary;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_executions_updated_at BEFORE UPDATE ON jailbreak_executions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_updated_at BEFORE UPDATE ON technique_performance
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
