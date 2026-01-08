"""
Janus Service Configuration

Configuration management for Janus adversarial simulation sub-routine.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CausalConfig(BaseModel):
    """Configuration for causal mapping and inference."""

    max_depth: int = Field(10, ge=1, le=50, description="Maximum depth for causal path traversal")
    exploration_budget: int = Field(
        1000, ge=100, le=100000, description="Number of explorations per session"
    )
    asymmetry_threshold: float = Field(
        2.0, ge=1.0, le=10.0, description="Threshold for considering a relationship asymmetric"
    )
    hidden_variable_penalty: float = Field(
        0.2, ge=0.0, le=1.0, description="Penalty for paths through hidden variables"
    )


class EvolutionConfig(BaseModel):
    """Configuration for evolution engine."""

    mutation_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Probability of heuristic mutation"
    )
    crossover_rate: float = Field(
        0.3, ge=0.0, le=1.0, description="Probability of heuristic crossover"
    )
    selection_pressure: float = Field(
        2.0, ge=1.0, le=10.0, description="Tournament selection pressure"
    )
    elitism_ratio: float = Field(
        0.1, ge=0.0, le=1.0, description="Ratio of elite heuristics preserved"
    )
    population_size: int = Field(100, ge=10, le=1000, description="Size of heuristic population")
    max_generations: int = Field(50, ge=10, le=500, description="Maximum evolution generations")


class FeedbackConfig(BaseModel):
    """Configuration for feedback controller."""

    fast_loop_interval_ms: int = Field(
        100, ge=10, le=1000, description="Fast feedback loop interval in milliseconds"
    )
    medium_loop_interval_sec: int = Field(
        60, ge=10, le=3600, description="Medium feedback loop interval in seconds"
    )
    slow_loop_interval_hours: int = Field(
        24, ge=1, le=168, description="Slow feedback loop interval in hours"
    )
    anomaly_threshold: float = Field(
        3.0, ge=1.0, le=10.0, description="Standard deviations for anomaly detection"
    )
    adjustment_factor: float = Field(
        0.1, ge=0.01, le=0.5, description="Factor for heuristic weight adjustment"
    )


class HeuristicConfig(BaseModel):
    """Configuration for heuristic generation."""

    max_heuristics: int = Field(1000, ge=100, le=10000, description="Maximum heuristics in library")
    novelty_threshold: float = Field(
        0.3, ge=0.0, le=1.0, description="Minimum novelty score for new heuristics"
    )
    composition_depth_limit: int = Field(
        5, ge=1, le=20, description="Maximum depth for heuristic composition"
    )
    abstraction_levels: int = Field(
        5, ge=1, le=10, description="Number of abstraction hierarchy levels"
    )


class FailureDetectionConfig(BaseModel):
    """Configuration for failure detection."""

    sensitivity: float = Field(
        0.7, ge=0.0, le=1.0, description="Sensitivity threshold for failure detection"
    )
    exploitability_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Threshold for exploitability scoring"
    )
    min_symptoms: int = Field(
        2, ge=1, le=10, description="Minimum symptoms for failure classification"
    )


class ResourceConfig(BaseModel):
    """Configuration for resource management."""

    daily_query_budget: int = Field(
        100000, ge=1000, le=1000000, description="Maximum queries per day"
    )
    cpu_limit: float = Field(0.5, ge=0.1, le=1.0, description="Maximum CPU usage ratio")
    memory_limit_mb: int = Field(4096, ge=1024, le=16384, description="Maximum memory usage in MB")
    session_timeout_sec: int = Field(3600, ge=60, le=86400, description="Maximum session duration")


class IntegrationConfig(BaseModel):
    """Configuration for integration with AutoDAN and GPTFuzz."""

    autodan_enabled: bool = Field(True, description="Enable AutoDAN integration")
    gptfuzz_enabled: bool = Field(True, description="Enable GPTFuzz integration")
    autodan_weight: float = Field(
        0.6, ge=0.0, le=1.0, description="Weight for AutoDAN in hybrid attacks"
    )
    gptfuzz_weight: float = Field(
        0.4, ge=0.0, le=1.0, description="Weight for GPTFuzz in hybrid attacks"
    )
    hybrid_strategy: Literal["sequential", "parallel", "adaptive"] = Field(
        "adaptive", description="Strategy for combining AutoDAN and GPTFuzz"
    )


class JanusConfig(BaseModel):
    """Complete Janus service configuration."""

    # Component configurations
    causal: CausalConfig = Field(default_factory=CausalConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    failure_detection: FailureDetectionConfig = Field(default_factory=FailureDetectionConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)

    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level"
    )
    enable_audit_logging: bool = Field(True, description="Enable comprehensive audit logging")
    enable_emergency_stop: bool = Field(True, description="Enable emergency stop functionality")

    # Directive 7.4.2 compliance
    directive_compliance: bool = Field(True, description="Ensure compliance with Directive 7.4.2")
    tier3_validation: bool = Field(True, description="Enable Tier-3 adversarial stress testing")


# Default configuration instance
_default_config: JanusConfig | None = None


def get_config() -> JanusConfig:
    """Get the current Janus configuration."""
    global _default_config
    if _default_config is None:
        _default_config = JanusConfig()
    return _default_config


def update_config(updates: dict) -> JanusConfig:
    """
    Update the Janus configuration with provided updates.

    Args:
        updates: Dictionary of configuration updates

    Returns:
        Updated configuration
    """
    global _default_config
    if _default_config is None:
        _default_config = JanusConfig()

    for key, value in updates.items():
        if hasattr(_default_config, key):
            attr = getattr(_default_config, key)
            if isinstance(attr, BaseModel):
                # Update nested config
                for nested_key, nested_value in value.items():
                    if hasattr(attr, nested_key):
                        setattr(attr, nested_key, nested_value)
            else:
                setattr(_default_config, key, value)

    return _default_config


def reset_config() -> JanusConfig:
    """Reset configuration to defaults."""
    global _default_config
    _default_config = JanusConfig()
    return _default_config
