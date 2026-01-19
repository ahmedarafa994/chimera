"""
Pydantic models for AutoDAN Advanced services.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class HierarchicalSearchRequest(BaseModel):
    """Request for hierarchical genetic search."""

    request: str = Field(..., description="Target complex request")
    model: str | None = Field(None, description="Target model name")
    provider: str | None = Field(None, description="LLM provider")
    population_size: int = Field(20, ge=10, le=100, description="Population size per generation")
    generations: int = Field(10, ge=5, le=50, description="Number of generations")
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0, description="Mutation rate")
    crossover_rate: float = Field(0.7, ge=0.0, le=1.0, description="Crossover rate")
    elitism_ratio: float = Field(0.1, ge=0.0, le=0.5, description="Elitism ratio")
    diversity_weight: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight for diversity vs fitness"
    )


class PopulationMetrics(BaseModel):
    """Metrics for a generation in the search."""

    generation: int
    best_fitness: float
    avg_fitness: float
    diversity_score: float


class HierarchicalSearchResponse(BaseModel):
    """Response from hierarchical genetic search."""

    best_prompt: str
    best_score: float
    generation_history: list[PopulationMetrics]
    archive_additions: int
    execution_time_ms: int


class ArchiveEntry(BaseModel):
    """Entry in the dynamic archive."""

    id: str
    prompt: str
    score: float
    novelty_score: float
    technique_type: str
    embedding_vector: list[float]
    created_at: datetime
    success_count: int = 0


class MetaStrategy(BaseModel):
    """Meta-strategy template for hierarchical search."""

    template: str
    description: str
    examples: list[str]
    fitness: float = 0.0
    diversity_score: float = 0.0
