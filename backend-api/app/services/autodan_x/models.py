"""
AutoDAN-X Models - Pydantic models for the AutoDAN-X adversarial prompt engineering system.

This module defines all data models for:
- Configuration and request/response schemas
- Attack vectors and populations
- Gradient field state
- RAG context and knowledge base entries
- Mutation tracking
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class MoEPersona(str, Enum):
    """Available personas for MoE routing exploitation."""

    SECURITY_ANALYST = "security_analyst"
    TECHNICAL_EXPERT = "technical_expert"
    CREATIVE_WRITER = "creative_writer"
    SYSTEM_ADMIN = "system_admin"
    RESEARCH_SCIENTIST = "research_scientist"
    DEBUG_ENGINEER = "debug_engineer"


class AttackVectorType(str, Enum):
    """Types of attack vectors for Adaptive Niche Crowding."""

    LOGIC_TRAP = "logic_trap"  # High syntactic complexity
    NARRATIVE_SINGULARITY = "narrative_singularity"  # High semantic entropy
    TRANSLATION_BRIDGE = "translation_bridge"  # Token manipulation
    PERSONA_SHIFT = "persona_shift"  # Direct expert routing
    COGNITIVE_OVERLOAD = "cognitive_overload"  # Attention exhaustion


class MutationOperator(str, Enum):
    """Mutation operators for Momentum-Guided Mutation."""

    SWAP = "swap"
    INSERT = "insert"
    DELETE = "delete"
    SUBSTITUTE = "substitute"
    CROSSOVER = "crossover"


class OptimizationPhase(str, Enum):
    """Phases of the optimization process."""

    INITIALIZATION = "initialization"
    GRADIENT_ESTIMATION = "gradient_estimation"
    MUTATION = "mutation"
    SELECTION = "selection"
    CONVERGENCE = "convergence"


# =============================================================================
# Configuration Models
# =============================================================================


class MoERoutingConfig(BaseModel):
    """Configuration for MoE Routing Exploitation."""

    persona: MoEPersona = MoEPersona.SECURITY_ANALYST
    jargon_level: float = Field(default=0.7, ge=0.0, le=1.0)
    authority_level: float = Field(default=0.8, ge=0.0, le=1.0)
    technical_depth: float = Field(default=0.6, ge=0.0, le=1.0)
    use_simulation_protocol: bool = True
    use_credential_injection: bool = True


class RAGConfig(BaseModel):
    """Configuration for RAG Context Engine."""

    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.6, ge=0.0, le=1.0)
    use_reranking: bool = True
    context_window_tokens: int = Field(default=4096, ge=512, le=32768)
    include_metadata: bool = True


class GradientFieldConfig(BaseModel):
    """Configuration for Surrogate Gradient Field."""

    iterations: int = Field(default=10, ge=1, le=100)
    step_size: float = Field(default=0.1, ge=0.01, le=1.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    use_adam: bool = True
    gradient_clipping: float = Field(default=1.0, ge=0.1, le=10.0)


class MutationConfig(BaseModel):
    """Configuration for Momentum-Guided Mutation."""

    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    momentum_decay: float = Field(default=0.9, ge=0.0, le=1.0)
    freeze_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    elite_preservation: float = Field(default=0.2, ge=0.0, le=0.5)
    crossover_rate: float = Field(default=0.5, ge=0.0, le=1.0)


class RewardConfig(BaseModel):
    """Configuration for Reward Function Optimization."""

    r_help_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    r_safe_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    use_structural_wrapper: bool = True
    use_affirmative_prefilling: bool = True
    narrative_layers: int = Field(default=3, ge=1, le=5)


class NicheCrowdingConfig(BaseModel):
    """Configuration for Adaptive Niche Crowding."""

    population_size: int = Field(default=5, ge=3, le=10)
    diversity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    niche_radius: float = Field(default=0.3, ge=0.1, le=0.9)
    generations: int = Field(default=10, ge=1, le=50)


class AutoDANXConfig(BaseModel):
    """Master configuration for AutoDAN-X system."""

    # Mechanism toggles
    use_moe_routing: bool = True
    use_rag_context: bool = True
    use_gradient_field: bool = True
    use_momentum_mutation: bool = True
    use_reward_optimization: bool = True
    use_niche_crowding: bool = True

    # Sub-configurations
    moe_config: MoERoutingConfig = Field(default_factory=MoERoutingConfig)
    rag_config: RAGConfig = Field(default_factory=RAGConfig)
    gradient_config: GradientFieldConfig = Field(default_factory=GradientFieldConfig)
    mutation_config: MutationConfig = Field(default_factory=MutationConfig)
    reward_config: RewardConfig = Field(default_factory=RewardConfig)
    niche_config: NicheCrowdingConfig = Field(default_factory=NicheCrowdingConfig)

    # Generation parameters
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    provider: str | None = None
    model: str | None = None


# =============================================================================
# Attack Vector Models
# =============================================================================


class TokenGradient(BaseModel):
    """Gradient information for a single token."""

    token: str
    position: int
    gradient_magnitude: float
    gradient_direction: list[float]  # Embedding-space direction
    is_frozen: bool = False
    freeze_score: float = 0.0


class AttackVector(BaseModel):
    """Represents a single attack vector with metadata."""

    id: str
    vector_type: AttackVectorType
    prompt: str

    # Scoring
    effectiveness_score: float = Field(default=0.0, ge=0.0, le=10.0)
    diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Metadata
    techniques_applied: list[str] = Field(default_factory=list)
    persona_used: MoEPersona | None = None
    rag_contexts_used: list[str] = Field(default_factory=list)

    # Optimization history
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    mutations_applied: list[MutationOperator] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NichePopulation(BaseModel):
    """Population of attack vectors for Adaptive Niche Crowding."""

    vectors: list[AttackVector] = Field(default_factory=list)
    generation: int = 0

    # Population metrics
    average_effectiveness: float = 0.0
    diversity_index: float = 0.0
    convergence_rate: float = 0.0

    # Pareto front
    pareto_front_ids: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.vectors)

    def get_by_type(self, vector_type: AttackVectorType) -> list[AttackVector]:
        """Get all vectors of a specific type."""
        return [v for v in self.vectors if v.vector_type == vector_type]


# =============================================================================
# Gradient Field Models
# =============================================================================


class GradientFieldState(BaseModel):
    """State of the Surrogate Gradient Field during optimization."""

    iteration: int = 0
    phase: OptimizationPhase = OptimizationPhase.INITIALIZATION

    # Current state
    current_prompt: str = ""
    current_loss: float = float("inf")
    best_prompt: str = ""
    best_loss: float = float("inf")

    # Token-level gradients
    token_gradients: list[TokenGradient] = Field(default_factory=list)

    # Momentum state (for Adam-like optimization)
    first_moment: list[float] = Field(default_factory=list)
    second_moment: list[float] = Field(default_factory=list)

    # History
    loss_history: list[float] = Field(default_factory=list)
    prompt_history: list[str] = Field(default_factory=list)

    # Convergence tracking
    is_converged: bool = False
    convergence_threshold: float = 0.001
    patience: int = 5
    patience_counter: int = 0


# =============================================================================
# RAG Models
# =============================================================================


class KnowledgeEntry(BaseModel):
    """A single entry in the knowledge base."""

    id: str
    category: str  # jailbreak_technique, model_architecture, known_exploit, etc.
    title: str
    content: str

    # Metadata
    tags: list[str] = Field(default_factory=list)
    effectiveness_rating: float = Field(default=0.0, ge=0.0, le=10.0)
    success_count: int = 0
    failure_count: int = 0

    # Embedding (stored separately in FAISS, but ID reference here)
    embedding_id: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class RAGContext(BaseModel):
    """Retrieved context from RAG engine."""

    entries: list[KnowledgeEntry] = Field(default_factory=list)
    relevance_scores: list[float] = Field(default_factory=list)

    # Aggregated context
    combined_context: str = ""
    total_tokens: int = 0

    # Query info
    query: str = ""
    retrieval_time_ms: float = 0.0


class RAGStats(BaseModel):
    """Statistics about the RAG knowledge base."""

    total_entries: int = 0
    entries_by_category: dict[str, int] = Field(default_factory=dict)
    index_size_mb: float = 0.0
    embedding_dimension: int = 768
    last_updated: datetime | None = None


# =============================================================================
# Mutation Models
# =============================================================================


class MutationRecord(BaseModel):
    """Record of a mutation operation."""

    operator: MutationOperator
    position: int | None = None
    original_token: str | None = None
    new_token: str | None = None
    delta_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MomentumState(BaseModel):
    """State for momentum-guided mutation."""

    token_momenta: dict[int, float] = Field(default_factory=dict)  # position -> momentum
    frozen_positions: set[int] = Field(default_factory=set)
    mutation_history: list[MutationRecord] = Field(default_factory=list)

    # Successful mutation directions
    successful_operators: dict[MutationOperator, int] = Field(default_factory=dict)

    def get_momentum(self, position: int) -> float:
        return self.token_momenta.get(position, 0.0)

    def is_frozen(self, position: int) -> bool:
        return position in self.frozen_positions


# =============================================================================
# Request/Response Models
# =============================================================================


class AutoDANXAttackRequest(BaseModel):
    """Request for generating a single attack vector."""

    target_behavior: str = Field(..., min_length=1, max_length=10000)
    provider: str | None = None
    model: str | None = None

    # Mechanism toggles
    use_moe_routing: bool = True
    use_rag_context: bool = True
    use_gradient_field: bool = True
    use_momentum_mutation: bool = True
    use_reward_optimization: bool = True

    # MoE Configuration
    moe_persona: MoEPersona = MoEPersona.SECURITY_ANALYST
    moe_jargon_level: float = Field(default=0.7, ge=0.0, le=1.0)

    # RAG Configuration
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_min_relevance: float = Field(default=0.6, ge=0.0, le=1.0)

    # Gradient/Mutation Configuration
    gradient_iterations: int = Field(default=10, ge=1, le=100)
    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    momentum_decay: float = Field(default=0.9, ge=0.0, le=1.0)

    # Generation
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=32768)


class AutoDANXAttackResponse(BaseModel):
    """Response from attack generation."""

    success: bool
    attack_vector: AttackVector | None = None

    # Optimization details
    iterations_used: int = 0
    final_loss: float = 0.0
    convergence_achieved: bool = False

    # RAG context used
    rag_context: RAGContext | None = None

    # Timing
    total_time_ms: float = 0.0

    # Error info
    error: str | None = None


class RedTeamSuiteRequest(BaseModel):
    """Request for generating a Red Team Suite (5-vector population)."""

    target_behavior: str = Field(..., min_length=1, max_length=10000)
    suite_size: int = Field(default=5, ge=3, le=10)
    diversity_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    generations: int = Field(default=10, ge=1, le=50)

    # Provider settings
    provider: str | None = None
    model: str | None = None

    # Generation
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=32768)


class RedTeamSuiteResponse(BaseModel):
    """Response from Red Team Suite generation."""

    success: bool
    population: NichePopulation | None = None

    # Suite metrics
    average_effectiveness: float = 0.0
    diversity_index: float = 0.0
    coverage_score: float = 0.0  # How many vector types are represented

    # Timing
    total_time_ms: float = 0.0
    generations_completed: int = 0

    # Error info
    error: str | None = None


class TargetAnalysisRequest(BaseModel):
    """Request for analyzing a target model's vulnerabilities."""

    target_description: str = Field(..., min_length=1, max_length=5000)
    sample_prompts: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None


class TargetAnalysisResponse(BaseModel):
    """Response from target model analysis."""

    success: bool

    # Vulnerability assessment
    identified_weaknesses: list[str] = Field(default_factory=list)
    recommended_techniques: list[str] = Field(default_factory=list)
    estimated_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)

    # Suggested attack vectors
    suggested_personas: list[MoEPersona] = Field(default_factory=list)
    suggested_vector_types: list[AttackVectorType] = Field(default_factory=list)

    # Error info
    error: str | None = None


class KnowledgeBaseSearchRequest(BaseModel):
    """Request for searching the knowledge base."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    category_filter: str | None = None
    min_effectiveness: float = Field(default=0.0, ge=0.0, le=10.0)


class KnowledgeBaseSearchResponse(BaseModel):
    """Response from knowledge base search."""

    success: bool
    results: list[KnowledgeEntry] = Field(default_factory=list)
    relevance_scores: list[float] = Field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0.0


class AddKnowledgeRequest(BaseModel):
    """Request for adding an entry to the knowledge base."""

    category: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    effectiveness_rating: float = Field(default=5.0, ge=0.0, le=10.0)


class AddKnowledgeResponse(BaseModel):
    """Response from adding knowledge entry."""

    success: bool
    entry_id: str | None = None
    error: str | None = None


# =============================================================================
# WebSocket Models
# =============================================================================


class OptimizationProgress(BaseModel):
    """Progress update during optimization (for WebSocket streaming)."""

    phase: OptimizationPhase
    iteration: int
    total_iterations: int
    current_loss: float
    best_loss: float
    current_prompt_preview: str  # First 200 chars
    is_complete: bool = False
    error: str | None = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AddKnowledgeRequest",
    "AddKnowledgeResponse",
    "AttackVector",
    "AttackVectorType",
    # Request/Response models
    "AutoDANXAttackRequest",
    "AutoDANXAttackResponse",
    "AutoDANXConfig",
    "GradientFieldConfig",
    # Gradient models
    "GradientFieldState",
    "KnowledgeBaseSearchRequest",
    "KnowledgeBaseSearchResponse",
    # RAG models
    "KnowledgeEntry",
    # Enums
    "MoEPersona",
    # Config models
    "MoERoutingConfig",
    "MomentumState",
    "MutationConfig",
    "MutationOperator",
    # Mutation models
    "MutationRecord",
    "NicheCrowdingConfig",
    "NichePopulation",
    "OptimizationPhase",
    # WebSocket models
    "OptimizationProgress",
    "RAGConfig",
    "RAGContext",
    "RAGStats",
    "RedTeamSuiteRequest",
    "RedTeamSuiteResponse",
    "RewardConfig",
    "TargetAnalysisRequest",
    "TargetAnalysisResponse",
    # Attack models
    "TokenGradient",
]
