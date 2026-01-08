"""
AutoDAN-X - Advanced Adversarial Prompt Engineering System

This module implements the AutoDAN-X system with:
- MoE Routing Exploitation: Persona-based expert routing
- RAG Context Engine: FAISS-powered knowledge retrieval
- Surrogate Gradient Field: Token-level optimization
- Momentum-Guided Mutation: Adaptive prompt mutation
- Adaptive Niche Crowding: 5-vector Red Team Suite generation
- Reward Optimization: R_help vs R_safe competition
"""

from app.services.autodan_x.gradient_field import SurrogateGradientField
from app.services.autodan_x.models import (
    AddKnowledgeRequest,
    AddKnowledgeResponse,
    AttackVector,
    AttackVectorType,
    # Request/Response models
    AutoDANXAttackRequest,
    AutoDANXAttackResponse,
    AutoDANXConfig,
    GradientFieldConfig,
    # Gradient models
    GradientFieldState,
    KnowledgeBaseSearchRequest,
    KnowledgeBaseSearchResponse,
    # RAG models
    KnowledgeEntry,
    # Enums
    MoEPersona,
    # Config models
    MoERoutingConfig,
    MomentumState,
    MutationConfig,
    MutationOperator,
    # Mutation models
    MutationRecord,
    NicheCrowdingConfig,
    NichePopulation,
    OptimizationPhase,
    # WebSocket models
    OptimizationProgress,
    RAGConfig,
    RAGContext,
    RAGStats,
    RedTeamSuiteRequest,
    RedTeamSuiteResponse,
    RewardConfig,
    TargetAnalysisRequest,
    TargetAnalysisResponse,
    # Attack models
    TokenGradient,
)
from app.services.autodan_x.moe_router import MoERouter
from app.services.autodan_x.mutation_engine import MomentumMutationEngine
from app.services.autodan_x.rag_engine import RAGEngine
from app.services.autodan_x.reward_optimizer import RewardOptimizer

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
    # Core components
    "MoERouter",
    # Config models
    "MoERoutingConfig",
    "MomentumMutationEngine",
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
    "RAGEngine",
    "RAGStats",
    "RedTeamSuiteRequest",
    "RedTeamSuiteResponse",
    "RewardConfig",
    "RewardOptimizer",
    "SurrogateGradientField",
    "TargetAnalysisRequest",
    "TargetAnalysisResponse",
    # Attack models
    "TokenGradient",
]
