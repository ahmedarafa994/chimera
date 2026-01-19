"""
Main V1 API Router Configuration

This module aggregates and registers all v1 service endpoints with appropriate
prefixes and tags to ensure logical separation and avoid route collisions.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    assessments,  # Security assessment management
    chat,
    cicd,  # CI/CD integration endpoints
    defense_engine,  # Defense engine endpoints
    execute,
    health,
    metrics,
    multimodal,  # Multi-modal attack testing
    research_lab,  # Research lab endpoints
    scheduled_testing,  # Scheduled testing endpoints
    technique_builder,  # Custom technique builder
    transformation,
    utils,  # Team workspaces & collaboration
)

api_router = APIRouter()

# --- Custom Technique Builder (Phase 3) ---
api_router.include_router(
    technique_builder.router,
    prefix="/techniques",
    tags=["techniques", "custom", "builder"],
)

# --- System & Core Endpoints ---
api_router.include_router(health.router, tags=["health"])
api_router.include_router(metrics.router, tags=["utils", "admin"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(transformation.router, prefix="/transform", tags=["transformation"])
api_router.include_router(execute.router, tags=["execute"])
api_router.include_router(utils.router, tags=["utils"])

# --- Defense Engine (Phase 3) ---
api_router.include_router(
    defense_engine.router,
    prefix="/defense-engine",
    tags=["defense", "protection", "security"],
)

# --- Research Lab (Phase 3) ---
api_router.include_router(
    research_lab.router,
    prefix="/research-lab",
    tags=["research", "experiments", "lab"],
)

# --- Multi-Modal Testing (Phase 4) ---
api_router.include_router(
    multimodal.router,
    prefix="/multimodal",
    tags=["multimodal", "vision", "audio", "security"],
)

# --- Scheduled Testing (Phase 3) ---
api_router.include_router(
    scheduled_testing.router,
    prefix="/scheduled-testing",
    tags=["scheduled", "testing", "automation"],
)

# --- CI/CD Integration (Phase 3) ---
api_router.include_router(
    cicd.router,
    prefix="/cicd",
    tags=["cicd", "integration", "automation"],
)

# --- Security Assessments (Phase 2) ---
api_router.include_router(
    assessments.router,
    prefix="/assessments",
    tags=["assessments", "security", "testing"],
)
