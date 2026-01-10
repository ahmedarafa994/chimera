"""
Main V1 API Router Configuration

This module aggregates and registers all v1 service endpoints with appropriate
prefixes and tags to ensure logical separation and avoid route collisions.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    admin,
    advanced_generation,
    adversarial,  # Unified adversarial endpoint (ADR-003)
    autoadv,
    autodan,
    autodan_enhanced,
    autodan_gradient,
    autodan_hierarchical,
    autodan_turbo,
    aegis,
    aegis_ws,  # Aegis WebSocket telemetry endpoint
    chat,
    connection,
    csrf,
    datasets,
    deepteam,
    execute,
    gptfuzz,
    gradient_optimization,
    health,
    intent_aware_generation,
    janus,
    jobs,
    metrics,
    model_selection,
    model_sync,
    mousetrap,
    overthink,
    pipeline_streaming,
    provider_config,
    providers,
    proxy_health,
    session,
    streaming,
    tokens,
    transformation,
    utils,
)

api_router = APIRouter()

# --- System & Core Endpoints ---
api_router.include_router(health.router, tags=["health"])
api_router.include_router(metrics.router, tags=["utils", "admin"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(transformation.router, prefix="/transform", tags=["transformation"])
api_router.include_router(execute.router, tags=["execute"])
api_router.include_router(utils.router, tags=["utils"])
# Datasets endpoints
api_router.include_router(datasets.router, tags=["datasets"])

# --- Unified Adversarial Endpoint (ADR-003) ---
# This is the recommended entry point for all adversarial generation
api_router.include_router(adversarial.router, tags=["adversarial"])

# --- AutoDAN Framework Endpoints (Legacy - will be deprecated) ---
# Unified Router (Primary High-Performance Entry Point)
api_router.include_router(autodan.router, prefix="/autodan", tags=["autodan"])

# Mousetrap: Chain of Iterative Chaos (Advanced Reasoning Model Jailbreaking)
api_router.include_router(mousetrap.router, prefix="/autodan", tags=["autodan", "mousetrap"])

# Specialized Research Routers (Unique Prefixes to Avoid Collision)
api_router.include_router(autodan_enhanced.router, tags=["autodan-research"])
api_router.include_router(
    autodan_hierarchical.router, prefix="/autodan-advanced", tags=["autodan-advanced"]
)
api_router.include_router(
    autodan_gradient.router, prefix="/autodan-gradient", tags=["autodan-advanced"]
)

# AutoDAN-Turbo (ICLR 2025 Lifelong Learning)
api_router.include_router(autodan_turbo.router, tags=["autodan-turbo"])

# --- Fuzzing & Optimization ---
api_router.include_router(gptfuzz.router, prefix="/gptfuzz", tags=["gptfuzz"])
api_router.include_router(
    gradient_optimization.router, prefix="/gradient", tags=["gradient-optimization"]
)

# --- Model & Session Management ---
api_router.include_router(session.router, prefix="/session", tags=["session", "model-sync"])
api_router.include_router(model_sync.router, prefix="/models", tags=["model-sync"])
api_router.include_router(model_selection.router, tags=["model-selection"])
api_router.include_router(connection.router, prefix="/connection", tags=["connection"])

# --- Generation Services ---
api_router.include_router(intent_aware_generation.router, tags=["intent-aware-generation"])
api_router.include_router(
    advanced_generation.router, prefix="/generation", tags=["advanced-generation"]
)
api_router.include_router(autoadv.router, prefix="/autoadv", tags=["autoadv"])
api_router.include_router(providers.router, tags=["providers"])
api_router.include_router(
    provider_config.router, prefix="/provider-config", tags=["provider-config", "providers"]
)

# --- Admin & Infrastructure ---
api_router.include_router(admin.router, tags=["admin"])

# --- Security Endpoints ---
api_router.include_router(csrf.router, tags=["security"])

# --- Background Jobs ---
api_router.include_router(jobs.router, tags=["background-jobs"])

# --- Janus Adversarial Simulation Endpoints ---
api_router.include_router(janus.router, prefix="/janus", tags=["janus", "tier-3", "adversarial"])

# --- DeepTeam Red Teaming Framework ---
api_router.include_router(
    deepteam.router, prefix="/deepteam", tags=["deepteam", "red-teaming", "security"]
)

# --- Project Aegis Red Teaming ---
api_router.include_router(aegis.router, prefix="/aegis", tags=["aegis", "red-teaming"])

# --- Aegis WebSocket Telemetry (Real-Time Campaign Monitoring) ---
api_router.include_router(
    aegis_ws.router, tags=["aegis", "websocket", "telemetry"]
)

# --- Streaming & Token Counting (Google GenAI SDK Enhancement) ---
api_router.include_router(streaming.router, tags=["streaming", "generation"])
api_router.include_router(tokens.router, tags=["tokens", "utilities"])

# --- Data Pipeline Streaming (Real-Time Metrics & Events) ---
api_router.include_router(pipeline_streaming.router, tags=["pipeline-streaming"])

# --- Proxy Health Monitoring (STORY-1.3) ---
api_router.include_router(proxy_health.router, tags=["proxy", "health"])

# --- OVERTHINK Reasoning Token Exploitation ---
api_router.include_router(
    overthink.router,
    prefix="/overthink",
    tags=["overthink", "reasoning", "adversarial"],
)
