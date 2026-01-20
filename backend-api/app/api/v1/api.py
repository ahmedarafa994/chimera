from fastapi import APIRouter

from app.api.v1.endpoints import (
    aegis_ws,
    api_keys,
    assessments,
    cicd,
    defense_engine,
    documentation,
    execute,
    generation,
    gptfuzz,
    gradient_optimization,
    health,
    intent_aware_generation,
    multimodal,
    onboarding,
    provider_health_dashboard,
    reports,
    research_lab,
    scheduled_testing,
    session,  # Added session (singular)
    sessions,
    technique_builder,
    templates,
    unified_providers, # Replaces multi_provider
    workspaces,
)

api_router = APIRouter()
api_router.include_router(onboarding.router, tags=["onboarding"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])
api_router.include_router(assessments.router, prefix="/assessments", tags=["assessments"])
api_router.include_router(templates.router, prefix="/templates", tags=["templates"])
# api_router.include_router(techniques.router, prefix="/techniques", tags=["techniques"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(aegis_ws.router)
# api_router.include_router(multi_provider.router, prefix="/multi-provider", tags=["multi-provider"]) # Removed broken module
api_router.include_router(unified_providers.router) # Included unified_providers (prefix defined in router)
api_router.include_router(health.router, tags=["health"])
api_router.include_router(provider_health_dashboard.router)
api_router.include_router(generation.router, prefix="/llm", tags=["llm"])
api_router.include_router(gptfuzz.router, prefix="/gptfuzz", tags=["gptfuzz"])
api_router.include_router(intent_aware_generation.router)
api_router.include_router(gradient_optimization.router, prefix="/gradient", tags=["gradient"])
api_router.include_router(execute.router, tags=["execute"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(session.router, prefix="/session", tags=["session"]) # Added session router
api_router.include_router(documentation.router, prefix="/docs", tags=["documentation"])
api_router.include_router(workspaces.router, prefix="/workspaces", tags=["workspaces"])
api_router.include_router(
    technique_builder.router,
    prefix="/technique-builder",
    tags=["technique-builder"],
)
api_router.include_router(cicd.router, prefix="/cicd", tags=["cicd"])
api_router.include_router(
    scheduled_testing.router,
    prefix="/scheduled-testing",
    tags=["scheduled-testing"],
)
api_router.include_router(multimodal.router, prefix="/multimodal", tags=["multimodal"])
api_router.include_router(research_lab.router, prefix="/research-lab", tags=["research-lab"])
api_router.include_router(defense_engine.router, prefix="/defense-engine", tags=["defense-engine"])
