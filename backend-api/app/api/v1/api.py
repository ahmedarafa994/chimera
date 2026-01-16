from fastapi import APIRouter

from app.api.v1.endpoints import (
    api_keys,
    assessments,
    execute,
    generation,
    gptfuzz,
    gradient_optimization,
    intent_aware_generation,
    multi_provider,
    onboarding,
    progress_ws,
    reports,
    session,
    techniques,
    templates,
)

api_router = APIRouter()
api_router.include_router(onboarding.router, tags=["onboarding"])
api_router.include_router(api_keys.router, prefix="/api-keys", tags=["api-keys"])
api_router.include_router(assessments.router, prefix="/assessments", tags=["assessments"])
api_router.include_router(templates.router, prefix="/templates", tags=["templates"])
api_router.include_router(techniques.router, prefix="/techniques", tags=["techniques"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(progress_ws.router, prefix="/ws", tags=["websockets"])
api_router.include_router(multi_provider.router, prefix="/multi-provider", tags=["multi-provider"])
api_router.include_router(generation.router, prefix="/llm", tags=["llm"])
api_router.include_router(gptfuzz.router, prefix="/gptfuzz", tags=["gptfuzz"])
api_router.include_router(intent_aware_generation.router)
api_router.include_router(gradient_optimization.router, prefix="/gradient", tags=["gradient"])
api_router.include_router(execute.router, tags=["execute"])
api_router.include_router(session.router, prefix="/session", tags=["session"])
