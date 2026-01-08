from fastapi import APIRouter

from app.api.v1.endpoints import (
    execute,
    generation,
    gptfuzz,
    gradient_optimization,
    intent_aware_generation,
    session,
)

api_router = APIRouter()
api_router.include_router(generation.router, prefix="/llm", tags=["llm"])
api_router.include_router(gptfuzz.router, prefix="/gptfuzz", tags=["gptfuzz"])
api_router.include_router(intent_aware_generation.router)
api_router.include_router(gradient_optimization.router, prefix="/gradient", tags=["gradient"])
api_router.include_router(execute.router, tags=["execute"])
api_router.include_router(session.router, prefix="/session", tags=["session"])
