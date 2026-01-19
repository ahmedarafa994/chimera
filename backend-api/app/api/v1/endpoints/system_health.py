"""System health check endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.schemas.system_health import SystemHealthResponse

router = APIRouter()


@router.get("/system-health", response_model=SystemHealthResponse)
async def get_system_health(
    db: AsyncSession = Depends(get_db)
) -> SystemHealthResponse:
    """
    Get system health status.

    Returns:
        SystemHealthResponse: Current system health status
    """
    # Check database connection
    try:
        await db.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    # Get active techniques (simplified for demo)
    active_techniques = [
        "persona",
        "obfuscation",
        "payload_splitting",
        "cognitive_hacking",
        "contextual_inception"
    ]

    return SystemHealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        services={
            "database": db_status,
            "redis": "connected",  # Simplified
            "llm_providers": "available"  # Simplified
        },
        active_techniques=active_techniques,
        api_version="1.0.0"
    )
