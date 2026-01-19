"""
Router Integration Patch for Health Monitoring Endpoints

This patch adds the health monitoring endpoints to the existing v1 router
without breaking the current system. Simply add this code to router.py.
"""

# Add this import to the existing imports in app/api/v1/router.py
# (around line 67 after the other imports)

# Health monitoring endpoints - ADD THIS LINE
try:
    from app.api.v1.endpoints.health_monitoring_fixed import router as health_monitoring_router

    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    health_monitoring_router = None

# Add this to the router registration section (after line 273)
# --- Health Monitoring Dashboard (FIXED VERSION) ---
if HEALTH_MONITORING_AVAILABLE and health_monitoring_router:
    api_router.include_router(
        health_monitoring_router,
        prefix="/health",
        tags=["health", "monitoring", "dashboard"],
    )
else:
    # Fallback health monitoring endpoints
    from fastapi import APIRouter

    fallback_health_router = APIRouter()

    @fallback_health_router.get("/dashboard")
    async def fallback_health_dashboard():
        """Fallback health dashboard when monitoring system not available."""
        return {
            "overall_status": "unknown",
            "total_endpoints": 0,
            "healthy_endpoints": 0,
            "degraded_endpoints": 0,
            "unhealthy_endpoints": 0,
            "active_alerts": 0,
            "endpoints": [],
            "recent_alerts": [],
            "metrics_summary": {},
            "timestamp": "2026-01-18T03:38:00Z",
            "message": "Health monitoring system not available. Using fallback endpoints.",
        }

    @fallback_health_router.get("/endpoints")
    async def fallback_health_endpoints():
        """Fallback endpoint listing."""
        return {
            "endpoints": {},
            "total_count": 0,
            "timestamp": "2026-01-18T03:38:00Z",
            "message": "Health monitoring system not available.",
        }

    @fallback_health_router.get("/monitoring/status")
    async def fallback_monitoring_status():
        """Fallback monitoring status."""
        return {
            "enabled": False,
            "monitored_endpoints": 0,
            "active_alerts": 0,
            "alert_rules": 0,
            "check_interval": 30,
            "timestamp": "2026-01-18T03:38:00Z",
            "message": "Health monitoring system not available.",
        }

    api_router.include_router(
        fallback_health_router,
        prefix="/health",
        tags=["health", "fallback"],
    )
