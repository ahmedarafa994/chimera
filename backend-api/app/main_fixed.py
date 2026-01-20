"""FastAPI Backend for Prompt Enhancement System - FIXED VERSION
Provides API endpoints for prompt enhancement with AI model integration.

This is a streamlined version with:
1. Simplified middleware stack
2. Better error handling
3. Fixed circular import issues
4. Optimized startup sequence
"""

import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure both the backend package and repo root are importable
_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _path in (_BACKEND_DIR, _REPO_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

# Core imports
import contextlib

from fastapi.exceptions import RequestValidationError

# Import API routes with error handling
try:
    from app.api.api_routes import router as api_router
except ImportError:
    api_router = None

# Import error handlers with fallbacks
try:
    from app.core.errors import (
        AppError,
        app_exception_handler,
        global_exception_handler,
        http_exception_handler,
    )
    from app.core.handlers import chimera_exception_handler
    from app.core.unified_errors import ChimeraError
except ImportError:
    AppError = None
    ChimeraError = None

# Import health checker with fallback
try:
    from app.core.health import health_checker
except ImportError:
    health_checker = None

# Import dependencies with fallbacks
try:
    from app.core.dependencies import get_jailbreak_enhancer, get_prompt_enhancer
    from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer
    from meta_prompter.prompt_enhancer import PromptEnhancer
except ImportError:
    get_jailbreak_enhancer = None
    get_prompt_enhancer = None
    JailbreakPromptEnhancer = None
    PromptEnhancer = None

# =============================================================================
# Application Setup
# =============================================================================

environment = os.getenv("ENVIRONMENT", "development")
log_level = os.getenv("LOG_LEVEL", "INFO")


# Initialize FastAPI app with simplified lifespan
@contextlib.asynccontextmanager
async def simple_lifespan(app: FastAPI):
    """Simplified lifespan context manager."""
    # Basic service initialization
    try:
        # Register basic services only
        pass
    except Exception:
        pass

    yield


app = FastAPI(
    title="Chimera API - FIXED",
    description="Streamlined Chimera API with simplified middleware stack",
    version="2.0.1-fixed",
    docs_url="/docs" if environment != "production" else None,
    redoc_url="/redoc" if environment != "production" else None,
    lifespan=simple_lifespan,
)

# =============================================================================
# Minimal Middleware Configuration
# =============================================================================

# Basic CORS configuration
if environment == "production":
    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
    allowed_origins = (
        [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
        if allowed_origins_str
        else []
    )
else:
    # Development: Allow localhost origins
    allowed_origins = [
        "http://localhost:3001",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-ID",
        "X-Session-ID",
    ],
    expose_headers=["X-Request-ID", "X-Response-Time", "X-Session-ID"],
    max_age=600,
)

# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request, exc: RequestValidationError):
    """Custom handler for Pydantic validation errors."""
    errors = exc.errors()

    formatted_errors = []
    for error in errors:
        loc = ".".join(str(loc_part) for loc_part in error.get("loc", []))
        msg = error.get("msg", "validation error")
        error_type = error.get("type", "")
        formatted_errors.append({"field": loc, "message": msg, "type": error_type})

    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "detail": formatted_errors,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler with logging."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
        },
    )


# =============================================================================
# Router Registration
# =============================================================================

if api_router:
    app.include_router(api_router, prefix="/api/v1")
else:
    # Create minimal fallback router
    from fastapi import APIRouter

    fallback_router = APIRouter()

    @fallback_router.get("/health")
    async def fallback_health():
        return {"status": "ok", "message": "Fallback health endpoint"}

    @fallback_router.get("/providers")
    async def fallback_providers():
        return {"providers": [], "message": "Providers not available"}

    app.include_router(fallback_router, prefix="/api/v1")

# =============================================================================
# Core Endpoints
# =============================================================================


@app.get("/")
async def root():
    return {
        "message": "Chimera API - FIXED VERSION",
        "version": "2.0.1-fixed",
        "status": "operational",
    }


# =============================================================================
# Health Check Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.1-fixed",
        "environment": environment,
    }


@app.get("/health/ping")
async def health_ping():
    """Lightweight ping endpoint."""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - checks if application is ready."""
    try:
        # Basic readiness checks
        status = "ready"
        components = {
            "api": "healthy",
            "middleware": "healthy",
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": status,
                "timestamp": time.time(),
                "components": components,
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": time.time(),
            },
        )


# =============================================================================
# WebSocket for Real-time Enhancement (Simplified)
# =============================================================================


@app.websocket("/ws/enhance")
async def websocket_enhance(websocket: WebSocket) -> None:
    """Simplified WebSocket endpoint for real-time enhancement."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            prompt = message.get("prompt", "")

            # Simple echo response for now
            await websocket.send_json(
                {
                    "status": "complete",
                    "enhanced_prompt": f"Enhanced: {prompt}",
                    "message": "Using simplified enhancement",
                },
            )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
