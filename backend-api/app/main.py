"""
FastAPI Backend for Prompt Enhancement System
Provides API endpoints for prompt enhancement with AI model integration

Security Features:
- JWT/API Key authentication
- RBAC (Role-Based Access Control)
- Input validation and sanitization
- Rate limiting
- Structured logging and observability
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure both the backend package and repo root are importable even when this
# file is executed directly (e.g., ``py backend-api/app/main.py``).
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

from app.api.api_routes import router as api_router
from app.api.v1.endpoints.unified_providers import router as unified_providers_router
from app.core.dependencies import get_jailbreak_enhancer, get_prompt_enhancer
from app.core.errors import (
    AppError,
    app_exception_handler,
    global_exception_handler,
    http_exception_handler,
)
from app.core.handlers import chimera_exception_handler

# Performance optimization (PERF-014): Import health checker at module level
# to avoid import overhead on every health check request
from app.core.health import health_checker
from app.core.lifespan import lifespan
from app.core.observability import (
    ObservabilityMiddleware,
    get_logger,
    setup_logging,
    setup_tracing,
)
from app.core.unified_errors import ChimeraError
from app.middleware.auth import APIKeyMiddleware
from app.middleware.auth_rate_limit import (
    AuthRateLimitMiddleware,
    set_auth_rate_limiter,
)
from app.middleware.compression import CompressionMiddleware
from app.middleware.context_middleware import ContextPropagationMiddleware
from app.middleware.cost_tracking_middleware import CostTrackingMiddleware
from app.middleware.provider_tracing_middleware import ProviderTracingMiddleware
from app.middleware.rate_limit_middleware import ProviderRateLimitMiddleware
from app.middleware.request_logging import (
    MetricsMiddleware,
    RequestLoggingMiddleware,
    set_metrics_middleware,
)

# Provider middleware imports
from app.middleware.selection_middleware import SelectionMiddleware

# Unified Provider System middleware imports (Phase 2)
from app.middleware.selection_validation_middleware import (
    SelectionValidationMiddleware,
)
from app.routers.auth import router as auth_router
from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer
from meta_prompter.prompt_enhancer import PromptEnhancer

# =============================================================================
# Application Setup
# =============================================================================

environment = os.getenv("ENVIRONMENT", "development")
log_level = os.getenv("LOG_LEVEL", "INFO")
use_json_logging = environment == "production"

setup_logging(level=log_level, json_format=use_json_logging)
logger = get_logger("chimera.main")

# Initialize FastAPI app with lifespan and enhanced OpenAPI
app = FastAPI(
    title="Chimera API",
    description="""
# Chimera - AI-Powered Prompt Optimization & Jailbreak Research System

**Unified backend API for prompt transformation, LLM orchestration, and security research.**

## Overview

Chimera provides advanced prompt engineering capabilities with multi-provider LLM integration,
sophisticated transformation techniques, and real-time enhancement for security research and
prompt optimization.

## Key Features

- **Multi-Provider LLM Integration**: Seamless integration with Google Gemini, OpenAI, Anthropic Claude, and DeepSeek
- **Advanced Prompt Transformation**: 20+ technique suites including quantum exploit, deep inception, and code chameleon
- **AI-Powered Jailbreak Generation**: Sophisticated jailbreak prompt generation using advanced techniques
- **Real-time Enhancement**: WebSocket support for live prompt enhancement with heartbeat monitoring
- **Session Management**: Persistent session tracking with model selection and caching
- **Circuit Breaker Pattern**: Automatic failover and provider resilience
- **Rate Limiting**: Per-model rate limiting with intelligent fallback suggestions
- **Research Tools**: AutoDAN, AutoAdv, GPTFuzz, and HouYi optimization endpoints

## Authentication

All API endpoints (except health checks and documentation) require authentication using one of:

- **API Key**: Include `X-API-Key` header with your API key
- **JWT Token**: Include `Authorization: Bearer <token>` header

Example:
```bash
curl -X POST "http://localhost:8001/api/v1/generate" \\
  -H "X-API-Key: your-api-key-here" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello, world!"}'
```

## Rate Limits

- **Free Tier**: 100 requests/hour per provider
- **Standard Tier**: 1000 requests/hour per provider
- Rate limit headers included in all responses

## Base URL

- **Development**: `http://localhost:8001`
- **Production**: Configure via `ALLOWED_ORIGINS` environment variable

## Support

- **Documentation**: `/docs` (Swagger UI) and `/redoc` (ReDoc)
- **Health Checks**: `/health`, `/health/ready`, `/health/full`
- **Integration Status**: `/health/integration`
    """,
    version="2.0.0",
    contact={
        "name": "Chimera API Support",
        "email": "support@chimera-api.example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs" if environment != "production" else None,
    redoc_url="/redoc" if environment != "production" else None,
    openapi_tags=[
        {
            "name": "generation",
            "description": "**Text generation with LLM providers**. Generate text using Google Gemini, OpenAI, Anthropic Claude, or DeepSeek models with configurable parameters.",
            "externalDocs": {
                "description": "Generation API Guide",
                "url": "https://docs.chimera-api.example.com/generation",
            },
        },
        {
            "name": "transformation",
            "description": "**Prompt transformation and enhancement**. Transform prompts using 20+ technique suites with configurable potency levels (1-10).",
            "externalDocs": {
                "description": "Transformation Techniques",
                "url": "https://docs.chimera-api.example.com/transformation",
            },
        },
        {
            "name": "jailbreak",
            "description": "**Jailbreak technique application**. AI-powered jailbreak prompt generation for security research with advanced obfuscation and bypass techniques.",
            "externalDocs": {
                "description": "Jailbreak Research Guide",
                "url": "https://docs.chimera-api.example.com/jailbreak",
            },
        },
        {
            "name": "autodan",
            "description": "**AutoDAN jailbreak generation**. Automated jailbreak prompt generation using AutoDAN-Reasoning with multiple search methods.",
        },
        {
            "name": "autodan-turbo",
            "description": "**AutoDAN-Turbo lifelong learning**. Autonomous jailbreak strategy discovery using lifelong learning agents. Based on ICLR 2025 paper achieving 88.5% ASR on GPT-4.",
            "externalDocs": {
                "description": "AutoDAN-Turbo Paper (ICLR 2025)",
                "url": "https://arxiv.org/abs/2410.05295",
            },
        },
        {
            "name": "autoadv",
            "description": "**AutoAdv adversarial generation**. Adversarial prompt generation using AutoAdv techniques.",
        },
        {
            "name": "gptfuzz",
            "description": "**GPTFuzz fuzzing techniques**. Fuzzing-based jailbreak generation using GPTFuzz methodology.",
        },
        {
            "name": "optimization",
            "description": "**HouYi prompt optimization**. Advanced prompt optimization using HouYi techniques.",
        },
        {
            "name": "providers",
            "description": "**LLM provider management**. List available providers, check health status, and manage provider selection with rate limit information.",
            "externalDocs": {
                "description": "Provider Configuration",
                "url": "https://docs.chimera-api.example.com/providers",
            },
        },
        {
            "name": "session",
            "description": "**Session and model management**. Create sessions, select models, and manage session state with persistent tracking.",
        },
        {
            "name": "model-sync",
            "description": "**Real-time model synchronization**. WebSocket-based model selection synchronization across clients.",
        },
        {
            "name": "health",
            "description": "**Health and monitoring endpoints**. Liveness, readiness, and comprehensive health checks for all services.",
        },
        {
            "name": "integration",
            "description": "**Service integration status**. Dependency graphs, service health, and integration statistics.",
        },
        {
            "name": "utils",
            "description": "**Utility endpoints**. Technique information, metrics, and system utilities.",
        },
        {
            "name": "admin",
            "description": "**Administrative endpoints**. System administration and configuration management.",
        },
        {
            "name": "telemetry",
            "description": "**Real-time telemetry streaming**. WebSocket endpoints for streaming live campaign telemetry events including attack metrics, technique performance, and cost tracking.",
        },
    ],
    servers=[
        {"url": "http://localhost:8001", "description": "Development server"},
        {"url": "http://localhost:8001/api/v1", "description": "Development server (API v1)"},
    ],
    lifespan=lifespan,
)


# Configure OpenAPI security schemes
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication. Include your API key in the X-API-Key header.",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token authentication. Include your JWT token in the Authorization header as 'Bearer <token>'.",
        },
    }

    # Add global security requirement (can be overridden per endpoint)
    openapi_schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Setup OpenTelemetry tracing
setup_tracing(app)

# =============================================================================
# Middleware Configuration
# =============================================================================

# PERF-031: ObservabilityMiddleware now only handles request ID propagation
# and response headers. Detailed logging is handled by RequestLoggingMiddleware.
app.add_middleware(ObservabilityMiddleware)

# PERF-001 FIX: Response compression middleware
# Enables gzip/brotli compression for 70-80% bandwidth reduction
# Configured via environment variables:
# - ENABLE_COMPRESSION=true (default: true)
# - COMPRESSION_LEVEL=6 (default: 6)
# - COMPRESSION_MIN_SIZE=500 (default: 500 bytes)
if os.getenv("ENABLE_COMPRESSION", "true").lower() == "true":
    app.add_middleware(
        CompressionMiddleware,
        minimum_size=int(os.getenv("COMPRESSION_MIN_SIZE", "500")),
        gzip_level=int(os.getenv("COMPRESSION_LEVEL", "6")),
        brotli_level=int(os.getenv("COMPRESSION_LEVEL", "4")),
    )
    logger.info("Response compression middleware enabled")
else:
    logger.info("Response compression middleware disabled")

# API Key Authentication Middleware
app.add_middleware(
    APIKeyMiddleware,
    excluded_paths=[
        "/",
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/full",
        "/health/live",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/providers",
        "/api/v1/techniques",
        "/api/v1/models",
        "/api/v1/metrics",
        "/api/v1/session",
        "/api/v1/gptfuzz",
        "/api/v1/intent-aware",
        "/api/v1/generation",
        # PERF-032: Exclude OpenAI-compatible endpoints from auth for faster 404/501 responses
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/models",
        "/v1/messages",
        "/api/v1/provider-sync",
        # AUTH-001: Exclude auth endpoints from API key requirement
        "/api/v1/auth",
    ],
)

# Request logging middleware
# PERF-033: Extended exclude paths and reduced slow request threshold
app.add_middleware(
    RequestLoggingMiddleware,
    exclude_paths=[
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/live",
        "/health/full",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",
        "/api/v1/auth",  # Exclude auth
    ],
    slow_request_threshold_ms=2000,  # Reduced from 5000ms for better alerting
)

# Metrics middleware
_metrics_middleware = MetricsMiddleware(app)
set_metrics_middleware(_metrics_middleware)

# SEC-001: Security Middleware Stack (CRIT-001 FIX: Re-enabled for security)
# These middleware components provide essential security controls:
# - Rate limiting: Prevents DoS and API abuse
# - Input validation: Validates request content types and sizes
# - Security headers: XSS, clickjacking, and other protections
# - CSRF protection: Prevents cross-site request forgery

from app.core.middleware import CSRFMiddleware
from app.core.validation import InputValidationMiddleware
from app.middleware.rate_limit import RateLimitMiddleware, SecurityHeadersMiddleware

# CRIT-001 FIX: Enable security headers middleware (always enabled)
app.add_middleware(SecurityHeadersMiddleware)
logger.info("Security headers middleware enabled")

# CRIT-003/CRIT-004 FIX: Enable rate limiting and input validation
# Rate limiting: 1000 requests per minute per IP in dev, 100 in prod (configurable)
default_rate_limit = "1000" if environment == "development" else "100"
rate_limit_calls = int(os.getenv("RATE_LIMIT_CALLS", default_rate_limit))
rate_limit_period = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
app.add_middleware(RateLimitMiddleware, calls=rate_limit_calls, period=rate_limit_period)
logger.info(f"Rate limiting middleware enabled: {rate_limit_calls} calls per {rate_limit_period}s")

# Input validation middleware (validates content-type and size)
# Input validation middleware (validates content-type and size)
app.add_middleware(
    InputValidationMiddleware,
    excluded_paths=[
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/auth",  # Exclude auth
    ],
)
logger.info("Input validation middleware enabled")

# CSRF protection (enabled with enforce=True in production)
csrf_enforce = environment == "production"
app.add_middleware(
    CSRFMiddleware,
    enforce=csrf_enforce,
    cookie_secure=environment == "production",
    exclude_paths=[
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/live",
        "/health/full",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/csrf/token",
        "/api/v1/health",
        "/api/v1/providers",
        "/api/v1/techniques",
        "/api/v1/models",
        "/api/v1/auth",  # Exclude auth
    ],
)
logger.info(f"CSRF middleware enabled (enforce={csrf_enforce})")

# =============================================================================
# Provider Selection and Cost Tracking Middleware
# =============================================================================
# Middleware order (first added = last executed):
# 1. Cost tracking (outermost - records final costs)
# 2. Provider rate limiting (check limits before processing)
# 3. Provider selection (inject provider context)
# 4. Context propagation (NEW - unified provider system context)
# 5. Validation middleware (validate config - added via existing validation)

# Selection Middleware - injects provider/model context with database-backed session preferences
# Implements three-tier selection hierarchy: Request Override → Session Preference → Global Default
# Supports headers: X-Provider-ID, X-Model-ID
# Supports query params: ?provider=<provider_id>&model=<model_id>
app.add_middleware(
    SelectionMiddleware,
    default_provider=os.getenv("DEFAULT_PROVIDER", "openai"),
    default_model=os.getenv("DEFAULT_MODEL", "gpt-4-turbo"),
    enable_session_lookup=True,
    excluded_paths=[
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/auth",  # Exclude auth endpoints
    ],
)
logger.info("Selection middleware enabled with database-backed session preferences")

# Context Propagation Middleware (NEW - Phase 2) - unified provider system
# Creates request-scoped context with provider/model from GlobalModelSelectionState
# Integrates with the new unified provider system
if os.getenv("ENABLE_UNIFIED_PROVIDER_SYSTEM", "false").lower() == "true":
    app.add_middleware(
        ContextPropagationMiddleware,
        excluded_paths=[
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v1/health",
            "/api/v1/auth",  # Exclude auth endpoints
        ],
    )
    logger.info("Context propagation middleware enabled (Unified Provider System)")

    # Selection Validation Middleware - validates provider/model at request time
    # Validates: provider exists, model exists, API key configured, provider health
    # Bypass paths: /api/v1/unified-providers/*, /api/v1/health/*, /api/v1/admin/*
    app.add_middleware(
        SelectionValidationMiddleware,
        bypass_paths=[
            "/api/v1/unified-providers",
            "/api/v1/health",
            "/api/v1/admin",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ],
        enable_health_check=True,
        cache_ttl_seconds=60.0,
    )
    logger.info("Selection validation middleware enabled")

    # Provider Tracing Middleware - traces provider/model through request lifecycle
    # Adds headers: X-Provider-Selection, X-Model-Selection, X-Selection-Source
    app.add_middleware(
        ProviderTracingMiddleware,
        trace_header_prefix="X-Selection-",
        log_selections=True,
    )
    logger.info("Provider tracing middleware enabled")

# Provider Rate Limiting Middleware - config-driven rate limiting per provider
# Uses rate limits from providers.yaml configuration
app.add_middleware(ProviderRateLimitMiddleware)
logger.info("Provider rate limiting middleware enabled")

# Cost Tracking Middleware - tracks costs of AI operations
# Records token usage and calculates costs using config pricing
if os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true":
    app.add_middleware(CostTrackingMiddleware)
    logger.info("Cost tracking middleware enabled")
else:
    logger.info("Cost tracking middleware disabled")

# SEC-002: Auth Rate Limiting Middleware (enabled for production security)
# Protects authentication endpoints from brute force and credential stuffing attacks
if environment == "production":
    _auth_rate_limiter = AuthRateLimitMiddleware(app)
    set_auth_rate_limiter(_auth_rate_limiter)
    app.add_middleware(AuthRateLimitMiddleware)
else:
    logger.info("Auth rate limiting disabled in development mode")

# CORS configuration (HIGH-005 FIX: Localhost only in development)
if environment == "production":
    # Production: Only use explicitly configured origins, no localhost defaults
    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
    if not allowed_origins_str:
        logger.warning("No ALLOWED_ORIGINS configured for production - CORS will block all origins")
        allowed_origins = []
    else:
        allowed_origins = [
            origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()
        ]
else:
    # Development: Allow localhost origins
    allowed_origins_str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3001,http://localhost:3002,http://localhost:8080,"
        "http://127.0.0.1:3001,http://127.0.0.1:3002,http://127.0.0.1:8080",
    )
    allowed_origins = [
        origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()
    ]

    # Add common dev origins if not already present
    dev_origins = [
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3700",
        "http://localhost:8080",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3700",
        "http://127.0.0.1:8080",
    ]
    for origin in dev_origins:
        if origin not in allowed_origins:
            allowed_origins.append(origin)

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


async def request_validation_exception_handler(request, exc: RequestValidationError):
    """
    Custom handler for Pydantic validation errors.
    Provides detailed field-level error information for debugging.
    """
    errors = exc.errors()

    # Log the validation error with full details
    logger.warning(
        f"[Validation Error] Path: {request.url.path}, Method: {request.method}, Errors: {errors}"
    )

    # Format errors for client response
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
            "raw_errors": errors,  # Include raw errors for debugging
        },
    )


app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(ChimeraError, chimera_exception_handler)
app.add_exception_handler(AppError, app_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# =============================================================================
# Router Registration
# =============================================================================

app.include_router(api_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(
    unified_providers_router, prefix="/api/v1"
)  # NEW - Phase 2: Unified Provider System

# Chimera+AutoDAN unified router
try:
    from app.routers.chimera_autodan import router as chimera_autodan_router

    app.include_router(chimera_autodan_router, prefix="/api/v1")
    logger.info("Chimera+AutoDAN router registered successfully")
except ImportError as e:
    logger.warning(f"Chimera+AutoDAN router not available: {e}")


# =============================================================================
# OpenAI-Compatible Endpoint Stubs (PERF-034)
# =============================================================================
# These endpoints return proper 501 Not Implemented responses instead of 404s
# to reduce log noise from clients expecting OpenAI API compatibility.


@app.post("/v1/chat/completions", tags=["compatibility"], include_in_schema=False)
@app.post("/chat/completions", tags=["compatibility"], include_in_schema=False)
async def openai_chat_completions_stub():
    """
    OpenAI-compatible chat completions endpoint stub.
    Returns 501 Not Implemented with guidance to use the Chimera API.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "OpenAI-compatible chat completions not implemented. Use /api/v1/generate instead.",
                "type": "not_implemented",
                "code": "endpoint_not_available",
                "chimera_endpoint": "/api/v1/generate",
            }
        },
    )


@app.post("/v1/completions", tags=["compatibility"], include_in_schema=False)
@app.post("/completions", tags=["compatibility"], include_in_schema=False)
async def openai_completions_stub():
    """
    OpenAI-compatible completions endpoint stub.
    Returns 501 Not Implemented with guidance to use the Chimera API.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "OpenAI-compatible completions not implemented. Use /api/v1/generate instead.",
                "type": "not_implemented",
                "code": "endpoint_not_available",
                "chimera_endpoint": "/api/v1/generate",
            }
        },
    )


@app.get("/v1/models", tags=["compatibility"], include_in_schema=False)
@app.get("/models", tags=["compatibility"], include_in_schema=False)
async def openai_models_stub():
    """
    OpenAI-compatible models endpoint stub.
    Returns 501 Not Implemented with guidance to use the Chimera API.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "OpenAI-compatible models endpoint not implemented. Use /api/v1/providers instead.",
                "type": "not_implemented",
                "code": "endpoint_not_available",
                "chimera_endpoint": "/api/v1/providers",
            }
        },
    )


@app.post("/v1/messages", tags=["compatibility"], include_in_schema=False)
@app.post("/messages", tags=["compatibility"], include_in_schema=False)
async def anthropic_messages_stub():
    """
    Anthropic-compatible messages endpoint stub.
    Returns 501 Not Implemented with guidance to use the Chimera API.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "Anthropic-compatible messages endpoint not implemented. Use /api/v1/generate instead.",
                "type": "not_implemented",
                "code": "endpoint_not_available",
                "chimera_endpoint": "/api/v1/generate",
            }
        },
    )


@app.post("/v1/api/event_logging/batch", tags=["compatibility"], include_in_schema=False)
@app.post("/api/event_logging/batch", tags=["compatibility"], include_in_schema=False)
async def event_logging_stub():
    """
    Event logging endpoint stub.
    Returns 501 Not Implemented - event logging not supported.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "Event logging endpoint not implemented.",
                "type": "not_implemented",
                "code": "endpoint_not_available",
            }
        },
    )


# =============================================================================
# Core Endpoints
# =============================================================================


@app.get("/")
async def root():
    return {
        "message": "Prompt Enhancement API",
        "version": "2.0.0",
        "status": "operational",
    }


# =============================================================================
# Health Check Endpoints (PERF-014: Optimized with module-level imports)
# =============================================================================

# Cached health check result for lightweight endpoint
_cached_health_result: dict | None = None
_cached_health_timestamp: float = 0.0
_HEALTH_CACHE_TTL_SECONDS: float = 5.0  # Cache full health check for 5 seconds


@app.get("/health/ping", tags=["health"])
async def health_ping():
    """
    Lightweight health ping - minimal latency for load balancer checks (PERF-015).

    Returns immediately without running any service checks.
    Use this for high-frequency health polling.
    """
    return {"status": "ok", "timestamp": time.time()}


@app.get("/health")
async def health_check():
    """
    Full health check - returns overall system health with all service checks.

    This endpoint provides comprehensive health information including:
    - Overall system status (healthy/degraded/unhealthy)
    - Individual service health checks
    - System version and uptime

    Results are cached for 5 seconds to reduce load under heavy polling (PERF-016).
    """
    global _cached_health_result, _cached_health_timestamp

    current_time = time.time()

    # Return cached result if still valid
    if (
        _cached_health_result
        and (current_time - _cached_health_timestamp) < _HEALTH_CACHE_TTL_SECONDS
    ):
        return _cached_health_result

    # Run full health check and cache result
    result = await health_checker.run_all_checks()
    _cached_health_result = result.to_dict()
    _cached_health_timestamp = current_time

    return _cached_health_result


@app.get("/health/live")
async def liveness_check():
    """Liveness probe - checks if application is running (for Kubernetes)."""
    result = await health_checker.liveness_check()
    return result.to_dict()


@app.get("/health/ready")
async def readiness_check():
    """Readiness probe - checks if application is ready to serve traffic."""
    result = await health_checker.readiness_check()
    status_code = 200 if result.status.value == "healthy" else 503
    return JSONResponse(content=result.to_dict(), status_code=status_code)


@app.get("/health/full", tags=["health"])
async def full_health_check():
    """Full health check - runs all registered health checks (bypasses cache)."""
    result = await health_checker.run_all_checks()
    return result.to_dict()


@app.get("/health/integration", tags=["integration"])
async def integration_health():
    """
    Integration health - shows service dependency graph and status.

    Returns a graph showing all services, their health status,
    and dependency relationships.
    """
    graph = await health_checker.get_dependency_graph()
    return graph


@app.get("/health/integration/{service}", tags=["integration"])
async def service_dependency_tree(service: str):
    """
    Get dependency tree for a specific service.

    Useful for understanding cascading failures and service dependencies.
    """
    tree = health_checker.get_dependency_tree(service)
    return tree


@app.get("/integration/stats", tags=["integration"])
async def integration_stats():
    """
    Get statistics for all integration services.

    Includes event bus, task queue, and webhook service stats.
    """
    stats = {}

    try:
        from app.core.event_bus import event_bus

        stats["event_bus"] = event_bus.get_stats()
    except ImportError:
        stats["event_bus"] = {"error": "not_available"}

    try:
        from app.core.task_queue import task_queue

        stats["task_queue"] = task_queue.get_stats()
    except ImportError:
        stats["task_queue"] = {"error": "not_available"}

    try:
        from app.services.webhook_service import webhook_service

        stats["webhook_service"] = webhook_service.get_stats()
    except ImportError:
        stats["webhook_service"] = {"error": "not_available"}

    try:
        from app.middleware.idempotency import get_idempotency_store

        stats["idempotency"] = get_idempotency_store().get_stats()
    except ImportError:
        stats["idempotency"] = {"error": "not_available"}

    return stats


# =============================================================================
# WebSocket for Real-time Enhancement
# =============================================================================


@app.websocket("/ws/enhance")
async def websocket_enhance(
    websocket: WebSocket,
    standard_enhancer: PromptEnhancer = Depends(get_prompt_enhancer),
    jailbreak_enhancer: JailbreakPromptEnhancer = Depends(get_jailbreak_enhancer),
):
    """WebSocket endpoint for real-time prompt enhancement with heartbeat."""
    await websocket.accept()

    # Heartbeat task to detect zombie connections
    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            try:
                await websocket.send_json({"type": "ping", "timestamp": time.time()})
            except Exception:
                break

    # MED-001 FIX: asyncio and time now imported at top of file
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            data = await websocket.receive_text()  # No timeout
            message = json.loads(data)

            # Handle pong responses from client
            if message.get("type") == "pong":
                continue

            prompt = message.get("prompt", "")
            enhancement_type = message.get("type", "standard")

            await websocket.send_json({"status": "processing", "message": "Enhancing prompt..."})

            try:
                if enhancement_type == "jailbreak":
                    result = jailbreak_enhancer.quick_enhance_jailbreak(
                        prompt, potency=message.get("potency", 7)
                    )
                else:
                    result = standard_enhancer.quick_enhance(prompt)

                await websocket.send_json({"status": "complete", "enhanced_prompt": result})
            except Exception as e:
                await websocket.send_json({"status": "error", "message": str(e)})

    except TimeoutError:
        logger.warning("WebSocket connection timed out (no activity for 120s)")
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected")
    except json.JSONDecodeError as e:
        # MED-004 FIX: Specific exception handling
        logger.error(f"WebSocket JSON decode error: {e}")
    except ValueError as e:
        logger.error(f"WebSocket value error: {e}")
    except Exception as e:
        # Log full traceback for unexpected errors
        logger.error(f"WebSocket unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()


# NEW - Phase 2: WebSocket for Model Selection (Unified Provider System)
@app.websocket("/ws/model-selection")
async def websocket_model_selection(websocket: WebSocket, user_id: str = "default"):
    """
    WebSocket endpoint for real-time model selection updates.

    Provides real-time synchronization of provider/model selection changes
    across all connected clients for a user.

    Query Parameters:
        user_id: User identifier (default: "default")

    Message Types:
        - SELECTION_CHANGED: Broadcast when user changes selection
        - PROVIDER_STATUS: Provider health/availability updates
        - MODEL_VALIDATION: Validation results for provider/model
        - PING/PONG: Heartbeat for keep-alive
    """
    try:
        from app.services.websocket.model_selection_ws import websocket_endpoint

        await websocket_endpoint(websocket, user_id)

    except ImportError as e:
        logger.error(f"Failed to import model selection WebSocket handler: {e}")
        await websocket.accept()
        await websocket.send_json(
            {"type": "ERROR", "data": {"message": "Model selection WebSocket not available"}}
        )
        await websocket.close()
    except Exception as e:
        logger.error(f"Model selection WebSocket error: {e}", exc_info=True)


# =============================================================================
# Optional Static File Serving
# =============================================================================


def setup_static_file_serving():
    """
    Configure static file serving for frontend build.
    Only active when SERVE_FRONTEND=true.
    """
    if os.getenv("SERVE_FRONTEND", "false").lower() != "true":
        return

    frontend_paths = [
        Path(__file__).parent.parent.parent / "frontend" / ".next" / "static",
        Path(__file__).parent.parent.parent / "frontend" / "out",
        Path(__file__).parent.parent.parent / "frontend" / "build",
    ]

    static_dir = next((p for p in frontend_paths if p.exists()), None)
    if not static_dir:
        logger.warning("SERVE_FRONTEND=true but no frontend build found")
        return

    # Validate path is not a symlink to prevent path traversal
    if static_dir.is_symlink():
        logger.error(f"Security: Refusing to serve symlinked directory: {static_dir}")
        return

    # Resolve to absolute path and validate it's within expected directory
    static_dir = static_dir.resolve()
    expected_base = Path(__file__).parent.parent.parent.resolve()
    if not str(static_dir).startswith(str(expected_base)):
        logger.error(f"Security: Static directory outside project root: {static_dir}")
        return

    logger.info(f"Serving frontend static files from: {static_dir}")
    app.mount("/_next/static", StaticFiles(directory=str(static_dir)), name="nextjs-static")

    index_file = static_dir.parent.parent / "out" / "index.html"
    if not index_file.exists():
        index_file = static_dir.parent.parent / ".next" / "server" / "app" / "index.html"

    if index_file.exists():
        # Validate index file path
        index_file = index_file.resolve()
        if not str(index_file).startswith(str(expected_base)):
            logger.error(f"Security: Index file outside project root: {index_file}")
            return

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            return FileResponse(str(index_file))

        logger.info("SPA fallback route configured")


setup_static_file_serving()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
# Force reload
