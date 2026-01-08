# Backend Integration Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance the already production-ready Chimera backend with request correlation IDs, service health dashboard, and improved observability for distributed tracing.

**Architecture:** The system already has excellent integration via ServiceRegistry, unified error handling through ChimeraError hierarchy, and comprehensive middleware stack. These enhancements add distributed tracing capabilities and operational visibility without disrupting existing architecture.

**Tech Stack:** FastAPI, OpenTelemetry, Python 3.11+, existing middleware infrastructure

---

## Task 1: Add Request Correlation IDs

**Files:**
- Modify: `backend-api/app/middleware/request_logging.py:1-220`
- Modify: `backend-api/app/core/observability.py:1-200`
- Test: Manual verification via API calls

**Step 1: Write the failing test**

```python
# backend-api/tests/test_correlation_ids.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_correlation_id_in_response_headers():
    """Test that correlation ID is added to response headers"""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

def test_correlation_id_propagates_through_services():
    """Test that correlation ID is available in request state"""
    response = client.post(
        "/api/v1/generate",
        json={"prompt": "test", "provider": "google"},
        headers={"X-API-Key": "test-key"}
    )
    assert "X-Request-ID" in response.headers
```

**Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/test_correlation_ids.py -v`
Expected: FAIL with "KeyError: 'X-Request-ID'"

**Step 3: Write minimal implementation**

```python
# backend-api/app/middleware/request_logging.py
import uuid
from fastapi import Request

class RequestLoggingMiddleware:
    async def dispatch(self, request: Request, call_next):
        # Generate or extract correlation ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
```

**Step 4: Run test to verify it passes**

Run: `cd backend-api && pytest tests/test_correlation_ids.py -v`
Expected: PASS

**Step 5: Update observability middleware to use correlation IDs**

```python
# backend-api/app/core/observability.py
from fastapi import Request

class ObservabilityMiddleware:
    async def dispatch(self, request: Request, call_next):
        # Get correlation ID from request state
        request_id = getattr(request.state, "request_id", "unknown")

        # Add to span attributes if tracing is enabled
        if hasattr(request.state, "span"):
            request.state.span.set_attribute("request.id", request_id)

        response = await call_next(request)
        return response
```

**Step 6: Commit**

```bash
git add backend-api/app/middleware/request_logging.py backend-api/app/core/observability.py backend-api/tests/test_correlation_ids.py
git commit -m "feat: add request correlation IDs for distributed tracing"
```

---

## Task 2: Create Service Health Dashboard Endpoint

**Files:**
- Create: `backend-api/app/api/v1/endpoints/health_dashboard.py`
- Modify: `backend-api/app/api/v1/router.py:1-50`
- Modify: `backend-api/app/core/service_registry.py:79-100`
- Test: `backend-api/tests/test_health_dashboard.py`

**Step 1: Write the failing test**

```python
# backend-api/tests/test_health_dashboard.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_dashboard_returns_service_status():
    """Test that health dashboard returns all service statuses"""
    response = client.get("/api/v1/health/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "services" in data
    assert "middleware" in data
    assert "providers" in data

def test_health_dashboard_includes_llm_service():
    """Test that LLM service is included in dashboard"""
    response = client.get("/api/v1/health/dashboard")
    data = response.json()
    assert "llm_service" in data["services"]
    assert data["services"]["llm_service"]["registered"] is True
```

**Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/test_health_dashboard.py -v`
Expected: FAIL with "404 Not Found"

**Step 3: Write minimal implementation**

```python
# backend-api/app/api/v1/endpoints/health_dashboard.py
from fastapi import APIRouter, Depends
from app.core.service_registry import service_registry
from app.services.llm_service import llm_service
from app.core.auth import get_current_user

router = APIRouter()

@router.get("/health/dashboard", tags=["health"])
async def health_dashboard(user=Depends(get_current_user)):
    """
    Comprehensive health dashboard showing all service statuses.

    Returns:
        - services: Status of all registered services
        - middleware: Middleware stack status
        - providers: Available LLM providers
    """
    # Get service status from registry
    services = service_registry.get_service_status()

    # Get middleware status
    middleware = {
        "observability": "active",
        "authentication": "active",
        "rate_limiting": "active",
        "validation": "active",
        "security_headers": "active"
    }

    # Get provider status
    providers = await llm_service.list_providers()

    return {
        "services": services,
        "middleware": middleware,
        "providers": providers.dict()
    }
```

**Step 4: Register the router**

```python
# backend-api/app/api/v1/router.py
from app.api.v1.endpoints import health_dashboard

# Add to router includes
api_router.include_router(
    health_dashboard.router,
    tags=["health"]
)
```

**Step 5: Run test to verify it passes**

Run: `cd backend-api && pytest tests/test_health_dashboard.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add backend-api/app/api/v1/endpoints/health_dashboard.py backend-api/app/api/v1/router.py backend-api/tests/test_health_dashboard.py
git commit -m "feat: add service health dashboard endpoint"
```

---

## Task 3: Enhance OpenTelemetry Exporter Configuration

**Files:**
- Modify: `backend-api/app/core/observability.py:1-200`
- Create: `backend-api/app/core/tracing_config.py`
- Modify: `backend-api/app/core/config.py:1-518`
- Test: Manual verification with Jaeger/Zipkin

**Step 1: Write the failing test**

```python
# backend-api/tests/test_tracing_config.py
import pytest
from app.core.tracing_config import TracingConfig

def test_tracing_config_loads_from_env():
    """Test that tracing configuration loads from environment"""
    config = TracingConfig()
    assert config.enabled is not None
    assert config.exporter_type in ["jaeger", "zipkin", "otlp", "console"]

def test_tracing_config_validates_endpoint():
    """Test that tracing endpoint is validated"""
    config = TracingConfig()
    if config.enabled and config.exporter_type != "console":
        assert config.endpoint is not None
        assert config.endpoint.startswith("http")
```

**Step 2: Run test to verify it fails**

Run: `cd backend-api && pytest tests/test_tracing_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.core.tracing_config'"

**Step 3: Write minimal implementation**

```python
# backend-api/app/core/tracing_config.py
from pydantic import Field
from pydantic_settings import BaseSettings

class TracingConfig(BaseSettings):
    """OpenTelemetry tracing configuration"""

    # Enable/disable tracing
    TRACING_ENABLED: bool = Field(default=False, description="Enable OpenTelemetry tracing")

    # Exporter type
    TRACING_EXPORTER: str = Field(
        default="console",
        description="Tracing exporter type: jaeger, zipkin, otlp, console"
    )

    # Exporter endpoint
    TRACING_ENDPOINT: str | None = Field(
        default=None,
        description="Tracing exporter endpoint URL"
    )

    # Service name
    TRACING_SERVICE_NAME: str = Field(
        default="chimera-backend",
        description="Service name for tracing"
    )

    # Sampling rate (0.0 to 1.0)
    TRACING_SAMPLE_RATE: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate"
    )

    @property
    def enabled(self) -> bool:
        return self.TRACING_ENABLED

    @property
    def exporter_type(self) -> str:
        return self.TRACING_EXPORTER

    @property
    def endpoint(self) -> str | None:
        return self.TRACING_ENDPOINT

    class Config:
        env_file = ".env"
        case_sensitive = True
```

**Step 4: Update observability.py to use new config**

```python
# backend-api/app/core/observability.py
from app.core.tracing_config import TracingConfig

def setup_tracing(app: FastAPI):
    """Setup OpenTelemetry tracing with configurable exporter"""
    tracing_config = TracingConfig()

    if not tracing_config.enabled:
        logger.info("Tracing disabled")
        return

    # Configure exporter based on type
    if tracing_config.exporter_type == "jaeger":
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        exporter = JaegerExporter(
            agent_host_name=tracing_config.endpoint or "localhost",
            agent_port=6831
        )
    elif tracing_config.exporter_type == "zipkin":
        from opentelemetry.exporter.zipkin.json import ZipkinExporter
        exporter = ZipkinExporter(
            endpoint=tracing_config.endpoint or "http://localhost:9411/api/v2/spans"
        )
    elif tracing_config.exporter_type == "otlp":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(
            endpoint=tracing_config.endpoint or "http://localhost:4317"
        )
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()

    # Setup tracer provider
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({
            "service.name": tracing_config.TRACING_SERVICE_NAME
        })
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set global tracer provider
    from opentelemetry import trace
    trace.set_tracer_provider(provider)

    logger.info(f"Tracing enabled with {tracing_config.exporter_type} exporter")
```

**Step 5: Run test to verify it passes**

Run: `cd backend-api && pytest tests/test_tracing_config.py -v`
Expected: PASS

**Step 6: Update .env.template with tracing variables**

```bash
# backend-api/.env.template
# OpenTelemetry Tracing Configuration
TRACING_ENABLED=false
TRACING_EXPORTER=console  # jaeger, zipkin, otlp, console
TRACING_ENDPOINT=http://localhost:4317
TRACING_SERVICE_NAME=chimera-backend
TRACING_SAMPLE_RATE=1.0
```

**Step 7: Commit**

```bash
git add backend-api/app/core/tracing_config.py backend-api/app/core/observability.py backend-api/tests/test_tracing_config.py backend-api/.env.template
git commit -m "feat: add configurable OpenTelemetry exporter support"
```

---

## Task 4: Update Integration Audit Documentation

**Files:**
- Modify: `backend-api/BACKEND_INTEGRATION_AUDIT.md:1-500`

**Step 1: Add enhancement implementation status**

```markdown
# backend-api/BACKEND_INTEGRATION_AUDIT.md

## Enhancement Implementation Status

### ✅ Implemented Enhancements

**1. Request Correlation IDs**
- **Status**: Implemented
- **Location**: `app/middleware/request_logging.py:15-25`
- **Usage**: Automatic correlation ID generation and propagation
- **Headers**: `X-Request-ID` in all responses

**2. Service Health Dashboard**
- **Status**: Implemented
- **Endpoint**: `GET /api/v1/health/dashboard`
- **Location**: `app/api/v1/endpoints/health_dashboard.py`
- **Returns**: Service status, middleware status, provider availability

**3. Enhanced OpenTelemetry Configuration**
- **Status**: Implemented
- **Location**: `app/core/tracing_config.py`
- **Supported Exporters**: Jaeger, Zipkin, OTLP, Console
- **Configuration**: Environment variables in `.env`

### Configuration Examples

**Jaeger Tracing**:
```bash
TRACING_ENABLED=true
TRACING_EXPORTER=jaeger
TRACING_ENDPOINT=localhost
TRACING_SERVICE_NAME=chimera-backend
```

**Zipkin Tracing**:
```bash
TRACING_ENABLED=true
TRACING_EXPORTER=zipkin
TRACING_ENDPOINT=http://localhost:9411/api/v2/spans
TRACING_SERVICE_NAME=chimera-backend
```

**OTLP (OpenTelemetry Protocol)**:
```bash
TRACING_ENABLED=true
TRACING_EXPORTER=otlp
TRACING_ENDPOINT=http://localhost:4317
TRACING_SERVICE_NAME=chimera-backend
```
```

**Step 2: Commit**

```bash
git add backend-api/BACKEND_INTEGRATION_AUDIT.md
git commit -m "docs: update audit with enhancement implementation status"
```

---

## Task 5: Integration Testing

**Files:**
- Create: `backend-api/tests/integration/test_enhanced_observability.py`

**Step 1: Write integration tests**

```python
# backend-api/tests/integration/test_enhanced_observability.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_correlation_id_end_to_end():
    """Test correlation ID flows through entire request lifecycle"""
    # Make request with custom correlation ID
    custom_id = "test-correlation-123"
    response = client.get(
        "/health",
        headers={"X-Request-ID": custom_id}
    )

    # Verify same ID is returned
    assert response.headers["X-Request-ID"] == custom_id

def test_health_dashboard_integration():
    """Test health dashboard returns complete system status"""
    response = client.get("/api/v1/health/dashboard")
    assert response.status_code == 200

    data = response.json()

    # Verify all sections present
    assert "services" in data
    assert "middleware" in data
    assert "providers" in data

    # Verify service registry integration
    assert "llm_service" in data["services"]
    assert "transformation_engine" in data["services"]

    # Verify provider integration
    assert data["providers"]["count"] > 0

def test_tracing_headers_propagation():
    """Test that tracing headers are properly propagated"""
    response = client.post(
        "/api/v1/generate",
        json={"prompt": "test", "provider": "google"},
        headers={
            "X-API-Key": "test-key",
            "X-Request-ID": "trace-test-456"
        }
    )

    # Verify correlation ID in response
    assert response.headers["X-Request-ID"] == "trace-test-456"
```

**Step 2: Run integration tests**

Run: `cd backend-api && pytest tests/integration/test_enhanced_observability.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add backend-api/tests/integration/test_enhanced_observability.py
git commit -m "test: add integration tests for enhanced observability"
```

---

## Summary

This plan implements three optional enhancements to the already production-ready Chimera backend:

1. **Request Correlation IDs**: Enables distributed tracing across services
2. **Service Health Dashboard**: Provides operational visibility into system status
3. **Enhanced OpenTelemetry**: Configurable tracing exporters for production monitoring

All enhancements are backward-compatible and can be enabled/disabled via environment variables. The existing architecture remains unchanged, with enhancements building on top of the solid foundation already in place.

**Total Implementation Time**: ~30-45 minutes for all tasks
**Testing**: Unit tests + integration tests included
**Documentation**: Audit report updated with implementation status
