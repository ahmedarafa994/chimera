---
name: Backend API Testing
description: Expert skill for testing and debugging the FastAPI backend. Use when working with API endpoints, database operations, authentication, or middleware troubleshooting.
---

# Backend API Testing Skill

## Overview

This skill provides expertise in testing, debugging, and developing the Chimera FastAPI backend, including API endpoints, database operations, authentication flows, and middleware configuration.

## When to Use This Skill

- Testing API endpoints (CRUD operations, transformations, generation)
- Debugging authentication/authorization issues
- Investigating database connection or query problems
- Troubleshooting middleware (rate limiting, CORS, security headers)
- Running pytest test suites
- Analyzing backend logs and error traces

## Technology Stack

### Core Framework

- **FastAPI 0.104+**: Modern async Python web framework
- **Pydantic V2**: Data validation with new `pattern`, `min_length` syntax
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Default development database (with `check_same_thread` considerations)

### Key Dependencies

```python
# From pyproject.toml
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.5.0"
sqlalchemy = "^2.0.23"
httpx = "^0.25.0"  # For async HTTP client
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
```

## Project Structure

```
backend-api/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── api.py              # Router registration
│   │       └── endpoints/          # API route handlers
│   │           ├── auth.py         # Authentication
│   │           ├── generate.py     # Content generation
│   │           ├── transform.py    # Prompt transformations
│   │           ├── aegis.py        # Aegis campaigns
│   │           └── aegis_ws.py     # WebSocket telemetry
│   ├── core/
│   │   ├── config.py               # Configuration management
│   │   ├── security.py             # API keys, CORS, headers
│   │   └── database.py             # DB session management
│   ├── middleware/
│   │   ├── rate_limit.py           # Rate limiting
│   │   ├── selection.py            # Model/provider selection
│   │   └── validation.py           # Input sanitization
│   ├── models/                     # SQLAlchemy models
│   ├── schemas/                    # Pydantic schemas
│   └── main.py                     # FastAPI application entry
├── tests/                          # Test suites
└── logs/                           # Application logs
```

## Common Testing Commands

### Run All Tests with Coverage

```bash
cd backend-api
poetry run pytest --cov=app --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

### Run Specific Test Files

```bash
# Authentication tests
poetry run pytest tests/test_auth.py -v

# API endpoint tests
poetry run pytest tests/test_api_endpoints.py -v

# Security tests
poetry run pytest tests/test_deepteam_security.py -m "security or owasp" -v
```

### Run Backend Server (Development)

```bash
# From project root
npm run dev:backend

# Or directly with uvicorn
cd backend-api
poetry run uvicorn app.main:app --reload --port 8001 --log-level debug
```

### Health Checks

```bash
# Check backend health
curl http://localhost:8001/health

# View API documentation
open http://localhost:8001/docs  # Swagger UI
open http://localhost:8001/redoc # ReDoc
```

## Common Issues and Solutions

### 1. Login Endpoint Hangs (httpx.ReadTimeout)

**Symptom**: `/api/v1/auth/login` returns 504 timeout
**Root Cause**: Database write lock in SQLite, often from `SelectionMiddleware`

**Solutions**:

```python
# Option A: Configure SQLite for web apps
# In app/core/database.py
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Allow multi-threaded access
    poolclass=StaticPool  # Use single connection pool
)

# Option B: Exclude auth routes from blocking middleware
# In app/middleware/selection.py
if request.url.path.startswith("/api/v1/auth"):
    return await call_next(request)  # Skip middleware
```

**Debugging**:

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
poetry run uvicorn app.main:app --reload --log-level debug

# Check for DB locks
# Add logging in middleware before/after DB operations
```

### 2. Pydantic V2 Migration Errors

**Symptom**: `AttributeError: 'BaseModel' has no attribute 'regex'` or `'min_items'`

**Root Cause**: Deprecated Pydantic V1 syntax used in V2

**Fixes**:

```python
# BEFORE (Pydantic V1)
from pydantic import BaseModel

class MySchema(BaseModel):
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    tags: List[str] = Field(..., min_items=1)

# AFTER (Pydantic V2)
from pydantic import BaseModel, Field

class MySchema(BaseModel):
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    tags: List[str] = Field(..., min_length=1)
```

### 3. ImportError for API_KEY_NAME_MAP

**Symptom**: `ImportError: cannot import name 'API_KEY_NAME_MAP' from 'app.core.config'`

**Root Cause**: Missing or incorrectly named constant in `config.py`

**Fix**:

```python
# In app/core/config.py
API_KEY_NAME_MAP = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY"
}
```

### 4. 404 Errors for New Endpoints

**Symptom**: `404 Not Found` for newly created endpoints

**Root Cause**: Router not registered in `api.py`

**Fix**:

```python
# In app/api/v1/api.py
from app.api.v1.endpoints import (
    auth, generate, transform, aegis, aegis_ws, sessions, techniques
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(aegis.router, prefix="/aegis", tags=["aegis"])
api_router.include_router(aegis_ws.router, prefix="/ws/aegis", tags=["websockets"])
# ... add all routers
```

### 5. CORS Errors from Frontend

**Symptom**: Browser console shows CORS policy errors

**Fix**:

```python
# In app/main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing Best Practices

### 1. Use Fixtures for Database Setup

```python
# In conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine)
    yield TestingSessionLocal()
```

### 2. Mock External API Calls

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_generate_endpoint():
    with patch("app.services.llm.call_api", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"content": "Test response"}
        # Test your endpoint
```

### 3. Test Authentication Flows

```python
def test_login_success(client):
    response = client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "secure_password"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()
```

### 4. Test Rate Limiting

```python
def test_rate_limit_exceeded(client):
    for _ in range(101):  # Assuming 100 req/min limit
        response = client.get("/api/v1/generate")
    assert response.status_code == 429  # Too Many Requests
```

## Debugging Techniques

### 1. Enable Debug Logging

```python
# In app/main.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### 2. Add Request/Response Logging Middleware

```python
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        logger.debug(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.debug(f"Response: {response.status_code}")
        return response
```

### 3. Use Pytest Verbose Flags

```bash
# Detailed test output
poetry run pytest -vv --tb=short

# Show print statements
poetry run pytest -s

# Stop on first failure
poetry run pytest -x
```

### 4. Inspect Database State

```python
# In tests, use fixtures to inspect DB
def test_campaign_created(test_db):
    # Create campaign via API
    # ...
    campaign = test_db.query(Campaign).first()
    assert campaign is not None
    assert campaign.status == "pending"
```

## Performance Optimization

### 1. Use Async Database Queries

```python
from sqlalchemy.ext.asyncio import AsyncSession

async def get_campaigns(db: AsyncSession):
    result = await db.execute(select(Campaign))
    return result.scalars().all()
```

### 2. Implement Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model_config(provider: str, model: str):
    # Expensive operation
    pass
```

### 3. Connection Pooling

```python
# In app/core/database.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True  # Verify connections before use
)
```

## OpenAPI Schema Validation

```bash
# Generate OpenAPI schema
cd backend-api
poetry run python -c "from app.main import app; import json; print(json.dumps(app.openapi(), indent=2))" > openapi.json

# Validate schema
npx @apidevtools/swagger-cli validate openapi.json
```

## References

- [backend-api/README.md](../../backend-api/README.md): Backend-specific documentation
- [docs/DEVELOPER_GUIDE.md](../../docs/DEVELOPER_GUIDE.md): Development guidelines
- [docs/openapi.yaml](../../docs/openapi.yaml): OpenAPI specification
- [.env.template](../../.env.template): Environment variable reference
