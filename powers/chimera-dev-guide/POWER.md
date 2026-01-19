---
name: "chimera-dev-guide"
displayName: "Chimera Development Guide"
description: "Best practices and development guidelines for the Chimera adversarial prompting platform. Covers backend FastAPI patterns, frontend Next.js conventions, and testing strategies."
keywords: ["chimera", "fastapi", "nextjs", "development", "best-practices"]
author: "Chimera Team"
---

# Chimera Development Guide

## Overview

This power provides development best practices and guidelines specifically for the Chimera adversarial prompting platform. It covers backend FastAPI patterns, frontend Next.js conventions, testing strategies, and common workflows to help developers work effectively with the codebase.

Chimera is a full-stack application with a FastAPI backend, Next.js frontend, and sophisticated AI/ML capabilities for adversarial prompt testing and red teaming.

## Core Principles

### 1. Layered Architecture

- **API Layer** → **Services** → **Domain** → **Infrastructure**
- Keep concerns separated and dependencies flowing inward
- Use dependency injection via FastAPI's `Depends()`

### 2. Type Safety Everywhere

- Python: Strict typing enforced outside tests (mypy)
- TypeScript: Strict mode enabled, no `any` types
- Share types between frontend and backend via OpenAPI schema

### 3. Test-Driven Development

- Write tests before or alongside code
- Maintain >80% coverage (enforced via `.coveragerc`)
- Use pytest markers for test organization

## Common Workflows

### Workflow 1: Starting Development

**Goal:** Get the full stack running locally

**Commands:**

```bash
# Install all dependencies
npm run install:all

# Start both backend and frontend
npm run dev

# Or start individually:
npm run dev:backend  # Backend only on port 8001
npm run dev:frontend # Frontend only on port 3000
```

**Verification:**

- Backend: <http://localhost:8001/docs> (Swagger UI)
- Frontend: <http://localhost:3000>
- Health check: `npm run health`

### Workflow 2: Adding a New API Endpoint

**Goal:** Add a new REST endpoint to the backend

**Steps:**

1. Define Pydantic schemas in `backend-api/app/schemas/`
2. Create endpoint in `backend-api/app/api/endpoints/`
3. Register route in `backend-api/app/api/api_routes.py`
4. Write tests in `backend-api/tests/api/`
5. Run tests: `pytest --cov=app`

**Example:**

```python
# schemas/my_feature.py
from pydantic import BaseModel

class MyFeatureRequest(BaseModel):
    input: str

class MyFeatureResponse(BaseModel):
    result: str

# api/endpoints/my_feature.py
from fastapi import APIRouter, Depends
from app.schemas.my_feature import MyFeatureRequest, MyFeatureResponse

router = APIRouter()

@router.post("/my-feature", response_model=MyFeatureResponse)
async def my_feature(request: MyFeatureRequest):
    return MyFeatureResponse(result=f"Processed: {request.input}")
```

### Workflow 3: Running Tests

**Goal:** Run tests with coverage reporting

**Commands:**

```bash
# Backend tests with coverage
cd backend-api
pytest --cov=app --cov=meta_prompter

# Frontend tests
cd frontend
npx vitest --run

# E2E tests
cd frontend
npx playwright test
```

**Test Markers:**

```bash
# Run only security tests
pytest -m security

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Best Practices

### Backend (FastAPI)

- **Use dependency injection** for services and database sessions
- **Implement circuit breakers** for external API calls
- **Add rate limiting** to public endpoints
- **Log structured data** using the observability module
- **Handle errors gracefully** with custom exception handlers

### Frontend (Next.js)

- **Use the API client layer** in `lib/api/core/` for all backend calls
- **Implement TanStack Query** for server state management
- **Follow shadcn/ui patterns** for component composition
- **Use React Hook Form + Zod** for form validation
- **Keep components small** and focused on single responsibility

### Testing

- **Write unit tests** for business logic
- **Write integration tests** for API endpoints
- **Write E2E tests** for critical user flows
- **Mock external dependencies** in tests
- **Use fixtures** for test data setup

### Code Quality

- **Run linters before commit**: `ruff check .` (Python), `npm run lint` (TypeScript)
- **Format code**: `black .` (Python), Prettier via ESLint (TypeScript)
- **Type check**: `mypy app/` (Python), `tsc --noEmit` (TypeScript)
- **Use pre-commit hooks** to enforce quality gates

## Troubleshooting

### Error: "Port already in use"

**Cause:** Another process is using port 8001 or 3000
**Solution:**

```bash
# Check what's using the port
node scripts/check-ports.js

# Kill the process (Windows)
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

### Error: "Module not found"

**Cause:** Dependencies not installed
**Solution:**

```bash
# Reinstall all dependencies
npm run install:all

# Or individually:
cd backend-api && poetry install
cd frontend && npm install
```

### Error: "Database migration failed"

**Cause:** Database schema out of sync
**Solution:**

```bash
cd backend-api
alembic upgrade head
```

## Configuration

### Environment Variables

**Backend** (`.env` in `backend-api/`):

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `GEMINI_API_KEY`: Google Gemini API key
- `OPENAI_API_KEY`: OpenAI API key

**Frontend** (`.env.local` in `frontend/`):

- `NEXT_PUBLIC_API_URL`: Backend API URL (default: <http://localhost:8001>)

### Development vs Production

- **Development**: Use `npm run dev` with hot reload
- **Production**: Use `npm run docker:prod` with Docker Compose

## Additional Resources

- **Architecture**: See `docs/ARCHITECTURE_GAP_ANALYSIS.md`
- **API Documentation**: <http://localhost:8001/docs> (when running)
- **Testing Strategy**: See `backend-api/TESTING_STRATEGY.md`
- **Deployment**: See `PRODUCTION_DEPLOYMENT_GUIDE.md`

---

**Project:** Chimera Adversarial Prompting Platform
**Stack:** FastAPI + Next.js + PostgreSQL + Redis
