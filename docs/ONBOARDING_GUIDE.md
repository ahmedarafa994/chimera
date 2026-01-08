# Chimera Developer Onboarding Guide

Welcome to the Chimera project! This guide will help you get up and running as a new developer.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Key Concepts](#key-concepts)
- [Development Workflow](#development-workflow)
- [Common Tasks](#common-tasks)
- [Resources](#resources)
- [Getting Help](#getting-help)

---

## Introduction

### What is Chimera?

Chimera is an AI-powered prompt optimization and jailbreak research platform. It provides:

- **Multi-provider LLM integration** (OpenAI, Google, Anthropic, DeepSeek)
- **Advanced transformation techniques** (AutoDAN, GCG, Deep Inception, etc.)
- **Real-time optimization** via WebSocket streaming
- **Research tools** for adversarial prompt engineering

### Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11+, FastAPI, Poetry |
| Frontend | Next.js 16, React 19, TypeScript |
| Database | PostgreSQL 14+, SQLite (dev) |
| Cache | Redis 7+ |
| Styling | Tailwind CSS v4 |
| Testing | pytest, Vitest, Playwright |

---

## Project Overview

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │
│   (Next.js)     │◀────│   (FastAPI)     │
│   Port: 3700    │     │   Port: 8001    │
└─────────────────┘     └─────────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │ OpenAI  │ │ Google  │ │Anthropic│
              │   API   │ │   API   │ │   API   │
              └─────────┘ └─────────┘ └─────────┘
```

### Key Components

1. **API Layer** (`backend-api/app/api/`) - REST endpoints and WebSocket handlers
2. **Services** (`backend-api/app/services/`) - Business logic and transformations
3. **Infrastructure** (`backend-api/app/infrastructure/`) - LLM providers, adapters
4. **Frontend** (`frontend/src/`) - React components and pages

---

## Development Setup

### Prerequisites

Install the following tools:

```bash
# Python 3.11+
python --version  # Should be 3.11 or higher

# Node.js 18+
node --version  # Should be 18 or higher

# Poetry (Python package manager)
curl -sSL https://install.python-poetry.org | python3 -

# Git
git --version
```

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/chimera.git
cd chimera
```

### Step 2: Install Dependencies

```bash
# Install all dependencies (backend + frontend)
npm run install:all

# Or install separately:
# Backend
cd backend-api
poetry install

# Frontend
cd ../frontend
npm install
```

### Step 3: Configure Environment

```bash
# Copy environment templates
cp .env.template .env
cp frontend/.env.example frontend/.env.local

# Edit .env with your API keys
# Required: OPENAI_API_KEY
# Recommended: GOOGLE_API_KEY
```

### Step 4: Start Development Servers

```bash
# Start both frontend and backend
npm run dev

# Or start separately:
# Terminal 1 - Backend
npm run dev:backend

# Terminal 2 - Frontend
npm run dev:frontend
```

### Step 5: Verify Installation

```bash
# Check backend health
curl http://localhost:8001/health

# Open frontend
open http://localhost:3700
```

---

## Project Structure

```
chimera/
├── backend-api/              # Backend service
│   ├── app/
│   │   ├── api/             # REST endpoints
│   │   │   └── v1/
│   │   │       └── endpoints/
│   │   ├── core/            # Configuration, utilities
│   │   ├── domain/          # Domain models
│   │   ├── infrastructure/  # External integrations
│   │   ├── middleware/      # Request processing
│   │   └── services/        # Business logic
│   │       ├── autodan/     # AutoDAN technique
│   │       └── transformers/ # Transformation techniques
│   ├── tests/               # Backend tests
│   └── alembic/             # Database migrations
│
├── frontend/                 # Frontend application
│   ├── src/
│   │   ├── app/             # Next.js pages
│   │   ├── components/      # React components
│   │   ├── lib/             # Utilities, API client
│   │   └── hooks/           # Custom React hooks
│   └── public/              # Static assets
│
├── docs/                     # Documentation
├── scripts/                  # Automation scripts
└── tests/                    # Integration tests
```

### Important Files

| File | Purpose |
|------|---------|
| `backend-api/app/main.py` | FastAPI application entry |
| `backend-api/app/core/config.py` | Backend configuration |
| `frontend/src/app/layout.tsx` | Root layout |
| `frontend/src/lib/api-enhanced.ts` | API client |
| `.env` | Environment variables |

---

## Key Concepts

### 1. Transformation Techniques

Chimera uses various techniques to transform prompts:

```python
# Example: Using transformation service
from app.services.transformation_service import transformation_engine

result = await transformation_engine.transform(
    prompt="How to improve security?",
    potency_level=5,
    technique_suite="autodan"
)
```

**Available Techniques:**
- `autodan` - AutoDAN adversarial optimization
- `deep_inception` - Nested context manipulation
- `cipher` - Encoding-based bypass
- `cognitive_hacking` - Psychological framing
- `gradient_injection` - GCG-style optimization

### 2. Potency Levels

Potency (1-10) controls transformation intensity:

| Level | Description |
|-------|-------------|
| 1-3 | Light transformation, subtle changes |
| 4-6 | Moderate transformation, noticeable changes |
| 7-9 | Heavy transformation, significant changes |
| 10 | Maximum transformation, all techniques applied |

### 3. LLM Providers

Chimera supports multiple LLM providers with automatic failover:

```python
# Provider configuration
from app.services.llm_service import llm_service

response = await llm_service.generate(
    prompt="Test prompt",
    provider="openai",  # or "google", "anthropic"
    model="gpt-4o",
    temperature=0.7
)
```

### 4. Circuit Breaker Pattern

The system uses circuit breakers for provider resilience:

```
CLOSED → OPEN → HALF_OPEN → CLOSED
  │        │         │
  │        │         └── Success: Close circuit
  │        └── Failure threshold reached
  └── Normal operation
```

### 5. Caching

Results are cached to improve performance:

```python
# Cache is automatic for transformations
# Clear cache if needed:
transformation_engine.clear_cache()
```

---

## Development Workflow

### Daily Development

1. **Pull latest changes**
   ```bash
   git pull origin main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feat/your-feature
   ```

3. **Make changes and test**
   ```bash
   # Run tests
   poetry run pytest
   npm run lint
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat(module): description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feat/your-feature
   ```

### Running Tests

```bash
# Backend tests
cd backend-api
poetry run pytest

# With coverage
poetry run pytest --cov=app --cov-report=html

# Specific test file
poetry run pytest tests/test_transformation_service.py -v

# Frontend tests
cd frontend
npm test
```

### Code Quality

```bash
# Format Python code
poetry run black .
poetry run ruff check --fix .

# Format TypeScript
npm run lint:fix

# Type checking
poetry run mypy app/
npm run type-check
```

---

## Common Tasks

### Adding a New API Endpoint

1. Create endpoint file:
   ```python
   # backend-api/app/api/v1/endpoints/new_feature.py
   from fastapi import APIRouter
   
   router = APIRouter()
   
   @router.get("/new-feature")
   async def get_new_feature():
       return {"message": "Hello"}
   ```

2. Register in router:
   ```python
   # backend-api/app/api/v1/router.py
   from app.api.v1.endpoints import new_feature
   
   api_router.include_router(
       new_feature.router,
       prefix="/new-feature",
       tags=["new-feature"]
   )
   ```

### Adding a New Transformation Technique

1. Create transformer:
   ```python
   # backend-api/app/services/transformers/my_technique.py
   from app.services.transformers.base import BaseTransformer
   
   class MyTechniqueTransformer(BaseTransformer):
       def transform(self, context):
           # Your transformation logic
           return transformed_prompt
   ```

2. Register in engine:
   ```python
   # backend-api/app/services/transformation_service.py
   self.my_technique = MyTechniqueTransformer()
   ```

### Adding a New React Component

1. Create component:
   ```typescript
   // frontend/src/components/MyComponent.tsx
   interface MyComponentProps {
     title: string;
   }
   
   export function MyComponent({ title }: MyComponentProps) {
     return <div>{title}</div>;
   }
   ```

2. Add tests:
   ```typescript
   // frontend/src/components/__tests__/MyComponent.test.tsx
   import { render } from '@testing-library/react';
   import { MyComponent } from '../MyComponent';
   
   test('renders title', () => {
     const { getByText } = render(<MyComponent title="Test" />);
     expect(getByText('Test')).toBeInTheDocument();
   });
   ```

### Debugging Tips

1. **Backend logging**:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.debug("Debug message")
   ```

2. **Frontend debugging**:
   ```typescript
   console.log('Debug:', variable);
   // Or use React DevTools
   ```

3. **API debugging**:
   - Use FastAPI Swagger UI: http://localhost:8001/docs
   - Use browser DevTools Network tab

---

## Resources

### Documentation

| Document | Location |
|----------|----------|
| API Reference | http://localhost:8001/docs |
| Coding Standards | `docs/CODING_STANDARDS.md` |
| Production Guide | `docs/PRODUCTION_DEPLOYMENT_GUIDE.md` |
| Contributing Guide | `CONTRIBUTING.md` |
| Architecture | `docs/ARCHITECTURE_REVIEW_SUMMARY.md` |

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [pytest Documentation](https://docs.pytest.org/)

### Code Examples

Look at existing implementations for patterns:

| Pattern | Example File |
|---------|--------------|
| API Endpoint | `backend-api/app/api/v1/endpoints/jailbreak.py` |
| Service | `backend-api/app/services/transformation_service.py` |
| Transformer | `backend-api/app/services/transformers/persona.py` |
| React Component | `frontend/src/components/jailbreak/JailbreakGenerator.tsx` |
| React Hook | `frontend/src/hooks/use-api.ts` |

---

## Getting Help

### Communication Channels

- **Code Questions**: Create a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@chimera.example.com

### Mentorship

New developers are paired with a mentor for the first 2 weeks. Your mentor can help with:

- Code reviews
- Architecture questions
- Best practices guidance
- Project conventions

### FAQ

**Q: Why isn't my environment working?**
A: Check:
1. Python version (3.11+)
2. Node version (18+)
3. API keys in `.env`
4. Dependencies installed (`npm run install:all`)

**Q: How do I add a new LLM provider?**
A: See `backend-api/app/infrastructure/providers/` for examples. Implement the `LLMProvider` interface.

**Q: Where are the API types defined?**
A: Backend: `backend-api/app/domain/models.py`
   Frontend: `frontend/src/types/`

**Q: How do I run a single test?**
A: `poetry run pytest tests/test_file.py::test_function -v`

---

## First Week Checklist

- [ ] Complete development setup
- [ ] Read `CONTRIBUTING.md`
- [ ] Read `CODING_STANDARDS.md`
- [ ] Explore the codebase structure
- [ ] Run all tests successfully
- [ ] Make a small contribution (typo fix, doc improvement)
- [ ] Set up IDE with linters/formatters
- [ ] Meet with your mentor
- [ ] Review recent PRs to understand patterns

---

**Welcome to the team! 🎉**

If you have any questions, don't hesitate to ask. We're here to help you succeed.

---

**Last Updated**: January 2026
**Version**: 1.0.0
