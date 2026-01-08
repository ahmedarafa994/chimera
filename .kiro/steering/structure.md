# Project Structure

## Root Organization

```
chimera/
├── backend-api/          # FastAPI backend service
├── frontend/             # Next.js frontend application
├── meta_prompter/        # Shared prompt enhancement library
├── chimera-orchestrator/ # Multi-agent orchestration system
├── autodan-turbo/        # AutoDAN-Turbo lifelong learning implementation
├── docs/                 # Project documentation
├── scripts/              # Automation scripts (health checks, port validation)
├── tests/                # Top-level integration and scenario tests
├── logs/                 # Runtime logs and attack logs
└── monitoring/           # Prometheus, Grafana, Loki configurations
```

## Backend Structure (`backend-api/app/`)

```
app/
├── api/                  # API layer
│   ├── endpoints/        # Individual endpoint modules
│   ├── routes/           # Route groupings
│   └── api_routes.py     # Main router aggregation
├── core/                 # Core infrastructure
│   ├── config.py         # Configuration management
│   ├── dependencies.py   # Dependency injection
│   ├── errors.py         # Error handling
│   ├── health.py         # Health check system
│   ├── lifespan.py       # Application lifecycle
│   ├── observability.py  # Logging, tracing, metrics
│   └── provider_*.py     # LLM provider management
├── services/             # Business logic layer
│   ├── autodan/          # AutoDAN implementation
│   ├── gptfuzz/          # GPTFuzz implementation
│   ├── jailbreak/        # Jailbreak techniques
│   ├── llm_adapters/     # LLM provider adapters
│   └── transformers/     # Prompt transformation services
├── engines/              # Adversarial engines
│   ├── autodan_engine.py # AutoDAN core engine
│   ├── transformer_engine.py # Transformation engine
│   └── autoadv/          # AutoAdv implementation
├── domain/               # Domain models and interfaces
│   ├── models.py         # Core domain models
│   ├── interfaces.py     # Service interfaces
│   └── jailbreak/        # Jailbreak domain logic
├── infrastructure/       # External integrations
│   ├── gemini_client.py  # Google Gemini client
│   ├── deepseek_client.py # DeepSeek client
│   └── redis_*.py        # Redis integrations
├── middleware/           # Request/response middleware
│   ├── auth.py           # Authentication
│   ├── rate_limit.py     # Rate limiting
│   └── request_logging.py # Request logging
├── schemas/              # Pydantic request/response schemas
├── models/               # SQLAlchemy database models
├── repositories/         # Data access layer
├── config/               # Configuration files (YAML)
├── data/                 # Static data (technique definitions)
└── main.py               # Application entry point
```

## Frontend Structure (`frontend/src/`)

```
src/
├── app/                  # Next.js App Router pages
│   ├── dashboard/        # Dashboard pages
│   ├── api/              # API routes (if any)
│   └── layout.tsx        # Root layout
├── components/           # React components
│   ├── ui/               # shadcn/ui primitives
│   ├── dashboard/        # Dashboard-specific components
│   └── shared/           # Shared components
├── lib/                  # Utility libraries
│   ├── api/              # API client layer
│   │   ├── core/         # Core API client (client.ts, config.ts, errors.ts)
│   │   ├── testing/      # Mock server and test utilities
│   │   └── migration/    # Backward compatibility layer
│   ├── utils.ts          # General utilities
│   └── hooks/            # Custom React hooks
├── styles/               # Global styles
└── types/                # TypeScript type definitions
```

## Key Architectural Patterns

### Backend Patterns

- **Layered Architecture**: API → Services → Domain → Infrastructure
- **Dependency Injection**: FastAPI's `Depends()` for service injection
- **Repository Pattern**: Data access abstraction in `repositories/`
- **Circuit Breaker**: Resilience pattern in `core/circuit_breaker.py`
- **Provider Registry**: Dynamic LLM provider registration in `core/provider_registry.py`
- **Middleware Stack**: Authentication → Rate Limiting → Logging → CORS
- **Health Checks**: Comprehensive health check system with dependency graphs

### Frontend Patterns

- **API Client Layer**: Unified client with retry, caching, circuit breaker in `lib/api/core/`
- **Component Composition**: Radix UI primitives wrapped with shadcn/ui styling
- **Server State**: TanStack Query for API data fetching and caching
- **Form Validation**: React Hook Form + Zod schemas
- **Type Safety**: Strict TypeScript with shared types from backend OpenAPI schema

## Configuration Locations

- **Backend Config**: `backend-api/app/core/config.py` (Python), `backend-api/.env`
- **Frontend Config**: `frontend/src/lib/api/core/config.ts`, `frontend/.env.local`
- **Technique Definitions**: `backend-api/data/jailbreak/techniques/*.yaml`
- **Database Migrations**: `backend-api/alembic/versions/`
- **Docker Compose**: `docker-compose.yml` (dev), `docker-compose.prod.yml` (prod)

## Testing Structure

- **Backend Unit Tests**: `backend-api/tests/` (mirrors `app/` structure)
- **Frontend Tests**: `frontend/src/**/*.test.tsx` (co-located with components)
- **Integration Tests**: `tests/` (root level, cross-service scenarios)
- **E2E Tests**: `frontend/` with Playwright (`playwright.config.ts`)

## Documentation Locations

- **API Docs**: `backend-api/API_INTEGRATION_GUIDE.md`, `API_INTEGRATION_ARCHITECTURE.md`
- **Architecture**: `docs/ARCHITECTURE_GAP_ANALYSIS.md`, `COMPREHENSIVE_PROJECT_REVIEW.md`
- **Development**: `docs/DEVELOPMENT_SETUP.md`, `docs/DEPLOYMENT_CHECKLIST.md`
- **Security**: `SECURITY_AUDIT_REPORT.md`, `PHASE_0_SECURITY_COMPLETION_REPORT.md`

## Naming Conventions

- **Python Modules**: `snake_case` (e.g., `llm_service.py`)
- **Python Classes**: `PascalCase` (e.g., `LLMService`, `PromptEnhancer`)
- **Python Functions**: `snake_case` (e.g., `generate_text()`)
- **TypeScript Files**: `kebab-case` (e.g., `api-client.ts`)
- **React Components**: `PascalCase.tsx` (e.g., `DashboardLayout.tsx`)
- **TypeScript Functions**: `camelCase` (e.g., `generateText()`)
- **API Endpoints**: `/api/v1/resource-name` (kebab-case, versioned)
