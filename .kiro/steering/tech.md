# Technology Stack

## Backend

- **Framework**: FastAPI 0.115+ with Uvicorn ASGI server
- **Language**: Python 3.11+ (strict typing enforced outside tests)
- **Database**: SQLAlchemy 2.0+ with Alembic migrations, PostgreSQL support
- **Caching**: Redis 5.0+ for rate limiting and execution tracking
- **AI/ML**: 
  - PyTorch 2.0+ for gradient-based optimization
  - Transformers 4.30+ for model loading
  - Google Generative AI SDK, OpenAI SDK, Anthropic SDK
- **Testing**: pytest 8.0+ with pytest-asyncio, httpx for API testing
- **Code Quality**: Ruff (linting), Black (formatting), mypy (type checking)

## Frontend

- **Framework**: Next.js 16.0+ with React 19.2+
- **Language**: TypeScript 5.7+
- **Styling**: Tailwind CSS 3.4+ with tailwindcss-animate
- **UI Components**: Radix UI primitives, shadcn/ui component library
- **State Management**: TanStack Query 5.90+ for server state
- **Forms**: React Hook Form 7.67+ with Zod 4.1+ validation
- **HTTP Client**: Axios 1.13+ with custom API client layer

## Development Tools

- **Package Management**: 
  - Python: Poetry (preferred) or pip with requirements.txt
  - Node: npm with package-lock.json
- **Monorepo**: Root package.json with concurrently for multi-service orchestration
- **Docker**: docker-compose.yml for development, docker-compose.prod.yml for production
- **Pre-commit**: Hooks for linting and formatting enforcement

## Common Commands

### Full Stack Development

```bash
# Install all dependencies (root, frontend, backend)
npm run install:all

# Start both backend and frontend in development mode
npm run dev

# Start production-like environment
npm start

# Build frontend for production
npm run build:frontend

# Check port availability
npm run check:ports

# Health check all services
npm run health
```

### Backend Only

```bash
cd backend-api

# Start development server (hot reload)
python run.py
# or
py run.py

# Run tests with coverage
pytest --cov=app --cov=meta_prompter

# Run specific test markers
pytest -m security
pytest -m integration

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Linting and formatting
ruff check .
black .
mypy app/
```

### Frontend Only

```bash
cd frontend

# Start development server with Turbopack
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Linting
npm run lint

# Run tests
npx vitest --run

# Run Playwright e2e tests
npx playwright test
```

### Docker

```bash
# Development environment
npm run docker:dev
# or
docker-compose up -d

# Production environment
npm run docker:prod
# or
docker-compose -f docker-compose.prod.yml up -d

# Build production images
npm run docker:build:prod
```

## Configuration Files

- **Python**: `pyproject.toml` (Poetry config, tool settings), `requirements.txt` (pip fallback)
- **Frontend**: `package.json`, `tsconfig.json`, `next.config.ts`, `tailwind.config.ts`
- **Testing**: `pytest.ini`, `.coveragerc`, `playwright.config.ts`
- **Code Quality**: `.pre-commit-config.yaml`, `ruff.toml` (via pyproject.toml)
- **Environment**: `.env.example` files in root, backend-api/, and frontend/

## API Documentation

- **Swagger UI**: http://localhost:8001/docs (development only)
- **ReDoc**: http://localhost:8001/redoc (development only)
- **OpenAPI Schema**: http://localhost:8001/openapi.json
