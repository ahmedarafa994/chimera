
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Chimera** is an AI-powered prompt optimization and jailbreak research system with a FastAPI backend and Next.js frontend. The system provides advanced prompt transformation techniques, multi-provider LLM integration, and real-time enhancement capabilities for security research and prompt engineering.

## Architecture

### Backend (FastAPI - Python 3.11+)
- **Location**: `backend-api/`
- **Entry Point**: `backend-api/run.py` or `backend-api/app/main.py`
- **Port**: 8001 (configurable via `PORT` environment variable)

**Key Components**:
- `app/main.py` - FastAPI application with middleware stack, WebSocket support, health checks
- `app/api/api_routes.py` - Main API router aggregating all v1 endpoints
- `app/services/llm_service.py` - Multi-provider LLM orchestration (Google, OpenAI, Anthropic, DeepSeek)
- `app/services/transformation_service.py` - Prompt transformation engine with 20+ technique suites
- `app/services/autodan/service.py` - AutoDAN adversarial prompt generation service
- `app/services/gptfuzz/service.py` - GPTFuzz mutation-based jailbreak testing
- `app/services/data_pipeline/` - **Data pipeline for LLM analytics and research tracking**
- `app/core/config.py` - Centralized configuration with proxy/direct mode support
- `meta_prompter/prompt_enhancer.py` - Comprehensive prompt enhancement system
- `meta_prompter/jailbreak_enhancer.py` - Jailbreak technique application

**Provider Architecture**:
- Supports both **proxy mode** (via AIClient-2-API Server on localhost:8080) and **direct mode** (native API calls)
- Connection mode configured via `API_CONNECTION_MODE` environment variable
- Provider endpoints: Google/Gemini, OpenAI, Anthropic/Claude, Qwen, DeepSeek, Cursor

### Frontend (Next.js 16 - React 19)
- **Location**: `frontend/`
- **Port**: 3000 (default Next.js dev server)
- **Tech Stack**: Next.js 16, React 19, TypeScript, Tailwind CSS 3, shadcn/ui, TanStack Query, Vitest

**Key Files**:
- `src/app/dashboard/` - Dashboard pages (generation, jailbreak, providers, health)
- `src/lib/api-enhanced.ts` - Enhanced API client with circuit breaker and retry logic
- `src/lib/api-config.ts` - Centralized API URL configuration
- `src/components/` - Reusable UI components

## Development Commands

### Backend Development

```bash
# Navigate to backend directory
cd backend-api

# Install dependencies (Poetry or pip)
poetry install
# OR
pip install -r requirements.txt

# Install spaCy model (required)
python -m spacy download en_core_web_sm

# Run development server (with auto-reload)
python run.py
# OR
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Run tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run specific test markers
pytest -m "unit"
pytest -m "integration"
pytest -m "security"
pytest -m "not slow"

# Run DeepTeam security tests
pytest tests/test_deepteam_security.py -m "security or owasp" -v

# Run with coverage
pytest --cov=app --cov-report=html

# Linting and formatting
ruff check .
ruff format .
black .
```

### Frontend Development

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Run linter
npm run lint

# Run tests (Vitest)
npm run test
```

### Full Stack Development

```bash
# From project root - runs both backend and frontend concurrently
npm run dev
# OR
npm start        # Same as npm run dev

# Or start individually
npm run backend  # Starts backend only (uses py run.py)
npm run frontend # Starts frontend only

# Alternative specific commands
npm run start:backend   # Starts backend
npm run start:frontend  # Starts frontend
npm run dev:backend     # Starts backend in dev mode
npm run dev:frontend    # Starts frontend in dev mode

# Install all dependencies (root, frontend, backend)
npm run install:all

# Build project
npm run build           # Builds frontend only
npm run build:frontend  # Builds frontend

# Port checking and health monitoring
npm run check:ports     # Check if ports are available
npm run health          # Health check
npm run wait-for-services  # Wait for services to start

# Docker development
npm run docker:dev
npm run docker:prod
npm run docker:build:prod
```

## Environment Configuration

Copy `.env.template` to `.env` and configure:

**Required Variables**:
```bash
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Server
PORT=8001

# Security (CHANGE IN PRODUCTION)
JWT_SECRET=<generate-secure-secret>
API_KEY=<generate-secure-api-key>
CHIMERA_API_KEY=<generate-secure-api-key>

# AI Provider API Keys (at least one required)
GOOGLE_API_KEY=<your-google-api-key>
OPENAI_API_KEY=<your-openai-api-key>
ANTHROPIC_API_KEY=<your-anthropic-api-key>

# Connection Mode
API_CONNECTION_MODE=direct  # or "proxy"

# Model Selection (optional)
OPENAI_MODEL=gpt-4          # or gpt-4-turbo, gpt-3.5-turbo
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # or claude-3-opus-20240229
GOOGLE_MODEL=gemini-1.5-pro # or gemini-pro, gemini-1.5-flash

# Redis (optional, for distributed caching)
REDIS_URL=redis://localhost:6379/0
```

**DeepTeam Security Testing**:
For security testing with DeepTeam, additional setup may be required:
```bash
# Install DeepTeam (if running full security tests)
pip install deepteam

# Run security tests
pytest tests/test_deepteam_security.py -m "security or owasp" -v
```

## API Architecture

### Main Endpoints

**Generation & Transformation**:
- `POST /api/v1/generate` - Generate text with LLM providers
- `POST /api/v1/transform` - Transform prompt without execution
- `POST /api/v1/execute` - Transform and execute prompt
- `POST /api/v1/generation/jailbreak/generate` - AI-powered jailbreak generation

**Advanced Jailbreak Research**:
- `POST /api/v1/autodan/*` - AutoDAN adversarial prompt optimization endpoints
- `POST /api/v1/gptfuzz/*` - GPTFuzz mutation-based testing endpoints
- `POST /api/v1/evasion/*` - Evasion technique endpoints
- `POST /api/v1/strategies/*` - Strategy management endpoints

**Provider Management**:
- `GET /api/v1/providers` - List available LLM providers and models
- `GET /api/v1/session/models` - Get available models for current session

**Health & Monitoring**:
- `GET /health` - Basic liveness check
- `GET /health/ready` - Readiness probe with dependency checks
- `GET /health/full` - Comprehensive health check
- `GET /health/integration` - Service dependency graph
- `GET /integration/stats` - Integration service statistics

**WebSocket**:
- `WS /ws/enhance` - Real-time prompt enhancement with heartbeat

## Data Pipeline Architecture

Chimera includes a production-grade data pipeline for LLM analytics, jailbreak research tracking, and compliance requirements.

**Documentation**:
- Architecture: `docs/DATA_PIPELINE_ARCHITECTURE.md`
- Deployment: `docs/PIPELINE_DEPLOYMENT_GUIDE.md`
- Summary: `docs/PIPELINE_IMPLEMENTATION_SUMMARY.md`

**Components**:
- **Batch Ingestion** (`app/services/data_pipeline/batch_ingestion.py`)
  - Hourly ETL processing with watermark tracking
  - Schema validation and dead letter queue
  - Parquet output with date/hour partitioning

- **Delta Lake Storage** (`app/services/data_pipeline/delta_lake_manager.py`)
  - ACID transactions for data consistency
  - Time travel queries for historical analysis
  - Z-order clustering and file optimization

- **Data Quality** (`app/services/data_pipeline/data_quality.py`)
  - Great Expectations validation suites
  - Automated quality checks (99%+ pass rate)
  - Alert generation for quality failures

**Orchestration**:
- **Airflow DAG** (`airflow/dags/chimera_etl_hourly.py`)
  - Hourly ETL schedule (SLA: 10 minutes)
  - Parallel extraction → Validation → dbt → Optimization
  - Automatic retries with exponential backoff

**Transformation**:
- **dbt Models** (`dbt/chimera/models/`)
  - Staging: Deduplication and validation
  - Marts: Analytics-ready dimensional models
  - Aggregations: Pre-computed provider metrics

**Monitoring**:
- **Prometheus Alerts** (`monitoring/prometheus/alerts/data_pipeline.yml`)
  - Pipeline health and SLA violations
  - Data freshness and quality checks
  - Performance and cost anomalies

**Quick Start**:
```bash
# Install dependencies
pip install -r backend-api/requirements-pipeline.txt

# Initialize Airflow
export AIRFLOW_HOME=/opt/airflow
airflow db init

# Setup dbt
cd dbt/chimera && dbt deps

# Run pipeline manually
airflow dags trigger chimera_etl_hourly
```

### Request/Response Models

Key Pydantic models in `app/domain/models.py`:
- `PromptRequest` - LLM generation request with config
- `PromptResponse` - LLM generation response with usage metadata
- `TransformationRequest` - Prompt transformation parameters
- `ExecutionRequest` - Combined transformation + execution
- `GenerationConfig` - LLM generation parameters (temperature, top_p, max_tokens)

## Testing Strategy

**Test Structure**:
- `backend-api/tests/` - Backend test suite
- `backend-api/tests/conftest.py` - Shared fixtures and test configuration
- Tests use `pytest` with `pytest-asyncio` for async support

**Running Tests**:
```bash
cd backend-api

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test markers
pytest -m "not slow"
pytest -m integration

# Run with coverage report
pytest --cov=app --cov-report=html
```

**Test Configuration** (`backend-api/pytest.ini`):
- Async mode: auto
- Timeout: 30s per test
- Markers: unit, integration, security, e2e, slow
- Coverage: 60% minimum threshold
- Coverage reports: terminal and HTML (coverage_html/)

## Key Implementation Patterns

### LLM Provider Integration

Providers implement `LLMProvider` interface (`app/domain/interfaces.py`):
```python
class LLMProvider(Protocol):
    async def generate(self, request: PromptRequest) -> PromptResponse:
        ...
```

Register providers in `app/core/lifespan.py`:
```python
from app.services.llm_service import llm_service
from app.infrastructure.providers.google_provider import GoogleProvider

llm_service.register_provider("google", GoogleProvider(), is_default=True)
```

### Circuit Breaker Pattern

LLM calls are protected by circuit breakers (`app/core/circuit_breaker.py`):
- Failure threshold: 3 consecutive failures
- Recovery timeout: 60 seconds
- Automatic fallback to alternative providers

### Prompt Enhancement Pipeline

The `PromptEnhancer` class (`meta_prompter/prompt_enhancer.py`) provides:
1. **Intent Analysis** - Detect user intent and categorize prompts
2. **Context Expansion** - Add domain knowledge and frameworks
3. **Virality Optimization** - Inject power words and emotional hooks
4. **SEO Optimization** - Generate keywords and meta descriptions
5. **Structure Optimization** - Hierarchical prompt structuring

Usage:
```python
from meta_prompter.prompt_enhancer import PromptEnhancer

enhancer = PromptEnhancer()
result = enhancer.enhance("Create a viral social media post")
enhanced_prompt = result["enhanced_prompt"]
```

### Transformation Techniques

The `TransformationEngine` (`app/services/transformation_service.py`) supports 20+ technique suites:
- **Basic**: simple, advanced, expert
- **Cognitive**: cognitive_hacking, hypothetical_scenario
- **Obfuscation**: advanced_obfuscation, typoglycemia
- **Persona**: hierarchical_persona, dan_persona
- **Context**: contextual_inception, nested_context
- **Logic**: logical_inference, conditional_logic
- **Multimodal**: multimodal_jailbreak, visual_context
- **Agentic**: agentic_exploitation, multi_agent
- **Payload**: payload_splitting, instruction_fragmentation
- **Advanced**: quantum_exploit, deep_inception, code_chameleon, cipher

### Jailbreak Techniques

Jailbreak transformations in `meta_prompter/jailbreak_enhancer.py`:
- Role hijacking
- Instruction injection
- Obfuscation (leet speak, homoglyphs, Caesar cipher)
- Neural bypass techniques
- Multilingual trojans
- Payload splitting

### Advanced Jailbreak Frameworks

**AutoDAN Service** (`app/services/autodan/`):
- Adversarial prompt optimization using genetic algorithms
- Multiple attack methods: vanilla, best-of-n, beam search
- Integrates with Chimera's LLM infrastructure via `ChimeraLLMAdapter`
- Supports reasoning models and hierarchical search strategies
- Configuration in `autodan/config.py` and `autodan/config_enhanced.py`

**GPTFuzz Service** (`app/services/gptfuzz/`):
- Mutation-based jailbreak testing framework
- Mutators: CrossOver, Expand, GenerateSimilar, Rephrase, Shorten
- MCTS-based selection policy for exploration
- Session-based testing with configurable parameters
- Components in `gptfuzz/components.py`

## Security Considerations

**Authentication**:
- API Key authentication via `X-API-Key` header
- JWT token support (configured via `JWT_SECRET`)
- Middleware: `app/middleware/auth.py`

**Input Validation**:
- Pydantic models validate all inputs
- Dangerous pattern detection in prompts
- Rate limiting via `app/core/rate_limit.py`
- CSRF protection via `app/core/middleware.py`

**Security Headers**:
- XSS protection
- Clickjacking prevention
- Content Security Policy
- Implemented in `SecurityHeadersMiddleware`

## Troubleshooting

**Port Conflicts**:
- Backend default: 8001
- Frontend default: 3000
- Check with: `netstat -ano | findstr :8001` (Windows) or `lsof -i :8001` (Unix)

**Missing Dependencies**:
- Backend: Ensure spaCy model installed (`python -m spacy download en_core_web_sm`)
- Frontend: Run `npm install` in frontend directory

**API Connection Issues**:
- Verify `API_CONNECTION_MODE` in `.env`
- Check provider API keys are set
- Review logs in `backend-api/logs/`

**Test Failures**:
- Ensure Redis is running if using distributed features
- Check API keys are configured for provider tests
- Review test output for specific error messages

## Project Structure Reference

```
chimera/
├── backend-api/           # FastAPI backend
│   ├── app/
│   │   ├── api/          # API routes and endpoints
│   │   │   └── v1/endpoints/  # Versioned API endpoints
│   │   ├── core/         # Core utilities (config, auth, logging)
│   │   ├── domain/       # Domain models and interfaces
│   │   ├── engines/      # Advanced transformation engines
│   │   ├── infrastructure/ # Provider implementations
│   │   ├── middleware/   # Custom middleware
│   │   ├── services/     # Business logic services
│   │   │   ├── autodan/  # AutoDAN adversarial framework
│   │   │   ├── gptfuzz/  # GPTFuzz mutation testing
│   │   │   ├── jailbreak/ # Jailbreak services
│   │   │   └── transformers/ # Transformation techniques
│   │   └── main.py       # FastAPI application
│   ├── tests/            # Test suite
│   ├── run.py            # Development entry point with port detection
│   ├── pytest.ini        # Test configuration
│   └── requirements.txt  # Python dependencies
├── frontend/             # Next.js frontend
│   ├── src/
│   │   ├── app/         # Next.js app router pages
│   │   ├── components/  # React components
│   │   ├── lib/         # Utilities and API client
│   │   └── types/       # TypeScript types
│   └── package.json     # Node dependencies
├── meta_prompter/       # Prompt enhancement library
│   ├── prompt_enhancer.py
│   └── jailbreak_enhancer.py
├── .env.template        # Environment configuration template
├── pyproject.toml       # Python project configuration
└── package.json         # Root package for concurrent dev
```

## Important Implementation Notes

**Port Configuration**:
- Backend uses centralized port configuration in `app/core/port_config.py`
- `run.py` includes automatic port conflict detection and resolution
- In development mode, automatically finds alternative ports if 8001 is occupied

**AutoDAN Integration**:
- Uses `ChimeraLLMAdapter` to integrate with Chimera's multi-provider LLM service
- Supports multiple optimization methods: vanilla, best_of_n, beam_search, mousetrap
- Configuration via `autodan/config.py` with retry strategies and model selection
- **Mousetrap Technique**: Advanced Chain of Iterative Chaos for reasoning model jailbreaking

**Mousetrap: Chain of Iterative Chaos**:
- Advanced jailbreaking technique specifically designed for reasoning-capable language models
- Creates multi-step chaotic reasoning chains that gradually introduce confusion and misdirection
- Adaptive configuration based on target model response patterns
- Configurable chaos escalation, semantic obfuscation, and iterative refinement
- API endpoints: `/api/v1/autodan/mousetrap`, `/api/v1/autodan/mousetrap/adaptive`

**GPTFuzz Integration**:
- Session-based mutation testing with configurable mutators
- MCTS exploration policy for intelligent prompt selection
- Integrates with `llm_service` for LLM predictions

**Development Workflow Standards**:
- All prompt files (`.prompt.md`) require markdown frontmatter with `mode` and `description` fields
- Agent files (`.agent.md`) follow similar frontmatter requirements
- Instruction files (`.instructions.md`) must have `applyTo` field specifying target file patterns
- Skills in `.github/skills/` must have proper `SKILL.md` with `name` matching folder name

**Multi-Provider Architecture**:
- Automatic provider detection based on configured API keys
- Mock provider always available as fallback
- Provider-specific model configuration via environment variables
- Detailed setup guide in `backend-api/docs/MULTI_MODEL_SETUP.md`


