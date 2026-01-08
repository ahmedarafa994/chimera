# Chimera Backend API

High-performance FastAPI backend for the Chimera adversarial prompting and red teaming platform.

## Technology Stack

- **Framework**: FastAPI 0.104+
- **Language**: Python 3.11+
- **Database**: SQLite (development), PostgreSQL (production)
- **Cache**: Redis (optional, for rate limiting)
- **Package Manager**: Poetry
- **ASGI Server**: Uvicorn

## Quick Start

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip

### Installation

```bash
# Using Poetry (recommended)
cd backend-api
poetry install

# Or using pip
pip install -r requirements.txt
```

### Development

```bash
# Using Poetry
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Or using the npm script from project root
npm run dev:backend

# Or using the CLI entry point
poetry run chimera-dev
```

### Production

```bash
# Start production server
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4

# Or with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001
```

## Project Structure

```
backend-api/
├── app/
│   ├── main.py                 # Application entry point
│   ├── config/                 # Configuration management
│   │   ├── settings.py         # Environment settings
│   │   └── logging.py          # Logging configuration
│   ├── core/                   # Core functionality
│   │   ├── middleware.py       # CSRF, auth middleware
│   │   ├── validation.py       # Input validation
│   │   └── security.py         # Security utilities
│   ├── middleware/             # Custom middleware
│   │   ├── auth.py             # API key authentication
│   │   └── rate_limit.py       # Rate limiting
│   ├── routers/                # API route handlers
│   │   ├── api.py              # Main API router
│   │   ├── providers.py        # LLM provider management
│   │   ├── models.py           # Model configuration
│   │   ├── generation.py       # Content generation
│   │   ├── jailbreak.py        # Jailbreak research
│   │   ├── transform.py        # Prompt transformation
│   │   ├── techniques.py       # Transformation techniques
│   │   └── websocket.py        # WebSocket handlers
│   ├── services/               # Business logic
│   │   ├── llm_service.py      # LLM provider integration
│   │   ├── prompt_service.py   # Prompt processing
│   │   └── cache_service.py    # Caching layer
│   └── schemas/                # Pydantic models
│       ├── request.py          # Request schemas
│       └── response.py         # Response schemas
├── alembic/                    # Database migrations
│   ├── versions/               # Migration scripts
│   └── alembic.ini             # Alembic configuration
├── tests/                      # Unit tests
├── pyproject.toml              # Poetry dependencies
└── requirements.txt            # pip dependencies
```

## API Endpoints

### Health & Status
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with service status |
| GET | `/api/v1/health/detailed` | Detailed health with provider status |

### Providers
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/providers` | List available LLM providers |
| GET | `/api/v1/providers/{id}` | Get provider details |
| POST | `/api/v1/providers/select` | Set active provider/model |

### Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/generate` | Generate content from LLM |
| POST | `/api/v1/transform` | Transform prompts |
| POST | `/api/v1/generation/jailbreak/generate` | Jailbreak research |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws/enhance` | Real-time prompt enhancement |
| `/ws/status` | Live status updates |

## Security Features

### Middleware Stack
The API includes comprehensive security middleware:

1. **SecurityHeadersMiddleware** - Sets security headers (CSP, X-Frame-Options, etc.)
2. **RateLimitMiddleware** - Request rate limiting (configurable)
3. **InputValidationMiddleware** - Payload validation and sanitization
4. **CSRFMiddleware** - CSRF protection for non-GET requests
5. **AuthMiddleware** - API key authentication

### Authentication
```bash
# Include API key in request header
curl -X GET "http://localhost:8001/api/v1/providers" \
  -H "X-API-Key: your-api-key"
```

### Rate Limiting
Configure via environment variables:
```env
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Server
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8001

# Security
API_KEY=your-secure-api-key
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000

# LLM Providers
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key

# Rate Limiting
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60

# Database (optional)
DATABASE_URL=sqlite:///./chimera.db

# Redis (optional, for distributed rate limiting)
REDIS_URL=redis://localhost:6379
```

## Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=app --cov-report=html

# Specific test file
poetry run pytest tests/test_providers.py -v

# Run security tests
poetry run pytest tests/ -m "security" -v
```

## Database Migrations

Using Alembic for database migrations:

```bash
# Create new migration
poetry run alembic revision --autogenerate -m "Description"

# Apply migrations
poetry run alembic upgrade head

# Rollback one version
poetry run alembic downgrade -1
```

## API Documentation

Once the server is running:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI JSON**: http://localhost:8001/openapi.json

## Performance

### Optimization Features
- Async/await throughout for non-blocking I/O
- Connection pooling for database and HTTP clients
- Response caching with configurable TTL
- Circuit breaker pattern for LLM providers

### Monitoring
- Prometheus metrics at `/metrics`
- Structured JSON logging
- Request tracing with correlation IDs

## Related Documentation

- [Architecture Overview](../docs/ARCHITECTURE.md)
- [API Reference](../docs/openapi.yaml)
- [Developer Guide](../docs/DEVELOPER_GUIDE.md)
- [Security Audit](../SECURITY_AUDIT_REPORT.md)
