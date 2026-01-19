# Project Chimera Backend API

**AI Prompt Transformation Engine for Security Research**

A production-grade FastAPI backend for prompt engineering, transformation, and multi-provider LLM integration. Built for AI safety research and red-team security testing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (port 8001)
python run.py
```

The server runs at `http://localhost:8001`.

## API Documentation

Interactive docs available when server is running:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## API Endpoints

### Public Endpoints (No Authentication)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and available endpoints |
| `/health` | GET | Health check with security status |
| `/api/v1/providers` | GET | List available LLM providers |
| `/api/v1/techniques` | GET | List available technique suites |
| `/api/v1/techniques/{suite_name}` | GET | Get technique suite details |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/validate` | POST | Validate model availability |

### Authenticated Endpoints (API Key Required)

#### Core Transformation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/transform` | POST | Transform a prompt without execution |
| `/api/v1/execute` | POST | Transform and execute with LLM |
| `/api/v1/generate` | POST | Generate content using LLM |
| `/api/v1/metrics` | GET | System metrics |

#### Chat & Sessions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/completions` | POST | Chat completion endpoint |
| `/api/v1/session/create` | POST | Create new session |
| `/api/v1/session/{id}` | GET | Get session details |

#### AutoDAN Integration
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/autodan/generate` | POST | Generate AutoDAN prompt |
| `/api/v1/autodan/strategies` | GET | List AutoDAN strategies |

#### Jailbreak System
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/jailbreak/generate` | POST | Generate jailbreak prompt |
| `/api/v1/jailbreak/techniques` | GET | List jailbreak techniques |
| `/api/v1/generation/jailbreak/generate` | POST | Advanced jailbreak generation |

#### HouYi Optimization
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/optimize/houyi` | POST | HouYi optimization |

### Advanced Transformation (v2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/advanced/transform` | POST | Multi-layered transformation |
| `/api/v2/advanced/analyze` | GET | Analyze prompt characteristics |
| `/api/v2/advanced/capabilities` | GET | Get transformation capabilities |
| `/api/v2/advanced/batch` | POST | Batch transform multiple prompts |
| `/api/v2/advanced/feedback` | POST | Submit transformation feedback |
| `/api/v2/advanced/statistics` | GET | Transformation statistics |
| `/api/v2/advanced/health` | GET | Advanced service health |

## Authentication

API key authentication is required for most endpoints. Provide your key via:

```bash
# Header (recommended)
Authorization: Bearer <your-api-key>
# or
X-API-Key: <your-api-key>

# Query parameter
?api_key=<your-api-key>
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
# Application
ENVIRONMENT=development
PORT=8001
LOG_LEVEL=INFO

# Security (REQUIRED - change in production)
CHIMERA_API_KEY=your-secure-api-key

# =============================================================================
# API CONNECTION CONFIGURATION
# =============================================================================
# Choose between "proxy" or "direct" mode
API_CONNECTION_MODE=proxy

# Option 1: Proxy Server Connection
# Route all API requests through a local proxy endpoint
USE_LLM_PROXY=true
LLM_PROXY_URL=http://localhost:8080/v1
LLM_PROXY_API_KEY=your-secure-proxy-api-key

# Option 2: Direct Gemini API Connection
# Connect directly to Google's Gemini API
GEMINI_DIRECT_API_KEY=your-gemini-api-key
GEMINI_DIRECT_BASE_URL=https://generativelanguage.googleapis.com/v1beta

# LLM Providers (for additional providers)
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# CORS Origins
BACKEND_CORS_ORIGINS=["http://localhost:3001"]

# Cache
ENABLE_CACHE=true
```

## API Connection Modes

The application supports two connection modes for AI API access:

### Option 1: Proxy Server Connection (Default)
Routes all API requests through a local proxy server at `http://localhost:8080/v1`.
- Best for development and debugging
- Supports multiple model providers through a single endpoint
- Requires the proxy server to be running

### Option 2: Direct Gemini API Connection
Connects directly to Google's Gemini API using your API key.
- No proxy server required
- Direct access to Gemini models
- Requires a valid Gemini API key

### Connection Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/connection/config` | GET | Get current connection configuration |
| `/api/v1/connection/status` | GET | Check current connection status |
| `/api/v1/connection/mode` | POST | Switch between proxy and direct modes |
| `/api/v1/connection/test` | POST | Test both connection methods |
| `/api/v1/connection/health` | GET | Quick health check for current connection |

## Directory Structure

```
backend-api/
├── app/
│   ├── api/           # Route handlers (v1/, v2/)
│   ├── core/          # Config, security, errors, logging
│   ├── domain/        # Domain models and interfaces
│   ├── infrastructure/# LLM provider clients
│   ├── middleware/    # Auth, rate limiting, security headers
│   ├── services/      # Business logic
│   ├── main.py        # FastAPI app factory
│   └── schemas.py     # Pydantic models
├── tests/             # Test suite
├── data/              # Data files
├── run.py             # Entry point
└── requirements.txt   # Dependencies
```

## Security Features

- API key authentication
- Rate limiting (60 req/min per IP)
- Request size limiting (10MB max)
- Security headers (CSP, HSTS, etc.)
- Audit logging
- Input validation via Pydantic

## Running with Uvicorn Directly

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Testing

```bash
pytest tests/ -v
```

## Related Documentation

- [JAILBREAK_SYSTEM.md](JAILBREAK_SYSTEM.md) - Jailbreak engine documentation
- [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) - Multi-provider LLM configuration
