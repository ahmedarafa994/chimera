# Design Document: Backend API Refactoring

## Overview

This design consolidates the fragmented `backend-api` workspace into a clean, production-grade FastAPI application. The existing `app/` directory already implements a well-structured FastAPI application with proper separation of concerns. This refactoring focuses on removing redundant files and standardizing the entry point and documentation.

## Architecture

The refactored backend follows a clean layered architecture:

```
backend-api/
├── app/                          # FastAPI Application (KEEP - already well-structured)
│   ├── api/                      # API Layer - Route handlers
│   │   ├── v1/                   # Version 1 endpoints
│   │   ├── v2/                   # Version 2 endpoints (advanced)
│   │   ├── routes.py             # Router aggregation
│   │   └── dependencies.py       # Shared dependencies
│   ├── core/                     # Core Layer - Cross-cutting concerns
│   │   ├── config.py             # Pydantic settings
│   │   ├── errors.py             # Error handling
│   │   ├── logging.py            # Logging configuration
│   │   └── security.py           # Security utilities
│   ├── domain/                   # Domain Layer - Business models
│   │   ├── models.py             # Domain entities
│   │   └── interfaces.py         # Abstract interfaces
│   ├── infrastructure/           # Infrastructure Layer - External services
│   │   ├── gemini_client.py      # Google Gemini integration
│   │   └── proxy_client.py       # LLM proxy clients
│   ├── middleware/               # Middleware Layer
│   │   ├── auth.py               # API key authentication
│   │   └── rate_limit.py         # Rate limiting
│   ├── services/                 # Service Layer - Business logic
│   │   ├── llm_service.py        # LLM orchestration
│   │   └── transformation_service.py
│   ├── main.py                   # FastAPI app factory
│   └── schemas.py                # Pydantic request/response models
├── tests/                        # Test suite (KEEP)
├── data/                         # Data files (KEEP)
├── run.py                        # Unified entry point (KEEP)
├── requirements.txt              # Dependencies (KEEP)
├── .env.example                  # Environment template (KEEP)
├── README.md                     # Documentation (UPDATE)
├── JAILBREAK_SYSTEM.md          # Feature docs (KEEP)
├── MULTI_MODEL_SETUP.md         # Feature docs (KEEP)
└── pytest.ini                    # Test config (KEEP)
```

## Components and Interfaces

### Entry Point (run.py)

The single entry point that starts the FastAPI application:

```python
#!/usr/bin/env python3
import uvicorn
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9250)),
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )
```

### Configuration (app/core/config.py)

Centralized configuration using pydantic-settings (already implemented):

- `API_V1_STR`: API version prefix ("/api/v1")
- `PORT`: Server port (default: 9250)
- `ENVIRONMENT`: development/production
- `CHIMERA_API_KEY`: API authentication key
- LLM provider API keys (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)

### API Routes Structure

```
/                           # Service info (public)
/health                     # Health check (public)
/api/v1/providers           # List LLM providers (public)
/api/v1/techniques          # List techniques (public)
/api/v1/models              # List models (public)
/api/v1/transform           # Transform prompt (authenticated)
/api/v1/execute             # Execute with LLM (authenticated)
/api/v1/jailbreak/*         # Jailbreak endpoints (authenticated)
/api/v2/advanced/*          # Advanced transformation (authenticated)
```

## Files to Remove

### Redundant Server Implementations
| File | Reason |
|------|--------|
| `server.js` | Node.js duplicate - use FastAPI |
| `simple_server.py` | Raw HTTP duplicate |
| `minimal_server.py` | Raw HTTP duplicate |
| `minimal_flask_server.py` | Flask duplicate |
| `working_server.py` | Raw HTTP duplicate |
| `chimera_server.py` | Duplicate server |
| `real_ai_server.py` | Duplicate server |
| `debug_server.py` | Debug duplicate |
| `appmain.py` | Duplicate entry |
| `server.py` | Duplicate server |

### Debug and Verification Scripts
| File | Reason |
|------|--------|
| `debug_env.py` | Debug script |
| `debug_providers.py` | Debug script |
| `verify_autodan*.py` (6 files) | Verification scripts |
| `verify_backend.py` | Verification script |
| `verify_execute.py` | Verification script |
| `verify_fix_*.py` (2 files) | Verification scripts |
| `verify_improvements.py` | Verification script |
| `check_google_models.py` | Debug script |
| `list_gemini_models.py` | Debug script |
| `intent_expander.py` | Orphaned utility |

### Test Scripts (outside tests/)
| File | Reason |
|------|--------|
| `run_test.py` | Move to tests/ or remove |
| `test_gemini_generation.py` | Move to tests/ or remove |
| `test_import.py` | Move to tests/ or remove |
| `test_providers.py` | Move to tests/ or remove |
| `test_server.py` | Move to tests/ or remove |
| `security_validation.py` | Move to tests/ or remove |
| `security_validation_simple.py` | Move to tests/ or remove |
| `legacy_simple_api.py` | Legacy code |
| `legacy_test_integration.py` | Legacy code |

### Artifacts and Logs
| File | Reason |
|------|--------|
| `error_log.txt` | Log artifact |
| `output.txt` | Output artifact |
| `traceback.txt` | Debug artifact |
| `test_server.log` | Log artifact |
| `security_validation_report.json` | Report artifact |
| `verify_output_8082.txt` | Debug artifact |
| `start_server.sh` | Shell script (use run.py) |

### Environment Files to Consolidate
| File | Action |
|------|--------|
| `.env` | KEEP |
| `.env.example` | KEEP (canonical template) |
| `.env.template` | REMOVE (duplicate) |

## Data Models

The existing Pydantic schemas in `app/schemas.py` are well-defined and should be retained:

- `TechniqueSuite` (Enum): All available technique suites
- `Provider` (Enum): LLM provider identifiers
- `TransformRequest`: Transform endpoint request
- `TransformResponse`: Transform endpoint response
- `ExecuteRequest`: Execute endpoint request
- `ExecuteResponse`: Execute endpoint response
- `ProviderListResponse`: Provider listing response
- `TechniqueListResponse`: Technique listing response

## Error Handling

The existing error handling in `app/core/errors.py` provides:

- `AppError`: Base application exception
- `app_exception_handler`: FastAPI exception handler
- `global_exception_handler`: Catch-all handler

All errors return consistent JSON:
```json
{
  "error": "Error message",
  "status_code": 400,
  "timestamp": "2025-01-01T00:00:00Z"
}
```

## Testing Strategy

Retain the existing test structure in `tests/`:
- `tests/conftest.py` - Shared fixtures
- `tests/test_api.py` - API endpoint tests
- `tests/test_integration_new.py` - Integration tests
- `tests/jailbreak/` - Jailbreak-specific tests

Run tests with: `pytest tests/ -v`

## Migration Steps

1. **Backup**: Create backup of current state
2. **Remove redundant files**: Delete all identified redundant files
3. **Update run.py**: Ensure proper port configuration
4. **Update README.md**: Reflect FastAPI and correct commands
5. **Verify**: Run tests and manual verification
