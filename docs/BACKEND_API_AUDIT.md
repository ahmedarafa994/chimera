# Backend API Technical Audit Report

**Project**: Chimera Backend API  
**Audit Date**: 2026-01-06  
**Phase**: 1 of Gap Analysis Project  
**Auditor**: Automated Technical Analysis  

---

## Executive Summary

The Chimera Backend API is a comprehensive FastAPI-based system designed for adversarial prompt engineering and LLM security research. The architecture demonstrates sophisticated patterns including:

- **Layered Architecture**: Clear separation between routers, services, domain models, and infrastructure
- **Multi-Provider Support**: Integration with Google/Gemini, OpenAI, Anthropic, DeepSeek, Qwen, and Cursor
- **Advanced Security**: Dual authentication (JWT + API Key), RBAC permissions, timing-safe comparisons
- **Real-time Capabilities**: WebSocket endpoints and SSE streaming for live updates
- **Resilience Patterns**: Circuit breakers, rate limiting, caching, and fallback providers
- **Comprehensive Observability**: Prometheus metrics, health checks, and structured logging

### Key Statistics

| Metric | Count |
|--------|-------|
| Total API Endpoints | 95+ |
| WebSocket Endpoints | 5 |
| SSE Streaming Endpoints | 6 |
| Pydantic Models | 50+ |
| OpenAPI Tags | 15 |
| LLM Providers | 8 |

---

## 1. API Endpoints Inventory

### 1.1 Health & System Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/health` | Comprehensive health check with component status | None |
| GET | `/health/live` | Kubernetes liveness probe | None |
| GET | `/health/ready` | Kubernetes readiness probe | None |
| GET | `/health/circuit-breakers` | Circuit breaker status for all providers | None |
| POST | `/health/circuit-breakers/{name}/reset` | Reset specific circuit breaker | None |
| GET | `/health/proxy` | Proxy server health (AIClient-2-API) | None |
| GET | `/health/integration` | Provider integration health monitoring | None |
| GET | `/health/integration/graph` | Service dependency graph visualization | None |
| GET | `/health/integration/history` | Provider health history over time | None |
| GET | `/health/integration/alerts` | Active health alerts | None |
| POST | `/health/integration/check` | Trigger immediate health check | None |

### 1.2 Generation Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/generation/generate` | Generate text with LLM provider | Yes |
| GET | `/api/v1/generation/health` | LLM provider availability status | Yes |

### 1.3 Streaming Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/streaming/generate/stream` | SSE streaming text generation | Yes |
| POST | `/api/v1/streaming/generate/stream/raw` | Raw text streaming (no JSON) | Yes |
| GET | `/api/v1/streaming/generate/stream/capabilities` | Provider streaming capabilities | Yes |

### 1.4 Provider Management Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/v1/providers/` | List all available providers | Yes |
| GET | `/api/v1/providers/{provider}/models` | Get models for specific provider | Yes |
| POST | `/api/v1/providers/select` | Select provider and model | Yes |
| GET | `/api/v1/providers/rate-limit` | Check rate limit status | Yes |
| GET | `/api/v1/providers/current` | Get current provider selection | Yes |
| GET | `/api/v1/providers/health` | Provider health status | Yes |
| WebSocket | `/api/v1/providers/ws/selection` | Real-time model selection sync | Yes |

### 1.5 Session Management Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/v1/session/models` | Get available models for session | Yes |
| POST | `/api/v1/session/models/validate` | Validate model selection | Yes |
| POST | `/api/v1/session` | Create new session | Yes |
| GET | `/api/v1/session` | Get current session info | Yes |
| DELETE | `/api/v1/session` | Delete current session | Yes |
| GET | `/api/v1/session/{session_id}` | Get session by ID | Yes |
| GET | `/api/v1/session/stats` | Session statistics (admin) | Yes |
| GET | `/api/v1/session/current-model` | Get current model selection | Yes |
| PUT | `/api/v1/session/model` | Update session model | Yes |

### 1.6 Transformation Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/transformation/` | Transform prompt with techniques | Permission |
| POST | `/api/v1/transformation/stream` | SSE streaming transformation | Permission |
| POST | `/api/v1/transformation/estimate-tokens` | Estimate token usage | Permission |
| GET | `/api/v1/transformation/cache/stats` | Transformation cache statistics | Yes |

### 1.7 Execute Pipeline Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/execute/execute` | Transform + execute with LLM | Permission |

### 1.8 Chat Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/chat/completions` | Chat completion with transformation | Yes |

### 1.9 Jailbreak Service Endpoints (DeepTeam)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/jailbreak/generate` | Generate jailbreak prompts | Yes |
| POST | `/api/v1/jailbreak/generate/quick` | Quick jailbreak generation | Yes |
| POST | `/api/v1/jailbreak/generate/batch` | Batch jailbreak generation | Yes |
| GET | `/api/v1/jailbreak/generate/stream` | SSE jailbreak streaming | Yes |
| WebSocket | `/api/v1/jailbreak/ws/generate` | WebSocket jailbreak streaming | Yes |
| GET | `/api/v1/jailbreak/strategies` | List attack strategies | Yes |
| GET | `/api/v1/jailbreak/strategies/{strategy_type}` | Get strategy details | Yes |
| GET | `/api/v1/jailbreak/vulnerabilities` | List vulnerability categories | Yes |
| GET | `/api/v1/jailbreak/cache/stats` | Jailbreak cache statistics | Yes |
| DELETE | `/api/v1/jailbreak/cache` | Clear jailbreak cache | Yes |
| GET | `/api/v1/jailbreak/session/{session_id}` | Get session result | Yes |
| DELETE | `/api/v1/jailbreak/session/{session_id}` | Cancel jailbreak session | Yes |
| GET | `/api/v1/jailbreak/health` | Jailbreak service health | Yes |

### 1.10 AutoDAN Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/autodan/jailbreak` | AutoDAN jailbreak generation | Yes |
| POST | `/api/v1/autodan/batch` | Batch AutoDAN generation | Yes |
| GET | `/api/v1/autodan/config` | Get AutoDAN configuration | Yes |
| POST | `/api/v1/autodan/lifelong` | Lifelong learning attack | Yes |

### 1.11 AutoDAN-Turbo Endpoints (Lifelong Learning)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/autodan-turbo/attack` | Single lifelong attack | Rate Limited |
| POST | `/api/v1/autodan-turbo/warmup` | Warm-up exploration phase | Rate Limited |
| POST | `/api/v1/autodan-turbo/lifelong` | Full lifelong learning loop | Yes |
| POST | `/api/v1/autodan-turbo/test` | Test stage with fixed library | Yes |
| GET | `/api/v1/autodan-turbo/strategies` | List strategies | Yes |
| GET | `/api/v1/autodan-turbo/strategies/{strategy_id}` | Get strategy details | Yes |
| POST | `/api/v1/autodan-turbo/strategies` | Create human-designed strategy | Yes |
| DELETE | `/api/v1/autodan-turbo/strategies/{strategy_id}` | Delete strategy | Yes |
| POST | `/api/v1/autodan-turbo/strategies/search` | Embedding-based strategy search | Yes |
| POST | `/api/v1/autodan-turbo/strategies/batch-inject` | Batch import strategies | Yes |
| GET | `/api/v1/autodan-turbo/progress` | Get learning progress | Yes |
| GET | `/api/v1/autodan-turbo/library/stats` | Library statistics | Yes |
| POST | `/api/v1/autodan-turbo/reset` | Reset engine progress | Yes |
| POST | `/api/v1/autodan-turbo/library/save` | Save library to disk | Yes |
| POST | `/api/v1/autodan-turbo/library/clear` | Clear library (destructive) | Yes |
| GET | `/api/v1/autodan-turbo/health` | Service health check | Yes |
| POST | `/api/v1/autodan-turbo/transfer/export` | Export library for transfer | Yes |
| POST | `/api/v1/autodan-turbo/transfer/import` | Import library | Yes |
| POST | `/api/v1/autodan-turbo/score` | Score adversarial response | Yes |
| POST | `/api/v1/autodan-turbo/extract` | Extract strategy from attack | Yes |

### 1.12 AutoAdv Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/autoadv/start` | Start AutoAdv job | Yes |
| WebSocket | `/api/v1/autoadv/ws` | Real-time AutoAdv updates | Yes |

### 1.13 GPTFuzz Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/gptfuzz/run` | Start GPTFuzz fuzzing session | Yes |
| GET | `/api/v1/gptfuzz/status/{session_id}` | Get fuzzing session status | Yes |

### 1.14 DeepTeam Red Team Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/deepteam/red-team` | Run comprehensive red teaming | Yes |
| POST | `/api/v1/deepteam/quick-scan` | Quick vulnerability scan | Yes |
| POST | `/api/v1/deepteam/security-audit` | Security-focused audit | Yes |
| POST | `/api/v1/deepteam/bias-audit` | Bias detection audit | Yes |
| POST | `/api/v1/deepteam/owasp-assessment` | OWASP Top 10 for LLMs | Yes |
| POST | `/api/v1/deepteam/assess-vulnerability` | Single vulnerability test | Yes |
| GET | `/api/v1/deepteam/sessions` | List red team sessions | Yes |
| GET | `/api/v1/deepteam/sessions/{session_id}` | Get session status | Yes |
| GET | `/api/v1/deepteam/sessions/{session_id}/result` | Get session result | Yes |
| GET | `/api/v1/deepteam/vulnerabilities` | List vulnerabilities | Yes |
| GET | `/api/v1/deepteam/attacks` | List attack methods | Yes |
| GET | `/api/v1/deepteam/presets` | List preset configs | Yes |
| GET | `/api/v1/deepteam/health` | DeepTeam health check | Yes |

### 1.15 DeepTeam Jailbreak Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/deepteam/jailbreak/generate` | Advanced jailbreak generation | Yes |
| POST | `/api/v1/deepteam/jailbreak/batch` | Batch jailbreak generation | Yes |
| GET | `/api/v1/deepteam/jailbreak/strategies` | List jailbreak strategies | Yes |
| GET | `/api/v1/deepteam/jailbreak/strategies/{strategy_type}` | Strategy details | Yes |
| DELETE | `/api/v1/deepteam/jailbreak/cache` | Clear jailbreak cache | Yes |
| GET | `/api/v1/deepteam/jailbreak/health` | Service health | Yes |
| WebSocket | `/api/v1/deepteam/jailbreak/ws/generate` | Streaming jailbreak | Yes |
| GET | `/api/v1/deepteam/jailbreak/generate/stream` | SSE jailbreak streaming | Yes |
| GET | `/api/v1/deepteam/jailbreak/sessions/{session_id}/prompts` | Get session prompts | Yes |
| GET | `/api/v1/deepteam/jailbreak/sessions/{session_id}/prompts/{prompt_id}` | Get specific prompt | Yes |
| DELETE | `/api/v1/deepteam/jailbreak/sessions/{session_id}` | Delete session | Yes |
| GET | `/api/v1/deepteam/jailbreak/sessions` | List jailbreak sessions | Yes |
| POST | `/api/v1/deepteam/jailbreak/sessions/{session_id}/cancel` | Cancel active session | Yes |

### 1.16 Admin Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/v1/admin/feature-flags` | List technique feature flags | Admin API Key |
| GET | `/api/v1/admin/feature-flags/stats` | Feature flag statistics | Admin API Key |
| POST | `/api/v1/admin/feature-flags/toggle` | Toggle technique enabled status | Admin API Key |
| POST | `/api/v1/admin/feature-flags/reload` | Reload config from disk | Admin API Key |
| GET | `/api/v1/admin/feature-flags/{technique_name}` | Get technique details | Admin API Key |
| GET | `/api/v1/admin/tenants` | List all tenants | Admin API Key |
| POST | `/api/v1/admin/tenants` | Create tenant | Admin API Key |
| GET | `/api/v1/admin/tenants/{tenant_id}` | Get tenant details | Admin API Key |
| DELETE | `/api/v1/admin/tenants/{tenant_id}` | Delete tenant | Admin API Key |
| GET | `/api/v1/admin/tenants/stats/summary` | Tenant statistics | Admin API Key |
| GET | `/api/v1/admin/usage/global` | Global usage statistics | Admin API Key |
| GET | `/api/v1/admin/usage/tenant/{tenant_id}` | Tenant usage summary | Admin API Key |
| GET | `/api/v1/admin/usage/techniques/top` | Top used techniques | Admin API Key |
| GET | `/api/v1/admin/usage/quota/{tenant_id}` | Check tenant quota | Admin API Key |

### 1.17 Metrics Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/api/v1/metrics/prometheus` | Prometheus exposition format | None |
| GET | `/api/v1/metrics/json` | JSON metrics format | None |
| GET | `/api/v1/metrics/circuit-breakers` | Circuit breaker status | None |
| POST | `/api/v1/metrics/circuit-breakers/{name}/reset` | Reset circuit breaker | None |
| POST | `/api/v1/metrics/circuit-breakers/reset-all` | Reset all circuit breakers | None |
| GET | `/api/v1/metrics/cache` | Transformation cache metrics | None |
| POST | `/api/v1/metrics/cache/clear` | Clear transformation cache | None |
| GET | `/api/v1/metrics/connection-pools` | Connection pool stats | None |
| POST | `/api/v1/metrics/connection-pools/reset` | Reset pool statistics | None |
| GET | `/api/v1/metrics/multi-level-cache` | L1/L2 cache metrics | None |
| POST | `/api/v1/metrics/multi-level-cache/clear` | Clear multi-level cache | None |

### 1.18 WebSocket Endpoints Summary

| Path | Purpose | Protocol |
|------|---------|----------|
| `/ws/enhance` | Real-time prompt enhancement | WebSocket |
| `/api/v1/providers/ws/selection` | Model selection sync with heartbeat | WebSocket |
| `/api/v1/jailbreak/ws/generate` | Jailbreak generation streaming | WebSocket |
| `/api/v1/deepteam/jailbreak/ws/generate` | DeepTeam jailbreak streaming | WebSocket |
| `/api/v1/autoadv/ws` | AutoAdv job updates | WebSocket |

---

## 2. Data Models & Schemas

### 2.1 Core Domain Models

#### [`PromptRequest`](backend-api/app/domain/models.py:62)
```python
class PromptRequest(BaseModel):
    prompt: str  # 1-50000 chars, validated for dangerous patterns
    system_instruction: str | None  # Max 10000 chars
    config: GenerationConfig | None
    model: str | None  # Specific model override
    provider: LLMProviderType | None
    api_key: str | None  # Optional API key override
    skip_validation: bool = False  # Skip dangerous content validation
```

#### [`PromptResponse`](backend-api/app/domain/models.py:186)
```python
class PromptResponse(BaseModel):
    text: str  # Max 50000 chars
    model_used: str
    provider: str
    usage_metadata: dict[str, Any] | None
    finish_reason: str | None
    latency_ms: float
```

#### [`GenerationConfig`](backend-api/app/domain/models.py:38)
```python
class GenerationConfig(BaseModel):
    temperature: float = 0.7  # 0.0-1.0
    top_p: float = 0.95  # 0.0-1.0
    top_k: int = 40  # >= 1
    max_output_tokens: int = 2048  # 1-8192
    stop_sequences: list[str] | None  # Max 10 sequences
    thinking_level: str | None  # "low" | "medium" | "high" for Gemini 3
```

### 2.2 LLM Provider Types

```python
class LLMProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"
    QWEN = "qwen"
    GEMINI_CLI = "gemini-cli"
    ANTIGRAVITY = "antigravity"
    KIRO = "kiro"
    CURSOR = "cursor"
    XAI = "xai"
    DEEPSEEK = "deepseek"
    MOCK = "mock"
```

### 2.3 Transformation Models

#### [`TransformationRequest`](backend-api/app/domain/models.py:147)
```python
class TransformationRequest(BaseModel):
    core_request: str  # 1-5000 chars
    potency_level: int  # 1-10
    technique_suite: str  # 1-50 chars, alphanumeric
```

#### [`ExecutionRequest`](backend-api/app/domain/models.py:165)
```python
class ExecutionRequest(TransformationRequest):
    provider: str | None = "openai"
    use_cache: bool = True
    model: str | None
    temperature: float | None = 0.7
    max_tokens: int | None = 2048
    top_p: float | None = 0.95
    frequency_penalty: float | None = 0.0
    presence_penalty: float | None = 0.0
    api_key: str | None
```

### 2.4 Jailbreak Models

#### [`JailbreakGenerationRequest`](backend-api/app/domain/models.py:353)
```python
class JailbreakGenerationRequest(BaseModel):
    core_request: str  # 1-5000 chars
    technique_suite: str
    potency_level: int  # 1-10
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048  # 256-8192
    density: float = 0.5
    # Content transformation flags
    use_leet_speak: bool = False
    use_homoglyphs: bool = False
    use_caesar_cipher: bool = False
    # Structural & semantic flags
    use_role_hijacking: bool = False
    use_instruction_injection: bool = False
    use_adversarial_suffixes: bool = False
    # Advanced neural flags
    use_neural_bypass: bool = False
    use_meta_prompting: bool = False
    # Research-driven flags
    use_multilingual_trojan: bool = False
    use_payload_splitting: bool = False
    use_contextual_interaction_attack: bool = False
```

### 2.5 Streaming Models

#### [`StreamChunk`](backend-api/app/domain/models.py:462)
```python
class StreamChunk(BaseModel):
    text: str
    is_final: bool = False
    finish_reason: str | None
    token_count: int | None
```

### 2.6 Error Response Model

#### [`ErrorResponse`](backend-api/app/domain/models.py:539)
```python
class ErrorResponse(BaseModel):
    error_code: str  # 1-50 chars
    message: str  # 1-500 chars
    status_code: int  # 400-599
    details: dict[str, Any] | None
    timestamp: str
    request_id: str | None
```

### 2.7 API Schema Models

#### Base Schemas ([`base_schemas.py`](backend-api/app/schemas/base_schemas.py))
- `BaseSchema` - Common Pydantic configuration
- `BaseRequest` - Base for all requests
- `BaseResponse` - Base with id, created_at, updated_at
- `ErrorResponse` - Standardized error format
- `SuccessResponse` - Standardized success format
- `PaginatedResponse` - Items with pagination metadata
- `HealthCheckResponse` - Health status with services

#### API-Specific Schemas ([`api_schemas.py`](backend-api/app/schemas/api_schemas.py))
- `LLMProvider` enum (openai, gemini, huggingface, custom)
- `EvasionTaskStatusEnum` enum (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- `LLMModelBase`, `LLMModelCreate`, `LLMModel`
- `MetamorphosisStrategyInfo`, `MetamorphosisStrategyConfig`
- `EvasionTaskConfig`, `EvasionTaskStatusResponse`, `EvasionAttemptResult`, `EvasionTaskResult`

---

## 3. Authentication & Security

### 3.1 Authentication Mechanisms

The system implements dual authentication methods:

#### JWT Bearer Token Authentication
- **Header**: `Authorization: Bearer <token>`
- **Algorithm**: HS256 (configurable via `JWT_ALGORITHM`)
- **Secret**: `JWT_SECRET` environment variable (warns if not set in production)
- **Access Token Expiry**: 1 hour (configurable via `JWT_EXPIRATION_HOURS`)
- **Refresh Token Expiry**: 7 days
- **Token Revocation**: Redis-backed (production) or in-memory (development)

#### API Key Authentication
- **Header**: `X-API-Key: <key>`
- **Validation**: Timing-safe comparison via `secrets.compare_digest()`
- **Environment Variable**: `CHIMERA_API_KEY`
- **Fallback**: Bearer token can also be an API key (auto-detected by JWT format check)

### 3.2 Role-Based Access Control (RBAC)

#### Roles ([`auth.py`](backend-api/app/core/auth.py:35))
```python
class Role(str, Enum):
    ADMIN = "admin"        # Full system access
    OPERATOR = "operator"  # Execute and manage operations
    DEVELOPER = "developer"  # API access for development
    VIEWER = "viewer"      # Read-only access
    API_CLIENT = "api_client"  # Programmatic API access
```

#### Permissions ([`auth.py`](backend-api/app/core/auth.py:45))
```python
class Permission(str, Enum):
    # Read permissions
    READ_PROMPTS = "read:prompts"
    READ_TECHNIQUES = "read:techniques"
    READ_PROVIDERS = "read:providers"
    READ_METRICS = "read:metrics"
    READ_LOGS = "read:logs"
    # Write permissions
    WRITE_PROMPTS = "write:prompts"
    WRITE_TECHNIQUES = "write:techniques"
    WRITE_PROVIDERS = "write:providers"
    # Execute permissions
    EXECUTE_TRANSFORM = "execute:transform"
    EXECUTE_ENHANCE = "execute:enhance"
    EXECUTE_JAILBREAK = "execute:jailbreak"
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"
```

### 3.3 Dependency Injection for Auth

```python
# Get current authenticated user
async def get_current_user(...) -> TokenPayload

# Require specific permission
@app.get("/transform")
async def transform(user: TokenPayload = Depends(require_permission(Permission.EXECUTE_TRANSFORM))):
    ...

# Require specific role
@app.get("/admin")
async def admin(user: TokenPayload = Depends(require_role(Role.ADMIN))):
    ...
```

### 3.4 Security Features

1. **Password Hashing**: Argon2 (OWASP recommended) with bcrypt fallback
2. **Timing-Safe Comparison**: All API key validations use `secrets.compare_digest()`
3. **Token Revocation**: Redis-backed with automatic TTL expiration
4. **Input Validation**: All models have strict field validators
5. **Dangerous Pattern Detection**: XSS/injection patterns blocked in prompts
6. **Rate Limiting**: Per-IP rate limiting with memory leak protection
7. **CORS**: Configurable allowed origins
8. **Audit Logging**: Authentication, authorization, and API access logging

### 3.5 Admin Endpoint Security

Admin endpoints use dedicated authentication:
```python
async def verify_admin_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    # Timing-safe comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, settings.CHIMERA_API_KEY):
        raise HTTPException(status_code=401, ...)
```

---

## 4. Error Handling Patterns

### 4.1 Custom Exception Classes ([`errors.py`](backend-api/app/core/errors.py))

```python
class AppError(Exception):
    """Base application error with status code"""
    def __init__(self, detail: str, status_code: int = 400):
        self.detail = detail
        self.status_code = status_code

class LLMProviderError(AppError):
    """LLM provider failures (502 Bad Gateway)"""
    status_code = 502

class ProviderNotAvailableError(AppError):
    """Provider unavailable (503 Service Unavailable)"""
    status_code = 503

class TransformationError(AppError):
    """Transformation failures (400 Bad Request)"""
    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}

class InvalidPotencyError(TransformationError):
    """Invalid potency level"""

class InvalidTechniqueError(TransformationError):
    """Invalid technique suite"""
```

### 4.2 Global Exception Handlers

```python
# Application errors - custom JSON response with CORS headers
async def app_exception_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=get_cors_headers(request)
    )

# HTTP exceptions - preserve status code
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=get_cors_headers(request)
    )

# Catch-all for unhandled exceptions
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
        headers=get_cors_headers(request)
    )
```

### 4.3 Error Response Format

All errors follow a consistent format:
```json
{
    "error_code": "VALIDATION_ERROR",
    "message": "Prompt cannot be empty",
    "status_code": 400,
    "details": {"field": "prompt", "constraint": "min_length"},
    "timestamp": "2023-10-27T10:00:00Z",
    "request_id": "req_a1b2c3d4"
}
```

### 4.4 HTTP Status Codes Used

| Status | Usage |
|--------|-------|
| 200 | Success |
| 400 | Bad Request - validation errors, invalid input |
| 401 | Unauthorized - missing or invalid authentication |
| 403 | Forbidden - insufficient permissions |
| 404 | Not Found - resource not found |
| 409 | Conflict - duplicate strategy, etc. |
| 422 | Unprocessable Entity - Pydantic validation errors |
| 429 | Too Many Requests - rate limit exceeded |
| 500 | Internal Server Error - unexpected errors |
| 501 | Not Implemented - streaming not supported |
| 502 | Bad Gateway - LLM provider error |
| 503 | Service Unavailable - provider unavailable |

---

## 5. Configuration Settings

### 5.1 Core Settings ([`config.py`](backend-api/app/core/config.py))

| Setting | Default | Description |
|---------|---------|-------------|
| `API_V1_STR` | `/api/v1` | API version prefix |
| `PROJECT_NAME` | `Chimera Backend` | Project name |
| `VERSION` | `1.0.0` | API version |
| `ENVIRONMENT` | `development` | Environment (development/production) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENABLE_CACHE` | `True` | Enable caching |

### 5.2 LLM Provider API Keys

| Setting | Description |
|---------|-------------|
| `GOOGLE_API_KEY` | Google/Gemini API key |
| `GOOGLE_MODEL` | Default: `gemini-3-pro-preview` |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Default: `gpt-4o` |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Default: `claude-3-5-sonnet-20241022` |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_MODEL` | Default: `deepseek-chat` |
| `QWEN_API_KEY` | Qwen API key |
| `CURSOR_API_KEY` | Cursor API key |

### 5.3 Connection Mode Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `API_CONNECTION_MODE` | `direct` | `direct` or `proxy` |
| `PROXY_MODE_ENDPOINT` | `http://localhost:8080` | AIClient-2-API Server URL |
| `PROXY_MODE_ENABLED` | `False` | Enable proxy mode |
| `PROXY_MODE_TIMEOUT` | `30` | Proxy request timeout (5-120s) |
| `PROXY_MODE_FALLBACK_TO_DIRECT` | `True` | Fallback to direct on proxy failure |
| `PROXY_MODE_HEALTH_CHECK_INTERVAL` | `30` | Health check interval (10-300s) |

### 5.4 Jailbreak Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `CHIMERA_API_KEY` | None | Master API key |
| `JAILBREAK_ENABLED` | `True` | Enable jailbreak features |
| `JAILBREAK_MAX_DAILY_EXECUTIONS` | `1000` | Daily execution limit |
| `JAILBREAK_MAX_CONCURRENT_EXECUTIONS` | `100` | Concurrent execution limit |
| `JAILBREAK_RATE_LIMIT_PER_MINUTE` | `60` | Per-minute rate limit |
| `JAILBREAK_REQUIRE_APPROVAL_FOR_HIGH_RISK` | `True` | Require approval for high-risk |
| `JAILBREAK_CACHE_ENABLED` | `True` | Enable jailbreak caching |
| `JAILBREAK_CACHE_TTL_SECONDS` | `3600` | Cache TTL (1 hour) |
| `JAILBREAK_MAX_PROMPT_LENGTH` | `50000` | Maximum prompt length |

### 5.5 Cache Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CACHE_MAX_MEMORY_ITEMS` | `5000` | Max items in memory cache |
| `CACHE_MAX_VALUE_SIZE_BYTES` | `1000000` | Max entry size (1MB) |
| `CACHE_DEFAULT_TTL` | `3600` | Default TTL (1 hour) |
| `CACHE_ENABLE_L2` | `False` | Enable Redis L2 cache |
| `CACHE_L1_TTL` | `300` | L1 cache TTL (5 minutes) |
| `CACHE_L2_TTL` | `3600` | L2 cache TTL (1 hour) |

### 5.6 Rate Limiting

| Setting | Default | Description |
|---------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `True` | Enable rate limiting |
| `RATE_LIMIT_DEFAULT_LIMIT` | `60` | Default requests per window |
| `RATE_LIMIT_DEFAULT_WINDOW` | `60` | Window size in seconds |

### 5.7 Redis Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_PASSWORD` | None | Redis password |
| `REDIS_SSL` | `False` | Enable SSL |
| `REDIS_CONNECTION_TIMEOUT` | `5` | Connection timeout |
| `REDIS_SOCKET_TIMEOUT` | `5` | Socket timeout |

### 5.8 PPO (Reinforcement Learning) Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `PPO_ENABLED` | `True` | Enable PPO-based selection |
| `PPO_LEARNING_RATE` | `3e-4` | Adam optimizer learning rate |
| `PPO_GAMMA` | `0.99` | Discount factor |
| `PPO_GAE_LAMBDA` | `0.95` | GAE lambda |
| `PPO_CLIP_EPSILON` | `0.2` | PPO clipping parameter |
| `PPO_MIN_SAMPLES` | `50` | Min samples before PPO |
| `PPO_PERSIST_WEIGHTS` | `True` | Persist weights |
| `PPO_STORAGE_PATH` | `./ppo_state` | Weights storage path |

### 5.9 Database

| Setting | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./chimera.db` | Database connection string |

**Note**: SQLite is not supported in production - PostgreSQL is enforced via validator.

---

## 6. Middleware Stack

The following middleware is applied in order:

1. **ObservabilityMiddleware** - Request tracing and metrics
2. **CompressionMiddleware** - gzip/brotli response compression
3. **APIKeyMiddleware** - API key extraction and validation
4. **RequestLoggingMiddleware** - Request/response logging
5. **MetricsMiddleware** - Prometheus metrics collection
6. **AuthRateLimitMiddleware** - Rate limiting (production only)
7. **CORSMiddleware** - Cross-origin resource sharing

---

## 7. Issues & Recommendations

### 7.1 Security Concerns

1. **JWT Secret in Development**: Warning logged when `JWT_SECRET` not set - should fail in production
2. **API Key Storage**: API keys should use secure vault storage (HashiCorp Vault, AWS Secrets Manager)
3. **Rate Limiter Memory**: In-memory rate limiting with `MAX_TRACKED_IPS = 10000` could be exhausted by distributed attacks

### 7.2 Architecture Observations

1. **Circular Import Risk**: `config.py` has conditional imports to avoid circular dependencies
2. **Duplicate Functionality**: Some jailbreak endpoints exist in both `/jailbreak/` and `/deepteam/jailbreak/`
3. **Missing OpenAPI Descriptions**: Some endpoints lack detailed descriptions

### 7.3 Performance Considerations

1. **Cache Hit Rate Monitoring**: Cache metrics endpoint available but alerting not configured
2. **Connection Pool Sizing**: Default connection pool sizes may need tuning for high load
3. **Background Task Monitoring**: AutoAdv and batch jobs run as background tasks without progress persistence

### 7.4 Documentation Gaps

1. **OpenAPI Security Schemes**: Both `ApiKeyAuth` and `BearerAuth` defined but not consistently documented per endpoint
2. **Rate Limit Headers**: `Retry-After` header documentation missing
3. **WebSocket Protocols**: WebSocket message formats not fully documented

---

## 8. OpenAPI Tags Reference

| Tag | Description |
|-----|-------------|
| `generation` | Text generation endpoints |
| `transformation` | Prompt transformation endpoints |
| `jailbreak` | Jailbreak generation endpoints |
| `autodan` | AutoDAN jailbreak endpoints |
| `autodan-turbo` | AutoDAN-Turbo lifelong learning |
| `autoadv` | AutoAdv adversarial endpoints |
| `gptfuzz` | GPTFuzz fuzzing endpoints |
| `providers` | Provider management endpoints |
| `session` | Session management endpoints |
| `model-sync` | Model synchronization (WebSocket) |
| `health` | Health check endpoints |
| `integration` | Provider integration health |
| `utils` | Utility endpoints |
| `admin` | Administrative endpoints |
| `streaming` | Streaming endpoints |

---

## 9. Appendix: Provider Models

### Google/Gemini Models
- gemini-3-pro-preview, gemini-3-pro-image-preview
- gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
- gemini-2.0-flash, gemini-2.0-flash-lite
- gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b

### OpenAI Models
- gpt-4o, gpt-4o-mini, gpt-4o-2024-11-20
- gpt-4-turbo, gpt-4-turbo-preview
- gpt-4, gpt-4-32k
- gpt-3.5-turbo, gpt-3.5-turbo-16k
- o1, o1-preview, o1-mini

### Anthropic Models
- claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307

### DeepSeek Models
- deepseek-chat (V3.2 non-thinking)
- deepseek-reasoner (V3.2 thinking mode)

### Qwen Models
- qwen-max, qwen-max-longcontext
- qwen-plus, qwen-turbo
- qwen-vl-max, qwen-vl-plus

---

*Report generated automatically as part of the Chimera Backend-Frontend Gap Analysis Project, Phase 1.*