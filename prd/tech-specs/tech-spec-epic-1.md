# Technical Specification: Multi-Provider Foundation

Date: 2026-01-02
Author: BMAD USER
Epic ID: Epic 1
Status: Draft

---

## Overview

Epic 1 establishes the foundational multi-provider LLM integration infrastructure for Chimera, enabling robust communication with Google Gemini, OpenAI, Anthropic Claude, and DeepSeek providers. This epic implements both direct API and proxy mode (AIClient-2-API) support, comprehensive health monitoring, circuit breaker patterns for production reliability, and a basic generation endpoint that serves as the entry point for all AI interactions.

## Objectives and Scope

**Objectives:**
- Establish multi-provider configuration management with encrypted API key storage
- Implement dual-mode connectivity (direct API and proxy mode)
- Build comprehensive provider health monitoring with metrics
- Deploy circuit breaker pattern for fault tolerance and automatic failover
- Create baseline generation endpoint with full parameter support
- Deliver provider management UI for selection and monitoring

**Scope:**
- 7 user stories covering provider configuration, dual-mode integration, health monitoring, circuit breaking, generation endpoint, and UI
- Supports 6 LLM providers: Google/Gemini, OpenAI, Anthropic/Claude, Qwen, DeepSeek, Cursor
- Provider configuration with hot-reload capability
- Circuit breaker state machine (CLOSED, OPEN, HALF_OPEN)
- Health metrics: latency, error rates, availability
- Provider selection UI with real-time status

**Out of Scope:**
- Transformation techniques (Epic 2)
- WebSocket real-time updates (Epic 3)
- Analytics pipeline (Epic 4)
- Cross-model strategy analysis (Epic 5)

## System Architecture Alignment

Epic 1 implements the **Provider Integration Layer** from the solution architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8001)              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌─────────────────────────────┐ │
│  │ Provider Config  │─────▶│  Circuit Breaker Manager    │ │
│  │  (encrypted)     │      │  (per-provider state)       │ │
│  └────────┬─────────┘      └───────────┬─────────────────┘ │
│           │                             │                   │
│           ▼                             ▼                   │
│  ┌──────────────────────────────────────────────────────┐ │
│  │          Multi-Provider LLM Service                  │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │ │
│  │  │ Google  │ │ OpenAI  │ │Anthropic│ │ DeepSeek│    │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │ │
│  └───────┼───────────┼───────────┼───────────┼─────────┘ │
│          │           │           │           │           │
│  ┌───────┴───────────┴───────────┴───────────┴─────────┐ │
│  │         Connection Mode Selector                    │ │
│  │    (API_CONNECTION_MODE: direct | proxy)            │ │
│  └───────┬──────────────────────────────────────────────┘ │
│           │                                                 │
│           ├─ Direct Mode ──────────▶ Provider APIs        │
│           │                                                 │
│           └─ Proxy Mode ───────────▶ AIClient-2-API        │
│                                          (localhost:8080) │
└─────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions Referenced:**
- **ADR-001**: Monolithic Full-Stack Architecture
- **ADR-002**: Separate Frontend/Backend Deployment

## Detailed Design

### Services and Modules

**Backend Services:**

1. **`app/core/config.py`** - Centralized Configuration Management
   - Environment-based configuration with precedence: env vars > config file > defaults
   - Encrypted API key storage (AES-256 at rest)
   - Hot-reload capability for runtime configuration changes
   - `API_CONNECTION_MODE` setting for direct/proxy selection
   - Provider-specific configurations: base URLs, model lists, rate limits

2. **`app/domain/interfaces.py`** - Provider Interface Definition
   ```python
   class LLMProvider(Protocol):
       async def generate(self, request: PromptRequest) -> PromptResponse: ...
       async def health_check(self) -> HealthStatus: ...
       def get_models(self) -> List[str]: ...
   ```

3. **`app/infrastructure/providers/`** - Provider Implementations
   - `google_provider.py` - Google Gemini integration
   - `openai_provider.py` - OpenAI GPT integration
   - `anthropic_provider.py` - Anthropic Claude integration
   - `deepseek_provider.py` - DeepSeek integration
   - Each implements `LLMProvider` interface with native API format handling

4. **`app/core/circuit_breaker.py`** - Circuit Breaker State Machine
   ```python
   class CircuitBreaker:
       # States: CLOSED (normal), OPEN (failing), HALF_OPEN (testing)
       failure_threshold: int = 3
       recovery_timeout: int = 60  # seconds
       async def call(self, provider, method, *args, **kwargs): ...
   ```

5. **`app/services/llm_service.py`** - Multi-Provider Orchestration
   - Provider registry with default provider selection
   - Request routing with automatic failover
   - Usage tracking (tokens, latency, costs)
   - Circuit breaker integration

6. **`app/services/integration_health_service.py`** - Health Monitoring
   - Per-provider health checks every 30 seconds (configurable)
   - Metrics: latency (p50, p95, p99), error rates, availability
   - Health status history for trend analysis
   - Prometheus metrics integration

7. **`app/api/v1/endpoints/generation.py`** - Generation Endpoint
   - `POST /api/v1/generate` - Main generation endpoint
   - `GET /api/v1/providers` - List available providers and models
   - `GET /api/v1/session/models` - Get current session models
   - Request validation with Pydantic models

**Frontend Components:**

1. **`src/app/dashboard/providers/page.tsx`** - Provider Management UI
   - Provider cards with status indicators (healthy/unhealthy)
   - Model selection dropdowns per provider
   - Provider metrics display (latency, success rate, request counts)
   - Connectivity test button with sample request
   - Real-time health status updates

### Data Models and Contracts

**Request Models (Pydantic):**

```python
class PromptRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None  # Uses default if not specified
    model: Optional[str] = None     # Uses provider default if not specified
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=32000)
    stream: bool = False

class ProviderConfig(BaseModel):
    provider_id: str
    api_key_encrypted: str
    base_url: str
    models: List[str]
    enabled: bool
```

**Response Models:**

```python
class PromptResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: TokenUsage
    timing: TimingInfo
    metadata: Dict[str, Any]

class ProviderHealth(BaseModel):
    provider_id: str
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: float
    error_rate: float
    uptime_percentage: float
    last_check: datetime
```

**Frontend TypeScript Types:**

```typescript
interface Provider {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  models: string[];
  metrics: ProviderMetrics;
}

interface GenerationRequest {
  prompt: string;
  provider?: string;
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
}
```

### APIs and Interfaces

**Backend API Endpoints:**

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/v1/providers` | List all configured providers with models and status | API Key |
| GET | `/api/v1/session/models` | Get available models for current session | API Key |
| POST | `/api/v1/generate` | Generate text using specified or default provider | API Key |
| GET | `/health/integration` | Provider health status and dependency graph | Internal |
| GET | `/health` | Basic liveness check | None |
| GET | `/health/ready` | Readiness probe with dependency checks | None |

**WebSocket Protocol:**

- **Endpoint:** `WS /ws/enhance`
- **Message Format:** JSON with `type` and `payload` fields
- **Heartbeat:** Every 30 seconds to maintain connection

**Frontend API Client:**

```typescript
// src/lib/api-client.ts
class ProviderClient {
  async getProviders(): Promise<Provider[]> { ... }
  async getModels(providerId: string): Promise<string[]> { ... }
  async generate(request: GenerationRequest): Promise<GenerationResponse> { ... }
  async testConnection(providerId: string): Promise<boolean> { ... }
}
```

### Workflows and Sequencing

**Provider Registration Flow:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ Application │    │  Lifespan   │    │  LLM Service    │
│   Startup   │───▶│   Hook      │───▶│   Register      │
└─────────────┘    └─────────────┘    └────────┬────────┘
                                              │
         ┌────────────────────────────────────┴──────────────┐
         │                                                     │
         ▼                                                     ▼
┌─────────────────┐                                ┌─────────────────┐
│  Direct Mode    │                                │   Proxy Mode    │
│  API_KEYS set   │                                │  localhost:8080 │
└────────┬────────┘                                └────────┬────────┘
         │                                                   │
         ▼                                                   ▼
┌─────────────────┐                                ┌─────────────────┐
│ Google Provider  │                                │ AIClient-2-API  │
│ OpenAI Provider  │                                │   Adapter       │
│ Anthropic Prov.  │                                └─────────────────┘
│ DeepSeek Prov.   │
└─────────────────┘
```

**Generation Request Flow with Circuit Breaker:**

```
┌─────────┐   POST   ┌──────────────┐   Select   ┌─────────────────┐
│ Client  │─────────▶│ /api/v1/     │───────────▶│ Primary Provider │
│         │         │ generate     │            │ (circuit: CLOSED)│
└─────────┘         └──────┬───────┘            └────────┬────────┘
                           │                              │
                           │                              ▼
                           │                     ┌─────────────────┐
                           │                     │ Call Provider   │
                           │                     │ API             │
                           │                     └────────┬────────┘
                           │                              │
                           │         ┌──────────────────────┘
                           │         │
                           │    ┌────┴────┐
                           │    │ Success?│
                           │    └────┬────┘
                           │         │
              ┌────────────┴─────────┴────────┐
              │                              │
             Yes                            No
              │                              │
              ▼                              ▼
      ┌─────────────┐               ┌─────────────────┐
      │ Return      │               │ Circuit: OPEN   │
      │ Response    │               │ Failover to      │
      └─────────────┘               │ Alt Provider     │
                                    └─────────────────┘
```

**Health Monitoring Loop:**

```
Every 30 seconds (configurable):
    For each provider:
        1. Send health check request
        2. Measure latency
        3. Update health status
        4. Check error rate threshold
        5. Trigger circuit breaker if needed
        6. Update Prometheus metrics
        7. Emit WebSocket update to connected clients
```

## Non-Functional Requirements

### Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Health Check Latency | <100ms | p50 latency |
| Generation Latency | <2s | Typical non-streaming request |
| Provider Switch Time | <500ms | Circuit breaker failover |
| Config Reload Time | <1s | Hot-reload without restart |
| Concurrent Requests | 100+ | Simultaneous API calls |

**Performance Optimization:**
- Async/await throughout request handling
- Connection pooling for HTTP clients
- Provider request caching where appropriate
- Circuit breaker prevents cascading failures

### Security

| Aspect | Implementation |
|--------|----------------|
| API Key Storage | AES-256 encryption at rest |
| API Key Transmission | TLS 1.3 for all external requests |
| Authentication | X-API-Key header with JWT support |
| Input Validation | Pydantic models with strict validation |
| Secret Rotation | Support for key rotation without restart |
| Audit Logging | All provider requests logged with sanitized data |

**Security Considerations:**
- API keys never logged in plaintext
- Proxy mode adds internal routing layer
- Circuit breaker prevents provider exhaustion attacks
- Rate limiting per provider (configurable)

### Reliability/Availability

| Metric | Target | Mechanism |
|--------|--------|-----------|
| Backend Uptime | 99.9% | Health checks and auto-restart |
| Provider Availability | Multi-provider redundancy with automatic failover |
| Circuit Breaker Recovery | 60s timeout | Automatic HALF_OPEN retry |
| Health Check Coverage | 100% of providers | Continuous monitoring |
| Graceful Degradation | Degraded mode with partial providers | Circuit breaker pattern |

**Reliability Features:**
- Circuit breaker prevents cascade failures
- Automatic failover to healthy providers
- Health monitoring with proactive alerts
- Retry with exponential backoff for transient failures

### Observability

| Aspect | Implementation |
|--------|----------------|
| Metrics | Prometheus exports for latency, error rates, usage |
| Logging | Structured JSON logs with request IDs |
| Tracing | Request ID propagation through all services |
| Health Endpoints | `/health`, `/health/ready`, `/health/integration` |
| Dashboards | Provider health, circuit breaker state, request metrics |

**Observable Metrics:**
- `provider_requests_total` - Per-provider request count
- `provider_latency_seconds` - Request latency by provider
- `provider_errors_total` - Error count by provider
- `circuit_breaker_state` - Current state per provider
- `health_check_status` - Health status by provider

## Dependencies and Integrations

**Internal Dependencies:**
- Epic 1 is foundational for all other epics
- Epic 3 (Real-Time Research Platform) builds on this provider layer
- Epic 5 (Cross-Model Intelligence) uses batch execution from this foundation

**External Dependencies:**

| Dependency | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.104+ | Web framework |
| Pydantic | 2.0+ | Request/response validation |
| httpx | 0.25+ | Async HTTP client |
| cryptography | 41.0+ | API key encryption |
| prometheus-client | 0.19+ | Metrics export |
| Next.js | 16 | Frontend framework |
| React | 19 | UI library |
| TanStack Query | 5.0+ | Data fetching and caching |

**Provider API Dependencies:**
- Google Gemini API - https://generativelanguage.googleapis.com
- OpenAI API - https://api.openai.com
- Anthropic Claude API - https://api.anthropic.com
- DeepSeek API - https://api.deepseek.com

**Optional Proxy:**
- AIClient-2-API Server - localhost:8080 (proxy mode only)

## Acceptance Criteria (Authoritative)

### Story MP-001: Provider Configuration Management
- [ ] Multiple providers configurable with separate sections
- [ ] API keys encrypted at rest using AES-256 or equivalent
- [ ] Both proxy and direct mode configuration supported
- [ ] Configuration validation verifies API key format and connectivity
- [ ] Invalid configurations provide clear error messages with remediation
- [ ] Provider configuration hot-reloadable without application restart

### Story MP-002: Direct API Integration
- [ ] Direct API mode routes requests to provider endpoints
- [ ] Each provider uses native API format and authentication
- [ ] Streaming and non-streaming modes supported
- [ ] Retry logic with exponential backoff for transient failures
- [ ] Response times meet benchmarks (<100ms health, <2s generation)
- [ ] Provider-specific rate limits respected and tracked
- [ ] Connection errors trigger failover to alternative providers

### Story MP-003: Proxy Mode Integration
- [ ] Proxy mode routes all requests through AIClient-2-API Server (localhost:8080)
- [ ] Proxy server handles provider-specific transformations
- [ ] Proxy communication uses efficient protocol when available
- [ ] Proxy server failures trigger graceful fallback or error handling
- [ ] Proxy mode supports all providers consistently
- [ ] Proxy health monitoring detects and reports proxy server status

### Story MP-004: Provider Health Monitoring
- [ ] Each provider has health status tracked (uptime, latency, error rates)
- [ ] Health checks run at configurable intervals (default: 30 seconds)
- [ ] Unhealthy providers marked for circuit breaker activation
- [ ] Health metrics exposed via `/health/integration` endpoint
- [ ] Provider health history maintained for trend analysis
- [ ] Health degradation triggers alerts before complete failure
- [ ] Health status visible in dashboard

### Story MP-005: Circuit Breaker Pattern
- [ ] Circuit breaker transitions to OPEN after 3 consecutive failures
- [ ] Requests stop routing to failed provider when OPEN
- [ ] Requests automatically failover to healthy alternative providers
- [ ] After 60 seconds, circuit attempts HALF_OPEN state
- [ ] Successful requests in HALF_OPEN transition circuit to CLOSED
- [ ] Continued failures keep circuit OPEN with backoff
- [ ] Circuit state visible in monitoring and logs

### Story MP-006: Basic Generation Endpoint
- [ ] `POST /api/v1/generate` processes requests with selected/default provider
- [ ] Supports parameters: model, temperature, top_p, max_tokens
- [ ] Response includes generated text and usage metadata (tokens, timing)
- [ ] Streaming and non-streaming modes supported
- [ ] Response times meet targets (<2s for typical requests)
- [ ] Errors provide clear, actionable messages
- [ ] Requests logged for audit and debugging

### Story MP-007: Provider Selection UI
- [ ] All configured providers displayed with status (healthy/unhealthy)
- [ ] Default provider selection capability
- [ ] Available models shown for each provider
- [ ] Provider metrics displayed (latency, success rate, request counts)
- [ ] Provider connectivity testable with sample request
- [ ] UI updates in real-time as provider health changes
- [ ] Provider selection persists across sessions

## Traceability Mapping

**Requirements from PRD:**

| PRD Requirement | Epic 1 Story | Implementation |
|----------------|--------------|----------------|
| FR-01: Multi-provider support | MP-001, MP-002, MP-003 | Provider configuration and dual-mode |
| FR-02: Provider health monitoring | MP-004 | Integration health service |
| FR-03: Automatic failover | MP-005 | Circuit breaker pattern |
| NFR-01: <100ms API response | MP-006 | Health check latency target |
| NFR-02: 99.9% uptime | MP-004, MP-005 | Health monitoring + circuit breaker |
| NFR-04: 100+ concurrent users | MP-006 | Async request handling |

**Epic-to-Architecture Mapping:**

| Architecture Component | Epic 1 Implementation |
|-----------------------|----------------------|
| Provider Integration Layer | All stories (MP-001 through MP-007) |
| Configuration Management | MP-001 |
| Circuit Breaker Pattern | MP-005 |
| Health Monitoring | MP-004 |
| API Gateway | MP-006 |
| Provider Management UI | MP-007 |

## Risks, Assumptions, Open Questions

**Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Provider API rate limits exceeded | High | Implement per-provider rate limiting and circuit breaker |
| Proxy server (AIClient-2-API) becomes SPOF in proxy mode | Medium | Graceful fallback to direct mode if proxy unavailable |
| Encrypted API key decryption failure | High | Secure key backup and rotation procedures |
| Circuit breaker false positives | Medium | Configurable thresholds and manual override capability |

**Assumptions:**
- All provider APIs are accessible from deployment environment
- AIClient-2-API Server is stable and well-documented for proxy mode
- Provider API documentation is accurate and up-to-date
- Encryption libraries meet security compliance requirements

**Open Questions:**
- Should provider-specific rate limits be hard-coded or configurable via UI? → **Decision: Configurable via API and UI**
- What is the maximum acceptable lag for health check updates in UI? → **Decision: 30 seconds is acceptable; can be reduced to 10s for production**
- Should failed providers be automatically re-enabled after recovery? → **Decision: Yes, via HALF_OPEN circuit breaker state**

## Test Strategy Summary

**Unit Tests:**
- Configuration loading and validation
- API key encryption/decryption
- Circuit breaker state transitions
- Health check logic and metrics calculation
- Provider interface implementations (mocked)

**Integration Tests:**
- Direct mode provider integration (with test API keys)
- Proxy mode routing (with local AIClient-2-API)
- Circuit breaker failover scenarios
- Health check endpoint responses
- Generation endpoint with various parameters

**End-to-End Tests:**
- Multi-provider configuration and selection
- Provider failover during outage simulation
- Configuration hot-reload without restart
- Full request flow from UI to provider API

**Performance Tests:**
- Concurrent request handling (100+ simultaneous)
- Circuit breaker recovery timing
- Health check overhead on provider latency
- Configuration reload performance

**Security Tests:**
- API key encryption verification
- TLS connection validation
- Input sanitization on generation endpoint
- Authentication bypass attempts

**Test Coverage Target:** 80%+ for backend services, 70%+ for frontend components

---

_This technical specification serves as the implementation guide for Epic 1: Multi-Provider Foundation. All development should reference this document for detailed design decisions, API contracts, and acceptance criteria._
