# Story 1.2: Direct API Integration

**Story ID:** MP-002  
**Epic:** Multi-Provider LLM Integration  
**Priority:** High  
**Status:** ✅ COMPLETED  
**Points:** 8  

---

## Story Description

As a **system**, I need to **communicate directly with LLM provider APIs** so that **prompts can be sent and responses received without intermediate services**.

---

## Acceptance Criteria

### AC1: Provider Client Implementations ✅
- [x] LLMProvider abstract interface defined with required methods
- [x] Google Gemini client implementation with full async support
- [x] OpenAI client implementation with streaming support
- [x] Anthropic Claude client implementation
- [x] DeepSeek client implementation
- [x] Qwen client implementation (NEW)
- [x] Cursor client implementation (NEW)

### AC2: Streaming and Non-Streaming Support ✅
- [x] All providers support non-streaming text generation
- [x] All providers support streaming text generation via `generate_stream()`
- [x] Streaming yields `StreamChunk` objects with text and metadata
- [x] Token counting support via `count_tokens()` method

### AC3: Retry Logic with Exponential Backoff ✅
- [x] Centralized `RetryHandler` with configurable strategies
- [x] Exponential, linear, constant, and Fibonacci backoff strategies
- [x] Jitter support to prevent thundering herd
- [x] Provider-specific retry configurations
- [x] Rate limit detection and respect for `retry-after` headers

### AC4: Provider Failover Mechanism ✅
- [x] `ProviderManager` with priority-based provider ordering
- [x] Automatic failover to backup providers on failure
- [x] Health check tracking per provider
- [x] Rate limit state tracking and cooldown periods
- [x] Provider status management (HEALTHY, DEGRADED, UNHEALTHY, RATE_LIMITED)

### AC5: Rate Limit Tracking ✅
- [x] Redis-based distributed rate limiting (`RedisRateLimiter`)
- [x] Sliding window algorithm for accurate rate calculations
- [x] Local fallback when Redis is unavailable
- [x] Per-provider rate limit tracking in `ProviderManager`

---

## Technical Implementation

### Files Created

| File | Purpose |
|------|---------|
| `backend-api/app/infrastructure/qwen_client.py` | Qwen (通义千问) provider via DashScope API |
| `backend-api/app/infrastructure/cursor_client.py` | Cursor AI provider |
| `backend-api/app/infrastructure/retry_handler.py` | Centralized retry logic with backoff strategies |
| `backend-api/app/infrastructure/provider_manager.py` | Provider failover and health management |
| `backend-api/app/infrastructure/__init__.py` | Infrastructure module exports |
| `backend-api/tests/test_direct_api_integration.py` | Comprehensive test suite |

### Files Enhanced

| File | Enhancement |
|------|-------------|
| `backend-api/app/domain/interfaces.py` | LLMProvider abstract base class |
| `backend-api/app/infrastructure/redis_rate_limiter.py` | Distributed rate limiting |
| `backend-api/app/services/llm_service.py` | Service layer with caching and circuit breaker |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Service Layer                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LLMService                                           │ │
│  │  - Response caching                                   │ │
│  │  - Request deduplication                              │ │
│  │  - Circuit breaker pattern                            │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Provider Manager                         │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  ProviderManager                                      │ │
│  │  - Provider registration & priority                   │ │
│  │  - Automatic failover                                 │ │
│  │  - Health tracking                                    │ │
│  │  - Rate limit state management                        │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Retry Handler                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  RetryHandler                                         │ │
│  │  - Exponential/Linear/Fibonacci backoff               │ │
│  │  - Jitter for thundering herd prevention              │ │
│  │  - Rate limit header respect                          │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM Provider Clients                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Gemini  │ │ OpenAI  │ │Anthropic│ │DeepSeek │           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│  ┌─────────┐ ┌─────────┐                                   │
│  │  Qwen   │ │ Cursor  │                                   │
│  └─────────┘ └─────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

### LLMProvider Interface

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: PromptRequest) -> PromptResponse:
        """Generate text from a prompt (non-streaming)."""
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the provider is healthy."""
        pass
    
    async def generate_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """Generate text with streaming support."""
        raise NotImplementedError("Streaming not supported")
    
    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens in text."""
        raise NotImplementedError("Token counting not supported")
```

### Retry Configuration

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.25
    strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    
    # Specific error handling
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True
    
    # Rate limit specific
    respect_retry_after: bool = True
    rate_limit_max_wait: float = 120.0
```

### Provider-Specific Defaults

| Provider | Max Retries | Initial Delay | Max Delay |
|----------|-------------|---------------|-----------|
| Google/Gemini | 3 | 1.0s | 30s |
| OpenAI | 5 | 1.0s | 60s |
| Anthropic | 4 | 1.5s | 45s |
| DeepSeek | 3 | 1.0s | 30s |
| Qwen | 3 | 1.0s | 30s |
| Cursor | 3 | 1.0s | 30s |

---

## API Endpoints

### Provider Health Check

```http
GET /api/providers/health
```

Returns health status for all registered providers.

### Provider List

```http
GET /api/providers
```

Returns list of available providers with their status and models.

### Generate Text (Non-Streaming)

```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "Hello, world!",
  "provider": "gemini",
  "model": "gemini-2.0-flash-exp",
  "config": {
    "temperature": 0.7,
    "max_output_tokens": 2048
  }
}
```

### Generate Text (Streaming)

```http
POST /api/generate/stream
Content-Type: application/json

{
  "prompt": "Write a story",
  "provider": "openai",
  "model": "gpt-4o",
  "stream": true
}
```

---

## Testing

### Test Coverage

- **Unit Tests:** Individual provider client tests
- **Integration Tests:** Failover chain, retry logic
- **Mock Tests:** Provider behavior simulation

### Running Tests

```bash
# Run all Story 1.2 tests
poetry run pytest backend-api/tests/test_direct_api_integration.py -v

# Run with coverage
poetry run pytest backend-api/tests/test_direct_api_integration.py --cov=app/infrastructure
```

### Test Categories

| Category | Tests |
|----------|-------|
| Interface Compliance | 4 tests |
| Retry Handler | 10 tests |
| Provider Manager | 8 tests |
| Qwen Client | 5 tests |
| Cursor Client | 4 tests |
| Integration | 3 tests |
| Backoff Strategies | 3 tests |

---

## Dependencies

### Python Packages

- `openai>=1.0.0` - OpenAI, DeepSeek, Qwen, Cursor clients
- `anthropic>=0.18.0` - Anthropic client
- `google-genai>=1.0.0` - Google Gemini client
- `tiktoken>=0.5.0` - Token counting
- `redis>=4.0.0` - Distributed rate limiting

### Environment Variables

```env
# Provider API Keys
GOOGLE_API_KEY=your-google-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
QWEN_API_KEY=your-qwen-key
CURSOR_API_KEY=your-cursor-key

# Redis (for rate limiting)
REDIS_URL=redis://localhost:6379/0
```

---

## Related Stories

| Story | Dependency |
|-------|------------|
| Story 1.1: Provider Configuration | Prerequisites - API key management |
| Story 1.3: Proxy Integration | Alternative connection mode |
| Story 1.4: Provider Health Monitoring | Enhanced health checks |

---

## Changelog

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-01-02 | 1.0.0 | System | Initial implementation |
| 2025-01-02 | 1.1.0 | System | Added Qwen and Cursor providers |
| 2025-01-02 | 1.2.0 | System | Added retry handler and provider manager |