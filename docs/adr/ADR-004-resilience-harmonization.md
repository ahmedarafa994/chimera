# ADR-004: Frontend/Backend Resilience Harmonization

## Status
Accepted

## Date
2026-01-02

## Context

The Chimera system operates in two modes:

1. **Proxy Mode** (Production): Frontend → Backend API → LLM Providers
2. **Direct Mode** (Development): Frontend → LLM Providers directly

The PROJECT_REVIEW_REPORT.md identified a critical issue: both the frontend `APIClient` and the backend had independent circuit breaker and retry logic. When operating in proxy mode, this created "double-wrapping" where:

- Frontend retries a failed request 3 times
- Each retry goes to the backend, which retries 3 times to the LLM provider
- Result: Up to 9x amplification of requests during failures ("thundering herd")

Additionally, both layers independently opened their circuit breakers, leading to inconsistent failure states and confusing error messages for users.

## Decision

**Implement mode-aware resilience configuration in the frontend:**

1. **Proxy Mode**: Disable frontend circuit breaker, minimize retries (maxRetries=1)
2. **Direct Mode**: Enable full resilience (circuit breaker + 3 retries)
3. **Security**: Block direct mode in production to prevent API key exposure

### Configuration Model

```typescript
type ApiMode = 'direct' | 'proxy';

interface ResilienceConfig {
  enableCircuitBreaker: boolean;
  enableRetry: boolean;
  maxRetries: number;
}

// Proxy mode: delegate resilience to backend
{ enableCircuitBreaker: false, enableRetry: true, maxRetries: 1 }

// Direct mode: full frontend resilience
{ enableCircuitBreaker: true, enableRetry: true, maxRetries: 3 }
```

### Security Enforcement

```typescript
setMode(mode: ApiMode): void {
  if (mode === 'direct' && this.config.environment === 'production') {
    throw new Error(
      'Direct mode is not allowed in production. ' +
      'Use proxy mode to keep API keys secure on the server.'
    );
  }
  // ...
}
```

## Consequences

### Positive
- **Eliminates request amplification**: Max 1x3=3 retries instead of 3x3=9
- **Single source of truth**: Backend circuit breaker state is authoritative
- **Consistent error handling**: Users see backend-generated errors, not frontend guesses
- **API key security**: Production enforces proxy mode, keeping secrets server-side
- **Reactive configuration**: `APIClient` subscribes to config changes via Zustand

### Negative
- **Development complexity**: Developers must understand mode differences
- **Local testing**: Direct mode only works with local `.env` API keys

### Neutral
- **Migration path**: Existing code continues to work; resilience degrades gracefully

## Implementation

### Files Modified

1. **`frontend/src/lib/api/core/config.ts`**
   - Added `ApiMode`, `ProxyConfig`, `ResilienceConfig` types
   - `getDefaultApiMode()`: Returns 'proxy' in production
   - `getResilienceConfig()`: Computes resilience based on mode
   - `ConfigManager.setMode()`: Validates and updates mode
   - `ConfigManager.isProxyMode()`, `shouldEnableCircuitBreaker()`: Convenience methods

2. **`frontend/src/lib/api/core/client.ts`**
   - `getDefaultClientConfig()`: Reads from `configManager.getResilienceConfig()`
   - Constructor subscribes to config changes
   - `destroy()`: Cleanup for subscription
   - `isCircuitBreakerEnabled()`, `isProxyMode()`: Diagnostic methods

### Usage Example

```typescript
// Check current mode
if (configManager.isProxyMode()) {
  console.log('Using backend for resilience');
}

// Check if circuit breaker is active
const client = new APIClient();
if (!client.isCircuitBreakerEnabled()) {
  console.log('Circuit breaker delegated to backend');
}

// Cleanup when component unmounts
client.destroy();
```

## References

- PROJECT_REVIEW_REPORT.md: Section on frontend/backend duplication
- Circuit Breaker Pattern: https://martinfowler.com/bliki/CircuitBreaker.html
- ADR-001: Frontend API Refactoring (TanStack Query integration)