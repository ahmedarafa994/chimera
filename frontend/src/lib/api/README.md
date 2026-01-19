# Chimera API Integration Architecture

A comprehensive, production-ready API integration layer for the Chimera project.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Migration Guide](#migration-guide)
- [API Reference](#api-reference)

## Overview

This API integration architecture provides:

- **Centralized Configuration**: Environment-based settings for development, staging, and production
- **Authentication & Security**: Token management, API key handling, and request signing
- **Performance Optimization**: Caching, request deduplication, batching, and connection pooling
- **Error Handling & Resilience**: Circuit breakers, retry logic with exponential backoff, and graceful degradation
- **Type Safety**: Comprehensive TypeScript interfaces for all API operations
- **Testing Infrastructure**: Mock servers, fixtures, and test utilities
- **Monitoring & Observability**: Request metrics, health checks, and alerting

## Quick Start

### Basic Usage

```typescript
import { apiClient, configManager } from '@/lib/api';

// Make a simple GET request
const response = await apiClient.get('/api/v1/health');

// Make a POST request with data
const chatResponse = await apiClient.post('/api/v1/chat', {
  messages: [{ role: 'user', content: 'Hello!' }],
  model: 'gpt-3.5-turbo',
});

// Use provider-specific endpoints
const geminiResponse = await apiClient.providerRequest('gemini', '/chat', {
  method: 'POST',
  data: { prompt: 'Hello from Gemini!' },
});
```

### With Error Handling

```typescript
import { apiClient, APIError, isNetworkError, isRateLimitError } from '@/lib/api';

try {
  const response = await apiClient.get('/api/v1/providers');
  console.log('Providers:', response.data);
} catch (error) {
  if (error instanceof APIError) {
    if (isRateLimitError(error)) {
      console.log('Rate limited, retry after:', error.retryAfter);
    } else if (isNetworkError(error)) {
      console.log('Network error, check connection');
    } else {
      console.log('API error:', error.message);
    }
  }
}
```

### With Caching

```typescript
import { apiClient, cacheManager } from '@/lib/api';

// Request with caching (default TTL)
const response = await apiClient.get('/api/v1/models', {
  cache: true,
});

// Request with custom cache TTL
const response = await apiClient.get('/api/v1/techniques', {
  cache: true,
  cacheTTL: 60000, // 1 minute
});

// Invalidate cache
cacheManager.invalidate('/api/v1/models');
cacheManager.invalidatePattern(/\/api\/v1\/providers/);
```

## Architecture

```
src/lib/api/
├── core/
│   ├── config.ts          # Configuration management
│   ├── auth.ts            # Authentication & token management
│   ├── client.ts          # Main API client
│   ├── errors.ts          # Error types and handling
│   ├── retry.ts           # Retry logic with exponential backoff
│   ├── circuit-breaker.ts # Circuit breaker pattern
│   ├── request-deduplication.ts # Request deduplication & batching
│   ├── monitoring.ts      # Metrics collection & health monitoring
│   ├── types.ts           # TypeScript interfaces
│   └── index.ts           # Core module exports
├── testing/
│   ├── mocks.ts           # Mock server & fixtures
│   ├── test-utils.ts      # Testing utilities
│   └── index.ts           # Testing module exports
├── migration/
│   └── compat.ts          # Backward compatibility layer
├── index.ts               # Main entry point
└── README.md              # This documentation
```

## Core Modules

### Configuration (`config.ts`)

Manages environment-based configuration for API endpoints, timeouts, and feature flags.

```typescript
import { configManager, Environment } from '@/lib/api';

// Get current environment
const env = configManager.getEnvironment(); // 'development' | 'staging' | 'production'

// Get active base URL
const baseUrl = configManager.getActiveBaseUrl();

// Get provider configuration
const geminiConfig = configManager.getProviderConfig('gemini');

// Update configuration
configManager.updateConfig({
  defaultTimeout: 60000,
  enableCaching: true,
});
```

### Authentication (`auth.ts`)

Handles token management, API keys, and request authentication.

```typescript
import { tokenManager, apiKeyManager } from '@/lib/api';

// Set authentication token
tokenManager.setToken('your-jwt-token');

// Token is automatically refreshed when needed
// Configure refresh behavior:
tokenManager.setRefreshCallback(async () => {
  const response = await fetch('/api/auth/refresh');
  const data = await response.json();
  return data.token;
});

// Manage API keys
apiKeyManager.setKey('gemini', 'your-gemini-api-key');
apiKeyManager.setKey('openai', 'your-openai-api-key');
```

### Error Handling (`errors.ts`)

Provides standardized error types and utilities.

```typescript
import {
  APIError,
  NetworkError,
  TimeoutError,
  RateLimitError,
  AuthenticationError,
  ValidationError,
  isRetryableError,
  mapBackendError,
} from '@/lib/api';

// Error types
class APIError extends Error {
  code: string;
  status?: number;
  details?: unknown;
  retryAfter?: number;
}

// Type guards
if (isRetryableError(error)) {
  // Safe to retry
}

// Map backend errors to typed errors
const typedError = mapBackendError(backendResponse);
```

### Retry Logic (`retry.ts`)

Implements retry with exponential backoff and jitter.

```typescript
import { executeWithRetry, DEFAULT_RETRY_CONFIG } from '@/lib/api';

// Execute with default retry config
const result = await executeWithRetry(async () => {
  return await apiClient.get('/api/v1/data');
});

// Custom retry configuration
const result = await executeWithRetry(
  async () => apiClient.get('/api/v1/data'),
  {
    maxRetries: 5,
    baseDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
    jitterFactor: 0.1,
    retryableStatusCodes: [408, 429, 500, 502, 503, 504],
  }
);
```

### Circuit Breaker (`circuit-breaker.ts`)

Prevents cascading failures with the circuit breaker pattern.

```typescript
import { CircuitBreaker, circuitBreakerRegistry } from '@/lib/api';

// Create a circuit breaker
const breaker = new CircuitBreaker('api-service', {
  failureThreshold: 5,
  resetTimeout: 30000,
  halfOpenRequests: 3,
});

// Execute with circuit breaker
const result = await breaker.execute(async () => {
  return await apiClient.get('/api/v1/data');
});

// Check circuit state
console.log('Circuit state:', breaker.getState()); // 'CLOSED' | 'OPEN' | 'HALF_OPEN'

// Use registry for multiple circuits
const providerBreaker = circuitBreakerRegistry.getCircuit('gemini');
```

### Request Deduplication (`request-deduplication.ts`)

Prevents duplicate concurrent requests and enables request batching.

```typescript
import { RequestDeduplicator, RequestBatcher } from '@/lib/api';

// Deduplication
const deduplicator = new RequestDeduplicator();
const result = await deduplicator.execute(
  'unique-request-key',
  async () => apiClient.get('/api/v1/data')
);

// Batching
const batcher = new RequestBatcher<string, UserData>({
  maxBatchSize: 10,
  maxWaitTime: 50,
  batchHandler: async (ids) => {
    const response = await apiClient.post('/api/v1/users/batch', { ids });
    return response.data;
  },
});

// Individual requests are automatically batched
const user1 = await batcher.add('user-1');
const user2 = await batcher.add('user-2');
```

### Monitoring (`monitoring.ts`)

Collects metrics and monitors API health.

```typescript
import { metricsCollector, healthMonitor } from '@/lib/api';

// Get metrics
const metrics = metricsCollector.getMetrics();
console.log('Total requests:', metrics.totalRequests);
console.log('Success rate:', metrics.successRate);
console.log('Average latency:', metrics.averageLatency);

// Health monitoring
healthMonitor.startMonitoring({
  interval: 30000,
  endpoints: ['/api/v1/health'],
  onHealthChange: (status) => {
    console.log('Health status changed:', status);
  },
});

// Get current health status
const health = healthMonitor.getStatus();
```

## Configuration

### Environment Variables

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_API_MODE=direct  # 'direct' only
NEXT_PUBLIC_ENVIRONMENT=development  # 'development' | 'staging' | 'production'

# Provider API Keys (for direct mode)
NEXT_PUBLIC_GEMINI_API_KEY=your-key
NEXT_PUBLIC_OPENAI_API_KEY=your-key
NEXT_PUBLIC_ANTHROPIC_API_KEY=your-key
NEXT_PUBLIC_DEEPSEEK_API_KEY=your-key

# Feature Flags
NEXT_PUBLIC_ENABLE_CACHING=true
NEXT_PUBLIC_ENABLE_RETRY=true
NEXT_PUBLIC_ENABLE_CIRCUIT_BREAKER=true
```

### Runtime Configuration

```typescript
import { configManager } from '@/lib/api';

configManager.updateConfig({
  // Timeouts
  defaultTimeout: 30000,
  uploadTimeout: 120000,
  
  // Retry
  enableRetry: true,
  maxRetries: 3,
  
  // Caching
  enableCaching: true,
  defaultCacheTTL: 300000,
  
  // Circuit Breaker
  enableCircuitBreaker: true,
  circuitBreakerThreshold: 5,
  
  // Logging
  enableLogging: true,
  logLevel: 'info',
});
```

## Authentication

### JWT Token Authentication

```typescript
import { tokenManager } from '@/lib/api';

// Set token (typically after login)
tokenManager.setToken(jwtToken);

// Token is automatically included in requests
// Authorization: Bearer <token>

// Set up automatic refresh
tokenManager.setRefreshCallback(async () => {
  const response = await fetch('/api/auth/refresh', {
    method: 'POST',
    credentials: 'include',
  });
  const data = await response.json();
  return data.accessToken;
});

// Clear token (on logout)
tokenManager.clearToken();
```

### API Key Authentication

```typescript
import { apiKeyManager } from '@/lib/api';

// Set provider API keys
apiKeyManager.setKey('gemini', process.env.GEMINI_API_KEY);
apiKeyManager.setKey('openai', process.env.OPENAI_API_KEY);

// Keys are automatically included for provider requests
// X-API-Key: <key> or provider-specific header
```

## Error Handling

### Error Types

| Error Type | Description | Retryable |
|------------|-------------|-----------|
| `NetworkError` | Network connectivity issues | Yes |
| `TimeoutError` | Request timeout | Yes |
| `RateLimitError` | Rate limit exceeded | Yes (with backoff) |
| `AuthenticationError` | Invalid/expired credentials | No |
| `AuthorizationError` | Insufficient permissions | No |
| `ValidationError` | Invalid request data | No |
| `NotFoundError` | Resource not found | No |
| `ServerError` | Server-side error | Yes |

### Error Handling Patterns

```typescript
import { apiClient, APIError } from '@/lib/api';

// Pattern 1: Try-catch with type guards
try {
  const response = await apiClient.get('/api/v1/data');
} catch (error) {
  if (error instanceof APIError) {
    switch (error.code) {
      case 'RATE_LIMITED':
        await delay(error.retryAfter || 1000);
        // Retry request
        break;
      case 'UNAUTHORIZED':
        // Redirect to login
        break;
      default:
        // Show error message
        console.error(error.message);
    }
  }
}

// Pattern 2: Using error utilities
import { isRetryableError, getErrorMessage } from '@/lib/api';

try {
  const response = await apiClient.get('/api/v1/data');
} catch (error) {
  if (isRetryableError(error)) {
    // Automatic retry is handled by the client
  }
  const message = getErrorMessage(error);
  showNotification({ type: 'error', message });
}
```

## Performance Optimization

### Caching Strategy

```typescript
import { cacheManager } from '@/lib/api';

// Cache configuration
cacheManager.configure({
  maxSize: 100,           // Maximum cache entries
  defaultTTL: 300000,     // 5 minutes
  cleanupInterval: 60000, // Cleanup every minute
});

// Manual cache operations
cacheManager.set('key', data, { ttl: 60000 });
const cached = cacheManager.get('key');
cacheManager.invalidate('key');
cacheManager.invalidatePattern(/^\/api\/v1\/users/);
cacheManager.clear();
```

### Request Batching

```typescript
import { RequestBatcher } from '@/lib/api';

// Configure batching for user lookups
const userBatcher = new RequestBatcher({
  maxBatchSize: 50,
  maxWaitTime: 100,
  batchHandler: async (userIds) => {
    const response = await apiClient.post('/api/v1/users/batch', { ids: userIds });
    return new Map(response.data.map(u => [u.id, u]));
  },
});

// Individual calls are automatically batched
const users = await Promise.all([
  userBatcher.add('user-1'),
  userBatcher.add('user-2'),
  userBatcher.add('user-3'),
]);
```

### Connection Pooling

The API client automatically manages connection pooling through Axios. Configuration:

```typescript
import { configManager } from '@/lib/api';

configManager.updateConfig({
  maxConcurrentRequests: 10,
  keepAlive: true,
  keepAliveMsecs: 1000,
});
```

## Testing

### Mock Server

```typescript
import { mockServer, fixtures } from '@/lib/api/testing';

// Use default mock handlers
mockServer.reset();

// Register custom handler
mockServer.register({
  method: 'GET',
  url: '/api/v1/custom',
  response: {
    data: { custom: 'data' },
    status: 200,
    statusText: 'OK',
    headers: { 'content-type': 'application/json' },
  },
});

// Dynamic response
mockServer.register({
  method: 'POST',
  url: '/api/v1/echo',
  response: (config) => ({
    data: JSON.parse(config.data),
    status: 200,
    statusText: 'OK',
    headers: {},
  }),
});
```

### Fixtures

```typescript
import { fixtures } from '@/lib/api/testing';

// Available fixtures
const healthCheck = fixtures.healthCheck();
const providers = fixtures.providers();
const models = fixtures.models('gemini');
const techniques = fixtures.techniques();
const chatResponse = fixtures.chatResponse('Hello!');
const jailbreakResponse = fixtures.jailbreakResponse('prompt', 'DAN');
const metrics = fixtures.metrics();
```

### Test Utilities

```typescript
import { createTestClient, waitFor, mockResponse, mockError } from '@/lib/api/testing';

// Create isolated test client
const testClient = createTestClient({
  enableRetry: false,
  enableCircuitBreaker: false,
});

// Wait for condition
await waitFor(() => someCondition, { timeout: 5000 });

// Create mock responses
const success = mockResponse({ data: 'test' });
const error = mockError('VALIDATION_ERROR', 'Invalid input', 400);
```

### Integration Tests

```typescript
import { apiClient } from '@/lib/api';
import { mockServer } from '@/lib/api/testing';

describe('API Integration', () => {
  beforeEach(() => {
    mockServer.reset();
  });

  it('should fetch providers', async () => {
    const response = await apiClient.get('/api/v1/providers');
    expect(response.data).toHaveLength(4);
    expect(response.data[0].name).toBe('gemini');
  });

  it('should handle errors', async () => {
    mockServer.register({
      method: 'GET',
      url: '/api/v1/error',
      response: mockError('SERVER_ERROR', 'Internal error', 500),
    });

    await expect(apiClient.get('/api/v1/error')).rejects.toThrow();
  });
});
```

## Migration Guide

### From Legacy API (`api.ts`, `api-enhanced.ts`)

The new architecture provides a backward-compatible layer:

```typescript
// Old code
import { api } from '@/lib/api';
const response = await api.get('/endpoint');

// New code (drop-in replacement)
import { legacyApi as api } from '@/lib/api/migration/compat';
const response = await api.get('/endpoint');

// Or migrate to new client
import { apiClient } from '@/lib/api';
const response = await apiClient.get('/endpoint');
```

### From `chimeraApi.ts`

```typescript
// Old code
import { chimeraApi } from '@/lib/chimeraApi';
const health = await chimeraApi.health.check();

// New code
import { apiClient } from '@/lib/api';
const health = await apiClient.get('/api/v1/health');

// Or use the compatibility layer
import { chimeraApiCompat } from '@/lib/api/migration/compat';
const health = await chimeraApiCompat.health.check();
```

### Migration Checklist

1. [ ] Update imports to use `@/lib/api`
2. [ ] Replace direct axios calls with `apiClient`
3. [ ] Update error handling to use new error types
4. [ ] Configure environment variables
5. [ ] Set up authentication tokens
6. [ ] Update tests to use mock server
7. [ ] Remove old API files after migration

## API Reference

### `apiClient`

Main API client instance.

```typescript
interface APIClient {
  get<T>(url: string, config?: RequestConfig): Promise<APIResponse<T>>;
  post<T>(url: string, data?: unknown, config?: RequestConfig): Promise<APIResponse<T>>;
  put<T>(url: string, data?: unknown, config?: RequestConfig): Promise<APIResponse<T>>;
  patch<T>(url: string, data?: unknown, config?: RequestConfig): Promise<APIResponse<T>>;
  delete<T>(url: string, config?: RequestConfig): Promise<APIResponse<T>>;
  providerRequest<T>(provider: ProviderName, endpoint: string, config?: RequestConfig): Promise<APIResponse<T>>;
}

interface RequestConfig {
  headers?: Record<string, string>;
  params?: Record<string, unknown>;
  timeout?: number;
  cache?: boolean;
  cacheTTL?: number;
  retry?: boolean | RetryConfig;
  circuitBreaker?: boolean;
}
```

### `configManager`

Configuration management.

```typescript
interface ConfigManager {
  getEnvironment(): Environment;
  getConfig(): APIConfig;
  updateConfig(updates: Partial<APIConfig>): void;
  getActiveBaseUrl(): string;
  getProviderConfig(provider: ProviderName): ProviderConfig;
}
```

### `tokenManager`

JWT token management.

```typescript
interface TokenManager {
  setToken(token: string): void;
  getToken(): string | null;
  clearToken(): void;
  isTokenValid(): boolean;
  setRefreshCallback(callback: () => Promise<string>): void;
}
```

### `metricsCollector`

Metrics collection.

```typescript
interface MetricsCollector {
  recordRequest(metrics: RequestMetrics): void;
  getMetrics(): AggregatedMetrics;
  getMetricsByEndpoint(endpoint: string): EndpointMetrics;
  reset(): void;
}
```

## Support

For issues or questions:

1. Check the [Troubleshooting Guide](../../../docs/TROUBLESHOOTING.md)
2. Review the [Frontend Architecture](../../../docs/FRONTEND_ARCHITECTURE_REPORT.md)
3. Open an issue in the repository

## License

MIT License - See LICENSE file for details.
