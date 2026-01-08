# Chimera API Integration Guide

## Overview

This document describes the comprehensive synchronization and integration system between the Chimera backend (FastAPI) and frontend (Next.js) applications.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Stores  │  │  Hooks   │  │ Services │  │  API Client      │ │
│  │ (Zustand)│  │ (React)  │  │          │  │  (Axios)         │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
│       │             │             │                  │           │
│       └─────────────┴─────────────┴──────────────────┘           │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              WebSocket Manager / Cache Manager             │  │
│  └───────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │   HTTP / WebSocket │
                     └─────────┬─────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                        Backend (FastAPI)                         │
├──────────────────────────────┴──────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    API Routers (v1)                        │  │
│  │  /providers  /sessions  /jailbreak  /autodan  /transform   │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                    Middleware Layer                        │  │
│  │  Auth / Rate Limit / CORS / Request Logging / Idempotency │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                    Service Layer                           │  │
│  │  LLM Service / Jailbreak Service / Session Service         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Frontend API Layer

### API Client (`frontend/src/lib/api/client.ts`)

The centralized API client handles all HTTP communication:

```typescript
import { apiClient } from '@/lib/api';

// GET request with caching
const response = await apiClient.get('/providers');

// POST request with retry
const result = await apiClient.post('/jailbreak', { prompt: '...' });

// Cancel request
apiClient.cancelRequest(requestId);
```

**Features:**
- Automatic token injection
- Request/response interceptors
- Retry with exponential backoff
- Request cancellation
- Response caching
- Request ID tracking

### Authentication Manager (`frontend/src/lib/api/auth-manager.ts`)

Handles token lifecycle:

```typescript
import { authManager } from '@/lib/api';

// Login
await authManager.login(email, password);

// Get access token (auto-refreshes if needed)
const token = await authManager.getAccessToken();

// Logout
authManager.logout();
```

### Cache Manager (`frontend/src/lib/api/cache-manager.ts`)

Intelligent response caching:

```typescript
import { apiCache } from '@/lib/api';

// Manual cache operations
apiCache.set('key', data, ttl);
const cached = apiCache.get('key');

// Cache invalidation
apiCache.invalidatePattern('/providers.*');

// Get or fetch with cache
const data = await apiCache.getOrFetch('key', fetchFn, ttl);
```

### WebSocket Manager (`frontend/src/lib/api/websocket-manager.ts`)

Real-time communication:

```typescript
import { wsManager, createStream } from '@/lib/api';

// Connect
await wsManager.connect({ url: 'ws://localhost:8000/ws' });

// Send messages
wsManager.sendData({ type: 'update', data: {} });

// Subscribe to messages
const unsubscribe = wsManager.onMessage((msg) => {
  console.log('Received:', msg);
});

// Stream handling
const stream = createStream('attack-123');
stream.onEvent((event) => {
  if (event.event === 'chunk') {
    console.log('Progress:', event.data.progress);
  }
});
```

## API Services

### Providers Service

```typescript
import { providersApi } from '@/lib/api';

// Get all providers
const { data } = await providersApi.getProviders();

// Get models for provider
const { data } = await providersApi.getProviderModels('gemini');

// Set default provider
await providersApi.setDefaultProvider({ provider_id: 'gemini', model_id: 'gemini-2.5-pro' });
```

### Sessions Service

```typescript
import { sessionsApi } from '@/lib/api';

// Create session
const { data: session } = await sessionsApi.createSession({ provider: 'gemini' });

// Send message
const { data: message } = await sessionsApi.sendMessage(sessionId, {
  content: 'Hello',
  stream: true,
});

// Get history
const { data } = await sessionsApi.getMessages(sessionId, page, pageSize);
```

### Jailbreak Service

```typescript
import { jailbreakApi } from '@/lib/api';

// Execute enhanced jailbreak
const { data } = await jailbreakApi.enhancedJailbreak({
  prompt: 'Test prompt',
  method: 'best_of_n',
  target_model: 'gemini-2.5-pro',
});

// Get configuration
const { data: config } = await jailbreakApi.getConfig();

// Transform prompt
const { data } = await jailbreakApi.transform({
  text: 'Original text',
  transformations: ['obfuscation', 'paraphrase'],
});
```

## State Management (Zustand Stores)

### Providers Store

```typescript
import { useProvidersStore } from '@/lib/api/stores';

function ProviderSelector() {
  const {
    providers,
    currentProvider,
    currentModel,
    isLoading,
    fetchProviders,
    setCurrentProvider,
  } = useProvidersStore();

  useEffect(() => {
    fetchProviders();
  }, []);

  return (
    <select onChange={(e) => setCurrentProvider(e.target.value)}>
      {providers.map((p) => (
        <option key={p.id} value={p.id}>{p.name}</option>
      ))}
    </select>
  );
}
```

### Session Store

```typescript
import { useSessionStore } from '@/lib/api/stores';

function ChatView() {
  const {
    messages,
    isSending,
    sendMessage,
    fetchMessages,
  } = useSessionStore();

  const handleSend = async (content: string) => {
    await sendMessage(content, true); // with streaming
  };

  return (
    <div>
      {messages.map((msg) => (
        <Message key={msg.id} {...msg} />
      ))}
    </div>
  );
}
```

### Jailbreak Store

```typescript
import { useJailbreakStore } from '@/lib/api/stores';

function JailbreakPanel() {
  const {
    currentResult,
    isExecuting,
    progress,
    executeJailbreak,
  } = useJailbreakStore();

  const handleExecute = async () => {
    await executeJailbreak({
      prompt: 'Test',
      method: 'autodan',
    });
  };

  return (
    <div>
      {isExecuting && <ProgressBar value={progress} />}
      {currentResult && <Result data={currentResult} />}
    </div>
  );
}
```

## React Hooks

### useApi Hook

```typescript
import { useApi, useMutation, useQuery } from '@/lib/api/hooks';

// Generic API hook
const { data, isLoading, error, execute } = useApi(providersApi.getProviders);

// Query hook (auto-fetch)
const { data } = useQuery(() => providersApi.getProviders(), {
  enabled: true,
  refetchInterval: 60000,
});

// Mutation hook
const { execute: createSession } = useMutation(sessionsApi.createSession, {
  onSuccess: (data) => console.log('Created:', data),
  onError: (err) => console.error('Failed:', err),
});
```

### useWebSocket Hook

```typescript
import { useWebSocket, useStream } from '@/lib/api/hooks';

function RealtimeComponent() {
  const { isConnected, send } = useWebSocket({
    url: 'ws://localhost:8000/ws/sync',
    autoConnect: true,
    onMessage: (msg) => console.log(msg),
  });

  const { isStreaming, events, start } = useStream({
    streamId: 'attack-123',
    onEvent: (event) => {
      if (event.event === 'end') {
        console.log('Complete!');
      }
    },
  });
}
```

## Validation (Zod Schemas)

```typescript
import { 
  promptRequestSchema, 
  jailbreakRequestSchema,
  validateRequest,
  getValidationErrors 
} from '@/lib/api/validation';

// Validate before sending
const validation = validateRequest(jailbreakRequestSchema, formData);

if (!validation.success) {
  const errors = getValidationErrors(validation.errors);
  // { prompt: ['Required'], technique: ['Invalid enum value'] }
  setFormErrors(errors);
  return;
}

await jailbreakApi.jailbreak(validation.data);
```

## Error Handling

### API Errors

```typescript
import { ApiError } from '@/lib/api/types';

try {
  await apiClient.post('/jailbreak', data);
} catch (error) {
  const apiError = error as ApiError;
  
  switch (apiError.code) {
    case 'RATE_LIMITED':
      showNotification('Too many requests. Please wait.');
      break;
    case 'VALIDATION_ERROR':
      setFormErrors(apiError.details);
      break;
    case 'UNAUTHORIZED':
      redirectToLogin();
      break;
    default:
      showNotification(apiError.message);
  }
}
```

### Retry Strategy

```typescript
import { createRetryStrategy } from '@/lib/api';

const strategy = createRetryStrategy()
  .maxRetries(5)
  .initialDelay(1000)
  .backoffMultiplier(2)
  .shouldRetry((error) => error.response?.status !== 400)
  .onRetry((error, attempt) => {
    console.log(`Retry ${attempt}:`, error.message);
  });

await strategy.execute(() => apiClient.get('/slow-endpoint'));
```

## Optimistic Updates

```typescript
import { useOptimisticMutation } from '@/lib/api/hooks';

const { execute, isOptimistic } = useOptimisticMutation(
  (data) => sessionsApi.sendMessage(sessionId, data),
  {
    optimisticUpdate: ([{ content }]) => ({
      id: `temp-${Date.now()}`,
      content,
      role: 'user',
      timestamp: new Date().toISOString(),
    }),
    rollback: (error, params) => {
      removeMessage(params[0].content);
      showError('Failed to send message');
    },
  }
);
```

## Backend Configuration

### CORS Settings

The backend is configured with proper CORS headers in `backend-api/app/config/cors.py`:

```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
]

CORS_ALLOW_HEADERS = [
    "Authorization",
    "X-Request-ID",
    "X-Tenant-ID",
    "X-API-Key",
]
```

### Response Schemas

All API responses follow a consistent structure:

```python
from app.schemas.api_response import APIResponse, PaginatedResponse, APIError

# Success response
@router.get("/items", response_model=APIResponse[List[Item]])
async def get_items():
    return APIResponse(data=items, status=200)

# Paginated response
@router.get("/items/paginated")
async def get_items_paginated(page: int = 1, page_size: int = 20):
    return PaginatedResponse.create(items, total, page, page_size)

# Error response
raise HTTPException(
    status_code=400,
    detail=APIError(
        message="Validation failed",
        code="VALIDATION_ERROR",
        status=400,
    ).model_dump()
)
```

## Testing

### Frontend Tests

```bash
# Run all API tests
cd frontend
npx vitest run src/lib/api/__tests__

# Watch mode
npx vitest src/lib/api/__tests__
```

### Backend Tests

```bash
# Run API integration tests
cd backend-api
poetry run pytest tests/api/ -v

# With coverage
poetry run pytest tests/api/ --cov=app/api
```

## Performance Monitoring

### Frontend Logging

```typescript
import { logger } from '@/lib/api';

// Get performance metrics
const metrics = logger.getPerformanceMetrics();
// { totalRequests, avgDuration, errorRate, p50Duration, p95Duration, p99Duration }

// Export logs for debugging
const logsJson = logger.exportLogs();
```

### Backend Logging

All requests are logged with:
- Request ID
- Method and URL
- Status code
- Response time
- Slow request warnings (>1000ms)

## Best Practices

1. **Always use typed responses**: Import types from `@/lib/api/types`
2. **Handle errors gracefully**: Use try-catch and display user-friendly messages
3. **Implement optimistic UI**: For better user experience on mutations
4. **Use caching strategically**: Set appropriate TTLs for different data types
5. **Cancel stale requests**: When components unmount or queries change
6. **Validate input**: Use Zod schemas before API calls
7. **Monitor performance**: Check logs and metrics regularly

## Migration Guide

If upgrading from the previous API implementation:

1. Replace direct fetch calls with `apiClient` methods
2. Use Zustand stores instead of React Context for global state
3. Implement validation schemas for all request types
4. Add proper error handling with `ApiError` type
5. Set up WebSocket connections for real-time features