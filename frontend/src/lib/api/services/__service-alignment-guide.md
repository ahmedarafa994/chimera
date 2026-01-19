# Frontend Service Alignment Guide

## Overview

This guide ensures all frontend services are properly aligned with backend API endpoints, using consistent patterns for paths, payload shapes, response mapping, and error handling.

## Service Standards

### 1. File Structure
```
frontend/src/lib/api/services/
├── __service-alignment-guide.md  (this file)
├── core/
│   ├── base-service.ts           (base service class)
│   ├── types.ts                  (shared API types)
│   └── constants.ts              (API constants)
├── [feature]-service.ts          (individual service files)
└── index.ts                      (service exports)
```

### 2. Import Pattern
```typescript
import { apiClient } from '../client';
import { ApiResponse, ApiError } from '../types';
import { apiErrorHandler } from '../../errors/api-error-handler';
```

### 3. URL Patterns
- **Correct**: `/api/v1/[endpoint]` - Full path for external calls
- **Backend Routes**: Backend uses `/api/v1/` prefix in FastAPI routers
- **WebSocket**: Use `apiClient.getWebSocketUrl()` for WS endpoints

### 4. Request/Response Types

#### Backend Pydantic Models → Frontend TypeScript
```typescript
// Backend: PromptRequest
export interface PromptRequest {
  prompt: string;
  provider?: string;
  model?: string;
  config?: GenerationConfig;
  api_key?: string;
}

// Backend: PromptResponse
export interface PromptResponse {
  text: string;
  provider: string;
  model_used: string;
  latency_ms: number;
  usage_metadata?: Record<string, number>;
  cached: boolean;
  error?: string;
}
```

#### Standard Response Wrapper
```typescript
export interface ApiResponse<T = any> {
  data: T;
  status: number;
  request_id?: string;
}
```

### 5. Error Handling Pattern
```typescript
export const serviceApi = {
  async serviceMethod(params: RequestType): Promise<ApiResponse<ResponseType>> {
    try {
      const response = await apiClient.post<ResponseType>('/api/v1/endpoint', params);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'ServiceMethod');
    }
  }
};
```

### 6. Authentication Headers
The `apiClient` automatically handles:
- `Authorization: Bearer <token>` (from authManager)
- `X-Request-ID: <uuid>` (auto-generated)
- `X-Tenant-ID: <tenant>` (if available)
- `Content-Type: application/json`

### 7. WebSocket Patterns
```typescript
const wsUrl = apiClient.getWebSocketUrl('/api/v1/ws/endpoint');
const ws = new WebSocket(wsUrl);
```

## Backend API Mapping

### Core Endpoints
- **Generation**: `/api/v1/generate` and `/api/v1/llm/generate`
- **Chat**: `/api/v1/chat/completions`
- **Transform**: `/api/v1/transform`
- **Execute**: `/api/v1/execute`

### Provider Management
- **List**: `GET /api/v1/providers`
- **Models**: `GET /api/v1/providers/{provider}/models`
- **Select**: `POST /api/v1/providers/select`
- **Current**: `GET /api/v1/providers/current`
- **Health**: `GET /api/v1/providers/health`
- **WebSocket**: `WS /api/v1/providers/ws/selection`

### Advanced Features
- **AutoDAN**: `/api/v1/autodan/*`
- **Jailbreak**: `/api/v1/jailbreak/*` and `/api/v1/generation/jailbreak/*`
- **GPTFuzz**: `/api/v1/gptfuzz/*`
- **Intent-Aware**: `/api/v1/intent-aware/*`
- **Gradient**: `/api/v1/gradient/*`

### Health & Monitoring
- **Health**: `GET /api/v1/health`
- **Metrics**: `GET /api/v1/metrics`
- **Provider Health**: `GET /api/v1/provider-health-dashboard/*`

### Session Management
- **Create**: `POST /api/v1/session`
- **Get**: `GET /api/v1/session/{id}`
- **Update Model**: `PUT /api/v1/session/model`
- **Stats**: `GET /api/v1/session/stats`

## Common Misalignments & Fixes

### 1. URL Path Issues
❌ **Wrong**: `/v1/providers` or `/providers`
✅ **Correct**: `/api/v1/providers`

### 2. Response Shape Mismatches
❌ **Wrong**: Expecting `{ providers: [] }` when backend returns `{ data: { providers: [] } }`
✅ **Correct**: Handle backend response structure properly

### 3. Error Response Handling
❌ **Wrong**: Not using consistent error types
✅ **Correct**: Use `ApiError` class and `apiErrorHandler`

### 4. Authentication
❌ **Wrong**: Manual header management
✅ **Correct**: Use `apiClient` which handles auth automatically

## Migration Checklist

### From `api-enhanced.ts` to New Pattern
1. ✅ Update imports to use `apiClient`
2. ✅ Change URL paths to include `/api/v1/` prefix
3. ✅ Update response type interfaces to match backend Pydantic models
4. ✅ Add proper error handling with `apiErrorHandler`
5. ✅ Use TypeScript generics for type safety
6. ✅ Add JSDoc comments for API methods
7. ✅ Test all endpoints for proper functionality

### Service Validation
- [ ] URLs match backend routes exactly
- [ ] Request/response types match Pydantic models
- [ ] Error handling follows standard pattern
- [ ] Authentication handled by apiClient
- [ ] WebSocket URLs use proper helper
- [ ] All methods have proper TypeScript types
- [ ] Services export consistent interface

## Example Service Implementation

```typescript
/**
 * Provider Service - Aligned with Backend API
 */
import { apiClient } from '../client';
import { ApiResponse } from '../types';
import { apiErrorHandler } from '../../errors/api-error-handler';

// Types matching backend Pydantic models
export interface ProviderInfo {
  provider: string;
  display_name: string;
  status: string;
  is_healthy: boolean;
  models: string[];
  default_model: string | null;
  latency_ms?: number;
}

export interface ProvidersListResponse {
  providers: ProviderInfo[];
  count: number;
  default_provider: string;
  default_model: string;
}

export const providerService = {
  /**
   * Get all available providers
   */
  async getProviders(): Promise<ApiResponse<ProvidersListResponse>> {
    try {
      return await apiClient.get<ProvidersListResponse>('/api/v1/providers');
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviders');
    }
  },

  /**
   * WebSocket for real-time provider updates
   */
  createWebSocket(): WebSocket {
    return new WebSocket(apiClient.getWebSocketUrl('/api/v1/providers/ws/selection'));
  }
};
```