# Frontend API Technical Audit Report

**Project**: Chimera Frontend  
**Audit Date**: 2026-01-06  
**Phase**: 2 of Gap Analysis Project  
**Auditor**: Automated Technical Analysis  

---

## Executive Summary

The Chimera Frontend is a comprehensive Next.js 14+ application built with TypeScript, React, and Tailwind CSS. The frontend demonstrates sophisticated patterns for API communication, state management, and real-time updates:

- **Architecture**: Next.js App Router with modular API client layer
- **State Management**: TanStack Query (React Query) + Zustand stores with optimistic updates
- **Real-time**: WebSocket + SSE managers with automatic reconnection
- **Authentication**: JWT Bearer tokens + API Key (X-API-Key header) via AuthManager singleton
- **Error Handling**: Comprehensive error hierarchy mirroring backend exceptions
- **UI Framework**: Shadcn UI components with Radix primitives

### Key Statistics

| Metric | Count |
|--------|-------|
| Expected API Endpoints | 75+ |
| WebSocket Implementations | 4 |
| SSE Streaming Implementations | 2 |
| TypeScript Type Files | 20+ |
| Zustand Stores | 4 |
| React Query Hooks | 10+ |
| Error Classes | 17 |

---

## 1. API Client Architecture

### 1.1 Client Layer Overview

The frontend uses multiple API client patterns:

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| [`frontend/src/lib/api-enhanced.ts`](frontend/src/lib/api-enhanced.ts) | 2485 | **DEPRECATED** | Monolithic API client (legacy) |
| [`frontend/src/api/jailbreak.ts`](frontend/src/api/jailbreak.ts) | 415 | Active | Jailbreak REST/WebSocket/SSE |
| [`frontend/src/lib/api/deepteam-client.ts`](frontend/src/lib/api/deepteam-client.ts) | 157 | Active | DeepTeam session/agent management |
| [`frontend/src/lib/api/auth-manager.ts`](frontend/src/lib/api/auth-manager.ts) | 303 | Active | Authentication token management |
| [`frontend/src/lib/sync/provider-sync-service.ts`](frontend/src/lib/sync/provider-sync-service.ts) | 1146 | Active | Provider synchronization |
| [`frontend/src/lib/sync/websocket-manager.ts`](frontend/src/lib/sync/websocket-manager.ts) | 746 | Active | Generic WebSocket management |
| [`frontend/src/lib/sync/sse-manager.ts`](frontend/src/lib/sync/sse-manager.ts) | 563 | Active | Generic SSE management |

### 1.2 Base URL Configuration

```typescript
// Primary configuration via environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
const API_V1_PREFIX = '/api/v1';

// Axios client defaults
const apiClient = axios.create({
  baseURL: '/api/v1',  // Uses Next.js proxy
  timeout: 30000,      // 30 second default
  headers: { 'Content-Type': 'application/json' }
});
```

### 1.3 Timeout Configuration

| Timeout Type | Duration | Usage |
|--------------|----------|-------|
| `DEFAULT_TIMEOUT` | 30,000ms | Standard API calls |
| `EXTENDED_TIMEOUT` | 600,000ms | Long operations (10 min) |
| `LLM_TIMEOUT` | 120,000ms | LLM generation (2 min) |
| `FAST_TIMEOUT` | 5,000ms | Health checks, quick operations |

### 1.4 Retry Configuration

```typescript
// Axios retry configuration
axiosRetry(apiClient, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) => 
    axiosRetry.isNetworkOrIdempotentRequestError(error) ||
    error.response?.status === 429 ||
    error.response?.status === 503
});
```

---

## 2. API Endpoints Expected by Frontend

### 2.1 Health & System Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| GET | `/health` | `HealthDashboard` | `OverallHealth` |
| GET | `/health/live` | `HealthDashboard` | `{ status: string }` |
| GET | `/health/ready` | N/A | N/A |
| GET | `/health/circuit-breakers` | N/A | `CircuitBreakerStatus[]` |

### 2.2 Authentication Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/auth/login` | `AuthManager.login()` | `{ email, password }` → `{ tokens, user }` |
| POST | `/api/v1/auth/refresh` | `AuthManager.refreshAccessToken()` | `{ refresh_token }` → `AuthTokens` |

### 2.3 Provider Management Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| GET | `/api/v1/providers/` | `useProviderConfig` | `Provider[]` |
| GET | `/api/v1/providers/current` | `useProviderConfig` | `ActiveProviderInfo` |
| GET | `/api/v1/providers/{provider}/models` | `enhancedApi.models.getForProvider()` | `Model[]` |
| POST | `/api/v1/providers/select` | `useProviderConfig` | `{ provider_id, model_id }` |
| GET | `/api/v1/providers/health` | `useProviderConfig` | `HealthStatus[]` |
| POST | `/api/v1/provider-config/providers` | `useProviderConfig` | `CreateProviderRequest` |
| PUT | `/api/v1/provider-config/providers/{id}` | `useProviderConfig` | `UpdateProviderRequest` |
| DELETE | `/api/v1/provider-config/providers/{id}` | `useProviderConfig` | N/A |
| POST | `/api/v1/provider-config/providers/{id}/test` | `useProviderConfig` | N/A |
| WebSocket | `/api/v1/provider-config/ws/updates` | `ProviderSyncContext` | `WebSocketMessage` |

### 2.4 Session Management Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| GET | `/api/v1/session/models` | `enhancedApi.session.getModels()` | `Model[]` |
| POST | `/api/v1/session/models/validate` | `enhancedApi.session.validateModel()` | `ValidationResult` |
| POST | `/api/v1/session` | `enhancedApi.session.create()` | `SessionConfig` |
| GET | `/api/v1/session` | `enhancedApi.session.getCurrent()` | `Session` |
| DELETE | `/api/v1/session` | `enhancedApi.session.delete()` | N/A |
| GET | `/api/v1/session/current-model` | `enhancedApi.session.getCurrentModel()` | `ModelInfo` |
| PUT | `/api/v1/session/model` | `enhancedApi.session.updateModel()` | `{ model_id }` |

### 2.5 Generation Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/generation/generate` | `enhancedApi.generate.text()` | `PromptRequest` → `PromptResponse` |
| GET | `/api/v1/generation/health` | `enhancedApi.generate.health()` | `HealthStatus` |
| POST | `/api/v1/streaming/generate/stream` | SSE streaming | `PromptRequest` → SSE events |
| GET | `/api/v1/streaming/generate/stream/capabilities` | N/A | `StreamingCapabilities` |

### 2.6 Jailbreak Endpoints (DeepTeam)

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/deepteam/jailbreak/generate` | `JailbreakAPI.generate()` | `JailbreakGenerationRequest` |
| POST | `/api/v1/deepteam/jailbreak/batch` | `JailbreakAPI.generateBatch()` | `BatchJailbreakRequest` |
| GET | `/api/v1/deepteam/jailbreak/strategies` | `JailbreakAPI.getStrategies()` | `Strategy[]` |
| GET | `/api/v1/deepteam/jailbreak/strategies/{type}` | `JailbreakAPI.getStrategyDetails()` | `StrategyDetails` |
| GET | `/api/v1/deepteam/jailbreak/vulnerabilities` | `JailbreakAPI.getVulnerabilities()` | `Vulnerability[]` |
| DELETE | `/api/v1/deepteam/jailbreak/cache` | `JailbreakAPI.clearCache()` | N/A |
| GET | `/api/v1/deepteam/jailbreak/health` | `JailbreakAPI.getHealth()` | `HealthStatus` |
| GET | `/api/v1/deepteam/jailbreak/sessions/{id}/prompts/{pid}` | `JailbreakAPI.getPrompt()` | `Prompt` |
| GET | `/api/v1/deepteam/jailbreak/sessions/{id}/prompts` | `JailbreakAPI.getSessionPrompts()` | `Prompt[]` |
| DELETE | `/api/v1/deepteam/jailbreak/sessions/{id}` | `JailbreakAPI.deleteSession()` | N/A |
| WebSocket | `/api/v1/deepteam/jailbreak/ws/generate` | `JailbreakWebSocket` | WebSocket messages |
| GET | `/api/v1/deepteam/jailbreak/generate/stream` | `JailbreakSSE` | SSE events |

### 2.7 AutoDAN Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/autodan/jailbreak` | `enhancedApi.autodan.jailbreak()` | `AutoDANRequest` |
| POST | `/api/v1/autodan/batch` | `enhancedApi.autodan.batch()` | `BatchAutoDANRequest` |
| GET | `/api/v1/autodan/config` | `enhancedApi.autodan.getConfig()` | `AutoDANConfig` |

### 2.8 AutoDAN-Turbo Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/autodan-turbo/attack` | `enhancedApi.autodanTurbo.attack()` | `AttackRequest` |
| POST | `/api/v1/autodan-turbo/warmup` | `enhancedApi.autodanTurbo.warmup()` | `WarmupRequest` |
| POST | `/api/v1/autodan-turbo/lifelong` | `enhancedApi.autodanTurbo.lifelong()` | `LifelongRequest` |
| POST | `/api/v1/autodan-turbo/test` | `enhancedApi.autodanTurbo.test()` | `TestRequest` |
| GET | `/api/v1/autodan-turbo/strategies` | `enhancedApi.autodanTurbo.getStrategies()` | `Strategy[]` |
| GET | `/api/v1/autodan-turbo/strategies/{id}` | `enhancedApi.autodanTurbo.getStrategy()` | `StrategyDetails` |
| POST | `/api/v1/autodan-turbo/strategies` | `enhancedApi.autodanTurbo.createStrategy()` | `CreateStrategyRequest` |
| DELETE | `/api/v1/autodan-turbo/strategies/{id}` | `enhancedApi.autodanTurbo.deleteStrategy()` | N/A |
| POST | `/api/v1/autodan-turbo/strategies/search` | `enhancedApi.autodanTurbo.searchStrategies()` | `SearchRequest` |
| POST | `/api/v1/autodan-turbo/strategies/batch-inject` | `enhancedApi.autodanTurbo.batchInject()` | `BatchInjectRequest` |
| GET | `/api/v1/autodan-turbo/progress` | `enhancedApi.autodanTurbo.getProgress()` | `LearningProgress` |
| GET | `/api/v1/autodan-turbo/library/stats` | `enhancedApi.autodanTurbo.getLibraryStats()` | `LibraryStats` |
| POST | `/api/v1/autodan-turbo/reset` | `enhancedApi.autodanTurbo.reset()` | N/A |
| POST | `/api/v1/autodan-turbo/library/save` | `enhancedApi.autodanTurbo.saveLibrary()` | N/A |
| POST | `/api/v1/autodan-turbo/library/clear` | `enhancedApi.autodanTurbo.clearLibrary()` | N/A |
| GET | `/api/v1/autodan-turbo/health` | `enhancedApi.autodanTurbo.health()` | `HealthStatus` |

### 2.9 DeepTeam Red Team Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/deepteam/red-team` | `DeepTeamApiClient` | `RedTeamRequest` |
| POST | `/api/v1/deepteam/quick-scan` | `DeepTeamApiClient` | `QuickScanRequest` |
| POST | `/api/v1/deepteam/security-audit` | `DeepTeamApiClient` | `SecurityAuditRequest` |
| POST | `/api/v1/deepteam/bias-audit` | `DeepTeamApiClient` | `BiasAuditRequest` |
| GET | `/api/v1/deepteam/sessions` | `DeepTeamApiClient.listSessions()` | `Session[]` |
| GET | `/api/v1/deepteam/sessions/{id}` | `DeepTeamApiClient.getSession()` | `SessionStatus` |
| GET | `/api/v1/deepteam/sessions/{id}/result` | `DeepTeamApiClient.getSessionResult()` | `SessionResult` |
| GET | `/api/v1/deepteam/vulnerabilities` | `DeepTeamApiClient` | `Vulnerability[]` |
| GET | `/api/v1/deepteam/attacks` | `DeepTeamApiClient` | `Attack[]` |
| GET | `/api/v1/deepteam/agents` | `DeepTeamApiClient.listAgents()` | `Agent[]` |
| GET | `/api/v1/deepteam/agents/{id}` | `DeepTeamApiClient.getAgent()` | `AgentDetails` |
| GET | `/api/v1/deepteam/evaluations` | `DeepTeamApiClient.listEvaluations()` | `Evaluation[]` |
| GET | `/api/v1/deepteam/evaluations/{id}` | `DeepTeamApiClient.getEvaluation()` | `EvaluationDetails` |
| GET | `/api/v1/deepteam/refinements` | `DeepTeamApiClient.listRefinements()` | `Refinement[]` |
| POST | `/api/v1/deepteam/refinements/apply` | `DeepTeamApiClient.applyRefinement()` | `ApplyRefinementRequest` |
| WebSocket | `/ws/sessions/{sessionId}` | `DeepTeamApiClient.createWebSocketConnection()` | WebSocket messages |

### 2.10 Provider Sync Endpoints

| Method | Path | Component/Service | Types Used |
|--------|------|-------------------|------------|
| POST | `/api/v1/provider-sync/sync` | `ProviderSyncService` | `SyncRequest` → `SyncResponse` |
| GET | `/api/v1/provider-sync/providers/{id}/availability` | `ProviderSyncService` | `ProviderAvailabilityInfo` |
| GET | `/api/v1/provider-sync/models/{id}/availability` | `ProviderSyncService` | `ModelAvailabilityInfo` |
| POST | `/api/v1/provider-sync/select/provider` | `ProviderSyncService.selectProvider()` | `SelectProviderRequest` |
| POST | `/api/v1/provider-sync/select/model` | `ProviderSyncService.selectModel()` | `SelectModelRequest` |
| GET | `/api/v1/provider-sync/active` | `ProviderSyncService.getActiveSelection()` | `ActiveSelection` |
| WebSocket | `/api/v1/provider-sync/ws` | `ProviderSyncService` | `WebSocketMessage` |

---

## 3. TypeScript Type Definitions

### 3.1 Core API Types ([`frontend/src/lib/api/types.ts`](frontend/src/lib/api/types.ts))

```typescript
// Authentication Types
interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

interface User {
  id: string;
  email: string;
  name: string;
  tenant_id: string;
  roles?: string[];
  permissions?: string[];
}

// Provider Types
interface Provider {
  id: string;
  name: string;
  type: LLMProviderType;
  enabled: boolean;
  models: Model[];
  health?: HealthStatus;
}

interface Model {
  id: string;
  name: string;
  provider_id: string;
  capabilities: string[];
  max_tokens: number;
  deprecated?: boolean;
}

// Generation Types
interface PromptRequest {
  prompt: string;
  system_instruction?: string;
  config?: GenerationConfig;
  model?: string;
  provider?: LLMProviderType;
}

interface PromptResponse {
  text: string;
  model_used: string;
  provider: string;
  usage_metadata?: Record<string, any>;
  finish_reason?: string;
  latency_ms: number;
}

interface GenerationConfig {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_output_tokens?: number;
  stop_sequences?: string[];
}
```

### 3.2 Provider Sync Types ([`frontend/src/types/provider-sync.ts`](frontend/src/types/provider-sync.ts))

```typescript
// Sync State Types
interface SyncState {
  metadata: SyncMetadata;
  providers: ProviderSyncInfo[];
  all_models: ModelSpecification[];
  active_provider_id?: string;
  active_model_id?: string;
}

interface SyncMetadata {
  version: number;
  last_sync_time: string;
  sync_type: 'full' | 'incremental';
}

// Sync Event Types
enum SyncEventType {
  FULL_SYNC = 'full_sync',
  INITIAL_STATE = 'initial_state',
  PROVIDER_ADDED = 'provider_added',
  PROVIDER_UPDATED = 'provider_updated',
  PROVIDER_REMOVED = 'provider_removed',
  PROVIDER_STATUS_CHANGED = 'provider_status_changed',
  MODEL_DEPRECATED = 'model_deprecated',
  ACTIVE_PROVIDER_CHANGED = 'active_provider_changed',
  ACTIVE_MODEL_CHANGED = 'active_model_changed',
  STATE_CHANGED = 'state_changed',
  HEARTBEAT = 'heartbeat',
  ERROR = 'error',
}

// Client Configuration
interface ProviderSyncConfig {
  apiBaseUrl: string;
  wsUrl: string;
  enableWebSocket: boolean;
  enableCache: boolean;
  cacheTtl: number;
  pollingInterval: number;
  syncTimeout: number;
  maxReconnectAttempts: number;
  reconnectBaseDelay: number;
  reconnectMaxDelay: number;
  heartbeatInterval: number;
  includeDeprecated: boolean;
}
```

### 3.3 Jailbreak Types ([`frontend/src/types/jailbreak.ts`](frontend/src/types/jailbreak.ts))

```typescript
interface JailbreakGenerationRequest {
  core_request: string;
  technique_suite: string;
  potency_level: number;
  temperature?: number;
  top_p?: number;
  max_new_tokens?: number;
  density?: number;
  // Content transformation flags
  use_leet_speak?: boolean;
  use_homoglyphs?: boolean;
  use_caesar_cipher?: boolean;
  // Structural & semantic flags
  use_role_hijacking?: boolean;
  use_instruction_injection?: boolean;
  use_adversarial_suffixes?: boolean;
  // Advanced neural flags
  use_neural_bypass?: boolean;
  use_meta_prompting?: boolean;
  // Research-driven flags
  use_multilingual_trojan?: boolean;
  use_payload_splitting?: boolean;
  use_contextual_interaction_attack?: boolean;
}

interface JailbreakResponse {
  session_id: string;
  prompts: GeneratedPrompt[];
  metadata: JailbreakMetadata;
}

interface GeneratedPrompt {
  id: string;
  content: string;
  technique: string;
  potency_score: number;
  safety_score: number;
}
```

### 3.4 WebSocket/SSE Types ([`frontend/src/lib/sync/types.ts`](frontend/src/lib/sync/types.ts))

```typescript
// WebSocket Types
interface WebSocketConfig {
  url: string;
  protocols?: string[];
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  maxReconnectDelay?: number;
  heartbeatInterval?: number;
  heartbeatTimeout?: number;
  queueSize?: number;
  debug?: boolean;
}

interface WebSocketMessage<T = unknown> {
  id: string;
  type: string;
  payload: T;
  timestamp: number;
  correlationId?: string;
}

// SSE Types
interface SSEConfig {
  url: string;
  headers?: Record<string, string>;
  autoReconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  debug?: boolean;
}

interface SSEMessage<T = unknown> {
  id?: string;
  event: string;
  data: T;
  timestamp: number;
}

// Connection State
type ConnectionState = 
  | 'disconnected' 
  | 'connecting' 
  | 'connected' 
  | 'reconnecting' 
  | 'error';

interface ConnectionInfo {
  state: ConnectionState;
  url: string;
  connectedAt?: Date;
  disconnectedAt?: Date;
  reconnectAttempts: number;
  latency?: number;
  error?: Error;
}
```

---

## 4. State Management Patterns

### 4.1 TanStack Query (React Query) Configuration

```typescript
// Query client configuration
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,      // 5 minutes
      gcTime: 30 * 60 * 1000,         // 30 minutes (formerly cacheTime)
      retry: 3,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});
```

### 4.2 Query Key Patterns

```typescript
// Provider query keys
const providerQueryKeys = {
  all: ['providers'] as const,
  lists: () => [...providerQueryKeys.all, 'list'] as const,
  list: (filters: string) => [...providerQueryKeys.lists(), filters] as const,
  details: () => [...providerQueryKeys.all, 'detail'] as const,
  detail: (id: string) => [...providerQueryKeys.details(), id] as const,
  health: () => [...providerQueryKeys.all, 'health'] as const,
  active: () => [...providerQueryKeys.all, 'active'] as const,
};
```

### 4.3 Zustand Stores

| Store | Purpose | Key State |
|-------|---------|-----------|
| `SessionStore` | Session management | `sessionId`, `model`, `provider` |
| `ProvidersStore` | Provider configuration | `providers`, `activeProvider`, `models` |
| `JailbreakStore` | Jailbreak state | `sessions`, `activeSession`, `results` |
| `ConfigStore` | App configuration | `settings`, `preferences`, `flags` |

### 4.4 Provider Sync Context

```typescript
// ProviderSyncContext provides:
interface ProviderSyncContextValue {
  // State
  status: SyncStatus;
  isConnected: boolean;
  error?: string;
  providers: ProviderSyncInfo[];
  models: ModelSpecification[];
  activeProvider?: ProviderSyncInfo;
  activeModel?: ModelSpecification;
  
  // Actions
  selectProvider: (providerId: string, modelId?: string) => Promise<SelectionResult>;
  selectModel: (modelId: string) => Promise<SelectionResult>;
  forceSync: () => Promise<void>;
  
  // Computed
  availableProviders: ProviderSyncInfo[];
  availableModels: ModelSpecification[];
}
```

---

## 5. Authentication Flow

### 5.1 AuthManager Singleton ([`frontend/src/lib/api/auth-manager.ts`](frontend/src/lib/api/auth-manager.ts))

```typescript
class AuthManager {
  // Storage keys
  private TOKEN_STORAGE_KEY = 'chimera_auth_tokens';
  private USER_STORAGE_KEY = 'chimera_user';
  private TENANT_STORAGE_KEY = 'chimera_tenant_id';
  
  // Configuration
  private TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before expiry
  
  // Core methods
  async login(email: string, password: string): Promise<boolean>;
  async getAccessToken(): Promise<string | null>;
  async refreshAccessToken(): Promise<string | null>;
  logout(): void;
  isAuthenticated(): boolean;
  hasRole(role: string): boolean;
  hasPermission(permission: string): boolean;
  onAuthStateChange(callback: (isAuthenticated: boolean) => void): () => void;
}

export const authManager = new AuthManager();
```

### 5.2 Token Flow

```
1. Login Request
   POST /api/v1/auth/login { email, password }
   Response: { tokens: AuthTokens, user: User }

2. Token Storage
   localStorage.setItem('chimera_auth_tokens', JSON.stringify(tokens))
   localStorage.setItem('chimera_user', JSON.stringify(user))

3. Request Authorization
   headers: { 'Authorization': `Bearer ${access_token}` }
   OR
   headers: { 'X-API-Key': api_key }

4. Token Refresh (5 min before expiry)
   POST /api/v1/auth/refresh { refresh_token }
   Response: AuthTokens

5. On 401 Response
   Attempt token refresh
   If refresh fails → logout() → redirect to login
```

### 5.3 API Key Support

```typescript
// DeepTeam client with X-API-Key
class DeepTeamApiClient {
  constructor(baseUrl: string, apiKey?: string) {
    this.headers = {
      'Content-Type': 'application/json',
      ...(apiKey && { 'X-API-Key': apiKey }),
    };
  }
}
```

---

## 6. WebSocket Implementations

### 6.1 WebSocket Manager ([`frontend/src/lib/sync/websocket-manager.ts`](frontend/src/lib/sync/websocket-manager.ts))

**Features:**
- Auto-reconnection with exponential backoff (1s → 30s max)
- Heartbeat ping/pong (30s interval, 10s timeout)
- Message queuing when disconnected (100 message limit)
- Request/response correlation with timeout
- Event-based architecture

**Configuration:**
```typescript
const DEFAULT_CONFIG: Required<WebSocketConfig> = {
  url: '',
  protocols: [],
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectDelay: 1000,
  maxReconnectDelay: 30000,
  heartbeatInterval: 30000,
  heartbeatTimeout: 10000,
  queueSize: 100,
  debug: false,
};
```

**React Hook:**
```typescript
const {
  connectionState,
  connectionInfo,
  isConnected,
  connect,
  disconnect,
  send,
  request,
  on,
  onEvent,
  error,
  latency,
} = useWebSocket({ url: 'ws://...', connectOnMount: true });
```

### 6.2 WebSocket Endpoints

| Endpoint | Component | Message Types |
|----------|-----------|---------------|
| `/api/v1/provider-config/ws/updates` | `ProviderSyncContext` | fullSync, providerAdded/Updated/Removed, heartbeat |
| `/api/v1/deepteam/jailbreak/ws/generate` | `JailbreakGenerator` | generation_start, generation_progress, prompt_generated, generation_complete, generation_error |
| `/ws/sessions/{sessionId}` | `DeepTeamApiClient` | session updates, agent status |
| `/api/v1/provider-sync/ws` | `ProviderSyncService` | sync events |

### 6.3 Jailbreak WebSocket ([`frontend/src/api/jailbreak.ts`](frontend/src/api/jailbreak.ts))

```typescript
class JailbreakWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private heartbeatInterval: NodeJS.Timer | null = null;

  async connect(
    request: JailbreakGenerationRequest,
    onMessage: (data: WebSocketMessage) => void,
    onError?: (error: Error) => void,
    onClose?: () => void
  ): Promise<void>;

  disconnect(): void;
  
  // ISSUE: Hardcoded URL
  private baseWsUrl = 'ws://localhost:8001';
}
```

---

## 7. SSE (Server-Sent Events) Implementations

### 7.1 SSE Manager ([`frontend/src/lib/sync/sse-manager.ts`](frontend/src/lib/sync/sse-manager.ts))

**Features:**
- Auto-reconnection with exponential backoff
- Event type filtering
- Connection state tracking
- React hook integration

**Configuration:**
```typescript
const DEFAULT_CONFIG: Required<SSEConfig> = {
  url: '',
  headers: {},
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectDelay: 1000,
  debug: false,
};
```

**React Hooks:**
```typescript
// General SSE hook
const { connectionState, lastMessage, messages, on } = useSSE({
  url: '/api/v1/streaming/...',
  eventTypes: ['text', 'complete', 'error'],
});

// Specialized streaming text hook
const { text, isStreaming, isComplete, start, stop, reset, error } = useStreamingText({
  url: '/api/v1/streaming/generate/stream',
  textEvent: 'text',
  completeEvent: 'complete',
  errorEvent: 'error',
});
```

### 7.2 SSE Endpoints

| Endpoint | Component | Event Types |
|----------|-----------|-------------|
| `/api/v1/streaming/generate/stream` | Generation | text, complete, error |
| `/api/v1/deepteam/jailbreak/generate/stream` | `JailbreakSSE` | generation_start, progress, prompt, complete, error |

### 7.3 Jailbreak SSE ([`frontend/src/api/jailbreak.ts`](frontend/src/api/jailbreak.ts))

```typescript
class JailbreakSSE {
  private eventSource: EventSource | null = null;

  connect(
    request: JailbreakGenerationRequest,
    onMessage: (data: SSEMessage) => void,
    onError?: (error: Error) => void,
    onComplete?: () => void
  ): void;

  disconnect(): void;
}
```

---

## 8. Error Handling

### 8.1 Error Class Hierarchy ([`frontend/src/lib/errors/api-errors.ts`](frontend/src/lib/errors/api-errors.ts))

```
APIError (abstract base)
├── Client Errors (4xx)
│   ├── ValidationError (400)
│   ├── AuthenticationError (401)
│   ├── AuthorizationError (403)
│   ├── NotFoundError (404)
│   ├── ConflictError (409)
│   └── RateLimitError (429)
├── Server Errors (5xx)
│   ├── InternalError (500)
│   ├── ServiceUnavailableError (503)
│   └── GatewayTimeoutError (504)
├── LLM Provider Errors
│   ├── LLMProviderError (500)
│   ├── LLMConnectionError (500)
│   ├── LLMTimeoutError (408)
│   ├── LLMQuotaExceededError (429)
│   ├── LLMInvalidResponseError (500)
│   └── LLMContentBlockedError (500)
├── Transformation Errors
│   ├── TransformationError (500)
│   ├── InvalidPotencyError (500)
│   └── InvalidTechniqueError (500)
├── Resilience Errors
│   └── CircuitBreakerOpenError (503)
└── Network Errors
    ├── NetworkError (0)
    └── RequestAbortedError (0)
```

### 8.2 Error Mapper ([`frontend/src/lib/errors/error-mapper.ts`](frontend/src/lib/errors/error-mapper.ts))

```typescript
// Maps Axios errors to frontend exception classes
export function mapBackendError(error: AxiosError): APIError;

// Utility functions
export function isAPIError(error: unknown): error is APIError;
export function isRetryableError(error: APIError): boolean;
export function getRetryDelay(error: APIError): number;
```

### 8.3 Global Error Handler ([`frontend/src/lib/errors/global-error-handler.ts`](frontend/src/lib/errors/global-error-handler.ts))

```typescript
// Main handler with toast notifications
export function handleApiError(
  error: unknown,
  context: string,
  options?: ErrorHandlerOptions
): APIError;

// Silent handler (no throw)
export function handleApiErrorSilent(
  error: unknown,
  context: string,
  options?: Omit<ErrorHandlerOptions, 'throwError'>
): APIError;

// React Query error handler factory
export function createQueryErrorHandler(context: string): (error: Error) => void;

// Form error handler factory
export function createFormErrorHandler(context: string): (error: unknown) => string;
```

### 8.4 Error Toast Configuration

| Error Type | Toast Type | Title | Action |
|------------|------------|-------|--------|
| `NetworkError` | error | "Connection Failed" | Retry button |
| `RateLimitError` | warning | "Rate Limited" | Auto-dismiss after `retryAfter` |
| `CircuitBreakerOpenError` | warning | "Service Recovering" | Auto-dismiss |
| `AuthenticationError` | error | "Authentication Required" | - |
| `AuthorizationError` | error | "Access Denied" | - |
| `LLMContentBlockedError` | warning | "Content Blocked" | - |
| `LLMProviderError` | error | "AI Provider Error" | - |
| `ValidationError` | warning | "Invalid Input" | - |
| `ServiceUnavailableError` | error | "Service Unavailable" | Retry button |

---

## 9. Issues & Inconsistencies

### 9.1 Critical Issues

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| Hardcoded WebSocket URL | [`frontend/src/api/jailbreak.ts:228`](frontend/src/api/jailbreak.ts:228) | **HIGH** | `ws://localhost:8001` hardcoded instead of using env variable |
| Deprecated API Client | [`frontend/src/lib/api-enhanced.ts`](frontend/src/lib/api-enhanced.ts) | MEDIUM | 2485-line monolithic file marked deprecated but still in use |
| Multiple API Client Patterns | Various | MEDIUM | Inconsistent client architecture (monolithic vs modular) |

### 9.2 Missing Backend Endpoints (Frontend expects but may not exist)

| Endpoint | Frontend Usage | Notes |
|----------|----------------|-------|
| `/api/v1/provider-sync/*` | `ProviderSyncService` | Full sync endpoint system |
| `/api/v1/deepteam/agents` | `DeepTeamApiClient` | Agent listing/details |
| `/api/v1/deepteam/evaluations` | `DeepTeamApiClient` | Evaluation listing/details |
| `/api/v1/deepteam/refinements` | `DeepTeamApiClient` | Refinement listing/apply |

### 9.3 Type Mismatches

| Frontend Type | Backend Type | Mismatch |
|---------------|--------------|----------|
| `expires_in: number` | `expires_in: int` | Units unclear (seconds vs ms) |
| `LLMProviderType` enum | Backend `LLMProviderType` | Frontend may be missing newer providers |

### 9.4 Configuration Issues

| Issue | Current | Recommended |
|-------|---------|-------------|
| Auth token refresh threshold | 5 minutes | Should be configurable |
| WebSocket reconnect attempts | 5-10 (varies) | Should be unified |
| Cache TTL | Various defaults | Should be centralized |

### 9.5 Architectural Observations

1. **Duplicate Jailbreak Clients**: Both `api/jailbreak.ts` and `lib/api-enhanced.ts` have jailbreak endpoints
2. **Inconsistent Error Handling**: Some components use `handleApiError`, others use direct try/catch
3. **Missing Loading States**: Some API calls don't track loading state
4. **No Request Deduplication**: Multiple components may trigger same API call

---

## 10. Recommendations

### 10.1 Immediate Actions

1. **Fix Hardcoded WebSocket URL**
   ```typescript
   // Change from:
   private baseWsUrl = 'ws://localhost:8001';
   // To:
   private baseWsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001';
   ```

2. **Migrate from Deprecated API Client**
   - Create modular services in `lib/api/services/`
   - Gradually move endpoints from `api-enhanced.ts`

3. **Unify WebSocket Configuration**
   - Create single `WebSocketConfig` for all connections
   - Use environment variables for all URLs

### 10.2 Medium-Term Improvements

1. **Centralize API Configuration**
   ```typescript
   // Create config/api.ts
   export const API_CONFIG = {
     baseUrl: process.env.NEXT_PUBLIC_API_URL,
     wsUrl: process.env.NEXT_PUBLIC_WS_URL,
     timeouts: { default: 30000, llm: 120000, extended: 600000 },
     retry: { attempts: 3, delay: 1000 },
   };
   ```

2. **Implement Request Deduplication**
   - Use React Query's built-in deduplication
   - Add request caching layer

3. **Add Comprehensive Error Boundaries**
   - Create feature-specific error boundaries
   - Implement error recovery UI

### 10.3 Long-Term Architecture

1. **API Client Refactoring**
   - Adopt a consistent pattern across all services
   - Generate types from OpenAPI spec

2. **State Management Consolidation**
   - Reduce Zustand stores where React Query suffices
   - Implement proper cache invalidation strategy

3. **Real-time Communication**
   - Unify WebSocket and SSE management
   - Implement proper offline support

---

## 11. Summary Statistics

### API Endpoint Coverage

| Category | Backend Endpoints | Frontend Coverage |
|----------|-------------------|-------------------|
| Health | 11 | 4 (36%) |
| Auth | 2 | 2 (100%) |
| Providers | 8 | 8 (100%) |
| Session | 9 | 9 (100%) |
| Generation | 3 | 3 (100%) |
| Streaming | 3 | 2 (67%) |
| Jailbreak | 15 | 12 (80%) |
| AutoDAN | 4 | 4 (100%) |
| AutoDAN-Turbo | 19 | 17 (89%) |
| DeepTeam | 14 | 10 (71%) |
| Admin | 14 | 0 (0%) |
| Metrics | 11 | 0 (0%) |
| **Total** | **95+** | **~75 (79%)** |

### WebSocket/SSE Coverage

| Backend WebSocket | Frontend Implementation |
|-------------------|------------------------|
| `/ws/enhance` | ❌ Not implemented |
| `/api/v1/providers/ws/selection` | ✅ `ProviderSyncContext` |
| `/api/v1/jailbreak/ws/generate` | ⚠️ Hardcoded URL |
| `/api/v1/deepteam/jailbreak/ws/generate` | ✅ `JailbreakWebSocket` |
| `/api/v1/autoadv/ws` | ❌ Not implemented |

### Type System Health

| Metric | Count |
|--------|-------|
| Total TypeScript files | 200+ |
| Type definition files | 20+ |
| `any` type usages | ~15 (acceptable) |
| Missing types | ~5 interfaces |

---

*Report generated automatically as part of the Chimera Backend-Frontend Gap Analysis Project, Phase 2.*