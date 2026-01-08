# Project Chimera - Comprehensive Full-Stack Integration Audit

**Date:** December 11, 2024  
**Auditor:** Antigravity AI  
**Version:** 1.0.0

---

## Executive Summary

This document presents a comprehensive audit of Project Chimera's frontend and backend architectures, identifies patterns and best practices in the backend that should be adopted by the frontend, and provides a robust integration strategy for creating a fully cohesive full-stack system.

### Key Findings

| Category | Backend Status | Frontend Status | Gap Severity |
|----------|---------------|-----------------|--------------|
| Error Handling | ✅ Excellent | ⚠️ Basic | **HIGH** |
| Provider Pattern | ✅ Implemented | ❌ Missing | **HIGH** |
| Circuit Breaker | ✅ Implemented | ❌ Missing | **MEDIUM** |
| Type Safety | ✅ Pydantic Models | ⚠️ TypeScript Interfaces | **MEDIUM** |
| Session Management | ✅ Full Service | ⚠️ API Calls Only | **MEDIUM** |
| WebSocket | ✅ Heartbeat+Reconnect | ⚠️ Basic Reconnect | **LOW** |
| Caching | ✅ Multi-layer | ❌ Missing | **MEDIUM** |
| Dependency Injection | ✅ FastAPI Depends | ⚠️ Context/Props | **LOW** |

---

## Part 1: Backend Architecture Audit

### 1.1 Architectural Patterns

#### 1.1.1 Domain-Driven Design (DDD)
```
backend-api/app/
├── domain/           # Domain layer (business entities)
│   ├── models.py     # Core domain models
│   ├── interfaces.py # Abstract interfaces
│   ├── jailbreak/    # Domain-specific subdomain
│   └── houyi/        # Domain-specific subdomain
├── services/         # Application services
├── engines/          # Core business logic engines
├── infrastructure/   # External integrations
└── api/              # Presentation layer (HTTP/WS)
```

**Pattern Rating:** ⭐⭐⭐⭐⭐ (Excellent)

#### 1.1.2 Service Layer Pattern
The backend implements a clean service layer with:

```python
# Example: LLMService (app/services/llm_service.py)
class LLMService:
    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._default_provider: str | None = None
    
    def register_provider(self, name: str, provider: LLMProvider, is_default: bool = False)
    def get_provider(self, name: str | None = None) -> LLMProvider
    async def generate_text(self, request: PromptRequest) -> PromptResponse
    async def list_providers(self) -> ProviderListResponse
```

**Key Services Identified:**
| Service | Purpose | Location |
|---------|---------|----------|
| `LLMService` | LLM provider orchestration | `services/llm_service.py` |
| `SessionService` | User session management | `services/session_service.py` |
| `TransformationEngine` | Prompt transformation | `services/transformation_service.py` |
| `JailbreakService` | Jailbreak technique execution | `services/jailbreak/` |
| `MetamorphService` | Dynamic transformations | `services/metamorph_service.py` |

#### 1.1.3 Provider Pattern with Registry
```python
# Provider Registration Pattern
class CircuitBreakerRegistry:
    _breakers: dict[str, CircuitBreakerState] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get(cls, name: str) -> CircuitBreakerState
    
    @classmethod
    def reset(cls, name: str)
    
    @classmethod
    def get_all_states(cls) -> dict[str, dict]
```

#### 1.1.4 Circuit Breaker Pattern
```python
# Circuit Breaker States
class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit tripped, fail fast
    HALF_OPEN = "half_open" # Testing recovery

# Usage Pattern
@circuit_breaker("gemini", failure_threshold=3, recovery_timeout=60)
async def call_gemini_api(prompt: str):
    ...
```

---

### 1.2 API Design Principles

#### 1.2.1 RESTful Endpoint Structure
```
GET    /api/v1/providers          # List providers
GET    /api/v1/techniques         # List techniques
POST   /api/v1/transform          # Transform prompt
POST   /api/v1/execute            # Transform and execute
POST   /api/v1/generate           # Direct LLM generation
POST   /api/v1/generation/jailbreak/generate  # Jailbreak generation
GET    /api/v1/session/info       # Session information
POST   /api/v1/session/create     # Create session
PUT    /api/v1/session/model      # Update model selection
GET    /api/v1/models/list        # List available models
POST   /api/v1/models/validate    # Validate model selection
WS     /ws/enhance               # WebSocket for real-time enhancement
```

#### 1.2.2 Request/Response Models
The backend uses Pydantic models with comprehensive validation:

```python
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=50000)
    system_instruction: str | None = Field(None, max_length=10000)
    config: GenerationConfig | None = Field(default_factory=GenerationConfig)
    model: str | None = Field(None)
    provider: LLMProviderType | None = Field(None)
    api_key: str | None = Field(None)
    skip_validation: bool = Field(False)

    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_dangerous_patterns(self):
        # Security validation
        ...
```

---

### 1.3 Data Flow Analysis

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   Client    │────▶│  API Layer  │────▶│   Services   │
│  (Frontend) │     │  (FastAPI)  │     │   (Logic)    │
└─────────────┘     └─────────────┘     └──────────────┘
                            │                   │
                            ▼                   ▼
                    ┌───────────────┐   ┌───────────────┐
                    │  Middleware   │   │    Engines    │
                    │  (Auth/CORS)  │   │ (Transform)   │
                    └───────────────┘   └───────────────┘
                            │                   │
                            ▼                   ▼
                    ┌───────────────┐   ┌───────────────┐
                    │    Domain     │   │  Providers    │
                    │   (Models)    │   │  (LLM APIs)   │
                    └───────────────┘   └───────────────┘
```

---

### 1.4 Security Models

#### 1.4.1 Authentication Middleware Stack
```python
# Middleware order (first = outermost)
app.add_middleware(ObservabilityMiddleware)      # Tracing/Metrics
app.add_middleware(APIKeyMiddleware, ...)        # Authentication
app.add_middleware(RequestLoggingMiddleware)     # Audit logging
app.add_middleware(RateLimitMiddleware)          # Rate limiting
app.add_middleware(InputValidationMiddleware)    # Input sanitization
app.add_middleware(CSRFMiddleware)               # CSRF protection
app.add_middleware(SecurityHeadersMiddleware)   # XSS/Clickjacking
app.add_middleware(JailbreakSecurityMiddleware) # Jailbreak controls
app.add_middleware(CORSMiddleware, ...)         # CORS
```

#### 1.4.2 Security Headers Implementation
- Content-Security-Policy
- X-Content-Type-Options
- X-Frame-Options
- Strict-Transport-Security
- X-XSS-Protection

---

### 1.5 Engine Logic Analysis

#### 1.5.1 Transformation Engine
```python
class TransformationStrategy(str, Enum):
    SIMPLE = "simple"
    LAYERED = "layered"
    RECURSIVE = "recursive"
    QUANTUM = "quantum"
    AI_BRAIN = "ai_brain"
    CODE_CHAMELEON = "code_chameleon"
    DEEP_INCEPTION = "deep_inception"
    CIPHER = "cipher"
    AUTODAN = "autodan"

class TransformationEngine:
    def __init__(self, enable_cache: bool = True)
    def transform(self, prompt, potency_level, technique_suite, use_cache=True)
    def _determine_strategy(self, potency_level, technique_suite)
    def _apply_transformation(self, prompt, potency_level, technique_suite, strategy)
```

#### 1.5.2 Transformer Engine (Base Pattern)
```python
class BaseTransformerEngine(abc.ABC, NextGenerationTransformationMixin):
    def __init__(self, llm_client, config)
    
    def transform(self, data: IntentData) -> TransformationResult:
        # Template Method Pattern
        prompt = self._generate_strategy_prompt(data)
        try:
            result = await self._call_llm(prompt)
            return self._process_llm_result(result, data)
        except Exception:
            return self._get_fallback_content(data)
    
    @abc.abstractmethod
    def _generate_strategy_prompt(self, data: IntentData) -> str
    
    @abc.abstractmethod
    def _get_fallback_content(self, data: IntentData) -> str
```

---

## Part 2: Frontend Current State Analysis

### 2.1 Architecture Overview

```
frontend/src/
├── app/              # Next.js App Router pages
│   ├── dashboard/    # Dashboard pages
│   └── gptfuzz/      # GPTFuzz pages
├── components/       # React components
│   ├── ui/           # Shadcn UI primitives
│   └── *.tsx         # Feature components
├── lib/              # Core utilities
│   ├── api-enhanced.ts   # API client
│   ├── api-config.ts     # Configuration
│   ├── api-types.ts      # Type definitions
│   └── use-websocket.ts  # WebSocket hooks
├── config/           # Configuration
├── providers/        # React context providers
└── types/            # TypeScript types
```

### 2.2 Current API Layer

**Strengths:**
- ✅ Axios-based HTTP client with interceptors
- ✅ Type-safe API calls with TypeScript
- ✅ Error handling with toast notifications
- ✅ Connection state tracking
- ✅ Multi-mode support (proxy/direct)
- ✅ API key management from environment

**Weaknesses:**
- ❌ No circuit breaker pattern
- ❌ No retry logic with exponential backoff
- ❌ No request queue/debouncing
- ❌ Error types not aligned with backend
- ❌ No data transformation layer
- ❌ No caching layer

### 2.3 Current Type Definitions

The frontend has types in `api-types.ts` that mirror backend models but:
- Types are duplicated in `api-enhanced.ts` and `api-types.ts`
- No runtime validation (unlike Pydantic)
- Some type misalignments with backend

### 2.4 Current WebSocket Implementation

```typescript
// Current implementation (use-websocket.ts)
function useWebSocket(endpoint: string, options: UseWebSocketOptions) {
  // ✅ Auto-connect
  // ✅ Message parsing
  // ✅ Reconnection attempts
  // ❌ No heartbeat handling (backend sends heartbeat)
  // ❌ No pong response mechanism
  // ❌ No connection quality monitoring
}
```

---

## Part 3: Gap Analysis & Integration Requirements

### 3.1 Critical Gaps

| Gap ID | Description | Backend Pattern | Frontend Missing | Priority |
|--------|-------------|-----------------|------------------|----------|
| GAP-001 | Error Hierarchy | `APIException` tree | Single Error type | **CRITICAL** |
| GAP-002 | Circuit Breaker | `CircuitBreakerRegistry` | None | **HIGH** |
| GAP-003 | Provider Registry | `LLMService._providers` | None | **HIGH** |
| GAP-004 | Session Service | `SessionService` | API calls only | **MEDIUM** |
| GAP-005 | Data Transform | Pydantic validators | None | **MEDIUM** |
| GAP-006 | Caching Layer | `TransformationCache` | None | **MEDIUM** |
| GAP-007 | WS Heartbeat | Server-side heartbeat | No pong handling | **LOW** |

### 3.2 Type Synchronization Requirements

```
Backend (Pydantic)          Frontend (TypeScript)
───────────────────         ────────────────────
PromptRequest        <--->  GenerateRequest  ✅ Aligned
PromptResponse       <--->  GenerateResponse ✅ Aligned
TransformationRequest <---> TransformRequest  ✅ Aligned
ExecutionRequest     <--->  ExecuteRequest   ✅ Aligned
JailbreakRequest     <--->  JailbreakRequest ✅ Aligned
APIException         <--->  ??? ❌ MISSING
LLMProviderError     <--->  ??? ❌ MISSING
CircuitBreakerState  <--->  ??? ❌ MISSING
SessionInfo          <--->  SessionInfoResponse ✅ Aligned
```

---

## Part 4: Integration Strategy

### 4.1 Phase 1: Error Handling Alignment (Priority: CRITICAL)

#### 4.1.1 Create Frontend Exception Hierarchy

```typescript
// lib/errors/api-errors.ts

export abstract class APIError extends Error {
  abstract readonly statusCode: number;
  abstract readonly errorCode: string;
  readonly details?: Record<string, unknown>;
  readonly timestamp: string;
  readonly requestId?: string;

  constructor(message: string, details?: Record<string, unknown>) {
    super(message);
    this.details = details;
    this.timestamp = new Date().toISOString();
  }

  toJSON() {
    return {
      error: this.errorCode,
      message: this.message,
      status_code: this.statusCode,
      details: this.details,
      timestamp: this.timestamp,
      request_id: this.requestId,
    };
  }
}

export class ValidationError extends APIError {
  readonly statusCode = 400;
  readonly errorCode = "VALIDATION_ERROR";
}

export class AuthenticationError extends APIError {
  readonly statusCode = 401;
  readonly errorCode = "AUTHENTICATION_ERROR";
}

export class AuthorizationError extends APIError {
  readonly statusCode = 403;
  readonly errorCode = "AUTHORIZATION_ERROR";
}

export class NotFoundError extends APIError {
  readonly statusCode = 404;
  readonly errorCode = "NOT_FOUND";
}

export class RateLimitError extends APIError {
  readonly statusCode = 429;
  readonly errorCode = "RATE_LIMIT_EXCEEDED";
  readonly retryAfter?: number;

  constructor(message: string, retryAfter?: number) {
    super(message, { retry_after: retryAfter });
    this.retryAfter = retryAfter;
  }
}

export class LLMProviderError extends APIError {
  readonly statusCode = 500;
  readonly errorCode = "LLM_PROVIDER_ERROR";
  readonly provider?: string;
}

export class CircuitBreakerOpenError extends APIError {
  readonly statusCode = 503;
  readonly errorCode = "CIRCUIT_BREAKER_OPEN";
  readonly retryAfter: number;

  constructor(providerName: string, retryAfter: number) {
    super(`Circuit breaker '${providerName}' is open. Retry after ${retryAfter}s`);
    this.retryAfter = retryAfter;
  }
}

export class TransformationError extends APIError {
  readonly statusCode = 500;
  readonly errorCode = "TRANSFORMATION_ERROR";
}
```

#### 4.1.2 Add Error Mapping from Backend

```typescript
// lib/errors/error-mapper.ts

import { AxiosError } from "axios";
import {
  APIError,
  ValidationError,
  AuthenticationError,
  RateLimitError,
  LLMProviderError,
  // ... other errors
} from "./api-errors";

interface BackendErrorResponse {
  error: string;
  message: string;
  status_code: number;
  details?: Record<string, unknown>;
  request_id?: string;
}

export function mapBackendError(error: AxiosError): APIError {
  const data = error.response?.data as BackendErrorResponse | undefined;
  const status = error.response?.status || 500;
  
  if (!data?.error) {
    // Network or unknown error
    if (error.code === "ERR_NETWORK") {
      return new ServiceUnavailableError("Backend server unavailable");
    }
    return new InternalError(error.message);
  }

  // Map backend error codes to frontend exceptions
  const errorMap: Record<string, new (message: string, details?: any) => APIError> = {
    "VALIDATION_ERROR": ValidationError,
    "AUTHENTICATION_ERROR": AuthenticationError,
    "AUTHORIZATION_ERROR": AuthorizationError,
    "NOT_FOUND": NotFoundError,
    "RATE_LIMIT_EXCEEDED": RateLimitError,
    "LLM_PROVIDER_ERROR": LLMProviderError,
    "LLM_CONNECTION_ERROR": LLMConnectionError,
    "LLM_TIMEOUT_ERROR": LLMTimeoutError,
    "TRANSFORMATION_ERROR": TransformationError,
    "CIRCUIT_BREAKER_OPEN": CircuitBreakerOpenError,
    // ... add all backend error codes
  };

  const ErrorClass = errorMap[data.error] || InternalError;
  return new ErrorClass(data.message, data.details);
}
```

### 4.2 Phase 2: Circuit Breaker Implementation (Priority: HIGH)

```typescript
// lib/resilience/circuit-breaker.ts

export enum CircuitState {
  CLOSED = "closed",
  OPEN = "open",
  HALF_OPEN = "half_open",
}

interface CircuitBreakerState {
  name: string;
  state: CircuitState;
  failureCount: number;
  successCount: number;
  lastFailureTime: number;
  lastStateChange: number;
}

class CircuitBreakerRegistry {
  private static breakers: Map<string, CircuitBreakerState> = new Map();

  static get(name: string): CircuitBreakerState {
    if (!this.breakers.has(name)) {
      this.breakers.set(name, {
        name,
        state: CircuitState.CLOSED,
        failureCount: 0,
        successCount: 0,
        lastFailureTime: 0,
        lastStateChange: Date.now(),
      });
    }
    return this.breakers.get(name)!;
  }

  static reset(name: string): void {
    const breaker = this.get(name);
    breaker.state = CircuitState.CLOSED;
    breaker.failureCount = 0;
    breaker.successCount = 0;
  }

  static getAllStates(): Record<string, CircuitBreakerState> {
    const states: Record<string, CircuitBreakerState> = {};
    this.breakers.forEach((v, k) => { states[k] = v; });
    return states;
  }
}

interface CircuitBreakerOptions {
  failureThreshold?: number;
  recoveryTimeout?: number;
  halfOpenMaxCalls?: number;
}

export function withCircuitBreaker<T>(
  name: string,
  fn: () => Promise<T>,
  options: CircuitBreakerOptions = {}
): Promise<T> {
  const {
    failureThreshold = 3,
    recoveryTimeout = 60000,
    halfOpenMaxCalls = 1,
  } = options;

  const breaker = CircuitBreakerRegistry.get(name);
  const now = Date.now();

  // Check if circuit should transition from OPEN to HALF_OPEN
  if (breaker.state === CircuitState.OPEN) {
    const timeSinceFailure = now - breaker.lastFailureTime;
    if (timeSinceFailure >= recoveryTimeout) {
      breaker.state = CircuitState.HALF_OPEN;
      breaker.successCount = 0;
      breaker.lastStateChange = now;
      console.log(`Circuit '${name}' transitioning to HALF_OPEN`);
    } else {
      const retryAfter = (recoveryTimeout - timeSinceFailure) / 1000;
      throw new CircuitBreakerOpenError(name, retryAfter);
    }
  }

  // Execute the function
  return fn()
    .then((result) => {
      // Success handling
      if (breaker.state === CircuitState.HALF_OPEN) {
        breaker.successCount++;
        if (breaker.successCount >= halfOpenMaxCalls) {
          breaker.state = CircuitState.CLOSED;
          breaker.failureCount = 0;
          breaker.lastStateChange = now;
          console.log(`Circuit '${name}' closed after recovery`);
        }
      } else if (breaker.state === CircuitState.CLOSED) {
        breaker.failureCount = Math.max(0, breaker.failureCount - 1);
      }
      return result;
    })
    .catch((error) => {
      // Failure handling
      breaker.failureCount++;
      breaker.lastFailureTime = now;

      if (breaker.state === CircuitState.HALF_OPEN) {
        breaker.state = CircuitState.OPEN;
        breaker.lastStateChange = now;
        console.warn(`Circuit '${name}' re-opened after recovery failure`);
      } else if (breaker.failureCount >= failureThreshold) {
        breaker.state = CircuitState.OPEN;
        breaker.lastStateChange = now;
        console.warn(`Circuit '${name}' opened after ${failureThreshold} failures`);
      }

      throw error;
    });
}
```

### 4.3 Phase 3: Provider Service Pattern (Priority: HIGH)

```typescript
// lib/services/provider-service.ts

import { withCircuitBreaker } from "../resilience/circuit-breaker";

export interface ProviderInfo {
  name: string;
  status: "active" | "inactive" | "error";
  models: string[];
  defaultModel: string;
}

interface ProviderConfig {
  name: string;
  baseUrl: string;
  apiKey?: string;
  isDefault?: boolean;
}

class ProviderService {
  private providers: Map<string, ProviderConfig> = new Map();
  private defaultProvider: string | null = null;

  registerProvider(config: ProviderConfig): void {
    this.providers.set(config.name, config);
    if (config.isDefault) {
      this.defaultProvider = config.name;
    }
    console.log(`Registered provider: ${config.name} (default=${config.isDefault})`);
  }

  getProvider(name?: string): ProviderConfig {
    const providerName = name || this.defaultProvider;
    if (!providerName || !this.providers.has(providerName)) {
      throw new LLMProviderError(`Provider not available: ${providerName || "default"}`);
    }
    return this.providers.get(providerName)!;
  }

  async callProvider<T>(
    providerName: string,
    operation: () => Promise<T>
  ): Promise<T> {
    return withCircuitBreaker(providerName, operation, {
      failureThreshold: 3,
      recoveryTimeout: 60000,
    });
  }

  listProviders(): ProviderInfo[] {
    return Array.from(this.providers.entries()).map(([name, config]) => ({
      name,
      status: "active", // Would check circuit breaker state
      models: [], // Would fetch from backend
      defaultModel: "",
    }));
  }
}

export const providerService = new ProviderService();
```

### 4.4 Phase 4: Session Management Service (Priority: MEDIUM)

```typescript
// lib/services/session-service.ts

import { enhancedApi } from "../api-enhanced";

interface Session {
  sessionId: string;
  provider: string;
  model: string;
  createdAt: Date;
  lastActivity: Date;
  requestCount: number;
}

class SessionService {
  private currentSession: Session | null = null;
  private sessionCheckInterval: NodeJS.Timeout | null = null;

  async initialize(provider?: string, model?: string): Promise<Session> {
    const response = await enhancedApi.session.create({ provider, model });
    
    this.currentSession = {
      sessionId: response.data.session_id,
      provider: response.data.provider,
      model: response.data.model,
      createdAt: new Date(),
      lastActivity: new Date(),
      requestCount: 0,
    };

    // Start session monitoring
    this.startSessionMonitoring();

    return this.currentSession;
  }

  get session(): Session | null {
    return this.currentSession;
  }

  get isActive(): boolean {
    return this.currentSession !== null;
  }

  async updateModel(provider: string, model: string): Promise<boolean> {
    if (!this.currentSession) {
      throw new Error("No active session");
    }

    const response = await enhancedApi.session.updateModel(
      this.currentSession.sessionId,
      { provider, model }
    );

    if (response.data.success) {
      this.currentSession.provider = response.data.provider;
      this.currentSession.model = response.data.model;
      this.currentSession.lastActivity = new Date();
      return true;
    }

    return false;
  }

  async refresh(): Promise<Session | null> {
    if (!this.currentSession) return null;

    try {
      const response = await enhancedApi.session.getInfo(this.currentSession.sessionId);
      this.currentSession.lastActivity = new Date(response.data.last_activity);
      this.currentSession.requestCount = response.data.request_count;
      return this.currentSession;
    } catch {
      this.currentSession = null;
      return null;
    }
  }

  async terminate(): Promise<void> {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
    }
    if (this.currentSession) {
      await enhancedApi.session.delete(this.currentSession.sessionId);
      this.currentSession = null;
    }
  }

  private startSessionMonitoring(): void {
    // Refresh session every 5 minutes
    this.sessionCheckInterval = setInterval(() => {
      this.refresh();
    }, 5 * 60 * 1000);
  }
}

export const sessionService = new SessionService();
```

### 4.5 Phase 5: Data Transformation Layer (Priority: MEDIUM)

```typescript
// lib/transforms/api-transforms.ts

import { z } from "zod";

// Zod schemas for runtime validation (like Pydantic in backend)
export const GenerationConfigSchema = z.object({
  temperature: z.number().min(0).max(1).default(0.7),
  top_p: z.number().min(0).max(1).default(0.95),
  top_k: z.number().min(1).default(40),
  max_output_tokens: z.number().min(1).max(8192).default(2048),
  stop_sequences: z.array(z.string().max(100)).max(10).optional(),
  thinking_level: z.enum(["low", "medium", "high"]).optional(),
});

export const PromptRequestSchema = z.object({
  prompt: z.string().min(1).max(50000).transform((s) => s.trim()),
  system_instruction: z.string().max(10000).optional(),
  config: GenerationConfigSchema.optional(),
  model: z.string().max(100).regex(/^[a-zA-Z0-9\-_.]+$/).optional(),
  provider: z.enum(["openai", "anthropic", "google", "gemini", "deepseek", "mock"]).optional(),
});

export const TransformRequestSchema = z.object({
  core_request: z.string().min(1).max(5000).transform((s) => s.trim()),
  potency_level: z.number().int().min(1).max(10),
  technique_suite: z.string().min(1).max(50).regex(/^[a-zA-Z0-9\-_]+$/),
});

// Transform functions
export function validatePromptRequest(data: unknown) {
  return PromptRequestSchema.parse(data);
}

export function validateTransformRequest(data: unknown) {
  return TransformRequestSchema.parse(data);
}

// Response normalization
export function normalizeTransformResponse(response: any): TransformResponse {
  return {
    success: response.success,
    original_prompt: response.original_prompt,
    transformed_prompt: response.transformed_prompt,
    metadata: {
      ...response.metadata,
      // Normalize layers_applied to always be an array
      layers_applied: Array.isArray(response.metadata.layers_applied)
        ? response.metadata.layers_applied
        : [response.metadata.layers_applied],
      // Ensure both technique fields exist
      techniques_used: response.metadata.techniques_used || response.metadata.applied_techniques || [],
      applied_techniques: response.metadata.applied_techniques || response.metadata.techniques_used || [],
    },
  };
}
```

### 4.6 Phase 6: Enhanced WebSocket with Heartbeat (Priority: LOW)

```typescript
// lib/use-websocket-enhanced.ts

import { useCallback, useEffect, useRef, useState } from "react";
import { getApiConfig } from "./api-config";

export function useWebSocketEnhanced(endpoint: string, options: UseWebSocketOptions = {}) {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    autoConnect = false,
    reconnectAttempts = 3,
    reconnectInterval = 3000,
    heartbeatInterval = 30000,  // Match backend (30s)
    heartbeatTimeout = 120000,  // Match backend (120s)
  } = options;

  const [status, setStatus] = useState<WebSocketStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionQuality, setConnectionQuality] = useState<"good" | "degraded" | "poor">("good");
  
  const wsRef = useRef<WebSocket | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastPingTimeRef = useRef<number>(0);
  const latencyHistoryRef = useRef<number[]>([]);

  const resetHeartbeatTimeout = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }
    heartbeatTimeoutRef.current = setTimeout(() => {
      console.warn("Heartbeat timeout - closing connection");
      wsRef.current?.close();
    }, heartbeatTimeout);
  }, [heartbeatTimeout]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data);
      
      // Handle heartbeat ping from server
      if (message.type === "ping") {
        // Calculate latency
        const latency = Date.now() - (message.timestamp * 1000);
        latencyHistoryRef.current.push(latency);
        if (latencyHistoryRef.current.length > 10) {
          latencyHistoryRef.current.shift();
        }

        // Update connection quality based on average latency
        const avgLatency = latencyHistoryRef.current.reduce((a, b) => a + b, 0) / latencyHistoryRef.current.length;
        if (avgLatency < 100) {
          setConnectionQuality("good");
        } else if (avgLatency < 500) {
          setConnectionQuality("degraded");
        } else {
          setConnectionQuality("poor");
        }

        // Send pong response
        wsRef.current?.send(JSON.stringify({ type: "pong" }));
        resetHeartbeatTimeout();
        return;
      }

      setLastMessage(message);
      onMessage?.(message);
    } catch (error) {
      console.error("Failed to parse WebSocket message:", error);
    }
  }, [onMessage, resetHeartbeatTimeout]);

  // ... rest of implementation
  
  return {
    status,
    lastMessage,
    connectionQuality,
    connect,
    disconnect,
    send,
    sendText,
    isConnected: status === "connected",
    isConnecting: status === "connecting",
    averageLatency: latencyHistoryRef.current.reduce((a, b) => a + b, 0) / (latencyHistoryRef.current.length || 1),
  };
}
```

---

## Part 5: API Endpoint Integration Map

### 5.1 Complete Endpoint Mapping

| Backend Endpoint | Method | Frontend Function | Status |
|------------------|--------|-------------------|--------|
| `/health` | GET | `enhancedApi.health.check()` | ✅ |
| `/providers` | GET | `enhancedApi.providers.list()` | ✅ |
| `/techniques` | GET | `enhancedApi.techniques.list()` | ✅ |
| `/techniques/{name}` | GET | `enhancedApi.techniques.get()` | ✅ |
| `/transform` | POST | `enhancedApi.transform.execute()` | ✅ |
| `/execute` | POST | `enhancedApi.execute.run()` | ✅ |
| `/generate` | POST | `enhancedApi.generate.text()` | ✅ |
| `/generation/jailbreak/generate` | POST | `enhancedApi.jailbreak.generate()` | ✅ |
| `/autodan/jailbreak` | POST | `enhancedApi.autodan.jailbreak()` | ✅ |
| `/optimize/optimize` | POST | `enhancedApi.optimize.houyi()` | ✅ |
| `/gptfuzz/run` | POST | `enhancedApi.gptfuzz.run()` | ✅ |
| `/gptfuzz/status/{id}` | GET | `enhancedApi.gptfuzz.status()` | ✅ |
| `/session/create` | POST | `enhancedApi.session.create()` | ✅ |
| `/session/{id}` | GET | `enhancedApi.session.getInfo()` | ✅ |
| `/session/{id}/model` | PUT | `enhancedApi.session.updateModel()` | ✅ |
| `/models/list` | GET | `enhancedApi.models.list()` | ✅ |
| `/models/validate` | POST | `enhancedApi.models.validate()` | ✅ |
| `/intent-aware/generate` | POST | `enhancedApi.intentAware.generate()` | ✅ |
| `/jailbreak/execute` | POST | ❌ Missing | ⚠️ |
| `/jailbreak/techniques` | GET | ❌ Missing | ⚠️ |
| `/jailbreak/statistics` | GET | ❌ Missing | ⚠️ |

### 5.2 Missing Frontend API Methods

```typescript
// Add to api-enhanced.ts

// Jailbreak Service endpoints (from jailbreak.py)
jailbreakService: {
  execute: (data: TechniqueExecutionRequest) =>
    apiClient.post<TechniqueExecutionResponse>("/jailbreak/execute", data),
  
  listTechniques: (params?: TechniqueSearchParams) =>
    apiClient.get<TechniqueListResponse>("/jailbreak/techniques", { params }),
  
  getTechnique: (techniqueId: string) =>
    apiClient.get<TechniqueDetail>(`/jailbreak/techniques/${techniqueId}`),
  
  validatePrompt: (prompt: string) =>
    apiClient.post<SafetyValidationResult>("/jailbreak/validate-prompt", { prompt }),
  
  getStatistics: (techniqueId?: string, timeRangeHours?: number) =>
    apiClient.get<ExecutionStatistics>("/jailbreak/statistics", {
      params: { technique_id: techniqueId, time_range_hours: timeRangeHours },
    }),
  
  searchTechniques: (query: string, limit?: number) =>
    apiClient.get<TechniqueSearchResult[]>("/jailbreak/search", {
      params: { query, limit },
    }),
  
  getHealth: () =>
    apiClient.get<JailbreakHealthStatus>("/jailbreak/health"),
  
  getAuditLogs: (params?: AuditLogParams) =>
    apiClient.get<AuditLogEntry[]>("/jailbreak/audit/logs", { params }),
},
```

---

## Part 6: Error Handling Routines

### 6.1 Global Error Handler

```typescript
// lib/errors/global-error-handler.ts

import { toast } from "sonner";
import { mapBackendError, APIError, RateLimitError, CircuitBreakerOpenError } from "./";

interface ErrorHandlerOptions {
  showToast?: boolean;
  logToConsole?: boolean;
  throwError?: boolean;
}

export async function handleApiError(
  error: unknown,
  context: string,
  options: ErrorHandlerOptions = {}
): Promise<never> {
  const { showToast = true, logToConsole = true, throwError = true } = options;

  // Map error to our type hierarchy
  const apiError = error instanceof APIError ? error : mapBackendError(error as AxiosError);

  if (logToConsole) {
    console.error(`[${context}] API Error:`, {
      code: apiError.errorCode,
      message: apiError.message,
      status: apiError.statusCode,
      details: apiError.details,
    });
  }

  if (showToast) {
    const toastConfig = getToastConfigForError(apiError);
    toast[toastConfig.type](toastConfig.title, {
      description: toastConfig.description,
      action: toastConfig.action,
    });
  }

  if (throwError) {
    throw apiError;
  }
}

function getToastConfigForError(error: APIError) {
  if (error instanceof RateLimitError) {
    return {
      type: "warning" as const,
      title: "Rate Limited",
      description: `Please wait ${error.retryAfter}s before retrying`,
      action: error.retryAfter 
        ? { label: "Retry", onClick: () => window.location.reload() }
        : undefined,
    };
  }

  if (error instanceof CircuitBreakerOpenError) {
    return {
      type: "error" as const,
      title: "Service Temporarily Unavailable",
      description: `The service is recovering. Retry in ${Math.ceil(error.retryAfter)}s`,
    };
  }

  // Default error toast
  return {
    type: "error" as const,
    title: "Error",
    description: error.message,
  };
}
```

### 6.2 Request Retry Strategy

```typescript
// lib/resilience/retry.ts

interface RetryOptions {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
  retryableStatuses?: number[];
}

export async function withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    backoffMultiplier = 2,
    retryableStatuses = [408, 429, 500, 502, 503, 504],
  } = options;

  let lastError: Error;
  let delay = initialDelay;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;

      // Check if error is retryable
      const status = error.response?.status || 0;
      const isRetryable = retryableStatuses.includes(status);

      if (!isRetryable || attempt === maxRetries) {
        throw error;
      }

      // Check for Retry-After header
      const retryAfter = error.response?.headers?.["retry-after"];
      if (retryAfter) {
        delay = parseInt(retryAfter, 10) * 1000;
      }

      console.log(`Retry attempt ${attempt + 1}/${maxRetries} after ${delay}ms`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      delay = Math.min(delay * backoffMultiplier, maxDelay);
    }
  }

  throw lastError!;
}
```

---

## Part 7: State Synchronization Mechanisms

### 7.1 React Context for Global State

```typescript
// providers/chimera-provider.tsx

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { sessionService } from "@/lib/services/session-service";
import { providerService, ProviderInfo } from "@/lib/services/provider-service";
import { CircuitBreakerRegistry, CircuitState } from "@/lib/resilience/circuit-breaker";

interface ChimeraContextValue {
  // Session State
  session: Session | null;
  isSessionActive: boolean;
  initializeSession: (provider?: string, model?: string) => Promise<void>;
  updateSessionModel: (provider: string, model: string) => Promise<void>;
  
  // Provider State
  providers: ProviderInfo[];
  defaultProvider: string | null;
  circuitStates: Record<string, { state: CircuitState; failureCount: number }>;
  
  // Connection State
  isBackendConnected: boolean;
  connectionMode: "proxy" | "direct";
  
  // Global Actions
  refreshProviders: () => Promise<void>;
  resetCircuitBreaker: (name: string) => void;
}

const ChimeraContext = createContext<ChimeraContextValue | null>(null);

export function ChimeraProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [circuitStates, setCircuitStates] = useState<Record<string, any>>({});

  // Sync circuit breaker states
  useEffect(() => {
    const interval = setInterval(() => {
      setCircuitStates(CircuitBreakerRegistry.getAllStates());
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // ... implement context methods

  return (
    <ChimeraContext.Provider value={contextValue}>
      {children}
    </ChimeraContext.Provider>
  );
}

export function useChimera() {
  const context = useContext(ChimeraContext);
  if (!context) {
    throw new Error("useChimera must be used within ChimeraProvider");
  }
  return context;
}
```

### 7.2 SWR/TanStack Query Integration

```typescript
// lib/hooks/use-providers.ts

import useSWR from "swr";
import { enhancedApi } from "@/lib/api-enhanced";

export function useProviders() {
  const { data, error, isLoading, mutate } = useSWR(
    "/api/v1/providers",
    async () => {
      const response = await enhancedApi.providers.list();
      return response.data;
    },
    {
      refreshInterval: 30000, // Refresh every 30s
      revalidateOnFocus: true,
      dedupingInterval: 5000,
    }
  );

  return {
    providers: data?.providers || [],
    defaultProvider: data?.default || null,
    count: data?.count || 0,
    error,
    isLoading,
    refresh: mutate,
  };
}

export function useTechniques() {
  const { data, error, isLoading, mutate } = useSWR(
    "/api/v1/techniques",
    async () => {
      const response = await enhancedApi.techniques.list();
      return response.data;
    },
    {
      revalidateOnFocus: false,
      dedupingInterval: 60000, // Cache for 1 minute
    }
  );

  return {
    techniques: data?.techniques || [],
    count: data?.count || 0,
    error,
    isLoading,
    refresh: mutate,
  };
}

export function useSession() {
  const { data, error, isLoading, mutate } = useSWR(
    sessionService.session ? `/api/v1/session/${sessionService.session.sessionId}` : null,
    async () => {
      return sessionService.refresh();
    },
    {
      refreshInterval: 60000, // Refresh every minute
    }
  );

  return {
    session: data,
    error,
    isLoading,
    refresh: mutate,
  };
}
```

---

## Part 8: Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Implement error hierarchy (`lib/errors/`)
- [ ] Add error mapping from backend responses
- [ ] Update API interceptors to use error mapping
- [ ] Add global error handler

### Phase 2: Resilience (Week 2)
- [ ] Implement circuit breaker pattern
- [ ] Add retry logic with exponential backoff
- [ ] Integrate with API client
- [ ] Add circuit breaker monitoring UI

### Phase 3: Services (Week 3)
- [ ] Implement ProviderService
- [ ] Implement SessionService
- [ ] Create ChimeraProvider context
- [ ] Add SWR/TanStack Query hooks

### Phase 4: Data Layer (Week 4)
- [ ] Add Zod schemas for runtime validation
- [ ] Create data transformation layer
- [ ] Normalize API responses
- [ ] Add request validation before API calls

### Phase 5: Enhancement (Week 5)
- [ ] Enhance WebSocket with heartbeat handling
- [ ] Add connection quality monitoring
- [ ] Implement missing API endpoints
- [ ] Add caching layer

### Phase 6: Testing & Polish (Week 6)
- [ ] Add unit tests for new services
- [ ] Add integration tests
- [ ] Performance optimization
- [ ] Documentation

---

## Appendix A: File Structure for New Code

```
frontend/src/lib/
├── errors/
│   ├── index.ts
│   ├── api-errors.ts           # Error class hierarchy
│   ├── error-mapper.ts         # Backend error mapping
│   └── global-error-handler.ts # Global handler
├── resilience/
│   ├── index.ts
│   ├── circuit-breaker.ts      # Circuit breaker implementation
│   └── retry.ts                # Retry with backoff
├── services/
│   ├── index.ts
│   ├── provider-service.ts     # Provider management
│   └── session-service.ts      # Session management
├── transforms/
│   ├── index.ts
│   ├── schemas.ts              # Zod validation schemas
│   └── api-transforms.ts       # Data transformations
├── hooks/
│   ├── use-providers.ts        # SWR hooks
│   ├── use-session.ts
│   └── use-techniques.ts
└── use-websocket-enhanced.ts   # Enhanced WebSocket

frontend/src/providers/
└── chimera-provider.tsx        # Global context
```

---

## Appendix B: Dependencies to Add

```json
{
  "dependencies": {
    "zod": "^3.22.0",
    "swr": "^2.2.0"
  }
}
```

Or with TanStack Query:
```json
{
  "dependencies": {
    "zod": "^3.22.0",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

---

## Conclusion

This audit identifies significant opportunities to improve the frontend by adopting proven patterns from the backend. The key priorities are:

1. **Error Handling Alignment** - Critical for user experience and debugging
2. **Resilience Patterns** - Circuit breakers prevent cascade failures
3. **Service Layer** - Better separation of concerns and testability
4. **Type Safety** - Runtime validation matches backend behavior

Implementing these changes will create a truly cohesive full-stack system with consistent behavior, better error handling, and improved reliability.
