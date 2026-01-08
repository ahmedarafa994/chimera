# Chimera Integration Technical Report

---

| **Document Information** | |
|--------------------------|---|
| **Title** | Chimera Integration Technical Report |
| **Version** | 1.0.0 |
| **Date** | 2026-01-06 |
| **Classification** | Internal - Technical |
| **Authors** | Technical Analysis Team |
| **Reviewers** | _TBD_ |
| **Status** | Draft |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Frontend-to-Backend Mapping Table](#2-frontend-to-backend-mapping-table)
3. [Type Alignment Analysis](#3-type-alignment-analysis)
4. [Authentication Flow Analysis](#4-authentication-flow-analysis)
5. [WebSocket & SSE Compatibility](#5-websocket--sse-compatibility)
6. [Error Handling Compatibility](#6-error-handling-compatibility)
7. [Configuration Dependencies](#7-configuration-dependencies)
8. [Appendices](#8-appendices)

---

## 1. Introduction

### 1.1 Purpose

This Technical Report provides a comprehensive mapping between every frontend API expectation and its corresponding backend implementation status in the Chimera platform. It serves as the definitive reference for developers working on integration tasks, debugging API issues, or implementing new features that span both frontend and backend systems.

### 1.2 Scope

This document covers:
- **95+ REST endpoints** across 17 backend categories
- **5 WebSocket endpoints** for real-time communication
- **6 SSE streaming endpoints** for server-sent events
- **50+ Pydantic models** on the backend
- **~75 expected API endpoints** from frontend perspective
- **20+ TypeScript type files** defining frontend contracts
- **4 Zustand stores** for state management
- **TanStack Query** caching layer integration

### 1.3 Methodology

The analysis was conducted through:
1. Static code analysis of backend FastAPI routers and Pydantic models
2. Static code analysis of frontend API clients, hooks, and TypeScript types
3. Cross-referencing OpenAPI specifications with frontend expectations
4. Runtime behavior verification of WebSocket/SSE connections

### 1.4 How to Read Status Indicators

| Indicator | Meaning |
|-----------|---------|
| ✅ | **Exists** - Backend endpoint fully implemented and compatible |
| ❌ | **Missing** - Backend endpoint does not exist but is expected by frontend |
| ⚠️ | **Mismatch** - Endpoint exists but has type, path, or behavior differences |
| 🔄 | **Partial** - Endpoint partially implemented or requires additional work |

---

## 2. Frontend-to-Backend Mapping Table

This section provides the comprehensive mapping between frontend expectations and backend implementations, organized by functional category.

### 2.1 Authentication & Authorization

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`AuthManager.login()`](../frontend/src/lib/api/auth-manager.ts:98) | `POST /api/v1/auth/login` | `{ email, password }` → `AuthTokens` | ❌ Missing | N/A | **GAP-001**: Auth router does not exist |
| [`AuthManager.refreshAccessToken()`](../frontend/src/lib/api/auth-manager.ts:142) | `POST /api/v1/auth/refresh` | `{ refresh_token }` → `AuthTokens` | ❌ Missing | N/A | **GAP-001**: Auth router does not exist |
| `AuthManager.logout()` | `POST /api/v1/auth/logout` | N/A | ❌ Missing | N/A | **GAP-001**: Auth router does not exist |
| API Key Header | `X-API-Key: <key>` | `string` | ✅ Exists | [`verify_admin_api_key()`](../backend-api/app/core/auth.py:503) | Timing-safe comparison |
| Bearer Token Header | `Authorization: Bearer <token>` | `string` | ✅ Exists | [`get_current_user()`](../backend-api/app/core/auth.py:194) | JWT validation |

### 2.2 Provider Management

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `GET /api/v1/providers/` | `Provider[]` | ✅ Exists | [`LLMProvider`](../backend-api/app/domain/models.py:20) | **GAP-002**: Type mismatch (4 vs 12 providers) |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `GET /api/v1/providers/current` | `ActiveProviderInfo` | ✅ Exists | `CurrentProvider` | Compatible |
| [`enhancedApi.models.getForProvider()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/providers/{provider}/models` | `Model[]` | ✅ Exists | `ModelInfo[]` | Compatible |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `POST /api/v1/providers/select` | `{ provider_id, model_id }` | ✅ Exists | `SelectionRequest` | Compatible |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `GET /api/v1/providers/health` | `HealthStatus[]` | ✅ Exists | `ProviderHealth` | Compatible |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `GET /api/v1/providers/rate-limit` | `RateLimitStatus` | ✅ Exists | `RateLimitInfo` | Compatible |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `POST /api/v1/provider-config/providers` | `CreateProviderRequest` | ⚠️ Mismatch | N/A | Different path prefix |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `PUT /api/v1/provider-config/providers/{id}` | `UpdateProviderRequest` | ⚠️ Mismatch | N/A | Different path prefix |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `DELETE /api/v1/provider-config/providers/{id}` | N/A | ⚠️ Mismatch | N/A | Different path prefix |
| [`useProviderConfig`](../frontend/src/hooks/useProviderConfig.ts) | `POST /api/v1/provider-config/providers/{id}/test` | N/A | ⚠️ Mismatch | N/A | Different path prefix |
| [`ProviderSyncContext`](../frontend/src/contexts/ProviderSyncContext.tsx) | `WebSocket /api/v1/provider-config/ws/updates` | `WebSocketMessage` | ⚠️ Mismatch | N/A | Uses `/api/v1/providers/ws/selection` |

### 2.3 Model Sync & Selection

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `POST /api/v1/provider-sync/sync` | `SyncRequest` → `SyncResponse` | ❌ Missing | N/A | Provider sync endpoints not implemented |
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `GET /api/v1/provider-sync/providers/{id}/availability` | `ProviderAvailabilityInfo` | ❌ Missing | N/A | Provider sync endpoints not implemented |
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `GET /api/v1/provider-sync/models/{id}/availability` | `ModelAvailabilityInfo` | ❌ Missing | N/A | Provider sync endpoints not implemented |
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `POST /api/v1/provider-sync/select/provider` | `SelectProviderRequest` | ❌ Missing | N/A | Use `/api/v1/providers/select` instead |
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `POST /api/v1/provider-sync/select/model` | `SelectModelRequest` | ❌ Missing | N/A | Use `/api/v1/session/model` instead |
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `GET /api/v1/provider-sync/active` | `ActiveSelection` | ❌ Missing | N/A | Use `/api/v1/providers/current` instead |
| [`ProviderSyncService`](../frontend/src/lib/sync/provider-sync-service.ts) | `WebSocket /api/v1/provider-sync/ws` | `WebSocketMessage` | ❌ Missing | N/A | Use `/api/v1/providers/ws/selection` |

### 2.4 Session Management

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`enhancedApi.session.getModels()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/session/models` | `Model[]` | ✅ Exists | `ModelInfo[]` | Compatible |
| [`enhancedApi.session.validateModel()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/session/models/validate` | `ValidationResult` | ✅ Exists | `ValidationResponse` | Compatible |
| [`enhancedApi.session.create()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/session` | `SessionConfig` | ✅ Exists | `SessionCreate` | Compatible |
| [`enhancedApi.session.getCurrent()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/session` | `Session` | ✅ Exists | `SessionInfo` | Compatible |
| [`enhancedApi.session.delete()`](../frontend/src/lib/api-enhanced.ts) | `DELETE /api/v1/session` | N/A | ✅ Exists | N/A | Compatible |
| [`enhancedApi.session.getCurrentModel()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/session/current-model` | `ModelInfo` | ✅ Exists | `ModelInfo` | Compatible |
| [`enhancedApi.session.updateModel()`](../frontend/src/lib/api-enhanced.ts) | `PUT /api/v1/session/model` | `{ model_id }` | ✅ Exists | `ModelUpdate` | Compatible |
| N/A | `GET /api/v1/session/{session_id}` | N/A | ✅ Exists | `SessionInfo` | Backend-only endpoint |
| N/A | `GET /api/v1/session/stats` | N/A | ✅ Exists | `SessionStats` | Admin endpoint, no frontend coverage |

### 2.5 Generation & Streaming

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`enhancedApi.generate.text()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/generation/generate` | `PromptRequest` → `PromptResponse` | ✅ Exists | [`PromptRequest`](../backend-api/app/domain/models.py:62) → [`PromptResponse`](../backend-api/app/domain/models.py:186) | Compatible |
| [`enhancedApi.generate.health()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/generation/health` | `HealthStatus` | ✅ Exists | `HealthCheck` | Compatible |
| SSE Streaming | `POST /api/v1/streaming/generate/stream` | `PromptRequest` → SSE events | ✅ Exists | [`StreamChunk`](../backend-api/app/domain/models.py:462) | Compatible |
| N/A | `POST /api/v1/streaming/generate/stream/raw` | N/A | ✅ Exists | Raw text stream | Backend-only |
| N/A | `GET /api/v1/streaming/generate/stream/capabilities` | `StreamingCapabilities` | ✅ Exists | `StreamingCapabilities` | Not implemented in frontend |

### 2.6 Jailbreak Operations (DeepTeam)

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`JailbreakAPI.generate()`](../frontend/src/api/jailbreak.ts) | `POST /api/v1/deepteam/jailbreak/generate` | [`JailbreakGenerationRequest`](../frontend/src/types/jailbreak.ts) | ✅ Exists | [`JailbreakGenerationRequest`](../backend-api/app/domain/models.py:353) | Compatible |
| [`JailbreakAPI.generateBatch()`](../frontend/src/api/jailbreak.ts) | `POST /api/v1/deepteam/jailbreak/batch` | `BatchJailbreakRequest` | ✅ Exists | `BatchRequest` | Compatible |
| [`JailbreakAPI.getStrategies()`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/strategies` | `Strategy[]` | ✅ Exists | `StrategyList` | **GAP-010**: Enum values mismatch |
| [`JailbreakAPI.getStrategyDetails()`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/strategies/{type}` | `StrategyDetails` | ✅ Exists | `StrategyInfo` | Compatible |
| [`JailbreakAPI.getVulnerabilities()`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/vulnerabilities` | `Vulnerability[]` | ⚠️ Mismatch | N/A | Path exists at `/api/v1/jailbreak/vulnerabilities` |
| [`JailbreakAPI.clearCache()`](../frontend/src/api/jailbreak.ts) | `DELETE /api/v1/deepteam/jailbreak/cache` | N/A | ✅ Exists | N/A | Compatible |
| [`JailbreakAPI.getHealth()`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/health` | `HealthStatus` | ✅ Exists | `HealthCheck` | Compatible |
| [`JailbreakAPI.getPrompt()`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/sessions/{id}/prompts/{pid}` | `Prompt` | ✅ Exists | `PromptInfo` | Compatible |
| [`JailbreakAPI.getSessionPrompts()`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/sessions/{id}/prompts` | `Prompt[]` | ✅ Exists | `PromptList` | Compatible |
| [`JailbreakAPI.deleteSession()`](../frontend/src/api/jailbreak.ts) | `DELETE /api/v1/deepteam/jailbreak/sessions/{id}` | N/A | ✅ Exists | N/A | Compatible |
| [`JailbreakWebSocket`](../frontend/src/api/jailbreak.ts:228) | `WebSocket /api/v1/deepteam/jailbreak/ws/generate` | WebSocket messages | ✅ Exists | WS Protocol | **GAP-003**: Hardcoded URL |
| [`JailbreakSSE`](../frontend/src/api/jailbreak.ts) | `GET /api/v1/deepteam/jailbreak/generate/stream` | SSE events | ✅ Exists | SSE Protocol | Compatible |

### 2.7 AutoDAN Operations

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`enhancedApi.autodan.jailbreak()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan/jailbreak` | `AutoDANRequest` | ✅ Exists | `AutoDANRequest` | Compatible |
| [`enhancedApi.autodan.batch()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan/batch` | `BatchAutoDANRequest` | ✅ Exists | `BatchRequest` | Compatible |
| [`enhancedApi.autodan.getConfig()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/autodan/config` | `AutoDANConfig` | ✅ Exists | `AutoDANConfig` | Compatible |
| N/A | `POST /api/v1/autodan/lifelong` | N/A | ✅ Exists | `LifelongRequest` | Not implemented in frontend |

### 2.8 AutoDAN-Turbo Operations

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`enhancedApi.autodanTurbo.attack()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/attack` | `AttackRequest` | ✅ Exists | `AttackRequest` | Rate limited |
| [`enhancedApi.autodanTurbo.warmup()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/warmup` | `WarmupRequest` | ✅ Exists | `WarmupRequest` | Rate limited |
| [`enhancedApi.autodanTurbo.lifelong()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/lifelong` | `LifelongRequest` | ✅ Exists | `LifelongRequest` | Compatible |
| [`enhancedApi.autodanTurbo.test()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/test` | `TestRequest` | ✅ Exists | `TestRequest` | Compatible |
| [`enhancedApi.autodanTurbo.getStrategies()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/autodan-turbo/strategies` | `Strategy[]` | ✅ Exists | `StrategyList` | Compatible |
| [`enhancedApi.autodanTurbo.getStrategy()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/autodan-turbo/strategies/{id}` | `StrategyDetails` | ✅ Exists | `StrategyInfo` | Compatible |
| [`enhancedApi.autodanTurbo.createStrategy()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/strategies` | `CreateStrategyRequest` | ✅ Exists | `StrategyCreate` | Compatible |
| [`enhancedApi.autodanTurbo.deleteStrategy()`](../frontend/src/lib/api-enhanced.ts) | `DELETE /api/v1/autodan-turbo/strategies/{id}` | N/A | ✅ Exists | N/A | Compatible |
| [`enhancedApi.autodanTurbo.searchStrategies()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/strategies/search` | `SearchRequest` | ✅ Exists | `SearchRequest` | Embedding-based |
| [`enhancedApi.autodanTurbo.batchInject()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/strategies/batch-inject` | `BatchInjectRequest` | ✅ Exists | `BatchInjectRequest` | Compatible |
| [`enhancedApi.autodanTurbo.getProgress()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/autodan-turbo/progress` | `LearningProgress` | ✅ Exists | `ProgressInfo` | Compatible |
| [`enhancedApi.autodanTurbo.getLibraryStats()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/autodan-turbo/library/stats` | `LibraryStats` | ✅ Exists | `LibraryStats` | Compatible |
| [`enhancedApi.autodanTurbo.reset()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/reset` | N/A | ✅ Exists | N/A | Compatible |
| [`enhancedApi.autodanTurbo.saveLibrary()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/library/save` | N/A | ✅ Exists | N/A | Compatible |
| [`enhancedApi.autodanTurbo.clearLibrary()`](../frontend/src/lib/api-enhanced.ts) | `POST /api/v1/autodan-turbo/library/clear` | N/A | ✅ Exists | N/A | Destructive operation |
| [`enhancedApi.autodanTurbo.health()`](../frontend/src/lib/api-enhanced.ts) | `GET /api/v1/autodan-turbo/health` | `HealthStatus` | ✅ Exists | `HealthCheck` | Compatible |
| N/A | `POST /api/v1/autodan-turbo/transfer/export` | N/A | ✅ Exists | `ExportResponse` | Not implemented in frontend |
| N/A | `POST /api/v1/autodan-turbo/transfer/import` | N/A | ✅ Exists | `ImportRequest` | Not implemented in frontend |

### 2.9 DeepTeam Red Team Operations

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`DeepTeamApiClient`](../frontend/src/lib/api/deepteam-client.ts) | `POST /api/v1/deepteam/red-team` | `RedTeamRequest` | ✅ Exists | `RedTeamConfig` | Compatible |
| [`DeepTeamApiClient`](../frontend/src/lib/api/deepteam-client.ts) | `POST /api/v1/deepteam/quick-scan` | `QuickScanRequest` | ✅ Exists | `QuickScanConfig` | Compatible |
| [`DeepTeamApiClient`](../frontend/src/lib/api/deepteam-client.ts) | `POST /api/v1/deepteam/security-audit` | `SecurityAuditRequest` | ✅ Exists | `SecurityAuditConfig` | Compatible |
| [`DeepTeamApiClient`](../frontend/src/lib/api/deepteam-client.ts) | `POST /api/v1/deepteam/bias-audit` | `BiasAuditRequest` | ✅ Exists | `BiasAuditConfig` | Compatible |
| N/A | `POST /api/v1/deepteam/owasp-assessment` | N/A | ✅ Exists | `OWASPConfig` | Not implemented in frontend |
| N/A | `POST /api/v1/deepteam/assess-vulnerability` | N/A | ✅ Exists | `VulnerabilityTest` | Not implemented in frontend |
| [`DeepTeamApiClient.listSessions()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/sessions` | `Session[]` | ✅ Exists | `SessionList` | Compatible |
| [`DeepTeamApiClient.getSession()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/sessions/{id}` | `SessionStatus` | ✅ Exists | `SessionInfo` | Compatible |
| [`DeepTeamApiClient.getSessionResult()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/sessions/{id}/result` | `SessionResult` | ✅ Exists | `SessionResult` | Compatible |
| [`DeepTeamApiClient`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/vulnerabilities` | `Vulnerability[]` | ✅ Exists | `VulnerabilityList` | Compatible |
| [`DeepTeamApiClient`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/attacks` | `Attack[]` | ✅ Exists | `AttackList` | Compatible |
| N/A | `GET /api/v1/deepteam/presets` | N/A | ✅ Exists | `PresetList` | Not implemented in frontend |
| N/A | `GET /api/v1/deepteam/health` | N/A | ✅ Exists | `HealthCheck` | Not implemented in frontend |
| [`DeepTeamApiClient.listAgents()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/agents` | `Agent[]` | ❌ Missing | N/A | Endpoint does not exist |
| [`DeepTeamApiClient.getAgent()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/agents/{id}` | `AgentDetails` | ❌ Missing | N/A | Endpoint does not exist |
| [`DeepTeamApiClient.listEvaluations()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/evaluations` | `Evaluation[]` | ❌ Missing | N/A | Endpoint does not exist |
| [`DeepTeamApiClient.getEvaluation()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/evaluations/{id}` | `EvaluationDetails` | ❌ Missing | N/A | Endpoint does not exist |
| [`DeepTeamApiClient.listRefinements()`](../frontend/src/lib/api/deepteam-client.ts) | `GET /api/v1/deepteam/refinements` | `Refinement[]` | ❌ Missing | N/A | Endpoint does not exist |
| [`DeepTeamApiClient.applyRefinement()`](../frontend/src/lib/api/deepteam-client.ts) | `POST /api/v1/deepteam/refinements/apply` | `ApplyRefinementRequest` | ❌ Missing | N/A | Endpoint does not exist |
| [`DeepTeamApiClient.createWebSocketConnection()`](../frontend/src/lib/api/deepteam-client.ts) | `WebSocket /ws/sessions/{sessionId}` | WebSocket messages | ⚠️ Mismatch | WS Protocol | Path may differ |

### 2.10 Admin & Metrics (0% Frontend Coverage)

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| None | `GET /api/v1/admin/feature-flags` | N/A | ✅ Exists | `FeatureFlagList` | **GAP-005**: No frontend coverage |
| None | `GET /api/v1/admin/feature-flags/stats` | N/A | ✅ Exists | `FeatureFlagStats` | Admin only |
| None | `POST /api/v1/admin/feature-flags/toggle` | N/A | ✅ Exists | `ToggleRequest` | Admin only |
| None | `POST /api/v1/admin/feature-flags/reload` | N/A | ✅ Exists | N/A | Admin only |
| None | `GET /api/v1/admin/feature-flags/{technique_name}` | N/A | ✅ Exists | `TechniqueInfo` | Admin only |
| None | `GET /api/v1/admin/tenants` | N/A | ✅ Exists | `TenantList` | Admin only |
| None | `POST /api/v1/admin/tenants` | N/A | ✅ Exists | `TenantCreate` | Admin only |
| None | `GET /api/v1/admin/tenants/{tenant_id}` | N/A | ✅ Exists | `TenantInfo` | Admin only |
| None | `DELETE /api/v1/admin/tenants/{tenant_id}` | N/A | ✅ Exists | N/A | Admin only |
| None | `GET /api/v1/admin/tenants/stats/summary` | N/A | ✅ Exists | `TenantStats` | Admin only |
| None | `GET /api/v1/admin/usage/global` | N/A | ✅ Exists | `GlobalUsage` | Admin only |
| None | `GET /api/v1/admin/usage/tenant/{tenant_id}` | N/A | ✅ Exists | `TenantUsage` | Admin only |
| None | `GET /api/v1/admin/usage/techniques/top` | N/A | ✅ Exists | `TopTechniques` | Admin only |
| None | `GET /api/v1/admin/usage/quota/{tenant_id}` | N/A | ✅ Exists | `QuotaInfo` | Admin only |
| None | `GET /api/v1/metrics/prometheus` | N/A | ✅ Exists | Prometheus format | **GAP-006**: No frontend coverage |
| None | `GET /api/v1/metrics/json` | N/A | ✅ Exists | JSON format | Metrics dashboard not implemented |
| None | `GET /api/v1/metrics/circuit-breakers` | N/A | ✅ Exists | `CircuitBreakerStatus` | Metrics dashboard not implemented |
| None | `POST /api/v1/metrics/circuit-breakers/{name}/reset` | N/A | ✅ Exists | N/A | Metrics dashboard not implemented |
| None | `POST /api/v1/metrics/circuit-breakers/reset-all` | N/A | ✅ Exists | N/A | Metrics dashboard not implemented |
| None | `GET /api/v1/metrics/cache` | N/A | ✅ Exists | `CacheMetrics` | Metrics dashboard not implemented |
| None | `POST /api/v1/metrics/cache/clear` | N/A | ✅ Exists | N/A | Metrics dashboard not implemented |
| None | `GET /api/v1/metrics/connection-pools` | N/A | ✅ Exists | `PoolStats` | Metrics dashboard not implemented |
| None | `POST /api/v1/metrics/connection-pools/reset` | N/A | ✅ Exists | N/A | Metrics dashboard not implemented |
| None | `GET /api/v1/metrics/multi-level-cache` | N/A | ✅ Exists | `CacheStats` | Metrics dashboard not implemented |
| None | `POST /api/v1/metrics/multi-level-cache/clear` | N/A | ✅ Exists | N/A | Metrics dashboard not implemented |

### 2.11 Health & Monitoring

| Frontend Expectation | Expected Endpoint | Frontend Type | Backend Status | Backend Model | Notes |
|---------------------|-------------------|---------------|----------------|---------------|-------|
| [`HealthDashboard`](../frontend/src/components/HealthDashboard.tsx) | `GET /health` | `OverallHealth` | ✅ Exists | [`HealthCheckResponse`](../backend-api/app/schemas/base_schemas.py) | Compatible |
| [`HealthDashboard`](../frontend/src/components/HealthDashboard.tsx) | `GET /health/live` | `{ status: string }` | ✅ Exists | `LivenessResponse` | Kubernetes liveness probe |
| N/A | `GET /health/ready` | N/A | ✅ Exists | `ReadinessResponse` | Kubernetes readiness probe |
| N/A | `GET /health/circuit-breakers` | `CircuitBreakerStatus[]` | ✅ Exists | `CircuitBreakerList` | Not implemented in frontend |
| N/A | `POST /health/circuit-breakers/{name}/reset` | N/A | ✅ Exists | N/A | Not implemented in frontend |
| N/A | `GET /health/proxy` | N/A | ✅ Exists | `ProxyHealth` | AIClient-2-API proxy health |
| N/A | `GET /health/integration` | N/A | ✅ Exists | `IntegrationHealth` | Provider integration health |
| N/A | `GET /health/integration/graph` | N/A | ✅ Exists | `DependencyGraph` | Service dependency visualization |
| N/A | `GET /health/integration/history` | N/A | ✅ Exists | `HealthHistory` | Historical health data |
| N/A | `GET /health/integration/alerts` | N/A | ✅ Exists | `AlertList` | Active health alerts |
| N/A | `POST /health/integration/check` | N/A | ✅ Exists | `CheckResult` | Trigger immediate health check |

---

## 3. Type Alignment Analysis

This section provides field-by-field comparison between frontend TypeScript interfaces and backend Pydantic models.

### 3.1 AuthTokens / TokenResponse

**Frontend Type** ([`frontend/src/lib/api/types.ts:283-289`](../frontend/src/lib/api/types.ts:283)):
```typescript
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';           // Capitalized
  expires_in: number;
  refresh_expires_in: number;     // ❌ DOES NOT EXIST IN BACKEND
}
```

**Backend Model** ([`backend-api/app/core/auth.py:126-132`](../backend-api/app/core/auth.py:126)):
```python
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"    # Lowercase
    expires_in: int               # Seconds
```

| Field | Frontend | Backend | Status | Notes |
|-------|----------|---------|--------|-------|
| `access_token` | `string` | `str` | ✅ | Compatible |
| `refresh_token` | `string` | `str` | ✅ | Compatible |
| `token_type` | `'Bearer'` (literal) | `"bearer"` (default) | ⚠️ **GAP-009** | Casing mismatch |
| `expires_in` | `number` | `int` | ✅ | Compatible (both seconds) |
| `refresh_expires_in` | `number` | N/A | ❌ **GAP-004** | Field does not exist in backend |

---

### 3.2 Provider / LLMProviderType

**Frontend Type** ([`frontend/src/lib/api/types.ts:43`](../frontend/src/lib/api/types.ts:43)):
```typescript
interface Provider {
  id: string;
  name: string;
  type: 'openai' | 'anthropic' | 'gemini' | 'deepseek';  // Only 4 types
  enabled: boolean;
  models: Model[];
  health?: HealthStatus;
}
```

**Backend Model** ([`backend-api/app/domain/models.py:20-35`](../backend-api/app/domain/models.py:20)):
```python
class LLMProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"            # ❌ Missing in frontend
    GEMINI = "gemini"
    QWEN = "qwen"                # ❌ Missing in frontend
    GEMINI_CLI = "gemini-cli"    # ❌ Missing in frontend
    ANTIGRAVITY = "antigravity"  # ❌ Missing in frontend
    KIRO = "kiro"                # ❌ Missing in frontend
    CURSOR = "cursor"            # ❌ Missing in frontend
    XAI = "xai"                  # ❌ Missing in frontend
    DEEPSEEK = "deepseek"
    MOCK = "mock"                # ❌ Missing in frontend
```

| Provider Type | Frontend | Backend | Status |
|---------------|----------|---------|--------|
| `openai` | ✅ | ✅ | Compatible |
| `anthropic` | ✅ | ✅ | Compatible |
| `gemini` | ✅ | ✅ | Compatible |
| `deepseek` | ✅ | ✅ | Compatible |
| `google` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `qwen` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `gemini-cli` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `antigravity` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `kiro` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `cursor` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `xai` | ❌ | ✅ | **GAP-002**: Missing in frontend |
| `mock` | ❌ | ✅ | **GAP-002**: Missing in frontend |

---

### 3.3 Model / ModelInfo

**Frontend Type** ([`frontend/src/lib/api/types.ts:250-258`](../frontend/src/lib/api/types.ts:250)):
```typescript
interface Model {
  id: string;
  name: string;
  provider_id: string;
  capabilities: string[];
  max_tokens: number;
  deprecated?: boolean;
}
```

**Backend Model** (inferred from [`backend-api/app/domain/models.py`](../backend-api/app/domain/models.py)):
```python
class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str              # Named 'provider' not 'provider_id'
    capabilities: list[str]
    max_tokens: int
    deprecated: bool = False
```

| Field | Frontend | Backend | Status | Notes |
|-------|----------|---------|--------|-------|
| `id` | `string` | `str` | ✅ | Compatible |
| `name` | `string` | `str` | ✅ | Compatible |
| `provider_id` | `string` | `provider` | ⚠️ | Different field name |
| `capabilities` | `string[]` | `list[str]` | ✅ | Compatible |
| `max_tokens` | `number` | `int` | ✅ | Compatible |
| `deprecated` | `boolean?` | `bool` | ✅ | Compatible (default false) |

---

### 3.4 PromptRequest / PromptResponse

**Frontend Type** ([`frontend/src/lib/api/types.ts:269-277`](../frontend/src/lib/api/types.ts:269)):
```typescript
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
```

**Backend Model** ([`backend-api/app/domain/models.py:62`](../backend-api/app/domain/models.py:62)):
```python
class PromptRequest(BaseModel):
    prompt: str                          # 1-50000 chars
    system_instruction: str | None
    config: GenerationConfig | None
    model: str | None
    provider: LLMProviderType | None
    api_key: str | None                  # ❌ Missing in frontend
    skip_validation: bool = False        # ❌ Missing in frontend

class PromptResponse(BaseModel):
    text: str                            # Max 50000 chars
    model_used: str
    provider: str
    usage_metadata: dict[str, Any] | None
    finish_reason: str | None
    latency_ms: float
```

| Field | Frontend | Backend | Status | Notes |
|-------|----------|---------|--------|-------|
| `prompt` | `string` | `str` (1-50000) | ✅ | Compatible |
| `system_instruction` | `string?` | `str \| None` | ✅ | Compatible |
| `config` | `GenerationConfig?` | `GenerationConfig \| None` | ✅ | Compatible |
| `model` | `string?` | `str \| None` | ✅ | Compatible |
| `provider` | `LLMProviderType?` | `LLMProviderType \| None` | ⚠️ | Provider enum mismatch |
| `api_key` | N/A | `str \| None` | 🔄 | Backend only field |
| `skip_validation` | N/A | `bool` | 🔄 | Backend only field |

---

### 3.5 JailbreakGenerationRequest

**Frontend Type** ([`frontend/src/types/jailbreak.ts:348-371`](../frontend/src/types/jailbreak.ts:348)):
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
```

**Backend Model** ([`backend-api/app/domain/models.py:353`](../backend-api/app/domain/models.py:353)):
```python
class JailbreakGenerationRequest(BaseModel):
    core_request: str                              # 1-5000 chars
    technique_suite: str
    potency_level: int                             # 1-10
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 2048                     # 256-8192
    density: float = 0.5
    use_leet_speak: bool = False
    use_homoglyphs: bool = False
    use_caesar_cipher: bool = False
    use_role_hijacking: bool = False
    use_instruction_injection: bool = False
    use_adversarial_suffixes: bool = False
    use_neural_bypass: bool = False
    use_meta_prompting: bool = False
    use_multilingual_trojan: bool = False
    use_payload_splitting: bool = False
    use_contextual_interaction_attack: bool = False
```

| Field | Frontend | Backend | Status |
|-------|----------|---------|--------|
| `core_request` | `string` | `str` (1-5000) | ✅ |
| `technique_suite` | `string` | `str` | ⚠️ **GAP-010**: Values may not align |
| `potency_level` | `number` | `int` (1-10) | ✅ |
| `temperature` | `number?` | `float = 0.7` | ✅ |
| `top_p` | `number?` | `float = 0.95` | ✅ |
| `max_new_tokens` | `number?` | `int = 2048` | ✅ |
| `density` | `number?` | `float = 0.5` | ✅ |
| All boolean flags | `boolean?` | `bool = False` | ✅ |

---

### 3.6 Error Types Mapping

**Frontend Error Classes** ([`frontend/src/lib/errors/api-errors.ts`](../frontend/src/lib/errors/api-errors.ts)):

| Frontend Error | HTTP Status | Backend Exception | Status |
|---------------|-------------|-------------------|--------|
| `ValidationError` | 400 | `ValidationError` | ✅ |
| `AuthenticationError` | 401 | `HTTPException(401)` | ✅ |
| `AuthorizationError` | 403 | `HTTPException(403)` | ✅ |
| `NotFoundError` | 404 | `HTTPException(404)` | ✅ |
| `ConflictError` | 409 | `ConflictError` | ✅ |
| `RateLimitError` | 429 | `RateLimitExceeded` | ✅ |
| `InternalError` | 500 | `AppError` | ✅ |
| `ServiceUnavailableError` | 503 | `ProviderNotAvailableError` | ✅ |
| `GatewayTimeoutError` | 504 | N/A | 🔄 |
| `LLMProviderError` | 502 | `LLMProviderError` | ✅ |
| `LLMConnectionError` | 500 | N/A | ❌ **GAP-008** |
| `LLMTimeoutError` | 408 | N/A | ❌ **GAP-008** |
| `LLMQuotaExceededError` | 429 | N/A | ❌ **GAP-008** |
| `LLMInvalidResponseError` | 500 | N/A | ❌ **GAP-008** |
| `LLMContentBlockedError` | 500 | N/A | ❌ **GAP-008** |
| `TransformationError` | 500 | `TransformationError` | ✅ |
| `CircuitBreakerOpenError` | 503 | N/A | 🔄 |
| `NetworkError` | 0 | N/A | Frontend only |
| N/A | N/A | `MissingFieldError` | ❌ **GAP-008** |
| N/A | N/A | `InvalidFieldError` | ❌ **GAP-008** |
| N/A | N/A | `PayloadTooLargeError` | ❌ **GAP-008** |
| N/A | N/A | `ProviderNotConfiguredError` | ❌ **GAP-008** |
| N/A | N/A | `CacheError` | ❌ **GAP-008** |
| N/A | N/A | `ConfigurationError` | ❌ **GAP-008** |

---

## 4. Authentication Flow Analysis

### 4.1 Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AUTHENTICATION FLOW COMPARISON                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FRONTEND EXPECTED FLOW                 BACKEND ACTUAL FLOW                 │
│  ─────────────────────────              ──────────────────────              │
│                                                                             │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ 1. User enters      │               │ 1. ❌ NO LOGIN      │             │
│  │    email/password   │               │    ENDPOINT EXISTS  │             │
│  └──────────┬──────────┘               └─────────────────────┘             │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ 2. POST /auth/login │──────────────▶│ 2. 404 NOT FOUND    │             │
│  │    { email, pass }  │               │    (no router)      │             │
│  └──────────┬──────────┘               └─────────────────────┘             │
│             │ ❌ FAILS                                                      │
│             ▼                                                               │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ 3. Store tokens in  │               │ 3. Token validation │             │
│  │    localStorage     │               │    works IF token   │             │
│  └──────────┬──────────┘               │    already exists   │             │
│             │                          └─────────────────────┘             │
│             ▼                                                               │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ 4. Attach Bearer    │──────────────▶│ 4. get_current_user │             │
│  │    token to requests│               │    validates token  │ ✅ WORKS   │
│  └──────────┬──────────┘               └─────────────────────┘             │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ 5. Token expires    │               │ 5. ❌ NO REFRESH    │             │
│  │    → refresh        │──────────────▶│    ENDPOINT EXISTS  │             │
│  └─────────────────────┘               └─────────────────────┘             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ALTERNATIVE: API KEY AUTHENTICATION (WORKS)                                │
│  ─────────────────────────────────────────────                              │
│                                                                             │
│  ┌─────────────────────┐               ┌─────────────────────┐             │
│  │ 1. Set X-API-Key    │──────────────▶│ 1. verify_api_key() │ ✅         │
│  │    header           │               │    timing-safe      │             │
│  └─────────────────────┘               └─────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Frontend Expected Flow

**Source:** [`frontend/src/lib/api/auth-manager.ts`](../frontend/src/lib/api/auth-manager.ts)

```typescript
// Step 1: Login
async login(email: string, password: string): Promise<boolean> {
  const response = await fetch('/api/v1/auth/login', {  // ❌ DOES NOT EXIST
    method: 'POST',
    body: JSON.stringify({ email, password })
  });
  const data = await response.json();
  this.storeTokens(data.tokens);  // { access_token, refresh_token, expires_in }
  return true;
}

// Step 2: Attach token to requests
getAuthHeaders(): Record<string, string> {
  const token = this.getAccessToken();
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

// Step 3: Refresh when expired
async refreshAccessToken(): Promise<string | null> {
  const response = await fetch('/api/v1/auth/refresh', {  // ❌ DOES NOT EXIST
    method: 'POST',
    body: JSON.stringify({ refresh_token: this.getRefreshToken() })
  });
  // ...
}
```

### 4.3 Backend Actual Implementation

**Source:** [`backend-api/app/core/auth.py`](../backend-api/app/core/auth.py)

```python
# Only validation dependency exists - NO HTTP ENDPOINTS

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get the current authenticated user from JWT token."""
    token = credentials.credentials
    
    # Check if it's an API key first
    if not token.startswith("eyJ"):  # Not a JWT
        api_key = settings.CHIMERA_API_KEY
        if api_key and secrets.compare_digest(token, api_key):
            return User(id="api", email="api@system", roles=[Role.API_CLIENT])
    
    # Decode JWT token
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    return User(**payload)
```

### 4.4 Gap Summary

| Authentication Step | Frontend | Backend | Status |
|---------------------|----------|---------|--------|
| Login endpoint | `POST /api/v1/auth/login` | Does not exist | ❌ **GAP-001** |
| Refresh endpoint | `POST /api/v1/auth/refresh` | Does not exist | ❌ **GAP-001** |
| Logout endpoint | `POST /api/v1/auth/logout` | Does not exist | ❌ **GAP-001** |
| Token validation | Bearer header | `get_current_user()` | ✅ Works |
| API Key auth | X-API-Key header | `verify_api_key()` | ✅ Works |
| Token storage | localStorage | N/A (frontend concern) | ✅ N/A |
| Token expiry | 1 hour expected | `JWT_EXPIRATION_HOURS` | ✅ Configurable |

### 4.5 Recommended Fix

Create a new auth router at [`backend-api/app/api/v1/endpoints/auth.py`](../backend-api/app/api/v1/endpoints/auth.py):

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"  # Note: Capitalized to match frontend
    expires_in: int

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    # Implement authentication logic
    ...

@router.post("/refresh", response_model=TokenResponse)
async def refresh(refresh_token: str):
    # Implement refresh logic
    ...

@router.post("/logout")
async def logout():
    # Implement logout logic (token revocation)
    ...
```

---

## 5. WebSocket & SSE Compatibility

### 5.1 WebSocket Endpoints

| Frontend URL | Backend URL | Message Format | Connection Params | Status |
|--------------|-------------|----------------|-------------------|--------|
| [`ws://localhost:8001/api/v1/deepteam/jailbreak/ws/generate`](../frontend/src/api/jailbreak.ts:228) | `/api/v1/deepteam/jailbreak/ws/generate` | JSON messages | Query: `request` (JSON) | ⚠️ **GAP-003**: Hardcoded URL |
| `/api/v1/provider-config/ws/updates` | `/api/v1/providers/ws/selection` | JSON messages | Auth: Bearer token | ⚠️ Path mismatch |
| `/ws/sessions/{sessionId}` | `/ws/sessions/{sessionId}` | JSON messages | Auth: Bearer token | ✅ Compatible |
| N/A | `/ws/enhance` | JSON messages | Auth: Bearer token | ❌ **GAP-012**: Not implemented |
| N/A | `/api/v1/autoadv/ws` | JSON messages | Auth: Bearer token | ❌ **GAP-012**: Not implemented |

### 5.2 WebSocket Message Formats

**Jailbreak WebSocket Messages:**

| Direction | Message Type | Payload |
|-----------|--------------|---------|
| Server → Client | `generation_start` | `{ session_id: string, total_prompts: number }` |
| Server → Client | `generation_progress` | `{ current: number, total: number, percentage: number }` |
| Server → Client | `prompt_generated` | `{ id: string, content: string, technique: string, potency_score: number }` |
| Server → Client | `generation_complete` | `{ session_id: string, prompts: Prompt[], duration_ms: number }` |
| Server → Client | `generation_error` | `{ error: string, code: string }` |
| Client → Server | `cancel` | `{ session_id: string }` |
| Both | `heartbeat` | `{ timestamp: number }` |

**Provider Sync WebSocket Messages:**

| Direction | Message Type | Payload |
|-----------|--------------|---------|
| Server → Client | `full_sync` | `SyncState` |
| Server → Client | `initial_state` | `SyncState` |
| Server → Client | `provider_added` | `ProviderSyncInfo` |
| Server → Client | `provider_updated` | `ProviderSyncInfo` |
| Server → Client | `provider_removed` | `{ provider_id: string }` |
| Server → Client | `provider_status_changed` | `{ provider_id: string, status: string }` |
| Server → Client | `model_deprecated` | `{ model_id: string, replacement?: string }` |
| Server → Client | `active_provider_changed` | `{ provider_id: string }` |
| Server → Client | `active_model_changed` | `{ model_id: string }` |
| Server → Client | `heartbeat` | `{ timestamp: number }` |
| Server → Client | `error` | `{ message: string, code: string }` |

### 5.3 SSE Endpoints

| Frontend URL | Backend URL | Event Types | Status |
|--------------|-------------|-------------|--------|
| `/api/v1/streaming/generate/stream` | `/api/v1/streaming/generate/stream` | `text`, `complete`, `error` | ✅ Compatible |
| `/api/v1/deepteam/jailbreak/generate/stream` | `/api/v1/deepteam/jailbreak/generate/stream` | `generation_start`, `progress`, `prompt`, `complete`, `error` | ✅ Compatible |
| N/A | `/api/v1/transformation/stream` | SSE events | ❌ Not implemented in frontend |
| N/A | `/api/v1/jailbreak/generate/stream` | SSE events | ⚠️ **GAP-014**: Duplicate path |
| N/A | `/api/v1/advanced/jailbreak/generate/stream` | SSE events | ❌ Not implemented |
| N/A | `/api/v1/advanced/code/generate/stream` | SSE events | ❌ Not implemented |

### 5.4 SSE Message Format

**Backend SSE Format:**
```
event: text
data: {"text": "Generated content...", "is_final": false}

event: complete
data: {"text": "Final content", "is_final": true, "finish_reason": "stop"}

event: error
data: {"error": "Error message", "code": "ERROR_CODE"}
```

**Frontend Parsing** ([`frontend/src/lib/sync/sse-manager.ts`](../frontend/src/lib/sync/sse-manager.ts)):
```typescript
eventSource.addEventListener('text', (event) => {
  const data = JSON.parse(event.data);
  // data: { text: string, is_final: boolean }
});

eventSource.addEventListener('complete', (event) => {
  const data = JSON.parse(event.data);
  // data: { text: string, is_final: true, finish_reason: string }
});
```

**Compatibility:** ✅ Format matches

### 5.5 WebSocket Configuration

**Frontend Configuration** ([`frontend/src/lib/sync/websocket-manager.ts`](../frontend/src/lib/sync/websocket-manager.ts)):

```typescript
const DEFAULT_CONFIG: Required<WebSocketConfig> = {
  url: '',
  protocols: [],
  autoReconnect: true,
  maxReconnectAttempts: 10,
  reconnectDelay: 1000,           // 1 second initial
  maxReconnectDelay: 30000,       // 30 seconds max
  heartbeatInterval: 30000,       // 30 seconds
  heartbeatTimeout: 10000,        // 10 seconds
  queueSize: 100,                 // Max queued messages
  debug: false,
};
```

**Backend Configuration:**
- Heartbeat: 30 second interval (configurable)
- Connection timeout: Matches client settings
- Max message size: Configurable per endpoint

---

## 6. Error Handling Compatibility

### 6.1 Error Response Format

**Backend Standard Format** ([`backend-api/app/domain/models.py:539`](../backend-api/app/domain/models.py:539)):
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

**Frontend Error Parsing** ([`frontend/src/lib/errors/error-mapper.ts`](../frontend/src/lib/errors/error-mapper.ts)):
```typescript
export function mapBackendError(error: AxiosError): APIError {
  const response = error.response;
  const data = response?.data as ErrorResponse;
  
  // Map based on status code and error_code
  switch (response?.status) {
    case 400: return new ValidationError(data.message, data.details);
    case 401: return new AuthenticationError(data.message);
    case 403: return new AuthorizationError(data.message);
    case 404: return new NotFoundError(data.message);
    case 429: return new RateLimitError(data.message, data.details?.retry_after);
    case 500: return new InternalError(data.message);
    case 502: return new LLMProviderError(data.message);
    case 503: return new ServiceUnavailableError(data.message);
    default: return new APIError(data.message, response?.status || 500);
  }
}
```

### 6.2 Error Mapping Table

| Frontend Error Class | Expected HTTP Status | Backend Exception Type | Actual HTTP Status | Message Format | Compatible |
|---------------------|---------------------|------------------------|-------------------|----------------|------------|
| `ValidationError` | 400 | `ValidationError` | 400 | `ErrorResponse` | ✅ |
| `AuthenticationError` | 401 | `HTTPException` | 401 | `{ detail: string }` | ⚠️ Different format |
| `AuthorizationError` | 403 | `HTTPException` | 403 | `{ detail: string }` | ⚠️ Different format |
| `NotFoundError` | 404 | `HTTPException` | 404 | `{ detail: string }` | ⚠️ Different format |
| `ConflictError` | 409 | `ConflictError` | 409 | `ErrorResponse` | ✅ |
| `RateLimitError` | 429 | `RateLimitExceeded` | 429 | `ErrorResponse` + `Retry-After` | ✅ |
| `InternalError` | 500 | `AppError` | 500 | `ErrorResponse` | ✅ |
| `LLMProviderError` | 502 | `LLMProviderError` | 502 | `ErrorResponse` | ✅ |
| `ServiceUnavailableError` | 503 | `ProviderNotAvailableError` | 503 | `ErrorResponse` | ✅ |
| `GatewayTimeoutError` | 504 | N/A | 504 | N/A | 🔄 Partial |
| `TransformationError` | 400 | `TransformationError` | 400 | `ErrorResponse` + `details` | ✅ |
| `InvalidPotencyError` | 400 | `InvalidPotencyError` | 400 | `ErrorResponse` | ✅ |
| `InvalidTechniqueError` | 400 | `InvalidTechniqueError` | 400 | `ErrorResponse` | ✅ |
| `CircuitBreakerOpenError` | 503 | N/A | N/A | Frontend only | 🔄 |
| `NetworkError` | 0 | N/A | N/A | Frontend only | 🔄 |
| `RequestAbortedError` | 0 | N/A | N/A | Frontend only | 🔄 |

### 6.3 Missing Backend Exceptions in Frontend

| Backend Exception | HTTP Status | Frontend Handling | Status |
|------------------|-------------|-------------------|--------|
| `MissingFieldError` | 400 | Falls to `ValidationError` | ⚠️ Lost specificity |
| `InvalidFieldError` | 400 | Falls to `ValidationError` | ⚠️ Lost specificity |
| `PayloadTooLargeError` | 413 | Not handled | ❌ **GAP-008** |
| `ProviderNotConfiguredError` | 400 | Falls to `ValidationError` | ⚠️ Lost specificity |
| `ProviderNotAvailableError` | 503 | Handled as `ServiceUnavailableError` | ✅ |
| `CacheError` | 500 | Falls to `InternalError` | ⚠️ Lost specificity |
| `ConfigurationError` | 500 | Falls to `InternalError` | ⚠️ Lost specificity |

### 6.4 HTTP Status Code Matrix

| Status Code | Backend Usage | Frontend Handling | Notes |
|-------------|--------------|-------------------|-------|
| 200 | Success | ✅ Handled | Compatible |
| 400 | Bad Request, Validation | `ValidationError` | Compatible |
| 401 | Unauthorized | `AuthenticationError` | Triggers token refresh |
| 403 | Forbidden | `AuthorizationError` | Permission denied |
| 404 | Not Found | `NotFoundError` | Resource not found |
| 409 | Conflict | `ConflictError` | Duplicate strategy, etc. |
| 413 | Payload Too Large | ❌ Not handled | **GAP-008** |
| 422 | Unprocessable Entity | Falls to `ValidationError` | Pydantic validation |
| 429 | Rate Limited | `RateLimitError` | Includes retry delay |
| 500 | Internal Error | `InternalError` | Generic server error |
| 501 | Not Implemented | ❌ Not handled | Streaming not supported |
| 502 | Bad Gateway | `LLMProviderError` | LLM provider failure |
| 503 | Service Unavailable | `ServiceUnavailableError` | Provider unavailable |
| 504 | Gateway Timeout | `GatewayTimeoutError` | Timeout |

---

## 7. Configuration Dependencies

### 7.1 Frontend Environment Variables

**File:** [`frontend/.env.example`](../frontend/.env.example)

| Variable | Default | Required | Backend Dependency |
|----------|---------|----------|-------------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8001` | Yes | Backend `HOST:PORT` |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8001` | Yes | Backend WebSocket URL |
| `NEXT_PUBLIC_API_VERSION` | `v1` | No | `API_V1_STR` |
| `NEXT_PUBLIC_APP_NAME` | `Chimera` | No | `PROJECT_NAME` |
| `NEXT_PUBLIC_ENABLE_ANALYTICS` | `false` | No | None |
| `NEXT_PUBLIC_SENTRY_DSN` | N/A | No | None |

### 7.2 Backend Environment Variables

**File:** [`backend-api/.env.example`](../backend-api/.env.example) and [`backend-api/app/core/config.py`](../backend-api/app/core/config.py)

| Variable | Default | Required | Frontend Dependency |
|----------|---------|----------|---------------------|
| `API_V1_STR` | `/api/v1` | No | Frontend API prefix |
| `PROJECT_NAME` | `Chimera Backend` | No | None |
| `VERSION` | `1.0.0` | No | None |
| `ENVIRONMENT` | `development` | Yes | None |
| `LOG_LEVEL` | `INFO` | No | None |
| `HOST` | `0.0.0.0` | Yes | `NEXT_PUBLIC_API_URL` |
| `PORT` | `8001` | Yes | `NEXT_PUBLIC_API_URL` |
| `JWT_SECRET` | N/A | **Yes** | Token validation |
| `JWT_ALGORITHM` | `HS256` | No | Token validation |
| `JWT_EXPIRATION_HOURS` | `1` | No | Token refresh timing |
| `CHIMERA_API_KEY` | N/A | **Yes** | X-API-Key header |

### 7.3 LLM Provider API Keys

| Variable | Provider | Frontend Usage |
|----------|----------|----------------|
| `GOOGLE_API_KEY` | Google/Gemini | Provider config |
| `GOOGLE_MODEL` | Default model | Model selection |
| `OPENAI_API_KEY` | OpenAI | Provider config |
| `OPENAI_MODEL` | Default model | Model selection |
| `ANTHROPIC_API_KEY` | Anthropic | Provider config |
| `ANTHROPIC_MODEL` | Default model | Model selection |
| `DEEPSEEK_API_KEY` | DeepSeek | Provider config |
| `DEEPSEEK_MODEL` | Default model | Model selection |
| `QWEN_API_KEY` | Qwen | Provider config |
| `CURSOR_API_KEY` | Cursor | Provider config |

### 7.4 Cross-Reference Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION DEPENDENCY GRAPH                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FRONTEND                              BACKEND                          │
│  ────────                              ───────                          │
│                                                                         │
│  NEXT_PUBLIC_API_URL ───────────────▶ HOST + PORT                      │
│  (http://localhost:8001)               (0.0.0.0:8001)                   │
│                                                                         │
│  NEXT_PUBLIC_WS_URL ────────────────▶ HOST + PORT (WebSocket)          │
│  (ws://localhost:8001)                                                  │
│                                                                         │
│  X-API-Key header ──────────────────▶ CHIMERA_API_KEY                  │
│                                                                         │
│  Authorization: Bearer ─────────────▶ JWT_SECRET + JWT_ALGORITHM       │
│                                                                         │
│  Token refresh timing ──────────────▶ JWT_EXPIRATION_HOURS             │
│  (5 min before expiry)                 (default: 1 hour)               │
│                                                                         │
│  /api/v1 prefix ────────────────────▶ API_V1_STR                       │
│                                        (default: /api/v1)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Rate Limiting Configuration

| Backend Setting | Default | Frontend Impact |
|----------------|---------|-----------------|
| `RATE_LIMIT_ENABLED` | `True` | 429 responses possible |
| `RATE_LIMIT_DEFAULT_LIMIT` | `60` | Requests per window |
| `RATE_LIMIT_DEFAULT_WINDOW` | `60` | Window size (seconds) |
| `JAILBREAK_RATE_LIMIT_PER_MINUTE` | `60` | Jailbreak endpoint limit |

### 7.6 Cache Configuration

| Backend Setting | Default | Frontend Impact |
|----------------|---------|-----------------|
| `ENABLE_CACHE` | `True` | Response caching |
| `CACHE_DEFAULT_TTL` | `3600` | Cache duration (1 hour) |
| `JAILBREAK_CACHE_ENABLED` | `True` | Jailbreak result caching |
| `JAILBREAK_CACHE_TTL_SECONDS` | `3600` | Jailbreak cache TTL |

---

## 8. Appendices

### Appendix A: Complete Backend Endpoint List

#### A.1 Health Endpoints (11 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | None | Comprehensive health check |
| GET | `/health/live` | None | Kubernetes liveness probe |
| GET | `/health/ready` | None | Kubernetes readiness probe |
| GET | `/health/circuit-breakers` | None | Circuit breaker status |
| POST | `/health/circuit-breakers/{name}/reset` | None | Reset specific circuit breaker |
| GET | `/health/proxy` | None | Proxy server health |
| GET | `/health/integration` | None | Provider integration health |
| GET | `/health/integration/graph` | None | Service dependency graph |
| GET | `/health/integration/history` | None | Health history |
| GET | `/health/integration/alerts` | None | Active alerts |
| POST | `/health/integration/check` | None | Trigger health check |

#### A.2 Generation Endpoints (2 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/generation/generate` | Yes | Generate text with LLM |
| GET | `/api/v1/generation/health` | Yes | LLM provider availability |

#### A.3 Streaming Endpoints (3 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/streaming/generate/stream` | Yes | SSE streaming generation |
| POST | `/api/v1/streaming/generate/stream/raw` | Yes | Raw text streaming |
| GET | `/api/v1/streaming/generate/stream/capabilities` | Yes | Streaming capabilities |

#### A.4 Provider Endpoints (7 total + 1 WebSocket)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/providers/` | Yes | List all providers |
| GET | `/api/v1/providers/{provider}/models` | Yes | Get provider models |
| POST | `/api/v1/providers/select` | Yes | Select provider/model |
| GET | `/api/v1/providers/rate-limit` | Yes | Check rate limit |
| GET | `/api/v1/providers/current` | Yes | Current selection |
| GET | `/api/v1/providers/health` | Yes | Provider health |
| WebSocket | `/api/v1/providers/ws/selection` | Yes | Real-time sync |

#### A.5 Session Endpoints (9 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/session/models` | Yes | Available models |
| POST | `/api/v1/session/models/validate` | Yes | Validate model |
| POST | `/api/v1/session` | Yes | Create session |
| GET | `/api/v1/session` | Yes | Get current session |
| DELETE | `/api/v1/session` | Yes | Delete session |
| GET | `/api/v1/session/{session_id}` | Yes | Get session by ID |
| GET | `/api/v1/session/stats` | Yes | Session statistics |
| GET | `/api/v1/session/current-model` | Yes | Current model |
| PUT | `/api/v1/session/model` | Yes | Update model |

#### A.6 Transformation Endpoints (4 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/transformation/` | Permission | Transform prompt |
| POST | `/api/v1/transformation/stream` | Permission | SSE transformation |
| POST | `/api/v1/transformation/estimate-tokens` | Permission | Estimate tokens |
| GET | `/api/v1/transformation/cache/stats` | Yes | Cache statistics |

#### A.7 Jailbreak Endpoints (15 total + 1 WebSocket)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/jailbreak/generate` | Yes | Generate jailbreak |
| POST | `/api/v1/jailbreak/generate/quick` | Yes | Quick generation |
| POST | `/api/v1/jailbreak/generate/batch` | Yes | Batch generation |
| GET | `/api/v1/jailbreak/generate/stream` | Yes | SSE streaming |
| WebSocket | `/api/v1/jailbreak/ws/generate` | Yes | WebSocket streaming |
| GET | `/api/v1/jailbreak/strategies` | Yes | List strategies |
| GET | `/api/v1/jailbreak/strategies/{type}` | Yes | Strategy details |
| GET | `/api/v1/jailbreak/vulnerabilities` | Yes | List vulnerabilities |
| GET | `/api/v1/jailbreak/cache/stats` | Yes | Cache statistics |
| DELETE | `/api/v1/jailbreak/cache` | Yes | Clear cache |
| GET | `/api/v1/jailbreak/session/{session_id}` | Yes | Get session |
| DELETE | `/api/v1/jailbreak/session/{session_id}` | Yes | Cancel session |
| GET | `/api/v1/jailbreak/health` | Yes | Service health |

#### A.8 AutoDAN Endpoints (4 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/autodan/jailbreak` | Yes | AutoDAN jailbreak |
| POST | `/api/v1/autodan/batch` | Yes | Batch generation |
| GET | `/api/v1/autodan/config` | Yes | Get configuration |
| POST | `/api/v1/autodan/lifelong` | Yes | Lifelong learning |

#### A.9 AutoDAN-Turbo Endpoints (19 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/autodan-turbo/attack` | Rate Limited | Single attack |
| POST | `/api/v1/autodan-turbo/warmup` | Rate Limited | Warm-up phase |
| POST | `/api/v1/autodan-turbo/lifelong` | Yes | Lifelong learning |
| POST | `/api/v1/autodan-turbo/test` | Yes | Test stage |
| GET | `/api/v1/autodan-turbo/strategies` | Yes | List strategies |
| GET | `/api/v1/autodan-turbo/strategies/{id}` | Yes | Strategy details |
| POST | `/api/v1/autodan-turbo/strategies` | Yes | Create strategy |
| DELETE | `/api/v1/autodan-turbo/strategies/{id}` | Yes | Delete strategy |
| POST | `/api/v1/autodan-turbo/strategies/search` | Yes | Search strategies |
| POST | `/api/v1/autodan-turbo/strategies/batch-inject` | Yes | Batch import |
| GET | `/api/v1/autodan-turbo/progress` | Yes | Learning progress |
| GET | `/api/v1/autodan-turbo/library/stats` | Yes | Library statistics |
| POST | `/api/v1/autodan-turbo/reset` | Yes | Reset progress |
| POST | `/api/v1/autodan-turbo/library/save` | Yes | Save library |
| POST | `/api/v1/autodan-turbo/library/clear` | Yes | Clear library |
| GET | `/api/v1/autodan-turbo/health` | Yes | Service health |
| POST | `/api/v1/autodan-turbo/transfer/export` | Yes | Export library |
| POST | `/api/v1/autodan-turbo/transfer/import` | Yes | Import library |
| POST | `/api/v1/autodan-turbo/score` | Yes | Score response |

#### A.10 DeepTeam Endpoints (14 total + WebSocket)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/deepteam/red-team` | Yes | Full red teaming |
| POST | `/api/v1/deepteam/quick-scan` | Yes | Quick scan |
| POST | `/api/v1/deepteam/security-audit` | Yes | Security audit |
| POST | `/api/v1/deepteam/bias-audit` | Yes | Bias audit |
| POST | `/api/v1/deepteam/owasp-assessment` | Yes | OWASP assessment |
| POST | `/api/v1/deepteam/assess-vulnerability` | Yes | Vulnerability test |
| GET | `/api/v1/deepteam/sessions` | Yes | List sessions |
| GET | `/api/v1/deepteam/sessions/{id}` | Yes | Get session |
| GET | `/api/v1/deepteam/sessions/{id}/result` | Yes | Get result |
| GET | `/api/v1/deepteam/vulnerabilities` | Yes | List vulnerabilities |
| GET | `/api/v1/deepteam/attacks` | Yes | List attacks |
| GET | `/api/v1/deepteam/presets` | Yes | List presets |
| GET | `/api/v1/deepteam/health` | Yes | Service health |

#### A.11 DeepTeam Jailbreak Endpoints (15 total + WebSocket)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/deepteam/jailbreak/generate` | Yes | Generate jailbreak |
| POST | `/api/v1/deepteam/jailbreak/batch` | Yes | Batch generation |
| GET | `/api/v1/deepteam/jailbreak/strategies` | Yes | List strategies |
| GET | `/api/v1/deepteam/jailbreak/strategies/{type}` | Yes | Strategy details |
| DELETE | `/api/v1/deepteam/jailbreak/cache` | Yes | Clear cache |
| GET | `/api/v1/deepteam/jailbreak/health` | Yes | Service health |
| WebSocket | `/api/v1/deepteam/jailbreak/ws/generate` | Yes | WebSocket streaming |
| GET | `/api/v1/deepteam/jailbreak/generate/stream` | Yes | SSE streaming |
| GET | `/api/v1/deepteam/jailbreak/sessions/{id}/prompts` | Yes | Session prompts |
| GET | `/api/v1/deepteam/jailbreak/sessions/{id}/prompts/{pid}` | Yes | Get prompt |
| DELETE | `/api/v1/deepteam/jailbreak/sessions/{id}` | Yes | Delete session |
| GET | `/api/v1/deepteam/jailbreak/sessions` | Yes | List sessions |
| POST | `/api/v1/deepteam/jailbreak/sessions/{id}/cancel` | Yes | Cancel session |

#### A.12 Admin Endpoints (14 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/admin/feature-flags` | Admin API Key | List feature flags |
| GET | `/api/v1/admin/feature-flags/stats` | Admin API Key | Feature flag stats |
| POST | `/api/v1/admin/feature-flags/toggle` | Admin API Key | Toggle technique |
| POST | `/api/v1/admin/feature-flags/reload` | Admin API Key | Reload config |
| GET | `/api/v1/admin/feature-flags/{technique}` | Admin API Key | Technique details |
| GET | `/api/v1/admin/tenants` | Admin API Key | List tenants |
| POST | `/api/v1/admin/tenants` | Admin API Key | Create tenant |
| GET | `/api/v1/admin/tenants/{id}` | Admin API Key | Get tenant |
| DELETE | `/api/v1/admin/tenants/{id}` | Admin API Key | Delete tenant |
| GET | `/api/v1/admin/tenants/stats/summary` | Admin API Key | Tenant stats |
| GET | `/api/v1/admin/usage/global` | Admin API Key | Global usage |
| GET | `/api/v1/admin/usage/tenant/{id}` | Admin API Key | Tenant usage |
| GET | `/api/v1/admin/usage/techniques/top` | Admin API Key | Top techniques |
| GET | `/api/v1/admin/usage/quota/{id}` | Admin API Key | Check quota |

#### A.13 Metrics Endpoints (11 total)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/metrics/prometheus` | None | Prometheus format |
| GET | `/api/v1/metrics/json` | None | JSON format |
| GET | `/api/v1/metrics/circuit-breakers` | None | Circuit breaker status |
| POST | `/api/v1/metrics/circuit-breakers/{name}/reset` | None | Reset circuit breaker |
| POST | `/api/v1/metrics/circuit-breakers/reset-all` | None | Reset all |
| GET | `/api/v1/metrics/cache` | None | Cache metrics |
| POST | `/api/v1/metrics/cache/clear` | None | Clear cache |
| GET | `/api/v1/metrics/connection-pools` | None | Pool stats |
| POST | `/api/v1/metrics/connection-pools/reset` | None | Reset pools |
| GET | `/api/v1/metrics/multi-level-cache` | None | L1/L2 cache |
| POST | `/api/v1/metrics/multi-level-cache/clear` | None | Clear multi-level |

---

### Appendix B: Complete Frontend API Call List

#### B.1 API Client Methods (by service)

**Authentication** ([`auth-manager.ts`](../frontend/src/lib/api/auth-manager.ts)):
- `login(email, password)` → `POST /api/v1/auth/login`
- `refreshAccessToken()` → `POST /api/v1/auth/refresh`
- `logout()` → `POST /api/v1/auth/logout`

**Providers** ([`useProviderConfig.ts`](../frontend/src/hooks/useProviderConfig.ts)):
- `getProviders()` → `GET /api/v1/providers/`
- `getCurrentProvider()` → `GET /api/v1/providers/current`
- `getModelsForProvider(provider)` → `GET /api/v1/providers/{provider}/models`
- `selectProvider(provider_id, model_id)` → `POST /api/v1/providers/select`
- `getProviderHealth()` → `GET /api/v1/providers/health`
- `createProvider(data)` → `POST /api/v1/provider-config/providers`
- `updateProvider(id, data)` → `PUT /api/v1/provider-config/providers/{id}`
- `deleteProvider(id)` → `DELETE /api/v1/provider-config/providers/{id}`
- `testProvider(id)` → `POST /api/v1/provider-config/providers/{id}/test`

**Session** ([`api-enhanced.ts`](../frontend/src/lib/api-enhanced.ts)):
- `session.getModels()` → `GET /api/v1/session/models`
- `session.validateModel(model_id)` → `POST /api/v1/session/models/validate`
- `session.create(config)` → `POST /api/v1/session`
- `session.getCurrent()` → `GET /api/v1/session`
- `session.delete()` → `DELETE /api/v1/session`
- `session.getCurrentModel()` → `GET /api/v1/session/current-model`
- `session.updateModel(model_id)` → `PUT /api/v1/session/model`

**Generation** ([`api-enhanced.ts`](../frontend/src/lib/api-enhanced.ts)):
- `generate.text(request)` → `POST /api/v1/generation/generate`
- `generate.health()` → `GET /api/v1/generation/health`
- `streaming.generate(request)` → `POST /api/v1/streaming/generate/stream` (SSE)

**Jailbreak** ([`jailbreak.ts`](../frontend/src/api/jailbreak.ts)):
- `JailbreakAPI.generate(request)` → `POST /api/v1/deepteam/jailbreak/generate`
- `JailbreakAPI.generateBatch(request)` → `POST /api/v1/deepteam/jailbreak/batch`
- `JailbreakAPI.getStrategies()` → `GET /api/v1/deepteam/jailbreak/strategies`
- `JailbreakAPI.getStrategyDetails(type)` → `GET /api/v1/deepteam/jailbreak/strategies/{type}`
- `JailbreakAPI.getVulnerabilities()` → `GET /api/v1/deepteam/jailbreak/vulnerabilities`
- `JailbreakAPI.clearCache()` → `DELETE /api/v1/deepteam/jailbreak/cache`
- `JailbreakAPI.getHealth()` → `GET /api/v1/deepteam/jailbreak/health`
- `JailbreakAPI.getPrompt(session_id, prompt_id)` → `GET /api/v1/deepteam/jailbreak/sessions/{id}/prompts/{pid}`
- `JailbreakAPI.getSessionPrompts(session_id)` → `GET /api/v1/deepteam/jailbreak/sessions/{id}/prompts`
- `JailbreakAPI.deleteSession(session_id)` → `DELETE /api/v1/deepteam/jailbreak/sessions/{id}`
- `JailbreakWebSocket.connect(request)` → `WebSocket /api/v1/deepteam/jailbreak/ws/generate`
- `JailbreakSSE.connect(request)` → `GET /api/v1/deepteam/jailbreak/generate/stream` (SSE)

**AutoDAN** ([`api-enhanced.ts`](../frontend/src/lib/api-enhanced.ts)):
- `autodan.jailbreak(request)` → `POST /api/v1/autodan/jailbreak`
- `autodan.batch(request)` → `POST /api/v1/autodan/batch`
- `autodan.getConfig()` → `GET /api/v1/autodan/config`

**AutoDAN-Turbo** ([`api-enhanced.ts`](../frontend/src/lib/api-enhanced.ts)):
- `autodanTurbo.attack(request)` → `POST /api/v1/autodan-turbo/attack`
- `autodanTurbo.warmup(request)` → `POST /api/v1/autodan-turbo/warmup`
- `autodanTurbo.lifelong(request)` → `POST /api/v1/autodan-turbo/lifelong`
- `autodanTurbo.test(request)` → `POST /api/v1/autodan-turbo/test`
- `autodanTurbo.getStrategies()` → `GET /api/v1/autodan-turbo/strategies`
- `autodanTurbo.getStrategy(id)` → `GET /api/v1/autodan-turbo/strategies/{id}`
- `autodanTurbo.createStrategy(data)` → `POST /api/v1/autodan-turbo/strategies`
- `autodanTurbo.deleteStrategy(id)` → `DELETE /api/v1/autodan-turbo/strategies/{id}`
- `autodanTurbo.searchStrategies(query)` → `POST /api/v1/autodan-turbo/strategies/search`
- `autodanTurbo.batchInject(data)` → `POST /api/v1/autodan-turbo/strategies/batch-inject`
- `autodanTurbo.getProgress()` → `GET /api/v1/autodan-turbo/progress`
- `autodanTurbo.getLibraryStats()` → `GET /api/v1/autodan-turbo/library/stats`
- `autodanTurbo.reset()` → `POST /api/v1/autodan-turbo/reset`
- `autodanTurbo.saveLibrary()` → `POST /api/v1/autodan-turbo/library/save`
- `autodanTurbo.clearLibrary()` → `POST /api/v1/autodan-turbo/library/clear`
- `autodanTurbo.health()` → `GET /api/v1/autodan-turbo/health`

**DeepTeam** ([`deepteam-client.ts`](../frontend/src/lib/api/deepteam-client.ts)):
- `redTeam(request)` → `POST /api/v1/deepteam/red-team`
- `quickScan(request)` → `POST /api/v1/deepteam/quick-scan`
- `securityAudit(request)` → `POST /api/v1/deepteam/security-audit`
- `biasAudit(request)` → `POST /api/v1/deepteam/bias-audit`
- `listSessions()` → `GET /api/v1/deepteam/sessions`
- `getSession(id)` → `GET /api/v1/deepteam/sessions/{id}`
- `getSessionResult(id)` → `GET /api/v1/deepteam/sessions/{id}/result`
- `listAgents()` → `GET /api/v1/deepteam/agents` (❌ Missing)
- `getAgent(id)` → `GET /api/v1/deepteam/agents/{id}` (❌ Missing)
- `listEvaluations()` → `GET /api/v1/deepteam/evaluations` (❌ Missing)
- `getEvaluation(id)` → `GET /api/v1/deepteam/evaluations/{id}` (❌ Missing)
- `listRefinements()` → `GET /api/v1/deepteam/refinements` (❌ Missing)
- `applyRefinement(request)` → `POST /api/v1/deepteam/refinements/apply` (❌ Missing)
- `createWebSocketConnection(sessionId)` → `WebSocket /ws/sessions/{sessionId}`

**Health** (various):
- `getHealth()` → `GET /health`
- `getLiveness()` → `GET /health/live`

---

### Appendix C: Glossary of Terms

| Term | Definition |
|------|------------|
| **API Key** | Static authentication token passed via `X-API-Key` header for machine-to-machine communication |
| **AutoDAN** | Automated jailbreak generation using gradient-based optimization |
| **AutoDAN-Turbo** | Enhanced AutoDAN with lifelong learning capabilities |
| **Bearer Token** | JWT-based authentication token passed via `Authorization: Bearer` header |
| **Circuit Breaker** | Resilience pattern that prevents cascading failures by stopping requests to unhealthy services |
| **DeepTeam** | Red teaming framework for adversarial testing of LLM systems |
| **GAP** | Identified integration gap between frontend and backend systems |
| **Jailbreak** | Technique to bypass LLM safety filters and content restrictions |
| **JWT** | JSON Web Token - compact, URL-safe means of representing claims between parties |
| **LLM** | Large Language Model - AI models trained on massive text datasets |
| **OWASP** | Open Web Application Security Project - security standards organization |
| **Potency Level** | 1-10 scale indicating the aggressiveness of jailbreak techniques |
| **PPO** | Proximal Policy Optimization - reinforcement learning algorithm |
| **Pydantic** | Python data validation library using type annotations |
| **RBAC** | Role-Based Access Control - permission system based on user roles |
| **Red Team** | Security testing approach simulating adversarial attacks |
| **SSE** | Server-Sent Events - one-way server-to-client streaming protocol |
| **TanStack Query** | React Query - data fetching and caching library (formerly React Query) |
| **Technique Suite** | Collection of jailbreak techniques applied together |
| **WebSocket** | Full-duplex communication protocol for real-time data exchange |
| **Zustand** | Lightweight state management library for React |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-06 | Technical Analysis Team | Initial release |

---

**Generated:** 2026-01-06T01:50:00Z
**Source Documents:** BACKEND_API_AUDIT.md, FRONTEND_API_AUDIT.md, GAP_ANALYSIS_REPORT.md, EXECUTIVE_SUMMARY.md
**Total Mappings Documented:** 150+
**Gaps Referenced:** 21 (from GAP_ANALYSIS_REPORT.md)