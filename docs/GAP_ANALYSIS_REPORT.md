# Backend-Frontend Gap Analysis Report

**Project:** Chimera  
**Date:** 2026-01-06  
**Phase:** 3 - Integration Gap Analysis  
**Author:** Code Skeptic Mode (Automated)  
**Status:** CRITICAL ISSUES IDENTIFIED

---

## 1. Executive Summary

This gap analysis reveals **severe integration issues** between the Chimera backend (FastAPI) and frontend (Next.js) systems. The analysis uncovered **3 CRITICAL**, **8 HIGH**, **6 MEDIUM**, and **4 LOW** priority gaps that must be addressed before production deployment.

### Key Findings

| Severity | Count | Impact |
|----------|-------|--------|
| **CRITICAL** | 3 | System will fail at runtime |
| **HIGH** | 8 | Major functionality broken |
| **MEDIUM** | 6 | Partial functionality issues |
| **LOW** | 4 | Minor inconsistencies |

### Overall Health Assessment: 🔴 **FAILING**

The frontend-backend integration is **NOT production-ready**. Critical authentication flow mismatches, provider type incompatibilities, and hardcoded configuration values will cause immediate failures.

---

## 2. Critical Gaps (Severity: CRITICAL)

These issues **WILL cause runtime failures**.

### GAP-001: Authentication Endpoints Do Not Exist

**Source Files:**
- Backend: [`backend-api/app/core/auth.py`](../backend-api/app/core/auth.py)
- Frontend: [`frontend/src/lib/api/types.ts`](../frontend/src/lib/api/types.ts:274-280)

**Problem:**  
The frontend expects `/api/v1/auth/login` and `/api/v1/auth/refresh` endpoints to exist. **They do not exist in the backend.** The backend `auth.py` only contains a `get_current_user()` dependency function, not HTTP endpoints.

**Evidence - Backend auth.py (lines 194-227):**
```python
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get the current authenticated user from JWT token."""
    # Only dependency injection, NO router endpoints
```

**Evidence - Frontend expectations (types.ts lines 280-285):**
```typescript
export type AuthEndpoints = {
  login: '/api/v1/auth/login';     // ❌ DOES NOT EXIST
  refresh: '/api/v1/auth/refresh'; // ❌ DOES NOT EXIST
  logout: '/api/v1/auth/logout';   // ❌ DOES NOT EXIST
};
```

**Impact:** Users cannot log in. All authenticated requests will fail.

**Recommended Fix:** Create `/api/v1/auth/` router with `login`, `refresh`, and `logout` endpoints.

---

### GAP-002: Provider Type Mismatch (8 Missing Providers)

**Source Files:**
- Backend: [`backend-api/app/domain/models.py`](../backend-api/app/domain/models.py:20-35)
- Frontend: [`frontend/src/lib/api/types.ts`](../frontend/src/lib/api/types.ts:43)

**Problem:**  
Backend supports **12 LLM provider types**, but frontend only defines **4 types**. This causes TypeScript compilation errors and runtime failures when using 8 providers.

**Backend LLMProviderType enum (12 values):**
```python
class LLMProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"        # ❌ Missing in frontend
    GEMINI = "gemini"
    QWEN = "qwen"            # ❌ Missing in frontend
    GEMINI_CLI = "gemini-cli" # ❌ Missing in frontend
    ANTIGRAVITY = "antigravity" # ❌ Missing in frontend
    KIRO = "kiro"            # ❌ Missing in frontend
    CURSOR = "cursor"        # ❌ Missing in frontend
    XAI = "xai"              # ❌ Missing in frontend
    DEEPSEEK = "deepseek"
    MOCK = "mock"            # ❌ Missing in frontend
```

**Frontend Provider type (4 values only):**
```typescript
type: 'openai' | 'anthropic' | 'gemini' | 'deepseek';
// Missing: google, qwen, gemini-cli, antigravity, kiro, cursor, xai, mock
```

**Impact:** Users cannot configure or use GOOGLE, QWEN, GEMINI_CLI, ANTIGRAVITY, KIRO, CURSOR, XAI, or MOCK providers.

**Recommended Fix:** Update `frontend/src/lib/api/types.ts` to include all 12 provider types.

---

### GAP-003: Hardcoded WebSocket URL

**Source File:** [`frontend/src/api/jailbreak.ts`](../frontend/src/api/jailbreak.ts:18-19)

**Problem:**  
WebSocket URL is hardcoded with a comment "HARDCODED FORCE FIX", making deployment to any non-localhost environment impossible.

**Evidence (jailbreak.ts lines 18-19):**
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
const WS_BASE_URL = 'ws://localhost:8001'; // HARDCODED FORCE FIX
```

**Impact:** WebSocket connections fail in production, staging, or any non-localhost deployment.

**Recommended Fix:** 
```typescript
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 
  API_BASE_URL.replace(/^http/, 'ws');
```

---

## 3. Major Gaps (Severity: HIGH)

These issues cause **significant functionality problems**.

### GAP-004: Extra Field in AuthTokens (refresh_expires_in)

**Source Files:**
- Backend: [`backend-api/app/core/auth.py`](../backend-api/app/core/auth.py:126-132)
- Frontend: [`frontend/src/lib/api/types.ts`](../frontend/src/lib/api/types.ts:283-289)

**Problem:**  
Frontend `AuthTokens` interface has a `refresh_expires_in` field that **does not exist** in backend `TokenResponse`.

**Backend TokenResponse:**
```python
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds only
```

**Frontend AuthTokens:**
```typescript
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';
  expires_in: number;
  refresh_expires_in: number;  // ❌ DOES NOT EXIST IN BACKEND
}
```

**Impact:** TypeScript expects a field that will never be returned, causing potential undefined access errors.

---

### GAP-005: Admin Endpoints - 0% Frontend Coverage

**Source File:** [`backend-api/app/api/v1/endpoints/admin.py`](../backend-api/app/api/v1/endpoints/admin.py)

**Problem:**  
14 admin endpoints exist in backend with **zero** frontend implementation.

| Backend Endpoint | Frontend Coverage |
|-----------------|-------------------|
| GET `/admin/feature-flags` | ❌ None |
| GET `/admin/feature-flags/stats` | ❌ None |
| POST `/admin/feature-flags/toggle` | ❌ None |
| POST `/admin/feature-flags/reload` | ❌ None |
| GET `/admin/feature-flags/{technique_name}` | ❌ None |
| GET `/admin/tenants` | ❌ None |
| POST `/admin/tenants` | ❌ None |
| GET `/admin/tenants/{tenant_id}` | ❌ None |
| DELETE `/admin/tenants/{tenant_id}` | ❌ None |
| GET `/admin/tenants/stats/summary` | ❌ None |
| GET `/admin/usage/global` | ❌ None |
| GET `/admin/usage/tenant/{tenant_id}` | ❌ None |
| GET `/admin/usage/techniques/top` | ❌ None |
| GET `/admin/usage/quota/{tenant_id}` | ❌ None |

**Impact:** Admin functionality is completely inaccessible from the UI.

---

### GAP-006: Metrics Endpoints - 0% Frontend Coverage

**Source File:** [`backend-api/app/api/v1/endpoints/metrics.py`](../backend-api/app/api/v1/endpoints/metrics.py)

**Problem:**  
11 metrics endpoints exist in backend with **zero** frontend implementation.

| Backend Endpoint | Frontend Coverage |
|-----------------|-------------------|
| GET `/metrics/prometheus` | ❌ None |
| GET `/metrics/json` | ❌ None |
| GET `/metrics/circuit-breakers` | ❌ None |
| POST `/metrics/circuit-breakers/{name}/reset` | ❌ None |
| POST `/metrics/circuit-breakers/reset-all` | ❌ None |
| GET `/metrics/cache` | ❌ None |
| POST `/metrics/cache/clear` | ❌ None |
| GET `/metrics/connection-pools` | ❌ None |
| POST `/metrics/connection-pools/reset` | ❌ None |
| GET `/metrics/multi-level-cache` | ❌ None |
| POST `/metrics/multi-level-cache/clear` | ❌ None |

**Impact:** Operations team cannot monitor system health through the UI.

---

### GAP-007: Deprecated API Client Still in Active Use

**Source Files:**
- Deprecated: [`frontend/src/lib/api-enhanced.ts`](../frontend/src/lib/api-enhanced.ts:1-93)
- Active User: [`frontend/src/lib/websocket-manager.ts`](../frontend/src/lib/websocket-manager.ts)

**Problem:**  
The 2,485-line `api-enhanced.ts` is explicitly marked `@deprecated` but is still imported and used by `websocket-manager.ts`.

**Evidence (api-enhanced.ts lines 1-15):**
```typescript
/**
 * @deprecated This file is deprecated and will be removed in a future release.
 * Please migrate to the new API architecture...
 */
```

**Evidence of active usage in websocket-manager.ts:**
```typescript
import { getCurrentApiUrl } from './api-enhanced';
```

**Impact:** Technical debt accumulation, inconsistent API patterns, maintenance burden.

---

### GAP-008: Missing Error Classes in Frontend

**Source Files:**
- Backend: [`backend-api/app/core/exceptions.py`](../backend-api/app/core/exceptions.py)
- Frontend: [`frontend/src/lib/errors/api-errors.ts`](../frontend/src/lib/errors/api-errors.ts)

**Problem:**  
Backend defines 7 additional exception types not mirrored in frontend:

| Backend Exception | Frontend Equivalent |
|------------------|---------------------|
| `MissingFieldError` | ❌ Missing |
| `InvalidFieldError` | ❌ Missing |
| `PayloadTooLargeError` | ❌ Missing |
| `ProviderNotConfiguredError` | ❌ Missing |
| `ProviderNotAvailableError` | ❌ Missing |
| `CacheError` | ❌ Missing |
| `ConfigurationError` | ❌ Missing |

**Impact:** Specific error types are downgraded to generic errors, losing diagnostic information.

---

### GAP-009: token_type Casing Mismatch

**Source Files:**
- Backend: [`backend-api/app/core/auth.py`](../backend-api/app/core/auth.py:129)
- Frontend: [`frontend/src/lib/api/types.ts`](../frontend/src/lib/api/types.ts:286)

**Problem:**  
Backend returns `"bearer"` (lowercase), frontend expects `'Bearer'` (capitalized).

**Backend:**
```python
token_type: str = "bearer"
```

**Frontend:**
```typescript
token_type: 'Bearer';  // Strict literal type
```

**Impact:** TypeScript type checking will fail on authentication responses.

---

### GAP-010: JailbreakTechnique vs TechniqueSuite Mismatch

**Source Files:**
- Backend: [`backend-api/app/domain/models.py`](../backend-api/app/domain/models.py:53-69)
- Frontend: [`frontend/src/types/jailbreak.ts`](../frontend/src/types/jailbreak.ts)

**Problem:**  
Frontend defines 12 `JailbreakTechnique` values that don't align with backend `TechniqueSuite` enum.

**Backend TechniqueSuite:**
```python
class TechniqueSuite(str, Enum):
    BASIC_INJECTION = "basic_injection"
    JAILBREAK_BASIC = "jailbreak_basic"
    # ... 12 technique categories
```

**Frontend JailbreakTechnique:**
```typescript
type JailbreakTechnique = 
  | 'prompt_injection'  // Different naming!
  | 'role_playing'
  // ... different set
```

**Impact:** Technique selection in UI may send invalid values to backend.

---

### GAP-011: WebSocket URL Inconsistency Across Files

**Source Files:**
- Hardcoded: [`frontend/src/api/jailbreak.ts`](../frontend/src/api/jailbreak.ts:19)
- Dynamic: [`frontend/src/lib/websocket-manager.ts`](../frontend/src/lib/websocket-manager.ts)

**Problem:**  
Two different WebSocket URL strategies:
1. `jailbreak.ts` - Hardcoded `ws://localhost:8001`
2. `websocket-manager.ts` - Dynamic via `getCurrentApiUrl()`

**Impact:** Inconsistent behavior between different WebSocket features.

---

## 4. Minor Gaps (Severity: MEDIUM)

These issues cause **partial functionality problems**.

### GAP-012: Missing WebSocket Endpoints in Frontend

**Problem:**  
Backend provides 5 WebSocket endpoints, frontend only implements 3-4.

| Backend WebSocket | Frontend Implementation |
|------------------|-------------------------|
| `/api/v1/provider-config/ws/updates` | ✅ SSEManager |
| `/api/v1/deepteam/jailbreak/ws/generate` | ✅ JailbreakWebSocket |
| `/ws/enhance` | ❌ Not implemented |
| `/api/v1/autoadv/ws` | ❌ Not implemented |
| `/api/v1/pipeline/streaming/metrics` | ❌ Not implemented |

---

### GAP-013: SSE Endpoint Path Mismatch

**Problem:**  
Frontend `JailbreakSSE` connects to `/api/v1/deepteam/jailbreak/generate/stream` but there's also:
- `/api/v1/jailbreak/generate/stream` (duplicate?)
- `/api/v1/advanced/jailbreak/generate/stream`

**Potential Issue:** Unclear which endpoint should be used.

---

### GAP-014: Duplicate Jailbreak Endpoint Paths

**Problem:**  
Backend has jailbreak endpoints duplicated across:
- `/api/v1/jailbreak/` prefix
- `/api/v1/deepteam/jailbreak/` prefix

Frontend primarily uses `/api/v1/deepteam/` but audit mentions both.

---

### GAP-015: Missing `any` Type Cleanup

**Problem:**  
Frontend audit identified ~15 `any` usages that could cause runtime type errors:
- `frontend/src/lib/api-enhanced.ts` - Multiple instances
- Various component files

---

### GAP-016: CamelCase Conversion Not Always Applied

**Source File:** [`backend-api/app/domain/models.py`](../backend-api/app/domain/models.py:10-18)

**Problem:**  
Backend has `CamelCaseModel` base class for automatic snake_case to camelCase conversion, but not all models inherit from it.

```python
class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
```

**Impact:** Some API responses use snake_case, others use camelCase.

---

### GAP-017: RBAC Permissions Not Exposed to Frontend

**Problem:**  
Backend has 15 granular permissions:
- READ_PROMPTS, READ_TECHNIQUES, READ_PROVIDERS, READ_METRICS, READ_LOGS
- WRITE_PROMPTS, WRITE_TECHNIQUES, WRITE_PROVIDERS
- EXECUTE_TRANSFORM, EXECUTE_ENHANCE, EXECUTE_JAILBREAK
- ADMIN_USERS, ADMIN_SYSTEM, ADMIN_AUDIT, ADMIN_CONFIG

Frontend has no permission constants or RBAC type definitions.

---

## 5. Low Priority Gaps (Severity: LOW)

These are **cosmetic or minor** issues.

### GAP-018: Inconsistent Error Message Defaults

Backend and frontend have slightly different default error messages for the same error types.

### GAP-019: Frontend Uses 5-min TanStack Query staleTime

May cause stale data display for rapidly changing backend state.

### GAP-020: Missing API Version Prefix Consistency

Some frontend calls use `/api/v1/`, some don't include version.

### GAP-021: Dual Error System (api-errors.ts + errors/index.ts)

Frontend has error classes defined in two places with slight variations.

---

## 6. Endpoint Coverage Matrix

### Backend Endpoints vs Frontend Coverage

| Category | Backend Endpoints | Frontend Coverage | % |
|----------|------------------|-------------------|---|
| Health | 3 | 3 | 100% |
| Auth | 0 (expected 3) | 3 expected | N/A |
| Providers | 8 | 6 | 75% |
| Session | 6 | 5 | 83% |
| Generation | 4 | 3 | 75% |
| Streaming (SSE) | 7 | 2 | 29% |
| Jailbreak | 8 | 6 | 75% |
| AutoDAN | 6 | 4 | 67% |
| AutoDAN-Turbo | 5 | 4 | 80% |
| DeepTeam | 12 | 8 | 67% |
| Admin | 14 | 0 | 0% |
| Metrics | 11 | 0 | 0% |
| Transformation | 5 | 3 | 60% |
| WebSocket | 5 | 3 | 60% |
| **TOTAL** | ~95 | ~50 | **~53%** |

---

## 7. Type Alignment Matrix

### Backend Pydantic Models vs Frontend TypeScript Types

| Backend Model | Frontend Type | Status | Notes |
|--------------|---------------|--------|-------|
| `LLMProviderType` (12 values) | Provider `type` (4 values) | ❌ MISMATCH | 8 missing |
| `TokenResponse` | `AuthTokens` | ❌ MISMATCH | Extra field, casing |
| `TechniqueSuite` | `JailbreakTechnique` | ❌ MISMATCH | Different values |
| `Permission` (15 values) | None | ❌ MISSING | Not defined |
| `Role` (5 values) | None | ❌ MISSING | Not defined |
| `PromptRequest` | `PromptRequest` | ⚠️ PARTIAL | Some fields missing |
| `StreamChunk` | SSE parsing | ✅ ALIGNED | Format matches |
| `APIException` hierarchy | `APIError` hierarchy | ⚠️ PARTIAL | 7 classes missing |

---

## 8. Authentication Flow Comparison

### Expected Flow (Frontend)

```
1. POST /api/v1/auth/login → { access_token, refresh_token, token_type: 'Bearer', expires_in, refresh_expires_in }
2. Store tokens in Zustand SessionStore
3. Attach Authorization: Bearer <token> to requests
4. POST /api/v1/auth/refresh when token expires
5. POST /api/v1/auth/logout to clear session
```

### Actual Flow (Backend)

```
1. ❌ /api/v1/auth/login DOES NOT EXIST
2. ❌ /api/v1/auth/refresh DOES NOT EXIST  
3. ❌ /api/v1/auth/logout DOES NOT EXIST
4. get_current_user() dependency validates tokens AFTER they exist
5. Backend assumes tokens are already created externally
```

### Gap Analysis

| Step | Frontend Expects | Backend Provides | Status |
|------|-----------------|------------------|--------|
| Login | POST `/auth/login` | Nothing | ❌ BROKEN |
| Token Refresh | POST `/auth/refresh` | Nothing | ❌ BROKEN |
| Logout | POST `/auth/logout` | Nothing | ❌ BROKEN |
| Token Validation | Dependency | `get_current_user()` | ✅ OK |
| API Key Auth | X-API-Key header | Supported | ✅ OK |

---

## 9. WebSocket/SSE Compatibility Analysis

### WebSocket Endpoints

| Backend Path | Frontend Implementation | Status |
|-------------|------------------------|--------|
| `/api/v1/provider-config/ws/updates` | SSEManager (uses EventSource) | ⚠️ TYPE MISMATCH |
| `/api/v1/deepteam/jailbreak/ws/generate` | JailbreakWebSocket | ✅ Compatible |
| `/ws/enhance` | Not implemented | ❌ Missing |
| `/api/v1/autoadv/ws` | Not implemented | ❌ Missing |
| `/api/v1/pipeline/streaming/metrics` | Not implemented | ❌ Missing |

### SSE Endpoints

| Backend Path | Frontend Implementation | Status |
|-------------|------------------------|--------|
| `/api/v1/streaming/generate` | chat-service.ts | ✅ Compatible |
| `/api/v1/transformation/stream` | Not found | ❌ Missing |
| `/api/v1/jailbreak/generate/stream` | JailbreakSSE | ✅ Compatible |
| `/api/v1/deepteam/jailbreak/generate/stream` | JailbreakSSE | ✅ Compatible |
| `/api/v1/advanced/jailbreak/generate/stream` | Not found | ❌ Missing |
| `/api/v1/advanced/code/generate/stream` | Not found | ❌ Missing |

### Message Format Compatibility

**Backend SSE Format:**
```
event: <event_type>
data: {"text": "...", "is_final": false, "finish_reason": null}

```

**Frontend SSE Parsing:**
```typescript
// JailbreakSSE
this.eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Expects: { text, is_final, finish_reason }
};
```

**Status:** ✅ Compatible

---

## 10. Recommendations

### Priority 1: Critical Fixes (Do Immediately)

1. **Create Auth Router** - Add `/api/v1/auth/` with login, refresh, logout endpoints
2. **Fix Provider Types** - Add all 12 provider types to frontend TypeScript
3. **Fix WebSocket URL** - Use environment variable instead of hardcoded localhost

### Priority 2: High Priority Fixes (This Sprint)

4. **Remove `refresh_expires_in`** - Or add it to backend TokenResponse
5. **Fix token_type casing** - Backend should return `"Bearer"` or frontend accept both
6. **Add missing error classes** - Mirror all 7 backend exceptions
7. **Deprecate api-enhanced.ts** - Migrate websocket-manager.ts to new API

### Priority 3: Medium Priority Fixes (Next Sprint)

8. **Implement admin UI** - Create admin dashboard for 14 endpoints
9. **Implement metrics UI** - Create monitoring dashboard for 11 endpoints
10. **Align TechniqueSuite** - Update frontend JailbreakTechnique enum
11. **Add missing WebSocket handlers** - `/ws/enhance`, `/api/v1/autoadv/ws`
12. **Add missing SSE handlers** - transformation, advanced generation

### Priority 4: Low Priority (Backlog)

13. **Clean up `any` types** - Replace with proper TypeScript types
14. **Document CamelCase convention** - Ensure consistency
15. **Add RBAC types to frontend** - Create Permission/Role enums
16. **Consolidate error classes** - Single source of truth

---

## Appendix A: Verification Commands Used

```bash
# Files examined
backend-api/app/core/auth.py
backend-api/app/domain/models.py
backend-api/app/core/exceptions.py
backend-api/app/api/v1/endpoints/admin.py
backend-api/app/api/v1/endpoints/metrics.py
backend-api/app/api/v1/endpoints/streaming.py
backend-api/app/api/v1/endpoints/jailbreak.py
backend-api/app/api/v1/endpoints/deepteam.py
frontend/src/lib/api/types.ts
frontend/src/types/jailbreak.ts
frontend/src/api/jailbreak.ts
frontend/src/lib/websocket-manager.ts
frontend/src/lib/api-enhanced.ts
frontend/src/lib/errors/api-errors.ts
frontend/src/lib/sync/sse-manager.ts
```

---

## Appendix B: Source File References

All findings are backed by direct source code examination. Line numbers reference the state of files at analysis time (2026-01-06).

| Gap ID | Primary Source Files |
|--------|---------------------|
| GAP-001 | auth.py:194-227, types.ts:280-285 |
| GAP-002 | models.py:20-35, types.ts:43 |
| GAP-003 | jailbreak.ts:18-19 |
| GAP-004 | auth.py:126-132, types.ts:283-289 |
| GAP-005 | admin.py (full file) |
| GAP-006 | metrics.py (full file) |
| GAP-007 | api-enhanced.ts:1-93 |
| GAP-008 | exceptions.py vs api-errors.ts |
| GAP-009 | auth.py:129, types.ts:286 |
| GAP-010 | models.py:53-69, jailbreak.ts |
| GAP-011 | jailbreak.ts:19, websocket-manager.ts |

---

**Report Generated:** 2026-01-06T01:36:33Z  
**Analysis Duration:** ~15 minutes  
**Files Analyzed:** 18  
**Total Gaps Identified:** 21