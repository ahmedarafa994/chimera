# Architecture Gap Analysis & Remediation Document

**Date:** 2025-12-04
**Author:** Kilo Code, Senior Solutions Architect
**Scope:** Frontend (Next.js) and Backend (FastAPI) Codebase Audit

## 1. High-Level Overview

The Chimera project aims to provide a sophisticated prompt enhancement and jailbreak testing platform. The architecture consists of a Next.js frontend and a FastAPI backend.

**Current Status:**
The integration is currently **fragmented and partially broken**. While the frontend has a rich UI with detailed configuration options, the backend implementation lags behind in specific areas, leading to "orphaned" frontend features that have no corresponding backend logic. There are significant mismatches in API endpoint paths and data schemas, particularly for the core jailbreak generation features.

## 2. Gap Analysis: Missing Services & Orphaned Components

The following table details the specific discrepancies identified between the client-side requirements and server-side implementations.

| Feature Area | Frontend Component / API Call | Backend Implementation Status | Gap Description |
| :--- | :--- | :--- | :--- |
| **Jailbreak Generation** | `POST /generation/jailbreak/generate` (in `api-enhanced.ts`) | `POST /api/v1/jailbreak/execute` (in `jailbreak.py`) | **Critical Path Mismatch.** Frontend calls a non-existent endpoint. Backend expects `/execute` but frontend calls `/generate`. |
| **HouYi Optimization** | `POST /optimize/houyi` | **Missing** | No corresponding router or endpoint found in `router.py` or `main.py`, despite `app/domain/houyi` existing. |
| **Intent-Aware Gen** | `POST /intent-aware/generate` | `POST /api/v1/intent-aware/generate` | **Aligned**, assuming base URL is correct. |
| **Legacy Enhancer** | `POST /api/enhance` | `POST /api/enhance` (in `main.py`) | **Aligned**, but exists outside the `/api/v1` versioned router, creating inconsistency. |
| **Models/Providers** | `GET /providers` | `GET /api/v1/providers` (in `utils.py`) | **Aligned**, assuming base URL is correct. |
| **Statistics** | `GET /metrics` | `GET /api/stats` (in `main.py`) | **Mismatch.** Frontend expects `/metrics` (likely via `api/v1`), backend provides `/api/stats` at root level. |

## 3. Payload Structure Mismatches

There are significant divergences in the data contracts (schemas) between the frontend and backend.

### 3.1 Jailbreak Request Schema
*   **Frontend (`JailbreakRequest`):** Highly granular. Includes fields like `use_leet_speak`, `use_homoglyphs`, `use_role_hijacking`, `use_neural_bypass`, `multilingual_target_language`, etc.
*   **Backend (`TechniqueExecutionRequest` / `JailbreakRequest`):**
    *   `main.py` defines a simple `JailbreakRequest` with only `technique_preference`, `obfuscation_level`, `target_model`.
    *   `jailbreak.py` uses `TechniqueExecutionRequest` (from `models.py`), which likely doesn't support the fine-grained flags sent by the frontend.
*   **Impact:** Frontend configuration options will be ignored or cause validation errors on the backend.

### 3.2 Execution Request Schema
*   **Frontend (`ExecuteRequest`):** Includes `provider`, `use_cache`, `model`, `temperature`, etc.
*   **Backend (`ExecuteRequest`):** Defined in `schemas.py`. Matches most fields but relies on `TechniqueSuite` enum.
*   **Risk:** Enum value mismatches between frontend strings and backend `TechniqueSuite` enum members could cause 422 Validation Errors.

## 4. Infrastructure & Non-Functional Gaps

*   **Caching:** Frontend requests `use_cache`, and backend has `Redis` references in `infrastructure`, but there is no clear evidence of a running Redis service in the provided file list (e.g., no `docker-compose.override.yml` ensuring it's up for dev).
*   **Message Queues:** `GPTFuzz` uses `BackgroundTasks` which is in-memory. For production reliability, a proper task queue (Celery/Redis) is recommended but missing.
*   **Error Logging:** Backend has `app/core/logging.py`, but frontend error handling is generic. No centralized error aggregation (Sentry, etc.) is visible.
*   **Security:**
    *   Frontend sends `X-API-Key` and `Authorization`.
    *   Backend `jailbreak.py` enforces `verify_api_key`.
    *   **Gap:** Not all endpoints enforce authentication. `main.py` endpoints like `/api/enhance` might be open if not protected by middleware.

## 5. Remediation Plan

### Phase 1: Critical API Alignment (Immediate)
1.  **Refactor Backend Routes:**
    *   Move `main.py` legacy routes (`/api/enhance`, etc.) to `app/api/v1/endpoints/legacy.py` or deprecate them.
    *   Ensure all endpoints are under `/api/v1`.
2.  **Fix Jailbreak Endpoint:**
    *   Update Frontend `api-enhanced.ts` to point to `/jailbreak/execute` OR Update Backend `jailbreak.py` to expose `/generate` alias.
    *   **Recommendation:** Update Frontend to match Backend RESTful standard: `POST /api/v1/jailbreak/execute`.
3.  **Implement HouYi Endpoint:**
    *   Create `app/api/v1/endpoints/houyi.py`.
    *   Implement `POST /optimize` endpoint connecting to `HouYiService`.
    *   Register router in `app/api/v1/router.py`.

### Phase 2: Schema Harmonization (Short Term)
1.  **Unified Data Models:**
    *   Create a shared definition or strictly map Frontend `JailbreakRequest` fields to Backend `TechniqueExecutionRequest` parameters.
    *   Update Backend Pydantic models to accept the granular flags (e.g., `use_leet_speak`) as an `options` dictionary or specific fields.
2.  **Enum Synchronization:**
    *   Verify `TechniqueSuite` enum values in `backend-api/app/schemas.py` match exactly with Frontend `types/schemas.ts`.

### Phase 3: Infrastructure Hardening (Medium Term)
1.  **Redis Integration:** Ensure Redis is configured and running. Implement `BoundedCacheManager` fully.
2.  **Async Processing:** Move `GPTFuzz` and long-running Jailbreak tasks to a proper Celery worker queue to avoid timeouts (Frontend timeout is 60s, generation can take longer).

### Phase 4: Security & Observability
1.  **Global Auth Middleware:** Enforce API Key validation globally in `middleware.py` rather than per-endpoint, with specific exclusions for public endpoints (health, docs).
2.  **Structured Logging:** Ensure frontend logs correlation IDs (`X-Request-ID`) returned by backend for end-to-end tracing.

## 6. Technical Recommendations

*   **Frontend:** Update `src/lib/api-enhanced.ts` to align with existing backend routes immediately.
*   **Backend:** Create a `HouYi` router immediately to unblock that feature.
*   **General:** Generate a `openapi.json` from the backend and use a tool like `openapi-typescript-codegen` to generate the frontend client automatically, preventing future drift.