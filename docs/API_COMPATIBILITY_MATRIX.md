# Chimera API Compatibility Matrix

**Version:** 1.0.0  
**Date:** 2026-01-06  
**Status:** Production Reference Document

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully Compatible - No changes required |
| ⚠️ | Partially Compatible - Works with limitations or type mismatches |
| ❌ | Not Compatible - Missing implementation or breaking differences |
| 🔄 | In Progress - Implementation underway |
| ➖ | Not Applicable - No frontend equivalent expected |

### Severity Indicators
| Tag | Impact |
|-----|--------|
| 🔴 BREAKING | Will cause runtime errors - must fix before deployment |
| 🟠 DEGRADED | Functionality works but with reduced capability |
| 🟡 COSMETIC | No runtime impact, but inconsistent behavior |
| 🟢 ALIGNED | Full compatibility confirmed |

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Backend Endpoints** | 106 |
| **Total Frontend API Calls** | 56 |
| **Overall Coverage** | 52.8% |
| **WebSocket Coverage** | 80% (4/5) |
| **SSE Coverage** | 33% (2/6) |
| **Breaking Changes** | 8 |
| **High-Priority Gaps** | 11 |

---

## 1. Coverage Summary Dashboard

| Category | Backend Endpoints | Frontend Coverage | Coverage % | Status | Priority |
|----------|-------------------|-------------------|------------|--------|----------|
| **Health** | 11 | 4 | 36% | ⚠️ | LOW |
| **Authentication** | 0 | 3 | N/A | ❌ | 🔴 CRITICAL |
| **Providers** | 7 + 1 WS | 8 | 100% | ✅ | - |
| **Models** | 2 | 2 | 100% | ✅ | - |
| **Session** | 9 | 5 | 56% | ⚠️ | MEDIUM |
| **Generation** | 2 | 2 | 100% | ✅ | - |
| **Streaming** | 3 | 2 | 67% | ⚠️ | MEDIUM |
| **Transformation** | 4 | 3 | 75% | ⚠️ | LOW |
| **Jailbreak** | 15 + 2 WS | 8 | 47% | ⚠️ | HIGH |
| **AutoDAN** | 4 | 3 | 75% | ⚠️ | MEDIUM |
| **AutoDAN-Turbo** | 19 | 7 | 37% | ❌ | HIGH |
| **DeepTeam** | 14 | 10 | 71% | ⚠️ | MEDIUM |
| **DeepTeam Jailbreak** | 15 + 1 WS | 10 | 67% | ⚠️ | MEDIUM |
| **Admin** | 14 | 0 | 0% | ❌ | HIGH |
| **Metrics** | 11 | 0 | 0% | ❌ | HIGH |
| **Webhook** | 3 | 0 | 0% | ❌ | LOW |
| **Tenant** | 2 | 0 | 0% | ❌ | LOW |

### Visual Coverage Breakdown

```
Health              ████░░░░░░░░░░░░░░░░  36%
Auth                ░░░░░░░░░░░░░░░░░░░░   0% ← CRITICAL GAP
Providers           ████████████████████ 100%
Models              ████████████████████ 100%
Session             ███████████░░░░░░░░░  56%
Generation          ████████████████████ 100%
Streaming           █████████████░░░░░░░  67%
Transformation      ███████████████░░░░░  75%
Jailbreak           █████████░░░░░░░░░░░  47%
AutoDAN             ███████████████░░░░░  75%
AutoDAN-Turbo       ███████░░░░░░░░░░░░░  37%
DeepTeam            ██████████████░░░░░░  71%
DeepTeam Jailbreak  █████████████░░░░░░░  67%
Admin               ░░░░░░░░░░░░░░░░░░░░   0% ← HIGH GAP
Metrics             ░░░░░░░░░░░░░░░░░░░░   0% ← HIGH GAP
```

---

## 2. Detailed Endpoint Matrix

### 2.1 Health Endpoints (`/api/v1/health/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/health` | GET | ✅ | ✅ | ✅ | ❌ | Basic health check |
| `/api/v1/health/ready` | GET | ✅ | ✅ | ✅ | ❌ | Kubernetes readiness |
| `/api/v1/health/live` | GET | ✅ | ✅ | ✅ | ❌ | Kubernetes liveness |
| `/api/v1/health/detailed` | GET | ✅ | ✅ | ✅ | ❌ | Extended health info |
| `/api/v1/health/status` | GET | ❌ | ➖ | ➖ | ❌ | Not implemented in FE |
| `/api/v1/health/redis` | GET | ❌ | ➖ | ➖ | ✅ | Admin only |
| `/api/v1/health/db` | GET | ❌ | ➖ | ➖ | ✅ | Admin only |
| `/api/v1/health/llm` | GET | ❌ | ➖ | ➖ | ✅ | Admin only |
| `/api/v1/health/cache` | GET | ❌ | ➖ | ➖ | ✅ | Admin only |
| `/api/v1/health/workers` | GET | ❌ | ➖ | ➖ | ✅ | Admin only |
| `/api/v1/health/metrics` | GET | ❌ | ➖ | ➖ | ✅ | Prometheus format |

**Coverage:** 4/11 (36%) | **Breaking:** 0 | **Status:** 🟢 Core health checks aligned

---

### 2.2 Authentication Endpoints (`/api/v1/auth/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/auth/login` | POST | ❌ | ❌ | ❌ | ❌ | 🔴 MISSING - FE expects this |
| `/api/v1/auth/refresh` | POST | ❌ | ❌ | ❌ | ✅ | 🔴 MISSING - FE expects this |
| `/api/v1/auth/logout` | POST | ❌ | ❌ | ❌ | ✅ | 🔴 MISSING - FE expects this |

**Coverage:** 0/3 expected (0%) | **Breaking:** 3 | **Status:** 🔴 CRITICAL GAP

> **GAP-001:** Frontend auth client (`authClient.ts`) expects these endpoints but backend has no `/auth` router. Frontend implements JWT refresh logic with no backend support.

---

### 2.3 Provider Endpoints (`/api/v1/providers/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/providers` | GET | ✅ | ✅ | ⚠️ | ❌ | Type mismatch: 4 FE vs 12 BE |
| `/api/v1/providers/{provider_id}` | GET | ✅ | ✅ | ⚠️ | ❌ | Same type issue |
| `/api/v1/providers/{provider_id}/health` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/providers/{provider_id}/models` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/providers/status` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/providers/sync` | POST | ✅ | ✅ | ✅ | ✅ | Admin only |
| `/api/v1/providers/refresh` | POST | ✅ | ✅ | ✅ | ✅ | |
| **WS** `/ws/providers/status` | WS | ✅ | ✅ | ✅ | ❌ | Real-time updates |

**Coverage:** 8/8 (100%) | **Breaking:** 1 | **Status:** ⚠️ Type mismatch needs fix

> **GAP-002:** `ProviderType` enum mismatch
> - Frontend: `openai | anthropic | google | mistral` (4 values)
> - Backend: `openai | anthropic | google | mistral | cohere | deepseek | grok | huggingface | together | groq | azure | aws` (12 values)

---

### 2.4 Model Endpoints (`/api/v1/models/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/models` | GET | ✅ | ✅ | ✅ | ❌ | List all models |
| `/api/v1/models/{model_id}` | GET | ✅ | ✅ | ✅ | ❌ | Get model details |

**Coverage:** 2/2 (100%) | **Breaking:** 0 | **Status:** 🟢 Fully aligned

---

### 2.5 Session Endpoints (`/api/v1/session/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/session` | POST | ✅ | ✅ | ✅ | ✅ | Create session |
| `/api/v1/session/{session_id}` | GET | ✅ | ✅ | ✅ | ✅ | Get session |
| `/api/v1/session/{session_id}` | DELETE | ✅ | ✅ | ✅ | ✅ | Delete session |
| `/api/v1/session/{session_id}/messages` | GET | ✅ | ✅ | ✅ | ✅ | Get messages |
| `/api/v1/session/{session_id}/messages` | POST | ✅ | ✅ | ✅ | ✅ | Add message |
| `/api/v1/session/list` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/session/{session_id}/export` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/session/{session_id}/context` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/session/bulk-delete` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |

**Coverage:** 5/9 (56%) | **Breaking:** 0 | **Status:** 🟢 Core ops aligned

---

### 2.6 Generation Endpoints (`/api/v1/generate/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/generate/prompt` | POST | ✅ | ⚠️ | ✅ | ✅ | See param matrix below |
| `/api/v1/generate/chat` | POST | ✅ | ⚠️ | ✅ | ✅ | See param matrix below |

**Coverage:** 2/2 (100%) | **Breaking:** 0 | **Status:** ⚠️ Minor param differences

---

### 2.7 Streaming Endpoints (`/api/v1/streaming/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/streaming/generate` | POST | ✅ | ✅ | ✅ | ✅ | SSE stream |
| `/api/v1/streaming/chat` | POST | ✅ | ✅ | ✅ | ✅ | SSE stream |
| `/api/v1/streaming/batch` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |

**Coverage:** 2/3 (67%) | **Breaking:** 0 | **Status:** 🟢 Core streaming works

---

### 2.8 Transformation Endpoints (`/api/v1/transform/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/transform/apply` | POST | ✅ | ✅ | ✅ | ✅ | Apply transformation |
| `/api/v1/transform/techniques` | GET | ✅ | ✅ | ✅ | ❌ | List techniques |
| `/api/v1/transform/preview` | POST | ✅ | ✅ | ✅ | ✅ | Preview result |
| `/api/v1/transform/batch` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |

**Coverage:** 3/4 (75%) | **Breaking:** 0 | **Status:** 🟢 Aligned

---

### 2.9 Jailbreak Endpoints (`/api/v1/jailbreak/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/jailbreak/run` | POST | ✅ | ⚠️ | ✅ | ✅ | Main jailbreak |
| `/api/v1/jailbreak/techniques` | GET | ✅ | ✅ | ✅ | ❌ | List techniques |
| `/api/v1/jailbreak/templates` | GET | ✅ | ✅ | ✅ | ❌ | Get templates |
| `/api/v1/jailbreak/templates/{id}` | GET | ✅ | ✅ | ✅ | ❌ | Get template |
| `/api/v1/jailbreak/evaluate` | POST | ✅ | ✅ | ✅ | ✅ | Evaluate result |
| `/api/v1/jailbreak/history` | GET | ✅ | ✅ | ✅ | ✅ | Get history |
| `/api/v1/jailbreak/history/{id}` | GET | ✅ | ✅ | ✅ | ✅ | Get attempt |
| `/api/v1/jailbreak/stats` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/export` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/import` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/analyze` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/compare` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/benchmark` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/report` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/jailbreak/config` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| **WS** `/ws/jailbreak/run` | WS | ⚠️ | ⚠️ | ✅ | ✅ | 🟠 Hardcoded URL |
| **SSE** `/api/v1/jailbreak/stream` | SSE | ✅ | ✅ | ✅ | ✅ | |

**Coverage:** 8/17 (47%) | **Breaking:** 1 | **Status:** ⚠️ WS URL hardcoded

> **GAP-003:** WebSocket URL hardcoded in `jailbreak.ts` line 127:
> ```typescript
> const ws = new WebSocket('ws://localhost:8001/ws/jailbreak/run');
> ```
> Should use `${config.wsBaseUrl}/ws/jailbreak/run`

---

### 2.10 AutoDAN Endpoints (`/api/v1/autodan/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/autodan/run` | POST | ✅ | ✅ | ✅ | ✅ | Start AutoDAN |
| `/api/v1/autodan/status/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | Get status |
| `/api/v1/autodan/cancel/{run_id}` | POST | ✅ | ✅ | ✅ | ✅ | Cancel run |
| `/api/v1/autodan/results/{run_id}` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |

**Coverage:** 3/4 (75%) | **Breaking:** 0 | **Status:** 🟢 Core ops aligned

---

### 2.11 AutoDAN-Turbo Endpoints (`/api/v1/autodan-turbo/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/autodan-turbo/run` | POST | ✅ | ⚠️ | ✅ | ✅ | Missing optional params |
| `/api/v1/autodan-turbo/status/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/autodan-turbo/cancel/{run_id}` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/autodan-turbo/results/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/autodan-turbo/strategies` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/autodan-turbo/strategies/{id}` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/autodan-turbo/strategies` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/autodan-turbo/library` | GET | ❌ | ➖ | ➖ | ❌ | Not in FE |
| `/api/v1/autodan-turbo/library/search` | POST | ❌ | ➖ | ➖ | ❌ | Not in FE |
| `/api/v1/autodan-turbo/library/{id}` | GET | ❌ | ➖ | ➖ | ❌ | Not in FE |
| `/api/v1/autodan-turbo/library/{id}` | PUT | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/library/{id}` | DELETE | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/warmup` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/config` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/config` | PUT | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/metrics` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/export` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/import` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/autodan-turbo/benchmark` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |

**Coverage:** 7/19 (37%) | **Breaking:** 0 | **Status:** ⚠️ Many advanced features missing

---

### 2.12 DeepTeam Endpoints (`/api/v1/deepteam/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/deepteam/run` | POST | ✅ | ✅ | ✅ | ✅ | Start DeepTeam |
| `/api/v1/deepteam/status/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam/cancel/{run_id}` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam/results/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam/attacks` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/deepteam/attacks/{id}` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/deepteam/vulnerabilities` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/deepteam/vulnerabilities/{id}` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/deepteam/report/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam/export/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam/config` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam/config` | PUT | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam/history` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam/stats` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |

**Coverage:** 10/14 (71%) | **Breaking:** 0 | **Status:** 🟢 Well aligned

---

### 2.13 DeepTeam Jailbreak Endpoints (`/api/v1/deepteam-jailbreak/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/deepteam-jailbreak/run` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/status/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/cancel/{run_id}` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/results/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/techniques` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/deepteam-jailbreak/techniques/{id}` | GET | ✅ | ✅ | ✅ | ❌ | |
| `/api/v1/deepteam-jailbreak/evaluate` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/history` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/report/{run_id}` | GET | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/export` | POST | ✅ | ✅ | ✅ | ✅ | |
| `/api/v1/deepteam-jailbreak/batch` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam-jailbreak/compare` | POST | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam-jailbreak/config` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam-jailbreak/config` | PUT | ❌ | ➖ | ➖ | ✅ | Not in FE |
| `/api/v1/deepteam-jailbreak/stats` | GET | ❌ | ➖ | ➖ | ✅ | Not in FE |
| **WS** `/ws/deepteam-jailbreak/run` | WS | ✅ | ✅ | ✅ | ✅ | Real-time updates |

**Coverage:** 10/16 (63%) | **Breaking:** 0 | **Status:** 🟢 Well aligned

---

### 2.14 Admin Endpoints (`/api/v1/admin/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/admin/users` | GET | ❌ | ➖ | ➖ | ✅ | 🟠 No admin panel |
| `/api/v1/admin/users/{id}` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/users/{id}` | PUT | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/users/{id}` | DELETE | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/roles` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/roles` | POST | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/permissions` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/audit-log` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/settings` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/settings` | PUT | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/cache/clear` | POST | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/backup` | POST | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/restore` | POST | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/admin/maintenance` | POST | ❌ | ➖ | ➖ | ✅ | |

**Coverage:** 0/14 (0%) | **Breaking:** 0 | **Status:** 🟠 Admin panel not implemented

> **GAP-004:** No frontend admin panel exists. All admin endpoints inaccessible from UI.

---

### 2.15 Metrics Endpoints (`/api/v1/metrics/*`)

| Endpoint | Method | Frontend | Params | Response | Auth | Notes |
|----------|--------|----------|--------|----------|------|-------|
| `/api/v1/metrics` | GET | ❌ | ➖ | ➖ | ✅ | Prometheus format |
| `/api/v1/metrics/summary` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/usage` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/usage/daily` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/usage/monthly` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/performance` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/errors` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/latency` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/throughput` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/providers` | GET | ❌ | ➖ | ➖ | ✅ | |
| `/api/v1/metrics/export` | GET | ❌ | ➖ | ➖ | ✅ | |

**Coverage:** 0/11 (0%) | **Breaking:** 0 | **Status:** 🟠 Metrics dashboard not implemented

> **GAP-005:** No metrics visualization in frontend. Backend has full Prometheus integration.

---

### 2.16 WebSocket Endpoints Summary

| Endpoint | Backend | Frontend | Params | Messages | Reconnect | Status |
|----------|---------|----------|--------|----------|-----------|--------|
| `/ws/providers/status` | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 |
| `/ws/jailbreak/run` | ✅ | ⚠️ | ✅ | ✅ | ❌ | 🟠 Hardcoded URL |
| `/ws/autodan-turbo/run` | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 |
| `/ws/deepteam/run` | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 |
| `/ws/deepteam-jailbreak/run` | ✅ | ✅ | ✅ | ✅ | ✅ | 🟢 |

**Coverage:** 4/5 working (80%) | **Breaking:** 1 | **Status:** ⚠️ One hardcoded URL

---

### 2.17 SSE Endpoints Summary

| Endpoint | Backend | Frontend | Event Format | Error Handling | Status |
|----------|---------|----------|--------------|----------------|--------|
| `/api/v1/streaming/generate` | ✅ | ✅ | ✅ | ✅ | 🟢 |
| `/api/v1/streaming/chat` | ✅ | ✅ | ✅ | ✅ | 🟢 |
| `/api/v1/streaming/batch` | ✅ | ❌ | ➖ | ➖ | 🟠 |
| `/api/v1/jailbreak/stream` | ✅ | ✅ | ✅ | ✅ | 🟢 |
| `/api/v1/autodan-turbo/stream` | ✅ | ❌ | ➖ | ➖ | 🟠 |
| `/api/v1/deepteam/stream` | ✅ | ❌ | ➖ | ➖ | 🟠 |

**Coverage:** 3/6 (50%) | **Breaking:** 0 | **Status:** 🟠 Some SSE endpoints unused

---

## 3. Parameter Alignment Matrix

### 3.1 POST `/api/v1/generate/prompt`

| Parameter | Backend Type | Required | Frontend Type | Match | Notes |
|-----------|--------------|----------|---------------|-------|-------|
| `prompt` | `str` | Yes | `string` | ✅ | |
| `provider` | `LLMProviderType` | Yes | `ProviderType` | ⚠️ | Enum mismatch |
| `model` | `str \| None` | No | `string?` | ✅ | |
| `temperature` | `float` | No | `number?` | ✅ | Default: 0.7 |
| `max_tokens` | `int \| None` | No | `number?` | ✅ | |
| `top_p` | `float` | No | `number?` | ✅ | Default: 1.0 |
| `frequency_penalty` | `float` | No | `number?` | ✅ | |
| `presence_penalty` | `float` | No | `number?` | ✅ | |
| `stop` | `list[str] \| None` | No | `string[]?` | ✅ | |
| `system_prompt` | `str \| None` | No | `string?` | ✅ | |
| `session_id` | `UUID \| None` | No | `string?` | ✅ | |

**Compatibility:** 10/11 params aligned (91%) | **Breaking:** 1 (provider enum)

---

### 3.2 POST `/api/v1/jailbreak/run`

| Parameter | Backend Type | Required | Frontend Type | Match | Notes |
|-----------|--------------|----------|---------------|-------|-------|
| `prompt` | `str` | Yes | `string` | ✅ | |
| `target_provider` | `LLMProviderType` | Yes | `ProviderType` | ⚠️ | Enum mismatch |
| `target_model` | `str` | Yes | `string` | ✅ | |
| `technique` | `JailbreakTechnique` | Yes | `TechniqueType` | ⚠️ | Enum mismatch |
| `max_iterations` | `int` | No | `number?` | ✅ | Default: 10 |
| `temperature` | `float` | No | `number?` | ✅ | |
| `success_threshold` | `float` | No | `number?` | ✅ | Default: 0.8 |
| `attacker_provider` | `LLMProviderType \| None` | No | ❌ | ❌ | Missing in FE |
| `attacker_model` | `str \| None` | No | ❌ | ❌ | Missing in FE |
| `judge_provider` | `LLMProviderType \| None` | No | ❌ | ❌ | Missing in FE |
| `judge_model` | `str \| None` | No | ❌ | ❌ | Missing in FE |
| `session_id` | `UUID \| None` | No | `string?` | ✅ | |

**Compatibility:** 8/12 params aligned (67%) | **Breaking:** 2 enums | **Missing:** 4 optional

---

### 3.3 POST `/api/v1/autodan-turbo/run`

| Parameter | Backend Type | Required | Frontend Type | Match | Notes |
|-----------|--------------|----------|---------------|-------|-------|
| `target_prompt` | `str` | Yes | `string` | ✅ | |
| `target_provider` | `LLMProviderType` | Yes | `ProviderType` | ⚠️ | |
| `target_model` | `str` | Yes | `string` | ✅ | |
| `attacker_provider` | `LLMProviderType` | No | `ProviderType?` | ⚠️ | |
| `attacker_model` | `str \| None` | No | `string?` | ✅ | |
| `max_iterations` | `int` | No | `number?` | ✅ | Default: 50 |
| `population_size` | `int` | No | ❌ | ❌ | Missing in FE |
| `mutation_rate` | `float` | No | ❌ | ❌ | Missing in FE |
| `crossover_rate` | `float` | No | ❌ | ❌ | Missing in FE |
| `fitness_threshold` | `float` | No | ❌ | ❌ | Missing in FE |
| `strategy_library_id` | `str \| None` | No | ❌ | ❌ | Missing in FE |
| `warm_start` | `bool` | No | ❌ | ❌ | Missing in FE |
| `session_id` | `UUID \| None` | No | `string?` | ✅ | |

**Compatibility:** 7/13 params aligned (54%) | **Breaking:** 2 enums | **Missing:** 6 optional

---

### 3.4 POST `/api/v1/deepteam/run`

| Parameter | Backend Type | Required | Frontend Type | Match | Notes |
|-----------|--------------|----------|---------------|-------|-------|
| `target_description` | `str` | Yes | `string` | ✅ | |
| `target_provider` | `LLMProviderType` | Yes | `ProviderType` | ⚠️ | |
| `target_model` | `str` | Yes | `string` | ✅ | |
| `attack_types` | `list[AttackType]` | No | `AttackType[]?` | ⚠️ | |
| `vulnerability_categories` | `list[VulnCategory]` | No | `VulnCategory[]?` | ⚠️ | |
| `max_attacks` | `int` | No | `number?` | ✅ | Default: 100 |
| `parallel_workers` | `int` | No | ❌ | ❌ | Missing in FE |
| `timeout` | `int` | No | ❌ | ❌ | Missing in FE |
| `session_id` | `UUID \| None` | No | `string?` | ✅ | |

**Compatibility:** 6/9 params aligned (67%) | **Breaking:** 3 enums | **Missing:** 2 optional

---

## 4. Response Schema Alignment

### 4.1 Generation Response

**Backend: `GenerationResponse`**
```python
class GenerationResponse(BaseModel):
    id: UUID
    content: str
    model: str
    provider: LLMProviderType
    usage: TokenUsage
    created_at: datetime
    latency_ms: float
    cached: bool = False
```

**Frontend: `GenerationResult`**
```typescript
interface GenerationResult {
    id: string;
    content: string;
    model: string;
    provider: ProviderType;  // ⚠️ Enum mismatch
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
    created_at: string;
    latency_ms: number;
    cached?: boolean;
}
```

| Field | Backend | Frontend | Match |
|-------|---------|----------|-------|
| `id` | `UUID` | `string` | ✅ |
| `content` | `str` | `string` | ✅ |
| `model` | `str` | `string` | ✅ |
| `provider` | `LLMProviderType` | `ProviderType` | ⚠️ |
| `usage` | `TokenUsage` | inline | ✅ |
| `created_at` | `datetime` | `string` | ✅ |
| `latency_ms` | `float` | `number` | ✅ |
| `cached` | `bool` | `boolean?` | ✅ |

**Match:** 7/8 fields (88%) | **Issue:** Provider enum

---

### 4.2 Jailbreak Result Response

**Backend: `JailbreakResult`**
```python
class JailbreakResult(BaseModel):
    id: UUID
    success: bool
    prompt: str
    response: str
    technique: JailbreakTechnique
    iterations: int
    score: float
    metadata: dict
    created_at: datetime
```

**Frontend: `JailbreakResult`**
```typescript
interface JailbreakResult {
    id: string;
    success: boolean;
    prompt: string;
    response: string;
    technique: TechniqueType;  // ⚠️ Enum mismatch
    iterations: number;
    score: number;
    metadata: Record<string, unknown>;
    created_at: string;
    // Additional FE fields:
    duration_ms?: number;  // ❌ Missing in BE
}
```

| Field | Backend | Frontend | Match |
|-------|---------|----------|-------|
| `id` | `UUID` | `string` | ✅ |
| `success` | `bool` | `boolean` | ✅ |
| `prompt` | `str` | `string` | ✅ |
| `response` | `str` | `string` | ✅ |
| `technique` | `JailbreakTechnique` | `TechniqueType` | ⚠️ |
| `iterations` | `int` | `number` | ✅ |
| `score` | `float` | `number` | ✅ |
| `metadata` | `dict` | `Record<...>` | ✅ |
| `created_at` | `datetime` | `string` | ✅ |
| `duration_ms` | ❌ | `number?` | ❌ FE extra |

**Match:** 8/10 fields (80%) | **Issues:** Enum, extra FE field

---

### 4.3 Provider Status Response

**Backend: `ProviderStatus`**
```python
class ProviderStatus(BaseModel):
    provider: LLMProviderType
    healthy: bool
    latency_ms: float | None
    error: str | None
    models_available: int
    last_check: datetime
```

**Frontend: `ProviderStatus`**
```typescript
interface ProviderStatus {
    provider: ProviderType;  // ⚠️ Only 4 values vs 12
    healthy: boolean;
    latency_ms?: number;
    error?: string;
    models_available: number;
    last_check: string;
}
```

| Field | Backend | Frontend | Match |
|-------|---------|----------|-------|
| `provider` | `LLMProviderType` | `ProviderType` | ⚠️ |
| `healthy` | `bool` | `boolean` | ✅ |
| `latency_ms` | `float \| None` | `number?` | ✅ |
| `error` | `str \| None` | `string?` | ✅ |
| `models_available` | `int` | `number` | ✅ |
| `last_check` | `datetime` | `string` | ✅ |

**Match:** 5/6 fields (83%) | **Issue:** Provider enum (8 values missing)

---

## 5. Enum/Constant Alignment

### 5.1 Provider Type Enum

| Backend `LLMProviderType` | Frontend `ProviderType` | Status |
|---------------------------|-------------------------|--------|
| `openai` | `openai` | ✅ |
| `anthropic` | `anthropic` | ✅ |
| `google` | `google` | ✅ |
| `mistral` | `mistral` | ✅ |
| `cohere` | ❌ | 🔴 Missing |
| `deepseek` | ❌ | 🔴 Missing |
| `grok` | ❌ | 🔴 Missing |
| `huggingface` | ❌ | 🔴 Missing |
| `together` | ❌ | 🔴 Missing |
| `groq` | ❌ | 🔴 Missing |
| `azure` | ❌ | 🔴 Missing |
| `aws` | ❌ | 🔴 Missing |

**Alignment:** 4/12 (33%) | **Status:** 🔴 BREAKING - 8 providers inaccessible from UI

---

### 5.2 Jailbreak Technique Enum

| Backend `JailbreakTechnique` | Frontend `TechniqueType` | Status |
|------------------------------|--------------------------|--------|
| `PAIR` | `pair` | ✅ |
| `GCG` | `gcg` | ✅ |
| `AutoDAN` | `autodan` | ✅ |
| `DeepInception` | `deep_inception` | ✅ |
| `TAP` | `tap` | ✅ |
| `BEAST` | ❌ | 🔴 Missing |
| `CipherChat` | ❌ | 🔴 Missing |
| `MultiLingual` | ❌ | 🔴 Missing |
| `Crescendo` | ❌ | 🔴 Missing |
| `ActorAttack` | ❌ | 🔴 Missing |

**Alignment:** 5/10 (50%) | **Status:** 🔴 BREAKING - 5 techniques inaccessible

---

### 5.3 Attack Type Enum (DeepTeam)

| Backend `AttackType` | Frontend `AttackType` | Status |
|----------------------|-----------------------|--------|
| `PROMPT_INJECTION` | `prompt_injection` | ✅ |
| `JAILBREAK` | `jailbreak` | ✅ |
| `DATA_EXTRACTION` | `data_extraction` | ✅ |
| `HALLUCINATION` | `hallucination` | ✅ |
| `BIAS` | `bias` | ✅ |
| `TOXICITY` | `toxicity` | ✅ |
| `PII_LEAK` | ❌ | 🟠 Missing |
| `MODEL_EXTRACTION` | ❌ | 🟠 Missing |

**Alignment:** 6/8 (75%) | **Status:** 🟠 DEGRADED - 2 attack types unavailable

---

### 5.4 Error Codes

| Backend Error Code | HTTP Status | Frontend Handling | Status |
|--------------------|-------------|-------------------|--------|
| `VALIDATION_ERROR` | 400 | ✅ Parsed | 🟢 |
| `UNAUTHORIZED` | 401 | ❌ No redirect | 🔴 |
| `FORBIDDEN` | 403 | ⚠️ Generic | 🟠 |
| `NOT_FOUND` | 404 | ✅ Handled | 🟢 |
| `RATE_LIMITED` | 429 | ⚠️ No retry | 🟠 |
| `PROVIDER_ERROR` | 502 | ⚠️ Generic | 🟠 |
| `TIMEOUT` | 504 | ⚠️ Generic | 🟠 |
| `INTERNAL_ERROR` | 500 | ✅ Handled | 🟢 |

**Alignment:** 4/8 (50%) | **Status:** 🟠 Some errors not specifically handled

---

## 6. Breaking vs Non-Breaking Changes

### 6.1 Breaking Changes (🔴 8 Total)

| ID | Issue | Impact | Fix Priority |
|----|-------|--------|--------------|
| B-001 | Auth endpoints missing | Login/logout non-functional | P0 |
| B-002 | Provider enum mismatch (8 missing) | 8 providers unusable | P0 |
| B-003 | Technique enum mismatch (5 missing) | 5 techniques unusable | P1 |
| B-004 | WebSocket URL hardcoded | Deployment fails | P0 |
| B-005 | 401 response not handled | No auth redirect | P1 |
| B-006 | JWT refresh not implemented | Sessions expire unexpectedly | P1 |
| B-007 | Attack type enum incomplete | 2 attack types unavailable | P2 |
| B-008 | Admin endpoints 0% coverage | No admin functionality | P2 |

---

### 6.2 Non-Breaking (Degraded) Changes (🟠 11 Total)

| ID | Issue | Impact | Fix Priority |
|----|-------|--------|--------------|
| D-001 | Metrics endpoints 0% coverage | No usage visibility | P2 |
| D-002 | Missing jailbreak optional params | Less control | P3 |
| D-003 | Missing AutoDAN-Turbo params | Suboptimal runs | P3 |
| D-004 | Session export not in FE | Manual export only | P3 |
| D-005 | Batch streaming unused | Single requests only | P3 |
| D-006 | Rate limit errors generic | Poor UX on limit | P3 |
| D-007 | Provider errors generic | Hard to debug | P3 |
| D-008 | SSE endpoints underutilized | Missing real-time | P3 |
| D-009 | Health deep checks unused | No DB/Redis status | P3 |
| D-010 | Webhook endpoints unused | No integrations | P4 |
| D-011 | Tenant endpoints unused | No multi-tenancy | P4 |

---

### 6.3 Cosmetic Issues (🟡 4 Total)

| ID | Issue | Impact |
|----|-------|--------|
| C-001 | Extra FE field `duration_ms` | Ignored by BE |
| C-002 | DateTime format differences | Auto-converted |
| C-003 | Case style differences | Compatible |
| C-004 | Optional field ordering | No impact |

---

## 7. Quick Reference Card

### ✅ Fully Compatible - Safe to Use

| Category | Endpoints |
|----------|-----------|
| **Health** | `/health`, `/health/ready`, `/health/live`, `/health/detailed` |
| **Models** | `/models`, `/models/{id}` |
| **Generation** | `/generate/prompt`, `/generate/chat` |
| **Streaming** | `/streaming/generate`, `/streaming/chat` |
| **Providers** | All 8 endpoints + WebSocket |
| **Session** | Core CRUD operations |
| **Transformation** | `/transform/apply`, `/transform/techniques`, `/transform/preview` |
| **DeepTeam** | Run, status, cancel, results, attacks, vulnerabilities |

### ⚠️ Partially Compatible - Use with Caution

| Category | Issue | Workaround |
|----------|-------|------------|
| **Jailbreak** | WS URL hardcoded, enum mismatch | Use SSE endpoint, limit to 5 techniques |
| **AutoDAN** | Missing advanced params | Use defaults |
| **AutoDAN-Turbo** | Missing 6 optional params | Basic runs only |
| **DeepTeam Jailbreak** | Enum partial coverage | Use 5 available techniques |

### ❌ Not Compatible - Do Not Use

| Category | Reason | Required Action |
|----------|--------|-----------------|
| **Authentication** | Endpoints don't exist | Build backend auth router |
| **Admin** | No frontend implementation | Build admin panel |
| **Metrics** | No frontend implementation | Build metrics dashboard |
| **Webhook** | No frontend implementation | Not critical |
| **Extended Providers** | Enum mismatch | Update FE enum |
| **Advanced Techniques** | Enum mismatch | Update FE enum |

---

## 8. Implementation Priority Matrix

| Priority | Items | Effort | Impact |
|----------|-------|--------|--------|
| **P0 - Critical** | Auth endpoints, Provider enum, WS URL | High | System functional |
| **P1 - High** | Technique enum, 401 handling, JWT refresh | Medium | Core features |
| **P2 - Medium** | Admin panel, Metrics dashboard, Attack types | High | Operations |
| **P3 - Low** | Optional params, SSE endpoints, Export | Low | Enhancement |
| **P4 - Backlog** | Webhook, Tenant, Batch operations | Medium | Future |

---

## Appendix A: Full Endpoint Count

| Router | REST | WebSocket | SSE | Total |
|--------|------|-----------|-----|-------|
| Health | 11 | 0 | 0 | 11 |
| Auth | 0 | 0 | 0 | 0 |
| Providers | 7 | 1 | 0 | 8 |
| Models | 2 | 0 | 0 | 2 |
| Session | 9 | 0 | 0 | 9 |
| Generation | 2 | 0 | 0 | 2 |
| Streaming | 3 | 0 | 3 | 6 |
| Transformation | 4 | 0 | 0 | 4 |
| Jailbreak | 15 | 2 | 1 | 18 |
| AutoDAN | 4 | 0 | 0 | 4 |
| AutoDAN-Turbo | 19 | 1 | 1 | 21 |
| DeepTeam | 14 | 1 | 1 | 16 |
| DeepTeam Jailbreak | 15 | 1 | 0 | 16 |
| Admin | 14 | 0 | 0 | 14 |
| Metrics | 11 | 0 | 0 | 11 |
| Webhook | 3 | 0 | 0 | 3 |
| Tenant | 2 | 0 | 0 | 2 |
| **TOTAL** | **135** | **6** | **6** | **147** |

---

## Appendix B: Cross-Reference to Gap Analysis

| Gap ID | Matrix Section | Status |
|--------|----------------|--------|
| GAP-001 | Section 2.2 | 🔴 Breaking |
| GAP-002 | Section 5.1 | 🔴 Breaking |
| GAP-003 | Section 2.9 | 🔴 Breaking |
| GAP-004 | Section 2.14 | 🟠 Degraded |
| GAP-005 | Section 2.15 | 🟠 Degraded |
| GAP-006-021 | Various | See Gap Analysis Report |

---

*Generated: 2026-01-06 | Document Version: 1.0.0*