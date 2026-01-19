# Chimera Integration Testing Checklist

| Document Info | |
|---------------|---|
| **Version** | 1.0.0 |
| **Date** | 2026-01-06 |
| **Status** | ACTIVE |
| **Source Documents** | GAP_ANALYSIS_REPORT.md, REMEDIATION_ROADMAP.md, API_COMPATIBILITY_MATRIX.md |
| **Total Gaps Covered** | 21 (3 CRITICAL, 8 HIGH, 6 MEDIUM, 4 LOW) |
| **Target Coverage** | 85%+ |

---

## Table of Contents

1. [Test Environment Setup](#1-test-environment-setup)
2. [Prerequisites](#2-prerequisites)
3. [Phase 0 Tests: Deploy Blockers](#3-phase-0-tests-deploy-blockers)
4. [Phase 1 Tests: Authentication](#4-phase-1-tests-authentication)
5. [Phase 2 Tests: Provider Alignment](#5-phase-2-tests-provider-alignment)
6. [Phase 3 Tests: Admin & Metrics](#6-phase-3-tests-admin--metrics)
7. [Integration Tests](#7-integration-tests)
8. [Error Handling Tests](#8-error-handling-tests)
9. [Regression Tests](#9-regression-tests)
10. [Performance Tests](#10-performance-tests)
11. [Security Tests](#11-security-tests)
12. [Test Summary Template](#12-test-summary-template)
13. [Automation Commands](#13-automation-commands)
14. [Sign-off Checklist](#14-sign-off-checklist)

---

## 1. Test Environment Setup

### Required Environment Variables

Create or update your `.env` files with the following configuration:

**Frontend ([`frontend/.env.local`](../frontend/.env.local)):**
```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8001/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8001

# Auth Configuration (for testing)
NEXT_PUBLIC_FEATURE_AUTH=true
```

**Backend ([`backend-api/.env`](../backend-api/.env)):**
```bash
# Admin Credentials (for testing)
CHIMERA_ADMIN_USER=admin
CHIMERA_ADMIN_PASSWORD=test-password-123

# JWT Configuration
JWT_SECRET_KEY=test-secret-key-for-development-only
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Start Services

```bash
# Terminal 1: Start backend
npm run dev:backend
# Or: cd backend-api && poetry run chimera-dev

# Terminal 2: Start frontend  
npm run dev:frontend
# Or: cd frontend && npm run dev

# Or start both together:
npm run dev
```

### Verify Services Running

```bash
# Check backend health
curl http://localhost:8001/api/v1/health

# Check frontend
curl http://localhost:3001

# Check ports
node scripts/check-ports.js
```

---

## 2. Prerequisites

Before running tests, ensure:

- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] Poetry installed and dependencies installed (`poetry install`)
- [ ] NPM dependencies installed (`npm run install:all`)
- [ ] Backend service running on port 8001
- [ ] Frontend service running on port 3000
- [ ] Environment variables configured as above
- [ ] Database migrations applied (if applicable)

---

## 3. Phase 0 Tests: Deploy Blockers

**Priority:** CRITICAL - Must pass before ANY deployment  
**Gaps Addressed:** GAP-003, GAP-001 (decision)

### TC-001: WebSocket URL Configuration (GAP-003)

**Objective:** Verify WebSocket URL is dynamically configured and not hardcoded.

**Pre-conditions:**
- Frontend application is running
- Backend WebSocket endpoint is available

#### Test Cases:

- [ ] **TC-001.1:** WS connects in development (localhost)
  - **Steps:**
    1. Start frontend on localhost:3001
    2. Navigate to jailbreak page
    3. Open browser DevTools → Network → WS
    4. Verify WebSocket connects to `ws://localhost:8001`
  - **Expected:** Connection established successfully
  - **Actual:** ________________

- [ ] **TC-001.2:** WS connects in staging (non-localhost)
  - **Steps:**
    1. Set `NEXT_PUBLIC_WS_URL=wss://staging.example.com`
    2. Build and deploy frontend
    3. Open browser DevTools → Network → WS
    4. Verify WebSocket connects to staging URL
  - **Expected:** Connection to `wss://staging.example.com`
  - **Actual:** ________________

- [ ] **TC-001.3:** WS URL falls back correctly if env var missing
  - **Steps:**
    1. Remove `NEXT_PUBLIC_WS_URL` from environment
    2. Restart frontend
    3. Navigate to WebSocket-using page
    4. Check connection URL in DevTools
  - **Expected:** Falls back to deriving WS URL from `window.location`
  - **Actual:** ________________

- [ ] **TC-001.4:** WSS (secure) used when page is HTTPS
  - **Steps:**
    1. Access frontend via HTTPS (e.g., https://localhost:3001)
    2. Open browser DevTools → Network → WS
    3. Verify protocol is `wss://` not `ws://`
  - **Expected:** Secure WebSocket connection (`wss://`)
  - **Actual:** ________________

**Validation Commands:**

```bash
# Manual test - Open browser console and check:
console.log('WS URL:', process.env.NEXT_PUBLIC_WS_URL);

# Check for hardcoded localhost in source
grep -r "ws://localhost" frontend/src/

# Automated test (when available)
npx playwright test --grep "websocket"
```

**Source Files to Verify:**
- [`frontend/src/api/jailbreak.ts`](../frontend/src/api/jailbreak.ts:18-19)
- [`frontend/src/lib/websocket-manager.ts`](../frontend/src/lib/websocket-manager.ts)

---

### TC-002: Auth Strategy Decision Verification

**Objective:** Verify authentication strategy has been decided and documented.

- [ ] **TC-002.1:** Auth strategy documented
  - **Steps:**
    1. Review REMEDIATION_ROADMAP.md Task 0.2
    2. Confirm decision recorded (Option A: Backend Auth OR Option B: External IdP)
  - **Expected:** Decision documented with rationale
  - **Actual:** ________________

---

## 4. Phase 1 Tests: Authentication

**Priority:** CRITICAL/HIGH  
**Gaps Addressed:** GAP-001, GAP-004, GAP-009, GAP-008

### TC-003: Login Endpoint (GAP-001)

**Objective:** Verify `/api/v1/auth/login` endpoint works correctly.

#### Test Cases:

- [ ] **TC-003.1:** POST /api/v1/auth/login returns 200 with valid credentials
  - **Steps:**
    1. Send POST request with valid username/password
    2. Verify 200 status code returned
    3. Verify response body structure
  - **Expected:** 200 OK with token response
  - **Actual:** ________________

- [ ] **TC-003.2:** POST /api/v1/auth/login returns 401 with invalid credentials
  - **Steps:**
    1. Send POST request with invalid password
    2. Verify 401 status code
    3. Verify error message is generic (no credential hints)
  - **Expected:** 401 Unauthorized
  - **Actual:** ________________

- [ ] **TC-003.3:** Response includes all required token fields
  - **Steps:**
    1. Perform successful login
    2. Verify response includes: `access_token`, `refresh_token`, `token_type`, `expires_in`, `refresh_expires_in`
  - **Expected:** All 5 fields present
  - **Actual:** ________________

- [ ] **TC-003.4:** `token_type` is `Bearer` (capital B) - GAP-009
  - **Steps:**
    1. Perform successful login
    2. Check `token_type` field value
  - **Expected:** `token_type: "Bearer"` (capital B)
  - **Actual:** ________________

- [ ] **TC-003.5:** `refresh_expires_in` field present - GAP-004
  - **Steps:**
    1. Perform successful login
    2. Verify `refresh_expires_in` is a positive integer (seconds)
  - **Expected:** `refresh_expires_in: 604800` (7 days default)
  - **Actual:** ________________

**Validation Command:**

```bash
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"test-password-123"}'
```

**Expected Response:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_expires_in": 604800,
  "user": {
    "id": "admin",
    "username": "admin",
    "role": "admin"
  }
}
```

---

### TC-004: Token Refresh Endpoint (GAP-001)

**Objective:** Verify `/api/v1/auth/refresh` endpoint works correctly.

- [ ] **TC-004.1:** POST /api/v1/auth/refresh returns new tokens
  - **Steps:**
    1. Login to get refresh token
    2. Send POST with refresh token
    3. Verify new access_token and refresh_token returned
  - **Expected:** 200 OK with new tokens
  - **Actual:** ________________

- [ ] **TC-004.2:** Expired refresh token returns 401
  - **Steps:**
    1. Use an expired refresh token
    2. Send POST request
    3. Verify 401 status
  - **Expected:** 401 Unauthorized with "expired" message
  - **Actual:** ________________

- [ ] **TC-004.3:** Invalid refresh token returns 401
  - **Steps:**
    1. Send POST with malformed/invalid token
    2. Verify 401 status
  - **Expected:** 401 Unauthorized
  - **Actual:** ________________

**Validation Command:**

```bash
# First login to get refresh token
TOKEN=$(curl -s -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"test-password-123"}' | jq -r '.refresh_token')

# Then refresh
curl -X POST http://localhost:8001/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\":\"$TOKEN\"}"
```

---

### TC-005: Logout Endpoint (GAP-001)

**Objective:** Verify `/api/v1/auth/logout` endpoint revokes tokens.

- [ ] **TC-005.1:** POST /api/v1/auth/logout revokes token
  - **Steps:**
    1. Login to get access token
    2. Call logout with access token
    3. Verify 204 No Content
  - **Expected:** 204 No Content
  - **Actual:** ________________

- [ ] **TC-005.2:** Subsequent requests with revoked token return 401
  - **Steps:**
    1. Use the revoked token from TC-005.1
    2. Make an authenticated request
    3. Verify 401 returned
  - **Expected:** 401 Unauthorized
  - **Actual:** ________________

**Validation Command:**

```bash
# Login and get token
ACCESS_TOKEN=$(curl -s -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"test-password-123"}' | jq -r '.access_token')

# Logout
curl -X POST http://localhost:8001/api/v1/auth/logout \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# Try to use revoked token (should fail)
curl http://localhost:8001/api/v1/auth/me \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

---

### TC-006: Frontend Auth Flow

**Objective:** Verify frontend authentication integration works end-to-end.

- [ ] **TC-006.1:** Login form submits correctly
  - **Steps:**
    1. Navigate to login page
    2. Enter credentials
    3. Submit form
    4. Verify redirect to dashboard
  - **Expected:** Successful login and redirect
  - **Actual:** ________________

- [ ] **TC-006.2:** Tokens stored in AuthManager
  - **Steps:**
    1. After login, open DevTools → Application → Storage
    2. Check for stored tokens (localStorage/sessionStorage)
  - **Expected:** Tokens stored securely
  - **Actual:** ________________

- [ ] **TC-006.3:** Bearer token sent on authenticated requests
  - **Steps:**
    1. Make any authenticated API call
    2. Check Network tab for Authorization header
  - **Expected:** `Authorization: Bearer <token>` header present
  - **Actual:** ________________

- [ ] **TC-006.4:** Auto-refresh occurs before token expiry
  - **Steps:**
    1. Login with short-lived token
    2. Wait until ~5 minutes before expiry
    3. Verify refresh request made
  - **Expected:** Token refreshed automatically
  - **Actual:** ________________

- [ ] **TC-006.5:** Logout clears tokens
  - **Steps:**
    1. Click logout button
    2. Verify tokens removed from storage
    3. Verify redirect to login page
  - **Expected:** All auth state cleared
  - **Actual:** ________________

---

### TC-007: Error Classes Implementation (GAP-008)

**Objective:** Verify all 7 new error classes are implemented.

- [ ] **TC-007.1:** RateLimitBudgetExceeded class exists
- [ ] **TC-007.2:** StreamingError class exists
- [ ] **TC-007.3:** WebSocketError class exists
- [ ] **TC-007.4:** SessionExpiredError class exists
- [ ] **TC-007.5:** ProviderQuotaExceeded class exists
- [ ] **TC-007.6:** ModelNotAvailableError class exists
- [ ] **TC-007.7:** TransformationTimeoutError class exists

**Validation Command:**

```bash
# Check error classes exist in frontend
grep -r "class RateLimitBudgetExceeded" frontend/src/
grep -r "class StreamingError" frontend/src/
grep -r "class WebSocketError" frontend/src/
grep -r "class SessionExpiredError" frontend/src/
grep -r "class ProviderQuotaExceeded" frontend/src/
grep -r "class ModelNotAvailableError" frontend/src/
grep -r "class TransformationTimeoutError" frontend/src/
```

---

## 5. Phase 2 Tests: Provider Alignment

**Priority:** HIGH/MEDIUM  
**Gaps Addressed:** GAP-002, GAP-010, GAP-007

### TC-008: Extended Provider Types (GAP-002)

**Objective:** Verify all 12 provider types work in frontend.

#### Provider Type Tests:

| Provider | Checkbox | Test Result | Notes |
|----------|----------|-------------|-------|
| openai | [ ] | | |
| anthropic | [ ] | | |
| gemini | [ ] | | |
| deepseek | [ ] | | |
| google | [ ] | | |
| qwen | [ ] | | |
| gemini-cli | [ ] | | |
| antigravity | [ ] | | |
| kiro | [ ] | | |
| cursor | [ ] | | |
| xai | [ ] | | |
| mock | [ ] | | |

**Test Steps for Each Provider:**

1. Navigate to provider configuration page
2. Select provider from dropdown
3. Verify provider appears in list
4. Save configuration
5. Verify TypeScript compiles without errors

**Validation Commands:**

```bash
# Check TypeScript types include all providers
grep -A 20 "type ProviderType" frontend/src/lib/api/types.ts

# Verify no TypeScript errors
cd frontend && npm run type-check
```

---

### TC-009: Technique Enum Alignment (GAP-010)

**Objective:** Verify all backend techniques are available in frontend.

#### Technique Enum Tests:

**Core Techniques:**
- [ ] role_play
- [ ] authority_impersonation
- [ ] hypothetical
- [ ] code_injection
- [ ] token_manipulation
- [ ] encoding

**Context Techniques:**
- [ ] context_confusion
- [ ] context_overflow
- [ ] translation

**Advanced Techniques:**
- [ ] autodan
- [ ] autodan_turbo
- [ ] gptfuzz
- [ ] pair
- [ ] tap
- [ ] crescendo

**Cipher Techniques:**
- [ ] cipher
- [ ] base64
- [ ] rot13

**Meta Techniques:**
- [ ] metamorph
- [ ] ensemble
- [ ] hybrid

**DeepTeam Techniques:**
- [ ] deep_inception
- [ ] code_chameleon
- [ ] quantum_exploit

**Validation Command:**

```bash
# Check technique types in frontend
grep -A 50 "type JailbreakTechnique" frontend/src/lib/api/types.ts
```

---

### TC-010: Deprecated API Client (GAP-007)

**Objective:** Verify old API client is properly deprecated.

- [ ] **TC-010.1:** No imports from `api-enhanced.ts` in new code
  - **Steps:**
    1. Search for imports from api-enhanced.ts
    2. Verify only legacy files import it
  - **Expected:** No new imports
  - **Actual:** ________________

- [ ] **TC-010.2:** Console shows deprecation warnings (if enabled)
  - **Steps:**
    1. Open any page that uses deprecated client
    2. Check browser console for warnings
  - **Expected:** Deprecation warning displayed
  - **Actual:** ________________

**Validation Command:**

```bash
# Find all files importing deprecated client
grep -r "from.*api-enhanced" frontend/src/ --include="*.ts" --include="*.tsx"
```

---

## 6. Phase 3 Tests: Admin & Metrics

**Priority:** HIGH/MEDIUM  
**Gaps Addressed:** GAP-005, GAP-006, GAP-008 (completion)

### TC-011: Admin Dashboard (GAP-005)

**Objective:** Verify admin dashboard is implemented and functional.

- [ ] **TC-011.1:** Admin page loads at /dashboard/admin
  - **Steps:**
    1. Login as admin user
    2. Navigate to /dashboard/admin
    3. Verify page loads without errors
  - **Expected:** Admin dashboard renders
  - **Actual:** ________________

- [ ] **TC-011.2:** User management functions work
  - **Steps:**
    1. View user list
    2. Create new user
    3. Edit user
    4. Delete user (test account)
  - **Expected:** All CRUD operations succeed
  - **Actual:** ________________

- [ ] **TC-011.3:** Role assignment works
  - **Steps:**
    1. Select a user
    2. Change their role
    3. Verify role persists after refresh
  - **Expected:** Role changes saved
  - **Actual:** ________________

- [ ] **TC-011.4:** System configuration editable
  - **Steps:**
    1. Access system settings
    2. Modify a setting
    3. Save changes
    4. Verify persistence
  - **Expected:** Settings saved successfully
  - **Actual:** ________________

- [ ] **TC-011.5:** RBAC prevents unauthorized access
  - **Steps:**
    1. Login as non-admin user
    2. Try to access /dashboard/admin
    3. Verify access denied
  - **Expected:** Redirect to unauthorized page
  - **Actual:** ________________

---

### TC-012: Metrics Dashboard (GAP-006)

**Objective:** Verify metrics dashboard is implemented and displays data.

- [ ] **TC-012.1:** Metrics page loads at /dashboard/metrics
  - **Steps:**
    1. Navigate to /dashboard/metrics
    2. Verify page loads
  - **Expected:** Metrics dashboard renders
  - **Actual:** ________________

- [ ] **TC-012.2:** Real-time metrics update
  - **Steps:**
    1. Generate some API traffic
    2. Observe metrics updating
  - **Expected:** Metrics refresh in real-time
  - **Actual:** ________________

- [ ] **TC-012.3:** Historical data displays
  - **Steps:**
    1. Select historical time range
    2. Verify data loads
  - **Expected:** Historical metrics displayed
  - **Actual:** ________________

- [ ] **TC-012.4:** Charts render correctly
  - **Steps:**
    1. Verify line/bar charts render
    2. Check for visual errors
    3. Test responsive behavior
  - **Expected:** Charts display correctly
  - **Actual:** ________________

---

## 7. Integration Tests

### TC-013: End-to-End Generation Flow

**Objective:** Verify complete generation workflow.

- [ ] **TC-013.1:** User can log in
- [ ] **TC-013.2:** User can select provider
- [ ] **TC-013.3:** User can select model
- [ ] **TC-013.4:** User can submit prompt
- [ ] **TC-013.5:** Response streams correctly
- [ ] **TC-013.6:** Response displays in UI

**Playwright Test:**

```typescript
// tests/e2e/generation-flow.spec.ts
import { test, expect } from '@playwright/test';

test('generation flow', async ({ page }) => {
  // Login
  await page.goto('/login');
  await page.fill('[data-testid="username-input"]', 'admin');
  await page.fill('[data-testid="password-input"]', 'test-password-123');
  await page.click('[data-testid="login-button"]');
  
  // Wait for dashboard
  await expect(page).toHaveURL(/dashboard/);
  
  // Enter prompt
  await page.fill('[data-testid="prompt-input"]', 'Test prompt');
  
  // Submit
  await page.click('[data-testid="generate-button"]');
  
  // Wait for response
  await expect(page.locator('[data-testid="response"]')).toBeVisible({
    timeout: 30000
  });
});
```

---

### TC-014: End-to-End Jailbreak Flow

**Objective:** Verify complete jailbreak workflow with WebSocket.

- [ ] **TC-014.1:** User can access jailbreak page
- [ ] **TC-014.2:** WebSocket connects successfully
- [ ] **TC-014.3:** Jailbreak generation starts
- [ ] **TC-014.4:** Streaming updates display
- [ ] **TC-014.5:** Final result shows

**Playwright Test:**

```typescript
// tests/e2e/jailbreak-flow.spec.ts
import { test, expect } from '@playwright/test';

test('jailbreak flow', async ({ page }) => {
  // Login first
  await page.goto('/login');
  await page.fill('[data-testid="username-input"]', 'admin');
  await page.fill('[data-testid="password-input"]', 'test-password-123');
  await page.click('[data-testid="login-button"]');
  
  // Navigate to jailbreak
  await page.goto('/dashboard/jailbreak');
  
  // Configure jailbreak
  await page.selectOption('[data-testid="technique-select"]', 'pair');
  await page.fill('[data-testid="target-prompt"]', 'Test target');
  
  // Start jailbreak
  await page.click('[data-testid="start-jailbreak"]');
  
  // Verify WebSocket connection indicator
  await expect(page.locator('[data-testid="ws-status"]')).toHaveText('Connected');
  
  // Wait for streaming updates
  await expect(page.locator('[data-testid="stream-output"]')).toBeVisible();
  
  // Wait for completion
  await expect(page.locator('[data-testid="result-score"]')).toBeVisible({
    timeout: 60000
  });
});
```

---

## 8. Error Handling Tests

### TC-015: Error Class Mapping

**Objective:** Verify frontend correctly handles all error types.

#### For each error class:

| Error Type | Test | Expected UI Behavior | Status |
|------------|------|---------------------|--------|
| RateLimitBudgetExceeded | [ ] | Toast + retry suggestion | |
| StreamingError | [ ] | Reconnect prompt | |
| WebSocketError | [ ] | Reconnect logic + indicator | |
| ModelNotFoundError | [ ] | Model selection prompt | |
| ProviderUnavailableError | [ ] | Fallback suggestion | |
| ConfigurationError | [ ] | Admin notification | |
| QuotaExceededError | [ ] | Upgrade prompt | |

---

### TC-016: HTTP Status Code Handling

**Objective:** Verify frontend handles all HTTP error codes correctly.

| Status Code | Test | Expected Behavior | Status |
|-------------|------|-------------------|--------|
| 400 Bad Request | [ ] | Validation error message | |
| 401 Unauthorized | [ ] | Redirect to login | |
| 403 Forbidden | [ ] | Permission denied message | |
| 404 Not Found | [ ] | Appropriate error | |
| 429 Too Many Requests | [ ] | Rate limit message + retry-after | |
| 500 Internal Server Error | [ ] | Generic error + support link | |

**Validation Commands:**

```bash
# Test 400 Bad Request
curl -X POST http://localhost:8001/api/v1/generate/prompt \
  -H "Content-Type: application/json" \
  -d '{"invalid": "body"}'

# Test 401 Unauthorized
curl http://localhost:8001/api/v1/auth/me

# Test 404 Not Found
curl http://localhost:8001/api/v1/nonexistent

# Test 429 Rate Limit (if rate limiting enabled)
for i in {1..100}; do
  curl http://localhost:8001/api/v1/health &
done
```

---

## 9. Regression Tests

### TC-017: Existing Functionality

**Objective:** Verify no regressions in existing features.

- [ ] **TC-017.1:** Health check endpoints still work
  - **Command:** `curl http://localhost:8001/api/v1/health`
  - **Expected:** `{"status": "healthy"}`
  - **Actual:** ________________

- [ ] **TC-017.2:** Existing provider configs unchanged
  - **Steps:**
    1. List existing providers
    2. Verify configurations intact
  - **Expected:** All configs preserved
  - **Actual:** ________________

- [ ] **TC-017.3:** Existing session management works
  - **Steps:**
    1. Create session
    2. Add messages
    3. Retrieve session
    4. Delete session
  - **Expected:** All CRUD operations work
  - **Actual:** ________________

- [ ] **TC-017.4:** Existing transformations work
  - **Steps:**
    1. Apply a transformation
    2. Verify result
  - **Expected:** Transformation succeeds
  - **Actual:** ________________

- [ ] **TC-017.5:** Existing streaming works
  - **Steps:**
    1. Make streaming request
    2. Verify SSE events received
  - **Expected:** Streaming functions correctly
  - **Actual:** ________________

---

## 10. Performance Tests

### TC-018: Authentication Performance

- [ ] **TC-018.1:** Login responds in < 500ms
  - **Command:**
    ```bash
    time curl -X POST http://localhost:8001/api/v1/auth/login \
      -H "Content-Type: application/json" \
      -d '{"username":"admin","password":"test-password-123"}'
    ```
  - **Expected:** < 500ms
  - **Actual:** ________ ms

- [ ] **TC-018.2:** Token refresh responds in < 200ms
  - **Expected:** < 200ms
  - **Actual:** ________ ms

- [ ] **TC-018.3:** No memory leaks in auth token refresh cycle
  - **Steps:**
    1. Run refresh cycle 100 times
    2. Monitor memory usage
    3. Verify stable memory
  - **Expected:** No memory growth
  - **Actual:** ________________

---

### TC-019: WebSocket Performance

- [ ] **TC-019.1:** WS connection established in < 1s
  - **Expected:** < 1000ms
  - **Actual:** ________ ms

- [ ] **TC-019.2:** Message latency < 100ms
  - **Expected:** < 100ms
  - **Actual:** ________ ms

- [ ] **TC-019.3:** Reconnection occurs within 5s of disconnect
  - **Steps:**
    1. Establish WS connection
    2. Kill backend server
    3. Measure reconnection time after restart
  - **Expected:** < 5000ms
  - **Actual:** ________ ms

---

## 11. Security Tests

### TC-020: Auth Security

- [ ] **TC-020.1:** JWT tokens expire correctly
  - **Steps:**
    1. Login to get token
    2. Wait for expiry time
    3. Verify token rejected
  - **Expected:** Token expires as configured
  - **Actual:** ________________

- [ ] **TC-020.2:** Revoked tokens cannot be reused
  - **Steps:**
    1. Logout to revoke token
    2. Try to use revoked token
    3. Verify rejection
  - **Expected:** 401 Unauthorized
  - **Actual:** ________________

- [ ] **TC-020.3:** Password not logged or exposed
  - **Steps:**
    1. Search logs for password content
    2. Check network responses for password
  - **Expected:** No password exposure
  - **Actual:** ________________

- [ ] **TC-020.4:** HTTPS enforced for auth endpoints in production
  - **Steps:**
    1. In production config, try HTTP
    2. Verify redirect to HTTPS or rejection
  - **Expected:** HTTPS required
  - **Actual:** ________________

---

### TC-021: API Security

- [ ] **TC-021.1:** Unauthenticated requests rejected for protected endpoints
  - **Command:**
    ```bash
    curl http://localhost:8001/api/v1/admin/users
    ```
  - **Expected:** 401 Unauthorized
  - **Actual:** ________________

- [ ] **TC-021.2:** CORS configured correctly
  - **Steps:**
    1. Make cross-origin request
    2. Verify CORS headers present
    3. Verify unauthorized origins rejected
  - **Expected:** Proper CORS headers
  - **Actual:** ________________

- [ ] **TC-021.3:** Rate limiting works
  - **Steps:**
    1. Make many rapid requests
    2. Verify 429 returned after limit
  - **Expected:** Rate limit enforced
  - **Actual:** ________________

---

## 12. Test Summary Template

Complete this summary after running all tests:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST EXECUTION SUMMARY                        │
├──────────────────┬──────────┬──────────┬──────────┬─────────────┤
│ Phase            │ Total    │ Passed   │ Failed   │ Skipped     │
├──────────────────┼──────────┼──────────┼──────────┼─────────────┤
│ Phase 0          │    5     │          │          │             │
│ Phase 1          │   18     │          │          │             │
│ Phase 2          │   17     │          │          │             │
│ Phase 3          │    9     │          │          │             │
│ Integration      │   11     │          │          │             │
│ Error Handling   │   13     │          │          │             │
│ Regression       │    5     │          │          │             │
│ Performance      │    6     │          │          │             │
│ Security         │    7     │          │          │             │
├──────────────────┼──────────┼──────────┼──────────┼─────────────┤
│ TOTAL            │   91     │          │          │             │
└──────────────────┴──────────┴──────────┴──────────┴─────────────┘

Test Date: ________________
Tester: ________________
Environment: ________________
Build Version: ________________
Backend Version: ________________
Frontend Version: ________________

Notes:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

## 13. Automation Commands

### Backend Tests

```bash
# Run all backend tests
cd backend-api && poetry run pytest --cov

# Run with coverage report
poetry run pytest --cov backend-api/app --cov meta_prompter --cov-report=html

# Run specific test file
poetry run pytest tests/test_auth_router.py -v

# Run tests matching pattern
poetry run pytest -k "auth" -v
```

### Frontend Tests

```bash
# Run frontend unit tests
cd frontend && npm run test
# Or: npx vitest --run

# Run with coverage
npx vitest --coverage

# Run specific test file
npx vitest src/__tests__/lib/errors.test.ts
```

### Playwright E2E Tests

```bash
# Run all Playwright tests
npx playwright test

# Run specific gap validation tests
npx playwright test --grep "GAP-"

# Run with UI mode (interactive)
npx playwright test --ui

# Generate test report
npx playwright show-report

# Run specific test file
npx playwright test tests/e2e/generation-flow.spec.ts
```

### Combined Test Commands

```bash
# Run all tests (backend + frontend + e2e)
npm run test:all

# Run lint checks
npm run lint

# Full CI pipeline simulation
npm run lint && npm run test:all && npm run build
```

### Health Checks

```bash
# Check service ports
node scripts/check-ports.js

# Run health check
node scripts/health-check.js

# Quick API health
curl -s http://localhost:8001/api/v1/health | jq
```

---

## 14. Sign-off Checklist

Complete this checklist before approving deployment:

### Critical Requirements (Must Pass)

- [ ] **All CRITICAL gap tests passed** (GAP-001, GAP-002, GAP-003)
- [ ] **All HIGH priority gap tests passed** (GAP-004 through GAP-011)
- [ ] **Authentication flow working end-to-end**
- [ ] **WebSocket URL properly configurable**
- [ ] **All 12 provider types selectable**

### Quality Requirements

- [ ] **No regression in existing functionality**
- [ ] **Performance benchmarks met:**
  - [ ] Login < 500ms
  - [ ] Token refresh < 200ms
  - [ ] WS connection < 1s
- [ ] **Security tests passed:**
  - [ ] Token expiry working
  - [ ] Revocation working
  - [ ] CORS configured
  - [ ] Rate limiting active

### Documentation Requirements

- [ ] **API documentation updated**
- [ ] **Changelog updated**
- [ ] **Migration guide published** (if breaking changes)
- [ ] **Release notes prepared**

### Deployment Readiness

- [ ] **Ready for staging deployment**
- [ ] **Ready for production deployment**

---

### Final Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| QA Engineer | | | |
| Tech Lead | | | |
| Product Owner | | | |

---

## Appendix A: Gap to Test Case Mapping

| Gap ID | Severity | Test Cases | Phase |
|--------|----------|------------|-------|
| GAP-001 | CRITICAL | TC-003, TC-004, TC-005, TC-006 | 1 |
| GAP-002 | CRITICAL | TC-008 | 2 |
| GAP-003 | CRITICAL | TC-001 | 0 |
| GAP-004 | HIGH | TC-003.5 | 1 |
| GAP-005 | HIGH | TC-011 | 3 |
| GAP-006 | HIGH | TC-012 | 3 |
| GAP-007 | HIGH | TC-010 | 2 |
| GAP-008 | HIGH | TC-007, TC-015 | 1, 3 |
| GAP-009 | HIGH | TC-003.4 | 1 |
| GAP-010 | HIGH | TC-009 | 2 |
| GAP-011 | MEDIUM | TC-001 | 0 |
| GAP-012-021 | MEDIUM/LOW | TC-017 (Regression) | 4 |

---

## Appendix B: Environment Matrix

| Environment | API URL | WS URL | Auth Required |
|-------------|---------|--------|---------------|
| Development | http://localhost:8001/api/v1 | ws://localhost:8001 | Yes |
| Staging | https://staging-api.example.com/api/v1 | wss://staging-api.example.com | Yes |
| Production | https://api.example.com/api/v1 | wss://api.example.com | Yes |

---

*Document Generated: 2026-01-06*  
*Based on: GAP_ANALYSIS_REPORT.md, REMEDIATION_ROADMAP.md, API_COMPATIBILITY_MATRIX.md*