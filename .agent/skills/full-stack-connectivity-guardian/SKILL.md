---
name: full-stack-connectivity-guardian
description: Autonomous full-stack connectivity verification. Self-healing E2E tests, frontend/backend validation, API contract monitoring. Use when validating app connectivity or generating adaptive tests.
skills: []
---

# Full-Stack Connectivity Guardian

> **Type**: Autonomous • Tool-Calling • Self-Healing • Continuous Verification

An autonomous IDE skill that continuously verifies frontend pages, backend services, and their connectivity, automatically generating, repairing, and evolving tests as the application changes.

Acts as a **background reliability engineer** inside the IDE.

---

## When This Skill Triggers

- On project open
- On frontend or backend code change
- On route, API, or schema modification
- On developer request ("verify app", "check connectivity")
- Before commit / before deploy

---

## Core Capabilities

### 1️⃣ Application Awareness

Detect frontend framework (React, Next.js, Vue, Angular, etc.) and backend stack (Node, Django, Laravel, Spring, etc.).

**Locate:**

- Routes
- API endpoints
- Auth mechanisms
- Environment configs

### 2️⃣ Autonomous Page Discovery

Enumerate all frontend routes:

- Static routes
- SPA router paths
- Auth-guarded pages

Crawl internal navigation and maintain a route graph.

### 3️⃣ Live Frontend Verification

For each discovered page:

- Load in real browser context
- Detect: JS runtime errors, console errors, missing DOM anchors, broken assets
- Measure load & render time

### 4️⃣ Frontend ↔ Backend Connectivity Validation

Intercept network calls and identify:

- REST
- GraphQL
- WebSocket

**Validate per request:**

- Status codes
- Payload existence
- Response structure
- Latency

**Detect:**

- Missing calls
- Silent failures
- Contract drift

### 5️⃣ Backend Independent Health Checks

- Auto-discover health endpoints
- Execute direct API tests
- Detect: Service degradation, auth failures, DB-backed endpoint instability

---

## 🧠 Self-Healing Engine (Core Feature)

When a test fails due to change, the skill must:

### 🔄 Diagnose

- UI selector drift
- Route rename
- API schema evolution
- Async timing changes

### 🛠 Heal

- Regenerate selectors using: Semantics, Roles, Text intent
- Infer new API schema from live traffic
- Adjust waits and assertions
- Re-run the test

### 🧾 Record

- What changed
- Why it changed
- What was updated

> The skill never hides breakage — it adapts but **flags breaking changes**.

---

## 🔁 End-to-End Flow Verification

- Login / logout
- Navigation flows
- CRUD interactions
- UI ↔ API state consistency

**Self-heal if:**

- Buttons move
- Forms change
- Validation logic shifts

---

## ⚠️ Failure Classification

Every issue is categorized as:

| Category | Description |
| ---------- | ------------- |
| Frontend Rendering Error | Component/layout failures |
| JavaScript Runtime Error | Uncaught exceptions |
| Backend API Error | 4xx/5xx responses |
| Network / Connectivity Error | Connection failures |
| Authentication Error | Auth flow failures |
| Performance Regression | Slow responses/renders |
| Breaking Contract Change | API schema drift |

Each includes:

- Evidence
- Root-cause hypothesis
- Severity
- Suggested fix

---

## 📊 IDE-Native Output

### Inline IDE Feedback

- Route health indicators
- API status badges
- Real-time warnings

### Generated Artifacts

- `connectivity-report.json`
- Auto-updated E2E tests
- Failure snapshots & traces

---

## 🔐 Execution Rules

1. Use real services (no mocks unless explicit)
2. Never fabricate passing results
3. Never suppress errors
4. Prefer adaptation over skipping
5. Continue execution unless system is unreachable

---

## 🏁 Success Conditions

The skill completes successfully when:

- All reachable pages are validated
- Frontend/backend connectivity is confirmed
- Tests are healed where possible
- All issues are explained, not ignored

---

## 🧠 Skill Philosophy

This skill behaves like a **senior full-stack reliability engineer** embedded in the IDE.

- If code changes → **adapt**
- If behavior breaks → **explain**
- If everything passes → **prove it**
