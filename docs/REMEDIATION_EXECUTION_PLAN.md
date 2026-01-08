# Remediation Execution Plan: Restoring Architectural Integrity

**Date:** December 10, 2025
**Status:** DRAFT
**Based on:** `CODEBASE_TECHNICAL_AUDIT.md`

---

## 1. Assessment Methodology
This plan utilizes the **Codebase Technical Audit (V1.0)** as the primary truth source. The assessment methodology focused on:
1.  **Static Analysis**: Structural review of `backend-api` vs `Project_Chimera`.
2.  **Logic Verification**: Line-by-line inspection of "AI" algorithms in `unified_transformation_engine.py`.
3.  **Configuration Audit**: Review of `config.py` against 12-Factor App methodology.
4.  **Dependency Mapping**: Container analysis via `docker-compose.yml`.

---

## 2. Identified Critical Organizational Issues

We define "Organizational Issues" here as structural defects that impede the software lifecycle and erode trust.

| ID | Issue Category | Description | Severity | Risk |
|----|----------------|-------------|----------|------|
| **ORG-01** | **Integrity & Trust** | "Complexity Theatre": Deceptive algorithms (Quantum/Evolutionary) that simulate work without value. | **CRITICAL** | Reputation loss, Latency, Maintenance costs for fake code. |
| **ORG-02** | **Architecture** | "Split Brain": Running both legacy Flask and modern FastAPI backends simultaneously. | **HIGH** | Data inconsistency, Double maintenance, Confusion. |
| **ORG-03** | **Agility** | "Rigid Config": ~300+ lines of hardcoded business logic in `config.py`. | **HIGH** | Deployment bottlenecks, Inability to hot-patch techniques. |
| **ORG-04** | **Maintainability** | "Monolithic Frontend": Oversized components (>500 lines) handling too many concerns. | **MEDIUM** | Fragility, High cost of change for UI. |

---

## 3. Implementation Timeline & resource Allocation

### Sprint 1: Operation "Truth" (Immediate - 1 Day)
**Objective**: Remove deceptive logic and establish a honest baseline.
**Resources**: Lead Developer (Agent).
**Deliverables**:
-   [x] **Refactor `unified_transformation_engine.py`**: Remove `_orchestrate_quantum_superposition` and `_orchestrate_evolutionary_pipeline`.
-   [x] **Implement Strategy Pattern**: Replace removed logic with simple, transparent pass-throughs or real LLM calls (if available).
-   [x] **Verify**: Ensure tests pass without the fake loops.

### Sprint 2: Configuration Decoupling (Days 2-3)
**Objective**: Enable business logic changes without code deployment.
**Resources**: Lead Developer (Agent).
**Deliverables**:
-   [x] **Create `config/techniques.yaml`**: Move the dictionary from `config.py` to YAML.
-   [x] **Update `Settings`**: Implement dynamic loader in `app/core/config.py`.
-   [x] **Validation**: Verify that modifying YAML updates the API response.

### Phase 3: Architectural Unification (Days 4-7)
**Objective**: Move to a single, robust backend.
**Resources**: Lead Developer (Agent) + User (Review).
**Deliverables**:
-   [x] **Migrate Routes**: Port any missing critical logic from Flask to FastAPI.
-   [x] **Update Docker**: Remove `legacy-backend` service.
-   [x] **Proxy Cleanup**: Simplify `config.py` connection mode logic.

### Sprint 4: Frontend Componentization (Bonus)
**Objective**: Improve maintainability of UI.
**Resources**: Lead Developer (Agent).
**Deliverables**:
-   [x] **Extract Data**: Move `TECHNIQUE_SUITES` to `config/techniques.ts`.
-   [x] **Split Components**: Create `InputPanel` and `ResultDisplay` sub-components.
-   [x] **Refactor Parent**: Simplify `IntentAwareGenerator.tsx`.

---

## 4. Success Metrics & Monitoring Strategy

| internal Metric | Target | Verification Method |
|-----------------|--------|---------------------|
| **Code Integrity** | 0 "Simulation" lines | Grep search for `ImportError` or `range(3)` loops in engines. |
| **Architecture** | 1 Backend Container | `docker ps` shows only `backend-api`. |
| **Config Agility** | 0 Rebuilds for Tweak | Change YAML -> Restart App -> Verify Change. |
| **Performance** | < 200ms Overhead | Benchmark transformation endpoint (formerly slowed by loops). |

---

## 5. Stakeholder Communication Protocols

*   **Decision Points**: User MUST approve the deletion of "featured" algorithms (Quantum/Evolutionary) acknowledging they were deceptive.
*   **Progress Updates**: Use `task_boundary` for real-time status.
*   **Blockers**: Use `notify_user` immediately if dependencies break during "Split Brain" resolution.

---

## 6. Contingency Planning

*   **Risk**: Removing "Quantum" logic breaks a dependent frontend component expecting that specific string format.
    *   *Mitigation*: Keep the *interface* identical, but change the *implementation* to return the prompt immediately or use a real simple transformation, ensuring the API contract remains valid.
*   **Risk**: Docker container fails to start after config refactor.
    *   *Mitigation*: Implement a `config_safe_mode` fallback in `loader.py` to use hardcoded defaults if YAML is missing/corrupt.
