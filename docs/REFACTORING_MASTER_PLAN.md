# Project Chimera - Refactoring Master Plan

## 1. Assessment Methodology & Findings
**Source of Truth**: `CODE_AUDIT_REPORT_V2.md`
**Key Finding**: The codebase suffers from "Complexity Theatre"—over-engineered, speculative implementations of AI logic (Quantum/Evolutionary simulations) that obscure the actual capabilities and introduce technical debt.

### Issue Prioritization Matrix
| Priority | Category | Issue | Risk Level |
|----------|----------|-------|------------|
| **P0** | **Architecture** | Redundant/Fake Service Engines (`ARCH-01`, `ARCH-02`) | High (Maintenance/Perf) |
| **P0** | **Config** | Hardcoded Transformation Logic (`SEC-01`) | High (Flexibility) |
| **P1** | **Frontend** | Monolithic Components (`FE-01`) | Med (Scalability) |
| **P2** | **Code Quality** | Circular Imports & Dead Code | Med (Stability) |

---

## 2. Resource Allocation Strategy
*   **Execution Team**: Pair Programming Unit (User + AI Agent).
*   **Role Division**:
    *   *AI (Antigravity)*: Strategy design, code scaffold generation, heavy refactoring execution.
    *   *User (Lead)*: Architectural review, approval of "destructive" changes (deletions), manual testing.

---

## 3. Implementation Timeline & Milestones

### Phase 1: Operation "Simplicity" (Backend Refactor)
*Objective: Remove speculative code and implement a standard Strategy Pattern.*

*   **Milestone 1.1: Core Interface Definition**
    *   Create `app/services/transform/base.py`.
    *   Define `TransformationStrategy` abstract base class.
*   **Milestone 1.2: Implementation Migration**
    *   Port functional logic from `unified_transformation_engine.py` to `BasicTransformationStrategy`.
    *   Port valid logic from `adaptive_transformation_engine.py` to `AdaptiveTransformationStrategy`.
*   **Milestone 1.3: The Great Purge**
    *   Delete `unified_transformation_engine.py`.
    *   Delete `adaptive_transformation_engine.py`.
    *   Delete `advanced_transformation_engine.py` (if redundant).
*   **Milestone 1.4: Integration**
    *   Update `main.py` dependencies.
    *   Verify tests pass.

### Phase 2: Configuration Liberty
*Objective: Decouple business rules from code.*

*   **Milestone 2.1: Schema Definition**
    *   Design `techniques.schema.json` or Pydantic model.
*   **Milestone 2.2: Externalization**
    *   Extract `TRANSFORMATION_TECHNIQUES` to `config/techniques.yml`.
*   **Milestone 2.3: Loader Implementation**
    *   Update `app/core/config.py` to load from YAML.

### Phase 3: Frontend Decomposition
*Objective: Improve maintainability of UI.*

*   **Milestone 3.1: Hook Extraction**
    *   Extract logic from `intent-aware-generator.tsx` into `useIntentGenerator` hook.
*   **Milestone 3.2: Component Splitting**
    *   Split UI into `GeneratorControls`, `GeneratorOutput`, `GeneratorHistory`.

---

## 4. Success Metrics & Monitoring

### KPIs (Key Performance Indicators)
1.  **Code Volume**: Reduction of >2,000 LOC (Lines of Code) in `app/services/`.
2.  **Cyclomatic Complexity**: Average function complexity < 10.
3.  **Startup Time**: backend startup < 2s.
4.  **Configuration**: 0 hardcoded techniques in Python files.

### Monitoring Framework
*   **Pre-Commit**: `ruff` linting to ensure no dead code remains.
*   **CI/CD**: `pytest` suite must pass after every "deletion" event.

---

## 5. Contingency Planning

### Rollback Strategy
*   **Feature Branching**: All destructive changes happen in `refactor/core-engines`.
*   **Verification Gates**: Tests must pass before merging to `main`.
*   **Backup**: Keep a copy of `unified_transformation_engine.py` in `legacy/` for one sprint if "simulations" need to be referenced (though unlikely).

## 6. Communication Protocols
*   **Task Updates**: Step-by-step reporting using `task_boundary` tool.
*   **Critical Decision points**: User approval required before mass deletion of files.

---

## 7. Immediate Next Steps (Action Items)
1.  [ ] Create `app/services/transform/base.py` (Strategy Interface).
2.  [ ] Scaffold `StandardStrategy` based on existing logic.
3.  [ ] **Notify User** for review of the new interface.
