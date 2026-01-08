# Recommendations & Implementation Roadmap

**Generated**: December 26, 2025
**Priority Level**: High

---

## 1. Critical Priority (Immediate Action Required)

### 1.1 Scalability: Migrate Background Jobs
**Issue**: The current `BackgroundJobService` runs in-memory. If the API server restarts or scales horizontally, job state is lost.
**Recommendation**:
- Replace in-memory queue with **Celery** backed by **Redis**.
- Persist job status and results to **PostgreSQL**.
- **Impact**: Enables reliable execution of long-running tasks like AutoDAN and Jailbreak generation.

### 1.2 Integration: Unify Frontend-Backend Resilience
**Issue**: Both Frontend (`APIClient`) and Backend (`LLMService`) implement retries and circuit breakers. This causes "retry storms" where 3 frontend retries x 3 backend retries = 9 total attempts per failure.
**Recommendation**:
- **Disable Circuit Breaker in Frontend** when using Proxy Mode (let backend handle upstream failures).
- **Reduce Frontend Retries** to 1 (for network connectivity only).
- **Trust Backend Status**: If backend returns 503 (Circuit Open), frontend should fail fast and show user feedback immediately.

---

## 2. High Priority (Next Sprint)

### 2.1 Configuration: Dynamic Model Synchronization
**Issue**: Frontend has hardcoded model lists in `client.ts` or config files that drift from the backend's master list.
**Recommendation**:
- **Single Source of Truth**: Frontend should *only* use the `/api/v1/providers` endpoint to populate model dropdowns.
- **Cache Strategy**: Cache this list in `localStorage` with a short TTL (e.g., 5 minutes) to avoid startup latency.

### 2.2 Security: Production Database Enforcement
**Issue**: SQLite is default. While convenient for dev, it is dangerous for production concurrency.
**Recommendation**:
- Strict enforcement of `DATABASE_URL` format in production environment variables (already partially implemented in `config.py`).
- Migration scripts to move dev data to Postgres if needed.

---

## 3. Medium Priority (Maintenance & Quality)

### 3.1 Observability: Distributed Tracing
**Recommendation**:
- Implement **OpenTelemetry** context propagation.
- Frontend should send `traceparent` headers.
- Backend should log these trace IDs to correlate user actions with server logs.

### 3.2 Documentation: API Contract
**Recommendation**:
- Auto-generate TypeScript interfaces from backend Pydantic models (using tools like `datamodel-code-generator`) to prevent type mismatches.

---

## 4. Implementation Checklist

- [ ] **Backend**: Install `celery`, `redis` dependencies.
- [ ] **Backend**: Create `app/worker.py` entry point.
- [ ] **Frontend**: Update `api-config.ts` to remove hardcoded model lists.
- [ ] **Frontend**: Modify `client.ts` to skip circuit breaker in proxy mode.
- [ ] **DevOps**: Add Redis service to `docker-compose.yml` (if not present).