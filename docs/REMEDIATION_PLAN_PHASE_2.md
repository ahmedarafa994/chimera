# Technical Remediation Plan - Phase 2

## Executive Summary
This phase addresses critical architectural weaknesses in the `JailbreakService` and API layer that pose significant risks to stability, scalability, and debuggability. The primary focus is eliminating memory leaks in the execution tracking and caching mechanisms, transitioning towards distributed state management, and improving error handling granularity.

## 1. Critical: Memory Leak in Execution Tracker
**Root Cause:** The `ExecutionTracker` class maintains an unbounded dictionary `_completed_executions` that stores the full history of every job processed by the service instance.
**Solution:**
*   **Immediate Fix:** Implement a `maxlen` constraint using `collections.deque` or a periodic cleanup mechanism to cap the number of stored execution records in memory.
*   **Strategic Fix:** Offload execution history to a persistent store (Database) or a distributed cache (Redis) with TTL. For this remediation, we will implement a bounded in-memory structure that mimics a sliding window of recent history, which is sufficient for the current requirements while preventing OOM.

## 2. High: Unbounded In-Memory Cache
**Root Cause:** The `CacheManager` uses a standard Python dictionary without a size limit or eviction policy. High-volume unique requests (e.g., fuzzing) can exhaust memory.
**Solution:**
*   **Immediate Fix:** Refactor `CacheManager` to use `OrderedDict` with a strict key limit and LRU (Least Recently Used) eviction policy, similar to the fix applied to the Observability module.
*   **Strategic Fix:** Abstract the cache interface to support a Redis backend, allowing for shared state across workers.

## 3. Moderate: Non-Distributed State (Scalability)
**Root Cause:** `RateLimiter`, `CacheManager`, and `ExecutionTracker` rely on process-local state.
**Solution:**
*   **Architecture:** Refactor these components to use the `RedisRateLimiter` pattern established in Phase 1.
*   **Implementation:**
    *   Update `RateLimiter` in `jailbreak_service.py` to use the core `get_rate_limiter()` factory which supports Redis.
    *   Ensure `CacheManager` and `ExecutionTracker` are designed to be swappable with Redis implementations in the future (Dependency Injection).

## 4. Moderate: Brittle Error Handling
**Root Cause:** API endpoints wrap logic in generic `try/except Exception` blocks, catching specific errors (like `ProviderNotAvailableError`) and re-raising them as generic 500 Internal Server Errors.
**Solution:**
*   **Refactoring:** Remove generic catch-all blocks where possible.
*   **Granularity:** Catch specific exceptions (`ValueError`, `ProviderNotAvailableError`, `TransformationError`) and map them to appropriate HTTP status codes (400, 503, 422).
*   **Middleware:** Rely on the global exception handler for truly unexpected errors.

## Implementation Steps
1.  **Refactor `JailbreakService`:**
    *   Update `ExecutionTracker` to use `deque` for completed executions.
    *   Update `CacheManager` to use `OrderedDict` with LRU eviction.
    *   Update `RateLimiter` to leverage the core distributed rate limiter.
2.  **Refactor `api_routes.py`:**
    *   Remove generic `try/except` blocks.
    *   Add specific exception handlers for domain errors.
3.  **Verification:**
    *   Add unit tests to verify memory bounds and error mapping.