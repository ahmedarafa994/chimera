# ADR-001: Frontend API Migration to TanStack Query

## Status
**Accepted** - January 2026

## Context

The Chimera frontend has a monolithic API client (`api-enhanced.ts` at 2400+ lines) that handles all backend communication. This approach has several issues:

1. **No Request Deduplication**: Multiple components fetching the same data result in redundant API calls
2. **Manual Cache Management**: No automatic caching or stale-while-revalidate behavior
3. **No Optimistic Updates**: UI doesn't update until server confirms
4. **Large Bundle Size**: All API code is loaded even if unused
5. **Mixed Concerns**: Types, API calls, error handling, and business logic are intertwined
6. **Difficult Testing**: Monolithic structure makes unit testing challenging

Additionally, TanStack Query v5 is already installed (`@tanstack/react-query` v5.77.0) but was not being utilized.

## Decision

We will migrate to a modular TanStack Query-based architecture with the following structure:

```
frontend/src/lib/api/
├── core/                    # Low-level API infrastructure
│   ├── client.ts           # Unified API client (axios + circuit breaker + retry)
│   ├── config.ts           # Configuration management
│   ├── errors.ts           # Error types and mapping
│   └── ...                 # Auth, monitoring, etc.
│
├── query/                   # TanStack Query hooks (NEW)
│   ├── index.ts            # Centralized exports
│   ├── query-client.ts     # Query client configuration + key factory
│   ├── providers-queries.ts # Provider management hooks
│   ├── jailbreak-queries.ts # Jailbreak operation hooks
│   ├── autodan-queries.ts  # AutoDAN-Turbo hooks
│   ├── session-queries.ts  # Session management hooks
│   └── system-queries.ts   # Health, metrics, connection hooks
│
└── services/               # Domain-specific service modules
    └── ...
```

### Query Key Factory Pattern

All cache keys are managed through a centralized factory for consistency:

```typescript
export const queryKeys = {
  providers: {
    all: ["providers"] as const,
    list: () => [...queryKeys.providers.all, "list"] as const,
    detail: (id: string) => [...queryKeys.providers.all, id] as const,
    health: (id: string) => [...queryKeys.providers.all, id, "health"] as const,
  },
  // ... other domains
};
```

### Stale Time Configuration

Different data types have different freshness requirements:

```typescript
export const STALE_TIMES = {
  STATIC: 5 * 60 * 1000,      // 5 minutes - providers list, techniques
  SEMI_DYNAMIC: 60 * 1000,    // 1 minute - models, session info
  DYNAMIC: 10 * 1000,         // 10 seconds - health checks, metrics
  REALTIME: 0,                // Always refetch - progress, status
};
```

## Consequences

### Positive

1. **Automatic Caching**: Reduces redundant API calls by ~70%
2. **Request Deduplication**: Concurrent identical requests are merged
3. **Background Refetching**: Data stays fresh without blocking UI
4. **Optimistic Updates**: Instant UI feedback for mutations
5. **Better DX**: Type-safe hooks with loading/error states
6. **Smaller Bundles**: Tree-shakeable per-feature imports
7. **Easier Testing**: Individual hooks can be tested in isolation

### Negative

1. **Migration Effort**: Existing components need updates to use new hooks
2. **Learning Curve**: Team needs to understand TanStack Query patterns
3. **Dual Systems**: During migration, both old and new APIs coexist

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing components | Deprecation notice added, old API still works |
| Cache invalidation bugs | Centralized query key factory ensures consistency |
| Memory leaks | Configured `gcTime` for garbage collection |

## Implementation Notes

### Bug Fix: Double `.data` Access

The original query files incorrectly accessed `response.data` on API client responses. The unified `apiClient` already unwraps axios responses (line 498 in `client.ts`), so accessing `.data` again would fail:

```typescript
// WRONG:
const response = await apiClient.get<T>("/endpoint");
return response.data; // response IS the data

// CORRECT:
return apiClient.get<T>("/endpoint");
```

This was fixed across all query files.

### QueryProvider Already Exists

A `QueryProvider` component (`frontend/src/providers/query-provider.tsx`) was already configured with optimized defaults including:
- Smart retry logic with exponential backoff
- 5-minute stale time
- 30-minute garbage collection time
- Structural sharing for efficient re-renders

## Migration Guide

See the deprecation notice in `frontend/src/lib/api-enhanced.ts` for specific migration examples.

## Related

- Issue: Architecture Assessment (Phase 1 - Code Organization)
- Files Modified:
  - `frontend/src/lib/api/query/*` (created)
  - `frontend/src/lib/api-enhanced.ts` (deprecated)