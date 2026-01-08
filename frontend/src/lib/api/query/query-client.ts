/**
 * TanStack Query Client Configuration
 * Centralized query client with optimized defaults for Chimera
 */

import { QueryClient } from "@tanstack/react-query";

/**
 * Default stale time for different query types (in milliseconds)
 */
export const STALE_TIMES = {
  /** Static data that rarely changes (providers list, techniques) */
  STATIC: 5 * 60 * 1000, // 5 minutes
  /** Semi-dynamic data (models, session info) */
  SEMI_DYNAMIC: 60 * 1000, // 1 minute
  /** Dynamic data (health checks, metrics) */
  DYNAMIC: 10 * 1000, // 10 seconds
  /** Real-time data (progress, status) */
  REALTIME: 0, // Always refetch
} as const;

/**
 * Query key factory for consistent cache key generation
 */
export const queryKeys = {
  // Providers
  providers: {
    all: ["providers"] as const,
    list: () => [...queryKeys.providers.all, "list"] as const,
    detail: (id: string) => [...queryKeys.providers.all, id] as const,
    health: (id: string) => [...queryKeys.providers.all, id, "health"] as const,
    active: () => [...queryKeys.providers.all, "active"] as const,
  },

  // Models
  models: {
    all: ["models"] as const,
    list: () => [...queryKeys.models.all, "list"] as const,
    byProvider: (provider: string) => [...queryKeys.models.all, provider] as const,
    health: () => [...queryKeys.models.all, "health"] as const,
    validate: (provider: string, model: string) =>
      [...queryKeys.models.all, "validate", provider, model] as const,
  },

  // Sessions
  sessions: {
    all: ["sessions"] as const,
    current: () => [...queryKeys.sessions.all, "current"] as const,
    detail: (id: string) => [...queryKeys.sessions.all, id] as const,
    stats: () => [...queryKeys.sessions.all, "stats"] as const,
  },

  // Jailbreak
  jailbreak: {
    all: ["jailbreak"] as const,
    techniques: () => [...queryKeys.jailbreak.all, "techniques"] as const,
    result: (requestId: string) => [...queryKeys.jailbreak.all, "result", requestId] as const,
  },

  // AutoDAN
  autodan: {
    all: ["autodan"] as const,
    strategies: () => [...queryKeys.autodan.all, "strategies"] as const,
    strategy: (id: string) => [...queryKeys.autodan.all, "strategy", id] as const,
    libraryStats: () => [...queryKeys.autodan.all, "library", "stats"] as const,
    progress: () => [...queryKeys.autodan.all, "progress"] as const,
    health: () => [...queryKeys.autodan.all, "health"] as const,
  },

  // GPTFuzz
  gptfuzz: {
    all: ["gptfuzz"] as const,
    session: (id: string) => [...queryKeys.gptfuzz.all, "session", id] as const,
  },

  // System
  system: {
    health: () => ["system", "health"] as const,
    metrics: () => ["system", "metrics"] as const,
    connection: () => ["system", "connection"] as const,
  },

  // Unified Provider System (NEW - Phase 2)
  unifiedProviders: {
    all: ["unified-providers"] as const,
    list: () => [...queryKeys.unifiedProviders.all, "list"] as const,
    models: (providerId: string) =>
      [...queryKeys.unifiedProviders.all, "models", providerId] as const,
    allModels: () => [...queryKeys.unifiedProviders.all, "models"] as const,
    selection: (sessionId?: string) =>
      sessionId
        ? [...queryKeys.unifiedProviders.all, "selection", sessionId] as const
        : [...queryKeys.unifiedProviders.all, "selection"] as const,
  },
} as const;

/**
 * Create and configure the query client
 */
export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: STALE_TIMES.SEMI_DYNAMIC,
        gcTime: 10 * 60 * 1000, // 10 minutes garbage collection
        retry: 2,
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
        refetchOnWindowFocus: false,
        refetchOnReconnect: true,
      },
      mutations: {
        retry: 1,
        retryDelay: 1000,
      },
    },
  });
}

/**
 * Singleton query client instance
 */
let queryClientInstance: QueryClient | null = null;

export function getQueryClient(): QueryClient {
  if (!queryClientInstance) {
    queryClientInstance = createQueryClient();
  }
  return queryClientInstance;
}