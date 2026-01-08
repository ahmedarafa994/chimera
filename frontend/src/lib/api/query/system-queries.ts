/**
 * System TanStack Query Hooks
 * Type-safe queries for system health, metrics, and connection status
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { queryKeys, STALE_TIMES } from "./query-client";
import { apiClient } from "../core/client";
import type {
  ConnectionStatusResponse,
  ConnectionTestResponse,
  MetricsResponse,
} from "../../api-enhanced";
import type { ModelsListResponse, ModelSyncHealthResponse } from "../../api-enhanced";

// ============================================================================
// Types
// ============================================================================

export interface HealthResponse {
  status: string;
  timestamp: string;
  version?: string;
  uptime?: number;
}

export interface ModelValidationRequest {
  provider: string;
  model: string;
}

export interface ModelValidationResponse {
  valid: boolean;
  message: string;
  fallback_model?: string;
  fallback_provider?: string;
}

// ============================================================================
// API Functions
// ============================================================================

async function fetchHealth(): Promise<HealthResponse> {
  return apiClient.get<HealthResponse>("/health", {
    baseURL: "/api", // Health endpoint is at root
  });
}

async function fetchMetrics(): Promise<MetricsResponse> {
  return apiClient.get<MetricsResponse>("/metrics");
}

async function fetchConnectionStatus(): Promise<ConnectionStatusResponse> {
  return apiClient.get<ConnectionStatusResponse>("/connection/status");
}

async function testConnection(config?: {
  provider?: string;
  model?: string;
}): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>("/connection/test", config || {});
}

async function fetchModels(): Promise<ModelsListResponse> {
  return apiClient.get<ModelsListResponse>("/models");
}

async function fetchModelsHealth(): Promise<ModelSyncHealthResponse> {
  return apiClient.get<ModelSyncHealthResponse>("/models/health");
}

async function validateModel(request: ModelValidationRequest): Promise<ModelValidationResponse> {
  return apiClient.post<ModelValidationResponse>("/models/validate", request);
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Fetch system health status
 */
export function useHealth(enabled = true) {
  return useQuery({
    queryKey: queryKeys.system.health(),
    queryFn: fetchHealth,
    staleTime: STALE_TIMES.DYNAMIC,
    refetchInterval: 30000, // Refetch every 30 seconds
    enabled,
  });
}

/**
 * Fetch system metrics
 */
export function useMetrics(enabled = true) {
  return useQuery({
    queryKey: queryKeys.system.metrics(),
    queryFn: fetchMetrics,
    staleTime: STALE_TIMES.DYNAMIC,
    refetchInterval: 60000, // Refetch every minute
    enabled,
  });
}

/**
 * Fetch connection status
 */
export function useConnectionStatus(enabled = true) {
  return useQuery({
    queryKey: queryKeys.system.connection(),
    queryFn: fetchConnectionStatus,
    staleTime: STALE_TIMES.DYNAMIC,
    refetchInterval: 30000,
    enabled,
  });
}

/**
 * Fetch available models
 */
export function useModels() {
  return useQuery({
    queryKey: queryKeys.models.list(),
    queryFn: fetchModels,
    staleTime: STALE_TIMES.STATIC,
  });
}

/**
 * Fetch models by provider
 */
export function useModelsByProvider(provider: string) {
  const modelsQuery = useModels();
  
  return {
    ...modelsQuery,
    data: modelsQuery.data?.providers.find((p) => p.provider === provider)?.available_models || [],
    providerData: modelsQuery.data?.providers.find((p) => p.provider === provider),
  };
}

/**
 * Fetch model sync health
 */
export function useModelsHealth() {
  return useQuery({
    queryKey: queryKeys.models.health(),
    queryFn: fetchModelsHealth,
    staleTime: STALE_TIMES.DYNAMIC,
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Test connection
 */
export function useTestConnection() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: testConnection,
    onSuccess: () => {
      // Refresh connection status after test
      queryClient.invalidateQueries({ queryKey: queryKeys.system.connection() });
    },
  });
}

/**
 * Validate model selection
 */
export function useValidateModel() {
  return useMutation({
    mutationFn: validateModel,
  });
}

// ============================================================================
// Prefetching Utilities
// ============================================================================

/**
 * Prefetch critical system data on app init
 */
export function usePrefetchSystemData() {
  const queryClient = useQueryClient();

  return async () => {
    await Promise.all([
      queryClient.prefetchQuery({
        queryKey: queryKeys.system.health(),
        queryFn: fetchHealth,
        staleTime: STALE_TIMES.DYNAMIC,
      }),
      queryClient.prefetchQuery({
        queryKey: queryKeys.models.list(),
        queryFn: fetchModels,
        staleTime: STALE_TIMES.STATIC,
      }),
    ]);
  };
}

/**
 * Check if backend is connected
 */
export function useIsBackendConnected() {
  const health = useHealth();
  return {
    isConnected: health.data?.status === "healthy",
    isLoading: health.isLoading,
    error: health.error,
    refetch: health.refetch,
  };
}