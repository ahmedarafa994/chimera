/**
 * Providers TanStack Query Hooks
 * Type-safe queries for provider management
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { queryKeys, STALE_TIMES } from "./query-client";
import { apiClient } from "../core/client";
import type {
  ProviderInfo,
  ProviderListResponse,
  ActiveProviderResponse,
  SetActiveProviderRequest,
  ProviderConfigResponse,
  UpdateProviderConfigRequest,
  ProviderHealthResponse,
} from "../../api-enhanced";

// ============================================================================
// API Functions
// ============================================================================

async function fetchProviders(): Promise<ProviderListResponse> {
  return apiClient.get<ProviderListResponse>("/provider-config/providers");
}

async function fetchActiveProvider(): Promise<ActiveProviderResponse> {
  return apiClient.get<ActiveProviderResponse>("/provider-config/active");
}

async function setActiveProvider(request: SetActiveProviderRequest): Promise<ActiveProviderResponse> {
  return apiClient.post<ActiveProviderResponse>("/provider-config/active", request);
}

async function fetchProviderConfig(providerId: string): Promise<ProviderConfigResponse> {
  return apiClient.get<ProviderConfigResponse>(
    `/provider-config/providers/${encodeURIComponent(providerId)}`
  );
}

async function updateProviderConfig(
  providerId: string,
  config: UpdateProviderConfigRequest
): Promise<ProviderConfigResponse> {
  return apiClient.put<ProviderConfigResponse>(
    `/provider-config/providers/${encodeURIComponent(providerId)}`,
    config
  );
}

async function fetchProviderHealth(providerId: string): Promise<ProviderHealthResponse> {
  return apiClient.get<ProviderHealthResponse>(
    `/provider-config/providers/${encodeURIComponent(providerId)}/health`
  );
}

async function fetchAllProvidersHealth(): Promise<{ providers: ProviderHealthResponse[] }> {
  return apiClient.get<{ providers: ProviderHealthResponse[] }>("/provider-config/health");
}

async function testProviderConnection(
  providerId: string,
  apiKey?: string
): Promise<{ success: boolean; message: string; latency_ms?: number }> {
  return apiClient.post<{ success: boolean; message: string; latency_ms?: number }>(
    `/provider-config/providers/${encodeURIComponent(providerId)}/test`,
    { api_key: apiKey }
  );
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Fetch list of all providers
 */
export function useProviders() {
  return useQuery({
    queryKey: queryKeys.providers.list(),
    queryFn: fetchProviders,
    staleTime: STALE_TIMES.STATIC,
  });
}

/**
 * Fetch the currently active provider
 */
export function useActiveProvider() {
  return useQuery({
    queryKey: queryKeys.providers.active(),
    queryFn: fetchActiveProvider,
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
  });
}

/**
 * Fetch configuration for a specific provider
 */
export function useProviderConfig(providerId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.providers.detail(providerId),
    queryFn: () => fetchProviderConfig(providerId),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: enabled && !!providerId,
  });
}

/**
 * Fetch health status for a specific provider
 */
export function useProviderHealth(providerId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.providers.health(providerId),
    queryFn: () => fetchProviderHealth(providerId),
    staleTime: STALE_TIMES.DYNAMIC,
    enabled: enabled && !!providerId,
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

/**
 * Fetch health status for all providers
 */
export function useAllProvidersHealth() {
  return useQuery({
    queryKey: [...queryKeys.providers.all, "health-all"],
    queryFn: fetchAllProvidersHealth,
    staleTime: STALE_TIMES.DYNAMIC,
    refetchInterval: 30000,
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Set the active provider
 */
export function useSetActiveProvider() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: setActiveProvider,
    onSuccess: (data) => {
      // Update active provider cache
      queryClient.setQueryData(queryKeys.providers.active(), data);
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.providers.all });
      queryClient.invalidateQueries({ queryKey: queryKeys.models.all });
    },
  });
}

/**
 * Update provider configuration
 */
export function useUpdateProviderConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ providerId, config }: { providerId: string; config: UpdateProviderConfigRequest }) =>
      updateProviderConfig(providerId, config),
    onSuccess: (data, { providerId }) => {
      // Update specific provider cache
      queryClient.setQueryData(queryKeys.providers.detail(providerId), data);
      // Invalidate provider list to reflect changes
      queryClient.invalidateQueries({ queryKey: queryKeys.providers.list() });
    },
  });
}

/**
 * Test provider connection
 */
export function useTestProviderConnection() {
  return useMutation({
    mutationFn: ({ providerId, apiKey }: { providerId: string; apiKey?: string }) =>
      testProviderConnection(providerId, apiKey),
  });
}