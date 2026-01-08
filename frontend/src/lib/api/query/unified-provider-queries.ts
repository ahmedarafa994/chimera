/**
 * TanStack Query hooks for the unified provider system.
 *
 * This module provides React Query hooks for the new unified provider API endpoints
 * that support database-backed session preferences with a three-tier selection hierarchy:
 * 1. Request Override (headers/query params)
 * 2. Session Preference (database)
 * 3. Global Default (environment)
 *
 * @module unified-provider-queries
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
  type UseQueryResult,
  type UseMutationResult,
} from "@tanstack/react-query";
import { apiClient } from "../client";
import { queryKeys, STALE_TIMES } from "./query-client";

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Selection scope indicating where the selection came from
 */
export type SelectionScope = "REQUEST" | "SESSION" | "GLOBAL";

/**
 * Provider information from unified API
 * Field names match backend ProviderResponse model
 */
export interface UnifiedProvider {
  provider_id: string;
  display_name: string;
  is_available: boolean;
  models_count: number;
  capabilities: string[];
  default_model?: string | null;
  status: string;
  has_api_key: boolean;
  // Computed/aliased fields for backwards compatibility
  id?: string;
  name?: string;
  description?: string;
  health_score?: number;
  supported_models?: string[];
}

/**
 * Model information from unified API
 */
export interface UnifiedModel {
  id: string;
  name: string;
  provider_id: string;
  description?: string;
  context_window?: number;
  max_tokens?: number;
  supports_streaming?: boolean;
  is_available: boolean;
}

/**
 * Current selection with source information
 */
export interface CurrentSelection {
  provider_id: string;
  model_id: string;
  scope: SelectionScope;
  session_id?: string;
  user_id?: string;
  created_at?: string;
  updated_at?: string;
}

/**
 * Request body for saving selection
 */
export interface SaveSelectionRequest {
  provider_id: string;
  model_id: string;
  session_id?: string;
  user_id?: string;
}

/**
 * Response from providers list endpoint
 */
export interface UnifiedProvidersResponse {
  providers: UnifiedProvider[];
  total: number;
}

/**
 * Response from models list endpoint
 */
export interface UnifiedModelsResponse {
  models: UnifiedModel[];
  provider_id: string;
  total: number;
}

// ============================================================================
// API Response Types (Backend Format)
// ============================================================================

/**
 * Provider info as returned by the backend API
 */
interface BackendProvider {
  provider: string;
  status: string;
  model: string;
  available_models: string[];
  models_detail?: Array<{
    id: string;
    name: string;
    provider: string;
    description?: string;
    max_tokens?: number;
    supports_streaming?: boolean;
    supports_vision?: boolean;
    is_default?: boolean;
    tier?: string;
  }>;
}

/**
 * Provider list response as returned by the backend API
 */
interface BackendProvidersResponse {
  providers: BackendProvider[];
  count?: number;
  default?: string;
  default_provider?: string;
  default_model?: string;
  total_models?: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Transform backend provider format to frontend format
 */
function transformProvider(backendProvider: BackendProvider): UnifiedProvider {
  return {
    provider_id: backendProvider.provider,
    display_name: backendProvider.provider.charAt(0).toUpperCase() + backendProvider.provider.slice(1),
    is_available: backendProvider.status === 'active',
    models_count: backendProvider.available_models?.length ?? 0,
    capabilities: [],
    default_model: backendProvider.model,
    status: backendProvider.status,
    has_api_key: backendProvider.status === 'active',
    // Backwards compatibility fields
    id: backendProvider.provider,
    name: backendProvider.provider.charAt(0).toUpperCase() + backendProvider.provider.slice(1),
    supported_models: backendProvider.available_models,
  };
}

/**
 * Fetch all available providers from unified API
 */
async function fetchUnifiedProviders(): Promise<UnifiedProvidersResponse> {
  const response = await apiClient.get<BackendProvidersResponse>("/providers");
  const backendData = response.data;

  // Transform backend format to frontend format
  const providers = (backendData.providers ?? []).map(transformProvider);

  return {
    providers,
    total: backendData.count ?? providers.length,
  };
}

/**
 * Backend model detail format
 */
interface BackendModelDetail {
  id: string;
  name: string;
  provider: string;
  description?: string;
  max_tokens?: number;
  supports_streaming?: boolean;
  supports_vision?: boolean;
  is_default?: boolean;
  tier?: string;
}

/**
 * Backend models response (same as providers but filtered)
 */
interface BackendModelsResponse {
  providers?: BackendProvider[];
  models?: BackendModelDetail[];
  default_provider?: string;
  default_model?: string;
  total_models?: number;
}

/**
 * Transform backend model to frontend format
 */
function transformModel(backendModel: BackendModelDetail): UnifiedModel {
  return {
    id: backendModel.id,
    name: backendModel.name,
    provider_id: backendModel.provider,
    description: backendModel.description,
    max_tokens: backendModel.max_tokens,
    supports_streaming: backendModel.supports_streaming,
    is_available: true,
  };
}

/**
 * Fetch models for a specific provider
 */
async function fetchUnifiedModels(providerId: string): Promise<UnifiedModelsResponse> {
  const response = await apiClient.get<BackendModelsResponse>(
    `/models`,
    { params: { provider: providerId } }
  );
  const backendData = response.data;

  // Extract models from the provider's models_detail array
  let models: UnifiedModel[] = [];

  if (backendData.providers) {
    const provider = backendData.providers.find(p => p.provider === providerId);
    if (provider?.models_detail) {
      models = provider.models_detail.map(transformModel);
    }
  } else if (backendData.models) {
    models = backendData.models
      .filter(m => m.provider === providerId)
      .map(transformModel);
  }

  return {
    models,
    provider_id: providerId,
    total: models.length,
  };
}

/**
 * Fetch all models across all providers
 */
async function fetchAllUnifiedModels(): Promise<UnifiedModelsResponse> {
  const response = await apiClient.get<BackendModelsResponse>("/models");
  const backendData = response.data;

  // Collect all models from all providers
  let models: UnifiedModel[] = [];

  if (backendData.providers) {
    for (const provider of backendData.providers) {
      if (provider.models_detail) {
        models = models.concat(provider.models_detail.map(transformModel));
      }
    }
  } else if (backendData.models) {
    models = backendData.models.map(transformModel);
  }

  return {
    models,
    provider_id: 'all',
    total: backendData.total_models ?? models.length,
  };
}

/**
 * Get current selection with source information
 * Transforms from backend format (provider, model) to frontend format (provider_id, model_id)
 */
async function fetchCurrentSelection(sessionId?: string): Promise<CurrentSelection> {
  const response = await apiClient.get<BackendModelSelectionResponse>(
    "/model-selection",
    { params: sessionId ? { session_id: sessionId } : undefined }
  );

  // Transform backend response to frontend format
  return {
    provider_id: response.data.provider ?? "",
    model_id: response.data.model ?? "",
    scope: "SESSION" as SelectionScope,
  };
}

/**
 * Backend request format for model selection
 */
interface BackendModelSelectionRequest {
  provider: string;
  model: string;
}

/**
 * Backend response format for model selection
 */
interface BackendModelSelectionResponse {
  provider: string | null;
  model: string | null;
}

/**
 * Save provider/model selection (creates session preference)
 * Transforms from frontend format (provider_id, model_id) to backend format (provider, model)
 */
async function saveSelection(request: SaveSelectionRequest): Promise<CurrentSelection> {
  // Transform frontend request to backend format
  const backendRequest: BackendModelSelectionRequest = {
    provider: request.provider_id,
    model: request.model_id,
  };

  const response = await apiClient.post<BackendModelSelectionResponse>(
    "/model-selection",
    backendRequest
  );

  // Transform backend response to frontend format
  return {
    provider_id: response.data.provider ?? "",
    model_id: response.data.model ?? "",
    scope: "SESSION" as SelectionScope,
  };
}

/**
 * Clear current selection (removes session preference, falls back to global default)
 */
async function clearSelection(sessionId?: string): Promise<void> {
  await apiClient.delete("/model-selection", {
    params: sessionId ? { session_id: sessionId } : undefined,
  });
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Hook to fetch all available providers.
 * Uses STATIC stale time since providers rarely change.
 *
 * @returns Query result with providers list
 *
 * @example
 * ```tsx
 * const { data, isLoading, error } = useUnifiedProviders();
 *
 * if (isLoading) return <Spinner />;
 * if (error) return <ErrorAlert error={error} />;
 *
 * return (
 *   <ProviderList providers={data?.providers ?? []} />
 * );
 * ```
 */
export function useUnifiedProviders(): UseQueryResult<UnifiedProvidersResponse, Error> {
  return useQuery({
    queryKey: queryKeys.unifiedProviders.list(),
    queryFn: fetchUnifiedProviders,
    staleTime: STALE_TIMES.STATIC,
  });
}

/**
 * Hook to fetch models for a specific provider.
 * Uses SEMI_DYNAMIC stale time for occasional updates.
 *
 * @param providerId - Provider ID to fetch models for
 * @param options - Query options
 * @returns Query result with models list
 *
 * @example
 * ```tsx
 * const { data, isLoading } = useUnifiedModels("openai", { enabled: !!selectedProvider });
 * ```
 */
export function useUnifiedModels(
  providerId: string | undefined,
  options?: { enabled?: boolean }
): UseQueryResult<UnifiedModelsResponse, Error> {
  return useQuery({
    queryKey: queryKeys.unifiedProviders.models(providerId ?? ""),
    queryFn: () => fetchUnifiedModels(providerId!),
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
    enabled: options?.enabled ?? !!providerId,
  });
}

/**
 * Hook to fetch all models across all providers.
 * Uses SEMI_DYNAMIC stale time for occasional updates.
 *
 * @returns Query result with all models
 */
export function useAllUnifiedModels(): UseQueryResult<UnifiedModelsResponse, Error> {
  return useQuery({
    queryKey: queryKeys.unifiedProviders.allModels(),
    queryFn: fetchAllUnifiedModels,
    staleTime: STALE_TIMES.SEMI_DYNAMIC,
  });
}

/**
 * Hook to fetch current selection with source information.
 * Uses DYNAMIC stale time for frequent updates.
 *
 * The response includes the selection scope indicating where it came from:
 * - REQUEST: From request headers or query parameters
 * - SESSION: From database session preference
 * - GLOBAL: From environment default
 *
 * @param sessionId - Optional session ID
 * @param options - Query options
 * @returns Query result with current selection
 *
 * @example
 * ```tsx
 * const { data: selection } = useCurrentSelection(sessionId);
 *
 * return (
 *   <div>
 *     <Badge variant={selection?.scope === "SESSION" ? "default" : "secondary"}>
 *       {selection?.scope}
 *     </Badge>
 *     <span>{selection?.provider_id} / {selection?.model_id}</span>
 *   </div>
 * );
 * ```
 */
export function useCurrentSelection(
  sessionId?: string,
  options?: { enabled?: boolean }
): UseQueryResult<CurrentSelection, Error> {
  return useQuery({
    queryKey: queryKeys.unifiedProviders.selection(sessionId),
    queryFn: () => fetchCurrentSelection(sessionId),
    staleTime: STALE_TIMES.DYNAMIC,
    enabled: options?.enabled ?? true,
  });
}

// ============================================================================
// Mutation Hooks
// ============================================================================

/**
 * Hook to save provider/model selection.
 * Implements optimistic updates and cache invalidation.
 *
 * After a successful save:
 * - Updates the selection cache optimistically
 * - Invalidates providers and models queries to refresh availability
 * - Broadcasts the change to connected clients via WebSocket
 *
 * @returns Mutation result
 *
 * @example
 * ```tsx
 * const saveMutation = useSaveSelection();
 *
 * const handleSelect = (providerId: string, modelId: string) => {
 *   saveMutation.mutate({
 *     provider_id: providerId,
 *     model_id: modelId,
 *     session_id: currentSessionId,
 *   });
 * };
 * ```
 */
export function useSaveSelection(): UseMutationResult<
  CurrentSelection,
  Error,
  SaveSelectionRequest,
  unknown
> {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: saveSelection,
    onMutate: async (variables) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({
        queryKey: queryKeys.unifiedProviders.selection(variables.session_id),
      });

      // Snapshot previous value
      const previousSelection = queryClient.getQueryData<CurrentSelection>(
        queryKeys.unifiedProviders.selection(variables.session_id)
      );

      // Optimistically update
      queryClient.setQueryData<CurrentSelection>(
        queryKeys.unifiedProviders.selection(variables.session_id),
        {
          provider_id: variables.provider_id,
          model_id: variables.model_id,
          scope: "SESSION",
          session_id: variables.session_id,
          user_id: variables.user_id,
        }
      );

      return { previousSelection };
    },
    onSuccess: (data, variables) => {
      // Update cache with server response
      queryClient.setQueryData(
        queryKeys.unifiedProviders.selection(variables.session_id),
        data
      );

      // Invalidate related queries
      queryClient.invalidateQueries({
        queryKey: queryKeys.unifiedProviders.all,
      });
    },
    onError: (error, variables, context) => {
      // Rollback on error
      if (context?.previousSelection) {
        queryClient.setQueryData(
          queryKeys.unifiedProviders.selection(variables.session_id),
          context.previousSelection
        );
      }
    },
  });
}

/**
 * Hook to clear current selection.
 * Removes session preference and falls back to global default.
 *
 * @returns Mutation result
 *
 * @example
 * ```tsx
 * const clearMutation = useClearSelection();
 *
 * const handleClear = () => {
 *   clearMutation.mutate({ session_id: currentSessionId });
 * };
 * ```
 */
export function useClearSelection(): UseMutationResult<
  void,
  Error,
  { session_id?: string },
  unknown
> {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ session_id }) => clearSelection(session_id),
    onSuccess: (_, variables) => {
      // Invalidate selection to trigger refetch (will return global default)
      queryClient.invalidateQueries({
        queryKey: queryKeys.unifiedProviders.selection(variables.session_id),
      });
    },
  });
}
