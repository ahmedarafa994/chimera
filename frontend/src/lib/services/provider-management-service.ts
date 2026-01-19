/**
 * Provider Management Service for Project Chimera Frontend
 * 
 * Provides API methods for provider management functionality including:
 * - Provider listing
 * - Model discovery
 * - Model selection
 * - Rate limiting
 * - Health monitoring
 */

import { enhancedApi } from "../api-enhanced";
import {
  ProviderInfo,
  AvailableProvidersResponse,
  ModelInfo,
  ProviderModelsResponse,
  ModelSelectionRequest,
  ModelSelectionResponse,
  CurrentSelectionResponse,
  RateLimitResponse,
  ProviderStatus,
  CircuitBreakerState,
  ProviderHealthStatus,
  AllProvidersHealthResponse,
  getOrCreateSessionId,
  SESSION_HEADER,
} from "../types/provider-management-types";

// =============================================================================
// Configuration
// =============================================================================

const PROVIDERS_BASE_PATH = "/providers";

// =============================================================================
// Session Management
// =============================================================================

/**
 * Get headers with session ID
 */
function getSessionHeaders(): Record<string, string> {
  return {
    [SESSION_HEADER]: getOrCreateSessionId(),
  };
}

// =============================================================================
// Mapping Helpers
// =============================================================================

type BackendProviderInfo = {
  provider: string;
  display_name?: string;
  status?: string;
  is_healthy?: boolean;
  models?: string[];
  default_model?: string | null;
  latency_ms?: number | null;
};

type BackendProviderModelsResponse = {
  provider: string;
  display_name?: string;
  models: Array<{
    id: string;
    name: string;
    description?: string | null;
    max_tokens?: number;
    is_default?: boolean;
    tier?: string;
  }>;
  default_model?: string | null;
  count: number;
};

type BackendCurrentSelection = {
  provider?: string;
  model?: string;
  display_name?: string;
  session_id?: string | null;
  is_default?: boolean;
};

type BackendRateLimitInfo = {
  allowed: boolean;
  remaining_requests: number;
  remaining_tokens: number;
  reset_at: string;
  retry_after_seconds?: number | null;
  limit_type?: string | null;
  tier?: string;
  fallback_provider?: string | null;
};

function mapProviderStatus(status?: string, isHealthy?: boolean): ProviderStatus {
  if (status === "rate_limited") return ProviderStatus.RATE_LIMITED;
  if (status === "degraded") return ProviderStatus.DEGRADED;
  if (status === "unavailable") return ProviderStatus.UNAVAILABLE;
  if (status === "unknown") return ProviderStatus.UNKNOWN;
  if (typeof isHealthy === "boolean") {
    return isHealthy ? ProviderStatus.AVAILABLE : ProviderStatus.UNAVAILABLE;
  }
  return ProviderStatus.AVAILABLE;
}

function mapProviderInfo(provider: BackendProviderInfo): ProviderInfo {
  return {
    provider_id: provider.provider,
    name: provider.display_name || provider.provider,
    status: mapProviderStatus(provider.status, provider.is_healthy),
    is_enabled: true,
    supported_model_types: ["chat"],
    metadata: {
      default_model: provider.default_model,
      latency_ms: provider.latency_ms,
    },
  };
}

function mapModelInfo(
  providerId: string,
  model: BackendProviderModelsResponse["models"][number]
): ModelInfo {
  return {
    model_id: model.id,
    name: model.name || model.id,
    description: model.description || undefined,
    provider_id: providerId,
    model_type: "chat",
    context_window: model.max_tokens,
    max_output_tokens: model.max_tokens,
    supports_streaming: true,
    supports_function_calling: false,
    supports_vision: false,
    metadata: {
      tier: model.tier,
      is_default: model.is_default,
    },
  };
}

// =============================================================================
// Provider API Methods
// =============================================================================

/**
 * List all available providers
 */
export async function listAvailableProviders(): Promise<AvailableProvidersResponse> {
  const response = await enhancedApi.get<{
    providers: BackendProviderInfo[];
    count: number;
    default_provider?: string;
    default_model?: string;
  }>(
    `${PROVIDERS_BASE_PATH}/available`,
    {
      headers: getSessionHeaders(),
    }
  );
  const providers = response.providers.map(mapProviderInfo);
  return {
    providers,
    total_count: response.count || providers.length,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Get models for a specific provider
 */
export async function getProviderModels(
  providerId: string
): Promise<ProviderModelsResponse> {
  const response = await enhancedApi.get<BackendProviderModelsResponse>(
    `${PROVIDERS_BASE_PATH}/${encodeURIComponent(providerId)}/models`,
    {
      headers: getSessionHeaders(),
    }
  );
  const models = response.models.map((model) => mapModelInfo(response.provider, model));
  return {
    provider_id: response.provider,
    models,
    total_count: response.count,
    last_updated: new Date().toISOString(),
  };
}

/**
 * Get a specific provider's info
 */
export async function getProviderInfo(
  providerId: string
): Promise<ProviderInfo> {
  const response = await enhancedApi.get<BackendProviderInfo>(
    `${PROVIDERS_BASE_PATH}/${encodeURIComponent(providerId)}`,
    {
      headers: getSessionHeaders(),
    }
  );
  return mapProviderInfo(response);
}

// =============================================================================
// Selection API Methods
// =============================================================================

/**
 * Select a model for the current session
 */
export async function selectModel(
  request: ModelSelectionRequest
): Promise<ModelSelectionResponse> {
  const sessionId = request.session_id || getOrCreateSessionId();

  const response = await enhancedApi.post<{
    success: boolean;
    message?: string;
    provider: string;
    model: string;
    session_id: string;
  }>(
    `${PROVIDERS_BASE_PATH}/select`,
    {
      provider: request.provider_id,
      model: request.model_id,
      session_id: sessionId,
      reason: request.reason,
    },
    {
      headers: getSessionHeaders(),
    }
  );
  return {
    success: response.success,
    session_id: response.session_id,
    provider_id: response.provider,
    model_id: response.model,
    selected_at: new Date().toISOString(),
    message: response.message,
  };
}

/**
 * Get the current model selection for the session
 */
export async function getCurrentSelection(): Promise<CurrentSelectionResponse> {
  const response = await enhancedApi.get<BackendCurrentSelection>(
    `${PROVIDERS_BASE_PATH}/current`,
    {
      headers: getSessionHeaders(),
    }
  );
  const hasSelection = !!response.provider && !!response.model;
  return {
    has_selection: hasSelection,
    session_id: response.session_id || undefined,
    provider_id: response.provider || undefined,
    model_id: response.model || undefined,
    selected_at: new Date().toISOString(),
  };
}

/**
 * Clear the current model selection
 */
export async function clearSelection(): Promise<{ success: boolean; message: string }> {
  const response = await enhancedApi.delete<{ success: boolean; message: string }>(
    `${PROVIDERS_BASE_PATH}/current`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
}

// =============================================================================
// Rate Limiting API Methods
// =============================================================================

/**
 * Get rate limit information for a provider/model
 */
export async function getRateLimitInfo(
  providerId: string,
  modelId: string
): Promise<RateLimitResponse> {
  const response = await enhancedApi.get<BackendRateLimitInfo>(
    `${PROVIDERS_BASE_PATH}/rate-limit`,
    {
      headers: getSessionHeaders(),
      params: {
        provider_id: providerId,
        model_id: modelId,
      },
    }
  );
  return {
    rate_limit: {
      provider_id: providerId,
      model_id: modelId,
      is_rate_limited: !response.allowed,
      requests_remaining: response.remaining_requests,
      tokens_remaining: response.remaining_tokens,
      reset_at: response.reset_at,
      retry_after_seconds: response.retry_after_seconds || undefined,
      tier: response.tier,
    },
    fallback_suggestions: response.fallback_provider
      ? [
          {
            provider_id: response.fallback_provider,
            model_id: "",
            reason: "Rate limit fallback",
            estimated_availability: "unknown",
            compatibility_score: 0.5,
          },
        ]
      : undefined,
  };
}

// =============================================================================
// Health API Methods
// =============================================================================

/**
 * Get health status for a specific provider
 */
export async function getProviderHealth(
  providerId: string
): Promise<ProviderHealthStatus> {
  const response = await enhancedApi.get<{
    provider?: string;
    is_healthy?: boolean;
    last_check?: string;
    latency_ms?: number | null;
    error_message?: string | null;
    consecutive_failures?: number;
  }>(
    `${PROVIDERS_BASE_PATH}/${encodeURIComponent(providerId)}/health`,
    {
      headers: getSessionHeaders(),
    }
  );
  return {
    provider_id: response.provider || providerId,
    status: response.is_healthy ? ProviderStatus.AVAILABLE : ProviderStatus.UNAVAILABLE,
    circuit_breaker_state: CircuitBreakerState.CLOSED,
    last_success: response.is_healthy ? response.last_check : undefined,
    last_failure: response.is_healthy ? undefined : response.last_check,
    failure_count: response.consecutive_failures || 0,
    success_rate: response.is_healthy ? 1 : 0,
    avg_latency_ms: response.latency_ms || undefined,
    checked_at: response.last_check || new Date().toISOString(),
  };
}

/**
 * Get health status for all providers
 */
export async function getAllProvidersHealth(): Promise<AllProvidersHealthResponse> {
  const response = await enhancedApi.get<{
    providers: Array<{
      provider: string;
      is_healthy: boolean;
      last_check: string;
      latency_ms?: number | null;
      error_message?: string | null;
      consecutive_failures?: number;
    }>;
    timestamp: string;
  }>(
    `${PROVIDERS_BASE_PATH}/health`,
    {
      headers: getSessionHeaders(),
    }
  );
  const providers = response.providers.map((provider) => ({
    provider_id: provider.provider,
    status: provider.is_healthy ? ProviderStatus.AVAILABLE : ProviderStatus.UNAVAILABLE,
    circuit_breaker_state: CircuitBreakerState.CLOSED,
    last_success: provider.is_healthy ? provider.last_check : undefined,
    last_failure: provider.is_healthy ? undefined : provider.last_check,
    failure_count: provider.consecutive_failures || 0,
    success_rate: provider.is_healthy ? 1 : 0,
    avg_latency_ms: provider.latency_ms || undefined,
    checked_at: provider.last_check,
  }));

  const systemHealth = providers.some((p) => p.status !== ProviderStatus.AVAILABLE)
    ? "degraded"
    : "healthy";

  return {
    providers,
    system_health: systemHealth,
    timestamp: response.timestamp,
  };
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Get all models from all providers
 */
export async function getAllModels(): Promise<Map<string, ModelInfo[]>> {
  const providersResponse = await listAvailableProviders();
  const modelsMap = new Map<string, ModelInfo[]>();

  await Promise.all(
    providersResponse.providers.map(async (provider) => {
      try {
        const modelsResponse = await getProviderModels(provider.provider_id);
        modelsMap.set(provider.provider_id, modelsResponse.models);
      } catch (error) {
        console.warn(`Failed to get models for provider ${provider.provider_id}:`, error);
        modelsMap.set(provider.provider_id, []);
      }
    })
  );

  return modelsMap;
}

/**
 * Find a model by ID across all providers
 */
export async function findModelById(
  modelId: string
): Promise<{ provider: ProviderInfo; model: ModelInfo } | null> {
  const providersResponse = await listAvailableProviders();

  for (const provider of providersResponse.providers) {
    try {
      const modelsResponse = await getProviderModels(provider.provider_id);
      const model = modelsResponse.models.find((m) => m.model_id === modelId);
      if (model) {
        return { provider, model };
      }
    } catch {
      // Continue to next provider
    }
  }

  return null;
}

/**
 * Select the best available model based on criteria
 */
export async function selectBestAvailableModel(
  criteria: {
    preferredProvider?: string;
    modelType?: string;
    minContextWindow?: number;
    requiresStreaming?: boolean;
    requiresFunctionCalling?: boolean;
  } = {}
): Promise<ModelSelectionResponse | null> {
  const providersResponse = await listAvailableProviders();
  const healthResponse = await getAllProvidersHealth();

  // Filter healthy providers
  const healthyProviders = providersResponse.providers.filter((provider) => {
    const health = healthResponse.providers.find(
      (h) => h.provider_id === provider.provider_id
    );
    return health && health.status !== "unavailable";
  });

  // Sort by preference
  const sortedProviders = healthyProviders.sort((a, b) => {
    if (criteria.preferredProvider) {
      if (a.provider_id === criteria.preferredProvider) return -1;
      if (b.provider_id === criteria.preferredProvider) return 1;
    }
    return 0;
  });

  // Find matching model
  for (const provider of sortedProviders) {
    try {
      const modelsResponse = await getProviderModels(provider.provider_id);

      const matchingModel = modelsResponse.models.find((model) => {
        if (criteria.modelType && model.model_type !== criteria.modelType) {
          return false;
        }
        if (
          criteria.minContextWindow &&
          model.context_window &&
          model.context_window < criteria.minContextWindow
        ) {
          return false;
        }
        if (criteria.requiresStreaming && !model.supports_streaming) {
          return false;
        }
        if (criteria.requiresFunctionCalling && !model.supports_function_calling) {
          return false;
        }
        return true;
      });

      if (matchingModel) {
        return selectModel({
          provider_id: provider.provider_id,
          model_id: matchingModel.model_id,
          reason: "Auto-selected based on criteria",
        });
      }
    } catch {
      // Continue to next provider
    }
  }

  return null;
}

// =============================================================================
// Service Object
// =============================================================================

export const providerManagementService = {
  // Provider methods
  listAvailableProviders,
  getProviderModels,
  getProviderInfo,

  // Selection methods
  selectModel,
  getCurrentSelection,
  clearSelection,

  // Rate limiting
  getRateLimitInfo,

  // Health
  getProviderHealth,
  getAllProvidersHealth,

  // Convenience
  getAllModels,
  findModelById,
  selectBestAvailableModel,
};

export default providerManagementService;
