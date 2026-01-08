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
// Provider API Methods
// =============================================================================

/**
 * List all available providers
 */
export async function listAvailableProviders(): Promise<AvailableProvidersResponse> {
  const response = await enhancedApi.get<AvailableProvidersResponse>(
    `${PROVIDERS_BASE_PATH}/available`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
}

/**
 * Get models for a specific provider
 */
export async function getProviderModels(
  providerId: string
): Promise<ProviderModelsResponse> {
  const response = await enhancedApi.get<ProviderModelsResponse>(
    `${PROVIDERS_BASE_PATH}/${encodeURIComponent(providerId)}/models`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
}

/**
 * Get a specific provider's info
 */
export async function getProviderInfo(
  providerId: string
): Promise<ProviderInfo> {
  const response = await enhancedApi.get<ProviderInfo>(
    `${PROVIDERS_BASE_PATH}/${encodeURIComponent(providerId)}`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
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

  const response = await enhancedApi.post<ModelSelectionResponse>(
    `${PROVIDERS_BASE_PATH}/select`,
    { ...request, session_id: sessionId },
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
}

/**
 * Get the current model selection for the session
 */
export async function getCurrentSelection(): Promise<CurrentSelectionResponse> {
  const response = await enhancedApi.get<CurrentSelectionResponse>(
    `${PROVIDERS_BASE_PATH}/current`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
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
  const response = await enhancedApi.get<RateLimitResponse>(
    `${PROVIDERS_BASE_PATH}/rate-limit`,
    {
      headers: getSessionHeaders(),
      params: {
        provider_id: providerId,
        model_id: modelId,
      },
    }
  );
  return response;
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
  const response = await enhancedApi.get<ProviderHealthStatus>(
    `${PROVIDERS_BASE_PATH}/${encodeURIComponent(providerId)}/health`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
}

/**
 * Get health status for all providers
 */
export async function getAllProvidersHealth(): Promise<AllProvidersHealthResponse> {
  const response = await enhancedApi.get<AllProvidersHealthResponse>(
    `${PROVIDERS_BASE_PATH}/health`,
    {
      headers: getSessionHeaders(),
    }
  );
  return response;
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