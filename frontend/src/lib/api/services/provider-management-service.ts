/**
 * Provider Management Service
 * 
 * Provides API methods for managing LLM providers,
 * including status monitoring, rate limits, and fallback suggestions.
 */

import { enhancedApi } from '@/lib/api-enhanced';
import type {
  ProviderStatus,
  RateLimitInfo,
  FallbackSuggestion,
  CircuitBreakerState,
  ProviderHealth,
} from '@/types/provider-management-types';

const API_BASE = '/api/v1/models';

/**
 * Provider Management Service API
 */
export const providerManagementService = {
  /**
   * Get all available providers
   */
  async getProviders(): Promise<ProviderStatus[]> {
    return enhancedApi.get<ProviderStatus[]>(`${API_BASE}/providers`);
  },

  /**
   * Get status of a specific provider
   */
  async getProviderStatus(providerId: string): Promise<ProviderStatus> {
    return enhancedApi.get<ProviderStatus>(`${API_BASE}/providers/${providerId}/status`);
  },

  /**
   * Get rate limit information for all providers
   */
  async getRateLimits(): Promise<RateLimitInfo[]> {
    return enhancedApi.get<RateLimitInfo[]>(`${API_BASE}/rate-limits`);
  },

  /**
   * Get rate limit for a specific provider/model
   */
  async getRateLimit(providerId: string, modelId?: string): Promise<RateLimitInfo> {
    const params = modelId ? { model_id: modelId } : {};
    return enhancedApi.get<RateLimitInfo>(
      `${API_BASE}/providers/${providerId}/rate-limit`,
      { params }
    );
  },

  /**
   * Get fallback suggestions for a provider/model
   */
  async getFallbackSuggestions(
    providerId: string,
    modelId?: string
  ): Promise<FallbackSuggestion[]> {
    const params = modelId ? { model_id: modelId } : {};
    return enhancedApi.get<FallbackSuggestion[]>(
      `${API_BASE}/fallback-suggestions`,
      { params: { provider_id: providerId, ...params } }
    );
  },

  /**
   * Get circuit breaker state for a provider
   */
  async getCircuitBreakerState(providerId: string): Promise<CircuitBreakerState> {
    return enhancedApi.get<CircuitBreakerState>(
      `${API_BASE}/providers/${providerId}/circuit-breaker`
    );
  },

  /**
   * Reset circuit breaker for a provider
   */
  async resetCircuitBreaker(providerId: string): Promise<{ success: boolean; message: string }> {
    return enhancedApi.post<{ success: boolean; message: string }>(
      `${API_BASE}/providers/${providerId}/circuit-breaker/reset`
    );
  },

  /**
   * Get health status of all providers
   */
  async getProvidersHealth(): Promise<ProviderHealth[]> {
    return enhancedApi.get<ProviderHealth[]>(`${API_BASE}/health`);
  },

  /**
   * Get available models for a provider
   */
  async getProviderModels(providerId: string): Promise<{
    models: {
      id: string;
      name: string;
      context_length: number;
      capabilities: string[];
    }[];
  }> {
    return enhancedApi.get<{
      models: {
        id: string;
        name: string;
        context_length: number;
        capabilities: string[];
      }[];
    }>(`${API_BASE}/providers/${providerId}/models`);
  },

  /**
   * Test provider connection
   */
  async testProviderConnection(providerId: string): Promise<{
    success: boolean;
    latency_ms: number;
    error?: string;
  }> {
    return enhancedApi.post<{
      success: boolean;
      latency_ms: number;
      error?: string;
    }>(`${API_BASE}/providers/${providerId}/test`);
  },

  /**
   * Get provider configuration
   */
  async getProviderConfig(providerId: string): Promise<{
    id: string;
    name: string;
    api_base_url: string;
    default_model: string;
    rate_limit_rpm: number;
    rate_limit_tpm: number;
    timeout_seconds: number;
  }> {
    return enhancedApi.get<{
      id: string;
      name: string;
      api_base_url: string;
      default_model: string;
      rate_limit_rpm: number;
      rate_limit_tpm: number;
      timeout_seconds: number;
    }>(`${API_BASE}/providers/${providerId}/config`);
  },

  /**
   * Update provider configuration
   */
  async updateProviderConfig(
    providerId: string,
    config: Partial<{
      default_model: string;
      rate_limit_rpm: number;
      rate_limit_tpm: number;
      timeout_seconds: number;
    }>
  ): Promise<{ success: boolean; message: string }> {
    return enhancedApi.put<{ success: boolean; message: string }>(
      `${API_BASE}/providers/${providerId}/config`,
      config
    );
  },
};

export default providerManagementService;