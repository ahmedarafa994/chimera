/**
 * Provider Management Service
 *
 * Provides API methods for managing LLM providers,
 * including status monitoring, rate limits, and fallback suggestions.
 */

import { apiClient } from '../client';
import type {
  ProviderStatus,
  RateLimitInfo,
  FallbackSuggestion,
  CircuitBreakerState,
  ProviderHealth,
} from '@/types/provider-management-types';

const API_BASE = '/provider-config';

/**
 * Provider Management Service API
 */
export const providerManagementService = {
  /**
   * Get all available providers
   */
  async getProviders(): Promise<ProviderStatus[]> {
    const response = await apiClient.get<ProviderStatus[]>(`${API_BASE}/providers`);
    return response.data;
  },

  /**
   * Get status of a specific provider
   */
  async getProviderStatus(providerId: string): Promise<ProviderStatus> {
    const response = await apiClient.get<ProviderStatus>(`${API_BASE}/providers/${providerId}/status`);
    return response.data;
  },

  /**
   * Get rate limit information for all providers
   */
  async getRateLimits(): Promise<RateLimitInfo[]> {
    const response = await apiClient.get<RateLimitInfo[]>(`${API_BASE}/rate-limits`);
    return response.data;
  },

  /**
   * Get rate limit for a specific provider/model
   */
  async getRateLimit(providerId: string, modelId?: string): Promise<RateLimitInfo> {
    const params = modelId ? { model_id: modelId } : {};
    const response = await apiClient.get<RateLimitInfo>(
      `${API_BASE}/providers/${providerId}/rate-limit`,
      { params }
    );
    return response.data;
  },

  /**
   * Get fallback suggestions for a provider/model
   */
  async getFallbackSuggestions(
    providerId: string,
    modelId?: string
  ): Promise<FallbackSuggestion[]> {
    const params = modelId ? { model_id: modelId } : {};
    const response = await apiClient.get<FallbackSuggestion[]>(
      `${API_BASE}/fallback-suggestions`,
      { params: { provider_id: providerId, ...params } }
    );
    return response.data;
  },

  /**
   * Get circuit breaker state for a provider
   */
  async getCircuitBreakerState(providerId: string): Promise<CircuitBreakerState> {
    const response = await apiClient.get<CircuitBreakerState>(
      `${API_BASE}/providers/${providerId}/circuit-breaker`
    );
    return response.data;
  },

  /**
   * Reset circuit breaker for a provider
   */
  async resetCircuitBreaker(providerId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>(
      `${API_BASE}/providers/${providerId}/circuit-breaker/reset`
    );
    return response.data;
  },

  /**
   * Get health status of all providers
   */
  async getProvidersHealth(): Promise<ProviderHealth[]> {
    const response = await apiClient.get<ProviderHealth[]>(`${API_BASE}/health`);
    return response.data;
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
    const response = await apiClient.get<any>(`${API_BASE}/providers/${providerId}/models`);
    return response.data;
  },

  /**
   * Test provider connection
   */
  async testProviderConnection(providerId: string): Promise<{
    success: boolean;
    latency_ms: number;
    error?: string;
  }> {
    const response = await apiClient.post<any>(`${API_BASE}/providers/${providerId}/test`);
    return response.data;
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
    const response = await apiClient.get<any>(`${API_BASE}/providers/${providerId}/config`);
    return response.data;
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
    const response = await apiClient.put<any>(
      `${API_BASE}/providers/${providerId}/config`,
      config
    );
    return response.data;
  },
};

export default providerManagementService;
