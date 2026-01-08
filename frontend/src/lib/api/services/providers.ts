/**
 * Providers API Service
 * Handles all provider-related API calls
 */

import { apiClient } from '../client';
import { Provider, ProviderModel, ApiResponse } from '../types';

export interface ProvidersResponse {
  providers: Provider[];
  default_provider: string;
}

export interface ProviderModelsResponse {
  models: ProviderModel[];
  provider: string;
}

export interface SetDefaultProviderRequest {
  provider_id: string;
  model_id?: string;
}

export const providersApi = {
  /**
   * Get all available providers
   */
  async getProviders(): Promise<ApiResponse<ProvidersResponse>> {
    return apiClient.get<ProvidersResponse>('/providers');
  },

  /**
   * Get available providers with status
   */
  async getAvailableProviders(): Promise<ApiResponse<ProvidersResponse>> {
    return apiClient.get<ProvidersResponse>('/providers/available');
  },

  /**
   * Get current provider and model
   */
  async getCurrentProvider(): Promise<ApiResponse<{ provider: string; model: string }>> {
    return apiClient.get('/providers/current');
  },

  /**
   * Get models for a specific provider
   */
  async getProviderModels(providerId: string): Promise<ApiResponse<ProviderModelsResponse>> {
    return apiClient.get<ProviderModelsResponse>(`/providers/${providerId}/models`);
  },

  /**
   * Set default provider and model
   */
  async setDefaultProvider(request: SetDefaultProviderRequest): Promise<ApiResponse<void>> {
    return apiClient.post('/providers/default', request);
  },

  /**
   * Test provider connection
   */
  async testProvider(providerId: string): Promise<ApiResponse<{ status: string; latency_ms: number }>> {
    return apiClient.post(`/providers/${providerId}/test`);
  },

  /**
   * Get provider health status
   */
  async getProviderHealth(providerId: string): Promise<ApiResponse<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    latency_ms: number;
    error?: string;
  }>> {
    return apiClient.get(`/providers/${providerId}/health`);
  },
};