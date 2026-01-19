/**
 * Provider Service - Aligned with Backend API
 *
 * This service is aligned with the actual backend provider endpoints:
 * - /api/v1/providers
 * - /api/v1/models
 * - /api/v1/provider-config/*
 * - /api/v1/provider-sync/*
 */

import { apiClient } from '@/lib/api/client';
import { apiErrorHandler } from '@/lib/errors/api-error-handler';

// ============================================================================
// Types
// ============================================================================

export interface ProviderInfo {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error';
  isDefault: boolean;
  currentModel: string;
  availableModels: ModelInfo[];
  modelsCount: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  description: string;
  maxTokens: number;
  supportsStreaming: boolean;
  supportsVision: boolean;
  isDefault: boolean;
  tier: string;
}

export interface ProviderSyncStatus {
  lastSync: string;
  totalProviders: number;
  activeProviders: number;
  totalModels: number;
  syncInProgress: boolean;
  errors: string[];
}

// ============================================================================
// Service
// ============================================================================

class ProviderService {
  private readonly baseUrl = '/api/v1';

  /**
   * List all available providers
   */
  async listProviders(): Promise<ProviderInfo[]> {
    try {
      const response = await apiClient.get<any>(`${this.baseUrl}/providers`);
      const data = response.data;

      return data.providers.map((provider: any) => ({
        id: provider.provider,
        name: provider.provider.charAt(0).toUpperCase() + provider.provider.slice(1),
        status: provider.status === 'active' ? 'active' : 'inactive',
        isDefault: provider.provider === data.default,
        currentModel: provider.model,
        availableModels: (provider.available_models || []).map((model: string) => ({
          id: `${provider.provider}:${model}`,
          name: model,
          provider: provider.provider,
          description: `${provider.provider} ${model}`,
          maxTokens: 2048,
          supportsStreaming: true,
          supportsVision: false,
          isDefault: false,
          tier: 'standard',
        })),
        modelsCount: (provider.available_models || []).length,
      }));
    } catch (error) {
      apiErrorHandler.handleError(error, 'List Providers');
      return [];
    }
  }

  /**
   * List all available models with detailed information
   */
  async listModels(): Promise<{ providers: ProviderInfo[]; totalModels: number; defaultProvider: string; defaultModel: string }> {
    try {
      const response = await apiClient.get<any>(`${this.baseUrl}/models`);
      const data = response.data;

      const providers = data.providers.map((provider: any) => ({
        id: provider.provider,
        name: provider.provider.charAt(0).toUpperCase() + provider.provider.slice(1),
        status: provider.status === 'active' ? 'active' : 'inactive',
        isDefault: provider.provider === data.default_provider,
        currentModel: provider.model,
        availableModels: (provider.models_detail || []).map((model: any) => ({
          id: model.id,
          name: model.name,
          provider: model.provider,
          description: model.description,
          maxTokens: model.max_tokens,
          supportsStreaming: model.supports_streaming,
          supportsVision: model.supports_vision,
          isDefault: model.is_default,
          tier: model.tier,
        })),
        modelsCount: (provider.models_detail || []).length,
      }));

      return {
        providers,
        totalModels: data.total_models,
        defaultProvider: data.default_provider,
        defaultModel: data.default_model,
      };
    } catch (error) {
      apiErrorHandler.handleError(error, 'List Models');
      return {
        providers: [],
        totalModels: 0,
        defaultProvider: 'deepseek',
        defaultModel: 'deepseek-chat',
      };
    }
  }

  /**
   * Set active provider
   */
  async setActiveProvider(providerId: string, modelId?: string): Promise<boolean> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/provider-config/active`, {
        provider_id: providerId,
        model_id: modelId,
      });

      return response.status === 200 || response.status === 204;
    } catch (error) {
      apiErrorHandler.handleError(error, 'Set Active Provider');
      return false;
    }
  }

  /**
   * Test provider connection
   */
  async testProvider(providerId: string, apiKey?: string): Promise<{ success: boolean; message: string; latencyMs?: number }> {
    try {
      const response = await apiClient.post<any>(`${this.baseUrl}/provider-config/providers/${encodeURIComponent(providerId)}/test`, {
        api_key: apiKey,
      });

      const data = response.data;

      return {
        success: response.status === 200,
        message: data.message || (response.status === 200 ? 'Connection successful' : 'Connection failed'),
        latencyMs: data.latency_ms,
      };
    } catch (error) {
      apiErrorHandler.handleError(error, 'Test Provider');
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Connection test failed',
      };
    }
  }
}

export const providerService = new ProviderService();
