/**
 * Provider Service - Aligned with Backend API
 *
 * This service is properly aligned with backend endpoints:
 * - GET /api/v1/providers (list all providers)
 * - GET /api/v1/providers/{provider}/models (get provider models)
 * - POST /api/v1/providers/select (select provider/model)
 * - GET /api/v1/providers/current (get current selection)
 * - GET /api/v1/providers/health (provider health status)
 * - WS /api/v1/providers/ws/selection (real-time updates)
 */

import { apiClient } from '../client';
import { apiErrorHandler } from '../../errors/api-error-handler';

// ============================================================================
// Types (matching backend Pydantic models exactly)
// ============================================================================

export interface ProviderModel {
  id: string;
  name: string;
  description?: string;
  max_tokens: number;
  is_default: boolean;
  tier: string;
}

export interface ProviderInfo {
  provider: string;
  display_name: string;
  status: string;
  is_healthy: boolean;
  models: string[];
  default_model?: string;
  latency_ms?: number;
}

export interface ProvidersListResponse {
  providers: ProviderInfo[];
  count: number;
  default_provider: string;
  default_model: string;
}

export interface ProviderModelsResponse {
  provider: string;
  display_name: string;
  models: ProviderModel[];
  default_model?: string;
  count: number;
}

export interface SelectProviderRequest {
  provider: string;
  model: string;
}

export interface SelectProviderResponse {
  success: boolean;
  message: string;
  provider: string;
  model: string;
  session_id: string;
}

export interface CurrentSelectionResponse {
  provider: string;
  model: string;
  display_name: string;
  session_id?: string;
  is_default: boolean;
}

export interface ProviderHealthStatus {
  provider: string;
  status: string;
  latency_ms?: number;
  last_check: string;
  error_message?: string;
}

export interface ProviderHealthResponse {
  providers: ProviderHealthStatus[];
  timestamp: string;
}

export interface RateLimitInfoResponse {
  allowed: boolean;
  remaining_requests: number;
  remaining_tokens: number;
  reset_at: string;
  retry_after_seconds?: number;
  limit_type?: string;
  tier: string;
  fallback_provider?: string;
}

// ============================================================================
// Provider Service Implementation
// ============================================================================

export class ProviderService {
  private wsConnection?: WebSocket;

  /**
   * Get all available providers with their status and models
   */
  async getProviders() {
    try {
      const response = await apiClient.get<ProvidersListResponse>('/api/v1/providers');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviders');
    }
  }

  /**
   * Get available providers (alias for getProviders)
   */
  async getAvailableProviders() {
    try {
      const response = await apiClient.get<ProvidersListResponse>('/api/v1/providers/available');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetAvailableProviders');
    }
  }

  /**
   * Get detailed information about a specific provider
   */
  async getProviderDetail(provider: string) {
    try {
      const response = await apiClient.get<ProviderInfo>(`/api/v1/providers/${encodeURIComponent(provider)}`);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviderDetail');
    }
  }

  /**
   * Get available models for a specific provider
   */
  async getProviderModels(provider: string) {
    try {
      const response = await apiClient.get<ProviderModelsResponse>(
        `/api/v1/providers/${encodeURIComponent(provider)}/models`
      );
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviderModels');
    }
  }

  /**
   * Select a provider and model for the session
   */
  async selectProvider(request: SelectProviderRequest) {
    try {
      const response = await apiClient.post<SelectProviderResponse>('/api/v1/providers/select', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'SelectProvider');
    }
  }

  /**
   * Get the current provider and model selection
   */
  async getCurrentSelection() {
    try {
      const response = await apiClient.get<CurrentSelectionResponse>('/api/v1/providers/current');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetCurrentSelection');
    }
  }

  /**
   * Clear the current provider selection (revert to defaults)
   */
  async clearCurrentSelection() {
    try {
      const response = await apiClient.delete('/api/v1/providers/current');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'ClearCurrentSelection');
    }
  }

  /**
   * Get health status for all providers
   */
  async getProviderHealth() {
    try {
      const response = await apiClient.get<ProviderHealthResponse>('/api/v1/providers/health');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviderHealth');
    }
  }

  /**
   * Get health status for a specific provider
   */
  async getProviderHealthDetail(provider: string) {
    try {
      const response = await apiClient.get<ProviderHealthStatus>(
        `/api/v1/providers/${encodeURIComponent(provider)}/health`
      );
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetProviderHealthDetail');
    }
  }

  /**
   * Test connection to a specific provider
   */
  async testProviderConnection(provider: string) {
    try {
      const response = await apiClient.post(`/api/v1/providers/${encodeURIComponent(provider)}/test`);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'TestProviderConnection');
    }
  }

  /**
   * Check rate limit status for a provider/model combination
   */
  async checkRateLimit(provider: string, model: string) {
    try {
      const response = await apiClient.get<RateLimitInfoResponse>(
        `/api/v1/providers/rate-limit?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}`
      );
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'CheckRateLimit');
    }
  }

  /**
   * Set default provider and model for new sessions
   */
  async setDefaultProvider(provider_id: string, model_id?: string) {
    try {
      const response = await apiClient.post('/api/v1/providers/default', {
        provider_id,
        model_id,
      });
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'SetDefaultProvider');
    }
  }

  // ============================================================================
  // WebSocket Methods
  // ============================================================================

  /**
   * Create WebSocket connection for real-time provider updates
   */
  createWebSocketConnection(): WebSocket {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      return this.wsConnection;
    }

    const wsUrl = apiClient.getWebSocketUrl('/api/v1/providers/ws/selection');
    this.wsConnection = new WebSocket(wsUrl);

    return this.wsConnection;
  }

  /**
   * Subscribe to real-time provider selection changes
   */
  subscribeToSelectionChanges(callback: (event: any) => void): () => void {
    const ws = this.createWebSocketConnection();

    const messageHandler = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'selection_change') {
          callback(message.data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.addEventListener('message', messageHandler);

    // Return cleanup function
    return () => {
      ws.removeEventListener('message', messageHandler);
    };
  }

  /**
   * Subscribe to provider health updates
   */
  subscribeToHealthUpdates(callback: (health: ProviderHealthStatus[]) => void): () => void {
    const ws = this.createWebSocketConnection();

    const messageHandler = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'health_update' && message.data?.providers) {
          callback(message.data.providers);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket health message:', error);
      }
    };

    ws.addEventListener('message', messageHandler);

    // Return cleanup function
    return () => {
      ws.removeEventListener('message', messageHandler);
    };
  }

  /**
   * Send a ping message to keep the WebSocket connection alive
   */
  pingWebSocket(): void {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString(),
      }));
    }
  }

  /**
   * Close WebSocket connection
   */
  closeWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = undefined;
    }
  }

  /**
   * Get current WebSocket connection status
   */
  getWebSocketStatus(): string {
    if (!this.wsConnection) return 'disconnected';

    switch (this.wsConnection.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'disconnected';
      default: return 'unknown';
    }
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const providerService = new ProviderService();

// ============================================================================
// Convenience functions for direct usage
// ============================================================================

export const providerApi = {
  getProviders: () => providerService.getProviders(),
  getAvailableProviders: () => providerService.getAvailableProviders(),
  getProviderDetail: (provider: string) => providerService.getProviderDetail(provider),
  getProviderModels: (provider: string) => providerService.getProviderModels(provider),
  selectProvider: (request: SelectProviderRequest) => providerService.selectProvider(request),
  getCurrentSelection: () => providerService.getCurrentSelection(),
  clearCurrentSelection: () => providerService.clearCurrentSelection(),
  getProviderHealth: () => providerService.getProviderHealth(),
  getProviderHealthDetail: (provider: string) => providerService.getProviderHealthDetail(provider),
  testProviderConnection: (provider: string) => providerService.testProviderConnection(provider),
  checkRateLimit: (provider: string, model: string) => providerService.checkRateLimit(provider, model),
  setDefaultProvider: (provider_id: string, model_id?: string) =>
    providerService.setDefaultProvider(provider_id, model_id),

  // WebSocket methods
  createWebSocket: () => providerService.createWebSocketConnection(),
  subscribeToSelectionChanges: (callback: (event: any) => void) =>
    providerService.subscribeToSelectionChanges(callback),
  subscribeToHealthUpdates: (callback: (health: ProviderHealthStatus[]) => void) =>
    providerService.subscribeToHealthUpdates(callback),
  pingWebSocket: () => providerService.pingWebSocket(),
  closeWebSocket: () => providerService.closeWebSocket(),
  getWebSocketStatus: () => providerService.getWebSocketStatus(),
};

export default providerService;