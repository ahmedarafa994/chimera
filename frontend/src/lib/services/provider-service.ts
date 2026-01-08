"use client";

/**
 * Provider Service
 *
 * Manages LLM providers, their health status, and selection logic.
 * Integrates with the new API Client and Circuit Breaker system.
 *
 * @module lib/services/provider-service
 */

import { apiClient, ENDPOINTS } from '../api/core';
import { Provider, ProviderName, ProviderStatus } from '../api/core/types';
import { circuitBreakerRegistry, CircuitState } from '../resilience/circuit-breaker';
import { handleApiError } from '../errors';

export interface ProviderRegistration {
  name: ProviderName;
  displayName: string;
  enabled: boolean;
  capabilities: {
    chat: boolean;
    completion: boolean;
    streaming: boolean;
  };
}

class ProviderService {
  private providers: Map<string, Provider> = new Map();
  private activeProviderId: string | null = null;
  private listeners: Set<(providers: Provider[]) => void> = new Set();
  private refreshInterval: NodeJS.Timeout | null = null;

  constructor() {
    // Basic auto-refresh every 5 minutes by default if in browser
    if (typeof window !== 'undefined') {
      this.startAutoRefresh(300000);
    }
  }

  /**
   * Register a provider or update an existing one
   */
  registerProvider(registration: ProviderRegistration): void {
    const existing = this.providers.get(registration.name);

    const provider: Provider = {
      id: registration.name,
      name: registration.name,
      displayName: registration.displayName,
      enabled: registration.enabled,
      configured: true, // If we're registering it here, assume it's configured
      models: existing?.models || [],
      capabilities: {
        chat: registration.capabilities.chat,
        completion: registration.capabilities.completion,
        embedding: false,
        imageGeneration: false,
        codeGeneration: registration.capabilities.chat, // Assume code if chat
        streaming: registration.capabilities.streaming,
        functionCalling: registration.capabilities.chat,
      },
      status: existing?.status || { available: true },
    };

    this.providers.set(registration.name, provider);
    this.notify();
  }

  /**
   * Fetch all providers from the backend
   */
  async refresh(): Promise<Provider[]> {
    try {
      const data = await apiClient.get<any[]>(ENDPOINTS.PROVIDERS_AVAILABLE);

      // Clear current providers and rebuild from backend data
      const newProviders = new Map<string, Provider>();

      data.forEach((p: any) => {
        const provider: Provider = {
          id: p.provider,
          name: p.provider as ProviderName,
          displayName: p.display_name || p.provider,
          enabled: p.enabled ?? true,
          configured: p.is_configured ?? true,
          models: (p.models || []).map((m: any) => ({
            id: typeof m === 'string' ? m : m.id,
            name: typeof m === 'string' ? m : m.name,
            provider: p.provider as ProviderName,
            displayName: typeof m === 'string' ? m : m.display_name || m.name,
            contextWindow: m.context_window || 4096,
            maxOutputTokens: m.max_output_tokens || 2048,
            inputPricePerToken: 0,
            outputPricePerToken: 0,
            capabilities: {
              chat: true,
              completion: true,
              embedding: false,
              vision: false,
              functionCalling: true,
              jsonMode: true,
              streaming: true,
            }
          })),
          capabilities: {
            chat: true,
            completion: true,
            embedding: false,
            imageGeneration: false,
            codeGeneration: true,
            streaming: true,
            functionCalling: true,
          },
          status: {
            available: p.is_healthy ?? true,
            latencyMs: p.latency_ms,
            lastCheck: p.last_check,
          },
        };
        newProviders.set(provider.id, provider);
      });

      this.providers = newProviders;
      this.notify();
      return this.list();
    } catch (error) {
      handleApiError(error, { context: 'ProviderService.refresh' });
      return this.list();
    }
  }

  /**
   * List all registered providers
   */
  list(): Provider[] {
    return Array.from(this.providers.values()).map(p => this.enrichStatus(p));
  }

  /**
   * Get a specific provider by ID
   */
  get(id: string): Provider | undefined {
    const provider = this.providers.get(id);
    return provider ? this.enrichStatus(provider) : undefined;
  }

  /**
   * Get the currently active provider
   */
  getActive(): Provider | undefined {
    if (!this.activeProviderId) return undefined;
    return this.get(this.activeProviderId);
  }

  /**
   * Resolve a provider with fallback logic
   */
  resolve(preference?: string): Provider | undefined {
    // 1. Try preference
    if (preference) {
      const p = this.get(preference);
      if (p && p.status.available) return p;
    }

    // 2. Try active provider
    if (this.activeProviderId) {
      const p = this.get(this.activeProviderId);
      if (p && p.status.available) return p;
    }

    // 3. Try any healthy provider
    const healthy = this.list().find(p => p.status.available && p.enabled);
    if (healthy) return healthy;

    // 4. Return first enabled provider even if unhealthy (last resort)
    return this.list().find(p => p.enabled);
  }

  /**
   * Set the active provider
   */
  async setActive(id: string, model?: string): Promise<boolean> {
    try {
      const p = this.get(id);
      if (!p) return false;

      const targetModel = model || p.models[0]?.id || 'default';

      await apiClient.post(ENDPOINTS.PROVIDERS_SELECT, {
        provider: p.name,
        model: targetModel,
      });

      this.activeProviderId = id;
      this.notify();
      return true;
    } catch (error) {
      handleApiError(error, { context: 'ProviderService.setActive' });
      return false;
    }
  }

  /**
   * Enrich provider status with frontend circuit breaker state
   */
  private enrichStatus(provider: Provider): Provider {
    // Check if there's an open circuit for this provider on the client side
    const circuitName = `provider:${provider.name}`;
    const stats = circuitBreakerRegistry.getStats(circuitName);

    if (stats && stats.state === CircuitState.OPEN) {
      return {
        ...provider,
        status: {
          ...provider.status,
          available: false,
          error: 'Circuit breaker is OPEN (high failure rate detected)',
        }
      };
    }

    return provider;
  }

  /**
   * Subscribe to provider updates
   */
  subscribe(listener: (providers: Provider[]) => void): () => void {
    this.listeners.add(listener);
    listener(this.list());
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    const list = this.list();
    this.listeners.forEach(l => l(list));
  }

  private startAutoRefresh(interval: number): void {
    if (this.refreshInterval) clearInterval(this.refreshInterval);
    this.refreshInterval = setInterval(() => this.refresh(), interval);
  }

  stopAutoRefresh(): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
  }
}

export const providerService = new ProviderService();
export default providerService;
