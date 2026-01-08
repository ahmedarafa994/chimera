"use client";

/**
 * useProviderModels Hook
 *
 * A React hook for fetching and managing AI provider and model data.
 * Features:
 * - Automatic data fetching on mount
 * - Caching with configurable TTL
 * - Loading and error states
 * - Refresh functionality
 * - Provider/model selection state management
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import { enhancedApi, Provider, ProvidersListResponse } from "@/lib/api-enhanced";

// ============================================================================
// Types
// ============================================================================

export interface ProviderModel {
  id: string;
  name: string;
  provider: string;
  isDefault: boolean;
}

export interface ProviderWithModels {
  id: string;
  name: string;
  displayName: string;
  status: "active" | "inactive" | "unknown";
  models: ProviderModel[];
  defaultModel: string | null;
  modelCount: number;
}

export interface UseProviderModelsOptions {
  /** Auto-fetch on mount */
  autoFetch?: boolean;
  /** Cache TTL in milliseconds (default: 5 minutes) */
  cacheTTL?: number;
  /** Initial selected provider */
  initialProvider?: string;
  /** Initial selected model */
  initialModel?: string;
  /** Callback when selection changes */
  onSelectionChange?: (provider: string, model: string) => void;
}

export interface UseProviderModelsReturn {
  /** List of providers with their models */
  providers: ProviderWithModels[];
  /** Flat list of all models */
  allModels: ProviderModel[];
  /** Currently selected provider ID */
  selectedProvider: string | null;
  /** Currently selected model ID */
  selectedModel: string | null;
  /** Default provider from backend */
  defaultProvider: string | null;
  /** Default model from backend */
  defaultModel: string | null;
  /** Loading state */
  isLoading: boolean;
  /** Refreshing state (for subsequent fetches) */
  isRefreshing: boolean;
  /** Error message if any */
  error: string | null;
  /** Total number of providers */
  providerCount: number;
  /** Total number of models */
  modelCount: number;
  /** Select a provider (auto-selects default model) */
  selectProvider: (providerId: string) => void;
  /** Select a model */
  selectModel: (modelId: string, providerId?: string) => void;
  /** Refresh data from backend */
  refresh: () => Promise<void>;
  /** Get models for a specific provider */
  getModelsForProvider: (providerId: string) => ProviderModel[];
  /** Get provider info by ID */
  getProvider: (providerId: string) => ProviderWithModels | undefined;
  /** Check if a model is available */
  isModelAvailable: (modelId: string, providerId?: string) => boolean;
  /** Reset selection to defaults */
  resetToDefaults: () => void;
}

// ============================================================================
// Cache Management
// ============================================================================

interface CacheEntry {
  data: ProvidersListResponse;
  timestamp: number;
}

let globalCache: CacheEntry | null = null;

function isCacheValid(cacheTTL: number): boolean {
  if (!globalCache) return false;
  return Date.now() - globalCache.timestamp < cacheTTL;
}

// ============================================================================
// Provider Display Names
// ============================================================================

const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  // Gemini AI Models
  google: "Gemini AI",
  gemini: "Gemini AI",
  "gemini-cli": "Gemini CLI (OAuth)",
  "gemini-openai": "Gemini (OpenAI Compatible)",
  // Hybrid Models
  antigravity: "Antigravity (Hybrid)",
  // OpenAI Models
  openai: "OpenAI",
  // Anthropic Claude Models
  anthropic: "Anthropic Claude",
  kiro: "Kiro (Claude)",
  // Other Providers
  qwen: "Qwen",
  deepseek: "DeepSeek",
  cursor: "Cursor",
};

function getDisplayName(providerId: string): string {
  return PROVIDER_DISPLAY_NAMES[providerId.toLowerCase()] ||
    providerId.charAt(0).toUpperCase() + providerId.slice(1);
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useProviderModels(options: UseProviderModelsOptions = {}): UseProviderModelsReturn {
  const {
    autoFetch = true,
    cacheTTL = 5 * 60 * 1000, // 5 minutes
    initialProvider,
    initialModel,
    onSelectionChange,
  } = options;

  // State
  const [rawProviders, setRawProviders] = useState<Provider[]>([]);
  const [defaultProvider, setDefaultProvider] = useState<string | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(initialProvider || null);
  const [selectedModel, setSelectedModel] = useState<string | null>(initialModel || null);
  const [isLoading, setIsLoading] = useState(autoFetch);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Transform raw providers to enriched format
  const providers: ProviderWithModels[] = useMemo(() => {
    return rawProviders.map((provider) => ({
      id: provider.provider,
      name: provider.provider,
      displayName: getDisplayName(provider.provider),
      status: provider.status as "active" | "inactive" | "unknown",
      models: provider.available_models.map((model) => ({
        id: model,
        name: model,
        provider: provider.provider,
        isDefault: model === provider.model,
      })),
      defaultModel: provider.model || null,
      modelCount: provider.available_models.length,
    }));
  }, [rawProviders]);

  // Flat list of all models
  const allModels: ProviderModel[] = useMemo(() => {
    return providers.flatMap((p) => p.models);
  }, [providers]);

  // Default model (from default provider)
  const defaultModel = useMemo(() => {
    if (!defaultProvider) return null;
    const provider = providers.find((p) => p.id === defaultProvider);
    return provider?.defaultModel || null;
  }, [defaultProvider, providers]);

  // Fetch providers from API
  const fetchProviders = useCallback(async (forceRefresh = false) => {
    // Check cache first
    if (!forceRefresh && isCacheValid(cacheTTL) && globalCache) {
      const data = globalCache.data;
      setRawProviders(data.providers);
      setDefaultProvider(data.default);

      // Set initial selection if not already set
      if (!selectedProvider && data.default) {
        setSelectedProvider(data.default);
        const defaultProv = data.providers.find((p) => p.provider === data.default);
        if (defaultProv && !selectedModel) {
          setSelectedModel(defaultProv.model || defaultProv.available_models[0] || null);
        }
      }

      setIsLoading(false);
      return;
    }

    try {
      setError(null);
      if (!isLoading) setIsRefreshing(true);

      const response = await enhancedApi.providers.list();
      const data = response.data;

      // Update cache
      globalCache = {
        data,
        timestamp: Date.now(),
      };

      setRawProviders(data.providers);
      setDefaultProvider(data.default);

      // Set initial selection if not already set
      if (!selectedProvider && data.default) {
        setSelectedProvider(data.default);
        const defaultProv = data.providers.find((p) => p.provider === data.default);
        if (defaultProv && !selectedModel) {
          const model = defaultProv.model || defaultProv.available_models[0] || null;
          setSelectedModel(model);
          if (model) {
            onSelectionChange?.(data.default, model);
          }
        }
      }
    } catch (err) {
      console.error("Failed to fetch providers:", err);
      setError(err instanceof Error ? err.message : "Failed to load providers");
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [cacheTTL, isLoading, selectedProvider, selectedModel, onSelectionChange]);

  // Auto-fetch on mount
  useEffect(() => {
    if (autoFetch) {
      fetchProviders();
    }
  }, [autoFetch, fetchProviders]);

  // Select provider
  const selectProvider = useCallback((providerId: string) => {
    const provider = providers.find((p) => p.id === providerId);
    if (!provider) {
      console.warn(`Provider '${providerId}' not found`);
      return;
    }

    setSelectedProvider(providerId);

    // Auto-select default model for this provider
    const model = provider.defaultModel || provider.models[0]?.id || null;
    setSelectedModel(model);

    if (model) {
      onSelectionChange?.(providerId, model);
    }
  }, [providers, onSelectionChange]);

  // Select model
  const selectModel = useCallback((modelId: string, providerId?: string) => {
    // Find which provider this model belongs to
    let targetProvider = providerId;

    if (!targetProvider) {
      const providerWithModel = providers.find((p) =>
        p.models.some((m) => m.id === modelId)
      );
      targetProvider = providerWithModel?.id;
    }

    if (!targetProvider) {
      console.warn(`Model '${modelId}' not found in any provider`);
      return;
    }

    setSelectedProvider(targetProvider);
    setSelectedModel(modelId);
    onSelectionChange?.(targetProvider, modelId);
  }, [providers, onSelectionChange]);

  // Get models for a specific provider
  const getModelsForProvider = useCallback((providerId: string): ProviderModel[] => {
    const provider = providers.find((p) => p.id === providerId);
    return provider?.models || [];
  }, [providers]);

  // Get provider by ID
  const getProvider = useCallback((providerId: string): ProviderWithModels | undefined => {
    return providers.find((p) => p.id === providerId);
  }, [providers]);

  // Check if model is available
  const isModelAvailable = useCallback((modelId: string, providerId?: string): boolean => {
    if (providerId) {
      const provider = providers.find((p) => p.id === providerId);
      return provider?.models.some((m) => m.id === modelId) || false;
    }
    return allModels.some((m) => m.id === modelId);
  }, [providers, allModels]);

  // Reset to defaults
  const resetToDefaults = useCallback(() => {
    if (defaultProvider) {
      setSelectedProvider(defaultProvider);
      const provider = providers.find((p) => p.id === defaultProvider);
      const model = provider?.defaultModel || provider?.models[0]?.id || null;
      setSelectedModel(model);
      if (model) {
        onSelectionChange?.(defaultProvider, model);
      }
    }
  }, [defaultProvider, providers, onSelectionChange]);

  // Refresh function
  const refresh = useCallback(async () => {
    await fetchProviders(true);
  }, [fetchProviders]);

  return {
    providers,
    allModels,
    selectedProvider,
    selectedModel,
    defaultProvider,
    defaultModel,
    isLoading,
    isRefreshing,
    error,
    providerCount: providers.length,
    modelCount: allModels.length,
    selectProvider,
    selectModel,
    refresh,
    getModelsForProvider,
    getProvider,
    isModelAvailable,
    resetToDefaults,
  };
}

export default useProviderModels;
