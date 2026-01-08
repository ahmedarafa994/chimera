"use client";

/**
 * Provider Management Hooks for Project Chimera Frontend
 * 
 * React hooks for provider management functionality including:
 * - Provider listing
 * - Model selection
 * - Health monitoring
 * - WebSocket sync
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { providerManagementService } from "@/lib/services/provider-management-service";
import {
  ProviderInfo,
  ModelInfo,
  CurrentSelectionResponse,
  AllProvidersHealthResponse,
  RateLimitInfo,
  ProviderStatus,
} from "@/lib/types/provider-management-types";
import {
  ProviderSyncManager,
  getProviderSyncManager,
} from "@/lib/websocket/provider-sync";
import {
  ModelUpdatesManager,
  getModelUpdatesManager,
} from "@/lib/websocket/model-updates";

// =============================================================================
// Providers Hook
// =============================================================================

export interface UseProvidersReturn {
  providers: ProviderInfo[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  getProvider: (providerId: string) => ProviderInfo | undefined;
}

export function useProviders(): UseProvidersReturn {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await providerManagementService.listAvailableProviders();
      setProviders(response.providers);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load providers");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getProvider = useCallback((providerId: string): ProviderInfo | undefined => {
    return providers.find((p) => p.provider_id === providerId);
  }, [providers]);

  // Load on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    providers,
    isLoading,
    error,
    refresh,
    getProvider,
  };
}

// =============================================================================
// Models Hook
// =============================================================================

export interface UseModelsReturn {
  modelsByProvider: Map<string, ModelInfo[]>;
  isLoading: boolean;
  error: string | null;
  loadModelsForProvider: (providerId: string) => Promise<void>;
  getModels: (providerId: string) => ModelInfo[];
  findModel: (modelId: string) => { provider: ProviderInfo; model: ModelInfo } | null;
}

export function useModels(): UseModelsReturn {
  const [modelsByProvider, setModelsByProvider] = useState<Map<string, ModelInfo[]>>(new Map());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { providers } = useProviders();

  const loadModelsForProvider = useCallback(async (providerId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await providerManagementService.getProviderModels(providerId);
      setModelsByProvider((prev) => {
        const updated = new Map(prev);
        updated.set(providerId, response.models);
        return updated;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getModels = useCallback((providerId: string): ModelInfo[] => {
    return modelsByProvider.get(providerId) || [];
  }, [modelsByProvider]);

  const findModel = useCallback((modelId: string): { provider: ProviderInfo; model: ModelInfo } | null => {
    for (const [providerId, models] of modelsByProvider.entries()) {
      const model = models.find((m) => m.model_id === modelId);
      if (model) {
        const provider = providers.find((p) => p.provider_id === providerId);
        if (provider) {
          return { provider, model };
        }
      }
    }
    return null;
  }, [modelsByProvider, providers]);

  return {
    modelsByProvider,
    isLoading,
    error,
    loadModelsForProvider,
    getModels,
    findModel,
  };
}

// =============================================================================
// Model Selection Hook
// =============================================================================

export interface UseModelSelectionReturn {
  currentSelection: CurrentSelectionResponse | null;
  isLoading: boolean;
  error: string | null;
  selectModel: (providerId: string, modelId: string) => Promise<void>;
  clearSelection: () => Promise<void>;
  refresh: () => Promise<void>;
}

export function useModelSelection(): UseModelSelectionReturn {
  const [currentSelection, setCurrentSelection] = useState<CurrentSelectionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await providerManagementService.getCurrentSelection();
      setCurrentSelection(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load selection");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const selectModel = useCallback(async (providerId: string, modelId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      await providerManagementService.selectModel({
        provider_id: providerId,
        model_id: modelId,
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to select model");
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [refresh]);

  const clearSelection = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      await providerManagementService.clearSelection();
      setCurrentSelection(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear selection");
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Load on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    currentSelection,
    isLoading,
    error,
    selectModel,
    clearSelection,
    refresh,
  };
}

// =============================================================================
// Provider Health Hook
// =============================================================================

export interface UseProviderHealthReturn {
  healthStatus: AllProvidersHealthResponse | null;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  getProviderHealth: (providerId: string) => ProviderStatus | null;
  isHealthy: (providerId: string) => boolean;
}

export function useProviderHealth(): UseProviderHealthReturn {
  const [healthStatus, setHealthStatus] = useState<AllProvidersHealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await providerManagementService.getAllProvidersHealth();
      setHealthStatus(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load health status");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getProviderHealth = useCallback((providerId: string): ProviderStatus | null => {
    if (!healthStatus) return null;
    const provider = healthStatus.providers.find((p) => p.provider_id === providerId);
    return provider?.status || null;
  }, [healthStatus]);

  const isHealthy = useCallback((providerId: string): boolean => {
    const status = getProviderHealth(providerId);
    return status === ProviderStatus.AVAILABLE;
  }, [getProviderHealth]);

  // Load on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    healthStatus,
    isLoading,
    error,
    refresh,
    getProviderHealth,
    isHealthy,
  };
}

// =============================================================================
// Provider Sync Hook (WebSocket)
// =============================================================================

export interface UseProviderSyncReturn {
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
  broadcastSelection: (providerId: string, modelId: string) => void;
  broadcastClear: () => void;
}

export function useProviderSync(
  onSelectionUpdate?: (providerId: string, modelId: string) => void,
  onSelectionCleared?: () => void
): UseProviderSyncReturn {
  const [isConnected, setIsConnected] = useState(false);
  const managerRef = useRef<ProviderSyncManager | null>(null);

  const connect = useCallback(() => {
    if (managerRef.current) {
      managerRef.current.disconnect();
    }

    managerRef.current = getProviderSyncManager({
      onSelectionUpdate,
      onSelectionCleared,
      onConnect: () => setIsConnected(true),
      onDisconnect: () => setIsConnected(false),
    });

    managerRef.current.connect();
  }, [onSelectionUpdate, onSelectionCleared]);

  const disconnect = useCallback(() => {
    if (managerRef.current) {
      managerRef.current.disconnect();
      managerRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const broadcastSelection = useCallback((providerId: string, modelId: string) => {
    if (managerRef.current) {
      managerRef.current.broadcastSelectionUpdate(providerId, modelId);
    }
  }, []);

  const broadcastClear = useCallback(() => {
    if (managerRef.current) {
      managerRef.current.broadcastSelectionCleared();
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (managerRef.current) {
        managerRef.current.disconnect();
      }
    };
  }, []);

  return {
    isConnected,
    connect,
    disconnect,
    broadcastSelection,
    broadcastClear,
  };
}

// =============================================================================
// Model Updates Hook (WebSocket)
// =============================================================================

export interface UseModelUpdatesReturn {
  isConnected: boolean;
  connect: () => void;
  disconnect: () => void;
  setProviderFilter: (providers: string[]) => void;
}

export function useModelUpdates(
  onModelAdded?: (providerId: string, model: ModelInfo) => void,
  onModelRemoved?: (providerId: string, modelId: string) => void,
  onProviderStatusChanged?: (providerId: string, status: ProviderStatus) => void
): UseModelUpdatesReturn {
  const [isConnected, setIsConnected] = useState(false);
  const managerRef = useRef<ModelUpdatesManager | null>(null);

  const connect = useCallback(() => {
    if (managerRef.current) {
      managerRef.current.disconnect();
    }

    managerRef.current = getModelUpdatesManager({
      onModelAdded,
      onModelRemoved,
      onProviderStatusChanged,
      onConnect: () => setIsConnected(true),
      onDisconnect: () => setIsConnected(false),
    });

    managerRef.current.connect();
  }, [onModelAdded, onModelRemoved, onProviderStatusChanged]);

  const disconnect = useCallback(() => {
    if (managerRef.current) {
      managerRef.current.disconnect();
      managerRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const setProviderFilter = useCallback((providers: string[]) => {
    if (managerRef.current) {
      managerRef.current.setProviderFilter(providers);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (managerRef.current) {
        managerRef.current.disconnect();
      }
    };
  }, []);

  return {
    isConnected,
    connect,
    disconnect,
    setProviderFilter,
  };
}