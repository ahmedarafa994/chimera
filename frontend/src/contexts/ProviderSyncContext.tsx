'use client';

/**
 * Provider Sync Context
 *
 * React context for managing provider/model synchronization state
 * with real-time updates and automatic reconnection.
 */

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
} from 'react';
import {
  SyncStatus,
  SyncState,
  SyncEvent,
  SyncEventType,
  ProviderSyncInfo,
  ModelSpecification,
  ProviderSyncConfig,
  ProviderAvailabilityInfo,
  ModelAvailabilityInfo,
  DEFAULT_SYNC_CONFIG,
} from '@/types/provider-sync';
import {
  ProviderSyncService,
  getProviderSyncService,
  destroyProviderSyncService,
} from '@/lib/sync/provider-sync-service';

// =============================================================================
// Context Types
// =============================================================================

interface ProviderSyncContextValue {
  // State
  status: SyncStatus;
  isConnected: boolean;
  isLoading: boolean;
  error: string | undefined;
  version: number;
  lastSyncTime: Date | undefined;

  // Providers
  providers: ProviderSyncInfo[];
  getProvider: (id: string) => ProviderSyncInfo | undefined;
  activeProvider: ProviderSyncInfo | undefined;
  activeProviderId: string | undefined;

  // Models
  models: ModelSpecification[];
  getModel: (id: string) => ModelSpecification | undefined;
  getProviderModels: (providerId: string) => ModelSpecification[];
  activeModel: ModelSpecification | undefined;
  activeModelId: string | undefined;

  // Actions
  forceSync: () => Promise<void>;
  getProviderAvailability: (providerId: string) => Promise<ProviderAvailabilityInfo | null>;
  getModelAvailability: (modelId: string, providerId?: string) => Promise<ModelAvailabilityInfo | null>;

  // Selection Actions
  selectProvider: (providerId: string, modelId?: string, options?: { persist?: boolean }) => Promise<{
    success: boolean;
    providerId: string;
    modelId?: string;
    fallbackApplied?: boolean;
    fallbackReason?: string;
    error?: string;
  }>;
  selectModel: (modelId: string, options?: { persist?: boolean }) => Promise<{
    success: boolean;
    modelId: string;
    providerId?: string;
    error?: string;
  }>;
}

const ProviderSyncContext = createContext<ProviderSyncContextValue | null>(null);

// =============================================================================
// Provider Component
// =============================================================================

interface ProviderSyncProviderProps {
  children: React.ReactNode;
  config?: Partial<ProviderSyncConfig>;
  onSyncEvent?: (event: SyncEvent) => void;
  onConnectionChange?: (connected: boolean) => void;
}

export function ProviderSyncProvider({
  children,
  config,
  onSyncEvent,
  onConnectionChange,
}: ProviderSyncProviderProps) {
  // Service reference
  const serviceRef = useRef<ProviderSyncService | null>(null);
  const initializedRef = useRef(false);

  // State
  const [status, setStatus] = useState<SyncStatus>(SyncStatus.DISCONNECTED);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();
  const [version, setVersion] = useState(0);
  const [lastSyncTime, setLastSyncTime] = useState<Date | undefined>();

  // Data state
  const [providers, setProviders] = useState<ProviderSyncInfo[]>([]);
  const [models, setModels] = useState<ModelSpecification[]>([]);
  const [activeProviderId, setActiveProviderId] = useState<string | undefined>();
  const [activeModelId, setActiveModelId] = useState<string | undefined>();

  // Initialize service
  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;

    const service = getProviderSyncService(config);
    serviceRef.current = service;

    // Set up event handlers
    service.setHandlers({
      onFullSync: (event) => {
        const state = event.data as SyncState;
        setProviders(state.providers);
        setModels(state.all_models);
        setActiveProviderId(state.active_provider_id);
        setActiveModelId(state.active_model_id);
        setVersion(state.metadata.version);
        setLastSyncTime(new Date(state.metadata.last_sync_time));
        setStatus(SyncStatus.SYNCED);
        setIsLoading(false);
        setError(undefined);
        onSyncEvent?.(event);
      },

      onProviderAdded: (event) => {
        const { provider } = event.data as { provider: ProviderSyncInfo };
        setProviders((prev) => [...prev, provider]);
        setModels((prev) => [...prev, ...provider.models]);
        setVersion(event.version);
        onSyncEvent?.(event);
      },

      onProviderUpdated: (event) => {
        const { provider } = event.data as { provider: ProviderSyncInfo };
        setProviders((prev) =>
          prev.map((p) => (p.id === provider.id ? provider : p))
        );
        // Update models for this provider
        setModels((prev) => {
          const otherModels = prev.filter((m) => m.provider_id !== provider.id);
          return [...otherModels, ...provider.models];
        });
        setVersion(event.version);
        onSyncEvent?.(event);
      },

      onProviderRemoved: (event) => {
        const providerId = event.provider_id;
        if (providerId) {
          setProviders((prev) => prev.filter((p) => p.id !== providerId));
          setModels((prev) => prev.filter((m) => m.provider_id !== providerId));
          setVersion(event.version);
        }
        onSyncEvent?.(event);
      },

      onProviderStatusChanged: (event) => {
        const { new_status, health } = event.data as {
          new_status: string;
          health?: ProviderSyncInfo['health'];
        };
        const providerId = event.provider_id;
        if (providerId) {
          setProviders((prev) =>
            prev.map((p) =>
              p.id === providerId ? { ...p, health } : p
            )
          );
        }
        onSyncEvent?.(event);
      },

      onModelDeprecated: (event) => {
        const modelId = event.model_id;
        const { deprecation_date, sunset_date, replacement_model_id } = event.data as {
          deprecation_date: string;
          sunset_date?: string;
          replacement_model_id?: string;
        };
        if (modelId) {
          setModels((prev) =>
            prev.map((m) =>
              m.id === modelId
                ? { ...m, deprecation_date, sunset_date, replacement_model_id }
                : m
            )
          );
        }
        onSyncEvent?.(event);
      },

      onActiveProviderChanged: (event) => {
        if (event.provider_id) {
          setActiveProviderId(event.provider_id);
        }
        onSyncEvent?.(event);
      },

      onHeartbeat: (event) => {
        // Just update connection status
        setIsConnected(true);
      },

      onError: (event) => {
        const { message } = event.data as { message: string };
        setError(message);
        setStatus(SyncStatus.ERROR);
        onSyncEvent?.(event);
      },

      onConnectionChange: (connected) => {
        setIsConnected(connected);
        if (!connected) {
          setStatus(SyncStatus.DISCONNECTED);
        }
        onConnectionChange?.(connected);
      },
    });

    // Initialize the service
    service.initialize().catch((err) => {
      console.error('[ProviderSyncContext] Initialization failed:', err);
      setError(err.message);
      setStatus(SyncStatus.ERROR);
      setIsLoading(false);
    });

    // Cleanup on unmount
    return () => {
      destroyProviderSyncService();
    };
  }, [config, onSyncEvent, onConnectionChange]);

  // Memoized getters
  const getProvider = useCallback(
    (id: string) => providers.find((p) => p.id === id),
    [providers]
  );

  const getModel = useCallback(
    (id: string) => models.find((m) => m.id === id),
    [models]
  );

  const getProviderModels = useCallback(
    (providerId: string) => models.filter((m) => m.provider_id === providerId),
    [models]
  );

  const activeProvider = useMemo(
    () => (activeProviderId ? getProvider(activeProviderId) : undefined),
    [activeProviderId, getProvider]
  );

  const activeModel = useMemo(
    () => (activeModelId ? getModel(activeModelId) : undefined),
    [activeModelId, getModel]
  );

  // Actions
  const forceSync = useCallback(async () => {
    setIsLoading(true);
    try {
      await serviceRef.current?.forceSync();
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getProviderAvailability = useCallback(
    async (providerId: string) => {
      return serviceRef.current?.getProviderAvailability(providerId) ?? null;
    },
    []
  );

  const getModelAvailability = useCallback(
    async (modelId: string, providerId?: string) => {
      return serviceRef.current?.getModelAvailability(modelId, providerId) ?? null;
    },
    []
  );

  // Selection actions with optimistic updates
  const selectProvider = useCallback(
    async (
      providerId: string,
      modelId?: string,
      options?: { persist?: boolean }
    ) => {
      const previousProviderId = activeProviderId;
      const previousModelId = activeModelId;

      // Optimistic update
      setActiveProviderId(providerId);
      if (modelId) {
        setActiveModelId(modelId);
      }

      const result = await serviceRef.current?.selectProvider(providerId, modelId, options);

      if (result?.success) {
        // Update to actual values from server
        setActiveProviderId(result.providerId);
        if (result.modelId) {
          setActiveModelId(result.modelId);
        }
        return result;
      } else {
        // Rollback on failure
        setActiveProviderId(previousProviderId);
        setActiveModelId(previousModelId);
        return result || {
          success: false,
          providerId,
          error: 'Service not available',
        };
      }
    },
    [activeProviderId, activeModelId]
  );

  const selectModel = useCallback(
    async (modelId: string, options?: { persist?: boolean }) => {
      const previousModelId = activeModelId;

      // Optimistic update
      setActiveModelId(modelId);

      const result = await serviceRef.current?.selectModel(modelId, options);

      if (result?.success) {
        setActiveModelId(result.modelId);
        return result;
      } else {
        // Rollback on failure
        setActiveModelId(previousModelId);
        return result || {
          success: false,
          modelId,
          error: 'Service not available',
        };
      }
    },
    [activeModelId]
  );

  // Context value
  const value = useMemo<ProviderSyncContextValue>(
    () => ({
      status,
      isConnected,
      isLoading,
      error,
      version,
      lastSyncTime,
      providers,
      getProvider,
      activeProvider,
      activeProviderId,
      models,
      getModel,
      getProviderModels,
      activeModel,
      activeModelId,
      forceSync,
      getProviderAvailability,
      getModelAvailability,
      selectProvider,
      selectModel,
    }),
    [
      status,
      isConnected,
      isLoading,
      error,
      version,
      lastSyncTime,
      providers,
      getProvider,
      activeProvider,
      activeProviderId,
      models,
      getModel,
      getProviderModels,
      activeModel,
      activeModelId,
      forceSync,
      getProviderAvailability,
      getModelAvailability,
      selectProvider,
      selectModel,
    ]
  );

  return (
    <ProviderSyncContext.Provider value={value}>
      {children}
    </ProviderSyncContext.Provider>
  );
}

// =============================================================================
// Hook
// =============================================================================

export function useProviderSync(): ProviderSyncContextValue {
  const context = useContext(ProviderSyncContext);

  if (!context) {
    throw new Error('useProviderSync must be used within a ProviderSyncProvider');
  }

  return context;
}

export function useProviderSyncSafe(): ProviderSyncContextValue | null {
  const context = useContext(ProviderSyncContext);
  return context ?? null;
}

export { useProviderSyncSafe };

// =============================================================================
// Selector Hooks
// =============================================================================

/**
 * Hook to get providers with optional filtering
 */
export function useProviders(options?: {
  enabledOnly?: boolean;
  configuredOnly?: boolean;
}) {
  const { providers } = useProviderSync();

  return useMemo(() => {
    let filtered = providers;

    if (options?.enabledOnly) {
      filtered = filtered.filter((p) => p.enabled);
    }

    if (options?.configuredOnly) {
      filtered = filtered.filter((p) => p.is_configured);
    }

    return filtered;
  }, [providers, options?.enabledOnly, options?.configuredOnly]);
}

/**
 * Hook to get models with optional filtering
 */
export function useModels(options?: {
  providerId?: string;
  availableOnly?: boolean;
  excludeDeprecated?: boolean;
}) {
  const { models, getProviderModels } = useProviderSync();

  return useMemo(() => {
    let filtered = options?.providerId
      ? getProviderModels(options.providerId)
      : models;

    if (options?.availableOnly) {
      filtered = filtered.filter((m) => m.is_available);
    }

    if (options?.excludeDeprecated) {
      filtered = filtered.filter((m) => m.deprecation_status === 'active');
    }

    return filtered;
  }, [models, getProviderModels, options?.providerId, options?.availableOnly, options?.excludeDeprecated]);
}

/**
 * Hook to get sync status information
 */
export function useSyncStatus() {
  const { status, isConnected, isLoading, error, version, lastSyncTime } = useProviderSync();

  return useMemo(
    () => ({
      status,
      isConnected,
      isLoading,
      error,
      version,
      lastSyncTime,
      isSynced: status === SyncStatus.SYNCED,
      isStale: status === SyncStatus.STALE,
      hasError: status === SyncStatus.ERROR,
    }),
    [status, isConnected, isLoading, error, version, lastSyncTime]
  );
}

/**
 * Hook to get active provider and model
 */
export function useActiveSelection() {
  const { activeProvider, activeModel, activeProviderId, activeModelId } = useProviderSync();

  return useMemo(
    () => ({
      provider: activeProvider,
      model: activeModel,
      providerId: activeProviderId,
      modelId: activeModelId,
    }),
    [activeProvider, activeModel, activeProviderId, activeModelId]
  );
}
