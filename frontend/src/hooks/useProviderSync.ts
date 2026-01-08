/**
 * Provider Sync Hooks
 * 
 * React Query hooks for provider/model synchronization with:
 * - Automatic caching and background refetching
 * - Optimistic updates
 * - Error handling and retry logic
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SyncStatus,
  SyncState,
  SyncRequest,
  SyncResponse,
  ProviderSyncInfo,
  ModelSpecification,
  ProviderAvailabilityInfo,
  ModelAvailabilityInfo,
  SyncEventType,
  WebSocketMessage,
} from '@/types/provider-sync';

// =============================================================================
// API Functions
// =============================================================================

const getApiBaseUrl = () => {
  return process.env.NEXT_PUBLIC_CHIMERA_API_URL || 'http://localhost:8001/api';
};

async function fetchSyncState(includeDeprecated = false): Promise<SyncState> {
  const response = await fetch(
    `${getApiBaseUrl()}/provider-sync/state?include_deprecated=${includeDeprecated}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to fetch sync state: ${response.status}`);
  }
  
  return response.json();
}

async function performSync(request: SyncRequest): Promise<SyncResponse> {
  const response = await fetch(`${getApiBaseUrl()}/provider-sync/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    throw new Error(`Sync failed: ${response.status}`);
  }
  
  return response.json();
}

async function fetchSyncVersion(): Promise<{ version: number; server_time: string }> {
  const response = await fetch(`${getApiBaseUrl()}/provider-sync/version`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch version: ${response.status}`);
  }
  
  return response.json();
}

async function fetchProviderAvailability(providerId: string): Promise<ProviderAvailabilityInfo> {
  const response = await fetch(
    `${getApiBaseUrl()}/provider-sync/providers/${providerId}/availability`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to fetch provider availability: ${response.status}`);
  }
  
  return response.json();
}

async function fetchModelAvailability(
  modelId: string,
  providerId?: string
): Promise<ModelAvailabilityInfo> {
  const params = new URLSearchParams();
  if (providerId) params.set('provider_id', providerId);
  
  const response = await fetch(
    `${getApiBaseUrl()}/provider-sync/models/${modelId}/availability?${params}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to fetch model availability: ${response.status}`);
  }
  
  return response.json();
}

async function fetchAllModels(options?: {
  providerId?: string;
  includeDeprecated?: boolean;
  capability?: string;
}): Promise<{ models: ModelSpecification[]; total: number }> {
  const params = new URLSearchParams();
  if (options?.providerId) params.set('provider_id', options.providerId);
  if (options?.includeDeprecated) params.set('include_deprecated', 'true');
  if (options?.capability) params.set('capability', options.capability);
  
  const response = await fetch(`${getApiBaseUrl()}/provider-sync/models?${params}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.status}`);
  }
  
  return response.json();
}

// =============================================================================
// Query Keys
// =============================================================================

export const providerSyncKeys = {
  all: ['provider-sync'] as const,
  state: () => [...providerSyncKeys.all, 'state'] as const,
  version: () => [...providerSyncKeys.all, 'version'] as const,
  providers: () => [...providerSyncKeys.all, 'providers'] as const,
  provider: (id: string) => [...providerSyncKeys.providers(), id] as const,
  providerAvailability: (id: string) => [...providerSyncKeys.provider(id), 'availability'] as const,
  models: () => [...providerSyncKeys.all, 'models'] as const,
  model: (id: string) => [...providerSyncKeys.models(), id] as const,
  modelAvailability: (id: string, providerId?: string) => 
    [...providerSyncKeys.model(id), 'availability', providerId] as const,
  providerModels: (providerId: string) => [...providerSyncKeys.models(), 'provider', providerId] as const,
};

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook to get the full sync state
 */
export function useSyncState(options?: { includeDeprecated?: boolean }) {
  return useQuery({
    queryKey: providerSyncKeys.state(),
    queryFn: () => fetchSyncState(options?.includeDeprecated),
    staleTime: 30000, // 30 seconds
    gcTime: 300000, // 5 minutes
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
  });
}

/**
 * Hook to get the current sync version
 */
export function useSyncVersion() {
  return useQuery({
    queryKey: providerSyncKeys.version(),
    queryFn: fetchSyncVersion,
    staleTime: 10000, // 10 seconds
    refetchInterval: 30000, // Poll every 30 seconds
  });
}

/**
 * Hook to perform a sync operation
 */
export function usePerformSync() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: performSync,
    onSuccess: (response) => {
      if (response.success && response.state) {
        // Update the cached state
        queryClient.setQueryData(providerSyncKeys.state(), response.state);
      }
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: providerSyncKeys.all });
    },
  });
}

/**
 * Hook to get all providers from sync state
 */
export function useProviderList() {
  const { data: state, ...rest } = useSyncState();
  
  const providers = useMemo(() => state?.providers ?? [], [state]);
  
  return {
    providers,
    activeProviderId: state?.active_provider_id,
    defaultProviderId: state?.default_provider_id,
    ...rest,
  };
}

/**
 * Hook to get a specific provider
 */
export function useProvider(providerId: string) {
  const { providers, ...rest } = useProviderList();
  
  const provider = useMemo(
    () => providers.find((p) => p.id === providerId),
    [providers, providerId]
  );
  
  return { provider, ...rest };
}

/**
 * Hook to get provider availability
 */
export function useProviderAvailability(providerId: string, enabled = true) {
  return useQuery({
    queryKey: providerSyncKeys.providerAvailability(providerId),
    queryFn: () => fetchProviderAvailability(providerId),
    enabled: enabled && !!providerId,
    staleTime: 10000, // 10 seconds
    refetchInterval: 30000, // Poll every 30 seconds
  });
}

/**
 * Hook to get all models
 */
export function useModelList(options?: {
  providerId?: string;
  includeDeprecated?: boolean;
  capability?: string;
}) {
  return useQuery({
    queryKey: options?.providerId 
      ? providerSyncKeys.providerModels(options.providerId)
      : providerSyncKeys.models(),
    queryFn: () => fetchAllModels(options),
    staleTime: 30000,
  });
}

/**
 * Hook to get models from sync state
 */
export function useModelsFromState(options?: {
  providerId?: string;
  availableOnly?: boolean;
  excludeDeprecated?: boolean;
}) {
  const { data: state, ...rest } = useSyncState();
  
  const models = useMemo(() => {
    let filtered = state?.all_models ?? [];
    
    if (options?.providerId) {
      filtered = filtered.filter((m) => m.provider_id === options.providerId);
    }
    
    if (options?.availableOnly) {
      filtered = filtered.filter((m) => m.is_available);
    }
    
    if (options?.excludeDeprecated) {
      filtered = filtered.filter((m) => m.deprecation_status === 'active');
    }
    
    return filtered;
  }, [state, options?.providerId, options?.availableOnly, options?.excludeDeprecated]);
  
  return { models, ...rest };
}

/**
 * Hook to get a specific model
 */
export function useModel(modelId: string) {
  const { data: state, ...rest } = useSyncState();
  
  const model = useMemo(
    () => state?.all_models.find((m) => m.id === modelId),
    [state, modelId]
  );
  
  return { model, ...rest };
}

/**
 * Hook to get model availability
 */
export function useModelAvailability(modelId: string, providerId?: string, enabled = true) {
  return useQuery({
    queryKey: providerSyncKeys.modelAvailability(modelId, providerId),
    queryFn: () => fetchModelAvailability(modelId, providerId),
    enabled: enabled && !!modelId,
    staleTime: 10000,
    refetchInterval: 30000,
  });
}

/**
 * Hook to get active provider and model
 */
export function useActiveSelection() {
  const { data: state, ...rest } = useSyncState();
  
  const activeProvider = useMemo(
    () => state?.providers.find((p) => p.id === state.active_provider_id),
    [state]
  );
  
  const activeModel = useMemo(
    () => state?.all_models.find((m) => m.id === state.active_model_id),
    [state]
  );
  
  return {
    activeProvider,
    activeModel,
    activeProviderId: state?.active_provider_id,
    activeModelId: state?.active_model_id,
    ...rest,
  };
}

// =============================================================================
// WebSocket Hook
// =============================================================================

interface UseSyncWebSocketOptions {
  enabled?: boolean;
  onEvent?: (event: WebSocketMessage) => void;
  onConnectionChange?: (connected: boolean) => void;
}

/**
 * Hook to connect to the sync WebSocket for real-time updates
 */
export function useSyncWebSocket(options: UseSyncWebSocketOptions = {}) {
  const { enabled = true, onEvent, onConnectionChange } = options;
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (!enabled || wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const apiUrl = getApiBaseUrl();
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = apiUrl.replace(/^https?:/, wsProtocol) + '/provider-sync/ws';

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[SyncWS] Connected');
        setIsConnected(true);
        setReconnectAttempts(0);
        onConnectionChange?.(true);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          
          // Handle different event types
          if (message.type === SyncEventType.INITIAL_STATE || 
              message.type === SyncEventType.FULL_SYNC) {
            // Update the cached state
            if (message.data) {
              queryClient.setQueryData(providerSyncKeys.state(), message.data);
            }
          } else if (
            message.type === SyncEventType.PROVIDER_ADDED ||
            message.type === SyncEventType.PROVIDER_UPDATED ||
            message.type === SyncEventType.PROVIDER_REMOVED ||
            message.type === SyncEventType.MODEL_DEPRECATED
          ) {
            // Invalidate queries to refetch
            queryClient.invalidateQueries({ queryKey: providerSyncKeys.state() });
          }
          
          onEvent?.(message);
        } catch (error) {
          console.error('[SyncWS] Failed to parse message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('[SyncWS] Disconnected:', event.code);
        setIsConnected(false);
        wsRef.current = null;
        onConnectionChange?.(false);

        // Attempt reconnection
        if (enabled && reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
          console.log(`[SyncWS] Reconnecting in ${delay}ms...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
            connect();
          }, delay);
        }
      };

      ws.onerror = () => {
        console.error('[SyncWS] Error occurred');
      };
    } catch (error) {
      console.error('[SyncWS] Failed to connect:', error);
    }
  }, [enabled, onEvent, onConnectionChange, queryClient, reconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    if (enabled) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    isConnected,
    reconnectAttempts,
    connect,
    disconnect,
    sendMessage,
  };
}

// =============================================================================
// Combined Hook
// =============================================================================

/**
 * Combined hook that provides sync state with WebSocket updates
 */
export function useProviderSyncWithWebSocket(options?: {
  includeDeprecated?: boolean;
  enableWebSocket?: boolean;
}) {
  const { includeDeprecated = false, enableWebSocket = true } = options ?? {};
  
  const stateQuery = useSyncState({ includeDeprecated });
  const syncMutation = usePerformSync();
  
  const { isConnected, reconnectAttempts } = useSyncWebSocket({
    enabled: enableWebSocket,
  });

  const forceSync = useCallback(async () => {
    await syncMutation.mutateAsync({
      sync_type: 'full',
      include_deprecated: includeDeprecated,
    });
  }, [syncMutation, includeDeprecated]);

  return {
    // State
    state: stateQuery.data,
    providers: stateQuery.data?.providers ?? [],
    models: stateQuery.data?.all_models ?? [],
    activeProviderId: stateQuery.data?.active_provider_id,
    activeModelId: stateQuery.data?.active_model_id,
    version: stateQuery.data?.metadata.version ?? 0,
    
    // Status
    isLoading: stateQuery.isLoading,
    isError: stateQuery.isError,
    error: stateQuery.error,
    isConnected,
    reconnectAttempts,
    isSyncing: syncMutation.isPending,
    
    // Actions
    forceSync,
    refetch: stateQuery.refetch,
  };
}