/**
 * useUnifiedProviderSelection Hook
 *
 * A comprehensive hook for managing provider and model selection with:
 * - React Query for data fetching with caching
 * - WebSocket integration for real-time updates
 * - Optimistic updates when selecting provider/model
 * - LocalStorage persistence and backend sync
 * - Auto-recovery on connection loss
 * - Version tracking for optimistic concurrency control
 * - Automatic retry with exponential backoff on conflicts
 *
 * @module hooks/useUnifiedProviderSelection
 */

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  ProviderInfo,
  ModelInfo,
  ProviderStatus,
  ConnectionStatus,
  UseUnifiedProviderSelectionReturn,
  WSMessage,
} from "../types/unified-providers";
import {
  getProviders,
  getProviderModels,
  selectProviderModel,
  clearSelection,
  getCurrentSelection,
  saveSelectionToLocalStorage,
  getSelectionFromLocalStorage,
  clearSelectionFromLocalStorage,
  getWebSocketUrl,
} from "../lib/api/unified-providers";

// =============================================================================
// Types for Optimistic Concurrency
// =============================================================================

interface SelectionState {
  provider: string | null;
  model: string | null;
  version: number;
}

interface ConflictError extends Error {
  status: 409;
  currentVersion: number;
  currentProvider: string;
  currentModel: string;
}

interface SyncResult {
  success: boolean;
  provider: string;
  model: string;
  version: number;
  conflict: boolean;
  serverTimestamp: string;
}

// =============================================================================
// Query Keys
// =============================================================================

export const UNIFIED_PROVIDER_QUERY_KEYS = {
  all: ["unified-providers"] as const,
  providers: () => [...UNIFIED_PROVIDER_QUERY_KEYS.all, "providers"] as const,
  provider: (id: string) => [...UNIFIED_PROVIDER_QUERY_KEYS.providers(), id] as const,
  models: (providerId: string) =>
    [...UNIFIED_PROVIDER_QUERY_KEYS.all, "models", providerId] as const,
  allModels: () => [...UNIFIED_PROVIDER_QUERY_KEYS.all, "models"] as const,
  selection: (sessionId?: string) =>
    [...UNIFIED_PROVIDER_QUERY_KEYS.all, "selection", sessionId] as const,
};

// =============================================================================
// Configuration
// =============================================================================

interface UseUnifiedProviderSelectionOptions {
  /** Session ID for session-scoped selection */
  sessionId?: string;
  /** Whether to automatically connect WebSocket */
  enableWebSocket?: boolean;
  /** Stale time for provider query (default: 5 minutes) */
  providerStaleTime?: number;
  /** Stale time for model query (default: 2 minutes) */
  modelStaleTime?: number;
  /** Whether to auto-restore selection from localStorage */
  autoRestoreSelection?: boolean;
  /** Callback when selection changes (from any source) */
  onSelectionChange?: (provider: string | null, model: string | null) => void;
  /** Maximum retry attempts on conflict (default: 3) */
  maxRetryAttempts?: number;
  /** Base delay for exponential backoff in ms (default: 100) */
  retryBaseDelay?: number;
  /** Callback when conflict is detected */
  onConflict?: (serverState: SelectionState) => void;
}

// =============================================================================
// Retry Helper
// =============================================================================

async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts: number,
  baseDelay: number,
  shouldRetry: (error: unknown) => boolean
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (!shouldRetry(error) || attempt === maxAttempts - 1) {
        throw error;
      }

      // Exponential backoff with jitter
      const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 100;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw lastError;
}

function isConflictError(error: unknown): error is ConflictError {
  return (
    error instanceof Error &&
    "status" in error &&
    (error as ConflictError).status === 409
  );
}

// =============================================================================
// WebSocket Hook
// =============================================================================

interface UseWebSocketReturn {
  status: ConnectionStatus;
  lastMessage: WSMessage | null;
  sendMessage: (message: unknown) => void;
  reconnect: () => void;
}

function useProviderWebSocket(
  sessionId?: string,
  enabled: boolean = true,
  onMessage?: (message: WSMessage) => void
): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const baseReconnectDelay = 1000;

  // Use ref to store the connect function to avoid circular dependency
  const connectRef = useRef<() => void>(() => {});

  // Define connect function
  const connect = useCallback(() => {
    if (!enabled || typeof window === "undefined") return;

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const url = getWebSocketUrl(sessionId);
      const ws = new WebSocket(url);
      wsRef.current = ws;
      setStatus("reconnecting");

      ws.onopen = () => {
        setStatus("connected");
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WSMessage;
          setLastMessage(message);
          onMessage?.(message);
        } catch (error) {
          console.warn("Failed to parse WebSocket message:", error);
        }
      };

      ws.onclose = (event) => {
        setStatus("disconnected");

        // Attempt reconnect if not intentionally closed
        if (!event.wasClean && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            // Use ref to call connect to avoid stale closure
            connectRef.current();
          }, delay);
        }
      };

      ws.onerror = () => {
        setStatus("disconnected");
      };
    } catch (error) {
      console.error("Failed to create WebSocket connection:", error);
      setStatus("disconnected");
    }
  }, [enabled, sessionId, onMessage]);

  // Keep connectRef in sync with connect
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const sendMessage = useCallback((message: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect]);

  // Connect on mount
  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [enabled, connect]);

  return { status, lastMessage, sendMessage, reconnect };
}

// =============================================================================
// Main Hook
// =============================================================================

export function useUnifiedProviderSelection(
  options: UseUnifiedProviderSelectionOptions = {}
): UseUnifiedProviderSelectionReturn {
  const {
    sessionId,
    enableWebSocket = true,
    providerStaleTime = 5 * 60 * 1000, // 5 minutes
    modelStaleTime = 2 * 60 * 1000, // 2 minutes
    autoRestoreSelection = true,
    onSelectionChange,
    maxRetryAttempts = 3,
    retryBaseDelay = 100,
    onConflict,
  } = options;

  const queryClient = useQueryClient();

  // Local state for selection with version tracking
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [selectionVersion, setSelectionVersion] = useState<number>(0);
  const [providerStatus, setProviderStatus] = useState<
    Record<string, ProviderStatus>
  >({});
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncError, setLastSyncError] = useState<Error | null>(null);

  // Track if we've restored from storage
  const hasRestoredRef = useRef(false);

  // Track pending optimistic updates for rollback
  const pendingUpdateRef = useRef<SelectionState | null>(null);

  // =============================================================================
  // Queries
  // =============================================================================

  // Fetch all providers
  const providersQuery = useQuery({
    queryKey: UNIFIED_PROVIDER_QUERY_KEYS.providers(),
    queryFn: () => getProviders({ available_only: true }),
    staleTime: providerStaleTime,
    gcTime: 10 * 60 * 1000, // 10 minutes
    refetchOnWindowFocus: false,
  });

  // Fetch models for selected provider
  const modelsQuery = useQuery({
    queryKey: UNIFIED_PROVIDER_QUERY_KEYS.models(selectedProvider || ""),
    queryFn: () => getProviderModels(selectedProvider!, { available_only: true }),
    enabled: !!selectedProvider,
    staleTime: modelStaleTime,
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
  });

  // Fetch current selection from backend
  const selectionQuery = useQuery({
    queryKey: UNIFIED_PROVIDER_QUERY_KEYS.selection(sessionId),
    queryFn: () => getCurrentSelection(sessionId),
    staleTime: 30 * 1000, // 30 seconds
    refetchOnWindowFocus: true,
    enabled: autoRestoreSelection,
  });

  // =============================================================================
  // Mutations with Optimistic Concurrency
  // =============================================================================

  // Sync selection to backend with version tracking
  const syncSelectionToBackend = useCallback(
    async (
      provider: string,
      model: string,
      expectedVersion?: number
    ): Promise<SyncResult> => {
      const response = await fetch(
        `/api/v1/unified-providers/selection/sync`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            provider,
            model,
            version: expectedVersion,
            source: "frontend",
            timestamp: new Date().toISOString(),
          }),
        }
      );

      if (response.status === 409) {
        const errorData = await response.json();
        const error = new Error("Version conflict") as ConflictError;
        error.status = 409;
        error.currentVersion = errorData.detail?.current_version;
        error.currentProvider = errorData.detail?.current_provider;
        error.currentModel = errorData.detail?.current_model;
        throw error;
      }

      if (!response.ok) {
        throw new Error(`Sync failed: ${response.statusText}`);
      }

      return response.json();
    },
    [sessionId]
  );

  // Save selection mutation with retry logic
  const saveSelectionMutation = useMutation({
    mutationFn: async ({
      provider,
      model,
    }: {
      provider: string;
      model: string;
    }) => {
      setIsSyncing(true);
      setLastSyncError(null);

      // Store current state for potential rollback
      pendingUpdateRef.current = {
        provider: selectedProvider,
        model: selectedModel,
        version: selectionVersion,
      };

      try {
        const result = await withRetry(
          () => syncSelectionToBackend(provider, model, selectionVersion),
          maxRetryAttempts,
          retryBaseDelay,
          (error) => {
            // Retry on conflict - fetch latest version and retry
            if (isConflictError(error)) {
              // Update local state with server state
              setSelectionVersion(error.currentVersion);
              onConflict?.({
                provider: error.currentProvider,
                model: error.currentModel,
                version: error.currentVersion,
              });
              return true;
            }
            // Retry on network errors
            return (
              error instanceof TypeError ||
              (error instanceof Error && error.message.includes("network"))
            );
          }
        );

        return result;
      } finally {
        setIsSyncing(false);
        pendingUpdateRef.current = null;
      }
    },
    onSuccess: (data) => {
      // Update version from server response
      setSelectionVersion(data.version);

      // Invalidate selection query
      queryClient.invalidateQueries({
        queryKey: UNIFIED_PROVIDER_QUERY_KEYS.selection(sessionId),
      });

      // Save to localStorage with version
      saveSelectionToLocalStorage(data.provider, data.model);
      localStorage.setItem("selection_version", String(data.version));
    },
    onError: (error) => {
      console.error("Failed to save selection:", error);
      setLastSyncError(error as Error);

      // Rollback optimistic update on failure
      if (pendingUpdateRef.current) {
        setSelectedProvider(pendingUpdateRef.current.provider);
        setSelectedModel(pendingUpdateRef.current.model);
        setSelectionVersion(pendingUpdateRef.current.version);
        pendingUpdateRef.current = null;
      }
    },
  });

  // Clear selection mutation
  const clearSelectionMutation = useMutation({
    mutationFn: async () => {
      return clearSelection(sessionId);
    },
    onSuccess: () => {
      setSelectedProvider(null);
      setSelectedModel(null);
      clearSelectionFromLocalStorage();

      queryClient.invalidateQueries({
        queryKey: UNIFIED_PROVIDER_QUERY_KEYS.selection(sessionId),
      });
    },
  });

  // =============================================================================
  // WebSocket Integration with Version Tracking
  // =============================================================================

  const handleWebSocketMessage = useCallback(
    (message: WSMessage) => {
      switch (message.type) {
        case "current_selection":
        case "selection_changed":
          // Update local state if the change came from another source
          if (
            message.session_id === sessionId ||
            !message.session_id
          ) {
            const newVersion = message.version || 0;
            // Only update if server version is newer
            if (newVersion >= selectionVersion) {
              setSelectedProvider(message.provider);
              setSelectedModel(message.model);
              setSelectionVersion(newVersion);
              onSelectionChange?.(message.provider, message.model);
            }
          }
          break;

        case "update_result":
          // Handle result from WebSocket-based updates
          if (message.success) {
            setSelectionVersion(message.version || selectionVersion + 1);
          } else if (message.conflict) {
            // Server rejected - fetch latest state
            onConflict?.({
              provider: message.provider,
              model: message.model,
              version: message.version || 0,
            });
          }
          break;

        case "SELECTION_CHANGED":
          // Legacy format support
          if (
            message.data?.session_id === sessionId ||
            !message.data?.session_id
          ) {
            setSelectedProvider(message.data.provider_id);
            setSelectedModel(message.data.model_id);
            if (message.data.version) {
              setSelectionVersion(message.data.version);
            }
            onSelectionChange?.(
              message.data.provider_id,
              message.data.model_id
            );
          }
          break;

        case "PROVIDER_STATUS":
          setProviderStatus((prev) => ({
            ...prev,
            [message.data.provider_id]: message.data.status as ProviderStatus,
          }));

          // Invalidate provider query if status changed significantly
          if (
            message.data.status === "unhealthy" ||
            message.data.status === "healthy"
          ) {
            queryClient.invalidateQueries({
              queryKey: UNIFIED_PROVIDER_QUERY_KEYS.providers(),
            });
          }
          break;

        case "MODEL_VALIDATION":
          if (!message.data.is_valid && message.data.errors?.length) {
            console.warn(
              `Model validation failed for ` +
                `${message.data.provider_id}/${message.data.model_id}:`,
              message.data.errors
            );
          }
          break;
      }
    },
    [sessionId, queryClient, onSelectionChange, selectionVersion, onConflict]
  );

  const websocket = useProviderWebSocket(sessionId, enableWebSocket, handleWebSocketMessage);

  // =============================================================================
  // Selection Restoration
  // =============================================================================

  // Restore selection from backend or localStorage on mount
  useEffect(() => {
    if (hasRestoredRef.current || !autoRestoreSelection) return;

    const restore = async () => {
      // First check backend selection
      if (selectionQuery.data) {
        setSelectedProvider(selectionQuery.data.provider_id);
        setSelectedModel(selectionQuery.data.model_id);
        // Set version from backend
        if (selectionQuery.data.version !== undefined) {
          setSelectionVersion(selectionQuery.data.version);
        }
        hasRestoredRef.current = true;
        return;
      }

      // Fallback to localStorage
      const stored = getSelectionFromLocalStorage();
      if (stored) {
        setSelectedProvider(stored.provider);
        setSelectedModel(stored.model);
        // Restore version from localStorage
        const storedVersion = localStorage.getItem("selection_version");
        if (storedVersion) {
          setSelectionVersion(parseInt(storedVersion, 10));
        }
        hasRestoredRef.current = true;
      }
    };

    if (!selectionQuery.isLoading) {
      restore();
    }
  }, [selectionQuery.data, selectionQuery.isLoading, autoRestoreSelection]);

  // =============================================================================
  // Actions
  // =============================================================================

  const selectProvider = useCallback(
    async (providerId: string) => {
      // Optimistic update
      setSelectedProvider(providerId);
      setSelectedModel(null); // Clear model when provider changes

      // Find the default model for this provider
      const provider = providersQuery.data?.find((p) => p.provider_id === providerId);
      if (provider?.default_model) {
        setSelectedModel(provider.default_model);

        // Persist to backend
        await saveSelectionMutation.mutateAsync({
          provider: providerId,
          model: provider.default_model,
        });
      }

      // Prefetch models for this provider
      queryClient.prefetchQuery({
        queryKey: UNIFIED_PROVIDER_QUERY_KEYS.models(providerId),
        queryFn: () => getProviderModels(providerId, { available_only: true }),
      });
    },
    [providersQuery.data, saveSelectionMutation, queryClient]
  );

  const selectModel = useCallback(
    async (modelId: string) => {
      if (!selectedProvider) {
        console.warn("Cannot select model without a provider");
        return;
      }

      // Optimistic update
      setSelectedModel(modelId);

      // Persist to backend
      await saveSelectionMutation.mutateAsync({
        provider: selectedProvider,
        model: modelId,
      });

      // Notify callback
      onSelectionChange?.(selectedProvider, modelId);
    },
    [selectedProvider, saveSelectionMutation, onSelectionChange]
  );

  const refreshProviders = useCallback(async () => {
    await queryClient.invalidateQueries({
      queryKey: UNIFIED_PROVIDER_QUERY_KEYS.providers(),
    });
  }, [queryClient]);

  const refreshModels = useCallback(async () => {
    if (selectedProvider) {
      await queryClient.invalidateQueries({
        queryKey: UNIFIED_PROVIDER_QUERY_KEYS.models(selectedProvider),
      });
    }
  }, [queryClient, selectedProvider]);

  const clearSelectionAction = useCallback(async () => {
    await clearSelectionMutation.mutateAsync();
  }, [clearSelectionMutation]);

  // =============================================================================
  // Helpers
  // =============================================================================

  const getProviderById = useCallback(
    (id: string): ProviderInfo | undefined => {
      return providersQuery.data?.find((p) => p.provider_id === id);
    },
    [providersQuery.data]
  );

  const getModelById = useCallback(
    (id: string): ModelInfo | undefined => {
      return modelsQuery.data?.find((m) => m.model_id === id);
    },
    [modelsQuery.data]
  );

  const getModelsForProvider = useCallback(
    (providerId: string): ModelInfo[] => {
      if (providerId === selectedProvider) {
        return modelsQuery.data || [];
      }
      // For other providers, return cached data if available
      const cachedData = queryClient.getQueryData<ModelInfo[]>(
        UNIFIED_PROVIDER_QUERY_KEYS.models(providerId)
      );
      return cachedData || [];
    },
    [selectedProvider, modelsQuery.data, queryClient]
  );

  // Build provider status map from query data
  const combinedProviderStatus = useMemo(() => {
    const statusMap: Record<string, ProviderStatus> = { ...providerStatus };

    providersQuery.data?.forEach((provider) => {
      if (!statusMap[provider.provider_id]) {
        statusMap[provider.provider_id] = provider.status;
      }
    });

    return statusMap;
  }, [providersQuery.data, providerStatus]);

  // =============================================================================
  // Return
  // =============================================================================

  // =============================================================================
  // Force Sync Helper
  // =============================================================================

  const forceSync = useCallback(async () => {
    // Fetch latest state from server and update local
    await queryClient.invalidateQueries({
      queryKey: UNIFIED_PROVIDER_QUERY_KEYS.selection(sessionId),
    });

    const data = await getCurrentSelection(sessionId);
    if (data) {
      setSelectedProvider(data.provider_id);
      setSelectedModel(data.model_id);
      if (data.version !== undefined) {
        setSelectionVersion(data.version);
      }
    }
  }, [queryClient, sessionId]);

  // =============================================================================
  // Return
  // =============================================================================

  return {
    // State
    providers: providersQuery.data || [],
    models: modelsQuery.data || [],
    selectedProvider,
    selectedModel,

    // Loading states
    isLoadingProviders: providersQuery.isLoading,
    isLoadingModels: modelsQuery.isLoading,
    isSaving: saveSelectionMutation.isPending,
    isSyncing,

    // Error states
    providersError: providersQuery.error as Error | null,
    modelsError: modelsQuery.error as Error | null,
    syncError: lastSyncError,

    // Actions
    selectProvider,
    selectModel,
    refreshProviders,
    refreshModels,
    clearSelection: clearSelectionAction,
    forceSync,

    // Status
    providerStatus: combinedProviderStatus,
    connectionStatus: websocket.status,

    // Version tracking (for advanced use)
    selectionVersion,

    // Helpers
    getProviderById,
    getModelById,
    getModelsForProvider,
  };
}

// =============================================================================
// Export query keys for external use
// =============================================================================

export { UNIFIED_PROVIDER_QUERY_KEYS as unifiedProviderQueryKeys };
