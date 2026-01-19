"use client";

/**
 * Enhanced Model Selection Hook
 *
 * An enhanced version of useModelSelection with:
 * - Real-time WebSocket synchronization (built-in)
 * - Optimistic updates with automatic rollback
 * - Session persistence via localStorage
 * - Comprehensive error handling and retry logic
 * - Rate limit awareness and visual feedback
 * - Debounced auto-save for better UX
 *
 * Use this hook when you need a self-contained model selection
 * solution with all features built-in. For more granular control,
 * use the individual hooks from use-provider-management.ts
 */

import { useState, useEffect, useCallback, useRef } from "react";

// Types
interface Provider {
  id: string;
  name: string;
  status: "healthy" | "degraded" | "offline";
  models: Model[];
  latency?: number;
}

interface Model {
  id: string;
  name: string;
  tier: "standard" | "premium" | "experimental";
  contextWindow?: number;
  maxTokens?: number;
}

interface ModelSelection {
  providerId: string;
  modelId: string;
}

interface UseEnhancedModelSelectionOptions {
  /** Enable WebSocket auto-sync (default: true) */
  autoSync?: boolean;
  /** Persist session ID to localStorage (default: true) */
  persistSession?: boolean;
  /** Callback when selection changes */
  onSelectionChange?: (selection: ModelSelection) => void;
  /** Callback on errors */
  onError?: (error: Error) => void;
  /** Callback on rate limit */
  onRateLimit?: (retryAfter?: number) => void;
  /** Custom API base URL */
  apiBaseUrl?: string;
}

interface UseEnhancedModelSelectionReturn {
  // State
  providers: Provider[];
  selectedProvider: string | null;
  selectedModel: string | null;
  isLoading: boolean;
  isSyncing: boolean;
  isConnected: boolean;
  error: string | null;
  sessionId: string | null;

  // Actions
  selectProvider: (providerId: string) => void;
  selectModel: (modelId: string) => void;
  setSelection: (providerId: string, modelId: string) => Promise<void>;
  refresh: () => Promise<void>;
  reconnect: () => void;
  clearError: () => void;

  // Computed
  currentProvider: Provider | null;
  currentModel: Model | null;
  availableModels: Model[];
  isRateLimited: boolean;
  healthyProviders: Provider[];
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api/v1";
const SESSION_KEY = "chimera_session_id";
const DEBOUNCE_MS = 400;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000;

// Type guard for provider data from API
interface ApiProviderData {
  provider?: string;
  provider_id?: string;
  id?: string;
  name?: string;
  display_name?: string;
  status?: string;
  is_healthy?: boolean;
  models?: Array<ApiModelData | string>;
  default_model?: string;
  latency?: number;
  latency_ms?: number;
}

interface ApiModelData {
  model_id?: string;
  id?: string;
  name?: string;
  tier?: string;
  context_window?: number;
  max_tokens?: number;
}

export function useEnhancedModelSelection(
  options: UseEnhancedModelSelectionOptions = {}
): UseEnhancedModelSelectionReturn {
  const {
    autoSync = true,
    persistSession = true,
    onSelectionChange,
    onError,
    onRateLimit,
    apiBaseUrl = DEFAULT_API_BASE,
  } = options;

  // State
  const [providers, setProviders] = useState<Provider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isRateLimited, setIsRateLimited] = useState(false);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const debounceTimer = useRef<NodeJS.Timeout | null>(null);
  const previousSelection = useRef<ModelSelection | null>(null);
  const wsBaseUrl = apiBaseUrl.replace(/^http/, "ws");

  // Initialize session
  useEffect(() => {
    if (persistSession && typeof window !== "undefined") {
      const stored = localStorage.getItem(SESSION_KEY);
      if (stored) {
        setSessionId(stored);
      } else {
        const newId = `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
        localStorage.setItem(SESSION_KEY, newId);
        setSessionId(newId);
      }
    } else if (!persistSession) {
      setSessionId(`temp_${Date.now()}_${Math.random().toString(36).slice(2)}`);
    }
  }, [persistSession]);

  // Fetch providers
  const fetchProviders = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (sessionId) {
        headers["X-Session-ID"] = sessionId;
      }

      const response = await fetch(`${apiBaseUrl}/providers`, { headers });

      if (!response.ok) {
        throw new Error(`Failed to fetch providers: ${response.status}`);
      }

      const data = await response.json();
      const providerList: ApiProviderData[] = Array.isArray(data)
        ? data
        : (data.providers || []);

      // Transform to internal format
      const transformed: Provider[] = providerList.map((p: ApiProviderData) => {
        const providerId = p.provider || p.provider_id || p.id || "";
        const displayName = p.display_name || p.name || providerId;
        const statusValue = p.status;
        const isHealthy = typeof p.is_healthy === "boolean" ? p.is_healthy : true;
        const normalizedStatus =
          statusValue === "healthy" || statusValue === "degraded" || statusValue === "offline"
            ? statusValue
            : isHealthy
              ? "healthy"
              : "offline";

        const models = Array.isArray(p.models)
          ? p.models.map((m) => {
              if (typeof m === "string") {
                return {
                  id: m,
                  name: m,
                  tier: "standard" as const,
                };
              }
              return {
                id: m.model_id || m.id || "",
                name: m.name || m.model_id || m.id || "",
                tier:
                  m.tier === "standard" || m.tier === "premium" || m.tier === "experimental"
                    ? m.tier
                    : "standard",
                contextWindow: m.context_window,
                maxTokens: m.max_tokens,
              };
            })
          : [];

        return {
          id: providerId,
          name: displayName,
          status: normalizedStatus,
          models,
          latency: p.latency_ms ?? p.latency,
        };
      });

      setProviders(transformed);

      // Set defaults if nothing selected
      if (!selectedProvider && transformed.length > 0) {
        const defaultProvider = transformed.find((p: Provider) => p.status === "healthy") || transformed[0];
        setSelectedProvider(defaultProvider.id);
        if (defaultProvider.models.length > 0) {
          setSelectedModel(defaultProvider.models[0].id);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load providers";
      setError(message);
      onError?.(err instanceof Error ? err : new Error(message));
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, selectedProvider, onError, apiBaseUrl]);

  // Fetch current selection from server
  const fetchCurrentSelection = useCallback(async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${apiBaseUrl}/session/${sessionId}/model`);
      if (response.ok) {
        const data = await response.json();
        const providerId = data.provider_id || data.provider;
        const modelId = data.model_id || data.model;
        if (providerId && modelId) {
          setSelectedProvider(providerId);
          setSelectedModel(modelId);
        }
      }
    } catch {
      // Silently fail - use local state
    }
  }, [sessionId, apiBaseUrl]);

  // Sync selection to server
  const syncSelection = useCallback(async (
    providerId: string,
    modelId: string
  ): Promise<boolean> => {
    if (!sessionId) return false;

    setIsSyncing(true);

    try {
      const response = await fetch(`${apiBaseUrl}/providers/select`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId,
        },
        body: JSON.stringify({
          provider: providerId,
          model: modelId,
        }),
      });

      if (response.status === 429) {
        setIsRateLimited(true);
        const data = await response.json();
        const retryAfter = data.retry_after || 60;
        onRateLimit?.(retryAfter);

        // Auto-clear rate limit after retry period
        setTimeout(() => setIsRateLimited(false), retryAfter * 1000);
        return false;
      }

      if (!response.ok) {
        throw new Error(`Sync failed: ${response.status}`);
      }

      setIsRateLimited(false);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Sync failed";
      setError(message);
      onError?.(err instanceof Error ? err : new Error(message));
      return false;
    } finally {
      setIsSyncing(false);
    }
  }, [sessionId, onError, onRateLimit, apiBaseUrl]);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (!autoSync || !sessionId) return;

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(
        `${wsBaseUrl}/providers/ws/selection?session_id=${sessionId}`
      );

      ws.onopen = () => {
        setIsConnected(true);
        reconnectAttempts.current = 0;
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "selection_change" || data.type === "current_selection") {
            const provider = data.data?.provider || data.provider_id;
            const model = data.data?.model || data.model_id;
            if (provider && model) {
              setSelectedProvider(provider);
              setSelectedModel(model);
              onSelectionChange?.({
                providerId: provider,
                modelId: model,
              });
            }
          } else if (data.type === "health_update") {
            const providersUpdate = Array.isArray(data.data?.providers)
              ? data.data.providers
              : [];
            if (providersUpdate.length > 0) {
              setProviders((prev) =>
                prev.map((p) => {
                  const updated = providersUpdate.find(
                    (item: { provider: string }) => item.provider === p.id
                  );
                  if (!updated) return p;
                  const nextStatus = updated.is_healthy ? "healthy" : "offline";
                  return { ...p, status: nextStatus as Provider["status"] };
                })
              );
            }
          } else if (data.type === "selection_update") {
            setSelectedProvider(data.provider_id);
            setSelectedModel(data.model_id);
            onSelectionChange?.({
              providerId: data.provider_id,
              modelId: data.model_id,
            });
          } else if (data.type === "rate_limit") {
            setIsRateLimited(true);
            onRateLimit?.(data.retry_after);
          }
        } catch {
          // Invalid message format - ignore
        }
      };

      ws.onclose = () => {
        setIsConnected(false);

        // Auto-reconnect with exponential backoff
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          const delay = RECONNECT_DELAY_MS * Math.pow(1.5, reconnectAttempts.current - 1);
          setTimeout(connectWebSocket, delay);
        }
      };

      ws.onerror = () => {
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch {
      // WebSocket creation failed
      setIsConnected(false);
    }
  }, [autoSync, sessionId, wsBaseUrl, onSelectionChange, onRateLimit]);

  // Initialize
  useEffect(() => {
    if (sessionId) {
      fetchProviders();
      fetchCurrentSelection();
    }
  }, [sessionId, fetchProviders, fetchCurrentSelection]);

  // Connect WebSocket
  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [sessionId, connectWebSocket]);

  // Debounced sync
  const debouncedSync = useCallback((providerId: string, modelId: string) => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(async () => {
      const success = await syncSelection(providerId, modelId);

      if (!success && previousSelection.current) {
        // Rollback on failure
        setSelectedProvider(previousSelection.current.providerId);
        setSelectedModel(previousSelection.current.modelId);
      }
    }, DEBOUNCE_MS);
  }, [syncSelection]);

  // Actions
  const selectProvider = useCallback((providerId: string) => {
    if (isRateLimited) return;

    previousSelection.current = selectedProvider && selectedModel
      ? { providerId: selectedProvider, modelId: selectedModel }
      : null;

    setSelectedProvider(providerId);

    // Auto-select first model of new provider
    const provider = providers.find((p) => p.id === providerId);
    if (provider && provider.models.length > 0) {
      const newModelId = provider.models[0].id;
      setSelectedModel(newModelId);
      debouncedSync(providerId, newModelId);
      onSelectionChange?.({ providerId, modelId: newModelId });
    }
  }, [providers, selectedProvider, selectedModel, isRateLimited, debouncedSync, onSelectionChange]);

  const selectModel = useCallback((modelId: string) => {
    if (!selectedProvider || isRateLimited) return;

    previousSelection.current = selectedModel
      ? { providerId: selectedProvider, modelId: selectedModel }
      : null;

    setSelectedModel(modelId);
    debouncedSync(selectedProvider, modelId);
    onSelectionChange?.({ providerId: selectedProvider, modelId });
  }, [selectedProvider, selectedModel, isRateLimited, debouncedSync, onSelectionChange]);

  const setSelection = useCallback(async (providerId: string, modelId: string) => {
    if (isRateLimited) return;

    previousSelection.current = selectedProvider && selectedModel
      ? { providerId: selectedProvider, modelId: selectedModel }
      : null;

    setSelectedProvider(providerId);
    setSelectedModel(modelId);

    const success = await syncSelection(providerId, modelId);

    if (success) {
      onSelectionChange?.({ providerId, modelId });
    } else if (previousSelection.current) {
      setSelectedProvider(previousSelection.current.providerId);
      setSelectedModel(previousSelection.current.modelId);
    }
  }, [selectedProvider, selectedModel, isRateLimited, syncSelection, onSelectionChange]);

  const refresh = useCallback(async () => {
    await fetchProviders();
    await fetchCurrentSelection();
  }, [fetchProviders, fetchCurrentSelection]);

  const reconnect = useCallback(() => {
    reconnectAttempts.current = 0;
    connectWebSocket();
  }, [connectWebSocket]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Computed values
  const currentProvider = providers.find((p) => p.id === selectedProvider) || null;
  const currentModel = currentProvider?.models?.find((m) => m.id === selectedModel) || null;
  const availableModels = currentProvider?.models || [];
  const healthyProviders = providers.filter((p) => p.status === "healthy");

  return {
    // State
    providers,
    selectedProvider,
    selectedModel,
    isLoading,
    isSyncing,
    isConnected,
    error,
    sessionId,

    // Actions
    selectProvider,
    selectModel,
    setSelection,
    refresh,
    reconnect,
    clearError,

    // Computed
    currentProvider,
    currentModel,
    availableModels,
    isRateLimited,
    healthyProviders,
  };
}

// Export types
export type {
  Provider,
  Model,
  ModelSelection,
  UseEnhancedModelSelectionOptions,
  UseEnhancedModelSelectionReturn,
};
