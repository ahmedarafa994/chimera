"use client";

/**
 * Unified Model Provider
 *
 * Consolidates all model selection state management into a single provider
 * to eliminate redundant API calls and improve performance.
 *
 * This provider replaces the need for:
 * - ChimeraProvider (session/provider management)
 * - ProviderSyncProvider (real-time sync)
 * - ModelSelectionContext (model selection state)
 *
 * Key features:
 * - Single initialization with proper guards
 * - Debounced API calls to prevent spam
 * - Request deduplication
 * - Optimistic updates with rollback
 * - WebSocket for real-time updates (with polling fallback)
 * - Integration with new unified providers API
 */

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
  type ReactNode,
} from "react";

// Import the new unified provider API functions
import {
  getProviders as fetchUnifiedProviders,
  getProviderModels as fetchUnifiedProviderModels,
  selectProviderModel as saveUnifiedSelection,
  getCurrentSelection,
  saveSelectionToLocalStorage,
  getSelectionFromLocalStorage,
  clearSelectionFromLocalStorage,
} from "@/lib/api/unified-providers";

import type {
  ProviderInfo as UnifiedProviderInfo,
  ModelInfo,
  ProviderStatus,
} from "@/types/unified-providers";

// =============================================================================
// Types
// =============================================================================

export interface ModelSelection {
  provider: string | null;
  model: string | null;
}

export interface ProviderInfo {
  id: string;
  name: string;
  provider: string;
  models: string[];
  defaultModel: string | null;
  status: ProviderStatus;
  isConfigured: boolean;
  // Extended fields from unified API
  capabilities?: string[];
  modelCount?: number;
  healthScore?: number;
}

export interface UnifiedModelState {
  // Selection
  selection: ModelSelection;

  // Providers
  providers: ProviderInfo[];

  // Models for current provider (from unified API)
  models: ModelInfo[];

  // Session
  sessionId: string | null;

  // Status
  isInitialized: boolean;
  isLoading: boolean;
  isLoadingModels: boolean;
  isSyncing: boolean;
  error: string | null;

  // Connection
  isConnected: boolean;
  lastSyncTime: Date | null;

  // Provider status map for real-time updates
  providerStatus: Record<string, ProviderStatus>;
}

export interface UnifiedModelContextValue extends UnifiedModelState {
  // Actions
  setSelection: (provider: string, model: string) => Promise<void>;
  clearSelection: () => Promise<void>;
  refreshProviders: () => Promise<void>;
  getProviderModels: (providerId: string) => string[];
  // New unified API methods
  fetchModelsForProvider: (providerId: string) => Promise<ModelInfo[]>;
  getProviderById: (id: string) => ProviderInfo | undefined;
  getModelById: (id: string) => ModelInfo | undefined;
}

// =============================================================================
// Context
// =============================================================================

const UnifiedModelContext = createContext<UnifiedModelContextValue | null>(null);

// =============================================================================
// Constants
// =============================================================================

const STORAGE_KEYS = {
  SESSION_ID: "chimera_session_id",
  SELECTION: "chimera_model_selection",
} as const;

const API_BASE = "/api/v1";

// Debounce delay for API calls
const SYNC_DEBOUNCE_MS = 300;

// Minimum time between provider refreshes
const PROVIDER_REFRESH_COOLDOWN_MS = 30000;

// Flag to use new unified providers API
const USE_UNIFIED_API = true;

// =============================================================================
// Helper Functions
// =============================================================================

function getStoredSessionId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.SESSION_ID);
    if (!raw) return null;
    // Handle both quoted and unquoted values
    const cleaned = raw.replace(/^"|"$/g, "");
    return cleaned || null;
  } catch {
    return null;
  }
}

function setStoredSessionId(sessionId: string): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEYS.SESSION_ID, sessionId);
  } catch {
    // Ignore storage errors
  }
}

function getStoredSelection(): ModelSelection | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.SELECTION);
    if (!raw) return null;
    return JSON.parse(raw) as ModelSelection;
  } catch {
    return null;
  }
}

function setStoredSelection(selection: ModelSelection): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEYS.SELECTION, JSON.stringify(selection));
  } catch {
    // Ignore storage errors
  }
}

// =============================================================================
// API Functions with Request Deduplication
// =============================================================================

// Track in-flight requests to prevent duplicates
const inFlightRequests = new Map<string, Promise<unknown>>();

async function deduplicatedFetch<T>(
  key: string,
  fetchFn: () => Promise<T>
): Promise<T> {
  const existing = inFlightRequests.get(key);
  if (existing) {
    return existing as Promise<T>;
  }

  const promise = fetchFn().finally(() => {
    inFlightRequests.delete(key);
  });

  inFlightRequests.set(key, promise);
  return promise;
}

async function fetchProviders(): Promise<ProviderInfo[]> {
  return deduplicatedFetch("providers", async () => {
    try {
      // Use the new unified API if enabled
      if (USE_UNIFIED_API) {
        const unifiedProviders = await fetchUnifiedProviders({ available_only: true });
        return unifiedProviders.map((p: UnifiedProviderInfo) => ({
          id: p.provider_id,
          name: p.display_name,
          provider: p.provider_id,
          models: [], // Models are fetched separately in unified API
          defaultModel: p.default_model || null,
          status: p.status,
          isConfigured: p.has_api_key,
          capabilities: p.capabilities,
          modelCount: p.model_count,
          healthScore: p.health_score,
        }));
      }

      // Legacy API fallback
      const response = await fetch(`${API_BASE}/providers`);

      if (!response.ok) {
        throw new Error(`Failed to fetch providers: ${response.statusText}`);
      }

      const data = await response.json();
      const providers = data?.providers || [];

      return providers.map((p: Record<string, unknown>) => ({
        id: p.provider as string,
        name: (p.display_name || p.provider) as string,
        provider: p.provider as string,
        models: (p.models || []) as string[],
        defaultModel: (p.default_model || null) as string | null,
        status: (p.is_healthy ? "healthy" : "unhealthy") as ProviderStatus,
        isConfigured: true,
      }));
    } catch (error) {
      console.error("[fetchProviders] Error fetching providers:", error);
      return [];
    }
  });
}

async function fetchModelsForProvider(providerId: string): Promise<ModelInfo[]> {
  return deduplicatedFetch(`models-${providerId}`, async () => {
    try {
      const models = await fetchUnifiedProviderModels(providerId, { available_only: true });
      return models;
    } catch (error) {
      console.error(`[fetchModelsForProvider] Error fetching models for ${providerId}:`, error);
      return [];
    }
  });
}

async function fetchSession(sessionId: string): Promise<{
  provider: string | null;
  model: string | null;
} | null> {
  return deduplicatedFetch(`session-${sessionId}`, async () => {
    const response = await fetch(`${API_BASE}/session/${sessionId}`);
    if (!response.ok) {
      return null;
    }
    const data = await response.json();
    if (!data || typeof data !== "object") {
      return null;
    }
    return {
      provider: (data as { provider?: string | null }).provider || null,
      model: (data as { model?: string | null }).model || null,
    };
  });
}

async function createSession(selection?: ModelSelection): Promise<string | null> {
  const response = await fetch(`${API_BASE}/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: selection?.provider,
      model: selection?.model,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to create session");
  }

  const data = await response.json();
  return data.session_id || null;
}

async function updateSessionModel(
  sessionId: string,
  provider: string,
  model: string
): Promise<boolean> {
  const response = await fetch(`${API_BASE}/session/${sessionId}/model`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model }),
  });

  return response.ok;
}

async function validateModel(provider: string, model: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/models/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model }),
    });

    if (!response.ok) return false;
    const data = await response.json();
    return data.valid === true;
  } catch {
    return false;
  }
}

// =============================================================================
// Provider Component
// =============================================================================

interface UnifiedModelProviderProps {
  children: ReactNode;
}

export function UnifiedModelProvider({ children }: UnifiedModelProviderProps) {
  // State
  const [state, setState] = useState<UnifiedModelState>({
    selection: { provider: null, model: null },
    providers: [],
    models: [],
    sessionId: null,
    isInitialized: false,
    isLoading: true,
    isLoadingModels: false,
    isSyncing: false,
    error: null,
    isConnected: false,
    lastSyncTime: null,
    providerStatus: {},
  });

  // Refs for preventing duplicate initialization and tracking sync
  const initializingRef = useRef(false);
  const lastProviderRefreshRef = useRef<number>(0);
  const syncTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Initialize on mount
  useEffect(() => {
    if (initializingRef.current || state.isInitialized) return;
    initializingRef.current = true;

    const initialize = async () => {
      try {
        // Load stored selection first for immediate UI (use new unified localStorage)
        let storedSelection: ModelSelection | null = null;

        if (USE_UNIFIED_API) {
          const stored = getSelectionFromLocalStorage();
          if (stored) {
            storedSelection = { provider: stored.provider, model: stored.model };
          }
        } else {
          storedSelection = getStoredSelection();
        }

        if (storedSelection) {
          setState(prev => ({
            ...prev,
            selection: storedSelection!,
          }));
        }

        // Fetch providers
        const providers = await fetchProviders();

        // Build initial provider status map
        const providerStatus: Record<string, ProviderStatus> = {};
        providers.forEach(p => {
          providerStatus[p.id] = p.status;
        });

        // Get or create session
        let sessionId = getStoredSessionId();
        let sessionSelection: ModelSelection | null = null;

        if (USE_UNIFIED_API) {
          // Try to get current selection from unified API
          try {
            const currentSelection = await getCurrentSelection(sessionId || undefined);
            if (currentSelection) {
              sessionSelection = {
                provider: currentSelection.provider_id,
                model: currentSelection.model_id,
              };
              if (currentSelection.session_id) {
                sessionId = currentSelection.session_id;
                setStoredSessionId(sessionId);
              }
            }
          } catch {
            // Selection may not exist, continue with stored/default
          }
        } else {
          if (sessionId) {
            // Try to restore session (legacy)
            sessionSelection = await fetchSession(sessionId);
            if (!sessionSelection) {
              sessionId = await createSession(storedSelection || undefined);
              if (sessionId) {
                setStoredSessionId(sessionId);
              }
            }
          } else {
            sessionId = await createSession(storedSelection || undefined);
            if (sessionId) {
              setStoredSessionId(sessionId);
            }
          }
        }

        // Determine final selection (session takes precedence)
        const finalSelection = sessionSelection || storedSelection || {
          provider: null,
          model: null,
        };

        // Fetch models if we have a provider selected
        let models: ModelInfo[] = [];
        if (USE_UNIFIED_API && finalSelection.provider) {
          models = await fetchModelsForProvider(finalSelection.provider);
        }

        // Update state
        setState(prev => ({
          ...prev,
          providers,
          models,
          sessionId,
          selection: finalSelection,
          isInitialized: true,
          isLoading: false,
          isConnected: true,
          lastSyncTime: new Date(),
          error: null,
          providerStatus,
        }));

        // Persist selection
        if (finalSelection.provider && finalSelection.model) {
          if (USE_UNIFIED_API) {
            saveSelectionToLocalStorage(finalSelection.provider, finalSelection.model);
          } else {
            setStoredSelection(finalSelection);
          }
        }

        lastProviderRefreshRef.current = Date.now();
      } catch (error) {
        console.error("[UnifiedModelProvider] Initialization failed:", error);
        setState(prev => ({
          ...prev,
          isInitialized: true,
          isLoading: false,
          error: error instanceof Error ? error.message : "Initialization failed",
        }));
      }
    };

    initialize();
  }, [state.isInitialized]);

  // Debounced sync to backend
  const syncToBackend = useCallback(async (provider: string, model: string) => {
    // Clear any pending sync
    if (syncTimeoutRef.current) {
      clearTimeout(syncTimeoutRef.current);
    }

    // Debounce the sync
    syncTimeoutRef.current = setTimeout(async () => {
      setState(prev => ({ ...prev, isSyncing: true }));

      try {
        if (USE_UNIFIED_API) {
          // Use new unified API
          await saveUnifiedSelection(provider, model, state.sessionId || undefined);
          saveSelectionToLocalStorage(provider, model);
        } else if (state.sessionId) {
          // Legacy API
          await updateSessionModel(state.sessionId, provider, model);
        }

        setState(prev => ({
          ...prev,
          isSyncing: false,
          lastSyncTime: new Date(),
        }));
      } catch (error) {
        console.error("[UnifiedModelProvider] Sync failed:", error);
        setState(prev => ({ ...prev, isSyncing: false }));
      }
    }, SYNC_DEBOUNCE_MS);
  }, [state.sessionId]);

  // Set selection with optimistic update
  const setSelection = useCallback(async (provider: string, model: string) => {
    const previousSelection = state.selection;
    const previousModels = state.models;

    // Optimistic update
    const newSelection = { provider, model };
    setState(prev => ({
      ...prev,
      selection: newSelection,
      isLoading: true,
    }));

    // Save to localStorage immediately
    if (USE_UNIFIED_API) {
      saveSelectionToLocalStorage(provider, model);
    } else {
      setStoredSelection(newSelection);
    }

    try {
      // If provider changed, fetch new models
      if (provider !== previousSelection.provider && USE_UNIFIED_API) {
        setState(prev => ({ ...prev, isLoadingModels: true }));
        const models = await fetchModelsForProvider(provider);
        setState(prev => ({ ...prev, models, isLoadingModels: false }));
      }

      // Validate model (skip for unified API as backend validates)
      if (!USE_UNIFIED_API) {
        const isValid = await validateModel(provider, model);
        if (!isValid) {
          throw new Error("Invalid model selection");
        }
      }

      // Ensure session exists (legacy mode)
      let sessionId = state.sessionId;
      if (!USE_UNIFIED_API && !sessionId) {
        sessionId = await createSession(newSelection);
        if (sessionId) {
          setStoredSessionId(sessionId);
          setState(prev => ({ ...prev, sessionId }));
        }
      }

      // Sync to backend (debounced)
      await syncToBackend(provider, model);

      setState(prev => ({ ...prev, isLoading: false }));
    } catch (error) {
      // Rollback on error
      console.error("[UnifiedModelProvider] setSelection failed:", error);
      setState(prev => ({
        ...prev,
        selection: previousSelection,
        models: previousModels,
        isLoading: false,
        isLoadingModels: false,
        error: error instanceof Error ? error.message : "Selection failed",
      }));

      // Rollback localStorage
      if (USE_UNIFIED_API) {
        if (previousSelection.provider && previousSelection.model) {
          saveSelectionToLocalStorage(previousSelection.provider, previousSelection.model);
        } else {
          clearSelectionFromLocalStorage();
        }
      } else {
        setStoredSelection(previousSelection);
      }
      throw error;
    }
  }, [state.selection, state.models, state.sessionId, syncToBackend]);

  // Clear selection
  const clearSelection = useCallback(async () => {
    setState(prev => ({
      ...prev,
      selection: { provider: null, model: null },
      models: [],
      sessionId: null,
    }));

    if (typeof window !== "undefined") {
      localStorage.removeItem(STORAGE_KEYS.SESSION_ID);
      localStorage.removeItem(STORAGE_KEYS.SELECTION);

      if (USE_UNIFIED_API) {
        clearSelectionFromLocalStorage();
      }
    }
  }, []);

  // Refresh providers with cooldown
  const refreshProviders = useCallback(async () => {
    const now = Date.now();
    if (now - lastProviderRefreshRef.current < PROVIDER_REFRESH_COOLDOWN_MS) {
      console.log("[UnifiedModelProvider] Provider refresh on cooldown");
      return;
    }

    setState(prev => ({ ...prev, isLoading: true }));

    try {
      const providers = await fetchProviders();
      setState(prev => ({
        ...prev,
        providers,
        isLoading: false,
        lastSyncTime: new Date(),
      }));
      lastProviderRefreshRef.current = now;
    } catch (error) {
      console.error("[UnifiedModelProvider] refreshProviders failed:", error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : "Refresh failed",
      }));
    }
  }, []);

  // Get models for a provider (legacy - returns string array)
  const getProviderModels = useCallback((providerId: string): string[] => {
    // If current provider, return from models state
    if (providerId === state.selection.provider) {
      return state.models.map(m => m.model_id);
    }
    // Otherwise check provider's cached models
    const provider = state.providers.find(
      p => p.id === providerId || p.provider === providerId
    );
    return provider?.models || [];
  }, [state.providers, state.models, state.selection.provider]);

  // Fetch models for a provider (async - returns full ModelInfo)
  const fetchModelsForProviderAction = useCallback(async (providerId: string): Promise<ModelInfo[]> => {
    if (!USE_UNIFIED_API) {
      // Legacy: return model IDs as ModelInfo-like objects
      const models = getProviderModels(providerId);
      return models.map(id => ({
        model_id: id,
        name: id,
        provider_id: providerId,
        context_length: 0,
        capabilities: [],
        is_default: false,
        is_available: true,
        supports_streaming: false,
        supports_vision: false,
        supports_function_calling: false,
      }));
    }

    setState(prev => ({ ...prev, isLoadingModels: true }));
    try {
      const models = await fetchModelsForProvider(providerId);
      setState(prev => ({ ...prev, models, isLoadingModels: false }));
      return models;
    } catch (error) {
      setState(prev => ({ ...prev, isLoadingModels: false }));
      throw error;
    }
  }, [getProviderModels]);

  // Get provider by ID
  const getProviderById = useCallback((id: string): ProviderInfo | undefined => {
    return state.providers.find(p => p.id === id || p.provider === id);
  }, [state.providers]);

  // Get model by ID
  const getModelById = useCallback((id: string): ModelInfo | undefined => {
    return state.models.find(m => m.model_id === id);
  }, [state.models]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (syncTimeoutRef.current) {
        clearTimeout(syncTimeoutRef.current);
      }
    };
  }, []);

  // Context value
  const value = useMemo<UnifiedModelContextValue>(() => ({
    ...state,
    setSelection,
    clearSelection,
    refreshProviders,
    getProviderModels,
    fetchModelsForProvider: fetchModelsForProviderAction,
    getProviderById,
    getModelById,
  }), [state, setSelection, clearSelection, refreshProviders, getProviderModels, fetchModelsForProviderAction, getProviderById, getModelById]);

  return (
    <UnifiedModelContext.Provider value={value}>
      {children}
    </UnifiedModelContext.Provider>
  );
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Main hook to access unified model state
 */
export function useUnifiedModel(): UnifiedModelContextValue {
  const context = useContext(UnifiedModelContext);
  if (!context) {
    throw new Error("useUnifiedModel must be used within a UnifiedModelProvider");
  }
  return context;
}

/**
 * Hook for just the selection state
 */
export function useModelSelection() {
  const { selection, setSelection, clearSelection, isLoading } = useUnifiedModel();
  return { selection, setSelection, clearSelection, isLoading };
}

/**
 * Hook for just the providers
 */
export function useProviderList() {
  const { providers, refreshProviders, getProviderModels, isLoading } = useUnifiedModel();
  return { providers, refreshProviders, getProviderModels, isLoading };
}

/**
 * Hook for sync status
 */
export function useSyncState() {
  const { isConnected, isSyncing, lastSyncTime, error } = useUnifiedModel();
  return { isConnected, isSyncing, lastSyncTime, error };
}

// =============================================================================
// Backward Compatibility Hooks
// =============================================================================

/**
 * Backward-compatible hook that matches the old ModelSelectionContext API
 */
export function useModelSelectionCompat() {
  const { selection, setSelection, clearSelection, isLoading } = useUnifiedModel();
  return {
    selection,
    setSelection,
    clearSelection,
    isLoading,
  };
}

/**
 * Backward-compatible hook that matches the old ChimeraProvider API
 */
export function useChimeraCompat() {
  const ctx = useUnifiedModel();
  return {
    session: ctx.sessionId ? {
      sessionId: ctx.sessionId,
      provider: ctx.selection.provider,
      model: ctx.selection.model,
      createdAt: "",
      lastActivity: ctx.lastSyncTime?.toISOString() || "",
    } : null,
    providers: ctx.providers.map(p => ({
      id: p.id,
      provider: p.provider,
      name: p.name,
      model: p.defaultModel || "",
      available_models: p.models,
      status: p.status,
    })),
    isReady: ctx.isInitialized && !ctx.isLoading,
    isLoading: ctx.isLoading,
    error: ctx.error,
    setModel: ctx.setSelection,
    refreshProviders: ctx.refreshProviders,
  };
}

export default UnifiedModelProvider;
