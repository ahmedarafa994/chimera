/**
 * Optimized Model Selection Store
 *
 * Performance improvements:
 * - Batched state updates to reduce re-renders
 * - Optimistic UI updates for instant feedback
 * - Selective subscriptions to prevent unnecessary re-renders
 * - Debounced API calls
 * - Memoized selectors
 */

import { enhancedApi } from "../api-enhanced";

// ============================================================================
// Types
// ============================================================================

export interface ProviderInfo {
  provider: string;
  displayName: string;
  status: string;
  isHealthy: boolean;
  models: string[];
  defaultModel: string | null;
  latencyMs: number | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  description?: string;
  maxTokens: number;
  isDefault: boolean;
  tier: string;
}

export interface ModelSelectionState {
  selectedProvider: string | null;
  selectedModel: string | null;
  providers: ProviderInfo[];
  models: ModelInfo[];
  sessionId: string | null;
  isDefault: boolean;
  isLoading: boolean;
  isSyncing: boolean; // Separate flag for background sync
  isInitialized: boolean;
  error: string | null;
  wsConnected: boolean;
}

// Selector types for granular subscriptions
export type StateSelector<T> = (state: ModelSelectionState) => T;

type StateListener = (state: ModelSelectionState) => void;
type SelectorListener<T> = (value: T) => void;

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = "chimera_model_selection";
const SESSION_ID_KEY = "chimera_session_id";
const WS_RECONNECT_DELAY = 3000;
const WS_MAX_RECONNECT_ATTEMPTS = 5;
const SYNC_DEBOUNCE_MS = 150;

// ============================================================================
// Optimized Model Selection Store
// ============================================================================

class OptimizedModelSelectionStore {
  private state: ModelSelectionState = {
    selectedProvider: null,
    selectedModel: null,
    providers: [],
    models: [],
    sessionId: null,
    isDefault: true,
    isLoading: false,
    isSyncing: false,
    isInitialized: false,
    error: null,
    wsConnected: false,
  };

  private listeners: Set<StateListener> = new Set();
  private selectorListeners: Map<StateSelector<unknown>, Set<SelectorListener<unknown>>> = new Map();
  private previousSelectorValues: Map<StateSelector<unknown>, unknown> = new Map();

  private ws: WebSocket | null = null;
  private wsReconnectAttempts = 0;
  private wsReconnectTimeout: NodeJS.Timeout | null = null;

  // Batching
  private pendingUpdates: Partial<ModelSelectionState>[] = [];
  private batchTimeout: NodeJS.Timeout | null = null;

  // Debouncing
  private syncTimeout: NodeJS.Timeout | null = null;
  private pendingSync: { provider: string; model: string } | null = null;

  // ============================================================================
  // State Management with Batching
  // ============================================================================

  getState(): ModelSelectionState {
    return this.state; // Return direct reference for performance (immutable updates)
  }

  private setState(partial: Partial<ModelSelectionState>, immediate = false) {
    if (immediate) {
      this.applyUpdate(partial);
      return;
    }

    this.pendingUpdates.push(partial);

    if (!this.batchTimeout) {
      this.batchTimeout = setTimeout(() => {
        this.flushUpdates();
      }, 0); // Microtask batching
    }
  }

  private flushUpdates() {
    if (this.pendingUpdates.length === 0) return;

    // Merge all pending updates
    const merged = this.pendingUpdates.reduce(
      (acc, update) => ({ ...acc, ...update }),
      {}
    );

    this.pendingUpdates = [];
    this.batchTimeout = null;

    this.applyUpdate(merged);
  }

  private applyUpdate(partial: Partial<ModelSelectionState>) {
    const prevState = this.state;
    this.state = { ...this.state, ...partial };

    // Notify full state listeners
    this.listeners.forEach((listener) => listener(this.state));

    // Notify selector listeners only if their selected value changed
    this.selectorListeners.forEach((listeners, selector) => {
      const prevValue = this.previousSelectorValues.get(selector);
      const newValue = selector(this.state);

      if (!Object.is(prevValue, newValue)) {
        this.previousSelectorValues.set(selector, newValue);
        listeners.forEach((listener) => listener(newValue));
      }
    });
  }

  // Full state subscription
  subscribe(listener: StateListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  // Granular selector subscription (prevents unnecessary re-renders)
  subscribeToSelector<T>(
    selector: StateSelector<T>,
    listener: SelectorListener<T>
  ): () => void {
    if (!this.selectorListeners.has(selector as StateSelector<unknown>)) {
      this.selectorListeners.set(selector as StateSelector<unknown>, new Set());
      this.previousSelectorValues.set(selector as StateSelector<unknown>, selector(this.state));
    }

    const listeners = this.selectorListeners.get(selector as StateSelector<unknown>)!;
    listeners.add(listener as SelectorListener<unknown>);

    // Immediately notify with current value
    listener(selector(this.state));

    return () => {
      listeners.delete(listener as SelectorListener<unknown>);
      if (listeners.size === 0) {
        this.selectorListeners.delete(selector as StateSelector<unknown>);
        this.previousSelectorValues.delete(selector as StateSelector<unknown>);
      }
    };
  }

  // ============================================================================
  // Initialization (Optimized)
  // ============================================================================

  async initialize(): Promise<void> {
    if (this.state.isInitialized) return;

    // Set loading immediately
    this.setState({ isLoading: true, error: null }, true);

    try {
      // Load from localStorage first for instant UI
      const storedSessionId = this.loadFromStorage(SESSION_ID_KEY);
      const storedSelection = this.loadFromStorage(STORAGE_KEY) as { provider?: string; model?: string } | null;

      if (storedSessionId || storedSelection) {
        this.setState({
          sessionId: storedSessionId,
          selectedProvider: storedSelection?.provider || null,
          selectedModel: storedSelection?.model || null,
        }, true);
      }

      // Fetch providers (this populates the UI)
      await this.fetchProviders();

      // Mark as initialized immediately so UI is responsive
      this.setState({ isInitialized: true, isLoading: false }, true);

      // Background tasks (don't block UI)
      this.ensureSession().catch(console.error);
      this.syncWithBackend().catch(console.error);
      this.connectWebSocket().catch(console.error);

    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to initialize";
      this.setState({ error: message, isLoading: false, isInitialized: true }, true);
    }
  }

  // ============================================================================
  // Provider/Model Fetching
  // ============================================================================

  async fetchProviders(): Promise<void> {
    try {
      const response = await fetch("/api/v1/providers/available", {
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch providers: ${response.statusText}`);
      }

      const data = await response.json();

      const providers: ProviderInfo[] = data.providers.map((p: Record<string, unknown>) => ({
        provider: p.provider as string,
        displayName: p.display_name as string,
        status: p.status as string,
        isHealthy: p.is_healthy as boolean,
        models: p.models as string[],
        defaultModel: p.default_model as string | null,
        latencyMs: p.latency_ms as number | null,
      }));

      // Only update if no selection exists
      const updates: Partial<ModelSelectionState> = { providers };
      if (!this.state.selectedProvider && data.default_provider) {
        updates.selectedProvider = data.default_provider;
      }
      if (!this.state.selectedModel && data.default_model) {
        updates.selectedModel = data.default_model;
      }

      this.setState(updates);

      // Fetch models for selected provider in background
      if (this.state.selectedProvider) {
        this.fetchModelsForProvider(this.state.selectedProvider).catch(console.error);
      }
    } catch (error) {
      console.error("Failed to fetch providers:", error);
      this.setDefaultProviders();
    }
  }

  private setDefaultProviders() {
    this.setState({
      providers: [
        {
          provider: "deepseek",
          displayName: "DeepSeek AI",
          status: "unknown",
          isHealthy: true,
          models: ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
          defaultModel: "deepseek-chat",
          latencyMs: null,
        },
        {
          provider: "google",
          displayName: "Gemini AI",
          status: "unknown",
          isHealthy: true,
          models: [
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
          ],
          defaultModel: "gemini-3-pro-preview",
          latencyMs: null,
        },
        {
          provider: "bigmodel",
          displayName: "BigModel (智谱AI)",
          status: "unknown",
          isHealthy: true,
          models: [
            "glm-4.7",
            "glm-4.6",
            "glm-4.5",
            "glm-4.5-flash",
            "glm-4-plus",
          ],
          defaultModel: "glm-4.7",
          latencyMs: null,
        },
        {
          provider: "routeway",
          displayName: "Routeway (AI Gateway)",
          status: "unknown",
          isHealthy: true,
          models: [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "llama-3.3-70b-instruct",
            "deepseek-chat",
          ],
          defaultModel: "gpt-4o-mini",
          latencyMs: null,
        },
      ],
    });
  }

  async fetchModelsForProvider(provider: string): Promise<void> {
    try {
      const response = await fetch(`/api/v1/providers/${provider}/models`, {
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) return;

      const data = await response.json();

      const models: ModelInfo[] = data.models.map((m: Record<string, unknown>) => ({
        id: m.id as string,
        name: m.name as string,
        description: m.description as string | undefined,
        maxTokens: (m.max_tokens as number) || 4096,
        isDefault: m.is_default as boolean,
        tier: (m.tier as string) || "standard",
      }));

      this.setState({ models });
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  }

  // ============================================================================
  // Selection Actions (Optimistic Updates)
  // ============================================================================

  async selectProvider(provider: string): Promise<boolean> {
    const providerInfo = this.state.providers.find((p) => p.provider === provider);
    if (!providerInfo) {
      this.setState({ error: `Provider '${provider}' not found` });
      return false;
    }

    // OPTIMISTIC UPDATE - Instant UI feedback
    const previousState = {
      selectedProvider: this.state.selectedProvider,
      selectedModel: this.state.selectedModel,
    };

    this.setState({
      selectedProvider: provider,
      selectedModel: providerInfo.defaultModel,
      error: null,
    }, true); // Immediate update

    // Fetch models in background
    this.fetchModelsForProvider(provider).catch(console.error);

    // Sync with backend (debounced)
    if (providerInfo.defaultModel) {
      this.debouncedSync(provider, providerInfo.defaultModel, previousState);
    }

    return true;
  }

  async selectModel(model: string): Promise<boolean> {
    if (!this.state.selectedProvider) {
      this.setState({ error: "No provider selected" });
      return false;
    }

    // OPTIMISTIC UPDATE - Instant UI feedback
    const previousState = {
      selectedProvider: this.state.selectedProvider,
      selectedModel: this.state.selectedModel,
    };

    this.setState({
      selectedModel: model,
      error: null,
    }, true); // Immediate update

    // Sync with backend (debounced)
    this.debouncedSync(this.state.selectedProvider, model, previousState);

    return true;
  }

  private debouncedSync(
    provider: string,
    model: string,
    previousState: { selectedProvider: string | null; selectedModel: string | null }
  ) {
    this.pendingSync = { provider, model };

    if (this.syncTimeout) {
      clearTimeout(this.syncTimeout);
    }

    this.syncTimeout = setTimeout(async () => {
      const sync = this.pendingSync;
      if (!sync) return;

      this.pendingSync = null;
      this.setState({ isSyncing: true });

      try {
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (this.state.sessionId) {
          headers["X-Session-ID"] = this.state.sessionId;
        }

        const response = await fetch("/api/v1/providers/select", {
          method: "POST",
          headers,
          body: JSON.stringify({ provider: sync.provider, model: sync.model }),
        });

        if (!response.ok) {
          throw new Error("Failed to sync selection");
        }

        const data = await response.json();

        this.setState({
          sessionId: data.session_id,
          isDefault: false,
          isSyncing: false,
        });

        // Persist to localStorage
        this.saveToStorage(SESSION_ID_KEY, data.session_id);
        this.saveToStorage(STORAGE_KEY, { provider: sync.provider, model: sync.model });

      } catch (error) {
        console.error("Sync failed:", error);

        // ROLLBACK on failure
        this.setState({
          selectedProvider: previousState.selectedProvider,
          selectedModel: previousState.selectedModel,
          error: "Failed to sync selection. Please try again.",
          isSyncing: false,
        }, true);
      }
    }, SYNC_DEBOUNCE_MS);
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  private async ensureSession(): Promise<void> {
    if (this.state.sessionId) {
      try {
        const response = await enhancedApi.session.get(this.state.sessionId);
        if (response.data) return;
      } catch {
        // Session invalid, create new
      }
    }

    try {
      const response = await enhancedApi.session.create({
        provider: this.state.selectedProvider || undefined,
        model: this.state.selectedModel || undefined,
      });

      if (response.data.success) {
        this.setState({ sessionId: response.data.session_id });
        this.saveToStorage(SESSION_ID_KEY, response.data.session_id);
      }
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  }

  private async syncWithBackend(): Promise<void> {
    try {
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (this.state.sessionId) {
        headers["X-Session-ID"] = this.state.sessionId;
      }

      const response = await fetch("/api/v1/providers/current", { headers });

      if (response.ok) {
        const data = await response.json();

        // Only update if different from current (avoid unnecessary re-renders)
        if (
          data.provider !== this.state.selectedProvider ||
          data.model !== this.state.selectedModel
        ) {
          this.setState({
            selectedProvider: data.provider,
            selectedModel: data.model,
            isDefault: data.is_default,
          });

          if (data.provider) {
            this.fetchModelsForProvider(data.provider).catch(console.error);
          }
        }
      }
    } catch (error) {
      console.error("Failed to sync with backend:", error);
    }
  }

  // ============================================================================
  // WebSocket (unchanged but with better error handling)
  // ============================================================================

  private async connectWebSocket(): Promise<void> {
    if (typeof window === "undefined") return;

    let wsApiBaseUrl: string | null = null;

    try {
      const resp = await fetch("/api/backend", { cache: "no-store" });
      if (resp.ok) {
        const info = (await resp.json()) as { apiBaseUrl?: string };
        if (info.apiBaseUrl) {
          wsApiBaseUrl = info.apiBaseUrl.replace("http://", "ws://").replace("https://", "wss://");
        }
      }
    } catch {
      // Fallback
    }

    if (!wsApiBaseUrl) {
      this.scheduleReconnect();
      return;
    }

    try {
      this.ws = new WebSocket(`${wsApiBaseUrl}/providers/ws/selection`);

      this.ws.onopen = () => {
        this.setState({ wsConnected: true });
        this.wsReconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      this.ws.onclose = () => {
        this.setState({ wsConnected: false });
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.setState({ wsConnected: false });
      };
    } catch (error) {
      console.error("Failed to connect WebSocket:", error);
      this.scheduleReconnect();
    }
  }

  private handleWebSocketMessage(message: { type: string; data: Record<string, unknown> }): void {
    switch (message.type) {
      case "selection_change":
        if (message.data.session_id === this.state.sessionId) {
          this.setState({
            selectedProvider: message.data.provider as string,
            selectedModel: message.data.model as string,
            isDefault: false,
          });
        }
        break;

      case "health_update":
        const healthData = message.data.providers as Array<{
          provider: string;
          is_healthy: boolean;
          latency_ms: number | null;
        }>;
        if (healthData) {
          const updatedProviders = this.state.providers.map((p) => {
            const health = healthData.find((h) => h.provider === p.provider);
            return health ? { ...p, isHealthy: health.is_healthy, latencyMs: health.latency_ms } : p;
          });
          this.setState({ providers: updatedProviders });
        }
        break;

      case "heartbeat":
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "ping" }));
        }
        break;
    }
  }

  private scheduleReconnect(): void {
    if (this.wsReconnectAttempts >= WS_MAX_RECONNECT_ATTEMPTS) return;

    if (this.wsReconnectTimeout) {
      clearTimeout(this.wsReconnectTimeout);
    }

    this.wsReconnectTimeout = setTimeout(() => {
      this.wsReconnectAttempts++;
      void this.connectWebSocket();
    }, WS_RECONNECT_DELAY);
  }

  disconnectWebSocket(): void {
    if (this.wsReconnectTimeout) clearTimeout(this.wsReconnectTimeout);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setState({ wsConnected: false });
  }

  // ============================================================================
  // Storage
  // ============================================================================

  private saveToStorage(key: string, value: unknown): void {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error("Failed to save to localStorage:", error);
    }
  }

  private loadFromStorage(key: string): string | null {
    if (typeof window === "undefined") return null;
    try {
      const value = localStorage.getItem(key);
      if (value) {
        try {
          const parsed = JSON.parse(value);
          return typeof parsed === "string" ? parsed : parsed;
        } catch {
          return value;
        }
      }
    } catch (error) {
      console.error("Failed to load from localStorage:", error);
    }
    return null;
  }

  // ============================================================================
  // Memoized Selectors
  // ============================================================================

  getSelectedProviderInfo(): ProviderInfo | null {
    return this.state.providers.find((p) => p.provider === this.state.selectedProvider) || null;
  }

  getSelectedModelInfo(): ModelInfo | null {
    return this.state.models.find((m) => m.id === this.state.selectedModel) || null;
  }

  reset(): void {
    this.disconnectWebSocket();
    if (typeof window !== "undefined") {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(SESSION_ID_KEY);
    }
    this.state = {
      selectedProvider: null,
      selectedModel: null,
      providers: [],
      models: [],
      sessionId: null,
      isDefault: true,
      isLoading: false,
      isSyncing: false,
      isInitialized: false,
      error: null,
      wsConnected: false,
    };
    this.listeners.forEach((listener) => listener(this.state));
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const optimizedModelSelectionStore = new OptimizedModelSelectionStore();

// ============================================================================
// Optimized React Hook with Selective Subscriptions
// ============================================================================

import { useState, useEffect, useCallback, useRef, useMemo } from "react";

// Selectors for granular subscriptions
export const selectors = {
  selection: (state: ModelSelectionState) => ({
    provider: state.selectedProvider,
    model: state.selectedModel,
  }),
  providers: (state: ModelSelectionState) => state.providers,
  models: (state: ModelSelectionState) => state.models,
  loading: (state: ModelSelectionState) => state.isLoading,
  syncing: (state: ModelSelectionState) => state.isSyncing,
  error: (state: ModelSelectionState) => state.error,
  wsConnected: (state: ModelSelectionState) => state.wsConnected,
  sessionId: (state: ModelSelectionState) => state.sessionId,
};

// Hook for full state (use sparingly)
export function useOptimizedModelSelection() {
  const [state, setState] = useState<ModelSelectionState>(optimizedModelSelectionStore.getState());
  const initRef = useRef(false);

  useEffect(() => {
    const unsubscribe = optimizedModelSelectionStore.subscribe(setState);

    if (!initRef.current) {
      initRef.current = true;
      optimizedModelSelectionStore.initialize();
    }

    return unsubscribe;
  }, []);

  const selectProvider = useCallback((provider: string) => {
    return optimizedModelSelectionStore.selectProvider(provider);
  }, []);

  const selectModel = useCallback((model: string) => {
    return optimizedModelSelectionStore.selectModel(model);
  }, []);

  const refresh = useCallback(async () => {
    await optimizedModelSelectionStore.fetchProviders();
  }, []);

  return useMemo(() => ({
    ...state,
    selectProvider,
    selectModel,
    refresh,
    getSelectedProviderInfo: () => optimizedModelSelectionStore.getSelectedProviderInfo(),
    getSelectedModelInfo: () => optimizedModelSelectionStore.getSelectedModelInfo(),
  }), [state, selectProvider, selectModel, refresh]);
}

// Hook for selection only (most common use case)
export function useModelSelectionValue() {
  const [selection, setSelection] = useState(() => selectors.selection(optimizedModelSelectionStore.getState()));
  const initRef = useRef(false);

  useEffect(() => {
    const unsubscribe = optimizedModelSelectionStore.subscribeToSelector(
      selectors.selection,
      setSelection
    );

    if (!initRef.current) {
      initRef.current = true;
      optimizedModelSelectionStore.initialize();
    }

    return unsubscribe;
  }, []);

  return selection;
}

// Hook for providers list only
export function useProvidersList() {
  const [providers, setProviders] = useState(() => selectors.providers(optimizedModelSelectionStore.getState()));

  useEffect(() => {
    return optimizedModelSelectionStore.subscribeToSelector(selectors.providers, setProviders);
  }, []);

  return providers;
}

// Hook for models list only
export function useModelsList() {
  const [models, setModels] = useState(() => selectors.models(optimizedModelSelectionStore.getState()));

  useEffect(() => {
    return optimizedModelSelectionStore.subscribeToSelector(selectors.models, setModels);
  }, []);

  return models;
}

// Hook for loading state only
export function useModelSelectionLoading() {
  const [isLoading, setIsLoading] = useState(() => selectors.loading(optimizedModelSelectionStore.getState()));
  const [isSyncing, setIsSyncing] = useState(() => selectors.syncing(optimizedModelSelectionStore.getState()));

  useEffect(() => {
    const unsub1 = optimizedModelSelectionStore.subscribeToSelector(selectors.loading, setIsLoading);
    const unsub2 = optimizedModelSelectionStore.subscribeToSelector(selectors.syncing, setIsSyncing);
    return () => {
      unsub1();
      unsub2();
    };
  }, []);

  return { isLoading, isSyncing };
}

// Hook for actions only (no state subscription)
export function useModelSelectionActions() {
  const selectProvider = useCallback((provider: string) => {
    return optimizedModelSelectionStore.selectProvider(provider);
  }, []);

  const selectModel = useCallback((model: string) => {
    return optimizedModelSelectionStore.selectModel(model);
  }, []);

  const refresh = useCallback(async () => {
    await optimizedModelSelectionStore.fetchProviders();
  }, []);

  return { selectProvider, selectModel, refresh };
}
