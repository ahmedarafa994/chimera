/**
 * Model Selection Store for Multi-Provider AI Model Selection
 *
 * This store manages the state for provider/model selection with:
 * - WebSocket real-time synchronization
 * - localStorage persistence for client preferences
 * - Backend session synchronization
 */

/**
 * @deprecated This store is deprecated and will be removed in a future update.
 * Please use `UnifiedModelProvider` and `useUnifiedModel` instead.
 *
 * Known issues:
 * - Causes hydration mismatches due to duplicate state management
 * - causing race conditions with UnifiedModelProvider
 */
import { enhancedApi } from "../api-enhanced";
// Note: getApiConfig imported but currently unused - kept for future use

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
  // Current selection
  selectedProvider: string | null;
  selectedModel: string | null;

  // Available options
  providers: ProviderInfo[];
  models: ModelInfo[];

  // Session info
  sessionId: string | null;
  isDefault: boolean;

  // Loading/error states
  isLoading: boolean;
  isInitialized: boolean;
  error: string | null;

  // WebSocket connection state
  wsConnected: boolean;
}

export interface WebSocketMessage {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

type StateListener = (state: ModelSelectionState) => void;

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEY = "chimera_model_selection";
const SESSION_ID_KEY = "chimera_session_id";
const WS_RECONNECT_DELAY = 3000;
const WS_MAX_RECONNECT_ATTEMPTS = 5;

// ============================================================================
// Model Selection Store Class
// ============================================================================

class ModelSelectionStoreClass {
  private state: ModelSelectionState = {
    selectedProvider: null,
    selectedModel: null,
    providers: [],
    models: [],
    sessionId: null,
    isDefault: true,
    isLoading: false,
    isInitialized: false,
    error: null,
    wsConnected: false,
  };

  private listeners: Set<StateListener> = new Set();
  private ws: WebSocket | null = null;
  private wsReconnectAttempts = 0;
  private wsReconnectTimeout: NodeJS.Timeout | null = null;

  // ============================================================================
  // State Management
  // ============================================================================

  getState(): ModelSelectionState {
    return { ...this.state };
  }

  private setState(partial: Partial<ModelSelectionState>) {
    this.state = { ...this.state, ...partial };
    this.notifyListeners();
  }

  subscribe(listener: StateListener): () => void {
    this.listeners.add(listener);
    // Immediately notify with current state
    listener(this.getState());
    return () => this.listeners.delete(listener);
  }

  private notifyListeners() {
    const state = this.getState();
    this.listeners.forEach((listener) => listener(state));
  }

  // ============================================================================
  // Initialization
  // ============================================================================

  async initialize(): Promise<void> {
    if (this.state.isInitialized) return;

    this.setState({ isLoading: true, error: null });

    try {
      // Load session ID from localStorage
      const storedSessionId = this.loadFromStorage(SESSION_ID_KEY);
      if (storedSessionId) {
        this.setState({ sessionId: storedSessionId });
      }

      // Fetch available providers
      await this.fetchProviders();

      // Get or create session
      await this.ensureSession();

      // Load current selection from backend
      await this.syncWithBackend();

      // Connect WebSocket for real-time updates
      await this.connectWebSocket();

      this.setState({ isInitialized: true, isLoading: false });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to initialize";
      this.setState({ error: message, isLoading: false });
      console.error("ModelSelectionStore initialization failed:", error);
    }
  }

  // ============================================================================
  // Provider/Model Fetching
  // ============================================================================

  async fetchProviders(): Promise<void> {
    try {
      // Use relative path for consistency with unified-model-provider
      // This relies on Next.js rewrites to proxy to the backend
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/providers`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch providers: ${response.statusText}`);
      }

      const data = await response.json();

      // Map the response exactly as UnifiedModelProvider does
      const providersList = data.providers || [];
      const providers: ProviderInfo[] = providersList.map((p: Record<string, unknown>) => ({
        provider: p.provider,
        displayName: p.display_name || p.provider,
        status: p.is_healthy ? "healthy" : "unhealthy",
        isHealthy: p.is_healthy,
        models: p.models || [],
        defaultModel: p.default_model || null,
        latencyMs: null,
      }));

      const defaultProvider = data.default_provider || (providers.length > 0 ? providers[0].provider : null);
      const defaultModel = data.default_model || (defaultProvider && providers.find((p: ProviderInfo) => p.provider === defaultProvider)?.defaultModel) || null;

      this.setState({
        providers,
        selectedProvider: this.state.selectedProvider || defaultProvider,
        selectedModel: this.state.selectedModel || defaultModel,
        isLoading: false,
        error: null
      });

      // If we have a selected provider, fetch its models (Legacy support)
      if (this.state.selectedProvider) {
        // keeping for safety
      }
    } catch (error) {
      console.error("Failed to fetch providers:", error);
      // Set default providers if fetch fails - Gemini AI as primary
      this.setState({
        providers: [
          {
            provider: "google",
            displayName: "Gemini AI",
            status: "unknown",
            isHealthy: true,
            models: [
              "gemini-3-pro-preview",
              "gemini-3-pro-image-preview",
              "gemini-2.5-pro",
              "gemini-2.5-pro-preview-06-05",
              "gemini-2.5-flash",
              "gemini-2.5-flash-lite",
              "gemini-2.5-flash-image",
            ],
            defaultModel: "gemini-3-pro-preview",
            latencyMs: null,
          },
          {
            provider: "gemini-cli",
            displayName: "Gemini CLI (OAuth)",
            status: "unknown",
            isHealthy: true,
            models: [
              "gemini-3-pro-preview",
              "gemini-2.5-pro",
              "gemini-2.5-flash",
              "gemini-2.5-flash-lite",
            ],
            defaultModel: "gemini-3-pro-preview",
            latencyMs: null,
          },
          {
            provider: "antigravity",
            displayName: "Antigravity (Hybrid)",
            status: "unknown",
            isHealthy: true,
            models: [
              "gemini-claude-sonnet-4-5-thinking",
              "gemini-claude-sonnet-4-5",
              "gemini-3-pro-preview",
              "gemini-2.5-flash",
            ],
            defaultModel: "gemini-claude-sonnet-4-5",
            latencyMs: null,
          },
          {
            provider: "anthropic",
            displayName: "Anthropic Claude",
            status: "unknown",
            isHealthy: true,
            models: [
              "claude-opus-4-5",
              "claude-sonnet-4-5",
              "claude-haiku-4-5",
            ],
            defaultModel: "claude-sonnet-4-5",
            latencyMs: null,
          },
          {
            provider: "openai",
            displayName: "OpenAI",
            status: "unknown",
            isHealthy: true,
            models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o3"],
            defaultModel: "gpt-4o",
            latencyMs: null,
          },
          {
            provider: "deepseek",
            displayName: "DeepSeek",
            status: "unknown",
            isHealthy: true,
            models: ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"],
            defaultModel: "deepseek-chat",
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
          {
            provider: "qwen",
            displayName: "Qwen",
            status: "unknown",
            isHealthy: true,
            models: ["qwen3-coder-plus", "qwen3-coder-flash"],
            defaultModel: "qwen3-coder-plus",
            latencyMs: null,
          },
        ],
      });
    }
  }

  async fetchModelsForProvider(provider: string): Promise<void> {
    try {
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/providers/${provider}/models`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

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
      console.error("Failed to fetch models for provider:", error);
    }
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  private async ensureSession(): Promise<void> {
    if (this.state.sessionId) {
      // Verify session exists
      try {
        const response = await enhancedApi.session.get(this.state.sessionId);
        if (response.data) {
          return; // Session is valid
        }
      } catch {
        // Session doesn't exist, create new one
      }
    }

    // Create new session
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
      const baseUrl = "/api/v1";

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (this.state.sessionId) {
        headers["X-Session-ID"] = this.state.sessionId;
      }

      const response = await fetch(`${baseUrl}/providers/current`, { headers });

      if (response.ok) {
        const data = await response.json();
        this.setState({
          selectedProvider: data.provider,
          selectedModel: data.model,
          isDefault: data.is_default,
        });

        // Fetch models for the current provider
        if (data.provider) {
          await this.fetchModelsForProvider(data.provider);
        }
      }
    } catch (error) {
      console.error("Failed to sync with backend:", error);
    }
  }

  // ============================================================================
  // Selection Actions
  // ============================================================================

  async selectProvider(provider: string): Promise<boolean> {
    const providerInfo = this.state.providers.find((p) => p.provider === provider);
    if (!providerInfo) {
      this.setState({ error: `Provider '${provider}' not found` });
      return false;
    }

    this.setState({
      selectedProvider: provider,
      selectedModel: providerInfo.defaultModel,
      isLoading: true,
      error: null,
    });

    // Fetch models for the new provider
    await this.fetchModelsForProvider(provider);

    // Update backend
    if (providerInfo.defaultModel) {
      return this.selectModel(providerInfo.defaultModel);
    }

    this.setState({ isLoading: false });
    return true;
  }

  async selectModel(model: string): Promise<boolean> {
    if (!this.state.selectedProvider) {
      this.setState({ error: "No provider selected" });
      return false;
    }

    this.setState({ isLoading: true, error: null });

    try {
      const baseUrl = "/api/v1";

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      if (this.state.sessionId) {
        headers["X-Session-ID"] = this.state.sessionId;
      }

      const response = await fetch(`${baseUrl}/providers/select`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          provider: this.state.selectedProvider,
          model,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to select model");
      }

      const data = await response.json();

      this.setState({
        selectedModel: data.model,
        selectedProvider: data.provider,
        sessionId: data.session_id,
        isDefault: false,
        isLoading: false,
      });

      // Save session ID
      this.saveToStorage(SESSION_ID_KEY, data.session_id);

      // Save selection to localStorage for quick restore
      this.saveToStorage(STORAGE_KEY, {
        provider: data.provider,
        model: data.model,
      });

      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to select model";
      this.setState({ error: message, isLoading: false });
      return false;
    }
  }

  // ============================================================================
  // WebSocket Connection
  // ============================================================================

  private async connectWebSocket(): Promise<void> {
    if (typeof window === "undefined") return; // SSR guard

    const apiBaseUrl = "/api/v1";

    let wsApiBaseUrl: string | null = null;
    if (apiBaseUrl.startsWith("http://") || apiBaseUrl.startsWith("https://")) {
      wsApiBaseUrl = apiBaseUrl.replace("http://", "ws://").replace("https://", "wss://");
    } else {
      // Direct mode uses same-origin HTTP; resolve the actual backend origin for WebSocket.
      try {
        const resp = await fetch("/api/backend", { cache: "no-store" });
        if (resp.ok) {
          const info = (await resp.json()) as { apiBaseUrl?: string };
          if (info.apiBaseUrl) {
            wsApiBaseUrl = info.apiBaseUrl.replace("http://", "ws://").replace("https://", "wss://");
          }
        }
      } catch {
        // Ignore and fall back to reconnect scheduling below.
      }
    }

    if (!wsApiBaseUrl) {
      this.setState({ wsConnected: false });
      this.scheduleReconnect();
      return;
    }

    try {
      this.ws = new WebSocket(`${wsApiBaseUrl}/providers/ws/selection`);

      this.ws.onopen = () => {
        console.log("ModelSelectionStore: WebSocket connected");
        this.setState({ wsConnected: true });
        this.wsReconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleWebSocketMessage(message);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      this.ws.onclose = () => {
        console.log("ModelSelectionStore: WebSocket disconnected");
        this.setState({ wsConnected: false });
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        // Fix: Use generic error message if error object is empty
        const errorMessage = error instanceof Event ? "Connection error" : String(error);
        console.error("ModelSelectionStore: WebSocket error:", errorMessage);
        this.setState({ wsConnected: false });
      };
    } catch (error) {
      console.error("Failed to connect WebSocket:", error);
      this.scheduleReconnect();
    }
  }

  private handleWebSocketMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case "selection_change":
        // Another client changed the selection for this session
        if (message.data.session_id === this.state.sessionId) {
          this.setState({
            selectedProvider: message.data.provider as string,
            selectedModel: message.data.model as string,
            isDefault: false,
          });
        }
        break;

      case "health_update":
        // Provider health status changed
        const healthData = message.data.providers as Array<{
          provider: string;
          is_healthy: boolean;
          latency_ms: number | null;
        }>;
        if (healthData) {
          const updatedProviders = this.state.providers.map((p) => {
            const health = healthData.find((h) => h.provider === p.provider);
            if (health) {
              return {
                ...p,
                isHealthy: health.is_healthy,
                latencyMs: health.latency_ms,
              };
            }
            return p;
          });
          this.setState({ providers: updatedProviders });
        }
        break;

      case "heartbeat":
        // Send pong response
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "ping" }));
        }
        break;

      case "connected":
        console.log("WebSocket connection confirmed:", message.data.client_id);
        break;
    }
  }

  private scheduleReconnect(): void {
    if (this.wsReconnectAttempts >= WS_MAX_RECONNECT_ATTEMPTS) {
      console.log("Max WebSocket reconnect attempts reached");
      return;
    }

    if (this.wsReconnectTimeout) {
      clearTimeout(this.wsReconnectTimeout);
    }

    this.wsReconnectTimeout = setTimeout(() => {
      this.wsReconnectAttempts++;
      console.log(`Attempting WebSocket reconnect (${this.wsReconnectAttempts}/${WS_MAX_RECONNECT_ATTEMPTS})`);
      void this.connectWebSocket();
    }, WS_RECONNECT_DELAY);
  }

  disconnectWebSocket(): void {
    if (this.wsReconnectTimeout) {
      clearTimeout(this.wsReconnectTimeout);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setState({ wsConnected: false });
  }

  // ============================================================================
  // Local Storage
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
        // Try to parse as JSON first
        try {
          const parsed = JSON.parse(value);
          // If it's a string, return it directly
          // If it's an object (like {provider, model}), return it as-is
          return typeof parsed === "string" ? parsed : parsed;
        } catch {
          // If JSON parsing fails, the value might be a plain string (e.g., old UUID format)
          // Return it directly
          return value;
        }
      }
    } catch (error) {
      console.error("Failed to load from localStorage:", error);
    }
    return null;
  }

  // ============================================================================
  // Utility Methods
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
      isInitialized: false,
      error: null,
      wsConnected: false,
    };
    this.notifyListeners();
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const ModelSelectionStore = new ModelSelectionStoreClass();

// ============================================================================
// React Hook
// ============================================================================

import { useState, useEffect, useCallback } from "react";

export function useModelSelection() {
  const [state, setState] = useState<ModelSelectionState>(ModelSelectionStore.getState());

  useEffect(() => {
    const unsubscribe = ModelSelectionStore.subscribe(setState);

    // Initialize on first mount
    if (!state.isInitialized && !state.isLoading) {
      ModelSelectionStore.initialize();
    }

    return unsubscribe;
  }, []);

  const selectProvider = useCallback((provider: string) => {
    return ModelSelectionStore.selectProvider(provider);
  }, []);

  const selectModel = useCallback((model: string) => {
    return ModelSelectionStore.selectModel(model);
  }, []);

  const refresh = useCallback(async () => {
    await ModelSelectionStore.fetchProviders();
  }, []);

  return {
    ...state,
    selectProvider,
    selectModel,
    refresh,
    getSelectedProviderInfo: () => ModelSelectionStore.getSelectedProviderInfo(),
    getSelectedModelInfo: () => ModelSelectionStore.getSelectedModelInfo(),
  };
}


