"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  Provider,
  ProviderStatus,
  ProviderModel,
} from "@/components/providers/ProviderSelector";
import { type ProviderType } from "@/lib/api/validation";
import type {
  ProviderConfigData,
} from "@/components/providers/ProviderConfigForm";

// =============================================================================
// API Configuration
// =============================================================================

// Use NEXT_PUBLIC_CHIMERA_API_URL which is the standard env var for this project
// It already includes /api/v1, so we just append the endpoint path
const CHIMERA_API_URL = process.env.NEXT_PUBLIC_CHIMERA_API_URL || "http://localhost:8001/api/v1";
const PROVIDER_CONFIG_API = `${CHIMERA_API_URL}/provider-config`;

// =============================================================================
// Types
// =============================================================================

export interface ApiProvider {
  provider_id: string;
  name: string;
  provider_type: ProviderType;
  base_url?: string;
  is_enabled: boolean;
  priority: number;
  models: ApiModel[];
  capabilities: string[];
  rate_limit_rpm?: number;
  rate_limit_tpm?: number;
  max_retries: number;
  timeout: number;
  custom_headers?: Record<string, string>;
  metadata?: Record<string, unknown>;
}

export interface ApiModel {
  model_id: string;
  name: string;
  description?: string;
  context_window?: number;
  max_output_tokens?: number;
  capabilities: string[];
  pricing?: {
    input_per_1k?: number;
    output_per_1k?: number;
  };
}

export interface ApiHealthStatus {
  provider_id: string;
  status: string;
  last_check?: string;
  response_time_ms?: number;
  error_rate?: number;
  consecutive_failures: number;
  last_error?: string;
  circuit_breaker_state: string;
}

export interface ProviderState {
  providers: Provider[];
  activeProviderId: string | null;
  activeModelId: string | null;
  healthStatuses: Record<string, ApiHealthStatus>;
  isConnected: boolean;
}

// Types for API responses (used by api-enhanced.ts)
export interface ProviderInfo {
  provider_id: string;
  name: string;
  provider_type: string;
  is_enabled: boolean;
  is_configured: boolean;
  status: string;
  models_count: number;
}



export interface ProviderHealthStatus {
  provider_id: string;
  status: string;
  latency_ms: number | null;
  last_check: string;
  error_message?: string;
}

export interface ProviderConfigResponse {
  provider_id: string;
  name: string;
  provider_type: string;
  base_url?: string;
  is_enabled: boolean;
  has_api_key: boolean;
  models: ApiModel[];
}

export interface ProviderListResponse {
  providers: ProviderInfo[];
  count: number;
}

export interface ActiveProviderResponse {
  provider_id: string;
  provider_name: string;
  model: string;
  status: string;
}

export interface SetActiveProviderRequest {
  provider_id: string;
  model_id?: string;
}

export interface UpdateProviderConfigRequest {
  api_key?: string;
  base_url?: string;
  is_enabled?: boolean;
  max_retries?: number;
  timeout?: number;
  custom_headers?: Record<string, string>;
}

export interface ProviderHealthResponse {
  provider_id: string;
  status: string;
  latency_ms: number | null;
  last_check: string;
  error_message?: string;
}

// =============================================================================
// API Functions
// =============================================================================

async function fetchProviders(): Promise<ApiProvider[]> {
  const response = await fetch(`${PROVIDER_CONFIG_API}/providers`);
  if (!response.ok) {
    throw new Error(`Failed to fetch providers: ${response.statusText}`);
  }
  const data = await response.json();
  return data.providers || [];
}

async function fetchActiveProvider(): Promise<{
  provider_id: string | null;
  model_id: string | null;
}> {
  const response = await fetch(`${PROVIDER_CONFIG_API}/providers/active`);
  if (!response.ok) {
    throw new Error(`Failed to fetch active provider: ${response.statusText}`);
  }
  return response.json();
}

async function setActiveProvider(
  providerId: string,
  modelId?: string
): Promise<void> {
  const response = await fetch(`${PROVIDER_CONFIG_API}/providers/active`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider_id: providerId, model_id: modelId }),
  });
  if (!response.ok) {
    throw new Error(`Failed to set active provider: ${response.statusText}`);
  }
}

async function fetchHealthStatuses(): Promise<Record<string, ApiHealthStatus>> {
  const response = await fetch(`${PROVIDER_CONFIG_API}/health`);
  if (!response.ok) {
    throw new Error(`Failed to fetch health statuses: ${response.statusText}`);
  }
  const data = await response.json();
  return data.health || {};
}

async function createProvider(config: ProviderConfigData): Promise<ApiProvider> {
  const payload = {
    provider_id: config.id || `${config.type}-${Date.now()}`,
    name: config.name,
    provider_type: config.type,
    api_key: config.apiKey,
    base_url: config.baseUrl,
    is_enabled: config.enabled ?? true,
    max_retries: config.maxRetries || 3,
    timeout: config.timeout || 30,
    custom_headers: config.customHeaders,
    metadata: {
      organization_id: config.organizationId,
      project_id: config.projectId,
    },
  };

  const response = await fetch(`${PROVIDER_CONFIG_API}/providers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to create provider: ${response.statusText}`);
  }

  return response.json();
}

async function updateProvider(
  providerId: string,
  config: Partial<ProviderConfigData>
): Promise<ApiProvider> {
  const payload: Record<string, unknown> = {};

  if (config.name !== undefined) payload.name = config.name;
  if (config.apiKey) payload.api_key = config.apiKey;
  if (config.baseUrl !== undefined) payload.base_url = config.baseUrl;
  if (config.enabled !== undefined) payload.is_enabled = config.enabled;
  if (config.maxRetries !== undefined) payload.max_retries = config.maxRetries;
  if (config.timeout !== undefined) payload.timeout = config.timeout;
  if (config.customHeaders !== undefined) payload.custom_headers = config.customHeaders;

  const response = await fetch(`${PROVIDER_CONFIG_API}/providers/${providerId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to update provider: ${response.statusText}`);
  }

  return response.json();
}

async function deleteProvider(providerId: string): Promise<void> {
  const response = await fetch(`${PROVIDER_CONFIG_API}/providers/${providerId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to delete provider: ${response.statusText}`);
  }
}

async function testProviderConnection(
  config: ProviderConfigData
): Promise<{ success: boolean; message: string }> {
  const payload = {
    provider_type: config.type,
    api_key: config.apiKey,
    base_url: config.baseUrl,
  };

  const response = await fetch(`${PROVIDER_CONFIG_API}/providers/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json();

  if (!response.ok) {
    return {
      success: false,
      message: data.detail || `Connection test failed: ${response.statusText}`,
    };
  }

  return {
    success: data.success ?? true,
    message: data.message || "Connection successful",
  };
}

async function refreshProviderHealth(providerId: string): Promise<ApiHealthStatus> {
  const response = await fetch(
    `${PROVIDER_CONFIG_API}/health/${providerId}/refresh`,
    { method: "POST" }
  );

  if (!response.ok) {
    throw new Error(`Failed to refresh health: ${response.statusText}`);
  }

  return response.json();
}

// =============================================================================
// Transform Functions
// =============================================================================

function transformApiProvider(api: ApiProvider, health?: ApiHealthStatus): Provider {
  const statusMap: Record<string, ProviderStatus> = {
    available: "available",
    unavailable: "unavailable",
    degraded: "degraded",
    unknown: "unknown",
  };

  return {
    id: api.provider_id,
    name: api.name,
    type: api.provider_type,
    status: health ? statusMap[health.status] || "unknown" : "unknown",
    isActive: false, // Will be set by the hook
    models: api.models.map((m) => ({
      id: m.model_id,
      name: m.name,
      description: m.description,
      contextWindow: m.context_window,
      maxTokens: m.max_output_tokens,
      capabilities: m.capabilities,
    })),
    description: api.metadata?.description as string | undefined,
    baseUrl: api.base_url,
    hasApiKey: true, // API doesn't expose this directly
    enabled: api.is_enabled,
    lastHealthCheck: health?.last_check,
    responseTime: health?.response_time_ms,
    errorRate: health?.error_rate,
  };
}

// =============================================================================
// Query Keys
// =============================================================================

export const providerQueryKeys = {
  all: ["providers"] as const,
  list: () => [...providerQueryKeys.all, "list"] as const,
  active: () => [...providerQueryKeys.all, "active"] as const,
  health: () => [...providerQueryKeys.all, "health"] as const,
  detail: (id: string) => [...providerQueryKeys.all, "detail", id] as const,
};

// =============================================================================
// Main Hook: useProviderConfig
// =============================================================================

export function useProviderConfig() {
  const queryClient = useQueryClient();

  // Fetch providers
  const providersQuery = useQuery({
    queryKey: providerQueryKeys.list(),
    queryFn: fetchProviders,
    staleTime: 30000, // 30 seconds
    refetchInterval: 60000, // 1 minute
  });

  // Fetch active provider
  const activeQuery = useQuery({
    queryKey: providerQueryKeys.active(),
    queryFn: fetchActiveProvider,
    staleTime: 10000, // 10 seconds
  });

  // Fetch health statuses
  const healthQuery = useQuery({
    queryKey: providerQueryKeys.health(),
    queryFn: fetchHealthStatuses,
    staleTime: 15000, // 15 seconds
    refetchInterval: 30000, // 30 seconds
  });

  // Transform providers with health data
  const providers: Provider[] = (providersQuery.data || []).map((api) => {
    const health = healthQuery.data?.[api.provider_id];
    const provider = transformApiProvider(api, health);
    provider.isActive = api.provider_id === activeQuery.data?.provider_id;
    return provider;
  });

  // Set active provider mutation
  const setActiveMutation = useMutation({
    mutationFn: ({ providerId, modelId }: { providerId: string; modelId?: string }) =>
      setActiveProvider(providerId, modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: providerQueryKeys.active() });
    },
  });

  // Create provider mutation
  const createMutation = useMutation({
    mutationFn: createProvider,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: providerQueryKeys.list() });
    },
  });

  // Update provider mutation
  const updateMutation = useMutation({
    mutationFn: ({ providerId, config }: { providerId: string; config: Partial<ProviderConfigData> }) =>
      updateProvider(providerId, config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: providerQueryKeys.list() });
    },
  });

  // Delete provider mutation
  const deleteMutation = useMutation({
    mutationFn: deleteProvider,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: providerQueryKeys.list() });
    },
  });

  // Refresh health mutation
  const refreshHealthMutation = useMutation({
    mutationFn: refreshProviderHealth,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: providerQueryKeys.health() });
    },
  });

  // Refresh all data
  const refreshAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: providerQueryKeys.all });
  }, [queryClient]);

  return {
    // Data
    providers,
    activeProviderId: activeQuery.data?.provider_id || null,
    activeModelId: activeQuery.data?.model_id || null,
    healthStatuses: healthQuery.data || {},

    // Loading states
    isLoading: providersQuery.isLoading || activeQuery.isLoading,
    isRefreshing: providersQuery.isFetching || healthQuery.isFetching,
    isChangingProvider: setActiveMutation.isPending,
    isSaving: createMutation.isPending || updateMutation.isPending,
    isDeleting: deleteMutation.isPending,

    // Errors
    error: providersQuery.error || activeQuery.error || healthQuery.error,

    // Actions
    setActiveProvider: (providerId: string, modelId?: string) =>
      setActiveMutation.mutateAsync({ providerId, modelId }),
    createProvider: (config: ProviderConfigData) =>
      createMutation.mutateAsync(config),
    updateProvider: (providerId: string, config: Partial<ProviderConfigData>) =>
      updateMutation.mutateAsync({ providerId, config }),
    deleteProvider: (providerId: string) =>
      deleteMutation.mutateAsync(providerId),
    testConnection: testProviderConnection,
    refreshHealth: (providerId: string) =>
      refreshHealthMutation.mutateAsync(providerId),
    refreshAll,
  };
}

// =============================================================================
// WebSocket Hook: useProviderWebSocket
// =============================================================================

interface WebSocketMessage {
  type: string;
  provider_id?: string;
  data?: unknown;
  timestamp?: string;
}

export function useProviderWebSocket(enabled: boolean = true) {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 3;
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const connect = useCallback(() => {
    // Don't connect if already connected or max attempts reached
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (reconnectAttempts.current >= maxReconnectAttempts) {
      console.log("[ProviderWS] Max reconnect attempts reached, using polling fallback");
      setConnectionError("WebSocket unavailable, using polling for updates");
      return;
    }

    try {
      // Build WebSocket URL from the provider config API URL
      const wsUrl = PROVIDER_CONFIG_API.replace(/^http/, "ws") + "/ws/updates";
      console.log("[ProviderWS] Connecting to:", wsUrl);
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("[ProviderWS] Connected");
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttempts.current = 0; // Reset on successful connection
      };

      ws.onmessage = (event) => {
        const rawData = event.data;
        if (typeof rawData === "string") {
          const trimmed = rawData.trim();
          if (trimmed === "ping") {
            ws.send("pong");
            return;
          }
          if (trimmed === "pong") {
            return;
          }
        }

        try {
          if (typeof rawData !== "string") {
            console.warn("[ProviderWS] Unsupported message type:", typeof rawData);
            return;
          }

          const message: WebSocketMessage = JSON.parse(rawData);
          setLastMessage(message);

          // Handle different message types
          switch (message.type) {
            case "initial_state":
            case "refresh_response":
              // Full state update - invalidate all queries
              queryClient.invalidateQueries({ queryKey: providerQueryKeys.all });
              break;

            case "provider_updated":
            case "provider_registered":
            case "provider_deregistered":
              // Provider list changed
              queryClient.invalidateQueries({ queryKey: providerQueryKeys.list() });
              break;

            case "provider_status_changed":
              // Health status changed
              queryClient.invalidateQueries({ queryKey: providerQueryKeys.health() });
              break;

            case "active_provider_changed":
              // Active provider changed
              queryClient.invalidateQueries({ queryKey: providerQueryKeys.active() });
              break;

            case "pong":
              // Heartbeat response - no action needed
              break;

            default:
              console.log("[ProviderWS] Unknown message type:", message.type);
          }
        } catch (error) {
          console.error("[ProviderWS] Failed to parse message:", error);
        }
      };

      ws.onclose = (event) => {
        console.log("[ProviderWS] Disconnected:", event.code, event.reason || "No reason provided");
        setIsConnected(false);
        wsRef.current = null;

        // Reconnect after delay (unless intentionally closed or max attempts reached)
        if (enabled && event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          const delay = Math.min(5000 * reconnectAttempts.current, 15000); // Exponential backoff, max 15s
          console.log(`[ProviderWS] Attempting reconnect ${reconnectAttempts.current}/${maxReconnectAttempts} in ${delay}ms...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connectImpl();
          }, delay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setConnectionError("WebSocket connection failed, using polling for updates");
        }
      };

      ws.onerror = () => {
        // WebSocket error events don't contain useful information for security reasons
        // The actual error will be handled in onclose
        console.warn("[ProviderWS] Connection error occurred (details unavailable for security)");
      };

      wsRef.current = ws;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      console.error("[ProviderWS] Failed to create WebSocket:", errorMessage);
      setConnectionError(`Failed to connect: ${errorMessage}`);
    }
  }, [enabled, queryClient]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, "Client disconnect");
      wsRef.current = null;
    }

    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  const requestRefresh = useCallback(() => {
    sendMessage({ type: "refresh" });
  }, [sendMessage]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send("ping");
      }
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [isConnected]);

  // Connect/disconnect based on enabled state
  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      disconnect();
    }

    return () => disconnect();
  }, [enabled, connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    connectionError,
    sendMessage,
    requestRefresh,
    connect,
    disconnect,
  };
}

// =============================================================================
// Combined Hook: useProviderSystem
// =============================================================================

export function useProviderSystem(options?: { enableWebSocket?: boolean }) {
  const { enableWebSocket = true } = options || {};

  const config = useProviderConfig();
  const ws = useProviderWebSocket(enableWebSocket);

  return {
    ...config,
    isWebSocketConnected: ws.isConnected,
    webSocketError: ws.connectionError,
    lastWebSocketMessage: ws.lastMessage,
    requestWebSocketRefresh: ws.requestRefresh,
  };
}

export default useProviderConfig;
