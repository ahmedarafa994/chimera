"use client";

/**
 * Provider Sync Provider Wrapper
 * 
 * Wraps the ProviderSyncProvider from contexts for use in the app layout.
 * This enables real-time synchronization of provider/model configurations
 * across the application.
 */

import { ProviderSyncProvider as SyncProvider } from "@/contexts/ProviderSyncContext";
import { ProviderSyncConfig } from "@/types/provider-sync";

interface ProviderSyncProviderProps {
  children: React.ReactNode;
  config?: Partial<ProviderSyncConfig>;
}

/**
 * Get the API base URL from environment or default
 */
function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    // Client-side: use environment variable or default to direct backend URL
    // NEXT_PUBLIC_CHIMERA_API_URL should include /api/v1 (e.g., http://localhost:8001/api/v1)
    return process.env.NEXT_PUBLIC_CHIMERA_API_URL || "http://localhost:8001/api/v1";
  }
  // Server-side: return empty (will be set on client)
  return "";
}

/**
 * Provider Sync Provider with default configuration
 * 
 * Provides real-time synchronization of AI provider configurations
 * with automatic WebSocket connection and polling fallback.
 */
export function ProviderSyncProvider({ children, config }: ProviderSyncProviderProps) {
  const apiBaseUrl = getApiBaseUrl();
  
  // Don't render on server side
  if (!apiBaseUrl) {
    return <>{children}</>;
  }

  // Check if apiBaseUrl already contains /api/v1
  const hasApiV1 = apiBaseUrl.includes('/api/v1');
  const syncPath = hasApiV1 ? '/provider-sync' : '/api/v1/provider-sync';
  
  const defaultConfig: Partial<ProviderSyncConfig> = {
    apiBaseUrl: `${apiBaseUrl}${syncPath}`,
    wsUrl: `${apiBaseUrl.replace("http", "ws")}${syncPath}/ws`,
    enableWebSocket: true,
    pollingInterval: 30000,
    maxReconnectAttempts: 5,
    reconnectBaseDelay: 1000,
    reconnectMaxDelay: 30000,
    heartbeatInterval: 25000,
    syncTimeout: 10000,
    includeDeprecated: false,
    enableCache: true,
    cacheTtl: 300000,
    ...config,
  };

  return (
    <SyncProvider config={defaultConfig}>
      {children}
    </SyncProvider>
  );
}

export default ProviderSyncProvider;