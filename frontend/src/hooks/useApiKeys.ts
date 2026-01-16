"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import type { ApiKeyFormData, ApiKeyItem, ProviderId, ProviderKeySummary } from "../api-keys";

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const API_KEYS_ENDPOINT = `${API_BASE_URL}/api/v1/api-keys`;

// =============================================================================
// Types
// =============================================================================

interface ApiKeyListResponse {
  keys: ApiKeyItem[];
  total: number;
  by_provider: Record<string, number>;
}

interface ProvidersSummaryResponse {
  providers: ProviderKeySummary[];
  total_keys: number;
  configured_providers: number;
}

interface ApiKeyTestResult {
  success: boolean;
  provider_id: string;
  latency_ms?: number;
  error?: string;
  models_available?: string[];
  tested_at: string;
}

interface UseApiKeysOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

interface UseApiKeysResult {
  keys: ApiKeyItem[];
  providers: ProviderKeySummary[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
  createKey: (data: ApiKeyFormData) => Promise<void>;
  updateKey: (keyId: string, data: Partial<ApiKeyFormData>) => Promise<void>;
  deleteKey: (keyId: string) => Promise<void>;
  testKey: (keyId: string) => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  testNewKey: (data: ApiKeyFormData) => Promise<{ success: boolean; message: string; latency_ms?: number }>;
  activateKey: (keyId: string) => Promise<void>;
  deactivateKey: (keyId: string) => Promise<void>;
  revokeKey: (keyId: string) => Promise<void>;
}

// =============================================================================
// API Helpers
// =============================================================================

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  // Add API key header if available
  const apiKey = typeof window !== "undefined"
    ? localStorage.getItem("chimera_api_key") || ""
    : "";

  if (apiKey) {
    (headers as Record<string, string>)["X-API-Key"] = apiKey;
  }

  const response = await fetch(`${API_KEYS_ENDPOINT}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API request failed: ${response.status}`);
  }

  return response.json();
}

// =============================================================================
// useApiKeys Hook
// =============================================================================

export function useApiKeys(options: UseApiKeysOptions = {}): UseApiKeysResult {
  const { autoRefresh = false, refreshInterval = 30000 } = options;

  const [keys, setKeys] = useState<ApiKeyItem[]>([]);
  const [providers, setProviders] = useState<ProviderKeySummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch all data
  const refresh = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Fetch keys and provider summaries in parallel
      const [keysResponse, providersResponse] = await Promise.all([
        apiRequest<ApiKeyListResponse>(""),
        apiRequest<ProvidersSummaryResponse>("/providers"),
      ]);

      setKeys(keysResponse.keys);
      setProviders(providersResponse.providers);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to fetch API keys"));
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Create a new key
  const createKey = useCallback(async (data: ApiKeyFormData) => {
    await apiRequest<ApiKeyItem>("", {
      method: "POST",
      body: JSON.stringify(data),
    });
    await refresh();
  }, [refresh]);

  // Update an existing key
  const updateKey = useCallback(async (keyId: string, data: Partial<ApiKeyFormData>) => {
    await apiRequest<ApiKeyItem>(`/${keyId}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
    await refresh();
  }, [refresh]);

  // Delete a key
  const deleteKey = useCallback(async (keyId: string) => {
    await apiRequest<{ success: boolean; message: string }>(`/${keyId}`, {
      method: "DELETE",
    });
    await refresh();
  }, [refresh]);

  // Test an existing key
  const testKey = useCallback(async (keyId: string) => {
    const result = await apiRequest<ApiKeyTestResult>(`/${keyId}/test`, {
      method: "POST",
    });
    return {
      success: result.success,
      message: result.success
        ? `Connection successful${result.models_available?.length ? ` (${result.models_available.length} models available)` : ""}`
        : result.error || "Test failed",
      latency_ms: result.latency_ms,
    };
  }, []);

  // Test a new key (before saving)
  const testNewKey = useCallback(async (data: ApiKeyFormData) => {
    const params = new URLSearchParams({
      provider_id: data.provider_id,
      api_key: data.api_key,
    });
    const result = await apiRequest<ApiKeyTestResult>(`/test?${params.toString()}`, {
      method: "POST",
    });
    return {
      success: result.success,
      message: result.success
        ? `Connection successful${result.models_available?.length ? ` (${result.models_available.length} models available)` : ""}`
        : result.error || "Test failed",
      latency_ms: result.latency_ms,
    };
  }, []);

  // Activate a key
  const activateKey = useCallback(async (keyId: string) => {
    await apiRequest<ApiKeyItem>(`/${keyId}/activate`, {
      method: "POST",
    });
    await refresh();
  }, [refresh]);

  // Deactivate a key
  const deactivateKey = useCallback(async (keyId: string) => {
    await apiRequest<ApiKeyItem>(`/${keyId}/deactivate`, {
      method: "POST",
    });
    await refresh();
  }, [refresh]);

  // Revoke a key
  const revokeKey = useCallback(async (keyId: string) => {
    await apiRequest<ApiKeyItem>(`/${keyId}/revoke`, {
      method: "POST",
    });
    await refresh();
  }, [refresh]);

  // Initial fetch
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      refreshTimerRef.current = setInterval(refresh, refreshInterval);
      return () => {
        if (refreshTimerRef.current) {
          clearInterval(refreshTimerRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, refresh]);

  return {
    keys,
    providers,
    isLoading,
    error,
    refresh,
    createKey,
    updateKey,
    deleteKey,
    testKey,
    testNewKey,
    activateKey,
    deactivateKey,
    revokeKey,
  };
}

export default useApiKeys;
