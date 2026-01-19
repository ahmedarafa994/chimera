/**
 * API Client Functions for Unified Provider System
 *
 * Provides typed functions for interacting with the unified providers API endpoints.
 * These functions handle HTTP requests, error handling, and response parsing.
 *
 * @module api/unified-providers
 */

import { getApiHeaders as getBaseApiHeaders } from "../api-config";
import type {
  ProviderInfo,
  ModelInfo,
  ProvidersListResponse,
  ModelsListResponse,
  CurrentSelectionResponse,
  SaveSelectionRequest,
  RefreshCatalogResponse,
  CatalogStatsResponse,
  SelectionScope,
} from "../../types/unified-providers";

// =============================================================================
// Configuration
// =============================================================================

// Backend API endpoints at /api/v1/unified-providers
const API_BASE = "/api/v1/unified-providers";

/**
 * Get the full API URL for a given path
 * Uses relative URLs in browser to go through the Next.js proxy, avoiding CORS issues
 * Falls back to absolute URL for server-side contexts
 */
function getApiUrl(path: string): string {
  // In browser context, use relative URL to leverage Next.js API proxy
  if (typeof window !== "undefined") {
    return `${API_BASE}${path}`;
  }
  // In server context (SSR), construct absolute URL
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3700";
  return `${baseUrl}${API_BASE}${path}`;
}

/**
 * Default fetch options with auth headers
 */
function getDefaultOptions(): RequestInit {
  const baseHeaders = getBaseApiHeaders();
  return {
    headers: {
      ...baseHeaders,
      Accept: "application/json",
    },
    credentials: "include" as RequestCredentials,
  };
}

// =============================================================================
// Error Handling
// =============================================================================

/**
 * Custom error class for API errors
 */
export class UnifiedProviderApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public statusText: string,
    public details?: unknown
  ) {
    super(message);
    this.name = "UnifiedProviderApiError";
  }
}

/**
 * Handle API response and throw appropriate errors
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let details: unknown;
    try {
      details = await response.json();
    } catch {
      details = await response.text().catch(() => null);
    }

    const message =
      typeof details === "object" && details !== null && "detail" in details
        ? String((details as { detail: unknown }).detail)
        : `API error: ${response.status} ${response.statusText}`;

    throw new UnifiedProviderApiError(
      message,
      response.status,
      response.statusText,
      details
    );
  }

  return response.json();
}

// =============================================================================
// Provider API Functions
// =============================================================================

/**
 * Fetch all available providers
 *
 * @param options.enabled_only - Only return enabled providers
 * @param options.available_only - Only return available providers (enabled + has API key)
 * @returns List of provider information
 *
 * @example
 * ```typescript
 * const providers = await getProviders({ enabled_only: true });
 * console.log(providers); // [{ provider_id: 'openai', display_name: 'OpenAI', ... }]
 * ```
 */
export async function getProviders(options?: {
  enabled_only?: boolean;
  available_only?: boolean;
}): Promise<ProviderInfo[]> {
  const params = new URLSearchParams();

  if (options?.enabled_only) {
    params.append("enabled_only", "true");
  }
  if (options?.available_only) {
    params.append("available_only", "true");
  }

  const queryString = params.toString();
  const url = getApiUrl(`/providers${queryString ? `?${queryString}` : ""}`);

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  const data = await handleResponse<ProvidersListResponse>(response);
  return data.providers;
}

/**
 * Fetch a specific provider by ID
 *
 * @param providerId - The provider identifier
 * @returns Provider information
 *
 * @example
 * ```typescript
 * const provider = await getProvider('openai');
 * console.log(provider.status); // 'healthy'
 * ```
 */
export async function getProvider(providerId: string): Promise<ProviderInfo> {
  const url = getApiUrl(`/providers/${encodeURIComponent(providerId)}`);

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  return handleResponse<ProviderInfo>(response);
}

/**
 * Fetch models for a specific provider
 *
 * @param providerId - The provider identifier
 * @param options.available_only - Only return available models
 * @returns List of model information
 *
 * @example
 * ```typescript
 * const models = await getProviderModels('openai');
 * console.log(models); // [{ model_id: 'gpt-4', name: 'GPT-4', ... }]
 * ```
 */
export async function getProviderModels(
  providerId: string,
  options?: { available_only?: boolean }
): Promise<ModelInfo[]> {
  const params = new URLSearchParams();

  if (options?.available_only) {
    params.append("available_only", "true");
  }

  const queryString = params.toString();
  const url = getApiUrl(
    `/providers/${encodeURIComponent(providerId)}/models${queryString ? `?${queryString}` : ""}`
  );

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  const data = await handleResponse<ModelsListResponse>(response);
  return data.models;
}

// =============================================================================
// Model API Functions
// =============================================================================

/**
 * Fetch all available models across all providers
 *
 * @param options.provider_id - Filter by provider
 * @param options.capability - Filter by capability
 * @param options.available_only - Only return available models
 * @returns List of model information
 *
 * @example
 * ```typescript
 * const visionModels = await getAllModels({ capability: 'vision' });
 * ```
 */
export async function getAllModels(options?: {
  provider_id?: string;
  capability?: string;
  available_only?: boolean;
}): Promise<ModelInfo[]> {
  const params = new URLSearchParams();

  if (options?.provider_id) {
    params.append("provider", options.provider_id);
  }
  if (options?.capability) {
    params.append("capability", options.capability);
  }
  if (options?.available_only) {
    params.append("available_only", "true");
  }

  const queryString = params.toString();
  const url = getApiUrl(`/models${queryString ? `?${queryString}` : ""}`);

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  const data = await handleResponse<{ models: ModelInfo[]; total: number }>(response);
  return data.models;
}

/**
 * Fetch a specific model by ID
 *
 * @param modelId - The model identifier
 * @returns Model information
 */
export async function getModel(modelId: string): Promise<ModelInfo> {
  const url = getApiUrl(`/models/${encodeURIComponent(modelId)}`);

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  return handleResponse<ModelInfo>(response);
}

// =============================================================================
// Selection API Functions
// =============================================================================

/**
 * Get the current provider/model selection
 *
 * @param sessionId - Optional session ID for session-scoped selection
 * @returns Current selection state
 *
 * @example
 * ```typescript
 * const selection = await getCurrentSelection();
 * console.log(selection); // { provider_id: 'openai', model_id: 'gpt-4', scope: 'SESSION' }
 * ```
 */
export async function getCurrentSelection(
  sessionId?: string
): Promise<CurrentSelectionResponse | null> {
  const params = new URLSearchParams();

  if (sessionId) {
    params.append("session_id", sessionId);
  }

  const queryString = params.toString();
  // Use /selection/current endpoint for getting current selection
  const url = getApiUrl(`/selection/current${queryString ? `?${queryString}` : ""}`);

  try {
    const response = await fetch(url, {
      ...getDefaultOptions(),
      method: "GET",
    });

    if (response.status === 404) {
      return null;
    }

    return handleResponse<CurrentSelectionResponse>(response);
  } catch (error) {
    if (error instanceof UnifiedProviderApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
}

/**
 * Save provider/model selection
 *
 * @param provider - Provider ID to select
 * @param model - Model ID to select
 * @param sessionId - Optional session ID for session-scoped selection
 * @returns The saved selection
 *
 * @example
 * ```typescript
 * await selectProviderModel('openai', 'gpt-4');
 * ```
 */
export async function selectProviderModel(
  provider: string,
  model: string,
  sessionId?: string
): Promise<CurrentSelectionResponse> {
  // Use /selection/sync endpoint for setting selection
  const url = getApiUrl("/selection/sync");

  const body = {
    session_id: sessionId || "default",
    provider: provider,
    model: model,
  };

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "POST",
    headers: {
      ...getDefaultOptions().headers,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  // Transform sync response to CurrentSelectionResponse format
  const syncResponse = await handleResponse<{
    success: boolean;
    session_id: string;
    provider: string;
    model: string;
    version: number;
  }>(response);

  return {
    provider_id: syncResponse.provider,
    model_id: syncResponse.model,
    scope: "SESSION",
    session_id: syncResponse.session_id,
    version: syncResponse.version,
  };
}

/**
 * Clear the current selection
 *
 * @param sessionId - Optional session ID for session-scoped selection
 * @param scope - Scope to clear (default: SESSION)
 */
export async function clearSelection(
  sessionId?: string,
  scope: SelectionScope = "SESSION"
): Promise<void> {
  // Use /selection/{session_id} DELETE endpoint
  const effectiveSessionId = sessionId || "default";
  const url = getApiUrl(`/selection/${effectiveSessionId}`);

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "DELETE",
  });

  if (!response.ok && response.status !== 404) {
    await handleResponse<void>(response);
  }
}

// =============================================================================
// Catalog Management API Functions
// =============================================================================

/**
 * Refresh the provider catalog (re-fetch all models from providers)
 *
 * @param providerIds - Optional list of specific providers to refresh
 * @returns Refresh result with stats
 *
 * @example
 * ```typescript
 * const result = await refreshProviderCatalog(['openai', 'anthropic']);
 * console.log(result.total_models); // 25
 * ```
 */
export async function refreshProviderCatalog(
  providerIds?: string[]
): Promise<RefreshCatalogResponse> {
  const url = getApiUrl("/catalog/refresh");

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "POST",
    body: JSON.stringify({
      provider_ids: providerIds,
    }),
  });

  return handleResponse<RefreshCatalogResponse>(response);
}

/**
 * Get catalog statistics
 *
 * @returns Catalog stats including provider/model counts
 */
export async function getCatalogStats(): Promise<CatalogStatsResponse> {
  const url = getApiUrl("/catalog/stats");

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  return handleResponse<CatalogStatsResponse>(response);
}

// =============================================================================
// Health Check API Functions
// =============================================================================

/**
 * Check health of a specific provider
 *
 * @param providerId - The provider identifier
 * @returns Provider health status
 */
export async function checkProviderHealth(
  providerId: string
): Promise<{ status: string; is_available: boolean; health_score?: number; message?: string }> {
  const url = getApiUrl(`/providers/${encodeURIComponent(providerId)}/health`);

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  return handleResponse<{
    status: string;
    is_available: boolean;
    health_score?: number;
    message?: string;
  }>(response);
}

/**
 * Check health of all providers
 *
 * @returns Map of provider ID to health status
 */
export async function checkAllProvidersHealth(): Promise<
  Record<string, { status: string; is_available: boolean; health_score?: number }>
> {
  const url = getApiUrl("/providers/health");

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "GET",
  });

  return handleResponse<
    Record<string, { status: string; is_available: boolean; health_score?: number }>
  >(response);
}

// =============================================================================
// Validation API Functions
// =============================================================================

/**
 * Validate a provider/model combination
 *
 * @param providerId - The provider identifier
 * @param modelId - The model identifier
 * @returns Validation result
 */
export async function validateSelection(
  providerId: string,
  modelId: string
): Promise<{ is_valid: boolean; errors?: string[]; warnings?: string[] }> {
  const url = getApiUrl("/models/validate");

  const response = await fetch(url, {
    ...getDefaultOptions(),
    method: "POST",
    body: JSON.stringify({
      provider_id: providerId,
      model_id: modelId,
    }),
  });

  return handleResponse<{ is_valid: boolean; errors?: string[]; warnings?: string[] }>(response);
}

// =============================================================================
// LocalStorage Helpers
// =============================================================================

const STORAGE_KEY = "chimera_unified_provider_selection";

/**
 * Selection stored in localStorage
 */
interface StoredSelection {
  provider: string;
  model: string;
  timestamp: number;
}

/**
 * Save selection to localStorage
 *
 * @param provider - Provider ID
 * @param model - Model ID
 */
export function saveSelectionToLocalStorage(provider: string, model: string): void {
  try {
    const data: StoredSelection = {
      provider,
      model,
      timestamp: Date.now(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (error) {
    console.warn("Failed to save selection to localStorage:", error);
  }
}

/**
 * Get selection from localStorage
 *
 * @returns Stored selection or null
 */
export function getSelectionFromLocalStorage(): StoredSelection | null {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return null;

    const parsed = JSON.parse(data) as StoredSelection;

    // Validate structure
    if (!parsed.provider || !parsed.model) {
      return null;
    }

    return parsed;
  } catch (error) {
    console.warn("Failed to get selection from localStorage:", error);
    return null;
  }
}

/**
 * Clear selection from localStorage
 */
export function clearSelectionFromLocalStorage(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.warn("Failed to clear selection from localStorage:", error);
  }
}

// =============================================================================
// WebSocket Connection Helpers
// =============================================================================

/**
 * Get WebSocket URL for real-time updates
 *
 * @param sessionId - Optional session ID
 * @returns WebSocket URL
 */
export function getWebSocketUrl(sessionId?: string): string {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  const wsProtocol = baseUrl.startsWith("https") ? "wss" : "ws";
  const wsBase = baseUrl.replace(/^https?/, wsProtocol);
  const effectiveSessionId = sessionId || "default";
  return `${wsBase}/api/v1/unified-providers/selection/subscribe/${encodeURIComponent(
    effectiveSessionId
  )}`;
}
