/**
 * API Migration Compatibility Layer
 *
 * Provides backward compatibility with existing API implementations
 * while allowing gradual migration to the new architecture.
 *
 * @module lib/api/migration/compat
 */

import { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { apiClient, APIClient, RequestOptions } from '../core/client';
import { configManager, ENDPOINTS } from '../core/config';
import { APIError, mapUnknownToError as mapBackendError } from '../../errors';

// ============================================================================
// Legacy API Enhanced Compatibility
// ============================================================================

/**
 * Legacy API interface compatible with api-enhanced.ts
 */
export interface LegacyAPI {
  get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  post<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T>;
  put<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T>;
  patch<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T>;
  delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  getAxiosInstance(): AxiosInstance;
}

/**
 * Create a legacy-compatible API instance
 *
 * @example
 * ```typescript
 * // Replace existing import
 * // import { api } from '@/lib/api-enhanced';
 * import { createLegacyAPI } from '@/lib/api/migration/compat';
 * const api = createLegacyAPI();
 *
 * // Existing code continues to work
 * const data = await api.get('/api/v1/providers');
 * ```
 */
export function createLegacyAPI(): LegacyAPI {
  return {
    async get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
      return apiClient.get<T>(url, convertAxiosConfig(config));
    },

    async post<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
      return apiClient.post<T>(url, data, convertAxiosConfig(config));
    },

    async put<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
      return apiClient.put<T>(url, data, convertAxiosConfig(config));
    },

    async patch<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
      return apiClient.patch<T>(url, data, convertAxiosConfig(config));
    },

    async delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T> {
      return apiClient.delete<T>(url, convertAxiosConfig(config));
    },

    getAxiosInstance(): AxiosInstance {
      return apiClient.getAxiosInstance();
    },
  };
}

/**
 * Convert AxiosRequestConfig to RequestOptions
 */
function convertAxiosConfig(config?: AxiosRequestConfig): RequestOptions {
  if (!config) return {};

  return {
    headers: config.headers as Record<string, string>,
    params: config.params,
    timeout: config.timeout,
    signal: config.signal,
  };
}

// ============================================================================
// Legacy ChimeraAPI Compatibility
// ============================================================================

/**
 * Legacy ChimeraAPI interface compatible with chimeraApi.ts
 */
export interface LegacyChimeraAPI {
  get<T = unknown>(endpoint: string): Promise<T>;
  post<T = unknown>(endpoint: string, data?: unknown): Promise<T>;
  put<T = unknown>(endpoint: string, data?: unknown): Promise<T>;
  delete<T = unknown>(endpoint: string): Promise<T>;
  setBaseURL(url: string): void;
  setAuthToken(token: string): void;
}

/**
 * Create a legacy ChimeraAPI-compatible instance
 *
 * @example
 * ```typescript
 * // Replace existing import
 * // import { chimeraApi } from '@/lib/chimeraApi';
 * import { createLegacyChimeraAPI } from '@/lib/api/migration/compat';
 * const chimeraApi = createLegacyChimeraAPI();
 *
 * // Existing code continues to work
 * const providers = await chimeraApi.get('/providers');
 * ```
 */
export function createLegacyChimeraAPI(): LegacyChimeraAPI {
  let baseURL = '/api/v1';

  return {
    async get<T = unknown>(endpoint: string): Promise<T> {
      const url = normalizeEndpoint(baseURL, endpoint);
      return apiClient.get<T>(url);
    },

    async post<T = unknown>(endpoint: string, data?: unknown): Promise<T> {
      const url = normalizeEndpoint(baseURL, endpoint);
      return apiClient.post<T>(url, data);
    },

    async put<T = unknown>(endpoint: string, data?: unknown): Promise<T> {
      const url = normalizeEndpoint(baseURL, endpoint);
      return apiClient.put<T>(url, data);
    },

    async delete<T = unknown>(endpoint: string): Promise<T> {
      const url = normalizeEndpoint(baseURL, endpoint);
      return apiClient.delete<T>(url);
    },

    setBaseURL(url: string): void {
      baseURL = url;
    },

    setAuthToken(token: string): void {
      // Use the new auth system
      import('../core/auth').then(({ setAuthToken }) => {
        setAuthToken({
          accessToken: token,
          tokenType: 'Bearer',
        });
      });
    },
  };
}

/**
 * Normalize endpoint path
 */
function normalizeEndpoint(baseURL: string, endpoint: string): string {
  // Remove leading slash from endpoint if base has trailing slash
  const base = baseURL.endsWith('/') ? baseURL.slice(0, -1) : baseURL;
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${base}${path}`;
}

// ============================================================================
// Service Layer Compatibility
// ============================================================================

/**
 * Base service class for backward compatibility with existing services
 */
export abstract class LegacyBaseService {
  protected client: APIClient;
  protected basePath: string;

  constructor(basePath: string = '') {
    this.client = apiClient;
    this.basePath = basePath;
  }

  protected async get<T>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.client.get<T>(`${this.basePath}${endpoint}`, options);
  }

  protected async post<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.client.post<T>(`${this.basePath}${endpoint}`, data, options);
  }

  protected async put<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.client.put<T>(`${this.basePath}${endpoint}`, data, options);
  }

  protected async patch<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.client.patch<T>(`${this.basePath}${endpoint}`, data, options);
  }

  protected async delete<T>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.client.delete<T>(`${this.basePath}${endpoint}`, options);
  }
}

// ============================================================================
// Hook Compatibility
// ============================================================================

/**
 * Create a hook-compatible fetch function
 *
 * @example
 * ```typescript
 * // In a custom hook
 * import { createFetchFn } from '@/lib/api/migration/compat';
 *
 * const fetchProviders = createFetchFn<Provider[]>('/api/v1/providers');
 *
 * // Use in hook
 * const { data, error, loading } = useQuery(fetchProviders);
 * ```
 */
export function createFetchFn<T>(
  url: string,
  options?: RequestOptions
): () => Promise<T> {
  return () => apiClient.get<T>(url, options);
}

/**
 * Create a mutation function for hooks
 */
export function createMutationFn<TData, TVariables>(
  url: string,
  method: 'POST' | 'PUT' | 'PATCH' | 'DELETE' = 'POST',
  options?: RequestOptions
): (variables: TVariables) => Promise<TData> {
  return async (variables: TVariables) => {
    switch (method) {
      case 'POST':
        return apiClient.post<TData>(url, variables, options);
      case 'PUT':
        return apiClient.put<TData>(url, variables, options);
      case 'PATCH':
        return apiClient.patch<TData>(url, variables, options);
      case 'DELETE':
        return apiClient.delete<TData>(url, options);
    }
  };
}

// ============================================================================
// Error Compatibility
// ============================================================================

/**
 * Convert new APIError to legacy error format
 */
export function toLegacyError(error: unknown): {
  message: string;
  code?: string;
  status?: number;
  data?: unknown;
} {
  if (error instanceof APIError) {
    const apiErr = error as APIError;
    return {
      message: apiErr.message,
      code: apiErr.errorCode,
      status: apiErr.statusCode,
      data: apiErr.details,
    };
  }

  if (error instanceof Error) {
    return {
      message: error.message,
    };
  }

  return {
    message: String(error),
  };
}

/**
 * Check if error is a specific type (legacy compatibility)
 */
export function isErrorType(error: unknown, type: string): boolean {
  if (error instanceof APIError) {
    return (error as APIError).errorCode === type;
  }
  return false;
}

// ============================================================================
// Configuration Compatibility
// ============================================================================

/**
 * Legacy configuration interface
 */
export interface LegacyConfig {
  baseURL: string;
  timeout: number;
  headers: Record<string, string>;
}

/**
 * Get legacy-compatible configuration
 */
export function getLegacyConfig(): LegacyConfig {
  const config = configManager.getConfig();
  return {
    baseURL: configManager.getActiveBaseUrl(),
    timeout: config.request.defaultTimeout,
    headers: configManager.getApiHeaders() as Record<string, string>,
  };
}

/**
 * Set configuration using legacy format
 */
export function setLegacyConfig(config: Partial<LegacyConfig>): void {
  if (config.timeout) {
    configManager.updateConfig({
      request: { ...configManager.getConfig().request, defaultTimeout: config.timeout },
    });
  }
}

// ============================================================================
// Endpoint Mapping
// ============================================================================

/**
 * Map legacy endpoint names to new endpoints
 */
export const LEGACY_ENDPOINT_MAP: Record<string, string> = {
  // Health
  '/health': ENDPOINTS.HEALTH,
  '/api/health': ENDPOINTS.HEALTH,

  // Providers
  '/providers': ENDPOINTS.PROVIDERS,
  '/api/providers': ENDPOINTS.PROVIDERS,

  // Models
  '/models': ENDPOINTS.MODELS,
  '/api/models': ENDPOINTS.MODELS,

  // Chat
  '/chat': ENDPOINTS.CHAT_COMPLETIONS,
  '/api/chat': ENDPOINTS.CHAT_COMPLETIONS,
  '/chat/completions': ENDPOINTS.CHAT_COMPLETIONS,

  // Jailbreak
  '/jailbreak': ENDPOINTS.JAILBREAK_GENERATE,
  '/jailbreak/generate': ENDPOINTS.JAILBREAK_GENERATE,
  '/api/jailbreak': ENDPOINTS.JAILBREAK_GENERATE,

  // Techniques
  '/techniques': ENDPOINTS.TECHNIQUES,
  '/api/techniques': ENDPOINTS.TECHNIQUES,

  // Metrics
  '/metrics': ENDPOINTS.METRICS,
  '/api/metrics': ENDPOINTS.METRICS,
};

/**
 * Resolve legacy endpoint to new endpoint
 */
export function resolveLegacyEndpoint(endpoint: string): string {
  return LEGACY_ENDPOINT_MAP[endpoint] || endpoint;
}

// ============================================================================
// Default Exports for Drop-in Replacement
// ============================================================================

/**
 * Drop-in replacement for api-enhanced.ts
 */
export const api = createLegacyAPI();

/**
 * Drop-in replacement for chimeraApi.ts
 */
export const chimeraApi = createLegacyChimeraAPI();
