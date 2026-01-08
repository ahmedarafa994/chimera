/**
 * Unified API Client
 *
 * Central API client that integrates all components:
 * - Configuration management
 * - Authentication & security
 * - Retry logic with exponential backoff
 * - Circuit breaker pattern
 * - Request deduplication
 * - Caching
 * - Monitoring & observability
 *
 * @module lib/api/core/client
 */

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from 'axios';
import { z } from 'zod';

import { configManager, AIProvider, ENDPOINTS } from './config';
import {
  APIError,
  NetworkError,
  TimeoutError,
  AuthenticationError,
  ValidationError,
  CircuitBreakerOpenError as CircuitBreakerError,
  mapBackendError,
  mapUnknownToError,
} from '../../errors';
import { withRetryResult as withRetry, type RetryConfig, DEFAULT_RETRY_CONFIG, type RetryResult } from '../../resilience/retry';
import {
  CircuitBreaker,
  circuitBreakerRegistry,
  type CircuitBreakerConfig,
  CircuitState,
} from '../../resilience/circuit-breaker';
import { requestDeduplicator, RequestBatcher, type BatchConfig } from './request-deduplication';
import {
  metricsCollector,
  trackRequest,
  apiLogger,
  generateRequestId,
} from './monitoring';
import { authHeaderBuilder, tokenManager, apiKeyManager } from './auth';

// ============================================================================
// Types
// ============================================================================

export interface RequestOptions extends AxiosRequestConfig {
  /** Skip retry logic */
  skipRetry?: boolean;
  /** Skip circuit breaker */
  skipCircuitBreaker?: boolean;
  /** Skip request deduplication */
  skipDeduplication?: boolean;
  /** Skip caching */
  skipCache?: boolean;
  /** Cache TTL in milliseconds */
  cacheTTL?: number;
  /** Provider for direct API calls */
  provider?: AIProvider;
  /** Require authentication */
  requireAuth?: boolean;
  /** Sign the request */
  signRequest?: boolean;
  /** Custom retry configuration */
  retryConfig?: Partial<RetryConfig>;
  /** Circuit breaker key override */
  circuitBreakerKey?: string;
  /** Request priority (for batching) */
  priority?: 'high' | 'normal' | 'low';
  /** Request tags for monitoring */
  tags?: Record<string, string>;
  /** Zod schema for request data validation */
  requestSchema?: z.ZodSchema<any>;
  /** Zod schema for response data validation */
  responseSchema?: z.ZodSchema<any>;
}

export interface APIClientConfig {
  /** Enable request deduplication */
  enableDeduplication: boolean;
  /** Enable circuit breaker */
  enableCircuitBreaker: boolean;
  /** Enable retry logic */
  enableRetry: boolean;
  /** Enable caching */
  enableCaching: boolean;
  /** Enable monitoring */
  enableMonitoring: boolean;
  /** Default retry configuration */
  defaultRetryConfig: RetryConfig;
  /** Default circuit breaker configuration */
  defaultCircuitBreakerConfig: CircuitBreakerConfig;
}

// ============================================================================
// Default Configuration
// ============================================================================

/**
 * Get default client config based on API mode
 *
 * In PROXY mode:
 * - Circuit breaker is DISABLED (backend handles this)
 * - Retries are MINIMAL (network errors only)
 * - This prevents "double-wrapping" and thundering herd problems
 *
 * In DIRECT mode:
 * - Full resilience enabled (frontend handles provider failures)
 */
function getDefaultClientConfig(): APIClientConfig {
  const appConfig = configManager.getConfig();
  const resilience = appConfig.resilience;

  return {
    enableDeduplication: true,
    enableCircuitBreaker: resilience.enableCircuitBreaker,
    enableRetry: resilience.enableRetry,
    enableCaching: true,
    enableMonitoring: true,
    defaultRetryConfig: {
      ...DEFAULT_RETRY_CONFIG,
      maxRetries: resilience.maxRetries,
    },
    defaultCircuitBreakerConfig: {
      name: 'default',
      failureThreshold: appConfig.circuitBreaker.failureThreshold,
      resetTimeout: appConfig.circuitBreaker.resetTimeout,
      successThreshold: 3,
      halfOpenRequests: appConfig.circuitBreaker.halfOpenRequests,
      failureWindow: 60000,
    },
  };
}

// Legacy constant for backward compatibility
const DEFAULT_CLIENT_CONFIG: APIClientConfig = getDefaultClientConfig();

// ============================================================================
// Simple In-Memory Cache
// ============================================================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class SimpleCache {
  private cache: Map<string, CacheEntry<unknown>> = new Map();
  private maxSize: number = 100;

  set<T>(key: string, data: T, ttl: number): void {
    // Enforce max size with LRU eviction
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check expiration
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data as T;
  }

  has(key: string): boolean {
    return this.get(key) !== null;
  }

  delete(key: string): void {
    this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  invalidatePattern(pattern: RegExp): void {
    const keysToDelete: string[] = [];
    this.cache.forEach((_, key) => {
      if (pattern.test(key)) {
        keysToDelete.push(key);
      }
    });
    keysToDelete.forEach(key => this.cache.delete(key));
  }
}

// ============================================================================
// API Client Class
// ============================================================================

interface BatchRequest {
  config: RequestOptions;
}

const ENDPOINT_TIMEOUTS: Record<string, number> = {
  [ENDPOINTS.JAILBREAK_GENERATE]: 60000,
  [ENDPOINTS.JAILBREAK_RUN_EXECUTE]: 90000,
  [ENDPOINTS.JAILBREAK_BATCH]: 120000,
  [ENDPOINTS.EXECUTE]: 45000,
  [ENDPOINTS.TRANSFORM]: 30000,
};

export class APIClient {
  private axiosInstance: AxiosInstance;
  private config: APIClientConfig;
  private cache: SimpleCache;
  private batcher: RequestBatcher<BatchRequest, { success: boolean; data?: unknown; error?: unknown }>;
  private configUnsubscribe: (() => void) | null = null;

  constructor(config: Partial<APIClientConfig> = {}) {
    // Get fresh config based on current API mode
    const defaultConfig = getDefaultClientConfig();
    this.config = { ...defaultConfig, ...config };

    // Subscribe to config changes to update resilience settings
    this.configUnsubscribe = configManager.subscribe((appConfig) => {
      const resilience = appConfig.resilience;
      this.config = {
        ...this.config,
        enableCircuitBreaker: resilience.enableCircuitBreaker,
        enableRetry: resilience.enableRetry,
        defaultRetryConfig: {
          ...this.config.defaultRetryConfig,
          maxRetries: resilience.maxRetries,
        },
      };

      // Recreate axios instance if base URL changed
      this.axiosInstance = this.createAxiosInstance();

      apiLogger.info('API client config updated', {
        mode: appConfig.mode,
        enableCircuitBreaker: this.config.enableCircuitBreaker,
        enableRetry: this.config.enableRetry,
      });
    });
    this.cache = new SimpleCache();

    const batchConfig: BatchConfig<BatchRequest, { success: boolean; data?: unknown; error?: unknown }> = {
      maxBatchSize: 10,
      maxWaitMs: 50,
      batchFn: async (requests: BatchRequest[]) => {
        // Process batched requests
        return Promise.all(
          requests.map(async (req) => {
            try {
              const response = await this.executeRequest(req.config);
              return { success: true, data: response.data };
            } catch (error) {
              return { success: false, error };
            }
          })
        );
      },
    };
    this.batcher = new RequestBatcher(batchConfig);

    // Create axios instance
    this.axiosInstance = this.createAxiosInstance();
  }

  // ============================================================================
  // Public Methods
  // ============================================================================

  /**
   * Make a GET request
   */
  async get<T = unknown>(url: string, options: RequestOptions = {}): Promise<T> {
    return this.request<T>({ ...options, method: 'GET', url });
  }

  /**
   * Make a POST request
   */
  async post<T = unknown>(
    url: string,
    data?: unknown,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>({ ...options, method: 'POST', url, data });
  }

  /**
   * Make a PUT request
   */
  async put<T = unknown>(
    url: string,
    data?: unknown,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>({ ...options, method: 'PUT', url, data });
  }

  /**
   * Make a PATCH request
   */
  async patch<T = unknown>(
    url: string,
    data?: unknown,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>({ ...options, method: 'PATCH', url, data });
  }

  /**
   * Make a DELETE request
   */
  async delete<T = unknown>(url: string, options: RequestOptions = {}): Promise<T> {
    return this.request<T>({ ...options, method: 'DELETE', url });
  }

  /**
   * Make a request with full options
   */
  async request<T = unknown>(options: RequestOptions): Promise<T> {
    const requestId = generateRequestId();
    const method = options.method || 'GET';
    const url = options.url || '';

    // Pick up default timeout for endpoint if not provided
    if (!options.timeout) {
      options.timeout = ENDPOINT_TIMEOUTS[url] || configManager.getTimeout();
    }

    apiLogger.debug(`Starting request`, { requestId, method, url });

    // Check cache for GET requests
    if (
      method === 'GET' &&
      this.config.enableCaching &&
      !options.skipCache
    ) {
      const cacheKey = this.getCacheKey(options);
      const cached = this.cache.get<T>(cacheKey);
      if (cached !== null) {
        apiLogger.debug(`Cache hit`, { requestId, cacheKey });
        return cached;
      }
    }

    // Deduplicate concurrent identical requests
    if (
      method === 'GET' &&
      this.config.enableDeduplication &&
      !options.skipDeduplication
    ) {
      const dedupeKey = this.getDedupeKey(options);
      return requestDeduplicator.execute(dedupeKey, () =>
        this.executeWithResilience<T>(options, requestId)
      );
    }

    // Validate request data if schema is provided
    if (options.requestSchema && options.data) {
      try {
        options.data = options.requestSchema.parse(options.data);
      } catch (err) {
        if (err instanceof z.ZodError) {
          const message = err.issues.map((e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`).join(', ');
          throw new ValidationError(`Request validation failed: ${message}`, { details: err.format() });
        }
        throw err;
      }
    }

    const result = await this.executeWithResilience<T>(options, requestId);

    // Validate response data if schema is provided
    if (options.responseSchema && result) {
      try {
        // We use safeParse for responses to avoid crashing on backend drift,
        // but we log warnings if validation fails.
        const validation = options.responseSchema.safeParse(result);
        if (!validation.success) {
          apiLogger.warn(`Response validation failed`, {
            requestId,
            url,
            error: validation.error.message
          });
          // In development we might want to be stricter, but for production
          // we're better off returning the data.
          if (process.env.NODE_ENV === 'development') {
            // throw new ValidationError(`Response validation failed: ${validation.error.message}`);
          }
        } else {
          return validation.data;
        }
      } catch (err) {
        apiLogger.error(`Error during response validation`, { requestId, error: err });
      }
    }

    return result;
  }

  /**
   * Invalidate cache entries matching a pattern
   */
  invalidateCache(pattern: RegExp): void {
    this.cache.invalidatePattern(pattern);
  }

  /**
   * Clear all cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get the underlying axios instance
   */
  getAxiosInstance(): AxiosInstance {
    return this.axiosInstance;
  }

  /**
   * Update client configuration
   */
  updateConfig(config: Partial<APIClientConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Destroy client and cleanup subscriptions
   */
  destroy(): void {
    if (this.configUnsubscribe) {
      this.configUnsubscribe();
      this.configUnsubscribe = null;
    }
    this.cache.clear();
  }

  /**
   * Check if circuit breaker is currently enabled
   */
  isCircuitBreakerEnabled(): boolean {
    return this.config.enableCircuitBreaker;
  }

  /**
   * Check if operating in proxy mode
   */
  isProxyMode(): boolean {
    return configManager.isProxyMode();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private createAxiosInstance(): AxiosInstance {
    const appConfig = configManager.getConfig();

    const instance = axios.create({
      baseURL: configManager.getActiveBaseUrl(),
      timeout: appConfig.request.defaultTimeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    instance.interceptors.request.use(
      async (config: InternalAxiosRequestConfig) => {
        // Add request ID
        const requestId = generateRequestId();
        config.headers.set('X-Request-ID', requestId);

        // Add auth headers
        const authHeaders = await authHeaderBuilder.buildHeaders({
          method: config.method || 'GET',
          url: config.url || '',
          body: config.data,
          requireAuth: false,
        });

        Object.entries(authHeaders).forEach(([key, value]) => {
          if (value) {
            config.headers.set(key, value);
          }
        });

        // Add custom headers from config
        const customHeaders = configManager.getApiHeaders();
        Object.entries(customHeaders).forEach(([key, value]) => {
          if (value) {
            config.headers.set(key, value);
          }
        });

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    instance.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => Promise.reject(mapBackendError(error))
    );

    return instance;
  }

  private async executeWithResilience<T>(
    options: RequestOptions,
    requestId: string
  ): Promise<T> {
    const method = options.method || 'GET';
    const url = options.url || '';

    // Start tracking
    const tracker = this.config.enableMonitoring
      ? trackRequest(method, url)
      : null;

    try {
      let result: T;

      // Apply circuit breaker
      if (this.config.enableCircuitBreaker && !options.skipCircuitBreaker) {
        const circuitKey = options.circuitBreakerKey || this.getCircuitBreakerKey(options);
        const circuitBreaker = circuitBreakerRegistry.getCircuit(
          circuitKey,
          this.config.defaultCircuitBreakerConfig
        );

        if (!circuitBreaker.canRequest()) {
          throw new CircuitBreakerError(
            circuitKey,
            circuitBreaker.getRetryAfter()
          );
        }

        try {
          result = await this.executeWithRetry<T>(options, requestId);
          circuitBreaker.recordSuccess();
        } catch (error) {
          circuitBreaker.recordFailure();
          throw error;
        }
      } else {
        result = await this.executeWithRetry<T>(options, requestId);
      }

      // Cache successful GET responses
      if (
        method === 'GET' &&
        this.config.enableCaching &&
        !options.skipCache
      ) {
        const cacheKey = this.getCacheKey(options);
        const ttl = options.cacheTTL || configManager.getConfig().cache.defaultTTL;
        this.cache.set(cacheKey, result, ttl);
      }

      // Record success
      if (tracker) {
        tracker.complete({
          success: true,
          status: 200,
          endpoint: this.extractEndpoint(url),
          provider: options.provider,
        });
      }

      return result;
    } catch (error) {
      const apiError = mapUnknownToError(error);
      // Record failure
      if (tracker) {
        tracker.complete({
          success: false,
          error: apiError,
          endpoint: this.extractEndpoint(url),
          provider: options.provider,
        });
      }

      throw apiError;
    }
  }

  private async executeWithRetry<T>(
    options: RequestOptions,
    requestId: string
  ): Promise<T> {
    if (this.config.enableRetry && !options.skipRetry) {
      const retryConfig: Partial<RetryConfig> = {
        ...options.retryConfig,
      };

      const result: RetryResult<AxiosResponse<T>> = await withRetry(
        () => this.executeRequest<T>(options),
        retryConfig
      );

      if (result.success && result.data) {
        return result.data.data;
      }

      throw mapUnknownToError(result.error);
    }

    const response = await this.executeRequest<T>(options);
    return response.data;
  }

  private async executeRequest<T>(
    options: RequestOptions
  ): Promise<AxiosResponse<T>> {
    const config: AxiosRequestConfig = {
      method: options.method,
      url: options.url,
      data: options.data,
      params: options.params,
      headers: options.headers,
      timeout: options.timeout,
      signal: options.signal,
    };

    // Handle provider-specific requests
    if (options.provider) {
      return this.executeProviderRequest<T>(options.provider, config);
    }

    return this.axiosInstance.request<T>(config);
  }

  private async executeProviderRequest<T>(
    provider: AIProvider,
    config: AxiosRequestConfig
  ): Promise<AxiosResponse<T>> {
    const providerConfig = configManager.getProviderConfig(provider);
    if (!providerConfig) {
      throw new Error(`Provider ${provider} is not configured`);
    }

    // Get provider-specific headers
    const providerHeaders = authHeaderBuilder.getProviderHeaders(provider);

    const providerAxios = axios.create({
      baseURL: providerConfig.apiUrl,
      timeout: configManager.getConfig().request.defaultTimeout,
      headers: {
        'Content-Type': 'application/json',
        ...providerHeaders,
      },
    });

    return providerAxios.request<T>(config);
  }

  private getCacheKey(options: RequestOptions): string {
    const parts = [
      options.method || 'GET',
      options.url || '',
      JSON.stringify(options.params || {}),
    ];
    return parts.join(':');
  }

  private getDedupeKey(options: RequestOptions): string {
    return this.getCacheKey(options);
  }

  private getCircuitBreakerKey(options: RequestOptions): string {
    if (options.provider) {
      return `provider:${options.provider}`;
    }
    const endpoint = this.extractEndpoint(options.url || '');
    return `endpoint:${endpoint}`;
  }

  private extractEndpoint(url: string): string {
    // Extract the main endpoint path
    const match = url.match(/^\/api\/v\d+\/([^/?]+)/);
    return match ? match[1] : url.split('?')[0];
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

export const apiClient = new APIClient();

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Make a GET request
 */
export function get<T = unknown>(url: string, options?: RequestOptions): Promise<T> {
  return apiClient.get<T>(url, options);
}

/**
 * Make a POST request
 */
export function post<T = unknown>(
  url: string,
  data?: unknown,
  options?: RequestOptions
): Promise<T> {
  return apiClient.post<T>(url, data, options);
}

/**
 * Make a PUT request
 */
export function put<T = unknown>(
  url: string,
  data?: unknown,
  options?: RequestOptions
): Promise<T> {
  return apiClient.put<T>(url, data, options);
}

/**
 * Make a PATCH request
 */
export function patch<T = unknown>(
  url: string,
  data?: unknown,
  options?: RequestOptions
): Promise<T> {
  return apiClient.patch<T>(url, data, options);
}

/**
 * Make a DELETE request
 */
export function del<T = unknown>(url: string, options?: RequestOptions): Promise<T> {
  return apiClient.delete<T>(url, options);
}

// ============================================================================
// Typed API Methods
// ============================================================================

/**
 * Health check endpoint
 */
export async function healthCheck(): Promise<{ status: string; timestamp: string }> {
  return apiClient.get(ENDPOINTS.HEALTH, { skipCache: true });
}

/**
 * Get available models
 */
export async function getModels(): Promise<unknown[]> {
  return apiClient.get(ENDPOINTS.MODELS, { cacheTTL: 60000 });
}

/**
 * Get available providers
 */
export async function getProviders(): Promise<unknown[]> {
  return apiClient.get(ENDPOINTS.PROVIDERS, { cacheTTL: 60000 });
}

/**
 * Generate jailbreak prompt
 */
export async function generateJailbreak(params: {
  prompt: string;
  technique?: string;
  model?: string;
}): Promise<unknown> {
  return apiClient.post(ENDPOINTS.JAILBREAK_GENERATE, params);
}

/**
 * Chat completion
 */
export async function chatCompletion(params: {
  messages: Array<{ role: string; content: string }>;
  model?: string;
  temperature?: number;
}): Promise<unknown> {
  return apiClient.post(ENDPOINTS.CHAT, params);
}

/**
 * Get techniques
 */
export async function getTechniques(): Promise<unknown[]> {
  return apiClient.get(ENDPOINTS.TECHNIQUES, { cacheTTL: 300000 });
}

/**
 * Get metrics
 */
export async function getMetrics(): Promise<unknown> {
  return apiClient.get(ENDPOINTS.METRICS, { skipCache: true });
}
