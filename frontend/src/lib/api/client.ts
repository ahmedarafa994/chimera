// @ts-nocheck
/**
 * Centralized API Client Configuration
 * Handles all HTTP communication between frontend and backend
 */

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios';
import { ApiError, ApiResponse } from './types';
import { authManager } from './auth-manager';
import { retryWithBackoff } from './retry-strategy';
import { apiCache } from './cache-manager';
import { logger } from './logger';

// ============================================================================
// Extended Types for Axios Metadata
// ============================================================================

/**
 * Extended request config with metadata
 */
interface RequestMetadata {
  startTime: number;
  requestId: string;
}

/**
 * Extended Axios config with custom properties
 */
interface ExtendedAxiosConfig extends AxiosRequestConfig {
  metadata?: RequestMetadata;
  disableCache?: boolean;
  cacheTTL?: number;
  _retry?: boolean;
}

/**
 * Error response data structure
 */
interface ErrorResponseData {
  message?: string;
  detail?: string;
  code?: string;
  details?: Record<string, unknown>;
}

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
const API_VERSION = '/api/v1';
const REQUEST_TIMEOUT = 30000;
const MAX_RETRIES = 3;

// ============================================================================
// Request ID Generation
// ============================================================================

function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substring(7)}`;
}

// ============================================================================
// API Client
// ============================================================================

class ApiClient {
  private client: AxiosInstance;
  private requestQueue: Map<string, AbortController>;

  constructor() {
    this.requestQueue = new Map();
    this.client = this.createClient();
  }

  private createClient(): AxiosInstance {
    const client = axios.create({
      baseURL: `${API_BASE_URL}${API_VERSION}`,
      timeout: REQUEST_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      withCredentials: true,
    });

    this.setupInterceptors(client);
    return client;
  }

  private setupInterceptors(client: AxiosInstance): void {
    // Request Interceptor
    client.interceptors.request.use(
      async (config: InternalAxiosRequestConfig) => {
        const startTime = Date.now();
        const requestId = generateRequestId();

        config.headers['X-Request-ID'] = requestId;

        const token = await authManager.getAccessToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        const tenantId = authManager.getTenantId();
        if (tenantId) {
          config.headers['X-Tenant-ID'] = tenantId;
        }

        const abortController = new AbortController();
        config.signal = abortController.signal;
        this.requestQueue.set(requestId, abortController);

        // Store metadata on config
        (config as unknown as ExtendedAxiosConfig).metadata = { startTime, requestId };

        logger.logRequest(
          requestId,
          config.method?.toUpperCase() || '',
          config.url || '',
          { headers: config.headers, params: config.params }
        );

        return config;
      },
      (error: AxiosError) => {
        logger.logError('Request interceptor error', error);
        return Promise.reject(error);
      }
    );

    // Response Interceptor
    client.interceptors.response.use(
      (response: AxiosResponse) => {
        const config = response.config as ExtendedAxiosConfig;
        const duration = Date.now() - (config.metadata?.startTime || 0);
        const requestId = config.metadata?.requestId;

        if (requestId) {
          this.requestQueue.delete(requestId);
        }

        logger.logResponse(requestId, response.status, duration, response.data);

        if (this.shouldCache(response.config)) {
          apiCache.set(
            this.getCacheKey(response.config),
            response.data,
            this.getCacheTTL(response.config)
          );
        }

        return response;
      },
      async (error: AxiosError<ErrorResponseData>) => {
        const config = error.config as ExtendedAxiosConfig | undefined;
        const requestId = config?.metadata?.requestId;

        if (requestId) {
          this.requestQueue.delete(requestId);
        }

        if (error.response?.status === 401 && config && !config._retry) {
          config._retry = true;

          try {
            const newToken = await authManager.refreshAccessToken();
            if (newToken) {
              if (!config.headers) {
                config.headers = {};
              }
              (config.headers as Record<string, string>).Authorization = `Bearer ${newToken}`;
              return this.client.request(config);
            }
          } catch (refreshError) {
            authManager.logout();
            throw refreshError;
          }
        }

        logger.logError('API Error', error, {
          requestId,
          status: error.response?.status,
          url: config?.url,
        });

        throw this.transformError(error);
      }
    );
  }

  private shouldCache(config: AxiosRequestConfig): boolean {
    if (config.method?.toUpperCase() !== 'GET') {
      return false;
    }
    if ((config as ExtendedAxiosConfig).disableCache) {
      return false;
    }
    return true;
  }

  private getCacheKey(config: AxiosRequestConfig): string {
    const url = config.url || '';
    const params = JSON.stringify(config.params || {});
    return `${url}:${params}`;
  }

  private getCacheTTL(config: AxiosRequestConfig): number {
    return (config as ExtendedAxiosConfig).cacheTTL || 5 * 60 * 1000;
  }

  private transformError(error: AxiosError<ErrorResponseData>): ApiError {
    const config = error.config as ExtendedAxiosConfig | undefined;
    const responseData = error.response?.data;

    const apiError: ApiError = {
      message: error.message,
      code: error.code || 'UNKNOWN_ERROR',
      status: error.response?.status || 500,
      details: responseData?.details,
      request_id: config?.metadata?.requestId,
      timestamp: new Date().toISOString(),
    };

    if (responseData) {
      apiError.message = responseData.message || responseData.detail || error.message;
      apiError.code = responseData.code || apiError.code;
    }

    return apiError;
  }

  // Public API Methods
  async get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const cacheKey = this.getCacheKey({ url, ...config });
    const cachedData = apiCache.get(cacheKey);

    if (cachedData && !config?.params?.skipCache) {
      logger.logCacheHit(cacheKey);
      return { data: cachedData as T, status: 200, headers: {} };
    }

    const response = await retryWithBackoff(
      () => this.client.get<T>(url, config),
      MAX_RETRIES
    );

    return {
      data: response.data,
      status: response.status,
      headers: response.headers as Record<string, string>,
      request_id: (response.config as ExtendedAxiosConfig)?.metadata?.requestId,
    };
  }

  async post<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await retryWithBackoff(
      () => this.client.post<T>(url, data, config),
      MAX_RETRIES
    );

    return {
      data: response.data,
      status: response.status,
      headers: response.headers as Record<string, string>,
      request_id: (response.config as ExtendedAxiosConfig)?.metadata?.requestId,
    };
  }

  async put<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await retryWithBackoff(
      () => this.client.put<T>(url, data, config),
      MAX_RETRIES
    );

    return {
      data: response.data,
      status: response.status,
      headers: response.headers as Record<string, string>,
      request_id: (response.config as ExtendedAxiosConfig)?.metadata?.requestId,
    };
  }

  async patch<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await retryWithBackoff(
      () => this.client.patch<T>(url, data, config),
      MAX_RETRIES
    );

    return {
      data: response.data,
      status: response.status,
      headers: response.headers as Record<string, string>,
      request_id: (response.config as ExtendedAxiosConfig)?.metadata?.requestId,
    };
  }

  async delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await retryWithBackoff(
      () => this.client.delete<T>(url, config),
      MAX_RETRIES
    );

    return {
      data: response.data,
      status: response.status,
      headers: response.headers as Record<string, string>,
      request_id: (response.config as ExtendedAxiosConfig)?.metadata?.requestId,
    };
  }

  cancelRequest(requestId: string): void {
    const controller = this.requestQueue.get(requestId);
    if (controller) {
      controller.abort();
      this.requestQueue.delete(requestId);
      logger.logInfo('Request cancelled', { requestId });
    }
  }

  cancelAllRequests(): void {
    this.requestQueue.forEach((controller, requestId) => {
      controller.abort();
      logger.logInfo('Request cancelled', { requestId });
    });
    this.requestQueue.clear();
  }

  getWebSocketUrl(path: string): string {
    const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss' : 'ws';
    const baseUrl = API_BASE_URL.replace(/^https?/, wsProtocol);
    return `${baseUrl}${path}`;
  }
}

export const apiClient = new ApiClient();
