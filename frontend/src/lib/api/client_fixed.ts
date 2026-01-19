/**
 * Fixed Frontend API Client
 *
 * Addresses common integration issues:
 * 1. Timeout handling
 * 2. Retry logic
 * 3. Error normalization
 * 4. Authentication handling
 * 5. Request/response interceptors
 */

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
const API_VERSION = '/api/v1';
const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// ============================================================================
// Types
// ============================================================================

export interface ApiResponse<T = any> {
  data: T;
  success: boolean;
  message?: string;
  request_id?: string;
  timestamp?: string;
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, any>;
  timestamp?: string;
  request_id?: string;
  path?: string;
}

export interface RequestConfig extends AxiosRequestConfig {
  skipRetry?: boolean;
  skipAuth?: boolean;
  timeout?: number;
}

// ============================================================================
// Authentication Manager
// ============================================================================

class AuthManager {
  private apiKey: string | null = null;
  private jwtToken: string | null = null;

  setApiKey(key: string) {
    this.apiKey = key;
    // Store in localStorage for persistence
    if (typeof window !== 'undefined') {
      localStorage.setItem('chimera_api_key', key);
    }
  }

  setJwtToken(token: string) {
    this.jwtToken = token;
    // Store in localStorage for persistence
    if (typeof window !== 'undefined') {
      localStorage.setItem('chimera_jwt_token', token);
    }
  }

  getApiKey(): string | null {
    if (this.apiKey) return this.apiKey;

    // Try to load from localStorage
    if (typeof window !== 'undefined') {
      this.apiKey = localStorage.getItem('chimera_api_key');
      return this.apiKey;
    }

    return null;
  }

  getJwtToken(): string | null {
    if (this.jwtToken) return this.jwtToken;

    // Try to load from localStorage
    if (typeof window !== 'undefined') {
      this.jwtToken = localStorage.getItem('chimera_jwt_token');
      return this.jwtToken;
    }

    return null;
  }

  clearAuth() {
    this.apiKey = null;
    this.jwtToken = null;

    if (typeof window !== 'undefined') {
      localStorage.removeItem('chimera_api_key');
      localStorage.removeItem('chimera_jwt_token');
    }
  }

  isAuthenticated(): boolean {
    return this.getApiKey() !== null || this.getJwtToken() !== null;
  }
}

// ============================================================================
// Retry Logic
// ============================================================================

function shouldRetry(error: AxiosError, attempt: number): boolean {
  if (attempt >= MAX_RETRIES) return false;

  // Don't retry on client errors (4xx) except for specific cases
  if (error.response) {
    const status = error.response.status;

    // Retry on server errors (5xx) and specific client errors
    if (status >= 500) return true;
    if (status === 429) return true; // Rate limit
    if (status === 408) return true; // Request timeout
    if (status === 409) return true; // Conflict (might be temporary)

    return false;
  }

  // Retry on network errors or no response
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') return true;
  if (error.message.includes('timeout')) return true;
  if (error.message.includes('Network Error')) return true;

  return false;
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function executeWithRetry<T>(
  requestFn: () => Promise<T>,
  config: RequestConfig
): Promise<T> {
  let lastError: any;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await requestFn();
    } catch (error) {
      lastError = error;

      // Skip retry if explicitly disabled
      if (config.skipRetry) throw error;

      // Check if we should retry
      if (error instanceof Error && shouldRetry(error as AxiosError, attempt)) {
        const delay = RETRY_DELAY * Math.pow(2, attempt - 1); // Exponential backoff
        console.warn(`Request failed (attempt ${attempt}/${MAX_RETRIES}), retrying in ${delay}ms...`, error.message);
        await sleep(delay);
        continue;
      }

      throw error;
    }
  }

  throw lastError;
}

// ============================================================================
// API Client
// ============================================================================

export class FixedApiClient {
  private client: AxiosInstance;
  private authManager: AuthManager;
  private requestIdCounter = 0;

  constructor() {
    this.authManager = new AuthManager();
    this.client = this.createAxiosInstance();
  }

  private createAxiosInstance(): AxiosInstance {
    const client = axios.create({
      baseURL: `${API_BASE_URL}${API_VERSION}`,
      timeout: REQUEST_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      // Don't use withCredentials by default to avoid CORS issues
      withCredentials: false,
    });

    this.setupInterceptors(client);
    return client;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${++this.requestIdCounter}`;
  }

  private setupInterceptors(client: AxiosInstance): void {
    // Request interceptor
    client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add request ID for tracking
        const requestId = this.generateRequestId();
        config.headers['X-Request-ID'] = requestId;

        // Add authentication headers if not skipped
        const extendedConfig = config as RequestConfig;
        if (!extendedConfig.skipAuth) {
          const apiKey = this.authManager.getApiKey();
          const jwtToken = this.authManager.getJwtToken();

          if (apiKey) {
            config.headers['X-API-Key'] = apiKey;
          } else if (jwtToken) {
            config.headers['Authorization'] = `Bearer ${jwtToken}`;
          }
        }

        console.debug(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, {
          requestId,
          hasAuth: !extendedConfig.skipAuth && this.authManager.isAuthenticated(),
        });

        return config;
      },
      (error) => {
        console.error('[API Request Error]', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    client.interceptors.response.use(
      (response: AxiosResponse) => {
        const requestId = response.config.headers['X-Request-ID'];
        console.debug(`[API Response] ${response.status} ${response.config.url}`, {
          requestId,
          status: response.status,
        });

        return response;
      },
      (error: AxiosError) => {
        const requestId = error.config?.headers?.['X-Request-ID'];
        console.error(`[API Response Error]`, {
          requestId,
          status: error.response?.status,
          message: error.message,
          url: error.config?.url,
        });

        return Promise.reject(this.normalizeError(error));
      }
    );
  }

  private normalizeError(error: AxiosError): ApiError {
    if (error.response) {
      // Server responded with error status
      const data = error.response.data as any;

      return {
        error: data?.error || 'API_ERROR',
        message: data?.message || error.message || 'An API error occurred',
        details: data?.details || { status: error.response.status },
        timestamp: data?.timestamp || new Date().toISOString(),
        request_id: data?.request_id,
        path: data?.path || error.config?.url,
      };
    } else if (error.request) {
      // Network error or no response
      return {
        error: 'NETWORK_ERROR',
        message: 'Network error - please check your connection',
        details: {
          code: error.code,
          originalMessage: error.message,
        },
        timestamp: new Date().toISOString(),
      };
    } else {
      // Request setup error
      return {
        error: 'REQUEST_ERROR',
        message: error.message || 'Failed to make request',
        details: {},
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Authentication methods
  setApiKey(key: string) {
    this.authManager.setApiKey(key);
  }

  setJwtToken(token: string) {
    this.authManager.setJwtToken(token);
  }

  clearAuth() {
    this.authManager.clearAuth();
  }

  isAuthenticated(): boolean {
    return this.authManager.isAuthenticated();
  }

  // HTTP methods with retry logic
  async get<T = any>(url: string, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    return executeWithRetry(
      () => this.client.get<T>(url, config).then(response => ({
        data: response.data,
        success: true,
        request_id: response.headers['x-request-id'],
      })),
      config
    );
  }

  async post<T = any>(url: string, data?: any, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    return executeWithRetry(
      () => this.client.post<T>(url, data, config).then(response => ({
        data: response.data,
        success: true,
        request_id: response.headers['x-request-id'],
      })),
      config
    );
  }

  async put<T = any>(url: string, data?: any, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    return executeWithRetry(
      () => this.client.put<T>(url, data, config).then(response => ({
        data: response.data,
        success: true,
        request_id: response.headers['x-request-id'],
      })),
      config
    );
  }

  async patch<T = any>(url: string, data?: any, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    return executeWithRetry(
      () => this.client.patch<T>(url, data, config).then(response => ({
        data: response.data,
        success: true,
        request_id: response.headers['x-request-id'],
      })),
      config
    );
  }

  async delete<T = any>(url: string, config: RequestConfig = {}): Promise<ApiResponse<T>> {
    return executeWithRetry(
      () => this.client.delete<T>(url, config).then(response => ({
        data: response.data,
        success: true,
        request_id: response.headers['x-request-id'],
      })),
      config
    );
  }

  // Convenience methods for common operations
  async healthCheck(): Promise<ApiResponse<{ status: string }>> {
    return this.get('/health', { skipAuth: true });
  }

  async getProviders(): Promise<ApiResponse<any>> {
    return this.get('/providers', { skipAuth: true });
  }

  async generateText(request: any): Promise<ApiResponse<any>> {
    return this.post('/generate', request);
  }

  async transformPrompt(request: any): Promise<ApiResponse<any>> {
    return this.post('/transform', request);
  }

  // Test connection with timeout
  async testConnection(timeoutMs: number = 5000): Promise<boolean> {
    try {
      const result = await this.get('/health', {
        skipAuth: true,
        timeout: timeoutMs,
        skipRetry: true,
      });
      return result.success;
    } catch (error) {
      console.warn('Connection test failed:', error);
      return false;
    }
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const apiClient = new FixedApiClient();

// Export types and classes
export { AuthManager };
export default apiClient;
