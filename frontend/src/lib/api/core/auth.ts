/**
 * Authentication and Security Layer
 *
 * Provides secure token management, automatic refresh mechanisms,
 * header injection, and request signing capabilities.
 *
 * @module lib/api/core/auth
 */

import { configManager, AIProvider } from './config';
import { AuthenticationError, TokenExpiredError } from '../../errors';

// ============================================================================
// Types
// ============================================================================

export interface TokenInfo {
  accessToken: string;
  refreshToken?: string;
  expiresAt?: number;
  tokenType: string;
  scope?: string;
}

export interface APIKeyInfo {
  key: string;
  provider: AIProvider;
  expiresAt?: number;
}

export interface AuthHeaders {
  Authorization?: string;
  'X-API-Key'?: string;
  'X-Request-Signature'?: string;
  'X-Request-Timestamp'?: string;
  'X-Client-ID'?: string;
  [key: string]: string | undefined;
}

export interface RefreshTokenResult {
  accessToken: string;
  refreshToken?: string;
  expiresIn?: number;
}

export type TokenRefreshFn = (refreshToken: string) => Promise<RefreshTokenResult>;

export interface AuthConfig {
  tokenStorageKey: string;
  apiKeyStoragePrefix: string;
  refreshThresholdMs: number;
  enableAutoRefresh: boolean;
  clientId?: string;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_AUTH_CONFIG: AuthConfig = {
  tokenStorageKey: 'chimera_auth_token',
  apiKeyStoragePrefix: 'chimera_api_key_',
  refreshThresholdMs: 5 * 60 * 1000, // 5 minutes before expiry
  enableAutoRefresh: true,
};

// ============================================================================
// Secure Storage
// ============================================================================

class SecureStorage {
  private memoryStorage: Map<string, string> = new Map();
  private useLocalStorage: boolean;

  constructor() {
    this.useLocalStorage = typeof window !== 'undefined' && !!window.localStorage;
  }

  set(key: string, value: string): void {
    if (this.useLocalStorage) {
      try {
        // Encode value to prevent XSS
        const encoded = btoa(encodeURIComponent(value));
        localStorage.setItem(key, encoded);
      } catch {
        // Fallback to memory storage
        this.memoryStorage.set(key, value);
      }
    } else {
      this.memoryStorage.set(key, value);
    }
  }

  get(key: string): string | null {
    if (this.useLocalStorage) {
      try {
        const encoded = localStorage.getItem(key);
        if (encoded) {
          return decodeURIComponent(atob(encoded));
        }
        return null;
      } catch {
        return this.memoryStorage.get(key) || null;
      }
    }
    return this.memoryStorage.get(key) || null;
  }

  remove(key: string): void {
    if (this.useLocalStorage) {
      try {
        localStorage.removeItem(key);
      } catch {
        // Ignore
      }
    }
    this.memoryStorage.delete(key);
  }

  clear(): void {
    if (this.useLocalStorage) {
      try {
        // Only clear our keys
        const keysToRemove: string[] = [];
        for (let i = 0; i < localStorage.length; i++) {
          const key = localStorage.key(i);
          if (key?.startsWith('chimera_')) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach(key => localStorage.removeItem(key));
      } catch {
        // Ignore
      }
    }
    this.memoryStorage.clear();
  }
}

// ============================================================================
// Token Manager
// ============================================================================

class TokenManager {
  private storage: SecureStorage;
  private config: AuthConfig;
  private refreshFn: TokenRefreshFn | null = null;
  private refreshPromise: Promise<TokenInfo> | null = null;
  private refreshTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(config: Partial<AuthConfig> = {}) {
    this.storage = new SecureStorage();
    this.config = { ...DEFAULT_AUTH_CONFIG, ...config };
  }

  /**
   * Set the token refresh function
   */
  setRefreshFunction(fn: TokenRefreshFn): void {
    this.refreshFn = fn;
  }

  /**
   * Store authentication token
   */
  setToken(tokenInfo: TokenInfo): void {
    this.storage.set(this.config.tokenStorageKey, JSON.stringify(tokenInfo));

    if (this.config.enableAutoRefresh && tokenInfo.expiresAt && tokenInfo.refreshToken) {
      this.scheduleRefresh(tokenInfo);
    }
  }

  /**
   * Get current token
   */
  getToken(): TokenInfo | null {
    const stored = this.storage.get(this.config.tokenStorageKey);
    if (!stored) return null;

    try {
      return JSON.parse(stored) as TokenInfo;
    } catch {
      return null;
    }
  }

  /**
   * Get access token string
   */
  getAccessToken(): string | null {
    const token = this.getToken();
    return token?.accessToken || null;
  }

  /**
   * Check if token is expired
   */
  isTokenExpired(): boolean {
    const token = this.getToken();
    if (!token || !token.expiresAt) return false;
    return Date.now() >= token.expiresAt;
  }

  /**
   * Check if token needs refresh
   */
  needsRefresh(): boolean {
    const token = this.getToken();
    if (!token || !token.expiresAt) return false;
    return Date.now() >= token.expiresAt - this.config.refreshThresholdMs;
  }

  /**
   * Refresh the token
   */
  async refreshToken(): Promise<TokenInfo> {
    // Return existing refresh promise if one is in progress
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    const currentToken = this.getToken();
    if (!currentToken?.refreshToken) {
      throw new AuthenticationError('No refresh token available');
    }

    if (!this.refreshFn) {
      throw new AuthenticationError('No refresh function configured');
    }

    this.refreshPromise = this.performRefresh(currentToken.refreshToken);

    try {
      const newToken = await this.refreshPromise;
      return newToken;
    } finally {
      this.refreshPromise = null;
    }
  }

  /**
   * Clear token
   */
  clearToken(): void {
    this.storage.remove(this.config.tokenStorageKey);
    this.cancelRefreshTimer();
  }

  /**
   * Get valid token, refreshing if necessary
   */
  async getValidToken(): Promise<string> {
    const token = this.getToken();

    if (!token) {
      throw new AuthenticationError('No authentication token');
    }

    if (this.isTokenExpired()) {
      throw new TokenExpiredError('Token has expired');
    }

    if (this.needsRefresh() && token.refreshToken) {
      try {
        const newToken = await this.refreshToken();
        return newToken.accessToken;
      } catch (error) {
        // If refresh fails but token is still valid, use it
        if (!this.isTokenExpired()) {
          return token.accessToken;
        }
        // Use (error as any).errorCode when checking property on generic Error.
        throw new TokenExpiredError('Token refresh failed', (error as any).errorCode || 'UNKNOWN_ERROR');
      }
    }

    return token.accessToken;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async performRefresh(refreshToken: string): Promise<TokenInfo> {
    if (!this.refreshFn) {
      throw new AuthenticationError('No refresh function configured');
    }

    const result = await this.refreshFn(refreshToken);

    const newToken: TokenInfo = {
      accessToken: result.accessToken,
      refreshToken: result.refreshToken || refreshToken,
      expiresAt: result.expiresIn
        ? Date.now() + result.expiresIn * 1000
        : undefined,
      tokenType: 'Bearer',
    };

    this.setToken(newToken);
    return newToken;
  }

  private scheduleRefresh(token: TokenInfo): void {
    this.cancelRefreshTimer();

    if (!token.expiresAt || !token.refreshToken) return;

    const refreshTime = token.expiresAt - this.config.refreshThresholdMs - Date.now();

    if (refreshTime > 0) {
      this.refreshTimer = setTimeout(() => {
        this.refreshToken().catch(console.error);
      }, refreshTime);
    }
  }

  private cancelRefreshTimer(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }
}

// ============================================================================
// API Key Manager
// ============================================================================

class APIKeyManager {
  private storage: SecureStorage;
  private config: AuthConfig;

  constructor(config: Partial<AuthConfig> = {}) {
    this.storage = new SecureStorage();
    this.config = { ...DEFAULT_AUTH_CONFIG, ...config };
  }

  /**
   * Store API key for a provider
   */
  setAPIKey(provider: AIProvider, key: string, expiresAt?: number): void {
    const info: APIKeyInfo = { key, provider, expiresAt };
    this.storage.set(
      `${this.config.apiKeyStoragePrefix}${provider}`,
      JSON.stringify(info)
    );
  }

  /**
   * Get API key for a provider
   */
  getAPIKey(provider: AIProvider): string | null {
    const stored = this.storage.get(`${this.config.apiKeyStoragePrefix}${provider}`);
    if (!stored) return null;

    try {
      const info = JSON.parse(stored) as APIKeyInfo;

      // Check expiration
      if (info.expiresAt && Date.now() >= info.expiresAt) {
        this.removeAPIKey(provider);
        return null;
      }

      return info.key;
    } catch {
      return null;
    }
  }

  /**
   * Remove API key for a provider
   */
  removeAPIKey(provider: AIProvider): void {
    this.storage.remove(`${this.config.apiKeyStoragePrefix}${provider}`);
  }

  /**
   * Get all stored API keys
   */
  getAllAPIKeys(): Map<AIProvider, string> {
    const keys = new Map<AIProvider, string>();
    const providers: AIProvider[] = ['gemini', 'deepseek', 'openai', 'anthropic'];

    for (const provider of providers) {
      const key = this.getAPIKey(provider);
      if (key) {
        keys.set(provider, key);
      }
    }

    return keys;
  }

  /**
   * Clear all API keys
   */
  clearAllAPIKeys(): void {
    const providers: AIProvider[] = ['gemini', 'deepseek', 'openai', 'anthropic'];
    providers.forEach(provider => this.removeAPIKey(provider));
  }
}

// ============================================================================
// Request Signer
// ============================================================================

class RequestSigner {
  private secretKey: string | null = null;

  /**
   * Set the secret key for signing
   */
  setSecretKey(key: string): void {
    this.secretKey = key;
  }

  /**
   * Sign a request
   */
  async sign(
    method: string,
    url: string,
    body?: unknown,
    timestamp?: number
  ): Promise<{ signature: string; timestamp: number }> {
    const ts = timestamp || Date.now();
    const payload = this.createSignaturePayload(method, url, body, ts);
    const signature = await this.computeSignature(payload);

    return { signature, timestamp: ts };
  }

  /**
   * Verify a signature
   */
  async verify(
    method: string,
    url: string,
    body: unknown,
    signature: string,
    timestamp: number
  ): Promise<boolean> {
    const payload = this.createSignaturePayload(method, url, body, timestamp);
    const expectedSignature = await this.computeSignature(payload);
    return signature === expectedSignature;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private createSignaturePayload(
    method: string,
    url: string,
    body: unknown,
    timestamp: number
  ): string {
    const parts = [
      method.toUpperCase(),
      url,
      timestamp.toString(),
    ];

    if (body) {
      parts.push(JSON.stringify(body));
    }

    return parts.join('\n');
  }

  private async computeSignature(payload: string): Promise<string> {
    if (!this.secretKey) {
      throw new Error('No secret key configured for request signing');
    }

    // Use Web Crypto API if available
    if (typeof window !== 'undefined' && window.crypto?.subtle) {
      const encoder = new TextEncoder();
      const keyData = encoder.encode(this.secretKey);
      const payloadData = encoder.encode(payload);

      const key = await window.crypto.subtle.importKey(
        'raw',
        keyData,
        { name: 'HMAC', hash: 'SHA-256' },
        false,
        ['sign']
      );

      const signature = await window.crypto.subtle.sign('HMAC', key, payloadData);
      return this.arrayBufferToHex(signature);
    }

    // Fallback: simple hash (not cryptographically secure, for development only)
    return this.simpleHash(payload + this.secretKey);
  }

  private arrayBufferToHex(buffer: ArrayBuffer): string {
    return Array.from(new Uint8Array(buffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16).padStart(8, '0');
  }
}

// ============================================================================
// Auth Header Builder
// ============================================================================

class AuthHeaderBuilder {
  private tokenManager: TokenManager;
  private apiKeyManager: APIKeyManager;
  private requestSigner: RequestSigner;
  private config: AuthConfig;

  constructor(
    tokenManager: TokenManager,
    apiKeyManager: APIKeyManager,
    requestSigner: RequestSigner,
    config: Partial<AuthConfig> = {}
  ) {
    this.tokenManager = tokenManager;
    this.apiKeyManager = apiKeyManager;
    this.requestSigner = requestSigner;
    this.config = { ...DEFAULT_AUTH_CONFIG, ...config };
  }

  /**
   * Build authentication headers for a request
   */
  async buildHeaders(options: {
    method: string;
    url: string;
    body?: unknown;
    provider?: AIProvider;
    requireAuth?: boolean;
    signRequest?: boolean;
  }): Promise<AuthHeaders> {
    const headers: AuthHeaders = {};

    // Add Bearer token if available
    if (options.requireAuth !== false) {
      try {
        const token = await this.tokenManager.getValidToken();
        headers.Authorization = `Bearer ${token}`;
      } catch {
        // Token not available or expired
        if (options.requireAuth === true) {
          throw new AuthenticationError('Authentication required');
        }
      }
    }

    // Add API key for provider if specified
    if (options.provider) {
      const apiKey = this.apiKeyManager.getAPIKey(options.provider);
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
    }

    // Add client ID if configured
    if (this.config.clientId) {
      headers['X-Client-ID'] = this.config.clientId;
    }

    // Sign request if required
    if (options.signRequest) {
      const { signature, timestamp } = await this.requestSigner.sign(
        options.method,
        options.url,
        options.body
      );
      headers['X-Request-Signature'] = signature;
      headers['X-Request-Timestamp'] = timestamp.toString();
    }

    return headers;
  }

  /**
   * Get headers for a specific provider
   */
  getProviderHeaders(provider: AIProvider): AuthHeaders {
    const headers: AuthHeaders = {};
    const providerConfig = configManager.getProviderConfig(provider);

    if (providerConfig?.apiKey) {
      // Different providers use different header names
      switch (provider) {
        case 'openai':
          headers.Authorization = `Bearer ${providerConfig.apiKey}`;
          break;
        case 'anthropic':
          headers['X-API-Key'] = providerConfig.apiKey;
          headers['anthropic-version'] = '2023-06-01';
          break;
        case 'gemini':
          // Gemini uses query parameter, but we can also set header
          headers['X-Goog-Api-Key'] = providerConfig.apiKey;
          break;
        case 'deepseek':
          headers.Authorization = `Bearer ${providerConfig.apiKey}`;
          break;
      }
    }

    return headers;
  }
}

// ============================================================================
// Singleton Instances
// ============================================================================

export const tokenManager = new TokenManager();
export const apiKeyManager = new APIKeyManager();
export const requestSigner = new RequestSigner();
export const authHeaderBuilder = new AuthHeaderBuilder(
  tokenManager,
  apiKeyManager,
  requestSigner
);

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Set authentication token
 */
export function setAuthToken(token: TokenInfo): void {
  tokenManager.setToken(token);
}

/**
 * Get current authentication token
 */
export function getAuthToken(): TokenInfo | null {
  return tokenManager.getToken();
}

/**
 * Clear authentication
 */
export function clearAuth(): void {
  tokenManager.clearToken();
  apiKeyManager.clearAllAPIKeys();
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  const token = tokenManager.getToken();
  return !!token && !tokenManager.isTokenExpired();
}

/**
 * Set API key for a provider
 */
export function setProviderAPIKey(provider: AIProvider, key: string): void {
  apiKeyManager.setAPIKey(provider, key);
}

/**
 * Get API key for a provider
 */
export function getProviderAPIKey(provider: AIProvider): string | null {
  return apiKeyManager.getAPIKey(provider);
}

/**
 * Configure token refresh
 */
export function configureTokenRefresh(refreshFn: TokenRefreshFn): void {
  tokenManager.setRefreshFunction(refreshFn);
}

/**
 * Configure request signing
 */
export function configureRequestSigning(secretKey: string): void {
  requestSigner.setSecretKey(secretKey);
}

/**
 * Build auth headers for a request
 */
export async function buildAuthHeaders(options: {
  method: string;
  url: string;
  body?: unknown;
  provider?: AIProvider;
  requireAuth?: boolean;
  signRequest?: boolean;
}): Promise<AuthHeaders> {
  return authHeaderBuilder.buildHeaders(options);
}
