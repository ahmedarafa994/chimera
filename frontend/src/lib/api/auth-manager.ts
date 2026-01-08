/**
 * Authentication Manager
 * Handles token management, refresh logic, and session handling
 */

import { AuthTokens, User } from './types';
import { logger } from './logger';

// ============================================================================
// Configuration
// ============================================================================

const TOKEN_STORAGE_KEY = 'chimera_auth_tokens';
const USER_STORAGE_KEY = 'chimera_user';
const TENANT_STORAGE_KEY = 'chimera_tenant_id';
const TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before expiry

// ============================================================================
// Auth Manager Implementation
// ============================================================================

class AuthManager {
  private tokens: AuthTokens | null = null;
  private user: User | null = null;
  private tenantId: string | null = null;
  private refreshPromise: Promise<string | null> | null = null;
  private tokenExpiryTime: number = 0;

  constructor() {
    this.loadFromStorage();
  }

  /**
   * Load auth state from storage
   */
  private loadFromStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      const tokensJson = localStorage.getItem(TOKEN_STORAGE_KEY);
      const userJson = localStorage.getItem(USER_STORAGE_KEY);
      const tenantId = localStorage.getItem(TENANT_STORAGE_KEY);

      if (tokensJson) {
        this.tokens = JSON.parse(tokensJson);
        if (this.tokens) {
          this.tokenExpiryTime = Date.now() + (this.tokens.expires_in * 1000);
        }
      }

      if (userJson) {
        this.user = JSON.parse(userJson);
      }

      if (tenantId) {
        this.tenantId = tenantId;
      }

      logger.logDebug('Auth state loaded from storage', {
        hasTokens: !!this.tokens,
        hasUser: !!this.user,
        tenantId: this.tenantId,
      });
    } catch (error) {
      logger.logError('Failed to load auth state from storage', error);
      this.clearStorage();
    }
  }

  /**
   * Save auth state to storage
   */
  private saveToStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      if (this.tokens) {
        localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(this.tokens));
      } else {
        localStorage.removeItem(TOKEN_STORAGE_KEY);
      }

      if (this.user) {
        localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(this.user));
      } else {
        localStorage.removeItem(USER_STORAGE_KEY);
      }

      if (this.tenantId) {
        localStorage.setItem(TENANT_STORAGE_KEY, this.tenantId);
      } else {
        localStorage.removeItem(TENANT_STORAGE_KEY);
      }
    } catch (error) {
      logger.logError('Failed to save auth state to storage', error);
    }
  }

  /**
   * Clear storage
   */
  private clearStorage(): void {
    if (typeof window === 'undefined') return;

    localStorage.removeItem(TOKEN_STORAGE_KEY);
    localStorage.removeItem(USER_STORAGE_KEY);
    localStorage.removeItem(TENANT_STORAGE_KEY);
  }

  /**
   * Set authentication tokens
   */
  setTokens(tokens: AuthTokens): void {
    this.tokens = tokens;
    this.tokenExpiryTime = Date.now() + (tokens.expires_in * 1000);
    this.saveToStorage();
    logger.logInfo('Auth tokens updated', { expiresIn: tokens.expires_in });
  }

  /**
   * Set user information
   */
  setUser(user: User): void {
    this.user = user;
    this.tenantId = user.tenant_id;
    this.saveToStorage();
    logger.logInfo('User set', { userId: user.id, tenantId: user.tenant_id });
  }

  /**
   * Get access token with automatic refresh if needed
   */
  async getAccessToken(): Promise<string | null> {
    if (!this.tokens) {
      return null;
    }

    // Check if token is about to expire
    const timeUntilExpiry = this.tokenExpiryTime - Date.now();

    if (timeUntilExpiry < TOKEN_REFRESH_THRESHOLD) {
      logger.logDebug('Token expiring soon, refreshing', { timeUntilExpiry });
      return this.refreshAccessToken();
    }

    return this.tokens.access_token;
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(): Promise<string | null> {
    // If already refreshing, wait for that to complete
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    if (!this.tokens?.refresh_token) {
      logger.logWarning('No refresh token available');
      return null;
    }

    this.refreshPromise = this.doRefreshToken();

    try {
      const result = await this.refreshPromise;
      return result;
    } finally {
      this.refreshPromise = null;
    }
  }

  /**
   * Perform token refresh
   */
  private async doRefreshToken(): Promise<string | null> {
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
      const response = await fetch(`${baseUrl}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refresh_token: this.tokens?.refresh_token,
        }),
      });

      if (!response.ok) {
        throw new Error(`Token refresh failed: ${response.status}`);
      }

      const newTokens: AuthTokens = await response.json();
      this.setTokens(newTokens);

      logger.logInfo('Token refreshed successfully');
      return newTokens.access_token;
    } catch (error) {
      logger.logError('Token refresh failed', error);
      this.logout();
      return null;
    }
  }

  /**
   * Get current user
   */
  getUser(): User | null {
    return this.user;
  }

  /**
   * Get tenant ID
   */
  getTenantId(): string | null {
    return this.tenantId;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.tokens && Date.now() < this.tokenExpiryTime;
  }

  /**
   * Check if user has specific role
   */
  hasRole(role: string): boolean {
    return this.user?.roles?.includes(role) ?? false;
  }

  /**
   * Check if user has specific permission
   */
  hasPermission(permission: string): boolean {
    return this.user?.permissions?.includes(permission) ?? false;
  }

  /**
   * Login with credentials
   */
  async login(email: string, password: string): Promise<boolean> {
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
      const response = await fetch(`${baseUrl}/api/v1/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        throw new Error(`Login failed: ${response.status}`);
      }

      const data = await response.json();
      this.setTokens(data.tokens);
      this.setUser(data.user);

      logger.logInfo('Login successful', { userId: data.user.id });
      return true;
    } catch (error) {
      logger.logError('Login failed', error);
      return false;
    }
  }

  /**
   * Logout and clear all auth state
   */
  logout(): void {
    this.tokens = null;
    this.user = null;
    this.tenantId = null;
    this.tokenExpiryTime = 0;
    this.clearStorage();
    logger.logInfo('User logged out');
  }

  /**
   * Register event listener for auth state changes
   */
  onAuthStateChange(callback: (isAuthenticated: boolean) => void): () => void {
    const handler = (event: StorageEvent) => {
      if (event.key === TOKEN_STORAGE_KEY) {
        this.loadFromStorage();
        callback(this.isAuthenticated());
      }
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('storage', handler);
      return () => window.removeEventListener('storage', handler);
    }

    return () => { };
  }
}

// Export singleton instance
export const authManager = new AuthManager();
