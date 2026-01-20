/**
 * Authentication Manager
 * Handles token management, refresh logic, and session handling
 */

import { AuthTokens, User } from './types';
import { logger } from './logger';

// ============================================================================
// Configuration
// ============================================================================

// Keys must match AuthContext.tsx AUTH_STORAGE_KEYS
const STORAGE_KEYS = {
  ACCESS_TOKEN: 'chimera_access_token',
  REFRESH_TOKEN: 'chimera_refresh_token',
  USER: 'chimera_auth_user',
  TOKEN_EXPIRY: 'chimera_token_expiry',
  REFRESH_EXPIRY: 'chimera_refresh_expiry',
  TENANT_ID: 'chimera_tenant_id',
};

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
      const accessToken = localStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN);
      const refreshToken = localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN);
      const userJson = localStorage.getItem(STORAGE_KEYS.USER);
      const tenantId = localStorage.getItem(STORAGE_KEYS.TENANT_ID);
      const expiryStr = localStorage.getItem(STORAGE_KEYS.TOKEN_EXPIRY);
      const refreshExpiryStr = localStorage.getItem(STORAGE_KEYS.REFRESH_EXPIRY);

      if (accessToken && refreshToken) {
        // AuthProvider stores tokens encoded (btoa(encodeURIComponent(value)))
        // We must decode them to get the raw JWT
        try {
          const decode = (val: string) => decodeURIComponent(atob(val));
          this.tokens = {
            access_token: decode(accessToken),
            refresh_token: decode(refreshToken),
            token_type: 'Bearer',
            expires_in: expiryStr ? (parseInt(expiryStr, 10) - Date.now()) / 1000 : 3600,
            refresh_expires_in: refreshExpiryStr ? (parseInt(refreshExpiryStr, 10) - Date.now()) / 1000 : 86400,
          };
        } catch (e) {
          logger.logError('Failed to decode tokens from storage', e);
          this.tokens = null; // Invalid tokens
        }

        if (expiryStr) {
          this.tokenExpiryTime = parseInt(expiryStr, 10);
        } else {
          // Fallback if expiry not stored
          this.tokenExpiryTime = Date.now() + 3600 * 1000;
        }
      }

      if (userJson) {
        try {
          // Check if userJson is double encoded or simple json
          // secureStorage in AuthProvider might encode it
          // implementation in AuthProvider uses btoa/encodeURIComponent?
          // "const encoded = btoa(encodeURIComponent(value));"
          // Wait, AuthProvider's SecureStorage DOES encode. Use decode logic.
          const decoded = decodeURIComponent(atob(userJson));
          this.user = JSON.parse(decoded);
        } catch {
          // Try standard JSON parse if decode fails
          try {
            this.user = JSON.parse(userJson);
          } catch (e) {
            logger.logError('Failed to parse user from storage', e);
          }
        }
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
        // AuthProvider expects encoded values.
        // We MUST encode if we write to the same keys.
        const encode = (val: string) => btoa(encodeURIComponent(val));

        localStorage.setItem(STORAGE_KEYS.ACCESS_TOKEN, encode(this.tokens.access_token));
        localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, encode(this.tokens.refresh_token));

        const expiresAt = Date.now() + this.tokens.expires_in * 1000;
        const refreshExpiresAt = Date.now() + this.tokens.refresh_expires_in * 1000;

        localStorage.setItem(STORAGE_KEYS.TOKEN_EXPIRY, String(expiresAt)); // Expiry is number, AuthProvider doesn't encode it?
        // Wait, AuthProvider lines 74-83: set(key, value) -> ALWAYS encodes.
        // AuthProvider line 249: storage.set(AUTH_STORAGE_KEYS.TOKEN_EXPIRY, String(expiresAt));
        // YES, it encodes expiry too!
        localStorage.setItem(STORAGE_KEYS.TOKEN_EXPIRY, encode(String(expiresAt)));
        localStorage.setItem(STORAGE_KEYS.REFRESH_EXPIRY, encode(String(refreshExpiresAt)));
      } else {
        localStorage.removeItem(STORAGE_KEYS.ACCESS_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
        localStorage.removeItem(STORAGE_KEYS.TOKEN_EXPIRY);
        localStorage.removeItem(STORAGE_KEYS.REFRESH_EXPIRY);
      }

      if (this.user) {
        const encode = (val: string) => btoa(encodeURIComponent(val));
        localStorage.setItem(STORAGE_KEYS.USER, encode(JSON.stringify(this.user)));
      } else {
        localStorage.removeItem(STORAGE_KEYS.USER);
      }

      if (this.tenantId) {
        localStorage.setItem(STORAGE_KEYS.TENANT_ID, this.tenantId);
      } else {
        localStorage.removeItem(STORAGE_KEYS.TENANT_ID);
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

    Object.values(STORAGE_KEYS).forEach(key => localStorage.removeItem(key));
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
    // Reload from storage to ensure we have latest from AuthProvider
    this.loadFromStorage();

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

    // Reload to ensure we have refresh token
    this.loadFromStorage();

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
    this.loadFromStorage();
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
    this.loadFromStorage();
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
      // Check for any relevant key change
      if (Object.values(STORAGE_KEYS).includes(event.key as any)) {
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
