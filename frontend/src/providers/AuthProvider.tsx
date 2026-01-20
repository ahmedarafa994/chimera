"use client";

/**
 * Authentication Provider
 *
 * Provides authentication functionality including:
 * - Login/logout with JWT tokens
 * - Token refresh logic with auto-refresh before expiry
 * - User registration
 * - Password reset flow
 * - Email verification
 * - Persistent auth state in localStorage
 * - Role-based access helpers
 *
 * @module providers/AuthProvider
 */

import React, {
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type ReactNode,
} from "react";

import {
  AuthContext,
  type AuthState,
  type AuthContextValue,
  type AuthUser,
  type AuthTokens,
  type LoginCredentials,
  type LoginResponse,
  type RegistrationData,
  type RegistrationResponse,
  type PasswordStrengthResult,
  type AuthError,
  type UserRole,
  AUTH_STORAGE_KEYS,
  TOKEN_REFRESH_THRESHOLD_MS,
  ROLE_HIERARCHY,
} from "@/contexts/AuthContext";

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = (process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8002").replace(/\/api\/v1\/?$/, "");
const AUTH_ENDPOINTS = {
  LOGIN: "/api/v1/auth/login",
  LOGOUT: "/api/v1/auth/logout",
  REFRESH: "/api/v1/auth/refresh",
  ME: "/api/v1/auth/me",
  REGISTER: "/api/v1/auth/register",
  VERIFY_EMAIL: "/api/v1/auth/verify-email",
  RESEND_VERIFICATION: "/api/v1/auth/resend-verification",
  FORGOT_PASSWORD: "/api/v1/auth/forgot-password",
  RESET_PASSWORD: "/api/v1/auth/reset-password",
  CHECK_PASSWORD_STRENGTH: "/api/v1/auth/check-password-strength",
} as const;

// =============================================================================
// Storage Helpers
// =============================================================================

class SecureStorage {
  private isAvailable: boolean;

  constructor() {
    this.isAvailable = typeof window !== "undefined" && !!window.localStorage;
  }

  set(key: string, value: string): void {
    if (!this.isAvailable) return;
    try {
      // Encode for XSS prevention
      const encoded = btoa(encodeURIComponent(value));
      localStorage.setItem(key, encoded);
    } catch {
      // Ignore storage errors
    }
  }

  get(key: string): string | null {
    if (!this.isAvailable) return null;
    try {
      const encoded = localStorage.getItem(key);
      if (!encoded) return null;
      return decodeURIComponent(atob(encoded));
    } catch {
      return null;
    }
  }

  remove(key: string): void {
    if (!this.isAvailable) return;
    try {
      localStorage.removeItem(key);
    } catch {
      // Ignore
    }
  }

  clear(): void {
    if (!this.isAvailable) return;
    try {
      Object.values(AUTH_STORAGE_KEYS).forEach((key) =>
        localStorage.removeItem(key)
      );
    } catch {
      // Ignore
    }
  }
}

const storage = new SecureStorage();

/**
 * Check if the backend service is available
 */
async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL.replace('/api/v1', '')}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout for health check
    });
    return response.ok;
  } catch (error) {
    console.warn('[AuthProvider] Backend health check failed:', error);
    return false;
  }
}

// =============================================================================
// API Helpers
// =============================================================================

/**
 * Make an authenticated API request with health checking and graceful fallback
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {},
  accessToken?: string
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  if (accessToken) {
    (headers as Record<string, string>)["Authorization"] =
      `Bearer ${accessToken}`;
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      // Add timeout to prevent hanging
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const error: AuthError = {
        message: errorData.detail?.message || errorData.detail || response.statusText,
        code: String(response.status),
        requires_verification: errorData.detail?.requires_verification,
        field_errors: errorData.detail?.errors,
      };
      throw error;
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  } catch (error) {
    // Handle network errors and service unavailable
    if (error instanceof TypeError && error.message.includes('fetch')) {
      const networkError: AuthError = {
        message: "Service temporarily unavailable. Please check your connection and try again.",
        code: "SERVICE_UNAVAILABLE",
      };
      throw networkError;
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
      const timeoutError: AuthError = {
        message: "Request timed out. Please try again.",
        code: "REQUEST_TIMEOUT",
      };
      throw timeoutError;
    }

    // Re-throw AuthError instances
    throw error;
  }
}

// =============================================================================
// Provider Component
// =============================================================================

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  // State
  const [state, setState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
    isInitialized: false,
    error: null,
    tokenExpiresAt: null,
    refreshTokenExpiresAt: null,
  });

  // Auth state readiness flag to prevent race conditions during login redirects
  const [isAuthStateReady, setIsAuthStateReady] = useState(false);

  // Refs for managing refresh timer and preventing race conditions
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const refreshPromiseRef = useRef<Promise<boolean> | null>(null);
  const initializingRef = useRef(false);

  // ==========================================================================
  // Storage Management
  // ==========================================================================

  /**
   * Save tokens to storage
   */
  const saveTokens = useCallback((tokens: AuthTokens) => {
    storage.set(AUTH_STORAGE_KEYS.ACCESS_TOKEN, tokens.access_token);
    storage.set(AUTH_STORAGE_KEYS.REFRESH_TOKEN, tokens.refresh_token);

    const expiresAt = Date.now() + tokens.expires_in * 1000;
    const refreshExpiresAt = Date.now() + tokens.refresh_expires_in * 1000;

    storage.set(AUTH_STORAGE_KEYS.TOKEN_EXPIRY, String(expiresAt));
    storage.set(AUTH_STORAGE_KEYS.REFRESH_EXPIRY, String(refreshExpiresAt));

    return { expiresAt, refreshExpiresAt };
  }, []);

  /**
   * Save user to storage
   */
  const saveUser = useCallback((user: AuthUser) => {
    storage.set(AUTH_STORAGE_KEYS.USER, JSON.stringify(user));
  }, []);

  /**
   * Load auth state from storage
   */
  const loadFromStorage = useCallback((): {
    user: AuthUser | null;
    accessToken: string | null;
    refreshToken: string | null;
    tokenExpiresAt: number | null;
    refreshTokenExpiresAt: number | null;
  } => {
    const accessToken = storage.get(AUTH_STORAGE_KEYS.ACCESS_TOKEN);
    const refreshToken = storage.get(AUTH_STORAGE_KEYS.REFRESH_TOKEN);
    const userJson = storage.get(AUTH_STORAGE_KEYS.USER);
    const expiryStr = storage.get(AUTH_STORAGE_KEYS.TOKEN_EXPIRY);
    const refreshExpiryStr = storage.get(AUTH_STORAGE_KEYS.REFRESH_EXPIRY);

    let user: AuthUser | null = null;
    if (userJson) {
      try {
        user = JSON.parse(userJson);
      } catch {
        // Invalid JSON, ignore
      }
    }

    return {
      user,
      accessToken,
      refreshToken,
      tokenExpiresAt: expiryStr ? parseInt(expiryStr, 10) : null,
      refreshTokenExpiresAt: refreshExpiryStr
        ? parseInt(refreshExpiryStr, 10)
        : null,
    };
  }, []);

  /**
   * Clear all auth storage
   */
  const clearStorage = useCallback(() => {
    storage.clear();
  }, []);

  // ==========================================================================
  // Token Management
  // ==========================================================================

  /**
   * Get access token, refreshing if needed
   */
  const getAccessToken = useCallback(async (): Promise<string | null> => {
    const { accessToken, tokenExpiresAt, refreshToken, refreshTokenExpiresAt } =
      loadFromStorage();

    if (!accessToken) {
      return null;
    }

    const now = Date.now();

    // Check if access token is still valid (with threshold)
    if (tokenExpiresAt && now < tokenExpiresAt - TOKEN_REFRESH_THRESHOLD_MS) {
      return accessToken;
    }

    // Access token expired or expiring soon, try to refresh
    if (refreshToken && refreshTokenExpiresAt && now < refreshTokenExpiresAt) {
      // Use existing refresh promise if one is in flight
      if (refreshPromiseRef.current) {
        const success = await refreshPromiseRef.current;
        if (success) {
          return loadFromStorage().accessToken;
        }
        return null;
      }

      // Start new refresh
      const refreshPromise = refreshTokenAction();
      refreshPromiseRef.current = refreshPromise;

      try {
        const success = await refreshPromise;
        if (success) {
          return loadFromStorage().accessToken;
        }
      } finally {
        refreshPromiseRef.current = null;
      }
    }

    return null;
  }, [loadFromStorage]);

  /**
   * Refresh the access token
   */
  const refreshTokenAction = useCallback(async (): Promise<boolean> => {
    const { refreshToken } = loadFromStorage();

    if (!refreshToken) {
      return false;
    }

    try {
      const response = await apiRequest<{
        access_token: string;
        refresh_token: string;
        token_type: string;
        expires_in: number;
        refresh_expires_in: number;
      }>(AUTH_ENDPOINTS.REFRESH, {
        method: "POST",
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      const tokens: AuthTokens = {
        access_token: response.access_token,
        refresh_token: response.refresh_token,
        token_type: "Bearer",
        expires_in: response.expires_in,
        refresh_expires_in: response.refresh_expires_in,
      };

      const { expiresAt, refreshExpiresAt } = saveTokens(tokens);

      setState((prev) => ({
        ...prev,
        tokenExpiresAt: expiresAt,
        refreshTokenExpiresAt: refreshExpiresAt,
      }));

      scheduleTokenRefresh(tokens.expires_in);

      return true;
    } catch (error) {
      // Refresh failed, user needs to re-authenticate
      clearStorage();
      setState((prev) => ({
        ...prev,
        user: null,
        isAuthenticated: false,
        tokenExpiresAt: null,
        refreshTokenExpiresAt: null,
        error: {
          message: "Session expired. Please log in again.",
          code: "SESSION_EXPIRED",
        },
      }));
      return false;
    }
  }, [loadFromStorage, saveTokens, clearStorage]);

  /**
   * Schedule automatic token refresh
   */
  const scheduleTokenRefresh = useCallback((expiresIn: number) => {
    // Clear existing timer
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
    }

    // Schedule refresh for 5 minutes before expiry
    const refreshTime = (expiresIn * 1000) - TOKEN_REFRESH_THRESHOLD_MS;

    if (refreshTime > 0) {
      refreshTimerRef.current = setTimeout(async () => {
        await refreshTokenAction();
      }, refreshTime);
    }
  }, [refreshTokenAction]);

  // ==========================================================================
  // Authentication Actions
  // ==========================================================================

  /**
   * Login with credentials
   */
  const login = useCallback(
    async (credentials: LoginCredentials): Promise<LoginResponse> => {
      console.log("[AuthProvider] login called with:", credentials.username);
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        // Check backend health before attempting login
        const isHealthy = await checkBackendHealth();
        if (!isHealthy) {
          const healthError: AuthError = {
            message: "Authentication service is currently unavailable. Please try again in a moment.",
            code: "SERVICE_UNAVAILABLE",
          };
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: healthError,
          }));
          throw healthError;
        }

        const response = await apiRequest<LoginResponse>(AUTH_ENDPOINTS.LOGIN, {
          method: "POST",
          body: JSON.stringify(credentials),
        });

        // Check if email verification is required
        if (response.requires_verification) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: {
              message: "Please verify your email before logging in.",
              code: "EMAIL_NOT_VERIFIED",
              requires_verification: true,
            },
          }));
          throw {
            message: "Please verify your email before logging in.",
            code: "EMAIL_NOT_VERIFIED",
            requires_verification: true,
          } as AuthError;
        }

        const tokens: AuthTokens = {
          access_token: response.access_token,
          refresh_token: response.refresh_token,
          token_type: response.token_type,
          expires_in: response.expires_in,
          refresh_expires_in: response.refresh_expires_in,
        };

        const { expiresAt, refreshExpiresAt } = saveTokens(tokens);
        saveUser(response.user);

        console.log("[AuthProvider] Setting authenticated state...");
        setState({
          user: response.user,
          isAuthenticated: true,
          isLoading: false,
          isInitialized: true,
          error: null,
          tokenExpiresAt: expiresAt,
          refreshTokenExpiresAt: refreshExpiresAt,
        });

        scheduleTokenRefresh(response.expires_in);

        // Set auth state ready after all state updates are complete
        // This prevents race conditions during login redirect
        setTimeout(() => {
          setIsAuthStateReady(true);
        }, 0);

        return response;
      } catch (error) {
        const authError = error as AuthError;
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: authError,
        }));
        throw error;
      }
    },
    [saveTokens, saveUser, scheduleTokenRefresh]
  );

  /**
   * Logout and clear all auth state
   */
  const logout = useCallback(async (): Promise<void> => {
    // Clear refresh timer
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }

    // Try to call logout endpoint (best effort)
    try {
      const accessToken = storage.get(AUTH_STORAGE_KEYS.ACCESS_TOKEN);
      if (accessToken) {
        await apiRequest(AUTH_ENDPOINTS.LOGOUT, { method: "POST" }, accessToken);
      }
    } catch {
      // Ignore logout API errors
    }

    // Clear storage
    clearStorage();

    // Reset auth state ready flag
    setIsAuthStateReady(false);

    // Reset state
    setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      isInitialized: true,
      error: null,
      tokenExpiresAt: null,
      refreshTokenExpiresAt: null,
    });
  }, [clearStorage]);

  /**
   * Register a new user
   */
  const register = useCallback(
    async (data: RegistrationData): Promise<RegistrationResponse> => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        const response = await apiRequest<RegistrationResponse>(
          AUTH_ENDPOINTS.REGISTER,
          {
            method: "POST",
            body: JSON.stringify(data),
          }
        );

        setState((prev) => ({
          ...prev,
          isLoading: false,
        }));

        return response;
      } catch (error) {
        const authError = error as AuthError;
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: authError,
        }));
        throw error;
      }
    },
    []
  );

  /**
   * Fetch current user info
   */
  const fetchCurrentUser = useCallback(async (): Promise<AuthUser | null> => {
    const accessToken = await getAccessToken();
    if (!accessToken) {
      return null;
    }

    try {
      const user = await apiRequest<AuthUser>(
        AUTH_ENDPOINTS.ME,
        { method: "GET" },
        accessToken
      );

      saveUser(user);
      setState((prev) => ({
        ...prev,
        user,
      }));

      return user;
    } catch {
      return null;
    }
  }, [getAccessToken, saveUser]);

  // ==========================================================================
  // Password Utilities
  // ==========================================================================

  /**
   * Check password strength
   */
  const checkPasswordStrength = useCallback(
    async (password: string): Promise<PasswordStrengthResult> => {
      try {
        const response = await apiRequest<PasswordStrengthResult>(
          AUTH_ENDPOINTS.CHECK_PASSWORD_STRENGTH,
          {
            method: "POST",
            body: JSON.stringify({ password }),
          }
        );
        return response;
      } catch {
        return {
          is_valid: false,
          score: 0,
          errors: ["Unable to check password strength"],
          warnings: [],
          suggestions: [],
        };
      }
    },
    []
  );

  /**
   * Request password reset
   */
  const requestPasswordReset = useCallback(
    async (email: string): Promise<boolean> => {
      try {
        await apiRequest(AUTH_ENDPOINTS.FORGOT_PASSWORD, {
          method: "POST",
          body: JSON.stringify({ email }),
        });
        return true;
      } catch {
        // Always return true to prevent email enumeration
        return true;
      }
    },
    []
  );

  /**
   * Reset password with token
   */
  const resetPassword = useCallback(
    async (token: string, newPassword: string): Promise<boolean> => {
      try {
        await apiRequest(AUTH_ENDPOINTS.RESET_PASSWORD, {
          method: "POST",
          body: JSON.stringify({ token, new_password: newPassword }),
        });
        return true;
      } catch (error) {
        const authError = error as AuthError;
        setState((prev) => ({
          ...prev,
          error: authError,
        }));
        return false;
      }
    },
    []
  );

  // ==========================================================================
  // Email Verification
  // ==========================================================================

  /**
   * Resend verification email
   */
  const resendVerificationEmail = useCallback(
    async (email: string): Promise<boolean> => {
      try {
        await apiRequest(AUTH_ENDPOINTS.RESEND_VERIFICATION, {
          method: "POST",
          body: JSON.stringify({ email }),
        });
        return true;
      } catch {
        return false;
      }
    },
    []
  );

  /**
   * Verify email with token
   */
  const verifyEmail = useCallback(async (token: string): Promise<boolean> => {
    try {
      await apiRequest(`${AUTH_ENDPOINTS.VERIFY_EMAIL}/${token}`, {
        method: "GET",
      });
      return true;
    } catch {
      return false;
    }
  }, []);

  // ==========================================================================
  // Role Helpers
  // ==========================================================================

  /**
   * Check if user has a specific role
   */
  const hasRole = useCallback(
    (role: UserRole): boolean => {
      if (!state.user) return false;
      return state.user.role === role;
    },
    [state.user]
  );

  /**
   * Check if user has any of the specified roles
   */
  const hasAnyRole = useCallback(
    (roles: UserRole[]): boolean => {
      if (!state.user) return false;
      return roles.includes(state.user.role);
    },
    [state.user]
  );

  /**
   * Check if user is admin
   */
  const isAdmin = useCallback((): boolean => {
    return hasRole("admin");
  }, [hasRole]);

  /**
   * Check if user is researcher
   */
  const isResearcher = useCallback((): boolean => {
    return hasAnyRole(["admin", "researcher"]);
  }, [hasAnyRole]);

  /**
   * Check if user is viewer (any authenticated user)
   */
  const isViewer = useCallback((): boolean => {
    return state.isAuthenticated;
  }, [state.isAuthenticated]);

  // ==========================================================================
  // Error Handling
  // ==========================================================================

  /**
   * Clear current error
   */
  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  // ==========================================================================
  // Initialization
  // ==========================================================================

  useEffect(() => {
    if (initializingRef.current || state.isInitialized) return;
    initializingRef.current = true;

    const initialize = async () => {
      try {
        const {
          user,
          accessToken,
          refreshToken,
          tokenExpiresAt,
          refreshTokenExpiresAt,
        } = loadFromStorage();

        const now = Date.now();

        // Check if we have valid tokens
        if (accessToken && tokenExpiresAt && now < tokenExpiresAt) {
          // Access token still valid
          setState({
            user,
            isAuthenticated: true,
            isLoading: false,
            isInitialized: true,
            error: null,
            tokenExpiresAt,
            refreshTokenExpiresAt,
          });

          // Set auth state ready for existing valid sessions
          setIsAuthStateReady(true);

          // Schedule refresh
          const remainingTime = Math.floor((tokenExpiresAt - now) / 1000);
          scheduleTokenRefresh(remainingTime);

          // Optionally refresh user data
          fetchCurrentUser().catch(() => { });
        } else if (
          refreshToken &&
          refreshTokenExpiresAt &&
          now < refreshTokenExpiresAt
        ) {
          // Access token expired but refresh token valid, try to refresh
          const success = await refreshTokenAction();
          if (success) {
            const newData = loadFromStorage();
            setState({
              user: newData.user,
              isAuthenticated: true,
              isLoading: false,
              isInitialized: true,
              error: null,
              tokenExpiresAt: newData.tokenExpiresAt,
              refreshTokenExpiresAt: newData.refreshTokenExpiresAt,
            });

            // Set auth state ready after successful refresh
            setIsAuthStateReady(true);
          } else {
            // Refresh failed
            clearStorage();
            setState({
              user: null,
              isAuthenticated: false,
              isLoading: false,
              isInitialized: true,
              error: null,
              tokenExpiresAt: null,
              refreshTokenExpiresAt: null,
            });
            setIsAuthStateReady(true);
          }
        } else {
          // No valid tokens
          clearStorage();
          setState({
            user: null,
            isAuthenticated: false,
            isLoading: false,
            isInitialized: true,
            error: null,
            tokenExpiresAt: null,
            refreshTokenExpiresAt: null,
          });
          setIsAuthStateReady(true);
        }
      } catch (error) {
        setState({
          user: null,
          isAuthenticated: false,
          isLoading: false,
          isInitialized: true,
          error: {
            message: "Failed to initialize authentication",
            code: "INIT_FAILED",
          },
          tokenExpiresAt: null,
          refreshTokenExpiresAt: null,
        });
        setIsAuthStateReady(true);
      }
    };

    initialize();
  }, [
    state.isInitialized,
    loadFromStorage,
    scheduleTokenRefresh,
    refreshTokenAction,
    fetchCurrentUser,
    clearStorage,
  ]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }
    };
  }, []);

  // Listen for storage changes (other tabs)
  useEffect(() => {
    if (typeof window === "undefined") return;

    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === AUTH_STORAGE_KEYS.ACCESS_TOKEN) {
        // Auth state changed in another tab
        const { user, tokenExpiresAt, refreshTokenExpiresAt } = loadFromStorage();
        if (event.newValue) {
          setState((prev) => ({
            ...prev,
            user,
            isAuthenticated: true,
            tokenExpiresAt,
            refreshTokenExpiresAt,
          }));
        } else {
          setState((prev) => ({
            ...prev,
            user: null,
            isAuthenticated: false,
            tokenExpiresAt: null,
            refreshTokenExpiresAt: null,
          }));
        }
      }
    };

    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, [loadFromStorage]);

  // ==========================================================================
  // Context Value
  // ==========================================================================

  const value = useMemo<AuthContextValue>(
    () => ({
      ...state,
      isAuthStateReady,
      login,
      logout,
      register,
      refreshToken: refreshTokenAction,
      getAccessToken,
      fetchCurrentUser,
      checkPasswordStrength,
      resendVerificationEmail,
      verifyEmail,
      requestPasswordReset,
      resetPassword,
      hasRole,
      hasAnyRole,
      isAdmin,
      isResearcher,
      isViewer,
      clearError,
    }),
    [
      state,
      isAuthStateReady,
      login,
      logout,
      register,
      refreshTokenAction,
      getAccessToken,
      fetchCurrentUser,
      checkPasswordStrength,
      resendVerificationEmail,
      verifyEmail,
      requestPasswordReset,
      resetPassword,
      hasRole,
      hasAnyRole,
      isAdmin,
      isResearcher,
      isViewer,
      clearError,
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export default AuthProvider;
