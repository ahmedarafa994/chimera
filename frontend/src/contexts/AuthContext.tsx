"use client";

/**
 * Authentication Context
 *
 * Provides authentication state and types for the application.
 * This context manages user authentication state, JWT tokens,
 * and role-based access control.
 *
 * @module contexts/AuthContext
 */

import { createContext, useContext } from "react";

// =============================================================================
// Types
// =============================================================================

/**
 * User role enum matching backend UserRole
 */
export type UserRole = "admin" | "researcher" | "viewer";

/**
 * User profile information returned from authentication
 */
export interface AuthUser {
  id: string;
  email: string;
  username: string;
  role: UserRole;
  is_verified: boolean;
  is_active?: boolean;
  created_at?: string;
  last_login?: string;
}

/**
 * Authentication tokens from login/refresh
 */
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: "Bearer";
  expires_in: number;
  refresh_expires_in: number;
}

/**
 * Login credentials
 */
export interface LoginCredentials {
  username: string;
  password: string;
}

/**
 * Registration data
 */
export interface RegistrationData {
  email: string;
  username: string;
  password: string;
}

/**
 * Login response from backend
 */
export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: "Bearer";
  expires_in: number;
  refresh_expires_in: number;
  user: AuthUser;
  requires_verification: boolean;
}

/**
 * Registration response from backend
 */
export interface RegistrationResponse {
  success: boolean;
  message: string;
  user: {
    id: string;
    email: string;
    username: string;
    role: UserRole;
    is_verified: boolean;
  };
  requires_email_verification: boolean;
}

/**
 * Password strength check result
 */
export interface PasswordStrengthResult {
  is_valid: boolean;
  score: number;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

/**
 * Authentication error with additional context
 */
export interface AuthError {
  message: string;
  code?: string;
  requires_verification?: boolean;
  field_errors?: Record<string, string[]>;
}

/**
 * Authentication state
 */
export interface AuthState {
  // Current user (null if not authenticated)
  user: AuthUser | null;

  // Authentication status
  isAuthenticated: boolean;
  isLoading: boolean;
  isInitialized: boolean;

  // Error state
  error: AuthError | null;

  // Token expiry tracking
  tokenExpiresAt: number | null;
  refreshTokenExpiresAt: number | null;
}

/**
 * Authentication context value including state and actions
 */
export interface AuthContextValue extends AuthState {
  // Auth state readiness flag for preventing race conditions
  isAuthStateReady: boolean;

  // Authentication actions
  login: (credentials: LoginCredentials) => Promise<LoginResponse>;
  logout: () => Promise<void>;
  register: (data: RegistrationData) => Promise<RegistrationResponse>;

  // Token management
  refreshToken: () => Promise<boolean>;
  getAccessToken: () => Promise<string | null>;

  // User info
  fetchCurrentUser: () => Promise<AuthUser | null>;

  // Password utilities
  checkPasswordStrength: (password: string) => Promise<PasswordStrengthResult>;

  // Email verification
  resendVerificationEmail: (email: string) => Promise<boolean>;
  verifyEmail: (token: string) => Promise<boolean>;

  // Password reset
  requestPasswordReset: (email: string) => Promise<boolean>;
  resetPassword: (token: string, newPassword: string) => Promise<boolean>;

  // Role-based access helpers
  hasRole: (role: UserRole) => boolean;
  hasAnyRole: (roles: UserRole[]) => boolean;
  isAdmin: () => boolean;
  isResearcher: () => boolean;
  isViewer: () => boolean;

  // Error handling
  clearError: () => void;
}

// =============================================================================
// Context
// =============================================================================

/**
 * Authentication context (null by default, must be used within AuthProvider)
 */
export const AuthContext = createContext<AuthContextValue | null>(null);

// =============================================================================
// Hook
// =============================================================================

/**
 * Hook to access authentication context
 *
 * @throws Error if used outside of AuthProvider
 */
export function useAuthContext(): AuthContextValue {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error(
      "useAuthContext must be used within an AuthProvider. " +
        "Make sure AuthProvider is in your component tree (usually in layout.tsx)."
    );
  }

  return context;
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Storage keys for auth state persistence
 */
export const AUTH_STORAGE_KEYS = {
  ACCESS_TOKEN: "chimera_access_token",
  REFRESH_TOKEN: "chimera_refresh_token",
  USER: "chimera_auth_user",
  TOKEN_EXPIRY: "chimera_token_expiry",
  REFRESH_EXPIRY: "chimera_refresh_expiry",
} as const;

/**
 * Token refresh threshold (refresh 5 minutes before expiry)
 */
export const TOKEN_REFRESH_THRESHOLD_MS = 5 * 60 * 1000;

/**
 * Role hierarchy for permission checks
 */
export const ROLE_HIERARCHY: Record<UserRole, number> = {
  admin: 3,
  researcher: 2,
  viewer: 1,
};

export default AuthContext;
