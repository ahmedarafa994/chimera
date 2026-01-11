/**
 * Authentication Hooks
 *
 * Provides convenient hooks for accessing authentication functionality.
 *
 * @module hooks/useAuth
 */

export {
  useAuthContext,
  type AuthUser,
  type AuthTokens,
  type LoginCredentials,
  type LoginResponse,
  type RegistrationData,
  type RegistrationResponse,
  type PasswordStrengthResult,
  type AuthError,
  type UserRole,
  type AuthState,
  type AuthContextValue,
  AUTH_STORAGE_KEYS,
  TOKEN_REFRESH_THRESHOLD_MS,
  ROLE_HIERARCHY,
} from "@/contexts/AuthContext";

import { useAuthContext, type UserRole } from "@/contexts/AuthContext";

// =============================================================================
// Main Hook
// =============================================================================

/**
 * Main authentication hook.
 *
 * Provides access to the full authentication context including:
 * - User state (user, isAuthenticated, isLoading)
 * - Auth actions (login, logout, register)
 * - Token management (refreshToken, getAccessToken)
 * - Role helpers (hasRole, isAdmin, isResearcher, isViewer)
 *
 * @example
 * ```tsx
 * const { user, isAuthenticated, login, logout } = useAuth();
 *
 * if (isAuthenticated) {
 *   return <div>Welcome, {user?.username}!</div>;
 * }
 * ```
 */
export function useAuth() {
  return useAuthContext();
}

// =============================================================================
// Convenience Hooks
// =============================================================================

/**
 * Hook for just the current user
 *
 * @example
 * ```tsx
 * const { user, isLoading } = useCurrentUser();
 * ```
 */
export function useCurrentUser() {
  const { user, isLoading, isAuthenticated, isInitialized } = useAuthContext();
  return { user, isLoading, isAuthenticated, isInitialized };
}

/**
 * Hook for authentication status only
 *
 * @example
 * ```tsx
 * const { isAuthenticated, isLoading } = useAuthStatus();
 * ```
 */
export function useAuthStatus() {
  const { isAuthenticated, isLoading, isInitialized } = useAuthContext();
  return { isAuthenticated, isLoading, isInitialized };
}

/**
 * Hook for login/logout actions
 *
 * @example
 * ```tsx
 * const { login, logout, isLoading, error } = useAuthActions();
 *
 * const handleLogin = async () => {
 *   try {
 *     await login({ username: 'user', password: 'pass' });
 *   } catch (e) {
 *     console.error(e);
 *   }
 * };
 * ```
 */
export function useAuthActions() {
  const { login, logout, register, isLoading, error, clearError } =
    useAuthContext();
  return { login, logout, register, isLoading, error, clearError };
}

/**
 * Hook for role-based access control
 *
 * @example
 * ```tsx
 * const { hasRole, isAdmin, isResearcher, isViewer } = useRoles();
 *
 * if (isAdmin()) {
 *   return <AdminPanel />;
 * }
 * ```
 */
export function useRoles() {
  const { user, hasRole, hasAnyRole, isAdmin, isResearcher, isViewer } =
    useAuthContext();
  return { user, hasRole, hasAnyRole, isAdmin, isResearcher, isViewer };
}

/**
 * Hook for token management
 *
 * @example
 * ```tsx
 * const { getAccessToken, refreshToken } = useTokens();
 *
 * const token = await getAccessToken();
 * ```
 */
export function useTokens() {
  const { getAccessToken, refreshToken, tokenExpiresAt, refreshTokenExpiresAt } =
    useAuthContext();
  return { getAccessToken, refreshToken, tokenExpiresAt, refreshTokenExpiresAt };
}

/**
 * Hook for password-related actions
 *
 * @example
 * ```tsx
 * const { checkPasswordStrength, requestPasswordReset, resetPassword } = usePassword();
 *
 * const result = await checkPasswordStrength('myPassword123!');
 * ```
 */
export function usePassword() {
  const { checkPasswordStrength, requestPasswordReset, resetPassword } =
    useAuthContext();
  return { checkPasswordStrength, requestPasswordReset, resetPassword };
}

/**
 * Hook for email verification
 *
 * @example
 * ```tsx
 * const { resendVerificationEmail, verifyEmail } = useEmailVerification();
 *
 * await resendVerificationEmail('user@example.com');
 * ```
 */
export function useEmailVerification() {
  const { resendVerificationEmail, verifyEmail } = useAuthContext();
  return { resendVerificationEmail, verifyEmail };
}

// =============================================================================
// Guard Hooks
// =============================================================================

/**
 * Hook that returns true if user has required role(s)
 *
 * @param requiredRoles - Single role or array of roles (any match passes)
 *
 * @example
 * ```tsx
 * const canAccess = useRequireRole('admin');
 * const canAccessEither = useRequireRole(['admin', 'researcher']);
 * ```
 */
export function useRequireRole(requiredRoles: UserRole | UserRole[]): boolean {
  const { user, isAuthenticated } = useAuthContext();

  if (!isAuthenticated || !user) {
    return false;
  }

  const roles = Array.isArray(requiredRoles) ? requiredRoles : [requiredRoles];
  return roles.includes(user.role);
}

/**
 * Hook that returns true if user has minimum role level
 *
 * Role hierarchy: viewer < researcher < admin
 *
 * @example
 * ```tsx
 * const canEdit = useMinimumRole('researcher'); // true for researcher and admin
 * ```
 */
export function useMinimumRole(minimumRole: UserRole): boolean {
  const { user, isAuthenticated } = useAuthContext();

  if (!isAuthenticated || !user) {
    return false;
  }

  const roleHierarchy: Record<UserRole, number> = {
    viewer: 1,
    researcher: 2,
    admin: 3,
  };

  return roleHierarchy[user.role] >= roleHierarchy[minimumRole];
}

// =============================================================================
// Default Export
// =============================================================================

export default useAuth;
