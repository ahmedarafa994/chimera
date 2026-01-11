/**
 * useAuth Hooks Tests
 *
 * Tests for all authentication convenience hooks.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import React from "react";

import {
  useAuth,
  useCurrentUser,
  useAuthStatus,
  useAuthActions,
  useRoles,
  useTokens,
  usePassword,
  useEmailVerification,
  useRequireRole,
  useMinimumRole,
} from "../useAuth";
import { AuthContext, type AuthContextValue, type AuthState, type UserRole } from "@/contexts/AuthContext";

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a mock auth context value
 */
function createMockAuthContextValue(
  overrides: Partial<AuthContextValue> = {}
): AuthContextValue {
  const defaultState: AuthState = {
    user: null,
    isAuthenticated: false,
    isLoading: false,
    isInitialized: true,
    error: null,
    tokenExpiresAt: null,
    refreshTokenExpiresAt: null,
  };

  const userRole = (overrides.user?.role as UserRole) || "viewer";

  return {
    ...defaultState,
    login: vi.fn().mockResolvedValue({}),
    logout: vi.fn().mockResolvedValue(undefined),
    register: vi.fn().mockResolvedValue({}),
    refreshToken: vi.fn().mockResolvedValue(true),
    getAccessToken: vi.fn().mockResolvedValue("test-token"),
    fetchCurrentUser: vi.fn().mockResolvedValue(null),
    checkPasswordStrength: vi.fn().mockResolvedValue({
      is_valid: true,
      score: 100,
      errors: [],
      warnings: [],
      suggestions: [],
    }),
    resendVerificationEmail: vi.fn().mockResolvedValue(true),
    verifyEmail: vi.fn().mockResolvedValue(true),
    requestPasswordReset: vi.fn().mockResolvedValue(true),
    resetPassword: vi.fn().mockResolvedValue(true),
    hasRole: (role: UserRole) => role === userRole,
    hasAnyRole: (roles: UserRole[]) => roles.includes(userRole),
    isAdmin: () => userRole === "admin",
    isResearcher: () => userRole === "admin" || userRole === "researcher",
    isViewer: () => overrides.isAuthenticated ?? false,
    clearError: vi.fn(),
    ...overrides,
  };
}

/**
 * Wrapper for hook testing
 */
function createWrapper(contextValue: AuthContextValue) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <AuthContext.Provider value={contextValue}>
        {children}
      </AuthContext.Provider>
    );
  };
}

// =============================================================================
// Tests
// =============================================================================

describe("useAuth", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return full auth context", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "test@example.com",
        username: "testuser",
        role: "admin",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.username).toBe("testuser");
    expect(typeof result.current.login).toBe("function");
    expect(typeof result.current.logout).toBe("function");
    expect(typeof result.current.register).toBe("function");
  });

  it("should throw error when used outside provider", () => {
    const consoleError = console.error;
    console.error = vi.fn();

    expect(() => {
      renderHook(() => useAuth());
    }).toThrow(/useAuthContext must be used within an AuthProvider/);

    console.error = consoleError;
  });
});

describe("useCurrentUser", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return user state", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isLoading: false,
      isInitialized: true,
      user: {
        id: "1",
        email: "test@example.com",
        username: "currentuser",
        role: "researcher",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useCurrentUser(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.user?.username).toBe("currentuser");
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.isInitialized).toBe(true);
  });

  it("should return null user when not authenticated", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      user: null,
    });

    const { result } = renderHook(() => useCurrentUser(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });
});

describe("useAuthStatus", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return authentication status", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isLoading: false,
      isInitialized: true,
    });

    const { result } = renderHook(() => useAuthStatus(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isInitialized).toBe(true);
  });

  it("should show loading status", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      isLoading: true,
      isInitialized: false,
    });

    const { result } = renderHook(() => useAuthStatus(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isInitialized).toBe(false);
  });
});

describe("useAuthActions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return auth actions", () => {
    const mockLogin = vi.fn();
    const mockLogout = vi.fn();
    const mockRegister = vi.fn();
    const mockClearError = vi.fn();

    const mockValue = createMockAuthContextValue({
      login: mockLogin,
      logout: mockLogout,
      register: mockRegister,
      clearError: mockClearError,
      isLoading: false,
      error: null,
    });

    const { result } = renderHook(() => useAuthActions(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.login).toBe(mockLogin);
    expect(result.current.logout).toBe(mockLogout);
    expect(result.current.register).toBe(mockRegister);
    expect(result.current.clearError).toBe(mockClearError);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("should return error state", () => {
    const mockValue = createMockAuthContextValue({
      error: { message: "Invalid credentials", code: "401" },
    });

    const { result } = renderHook(() => useAuthActions(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.error?.message).toBe("Invalid credentials");
  });
});

describe("useRoles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return role checking functions", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useRoles(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.user?.role).toBe("admin");
    expect(typeof result.current.hasRole).toBe("function");
    expect(typeof result.current.hasAnyRole).toBe("function");
    expect(typeof result.current.isAdmin).toBe("function");
    expect(typeof result.current.isResearcher).toBe("function");
    expect(typeof result.current.isViewer).toBe("function");
  });

  it("should correctly check admin role", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useRoles(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.isAdmin()).toBe(true);
    expect(result.current.hasRole("admin")).toBe(true);
    expect(result.current.hasAnyRole(["admin", "researcher"])).toBe(true);
  });

  it("should correctly check researcher role", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "researcher@example.com",
        username: "researcher",
        role: "researcher",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useRoles(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.isAdmin()).toBe(false);
    expect(result.current.isResearcher()).toBe(true);
    expect(result.current.hasRole("researcher")).toBe(true);
  });
});

describe("useTokens", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return token management functions", () => {
    const mockGetAccessToken = vi.fn().mockResolvedValue("access-token");
    const mockRefreshToken = vi.fn().mockResolvedValue(true);

    const mockValue = createMockAuthContextValue({
      getAccessToken: mockGetAccessToken,
      refreshToken: mockRefreshToken,
      tokenExpiresAt: Date.now() + 3600000,
      refreshTokenExpiresAt: Date.now() + 86400000,
    });

    const { result } = renderHook(() => useTokens(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.getAccessToken).toBe(mockGetAccessToken);
    expect(result.current.refreshToken).toBe(mockRefreshToken);
    expect(result.current.tokenExpiresAt).toBeGreaterThan(Date.now());
    expect(result.current.refreshTokenExpiresAt).toBeGreaterThan(Date.now());
  });

  it("should return null expiry when not authenticated", () => {
    const mockValue = createMockAuthContextValue({
      tokenExpiresAt: null,
      refreshTokenExpiresAt: null,
    });

    const { result } = renderHook(() => useTokens(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.tokenExpiresAt).toBeNull();
    expect(result.current.refreshTokenExpiresAt).toBeNull();
  });
});

describe("usePassword", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return password functions", () => {
    const mockCheckPasswordStrength = vi.fn().mockResolvedValue({
      is_valid: true,
      score: 85,
      errors: [],
      warnings: [],
      suggestions: [],
    });
    const mockRequestPasswordReset = vi.fn().mockResolvedValue(true);
    const mockResetPassword = vi.fn().mockResolvedValue(true);

    const mockValue = createMockAuthContextValue({
      checkPasswordStrength: mockCheckPasswordStrength,
      requestPasswordReset: mockRequestPasswordReset,
      resetPassword: mockResetPassword,
    });

    const { result } = renderHook(() => usePassword(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.checkPasswordStrength).toBe(mockCheckPasswordStrength);
    expect(result.current.requestPasswordReset).toBe(mockRequestPasswordReset);
    expect(result.current.resetPassword).toBe(mockResetPassword);
  });
});

describe("useEmailVerification", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return email verification functions", () => {
    const mockResendVerificationEmail = vi.fn().mockResolvedValue(true);
    const mockVerifyEmail = vi.fn().mockResolvedValue(true);

    const mockValue = createMockAuthContextValue({
      resendVerificationEmail: mockResendVerificationEmail,
      verifyEmail: mockVerifyEmail,
    });

    const { result } = renderHook(() => useEmailVerification(), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current.resendVerificationEmail).toBe(mockResendVerificationEmail);
    expect(result.current.verifyEmail).toBe(mockVerifyEmail);
  });
});

describe("useRequireRole", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return true when user has required role", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useRequireRole("admin"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(true);
  });

  it("should return false when user lacks required role", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "viewer@example.com",
        username: "viewer",
        role: "viewer",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useRequireRole("admin"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(false);
  });

  it("should return true when user has one of required roles (array)", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "researcher@example.com",
        username: "researcher",
        role: "researcher",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useRequireRole(["admin", "researcher"]), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(true);
  });

  it("should return false when not authenticated", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      user: null,
    });

    const { result } = renderHook(() => useRequireRole("admin"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(false);
  });
});

describe("useMinimumRole", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return true when user meets minimum role (exact match)", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "researcher@example.com",
        username: "researcher",
        role: "researcher",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useMinimumRole("researcher"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(true);
  });

  it("should return true when user exceeds minimum role", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useMinimumRole("researcher"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(true);
  });

  it("should return false when user is below minimum role", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "viewer@example.com",
        username: "viewer",
        role: "viewer",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useMinimumRole("researcher"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(false);
  });

  it("should return true for viewer minimum when authenticated", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "viewer@example.com",
        username: "viewer",
        role: "viewer",
        is_verified: true,
      },
    });

    const { result } = renderHook(() => useMinimumRole("viewer"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(true);
  });

  it("should return false when not authenticated", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      user: null,
    });

    const { result } = renderHook(() => useMinimumRole("viewer"), {
      wrapper: createWrapper(mockValue),
    });

    expect(result.current).toBe(false);
  });

  it("should validate role hierarchy correctly", () => {
    // Test that admin (level 3) > researcher (level 2) > viewer (level 1)
    const adminValue = createMockAuthContextValue({
      isAuthenticated: true,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    const { result: adminViewerCheck } = renderHook(() => useMinimumRole("viewer"), {
      wrapper: createWrapper(adminValue),
    });
    expect(adminViewerCheck.current).toBe(true);

    const { result: adminResearcherCheck } = renderHook(() => useMinimumRole("researcher"), {
      wrapper: createWrapper(adminValue),
    });
    expect(adminResearcherCheck.current).toBe(true);

    const { result: adminAdminCheck } = renderHook(() => useMinimumRole("admin"), {
      wrapper: createWrapper(adminValue),
    });
    expect(adminAdminCheck.current).toBe(true);
  });
});
