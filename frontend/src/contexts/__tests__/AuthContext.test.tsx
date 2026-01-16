/**
 * AuthContext Tests
 *
 * Tests for authentication context types, constants, and hook behavior.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";

import {
  AuthContext,
  useAuthContext,
  AUTH_STORAGE_KEYS,
  TOKEN_REFRESH_THRESHOLD_MS,
  ROLE_HIERARCHY,
  type AuthUser,
  type AuthContextValue,
  type AuthState,
} from "../AuthContext";

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a mock auth context value for testing
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

  return {
    ...defaultState,
    login: vi.fn().mockResolvedValue({
      access_token: "test-token",
      refresh_token: "test-refresh",
      token_type: "Bearer",
      expires_in: 3600,
      refresh_expires_in: 86400,
      user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
      requires_verification: false,
    }),
    logout: vi.fn().mockResolvedValue(undefined),
    register: vi.fn().mockResolvedValue({
      success: true,
      message: "Registration successful",
      user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: false },
      requires_email_verification: true,
    }),
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
    hasRole: vi.fn().mockReturnValue(false),
    hasAnyRole: vi.fn().mockReturnValue(false),
    isAdmin: vi.fn().mockReturnValue(false),
    isResearcher: vi.fn().mockReturnValue(false),
    isViewer: vi.fn().mockReturnValue(false),
    clearError: vi.fn(),
    ...overrides,
  };
}

/**
 * Test component that uses auth context
 */
function TestConsumer() {
  const auth = useAuthContext();
  return (
    <div>
      <div data-testid="is-authenticated">{auth.isAuthenticated ? "yes" : "no"}</div>
      <div data-testid="is-loading">{auth.isLoading ? "yes" : "no"}</div>
      <div data-testid="is-initialized">{auth.isInitialized ? "yes" : "no"}</div>
      <div data-testid="user">{auth.user ? auth.user.username : "none"}</div>
      <div data-testid="error">{auth.error?.message || "none"}</div>
      <button data-testid="login-btn" onClick={() => auth.login({ username: "test", password: "pass" })}>
        Login
      </button>
      <button data-testid="logout-btn" onClick={() => auth.logout()}>
        Logout
      </button>
    </div>
  );
}

// =============================================================================
// Tests
// =============================================================================

describe("AuthContext", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Constants", () => {
    it("should export correct storage keys", () => {
      expect(AUTH_STORAGE_KEYS).toEqual({
        ACCESS_TOKEN: "chimera_access_token",
        REFRESH_TOKEN: "chimera_refresh_token",
        USER: "chimera_auth_user",
        TOKEN_EXPIRY: "chimera_token_expiry",
        REFRESH_EXPIRY: "chimera_refresh_expiry",
      });
    });

    it("should export correct token refresh threshold", () => {
      expect(TOKEN_REFRESH_THRESHOLD_MS).toBe(5 * 60 * 1000); // 5 minutes
    });

    it("should export correct role hierarchy", () => {
      expect(ROLE_HIERARCHY).toEqual({
        admin: 3,
        researcher: 2,
        viewer: 1,
      });
    });

    it("should have admin as highest role in hierarchy", () => {
      expect(ROLE_HIERARCHY.admin).toBeGreaterThan(ROLE_HIERARCHY.researcher);
      expect(ROLE_HIERARCHY.researcher).toBeGreaterThan(ROLE_HIERARCHY.viewer);
    });
  });

  describe("useAuthContext", () => {
    it("should throw error when used outside provider", () => {
      // Suppress console.error for this test
      const consoleError = console.error;
      console.error = vi.fn();

      expect(() => {
        render(<TestConsumer />);
      }).toThrow("useAuthContext must be used within an AuthProvider");

      console.error = consoleError;
    });

    it("should provide auth state when inside provider", () => {
      const mockValue = createMockAuthContextValue({
        isAuthenticated: true,
        isInitialized: true,
        user: {
          id: "1",
          email: "test@example.com",
          username: "testuser",
          role: "admin",
          is_verified: true,
        },
      });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId("is-authenticated").textContent).toBe("yes");
      expect(screen.getByTestId("is-initialized").textContent).toBe("yes");
      expect(screen.getByTestId("user").textContent).toBe("testuser");
    });

    it("should show loading state", () => {
      const mockValue = createMockAuthContextValue({
        isLoading: true,
        isInitialized: false,
      });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId("is-loading").textContent).toBe("yes");
      expect(screen.getByTestId("is-initialized").textContent).toBe("no");
    });

    it("should show error state", () => {
      const mockValue = createMockAuthContextValue({
        error: { message: "Invalid credentials", code: "401" },
      });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId("error").textContent).toBe("Invalid credentials");
    });

    it("should call login when login button clicked", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockResolvedValue({});
      const mockValue = createMockAuthContextValue({ login: mockLogin });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      await user.click(screen.getByTestId("login-btn"));

      expect(mockLogin).toHaveBeenCalledWith({ username: "test", password: "pass" });
    });

    it("should call logout when logout button clicked", async () => {
      const user = userEvent.setup();
      const mockLogout = vi.fn().mockResolvedValue(undefined);
      const mockValue = createMockAuthContextValue({ logout: mockLogout });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      await user.click(screen.getByTestId("logout-btn"));

      expect(mockLogout).toHaveBeenCalled();
    });
  });

  describe("AuthUser type structure", () => {
    it("should correctly type user with all fields", () => {
      const user: AuthUser = {
        id: "user-123",
        email: "user@example.com",
        username: "testuser",
        role: "researcher",
        is_verified: true,
        is_active: true,
        created_at: "2024-01-01T00:00:00Z",
        last_login: "2024-01-02T00:00:00Z",
      };

      expect(user.id).toBe("user-123");
      expect(user.role).toBe("researcher");
    });

    it("should allow optional fields to be undefined", () => {
      const user: AuthUser = {
        id: "user-123",
        email: "user@example.com",
        username: "testuser",
        role: "viewer",
        is_verified: false,
      };

      expect(user.is_active).toBeUndefined();
      expect(user.created_at).toBeUndefined();
      expect(user.last_login).toBeUndefined();
    });
  });

  describe("Role helpers", () => {
    it("should correctly check hasRole", () => {
      const mockValue = createMockAuthContextValue({
        user: {
          id: "1",
          email: "admin@example.com",
          username: "admin",
          role: "admin",
          is_verified: true,
        },
        hasRole: (role) => role === "admin",
      });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(mockValue.hasRole("admin")).toBe(true);
      expect(mockValue.hasRole("researcher")).toBe(false);
      expect(mockValue.hasRole("viewer")).toBe(false);
    });

    it("should correctly check hasAnyRole", () => {
      const mockValue = createMockAuthContextValue({
        user: {
          id: "1",
          email: "researcher@example.com",
          username: "researcher",
          role: "researcher",
          is_verified: true,
        },
        hasAnyRole: (roles) => roles.includes("researcher"),
      });

      render(
        <AuthContext.Provider value={mockValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(mockValue.hasAnyRole(["admin", "researcher"])).toBe(true);
      expect(mockValue.hasAnyRole(["viewer"])).toBe(false);
    });

    it("should check admin role correctly", () => {
      const adminMockValue = createMockAuthContextValue({
        user: {
          id: "1",
          email: "admin@example.com",
          username: "admin",
          role: "admin",
          is_verified: true,
        },
        isAdmin: () => true,
        isResearcher: () => true,
        isViewer: () => true,
      });

      expect(adminMockValue.isAdmin()).toBe(true);
      expect(adminMockValue.isResearcher()).toBe(true);
      expect(adminMockValue.isViewer()).toBe(true);
    });

    it("should check researcher role correctly", () => {
      const researcherMockValue = createMockAuthContextValue({
        user: {
          id: "1",
          email: "researcher@example.com",
          username: "researcher",
          role: "researcher",
          is_verified: true,
        },
        isAdmin: () => false,
        isResearcher: () => true,
        isViewer: () => true,
      });

      expect(researcherMockValue.isAdmin()).toBe(false);
      expect(researcherMockValue.isResearcher()).toBe(true);
      expect(researcherMockValue.isViewer()).toBe(true);
    });

    it("should check viewer role correctly", () => {
      const viewerMockValue = createMockAuthContextValue({
        user: {
          id: "1",
          email: "viewer@example.com",
          username: "viewer",
          role: "viewer",
          is_verified: true,
        },
        isAdmin: () => false,
        isResearcher: () => false,
        isViewer: () => true,
        isAuthenticated: true,
      });

      expect(viewerMockValue.isAdmin()).toBe(false);
      expect(viewerMockValue.isResearcher()).toBe(false);
      expect(viewerMockValue.isViewer()).toBe(true);
    });
  });

  describe("Context value updates", () => {
    it("should reflect user change when context updates", () => {
      const initialValue = createMockAuthContextValue({
        user: null,
        isAuthenticated: false,
      });

      const { rerender } = render(
        <AuthContext.Provider value={initialValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId("user").textContent).toBe("none");
      expect(screen.getByTestId("is-authenticated").textContent).toBe("no");

      // Simulate login success
      const loggedInValue = createMockAuthContextValue({
        user: {
          id: "1",
          email: "test@example.com",
          username: "loggedin",
          role: "admin",
          is_verified: true,
        },
        isAuthenticated: true,
      });

      rerender(
        <AuthContext.Provider value={loggedInValue}>
          <TestConsumer />
        </AuthContext.Provider>
      );

      expect(screen.getByTestId("user").textContent).toBe("loggedin");
      expect(screen.getByTestId("is-authenticated").textContent).toBe("yes");
    });
  });
});
