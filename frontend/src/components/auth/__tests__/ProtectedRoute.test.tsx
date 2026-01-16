/**
 * ProtectedRoute Component Tests
 *
 * Tests for the protected route wrapper component including
 * authentication checks, redirects, and loading states.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import React from "react";

import { ProtectedRoute, withProtectedRoute, useProtectedPage } from "../ProtectedRoute";
import { AuthContext, type AuthContextValue, type AuthState } from "@/contexts/AuthContext";

// =============================================================================
// Mock Next.js router
// =============================================================================

const mockReplace = vi.fn();
const mockPathname = vi.fn().mockReturnValue("/protected-page");

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: mockReplace,
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
  usePathname: () => mockPathname(),
  useSearchParams: () => new URLSearchParams(),
}));

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
 * Render ProtectedRoute with mock provider
 */
function renderProtectedRoute(
  children: React.ReactNode,
  props: Partial<React.ComponentProps<typeof ProtectedRoute>> = {},
  contextOverrides: Partial<AuthContextValue> = {}
) {
  const mockValue = createMockAuthContextValue(contextOverrides);

  const result = render(
    <AuthContext.Provider value={mockValue}>
      <ProtectedRoute {...props}>{children}</ProtectedRoute>
    </AuthContext.Provider>
  );

  return {
    ...result,
    mockValue,
  };
}

// =============================================================================
// Tests
// =============================================================================

describe("ProtectedRoute", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPathname.mockReturnValue("/protected-page");
  });

  describe("Authentication Check", () => {
    it("should render children when user is authenticated", () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {},
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "test@example.com",
            username: "testuser",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
      expect(screen.getByText("Protected Content")).toBeInTheDocument();
    });

    it("should redirect to login when user is not authenticated", async () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {},
        {
          isAuthenticated: false,
          isInitialized: true,
          isLoading: false,
        }
      );

      await waitFor(() => {
        expect(mockReplace).toHaveBeenCalledWith(
          "/login?redirect=%2Fprotected-page"
        );
      });
    });

    it("should not render children when redirecting", () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {},
        {
          isAuthenticated: false,
          isInitialized: true,
          isLoading: false,
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
    });

    it("should preserve return URL in redirect", async () => {
      mockPathname.mockReturnValue("/dashboard/settings/profile");

      renderProtectedRoute(
        <div>Content</div>,
        {},
        {
          isAuthenticated: false,
          isInitialized: true,
          isLoading: false,
        }
      );

      await waitFor(() => {
        expect(mockReplace).toHaveBeenCalledWith(
          expect.stringContaining("redirect=%2Fdashboard%2Fsettings%2Fprofile")
        );
      });
    });
  });

  describe("Loading State", () => {
    it("should show loading state while initializing", () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {},
        {
          isAuthenticated: false,
          isInitialized: false,
          isLoading: true,
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
      expect(screen.getByText(/verifying authentication/i)).toBeInTheDocument();
    });

    it("should show loading state while loading", () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {},
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: true,
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
    });

    it("should show custom loading component when provided", () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {
          loadingComponent: <div data-testid="custom-loader">Custom Loading...</div>,
        },
        {
          isAuthenticated: false,
          isInitialized: false,
          isLoading: true,
        }
      );

      expect(screen.getByTestId("custom-loader")).toBeInTheDocument();
      expect(screen.queryByText(/verifying authentication/i)).not.toBeInTheDocument();
    });

    it("should show custom loading message when provided", () => {
      renderProtectedRoute(
        <div data-testid="protected-content">Protected Content</div>,
        {
          loadingMessage: "Please wait, checking access...",
        },
        {
          isAuthenticated: false,
          isInitialized: false,
          isLoading: true,
        }
      );

      expect(screen.getByText("Please wait, checking access...")).toBeInTheDocument();
    });
  });

  describe("Custom Redirect", () => {
    it("should redirect to custom URL when redirectTo prop is provided", async () => {
      renderProtectedRoute(
        <div>Content</div>,
        { redirectTo: "/auth/signin" },
        {
          isAuthenticated: false,
          isInitialized: true,
          isLoading: false,
        }
      );

      await waitFor(() => {
        expect(mockReplace).toHaveBeenCalledWith(
          expect.stringContaining("/auth/signin")
        );
      });
    });
  });

  describe("Redirect Callback", () => {
    it("should call onRedirect callback when redirecting", async () => {
      const onRedirect = vi.fn();

      renderProtectedRoute(
        <div>Content</div>,
        { onRedirect },
        {
          isAuthenticated: false,
          isInitialized: true,
          isLoading: false,
        }
      );

      await waitFor(() => {
        expect(onRedirect).toHaveBeenCalled();
      });
    });

    it("should not call onRedirect when user is authenticated", () => {
      const onRedirect = vi.fn();

      renderProtectedRoute(
        <div>Content</div>,
        { onRedirect },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "test@example.com",
            username: "testuser",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      expect(onRedirect).not.toHaveBeenCalled();
    });
  });
});

describe("withProtectedRoute HOC", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should wrap component with ProtectedRoute", () => {
    const TestComponent = () => <div data-testid="wrapped">Wrapped Content</div>;
    const WrappedComponent = withProtectedRoute(TestComponent);

    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "test@example.com",
        username: "testuser",
        role: "viewer",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <WrappedComponent />
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("wrapped")).toBeInTheDocument();
  });

  it("should set correct displayName", () => {
    const TestComponent = () => <div>Test</div>;
    TestComponent.displayName = "MyTestComponent";

    const WrappedComponent = withProtectedRoute(TestComponent);

    expect(WrappedComponent.displayName).toBe("withProtectedRoute(MyTestComponent)");
  });

  it("should use component name when displayName not set", () => {
    function NamedComponent() {
      return <div>Test</div>;
    }

    const WrappedComponent = withProtectedRoute(NamedComponent);

    expect(WrappedComponent.displayName).toBe("withProtectedRoute(NamedComponent)");
  });

  it("should pass options to ProtectedRoute", async () => {
    const TestComponent = () => <div>Content</div>;
    const WrappedComponent = withProtectedRoute(TestComponent, {
      redirectTo: "/custom-login",
      loadingMessage: "Custom loading...",
    });

    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      isInitialized: true,
      isLoading: false,
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <WrappedComponent />
      </AuthContext.Provider>
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith(
        expect.stringContaining("/custom-login")
      );
    });
  });
});

describe("useProtectedPage Hook", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  /**
   * Test component for useProtectedPage hook
   */
  function TestHookComponent({ redirectTo }: { redirectTo?: string }) {
    const { isChecking, isProtected } = useProtectedPage(redirectTo);
    return (
      <div>
        <div data-testid="is-checking">{isChecking ? "yes" : "no"}</div>
        <div data-testid="is-protected">{isProtected ? "yes" : "no"}</div>
      </div>
    );
  }

  it("should return isChecking=true while loading", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      isInitialized: false,
      isLoading: true,
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <TestHookComponent />
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("is-checking").textContent).toBe("yes");
  });

  it("should return isProtected=true when authenticated", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "test@example.com",
        username: "testuser",
        role: "viewer",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <TestHookComponent />
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("is-protected").textContent).toBe("yes");
    expect(screen.getByTestId("is-checking").textContent).toBe("no");
  });

  it("should return isProtected=false and redirect when not authenticated", async () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      isInitialized: true,
      isLoading: false,
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <TestHookComponent />
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("is-protected").textContent).toBe("no");

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalled();
    });
  });

  it("should redirect to custom URL when provided", async () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: false,
      isInitialized: true,
      isLoading: false,
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <TestHookComponent redirectTo="/auth/signin" />
      </AuthContext.Provider>
    );

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith(
        expect.stringContaining("/auth/signin")
      );
    });
  });
});
