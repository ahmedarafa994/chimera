/**
 * RoleGuard Component Tests
 *
 * Tests for the role-based access control component including
 * role verification, access denied states, and convenience components.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";

import {
  RoleGuard,
  AccessDenied,
  withRoleGuard,
  AdminOnly,
  ResearcherOnly,
} from "../RoleGuard";
import { AuthContext, type AuthContextValue, type AuthState, type UserRole } from "@/contexts/AuthContext";

// =============================================================================
// Mock Next.js router
// =============================================================================

const mockReplace = vi.fn();
const mockBack = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: mockReplace,
    prefetch: vi.fn(),
    back: mockBack,
  }),
  usePathname: () => "/admin",
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

  // Default user role for role checks
  const userRole: UserRole = (overrides.user?.role as UserRole) || "viewer";

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
    isViewer: () => true,
    clearError: vi.fn(),
    ...overrides,
  };
}

/**
 * Render RoleGuard with mock provider
 */
function renderRoleGuard(
  children: React.ReactNode,
  props: Partial<React.ComponentProps<typeof RoleGuard>> = {},
  contextOverrides: Partial<AuthContextValue> = {}
) {
  const mockValue = createMockAuthContextValue(contextOverrides);

  const result = render(
    <AuthContext.Provider value={mockValue}>
      <RoleGuard {...props}>{children}</RoleGuard>
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

describe("RoleGuard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Role-based Access", () => {
    it("should render children when user has required role", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Admin Content</div>,
        { allowedRoles: ["admin"] },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "admin@example.com",
            username: "admin",
            role: "admin",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    });

    it("should render children when user has one of allowed roles", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Research Content</div>,
        { allowedRoles: ["admin", "researcher"] },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "researcher@example.com",
            username: "researcher",
            role: "researcher",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    });

    it("should show access denied when user lacks required role", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Admin Content</div>,
        { allowedRoles: ["admin"] },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "viewer@example.com",
            username: "viewer",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
      expect(screen.getByText(/access denied/i)).toBeInTheDocument();
    });

    it("should not render when user is not authenticated", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Content</div>,
        { allowedRoles: ["admin"] },
        {
          isAuthenticated: false,
          isInitialized: true,
          isLoading: false,
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
      // Should not show access denied since user is not authenticated
      expect(screen.queryByText(/access denied/i)).not.toBeInTheDocument();
    });
  });

  describe("Minimum Role Hierarchy", () => {
    it("should render children when user meets minimum role (admin >= researcher)", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Researcher Content</div>,
        { minimumRole: "researcher" },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "admin@example.com",
            username: "admin",
            role: "admin",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    });

    it("should render children when user is exact minimum role", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Researcher Content</div>,
        { minimumRole: "researcher" },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "researcher@example.com",
            username: "researcher",
            role: "researcher",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    });

    it("should show access denied when user is below minimum role", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Researcher Content</div>,
        { minimumRole: "researcher" },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "viewer@example.com",
            username: "viewer",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
      expect(screen.getByText(/access denied/i)).toBeInTheDocument();
    });

    it("should render children for any authenticated user with minimumRole=viewer", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Viewer Content</div>,
        { minimumRole: "viewer" },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "viewer@example.com",
            username: "viewer",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    });
  });

  describe("Loading State", () => {
    it("should show loading state while initializing", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Content</div>,
        { allowedRoles: ["admin"] },
        {
          isAuthenticated: false,
          isInitialized: false,
          isLoading: true,
        }
      );

      expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
      expect(screen.getByText(/checking permissions/i)).toBeInTheDocument();
    });

    it("should show custom loading component when provided", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Content</div>,
        {
          allowedRoles: ["admin"],
          loadingComponent: <div data-testid="custom-loader">Loading roles...</div>,
        },
        {
          isAuthenticated: false,
          isInitialized: false,
          isLoading: true,
        }
      );

      expect(screen.getByTestId("custom-loader")).toBeInTheDocument();
    });

    it("should hide loading when showLoading is false", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Content</div>,
        { allowedRoles: ["admin"], showLoading: false },
        {
          isAuthenticated: false,
          isInitialized: false,
          isLoading: true,
        }
      );

      expect(screen.queryByText(/checking permissions/i)).not.toBeInTheDocument();
    });
  });

  describe("Redirect on Access Denied", () => {
    it("should redirect when redirectTo is provided", async () => {
      renderRoleGuard(
        <div data-testid="protected-content">Admin Content</div>,
        { allowedRoles: ["admin"], redirectTo: "/dashboard" },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "viewer@example.com",
            username: "viewer",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      await waitFor(() => {
        expect(mockReplace).toHaveBeenCalledWith("/dashboard");
      });
    });

    it("should call onAccessDenied callback", async () => {
      const onAccessDenied = vi.fn();

      renderRoleGuard(
        <div data-testid="protected-content">Admin Content</div>,
        { allowedRoles: ["admin"], onAccessDenied },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "viewer@example.com",
            username: "viewer",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      await waitFor(() => {
        expect(onAccessDenied).toHaveBeenCalled();
      });
    });
  });

  describe("Custom Fallback", () => {
    it("should render custom fallback when access denied", () => {
      renderRoleGuard(
        <div data-testid="protected-content">Admin Content</div>,
        {
          allowedRoles: ["admin"],
          fallback: <div data-testid="custom-fallback">You need admin access</div>,
        },
        {
          isAuthenticated: true,
          isInitialized: true,
          isLoading: false,
          user: {
            id: "1",
            email: "viewer@example.com",
            username: "viewer",
            role: "viewer",
            is_verified: true,
          },
        }
      );

      expect(screen.getByTestId("custom-fallback")).toBeInTheDocument();
      expect(screen.queryByText(/access denied/i)).not.toBeInTheDocument();
    });
  });
});

describe("AccessDenied", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render with default title and description", () => {
    render(<AccessDenied />);

    expect(screen.getByText("Access Denied")).toBeInTheDocument();
    expect(screen.getByText(/don't have permission/i)).toBeInTheDocument();
  });

  it("should render custom title and description", () => {
    render(
      <AccessDenied
        title="Admin Required"
        description="This section requires administrator access."
      />
    );

    expect(screen.getByText("Admin Required")).toBeInTheDocument();
    expect(screen.getByText(/requires administrator access/i)).toBeInTheDocument();
  });

  it("should display role information", () => {
    render(<AccessDenied requiredRole="admin" currentRole="viewer" />);

    expect(screen.getByText(/your role/i)).toBeInTheDocument();
    expect(screen.getByText(/viewer/i)).toBeInTheDocument();
    expect(screen.getByText(/admin or higher/i)).toBeInTheDocument();
  });

  it("should call router.back when Go Back button is clicked", async () => {
    const user = userEvent.setup();
    render(<AccessDenied showBackButton />);

    const backButton = screen.getByRole("button", { name: /go back/i });
    await user.click(backButton);

    expect(mockBack).toHaveBeenCalled();
  });

  it("should have link to dashboard", () => {
    render(<AccessDenied />);

    const dashboardLink = screen.getByRole("link", { name: /return to dashboard/i });
    expect(dashboardLink).toHaveAttribute("href", "/dashboard");
  });

  it("should render custom action", () => {
    render(
      <AccessDenied
        action={<button data-testid="custom-action">Request Access</button>}
      />
    );

    expect(screen.getByTestId("custom-action")).toBeInTheDocument();
  });
});

describe("withRoleGuard HOC", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should wrap component with RoleGuard", () => {
    const TestComponent = () => <div data-testid="wrapped">Admin Content</div>;
    const WrappedComponent = withRoleGuard(TestComponent, {
      allowedRoles: ["admin"],
    });

    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
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
    TestComponent.displayName = "AdminPanel";

    const WrappedComponent = withRoleGuard(TestComponent, {
      allowedRoles: ["admin"],
    });

    expect(WrappedComponent.displayName).toBe("withRoleGuard(AdminPanel)");
  });

  it("should deny access when role requirement not met", () => {
    const TestComponent = () => <div data-testid="wrapped">Admin Content</div>;
    const WrappedComponent = withRoleGuard(TestComponent, {
      allowedRoles: ["admin"],
    });

    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "viewer@example.com",
        username: "viewer",
        role: "viewer",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <WrappedComponent />
      </AuthContext.Provider>
    );

    expect(screen.queryByTestId("wrapped")).not.toBeInTheDocument();
    expect(screen.getByText(/access denied/i)).toBeInTheDocument();
  });
});

describe("AdminOnly", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render children for admin users", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <AdminOnly>
          <div data-testid="admin-content">Admin Only</div>
        </AdminOnly>
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("admin-content")).toBeInTheDocument();
  });

  it("should show access denied for non-admin users", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "researcher@example.com",
        username: "researcher",
        role: "researcher",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <AdminOnly>
          <div data-testid="admin-content">Admin Only</div>
        </AdminOnly>
      </AuthContext.Provider>
    );

    expect(screen.queryByTestId("admin-content")).not.toBeInTheDocument();
    expect(screen.getByText(/access denied/i)).toBeInTheDocument();
  });

  it("should support custom fallback", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "viewer@example.com",
        username: "viewer",
        role: "viewer",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <AdminOnly fallback={<div data-testid="fallback">Upgrade Required</div>}>
          <div data-testid="admin-content">Admin Only</div>
        </AdminOnly>
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("fallback")).toBeInTheDocument();
  });
});

describe("ResearcherOnly", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render children for researcher users", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "researcher@example.com",
        username: "researcher",
        role: "researcher",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <ResearcherOnly>
          <div data-testid="research-content">Research Tools</div>
        </ResearcherOnly>
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("research-content")).toBeInTheDocument();
  });

  it("should render children for admin users (higher role)", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "admin@example.com",
        username: "admin",
        role: "admin",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <ResearcherOnly>
          <div data-testid="research-content">Research Tools</div>
        </ResearcherOnly>
      </AuthContext.Provider>
    );

    expect(screen.getByTestId("research-content")).toBeInTheDocument();
  });

  it("should show access denied for viewer users", () => {
    const mockValue = createMockAuthContextValue({
      isAuthenticated: true,
      isInitialized: true,
      isLoading: false,
      user: {
        id: "1",
        email: "viewer@example.com",
        username: "viewer",
        role: "viewer",
        is_verified: true,
      },
    });

    render(
      <AuthContext.Provider value={mockValue}>
        <ResearcherOnly>
          <div data-testid="research-content">Research Tools</div>
        </ResearcherOnly>
      </AuthContext.Provider>
    );

    expect(screen.queryByTestId("research-content")).not.toBeInTheDocument();
    expect(screen.getByText(/access denied/i)).toBeInTheDocument();
  });
});
