/**
 * LoginForm Component Tests
 *
 * Tests for the login form component including validation, error handling,
 * and user interactions.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";

import { LoginForm } from "../LoginForm";
import { AuthContext, type AuthContextValue, type AuthState } from "@/contexts/AuthContext";

// =============================================================================
// Mock Next.js router
// =============================================================================

const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
  }),
  usePathname: () => "/login",
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
 * Render LoginForm with mock provider
 */
function renderLoginForm(
  props: Partial<React.ComponentProps<typeof LoginForm>> = {},
  contextOverrides: Partial<AuthContextValue> = {}
) {
  const mockValue = createMockAuthContextValue(contextOverrides);

  const result = render(
    <AuthContext.Provider value={mockValue}>
      <LoginForm {...props} />
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

describe("LoginForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Clear localStorage
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe("Rendering", () => {
    it("should render login form with all fields", () => {
      renderLoginForm();

      expect(screen.getByLabelText(/email or username/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
      expect(screen.getByRole("checkbox", { name: /remember me/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
    });

    it("should have link to forgot password", () => {
      renderLoginForm();

      const forgotLink = screen.getByRole("link", { name: /forgot password/i });
      expect(forgotLink).toHaveAttribute("href", "/forgot-password");
    });

    it("should have link to register", () => {
      renderLoginForm();

      const registerLink = screen.getByRole("link", { name: /create one now/i });
      expect(registerLink).toHaveAttribute("href", "/register");
    });

    it("should show password input as password type by default", () => {
      renderLoginForm();

      const passwordInput = screen.getByLabelText(/password/i);
      expect(passwordInput).toHaveAttribute("type", "password");
    });
  });

  describe("Password Visibility Toggle", () => {
    it("should toggle password visibility when clicking eye button", async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const passwordInput = screen.getByLabelText(/password/i);
      const toggleButton = screen.getByRole("button", { name: /show password/i });

      expect(passwordInput).toHaveAttribute("type", "password");

      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute("type", "text");

      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute("type", "password");
    });
  });

  describe("Form Validation", () => {
    it("should show error when submitting empty username", async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const passwordInput = screen.getByLabelText(/password/i);
      await user.type(passwordInput, "testpassword");

      const submitButton = screen.getByRole("button", { name: /sign in/i });
      await user.click(submitButton);

      expect(await screen.findByText(/email or username is required/i)).toBeInTheDocument();
    });

    it("should show error when submitting empty password", async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const usernameInput = screen.getByLabelText(/email or username/i);
      await user.type(usernameInput, "testuser");

      const submitButton = screen.getByRole("button", { name: /sign in/i });
      await user.click(submitButton);

      expect(await screen.findByText(/password is required/i)).toBeInTheDocument();
    });

    it("should clear field errors when typing", async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const submitButton = screen.getByRole("button", { name: /sign in/i });
      await user.click(submitButton);

      expect(await screen.findByText(/email or username is required/i)).toBeInTheDocument();

      const usernameInput = screen.getByLabelText(/email or username/i);
      await user.type(usernameInput, "t");

      await waitFor(() => {
        expect(screen.queryByText(/email or username is required/i)).not.toBeInTheDocument();
      });
    });
  });

  describe("Form Submission", () => {
    it("should call login with correct credentials on submit", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockResolvedValue({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser@example.com");
      await user.type(passwordInput, "mypassword123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalledWith({
          username: "testuser@example.com",
          password: "mypassword123",
        });
      });
    });

    it("should redirect to dashboard on successful login", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockResolvedValue({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith("/dashboard");
      });
    });

    it("should redirect to custom URL when redirectTo prop is provided", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockResolvedValue({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });

      renderLoginForm({ redirectTo: "/custom/path" }, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith("/custom/path");
      });
    });

    it("should call onSuccess callback after successful login", async () => {
      const user = userEvent.setup();
      const onSuccess = vi.fn();
      const mockLogin = vi.fn().mockResolvedValue({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });

      renderLoginForm({ onSuccess }, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(onSuccess).toHaveBeenCalled();
      });
    });
  });

  describe("Error Handling", () => {
    it("should display error for invalid credentials (401)", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockRejectedValue({
        message: "Invalid credentials",
        code: "401",
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "wrongpassword");
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/invalid email\/username or password/i)).toBeInTheDocument();
      });
    });

    it("should display error for rate limiting (429)", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockRejectedValue({
        message: "Too many requests",
        code: "429",
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/too many login attempts/i)).toBeInTheDocument();
      });
    });

    it("should display verification required message and link", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockRejectedValue({
        message: "Email not verified",
        code: "EMAIL_NOT_VERIFIED",
        requires_verification: true,
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "unverified@example.com");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/verify your email/i)).toBeInTheDocument();
        expect(screen.getByRole("link", { name: /resend verification email/i })).toBeInTheDocument();
      });
    });

    it("should display generic error for unknown errors", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockRejectedValue({
        message: "Something went wrong",
        code: "500",
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
      });
    });
  });

  describe("Loading State", () => {
    it("should show loading state during login", async () => {
      const user = userEvent.setup();
      let resolveLogin: (value: unknown) => void;
      const loginPromise = new Promise((resolve) => {
        resolveLogin = resolve;
      });
      const mockLogin = vi.fn().mockImplementation(() => loginPromise);

      renderLoginForm({}, { login: mockLogin, isLoading: false });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "testuser");
      await user.type(passwordInput, "password123");
      await user.click(submitButton);

      // Button should still be clickable but form should prevent double-submit
      expect(mockLogin).toHaveBeenCalledTimes(1);

      // Resolve the promise
      resolveLogin!({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });
    });

    it("should disable inputs when loading", () => {
      renderLoginForm({}, { isLoading: true });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const submitButton = screen.getByRole("button", { name: /signing in/i });

      expect(usernameInput).toBeDisabled();
      expect(passwordInput).toBeDisabled();
      expect(submitButton).toBeDisabled();
    });
  });

  describe("Remember Me", () => {
    it("should save username to localStorage when remember me is checked", async () => {
      const user = userEvent.setup();
      const mockLogin = vi.fn().mockResolvedValue({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const rememberMeCheckbox = screen.getByRole("checkbox", { name: /remember me/i });
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      await user.type(usernameInput, "remembereduser");
      await user.type(passwordInput, "password123");
      await user.click(rememberMeCheckbox);
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalled();
      });

      // Check localStorage
      const storedValue = localStorage.getItem("chimera_remember_username");
      expect(storedValue).not.toBeNull();
      // Decode the stored value
      const decoded = decodeURIComponent(atob(storedValue!));
      expect(decoded).toBe("remembereduser");
    });

    it("should remove username from localStorage when remember me is unchecked", async () => {
      const user = userEvent.setup();

      // Pre-set a remembered username
      const encoded = btoa(encodeURIComponent("olduser"));
      localStorage.setItem("chimera_remember_username", encoded);

      const mockLogin = vi.fn().mockResolvedValue({
        access_token: "token",
        refresh_token: "refresh",
        token_type: "Bearer",
        expires_in: 3600,
        refresh_expires_in: 86400,
        user: { id: "1", email: "test@example.com", username: "testuser", role: "viewer", is_verified: true },
        requires_verification: false,
      });

      renderLoginForm({}, { login: mockLogin });

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const rememberMeCheckbox = screen.getByRole("checkbox", { name: /remember me/i });
      const submitButton = screen.getByRole("button", { name: /sign in/i });

      // Clear the pre-filled value and enter new username
      await user.clear(usernameInput);
      await user.type(usernameInput, "newuser");
      await user.type(passwordInput, "password123");

      // Ensure remember me is unchecked (it may be checked from stored value)
      if ((rememberMeCheckbox as HTMLInputElement).checked) {
        await user.click(rememberMeCheckbox);
      }

      await user.click(submitButton);

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalled();
      });

      // Check localStorage
      expect(localStorage.getItem("chimera_remember_username")).toBeNull();
    });
  });

  describe("Accessibility", () => {
    it("should have proper aria-invalid attributes on error", async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const submitButton = screen.getByRole("button", { name: /sign in/i });
      await user.click(submitButton);

      const usernameInput = screen.getByLabelText(/email or username/i);
      const passwordInput = screen.getByLabelText(/password/i);

      await waitFor(() => {
        expect(usernameInput).toHaveAttribute("aria-invalid", "true");
        expect(passwordInput).toHaveAttribute("aria-invalid", "true");
      });
    });

    it("should have proper aria-describedby when errors present", async () => {
      const user = userEvent.setup();
      renderLoginForm();

      const submitButton = screen.getByRole("button", { name: /sign in/i });
      await user.click(submitButton);

      const usernameInput = screen.getByLabelText(/email or username/i);

      await waitFor(() => {
        expect(usernameInput).toHaveAttribute("aria-describedby", "username-error");
      });
    });

    it("should have accessible toggle button for password visibility", () => {
      renderLoginForm();

      const toggleButton = screen.getByRole("button", { name: /show password/i });
      expect(toggleButton).toHaveAttribute("aria-label");
    });
  });
});
