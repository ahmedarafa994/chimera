"use client";

/**
 * Login Form Component
 *
 * A comprehensive login form with:
 * - Email/username and password fields
 * - Client-side validation
 * - Error display for invalid credentials
 * - Remember me option
 * - Links to forgot password and register
 *
 * @module components/auth/LoginForm
 */

import React, { useState, useCallback } from "react";
import Link from "next/link";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  ArrowRight,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import type { AuthError } from "@/contexts/AuthContext";

// =============================================================================
// Types
// =============================================================================

interface LoginFormProps {
  /** Redirect URL after successful login */
  redirectTo?: string;
  /** Callback after successful login */
  onSuccess?: () => void;
  /** Additional class names */
  className?: string;
}

interface FormState {
  username: string;
  password: string;
  rememberMe: boolean;
}

interface FormErrors {
  username?: string;
  password?: string;
  general?: string;
}

// =============================================================================
// Constants
// =============================================================================

const REMEMBER_ME_KEY = "chimera_remember_username";

// =============================================================================
// Component
// =============================================================================

export function LoginForm({
  redirectTo = "/dashboard",
  onSuccess,
  className,
}: LoginFormProps) {
  const { login, isLoading, error: authError, clearError } = useAuth();

  // Form state
  const [formState, setFormState] = useState<FormState>(() => {
    // Check for remembered username
    if (typeof window !== "undefined") {
      const remembered = localStorage.getItem(REMEMBER_ME_KEY);
      if (remembered) {
        try {
          const decoded = decodeURIComponent(atob(remembered));
          return { username: decoded, password: "", rememberMe: true };
        } catch {
          // Invalid stored value, ignore
        }
      }
    }
    return { username: "", password: "", rememberMe: false };
  });

  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [showPassword, setShowPassword] = useState(false);
  const [requiresVerification, setRequiresVerification] = useState(false);

  // ==========================================================================
  // Handlers
  // ==========================================================================

  /**
   * Update form field
   */
  const handleInputChange = useCallback(
    (field: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.type === "checkbox" ? e.target.checked : e.target.value;
      setFormState((prev) => ({ ...prev, [field]: value }));
      // Clear field-specific error on change
      setFormErrors((prev) => ({ ...prev, [field]: undefined, general: undefined }));
      clearError();
    },
    [clearError]
  );

  /**
   * Handle remember me checkbox change
   */
  const handleRememberMeChange = useCallback((checked: boolean) => {
    setFormState((prev) => ({ ...prev, rememberMe: checked }));
  }, []);

  /**
   * Validate form before submission
   */
  const validateForm = useCallback((): boolean => {
    const errors: FormErrors = {};

    // Username/email validation
    if (!formState.username.trim()) {
      errors.username = "Email or username is required";
    }

    // Password validation
    if (!formState.password) {
      errors.password = "Password is required";
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  }, [formState]);

  /**
   * Handle form submission
   */
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      // Clear previous errors
      setFormErrors({});
      setRequiresVerification(false);
      clearError();
      console.log("[LoginForm] Submitting login form...");

      // Validate form
      if (!validateForm()) {
        return;
      }

      try {
        // Attempt login
        await login({
          username: formState.username.trim(),
          password: formState.password,
        });
        console.log("[LoginForm] Login successful");

        // Handle remember me
        if (formState.rememberMe) {
          try {
            const encoded = btoa(encodeURIComponent(formState.username));
            localStorage.setItem(REMEMBER_ME_KEY, encoded);
          } catch {
            // Storage error, ignore
          }
        } else {
          localStorage.removeItem(REMEMBER_ME_KEY);
        }

        // Call success callback if provided
        if (onSuccess) {
          onSuccess();
        }

        // Let the login page handle redirect after auth state is fully updated
        // This prevents race conditions between auth state update and navigation
      } catch (err) {
        const error = err as AuthError;

        // Check if email verification is required
        if (error.requires_verification) {
          setRequiresVerification(true);
          setFormErrors({
            general: "Please verify your email address before logging in.",
          });
        } else if (error.code === "401") {
          setFormErrors({
            general: "Invalid email/username or password. Please try again.",
          });
        } else if (error.code === "429") {
          setFormErrors({
            general: "Too many login attempts. Please try again later.",
          });
        } else {
          setFormErrors({
            general: error.message || "An unexpected error occurred. Please try again.",
          });
        }
      }
    },
    [formState, login, validateForm, clearError, onSuccess]
  );

  /**
   * Toggle password visibility
   */
  const togglePasswordVisibility = useCallback(() => {
    setShowPassword((prev) => !prev);
  }, []);

  // ==========================================================================
  // Render
  // ==========================================================================

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-6", className)}>
      {/* General Error Alert */}
      {(formErrors.general || authError) && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {formErrors.general || authError?.message}
            {requiresVerification && (
              <div className="mt-2">
                <Link
                  href={`/verify-email?email=${encodeURIComponent(formState.username)}`}
                  className="text-primary hover:underline font-medium"
                >
                  Resend verification email
                </Link>
              </div>
            )}
          </AlertDescription>
        </Alert>
      )}

      {/* Username/Email Field */}
      <div className="space-y-2">
        <Label htmlFor="login-username" className="text-foreground/90">
          Email or Username
        </Label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="login-username"
            type="text"
            placeholder="you@example.com"
            value={formState.username}
            onChange={handleInputChange("username")}
            className={cn(
              "pl-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.username && "border-destructive focus:border-destructive"
            )}
            autoComplete="username"
            autoFocus
            disabled={isLoading}
            aria-invalid={!!formErrors.username}
            aria-describedby={formErrors.username ? "username-error" : undefined}
          />
        </div>
        {formErrors.username && (
          <p id="username-error" className="text-sm text-destructive animate-fade-in">
            {formErrors.username}
          </p>
        )}
      </div>

      {/* Password Field */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="login-password" className="text-foreground/90">
            Password
          </Label>
          <Link
            href="/forgot-password"
            className="text-sm text-primary hover:text-primary/80 hover:underline transition-colors"
            tabIndex={-1}
          >
            Forgot password?
          </Link>
        </div>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="login-password"
            type={showPassword ? "text" : "password"}
            placeholder="Enter your password"
            value={formState.password}
            onChange={handleInputChange("password")}
            className={cn(
              "pl-10 pr-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.password && "border-destructive focus:border-destructive"
            )}
            autoComplete="current-password"
            disabled={isLoading}
            aria-invalid={!!formErrors.password}
            aria-describedby={formErrors.password ? "password-error" : undefined}
          />
          <button
            type="button"
            onClick={togglePasswordVisibility}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            tabIndex={-1}
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
        </div>
        {formErrors.password && (
          <p id="password-error" className="text-sm text-destructive animate-fade-in">
            {formErrors.password}
          </p>
        )}
      </div>

      {/* Remember Me Checkbox */}
      <div className="flex items-center space-x-2">
        <Checkbox
          id="login-remember"
          checked={formState.rememberMe}
          onCheckedChange={handleRememberMeChange}
          disabled={isLoading}
        />
        <Label
          htmlFor="login-remember"
          className="text-sm text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
        >
          Remember me
        </Label>
      </div>

      {/* Submit Button */}
      <Button
        type="submit"
        size="lg"
        className="w-full h-11 gap-2 text-base font-medium shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-all"
        disabled={isLoading}
      >
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Signing in...
          </>
        ) : (
          <>
            Sign In
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>

      {/* Register Link */}
      <div className="text-center text-sm text-muted-foreground">
        Don&apos;t have an account?{" "}
        <Link
          href="/register"
          className="text-primary hover:text-primary/80 font-medium hover:underline transition-colors"
        >
          Create one now
        </Link>
      </div>
    </form>
  );
}

export default LoginForm;
