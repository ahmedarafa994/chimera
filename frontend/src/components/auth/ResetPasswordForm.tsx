"use client";

/**
 * Reset Password Form Component
 *
 * A form for resetting password with:
 * - New password input with strength indicator
 * - Confirm password field
 * - Client-side and server-side validation
 * - Success state with redirect to login
 *
 * @module components/auth/ResetPasswordForm
 */

import React, { useState, useCallback, useEffect, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Lock,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  Check,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import {
  PasswordStrengthMeter,
  PasswordRequirements,
} from "@/components/auth/PasswordStrengthMeter";
import type { PasswordStrengthResult } from "@/contexts/AuthContext";

// =============================================================================
// Types
// =============================================================================

interface ResetPasswordFormProps {
  /** Reset token from URL */
  token: string;
  /** Callback after successful reset */
  onSuccess?: () => void;
  /** Additional class names */
  className?: string;
}

interface FormState {
  password: string;
  confirmPassword: string;
}

interface FormErrors {
  password?: string;
  confirmPassword?: string;
  token?: string;
  general?: string;
}

// =============================================================================
// Constants
// =============================================================================

const PASSWORD_CHECK_DEBOUNCE_MS = 500;

// =============================================================================
// Component
// =============================================================================

export function ResetPasswordForm({
  token,
  onSuccess,
  className,
}: ResetPasswordFormProps) {
  const router = useRouter();
  const { resetPassword, checkPasswordStrength, isLoading, clearError } =
    useAuth();

  // Form state
  const [formState, setFormState] = useState<FormState>({
    password: "",
    confirmPassword: "",
  });
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [redirectCountdown, setRedirectCountdown] = useState(5);

  // Password strength state
  const [passwordStrength, setPasswordStrength] =
    useState<PasswordStrengthResult | null>(null);
  const [isCheckingStrength, setIsCheckingStrength] = useState(false);

  // Token validation state
  const [isTokenValid, setIsTokenValid] = useState(true);
  const [tokenExpired, setTokenExpired] = useState(false);

  // ==========================================================================
  // Password Strength Check
  // ==========================================================================

  useEffect(() => {
    if (!formState.password) {
      setPasswordStrength(null);
      return;
    }

    const timeoutId = setTimeout(async () => {
      setIsCheckingStrength(true);
      try {
        const result = await checkPasswordStrength(formState.password);
        setPasswordStrength(result);
      } finally {
        setIsCheckingStrength(false);
      }
    }, PASSWORD_CHECK_DEBOUNCE_MS);

    return () => clearTimeout(timeoutId);
  }, [formState.password, checkPasswordStrength]);

  // ==========================================================================
  // Redirect Countdown
  // ==========================================================================

  useEffect(() => {
    if (!isSuccess) return;

    const timer = setInterval(() => {
      setRedirectCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          router.push("/login?password_reset=true");
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isSuccess, router]);

  // ==========================================================================
  // Computed Values
  // ==========================================================================

  const passwordsMatch = useMemo(() => {
    return (
      formState.password &&
      formState.confirmPassword &&
      formState.password === formState.confirmPassword
    );
  }, [formState.password, formState.confirmPassword]);

  // ==========================================================================
  // Handlers
  // ==========================================================================

  /**
   * Update form field
   */
  const handleInputChange = useCallback(
    (field: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => {
      setFormState((prev) => ({ ...prev, [field]: e.target.value }));
      // Clear field-specific error on change
      setFormErrors((prev) => ({
        ...prev,
        [field]: undefined,
        general: undefined,
      }));
      clearError();
    },
    [clearError]
  );

  /**
   * Validate form before submission
   */
  const validateForm = useCallback((): boolean => {
    const errors: FormErrors = {};

    // Password validation
    if (!formState.password) {
      errors.password = "Password is required";
    } else if (passwordStrength && !passwordStrength.is_valid) {
      errors.password = "Please choose a stronger password";
    }

    // Confirm password validation
    if (!formState.confirmPassword) {
      errors.confirmPassword = "Please confirm your password";
    } else if (formState.password !== formState.confirmPassword) {
      errors.confirmPassword = "Passwords do not match";
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  }, [formState, passwordStrength]);

  /**
   * Handle form submission
   */
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      // Clear previous errors
      setFormErrors({});
      clearError();

      // Validate form
      if (!validateForm()) {
        return;
      }

      setIsSubmitting(true);

      try {
        const success = await resetPassword(token, formState.password);

        if (success) {
          setIsSuccess(true);

          // Call success callback if provided
          if (onSuccess) {
            onSuccess();
          }
        } else {
          // Check for specific error types
          setFormErrors({
            general: "Failed to reset password. The link may have expired.",
          });
          setTokenExpired(true);
        }
      } catch (error: unknown) {
        const err = error as { message?: string; code?: string };
        if (err.code === "400" || err.message?.includes("expired")) {
          setTokenExpired(true);
          setIsTokenValid(false);
          setFormErrors({
            token: "This password reset link has expired.",
          });
        } else {
          setFormErrors({
            general:
              err.message || "An unexpected error occurred. Please try again.",
          });
        }
      } finally {
        setIsSubmitting(false);
      }
    },
    [formState, token, resetPassword, validateForm, clearError, onSuccess]
  );

  /**
   * Toggle password visibility
   */
  const togglePasswordVisibility = useCallback(() => {
    setShowPassword((prev) => !prev);
  }, []);

  const toggleConfirmPasswordVisibility = useCallback(() => {
    setShowConfirmPassword((prev) => !prev);
  }, []);

  // ==========================================================================
  // Render Invalid Token State
  // ==========================================================================

  if (!isTokenValid || tokenExpired) {
    return (
      <div className={cn("space-y-6", className)}>
        {/* Error Alert */}
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-5 w-5" />
          <AlertTitle className="font-semibold">Link Expired</AlertTitle>
          <AlertDescription className="mt-2">
            This password reset link has expired or is invalid. Please request a
            new password reset link.
          </AlertDescription>
        </Alert>

        {/* Actions */}
        <div className="flex flex-col gap-3 pt-2">
          <Link href="/forgot-password" className="w-full">
            <Button
              type="button"
              size="lg"
              className="w-full h-11 gap-2 text-base font-medium"
            >
              Request New Link
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>

          <Link href="/login" className="w-full">
            <Button
              type="button"
              variant="ghost"
              className="w-full h-11 gap-2 text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to login
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  // ==========================================================================
  // Render Success State
  // ==========================================================================

  if (isSuccess) {
    return (
      <div className={cn("space-y-6", className)}>
        {/* Success Alert */}
        <Alert className="bg-green-500/10 border-green-500/20 animate-fade-in">
          <CheckCircle2 className="h-5 w-5 text-green-400" />
          <AlertTitle className="text-green-400 font-semibold">
            Password Reset Successfully
          </AlertTitle>
          <AlertDescription className="text-muted-foreground mt-2">
            Your password has been changed. You can now log in with your new
            password.
          </AlertDescription>
        </Alert>

        {/* Redirect Notice */}
        <p className="text-sm text-center text-muted-foreground">
          Redirecting to login in{" "}
          <span className="font-medium text-foreground">
            {redirectCountdown}
          </span>{" "}
          seconds...
        </p>

        {/* Immediate Login Button */}
        <Link href="/login?password_reset=true" className="block">
          <Button
            type="button"
            size="lg"
            className="w-full h-11 gap-2 text-base font-medium shadow-lg shadow-primary/20"
          >
            Go to Login
            <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>
    );
  }

  // ==========================================================================
  // Render Form
  // ==========================================================================

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-6", className)}>
      {/* Description */}
      <p className="text-sm text-muted-foreground">
        Create a new password for your account. Make sure it&apos;s strong and
        unique.
      </p>

      {/* General Error Alert */}
      {formErrors.general && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formErrors.general}</AlertDescription>
        </Alert>
      )}

      {/* Token Error Alert */}
      {formErrors.token && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formErrors.token}</AlertDescription>
        </Alert>
      )}

      {/* New Password Field */}
      <div className="space-y-2">
        <Label htmlFor="reset-password" className="text-foreground/90">
          New Password
        </Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="reset-password"
            type={showPassword ? "text" : "password"}
            placeholder="Enter your new password"
            value={formState.password}
            onChange={handleInputChange("password")}
            className={cn(
              "pl-10 pr-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.password && "border-destructive focus:border-destructive"
            )}
            autoComplete="new-password"
            autoFocus
            disabled={isSubmitting || isLoading}
            aria-invalid={!!formErrors.password}
            aria-describedby={
              formErrors.password ? "new-password-error" : undefined
            }
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
          <p
            id="new-password-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.password}
          </p>
        )}

        {/* Password Requirements Checklist */}
        {formState.password && (
          <div className="pt-2">
            <PasswordRequirements password={formState.password} compact />
          </div>
        )}

        {/* Password Strength Meter */}
        {formState.password && (
          <div className="pt-2">
            <PasswordStrengthMeter
              result={passwordStrength}
              isLoading={isCheckingStrength}
              showDetails
              compact
            />
          </div>
        )}
      </div>

      {/* Confirm Password Field */}
      <div className="space-y-2">
        <Label htmlFor="reset-confirm-password" className="text-foreground/90">
          Confirm New Password
        </Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="reset-confirm-password"
            type={showConfirmPassword ? "text" : "password"}
            placeholder="Confirm your new password"
            value={formState.confirmPassword}
            onChange={handleInputChange("confirmPassword")}
            className={cn(
              "pl-10 pr-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.confirmPassword &&
                "border-destructive focus:border-destructive",
              passwordsMatch && "border-green-500/50 focus:border-green-500/50"
            )}
            autoComplete="new-password"
            disabled={isSubmitting || isLoading}
            aria-invalid={!!formErrors.confirmPassword}
            aria-describedby={
              formErrors.confirmPassword ? "confirm-password-error" : undefined
            }
          />
          <button
            type="button"
            onClick={toggleConfirmPasswordVisibility}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            tabIndex={-1}
            aria-label={
              showConfirmPassword ? "Hide password" : "Show password"
            }
          >
            {showConfirmPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* Password Match Indicator */}
        {formState.confirmPassword && (
          <div className="flex items-center gap-2 animate-fade-in">
            {passwordsMatch ? (
              <>
                <Check className="h-4 w-4 text-green-400" />
                <span className="text-sm text-green-400">Passwords match</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-4 w-4 text-destructive" />
                <span className="text-sm text-destructive">
                  Passwords do not match
                </span>
              </>
            )}
          </div>
        )}

        {formErrors.confirmPassword && (
          <p
            id="confirm-password-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.confirmPassword}
          </p>
        )}
      </div>

      {/* Submit Button */}
      <Button
        type="submit"
        size="lg"
        className="w-full h-11 gap-2 text-base font-medium shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-all"
        disabled={
          isSubmitting ||
          isLoading ||
          isCheckingStrength ||
          (passwordStrength && !passwordStrength.is_valid) ||
          !passwordsMatch
        }
      >
        {isSubmitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Resetting password...
          </>
        ) : (
          <>
            Reset Password
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>

      {/* Back to Login Link */}
      <div className="text-center">
        <Link
          href="/login"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to login
        </Link>
      </div>
    </form>
  );
}

export default ResetPasswordForm;
