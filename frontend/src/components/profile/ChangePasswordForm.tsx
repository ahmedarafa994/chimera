"use client";

/**
 * Change Password Form Component
 *
 * A form for changing user password including:
 * - Current password verification
 * - New password with strength indicator
 * - Confirm password with match indicator
 *
 * @module components/profile/ChangePasswordForm
 */

import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  Lock,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Check,
  Key,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import {
  PasswordStrengthMeter,
  PasswordRequirements,
} from "@/components/auth/PasswordStrengthMeter";
import type { PasswordStrengthResult } from "@/contexts/AuthContext";
import { getApiConfig, getApiHeaders } from "@/lib/api-config";

// =============================================================================
// Types
// =============================================================================

interface ChangePasswordFormProps {
  /** Optional callback after successful change */
  onSuccess?: () => void;
  /** Additional class names */
  className?: string;
}

interface FormState {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

interface FormErrors {
  currentPassword?: string;
  newPassword?: string;
  confirmPassword?: string;
  general?: string;
}

// =============================================================================
// Constants
// =============================================================================

const PASSWORD_CHECK_DEBOUNCE_MS = 500;

// =============================================================================
// Component
// =============================================================================

export function ChangePasswordForm({
  onSuccess,
  className,
}: ChangePasswordFormProps) {
  const { checkPasswordStrength } = useAuth();

  // Form state
  const [formState, setFormState] = useState<FormState>({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  // Password strength state
  const [passwordStrength, setPasswordStrength] =
    useState<PasswordStrengthResult | null>(null);
  const [isCheckingStrength, setIsCheckingStrength] = useState(false);

  // ==========================================================================
  // Password Strength Check
  // ==========================================================================

  useEffect(() => {
    if (!formState.newPassword) {
      setPasswordStrength(null);
      return;
    }

    const timeoutId = setTimeout(async () => {
      setIsCheckingStrength(true);
      try {
        const result = await checkPasswordStrength(formState.newPassword);
        setPasswordStrength(result);
      } finally {
        setIsCheckingStrength(false);
      }
    }, PASSWORD_CHECK_DEBOUNCE_MS);

    return () => clearTimeout(timeoutId);
  }, [formState.newPassword, checkPasswordStrength]);

  // ==========================================================================
  // Computed Values
  // ==========================================================================

  const passwordsMatch = useMemo(() => {
    return (
      formState.newPassword &&
      formState.confirmPassword &&
      formState.newPassword === formState.confirmPassword
    );
  }, [formState.newPassword, formState.confirmPassword]);

  // ==========================================================================
  // Handlers
  // ==========================================================================

  /**
   * Update form field
   */
  const handleInputChange = useCallback(
    (field: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => {
      setFormState((prev) => ({ ...prev, [field]: e.target.value }));
      setFormErrors((prev) => ({
        ...prev,
        [field]: undefined,
        general: undefined,
      }));
      setIsSuccess(false);
    },
    []
  );

  /**
   * Validate form before submission
   */
  const validateForm = useCallback((): boolean => {
    const errors: FormErrors = {};

    // Current password validation
    if (!formState.currentPassword) {
      errors.currentPassword = "Current password is required";
    }

    // New password validation
    if (!formState.newPassword) {
      errors.newPassword = "New password is required";
    } else if (passwordStrength && !passwordStrength.is_valid) {
      errors.newPassword = "Please choose a stronger password";
    } else if (formState.newPassword === formState.currentPassword) {
      errors.newPassword = "New password must be different from current password";
    }

    // Confirm password validation
    if (!formState.confirmPassword) {
      errors.confirmPassword = "Please confirm your new password";
    } else if (formState.newPassword !== formState.confirmPassword) {
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
      setIsSuccess(false);

      // Validate form
      if (!validateForm()) {
        return;
      }

      setIsSubmitting(true);

      try {
        const apiConfig = getApiConfig();
        const headers = getApiHeaders();

        const response = await fetch(
          `${apiConfig.backendApiUrl}/api/v1/users/me/change-password`,
          {
            method: "POST",
            headers: {
              ...headers,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              current_password: formState.currentPassword,
              new_password: formState.newPassword,
            }),
          }
        );

        const data = await response.json();

        if (!response.ok) {
          // Handle specific error cases
          if (response.status === 401) {
            setFormErrors({
              currentPassword: "Current password is incorrect",
            });
          } else if (response.status === 429) {
            setFormErrors({
              general: "Too many attempts. Please try again later.",
            });
          } else if (data.detail?.password_errors) {
            setFormErrors({
              newPassword: data.detail.password_errors.join(". "),
            });
          } else {
            setFormErrors({
              general: data.detail?.error || data.detail || "Failed to change password",
            });
          }
          return;
        }

        // Success - reset form
        setFormState({
          currentPassword: "",
          newPassword: "",
          confirmPassword: "",
        });
        setPasswordStrength(null);
        setIsSuccess(true);

        if (onSuccess) {
          onSuccess();
        }
      } catch (error: unknown) {
        const err = error as { message?: string };
        setFormErrors({
          general: err.message || "An unexpected error occurred",
        });
      } finally {
        setIsSubmitting(false);
      }
    },
    [formState, validateForm, onSuccess]
  );

  /**
   * Toggle password visibility
   */
  const togglePasswordVisibility = useCallback(
    (field: "current" | "new" | "confirm") => () => {
      if (field === "current") setShowCurrentPassword((prev) => !prev);
      else if (field === "new") setShowNewPassword((prev) => !prev);
      else setShowConfirmPassword((prev) => !prev);
    },
    []
  );

  // ==========================================================================
  // Render Success State
  // ==========================================================================

  if (isSuccess) {
    return (
      <div className={cn("space-y-6", className)}>
        <Alert className="bg-green-500/10 border-green-500/20 animate-fade-in">
          <CheckCircle2 className="h-5 w-5 text-green-400" />
          <AlertDescription className="text-green-400 font-medium">
            Password changed successfully!
          </AlertDescription>
        </Alert>

        <Button
          type="button"
          variant="outline"
          className="w-full"
          onClick={() => setIsSuccess(false)}
        >
          Change Password Again
        </Button>
      </div>
    );
  }

  // ==========================================================================
  // Render Form
  // ==========================================================================

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-6", className)}>
      {/* General Error Alert */}
      {formErrors.general && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formErrors.general}</AlertDescription>
        </Alert>
      )}

      {/* Current Password Field */}
      <div className="space-y-2">
        <Label htmlFor="current-password" className="text-foreground/90">
          Current Password
        </Label>
        <div className="relative">
          <Key className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="current-password"
            type={showCurrentPassword ? "text" : "password"}
            placeholder="Enter your current password"
            value={formState.currentPassword}
            onChange={handleInputChange("currentPassword")}
            className={cn(
              "pl-10 pr-10 h-11 bg-background border-input focus:border-primary/50",
              formErrors.currentPassword &&
                "border-destructive focus:border-destructive"
            )}
            autoComplete="current-password"
            disabled={isSubmitting}
            aria-invalid={!!formErrors.currentPassword}
            aria-describedby={
              formErrors.currentPassword ? "current-password-error" : undefined
            }
          />
          <button
            type="button"
            onClick={togglePasswordVisibility("current")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            tabIndex={-1}
            aria-label={showCurrentPassword ? "Hide password" : "Show password"}
          >
            {showCurrentPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
        </div>
        {formErrors.currentPassword && (
          <p
            id="current-password-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.currentPassword}
          </p>
        )}
      </div>

      {/* New Password Field */}
      <div className="space-y-2">
        <Label htmlFor="new-password" className="text-foreground/90">
          New Password
        </Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="new-password"
            type={showNewPassword ? "text" : "password"}
            placeholder="Enter your new password"
            value={formState.newPassword}
            onChange={handleInputChange("newPassword")}
            className={cn(
              "pl-10 pr-10 h-11 bg-background border-input focus:border-primary/50",
              formErrors.newPassword &&
                "border-destructive focus:border-destructive"
            )}
            autoComplete="new-password"
            disabled={isSubmitting}
            aria-invalid={!!formErrors.newPassword}
            aria-describedby={
              formErrors.newPassword ? "new-password-error" : undefined
            }
          />
          <button
            type="button"
            onClick={togglePasswordVisibility("new")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            tabIndex={-1}
            aria-label={showNewPassword ? "Hide password" : "Show password"}
          >
            {showNewPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
        </div>
        {formErrors.newPassword && (
          <p
            id="new-password-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.newPassword}
          </p>
        )}

        {/* Password Requirements Checklist */}
        {formState.newPassword && (
          <div className="pt-2">
            <PasswordRequirements password={formState.newPassword} compact />
          </div>
        )}

        {/* Password Strength Meter */}
        {formState.newPassword && (
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
        <Label htmlFor="confirm-new-password" className="text-foreground/90">
          Confirm New Password
        </Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="confirm-new-password"
            type={showConfirmPassword ? "text" : "password"}
            placeholder="Confirm your new password"
            value={formState.confirmPassword}
            onChange={handleInputChange("confirmPassword")}
            className={cn(
              "pl-10 pr-10 h-11 bg-background border-input focus:border-primary/50",
              formErrors.confirmPassword &&
                "border-destructive focus:border-destructive",
              passwordsMatch && "border-green-500/50 focus:border-green-500/50"
            )}
            autoComplete="new-password"
            disabled={isSubmitting}
            aria-invalid={!!formErrors.confirmPassword}
            aria-describedby={
              formErrors.confirmPassword ? "confirm-password-error" : undefined
            }
          />
          <button
            type="button"
            onClick={togglePasswordVisibility("confirm")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            tabIndex={-1}
            aria-label={showConfirmPassword ? "Hide password" : "Show password"}
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
          isCheckingStrength ||
          (passwordStrength && !passwordStrength.is_valid) ||
          !passwordsMatch ||
          !formState.currentPassword
        }
      >
        {isSubmitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Changing password...
          </>
        ) : (
          <>
            <Lock className="h-4 w-4" />
            Change Password
          </>
        )}
      </Button>
    </form>
  );
}

export default ChangePasswordForm;
