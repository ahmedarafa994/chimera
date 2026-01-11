"use client";

/**
 * Register Form Component
 *
 * A comprehensive registration form with:
 * - Email, username, and password fields
 * - Password strength indicator with real-time feedback
 * - Confirm password validation
 * - Terms acceptance checkbox
 * - Client-side and server-side validation
 * - Link to login
 *
 * @module components/auth/RegisterForm
 */

import React, { useState, useCallback, useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Mail,
  Lock,
  User,
  Eye,
  EyeOff,
  Loader2,
  AlertCircle,
  ArrowRight,
  Check,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import {
  PasswordStrengthMeter,
  PasswordRequirements,
} from "@/components/auth/PasswordStrengthMeter";
import type { AuthError, PasswordStrengthResult } from "@/contexts/AuthContext";

// =============================================================================
// Types
// =============================================================================

interface RegisterFormProps {
  /** Callback after successful registration */
  onSuccess?: () => void;
  /** Additional class names */
  className?: string;
}

interface FormState {
  email: string;
  username: string;
  password: string;
  confirmPassword: string;
  acceptTerms: boolean;
}

interface FormErrors {
  email?: string;
  username?: string;
  password?: string;
  confirmPassword?: string;
  acceptTerms?: string;
  general?: string;
}

// =============================================================================
// Constants
// =============================================================================

/** Debounce delay for password strength check (ms) */
const PASSWORD_CHECK_DEBOUNCE = 500;

/** Minimum password length to start checking strength */
const MIN_PASSWORD_LENGTH_FOR_CHECK = 4;

// =============================================================================
// Component
// =============================================================================

export function RegisterForm({ onSuccess, className }: RegisterFormProps) {
  const router = useRouter();
  const { register, checkPasswordStrength, isLoading, clearError } = useAuth();

  // Form state
  const [formState, setFormState] = useState<FormState>({
    email: "",
    username: "",
    password: "",
    confirmPassword: "",
    acceptTerms: false,
  });

  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [registrationSuccess, setRegistrationSuccess] = useState(false);

  // Password strength state
  const [passwordStrength, setPasswordStrength] =
    useState<PasswordStrengthResult | null>(null);
  const [isCheckingPassword, setIsCheckingPassword] = useState(false);
  const passwordCheckTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );

  // ==========================================================================
  // Password Strength Check
  // ==========================================================================

  // Debounced password strength check
  useEffect(() => {
    // Clear existing timeout
    if (passwordCheckTimeoutRef.current) {
      clearTimeout(passwordCheckTimeoutRef.current);
    }

    // Don't check if password is too short
    if (formState.password.length < MIN_PASSWORD_LENGTH_FOR_CHECK) {
      setPasswordStrength(null);
      setIsCheckingPassword(false);
      return;
    }

    // Show loading indicator
    setIsCheckingPassword(true);

    // Debounce the API call
    passwordCheckTimeoutRef.current = setTimeout(async () => {
      try {
        const result = await checkPasswordStrength(formState.password);
        setPasswordStrength(result);
      } catch {
        // Error checking password, show generic message
        setPasswordStrength({
          is_valid: false,
          score: 0,
          errors: ["Unable to check password strength"],
          warnings: [],
          suggestions: [],
        });
      } finally {
        setIsCheckingPassword(false);
      }
    }, PASSWORD_CHECK_DEBOUNCE);

    return () => {
      if (passwordCheckTimeoutRef.current) {
        clearTimeout(passwordCheckTimeoutRef.current);
      }
    };
  }, [formState.password, checkPasswordStrength]);

  // ==========================================================================
  // Handlers
  // ==========================================================================

  /**
   * Update form field
   */
  const handleInputChange = useCallback(
    (field: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => {
      const value =
        e.target.type === "checkbox" ? e.target.checked : e.target.value;
      setFormState((prev) => ({ ...prev, [field]: value }));
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
   * Handle terms checkbox change
   */
  const handleTermsChange = useCallback((checked: boolean) => {
    setFormState((prev) => ({ ...prev, acceptTerms: checked }));
    setFormErrors((prev) => ({ ...prev, acceptTerms: undefined }));
  }, []);

  /**
   * Validate email format
   */
  const validateEmail = useCallback((email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }, []);

  /**
   * Validate username format
   */
  const validateUsername = useCallback((username: string): boolean => {
    // Alphanumeric, underscores, hyphens. 3-30 chars. Must start with letter
    const usernameRegex = /^[a-zA-Z][a-zA-Z0-9_-]{2,29}$/;
    return usernameRegex.test(username);
  }, []);

  /**
   * Validate form before submission
   */
  const validateForm = useCallback((): boolean => {
    const errors: FormErrors = {};

    // Email validation
    if (!formState.email.trim()) {
      errors.email = "Email is required";
    } else if (!validateEmail(formState.email)) {
      errors.email = "Please enter a valid email address";
    }

    // Username validation
    if (!formState.username.trim()) {
      errors.username = "Username is required";
    } else if (!validateUsername(formState.username)) {
      errors.username =
        "Username must be 3-30 characters, start with a letter, and contain only letters, numbers, underscores, or hyphens";
    }

    // Password validation
    if (!formState.password) {
      errors.password = "Password is required";
    } else if (passwordStrength && !passwordStrength.is_valid) {
      errors.password = "Password does not meet requirements";
    }

    // Confirm password validation
    if (!formState.confirmPassword) {
      errors.confirmPassword = "Please confirm your password";
    } else if (formState.password !== formState.confirmPassword) {
      errors.confirmPassword = "Passwords do not match";
    }

    // Terms validation
    if (!formState.acceptTerms) {
      errors.acceptTerms = "You must accept the Terms of Service";
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  }, [formState, passwordStrength, validateEmail, validateUsername]);

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
        // Register user
        const response = await register({
          email: formState.email.trim().toLowerCase(),
          username: formState.username.trim(),
          password: formState.password,
        });

        // Show success state
        setRegistrationSuccess(true);

        // Call success callback if provided
        if (onSuccess) {
          onSuccess();
        }

        // Redirect to login with success message after a brief delay
        setTimeout(() => {
          router.push("/login?registered=true");
        }, 2000);
      } catch (err) {
        const error = err as AuthError;

        // Handle specific error cases
        if (error.code === "409" || error.message?.includes("already")) {
          if (error.message?.toLowerCase().includes("email")) {
            setFormErrors({
              email: "This email is already registered",
            });
          } else if (error.message?.toLowerCase().includes("username")) {
            setFormErrors({
              username: "This username is already taken",
            });
          } else {
            setFormErrors({
              general: error.message || "Email or username already exists",
            });
          }
        } else if (error.field_errors) {
          // Handle field-specific errors from backend
          const fieldErrors: FormErrors = {};
          for (const [field, messages] of Object.entries(error.field_errors)) {
            if (field in formState) {
              fieldErrors[field as keyof FormErrors] = messages.join(". ");
            }
          }
          setFormErrors(fieldErrors);
        } else if (error.code === "429") {
          setFormErrors({
            general: "Too many registration attempts. Please try again later.",
          });
        } else {
          setFormErrors({
            general:
              error.message || "An unexpected error occurred. Please try again.",
          });
        }
      } finally {
        setIsSubmitting(false);
      }
    },
    [formState, register, validateForm, clearError, onSuccess, router]
  );

  /**
   * Toggle password visibility
   */
  const togglePasswordVisibility = useCallback(() => {
    setShowPassword((prev) => !prev);
  }, []);

  /**
   * Toggle confirm password visibility
   */
  const toggleConfirmPasswordVisibility = useCallback(() => {
    setShowConfirmPassword((prev) => !prev);
  }, []);

  // ==========================================================================
  // Render Success State
  // ==========================================================================

  if (registrationSuccess) {
    return (
      <div className={cn("text-center space-y-6 py-8", className)}>
        <div className="mx-auto w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center animate-fade-in">
          <Check className="w-8 h-8 text-green-400" />
        </div>

        <div className="space-y-2 animate-fade-in-up">
          <h3 className="text-xl font-semibold text-foreground">
            Registration Successful!
          </h3>
          <p className="text-muted-foreground text-sm">
            We&apos;ve sent a verification email to{" "}
            <span className="text-primary font-medium">{formState.email}</span>.
            <br />
            Please check your inbox to verify your account.
          </p>
        </div>

        <div
          className="flex items-center justify-center gap-2 text-muted-foreground text-sm animate-fade-in"
          style={{ animationDelay: "0.5s" }}
        >
          <Loader2 className="w-4 h-4 animate-spin" />
          Redirecting to login...
        </div>
      </div>
    );
  }

  // ==========================================================================
  // Render Form
  // ==========================================================================

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-5", className)}>
      {/* General Error Alert */}
      {formErrors.general && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formErrors.general}</AlertDescription>
        </Alert>
      )}

      {/* Email Field */}
      <div className="space-y-2">
        <Label htmlFor="register-email" className="text-foreground/90">
          Email
        </Label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="register-email"
            type="email"
            placeholder="you@example.com"
            value={formState.email}
            onChange={handleInputChange("email")}
            className={cn(
              "pl-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.email && "border-destructive focus:border-destructive"
            )}
            autoComplete="email"
            autoFocus
            disabled={isSubmitting || isLoading}
            aria-invalid={!!formErrors.email}
            aria-describedby={formErrors.email ? "email-error" : undefined}
          />
        </div>
        {formErrors.email && (
          <p id="email-error" className="text-sm text-destructive animate-fade-in">
            {formErrors.email}
          </p>
        )}
      </div>

      {/* Username Field */}
      <div className="space-y-2">
        <Label htmlFor="register-username" className="text-foreground/90">
          Username
        </Label>
        <div className="relative">
          <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="register-username"
            type="text"
            placeholder="johndoe"
            value={formState.username}
            onChange={handleInputChange("username")}
            className={cn(
              "pl-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.username &&
                "border-destructive focus:border-destructive"
            )}
            autoComplete="username"
            disabled={isSubmitting || isLoading}
            aria-invalid={!!formErrors.username}
            aria-describedby={formErrors.username ? "username-error" : undefined}
          />
        </div>
        {formErrors.username && (
          <p
            id="username-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.username}
          </p>
        )}
      </div>

      {/* Password Field */}
      <div className="space-y-2">
        <Label htmlFor="register-password" className="text-foreground/90">
          Password
        </Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="register-password"
            type={showPassword ? "text" : "password"}
            placeholder="Create a strong password"
            value={formState.password}
            onChange={handleInputChange("password")}
            className={cn(
              "pl-10 pr-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.password &&
                "border-destructive focus:border-destructive"
            )}
            autoComplete="new-password"
            disabled={isSubmitting || isLoading}
            aria-invalid={!!formErrors.password}
            aria-describedby={
              formErrors.password ? "password-error" : "password-strength"
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
            id="password-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.password}
          </p>
        )}

        {/* Password Strength Indicator */}
        <div id="password-strength" className="pt-1">
          {formState.password.length >= MIN_PASSWORD_LENGTH_FOR_CHECK ? (
            <PasswordStrengthMeter
              result={passwordStrength}
              isLoading={isCheckingPassword}
              showDetails
              compact
            />
          ) : (
            formState.password.length > 0 && (
              <PasswordRequirements
                password={formState.password}
                compact
              />
            )
          )}
        </div>
      </div>

      {/* Confirm Password Field */}
      <div className="space-y-2">
        <Label
          htmlFor="register-confirm-password"
          className="text-foreground/90"
        >
          Confirm Password
        </Label>
        <div className="relative">
          <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="register-confirm-password"
            type={showConfirmPassword ? "text" : "password"}
            placeholder="Confirm your password"
            value={formState.confirmPassword}
            onChange={handleInputChange("confirmPassword")}
            className={cn(
              "pl-10 pr-10 h-11 bg-white/[0.03] border-white/[0.08] focus:border-primary/50",
              formErrors.confirmPassword &&
                "border-destructive focus:border-destructive",
              formState.confirmPassword &&
                formState.password === formState.confirmPassword &&
                "border-green-500/50 focus:border-green-500"
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
              showConfirmPassword
                ? "Hide confirm password"
                : "Show confirm password"
            }
          >
            {showConfirmPassword ? (
              <EyeOff className="h-4 w-4" />
            ) : (
              <Eye className="h-4 w-4" />
            )}
          </button>
        </div>
        {formErrors.confirmPassword && (
          <p
            id="confirm-password-error"
            className="text-sm text-destructive animate-fade-in"
          >
            {formErrors.confirmPassword}
          </p>
        )}
        {/* Password match indicator */}
        {formState.confirmPassword &&
          formState.password === formState.confirmPassword && (
            <p className="text-sm text-green-400 flex items-center gap-1.5 animate-fade-in">
              <Check className="w-3.5 h-3.5" />
              Passwords match
            </p>
          )}
      </div>

      {/* Terms Checkbox */}
      <div className="space-y-2">
        <div className="flex items-start space-x-2">
          <Checkbox
            id="register-terms"
            checked={formState.acceptTerms}
            onCheckedChange={handleTermsChange}
            disabled={isSubmitting || isLoading}
            className="mt-1"
          />
          <Label
            htmlFor="register-terms"
            className={cn(
              "text-sm text-muted-foreground cursor-pointer leading-relaxed",
              formErrors.acceptTerms && "text-destructive"
            )}
          >
            I agree to the{" "}
            <Link
              href="/terms"
              className="text-primary hover:text-primary/80 hover:underline"
              target="_blank"
            >
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link
              href="/privacy"
              className="text-primary hover:text-primary/80 hover:underline"
              target="_blank"
            >
              Privacy Policy
            </Link>
          </Label>
        </div>
        {formErrors.acceptTerms && (
          <p className="text-sm text-destructive animate-fade-in">
            {formErrors.acceptTerms}
          </p>
        )}
      </div>

      {/* Submit Button */}
      <Button
        type="submit"
        size="lg"
        className="w-full h-11 gap-2 text-base font-medium shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-all"
        disabled={isSubmitting || isLoading}
      >
        {isSubmitting || isLoading ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Creating account...
          </>
        ) : (
          <>
            Create Account
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>

      {/* Login Link */}
      <div className="text-center text-sm text-muted-foreground">
        Already have an account?{" "}
        <Link
          href="/login"
          className="text-primary hover:text-primary/80 font-medium hover:underline transition-colors"
        >
          Sign in
        </Link>
      </div>
    </form>
  );
}

export default RegisterForm;
