"use client";

/**
 * Forgot Password Form Component
 *
 * A form for requesting password reset with:
 * - Email input field
 * - Client-side validation
 * - Success message on submission
 * - Rate limiting feedback
 *
 * @module components/auth/ForgotPasswordForm
 */

import React, { useState, useCallback } from "react";
import Link from "next/link";
import {
  Mail,
  Loader2,
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Send,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";

// =============================================================================
// Types
// =============================================================================

interface ForgotPasswordFormProps {
  /** Pre-filled email (e.g., from login page) */
  initialEmail?: string;
  /** Callback after successful submission */
  onSuccess?: () => void;
  /** Additional class names */
  className?: string;
}

interface FormState {
  email: string;
}

interface FormErrors {
  email?: string;
  general?: string;
}

// =============================================================================
// Component
// =============================================================================

export function ForgotPasswordForm({
  initialEmail = "",
  onSuccess,
  className,
}: ForgotPasswordFormProps) {
  const { requestPasswordReset, isLoading, clearError } = useAuth();

  // Form state
  const [formState, setFormState] = useState<FormState>({
    email: initialEmail,
  });
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

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
      setFormErrors((prev) => ({ ...prev, [field]: undefined, general: undefined }));
      clearError();
    },
    [clearError]
  );

  /**
   * Validate form before submission
   */
  const validateForm = useCallback((): boolean => {
    const errors: FormErrors = {};

    // Email validation
    const email = formState.email.trim();
    if (!email) {
      errors.email = "Email address is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      errors.email = "Please enter a valid email address";
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
      clearError();

      // Validate form
      if (!validateForm()) {
        return;
      }

      setIsSubmitting(true);

      try {
        // Request password reset
        await requestPasswordReset(formState.email.trim());

        // Always show success (to prevent email enumeration)
        setIsSuccess(true);

        // Call success callback if provided
        if (onSuccess) {
          onSuccess();
        }
      } catch {
        // Even on error, show success message for security
        setIsSuccess(true);
      } finally {
        setIsSubmitting(false);
      }
    },
    [formState, requestPasswordReset, validateForm, clearError, onSuccess]
  );

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
            Check your email
          </AlertTitle>
          <AlertDescription className="text-muted-foreground mt-2">
            If an account exists with the email{" "}
            <span className="font-medium text-foreground">
              {formState.email}
            </span>
            , you will receive a password reset link shortly.
          </AlertDescription>
        </Alert>

        {/* Instructions */}
        <div className="space-y-3 text-sm text-muted-foreground">
          <p>The link will expire in 1 hour. If you don&apos;t see the email:</p>
          <ul className="list-disc list-inside space-y-1 pl-2">
            <li>Check your spam or junk folder</li>
            <li>Make sure you entered the correct email</li>
            <li>Wait a few minutes and try again</li>
          </ul>
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-3 pt-2">
          <Button
            type="button"
            variant="outline"
            className="w-full h-11 gap-2"
            onClick={() => {
              setIsSuccess(false);
              setFormState({ email: "" });
            }}
          >
            <Send className="h-4 w-4" />
            Send another email
          </Button>

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
  // Render Form
  // ==========================================================================

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-6", className)}>
      {/* Description */}
      <p className="text-sm text-muted-foreground">
        Enter your email address and we&apos;ll send you a link to reset your
        password.
      </p>

      {/* General Error Alert */}
      {formErrors.general && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formErrors.general}</AlertDescription>
        </Alert>
      )}

      {/* Email Field */}
      <div className="space-y-2">
        <Label htmlFor="forgot-email" className="text-foreground/90">
          Email Address
        </Label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="forgot-email"
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

      {/* Submit Button */}
      <Button
        type="submit"
        size="lg"
        className="w-full h-11 gap-2 text-base font-medium shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-all"
        disabled={isSubmitting || isLoading}
      >
        {isSubmitting ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Sending reset link...
          </>
        ) : (
          <>
            <Send className="h-4 w-4" />
            Send Reset Link
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

export default ForgotPasswordForm;
