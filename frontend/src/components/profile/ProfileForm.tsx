"use client";

/**
 * Profile Form Component
 *
 * A form for viewing and editing user profile information including:
 * - User avatar with initials
 * - Username editing
 * - Role and status display
 * - Email display (read-only)
 *
 * @module components/profile/ProfileForm
 */

import React, { useState, useCallback } from "react";
import {
  User,
  AtSign,
  Mail,
  Shield,
  FlaskConical,
  Eye,
  Loader2,
  Save,
  AlertCircle,
  CheckCircle2,
  Calendar,
  Clock,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { useAuth, type AuthUser, type UserRole } from "@/hooks/useAuth";
import { getApiConfig, getApiHeaders } from "@/lib/api-config";

// =============================================================================
// Types
// =============================================================================

interface ProfileFormProps {
  /** Optional callback after successful update */
  onUpdate?: (user: AuthUser) => void;
  /** Additional class names */
  className?: string;
}

interface FormState {
  username: string;
}

interface FormErrors {
  username?: string;
  general?: string;
}

// =============================================================================
// Role Configuration
// =============================================================================

interface RoleConfig {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badgeClassName: string;
  description: string;
}

const ROLE_CONFIG: Record<UserRole, RoleConfig> = {
  admin: {
    label: "Administrator",
    icon: Shield,
    badgeClassName:
      "bg-gradient-to-r from-red-500/20 to-orange-500/20 text-orange-400 border-orange-500/30",
    description: "Full system access and user management",
  },
  researcher: {
    label: "Researcher",
    icon: FlaskConical,
    badgeClassName:
      "bg-gradient-to-r from-blue-500/20 to-cyan-500/20 text-cyan-400 border-cyan-500/30",
    description: "Create and edit campaigns and prompts",
  },
  viewer: {
    label: "Viewer",
    icon: Eye,
    badgeClassName:
      "bg-gradient-to-r from-gray-500/20 to-slate-500/20 text-slate-400 border-slate-500/30",
    description: "Read-only access to campaigns and results",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate avatar initials from username or email
 */
function getInitials(username?: string, email?: string): string {
  if (username) {
    return username.slice(0, 2).toUpperCase();
  }
  if (email) {
    const parts = email.split("@");
    if (parts.length === 2 && parts[0] && parts[1]) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return email.slice(0, 2).toUpperCase();
  }
  return "??";
}

/**
 * Generate a consistent color based on username for avatar background
 */
function getAvatarColor(username?: string): string {
  if (!username) return "bg-primary";

  let hash = 0;
  for (let i = 0; i < username.length; i++) {
    hash = username.charCodeAt(i) + ((hash << 5) - hash);
  }

  const colors = [
    "bg-gradient-to-br from-violet-500 to-purple-600",
    "bg-gradient-to-br from-blue-500 to-cyan-600",
    "bg-gradient-to-br from-emerald-500 to-teal-600",
    "bg-gradient-to-br from-orange-500 to-amber-600",
    "bg-gradient-to-br from-pink-500 to-rose-600",
    "bg-gradient-to-br from-indigo-500 to-blue-600",
  ];

  return colors[Math.abs(hash) % colors.length] || "bg-primary";
}

/**
 * Format date for display
 */
function formatDate(dateStr?: string | null): string {
  if (!dateStr) return "Never";
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "Unknown";
  }
}

// =============================================================================
// Component
// =============================================================================

export function ProfileForm({ onUpdate, className }: ProfileFormProps) {
  const { user, isLoading: authLoading, fetchCurrentUser } = useAuth();

  // Form state - initialized with user data
  const [formState, setFormState] = useState<FormState>({
    username: user?.username || "",
  });
  const [formErrors, setFormErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // ==========================================================================
  // Update form when user changes
  // ==========================================================================

  React.useEffect(() => {
    if (user) {
      setFormState({ username: user.username });
      setHasChanges(false);
    }
  }, [user]);

  // ==========================================================================
  // Handlers
  // ==========================================================================

  /**
   * Update form field
   */
  const handleInputChange = useCallback(
    (field: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = e.target.value;
      setFormState((prev) => ({ ...prev, [field]: newValue }));
      setFormErrors((prev) => ({ ...prev, [field]: undefined, general: undefined }));
      setIsSuccess(false);
      setHasChanges(newValue !== user?.username);
    },
    [user?.username]
  );

  /**
   * Validate form before submission
   */
  const validateForm = useCallback((): boolean => {
    const errors: FormErrors = {};

    // Username validation
    if (!formState.username.trim()) {
      errors.username = "Username is required";
    } else if (formState.username.length < 3) {
      errors.username = "Username must be at least 3 characters";
    } else if (formState.username.length > 50) {
      errors.username = "Username cannot exceed 50 characters";
    } else if (!/^[a-zA-Z0-9_-]+$/.test(formState.username)) {
      errors.username =
        "Username can only contain letters, numbers, underscores, and hyphens";
    } else if (/^[0-9]/.test(formState.username)) {
      errors.username = "Username cannot start with a number";
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  }, [formState.username]);

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

      // Check if there are changes
      if (formState.username === user?.username) {
        setIsSuccess(true);
        return;
      }

      setIsSubmitting(true);

      try {
        const apiConfig = getApiConfig();
        const headers = getApiHeaders();

        const response = await fetch(`${apiConfig.backendApiUrl}/api/v1/users/me`, {
          method: "PUT",
          headers: {
            ...headers,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            username: formState.username.toLowerCase(),
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          const errorMessage =
            data.detail?.error || data.detail || "Failed to update profile";

          if (response.status === 409) {
            setFormErrors({ username: "This username is already taken" });
          } else {
            setFormErrors({ general: errorMessage });
          }
          return;
        }

        // Refresh user data
        const updatedUser = await fetchCurrentUser();

        if (updatedUser && onUpdate) {
          onUpdate(updatedUser);
        }

        setIsSuccess(true);
        setHasChanges(false);
      } catch (error: unknown) {
        const err = error as { message?: string };
        setFormErrors({
          general: err.message || "An unexpected error occurred",
        });
      } finally {
        setIsSubmitting(false);
      }
    },
    [formState.username, user?.username, validateForm, fetchCurrentUser, onUpdate]
  );

  /**
   * Reset form to original values
   */
  const handleReset = useCallback(() => {
    if (user) {
      setFormState({ username: user.username });
      setFormErrors({});
      setIsSuccess(false);
      setHasChanges(false);
    }
  }, [user]);

  // ==========================================================================
  // Loading State
  // ==========================================================================

  if (authLoading || !user) {
    return (
      <div className={cn("space-y-6 animate-pulse", className)}>
        <div className="flex items-center gap-6">
          <div className="h-24 w-24 rounded-full bg-muted" />
          <div className="space-y-3">
            <div className="h-6 w-48 bg-muted rounded" />
            <div className="h-4 w-32 bg-muted rounded" />
            <div className="h-6 w-24 bg-muted rounded" />
          </div>
        </div>
        <div className="h-12 w-full bg-muted rounded" />
        <div className="h-12 w-full bg-muted rounded" />
      </div>
    );
  }

  const initials = getInitials(user.username, user.email);
  const avatarColor = getAvatarColor(user.username);
  const roleConfig = ROLE_CONFIG[user.role];
  const RoleIcon = roleConfig.icon;

  // ==========================================================================
  // Render Form
  // ==========================================================================

  return (
    <form onSubmit={handleSubmit} className={cn("space-y-8", className)}>
      {/* Avatar and User Info Header */}
      <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
        <Avatar className="h-24 w-24 ring-4 ring-primary/20 shadow-xl">
          <AvatarFallback
            className={`${avatarColor} text-white text-2xl font-bold`}
          >
            {initials}
          </AvatarFallback>
        </Avatar>

        <div className="flex flex-col items-center sm:items-start gap-2 text-center sm:text-left">
          <h2 className="text-2xl font-bold text-foreground">{user.username}</h2>
          <p className="text-muted-foreground">{user.email}</p>
          <Badge
            variant="outline"
            className={`${roleConfig.badgeClassName} flex items-center gap-1.5`}
          >
            <RoleIcon className="h-3.5 w-3.5" />
            {roleConfig.label}
          </Badge>
          <p className="text-xs text-muted-foreground mt-1">
            {roleConfig.description}
          </p>
        </div>
      </div>

      <Separator />

      {/* Success Alert */}
      {isSuccess && (
        <Alert className="bg-green-500/10 border-green-500/20 animate-fade-in">
          <CheckCircle2 className="h-4 w-4 text-green-400" />
          <AlertDescription className="text-green-400">
            Profile updated successfully!
          </AlertDescription>
        </Alert>
      )}

      {/* General Error Alert */}
      {formErrors.general && (
        <Alert variant="destructive" className="animate-fade-in">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formErrors.general}</AlertDescription>
        </Alert>
      )}

      {/* Username Field */}
      <div className="space-y-2">
        <Label htmlFor="profile-username" className="text-foreground/90">
          Username
        </Label>
        <div className="relative">
          <AtSign className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="profile-username"
            type="text"
            placeholder="Enter your username"
            value={formState.username}
            onChange={handleInputChange("username")}
            className={cn(
              "pl-10 h-11 bg-background border-input focus:border-primary/50",
              formErrors.username && "border-destructive focus:border-destructive"
            )}
            autoComplete="username"
            disabled={isSubmitting}
            aria-invalid={!!formErrors.username}
            aria-describedby={formErrors.username ? "username-error" : undefined}
          />
        </div>
        {formErrors.username && (
          <p id="username-error" className="text-sm text-destructive animate-fade-in">
            {formErrors.username}
          </p>
        )}
        <p className="text-xs text-muted-foreground">
          3-50 characters, letters, numbers, underscores, and hyphens only
        </p>
      </div>

      {/* Email Field (Read-only) */}
      <div className="space-y-2">
        <Label htmlFor="profile-email" className="text-foreground/90">
          Email Address
        </Label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="profile-email"
            type="email"
            value={user.email}
            className="pl-10 h-11 bg-muted/50 border-input cursor-not-allowed"
            disabled
            readOnly
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Email changes require re-verification. Contact support to update.
        </p>
      </div>

      <Separator />

      {/* Account Information */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/30 border border-border/50">
          <Calendar className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-xs text-muted-foreground">Member Since</p>
            <p className="text-sm font-medium">{formatDate(user.created_at)}</p>
          </div>
        </div>
        <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/30 border border-border/50">
          <Clock className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="text-xs text-muted-foreground">Last Login</p>
            <p className="text-sm font-medium">{formatDate(user.last_login)}</p>
          </div>
        </div>
      </div>

      {/* Account Status */}
      <div className="flex flex-wrap gap-2">
        <Badge
          variant="outline"
          className={
            user.is_verified
              ? "bg-green-500/10 text-green-400 border-green-500/30"
              : "bg-yellow-500/10 text-yellow-400 border-yellow-500/30"
          }
        >
          {user.is_verified ? "Email Verified" : "Email Not Verified"}
        </Badge>
        {user.is_active !== undefined && (
          <Badge
            variant="outline"
            className={
              user.is_active
                ? "bg-green-500/10 text-green-400 border-green-500/30"
                : "bg-red-500/10 text-red-400 border-red-500/30"
            }
          >
            {user.is_active ? "Account Active" : "Account Inactive"}
          </Badge>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-3 pt-4">
        <Button
          type="submit"
          size="lg"
          className="h-11 gap-2 text-base font-medium shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-all"
          disabled={isSubmitting || !hasChanges}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Saving...
            </>
          ) : (
            <>
              <Save className="h-4 w-4" />
              Save Changes
            </>
          )}
        </Button>

        {hasChanges && (
          <Button
            type="button"
            variant="outline"
            size="lg"
            className="h-11"
            onClick={handleReset}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
        )}
      </div>
    </form>
  );
}

export default ProfileForm;
