"use client";

/**
 * Role Guard Component
 *
 * A wrapper component for role-based access control.
 * Handles:
 * - Role verification after authentication
 * - Minimum role level requirements
 * - Access denied UI for insufficient permissions
 * - Optional redirect on access denied
 *
 * @module components/auth/RoleGuard
 */

import { useEffect, ReactNode } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ShieldX, ArrowLeft, Lock } from "lucide-react";

import { useAuth, useMinimumRole, useRequireRole } from "@/hooks/useAuth";
import { type UserRole } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Spinner } from "@/components/enhanced/LoadingStates";
import { cn } from "@/lib/utils";

// =============================================================================
// Types
// =============================================================================

export interface RoleGuardProps {
  /** Content to render when authorized */
  children: ReactNode;
  /** Required role(s) - user must have one of these roles */
  allowedRoles?: UserRole[];
  /** Minimum role required (uses role hierarchy) */
  minimumRole?: UserRole;
  /** Redirect URL when access is denied (if not provided, shows access denied UI) */
  redirectTo?: string;
  /** Custom access denied component */
  fallback?: ReactNode;
  /** Show loading state while checking */
  showLoading?: boolean;
  /** Custom loading component */
  loadingComponent?: ReactNode;
  /** Callback when access is denied */
  onAccessDenied?: () => void;
}

export interface AccessDeniedProps {
  /** Title for the access denied message */
  title?: string;
  /** Description for the access denied message */
  description?: string;
  /** Required role (for display) */
  requiredRole?: UserRole;
  /** User's current role (for display) */
  currentRole?: UserRole;
  /** Show go back button */
  showBackButton?: boolean;
  /** Custom action component */
  action?: ReactNode;
  /** Additional class names */
  className?: string;
}

// =============================================================================
// Access Denied Component
// =============================================================================

/**
 * Default Access Denied UI
 *
 * Displays when a user doesn't have sufficient permissions.
 */
export function AccessDenied({
  title = "Access Denied",
  description = "You don't have permission to access this page.",
  requiredRole,
  currentRole,
  showBackButton = true,
  action,
  className,
}: AccessDeniedProps) {
  const router = useRouter();

  return (
    <div
      className={cn(
        "min-h-screen bg-background flex items-center justify-center p-4",
        className
      )}
    >
      <Card className="w-full max-w-md bg-white/[0.03] backdrop-blur-xl border-white/[0.08] shadow-2xl">
        <CardHeader className="text-center">
          {/* Icon */}
          <div className="mx-auto mb-4 relative">
            <div className="absolute inset-0 blur-2xl bg-destructive/30 rounded-full" />
            <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-destructive/20 to-destructive/5 flex items-center justify-center border border-destructive/20">
              <ShieldX className="w-8 h-8 text-destructive" />
            </div>
          </div>

          <CardTitle className="text-2xl font-bold">{title}</CardTitle>
          <CardDescription className="text-muted-foreground">
            {description}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Role info */}
          {(requiredRole || currentRole) && (
            <div className="rounded-lg bg-white/[0.02] border border-white/[0.08] p-4 space-y-3">
              {currentRole && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Your role:</span>
                  <span className="font-medium capitalize flex items-center gap-2">
                    <Lock className="w-3 h-3" />
                    {currentRole}
                  </span>
                </div>
              )}
              {requiredRole && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Required role:</span>
                  <span className="font-medium text-primary capitalize flex items-center gap-2">
                    <ShieldX className="w-3 h-3" />
                    {requiredRole} or higher
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex flex-col gap-3">
            {action}
            {showBackButton && (
              <Button
                variant="outline"
                onClick={() => router.back()}
                className="w-full gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Go Back
              </Button>
            )}
            <Link href="/dashboard" className="block">
              <Button variant="ghost" className="w-full">
                Return to Dashboard
              </Button>
            </Link>
          </div>

          {/* Help text */}
          <p className="text-center text-xs text-muted-foreground/70">
            If you believe this is an error, please contact your administrator.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

// =============================================================================
// Loading State
// =============================================================================

function RoleLoadingState() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="flex flex-col items-center gap-4 animate-fade-in">
        <Spinner size="lg" />
        <p className="text-sm text-muted-foreground animate-pulse">
          Checking permissions...
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// RoleGuard Component
// =============================================================================

/**
 * RoleGuard Component
 *
 * Wraps content that requires specific roles. Works in two modes:
 * 1. allowedRoles: User must have one of the specified roles
 * 2. minimumRole: User must have at least this role level (uses hierarchy)
 *
 * IMPORTANT: This component assumes the user is already authenticated.
 * Use it inside a ProtectedRoute or after authentication check.
 *
 * @example
 * ```tsx
 * // Require specific roles
 * <RoleGuard allowedRoles={['admin']}>
 *   <AdminPanel />
 * </RoleGuard>
 *
 * // Multiple allowed roles
 * <RoleGuard allowedRoles={['admin', 'researcher']}>
 *   <ResearchTools />
 * </RoleGuard>
 *
 * // Minimum role level (viewer < researcher < admin)
 * <RoleGuard minimumRole="researcher">
 *   <ResearcherContent />
 * </RoleGuard>
 *
 * // With redirect instead of access denied UI
 * <RoleGuard allowedRoles={['admin']} redirectTo="/dashboard">
 *   <AdminContent />
 * </RoleGuard>
 *
 * // With custom fallback
 * <RoleGuard
 *   allowedRoles={['admin']}
 *   fallback={<UpgradePrompt />}
 * >
 *   <PremiumFeature />
 * </RoleGuard>
 * ```
 */
export function RoleGuard({
  children,
  allowedRoles,
  minimumRole,
  redirectTo,
  fallback,
  showLoading = true,
  loadingComponent,
  onAccessDenied,
}: RoleGuardProps) {
  const router = useRouter();
  const { user, isLoading, isInitialized, isAuthenticated } = useAuth();

  // Check role access
  const hasRequiredRole = useRequireRole(allowedRoles || []);
  const hasMinimumRole = useMinimumRole(minimumRole || "viewer");

  // Determine if user has access
  const hasAccess = allowedRoles
    ? hasRequiredRole
    : minimumRole
      ? hasMinimumRole
      : true;

  // Handle redirect on access denied
  useEffect(() => {
    if (isInitialized && !isLoading && isAuthenticated && !hasAccess) {
      // Call access denied callback
      if (onAccessDenied) {
        onAccessDenied();
      }

      // Redirect if URL provided
      if (redirectTo) {
        router.replace(redirectTo);
      }
    }
  }, [isInitialized, isLoading, isAuthenticated, hasAccess, redirectTo, router, onAccessDenied]);

  // Show loading while checking
  if (!isInitialized || isLoading) {
    if (!showLoading) {
      return null;
    }
    if (loadingComponent) {
      return <>{loadingComponent}</>;
    }
    return <RoleLoadingState />;
  }

  // If not authenticated, don't render (should be handled by ProtectedRoute)
  if (!isAuthenticated) {
    return null;
  }

  // Access denied
  if (!hasAccess) {
    // If redirecting, show loading
    if (redirectTo) {
      if (loadingComponent) {
        return <>{loadingComponent}</>;
      }
      return <RoleLoadingState />;
    }

    // Show custom fallback
    if (fallback) {
      return <>{fallback}</>;
    }

    // Show default access denied
    return (
      <AccessDenied
        requiredRole={minimumRole || allowedRoles?.[0]}
        currentRole={user?.role}
      />
    );
  }

  // User has access, render children
  return <>{children}</>;
}

// =============================================================================
// Higher-Order Component (HOC) Version
// =============================================================================

export interface WithRoleGuardOptions {
  /** Required role(s) */
  allowedRoles?: UserRole[];
  /** Minimum role required */
  minimumRole?: UserRole;
  /** Redirect URL when denied */
  redirectTo?: string;
  /** Custom fallback component */
  fallback?: ReactNode;
}

/**
 * Higher-Order Component for role-based access control
 *
 * @example
 * ```tsx
 * // Admin-only component
 * const AdminPanel = withRoleGuard(AdminPanelContent, {
 *   allowedRoles: ['admin'],
 * });
 *
 * // Researcher or above
 * const ResearchTools = withRoleGuard(ResearchToolsContent, {
 *   minimumRole: 'researcher',
 * });
 * ```
 */
export function withRoleGuard<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: WithRoleGuardOptions
) {
  const WithRoleGuard = (props: P) => {
    return (
      <RoleGuard
        allowedRoles={options.allowedRoles}
        minimumRole={options.minimumRole}
        redirectTo={options.redirectTo}
        fallback={options.fallback}
      >
        <WrappedComponent {...props} />
      </RoleGuard>
    );
  };

  // Set display name for debugging
  const wrappedComponentName =
    WrappedComponent.displayName || WrappedComponent.name || "Component";
  WithRoleGuard.displayName = `withRoleGuard(${wrappedComponentName})`;

  return WithRoleGuard;
}

// =============================================================================
// Convenience Components
// =============================================================================

/**
 * Admin-only guard component
 *
 * @example
 * ```tsx
 * <AdminOnly>
 *   <AdminSettings />
 * </AdminOnly>
 * ```
 */
export function AdminOnly({
  children,
  fallback,
  redirectTo,
}: {
  children: ReactNode;
  fallback?: ReactNode;
  redirectTo?: string;
}) {
  return (
    <RoleGuard
      allowedRoles={["admin"]}
      fallback={fallback}
      redirectTo={redirectTo}
    >
      {children}
    </RoleGuard>
  );
}

/**
 * Researcher or above guard component
 *
 * @example
 * ```tsx
 * <ResearcherOnly>
 *   <ResearchTools />
 * </ResearcherOnly>
 * ```
 */
export function ResearcherOnly({
  children,
  fallback,
  redirectTo,
}: {
  children: ReactNode;
  fallback?: ReactNode;
  redirectTo?: string;
}) {
  return (
    <RoleGuard
      minimumRole="researcher"
      fallback={fallback}
      redirectTo={redirectTo}
    >
      {children}
    </RoleGuard>
  );
}

export default RoleGuard;
