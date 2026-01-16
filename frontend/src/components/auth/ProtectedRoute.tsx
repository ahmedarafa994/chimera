"use client";

/**
 * Protected Route Component
 *
 * A wrapper component that protects routes requiring authentication.
 * Handles:
 * - Authentication status checking
 * - Redirect to login for unauthenticated users
 * - Preserving intended destination via returnUrl
 * - Loading state display during auth initialization
 *
 * @module components/auth/ProtectedRoute
 */

import { useEffect, ReactNode } from "react";
import { useRouter, usePathname } from "next/navigation";

import { useAuth } from "@/hooks/useAuth";
import { Spinner } from "@/components/enhanced/LoadingStates";

// =============================================================================
// Types
// =============================================================================

export interface ProtectedRouteProps {
  /** Content to render when authenticated */
  children: ReactNode;
  /** Custom redirect URL for unauthenticated users (default: /login) */
  redirectTo?: string;
  /** Custom loading component */
  loadingComponent?: ReactNode;
  /** Custom loading message */
  loadingMessage?: string;
  /** Callback when redirect happens */
  onRedirect?: () => void;
}

// =============================================================================
// Loading Component
// =============================================================================

function DefaultLoadingState({ message }: { message?: string }) {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="flex flex-col items-center gap-4 animate-fade-in">
        <Spinner size="lg" />
        <p className="text-sm text-muted-foreground animate-pulse">
          {message || "Verifying authentication..."}
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// Component
// =============================================================================

/**
 * ProtectedRoute Component
 *
 * Wraps content that requires authentication. If the user is not authenticated,
 * they are redirected to the login page with the current path preserved as a
 * return URL query parameter.
 *
 * @example
 * ```tsx
 * // Basic usage in a page
 * export default function DashboardPage() {
 *   return (
 *     <ProtectedRoute>
 *       <DashboardContent />
 *     </ProtectedRoute>
 *   );
 * }
 *
 * // With custom loading
 * <ProtectedRoute
 *   loadingComponent={<CustomLoader />}
 *   loadingMessage="Loading your profile..."
 * >
 *   <ProfileContent />
 * </ProtectedRoute>
 *
 * // With custom redirect
 * <ProtectedRoute redirectTo="/auth/signin">
 *   <SecretContent />
 * </ProtectedRoute>
 * ```
 */
export function ProtectedRoute({
  children,
  redirectTo = "/login",
  loadingComponent,
  loadingMessage,
  onRedirect,
}: ProtectedRouteProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, isLoading, isInitialized } = useAuth();

  // Handle redirect for unauthenticated users
  useEffect(() => {
    if (isInitialized && !isLoading && !isAuthenticated) {
      // Build redirect URL with return path
      const returnUrl = encodeURIComponent(pathname);
      const loginUrl = `${redirectTo}?redirect=${returnUrl}`;

      // Call redirect callback if provided
      if (onRedirect) {
        onRedirect();
      }

      router.replace(loginUrl);
    }
  }, [isInitialized, isLoading, isAuthenticated, pathname, redirectTo, router, onRedirect]);

  // Show loading state while initializing or if not yet authenticated
  if (!isInitialized || isLoading) {
    if (loadingComponent) {
      return <>{loadingComponent}</>;
    }
    return <DefaultLoadingState message={loadingMessage} />;
  }

  // Don't render children while redirecting
  if (!isAuthenticated) {
    if (loadingComponent) {
      return <>{loadingComponent}</>;
    }
    return <DefaultLoadingState message="Redirecting to login..." />;
  }

  // User is authenticated, render children
  return <>{children}</>;
}

// =============================================================================
// Higher-Order Component (HOC) Version
// =============================================================================

export interface WithProtectedRouteOptions {
  /** Custom redirect URL */
  redirectTo?: string;
  /** Custom loading component */
  loadingComponent?: ReactNode;
  /** Custom loading message */
  loadingMessage?: string;
}

/**
 * Higher-Order Component for protecting routes
 *
 * @example
 * ```tsx
 * // Wrap a component
 * const ProtectedDashboard = withProtectedRoute(DashboardContent);
 *
 * // With options
 * const ProtectedAdmin = withProtectedRoute(AdminPanel, {
 *   redirectTo: '/auth/signin',
 *   loadingMessage: 'Checking admin access...',
 * });
 * ```
 */
export function withProtectedRoute<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: WithProtectedRouteOptions = {}
) {
  const WithProtectedRoute = (props: P) => {
    return (
      <ProtectedRoute
        redirectTo={options.redirectTo}
        loadingComponent={options.loadingComponent}
        loadingMessage={options.loadingMessage}
      >
        <WrappedComponent {...props} />
      </ProtectedRoute>
    );
  };

  // Set display name for debugging
  const wrappedComponentName =
    WrappedComponent.displayName || WrappedComponent.name || "Component";
  WithProtectedRoute.displayName = `withProtectedRoute(${wrappedComponentName})`;

  return WithProtectedRoute;
}

// =============================================================================
// Utility Hook
// =============================================================================

/**
 * Hook to check authentication and redirect if needed
 *
 * Use this for programmatic protection where component wrapper isn't suitable.
 *
 * @example
 * ```tsx
 * function MyPage() {
 *   const { isProtected, isChecking } = useProtectedPage();
 *
 *   if (isChecking || !isProtected) {
 *     return <Loading />;
 *   }
 *
 *   return <Content />;
 * }
 * ```
 */
export function useProtectedPage(redirectTo = "/login") {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, isLoading, isInitialized } = useAuth();

  useEffect(() => {
    if (isInitialized && !isLoading && !isAuthenticated) {
      const returnUrl = encodeURIComponent(pathname);
      router.replace(`${redirectTo}?redirect=${returnUrl}`);
    }
  }, [isInitialized, isLoading, isAuthenticated, pathname, redirectTo, router]);

  return {
    /** Whether auth is being checked */
    isChecking: !isInitialized || isLoading,
    /** Whether user is authenticated (protected) */
    isProtected: isAuthenticated,
  };
}

export default ProtectedRoute;
