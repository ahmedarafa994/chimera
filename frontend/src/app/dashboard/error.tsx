'use client';

import { useEffect, useCallback } from 'react';
import { AlertCircle, RefreshCw, ArrowLeft, LogIn, ShieldX } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';

interface DashboardErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

/**
 * Dashboard Error Boundary
 *
 * Handles errors in the dashboard, with special handling for:
 * - Session expiry / authentication errors (401/403)
 * - Network errors
 * - General application errors
 */
export default function DashboardError({ error, reset }: DashboardErrorProps) {
  const router = useRouter();
  const { isAuthenticated, logout, refreshToken } = useAuth();

  // Check if this is an authentication-related error
  const isAuthError = error.message?.toLowerCase().includes('unauthorized') ||
                      error.message?.toLowerCase().includes('401') ||
                      error.message?.toLowerCase().includes('403') ||
                      error.message?.toLowerCase().includes('session') ||
                      error.message?.toLowerCase().includes('token') ||
                      error.message?.toLowerCase().includes('authentication');

  // Check if this is a network error
  const isNetworkError = error.message?.toLowerCase().includes('network') ||
                         error.message?.toLowerCase().includes('fetch') ||
                         error.message?.toLowerCase().includes('connection');

  useEffect(() => {
    // Log the error for debugging (but not in production)
    if (process.env.NODE_ENV === 'development') {
      console.error('Dashboard error:', error);
    }
  }, [error]);

  // Handle session refresh attempt
  const handleRefreshSession = useCallback(async () => {
    try {
      const success = await refreshToken();
      if (success) {
        // Token refreshed successfully, retry the operation
        reset();
      } else {
        // Refresh failed, redirect to login
        router.push('/login?session_expired=true');
      }
    } catch {
      // Refresh threw an error, redirect to login
      router.push('/login?session_expired=true');
    }
  }, [refreshToken, reset, router]);

  // Handle logout and redirect to login
  const handleLogout = useCallback(async () => {
    try {
      await logout();
    } catch {
      // Ignore logout errors
    }
    router.push('/login?session_expired=true');
  }, [logout, router]);

  // Render authentication error UI
  if (isAuthError) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center p-4">
        <div className="w-full max-w-lg space-y-6 text-center">
          <div className="flex justify-center">
            <div className="rounded-full bg-amber-500/10 p-4">
              <ShieldX className="h-10 w-10 text-amber-500" aria-hidden="true" />
            </div>
          </div>

          <div className="space-y-2">
            <h2 className="text-xl font-semibold">Session Expired</h2>
            <p className="text-sm text-muted-foreground">
              Your session has expired or you don&apos;t have permission to access this page.
              Please sign in again to continue.
            </p>
          </div>

          {process.env.NODE_ENV === 'development' && error.message && (
            <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-left">
              <p className="text-xs font-mono text-amber-500 break-all">
                {error.message}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-2 sm:flex-row sm:justify-center">
            <Button
              onClick={handleRefreshSession}
              variant="default"
              size="sm"
              className="inline-flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Refresh Session
            </Button>

            <Button
              onClick={handleLogout}
              variant="outline"
              size="sm"
              className="inline-flex items-center gap-2"
            >
              <LogIn className="h-4 w-4" aria-hidden="true" />
              Sign In Again
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Render network error UI
  if (isNetworkError) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center p-4">
        <div className="w-full max-w-lg space-y-6 text-center">
          <div className="flex justify-center">
            <div className="rounded-full bg-blue-500/10 p-4">
              <AlertCircle className="h-10 w-10 text-blue-500" aria-hidden="true" />
            </div>
          </div>

          <div className="space-y-2">
            <h2 className="text-xl font-semibold">Connection Error</h2>
            <p className="text-sm text-muted-foreground">
              Unable to connect to the server. Please check your internet connection
              and try again.
            </p>
          </div>

          {process.env.NODE_ENV === 'development' && error.message && (
            <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3 text-left">
              <p className="text-xs font-mono text-blue-500 break-all">
                {error.message}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-2 sm:flex-row sm:justify-center">
            <Button
              onClick={reset}
              variant="default"
              size="sm"
              className="inline-flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Default error UI
  return (
    <div className="flex min-h-[60vh] items-center justify-center p-4">
      <div className="w-full max-w-lg space-y-6 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-destructive/10 p-4">
            <AlertCircle className="h-10 w-10 text-destructive" aria-hidden="true" />
          </div>
        </div>

        <div className="space-y-2">
          <h2 className="text-xl font-semibold">Dashboard Error</h2>
          <p className="text-sm text-muted-foreground">
            Unable to load the dashboard. Please try refreshing the page.
          </p>
        </div>

        {process.env.NODE_ENV === 'development' && error.message && (
          <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-left">
            <p className="text-xs font-mono text-destructive break-all">
              {error.message}
            </p>
          </div>
        )}

        <div className="flex flex-col gap-2 sm:flex-row sm:justify-center">
          <Button
            onClick={reset}
            variant="default"
            size="sm"
            className="inline-flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
            Retry
          </Button>

          <Button
            onClick={() => router.push('/')}
            variant="outline"
            size="sm"
            className="inline-flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            Go Back
          </Button>
        </div>
      </div>
    </div>
  );
}
