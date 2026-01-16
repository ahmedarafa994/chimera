"use client";

/**
 * Reset Password Page
 *
 * Password reset page with token validation:
 * - Premium glassmorphism design matching login page
 * - Animated background effects
 * - New password form with strength indicator
 * - Confirm password field
 * - Token validation and error handling
 * - Redirect if already authenticated
 *
 * @module app/reset-password/page
 */

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { ShieldAlert, ArrowLeft, Lock, AlertCircle, ArrowRight } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import { ResetPasswordForm } from "@/components/auth/ResetPasswordForm";

// =============================================================================
// Background Component
// =============================================================================

function AnimatedBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      {/* Gradient orbs */}
      <div className="absolute top-1/4 -left-32 w-96 h-96 bg-primary/20 rounded-full blur-[100px] animate-pulse" />
      <div
        className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/15 rounded-full blur-[100px] animate-pulse"
        style={{ animationDelay: "2s" }}
      />
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-[150px]"
      />

      {/* Grid pattern overlay */}
      <div
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: "50px 50px",
        }}
      />
    </div>
  );
}

// =============================================================================
// Missing Token Component
// =============================================================================

function MissingTokenError() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AnimatedBackground />

      {/* Header */}
      <header className="relative z-10 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link
            href="/login"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm font-medium">Back to Login</span>
          </Link>

          <Link href="/" className="flex items-center gap-2">
            <ShieldAlert className="w-6 h-6 text-primary" />
            <span className="text-lg font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
              Chimera
            </span>
          </Link>

          <div className="w-[100px]" /> {/* Spacer for balance */}
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex items-center justify-center p-4 sm:p-8">
        <div className="w-full max-w-md animate-fade-in-up">
          <Card
            className={cn(
              "bg-white/[0.03] backdrop-blur-xl border-white/[0.08]",
              "shadow-2xl shadow-black/20"
            )}
          >
            <CardHeader className="text-center pb-2">
              <div className="mx-auto mb-4 relative">
                <div className="absolute inset-0 blur-2xl bg-red-500/30 rounded-full" />
                <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-red-500/20 to-red-500/5 flex items-center justify-center border border-red-500/20">
                  <AlertCircle className="w-8 h-8 text-red-400" />
                </div>
              </div>

              <CardTitle className="text-2xl font-bold">
                Invalid Reset Link
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                This password reset link is invalid or has expired
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4 space-y-6">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Missing Reset Token</AlertTitle>
                <AlertDescription>
                  The password reset link you followed is missing the required
                  token. This can happen if the link was not copied correctly
                  from your email.
                </AlertDescription>
              </Alert>

              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Please try one of the following:
                </p>
                <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1 pl-2">
                  <li>Copy the complete link from your email</li>
                  <li>Request a new password reset link</li>
                  <li>Contact support if the issue persists</li>
                </ul>
              </div>

              <div className="flex flex-col gap-3">
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
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// Reset Password Page Content (with search params)
// =============================================================================

function ResetPasswordPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isAuthenticated, isLoading, isInitialized } = useAuth();
  const [mounted, setMounted] = useState(false);

  // Get token from query params
  const token = searchParams.get("token") || "";

  useEffect(() => {
    setMounted(true);
  }, []);

  // Redirect if already authenticated
  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      router.replace("/dashboard");
    }
  }, [isInitialized, isAuthenticated, router]);

  // Show loading state
  if (!mounted || isLoading || (isInitialized && isAuthenticated)) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Show error if no token provided
  if (!token) {
    return <MissingTokenError />;
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AnimatedBackground />

      {/* Header */}
      <header className="relative z-10 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link
            href="/login"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm font-medium">Back to Login</span>
          </Link>

          <Link href="/" className="flex items-center gap-2">
            <ShieldAlert className="w-6 h-6 text-primary" />
            <span className="text-lg font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
              Chimera
            </span>
          </Link>

          <div className="w-[100px]" /> {/* Spacer for balance */}
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex items-center justify-center p-4 sm:p-8">
        <div className="w-full max-w-md animate-fade-in-up">
          {/* Reset Password Card */}
          <Card
            className={cn(
              "bg-white/[0.03] backdrop-blur-xl border-white/[0.08]",
              "shadow-2xl shadow-black/20"
            )}
          >
            <CardHeader className="text-center pb-2">
              {/* Logo */}
              <div className="mx-auto mb-4 relative">
                <div className="absolute inset-0 blur-2xl bg-primary/30 rounded-full" />
                <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center border border-primary/20">
                  <Lock className="w-8 h-8 text-primary" />
                </div>
              </div>

              <CardTitle className="text-2xl font-bold">
                Reset Password
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Create a new password for your account
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4">
              <ResetPasswordForm token={token} />
            </CardContent>
          </Card>

          {/* Footer */}
          <p className="mt-8 text-center text-xs text-muted-foreground/70">
            Remember your password?{" "}
            <Link href="/login" className="hover:text-primary transition-colors">
              Sign in instead
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// Main Page Export with Suspense
// =============================================================================

export default function ResetPasswordPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <ResetPasswordPageContent />
    </Suspense>
  );
}
