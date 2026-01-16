"use client";

/**
 * Login Page
 *
 * Authentication page for user login with:
 * - Premium glassmorphism design
 * - Animated background effects
 * - Login form with validation
 * - Links to register and forgot password
 * - Redirect on success
 *
 * @module app/login/page
 */

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { ShieldAlert, ArrowLeft } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import { LoginForm } from "@/components/auth/LoginForm";

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
// Login Page Content (with search params)
// =============================================================================

function LoginPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isAuthenticated, isLoading, isInitialized } = useAuth();
  const [mounted, setMounted] = useState(false);

  // Get redirect URL from query params
  const redirectTo = searchParams.get("redirect") || "/dashboard";
  const registered = searchParams.get("registered") === "true";
  const verified = searchParams.get("verified") === "true";
  const passwordReset = searchParams.get("password_reset") === "true";
  const sessionExpired = searchParams.get("session_expired") === "true";

  useEffect(() => {
    setMounted(true);
  }, []);

  // Redirect if already authenticated
  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      router.replace(redirectTo);
    }
  }, [isInitialized, isAuthenticated, router, redirectTo]);

  // Show loading state
  if (!mounted || isLoading || (isInitialized && isAuthenticated)) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AnimatedBackground />

      {/* Header */}
      <header className="relative z-10 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <Link
            href="/"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm font-medium">Back to Home</span>
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
          {/* Success Messages */}
          {registered && (
            <div className="mb-6 p-4 rounded-lg bg-green-500/10 border border-green-500/20 text-green-400 text-sm animate-fade-in">
              <strong>Registration successful!</strong> Please check your email to verify
              your account, then log in.
            </div>
          )}

          {verified && (
            <div className="mb-6 p-4 rounded-lg bg-green-500/10 border border-green-500/20 text-green-400 text-sm animate-fade-in">
              <strong>Email verified!</strong> You can now log in to your account.
            </div>
          )}

          {passwordReset && (
            <div className="mb-6 p-4 rounded-lg bg-green-500/10 border border-green-500/20 text-green-400 text-sm animate-fade-in">
              <strong>Password reset successful!</strong> You can now log in with your
              new password.
            </div>
          )}

          {sessionExpired && (
            <div className="mb-6 p-4 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm animate-fade-in">
              <strong>Session expired.</strong> Please sign in again to continue.
            </div>
          )}

          {/* Login Card */}
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
                  <ShieldAlert className="w-8 h-8 text-primary" />
                </div>
              </div>

              <CardTitle className="text-2xl font-bold">
                Welcome Back
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Sign in to access your account
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4">
              <LoginForm redirectTo={redirectTo} />
            </CardContent>
          </Card>

          {/* Footer */}
          <p className="mt-8 text-center text-xs text-muted-foreground/70">
            By signing in, you agree to our{" "}
            <Link href="/terms" className="hover:text-primary transition-colors">
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link href="/privacy" className="hover:text-primary transition-colors">
              Privacy Policy
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

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <LoginPageContent />
    </Suspense>
  );
}
