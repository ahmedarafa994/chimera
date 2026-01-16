"use client";

/**
 * Email Verification Page
 *
 * Handles email verification from email link with:
 * - Token extraction from URL query params
 * - Auto-verification on page load
 * - Success/failure/loading states
 * - Premium glassmorphism design matching login page
 * - Link to login on success
 * - Option to resend verification if expired
 * - Redirect if already authenticated
 *
 * @module app/verify-email/page
 */

import { useEffect, useState, Suspense, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ShieldAlert,
  ArrowLeft,
  Mail,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Loader2,
  RefreshCcw,
} from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";

// =============================================================================
// Types
// =============================================================================

type VerificationState =
  | "loading"
  | "success"
  | "error"
  | "expired"
  | "missing_token";

interface VerificationResult {
  state: VerificationState;
  message: string;
  email?: string;
  username?: string;
}

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
// Loading State Component
// =============================================================================

function VerifyingState() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AnimatedBackground />

      {/* Header */}
      <header className="relative z-10 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="w-[100px]" /> {/* Spacer for balance */}

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
                <div className="absolute inset-0 blur-2xl bg-primary/30 rounded-full" />
                <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center border border-primary/20">
                  <Loader2 className="w-8 h-8 text-primary animate-spin" />
                </div>
              </div>

              <CardTitle className="text-2xl font-bold">
                Verifying Your Email
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Please wait while we verify your email address...
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4">
              <div className="flex justify-center">
                <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                  <div className="h-full w-1/2 bg-primary rounded-full animate-pulse" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// Success State Component
// =============================================================================

function VerificationSuccess({
  email,
  username,
}: {
  email?: string;
  username?: string;
}) {
  const router = useRouter();
  const [countdown, setCountdown] = useState(5);

  // Countdown and auto-redirect
  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    } else {
      router.replace("/login?verified=true");
    }
  }, [countdown, router]);

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <AnimatedBackground />

      {/* Header */}
      <header className="relative z-10 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="w-[100px]" /> {/* Spacer for balance */}

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
                <div className="absolute inset-0 blur-2xl bg-green-500/30 rounded-full" />
                <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-green-500/20 to-green-500/5 flex items-center justify-center border border-green-500/20">
                  <CheckCircle2 className="w-8 h-8 text-green-400" />
                </div>
              </div>

              <CardTitle className="text-2xl font-bold">
                Email Verified!
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Your email has been successfully verified
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4 space-y-6">
              <Alert className="bg-green-500/10 border-green-500/20 text-green-400">
                <CheckCircle2 className="h-4 w-4" />
                <AlertTitle>Verification Complete</AlertTitle>
                <AlertDescription>
                  {username ? (
                    <>
                      Welcome, <strong>{username}</strong>! Your account is now
                      active and you can sign in.
                    </>
                  ) : (
                    "Your account is now active and you can sign in."
                  )}
                </AlertDescription>
              </Alert>

              {email && (
                <div className="text-center text-sm text-muted-foreground">
                  Verified email: <strong>{email}</strong>
                </div>
              )}

              <div className="flex flex-col gap-3">
                <Link href="/login?verified=true" className="w-full">
                  <Button
                    type="button"
                    size="lg"
                    className="w-full h-11 gap-2 text-base font-medium"
                  >
                    Sign In Now
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>

                <p className="text-center text-xs text-muted-foreground/70">
                  Redirecting to login in {countdown} seconds...
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

// =============================================================================
// Error State Component (with Resend Option)
// =============================================================================

function VerificationError({
  isExpired,
  errorMessage,
}: {
  isExpired: boolean;
  errorMessage: string;
}) {
  const { resendVerificationEmail } = useAuth();
  const [email, setEmail] = useState("");
  const [isResending, setIsResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);
  const [resendError, setResendError] = useState<string | null>(null);
  const [showResendForm, setShowResendForm] = useState(false);

  const handleResend = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email.trim()) {
      setResendError("Please enter your email address");
      return;
    }

    setIsResending(true);
    setResendError(null);

    try {
      const success = await resendVerificationEmail(email.trim());
      if (success) {
        setResendSuccess(true);
      } else {
        setResendError("Failed to send verification email. Please try again.");
      }
    } catch {
      setResendError("An unexpected error occurred. Please try again.");
    } finally {
      setIsResending(false);
    }
  };

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
                {isExpired ? "Verification Link Expired" : "Verification Failed"}
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                {isExpired
                  ? "This verification link has expired"
                  : "We couldn't verify your email"}
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4 space-y-6">
              {!resendSuccess ? (
                <>
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>
                      {isExpired ? "Link Expired" : "Verification Error"}
                    </AlertTitle>
                    <AlertDescription>{errorMessage}</AlertDescription>
                  </Alert>

                  {!showResendForm ? (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        {isExpired
                          ? "Don't worry! You can request a new verification email."
                          : "Please try the following:"}
                      </p>
                      {!isExpired && (
                        <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1 pl-2">
                          <li>Make sure you copied the complete link from your email</li>
                          <li>Check that the link hasn&apos;t been modified</li>
                          <li>Request a new verification email if needed</li>
                        </ul>
                      )}

                      <div className="flex flex-col gap-3 pt-2">
                        <Button
                          type="button"
                          size="lg"
                          onClick={() => setShowResendForm(true)}
                          className="w-full h-11 gap-2 text-base font-medium"
                        >
                          <RefreshCcw className="h-4 w-4" />
                          Resend Verification Email
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
                  ) : (
                    <form onSubmit={handleResend} className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="resend-email">Email Address</Label>
                        <div className="relative">
                          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                          <Input
                            id="resend-email"
                            type="email"
                            placeholder="Enter your email"
                            value={email}
                            onChange={(e) => {
                              setEmail(e.target.value);
                              setResendError(null);
                            }}
                            className={cn(
                              "h-11 pl-10 bg-white/[0.02] border-white/[0.08]",
                              "focus-visible:ring-primary/50",
                              resendError && "border-red-500/50"
                            )}
                            disabled={isResending}
                          />
                        </div>
                        {resendError && (
                          <p className="text-xs text-red-400">{resendError}</p>
                        )}
                      </div>

                      <div className="flex flex-col gap-2">
                        <Button
                          type="submit"
                          size="lg"
                          disabled={isResending}
                          className="w-full h-11 gap-2 text-base font-medium"
                        >
                          {isResending ? (
                            <>
                              <Loader2 className="h-4 w-4 animate-spin" />
                              Sending...
                            </>
                          ) : (
                            <>
                              <Mail className="h-4 w-4" />
                              Send Verification Email
                            </>
                          )}
                        </Button>

                        <Button
                          type="button"
                          variant="ghost"
                          onClick={() => {
                            setShowResendForm(false);
                            setResendError(null);
                          }}
                          className="w-full h-11 gap-2 text-muted-foreground hover:text-foreground"
                          disabled={isResending}
                        >
                          Cancel
                        </Button>
                      </div>
                    </form>
                  )}
                </>
              ) : (
                <div className="space-y-4">
                  <Alert className="bg-green-500/10 border-green-500/20 text-green-400">
                    <CheckCircle2 className="h-4 w-4" />
                    <AlertTitle>Email Sent!</AlertTitle>
                    <AlertDescription>
                      If <strong>{email}</strong> is registered in our system,
                      you&apos;ll receive a new verification email shortly.
                    </AlertDescription>
                  </Alert>

                  <div className="flex flex-col gap-3">
                    <p className="text-sm text-muted-foreground text-center">
                      Please check your inbox and spam folder for the verification
                      link.
                    </p>

                    <Link href="/login" className="w-full">
                      <Button
                        type="button"
                        size="lg"
                        className="w-full h-11 gap-2 text-base font-medium"
                      >
                        <ArrowRight className="h-4 w-4" />
                        Go to Login
                      </Button>
                    </Link>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
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
                <div className="absolute inset-0 blur-2xl bg-amber-500/30 rounded-full" />
                <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-500/20 to-amber-500/5 flex items-center justify-center border border-amber-500/20">
                  <Mail className="w-8 h-8 text-amber-400" />
                </div>
              </div>

              <CardTitle className="text-2xl font-bold">
                Missing Verification Token
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                The verification link appears to be incomplete
              </CardDescription>
            </CardHeader>

            <CardContent className="pt-4 space-y-6">
              <Alert className="bg-amber-500/10 border-amber-500/20 text-amber-400">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Link Problem</AlertTitle>
                <AlertDescription>
                  This verification link is missing the required token. This can
                  happen if the link was not copied correctly from your email.
                </AlertDescription>
              </Alert>

              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Please try one of the following:
                </p>
                <ul className="text-sm text-muted-foreground list-disc list-inside space-y-1 pl-2">
                  <li>Copy the complete link from your verification email</li>
                  <li>Click the link directly from your email</li>
                  <li>Request a new verification email if the issue persists</li>
                </ul>
              </div>

              <div className="flex flex-col gap-3">
                <Link href="/register" className="w-full">
                  <Button
                    type="button"
                    size="lg"
                    className="w-full h-11 gap-2 text-base font-medium"
                  >
                    Register Again
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
// Verify Email Page Content (with search params)
// =============================================================================

function VerifyEmailPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { verifyEmail, isAuthenticated, isLoading: authLoading, isInitialized } = useAuth();
  const [mounted, setMounted] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null);

  // Get token from query params
  const token = searchParams.get("token") || "";

  // Perform verification
  const performVerification = useCallback(async (verifyToken: string) => {
    try {
      const success = await verifyEmail(verifyToken);

      if (success) {
        setVerificationResult({
          state: "success",
          message: "Your email has been verified successfully!",
        });
      } else {
        // Check if it might be expired
        setVerificationResult({
          state: "error",
          message:
            "This verification link is invalid or has already been used. Please request a new verification email.",
        });
      }
    } catch (error: unknown) {
      const err = error as { token_expired?: boolean };
      // Check for expired token error
      if (err?.token_expired) {
        setVerificationResult({
          state: "expired",
          message:
            "This verification link has expired. Verification links are valid for 24 hours. Please request a new one.",
        });
      } else {
        setVerificationResult({
          state: "error",
          message:
            "An error occurred while verifying your email. Please try again or request a new verification link.",
        });
      }
    }
  }, [verifyEmail]);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Redirect if already authenticated
  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      router.replace("/dashboard");
    }
  }, [isInitialized, isAuthenticated, router]);

  // Perform verification when token is available
  useEffect(() => {
    if (mounted && token && !verificationResult) {
      performVerification(token);
    }
  }, [mounted, token, verificationResult, performVerification]);

  // Show global loading state
  if (!mounted || authLoading || (isInitialized && isAuthenticated)) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Show missing token error
  if (!token) {
    return <MissingTokenError />;
  }

  // Show verification states
  if (!verificationResult) {
    return <VerifyingState />;
  }

  if (verificationResult.state === "success") {
    return (
      <VerificationSuccess
        email={verificationResult.email}
        username={verificationResult.username}
      />
    );
  }

  if (verificationResult.state === "expired" || verificationResult.state === "error") {
    return (
      <VerificationError
        isExpired={verificationResult.state === "expired"}
        errorMessage={verificationResult.message}
      />
    );
  }

  // Fallback - should not reach here
  return <VerifyingState />;
}

// =============================================================================
// Main Page Export with Suspense
// =============================================================================

export default function VerifyEmailPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-background flex items-center justify-center">
          <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <VerifyEmailPageContent />
    </Suspense>
  );
}
