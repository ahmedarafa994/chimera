"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, RefreshCw, Home, Bug } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Global Error:", error);
    
    // In production, you would send this to an error tracking service
    // e.g., Sentry, LogRocket, etc.
    if (process.env.NODE_ENV === "production") {
      // sendToErrorTracking(error);
    }
  }, [error]);

  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="max-w-lg w-full">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10">
              <AlertTriangle className="h-8 w-8 text-destructive" />
            </div>
            <CardTitle className="text-2xl">Something went wrong!</CardTitle>
            <CardDescription>
              An unexpected error occurred. Our team has been notified.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Error Details (only in development) */}
            {process.env.NODE_ENV === "development" && (
              <div className="rounded-md bg-muted p-4 space-y-2">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Bug className="h-4 w-4" />
                  Error Details
                </div>
                <p className="text-sm text-muted-foreground font-mono break-all">
                  {error.message}
                </p>
                {error.digest && (
                  <p className="text-xs text-muted-foreground">
                    Error ID: {error.digest}
                  </p>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-2">
              <Button onClick={reset} className="flex-1">
                <RefreshCw className="mr-2 h-4 w-4" />
                Try Again
              </Button>
              <Button 
                variant="outline" 
                onClick={() => window.location.href = "/"}
                className="flex-1"
              >
                <Home className="mr-2 h-4 w-4" />
                Go Home
              </Button>
            </div>

            {/* Support Information */}
            <p className="text-center text-xs text-muted-foreground">
              If this problem persists, please contact support with error ID:{" "}
              <code className="bg-muted px-1 py-0.5 rounded">
                {error.digest || "unknown"}
              </code>
            </p>
          </CardContent>
        </Card>
      </body>
    </html>
  );
}
