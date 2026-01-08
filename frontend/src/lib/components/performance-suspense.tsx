'use client';

import React, { Suspense, Component, ReactNode } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface PerformantSuspenseProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  errorFallback?: React.ComponentType<{ error: Error; resetErrorBoundary: () => void }>;
  timeout?: number;
}

// Simple Error Boundary implementation
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: React.ComponentType<{ error: Error; resetErrorBoundary: () => void }>;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  resetErrorBoundary = () => {
    this.setState({ hasError: false, error: undefined });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      const FallbackComponent = this.props.fallback;
      return <FallbackComponent error={this.state.error} resetErrorBoundary={this.resetErrorBoundary} />;
    }

    return this.props.children;
  }
}

// Default loading component with skeleton
const DefaultLoading = () => (
  <Card>
    <CardHeader>
      <div className="space-y-2">
        <div className="h-4 w-1/3 bg-gray-200 rounded animate-pulse" />
        <div className="h-3 w-1/2 bg-gray-200 rounded animate-pulse" />
      </div>
    </CardHeader>
    <CardContent>
      <div className="space-y-2">
        <div className="h-4 w-full bg-gray-200 rounded animate-pulse" />
        <div className="h-4 w-3/4 bg-gray-200 rounded animate-pulse" />
        <div className="h-4 w-1/2 bg-gray-200 rounded animate-pulse" />
      </div>
    </CardContent>
  </Card>
);

// Default error boundary component
const DefaultErrorFallback = ({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) => (
  <Card className="border-destructive">
    <CardHeader>
      <CardTitle className="flex items-center space-x-2 text-destructive">
        <AlertTriangle className="h-5 w-5" />
        <span>Component Error</span>
      </CardTitle>
      <CardDescription>
        Failed to load component: {error.message}
      </CardDescription>
    </CardHeader>
    <CardContent>
      <Button
        variant="outline"
        onClick={resetErrorBoundary}
        className="flex items-center space-x-2"
      >
        <RefreshCw className="h-4 w-4" />
        <span>Retry</span>
      </Button>
    </CardContent>
  </Card>
);

/**
 * Performance-optimized Suspense wrapper with error boundaries
 * Includes timeout handling and graceful degradation
 */
export function PerformantSuspense({
  children,
  fallback = <DefaultLoading />,
  errorFallback = DefaultErrorFallback,
  timeout = 10000, // 10 second timeout
}: PerformantSuspenseProps) {
  const [hasTimedOut, setHasTimedOut] = React.useState(false);

  React.useEffect(() => {
    const timer = setTimeout(() => {
      setHasTimedOut(true);
    }, timeout);

    return () => clearTimeout(timer);
  }, [timeout]);

  if (hasTimedOut) {
    return (
      <Card className="border-yellow-500">
        <CardHeader>
          <CardTitle className="text-yellow-600">Loading Timeout</CardTitle>
          <CardDescription>
            This component is taking longer than expected to load.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            variant="outline"
            onClick={() => setHasTimedOut(false)}
            className="flex items-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Retry</span>
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <ErrorBoundary
      fallback={errorFallback}
    >
      <Suspense fallback={fallback}>
        {children}
      </Suspense>
    </ErrorBoundary>
  );
}

// Hook for progressive enhancement
export function useProgressiveEnhancement() {
  const [isEnhanced, setIsEnhanced] = React.useState(false);

  React.useEffect(() => {
    // Check if we should load enhanced features
    const shouldEnhance =
      'requestIdleCallback' in window &&
      navigator.hardwareConcurrency > 2 &&
      (navigator as any).connection?.effectiveType !== 'slow-2g';

    if (shouldEnhance) {
      // Delay enhancement to not block initial render
      requestIdleCallback(() => {
        setIsEnhanced(true);
      });
    }
  }, []);

  return isEnhanced;
}

// Component for conditional feature loading
interface ConditionalFeatureProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  condition?: boolean;
}

export function ConditionalFeature({ children, fallback = null, condition = true }: ConditionalFeatureProps) {
  const isEnhanced = useProgressiveEnhancement();

  if (!condition || !isEnhanced) {
    return <>{fallback}</>;
  }

  return <PerformantSuspense>{children}</PerformantSuspense>;
}