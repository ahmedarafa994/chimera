'use client';

/**
 * Loading State Components
 * 
 * Provides various loading indicators:
 * - Skeleton loaders with shimmer
 * - Spinner variants
 * - Progress indicators
 * - Content placeholders
 * - Suspense boundaries
 * 
 * Built on Shadcn Skeleton primitive
 */

import * as React from 'react';
import { forwardRef, Suspense } from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
// ============================================================================

export type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type SpinnerVariant = 'default' | 'dots' | 'bars' | 'pulse' | 'ring';

// ============================================================================
// Spinner Component
// ============================================================================

export interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Size of the spinner */
  size?: SpinnerSize;
  /** Visual variant */
  variant?: SpinnerVariant;
  /** Color */
  color?: 'primary' | 'secondary' | 'muted' | 'white';
  /** Label for accessibility */
  label?: string;
}

const spinnerSizes: Record<SpinnerSize, string> = {
  xs: 'h-3 w-3',
  sm: 'h-4 w-4',
  md: 'h-6 w-6',
  lg: 'h-8 w-8',
  xl: 'h-12 w-12',
};

const spinnerColors = {
  primary: 'text-primary',
  secondary: 'text-secondary',
  muted: 'text-muted-foreground',
  white: 'text-white',
};

export const Spinner = forwardRef<HTMLDivElement, SpinnerProps>(
  function Spinner(
    {
      className,
      size = 'md',
      variant = 'default',
      color = 'primary',
      label = 'Loading...',
      ...props
    },
    ref
  ) {
    const sizeClass = spinnerSizes[size];
    const colorClass = spinnerColors[color];

    // Default spinning circle
    if (variant === 'default') {
      return (
        <div
          ref={ref}
          role="status"
          aria-label={label}
          className={cn('animate-spin', sizeClass, colorClass, className)}
          {...props}
        >
          <svg
            className="h-full w-full"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span className="sr-only">{label}</span>
        </div>
      );
    }

    // Dots variant
    if (variant === 'dots') {
      return (
        <div
          ref={ref}
          role="status"
          aria-label={label}
          className={cn('flex items-center gap-1', colorClass, className)}
          {...props}
        >
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={cn(
                'rounded-full bg-current animate-bounce',
                size === 'xs' && 'h-1 w-1',
                size === 'sm' && 'h-1.5 w-1.5',
                size === 'md' && 'h-2 w-2',
                size === 'lg' && 'h-3 w-3',
                size === 'xl' && 'h-4 w-4'
              )}
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
          <span className="sr-only">{label}</span>
        </div>
      );
    }

    // Bars variant
    if (variant === 'bars') {
      return (
        <div
          ref={ref}
          role="status"
          aria-label={label}
          className={cn('flex items-end gap-0.5', colorClass, className)}
          {...props}
        >
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className={cn(
                'bg-current animate-pulse rounded-sm',
                size === 'xs' && 'w-0.5',
                size === 'sm' && 'w-1',
                size === 'md' && 'w-1.5',
                size === 'lg' && 'w-2',
                size === 'xl' && 'w-3'
              )}
              style={{
                height: `${(i + 1) * 25}%`,
                animationDelay: `${i * 0.15}s`,
              }}
            />
          ))}
          <span className="sr-only">{label}</span>
        </div>
      );
    }

    // Pulse variant
    if (variant === 'pulse') {
      return (
        <div
          ref={ref}
          role="status"
          aria-label={label}
          className={cn('relative', sizeClass, className)}
          {...props}
        >
          <div
            className={cn(
              'absolute inset-0 rounded-full bg-current opacity-75 animate-ping',
              colorClass
            )}
          />
          <div
            className={cn(
              'relative rounded-full bg-current',
              sizeClass,
              colorClass
            )}
          />
          <span className="sr-only">{label}</span>
        </div>
      );
    }

    // Ring variant
    if (variant === 'ring') {
      return (
        <div
          ref={ref}
          role="status"
          aria-label={label}
          className={cn(
            'rounded-full border-2 border-current border-t-transparent animate-spin',
            sizeClass,
            colorClass,
            className
          )}
          {...props}
        >
          <span className="sr-only">{label}</span>
        </div>
      );
    }

    return null;
  }
);

// ============================================================================
// Skeleton Variants
// ============================================================================

export interface SkeletonTextProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Number of lines */
  lines?: number;
  /** Last line width */
  lastLineWidth?: 'full' | '3/4' | '1/2' | '1/4';
  /** Gap between lines */
  gap?: 'sm' | 'md' | 'lg';
}

const lastLineWidthClasses = {
  full: 'w-full',
  '3/4': 'w-3/4',
  '1/2': 'w-1/2',
  '1/4': 'w-1/4',
};

const gapClasses = {
  sm: 'space-y-1',
  md: 'space-y-2',
  lg: 'space-y-3',
};

export const SkeletonText = forwardRef<HTMLDivElement, SkeletonTextProps>(
  function SkeletonText(
    { className, lines = 3, lastLineWidth = '3/4', gap = 'md', ...props },
    ref
  ) {
    return (
      <div ref={ref} className={cn(gapClasses[gap], className)} {...props}>
        {Array.from({ length: lines }).map((_, i) => (
          <Skeleton
            key={i}
            className={cn(
              'h-4',
              i === lines - 1 ? lastLineWidthClasses[lastLineWidth] : 'w-full'
            )}
          />
        ))}
      </div>
    );
  }
);

export interface SkeletonAvatarProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Size */
  size?: 'sm' | 'md' | 'lg' | 'xl';
  /** Shape */
  shape?: 'circle' | 'square';
}

const avatarSizes = {
  sm: 'h-8 w-8',
  md: 'h-10 w-10',
  lg: 'h-12 w-12',
  xl: 'h-16 w-16',
};

export const SkeletonAvatar = forwardRef<HTMLDivElement, SkeletonAvatarProps>(
  function SkeletonAvatar(
    { className, size = 'md', shape = 'circle', ...props },
    ref
  ) {
    // Skeleton component may not support ref directly, wrap in div if needed
    return (
      <div ref={ref} className={cn(
        avatarSizes[size],
        shape === 'circle' ? 'rounded-full' : 'rounded-md',
        className
      )}>
        <Skeleton
          className="w-full h-full"
          {...props}
        />
      </div>
    );
  }
);

export interface SkeletonCardProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Show image placeholder */
  hasImage?: boolean;
  /** Image height */
  imageHeight?: number;
  /** Number of text lines */
  lines?: number;
  /** Show footer */
  hasFooter?: boolean;
}

export const SkeletonCard = forwardRef<HTMLDivElement, SkeletonCardProps>(
  function SkeletonCard(
    {
      className,
      hasImage = true,
      imageHeight = 200,
      lines = 3,
      hasFooter = true,
      ...props
    },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn(
          'rounded-lg border border-border bg-card overflow-hidden',
          className
        )}
        {...props}
      >
        {hasImage && (
          <Skeleton className="w-full" style={{ height: imageHeight }} />
        )}
        <div className="p-4 space-y-4">
          <div className="space-y-2">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
          </div>
          <SkeletonText lines={lines} />
          {hasFooter && (
            <div className="flex items-center justify-between pt-2">
              <Skeleton className="h-9 w-24" />
              <Skeleton className="h-9 w-9 rounded-full" />
            </div>
          )}
        </div>
      </div>
    );
  }
);

export interface SkeletonTableProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Number of rows */
  rows?: number;
  /** Number of columns */
  columns?: number;
  /** Show header */
  hasHeader?: boolean;
}

export const SkeletonTable = forwardRef<HTMLDivElement, SkeletonTableProps>(
  function SkeletonTable(
    { className, rows = 5, columns = 4, hasHeader = true, ...props },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn('rounded-lg border border-border overflow-hidden', className)}
        {...props}
      >
        {hasHeader && (
          <div className="flex gap-4 p-4 bg-muted/50 border-b border-border">
            {Array.from({ length: columns }).map((_, i) => (
              <Skeleton key={i} className="h-4 flex-1" />
            ))}
          </div>
        )}
        <div className="divide-y divide-border">
          {Array.from({ length: rows }).map((_, rowIndex) => (
            <div key={rowIndex} className="flex gap-4 p-4">
              {Array.from({ length: columns }).map((_, colIndex) => (
                <Skeleton
                  key={colIndex}
                  className={cn(
                    'h-4 flex-1',
                    colIndex === 0 && 'w-1/4 flex-none'
                  )}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }
);

// ============================================================================
// Progress Indicators
// ============================================================================

export interface LoadingProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Progress value (0-100) */
  value?: number;
  /** Indeterminate mode */
  indeterminate?: boolean;
  /** Show percentage */
  showPercentage?: boolean;
  /** Label */
  label?: string;
  /** Size */
  size?: 'sm' | 'md' | 'lg';
}

const progressSizes = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-3',
};

export const LoadingProgress = forwardRef<HTMLDivElement, LoadingProgressProps>(
  function LoadingProgress(
    {
      className,
      value = 0,
      indeterminate = false,
      showPercentage = false,
      label,
      size = 'md',
      ...props
    },
    ref
  ) {
    return (
      <div ref={ref} className={cn('w-full space-y-2', className)} {...props}>
        {(label || showPercentage) && (
          <div className="flex items-center justify-between text-sm">
            {label && <span className="text-muted-foreground">{label}</span>}
            {showPercentage && !indeterminate && (
              <span className="font-medium">{Math.round(value)}%</span>
            )}
          </div>
        )}
        <div className={cn('relative overflow-hidden rounded-full bg-muted', progressSizes[size])}>
          {indeterminate ? (
            <div
              className="absolute inset-0 bg-primary animate-indeterminate-progress"
              style={{
                width: '50%',
              }}
            />
          ) : (
            <Progress value={value} className={progressSizes[size]} />
          )}
        </div>
      </div>
    );
  }
);

// ============================================================================
// Loading Overlay
// ============================================================================

export interface LoadingOverlayProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Show overlay */
  visible?: boolean;
  /** Blur background */
  blur?: boolean;
  /** Loading message */
  message?: string;
  /** Spinner size */
  spinnerSize?: SpinnerSize;
  /** Full screen */
  fullScreen?: boolean;
}

export const LoadingOverlay = forwardRef<HTMLDivElement, LoadingOverlayProps>(
  function LoadingOverlay(
    {
      className,
      visible = true,
      blur = true,
      message,
      spinnerSize = 'lg',
      fullScreen = false,
      ...props
    },
    ref
  ) {
    if (!visible) return null;

    return (
      <div
        ref={ref}
        className={cn(
          'flex flex-col items-center justify-center gap-4',
          fullScreen ? 'fixed inset-0 z-50' : 'absolute inset-0 z-10',
          blur ? 'bg-background/80 backdrop-blur-sm' : 'bg-background/90',
          'animate-in fade-in duration-200',
          className
        )}
        {...props}
      >
        <Spinner size={spinnerSize} />
        {message && (
          <p className="text-sm text-muted-foreground animate-pulse">{message}</p>
        )}
      </div>
    );
  }
);

// ============================================================================
// Suspense Wrapper
// ============================================================================

export interface SuspenseWrapperProps {
  /** Children to render */
  children: React.ReactNode;
  /** Fallback component */
  fallback?: React.ReactNode;
  /** Loading message */
  loadingMessage?: string;
}

export function SuspenseWrapper({
  children,
  fallback,
  loadingMessage = 'Loading...',
}: SuspenseWrapperProps) {
  const defaultFallback = (
    <div className="flex items-center justify-center p-8">
      <div className="flex flex-col items-center gap-4">
        <Spinner size="lg" />
        <p className="text-sm text-muted-foreground">{loadingMessage}</p>
      </div>
    </div>
  );

  return <Suspense fallback={fallback || defaultFallback}>{children}</Suspense>;
}

// ============================================================================
// Content Placeholder
// ============================================================================

export interface ContentPlaceholderProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Icon */
  icon?: React.ReactNode;
  /** Title */
  title?: string;
  /** Description */
  description?: string;
  /** Action button */
  action?: React.ReactNode;
}

export const ContentPlaceholder = forwardRef<HTMLDivElement, ContentPlaceholderProps>(
  function ContentPlaceholder(
    { className, icon, title, description, action, ...props },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn(
          'flex flex-col items-center justify-center p-8 text-center',
          className
        )}
        {...props}
      >
        {icon && (
          <div className="mb-4 p-4 rounded-full bg-muted text-muted-foreground">
            {icon}
          </div>
        )}
        {title && (
          <h3 className="text-lg font-semibold mb-2">{title}</h3>
        )}
        {description && (
          <p className="text-sm text-muted-foreground max-w-sm mb-4">
            {description}
          </p>
        )}
        {action}
      </div>
    );
  }
);

// ============================================================================
// Shimmer Effect (for custom skeletons)
// ============================================================================

export interface ShimmerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Width */
  width?: string | number;
  /** Height */
  height?: string | number;
  /** Border radius */
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'full';
}

const roundedClasses = {
  none: 'rounded-none',
  sm: 'rounded-sm',
  md: 'rounded-md',
  lg: 'rounded-lg',
  full: 'rounded-full',
};

export const Shimmer = forwardRef<HTMLDivElement, ShimmerProps>(
  function Shimmer(
    { className, width, height, rounded = 'md', style, ...props },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn(
          'relative overflow-hidden bg-muted',
          roundedClasses[rounded],
          'before:absolute before:inset-0',
          'before:-translate-x-full',
          'before:animate-shimmer',
          'before:bg-gradient-to-r',
          'before:from-transparent before:via-white/20 before:to-transparent',
          className
        )}
        style={{
          width,
          height,
          ...style,
        }}
        {...props}
      />
    );
  }
);

// ============================================================================
// Add shimmer animation to global styles
// ============================================================================

// Add this to your globals.css:
// @keyframes shimmer {
//   100% {
//     transform: translateX(100%);
//   }
// }
// .animate-shimmer {
//   animation: shimmer 2s infinite;
// }
// @keyframes indeterminate-progress {
//   0% {
//     transform: translateX(-100%);
//   }
//   100% {
//     transform: translateX(200%);
//   }
// }
// .animate-indeterminate-progress {
//   animation: indeterminate-progress 1.5s infinite ease-in-out;
// }