'use client';

/**
 * Animated Card Component
 * 
 * Enhanced card with micro-interactions:
 * - Hover lift effect with shadow
 * - Focus ring for accessibility
 * - Staggered entrance animations
 * - Glassmorphism variant
 * - Loading skeleton state
 * 
 * Built on Shadcn Card primitive
 */

import * as React from 'react';
import { forwardRef, useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
// ============================================================================

export type CardVariant = 'default' | 'glass' | 'gradient' | 'outline' | 'elevated';
export type CardAnimation = 'none' | 'fade' | 'slide' | 'scale' | 'flip';

export interface AnimatedCardProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Visual variant */
  variant?: CardVariant;
  /** Entrance animation */
  animation?: CardAnimation;
  /** Animation delay (for staggered lists) */
  animationDelay?: number;
  /** Enable hover effects */
  hoverable?: boolean;
  /** Enable click effects */
  clickable?: boolean;
  /** Loading state */
  loading?: boolean;
  /** Skeleton height when loading */
  skeletonHeight?: number;
  /** Disable animations (for reduced motion) */
  disableAnimations?: boolean;
  /** Glow effect on hover */
  glowOnHover?: boolean;
  /** Border gradient */
  borderGradient?: boolean;
}

export interface AnimatedCardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Icon to display */
  icon?: React.ReactNode;
  /** Badge/status indicator */
  badge?: React.ReactNode;
  /** Actions (buttons, menu) */
  actions?: React.ReactNode;
}

export interface AnimatedCardContentProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Padding size */
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

// ============================================================================
// Variant Styles
// ============================================================================

const variantStyles: Record<CardVariant, string> = {
  default: 'bg-card text-card-foreground border border-border',
  glass: 'bg-background/60 backdrop-blur-xl border border-white/10 shadow-xl',
  gradient: 'bg-gradient-to-br from-primary/10 via-background to-secondary/10 border border-border',
  outline: 'bg-transparent border-2 border-border hover:border-primary/50',
  elevated: 'bg-card shadow-lg hover:shadow-xl border-0',
};

const animationStyles: Record<CardAnimation, string> = {
  none: '',
  fade: 'animate-in fade-in duration-500',
  slide: 'animate-in slide-in-from-bottom-4 fade-in duration-500',
  scale: 'animate-in zoom-in-95 fade-in duration-500',
  flip: 'animate-in spin-in-180 fade-in duration-700',
};

const paddingStyles = {
  none: 'p-0',
  sm: 'p-3',
  md: 'p-6',
  lg: 'p-8',
};

// ============================================================================
// Animated Card Component
// ============================================================================

export const AnimatedCard = forwardRef<HTMLDivElement, AnimatedCardProps>(
  function AnimatedCard(
    {
      className,
      variant = 'default',
      animation = 'fade',
      animationDelay = 0,
      hoverable = true,
      clickable = false,
      loading = false,
      skeletonHeight = 200,
      disableAnimations = false,
      glowOnHover = false,
      borderGradient = false,
      children,
      style,
      ...props
    },
    ref
  ) {
    const [isVisible, setIsVisible] = useState(false);
    const [isPressed, setIsPressed] = useState(false);
    const cardRef = useRef<HTMLDivElement>(null);

    // Check for reduced motion preference
    const prefersReducedMotion = 
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const shouldAnimate = !disableAnimations && !prefersReducedMotion;

    // Intersection observer for entrance animation
    useEffect(() => {
      if (!shouldAnimate || animation === 'none') {
        setIsVisible(true);
        return;
      }

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setTimeout(() => setIsVisible(true), animationDelay);
            observer.disconnect();
          }
        },
        { threshold: 0.1 }
      );

      const element = cardRef.current;
      if (element) {
        observer.observe(element);
      }

      return () => observer.disconnect();
    }, [shouldAnimate, animation, animationDelay]);

    // Loading skeleton
    if (loading) {
      return (
        <Card
          ref={ref}
          className={cn(
            variantStyles[variant],
            'overflow-hidden',
            className
          )}
          {...props}
        >
          <CardHeader>
            <Skeleton className="h-6 w-3/4" />
            <Skeleton className="h-4 w-1/2 mt-2" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6 mt-2" />
            <Skeleton className="h-4 w-4/6 mt-2" />
          </CardContent>
          <CardFooter>
            <Skeleton className="h-10 w-24" />
          </CardFooter>
        </Card>
      );
    }

    return (
      <div
        ref={cardRef}
        className={cn(
          'relative',
          borderGradient && 'p-[1px] rounded-lg bg-gradient-to-r from-primary via-secondary to-primary'
        )}
      >
        {/* Glow effect */}
        {glowOnHover && (
          <div
            className={cn(
              'absolute inset-0 rounded-lg opacity-0 transition-opacity duration-300',
              'bg-gradient-to-r from-primary/20 via-secondary/20 to-primary/20 blur-xl',
              'group-hover:opacity-100 -z-10'
            )}
          />
        )}

        <Card
          ref={ref}
          className={cn(
            variantStyles[variant],
            'relative overflow-hidden transition-all duration-300',
            // Visibility
            shouldAnimate && !isVisible && 'opacity-0',
            shouldAnimate && isVisible && animationStyles[animation],
            // Hover effects
            hoverable && [
              'hover:-translate-y-1',
              'hover:shadow-lg',
              variant === 'glass' && 'hover:bg-background/70',
              variant === 'outline' && 'hover:border-primary',
            ],
            // Click effects
            clickable && [
              'cursor-pointer',
              'active:scale-[0.98]',
              isPressed && 'scale-[0.98]',
            ],
            // Focus
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
            className
          )}
          style={{
            ...style,
            animationDelay: shouldAnimate ? `${animationDelay}ms` : undefined,
          }}
          onMouseDown={() => clickable && setIsPressed(true)}
          onMouseUp={() => clickable && setIsPressed(false)}
          onMouseLeave={() => clickable && setIsPressed(false)}
          tabIndex={clickable ? 0 : undefined}
          role={clickable ? 'button' : undefined}
          {...props}
        >
          {children}
        </Card>
      </div>
    );
  }
);

// ============================================================================
// Animated Card Header
// ============================================================================

export const AnimatedCardHeader = forwardRef<HTMLDivElement, AnimatedCardHeaderProps>(
  function AnimatedCardHeader(
    { className, icon, badge, actions, children, ...props },
    ref
  ) {
    return (
      <CardHeader
        ref={ref}
        className={cn('flex flex-row items-start justify-between space-y-0', className)}
        {...props}
      >
        <div className="flex items-start gap-3">
          {icon && (
            <div className="flex-shrink-0 p-2 rounded-lg bg-primary/10 text-primary">
              {icon}
            </div>
          )}
          <div className="space-y-1">
            {children}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {badge}
          {actions}
        </div>
      </CardHeader>
    );
  }
);

// ============================================================================
// Animated Card Content
// ============================================================================

export const AnimatedCardContent = forwardRef<HTMLDivElement, AnimatedCardContentProps>(
  function AnimatedCardContent(
    { className, padding = 'md', children, ...props },
    ref
  ) {
    return (
      <CardContent
        ref={ref}
        className={cn(paddingStyles[padding], className)}
        {...props}
      >
        {children}
      </CardContent>
    );
  }
);

// ============================================================================
// Animated Card Title & Description (re-export with animations)
// ============================================================================

export const AnimatedCardTitle = forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(function AnimatedCardTitle({ className, ...props }, ref) {
  return (
    <CardTitle
      ref={ref}
      className={cn(
        'text-lg font-semibold leading-none tracking-tight',
        'transition-colors duration-200',
        className
      )}
      {...props}
    />
  );
});

export const AnimatedCardDescription = forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(function AnimatedCardDescription({ className, ...props }, ref) {
  return (
    <CardDescription
      ref={ref}
      className={cn(
        'text-sm text-muted-foreground',
        'transition-colors duration-200',
        className
      )}
      {...props}
    />
  );
});

// ============================================================================
// Animated Card Footer
// ============================================================================

export const AnimatedCardFooter = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function AnimatedCardFooter({ className, ...props }, ref) {
  return (
    <CardFooter
      ref={ref}
      className={cn(
        'flex items-center justify-between pt-4',
        'border-t border-border/50',
        className
      )}
      {...props}
    />
  );
});

// ============================================================================
// Card Grid Component
// ============================================================================

export interface CardGridProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Number of columns */
  columns?: 1 | 2 | 3 | 4;
  /** Gap between cards */
  gap?: 'sm' | 'md' | 'lg';
  /** Stagger animation delay between cards */
  staggerDelay?: number;
}

const columnStyles = {
  1: 'grid-cols-1',
  2: 'grid-cols-1 md:grid-cols-2',
  3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
  4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
};

const gapStyles = {
  sm: 'gap-3',
  md: 'gap-6',
  lg: 'gap-8',
};

export const CardGrid = forwardRef<HTMLDivElement, CardGridProps>(
  function CardGrid(
    { className, columns = 3, gap = 'md', staggerDelay = 100, children, ...props },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn('grid', columnStyles[columns], gapStyles[gap], className)}
        {...props}
      >
        {React.Children.map(children, (child, index) => {
          if (React.isValidElement<AnimatedCardProps>(child)) {
            return React.cloneElement(child, {
              animationDelay: index * staggerDelay,
            });
          }
          return child;
        })}
      </div>
    );
  }
);

// ============================================================================
// Stat Card Component
// ============================================================================

export interface StatCardProps extends Omit<AnimatedCardProps, 'children'> {
  /** Stat label */
  label: string;
  /** Stat value */
  value: string | number;
  /** Change indicator */
  change?: {
    value: number;
    type: 'increase' | 'decrease' | 'neutral';
  };
  /** Icon */
  icon?: React.ReactNode;
  /** Trend chart (mini sparkline) */
  trend?: React.ReactNode;
}

export const StatCard = forwardRef<HTMLDivElement, StatCardProps>(
  function StatCard(
    { label, value, change, icon, trend, className, ...props },
    ref
  ) {
    const changeColors = {
      increase: 'text-green-500',
      decrease: 'text-red-500',
      neutral: 'text-muted-foreground',
    };

    const changeIcons = {
      increase: '↑',
      decrease: '↓',
      neutral: '→',
    };

    return (
      <AnimatedCard
        ref={ref}
        className={cn('group', className)}
        {...props}
      >
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">
                {label}
              </p>
              <p className="text-3xl font-bold tracking-tight">
                {value}
              </p>
              {change && (
                <p className={cn('text-sm font-medium flex items-center gap-1', changeColors[change.type])}>
                  <span>{changeIcons[change.type]}</span>
                  <span>{Math.abs(change.value)}%</span>
                  <span className="text-muted-foreground">vs last period</span>
                </p>
              )}
            </div>
            {icon && (
              <div className="p-3 rounded-full bg-primary/10 text-primary transition-transform duration-300 group-hover:scale-110">
                {icon}
              </div>
            )}
          </div>
          {trend && (
            <div className="mt-4 h-12">
              {trend}
            </div>
          )}
        </CardContent>
      </AnimatedCard>
    );
  }
);

// ============================================================================
// Feature Card Component
// ============================================================================

export interface FeatureCardProps extends Omit<AnimatedCardProps, 'children'> {
  /** Feature title */
  title: string;
  /** Feature description */
  description: string;
  /** Feature icon */
  icon: React.ReactNode;
  /** Call to action */
  action?: React.ReactNode;
}

export const FeatureCard = forwardRef<HTMLDivElement, FeatureCardProps>(
  function FeatureCard(
    { title, description, icon, action, className, ...props },
    ref
  ) {
    return (
      <AnimatedCard
        ref={ref}
        variant="outline"
        hoverable
        className={cn('group', className)}
        {...props}
      >
        <CardContent className="p-6 space-y-4">
          <div className="p-3 w-fit rounded-lg bg-primary/10 text-primary transition-all duration-300 group-hover:bg-primary group-hover:text-primary-foreground group-hover:scale-110">
            {icon}
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold text-lg">{title}</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {description}
            </p>
          </div>
          {action && (
            <div className="pt-2">
              {action}
            </div>
          )}
        </CardContent>
      </AnimatedCard>
    );
  }
);