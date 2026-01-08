'use client';

/**
 * Responsive Layout Components
 * 
 * Provides responsive layout primitives:
 * - Container with max-width constraints
 * - Responsive grid system
 * - Stack (vertical/horizontal)
 * - Responsive visibility
 * - Aspect ratio containers
 * 
 * Built with Tailwind CSS utilities
 */

import * as React from 'react';
import { forwardRef } from 'react';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
// ============================================================================

export type Breakpoint = 'sm' | 'md' | 'lg' | 'xl' | '2xl';
export type Spacing = 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
export type Alignment = 'start' | 'center' | 'end' | 'stretch' | 'baseline';
export type Justify = 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly';

// ============================================================================
// Spacing Utilities
// ============================================================================

const spacingMap: Record<Spacing, string> = {
  none: '0',
  xs: '1',
  sm: '2',
  md: '4',
  lg: '6',
  xl: '8',
  '2xl': '12',
};

const gapClasses: Record<Spacing, string> = {
  none: 'gap-0',
  xs: 'gap-1',
  sm: 'gap-2',
  md: 'gap-4',
  lg: 'gap-6',
  xl: 'gap-8',
  '2xl': 'gap-12',
};

const paddingClasses: Record<Spacing, string> = {
  none: 'p-0',
  xs: 'p-1',
  sm: 'p-2',
  md: 'p-4',
  lg: 'p-6',
  xl: 'p-8',
  '2xl': 'p-12',
};

// ============================================================================
// Container Component
// ============================================================================

export interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Maximum width */
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full' | 'prose';
  /** Center the container */
  centered?: boolean;
  /** Horizontal padding */
  padding?: Spacing;
  /** As different element */
  as?: 'div' | 'section' | 'article' | 'main' | 'aside';
}

const maxWidthClasses = {
  sm: 'max-w-screen-sm',
  md: 'max-w-screen-md',
  lg: 'max-w-screen-lg',
  xl: 'max-w-screen-xl',
  '2xl': 'max-w-screen-2xl',
  full: 'max-w-full',
  prose: 'max-w-prose',
};

export const Container = forwardRef<HTMLDivElement, ContainerProps>(
  function Container(
    {
      className,
      maxWidth = 'xl',
      centered = true,
      padding = 'md',
      as: Component = 'div',
      children,
      ...props
    },
    ref
  ) {
    return (
      <Component
        ref={ref}
        className={cn(
          'w-full',
          maxWidthClasses[maxWidth],
          centered && 'mx-auto',
          `px-${spacingMap[padding]}`,
          className
        )}
        {...props}
      >
        {children}
      </Component>
    );
  }
);

// ============================================================================
// Grid Component
// ============================================================================

export interface GridProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Number of columns */
  columns?: 1 | 2 | 3 | 4 | 5 | 6 | 12 | 'auto';
  /** Responsive columns */
  responsiveColumns?: {
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
  };
  /** Gap between items */
  gap?: Spacing;
  /** Row gap */
  rowGap?: Spacing;
  /** Column gap */
  columnGap?: Spacing;
  /** Align items */
  align?: Alignment;
  /** Justify items */
  justify?: Justify;
  /** Auto-fit columns */
  autoFit?: boolean;
  /** Minimum column width for auto-fit */
  minColumnWidth?: string;
}

const columnClasses: Record<number | 'auto', string> = {
  1: 'grid-cols-1',
  2: 'grid-cols-2',
  3: 'grid-cols-3',
  4: 'grid-cols-4',
  5: 'grid-cols-5',
  6: 'grid-cols-6',
  12: 'grid-cols-12',
  auto: 'grid-cols-none',
};

const alignClasses: Record<Alignment, string> = {
  start: 'items-start',
  center: 'items-center',
  end: 'items-end',
  stretch: 'items-stretch',
  baseline: 'items-baseline',
};

const justifyClasses: Record<Justify, string> = {
  start: 'justify-start',
  center: 'justify-center',
  end: 'justify-end',
  between: 'justify-between',
  around: 'justify-around',
  evenly: 'justify-evenly',
};

export const Grid = forwardRef<HTMLDivElement, GridProps>(
  function Grid(
    {
      className,
      columns = 1,
      responsiveColumns,
      gap = 'md',
      rowGap,
      columnGap,
      align,
      justify,
      autoFit = false,
      minColumnWidth = '250px',
      style,
      children,
      ...props
    },
    ref
  ) {
    const responsiveClasses = responsiveColumns
      ? [
        responsiveColumns.sm && `sm:grid-cols-${responsiveColumns.sm}`,
        responsiveColumns.md && `md:grid-cols-${responsiveColumns.md}`,
        responsiveColumns.lg && `lg:grid-cols-${responsiveColumns.lg}`,
        responsiveColumns.xl && `xl:grid-cols-${responsiveColumns.xl}`,
      ].filter(Boolean)
      : [];

    return (
      <div
        ref={ref}
        className={cn(
          'grid',
          !autoFit && columnClasses[columns as keyof typeof columnClasses],
          ...responsiveClasses,
          rowGap ? `row-gap-${spacingMap[rowGap]}` : null,
          columnGap ? `col-gap-${spacingMap[columnGap]}` : null,
          !rowGap && !columnGap && gapClasses[gap],
          align && alignClasses[align],
          justify && justifyClasses[justify],
          className
        )}
        style={{
          ...style,
          ...(autoFit && {
            gridTemplateColumns: `repeat(auto-fit, minmax(${minColumnWidth}, 1fr))`,
          }),
        }}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Stack Component
// ============================================================================

export interface StackProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Direction */
  direction?: 'row' | 'column' | 'row-reverse' | 'column-reverse';
  /** Responsive direction */
  responsiveDirection?: {
    sm?: 'row' | 'column';
    md?: 'row' | 'column';
    lg?: 'row' | 'column';
  };
  /** Gap between items */
  gap?: Spacing;
  /** Align items */
  align?: Alignment;
  /** Justify content */
  justify?: Justify;
  /** Wrap items */
  wrap?: boolean;
  /** Divider between items */
  divider?: React.ReactNode;
  /** As different element */
  as?: 'div' | 'ul' | 'ol' | 'nav' | 'section';
}

const directionClasses = {
  row: 'flex-row',
  column: 'flex-col',
  'row-reverse': 'flex-row-reverse',
  'column-reverse': 'flex-col-reverse',
};

export const Stack = forwardRef<HTMLDivElement, StackProps>(
  function Stack(
    {
      className,
      direction = 'column',
      responsiveDirection,
      gap = 'md',
      align,
      justify,
      wrap = false,
      divider,
      as: Component = 'div',
      children,
      ...props
    },
    ref
  ) {
    const responsiveClasses = responsiveDirection
      ? [
        responsiveDirection.sm && `sm:flex-${responsiveDirection.sm}`,
        responsiveDirection.md && `md:flex-${responsiveDirection.md}`,
        responsiveDirection.lg && `lg:flex-${responsiveDirection.lg}`,
      ].filter(Boolean)
      : [];

    // Add dividers between children
    const childrenWithDividers = divider
      ? React.Children.toArray(children).reduce<React.ReactNode[]>(
        (acc, child, index, array) => {
          acc.push(child);
          if (index < array.length - 1) {
            acc.push(
              <div key={`divider-${index}`} className="flex-shrink-0">
                {divider}
              </div>
            );
          }
          return acc;
        },
        []
      )
      : children;

    // Polymorphic component pattern requires type assertion
    // This is a standard pattern for components with `as` prop
    const Comp = Component as React.ElementType;

    // For polymorphic components, we need to handle ref carefully
    // Only pass ref to elements that support it
    const componentProps = {
      className: cn(
        'flex',
        directionClasses[direction],
        ...responsiveClasses,
        gapClasses[gap],
        align && alignClasses[align],
        justify && justifyClasses[justify],
        wrap && 'flex-wrap',
        className
      ),
      ...props,
    };

    // Only add ref for standard HTML elements
    if (typeof Component === 'string') {
      return (
        <Comp ref={ref} {...componentProps}>
          {childrenWithDividers}
        </Comp>
      );
    }

    return (
      <Comp {...componentProps}>
        {childrenWithDividers}
      </Comp>
    );
  }
);

// ============================================================================
// HStack & VStack Shortcuts
// ============================================================================

export const HStack = forwardRef<HTMLDivElement, Omit<StackProps, 'direction'>>(
  function HStack(props, ref) {
    return <Stack ref={ref} direction="row" {...props} />;
  }
);

export const VStack = forwardRef<HTMLDivElement, Omit<StackProps, 'direction'>>(
  function VStack(props, ref) {
    return <Stack ref={ref} direction="column" {...props} />;
  }
);

// ============================================================================
// Responsive Visibility
// ============================================================================

export interface ShowProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Show above this breakpoint */
  above?: Breakpoint;
  /** Show below this breakpoint */
  below?: Breakpoint;
  /** Show only at this breakpoint */
  at?: Breakpoint;
}

const showAboveClasses: Record<Breakpoint, string> = {
  sm: 'hidden sm:block',
  md: 'hidden md:block',
  lg: 'hidden lg:block',
  xl: 'hidden xl:block',
  '2xl': 'hidden 2xl:block',
};

const showBelowClasses: Record<Breakpoint, string> = {
  sm: 'sm:hidden',
  md: 'md:hidden',
  lg: 'lg:hidden',
  xl: 'xl:hidden',
  '2xl': '2xl:hidden',
};

const showAtClasses: Record<Breakpoint, string> = {
  sm: 'hidden sm:block md:hidden',
  md: 'hidden md:block lg:hidden',
  lg: 'hidden lg:block xl:hidden',
  xl: 'hidden xl:block 2xl:hidden',
  '2xl': 'hidden 2xl:block',
};

export const Show = forwardRef<HTMLDivElement, ShowProps>(
  function Show({ className, above, below, at, children, ...props }, ref) {
    let visibilityClass = '';

    if (at) {
      visibilityClass = showAtClasses[at];
    } else if (above) {
      visibilityClass = showAboveClasses[above];
    } else if (below) {
      visibilityClass = showBelowClasses[below];
    }

    return (
      <div ref={ref} className={cn(visibilityClass, className)} {...props}>
        {children}
      </div>
    );
  }
);

export const Hide = forwardRef<HTMLDivElement, ShowProps>(
  function Hide({ className, above, below, at, children, ...props }, ref) {
    let visibilityClass = '';

    if (at) {
      // Invert the at logic
      visibilityClass = `block ${at}:hidden`;
    } else if (above) {
      visibilityClass = showBelowClasses[above];
    } else if (below) {
      visibilityClass = showAboveClasses[below];
    }

    return (
      <div ref={ref} className={cn(visibilityClass, className)} {...props}>
        {children}
      </div>
    );
  }
);

// ============================================================================
// Aspect Ratio Container
// ============================================================================

export interface AspectRatioProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Aspect ratio (width/height) */
  ratio?: number | '16/9' | '4/3' | '1/1' | '21/9' | '9/16';
}

const ratioMap: Record<string, number> = {
  '16/9': 16 / 9,
  '4/3': 4 / 3,
  '1/1': 1,
  '21/9': 21 / 9,
  '9/16': 9 / 16,
};

export const AspectRatio = forwardRef<HTMLDivElement, AspectRatioProps>(
  function AspectRatio({ className, ratio = '16/9', style, children, ...props }, ref) {
    const numericRatio = typeof ratio === 'number' ? ratio : ratioMap[ratio];

    return (
      <div
        ref={ref}
        className={cn('relative w-full', className)}
        style={{
          ...style,
          paddingBottom: `${(1 / numericRatio) * 100}%`,
        }}
        {...props}
      >
        <div className="absolute inset-0">{children}</div>
      </div>
    );
  }
);

// ============================================================================
// Spacer Component
// ============================================================================

export interface SpacerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Size of the spacer */
  size?: Spacing;
  /** Axis (for flex containers) */
  axis?: 'horizontal' | 'vertical' | 'both';
}

export const Spacer = forwardRef<HTMLDivElement, SpacerProps>(
  function Spacer({ className, size = 'md', axis = 'both', ...props }, ref) {
    const sizeClass = `h-${spacingMap[size]} w-${spacingMap[size]}`;

    return (
      <div
        ref={ref}
        className={cn(
          axis === 'horizontal' && `w-${spacingMap[size]} h-0`,
          axis === 'vertical' && `h-${spacingMap[size]} w-0`,
          axis === 'both' && sizeClass,
          'flex-shrink-0',
          className
        )}
        aria-hidden="true"
        {...props}
      />
    );
  }
);

// ============================================================================
// Center Component
// ============================================================================

export interface CenterProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Inline (horizontal only) */
  inline?: boolean;
}

export const Center = forwardRef<HTMLDivElement, CenterProps>(
  function Center({ className, inline = false, children, ...props }, ref) {
    return (
      <div
        ref={ref}
        className={cn(
          'flex items-center justify-center',
          !inline && 'h-full w-full',
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Divider Component
// ============================================================================

export interface DividerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Variant */
  variant?: 'solid' | 'dashed' | 'dotted';
  /** Label */
  label?: React.ReactNode;
}

export const Divider = forwardRef<HTMLDivElement, DividerProps>(
  function Divider(
    { className, orientation = 'horizontal', variant = 'solid', label, ...props },
    ref
  ) {
    const borderStyle = {
      solid: 'border-solid',
      dashed: 'border-dashed',
      dotted: 'border-dotted',
    };

    if (label) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center',
            orientation === 'vertical' && 'flex-col h-full',
            className
          )}
          role="separator"
          {...props}
        >
          <div
            className={cn(
              'flex-1 border-border',
              borderStyle[variant],
              orientation === 'horizontal' ? 'border-t' : 'border-l'
            )}
          />
          <span className="px-3 text-sm text-muted-foreground">{label}</span>
          <div
            className={cn(
              'flex-1 border-border',
              borderStyle[variant],
              orientation === 'horizontal' ? 'border-t' : 'border-l'
            )}
          />
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn(
          'border-border',
          borderStyle[variant],
          orientation === 'horizontal' ? 'w-full border-t' : 'h-full border-l',
          className
        )}
        role="separator"
        {...props}
      />
    );
  }
);

// ============================================================================
// Wrap Component
// ============================================================================

export interface WrapProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Gap between items */
  gap?: Spacing;
  /** Align items */
  align?: Alignment;
  /** Justify content */
  justify?: Justify;
}

export const Wrap = forwardRef<HTMLDivElement, WrapProps>(
  function Wrap({ className, gap = 'md', align, justify, children, ...props }, ref) {
    return (
      <div
        ref={ref}
        className={cn(
          'flex flex-wrap',
          gapClasses[gap],
          align && alignClasses[align],
          justify && justifyClasses[justify],
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);