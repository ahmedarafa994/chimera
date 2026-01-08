'use client';

/**
 * Animated Button Component
 * 
 * Enhanced button with micro-interactions:
 * - Ripple effect on click
 * - Loading spinner with text
 * - Icon animations
 * - Magnetic hover effect
 * - Success/error state animations
 * 
 * Built on Shadcn Button primitive
 */

import * as React from 'react';
import { forwardRef, useState, useRef, useCallback } from 'react';
import { Button, ButtonProps } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Loader2, Check, X } from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export type ButtonState = 'idle' | 'loading' | 'success' | 'error';

export interface AnimatedButtonProps extends ButtonProps {
  /** Current state */
  state?: ButtonState;
  /** Loading text */
  loadingText?: string;
  /** Success text */
  successText?: string;
  /** Error text */
  errorText?: string;
  /** Enable ripple effect */
  ripple?: boolean;
  /** Enable magnetic hover effect */
  magnetic?: boolean;
  /** Icon (left side) */
  leftIcon?: React.ReactNode;
  /** Icon (right side) */
  rightIcon?: React.ReactNode;
  /** Animate icon on hover */
  animateIcon?: boolean;
  /** Auto-reset state after success/error (ms) */
  autoResetDelay?: number;
  /** Callback when state resets */
  onStateReset?: () => void;
  /** Pulse animation */
  pulse?: boolean;
  /** Glow effect */
  glow?: boolean;
}

interface RippleStyle {
  left: number;
  top: number;
  width: number;
  height: number;
}

// ============================================================================
// Ripple Effect Hook
// ============================================================================

function useRipple() {
  const [ripples, setRipples] = useState<RippleStyle[]>([]);

  const addRipple = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    const button = event.currentTarget;
    const rect = button.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;

    const newRipple: RippleStyle = {
      left: x,
      top: y,
      width: size,
      height: size,
    };

    setRipples((prev) => [...prev, newRipple]);

    // Remove ripple after animation
    setTimeout(() => {
      setRipples((prev) => prev.slice(1));
    }, 600);
  }, []);

  return { ripples, addRipple };
}

// ============================================================================
// Magnetic Effect Hook
// ============================================================================

function useMagnetic(enabled: boolean) {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0 });

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      if (!enabled || !buttonRef.current) return;

      const rect = buttonRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      const deltaX = (event.clientX - centerX) * 0.2;
      const deltaY = (event.clientY - centerY) * 0.2;

      setTransform({ x: deltaX, y: deltaY });
    },
    [enabled]
  );

  const handleMouseLeave = useCallback(() => {
    setTransform({ x: 0, y: 0 });
  }, []);

  return { buttonRef, transform, handleMouseMove, handleMouseLeave };
}

// ============================================================================
// Animated Button Component
// ============================================================================

export const AnimatedButton = forwardRef<HTMLButtonElement, AnimatedButtonProps>(
  function AnimatedButton(
    {
      className,
      children,
      state = 'idle',
      loadingText,
      successText,
      errorText,
      ripple = true,
      magnetic = false,
      leftIcon,
      rightIcon,
      animateIcon = true,
      autoResetDelay = 2000,
      onStateReset,
      pulse = false,
      glow = false,
      disabled,
      onClick,
      ...props
    },
    ref
  ) {
    const { ripples, addRipple } = useRipple();
    const { buttonRef: magneticRef, transform, handleMouseMove, handleMouseLeave } = useMagnetic(magnetic);

    // Combine refs - inline to avoid hook complexity
    const combinedRef = useCallback((node: HTMLButtonElement | null) => {
      if (typeof ref === 'function') {
        ref(node);
      } else if (ref) {
        ref.current = node;
      }
      if (magneticRef && 'current' in magneticRef) {
        magneticRef.current = node;
      }
    }, [ref, magneticRef]);

    // Auto-reset state
    React.useEffect(() => {
      if ((state === 'success' || state === 'error') && autoResetDelay > 0) {
        const timer = setTimeout(() => {
          onStateReset?.();
        }, autoResetDelay);
        return () => clearTimeout(timer);
      }
    }, [state, autoResetDelay, onStateReset]);

    // Handle click with ripple
    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
      if (ripple && state === 'idle') {
        addRipple(event);
      }
      onClick?.(event);
    };

    // Determine content based on state
    const renderContent = () => {
      switch (state) {
        case 'loading':
          return (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              {loadingText && <span className="ml-2">{loadingText}</span>}
            </>
          );
        case 'success':
          return (
            <>
              <Check className="h-4 w-4 animate-in zoom-in duration-200" />
              {successText && <span className="ml-2">{successText}</span>}
            </>
          );
        case 'error':
          return (
            <>
              <X className="h-4 w-4 animate-in zoom-in duration-200" />
              {errorText && <span className="ml-2">{errorText}</span>}
            </>
          );
        default:
          return (
            <>
              {leftIcon && (
                <span
                  className={cn(
                    'transition-transform duration-200',
                    animateIcon && 'group-hover:-translate-x-0.5'
                  )}
                >
                  {leftIcon}
                </span>
              )}
              {children}
              {rightIcon && (
                <span
                  className={cn(
                    'transition-transform duration-200',
                    animateIcon && 'group-hover:translate-x-0.5'
                  )}
                >
                  {rightIcon}
                </span>
              )}
            </>
          );
      }
    };

    // State-based styles
    const stateStyles = {
      idle: '',
      loading: 'cursor-wait',
      success: 'bg-green-500 hover:bg-green-500 text-white border-green-500',
      error: 'bg-red-500 hover:bg-red-500 text-white border-red-500',
    };

    return (
      <Button
        ref={combinedRef}
        className={cn(
          'group relative overflow-hidden transition-all duration-200',
          stateStyles[state],
          pulse && 'animate-pulse',
          glow && [
            'shadow-lg',
            'shadow-primary/25',
            'hover:shadow-xl',
            'hover:shadow-primary/30',
          ],
          className
        )}
        style={{
          transform: magnetic ? `translate(${transform.x}px, ${transform.y}px)` : undefined,
        }}
        disabled={disabled || state === 'loading'}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        {...props}
      >
        {/* Ripple effects */}
        {ripples.map((ripple, index) => (
          <span
            key={index}
            className="absolute rounded-full bg-white/30 animate-ripple pointer-events-none"
            style={{
              left: ripple.left,
              top: ripple.top,
              width: ripple.width,
              height: ripple.height,
            }}
          />
        ))}

        {/* Button content */}
        <span className="relative z-10 flex items-center justify-center gap-2">
          {renderContent()}
        </span>
      </Button>
    );
  }
);

// ============================================================================
// Icon Button Component
// ============================================================================

export interface IconButtonProps extends Omit<AnimatedButtonProps, 'children'> {
  /** Icon to display */
  icon: React.ReactNode;
  /** Accessible label */
  label: string;
  /** Show tooltip */
  tooltip?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton({ icon, label, tooltip = true, className, ...props }, ref) {
    return (
      <AnimatedButton
        ref={ref}
        variant="ghost"
        size="icon"
        className={cn('relative', className)}
        aria-label={label}
        title={tooltip ? label : undefined}
        {...props}
      >
        {icon}
      </AnimatedButton>
    );
  }
);

// ============================================================================
// Button Group Component
// ============================================================================

export interface ButtonGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Size for all buttons */
  size?: ButtonProps['size'];
  /** Variant for all buttons */
  variant?: ButtonProps['variant'];
}

export const ButtonGroup = forwardRef<HTMLDivElement, ButtonGroupProps>(
  function ButtonGroup(
    { className, orientation = 'horizontal', size, variant, children, ...props },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex',
          orientation === 'horizontal' ? 'flex-row' : 'flex-col',
          '[&>button]:rounded-none',
          orientation === 'horizontal' && [
            '[&>button:first-child]:rounded-l-md',
            '[&>button:last-child]:rounded-r-md',
            '[&>button:not(:last-child)]:border-r-0',
          ],
          orientation === 'vertical' && [
            '[&>button:first-child]:rounded-t-md',
            '[&>button:last-child]:rounded-b-md',
            '[&>button:not(:last-child)]:border-b-0',
          ],
          className
        )}
        role="group"
        {...props}
      >
        {React.Children.map(children, (child) => {
          if (React.isValidElement<ButtonProps>(child)) {
            return React.cloneElement(child, {
              size: size ?? child.props.size,
              variant: variant ?? child.props.variant,
            });
          }
          return child;
        })}
      </div>
    );
  }
);

// ============================================================================
// Async Button Component
// ============================================================================

export interface AsyncButtonProps extends Omit<AnimatedButtonProps, 'state' | 'onClick'> {
  /** Async click handler */
  onClick: () => Promise<void>;
  /** Show success state on completion */
  showSuccess?: boolean;
  /** Show error state on failure */
  showError?: boolean;
}

export const AsyncButton = forwardRef<HTMLButtonElement, AsyncButtonProps>(
  function AsyncButton(
    { onClick, showSuccess = true, showError = true, ...props },
    ref
  ) {
    const [state, setState] = useState<ButtonState>('idle');

    const handleClick = async () => {
      setState('loading');
      try {
        await onClick();
        if (showSuccess) {
          setState('success');
        } else {
          setState('idle');
        }
      } catch {
        if (showError) {
          setState('error');
        } else {
          setState('idle');
        }
      }
    };

    return (
      <AnimatedButton
        ref={ref}
        state={state}
        onClick={handleClick}
        onStateReset={() => setState('idle')}
        {...props}
      />
    );
  }
);

// ============================================================================
// Copy Button Component
// ============================================================================

export interface CopyButtonProps extends Omit<AnimatedButtonProps, 'onClick' | 'state'> {
  /** Text to copy */
  text: string;
  /** Callback on copy */
  onCopy?: () => void;
}

export const CopyButton = forwardRef<HTMLButtonElement, CopyButtonProps>(
  function CopyButton({ text, onCopy, children, ...props }, ref) {
    const [state, setState] = useState<ButtonState>('idle');

    const handleCopy = async () => {
      try {
        await navigator.clipboard.writeText(text);
        setState('success');
        onCopy?.();
      } catch {
        setState('error');
      }
    };

    return (
      <AnimatedButton
        ref={ref}
        state={state}
        onClick={handleCopy}
        onStateReset={() => setState('idle')}
        successText="Copied!"
        errorText="Failed"
        {...props}
      >
        {children || 'Copy'}
      </AnimatedButton>
    );
  }
);

// ============================================================================
// Add ripple animation to global styles
// ============================================================================

// Add this to your globals.css:
// @keyframes ripple {
//   to {
//     transform: scale(4);
//     opacity: 0;
//   }
// }
// .animate-ripple {
//   animation: ripple 0.6s linear;
// }