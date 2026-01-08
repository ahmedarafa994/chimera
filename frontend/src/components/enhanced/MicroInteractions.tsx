'use client';

/**
 * Micro-Interaction Components
 * 
 * Provides subtle animations and feedback:
 * - Hover effects
 * - Click feedback
 * - Scroll animations
 * - Transition wrappers
 * - Gesture handlers
 * 
 * Enhances user experience with delightful interactions
 */

import * as React from 'react';
import { forwardRef, useRef, useState, useEffect, useCallback } from 'react';
import { cn } from '@/lib/utils';

// ============================================================================
// Types
// ============================================================================

export type EasingFunction = 
  | 'linear'
  | 'ease'
  | 'ease-in'
  | 'ease-out'
  | 'ease-in-out'
  | 'spring';

export type AnimationDirection = 'up' | 'down' | 'left' | 'right';

// ============================================================================
// Hover Scale Component
// ============================================================================

export interface HoverScaleProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Scale factor on hover */
  scale?: number;
  /** Animation duration (ms) */
  duration?: number;
  /** Easing function */
  easing?: EasingFunction;
  /** Disable on touch devices */
  disableOnTouch?: boolean;
}

export const HoverScale = forwardRef<HTMLDivElement, HoverScaleProps>(
  function HoverScale(
    {
      className,
      scale = 1.05,
      duration = 200,
      easing = 'ease-out',
      disableOnTouch = true,
      style,
      children,
      ...props
    },
    ref
  ) {
    const [isHovered, setIsHovered] = useState(false);
    const [isTouchDevice, setIsTouchDevice] = useState(false);

    useEffect(() => {
      setIsTouchDevice('ontouchstart' in window);
    }, []);

    const shouldAnimate = !disableOnTouch || !isTouchDevice;

    return (
      <div
        ref={ref}
        className={cn('transition-transform', className)}
        style={{
          ...style,
          transform: shouldAnimate && isHovered ? `scale(${scale})` : 'scale(1)',
          transitionDuration: `${duration}ms`,
          transitionTimingFunction: easing,
        }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Hover Lift Component
// ============================================================================

export interface HoverLiftProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Lift distance (px) */
  distance?: number;
  /** Shadow intensity */
  shadowIntensity?: 'none' | 'sm' | 'md' | 'lg';
  /** Animation duration (ms) */
  duration?: number;
}

const shadowClasses = {
  none: '',
  sm: 'hover:shadow-md',
  md: 'hover:shadow-lg',
  lg: 'hover:shadow-xl',
};

export const HoverLift = forwardRef<HTMLDivElement, HoverLiftProps>(
  function HoverLift(
    {
      className,
      distance = 4,
      shadowIntensity = 'md',
      duration = 200,
      style,
      children,
      ...props
    },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn(
          'transition-all',
          shadowClasses[shadowIntensity],
          className
        )}
        style={{
          ...style,
          transitionDuration: `${duration}ms`,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = `translateY(-${distance}px)`;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
        }}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Press Effect Component
// ============================================================================

export interface PressEffectProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Scale when pressed */
  pressScale?: number;
  /** Animation duration (ms) */
  duration?: number;
}

export const PressEffect = forwardRef<HTMLDivElement, PressEffectProps>(
  function PressEffect(
    { className, pressScale = 0.97, duration = 100, style, children, ...props },
    ref
  ) {
    const [isPressed, setIsPressed] = useState(false);

    return (
      <div
        ref={ref}
        className={cn('transition-transform cursor-pointer select-none', className)}
        style={{
          ...style,
          transform: isPressed ? `scale(${pressScale})` : 'scale(1)',
          transitionDuration: `${duration}ms`,
        }}
        onMouseDown={() => setIsPressed(true)}
        onMouseUp={() => setIsPressed(false)}
        onMouseLeave={() => setIsPressed(false)}
        onTouchStart={() => setIsPressed(true)}
        onTouchEnd={() => setIsPressed(false)}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Tilt Effect Component
// ============================================================================

export interface TiltEffectProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Maximum tilt angle (degrees) */
  maxTilt?: number;
  /** Perspective (px) */
  perspective?: number;
  /** Scale on hover */
  scale?: number;
  /** Glare effect */
  glare?: boolean;
  /** Glare opacity */
  glareOpacity?: number;
}

export const TiltEffect = forwardRef<HTMLDivElement, TiltEffectProps>(
  function TiltEffect(
    {
      className,
      maxTilt = 10,
      perspective = 1000,
      scale = 1.02,
      glare = false,
      glareOpacity = 0.2,
      style,
      children,
      ...props
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [transform, setTransform] = useState('');
    const [glarePosition, setGlarePosition] = useState({ x: 50, y: 50 });

    const handleMouseMove = useCallback(
      (e: React.MouseEvent<HTMLDivElement>) => {
        const container = containerRef.current;
        if (!container) return;

        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = ((y - centerY) / centerY) * -maxTilt;
        const rotateY = ((x - centerX) / centerX) * maxTilt;

        setTransform(
          `perspective(${perspective}px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(${scale})`
        );

        if (glare) {
          setGlarePosition({
            x: (x / rect.width) * 100,
            y: (y / rect.height) * 100,
          });
        }
      },
      [maxTilt, perspective, scale, glare]
    );

    const handleMouseLeave = useCallback(() => {
      setTransform('');
      setGlarePosition({ x: 50, y: 50 });
    }, []);

    return (
      <div
        ref={(node) => {
          (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
        }}
        className={cn('relative transition-transform duration-200', className)}
        style={{
          ...style,
          transform: transform || undefined,
          transformStyle: 'preserve-3d',
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        {...props}
      >
        {children}
        {glare && (
          <div
            className="absolute inset-0 pointer-events-none rounded-inherit"
            style={{
              background: `radial-gradient(circle at ${glarePosition.x}% ${glarePosition.y}%, rgba(255,255,255,${glareOpacity}) 0%, transparent 60%)`,
            }}
          />
        )}
      </div>
    );
  }
);

// ============================================================================
// Scroll Reveal Component
// ============================================================================

export interface ScrollRevealProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Animation direction */
  direction?: AnimationDirection;
  /** Animation distance (px) */
  distance?: number;
  /** Animation duration (ms) */
  duration?: number;
  /** Animation delay (ms) */
  delay?: number;
  /** Trigger threshold (0-1) */
  threshold?: number;
  /** Only animate once */
  once?: boolean;
  /** Easing function */
  easing?: EasingFunction;
}

const directionTransforms: Record<AnimationDirection, (distance: number) => string> = {
  up: (d) => `translateY(${d}px)`,
  down: (d) => `translateY(-${d}px)`,
  left: (d) => `translateX(${d}px)`,
  right: (d) => `translateX(-${d}px)`,
};

export const ScrollReveal = forwardRef<HTMLDivElement, ScrollRevealProps>(
  function ScrollReveal(
    {
      className,
      direction = 'up',
      distance = 30,
      duration = 600,
      delay = 0,
      threshold = 0.1,
      once = true,
      easing = 'ease-out',
      style,
      children,
      ...props
    },
    ref
  ) {
    const elementRef = useRef<HTMLDivElement>(null);
    const [isVisible, setIsVisible] = useState(false);
    const hasAnimated = useRef(false);

    useEffect(() => {
      const element = elementRef.current;
      if (!element) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            if (once && hasAnimated.current) return;
            setIsVisible(true);
            hasAnimated.current = true;
          } else if (!once) {
            setIsVisible(false);
          }
        },
        { threshold }
      );

      observer.observe(element);
      return () => observer.disconnect();
    }, [threshold, once]);

    const initialTransform = directionTransforms[direction](distance);

    return (
      <div
        ref={(node) => {
          (elementRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
        }}
        className={cn('transition-all', className)}
        style={{
          ...style,
          opacity: isVisible ? 1 : 0,
          transform: isVisible ? 'translate(0)' : initialTransform,
          transitionDuration: `${duration}ms`,
          transitionDelay: `${delay}ms`,
          transitionTimingFunction: easing,
        }}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Stagger Children Component
// ============================================================================

export interface StaggerChildrenProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Delay between each child (ms) */
  staggerDelay?: number;
  /** Initial delay (ms) */
  initialDelay?: number;
  /** Animation duration (ms) */
  duration?: number;
  /** Animation direction */
  direction?: AnimationDirection;
  /** Distance */
  distance?: number;
}

export const StaggerChildren = forwardRef<HTMLDivElement, StaggerChildrenProps>(
  function StaggerChildren(
    {
      className,
      staggerDelay = 100,
      initialDelay = 0,
      duration = 400,
      direction = 'up',
      distance = 20,
      children,
      ...props
    },
    ref
  ) {
    const [isVisible, setIsVisible] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
            observer.disconnect();
          }
        },
        { threshold: 0.1 }
      );

      const element = containerRef.current;
      if (element) {
        observer.observe(element);
      }

      return () => observer.disconnect();
    }, []);

    const initialTransform = directionTransforms[direction](distance);

    return (
      <div
        ref={(node) => {
          (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
        }}
        className={className}
        {...props}
      >
        {React.Children.map(children, (child, index) => (
          <div
            className="transition-all"
            style={{
              opacity: isVisible ? 1 : 0,
              transform: isVisible ? 'translate(0)' : initialTransform,
              transitionDuration: `${duration}ms`,
              transitionDelay: `${initialDelay + index * staggerDelay}ms`,
              transitionTimingFunction: 'ease-out',
            }}
          >
            {child}
          </div>
        ))}
      </div>
    );
  }
);

// ============================================================================
// Parallax Component
// ============================================================================

export interface ParallaxProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Parallax speed (negative = opposite direction) */
  speed?: number;
  /** Parallax direction */
  direction?: 'vertical' | 'horizontal';
}

export const Parallax = forwardRef<HTMLDivElement, ParallaxProps>(
  function Parallax(
    { className, speed = 0.5, direction = 'vertical', style, children, ...props },
    ref
  ) {
    const [offset, setOffset] = useState(0);
    const elementRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      const handleScroll = () => {
        const element = elementRef.current;
        if (!element) return;

        const rect = element.getBoundingClientRect();
        const scrolled = window.scrollY;
        const elementTop = rect.top + scrolled;
        const relativeScroll = scrolled - elementTop + window.innerHeight;

        setOffset(relativeScroll * speed);
      };

      window.addEventListener('scroll', handleScroll, { passive: true });
      handleScroll();

      return () => window.removeEventListener('scroll', handleScroll);
    }, [speed]);

    const transform =
      direction === 'vertical'
        ? `translateY(${offset}px)`
        : `translateX(${offset}px)`;

    return (
      <div
        ref={(node) => {
          (elementRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
        }}
        className={cn('will-change-transform', className)}
        style={{
          ...style,
          transform,
        }}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Magnetic Effect Component
// ============================================================================

export interface MagneticProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Magnetic strength */
  strength?: number;
  /** Magnetic radius (px) */
  radius?: number;
}

export const Magnetic = forwardRef<HTMLDivElement, MagneticProps>(
  function Magnetic(
    { className, strength = 0.3, radius = 100, style, children, ...props },
    ref
  ) {
    const elementRef = useRef<HTMLDivElement>(null);
    const [position, setPosition] = useState({ x: 0, y: 0 });

    const handleMouseMove = useCallback(
      (e: MouseEvent) => {
        const element = elementRef.current;
        if (!element) return;

        const rect = element.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        const distanceX = e.clientX - centerX;
        const distanceY = e.clientY - centerY;
        const distance = Math.sqrt(distanceX ** 2 + distanceY ** 2);

        if (distance < radius) {
          const factor = (1 - distance / radius) * strength;
          setPosition({
            x: distanceX * factor,
            y: distanceY * factor,
          });
        } else {
          setPosition({ x: 0, y: 0 });
        }
      },
      [strength, radius]
    );

    const handleMouseLeave = useCallback(() => {
      setPosition({ x: 0, y: 0 });
    }, []);

    useEffect(() => {
      window.addEventListener('mousemove', handleMouseMove);
      return () => window.removeEventListener('mousemove', handleMouseMove);
    }, [handleMouseMove]);

    return (
      <div
        ref={(node) => {
          (elementRef as React.MutableRefObject<HTMLDivElement | null>).current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
        }}
        className={cn('transition-transform duration-200', className)}
        style={{
          ...style,
          transform: `translate(${position.x}px, ${position.y}px)`,
        }}
        onMouseLeave={handleMouseLeave}
        {...props}
      >
        {children}
      </div>
    );
  }
);

// ============================================================================
// Typewriter Effect Component
// ============================================================================

export interface TypewriterProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Text to type */
  text: string;
  /** Typing speed (ms per character) */
  speed?: number;
  /** Delay before starting (ms) */
  delay?: number;
  /** Show cursor */
  cursor?: boolean;
  /** Cursor character */
  cursorChar?: string;
  /** Loop the animation */
  loop?: boolean;
  /** Pause at end (ms) */
  pauseAtEnd?: number;
  /** Callback when typing completes */
  onComplete?: () => void;
}

export const Typewriter = forwardRef<HTMLSpanElement, TypewriterProps>(
  function Typewriter(
    {
      className,
      text,
      speed = 50,
      delay = 0,
      cursor = true,
      cursorChar = '|',
      loop = false,
      pauseAtEnd = 1000,
      onComplete,
      ...props
    },
    ref
  ) {
    const [displayText, setDisplayText] = useState('');
    const [isTyping, setIsTyping] = useState(false);

    useEffect(() => {
      let timeout: ReturnType<typeof setTimeout>;
      let charIndex = 0;

      const startTyping = () => {
        setIsTyping(true);
        setDisplayText('');
        charIndex = 0;

        const type = () => {
          if (charIndex < text.length) {
            setDisplayText(text.slice(0, charIndex + 1));
            charIndex++;
            timeout = setTimeout(type, speed);
          } else {
            setIsTyping(false);
            onComplete?.();

            if (loop) {
              timeout = setTimeout(startTyping, pauseAtEnd);
            }
          }
        };

        timeout = setTimeout(type, delay);
      };

      startTyping();

      return () => clearTimeout(timeout);
    }, [text, speed, delay, loop, pauseAtEnd, onComplete]);

    return (
      <span ref={ref} className={className} {...props}>
        {displayText}
        {cursor && (
          <span
            className={cn(
              'inline-block ml-0.5',
              isTyping ? 'animate-pulse' : 'animate-blink'
            )}
          >
            {cursorChar}
          </span>
        )}
      </span>
    );
  }
);

// ============================================================================
// Counter Animation Component
// ============================================================================

export interface CounterProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Target value */
  value: number;
  /** Animation duration (ms) */
  duration?: number;
  /** Decimal places */
  decimals?: number;
  /** Prefix */
  prefix?: string;
  /** Suffix */
  suffix?: string;
  /** Easing function */
  easing?: EasingFunction;
  /** Start animation on mount */
  autoStart?: boolean;
}

export const Counter = forwardRef<HTMLSpanElement, CounterProps>(
  function Counter(
    {
      className,
      value,
      duration = 2000,
      decimals = 0,
      prefix = '',
      suffix = '',
      easing = 'ease-out',
      autoStart = true,
      ...props
    },
    ref
  ) {
    const [displayValue, setDisplayValue] = useState(0);
    const elementRef = useRef<HTMLSpanElement>(null);
    const hasStarted = useRef(false);

    useEffect(() => {
      if (!autoStart) return;

      const element = elementRef.current;
      if (!element) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting && !hasStarted.current) {
            hasStarted.current = true;
            animateValue();
            observer.disconnect();
          }
        },
        { threshold: 0.1 }
      );

      observer.observe(element);
      return () => observer.disconnect();
    }, [autoStart, value, duration]);

    const animateValue = () => {
      const startTime = performance.now();
      const startValue = 0;

      const animate = (currentTime: number) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Apply easing
        let easedProgress = progress;
        if (easing === 'ease-out') {
          easedProgress = 1 - Math.pow(1 - progress, 3);
        } else if (easing === 'ease-in') {
          easedProgress = Math.pow(progress, 3);
        } else if (easing === 'ease-in-out') {
          easedProgress =
            progress < 0.5
              ? 4 * Math.pow(progress, 3)
              : 1 - Math.pow(-2 * progress + 2, 3) / 2;
        }

        const currentValue = startValue + (value - startValue) * easedProgress;
        setDisplayValue(currentValue);

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      requestAnimationFrame(animate);
    };

    return (
      <span
        ref={(node) => {
          (elementRef as React.MutableRefObject<HTMLSpanElement | null>).current = node;
          if (typeof ref === 'function') ref(node);
          else if (ref) ref.current = node;
        }}
        className={className}
        {...props}
      >
        {prefix}
        {displayValue.toFixed(decimals)}
        {suffix}
      </span>
    );
  }
);

// ============================================================================
// Add blink animation to global styles
// ============================================================================

// Add this to your globals.css:
// @keyframes blink {
//   0%, 50% { opacity: 1; }
//   51%, 100% { opacity: 0; }
// }
// .animate-blink {
//   animation: blink 1s step-end infinite;
// }