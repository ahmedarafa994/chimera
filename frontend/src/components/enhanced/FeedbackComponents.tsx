'use client';

/**
 * Feedback Components
 * 
 * Provides user feedback UI elements:
 * - Toast notifications
 * - Alert banners
 * - Confirmation dialogs
 * - Status indicators
 * - Empty states
 * - Error boundaries
 * 
 * Built on Shadcn/Radix primitives
 */

import * as React from 'react';
import { forwardRef, useState, useCallback, createContext, useContext } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Info,
  X,
  RefreshCw,
  Wifi,
  WifiOff,
  Clock,
  Zap,
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

export type FeedbackType = 'success' | 'error' | 'warning' | 'info';
export type StatusType = 'online' | 'offline' | 'busy' | 'away' | 'idle';

// ============================================================================
// Alert Banner Component
// ============================================================================

export interface AlertBannerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Alert type */
  type?: FeedbackType;
  /** Title */
  title?: string;
  /** Description */
  description?: string;
  /** Dismissible */
  dismissible?: boolean;
  /** Action button */
  action?: {
    label: string;
    onClick: () => void;
  };
  /** Icon override */
  icon?: React.ReactNode;
  /** Callback when dismissed */
  onDismiss?: () => void;
}

const alertIcons: Record<FeedbackType, React.ReactNode> = {
  success: <CheckCircle className="h-4 w-4" />,
  error: <XCircle className="h-4 w-4" />,
  warning: <AlertTriangle className="h-4 w-4" />,
  info: <Info className="h-4 w-4" />,
};

const alertStyles: Record<FeedbackType, string> = {
  success: 'border-green-500/50 bg-green-500/10 text-green-700 dark:text-green-400',
  error: 'border-red-500/50 bg-red-500/10 text-red-700 dark:text-red-400',
  warning: 'border-yellow-500/50 bg-yellow-500/10 text-yellow-700 dark:text-yellow-400',
  info: 'border-blue-500/50 bg-blue-500/10 text-blue-700 dark:text-blue-400',
};

export const AlertBanner = forwardRef<HTMLDivElement, AlertBannerProps>(
  function AlertBanner(
    {
      className,
      type = 'info',
      title,
      description,
      dismissible = false,
      action,
      icon,
      onDismiss,
      ...props
    },
    ref
  ) {
    const [isDismissed, setIsDismissed] = useState(false);

    const handleDismiss = () => {
      setIsDismissed(true);
      onDismiss?.();
    };

    if (isDismissed) return null;

    return (
      <Alert
        ref={ref}
        className={cn(
          'relative',
          alertStyles[type],
          'animate-in slide-in-from-top-2 fade-in duration-300',
          className
        )}
        {...props}
      >
        <div className="flex items-start gap-3">
          <span className="flex-shrink-0 mt-0.5">
            {icon || alertIcons[type]}
          </span>
          <div className="flex-1 min-w-0">
            {title && <AlertTitle className="font-semibold">{title}</AlertTitle>}
            {description && (
              <AlertDescription className="mt-1 text-sm opacity-90">
                {description}
              </AlertDescription>
            )}
            {action && (
              <Button
                variant="link"
                size="sm"
                className="mt-2 h-auto p-0 font-medium"
                onClick={action.onClick}
              >
                {action.label}
              </Button>
            )}
          </div>
          {dismissible && (
            <button
              onClick={handleDismiss}
              className="flex-shrink-0 p-1 rounded-md hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
              aria-label="Dismiss"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </Alert>
    );
  }
);

// ============================================================================
// Status Indicator Component
// ============================================================================

export interface StatusIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Status type */
  status: StatusType;
  /** Show label */
  showLabel?: boolean;
  /** Custom label */
  label?: string;
  /** Size */
  size?: 'sm' | 'md' | 'lg';
  /** Pulse animation */
  pulse?: boolean;
}

const statusColors: Record<StatusType, string> = {
  online: 'bg-green-500',
  offline: 'bg-gray-400',
  busy: 'bg-red-500',
  away: 'bg-yellow-500',
  idle: 'bg-gray-400',
};

const statusLabels: Record<StatusType, string> = {
  online: 'Online',
  offline: 'Offline',
  busy: 'Busy',
  away: 'Away',
  idle: 'Idle',
};

const statusSizes = {
  sm: 'h-2 w-2',
  md: 'h-3 w-3',
  lg: 'h-4 w-4',
};

export const StatusIndicator = forwardRef<HTMLDivElement, StatusIndicatorProps>(
  function StatusIndicator(
    {
      className,
      status,
      showLabel = false,
      label,
      size = 'md',
      pulse = true,
      ...props
    },
    ref
  ) {
    return (
      <div
        ref={ref}
        className={cn('flex items-center gap-2', className)}
        {...props}
      >
        <span className="relative flex">
          <span
            className={cn(
              'rounded-full',
              statusColors[status],
              statusSizes[size]
            )}
          />
          {pulse && status === 'online' && (
            <span
              className={cn(
                'absolute inset-0 rounded-full animate-ping',
                statusColors[status],
                'opacity-75'
              )}
            />
          )}
        </span>
        {showLabel && (
          <span className="text-sm text-muted-foreground">
            {label || statusLabels[status]}
          </span>
        )}
      </div>
    );
  }
);

// ============================================================================
// Connection Status Component
// ============================================================================

export interface ConnectionStatusProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Is connected */
  isConnected: boolean;
  /** Is reconnecting */
  isReconnecting?: boolean;
  /** Latency (ms) */
  latency?: number;
  /** Show details */
  showDetails?: boolean;
}

export const ConnectionStatus = forwardRef<HTMLDivElement, ConnectionStatusProps>(
  function ConnectionStatus(
    {
      className,
      isConnected,
      isReconnecting = false,
      latency,
      showDetails = false,
      ...props
    },
    ref
  ) {
    const getLatencyColor = (ms: number) => {
      if (ms < 100) return 'text-green-500';
      if (ms < 300) return 'text-yellow-500';
      return 'text-red-500';
    };

    return (
      <div
        ref={ref}
        className={cn('flex items-center gap-2', className)}
        {...props}
      >
        {isReconnecting ? (
          <>
            <RefreshCw className="h-4 w-4 animate-spin text-yellow-500" />
            <span className="text-sm text-yellow-500">Reconnecting...</span>
          </>
        ) : isConnected ? (
          <>
            <Wifi className="h-4 w-4 text-green-500" />
            {showDetails && (
              <span className="text-sm text-muted-foreground">
                Connected
                {latency !== undefined && (
                  <span className={cn('ml-1', getLatencyColor(latency))}>
                    ({latency}ms)
                  </span>
                )}
              </span>
            )}
          </>
        ) : (
          <>
            <WifiOff className="h-4 w-4 text-red-500" />
            {showDetails && (
              <span className="text-sm text-red-500">Disconnected</span>
            )}
          </>
        )}
      </div>
    );
  }
);

// ============================================================================
// Confirmation Dialog Component
// ============================================================================

export interface ConfirmDialogProps {
  /** Is open */
  open: boolean;
  /** On open change */
  onOpenChange: (open: boolean) => void;
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** Confirm button text */
  confirmText?: string;
  /** Cancel button text */
  cancelText?: string;
  /** Confirm button variant */
  confirmVariant?: 'default' | 'destructive';
  /** On confirm */
  onConfirm: () => void | Promise<void>;
  /** On cancel */
  onCancel?: () => void;
  /** Loading state */
  loading?: boolean;
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  confirmVariant = 'default',
  onConfirm,
  onCancel,
  loading = false,
}: ConfirmDialogProps) {
  const handleConfirm = async () => {
    await onConfirm();
    onOpenChange(false);
  };

  const handleCancel = () => {
    onCancel?.();
    onOpenChange(false);
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={handleCancel} disabled={loading}>
            {cancelText}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleConfirm}
            disabled={loading}
            className={cn(
              confirmVariant === 'destructive' &&
              'bg-destructive text-destructive-foreground hover:bg-destructive/90'
            )}
          >
            {loading ? (
              <RefreshCw className="h-4 w-4 animate-spin mr-2" />
            ) : null}
            {confirmText}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// ============================================================================
// Empty State Component
// ============================================================================

export interface EmptyStateProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Icon */
  icon?: React.ReactNode;
  /** Title */
  title: string;
  /** Description */
  description?: string;
  /** Primary action */
  action?: {
    label: string;
    onClick: () => void;
  };
  /** Secondary action */
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  /** Size */
  size?: 'sm' | 'md' | 'lg';
}

const emptySizes = {
  sm: {
    container: 'py-8',
    icon: 'h-8 w-8',
    title: 'text-base',
    description: 'text-sm',
  },
  md: {
    container: 'py-12',
    icon: 'h-12 w-12',
    title: 'text-lg',
    description: 'text-sm',
  },
  lg: {
    container: 'py-16',
    icon: 'h-16 w-16',
    title: 'text-xl',
    description: 'text-base',
  },
};

export const EmptyState = forwardRef<HTMLDivElement, EmptyStateProps>(
  function EmptyState(
    {
      className,
      icon,
      title,
      description,
      action,
      secondaryAction,
      size = 'md',
      ...props
    },
    ref
  ) {
    const sizeStyles = emptySizes[size];

    return (
      <div
        ref={ref}
        className={cn(
          'flex flex-col items-center justify-center text-center',
          sizeStyles.container,
          className
        )}
        {...props}
      >
        {icon && (
          <div
            className={cn(
              'mb-4 text-muted-foreground/50',
              sizeStyles.icon
            )}
          >
            {icon}
          </div>
        )}
        <h3 className={cn('font-semibold', sizeStyles.title)}>{title}</h3>
        {description && (
          <p
            className={cn(
              'mt-2 text-muted-foreground max-w-sm',
              sizeStyles.description
            )}
          >
            {description}
          </p>
        )}
        {(action || secondaryAction) && (
          <div className="mt-6 flex items-center gap-3">
            {action && (
              <Button onClick={action.onClick}>{action.label}</Button>
            )}
            {secondaryAction && (
              <Button variant="outline" onClick={secondaryAction.onClick}>
                {secondaryAction.label}
              </Button>
            )}
          </div>
        )}
      </div>
    );
  }
);

// ============================================================================
// Error State Component
// ============================================================================

export interface ErrorStateProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Error title */
  title?: string;
  /** Error message */
  message?: string;
  /** Error object */
  error?: Error | null;
  /** Retry action */
  onRetry?: () => void;
  /** Show stack trace (dev only) */
  showStack?: boolean;
}

export const ErrorState = forwardRef<HTMLDivElement, ErrorStateProps>(
  function ErrorState(
    {
      className,
      title = 'Something went wrong',
      message,
      error,
      onRetry,
      showStack = process.env.NODE_ENV === 'development',
      ...props
    },
    ref
  ) {
    const errorMessage = message || error?.message || 'An unexpected error occurred';

    return (
      <div
        ref={ref}
        className={cn(
          'flex flex-col items-center justify-center text-center py-12',
          className
        )}
        {...props}
      >
        <div className="mb-4 p-4 rounded-full bg-red-500/10">
          <XCircle className="h-8 w-8 text-red-500" />
        </div>
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="mt-2 text-sm text-muted-foreground max-w-sm">
          {errorMessage}
        </p>
        {showStack && error?.stack && (
          <pre className="mt-4 p-4 bg-muted rounded-lg text-xs text-left overflow-auto max-w-full max-h-40">
            {error.stack}
          </pre>
        )}
        {onRetry && (
          <Button onClick={onRetry} className="mt-6">
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
        )}
      </div>
    );
  }
);

// ============================================================================
// Progress Badge Component
// ============================================================================

export interface ProgressBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Current value */
  value: number;
  /** Maximum value */
  max?: number;
  /** Show percentage */
  showPercentage?: boolean;
  /** Label */
  label?: string;
  /** Variant */
  variant?: 'default' | 'success' | 'warning' | 'error';
}

const progressVariants = {
  default: 'bg-primary/10 text-primary',
  success: 'bg-green-500/10 text-green-600',
  warning: 'bg-yellow-500/10 text-yellow-600',
  error: 'bg-red-500/10 text-red-600',
};

export const ProgressBadge = forwardRef<HTMLDivElement, ProgressBadgeProps>(
  function ProgressBadge(
    {
      className,
      value,
      max = 100,
      showPercentage = true,
      label,
      variant = 'default',
      ...props
    },
    ref
  ) {
    const percentage = Math.round((value / max) * 100);

    return (
      <Badge
        ref={ref}
        variant="outline"
        className={cn(
          'font-medium',
          progressVariants[variant],
          className
        )}
        {...props}
      >
        {label && <span className="mr-1">{label}</span>}
        {showPercentage ? `${percentage}%` : `${value}/${max}`}
      </Badge>
    );
  }
);

// ============================================================================
// Time Ago Component
// ============================================================================

export interface TimeAgoProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Date to display */
  date: Date | string | number;
  /** Update interval (ms) */
  updateInterval?: number;
  /** Show icon */
  showIcon?: boolean;
}

function getTimeAgo(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 60) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return date.toLocaleDateString();
}

export const TimeAgo = forwardRef<HTMLSpanElement, TimeAgoProps>(
  function TimeAgo(
    { className, date, updateInterval = 60000, showIcon = false, ...props },
    ref
  ) {
    const [timeAgo, setTimeAgo] = useState(() =>
      getTimeAgo(new Date(date))
    );

    React.useEffect(() => {
      const interval = setInterval(() => {
        setTimeAgo(getTimeAgo(new Date(date)));
      }, updateInterval);

      return () => clearInterval(interval);
    }, [date, updateInterval]);

    return (
      <span
        ref={ref}
        className={cn('text-sm text-muted-foreground', className)}
        title={new Date(date).toLocaleString()}
        {...props}
      >
        {showIcon && <Clock className="inline h-3 w-3 mr-1" />}
        {timeAgo}
      </span>
    );
  }
);

// ============================================================================
// Notification Context & Provider
// ============================================================================

export interface Notification {
  id: string;
  type: FeedbackType;
  title: string;
  description?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationContextValue {
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = useCallback(
    (notification: Omit<Notification, 'id'>) => {
      const id = Math.random().toString(36).slice(2);
      const newNotification = { ...notification, id };

      setNotifications((prev) => [...prev, newNotification]);

      // Auto-remove after duration
      if (notification.duration !== 0) {
        setTimeout(() => {
          setNotifications((prev) => prev.filter((n) => n.id !== id));
        }, notification.duration || 5000);
      }

      return id;
    },
    []
  );

  const removeNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const clearNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        addNotification,
        removeNotification,
        clearNotifications,
      }}
    >
      {children}
      <NotificationContainer notifications={notifications} onDismiss={removeNotification} />
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
}

// ============================================================================
// Notification Container
// ============================================================================

interface NotificationContainerProps {
  notifications: Notification[];
  onDismiss: (id: string) => void;
}

function NotificationContainer({ notifications, onDismiss }: NotificationContainerProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {notifications.map((notification) => (
        <AlertBanner
          key={notification.id}
          type={notification.type}
          title={notification.title}
          description={notification.description}
          action={notification.action}
          dismissible
          onDismiss={() => onDismiss(notification.id)}
          className="shadow-lg"
        />
      ))}
    </div>
  );
}

// ============================================================================
// Quick Feedback Component
// ============================================================================

export interface QuickFeedbackProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Feedback type */
  type: FeedbackType;
  /** Message */
  message: string;
  /** Auto-hide duration (ms, 0 to disable) */
  duration?: number;
  /** On hide callback */
  onHide?: () => void;
}

export const QuickFeedback = forwardRef<HTMLDivElement, QuickFeedbackProps>(
  function QuickFeedback(
    { className, type, message, duration = 3000, onHide, ...props },
    ref
  ) {
    const [isVisible, setIsVisible] = useState(true);

    React.useEffect(() => {
      if (duration > 0) {
        const timer = setTimeout(() => {
          setIsVisible(false);
          onHide?.();
        }, duration);
        return () => clearTimeout(timer);
      }
    }, [duration, onHide]);

    if (!isVisible) return null;

    return (
      <div
        ref={ref}
        className={cn(
          'fixed bottom-4 left-1/2 -translate-x-1/2 z-50',
          'px-4 py-2 rounded-full shadow-lg',
          'flex items-center gap-2',
          'animate-in slide-in-from-bottom-4 fade-in duration-300',
          type === 'success' && 'bg-green-500 text-white',
          type === 'error' && 'bg-red-500 text-white',
          type === 'warning' && 'bg-yellow-500 text-white',
          type === 'info' && 'bg-blue-500 text-white',
          className
        )}
        {...props}
      >
        {alertIcons[type]}
        <span className="text-sm font-medium">{message}</span>
      </div>
    );
  }
);