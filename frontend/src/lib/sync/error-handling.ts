/**
 * Error Handling Module
 * 
 * Comprehensive error handling with:
 * - Error classification and mapping
 * - User-friendly error messages
 * - Error recovery strategies
 * - Error boundary integration
 * - Logging and reporting
 * 
 * @module lib/sync/error-handling
 */

import { useCallback, useState, useEffect, useRef } from 'react';
import { getEventBus } from './event-bus';

// ============================================================================
// Error Types
// ============================================================================

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';
export type ErrorCategory =
  | 'network'
  | 'validation'
  | 'authentication'
  | 'authorization'
  | 'server'
  | 'client'
  | 'timeout'
  | 'rate_limit'
  | 'unknown';

export interface AppError {
  name: string;
  message: string;
  code: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  userMessage: string;
  technicalMessage: string;
  recoverable: boolean;
  retryable: boolean;
  context?: Record<string, unknown>;
  originalError?: Error;
  timestamp: Date;
  stack?: string;
}

export interface ErrorRecoveryAction {
  label: string;
  action: () => void | Promise<void>;
  primary?: boolean;
}

export interface ErrorDisplayConfig {
  title: string;
  message: string;
  severity: ErrorSeverity;
  actions: ErrorRecoveryAction[];
  dismissible: boolean;
  autoHide?: number;
}

// ============================================================================
// Error Codes
// ============================================================================

export const ERROR_CODES = {
  // Network errors
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT: 'TIMEOUT',
  CONNECTION_LOST: 'CONNECTION_LOST',
  DNS_ERROR: 'DNS_ERROR',

  // HTTP errors
  BAD_REQUEST: 'BAD_REQUEST',
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  METHOD_NOT_ALLOWED: 'METHOD_NOT_ALLOWED',
  CONFLICT: 'CONFLICT',
  UNPROCESSABLE_ENTITY: 'UNPROCESSABLE_ENTITY',
  TOO_MANY_REQUESTS: 'TOO_MANY_REQUESTS',
  INTERNAL_SERVER_ERROR: 'INTERNAL_SERVER_ERROR',
  BAD_GATEWAY: 'BAD_GATEWAY',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',
  GATEWAY_TIMEOUT: 'GATEWAY_TIMEOUT',

  // Application errors
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  PARSE_ERROR: 'PARSE_ERROR',
  STATE_ERROR: 'STATE_ERROR',
  SYNC_ERROR: 'SYNC_ERROR',
  WEBSOCKET_ERROR: 'WEBSOCKET_ERROR',

  // Auth errors
  SESSION_EXPIRED: 'SESSION_EXPIRED',
  INVALID_TOKEN: 'INVALID_TOKEN',
  INSUFFICIENT_PERMISSIONS: 'INSUFFICIENT_PERMISSIONS',

  // Unknown
  UNKNOWN_ERROR: 'UNKNOWN_ERROR',
} as const;

export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];

// ============================================================================
// Error Messages
// ============================================================================

const ERROR_MESSAGES: Record<ErrorCode, { user: string; technical: string }> = {
  [ERROR_CODES.NETWORK_ERROR]: {
    user: 'Unable to connect to the server. Please check your internet connection.',
    technical: 'Network request failed - no response received',
  },
  [ERROR_CODES.TIMEOUT]: {
    user: 'The request took too long. Please try again.',
    technical: 'Request timeout exceeded',
  },
  [ERROR_CODES.CONNECTION_LOST]: {
    user: 'Connection lost. Attempting to reconnect...',
    technical: 'WebSocket/SSE connection terminated unexpectedly',
  },
  [ERROR_CODES.DNS_ERROR]: {
    user: 'Unable to reach the server. Please try again later.',
    technical: 'DNS resolution failed',
  },
  [ERROR_CODES.BAD_REQUEST]: {
    user: 'Invalid request. Please check your input and try again.',
    technical: 'Server returned 400 Bad Request',
  },
  [ERROR_CODES.UNAUTHORIZED]: {
    user: 'Please sign in to continue.',
    technical: 'Server returned 401 Unauthorized',
  },
  [ERROR_CODES.FORBIDDEN]: {
    user: "You don't have permission to perform this action.",
    technical: 'Server returned 403 Forbidden',
  },
  [ERROR_CODES.NOT_FOUND]: {
    user: 'The requested resource was not found.',
    technical: 'Server returned 404 Not Found',
  },
  [ERROR_CODES.METHOD_NOT_ALLOWED]: {
    user: 'This action is not allowed.',
    technical: 'Server returned 405 Method Not Allowed',
  },
  [ERROR_CODES.CONFLICT]: {
    user: 'This action conflicts with existing data. Please refresh and try again.',
    technical: 'Server returned 409 Conflict',
  },
  [ERROR_CODES.UNPROCESSABLE_ENTITY]: {
    user: 'The provided data is invalid. Please check and try again.',
    technical: 'Server returned 422 Unprocessable Entity',
  },
  [ERROR_CODES.TOO_MANY_REQUESTS]: {
    user: 'Too many requests. Please wait a moment and try again.',
    technical: 'Server returned 429 Too Many Requests',
  },
  [ERROR_CODES.INTERNAL_SERVER_ERROR]: {
    user: 'Something went wrong on our end. Please try again later.',
    technical: 'Server returned 500 Internal Server Error',
  },
  [ERROR_CODES.BAD_GATEWAY]: {
    user: 'Service temporarily unavailable. Please try again.',
    technical: 'Server returned 502 Bad Gateway',
  },
  [ERROR_CODES.SERVICE_UNAVAILABLE]: {
    user: 'Service is temporarily unavailable. Please try again later.',
    technical: 'Server returned 503 Service Unavailable',
  },
  [ERROR_CODES.GATEWAY_TIMEOUT]: {
    user: 'The server took too long to respond. Please try again.',
    technical: 'Server returned 504 Gateway Timeout',
  },
  [ERROR_CODES.VALIDATION_ERROR]: {
    user: 'Please check your input and try again.',
    technical: 'Client-side validation failed',
  },
  [ERROR_CODES.PARSE_ERROR]: {
    user: 'Unable to process the response. Please try again.',
    technical: 'Failed to parse server response',
  },
  [ERROR_CODES.STATE_ERROR]: {
    user: 'Application state error. Please refresh the page.',
    technical: 'Invalid application state detected',
  },
  [ERROR_CODES.SYNC_ERROR]: {
    user: 'Unable to sync data. Your changes will be saved locally.',
    technical: 'State synchronization failed',
  },
  [ERROR_CODES.WEBSOCKET_ERROR]: {
    user: 'Real-time connection error. Reconnecting...',
    technical: 'WebSocket error occurred',
  },
  [ERROR_CODES.SESSION_EXPIRED]: {
    user: 'Your session has expired. Please sign in again.',
    technical: 'Authentication token expired',
  },
  [ERROR_CODES.INVALID_TOKEN]: {
    user: 'Authentication error. Please sign in again.',
    technical: 'Invalid or malformed authentication token',
  },
  [ERROR_CODES.INSUFFICIENT_PERMISSIONS]: {
    user: "You don't have permission to access this resource.",
    technical: 'User lacks required permissions',
  },
  [ERROR_CODES.UNKNOWN_ERROR]: {
    user: 'An unexpected error occurred. Please try again.',
    technical: 'Unknown error',
  },
};

// ============================================================================
// Error Classification
// ============================================================================

/**
 * Classify error by HTTP status code
 */
function classifyByStatusCode(status: number): {
  code: ErrorCode;
  category: ErrorCategory;
  severity: ErrorSeverity;
  recoverable: boolean;
  retryable: boolean;
} {
  switch (status) {
    case 400:
      return {
        code: ERROR_CODES.BAD_REQUEST,
        category: 'validation',
        severity: 'medium',
        recoverable: true,
        retryable: false,
      };
    case 401:
      return {
        code: ERROR_CODES.UNAUTHORIZED,
        category: 'authentication',
        severity: 'high',
        recoverable: true,
        retryable: false,
      };
    case 403:
      return {
        code: ERROR_CODES.FORBIDDEN,
        category: 'authorization',
        severity: 'high',
        recoverable: false,
        retryable: false,
      };
    case 404:
      return {
        code: ERROR_CODES.NOT_FOUND,
        category: 'client',
        severity: 'medium',
        recoverable: true,
        retryable: false,
      };
    case 405:
      return {
        code: ERROR_CODES.METHOD_NOT_ALLOWED,
        category: 'client',
        severity: 'medium',
        recoverable: false,
        retryable: false,
      };
    case 409:
      return {
        code: ERROR_CODES.CONFLICT,
        category: 'client',
        severity: 'medium',
        recoverable: true,
        retryable: false,
      };
    case 422:
      return {
        code: ERROR_CODES.UNPROCESSABLE_ENTITY,
        category: 'validation',
        severity: 'medium',
        recoverable: true,
        retryable: false,
      };
    case 429:
      return {
        code: ERROR_CODES.TOO_MANY_REQUESTS,
        category: 'rate_limit',
        severity: 'medium',
        recoverable: true,
        retryable: true,
      };
    case 500:
      return {
        code: ERROR_CODES.INTERNAL_SERVER_ERROR,
        category: 'server',
        severity: 'high',
        recoverable: true,
        retryable: true,
      };
    case 502:
      return {
        code: ERROR_CODES.BAD_GATEWAY,
        category: 'server',
        severity: 'high',
        recoverable: true,
        retryable: true,
      };
    case 503:
      return {
        code: ERROR_CODES.SERVICE_UNAVAILABLE,
        category: 'server',
        severity: 'high',
        recoverable: true,
        retryable: true,
      };
    case 504:
      return {
        code: ERROR_CODES.GATEWAY_TIMEOUT,
        category: 'timeout',
        severity: 'high',
        recoverable: true,
        retryable: true,
      };
    default:
      if (status >= 400 && status < 500) {
        return {
          code: ERROR_CODES.BAD_REQUEST,
          category: 'client',
          severity: 'medium',
          recoverable: true,
          retryable: false,
        };
      }
      if (status >= 500) {
        return {
          code: ERROR_CODES.INTERNAL_SERVER_ERROR,
          category: 'server',
          severity: 'high',
          recoverable: true,
          retryable: true,
        };
      }
      return {
        code: ERROR_CODES.UNKNOWN_ERROR,
        category: 'unknown',
        severity: 'medium',
        recoverable: true,
        retryable: false,
      };
  }
}

/**
 * Classify error by error type/message
 */
function classifyByErrorType(error: Error): {
  code: ErrorCode;
  category: ErrorCategory;
  severity: ErrorSeverity;
  recoverable: boolean;
  retryable: boolean;
} {
  const message = error.message.toLowerCase();
  const name = error.name.toLowerCase();

  // Network errors
  if (
    name === 'typeerror' && message.includes('failed to fetch') ||
    message.includes('network') ||
    message.includes('econnrefused') ||
    message.includes('enotfound')
  ) {
    return {
      code: ERROR_CODES.NETWORK_ERROR,
      category: 'network',
      severity: 'high',
      recoverable: true,
      retryable: true,
    };
  }

  // Timeout errors
  if (
    name === 'aborterror' ||
    message.includes('timeout') ||
    message.includes('etimedout')
  ) {
    return {
      code: ERROR_CODES.TIMEOUT,
      category: 'timeout',
      severity: 'medium',
      recoverable: true,
      retryable: true,
    };
  }

  // Parse errors
  if (
    name === 'syntaxerror' ||
    message.includes('json') ||
    message.includes('parse')
  ) {
    return {
      code: ERROR_CODES.PARSE_ERROR,
      category: 'client',
      severity: 'medium',
      recoverable: true,
      retryable: false,
    };
  }

  // WebSocket errors
  if (message.includes('websocket')) {
    return {
      code: ERROR_CODES.WEBSOCKET_ERROR,
      category: 'network',
      severity: 'medium',
      recoverable: true,
      retryable: true,
    };
  }

  return {
    code: ERROR_CODES.UNKNOWN_ERROR,
    category: 'unknown',
    severity: 'medium',
    recoverable: true,
    retryable: false,
  };
}

// ============================================================================
// Error Factory
// ============================================================================

/**
 * Create an AppError from any error
 */
export function createAppError(
  error: unknown,
  context?: Record<string, unknown>
): AppError {
  // Already an AppError
  if (isAppError(error)) {
    return {
      ...error,
      context: { ...error.context, ...context },
    };
  }

  // Axios-like error with response
  if (isAxiosError(error)) {
    const status = error.response?.status || 0;
    const classification = classifyByStatusCode(status);
    const messages = ERROR_MESSAGES[classification.code];

    const appError: AppError = {
      name: 'AppError',
      message: messages.user,
      code: classification.code,
      category: classification.category,
      severity: classification.severity,
      userMessage: messages.user,
      technicalMessage: error.response?.data?.message || messages.technical,
      recoverable: classification.recoverable,
      retryable: classification.retryable,
      context: {
        ...context,
        status,
        url: error.config?.url,
        method: error.config?.method,
        responseData: error.response?.data,
      },
      originalError: error as Error,
      timestamp: new Date(),
    };

    return appError;
  }

  // Standard Error
  if (error instanceof Error) {
    const classification = classifyByErrorType(error);
    const messages = ERROR_MESSAGES[classification.code];

    const appError: AppError = {
      name: 'AppError',
      message: messages.user,
      code: classification.code,
      category: classification.category,
      severity: classification.severity,
      userMessage: messages.user,
      technicalMessage: error.message || messages.technical,
      recoverable: classification.recoverable,
      retryable: classification.retryable,
      context,
      originalError: error,
      timestamp: new Date(),
    };

    return appError;
  }

  // Unknown error type
  const messages = ERROR_MESSAGES[ERROR_CODES.UNKNOWN_ERROR];
  return {
    name: 'AppError',
    message: messages.user,
    code: ERROR_CODES.UNKNOWN_ERROR,
    category: 'unknown',
    severity: 'medium',
    userMessage: messages.user,
    technicalMessage: String(error),
    recoverable: true,
    retryable: false,
    context,
    timestamp: new Date(),
  };
}

/**
 * Type guard for AppError
 */
export function isAppError(error: unknown): error is AppError {
  return (
    error !== null &&
    typeof error === 'object' &&
    'code' in error &&
    'category' in error &&
    'severity' in error &&
    'userMessage' in error
  );
}

/**
 * Type guard for Axios-like errors
 */
function isAxiosError(error: unknown): error is {
  response?: { status: number; data?: { message?: string } };
  config?: { url?: string; method?: string };
  message: string;
} {
  return (
    error !== null &&
    typeof error === 'object' &&
    'message' in error &&
    ('response' in error || 'config' in error)
  );
}

// ============================================================================
// Error Handler
// ============================================================================

export interface ErrorHandlerConfig {
  onError?: (error: AppError) => void;
  onRecovery?: (error: AppError) => void;
  logErrors?: boolean;
  reportErrors?: boolean;
  reportEndpoint?: string;
}

class ErrorHandler {
  private config: ErrorHandlerConfig = {
    logErrors: true,
    reportErrors: false,
  };
  private eventBus = getEventBus();
  private errorQueue: AppError[] = [];
  private isReporting = false;

  /**
   * Configure error handler
   */
  configure(config: Partial<ErrorHandlerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Handle an error
   */
  handle(error: unknown, context?: Record<string, unknown>): AppError {
    const appError = createAppError(error, context);

    // Log error
    if (this.config.logErrors) {
      this.logError(appError);
    }

    // Report error
    if (this.config.reportErrors) {
      this.queueErrorReport(appError);
    }

    // Emit event
    this.eventBus.emit('error:occurred', appError);

    // Call handler
    this.config.onError?.(appError);

    return appError;
  }

  /**
   * Report recovery
   */
  reportRecovery(error: AppError): void {
    this.eventBus.emit('error:recovered', error);
    this.config.onRecovery?.(error);
  }

  /**
   * Log error to console
   */
  private logError(error: AppError): void {
    const logData = {
      code: error.code,
      category: error.category,
      severity: error.severity,
      message: error.technicalMessage,
      context: error.context,
      timestamp: error.timestamp.toISOString(),
    };

    switch (error.severity) {
      case 'critical':
      case 'high':
        console.error('[ERROR]', logData);
        break;
      case 'medium':
        console.warn('[WARN]', logData);
        break;
      case 'low':
        console.info('[INFO]', logData);
        break;
    }
  }

  /**
   * Queue error for reporting
   */
  private queueErrorReport(error: AppError): void {
    this.errorQueue.push(error);
    this.flushErrorQueue();
  }

  /**
   * Flush error queue to reporting endpoint
   */
  private async flushErrorQueue(): Promise<void> {
    if (this.isReporting || this.errorQueue.length === 0) return;
    if (!this.config.reportEndpoint) return;

    this.isReporting = true;
    const errors = [...this.errorQueue];
    this.errorQueue = [];

    try {
      await fetch(this.config.reportEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          errors: errors.map((e) => ({
            code: e.code,
            category: e.category,
            severity: e.severity,
            message: e.technicalMessage,
            context: e.context,
            timestamp: e.timestamp.toISOString(),
            stack: e.originalError?.stack,
          })),
        }),
      });
    } catch {
      // Re-queue errors on failure
      this.errorQueue.unshift(...errors);
    } finally {
      this.isReporting = false;
    }
  }
}

// Singleton instance
let errorHandler: ErrorHandler | null = null;

export function getErrorHandler(): ErrorHandler {
  if (!errorHandler) {
    errorHandler = new ErrorHandler();
  }
  return errorHandler;
}

// ============================================================================
// Error Display Helpers
// ============================================================================

/**
 * Get display configuration for an error
 */
export function getErrorDisplayConfig(
  error: AppError,
  options: {
    onRetry?: () => void;
    onDismiss?: () => void;
    onSignIn?: () => void;
    onRefresh?: () => void;
  } = {}
): ErrorDisplayConfig {
  const actions: ErrorRecoveryAction[] = [];

  // Add retry action for retryable errors
  if (error.retryable && options.onRetry) {
    actions.push({
      label: 'Try Again',
      action: options.onRetry,
      primary: true,
    });
  }

  // Add sign in action for auth errors
  if (error.category === 'authentication' && options.onSignIn) {
    actions.push({
      label: 'Sign In',
      action: options.onSignIn,
      primary: true,
    });
  }

  // Add refresh action for state errors
  if (error.code === ERROR_CODES.STATE_ERROR && options.onRefresh) {
    actions.push({
      label: 'Refresh Page',
      action: options.onRefresh,
      primary: true,
    });
  }

  // Add dismiss action
  if (options.onDismiss) {
    actions.push({
      label: 'Dismiss',
      action: options.onDismiss,
    });
  }

  // Determine title based on category
  let title: string;
  switch (error.category) {
    case 'network':
      title = 'Connection Error';
      break;
    case 'authentication':
      title = 'Authentication Required';
      break;
    case 'authorization':
      title = 'Access Denied';
      break;
    case 'validation':
      title = 'Invalid Input';
      break;
    case 'server':
      title = 'Server Error';
      break;
    case 'timeout':
      title = 'Request Timeout';
      break;
    case 'rate_limit':
      title = 'Rate Limited';
      break;
    default:
      title = 'Error';
  }

  return {
    title,
    message: error.userMessage,
    severity: error.severity,
    actions,
    dismissible: error.severity !== 'critical',
    autoHide: error.severity === 'low' ? 5000 : undefined,
  };
}

// ============================================================================
// React Hooks
// ============================================================================

/**
 * Hook for error handling
 */
export function useErrorHandler(): {
  error: AppError | null;
  handleError: (error: unknown, context?: Record<string, unknown>) => AppError;
  clearError: () => void;
  displayConfig: ErrorDisplayConfig | null;
} {
  const [error, setError] = useState<AppError | null>(null);
  const handler = getErrorHandler();

  const handleError = useCallback(
    (err: unknown, context?: Record<string, unknown>) => {
      const appError = handler.handle(err, context);
      setError(appError);
      return appError;
    },
    [handler]
  );

  const clearError = useCallback(() => {
    if (error) {
      handler.reportRecovery(error);
    }
    setError(null);
  }, [error, handler]);

  const displayConfig = error
    ? getErrorDisplayConfig(error, {
      onRetry: error.retryable ? clearError : undefined,
      onDismiss: clearError,
      onRefresh: () => window.location.reload(),
    })
    : null;

  return { error, handleError, clearError, displayConfig };
}

/**
 * Hook for async error handling
 */
export function useAsyncError<T>(
  asyncFn: () => Promise<T>,
  options: {
    onSuccess?: (data: T) => void;
    onError?: (error: AppError) => void;
    retryOnError?: boolean;
    maxRetries?: number;
  } = {}
): {
  execute: () => Promise<T | undefined>;
  data: T | undefined;
  error: AppError | null;
  isLoading: boolean;
  retry: () => Promise<T | undefined>;
  reset: () => void;
} {
  const [data, setData] = useState<T | undefined>();
  const [error, setError] = useState<AppError | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const retryCountRef = useRef(0);
  const handler = getErrorHandler();

  const execute = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await asyncFn();
      setData(result);
      retryCountRef.current = 0;
      options.onSuccess?.(result);
      return result;
    } catch (err) {
      const appError = handler.handle(err);
      setError(appError);
      options.onError?.(appError);

      // Auto-retry if enabled
      if (
        options.retryOnError &&
        appError.retryable &&
        retryCountRef.current < (options.maxRetries || 3)
      ) {
        retryCountRef.current++;
        return execute();
      }

      return undefined;
    } finally {
      setIsLoading(false);
    }
  }, [asyncFn, handler, options]);

  const retry = useCallback(() => {
    retryCountRef.current = 0;
    return execute();
  }, [execute]);

  const reset = useCallback(() => {
    setData(undefined);
    setError(null);
    setIsLoading(false);
    retryCountRef.current = 0;
  }, []);

  return { execute, data, error, isLoading, retry, reset };
}

/**
 * Hook for error boundary fallback
 */
export function useErrorBoundaryFallback(
  error: Error,
  resetErrorBoundary: () => void
): {
  appError: AppError;
  displayConfig: ErrorDisplayConfig;
} {
  const appError = createAppError(error);
  const displayConfig = getErrorDisplayConfig(appError, {
    onRetry: appError.retryable ? resetErrorBoundary : undefined,
    onRefresh: () => window.location.reload(),
  });

  return { appError, displayConfig };
}

/**
 * Hook for global error listening
 */
export function useGlobalErrorListener(
  onError: (error: AppError) => void
): void {
  const eventBus = getEventBus();

  useEffect(() => {
    const subscription = eventBus.on<AppError>('error:occurred', (event) => {
      onError(event.payload);
    });
    return () => subscription.unsubscribe();
  }, [eventBus, onError]);
}

// ============================================================================
// Error Recovery Strategies
// ============================================================================

export interface RecoveryStrategy {
  canRecover: (error: AppError) => boolean;
  recover: (error: AppError) => Promise<void>;
}

/**
 * Create a recovery strategy for network errors
 */
export function createNetworkRecoveryStrategy(
  onOnline: () => void
): RecoveryStrategy {
  return {
    canRecover: (error) => error.category === 'network',
    recover: async (error) => {
      return new Promise((resolve) => {
        const handleOnline = () => {
          window.removeEventListener('online', handleOnline);
          onOnline();
          getErrorHandler().reportRecovery(error);
          resolve();
        };

        if (navigator.onLine) {
          onOnline();
          getErrorHandler().reportRecovery(error);
          resolve();
        } else {
          window.addEventListener('online', handleOnline);
        }
      });
    },
  };
}

/**
 * Create a recovery strategy for auth errors
 */
export function createAuthRecoveryStrategy(
  refreshToken: () => Promise<boolean>
): RecoveryStrategy {
  return {
    canRecover: (error) =>
      error.category === 'authentication' &&
      error.code !== ERROR_CODES.INVALID_TOKEN,
    recover: async (error) => {
      const success = await refreshToken();
      if (success) {
        getErrorHandler().reportRecovery(error);
      } else {
        throw new Error('Token refresh failed');
      }
    },
  };
}

/**
 * Apply recovery strategies
 */
export async function attemptRecovery(
  error: AppError,
  strategies: RecoveryStrategy[]
): Promise<boolean> {
  for (const strategy of strategies) {
    if (strategy.canRecover(error)) {
      try {
        await strategy.recover(error);
        return true;
      } catch {
        continue;
      }
    }
  }
  return false;
}