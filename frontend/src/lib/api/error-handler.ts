/**
 * Error Handler Utilities
 * Centralized error handling and user-friendly messages
 */

import { ApiError } from './types';
import { logger } from './logger';

// ============================================================================
// Error Messages
// ============================================================================

const ERROR_MESSAGES: Record<string, string> = {
  // Network errors
  NETWORK_ERROR: 'Unable to connect to the server. Please check your internet connection.',
  TIMEOUT: 'The request timed out. Please try again.',
  ECONNABORTED: 'Connection was aborted. Please try again.',

  // Authentication errors
  UNAUTHORIZED: 'Your session has expired. Please log in again.',
  FORBIDDEN: 'You do not have permission to perform this action.',
  INVALID_TOKEN: 'Invalid authentication token. Please log in again.',
  TOKEN_EXPIRED: 'Your session has expired. Please log in again.',

  // Validation errors
  VALIDATION_ERROR: 'Please check your input and try again.',
  INVALID_INPUT: 'Invalid input provided.',
  MISSING_FIELD: 'Required field is missing.',

  // Resource errors
  NOT_FOUND: 'The requested resource was not found.',
  ALREADY_EXISTS: 'This resource already exists.',
  CONFLICT: 'A conflict occurred. Please refresh and try again.',

  // Server errors
  INTERNAL_ERROR: 'An unexpected error occurred. Please try again later.',
  SERVICE_UNAVAILABLE: 'The service is temporarily unavailable. Please try again later.',

  // Rate limiting
  RATE_LIMITED: 'Too many requests. Please wait a moment and try again.',
  QUOTA_EXCEEDED: 'You have exceeded your quota. Please upgrade your plan.',

  // Provider errors
  PROVIDER_ERROR: 'AI provider error. Please try again or switch providers.',
  MODEL_UNAVAILABLE: 'The selected model is currently unavailable.',

  // Jailbreak errors
  JAILBREAK_FAILED: 'Jailbreak attempt failed. Try a different technique.',
  ATTACK_TIMEOUT: 'The attack timed out. Try reducing iterations.',
};

// ============================================================================
// Error Handler Class
// ============================================================================

export class ErrorHandler {
  private static instance: ErrorHandler;
  private errorListeners: Set<(error: ApiError) => void> = new Set();

  private constructor() {}

  static getInstance(): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler();
    }
    return ErrorHandler.instance;
  }

  /**
   * Get user-friendly error message
   */
  getMessage(error: ApiError): string {
    // Check for specific error code message
    if (error.code && ERROR_MESSAGES[error.code]) {
      return ERROR_MESSAGES[error.code];
    }

    // Check for HTTP status-based message
    const statusMessages: Record<number, string> = {
      400: 'Invalid request. Please check your input.',
      401: 'Please log in to continue.',
      403: 'Access denied.',
      404: 'Resource not found.',
      408: 'Request timed out.',
      429: 'Too many requests. Please slow down.',
      500: 'Server error. Please try again later.',
      502: 'Service temporarily unavailable.',
      503: 'Service maintenance in progress.',
      504: 'Gateway timeout. Please try again.',
    };

    if (error.status && statusMessages[error.status]) {
      return statusMessages[error.status];
    }

    // Return original message or generic error
    return error.message || 'An unexpected error occurred.';
  }

  /**
   * Handle error with logging and notification
   */
  handle(error: ApiError, context?: string): void {
    const message = this.getMessage(error);

    logger.logError(context || 'API Error', error, {
      code: error.code,
      status: error.status,
      requestId: error.request_id,
    });

    // Notify listeners
    this.errorListeners.forEach(listener => listener(error));
  }

  /**
   * Subscribe to errors
   */
  onError(callback: (error: ApiError) => void): () => void {
    this.errorListeners.add(callback);
    return () => this.errorListeners.delete(callback);
  }

  /**
   * Check if error is retryable
   */
  isRetryable(error: ApiError): boolean {
    const retryableCodes = new Set([
      'TIMEOUT',
      'SERVICE_UNAVAILABLE',
      'RATE_LIMITED',
      'PROVIDER_ERROR',
    ]);

    const retryableStatuses = new Set([408, 429, 500, 502, 503, 504]);

    return Boolean(
      (error.code && retryableCodes.has(error.code)) ||
      (error.status && retryableStatuses.has(error.status))
    );
  }

  /**
   * Check if error requires re-authentication
   */
  requiresAuth(error: ApiError): boolean {
    const authCodes = new Set(['UNAUTHORIZED', 'INVALID_TOKEN', 'TOKEN_EXPIRED']);
    return error.status === 401 || Boolean(error.code && authCodes.has(error.code));
  }

  /**
   * Create error from HTTP response
   */
  fromResponse(status: number, data: ErrorResponseData | null, requestId?: string): ApiError {
    return {
      message: data?.message || data?.detail || 'Request failed',
      code: data?.code || 'UNKNOWN_ERROR',
      status,
      details: data?.details,
      request_id: requestId,
      timestamp: new Date().toISOString(),
    };
  }
}

/**
 * Error response data interface
 */
interface ErrorResponseData {
  message?: string;
  detail?: string;
  code?: string;
  details?: Record<string, unknown>;
}

// Export singleton instance
export const errorHandler = ErrorHandler.getInstance();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format validation errors for form display
 */
export function formatValidationErrors(
  details: Array<{ field: string; message: string }> | undefined
): Record<string, string[]> {
  if (!details) return {};

  const errors: Record<string, string[]> = {};

  for (const detail of details) {
    if (!errors[detail.field]) {
      errors[detail.field] = [];
    }
    errors[detail.field].push(detail.message);
  }

  return errors;
}

/**
 * Check if error is a network error
 */
export function isNetworkError(error: unknown): boolean {
  if (typeof error !== 'object' || error === null) {
    return false;
  }

  const err = error as { code?: string; message?: string };
  return (
    err.code === 'NETWORK_ERROR' ||
    err.code === 'ECONNABORTED' ||
    (typeof err.message === 'string' && err.message.includes('Network Error')) ||
    (typeof err.message === 'string' && err.message.includes('timeout'))
  );
}

/**
 * Get retry delay based on error
 */
export function getRetryDelay(error: ApiError, attempt: number): number {
  // Check for Retry-After header value
  const retryAfter = error.details?.retryAfter;
  if (typeof retryAfter === 'number') {
    return retryAfter * 1000;
  }

  // Exponential backoff with jitter
  const baseDelay = 1000;
  const maxDelay = 30000;
  const exponentialDelay = baseDelay * Math.pow(2, attempt - 1);
  const jitter = exponentialDelay * 0.1 * Math.random();

  return Math.min(exponentialDelay + jitter, maxDelay);
}
