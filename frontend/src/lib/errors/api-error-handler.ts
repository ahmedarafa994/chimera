/**
 * Unified Error Handler for API Client
 *
 * Provides consistent error handling, logging, and user-friendly messages
 * across all API calls in the application.
 */

interface ApiErrorDetails {
  code?: string;
  field?: string;
  context?: Record<string, any>;
}

export class ApiError extends Error {
  public readonly status: number;
  public readonly code: string;
  public readonly details?: ApiErrorDetails;
  public readonly timestamp: string;
  public readonly requestId?: string;

  constructor(
    message: string,
    status: number = 500,
    code: string = 'UNKNOWN_ERROR',
    details?: ApiErrorDetails,
    requestId?: string
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
    this.timestamp = new Date().toISOString();
    this.requestId = requestId;
  }

  /**
   * Convert error to user-friendly message
   */
  toUserMessage(): string {
    switch (this.status) {
      case 401:
        return 'Authentication required. Please log in and try again.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return 'The requested resource was not found.';
      case 409:
        return 'This action conflicts with the current state. Please refresh and try again.';
      case 429:
        return 'Too many requests. Please wait a moment and try again.';
      case 500:
        return 'A server error occurred. Please try again later.';
      case 502:
      case 503:
      case 504:
        return 'The service is temporarily unavailable. Please try again later.';
      default:
        return this.message || 'An unexpected error occurred.';
    }
  }

  /**
   * Check if error is retryable
   */
  isRetryable(): boolean {
    return [429, 502, 503, 504].includes(this.status);
  }

  /**
   * Check if error indicates network issues
   */
  isNetworkError(): boolean {
    return [0, 502, 503, 504].includes(this.status) || this.code === 'NETWORK_ERROR';
  }
}

export interface ErrorHandlerOptions {
  /** Show user-friendly toast notification */
  showToast?: boolean;
  /** Log error to console/service */
  logError?: boolean;
  /** Include request context in logs */
  includeContext?: boolean;
  /** Custom error message */
  customMessage?: string;
}

export class ApiErrorHandler {
  private static instance: ApiErrorHandler;

  public static getInstance(): ApiErrorHandler {
    if (!ApiErrorHandler.instance) {
      ApiErrorHandler.instance = new ApiErrorHandler();
    }
    return ApiErrorHandler.instance;
  }

  /**
   * Handle API errors with consistent logging and user feedback
   */
  handleError(
    error: unknown,
    context: string = 'API Call',
    options: ErrorHandlerOptions = {}
  ): ApiError {
    const {
      showToast = true,
      logError = true,
      includeContext = true,
      customMessage
    } = options;

    let apiError: ApiError;

    if (error instanceof ApiError) {
      apiError = error;
    } else if (error instanceof Error) {
      // Transform common error types
      if (error.name === 'AbortError') {
        apiError = new ApiError('Request was cancelled', 0, 'REQUEST_ABORTED');
      } else if (error.message.includes('fetch')) {
        apiError = new ApiError('Network connection failed', 0, 'NETWORK_ERROR');
      } else if (error.message.includes('timeout')) {
        apiError = new ApiError('Request timed out', 408, 'TIMEOUT_ERROR');
      } else {
        apiError = new ApiError(error.message, 500, 'CLIENT_ERROR');
      }
    } else {
      apiError = new ApiError('Unknown error occurred', 500, 'UNKNOWN_ERROR');
    }

    // Log error if enabled
    if (logError) {
      const logData = {
        message: apiError.message,
        status: apiError.status,
        code: apiError.code,
        context,
        timestamp: apiError.timestamp,
        requestId: apiError.requestId,
        ...(includeContext && apiError.details ? { details: apiError.details } : {}),
      };

      if (apiError.status >= 500) {
        console.error('[ApiError] Server Error:', logData);
      } else if (apiError.status >= 400) {
        console.warn('[ApiError] Client Error:', logData);
      } else {
        console.info('[ApiError] Request Error:', logData);
      }
    }

    // Show user notification if enabled
    if (showToast && typeof window !== 'undefined') {
      const userMessage = customMessage || apiError.toUserMessage();
      this.showUserNotification(userMessage, apiError.status >= 500 ? 'error' : 'warning');
    }

    return apiError;
  }

  /**
   * Transform Axios-like errors to ApiError
   */
  transformAxiosError(error: any): ApiError {
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data;

      let message = error.message;
      let code = 'HTTP_ERROR';
      let details: ApiErrorDetails | undefined;

      if (data) {
        if (typeof data === 'string') {
          message = data;
        } else if (data.message) {
          message = data.message;
        } else if (data.detail) {
          message = data.detail;
        } else if (data.error) {
          message = typeof data.error === 'string' ? data.error : data.error.message;
        }

        code = data.code || data.error_code || `HTTP_${status}`;

        if (data.details || data.field || data.context) {
          details = {
            field: data.field,
            context: data.context,
            code: data.code,
          };
        }
      }

      return new ApiError(message, status, code, details, error.config?.headers?.['X-Request-ID']);
    } else if (error.request) {
      // Network error
      return new ApiError('Network error - please check your connection', 0, 'NETWORK_ERROR');
    } else {
      // Something else
      return new ApiError(error.message || 'Request configuration error', 500, 'REQUEST_ERROR');
    }
  }

  /**
   * Show user notification (override in app-specific implementations)
   */
  private showUserNotification(message: string, type: 'error' | 'warning' | 'info'): void {
    // Default console fallback - override this in your app
    if (type === 'error') {
      console.error('[User Notification]', message);
    } else if (type === 'warning') {
      console.warn('[User Notification]', message);
    } else {
      console.info('[User Notification]', message);
    }

    // Integration with toast libraries would go here
    // Example: toast.error(message) or showNotification(message, type)
  }

  /**
   * Set custom notification handler
   */
  setNotificationHandler(handler: (message: string, type: 'error' | 'warning' | 'info') => void): void {
    this.showUserNotification = handler;
  }

  /**
   * Create standardized error response for catch blocks
   */
  createErrorResponse<T>(
    error: unknown,
    context: string,
    fallbackData?: T,
    options?: ErrorHandlerOptions
  ): { success: false; error: ApiError; data?: T } {
    const apiError = this.handleError(error, context, options);
    return {
      success: false,
      error: apiError,
      ...(fallbackData !== undefined && { data: fallbackData }),
    };
  }
}

// Export singleton instance
export const apiErrorHandler = ApiErrorHandler.getInstance();