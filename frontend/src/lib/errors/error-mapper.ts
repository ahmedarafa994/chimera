import type { AxiosError } from 'axios';
import {
    APIError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    NetworkError,
    LLMProviderError,
    CircuitBreakerOpenError,
    TransformationError,
    InternalError,
    ServiceUnavailableError,
    GatewayTimeoutError,
    TimeoutError
} from './api-errors';

/**
 * Maps any unknown error to a typed APIError
 */
export function mapUnknownToError(error: unknown): APIError {
    if (error instanceof APIError) {
        return error;
    }

    // Handle Axios-like errors
    if (isAxiosErrorCheck(error)) {
        if (error.code === 'ECONNABORTED') {
            return new TimeoutError(`Request timed out`);
        }
        if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
            return new NetworkError(error.message);
        }

        if (error.response) {
            const status = error.response.status;
            const data = error.response.data as Record<string, any> | undefined;
            const message = (data?.message || data?.detail || error.message) as string;
            const requestId = (error.response.headers as any)?.['x-request-id'] as string | undefined;

            switch (status) {
                case 400:
                case 422:
                    return new ValidationError(message, data, requestId);
                case 401:
                    return new AuthenticationError(message, data, requestId);
                case 403:
                    return new AuthorizationError(message, data, requestId);
                case 404:
                    return new NotFoundError(message, data, requestId);
                case 429:
                    return new RateLimitError(message, typeof data?.retry_after === 'number' ? data.retry_after : undefined, requestId);
                case 503:
                    return new ServiceUnavailableError(message, data, requestId);
                default:
                    return new InternalError(message, data, requestId);
            }
        }

        return new NetworkError(error.message);
    }

    if (error instanceof Error) {
        return new InternalError(error.message);
    }

    return new InternalError('An unexpected error occurred');
}

/**
 * Maps backend errors to typed errors.
 * (Alias for mapUnknownToError for backward compatibility if needed)
 */
export function mapBackendError(error: unknown): APIError {
    return mapUnknownToError(error);
}

/**
 * Helper to identify axios errors safely
 */
function isAxiosErrorCheck(error: unknown): error is AxiosError {
    return typeof error === 'object' && error !== null && 'isAxiosError' in error;
}

/**
 * Check if error is an APIError instance
 */
export function isAPIError(error: unknown): error is APIError {
    return error instanceof APIError;
}

/**
 * Check if error is a specific error type by code
 */
export function isErrorType(error: unknown, errorCode: string): boolean {
    if (error instanceof APIError) {
        return error.errorCode === errorCode;
    }
    return false;
}

/**
 * Check if error is retryable
 */
export function isRetryableError(error: unknown): boolean {
    if (error instanceof NetworkError || error instanceof GatewayTimeoutError || error instanceof TimeoutError) {
        return true;
    }
    if (error instanceof APIError) {
        // 5xx errors and 429 are generally retryable
        return error.statusCode >= 500 || error.statusCode === 429;
    }
    return false;
}

/**
 * Get retry delay based on error type
 */
export function getRetryDelay(error: unknown, attempt: number = 1): number {
    const baseDelay = 1000; // 1 second base
    const maxDelay = 30000; // 30 seconds max

    if (error instanceof RateLimitError && error.retryAfter !== undefined) {
        return Math.min(error.retryAfter * 1000, maxDelay);
    }

    // Exponential backoff with jitter
    const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), maxDelay);
    const jitter = Math.random() * 0.1 * delay;
    return delay + jitter;
}

/**
 * Generates a user-friendly message for display
 */
export function getUserMessage(error: APIError): string {
    if (error instanceof NetworkError) {
        return 'Unable to connect to the server. Please check your internet connection.';
    }

    if (error instanceof TimeoutError || error instanceof GatewayTimeoutError) {
        return 'The request timed out. Please try again.';
    }

    if (error instanceof ValidationError) {
        return error.message || 'Please check your input and try again.';
    }

    if (error instanceof RateLimitError) {
        return `Too many requests. Please try again in ${error.retryAfter || 60} seconds.`;
    }

    if (error instanceof InternalError && !error.isOperational) {
        return 'Something went wrong. Please try again later.';
    }

    return error.message;
}
