import { describe, it, expect } from 'vitest';
import {
    APIError,
    ValidationError,
    RateLimitError,
    NetworkError,
    TimeoutError
} from '../errors/api-errors';
import { mapUnknownToError, isRetryableError, getUserMessage } from '../errors/error-mapper';

describe('Error Hierarchy', () => {
    it('should create a ValidationError with correct properties', () => {
        const details = { field: 'email', reason: 'invalid' };
        const error = new ValidationError('Invalid input', details);

        expect(error).toBeInstanceOf(APIError);
        expect(error.message).toBe('Invalid input');
        expect(error.errorCode).toBe('VALIDATION_ERROR');
        expect(error.statusCode).toBe(400);
        expect(error.details).toEqual(details);
    });

    it('should create a RateLimitError with retryAfter', () => {
        const error = new RateLimitError('Too many requests', 30);
        expect(error.statusCode).toBe(429);
        expect(error.retryAfter).toBe(30);
    });
});

describe('Error Mapper', () => {
    it('should map Axios network error to NetworkError', () => {
        const axiosError = {
            isAxiosError: true,
            message: 'Network Error',
            code: 'ERR_NETWORK'
        };

        const mapped = mapUnknownToError(axiosError);
        expect(mapped).toBeInstanceOf(NetworkError);
        expect(mapped.message).toBe('Network Error');
    });

    it('should map Axios timeout to TimeoutError', () => {
        const axiosError = {
            isAxiosError: true,
            code: 'ECONNABORTED',
            message: 'timeout'
        };

        const mapped = mapUnknownToError(axiosError);
        expect(mapped).toBeInstanceOf(TimeoutError);
    });

    it('should map HTTP 429 to RateLimitError', () => {
        const axiosError = {
            isAxiosError: true,
            response: {
                status: 429,
                data: { message: 'Slow down', retry_after: 45 },
                headers: {}
            }
        };

        const mapped = mapUnknownToError(axiosError);
        expect(mapped).toBeInstanceOf(RateLimitError);
        expect((mapped as RateLimitError).retryAfter).toBe(45);
    });
});

describe('Error Utilities', () => {
    it('should identify retryable errors', () => {
        expect(isRetryableError(new NetworkError())).toBe(true);
        expect(isRetryableError(new RateLimitError())).toBe(true);
        expect(isRetryableError(new ValidationError())).toBe(false);
    });

    it('should generate user-friendly messages', () => {
        expect(getUserMessage(new NetworkError())).toContain('internet connection');
        expect(getUserMessage(new TimeoutError())).toContain('timed out');
    });
});
