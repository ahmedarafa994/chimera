import { describe, it, expect, vi } from 'vitest';
import { CircuitBreaker, CircuitState } from '../resilience/circuit-breaker';
import { withRetry } from '../resilience/retry';

describe('Circuit Breaker', () => {
    it('should transition from CLOSED to OPEN after failures', async () => {
        const breaker = new CircuitBreaker({
            name: 'test-circuit',
            failureThreshold: 2,
            resetTimeout: 1000,
            successThreshold: 1,
            failureWindow: 5000
        });

        expect(breaker.getState()).toBe(CircuitState.CLOSED);

        const failFn = async () => { throw new Error('fail'); };

        await expect(breaker.execute(failFn)).rejects.toThrow('fail');
        await expect(breaker.execute(failFn)).rejects.toThrow('fail');

        expect(breaker.getState()).toBe(CircuitState.OPEN);

        // Should throw CircuitBreakerOpenError now
        await expect(breaker.execute(failFn)).rejects.toThrow(/circuit breaker/i);
    });
});

describe('Retry Logic', () => {
    it('should retry a specified number of times', async () => {
        const fn = vi.fn().mockRejectedValue(new Error('network error'));

        await expect(withRetry(fn, {
            maxRetries: 2,
            baseDelay: 10,
            useJitter: false
        })).rejects.toThrow('network error');

        expect(fn).toHaveBeenCalledTimes(3); // Initial + 2 retries
    });

    it('should return value on success after retry', async () => {
        const fn = vi.fn()
            .mockRejectedValueOnce(new Error('network fail'))
            .mockResolvedValueOnce('success');

        const result = await withRetry(fn, {
            maxRetries: 2,
            baseDelay: 10,
            useJitter: false
        });

        expect(result).toBe('success');
        expect(fn).toHaveBeenCalledTimes(2);
    });
});
