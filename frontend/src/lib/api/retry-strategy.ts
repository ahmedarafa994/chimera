/**
 * Retry Strategy with Exponential Backoff
 * Handles failed requests with intelligent retry logic
 */

import { AxiosError, AxiosResponse } from 'axios';
import { logger } from './logger';

// ============================================================================
// Configuration
// ============================================================================

const INITIAL_RETRY_DELAY = 1000;
const MAX_RETRY_DELAY = 30000;
const BACKOFF_MULTIPLIER = 2;
const JITTER_FACTOR = 0.1;

const RETRYABLE_STATUS_CODES = new Set([408, 429, 500, 502, 503, 504]);
const RETRYABLE_ERROR_CODES = new Set(['ECONNABORTED', 'ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', 'ENETUNREACH']);

// ============================================================================
// Types
// ============================================================================

export interface RetryConfig {
  maxRetries?: number;
  initialDelay?: number;
  maxDelay?: number;
  backoffMultiplier?: number;
  shouldRetry?: (error: AxiosError) => boolean;
  onRetry?: (error: AxiosError, attempt: number) => void;
}

// ============================================================================
// Helper Functions
// ============================================================================

function shouldRetry(error: AxiosError): boolean {
  if (error.code === 'ERR_CANCELED') {
    return false;
  }

  if (error.code && RETRYABLE_ERROR_CODES.has(error.code)) {
    return true;
  }

  if (error.response?.status && RETRYABLE_STATUS_CODES.has(error.response.status)) {
    return true;
  }

  if (error.response?.status && error.response.status >= 400 && error.response.status < 500) {
    return false;
  }

  return false;
}

function calculateDelay(
  attempt: number,
  initialDelay: number = INITIAL_RETRY_DELAY,
  maxDelay: number = MAX_RETRY_DELAY,
  multiplier: number = BACKOFF_MULTIPLIER
): number {
  const exponentialDelay = initialDelay * Math.pow(multiplier, attempt - 1);
  const cappedDelay = Math.min(exponentialDelay, maxDelay);
  const jitter = cappedDelay * JITTER_FACTOR * (Math.random() - 0.5);
  return Math.floor(cappedDelay + jitter);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// Main Retry Function
// ============================================================================

export async function retryWithBackoff<T>(
  requestFn: () => Promise<AxiosResponse<T>>,
  maxRetries: number = 3,
  config: RetryConfig = {}
): Promise<AxiosResponse<T>> {
  const {
    initialDelay = INITIAL_RETRY_DELAY,
    maxDelay = MAX_RETRY_DELAY,
    backoffMultiplier = BACKOFF_MULTIPLIER,
    shouldRetry: customShouldRetry = shouldRetry,
    onRetry,
  } = config;

  let lastError: AxiosError;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await requestFn();

      if (attempt > 0) {
        logger.logInfo('Request succeeded after retry', {
          attempt,
          totalAttempts: attempt + 1,
        });
      }

      return response;
    } catch (error) {
      lastError = error as AxiosError;

      const willRetry = attempt < maxRetries && customShouldRetry(lastError);

      if (!willRetry) {
        logger.logError('Request failed, no retry', lastError, {
          attempt: attempt + 1,
          totalAttempts: maxRetries + 1,
        });
        throw lastError;
      }

      const delay = calculateDelay(attempt + 1, initialDelay, maxDelay, backoffMultiplier);

      const retryAfter = lastError.response?.headers?.['retry-after'];
      const actualDelay = retryAfter ? parseInt(retryAfter, 10) * 1000 : delay;

      logger.logWarning('Request failed, will retry', {
        attempt: attempt + 1,
        maxRetries: maxRetries + 1,
        delay: actualDelay,
        error: lastError.message,
        status: lastError.response?.status,
      });

      if (onRetry) {
        onRetry(lastError, attempt + 1);
      }

      await sleep(actualDelay);
    }
  }

  throw lastError!;
}

// ============================================================================
// Retry Strategy Builder
// ============================================================================

export class RetryStrategy {
  private config: RetryConfig = {};

  maxRetries(count: number): this {
    this.config.maxRetries = count;
    return this;
  }

  initialDelay(ms: number): this {
    this.config.initialDelay = ms;
    return this;
  }

  maxDelay(ms: number): this {
    this.config.maxDelay = ms;
    return this;
  }

  backoffMultiplier(multiplier: number): this {
    this.config.backoffMultiplier = multiplier;
    return this;
  }

  shouldRetry(fn: (error: AxiosError) => boolean): this {
    this.config.shouldRetry = fn;
    return this;
  }

  onRetry(fn: (error: AxiosError, attempt: number) => void): this {
    this.config.onRetry = fn;
    return this;
  }

  async execute<T>(requestFn: () => Promise<AxiosResponse<T>>): Promise<AxiosResponse<T>> {
    return retryWithBackoff(requestFn, this.config.maxRetries || 3, this.config);
  }
}

export function createRetryStrategy(): RetryStrategy {
  return new RetryStrategy();
}