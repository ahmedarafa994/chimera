/**
 * Rate limiter configuration
 */
export interface RateLimiterConfig {
  maxRequests: number;
  windowMs: number;
}

/**
 * Client-side rate limiter with sliding window
 */
export class RateLimiter {
  private requests: number[] = [];
  private maxRequests: number;
  private windowMs: number;

  constructor(config: RateLimiterConfig) {
    this.maxRequests = config.maxRequests;
    this.windowMs = config.windowMs;
  }

  /**
   * Check if a request can be made
   */
  canMakeRequest(): boolean {
    this.cleanup();
    return this.requests.length < this.maxRequests;
  }

  /**
   * Record a request
   */
  recordRequest(): void {
    this.cleanup();
    this.requests.push(Date.now());
  }

  /**
   * Attempt to make a request
   * Returns true if allowed, false if rate limited
   */
  attempt(): boolean {
    if (this.canMakeRequest()) {
      this.recordRequest();
      return true;
    }
    return false;
  }

  /**
   * Get time until next request is allowed (in ms)
   * Returns 0 if request can be made immediately
   */
  getTimeUntilNextRequest(): number {
    this.cleanup();

    if (this.requests.length < this.maxRequests) {
      return 0;
    }

    const oldestRequest = this.requests[0];
    const timeUntilExpiry = (oldestRequest + this.windowMs) - Date.now();
    return Math.max(0, timeUntilExpiry);
  }

  /**
   * Get current request count in window
   */
  getCurrentCount(): number {
    this.cleanup();
    return this.requests.length;
  }

  /**
   * Get remaining requests in current window
   */
  getRemainingRequests(): number {
    this.cleanup();
    return Math.max(0, this.maxRequests - this.requests.length);
  }

  /**
   * Reset the rate limiter
   */
  reset(): void {
    this.requests = [];
  }

  /**
   * Remove expired requests from the window
   */
  private cleanup(): void {
    const now = Date.now();
    this.requests = this.requests.filter(timestamp => now - timestamp < this.windowMs);
  }
}

/**
 * Pre-configured rate limiters for common use cases
 */
export const RateLimiters = {
  /**
   * Standard API rate limiter: 100 requests per minute
   */
  api: new RateLimiter({
    maxRequests: 100,
    windowMs: 60000,
  }),

  /**
   * LLM generation rate limiter: 10 requests per minute
   */
  generation: new RateLimiter({
    maxRequests: 10,
    windowMs: 60000,
  }),

  /**
   * Search/transform rate limiter: 30 requests per minute
   */
  search: new RateLimiter({
    maxRequests: 30,
    windowMs: 60000,
  }),

  /**
   * Health check rate limiter: 1 request per 10 seconds
   */
  health: new RateLimiter({
    maxRequests: 1,
    windowMs: 10000,
  }),
};
