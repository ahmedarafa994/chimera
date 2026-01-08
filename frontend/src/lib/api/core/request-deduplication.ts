/**
 * Request Deduplication
 * 
 * Prevents redundant API calls by tracking in-flight requests
 * and returning the same promise for identical concurrent requests.
 * 
 * @module lib/api/core/request-deduplication
 */

// ============================================================================
// Types
// ============================================================================

export interface DeduplicationConfig {
  /** Enable deduplication */
  enabled: boolean;
  /** Time window for considering requests as duplicates (ms) */
  windowMs: number;
  /** Maximum number of tracked requests */
  maxTracked: number;
  /** Custom key generator */
  keyGenerator?: (method: string, url: string, data?: unknown) => string;
}

interface PendingRequest<T> {
  promise: Promise<T>;
  timestamp: number;
  subscribers: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: DeduplicationConfig = {
  enabled: true,
  windowMs: 100, // 100ms window for deduplication
  maxTracked: 1000,
};

// ============================================================================
// Key Generation
// ============================================================================

/**
 * Generate a unique key for a request
 */
export function generateRequestKey(
  method: string,
  url: string,
  data?: unknown
): string {
  const normalizedMethod = method.toUpperCase();
  const normalizedUrl = url.toLowerCase();
  
  // For GET requests, only use method and URL
  if (normalizedMethod === 'GET' || normalizedMethod === 'HEAD') {
    return `${normalizedMethod}:${normalizedUrl}`;
  }
  
  // For other methods, include a hash of the data
  const dataHash = data ? hashObject(data) : '';
  return `${normalizedMethod}:${normalizedUrl}:${dataHash}`;
}

/**
 * Simple hash function for objects
 */
function hashObject(obj: unknown): string {
  const str = JSON.stringify(obj, Object.keys(obj as object).sort());
  let hash = 0;
  
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  
  return hash.toString(36);
}

// ============================================================================
// Request Deduplicator
// ============================================================================

export class RequestDeduplicator {
  private pending: Map<string, PendingRequest<unknown>> = new Map();
  private config: DeduplicationConfig;
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: Partial<DeduplicationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    // Start cleanup interval
    if (typeof window !== 'undefined') {
      this.cleanupInterval = setInterval(() => this.cleanup(), 5000);
    }
  }

  /**
   * Execute a request with deduplication
   */
  async execute<T>(
    key: string,
    requestFn: () => Promise<T>
  ): Promise<T> {
    if (!this.config.enabled) {
      return requestFn();
    }

    // Check for existing pending request
    const existing = this.pending.get(key);
    if (existing && this.isWithinWindow(existing.timestamp)) {
      existing.subscribers++;
      return existing.promise as Promise<T>;
    }

    // Create new request
    const promise = this.createTrackedRequest(key, requestFn);
    return promise;
  }

  /**
   * Execute with automatic key generation
   */
  async executeWithKey<T>(
    method: string,
    url: string,
    data: unknown,
    requestFn: () => Promise<T>
  ): Promise<T> {
    const key = this.config.keyGenerator
      ? this.config.keyGenerator(method, url, data)
      : generateRequestKey(method, url, data);
    
    return this.execute(key, requestFn);
  }

  /**
   * Check if a request is currently pending
   */
  isPending(key: string): boolean {
    const existing = this.pending.get(key);
    return existing !== undefined && this.isWithinWindow(existing.timestamp);
  }

  /**
   * Get number of pending requests
   */
  getPendingCount(): number {
    return this.pending.size;
  }

  /**
   * Cancel a pending request (removes from tracking)
   */
  cancel(key: string): boolean {
    return this.pending.delete(key);
  }

  /**
   * Clear all pending requests
   */
  clear(): void {
    this.pending.clear();
  }

  /**
   * Destroy the deduplicator
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.pending.clear();
  }

  /**
   * Get statistics
   */
  getStats(): {
    pendingRequests: number;
    totalSubscribers: number;
  } {
    let totalSubscribers = 0;
    this.pending.forEach(req => {
      totalSubscribers += req.subscribers;
    });

    return {
      pendingRequests: this.pending.size,
      totalSubscribers,
    };
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private async createTrackedRequest<T>(
    key: string,
    requestFn: () => Promise<T>
  ): Promise<T> {
    // Enforce max tracked limit
    if (this.pending.size >= this.config.maxTracked) {
      this.cleanup();
      
      // If still at limit, remove oldest
      if (this.pending.size >= this.config.maxTracked) {
        const oldestKey = this.pending.keys().next().value;
        if (oldestKey) {
          this.pending.delete(oldestKey);
        }
      }
    }

    const pendingRequest: PendingRequest<T> = {
      promise: requestFn().finally(() => {
        // Remove from pending after completion
        this.pending.delete(key);
      }),
      timestamp: Date.now(),
      subscribers: 1,
    };

    this.pending.set(key, pendingRequest as PendingRequest<unknown>);
    return pendingRequest.promise;
  }

  private isWithinWindow(timestamp: number): boolean {
    return Date.now() - timestamp < this.config.windowMs;
  }

  private cleanup(): void {
    const now = Date.now();
    const expiredKeys: string[] = [];

    this.pending.forEach((request, key) => {
      if (now - request.timestamp > this.config.windowMs * 10) {
        expiredKeys.push(key);
      }
    });

    expiredKeys.forEach(key => this.pending.delete(key));
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

export const requestDeduplicator = new RequestDeduplicator();

// ============================================================================
// Decorator
// ============================================================================

/**
 * Wrap a function with request deduplication
 */
export function deduplicated<T extends unknown[], R>(
  keyGenerator: (...args: T) => string,
  fn: (...args: T) => Promise<R>
): (...args: T) => Promise<R> {
  return async (...args: T): Promise<R> => {
    const key = keyGenerator(...args);
    return requestDeduplicator.execute(key, () => fn(...args));
  };
}

// ============================================================================
// Request Batching
// ============================================================================

export interface BatchConfig<T, R> {
  /** Maximum batch size */
  maxBatchSize: number;
  /** Maximum wait time before executing batch (ms) */
  maxWaitMs: number;
  /** Batch executor function */
  batchFn: (items: T[]) => Promise<R[]>;
  /** Key generator for deduplication within batch */
  keyFn?: (item: T) => string;
}

export interface BatchResult<T> {
  success: boolean;
  data?: T;
  error?: Error;
}

interface BatchItem<T, R> {
  item: T;
  resolve: (value: R) => void;
  reject: (error: Error) => void;
}

export class RequestBatcher<T, R> {
  private queue: BatchItem<T, R>[] = new Map() as unknown as BatchItem<T, R>[];
  private pendingKeys: Set<string> = new Set();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private config: BatchConfig<T, R>;

  constructor(config: BatchConfig<T, R>) {
    this.config = config;
    this.queue = [];
  }

  /**
   * Add an item to the batch
   */
  async add(item: T): Promise<R> {
    // Check for duplicate if key function provided
    if (this.config.keyFn) {
      const key = this.config.keyFn(item);
      if (this.pendingKeys.has(key)) {
        // Wait for existing batch to complete
        return new Promise((resolve, reject) => {
          this.queue.push({ item, resolve, reject });
        });
      }
      this.pendingKeys.add(key);
    }

    return new Promise((resolve, reject) => {
      this.queue.push({ item, resolve, reject });

      // Check if batch is full
      if (this.queue.length >= this.config.maxBatchSize) {
        this.flush();
      } else if (!this.timer) {
        // Start timer for max wait
        this.timer = setTimeout(() => this.flush(), this.config.maxWaitMs);
      }
    });
  }

  /**
   * Flush the current batch
   */
  async flush(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    if (this.queue.length === 0) return;

    const batch = [...this.queue];
    this.queue = [];
    this.pendingKeys.clear();

    try {
      const items = batch.map(b => b.item);
      const results = await this.config.batchFn(items);

      // Resolve each promise with corresponding result
      batch.forEach((batchItem, index) => {
        if (index < results.length) {
          batchItem.resolve(results[index]);
        } else {
          batchItem.reject(new Error('Batch result missing'));
        }
      });
    } catch (error) {
      // Reject all promises on error
      const err = error instanceof Error ? error : new Error(String(error));
      batch.forEach(batchItem => batchItem.reject(err));
    }
  }

  /**
   * Get current queue size
   */
  getQueueSize(): number {
    return this.queue.length;
  }

  /**
   * Clear the queue without executing
   */
  clear(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    const error = new Error('Batch cleared');
    this.queue.forEach(item => item.reject(error));
    this.queue = [];
    this.pendingKeys.clear();
  }
}

// ============================================================================
// Debounced Request
// ============================================================================

export interface DebouncedRequestConfig {
  /** Debounce delay in ms */
  delayMs: number;
  /** Leading edge execution */
  leading?: boolean;
  /** Trailing edge execution */
  trailing?: boolean;
}

/**
 * Create a debounced version of an async function
 */
export function debouncedRequest<T extends unknown[], R>(
  fn: (...args: T) => Promise<R>,
  config: DebouncedRequestConfig
): {
  (...args: T): Promise<R>;
  cancel: () => void;
  flush: () => Promise<R | undefined>;
} {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pendingPromise: {
    resolve: (value: R) => void;
    reject: (error: Error) => void;
  } | null = null;
  let lastArgs: T | null = null;
  let lastResult: R | undefined;

  const { delayMs, leading = false, trailing = true } = config;

  const debouncedFn = (...args: T): Promise<R> => {
    lastArgs = args;

    return new Promise((resolve, reject) => {
      const callNow = leading && !timer;

      if (timer) {
        clearTimeout(timer);
      }

      pendingPromise = { resolve, reject };

      timer = setTimeout(async () => {
        timer = null;

        if (trailing && lastArgs) {
          try {
            lastResult = await fn(...lastArgs);
            pendingPromise?.resolve(lastResult);
          } catch (error) {
            pendingPromise?.reject(error instanceof Error ? error : new Error(String(error)));
          }
        }

        pendingPromise = null;
        lastArgs = null;
      }, delayMs);

      if (callNow) {
        fn(...args)
          .then(result => {
            lastResult = result;
            resolve(result);
          })
          .catch(reject);
      }
    });
  };

  debouncedFn.cancel = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    pendingPromise?.reject(new Error('Debounced request cancelled'));
    pendingPromise = null;
    lastArgs = null;
  };

  debouncedFn.flush = async (): Promise<R | undefined> => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }

    if (lastArgs) {
      try {
        lastResult = await fn(...lastArgs);
        pendingPromise?.resolve(lastResult);
      } catch (error) {
        pendingPromise?.reject(error instanceof Error ? error : new Error(String(error)));
      }
      pendingPromise = null;
      lastArgs = null;
    }

    return lastResult;
  };

  return debouncedFn;
}

// ============================================================================
// Throttled Request
// ============================================================================

/**
 * Create a throttled version of an async function
 */
export function throttledRequest<T extends unknown[], R>(
  fn: (...args: T) => Promise<R>,
  limitMs: number
): (...args: T) => Promise<R> {
  let lastCall = 0;
  let pendingPromise: Promise<R> | null = null;

  return async (...args: T): Promise<R> => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall;

    if (timeSinceLastCall >= limitMs) {
      lastCall = now;
      pendingPromise = fn(...args);
      return pendingPromise;
    }

    // Return pending promise if exists
    if (pendingPromise) {
      return pendingPromise;
    }

    // Wait for throttle period
    await new Promise(resolve => setTimeout(resolve, limitMs - timeSinceLastCall));
    lastCall = Date.now();
    pendingPromise = fn(...args);
    return pendingPromise;
  };
}