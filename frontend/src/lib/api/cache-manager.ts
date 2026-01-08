/**
 * API Cache Manager
 * Implements intelligent caching for API responses
 */

import { logger } from './logger';

// ============================================================================
// Types
// ============================================================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
  hits: number;
}

interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  evictions: number;
}

// ============================================================================
// Cache Manager
// ============================================================================

class ApiCacheManager {
  private cache: Map<string, CacheEntry<any>>;
  private stats: CacheStats;
  private maxSize: number;
  private cleanupInterval: number;
  private cleanupTimer?: ReturnType<typeof setInterval>;

  constructor(maxSize: number = 100, cleanupInterval: number = 60000) {
    this.cache = new Map();
    this.stats = {
      hits: 0,
      misses: 0,
      size: 0,
      evictions: 0,
    };
    this.maxSize = maxSize;
    this.cleanupInterval = cleanupInterval;
    this.startCleanup();
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) {
      this.stats.misses++;
      return null;
    }

    const now = Date.now();
    if (now - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      this.stats.misses++;
      this.stats.size--;
      return null;
    }

    entry.hits++;
    this.stats.hits++;
    logger.logDebug('Cache hit', { key, hits: entry.hits });
    return entry.data as T;
  }

  set<T>(key: string, data: T, ttl: number = 300000): void {
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictLeastUsed();
    }

    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      ttl,
      hits: 0,
    };

    const isUpdate = this.cache.has(key);
    this.cache.set(key, entry);

    if (!isUpdate) {
      this.stats.size++;
    }

    logger.logDebug('Cache set', { key, ttl, isUpdate });
  }

  delete(key: string): boolean {
    const deleted = this.cache.delete(key);
    if (deleted) {
      this.stats.size--;
      logger.logDebug('Cache deleted', { key });
    }
    return deleted;
  }

  clear(): void {
    const previousSize = this.cache.size;
    this.cache.clear();
    this.stats.size = 0;
    logger.logInfo('Cache cleared', { previousSize });
  }

  invalidatePattern(pattern: string | RegExp): number {
    let count = 0;
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;

    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
        count++;
      }
    }

    this.stats.size -= count;
    logger.logInfo('Cache invalidated by pattern', { pattern: pattern.toString(), count });
    return count;
  }

  private evictLeastUsed(): void {
    let leastUsedKey: string | null = null;
    let leastHits = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.hits < leastHits) {
        leastHits = entry.hits;
        leastUsedKey = key;
      }
    }

    if (leastUsedKey) {
      this.cache.delete(leastUsedKey);
      this.stats.size--;
      this.stats.evictions++;
      logger.logDebug('Cache eviction', { key: leastUsedKey, hits: leastHits });
    }
  }

  private cleanup(): void {
    const now = Date.now();
    let cleaned = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.cache.delete(key);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      this.stats.size -= cleaned;
      logger.logInfo('Cache cleanup', { cleaned, remaining: this.cache.size });
    }
  }

  private startCleanup(): void {
    if (typeof window !== 'undefined') {
      this.cleanupTimer = setInterval(() => {
        this.cleanup();
      }, this.cleanupInterval);
    }
  }

  stopCleanup(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = undefined;
    }
  }

  getStats(): CacheStats {
    return { ...this.stats };
  }

  getHitRate(): number {
    const total = this.stats.hits + this.stats.misses;
    return total > 0 ? this.stats.hits / total : 0;
  }

  async prefetch<T>(key: string, fetchFn: () => Promise<T>, ttl?: number): Promise<void> {
    try {
      const data = await fetchFn();
      this.set(key, data, ttl);
      logger.logInfo('Prefetch successful', { key });
    } catch (error) {
      logger.logError('Prefetch failed', error as Error, { key });
    }
  }

  async getOrFetch<T>(key: string, fetchFn: () => Promise<T>, ttl?: number): Promise<T> {
    const cached = this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    const data = await fetchFn();
    this.set(key, data, ttl);
    return data;
  }
}

export const apiCache = new ApiCacheManager();