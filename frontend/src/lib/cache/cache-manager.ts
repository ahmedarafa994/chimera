/**
 * Cache Manager for Project Chimera Frontend
 * In-memory cache with TTL support for API response caching
 *
 * @module lib/cache/cache-manager
 */

// ============================================================================
// Types
// ============================================================================

export interface CacheEntry<T> {
    value: T;
    expiresAt: number;
    createdAt: number;
    accessCount: number;
    lastAccessedAt: number;
}

export interface CacheConfig {
    /** Default TTL in milliseconds */
    defaultTTL: number;
    /** Maximum number of entries */
    maxEntries: number;
    /** Enable LRU eviction when max entries reached */
    enableLRU: boolean;
}

export interface CacheStats {
    entries: number;
    hits: number;
    misses: number;
    hitRate: number;
    memoryEstimate: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: CacheConfig = {
    defaultTTL: 5 * 60 * 1000, // 5 minutes
    maxEntries: 100,
    enableLRU: true,
};

// ============================================================================
// Cache Manager Implementation
// ============================================================================

export class CacheManager {
    private cache: Map<string, CacheEntry<unknown>> = new Map();
    private config: CacheConfig;
    private hits = 0;
    private misses = 0;

    constructor(config: Partial<CacheConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
    }

    /**
     * Get a value from the cache
     */
    get<T>(key: string): T | null {
        const entry = this.cache.get(key);

        if (!entry) {
            this.misses++;
            return null;
        }

        // Check if expired
        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            this.misses++;
            return null;
        }

        // Update access stats
        entry.accessCount++;
        entry.lastAccessedAt = Date.now();
        this.hits++;

        return entry.value as T;
    }

    /**
     * Set a value in the cache
     */
    set<T>(key: string, value: T, ttl?: number): void {
        // Evict if at capacity
        if (this.cache.size >= this.config.maxEntries) {
            this.evict();
        }

        const now = Date.now();
        const entry: CacheEntry<T> = {
            value,
            expiresAt: now + (ttl ?? this.config.defaultTTL),
            createdAt: now,
            accessCount: 0,
            lastAccessedAt: now,
        };

        this.cache.set(key, entry);
    }

    /**
     * Check if a key exists and is not expired
     */
    has(key: string): boolean {
        const entry = this.cache.get(key);
        if (!entry) return false;
        if (Date.now() > entry.expiresAt) {
            this.cache.delete(key);
            return false;
        }
        return true;
    }

    /**
     * Delete a specific key
     */
    delete(key: string): boolean {
        return this.cache.delete(key);
    }

    /**
     * Invalidate entries matching a pattern
     * Supports wildcards: * matches any sequence, ? matches single char
     */
    invalidate(pattern: string): number {
        const regex = this.patternToRegex(pattern);
        let count = 0;

        for (const key of this.cache.keys()) {
            if (regex.test(key)) {
                this.cache.delete(key);
                count++;
            }
        }

        return count;
    }

    /**
     * Clear all entries
     */
    clear(): void {
        this.cache.clear();
        this.hits = 0;
        this.misses = 0;
    }

    /**
     * Get cache statistics
     */
    getStats(): CacheStats {
        const totalRequests = this.hits + this.misses;
        const hitRate = totalRequests > 0 ? this.hits / totalRequests : 0;

        // Rough memory estimate (not accurate, just for monitoring)
        let memoryEstimate = 0;
        for (const entry of this.cache.values()) {
            memoryEstimate += JSON.stringify(entry.value).length * 2; // UTF-16
        }

        return {
            entries: this.cache.size,
            hits: this.hits,
            misses: this.misses,
            hitRate,
            memoryEstimate,
        };
    }

    /**
     * Get or set pattern - returns cached value or executes function
     */
    async getOrSet<T>(
        key: string,
        fn: () => Promise<T>,
        ttl?: number
    ): Promise<T> {
        const cached = this.get<T>(key);
        if (cached !== null) {
            return cached;
        }

        const value = await fn();
        this.set(key, value, ttl);
        return value;
    }

    /**
     * Prune expired entries
     */
    prune(): number {
        const now = Date.now();
        let count = 0;

        for (const [key, entry] of this.cache.entries()) {
            if (now > entry.expiresAt) {
                this.cache.delete(key);
                count++;
            }
        }

        return count;
    }

    // ============================================================================
    // Private Methods
    // ============================================================================

    /**
     * Evict entries when at capacity
     */
    private evict(): void {
        if (!this.config.enableLRU) {
            // Simple eviction: remove oldest
            const firstKey = this.cache.keys().next().value;
            if (firstKey) {
                this.cache.delete(firstKey);
            }
            return;
        }

        // LRU eviction: remove least recently accessed
        let lruKey: string | null = null;
        let lruTime = Infinity;

        for (const [key, entry] of this.cache.entries()) {
            if (entry.lastAccessedAt < lruTime) {
                lruTime = entry.lastAccessedAt;
                lruKey = key;
            }
        }

        if (lruKey) {
            this.cache.delete(lruKey);
        }
    }

    /**
     * Convert glob pattern to regex
     */
    private patternToRegex(pattern: string): RegExp {
        const escaped = pattern
            .replace(/[.+^${}()|[\]\\]/g, "\\$&")
            .replace(/\*/g, ".*")
            .replace(/\?/g, ".");
        return new RegExp(`^${escaped}$`);
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

export const globalCache = new CacheManager();

// ============================================================================
// Cache Key Utilities
// ============================================================================

/**
 * Generate a deterministic cache key from endpoint and parameters
 */
export function generateCacheKey(
    endpoint: string,
    params?: object
): string {
    const sortedParams = params
        ? Object.keys(params)
            .sort()
            .reduce((acc, key) => {
                acc[key] = (params as Record<string, unknown>)[key];
                return acc;
            }, {} as Record<string, unknown>)
        : {};

    return `${endpoint}:${JSON.stringify(sortedParams)}`;
}

/**
 * Pre-defined cache TTLs for different content types
 */
export const CACHE_TTL = {
    /** Static content that rarely changes */
    STATIC: 60 * 60 * 1000, // 1 hour
    /** Provider and model lists */
    PROVIDERS: 5 * 60 * 1000, // 5 minutes
    /** Technique lists */
    TECHNIQUES: 10 * 60 * 1000, // 10 minutes
    /** Session data */
    SESSION: 1 * 60 * 1000, // 1 minute
    /** AI generation results (short-lived) */
    GENERATION: 30 * 1000, // 30 seconds
    /** Health checks */
    HEALTH: 10 * 1000, // 10 seconds
} as const;

// ============================================================================
// Automatic Cleanup
// ============================================================================

// Run cleanup every 5 minutes
if (typeof window !== "undefined") {
    setInterval(() => {
        globalCache.prune();
    }, 5 * 60 * 1000);
}
