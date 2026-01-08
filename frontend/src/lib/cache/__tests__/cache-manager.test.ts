/**
 * Unit Tests for Cache Manager
 * Tests LRU caching, TTL expiration, and pattern invalidation
 *
 * @module lib/cache/__tests__/cache-manager.test
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { CacheManager, generateCacheKey, CACHE_TTL } from "../cache-manager";

describe("CacheManager", () => {
    let cache: CacheManager;

    beforeEach(() => {
        cache = new CacheManager({
            defaultTTL: 1000, // 1 second for faster tests
            maxEntries: 5,
            enableLRU: true,
        });
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    describe("get/set operations", () => {
        it("should store and retrieve values", () => {
            cache.set("key1", "value1");
            expect(cache.get<string>("key1")).toBe("value1");
        });

        it("should return null for non-existent keys", () => {
            expect(cache.get<string>("nonexistent")).toBeNull();
        });

        it("should handle different value types", () => {
            cache.set("string", "hello");
            cache.set("number", 42);
            cache.set("object", { foo: "bar" });
            cache.set("array", [1, 2, 3]);

            expect(cache.get<string>("string")).toBe("hello");
            expect(cache.get<number>("number")).toBe(42);
            expect(cache.get<{ foo: string }>("object")).toEqual({ foo: "bar" });
            expect(cache.get<number[]>("array")).toEqual([1, 2, 3]);
        });
    });

    describe("TTL expiration", () => {
        it("should expire entries after TTL", () => {
            cache.set("expiring", "value", 500); // 500ms TTL

            expect(cache.get<string>("expiring")).toBe("value");

            vi.advanceTimersByTime(600);

            expect(cache.get<string>("expiring")).toBeNull();
        });

        it("should use default TTL when not specified", () => {
            cache.set("default-ttl", "value");

            expect(cache.get<string>("default-ttl")).toBe("value");

            vi.advanceTimersByTime(1100); // Default is 1000ms

            expect(cache.get<string>("default-ttl")).toBeNull();
        });
    });

    describe("LRU eviction", () => {
        it("should evict least recently used when at capacity", () => {
            // Fill cache to capacity (5 entries), advancing time between each
            cache.set("key1", "value1");
            vi.advanceTimersByTime(1);
            cache.set("key2", "value2");
            vi.advanceTimersByTime(1);
            cache.set("key3", "value3");
            vi.advanceTimersByTime(1);
            cache.set("key4", "value4");
            vi.advanceTimersByTime(1);
            cache.set("key5", "value5");
            vi.advanceTimersByTime(1);

            // Access key1 and key2 to make them recently used
            cache.get("key1");
            vi.advanceTimersByTime(1);
            cache.get("key2");
            vi.advanceTimersByTime(1);

            // Add another entry, should evict key3 (least recently used among non-accessed keys)
            // key1, key2 were accessed, key3 is the oldest among key3, key4, key5 that weren't accessed
            cache.set("key6", "value6");

            expect(cache.get<string>("key1")).toBe("value1");
            expect(cache.get<string>("key2")).toBe("value2");
            expect(cache.get<string>("key3")).toBeNull(); // Evicted - was created at t=2 and never accessed
            expect(cache.get<string>("key6")).toBe("value6");
        });
    });

    describe("has()", () => {
        it("should return true for existing keys", () => {
            cache.set("exists", "value");
            expect(cache.has("exists")).toBe(true);
        });

        it("should return false for non-existent keys", () => {
            expect(cache.has("nonexistent")).toBe(false);
        });

        it("should return false for expired keys", () => {
            cache.set("expiring", "value", 100);
            expect(cache.has("expiring")).toBe(true);

            vi.advanceTimersByTime(150);

            expect(cache.has("expiring")).toBe(false);
        });
    });

    describe("delete()", () => {
        it("should delete existing keys", () => {
            cache.set("toDelete", "value");
            expect(cache.delete("toDelete")).toBe(true);
            expect(cache.get<string>("toDelete")).toBeNull();
        });

        it("should return false for non-existent keys", () => {
            expect(cache.delete("nonexistent")).toBe(false);
        });
    });

    describe("invalidate()", () => {
        it("should invalidate entries matching wildcard pattern", () => {
            cache.set("user:1:name", "Alice");
            cache.set("user:1:email", "alice@example.com");
            cache.set("user:2:name", "Bob");
            cache.set("product:1", "Widget");

            const count = cache.invalidate("user:1:*");

            expect(count).toBe(2);
            expect(cache.get<string>("user:1:name")).toBeNull();
            expect(cache.get<string>("user:1:email")).toBeNull();
            expect(cache.get<string>("user:2:name")).toBe("Bob");
            expect(cache.get<string>("product:1")).toBe("Widget");
        });

        it("should invalidate entries matching prefix pattern", () => {
            cache.set("api:users", "data1");
            cache.set("api:products", "data2");
            cache.set("cache:users", "data3");

            const count = cache.invalidate("api:*");

            expect(count).toBe(2);
            expect(cache.get<string>("cache:users")).toBe("data3");
        });
    });

    describe("clear()", () => {
        it("should remove all entries", () => {
            cache.set("key1", "value1");
            cache.set("key2", "value2");

            cache.clear();

            expect(cache.get<string>("key1")).toBeNull();
            expect(cache.get<string>("key2")).toBeNull();
        });

        it("should reset stats", () => {
            cache.set("key", "value");
            cache.get("key"); // hit
            cache.get("nonexistent"); // miss

            cache.clear();

            const stats = cache.getStats();
            expect(stats.hits).toBe(0);
            expect(stats.misses).toBe(0);
        });
    });

    describe("getStats()", () => {
        it("should track hits and misses", () => {
            cache.set("key", "value");
            cache.get("key"); // hit
            cache.get("key"); // hit
            cache.get("nonexistent"); // miss

            const stats = cache.getStats();

            expect(stats.hits).toBe(2);
            expect(stats.misses).toBe(1);
            expect(stats.hitRate).toBeCloseTo(0.667, 2);
        });
    });

    describe("getOrSet()", () => {
        it("should return cached value if exists", async () => {
            cache.set("key", "cached");
            const fn = vi.fn().mockResolvedValue("fresh");

            const result = await cache.getOrSet("key", fn);

            expect(result).toBe("cached");
            expect(fn).not.toHaveBeenCalled();
        });

        it("should call function and cache result if not exists", async () => {
            const fn = vi.fn().mockResolvedValue("fresh");

            const result = await cache.getOrSet("key", fn);

            expect(result).toBe("fresh");
            expect(fn).toHaveBeenCalledTimes(1);
            expect(cache.get<string>("key")).toBe("fresh");
        });
    });

    describe("prune()", () => {
        it("should remove expired entries", () => {
            cache.set("short", "value", 100);
            cache.set("long", "value", 10000);

            vi.advanceTimersByTime(150);

            const pruned = cache.prune();

            expect(pruned).toBe(1);
            expect(cache.get<string>("short")).toBeNull();
            expect(cache.get<string>("long")).toBe("value");
        });
    });
});

describe("generateCacheKey", () => {
    it("should generate consistent keys for same inputs", () => {
        const key1 = generateCacheKey("endpoint", { a: 1, b: 2 });
        const key2 = generateCacheKey("endpoint", { a: 1, b: 2 });
        expect(key1).toBe(key2);
    });

    it("should sort object keys for consistency", () => {
        const key1 = generateCacheKey("endpoint", { b: 2, a: 1 });
        const key2 = generateCacheKey("endpoint", { a: 1, b: 2 });
        expect(key1).toBe(key2);
    });

    it("should include endpoint in key", () => {
        const key = generateCacheKey("users/list", { page: 1 });
        expect(key).toContain("users/list");
    });
});

describe("CACHE_TTL constants", () => {
    it("should have correct TTL values", () => {
        expect(CACHE_TTL.STATIC).toBe(60 * 60 * 1000);
        expect(CACHE_TTL.PROVIDERS).toBe(5 * 60 * 1000);
        expect(CACHE_TTL.GENERATION).toBe(30 * 1000);
        expect(CACHE_TTL.HEALTH).toBe(10 * 1000);
    });
});
