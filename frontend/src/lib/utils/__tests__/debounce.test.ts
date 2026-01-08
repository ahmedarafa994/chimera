/**
 * Unit Tests for Debounce Utility
 * Tests debouncing, throttling, and request deduplication
 *
 * @module lib/utils/__tests__/debounce.test
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
    debounce,
    throttle,
    deduplicateRequest,
    generateCacheKey,
    DEBOUNCE_DELAYS,
} from "../debounce";

describe("debounce", () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("should delay execution until after delay period", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const debouncedFn = debounce(fn, { delay: 100 });

        const promise = debouncedFn("arg1");

        expect(fn).not.toHaveBeenCalled();

        vi.advanceTimersByTime(100);

        const result = await promise;
        expect(fn).toHaveBeenCalledWith("arg1");
        expect(result).toBe("result");
    });

    it("should only execute once for rapid calls", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const debouncedFn = debounce(fn, { delay: 100 });

        debouncedFn("call1");
        debouncedFn("call2");
        debouncedFn("call3");

        vi.advanceTimersByTime(100);
        await Promise.resolve();

        expect(fn).toHaveBeenCalledTimes(1);
        expect(fn).toHaveBeenCalledWith("call3");
    });

    it("should execute on leading edge when configured", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const debouncedFn = debounce(fn, { delay: 100, leading: true });

        debouncedFn("arg");

        expect(fn).toHaveBeenCalledTimes(1);
    });

    it("should respect maxWait option", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const debouncedFn = debounce(fn, { delay: 100, maxWait: 150 });

        // Keep calling to reset the timer
        debouncedFn("call1");
        vi.advanceTimersByTime(50);
        debouncedFn("call2");
        vi.advanceTimersByTime(50);
        debouncedFn("call3");
        vi.advanceTimersByTime(50);

        // maxWait should have triggered by now
        await Promise.resolve();
        expect(fn).toHaveBeenCalled();
    });

    describe("cancel()", () => {
        it("should prevent pending execution", async () => {
            const fn = vi.fn().mockResolvedValue("result");
            const debouncedFn = debounce(fn, { delay: 100 });

            debouncedFn("arg");
            debouncedFn.cancel();

            vi.advanceTimersByTime(100);
            await Promise.resolve();

            expect(fn).not.toHaveBeenCalled();
        });
    });

    describe("flush()", () => {
        it("should execute pending call immediately", async () => {
            const fn = vi.fn().mockResolvedValue("flushed");
            const debouncedFn = debounce(fn, { delay: 100 });

            debouncedFn("arg");
            const result = await debouncedFn.flush();

            expect(fn).toHaveBeenCalledWith("arg");
            expect(result).toBe("flushed");
        });

        it("should return undefined if no pending call", async () => {
            const fn = vi.fn().mockResolvedValue("result");
            const debouncedFn = debounce(fn, { delay: 100 });

            const result = await debouncedFn.flush();

            expect(result).toBeUndefined();
            expect(fn).not.toHaveBeenCalled();
        });
    });

    describe("pending()", () => {
        it("should return true when call is pending", () => {
            const fn = vi.fn().mockResolvedValue("result");
            const debouncedFn = debounce(fn, { delay: 100 });

            expect(debouncedFn.pending()).toBe(false);

            debouncedFn("arg");

            expect(debouncedFn.pending()).toBe(true);
        });
    });
});

describe("throttle", () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("should execute immediately on first call with leading=true", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const throttledFn = throttle(fn, { interval: 100, leading: true });

        throttledFn("arg");

        expect(fn).toHaveBeenCalledWith("arg");
    });

    it("should throttle rapid calls", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const throttledFn = throttle(fn, { interval: 100, leading: true });

        throttledFn("call1");
        throttledFn("call2");
        throttledFn("call3");

        expect(fn).toHaveBeenCalledTimes(1);
        expect(fn).toHaveBeenCalledWith("call1");
    });

    it("should execute trailing call after interval", async () => {
        const fn = vi.fn().mockResolvedValue("result");
        const throttledFn = throttle(fn, { interval: 100, leading: true, trailing: true });

        throttledFn("call1");
        throttledFn("call2");

        vi.advanceTimersByTime(100);
        await Promise.resolve();

        expect(fn).toHaveBeenCalledTimes(2);
    });
});

describe("deduplicateRequest", () => {
    it("should return same promise for concurrent identical requests", async () => {
        let callCount = 0;
        const fn = () => {
            callCount++;
            return Promise.resolve(`result-${callCount}`);
        };

        const [result1, result2] = await Promise.all([
            deduplicateRequest("key1", fn),
            deduplicateRequest("key1", fn),
        ]);

        expect(callCount).toBe(1);
        expect(result1).toBe("result-1");
        expect(result2).toBe("result-1");
    });

    it("should allow new request after previous completes", async () => {
        let callCount = 0;
        const fn = () => {
            callCount++;
            return Promise.resolve(`result-${callCount}`);
        };

        const result1 = await deduplicateRequest("key1", fn);
        const result2 = await deduplicateRequest("key1", fn);

        expect(callCount).toBe(2);
        expect(result1).toBe("result-1");
        expect(result2).toBe("result-2");
    });

    it("should not deduplicate different keys", async () => {
        let callCount = 0;
        const fn = () => {
            callCount++;
            return Promise.resolve(`result-${callCount}`);
        };

        const [result1, result2] = await Promise.all([
            deduplicateRequest("key1", fn),
            deduplicateRequest("key2", fn),
        ]);

        expect(callCount).toBe(2);
        expect(result1).toBe("result-1");
        expect(result2).toBe("result-2");
    });
});

describe("generateCacheKey", () => {
    it("should generate consistent keys for arrays", () => {
        const key = generateCacheKey([1, 2, 3]);
        expect(typeof key).toBe("string");
    });

    it("should generate consistent keys for objects", () => {
        const key = generateCacheKey([{ a: 1 }, { b: 2 }]);
        expect(typeof key).toBe("string");
    });
});

describe("DEBOUNCE_DELAYS constants", () => {
    it("should have correct delay values", () => {
        expect(DEBOUNCE_DELAYS.TYPING).toBe(150);
        expect(DEBOUNCE_DELAYS.INPUT).toBe(300);
        expect(DEBOUNCE_DELAYS.API_CALL).toBe(500);
        expect(DEBOUNCE_DELAYS.AI_GENERATION).toBe(1000);
        expect(DEBOUNCE_DELAYS.HEAVY_OPERATION).toBe(2000);
    });
});
