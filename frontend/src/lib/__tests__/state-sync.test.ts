/**
 * Unit Tests for State Sync Utilities
 * Tests optimistic updates, event bus, and state recovery
 *
 * @module lib/__tests__/state-sync.test
 */

import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import {
    useOptimisticUpdate,
    useStateRecovery,
    eventBus,
    useEventDrivenState,
    stateSyncManager,
    useSyncedState,
    STATE_SYNC_EVENTS,
} from "../state-sync";

describe("useOptimisticUpdate", () => {
    // Don't use fake timers - waitFor depends on real timers

    it("should update value optimistically", async () => {
        const updateFn = vi.fn().mockResolvedValue("confirmed");

        const { result } = renderHook(() =>
            useOptimisticUpdate("initial", updateFn)
        );

        act(() => {
            result.current[1]("optimistic");
        });

        expect(result.current[0].value).toBe("optimistic");
        expect(result.current[0].isPending).toBe(true);
    });

    it("should confirm value on successful update", async () => {
        const updateFn = vi.fn().mockResolvedValue("confirmed");

        const { result } = renderHook(() =>
            useOptimisticUpdate("initial", updateFn)
        );

        act(() => {
            result.current[1]("optimistic");
        });

        await waitFor(() => {
            expect(result.current[0].isPending).toBe(false);
        }, { timeout: 5000 });

        expect(result.current[0].value).toBe("confirmed");
        expect(result.current[0].confirmedValue).toBe("confirmed");
    });

    it("should rollback on failure", async () => {
        const updateFn = vi.fn().mockRejectedValue(new Error("Failed"));

        const { result } = renderHook(() =>
            useOptimisticUpdate("initial", updateFn)
        );

        act(() => {
            result.current[1]("optimistic");
        });

        await waitFor(() => {
            expect(result.current[0].isPending).toBe(false);
        }, { timeout: 5000 });

        expect(result.current[0].value).toBe("initial"); // Rolled back
        expect(result.current[0].rolledBack).toBe(true);
        expect(result.current[0].error).toBeInstanceOf(Error);
    });

    it("should call onRollback callback on failure", async () => {
        const onRollback = vi.fn();
        const updateFn = vi.fn().mockRejectedValue(new Error("Failed"));

        const { result } = renderHook(() =>
            useOptimisticUpdate("initial", updateFn, { onRollback })
        );

        act(() => {
            result.current[1]("optimistic");
        });

        await waitFor(() => {
            expect(onRollback).toHaveBeenCalled();
        }, { timeout: 5000 });

        expect(onRollback).toHaveBeenCalledWith(
            expect.any(Error),
            "initial"
        );
    });

    it("should cancel pending update", async () => {
        // Use a promise we control instead of setTimeout
        let resolvePromise: ((value: string) => void) | undefined;
        const updateFn = vi.fn().mockImplementation(
            () => new Promise<string>((resolve) => {
                resolvePromise = resolve;
            })
        );

        const { result } = renderHook(() =>
            useOptimisticUpdate("initial", updateFn)
        );

        act(() => {
            result.current[1]("optimistic");
        });
        
        // Cancel immediately
        act(() => {
            result.current[2](); // cancel
        });

        expect(result.current[0].value).toBe("initial");
        expect(result.current[0].isPending).toBe(false);
        expect(result.current[0].rolledBack).toBe(true);
    });
});

describe("useStateRecovery", () => {
    const mockStorage: Record<string, string> = {};

    beforeEach(() => {
        // Mock localStorage
        vi.stubGlobal("localStorage", {
            getItem: vi.fn((key: string) => mockStorage[key] || null),
            setItem: vi.fn((key: string, value: string) => {
                mockStorage[key] = value;
            }),
            removeItem: vi.fn((key: string) => {
                delete mockStorage[key];
            }),
        });
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        Object.keys(mockStorage).forEach((key) => delete mockStorage[key]);
    });

    it("should use initial value when no stored value exists", () => {
        const { result } = renderHook(() =>
            useStateRecovery("testKey", "initial")
        );

        expect(result.current[0]).toBe("initial");
    });

    it("should persist value to storage", () => {
        const { result } = renderHook(() =>
            useStateRecovery("testKey", "initial")
        );

        act(() => {
            result.current[1]("updated");
        });

        expect(result.current[0]).toBe("updated");
        expect(localStorage.setItem).toHaveBeenCalled();
    });

    it("should clear recovery state", () => {
        const { result } = renderHook(() =>
            useStateRecovery("testKey", "initial")
        );

        act(() => {
            result.current[1]("value");
            result.current[2](); // clearRecovery
        });

        expect(localStorage.removeItem).toHaveBeenCalledWith("chimera:state:testKey");
    });
});

describe("eventBus", () => {
    beforeEach(() => {
        eventBus.clear();
    });

    it("should emit and receive events", () => {
        const callback = vi.fn();
        eventBus.on("test-event", callback);

        eventBus.emit("test-event", { data: "value" });

        expect(callback).toHaveBeenCalledWith({ data: "value" });
    });

    it("should unsubscribe from events", () => {
        const callback = vi.fn();
        const unsubscribe = eventBus.on("test-event", callback);

        unsubscribe();
        eventBus.emit("test-event", { data: "value" });

        expect(callback).not.toHaveBeenCalled();
    });

    it("should handle once subscription", () => {
        const callback = vi.fn();
        eventBus.once("test-event", callback);

        eventBus.emit("test-event", "first");
        eventBus.emit("test-event", "second");

        expect(callback).toHaveBeenCalledTimes(1);
        expect(callback).toHaveBeenCalledWith("first");
    });

    it("should clear specific event listeners", () => {
        const callback = vi.fn();
        eventBus.on("test-event", callback);

        eventBus.off("test-event");
        eventBus.emit("test-event", "data");

        expect(callback).not.toHaveBeenCalled();
    });
});

describe("useEventDrivenState", () => {
    beforeEach(() => {
        eventBus.clear();
    });

    it("should update state when event is emitted", async () => {
        const { result } = renderHook(() =>
            useEventDrivenState("counter", 0)
        );

        expect(result.current).toBe(0);

        act(() => {
            eventBus.emit("counter", 5);
        });

        expect(result.current).toBe(5);
    });

    it("should apply reducer function to event data", async () => {
        const reducer = (current: number, data: unknown) => current + (data as number);

        const { result } = renderHook(() =>
            useEventDrivenState("counter", 10, reducer)
        );

        act(() => {
            eventBus.emit("counter", 5);
        });

        expect(result.current).toBe(15);
    });
});

describe("stateSyncManager", () => {
    beforeEach(() => {
        stateSyncManager.reset();
    });

    it("should get and set values", () => {
        stateSyncManager.set("key", "value");
        expect(stateSyncManager.get<string>("key")).toBe("value");
    });

    it("should notify subscribers on set", () => {
        const callback = vi.fn();
        stateSyncManager.subscribe("key", callback);

        stateSyncManager.set("key", "value");

        expect(callback).toHaveBeenCalledWith("value");
    });

    it("should unsubscribe from changes", () => {
        const callback = vi.fn();
        const unsubscribe = stateSyncManager.subscribe("key", callback);

        unsubscribe();
        stateSyncManager.set("key", "value");

        expect(callback).not.toHaveBeenCalled();
    });
});

describe("useSyncedState", () => {
    beforeEach(() => {
        stateSyncManager.reset();
    });

    it("should sync state across components", () => {
        const { result: result1 } = renderHook(() =>
            useSyncedState("shared", "initial")
        );
        const { result: result2 } = renderHook(() =>
            useSyncedState("shared", "initial")
        );

        act(() => {
            result1.current[1]("updated");
        });

        expect(result1.current[0]).toBe("updated");
        expect(result2.current[0]).toBe("updated");
    });
});

describe("STATE_SYNC_EVENTS", () => {
    it("should have correct event names", () => {
        expect(STATE_SYNC_EVENTS.PROVIDER_STATUS_CHANGED).toBe("state:provider:status");
        expect(STATE_SYNC_EVENTS.SESSION_UPDATED).toBe("state:session:updated");
        expect(STATE_SYNC_EVENTS.CACHE_INVALIDATED).toBe("state:cache:invalidated");
    });
});
