import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ChimeraProvider, useChimera } from "../chimera-provider";
import { providerService } from "../../lib/services/provider-service";
import { sessionService } from "../../lib/services/session-service";
import { apiClient } from "../../lib/api/core";
import React from "react";

// Mock the services
vi.mock("../../lib/services/provider-service", () => ({
    providerService: {
        subscribe: vi.fn((cb) => {
            cb([]);
            return () => { };
        }),
        refresh: vi.fn(),
        list: vi.fn(() => []),
    }
}));

vi.mock("../../lib/services/session-service", () => ({
    sessionService: {
        subscribe: vi.fn((cb) => {
            cb(null);
            return () => { };
        }),
        initialize: vi.fn(),
        getSession: vi.fn(() => null),
    }
}));

const TestComponent = () => {
    const { session, providers, loading } = useChimera();
    return (
        <div>
            <div data-testid="loading">{loading ? "loading" : "ready"}</div>
            <div data-testid="session">{session ? "has-session" : "no-session"}</div>
            <div data-testid="providers-count">{providers.length}</div>
        </div>
    );
};

describe("ChimeraProvider", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should initialize services on mount", async () => {
        render(
            <ChimeraProvider>
                <TestComponent />
            </ChimeraProvider>
        );

        expect(sessionService.initialize).toHaveBeenCalled();
        expect(providerService.refresh).toHaveBeenCalled();
    });

    it("should provide session and providers state", async () => {
        const mockSession = { sessionId: "test-session" };
        const mockProviders = [{ id: "test-provider", displayName: "Test" }];

        // Update mocks to trigger subscriptions
        (sessionService.subscribe as any).mockImplementation((cb: any) => {
            cb(mockSession);
            return () => { };
        });
        (providerService.subscribe as any).mockImplementation((cb: any) => {
            cb(mockProviders);
            return () => { };
        });

        render(
            <ChimeraProvider>
                <TestComponent />
            </ChimeraProvider>
        );

        expect(screen.getByTestId("session").textContent).toBe("has-session");
        expect(screen.getByTestId("providers-count").textContent).toBe("1");
    });
});
