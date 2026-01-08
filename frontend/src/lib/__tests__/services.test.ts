import { describe, it, expect, vi, beforeEach } from "vitest";
import { providerService } from "../services/provider-service";
import { sessionService } from "../services/session-service";
import { apiClient } from "../api/core";

// Mock the apiClient
vi.mock("../api/core", async () => {
    const actual = await vi.importActual("../api/core") as any;
    return {
        ...actual,
        apiClient: {
            get: vi.fn(),
            post: vi.fn(),
            put: vi.fn(),
            delete: vi.fn(),
        },
    };
});

describe("Services Integration", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        // Reset services if they have state (providerService and sessionService are singletons)
        // For testing we might need to reset their internal state manually if they don't have a reset method
        // providerService does not have one, so we'll just be careful with the tests
    });

    describe("ProviderService", () => {
        it("should refresh and list providers", async () => {
            const mockProviders = [
                {
                    provider: "google",
                    display_name: "Google AI",
                    is_healthy: true,
                    models: ["gemini-2.0-flash-exp"]
                }
            ];

            (apiClient.get as any).mockResolvedValue(mockProviders);

            const providers = await providerService.refresh();

            expect(apiClient.get).toHaveBeenCalled();
            expect(providers).toHaveLength(1);
            expect(providers[0].name).toBe("google");
            expect(providers[0].displayName).toBe("Google AI");
        });

        it("should setActive and call backend", async () => {
            (apiClient.post as any).mockResolvedValue({ success: true });

            // First register a provider so we can set it active
            providerService.registerProvider({
                name: "google" as any,
                displayName: "Google",
                enabled: true,
                capabilities: { chat: true, completion: true, streaming: true }
            });

            const success = await providerService.setActive("google", "gemini-pro");

            expect(success).toBe(true);
            expect(apiClient.post).toHaveBeenCalled();
        });
    });

    describe("SessionService", () => {
        it("should initialize by creating a new session if none exists", async () => {
            (apiClient.get as any).mockResolvedValue(null); // No existing session
            (apiClient.post as any).mockResolvedValue({
                session_id: "new-session-id",
                provider: "google",
                model: "gemini-pro"
            });

            const session = await sessionService.initialize();

            expect(session?.sessionId).toBe("new-session-id");
            expect(apiClient.post).toHaveBeenCalled();
        });

        it("should refresh session data", async () => {
            (apiClient.get as any).mockResolvedValue({
                session_id: "existing-id",
                provider: "openai",
                model: "gpt-4"
            });

            const session = await sessionService.refresh();

            expect(session?.provider).toBe("openai");
            expect(apiClient.get).toHaveBeenCalled();
        });
    });
});
