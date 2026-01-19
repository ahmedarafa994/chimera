/**
 * Unit Tests for Security Utilities
 * Tests secret detection, sanitization, and XSS prevention
 *
 * @module lib/__tests__/security.test
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
    checkForSecrets,
    isEnvSafeForClient,
    sanitizeHeaders,
    sanitizeForLogging,
    escapeHtml,
    generateCSPNonce,
} from "../security";

describe("checkForSecrets", () => {
    it("should detect OpenAI API keys", () => {
        // Use a key that matches the regex /sk-[a-zA-Z0-9]{20,}/g
        // eslint-disable-next-line no-restricted-syntax
        const content = 'const key = "sk-dummyOpenAIKeyForTestingPurposesOnly12345"';
        const result = checkForSecrets(content);

        expect(result.safe).toBe(false);
        expect(result.violations.length).toBeGreaterThan(0);
        expect(result.violations[0].type).toBe("api_key_exposure");
    });

    it("should detect Google API keys", () => {
        // eslint-disable-next-line no-restricted-syntax
        const content = 'const key = "AIzaSyDummyGoogleKeyForTestingPurposesOnly12"';
        const result = checkForSecrets(content);

        expect(result.safe).toBe(false);
        expect(result.violations.length).toBeGreaterThan(0);
    });

    it("should detect private keys", () => {
        // eslint-disable-next-line no-restricted-syntax
        const content = '-----BEGIN RSA PRIVATE KEY-----';
        const result = checkForSecrets(content);

        expect(result.safe).toBe(false);
        expect(result.violations.some((v) => v.pattern?.includes("PRIVATE KEY"))).toBe(true);
    });

    it("should pass for safe content", () => {
        const content = 'const message = "Hello, world!"';
        const result = checkForSecrets(content);

        expect(result.safe).toBe(true);
        expect(result.violations).toHaveLength(0);
    });

    it("should detect GitHub tokens", () => {
        const content = 'const token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"';
        const result = checkForSecrets(content);

        expect(result.safe).toBe(false);
    });

    it("should detect Slack tokens", () => {
        const content = 'const token = "xoxb-123456789012-1234567890123-abcdefghijklmnopqrstuvwx"';
        const result = checkForSecrets(content);

        expect(result.safe).toBe(false);
    });
});

describe("isEnvSafeForClient", () => {
    it("should allow NEXT_PUBLIC_ prefixed vars", () => {
        expect(isEnvSafeForClient("NEXT_PUBLIC_API_URL")).toBe(true);
        expect(isEnvSafeForClient("NEXT_PUBLIC_VERSION")).toBe(true);
    });

    it("should block non-public vars", () => {
        expect(isEnvSafeForClient("API_KEY")).toBe(false);
        expect(isEnvSafeForClient("DATABASE_URL")).toBe(false);
        expect(isEnvSafeForClient("GEMINI_API_KEY")).toBe(false);
    });
});

describe("sanitizeHeaders", () => {
    it("should redact authorization headers", () => {
        const headers = {
            "Authorization": "Bearer secret-token",
            "Content-Type": "application/json",
        };

        const sanitized = sanitizeHeaders(headers);

        expect(sanitized["Authorization"]).toBe("[REDACTED]");
        expect(sanitized["Content-Type"]).toBe("application/json");
    });

    it("should redact api-key headers", () => {
        const headers = {
            "X-Api-Key": "my-secret-key",
            "Accept": "application/json",
        };

        const sanitized = sanitizeHeaders(headers);

        expect(sanitized["X-Api-Key"]).toBe("[REDACTED]");
        expect(sanitized["Accept"]).toBe("application/json");
    });

    it("should redact cookie headers", () => {
        const headers = {
            "Cookie": "session=abc123",
            "Set-Cookie": "token=xyz789",
        };

        const sanitized = sanitizeHeaders(headers);

        expect(sanitized["Cookie"]).toBe("[REDACTED]");
        expect(sanitized["Set-Cookie"]).toBe("[REDACTED]");
    });
});

describe("sanitizeForLogging", () => {
    it("should redact sensitive keys", () => {
        const obj = {
            username: "john",
            password: "secret123",
            api_key: "sk-xxx",
            data: "public info",
        };

        const sanitized = sanitizeForLogging(obj);

        expect(sanitized.username).toBe("john");
        expect(sanitized.password).toBe("[REDACTED]");
        expect(sanitized.api_key).toBe("[REDACTED]");
        expect(sanitized.data).toBe("public info");
    });

    it("should handle nested objects", () => {
        const obj = {
            user: {
                name: "john",
                apiKey: "secret",
            },
        };

        const sanitized = sanitizeForLogging(obj);

        expect((sanitized.user as Record<string, unknown>).name).toBe("john");
        expect((sanitized.user as Record<string, unknown>).apiKey).toBe("[REDACTED]");
    });

    it("should use custom sensitive keys", () => {
        const obj = {
            customSecret: "value",
            normalField: "public",
        };

        const sanitized = sanitizeForLogging(obj, ["customSecret"]);

        expect(sanitized.customSecret).toBe("[REDACTED]");
        expect(sanitized.normalField).toBe("public");
    });
});

describe("escapeHtml", () => {
    it("should escape HTML special characters", () => {
        const unsafe = '<script>alert("XSS")</script>';
        const safe = escapeHtml(unsafe);

        expect(safe).toBe("&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;");
    });

    it("should escape ampersands", () => {
        expect(escapeHtml("Tom & Jerry")).toBe("Tom &amp; Jerry");
    });

    it("should escape quotes", () => {
        expect(escapeHtml('"quoted" and \'apostrophe\'')).toBe(
            "&quot;quoted&quot; and &#039;apostrophe&#039;"
        );
    });

    it("should handle safe strings unchanged", () => {
        expect(escapeHtml("Hello World")).toBe("Hello World");
    });
});

describe("generateCSPNonce", () => {
    it("should generate a base64 string", () => {
        const nonce = generateCSPNonce();

        expect(typeof nonce).toBe("string");
        expect(nonce.length).toBeGreaterThan(0);
    });

    it("should generate unique nonces", () => {
        const nonce1 = generateCSPNonce();
        const nonce2 = generateCSPNonce();

        // Note: There's a tiny chance of collision, but effectively unique
        expect(nonce1).not.toBe(nonce2);
    });
});
