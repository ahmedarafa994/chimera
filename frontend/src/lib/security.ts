/**
 * Security Utilities for Project Chimera Frontend
 * Ensures API keys are never exposed in client-side code
 *
 * @module lib/security
 */

// ============================================================================
// Types
// ============================================================================

export interface SecurityConfig {
    /** Patterns that should never appear in client code */
    forbiddenPatterns: RegExp[];
    /** Environment variable prefixes that are safe for client */
    safeEnvPrefixes: string[];
}

export interface SecurityCheckResult {
    safe: boolean;
    violations: SecurityViolation[];
}

export interface SecurityViolation {
    type: "api_key_exposure" | "forbidden_pattern" | "unsafe_env_access";
    message: string;
    pattern?: string;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: SecurityConfig = {
    forbiddenPatterns: [
        // API key patterns
        /sk-[a-zA-Z0-9]{20,}/g,           // OpenAI keys
        /AIza[a-zA-Z0-9_-]{35}/g,         // Google API keys
        /sk-ant-[a-zA-Z0-9_-]{40,}/g,     // Anthropic keys
        /ghp_[a-zA-Z0-9]{36}/g,           // GitHub tokens
        /gho_[a-zA-Z0-9]{36}/g,           // GitHub OAuth
        /github_pat_[a-zA-Z0-9_]{22,}/g,  // GitHub PATs
        /xox[baprs]-[0-9a-zA-Z-]{10,}/g,  // Slack tokens
        /-----BEGIN.*PRIVATE KEY-----/g,   // Private keys
    ],
    safeEnvPrefixes: [
        "NEXT_PUBLIC_",
    ],
};

// ============================================================================
// Security Checks
// ============================================================================

/**
 * Check if a string contains any forbidden patterns
 */
export function checkForSecrets(
    content: string,
    config: SecurityConfig = DEFAULT_CONFIG
): SecurityCheckResult {
    const violations: SecurityViolation[] = [];

    for (const pattern of config.forbiddenPatterns) {
        // Reset regex state
        pattern.lastIndex = 0;

        if (pattern.test(content)) {
            violations.push({
                type: "api_key_exposure",
                message: `Potential API key or secret detected`,
                pattern: pattern.source,
            });
        }
    }

    return {
        safe: violations.length === 0,
        violations,
    };
}

/**
 * Check if an environment variable is safe to use client-side
 */
export function isEnvSafeForClient(
    envName: string,
    config: SecurityConfig = DEFAULT_CONFIG
): boolean {
    return config.safeEnvPrefixes.some((prefix) =>
        envName.startsWith(prefix)
    );
}

/**
 * Get all client-safe environment variables
 */
export function getClientSafeEnvVars(): Record<string, string> {
    if (typeof process === "undefined") return {};

    const safeVars: Record<string, string> = {};

    for (const [key, value] of Object.entries(process.env)) {
        if (isEnvSafeForClient(key) && value) {
            safeVars[key] = value;
        }
    }

    return safeVars;
}

// ============================================================================
// Runtime Security Guards
// ============================================================================

/**
 * Sanitize request headers to remove sensitive data before logging
 */
export function sanitizeHeaders(
    headers: Record<string, string>
): Record<string, string> {
    const sensitiveHeaders = [
        "authorization",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "cookie",
        "set-cookie",
    ];

    const sanitized: Record<string, string> = {};

    for (const [key, value] of Object.entries(headers)) {
        if (sensitiveHeaders.includes(key.toLowerCase())) {
            sanitized[key] = "[REDACTED]";
        } else {
            sanitized[key] = value;
        }
    }

    return sanitized;
}

/**
 * Sanitize an object for logging (removes potential secrets)
 */
export function sanitizeForLogging(
    obj: Record<string, unknown>,
    sensitiveKeys: string[] = ["api_key", "apiKey", "password", "token", "secret"]
): Record<string, unknown> {
    const sanitized: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(obj)) {
        const keyLower = key.toLowerCase();
        const isSensitive = sensitiveKeys.some(
            (sk) => keyLower.includes(sk.toLowerCase())
        );

        if (isSensitive) {
            sanitized[key] = "[REDACTED]";
        } else if (typeof value === "object" && value !== null) {
            sanitized[key] = sanitizeForLogging(
                value as Record<string, unknown>,
                sensitiveKeys
            );
        } else {
            sanitized[key] = value;
        }
    }

    return sanitized;
}

// ============================================================================
// API Key Validation
// ============================================================================

/**
 * Validate that API key is not being used client-side inappropriately
 * This is a development-time check
 */
export function validateApiKeyUsage(): void {
    if (typeof window === "undefined") return; // Server-side is OK

    if (process.env.NODE_ENV !== "production") {
        // Check for common mistakes in development
        const dangerousEnvVars = [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
        ];

        for (const envVar of dangerousEnvVars) {
            if (process.env[envVar]) {
                console.warn(
                    `⚠️ Security Warning: ${envVar} is accessible in client-side code. ` +
                    `This key should only be used on the server. ` +
                    `Use NEXT_PUBLIC_ prefix only for values safe to expose.`
                );
            }
        }
    }
}

// ============================================================================
// Content Security
// ============================================================================

/**
 * Generate a nonce for Content Security Policy
 */
export function generateCSPNonce(): string {
    const array = new Uint8Array(16);
    if (typeof crypto !== "undefined" && crypto.getRandomValues) {
        crypto.getRandomValues(array);
    } else {
        // Fallback for environments without crypto
        for (let i = 0; i < array.length; i++) {
            array[i] = Math.floor(Math.random() * 256);
        }
    }
    return btoa(String.fromCharCode(...array));
}

/**
 * Escape user input to prevent XSS
 */
export function escapeHtml(unsafe: string): string {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ============================================================================
// Security Initialization
// ============================================================================

/**
 * Initialize security checks on app startup
 * Call this in your app's entry point
 */
export function initializeSecurity(): void {
    if (process.env.NODE_ENV === "development") {
        validateApiKeyUsage();
    }
}

// Run security checks on module load in development
if (process.env.NODE_ENV === "development" && typeof window !== "undefined") {
    initializeSecurity();
}
