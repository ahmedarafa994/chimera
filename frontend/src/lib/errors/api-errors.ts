/**
 * API Error Hierarchy for Project Chimera Frontend
 * Mirrors backend's exception system from app/core/exceptions.py
 *
 * Usage:
 *   import { ValidationError, RateLimitError } from '@/lib/errors';
 *   throw new ValidationError("Invalid input", { field: "prompt" });
 */

// ============================================================================
// Base Error Class
// ============================================================================

/**
 * Base exception class for all API exceptions.
 * Mirrors backend's APIException class.
 */
export abstract class APIError extends Error {
    abstract get statusCode(): number;
    abstract get errorCode(): string;
    readonly details?: Record<string, unknown>;
    readonly timestamp: string;
    readonly requestId?: string;
    readonly isOperational: boolean;

    constructor(
        message: string,
        details?: Record<string, unknown>,
        requestId?: string,
        isOperational: boolean = true
    ) {
        super(message);
        this.name = this.constructor.name;
        this.details = details;
        this.timestamp = new Date().toISOString();
        this.requestId = requestId;
        this.isOperational = isOperational;

        // Maintains proper stack trace in V8 environments
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, this.constructor);
        }
    }

    /**
     * Convert exception to dictionary for logging/serialization.
     */
    toJSON(): Record<string, unknown> {
        return {
            error: this.errorCode,
            message: this.message,
            status_code: this.statusCode,
            details: this.details,
            timestamp: this.timestamp,
            request_id: this.requestId,
        };
    }
}

// ============================================================================
// Client Errors (4xx)
// ============================================================================

/** Raised when request validation fails. */
export class ValidationError extends APIError {
    get statusCode() { return 400; }
    get errorCode() { return "VALIDATION_ERROR"; }

    constructor(message = "Request validation failed", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when authentication fails. */
export class AuthenticationError extends APIError {
    get statusCode() { return 401; }
    get errorCode() { return "AUTHENTICATION_ERROR"; }

    constructor(message = "Authentication failed", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when authentication token has expired. */
export class TokenExpiredError extends AuthenticationError {
    get errorCode() { return "TOKEN_EXPIRED"; }

    constructor(message = "Authentication token has expired", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when session has expired. */
export class SessionExpiredError extends APIError {
    get statusCode() { return 401; }
    get errorCode() { return "SESSION_EXPIRED"; }

    constructor(message = "Session has expired", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when authorization fails. */
export class AuthorizationError extends APIError {
    get statusCode() { return 403; }
    get errorCode() { return "AUTHORIZATION_ERROR"; }

    constructor(message = "You don't have permission to access this resource", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when a requested resource is not found. */
export class NotFoundError extends APIError {
    get statusCode() { return 404; }
    get errorCode() { return "NOT_FOUND"; }

    constructor(message = "Resource not found", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when there's a conflict with the current state. */
export class ConflictError extends APIError {
    get statusCode() { return 409; }
    get errorCode() { return "CONFLICT_ERROR"; }

    constructor(message = "The request conflicts with the current state", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when rate limit is exceeded. */
export class RateLimitError extends APIError {
    get statusCode() { return 429; }
    get errorCode() { return "RATE_LIMIT_EXCEEDED"; }
    readonly retryAfter?: number;

    constructor(message = "Rate limit exceeded. Please try again later", retryAfter?: number, requestId?: string) {
        super(message, retryAfter ? { retry_after: retryAfter } : undefined, requestId);
        this.retryAfter = retryAfter;
    }
}

/** Raised when rate limit budget is exceeded. */
export class RateLimitBudgetExceeded extends RateLimitError {
    get statusCode() { return 429; }
    get errorCode() { return "RATE_LIMIT_BUDGET_EXCEEDED"; }
    readonly budget: string;
    readonly current: number;
    readonly limit: number;

    constructor(budget: string, current: number, limit: number, retryAfter?: number) {
        super(`Rate limit budget "${budget}" exceeded: ${current}/${limit}`, retryAfter);
        this.budget = budget;
        this.current = current;
        this.limit = limit;
    }
}

/** Raised when provider quota is exceeded. */
export class ProviderQuotaExceeded extends RateLimitError {
    get statusCode() { return 429; }
    get errorCode() { return "PROVIDER_QUOTA_EXCEEDED"; }
    readonly provider: string;
    readonly quotaType: string;

    constructor(provider: string, quotaType: string = "requests", retryAfter?: number) {
        super(`Provider "${provider}" quota exceeded for ${quotaType}`, retryAfter);
        this.provider = provider;
        this.quotaType = quotaType;
    }
}

// ============================================================================
// Server Errors (5xx)
// ============================================================================

/** Generic internal server error. */
export class InternalError extends APIError {
    get statusCode() { return 500; }
    get errorCode() { return "INTERNAL_ERROR"; }

    constructor(message = "An internal error occurred", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when a service is temporarily unavailable. */
export class ServiceUnavailableError extends APIError {
    get statusCode() { return 503; }
    get errorCode() { return "SERVICE_UNAVAILABLE"; }

    constructor(message = "The service is temporarily unavailable", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when a request times out. */
export class GatewayTimeoutError extends APIError {
    get statusCode() { return 504; }
    get errorCode() { return "GATEWAY_TIMEOUT"; }

    constructor(message = "Request timed out", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

// ============================================================================
// LLM Provider Errors
// ============================================================================

/** Base exception for LLM provider errors. */
export class LLMProviderError extends APIError {
    get statusCode() { return 500; }
    get errorCode() { return "LLM_PROVIDER_ERROR"; }
    readonly provider?: string;

    constructor(message = "LLM provider error occurred", provider?: string, details?: Record<string, unknown>, requestId?: string) {
        super(message, { ...details, provider }, requestId);
        this.provider = provider;
    }
}

/** Raised when connection to LLM provider fails. */
export class LLMConnectionError extends LLMProviderError {
    get errorCode() { return "LLM_CONNECTION_ERROR"; }

    constructor(message = "Failed to connect to LLM provider", provider?: string, requestId?: string) {
        super(message, provider, undefined, requestId);
    }
}

/** Raised when LLM request times out. */
export class LLMTimeoutError extends LLMProviderError {
    get statusCode() { return 408; }
    get errorCode() { return "LLM_TIMEOUT_ERROR"; }

    constructor(message = "LLM request timed out", provider?: string, requestId?: string) {
        super(message, provider, undefined, requestId);
    }
}

/** Raised when LLM quota is exceeded. */
export class LLMQuotaExceededError extends LLMProviderError {
    get statusCode() { return 429; }
    get errorCode() { return "LLM_QUOTA_EXCEEDED"; }
    readonly retryAfter?: number;

    constructor(message = "LLM API quota exceeded", retryAfter?: number, provider?: string, requestId?: string) {
        super(message, provider, retryAfter ? { retry_after: retryAfter } : undefined, requestId);
        this.retryAfter = retryAfter;
    }
}

/** Raised when model is not available on provider. */
export class ModelNotAvailableError extends LLMProviderError {
    get statusCode() { return 503; }
    get errorCode() { return "MODEL_NOT_AVAILABLE"; }
    readonly model: string;

    constructor(provider: string, model: string, reason?: string) {
        super(
            `Model "${model}" on provider "${provider}" is not available${reason ? `: ${reason}` : ""}`,
            provider,
            { model, reason }
        );
        this.model = model;
    }
}

/** Raised when LLM returns invalid response. */
export class LLMInvalidResponseError extends LLMProviderError {
    get errorCode() { return "LLM_INVALID_RESPONSE"; }

    constructor(message = "Invalid response from LLM provider", provider?: string, requestId?: string) {
        super(message, provider, undefined, requestId);
    }
}

/** Raised when content is blocked by safety filters. */
export class LLMContentBlockedError extends LLMProviderError {
    get errorCode() { return "LLM_CONTENT_BLOCKED"; }
    readonly blockReason?: string;

    constructor(message = "Content blocked by safety filters", blockReason?: string, provider?: string, requestId?: string) {
        super(message, provider, blockReason ? { block_reason: blockReason } : undefined, requestId);
        this.blockReason = blockReason;
    }
}

// ============================================================================
// Transformation Errors
// ============================================================================

/** Raised when prompt transformation fails. */
export class TransformationError extends APIError {
    get statusCode() { return 500; }
    get errorCode() { return "TRANSFORMATION_ERROR"; }

    constructor(message = "Prompt transformation failed", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when potency level is invalid. */
export class InvalidPotencyError extends TransformationError {
    get errorCode() { return "INVALID_POTENCY"; }

    constructor(message = "Invalid potency level. Must be between 1 and 10") {
        super(message);
    }
}

/** Raised when technique suite is invalid or not found. */
export class InvalidTechniqueError extends TransformationError {
    get errorCode() { return "INVALID_TECHNIQUE"; }

    constructor(message = "Invalid or unknown technique suite", techniqueName?: string) {
        super(message, techniqueName ? { technique: techniqueName } : undefined);
    }
}

/** Raised when transformation times out. */
export class TransformationTimeoutError extends TransformationError {
    get statusCode() { return 504; }
    get errorCode() { return "TRANSFORMATION_TIMEOUT"; }
    readonly timeoutMs: number;
    readonly technique?: string;

    constructor(timeoutMs: number, technique?: string) {
        super(
            `Transformation timed out after ${timeoutMs}ms${technique ? ` for technique "${technique}"` : ""}`,
            { timeoutMs, technique }
        );
        this.timeoutMs = timeoutMs;
        this.technique = technique;
    }
}

// ============================================================================
// Streaming & WebSocket Errors
// ============================================================================

/** Raised when streaming error occurs. */
export class StreamingError extends APIError {
    get statusCode() { return 500; }
    get errorCode() { return "STREAMING_ERROR"; }
    readonly streamId?: string;

    constructor(message = "Streaming error occurred", streamId?: string, details?: Record<string, unknown>, requestId?: string) {
        super(message, { ...details, streamId }, requestId);
        this.streamId = streamId;
    }
}

/** Raised when WebSocket error occurs. */
export class WebSocketError extends APIError {
    get statusCode() { return 500; }
    get errorCode() { return "WEBSOCKET_ERROR"; }
    readonly wsCode?: number;
    readonly wsReason?: string;

    constructor(message = "WebSocket error occurred", wsCode?: number, wsReason?: string, details?: Record<string, unknown>, requestId?: string) {
        super(message, { ...details, wsCode, wsReason }, requestId);
        this.wsCode = wsCode;
        this.wsReason = wsReason;
    }

    static fromCloseEvent(event: CloseEvent): WebSocketError {
        return new WebSocketError(
            `WebSocket closed: ${event.reason || "Unknown reason"}`,
            event.code,
            event.reason
        );
    }
}

// ============================================================================
// Resilience Errors
// ============================================================================

/** Raised when circuit breaker is open. */
export class CircuitBreakerOpenError extends APIError {
    get statusCode() { return 503; }
    get errorCode() { return "CIRCUIT_BREAKER_OPEN"; }
    readonly retryAfter: number;
    readonly circuitName: string;

    constructor(circuitName: string, retryAfter: number) {
        super(`Circuit breaker '${circuitName}' is open. Retry after ${retryAfter.toFixed(1)}s`, {
            circuit_name: circuitName,
            retry_after: retryAfter,
        });
        this.circuitName = circuitName;
        this.retryAfter = retryAfter;
    }
}

// ============================================================================
// Network Errors
// ============================================================================

/** Raised when network request fails (no response from server). */
export class NetworkError extends APIError {
    get statusCode() { return 0; }
    get errorCode() { return "NETWORK_ERROR"; }

    constructor(message = "Network error - unable to reach server", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when request is aborted/cancelled. */
export class RequestAbortedError extends APIError {
    get statusCode() { return 0; }
    get errorCode() { return "REQUEST_ABORTED"; }

    constructor(message = "Request was aborted", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}

/** Raised when a generic timeout occurs. */
export class TimeoutError extends APIError {
    get statusCode() { return 408; }
    get errorCode() { return "TIMEOUT_ERROR"; }

    constructor(message = "Request timed out", details?: Record<string, unknown>, requestId?: string) {
        super(message, details, requestId);
    }
}
