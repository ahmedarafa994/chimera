/**
 * AI Service Layer for Project Chimera Frontend
 * Unified abstraction for all AI model interactions
 *
 * Features:
 * - Provider selection with automatic fallback
 * - Circuit breaker integration for resilience
 * - Request debouncing for rapid calls
 * - Response caching with TTL
 * - Retry with exponential backoff
 * - Comprehensive error handling
 * - Zod schema validation
 *
 * @module lib/services/ai-service
 */

import { apiClient, ENDPOINTS } from "../api/core";
import { CircuitState } from "../resilience/circuit-breaker";
import { type RetryConfig } from "../resilience/retry";
import { globalCache, generateCacheKey, CACHE_TTL } from "../cache/cache-manager";
import { debounce, DEBOUNCE_DELAYS, type DebouncedFunction } from "../utils/debounce";
import { providerService } from "./provider-service";
import * as schemas from "../transforms/schemas";

// ============================================================================
// Types
// ============================================================================

export interface AIServiceConfig {
    /** Enable response caching */
    enableCaching: boolean;
    /** Enable request debouncing */
    enableDebouncing: boolean;
    /** Enable circuit breaker */
    enableCircuitBreaker: boolean;
    /** Enable retry with backoff */
    enableRetry: boolean;
    /** Custom retry configuration */
    retryConfig?: Partial<RetryConfig>;
    /** Debounce delay in ms */
    debounceDelay?: number;
}

export interface AIServiceState {
    initialized: boolean;
    activeProvider: string | null;
    circuitState: CircuitState;
    cacheStats: {
        hits: number;
        misses: number;
        hitRate: number;
    };
    lastError: Error | null;
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: AIServiceConfig = {
    enableCaching: true,
    enableDebouncing: true,
    enableCircuitBreaker: true,
    enableRetry: true,
    retryConfig: {
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
    },
    debounceDelay: DEBOUNCE_DELAYS.AI_GENERATION,
};

// ============================================================================
// AI Service Implementation
// ============================================================================

class AIServiceClass {
    private config: AIServiceConfig;
    private initialized: boolean = false;
    private lastError: Error | null = null;

    // Debounced function instances
    private debouncedGenerate: DebouncedFunction<schemas.GenerateRequest, schemas.GenerateResponse> | null = null;
    private debouncedTransform: DebouncedFunction<schemas.TransformRequest, schemas.TransformResponse> | null = null;

    constructor(config: Partial<AIServiceConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.initializeDebouncedFunctions();
    }

    /**
     * Initialize the AI service
     */
    async initialize(): Promise<void> {
        if (this.initialized) return;
        this.initialized = true;
    }

    /**
     * Initialize debounced function wrappers
     */
    private initializeDebouncedFunctions(): void {
        if (!this.config.enableDebouncing) return;

        this.debouncedGenerate = debounce(
            (request: schemas.GenerateRequest) => this.executeGenerate(request),
            { delay: this.config.debounceDelay ?? DEBOUNCE_DELAYS.AI_GENERATION }
        );

        this.debouncedTransform = debounce(
            (request: schemas.TransformRequest) => this.executeTransform(request),
            { delay: this.config.debounceDelay ?? DEBOUNCE_DELAYS.API_CALL }
        );
    }

    /**
     * Generate text using the AI model
     */
    async generate(request: schemas.GenerateRequest): Promise<schemas.GenerateResponse> {
        await this.ensureInitialized();
        if (this.config.enableDebouncing && this.debouncedGenerate) {
            return this.debouncedGenerate(request);
        }
        return this.executeGenerate(request);
    }

    /**
     * Transform a prompt
     */
    async transform(request: schemas.TransformRequest): Promise<schemas.TransformResponse> {
        await this.ensureInitialized();
        if (this.config.enableDebouncing && this.debouncedTransform) {
            return this.debouncedTransform(request);
        }
        return this.executeTransform(request);
    }

    /**
     * Execute full pipeline
     */
    async execute(request: schemas.ExecuteRequest): Promise<schemas.ExecuteResponse> {
        await this.ensureInitialized();
        return apiClient.post<schemas.ExecuteResponse>(ENDPOINTS.EXECUTE, request, {
            requestSchema: schemas.ExecuteRequestSchema,
            responseSchema: schemas.ExecuteResponseSchema
        });
    }

    /**
     * Jailbreak generation
     */
    async jailbreak(request: schemas.JailbreakRequest): Promise<schemas.JailbreakResponse> {
        await this.ensureInitialized();
        return apiClient.post<schemas.JailbreakResponse>(ENDPOINTS.JAILBREAK_GENERATE, request, {
            requestSchema: schemas.JailbreakRequestSchema,
            responseSchema: schemas.JailbreakResponseSchema,
        });
    }

    /**
     * Intent-aware generation
     */
    async intentAware(request: schemas.IntentAwareRequest): Promise<schemas.IntentAwareResponse> {
        await this.ensureInitialized();
        return apiClient.post<schemas.IntentAwareResponse>(ENDPOINTS.INTENT_AWARE_GENERATE, request, {
            requestSchema: schemas.IntentAwareRequestSchema,
            responseSchema: schemas.IntentAwareResponseSchema
        });
    }

    private async executeGenerate(request: schemas.GenerateRequest): Promise<schemas.GenerateResponse> {
        if (this.config.enableCaching) {
            const cacheKey = generateCacheKey("generate", request);
            const cached = globalCache.get<schemas.GenerateResponse>(cacheKey);
            if (cached) return cached;
        }

        const resolvedRequest = { ...request };
        if (!resolvedRequest.provider) {
            const provider = providerService.getActive();
            if (provider) {
                resolvedRequest.provider = provider.name as any;
                resolvedRequest.model = provider.models[0]?.id || "default";
            }
        }

        const result = await apiClient.post<schemas.GenerateResponse>(ENDPOINTS.GENERATE, resolvedRequest, {
            requestSchema: schemas.GenerateRequestSchema,
            responseSchema: schemas.GenerateResponseSchema,
            skipCache: true, // We handle caching here
        });

        if (this.config.enableCaching) {
            const cacheKey = generateCacheKey("generate", request);
            globalCache.set(cacheKey, result, CACHE_TTL.GENERATION);
        }

        return result;
    }

    private async executeTransform(request: schemas.TransformRequest): Promise<schemas.TransformResponse> {
        if (this.config.enableCaching) {
            const cacheKey = generateCacheKey("transform", request);
            const cached = globalCache.get<schemas.TransformResponse>(cacheKey);
            if (cached) return cached;
        }

        const result = await apiClient.post<schemas.TransformResponse>(ENDPOINTS.TRANSFORM, request, {
            requestSchema: schemas.TransformRequestSchema,
            responseSchema: schemas.TransformResponseSchema,
            skipCache: true,
        });

        if (this.config.enableCaching) {
            const cacheKey = generateCacheKey("transform", request);
            globalCache.set(cacheKey, result, CACHE_TTL.GENERATION);
        }

        return result;
    }

    private async ensureInitialized(): Promise<void> {
        if (!this.initialized) await this.initialize();
    }

    getState(): AIServiceState {
        const cacheStats = globalCache.getStats();
        // Since we use apiClient which handles circuit breakers internally per endpoint,
        // we might not have a single "ai-service" circuit.
        // But for UI compatibility we check the main 'text' endpoint.
        const cb = (apiClient as any).getCircuitBreaker ? (apiClient as any).getCircuitBreaker(ENDPOINTS.GENERATE) : null;

        return {
            initialized: this.initialized,
            activeProvider: providerService.getActive()?.name ?? null,
            circuitState: cb?.getState() ?? CircuitState.CLOSED,
            cacheStats: {
                hits: cacheStats.hits,
                misses: cacheStats.misses,
                hitRate: cacheStats.hitRate,
            },
            lastError: this.lastError,
        };
    }

    updateConfig(config: Partial<AIServiceConfig>): void {
        this.config = { ...this.config, ...config };
        this.initializeDebouncedFunctions();
    }

    /**
     * Generate and execute a jailbreak attack
     */
    async executeJailbreak(data: schemas.JailbreakExecuteRequest): Promise<schemas.JailbreakExecuteResponse> {
        return apiClient.post<schemas.JailbreakExecuteResponse>(ENDPOINTS.JAILBREAK_RUN_EXECUTE, data, {
            requestSchema: schemas.JailbreakExecuteRequestSchema,
            responseSchema: schemas.JailbreakExecuteResponseSchema,
        });
    }

    /**
     * Get jailbreak statistics
     */
    async getJailbreakStatistics(): Promise<schemas.JailbreakStatistics> {
        return apiClient.get<schemas.JailbreakStatistics>(ENDPOINTS.JAILBREAK_STATS, {
            responseSchema: schemas.JailbreakStatisticsSchema,
        });
    }

    /**
     * Validate a prompt for jailbreak potential/risk
     */
    async validateJailbreakPrompt(prompt: string): Promise<schemas.JailbreakValidateResponse> {
        return apiClient.post<schemas.JailbreakValidateResponse>(ENDPOINTS.JAILBREAK_PROMPT_VALIDATE, { prompt }, {
            requestSchema: schemas.JailbreakValidateRequestSchema,
            responseSchema: schemas.JailbreakValidateResponseSchema,
        });
    }

    /**
     * Search through historical jailbreaks
     */
    async searchJailbreaks(query: string, limit: number = 10): Promise<any[]> {
        return apiClient.post<any[]>(ENDPOINTS.JAILBREAK_PROMPT_SEARCH, { query, limit }, {
            requestSchema: schemas.JailbreakSearchRequestSchema,
        });
    }

    /**
     * Get jailbreak audit logs
     */
    async getJailbreakAuditLogs(): Promise<schemas.JailbreakAuditLog[]> {
        return apiClient.get<schemas.JailbreakAuditLog[]>(ENDPOINTS.JAILBREAK_AUDIT);
    }

    /**
     * Get available transformation techniques (cached)
     */
    async getTechniques(): Promise<any[]> {
        const cacheKey = generateCacheKey("techniques");
        const cached = globalCache.get<any[]>(cacheKey);
        if (cached) return cached;

        const techniques = await apiClient.get<any[]>(ENDPOINTS.TECHNIQUES);
        globalCache.set(cacheKey, techniques, CACHE_TTL.TECHNIQUES);
        return techniques;
    }

    reset(): void {
        this.debouncedGenerate?.cancel();
        this.debouncedTransform?.cancel();
        this.initialized = false;
        this.lastError = null;
    }
}

export const AIService = new AIServiceClass();

export function getAIServiceSnapshot(): AIServiceState {
    return AIService.getState();
}

export function subscribeToAIService(callback: (state: AIServiceState) => void): () => void {
    const interval = setInterval(() => callback(AIService.getState()), 2000);
    return () => clearInterval(interval);
}
