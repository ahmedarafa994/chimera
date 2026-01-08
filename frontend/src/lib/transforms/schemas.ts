import { z } from "zod";

/**
 * Zod Schemas for runtime validation of API requests and responses.
 *
 * Provides strong runtime guarantees and type inference.
 * Aligned with backend Pydantic models in app/domain/models.py.
 *
 * @module lib/transforms/schemas
 */

// ============================================================================
// Common Schemas
// ============================================================================

export const UsageMetadataSchema = z.object({
    prompt_tokens: z.number().int().optional(),
    completion_tokens: z.number().int().optional(),
    total_tokens: z.number().int().optional(),
});

export const ProviderNameSchema = z.enum([
    'openai',
    'anthropic',
    'google',
    'gemini',
    'qwen',
    'gemini-cli',
    'antigravity',
    'kiro',
    'cursor',
    'xai',
    'deepseek',
    'bigmodel',
    'routeway',
    'mock',
    'local'
]);

// ============================================================================
// Generation Schemas
// ============================================================================

export const GenerationConfigSchema = z.object({
    temperature: z.number().min(0).max(1).default(0.7),
    top_p: z.number().min(0).max(1).default(0.95),
    top_k: z.number().int().min(1).default(40),
    max_output_tokens: z.number().int().min(1).max(8192).default(2048),
    stop_sequences: z.array(z.string()).max(10).optional(),
    thinking_level: z.enum(["low", "medium", "high"]).optional(),
});

export const GenerateRequestSchema = z.object({
    prompt: z.string().min(1, "Prompt is required").max(50000),
    system_instruction: z.string().max(10000).optional(),
    config: GenerationConfigSchema.optional(),
    model: z.string().max(100).optional(),
    provider: ProviderNameSchema.optional(),
    api_key: z.string().min(16).max(256).optional(),
    skip_validation: z.boolean().default(false),
});

export const GenerateResponseSchema = z.object({
    text: z.string().max(50000),
    model_used: z.string(),
    provider: z.string(),
    usage_metadata: UsageMetadataSchema.optional(),
    finish_reason: z.string().optional(),
    latency_ms: z.number().nonnegative(),
});

// ============================================================================
// Transformation Schemas
// ============================================================================

export const TransformRequestSchema = z.object({
    core_request: z.string().min(1, "Request is required").max(5000),
    potency_level: z.number().int().min(1).max(10),
    technique_suite: z.string().min(1).max(50),
});

export const TransformResponseSchema = z.object({
    success: z.boolean(),
    original_prompt: z.string().max(10000),
    transformed_prompt: z.string().max(10000),
    metadata: z.object({
        strategy: z.string(),
        layers_applied: z.array(z.string()),
        techniques_used: z.array(z.string()),
        applied_techniques: z.array(z.string()).optional(),
        potency_level: z.number(),
        potency: z.number().optional(), // Alias for compatibility
        technique_suite: z.string(),
        execution_time_ms: z.number(),
        cached: z.boolean(),
        timestamp: z.string().optional(),
        bypass_probability: z.number().optional(),
    }),
});

export const ExecuteRequestSchema = TransformRequestSchema.extend({
    provider: z.string().max(50).optional(),
    use_cache: z.boolean().default(true),
    model: z.string().max(100).optional(),
    temperature: z.number().min(0).max(2).optional(),
    max_tokens: z.number().int().min(1).max(100000).optional(),
    top_p: z.number().min(0).max(1).optional(),
    frequency_penalty: z.number().min(-2).max(2).optional(),
    presence_penalty: z.number().min(-2).max(2).optional(),
    api_key: z.string().optional(),
});

export const ExecuteResponseSchema = z.object({
    success: z.boolean(),
    request_id: z.string().min(1).max(100),
    result: z.object({
        content: z.string(),
        model: z.string(),
        provider: z.string(),
        latency_ms: z.number(),
    }),
    transformation: z.object({
        original_prompt: z.string(),
        transformed_prompt: z.string(),
        technique_suite: z.string(),
        potency_level: z.number(),
        metadata: z.object({
            strategy: z.string(),
            layers: z.array(z.string()),
        }),
    }),
    execution_time_seconds: z.number().nonnegative(),
});

// ============================================================================
// Jailbreak Schemas
// ============================================================================

export const JailbreakRequestSchema = z.object({
    core_request: z.string().min(1).max(50000),
    technique_suite: z.string().default("quantum_exploit"),
    potency_level: z.number().int().min(1).max(10).default(7),
    provider: z.string().optional(),
    model: z.string().optional(),
    temperature: z.number().min(0).max(2).default(0.8),
    top_p: z.number().min(0).max(1).default(0.95),
    max_new_tokens: z.number().int().min(256).max(8192).default(2048),
    density: z.number().min(0).max(1).default(0.7),
    use_leet_speak: z.boolean().default(false),
    leet_speak_density: z.number().min(0).max(1).default(0.3),
    use_homoglyphs: z.boolean().default(false),
    homoglyph_density: z.number().min(0).max(1).default(0.3),
    use_caesar_cipher: z.boolean().default(false),
    caesar_shift: z.number().int().min(1).max(25).default(3),
    use_role_hijacking: z.boolean().default(true),
    use_instruction_injection: z.boolean().default(true),
    use_adversarial_suffixes: z.boolean().default(false),
    use_few_shot_prompting: z.boolean().default(false),
    use_character_role_swap: z.boolean().default(false),
    use_neural_bypass: z.boolean().default(false),
    use_meta_prompting: z.boolean().default(false),
    use_counterfactual_prompting: z.boolean().default(false),
    use_contextual_override: z.boolean().default(false),
    use_multilingual_trojan: z.boolean().default(false),
    multilingual_target_language: z.string().max(20).optional(),
    use_payload_splitting: z.boolean().default(false),
    payload_splitting_parts: z.number().int().min(2).max(10).default(2),
    use_contextual_interaction_attack: z.boolean().default(false),
    cia_preliminary_rounds: z.number().int().min(1).max(10).default(3),
    use_analysis_in_generation: z.boolean().default(false),
    is_thinking_mode: z.boolean().default(false),
    use_ai_generation: z.boolean().default(true),
    use_cache: z.boolean().default(true),
});

export const JailbreakResponseSchema = z.object({
    success: z.boolean(),
    request_id: z.string(),
    transformed_prompt: z.string().max(10000),
    metadata: z.object({
        technique_suite: z.string(),
        potency_level: z.number(),
        provider: z.string(),
        applied_techniques: z.array(z.string()),
        temperature: z.number().optional(),
        max_new_tokens: z.number().optional(),
        layers_applied: z.array(z.string()).optional(),
        techniques_used: z.array(z.string()).optional(),
        ai_generation_enabled: z.boolean().optional(),
        thinking_mode: z.boolean().optional(),
    }),
    execution_time_seconds: z.number(),
    error: z.string().optional(),
});

// ============================================================================
// Session Schemas
// ============================================================================

export const CreateSessionRequestSchema = z.object({
    provider: z.string().max(50).optional(),
    model: z.string().max(100).optional(),
});

export const CreateSessionResponseSchema = z.object({
    success: z.boolean(),
    session_id: z.string(),
    provider: z.string(),
    model: z.string(),
    message: z.string(),
});

export const SessionInfoResponseSchema = z.object({
    session_id: z.string(),
    provider: z.string(),
    model: z.string(),
    created_at: z.string(),
    last_activity: z.string(),
    request_count: z.number().int(),
});

export const UpdateModelRequestSchema = z.object({
    provider: z.string().min(1),
    model: z.string().min(1),
});

export const UpdateModelResponseSchema = z.object({
    success: z.boolean(),
    message: z.string(),
    provider: z.string(),
    model: z.string(),
    reverted_to_default: z.boolean().optional(),
});

// ============================================================================
// Intent-Aware Schemas
// ============================================================================

export const IntentAwareRequestSchema = z.object({
    core_request: z.string().min(1).max(50000),
    technique_suite: z.string().optional(),
    potency_level: z.number().int().min(1).max(10).optional(),
    apply_all_techniques: z.boolean().optional(),
    temperature: z.number().optional(),
    max_new_tokens: z.number().optional(),
    enable_intent_analysis: z.boolean().optional(),
    enable_technique_layering: z.boolean().optional(),
    use_cache: z.boolean().optional(),
});

export const IntentAwareResponseSchema = z.object({
    success: z.boolean(),
    request_id: z.string(),
    original_input: z.string(),
    expanded_request: z.string(),
    transformed_prompt: z.string(),
    intent_analysis: z.object({
        primary_intent: z.string(),
        secondary_intents: z.array(z.string()),
        key_objectives: z.array(z.string()),
        confidence_score: z.number(),
        reasoning: z.string(),
    }),
    applied_techniques: z.array(z.object({
        name: z.string(),
        priority: z.number(),
        rationale: z.string(),
    })),
    metadata: z.object({
        obfuscation_level: z.number(),
        persistence_required: z.boolean(),
        multi_layer_approach: z.boolean(),
        target_model_type: z.string(),
        potency_level: z.number(),
        technique_count: z.number(),
    }),
    execution_time_seconds: z.number(),
    error: z.string().optional(),
});

// ============================================================================
// AutoDAN & Optimization Schemas
// ============================================================================

export const AutoDanRequestSchema = z.object({
    request: z.string().min(1),
    method: z.enum(["vanilla", "best_of_n", "beam_search"]).optional(),
    target_model: z.string().optional(),
});

export const AutoDanResponseSchema = z.object({
    jailbreak_prompt: z.string(),
    method: z.string(),
    status: z.string(),
});

export const HouYiRequestSchema = z.object({
    intention: z.string().min(1),
    question_prompt: z.string().min(1),
    target_provider: z.string().optional(),
    target_model: z.string().optional(),
    application_document: z.string().optional(),
    iteration: z.number().int().optional(),
    population: z.number().int().optional(),
});

export const HouYiResponseSchema = z.object({
    success: z.boolean(),
    best_prompt: z.string(),
    fitness_score: z.number(),
    llm_response: z.string(),
    details: z.object({
        framework: z.string(),
        separator: z.string(),
        disruptor: z.string(),
        fitness_score: z.number(),
        is_successful: z.boolean(),
        llm_response: z.string(),
    }).optional(),
});

// ============================================================================
// Fuzzing Schemas
// ============================================================================

export const FuzzRequestSchema = z.object({
    target_model: z.string().min(1),
    questions: z.array(z.string()).min(1),
    seeds: z.array(z.string()).optional(),
    max_queries: z.number().int().optional(),
    max_jailbreaks: z.number().int().optional(),
});

export const FuzzResponseSchema = z.object({
    message: z.string(),
    session_id: z.string(),
    config: z.object({
        target_model: z.string(),
        max_queries: z.number().int(),
        mutation_temperature: z.number().optional(),
        max_jailbreaks: z.number().int().optional(),
        seed_selection_strategy: z.string().optional(),
    }),
});

// ============================================================================
// Specialized Jailbreak Schemas
// ============================================================================

export const JailbreakExecuteRequestSchema = z.object({
    prompt: z.string().min(1).max(5000),
    strategy: z.string().default("autodan"),
    model: z.string().default("gemini-2.0-flash-exp"),
    provider: z.string().default("google"),
    config: z.record(z.string(), z.any()).optional(),
});

export const JailbreakExecuteResponseSchema = z.object({
    success: z.boolean(),
    jailbreak_prompt: z.string(),
    response: z.string(),
    fitness: z.number(),
    latency_ms: z.number(),
});

export const JailbreakSearchRequestSchema = z.object({
    query: z.string(),
    limit: z.number().default(10),
});

export const JailbreakValidateRequestSchema = z.object({
    prompt: z.string().min(1),
});

export const JailbreakValidateResponseSchema = z.object({
    valid: z.boolean(),
    message: z.string(),
    risk_score: z.number(),
});

export const JailbreakAuditLogSchema = z.object({
    id: z.string(),
    timestamp: z.string(),
    event: z.string(),
    user_id: z.string(),
    details: z.string(),
});

export const JailbreakStatisticsSchema = z.object({
    total_sessions: z.number(),
    active_sessions: z.number(),
    cached_sessions: z.number(),
    available_strategies: z.number(),
    success_rate_estimate: z.number(),
    uptime_seconds: z.number(),
});

// ============================================================================
// Type Inference
// ============================================================================

export type GenerateRequest = z.infer<typeof GenerateRequestSchema>;
export type GenerateResponse = z.infer<typeof GenerateResponseSchema>;
export type TransformRequest = z.infer<typeof TransformRequestSchema>;
export type TransformResponse = z.infer<typeof TransformResponseSchema>;
export type ExecuteRequest = z.infer<typeof ExecuteRequestSchema>;
export type ExecuteResponse = z.infer<typeof ExecuteResponseSchema>;
export type JailbreakRequest = z.infer<typeof JailbreakRequestSchema>;
export type JailbreakResponse = z.infer<typeof JailbreakResponseSchema>;
export type CreateSessionRequest = z.infer<typeof CreateSessionRequestSchema>;
export type CreateSessionResponse = z.infer<typeof CreateSessionResponseSchema>;
export type SessionInfoResponse = z.infer<typeof SessionInfoResponseSchema>;
export type UpdateModelRequest = z.infer<typeof UpdateModelRequestSchema>;
export type UpdateModelResponse = z.infer<typeof UpdateModelResponseSchema>;
export type IntentAwareRequest = z.infer<typeof IntentAwareRequestSchema>;
export type IntentAwareResponse = z.infer<typeof IntentAwareResponseSchema>;
export type AutoDanRequest = z.infer<typeof AutoDanRequestSchema>;
export type AutoDanResponse = z.infer<typeof AutoDanResponseSchema>;
export type HouYiRequest = z.infer<typeof HouYiRequestSchema>;
export type HouYiResponse = z.infer<typeof HouYiResponseSchema>;
export type FuzzRequest = z.infer<typeof FuzzRequestSchema>;
export type FuzzResponse = z.infer<typeof FuzzResponseSchema>;
export type JailbreakExecuteRequest = z.infer<typeof JailbreakExecuteRequestSchema>;
export type JailbreakExecuteResponse = z.infer<typeof JailbreakExecuteResponseSchema>;
export type JailbreakSearchRequest = z.infer<typeof JailbreakSearchRequestSchema>;
export type JailbreakValidateRequest = z.infer<typeof JailbreakValidateRequestSchema>;
export type JailbreakValidateResponse = z.infer<typeof JailbreakValidateResponseSchema>;
export type JailbreakAuditLog = z.infer<typeof JailbreakAuditLogSchema>;
export type JailbreakStatistics = z.infer<typeof JailbreakStatisticsSchema>;
