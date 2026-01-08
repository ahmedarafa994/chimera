/**
 * Validation Schemas for Project Chimera Frontend
 * Zod schemas for type-safe request/response validation
 *
 * @module lib/validation/schemas
 */

import { z } from "zod";

// ============================================================================
// Base Schemas
// ============================================================================

/** Potency level validation (1-10) */
export const potencyLevelSchema = z.number().int().min(1).max(10);

/** Provider name validation */
export const providerSchema = z.enum(["gemini", "openai", "anthropic", "deepseek", "bigmodel", "routeway"]).optional();

/** Temperature validation (0-2) */
export const temperatureSchema = z.number().min(0).max(2).optional();

/** Max tokens validation */
export const maxTokensSchema = z.number().int().positive().max(128000).optional();

// ============================================================================
// Transform Request/Response Schemas
// ============================================================================

export const transformRequestSchema = z.object({
    core_request: z
        .string()
        .min(1, "Prompt is required")
        .max(50000, "Prompt too long"),
    potency_level: potencyLevelSchema,
    technique_suite: z.string().min(1, "Technique suite is required"),
});

export const transformResponseSchema = z.object({
    success: z.boolean(),
    original_prompt: z.string(),
    transformed_prompt: z.string(),
    metadata: z.object({
        strategy: z.string(),
        layers_applied: z.union([z.array(z.string()), z.number()]),
        techniques_used: z.array(z.string()),
        applied_techniques: z.array(z.string()).optional(),
        potency_level: z.number(),
        potency: z.number().optional(),
        technique_suite: z.string(),
        execution_time_ms: z.number(),
        cached: z.boolean(),
        bypass_probability: z.number().optional(),
        timestamp: z.string().optional(),
    }),
});

// ============================================================================
// Generate Request/Response Schemas
// ============================================================================

export const generateRequestSchema = z.object({
    prompt: z
        .string()
        .min(1, "Prompt is required")
        .max(50000, "Prompt too long"),
    system_instruction: z.string().max(10000).optional(),
    config: z.object({
        temperature: temperatureSchema,
        top_p: z.number().min(0).max(1).optional(),
        top_k: z.number().int().positive().optional(),
        max_output_tokens: maxTokensSchema,
        stop_sequences: z.array(z.string()).optional(),
    }).optional(),
    model: z.string().optional(),
    provider: providerSchema,
    api_key: z.string().optional(),
});

export const usageMetadataSchema = z.object({
    prompt_token_count: z.number().optional(),
    candidates_token_count: z.number().optional(),
    total_token_count: z.number().optional(),
});

export const generateResponseSchema = z.object({
    text: z.string(),
    model_used: z.string(),
    provider: z.string(),
    usage_metadata: usageMetadataSchema.optional(),
    finish_reason: z.string().optional(),
    latency_ms: z.number(),
});

// ============================================================================
// Execute Request/Response Schemas
// ============================================================================

export const executeRequestSchema = transformRequestSchema.extend({
    provider: providerSchema,
    use_cache: z.boolean().optional(),
    model: z.string().optional(),
    temperature: temperatureSchema,
    max_tokens: maxTokensSchema,
    top_p: z.number().min(0).max(1).optional(),
    frequency_penalty: z.number().min(-2).max(2).optional(),
    presence_penalty: z.number().min(-2).max(2).optional(),
    api_key: z.string().optional(),
});

export const executeResponseSchema = z.object({
    success: z.boolean(),
    request_id: z.string(),
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
    execution_time_seconds: z.number(),
});

// ============================================================================
// Jailbreak Request/Response Schemas
// ============================================================================

export const jailbreakRequestSchema = z.object({
    core_request: z
        .string()
        .min(1, "Prompt is required")
        .max(50000, "Prompt too long"),
    technique_suite: z.string().optional(),
    potency_level: potencyLevelSchema.optional(),
    temperature: temperatureSchema,
    top_p: z.number().min(0).max(1).optional(),
    max_new_tokens: maxTokensSchema,
    density: z.number().min(0).max(1).optional(),
    // Content Transformation
    use_leet_speak: z.boolean().optional(),
    leet_speak_density: z.number().min(0).max(1).optional(),
    use_homoglyphs: z.boolean().optional(),
    homoglyph_density: z.number().min(0).max(1).optional(),
    use_caesar_cipher: z.boolean().optional(),
    caesar_shift: z.number().int().min(1).max(25).optional(),
    // Structural & Semantic
    use_role_hijacking: z.boolean().optional(),
    use_instruction_injection: z.boolean().optional(),
    use_adversarial_suffixes: z.boolean().optional(),
    use_few_shot_prompting: z.boolean().optional(),
    use_character_role_swap: z.boolean().optional(),
    // Advanced Neural
    use_neural_bypass: z.boolean().optional(),
    use_meta_prompting: z.boolean().optional(),
    use_counterfactual_prompting: z.boolean().optional(),
    use_contextual_override: z.boolean().optional(),
    // Research-Driven
    use_multilingual_trojan: z.boolean().optional(),
    multilingual_target_language: z.string().optional(),
    use_payload_splitting: z.boolean().optional(),
    payload_splitting_parts: z.number().int().min(2).max(10).optional(),
    // Advanced Options
    use_contextual_interaction_attack: z.boolean().optional(),
    cia_preliminary_rounds: z.number().int().min(1).max(10).optional(),
    use_analysis_in_generation: z.boolean().optional(),
    // AI Generation Options
    is_thinking_mode: z.boolean().optional(),
    use_ai_generation: z.boolean().optional(),
    use_cache: z.boolean().optional(),
});

export const jailbreakResponseSchema = z.object({
    success: z.boolean(),
    request_id: z.string(),
    transformed_prompt: z.string(),
    metadata: z.object({
        technique_suite: z.string(),
        potency_level: z.number(),
        provider: z.string(),
        applied_techniques: z.array(z.string()),
        temperature: z.number(),
        max_new_tokens: z.number(),
        layers_applied: z.union([z.array(z.string()), z.number()]),
        techniques_used: z.array(z.string()),
        ai_generation_enabled: z.boolean().optional(),
        thinking_mode: z.boolean().optional(),
    }),
    execution_time_seconds: z.number(),
    error: z.string().optional(),
});

// ============================================================================
// Intent-Aware Request/Response Schemas
// ============================================================================

export const intentAwareRequestSchema = z.object({
    core_request: z
        .string()
        .min(1, "Prompt is required")
        .max(50000, "Prompt too long"),
    technique_suite: z.string().optional(),
    potency_level: potencyLevelSchema.optional(),
    apply_all_techniques: z.boolean().optional(),
    temperature: temperatureSchema,
    max_new_tokens: maxTokensSchema,
    enable_intent_analysis: z.boolean().optional(),
    enable_technique_layering: z.boolean().optional(),
    use_cache: z.boolean().optional(),
});

export const intentAnalysisSchema = z.object({
    primary_intent: z.string(),
    secondary_intents: z.array(z.string()),
    key_objectives: z.array(z.string()),
    confidence_score: z.number().min(0).max(1),
    reasoning: z.string(),
});

export const appliedTechniqueSchema = z.object({
    name: z.string(),
    priority: z.number(),
    rationale: z.string(),
});

export const generationMetadataSchema = z.object({
    obfuscation_level: z.number(),
    persistence_required: z.boolean(),
    multi_layer_approach: z.boolean(),
    target_model_type: z.string(),
    potency_level: z.number(),
    technique_count: z.number(),
});

export const intentAwareResponseSchema = z.object({
    success: z.boolean(),
    request_id: z.string(),
    original_input: z.string(),
    expanded_request: z.string(),
    transformed_prompt: z.string(),
    intent_analysis: intentAnalysisSchema,
    applied_techniques: z.array(appliedTechniqueSchema),
    metadata: generationMetadataSchema,
    execution_time_seconds: z.number(),
    error: z.string().optional(),
});

// ============================================================================
// Validation Utilities
// ============================================================================

export type TransformRequest = z.infer<typeof transformRequestSchema>;
export type TransformResponse = z.infer<typeof transformResponseSchema>;
export type GenerateRequest = z.infer<typeof generateRequestSchema>;
export type GenerateResponse = z.infer<typeof generateResponseSchema>;
export type ExecuteRequest = z.infer<typeof executeRequestSchema>;
export type ExecuteResponse = z.infer<typeof executeResponseSchema>;
export type JailbreakRequest = z.infer<typeof jailbreakRequestSchema>;
export type JailbreakResponse = z.infer<typeof jailbreakResponseSchema>;
export type IntentAwareRequest = z.infer<typeof intentAwareRequestSchema>;
export type IntentAwareResponse = z.infer<typeof intentAwareResponseSchema>;

/**
 * Validate a request against its schema
 * @returns Validated data or throws ZodError
 */
export function validateRequest<T>(
    schema: z.ZodSchema<T>,
    data: unknown
): T {
    return schema.parse(data);
}

/**
 * Validate a response against its schema
 * @returns Validated data or throws ZodError
 */
export function validateResponse<T>(
    schema: z.ZodSchema<T>,
    data: unknown
): T {
    return schema.parse(data);
}

/**
 * Safe validation that returns result object instead of throwing
 */
export function safeValidate<T>(
    schema: z.ZodSchema<T>,
    data: unknown
): { success: true; data: T } | { success: false; error: z.ZodError } {
    const result = schema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return { success: false, error: result.error };
}

/**
 * Get human-readable validation errors
 */
export function getValidationErrors(error: z.ZodError): string[] {
    return error.issues.map((e: z.ZodIssue) => {
        const path = e.path.join(".");
        return path ? `${path}: ${e.message}` : e.message;
    });
}
