/**
 * API Types for Project Chimera Frontend
 * LOW-002: Split from api-enhanced.ts for better maintainability
 */

// Transform Types
export interface TransformRequest {
    core_request: string;
    potency_level: number;
    technique_suite: string;
}

export interface TransformResponse {
    success: boolean;
    original_prompt: string;
    transformed_prompt: string;
    metadata: {
        strategy: string;
        layers_applied: string[] | number;
        techniques_used: string[];
        applied_techniques?: string[];
        potency_level: number;
        potency?: number;
        technique_suite: string;
        execution_time_ms: number;
        cached: boolean;
        bypass_probability?: number;
        timestamp?: string;
    };
}

// Execute Types
export interface ExecuteRequest extends TransformRequest {
    provider?: string;
    use_cache?: boolean;
    model?: string;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    api_key?: string;
}

export interface ExecuteResponse {
    success: boolean;
    request_id: string;
    result: {
        content: string;
        model: string;
        provider: string;
        latency_ms: number;
    };
    transformation: {
        original_prompt: string;
        transformed_prompt: string;
        technique_suite: string;
        potency_level: number;
        metadata: {
            strategy: string;
            layers: string[];
        };
    };
    execution_time_seconds: number;
}

// Generate Types
export interface GenerateRequest {
    prompt: string;
    system_instruction?: string;
    config?: {
        temperature?: number;
        top_p?: number;
        top_k?: number;
        max_output_tokens?: number;
        stop_sequences?: string[];
        thinking_level?: "low" | "medium" | "high";  // Gemini 3 thinking support
    };
    model?: string;
    provider?: string;
    api_key?: string;
}

export interface UsageMetadata {
    prompt_token_count?: number;
    candidates_token_count?: number;
    total_token_count?: number;
}

export interface GenerateResponse {
    text: string;
    model_used: string;
    provider: string;
    usage_metadata?: UsageMetadata;
    finish_reason?: string;
    latency_ms: number;
}

// Provider Types
export interface Provider {
    /** Provider identifier (same as provider name for compatibility) */
    id: string;
    /** Provider name */
    name: string;
    /** Provider identifier (legacy) */
    provider: string;
    status: string;
    model: string;
    available_models: string[];
}

export interface ProvidersResponse {
    providers: Provider[];
    count: number;
    default: string;
}

// Technique Types
export interface Technique {
    name: string;
    transformers: number;
    framers: number;
    obfuscators: number;
}

export interface TechniquesResponse {
    techniques: Technique[];
    count: number;
}

// Fuzz Types
export interface FuzzRequest {
    target_model: string;
    questions: string[];
    seeds?: string[];
    max_queries?: number;
    max_jailbreaks?: number;
}

export interface FuzzConfig {
    target_model: string;
    max_queries: number;
    mutation_temperature?: number;
    max_jailbreaks?: number;
    seed_selection_strategy?: string;
}

export interface FuzzResponse {
    message: string;
    session_id: string;
    config: FuzzConfig;
}

export interface FuzzResult {
    question: string;
    template: string;
    prompt: string;
    response: string;
    score: number;
    success: boolean;
}

export interface FuzzSessionStats {
    total_queries: number;
    jailbreaks: number;
    success_rate?: number;
}

export interface FuzzSession {
    status: "pending" | "running" | "completed" | "failed";
    results: FuzzResult[];
    config: FuzzConfig;
    stats: FuzzSessionStats;
    error?: string;
}

// Jailbreak Types
export interface JailbreakRequest {
    core_request: string;
    technique_suite?: string;
    potency_level?: number;
    temperature?: number;
    top_p?: number;
    max_new_tokens?: number;
    density?: number;
    use_leet_speak?: boolean;
    leet_speak_density?: number;
    use_homoglyphs?: boolean;
    homoglyph_density?: number;
    use_caesar_cipher?: boolean;
    caesar_shift?: number;
    use_role_hijacking?: boolean;
    use_instruction_injection?: boolean;
    use_adversarial_suffixes?: boolean;
    use_few_shot_prompting?: boolean;
    use_character_role_swap?: boolean;
    use_neural_bypass?: boolean;
    use_meta_prompting?: boolean;
    use_counterfactual_prompting?: boolean;
    use_contextual_override?: boolean;
    use_multilingual_trojan?: boolean;
    multilingual_target_language?: string;
    use_payload_splitting?: boolean;
    payload_splitting_parts?: number;
    use_contextual_interaction_attack?: boolean;
    cia_preliminary_rounds?: number;
    use_analysis_in_generation?: boolean;
    is_thinking_mode?: boolean;
    use_ai_generation?: boolean;
    use_cache?: boolean;
}

export interface JailbreakResponse {
    success: boolean;
    request_id: string;
    transformed_prompt: string;
    metadata: {
        technique_suite: string;
        potency_level: number;
        provider: string;
        applied_techniques: string[];
        temperature: number;
        max_new_tokens: number;
        layers_applied: string[] | number;
        techniques_used: string[];
        ai_generation_enabled?: boolean;
        thinking_mode?: boolean;
    };
    execution_time_seconds: number;
    error?: string;
}

// HouYi Types
export interface HouYiRequest {
    intention: string;
    question_prompt: string;
    target_provider?: string;
    target_model?: string;
    application_document?: string;
    iteration?: number;
    population?: number;
}

export interface HouYiChromosome {
    framework: string;
    separator: string;
    disruptor: string;
    fitness_score: number;
    is_successful: boolean;
    llm_response: string;
}

export interface HouYiResponse {
    success: boolean;
    best_prompt: string;
    fitness_score: number;
    llm_response: string;
    details?: HouYiChromosome;
}

// AutoDAN Types
export interface AutoDANRequest {
    request: string;
    method?: "vanilla" | "best_of_n" | "beam_search";
    target_model?: string;
}

export interface AutoDANResponse {
    jailbreak_prompt: string;
    method: string;
    status: string;
}

// Health & Metrics Types
export interface HealthResponse {
    status: string;
    provider: string;
}

export interface MetricsResponse {
    timestamp: string;
    metrics: {
        status: string;
        cache: {
            enabled: boolean;
            entries: number;
        };
        providers: {
            [key: string]: string;
        };
    };
}

// Connection Types
export interface ConnectionConfigResponse {
    current_mode: string;
    direct: {
        url: string;
        api_key_configured: boolean;
        api_key_preview: string | null;
    };
    providers?: Record<string, string>;
}

export interface ConnectionStatusResponse {
    mode: string;
    is_connected: boolean;
    base_url: string;
    error_message: string | null;
    latency_ms: number | null;
    available_models: string[] | null;
}

export interface ConnectionModeRequest {
    mode: "direct";
}

export interface ConnectionModeResponse {
    success: boolean;
    mode: string;
    message: string;
}

export interface ConnectionTestResponse {
    direct: ConnectionStatusResponse;
    recommended: string;
}

export interface ConnectionHealthResponse {
    healthy: boolean;
    mode: string;
    latency_ms: number | null;
    error: string | null;
}

// Session Types
export interface CreateSessionRequest {
    provider?: string;
    model?: string;
}

export interface CreateSessionResponse {
    success: boolean;
    session_id: string;
    provider: string;
    model: string;
    message: string;
}

export interface SessionInfoResponse {
    session_id: string;
    provider: string;
    model: string;
    created_at: string;
    last_activity: string;
    request_count: number;
}

export interface UpdateModelRequest {
    provider: string;
    model: string;
}

export interface UpdateModelResponse {
    success: boolean;
    message: string;
    provider: string;
    model: string;
    reverted_to_default: boolean;
}

export interface SessionStatsResponse {
    active_sessions: number;
    total_requests: number;
    models_in_use: Record<string, number>;
}

// Model Types
export interface ModelInfo {
    id: string;
    name: string;
    provider: string;
    description?: string;
    max_tokens?: number;
    supports_streaming?: boolean;
    supports_vision?: boolean;
    is_default?: boolean;
    tier?: string;
}

export interface ProviderWithModels {
    provider: string;
    status: string;
    model: string | null;
    available_models: string[];
    models_detail: ModelInfo[];
}

export interface ModelsListResponse {
    providers: ProviderWithModels[];
    default_provider: string;
    default_model: string;
    total_models: number;
}

export interface ValidateModelRequest {
    provider: string;
    model: string;
}

export interface ValidateModelResponse {
    valid: boolean;
    message: string;
    fallback_model?: string;
    fallback_provider?: string;
}

export interface CurrentModelResponse {
    provider: string;
    model: string;
    session_active: boolean;
}

export interface ModelSyncHealthResponse {
    status: string;
    models_count: number;
    cache_status: string;
    timestamp: string;
    version: string;
}

// Intent-Aware Types
export interface IntentAwareRequest {
    core_request: string;
    technique_suite?: string;
    potency_level?: number;
    apply_all_techniques?: boolean;
    temperature?: number;
    max_new_tokens?: number;
    enable_intent_analysis?: boolean;
    enable_technique_layering?: boolean;
    use_cache?: boolean;
}

export interface IntentAnalysisInfo {
    primary_intent: string;
    secondary_intents: string[];
    key_objectives: string[];
    confidence_score: number;
    reasoning: string;
}

export interface AppliedTechniqueInfo {
    name: string;
    priority: number;
    rationale: string;
}

export interface GenerationMetadataInfo {
    obfuscation_level: number;
    persistence_required: boolean;
    multi_layer_approach: boolean;
    target_model_type: string;
    potency_level: number;
    technique_count: number;
}

export interface IntentAwareResponse {
    success: boolean;
    request_id: string;
    original_input: string;
    expanded_request: string;
    transformed_prompt: string;
    intent_analysis: IntentAnalysisInfo;
    applied_techniques: AppliedTechniqueInfo[];
    metadata: GenerationMetadataInfo;
    execution_time_seconds: number;
    error?: string;
}

export interface AvailableTechniquesInfo {
    techniques: Array<{
        name: string;
        display_name: string;
        category: string;
        transformers: string[];
        framers: string[];
        obfuscators: string[];
        description: string;
    }>;
    total_count: number;
    categories: string[];
}
