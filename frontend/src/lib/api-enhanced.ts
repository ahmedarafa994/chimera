/**
 * @deprecated This file is deprecated and will be removed in a future release.
 *
 * MIGRATION GUIDE:
 * ================
 *
 * This monolithic API client (2400+ lines) has been replaced with modular TanStack Query hooks.
 * The new approach provides:
 * - Automatic caching and deduplication
 * - Background refetching and stale-while-revalidate
 * - Optimistic updates
 * - Better TypeScript support
 * - Smaller bundle size per feature
 *
 * ## How to Migrate:
 *
 * ### For Provider Operations:
 * ```tsx
 * // OLD:
 * import { enhancedApi } from "@/lib/api-enhanced";
 * const { data } = await enhancedApi.providers.list();
 *
 * // NEW:
 * import { useProviders, useSetActiveProvider } from "@/lib/api/query";
 * const { data, isLoading, error } = useProviders();
 * const setActiveProvider = useSetActiveProvider();
 * ```
 *
 * ### For Jailbreak Operations:
 * ```tsx
 * // OLD:
 * const result = await enhancedApi.jailbreak(request);
 *
 * // NEW:
 * import { useJailbreakGenerate, useTechniques } from "@/lib/api/query";
 * const { data: techniques } = useTechniques();
 * const jailbreak = useJailbreakGenerate();
 * await jailbreak.mutateAsync(request);
 * ```
 *
 * ### For AutoDAN-Turbo Operations:
 * ```tsx
 * // OLD:
 * const { data } = await enhancedApi.autodanTurbo.strategies.list();
 *
 * // NEW:
 * import { useStrategies, useAttack, useWarmup } from "@/lib/api/query";
 * const { data, isLoading } = useStrategies();
 * const attack = useAttack();
 * ```
 *
 * ### For Session Management:
 * ```tsx
 * // OLD:
 * const session = await enhancedApi.session.create();
 *
 * // NEW:
 * import { useCurrentSession, useCreateSession } from "@/lib/api/query";
 * const { sessionId, initializeSession } = useCurrentSession();
 * ```
 *
 * ### For System Health/Metrics:
 * ```tsx
 * // OLD:
 * const health = await enhancedApi.health();
 *
 * // NEW:
 * import { useHealth, useMetrics, useIsBackendConnected } from "@/lib/api/query";
 * const { isConnected } = useIsBackendConnected();
 * ```
 *
 * ## New Module Locations:
 * - @/lib/api/query - TanStack Query hooks (recommended for React components)
 * - @/lib/api/core/client - Low-level API client with circuit breaker, retry, caching
 * - @/lib/api/services - Domain-specific service modules
 *
 * @see {@link file://./api/query/index.ts} for all available query hooks
 * @see {@link file://./api/core/client.ts} for the new unified API client
 *
 * ============================================================================
 *
 * Enhanced API client with improved error handling, retry logic, and timeout configuration.
 *
 * This module provides a robust API client that:
 * - Handles backend connection failures gracefully
 * - Provides consistent error mapping
 * - Supports configurable timeouts for long-running operations
 * - Includes health check functionality
 *
 * API Path Convention:
 * - Use "/v1/..." for API v1 endpoints (e.g., "/v1/generate", "/v1/providers")
 * - Use "/..." for root-level endpoints (e.g., "/health", "/integration/stats")
 */

import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from "axios";
import { mapBackendError } from "./errors/index";
import { getApiConfig, getActiveApiUrl, getApiHeaders, saveApiConfig, ApiMode } from "./api-config";

// Import AutoDAN-Turbo types
import type {
  AttackRequest,
  AttackResponse,
  WarmupRequest,
  WarmupResponse,
  LifelongLoopRequest,
  LifelongLoopResponse,
  TestStageRequest,
  TestStageResponse,
  StrategyListResponse,
  StrategySearchRequest,
  StrategySearchResponse,
  StrategyCreateRequest,
  JailbreakStrategy,
  LibraryStatsResponse,
  ProgressResponse,
  HealthResponse,
  ExportedLibrary,
  ImportLibraryResponse,
  DeleteStrategyResponse,
  ResetResponse,
  SaveLibraryResponse,
  ClearLibraryResponse,
} from "./types/autodan-turbo-types";

// Import and re-export types from api-types
import type { MetricsResponse } from "./api-types";
export type { MetricsResponse };

// Provider configuration types (defined inline to avoid circular dependencies)
export interface ProviderInfo {
  provider_id: string;
  name: string;
  provider_type: string;
  is_enabled: boolean;
  is_configured: boolean;
  status: string;
  models_count: number;
}

export interface ProviderHealthStatus {
  provider_id: string;
  status: string;
  latency_ms: number | null;
  last_check: string;
  error_message?: string;
}

export interface ProviderConfigResponse {
  provider_id: string;
  name: string;
  provider_type: string;
  base_url?: string;
  is_enabled: boolean;
  has_api_key: boolean;
  models: Array<{
    model_id: string;
    name: string;
    description?: string;
    context_window?: number;
    max_output_tokens?: number;
    capabilities: string[];
  }>;
}

export interface ProviderListResponse {
  providers: ProviderInfo[];
  count: number;
}

export interface ActiveProviderResponse {
  provider_id: string;
  provider_name: string;
  model: string;
  status: string;
}

export interface SetActiveProviderRequest {
  provider_id: string;
  model_id?: string;
}

export interface UpdateProviderConfigRequest {
  api_key?: string;
  base_url?: string;
  is_enabled?: boolean;
  max_retries?: number;
  timeout?: number;
  custom_headers?: Record<string, string>;
}

export interface ProviderHealthResponse {
  provider_id: string;
  status: string;
  latency_ms: number | null;
  last_check: string;
  error_message?: string;
}

// Re-export getCurrentApiUrl from api-config for backwards compatibility
export { getCurrentApiUrl, getActiveApiUrl, getApiConfig, getApiHeaders } from "./api-config";

// Environment-based configuration
const isDevelopment = process.env.NODE_ENV === "development";

// PERF-001 FIX: Proper timeout configuration to prevent hanging requests
// Timeout hierarchy aligned with backend TimeoutConfig:
// - Standard API calls: 30s
// - LLM provider calls: 2min
// - Long-running operations (AutoDAN, GPTFuzz): 10min
const DEFAULT_TIMEOUT_MS = 30000; // 30s - Standard API calls

// Long-running operations timeout (AutoDAN, GPTFuzz optimization)
export const EXTENDED_TIMEOUT_MS = 600000; // 10min - Long-running operations

// LLM provider timeout
export const LLM_TIMEOUT_MS = 120000; // 2min - LLM provider calls

// Fast operations timeout (health checks, connection tests)
export const FAST_TIMEOUT_MS = 5000; // 5s - Quick operations

// Backend API base URL - uses Next.js API routes as proxy
// The proxy at /api/[...path] forwards to backend origin
// The proxy at /api/v1/[...path] forwards to backend /api/v1
const API_BASE_URL = "/api/v1";

/**
 * Debug logger for API operations
 */
const debugLog = {
  log: (...args: unknown[]) => isDevelopment && console.log("[API-Enhanced]", ...args),
  error: (...args: unknown[]) => console.error("[API-Enhanced]", ...args),
  warn: (...args: unknown[]) => isDevelopment && console.warn("[API-Enhanced]", ...args),
};

/**
 * Create the base axios instance with default configuration
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: DEFAULT_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
  },
  // Ensure we don't transform the response data
  transformResponse: [(data) => {
    try {
      return typeof data === "string" ? JSON.parse(data) : data;
    } catch {
      return data;
    }
  }],
});

/**
 * Request interceptor for logging and adding common headers (including auth)
 */
apiClient.interceptors.request.use(
  (config) => {
    debugLog.log(`Request: ${config.method?.toUpperCase()} ${config.url}`);

    // Add authentication headers from config
    const authHeaders = getApiHeaders();
    config.headers = config.headers || {};
    Object.entries(authHeaders).forEach(([key, value]) => {
      // Don't override Content-Type if already set
      if (key !== "Content-Type" || !config.headers["Content-Type"]) {
        config.headers[key] = value;
      }
    });

    return config;
  },
  (error) => {
    debugLog.error("Request interceptor error:", error);
    return Promise.reject(error);
  }
);

/**
 * Response interceptor for logging and error transformation
 */
apiClient.interceptors.response.use(
  (response) => {
    debugLog.log(`Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error: AxiosError) => {
    debugLog.error(`Response error: ${error.message}`);

    // Transform to APIError for consistent handling
    const apiError = mapBackendError(error);
    return Promise.reject(apiError);
  }
);

/**
 * Check if the backend is available
 * @returns Promise<boolean> - true if backend is reachable
 */
export async function checkBackendConnection(): Promise<boolean> {
  try {
    const response = await apiClient.get("/health", {
      timeout: FAST_TIMEOUT_MS, // PERF-001 FIX: Use 5s timeout for health checks
      baseURL: "/api", // Health is at /api/health, not /api/v1/health
      // Don't throw on non-2xx status
      validateStatus: (status) => status < 500,
    });
    return response.status === 200;
  } catch (error) {
    debugLog.warn("Backend connection check failed:", error);
    return false;
  }
}

/**
 * Session API response types
 */
interface SessionResponse {
  data: {
    success: boolean;
    session_id: string;
    provider?: string;
    model?: string;
  };
}

interface SessionGetResponse {
  data: SessionInfoResponse | null;
}

/**
 * Model information type
 */
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

/**
 * Provider type - represents an AI provider with its available models
 */
export interface Provider {
  provider: string;
  status: string;
  model?: string;
  available_models: string[];
  models_detail?: ModelInfo[];
}

/**
 * Provider with models type (alias for Provider)
 */
export type ProviderWithModels = Provider;

/**
 * Providers list response type
 */
export interface ProvidersListResponse {
  providers: Provider[];
  default: string;
  count: number;
}

/**
 * Models list response type
 */
export interface ModelsListResponse {
  providers: ProviderWithModels[];
  default_provider: string;
  default_model: string;
  total_models: number;
}

/**
 * Model sync health response type
 */
export interface ModelSyncHealthResponse {
  status: "healthy" | "unhealthy";
  models_count: number;
  cache_status: string;
  timestamp: string;
  version: string;
  error?: string;
}

/**
 * Session stats response type
 */
export interface SessionStatsResponse {
  active_sessions: number;
  total_requests: number;
  models_in_use: Record<string, number>;
}

/**
 * Generate response type
 */
export interface GenerateResponse {
  success: boolean;
  result?: string;
  error?: string;
  text?: string;
  model_used?: string;
  provider?: string;
  latency_ms?: number;
  finish_reason?: string;
  usage_metadata?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
    prompt_token_count?: number;
    candidates_token_count?: number;
    total_token_count?: number;
  };
}

/**
 * Generate request type
 */
export interface GenerateRequest {
  prompt: string;
  model?: string;
  provider?: string;
  system_instruction?: string;
  config?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    max_output_tokens?: number;
    stop_sequences?: string[];
    thinking_level?: "low" | "medium" | "high";
  };
  temperature?: number;
  max_tokens?: number;
}

/**
 * Jailbreak request type
 */
export interface JailbreakRequest {
  prompt?: string;
  core_request?: string;
  technique?: string;
  technique_suite?: string;
  potency_level?: number;
  provider?: string;
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_new_tokens?: number;
  density?: number;
  // Content Transformation
  use_leet_speak?: boolean;
  leet_speak_density?: number;
  use_homoglyphs?: boolean;
  homoglyph_density?: number;
  use_caesar_cipher?: boolean;
  caesar_shift?: number;
  // Structural & Semantic
  use_role_hijacking?: boolean;
  use_instruction_injection?: boolean;
  use_adversarial_suffixes?: boolean;
  use_few_shot_prompting?: boolean;
  use_character_role_swap?: boolean;
  // Advanced Neural
  use_neural_bypass?: boolean;
  use_meta_prompting?: boolean;
  use_counterfactual_prompting?: boolean;
  use_contextual_override?: boolean;
  // Research-Driven
  use_multilingual_trojan?: boolean;
  multilingual_target_language?: string;
  use_payload_splitting?: boolean;
  payload_splitting_parts?: number;
  use_contextual_interaction_attack?: boolean;
  cia_preliminary_rounds?: number;
  // AI Generation Options
  use_ai_generation?: boolean;
  is_thinking_mode?: boolean;
  use_cache?: boolean;
}

/**
 * Jailbreak response metadata type
 */
export interface JailbreakMetadata {
  technique_suite?: string;
  potency_level?: number;
  applied_techniques?: string[];
  techniques_used?: string[];  // Alias for applied_techniques used by some components
  layers_applied?: string[] | number;
  ai_generation_enabled?: boolean;
  thinking_mode?: boolean;
  provider?: string;
  model?: string;
  temperature?: number;
  [key: string]: unknown;
}

/**
 * Jailbreak response type
 */
export interface JailbreakResponse {
  success: boolean;
  result?: string;
  error?: string;
  jailbreak_prompt?: string;
  transformed_prompt?: string;
  technique?: string;
  provider?: string;
  model_used?: string;
  latency_ms?: number;
  usage_metadata?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  execution_time_seconds?: number;
  request_id?: string;
  metadata?: JailbreakMetadata;
}

/**
 * AutoDAN request type
 */
export interface AutoDANRequest {
  prompt?: string;
  request?: string;
  target_model?: string;
  method?: string;
  provider?: string;
  max_attempts?: number;
  technique?: string;
}

/**
 * AutoDAN response type
 */
export interface AutoDANResponse {
  success: boolean;
  result?: string;
  error?: string;
  jailbreak_prompt?: string;
  method?: string;
  status?: string;
  prompt?: string;
  response?: string;
  score?: number;
  is_jailbreak?: boolean;
  strategy_id?: string;
  strategy_extracted?: boolean;
}

/**
 * Extended AutoDAN response type
 */
export interface ExtendedAutoDANResponse extends AutoDANResponse {
  jailbreak_prompt: string;
  method: string;
  status: string;
}

/**
 * Execute request type
 */
export interface ExecuteRequest {
  prompt?: string;
  core_request?: string;
  transformation?: string;
  provider?: string;
  model?: string;
  potency_level?: number;
  technique_suite?: string;
}

/**
 * Execute response type
 */
export interface ExecuteResponse {
  success: boolean;
  result?: string | {
    provider?: string;
    model?: string;
    latency_ms?: number;
    text?: string;
  };
  error?: string;
  transformation?: string;
  provider?: string;
  model?: string;
  latency_ms?: number;
}

/**
 * Fuzz request type
 */
export interface FuzzRequest {
  target_prompt?: string;
  target_model?: string;
  questions?: string[];
  seeds?: string[];
  num_attempts?: number;
  techniques?: string[];
  provider?: string;
  model?: string;
  max_queries?: number;
  max_jailbreaks?: number;
}

/**
 * Fuzz session type
 */
export interface FuzzSession {
  session_id: string;
  status: string;
  created_at: string;
  results?: FuzzResult[];
  error?: string;
  message?: string;
  stats?: {
    jailbreaks: number;
    total_queries: number;
  };
}

/**
 * Fuzz result type
 */
export interface FuzzResult {
  attempt_number: number;
  technique: string;
  prompt: string;
  response: string;
  success: boolean;
  score?: number;
  question?: string;
  template?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Gradient optimization request type
 */
export interface GradientOptimizationRequest {
  core_request: string;
  technique: "hotflip" | "gcg";
  potency_level: number;
  num_steps?: number;
  beam_width?: number;
  target_model?: string;
  provider?: string;
  model?: string;
  use_cache?: boolean;
}

/**
 * Gradient optimization response type
 */
export interface GradientOptimizationResponse {
  success: boolean;
  request_id: string;
  optimized_prompt: string;
  metadata: {
    technique: string;
    potency_level: number;
    num_steps: number;
    beam_width?: number;
    provider?: string;
    model?: string;
    used_fallback?: boolean;
  };
  execution_time_seconds: number;
}

/**
 * HouYi request type
 */
export interface HouYiRequest {
  intention: string;
  question_prompt: string;
  target_provider?: string;
  target_model?: string;
  application_document?: string;
  iteration?: number;
  population?: number;
}

/**
 * HouYi response type
 */
export interface HouYiResponse {
  success: boolean;
  best_prompt: string;
  fitness_score: number;
  llm_response: string;
  details?: {
    framework?: string;
    separator?: string;
    disruptor?: string;
    question_prompt?: string;
    [key: string]: unknown;
  };
}

/**
 * Intent-aware request type
 */
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

/**
 * Intent-aware response type
 */
export interface IntentAwareResponse {
  success: boolean;
  request_id: string;
  original_input: string;
  expanded_request: string;
  transformed_prompt: string;
  intent_analysis: IntentAnalysisInfo;
  applied_techniques: AppliedTechniqueInfo[];
  metadata: GenerationMetadata;
  execution_time_seconds: number;
  error?: string;
}

/**
 * Applied technique info type
 */
export interface AppliedTechniqueInfo {
  name: string;
  priority: number;
  rationale: string;
}

/**
 * Generation metadata type
 */
export interface GenerationMetadata {
  obfuscation_level: number;
  persistence_required: boolean;
  multi_layer_approach: boolean;
  target_model_type: string;
  potency_level: number;
  technique_count: number;
}

/**
 * Intent analysis info type
 */
export interface IntentAnalysisInfo {
  primary_intent: string;
  secondary_intents: string[];
  key_objectives: string[];
  confidence_score: number;
  reasoning: string;
}

/**
 * Hierarchical search request type
 */
export interface HierarchicalSearchRequest {
  query?: string;
  request?: string;
  search_depth?: number;
  max_results?: number;
  population_size?: number;
  generations?: number;
  mutation_rate?: number;
  crossover_rate?: number;
}

/**
 * Hierarchical search response type
 */
export interface HierarchicalSearchResponse {
  success: boolean;
  results?: string[];
  error?: string;
  best_score?: number;
  execution_time_ms?: number;
  best_prompt?: string;
  generation_history?: Array<{
    generation: number;
    best_score: number;
    avg_score: number;
  }>;
  search_metadata?: {
    depth_reached: number;
    total_nodes: number;
    execution_time: number;
  };
}

/**
 * Transform request type
 */
export interface TransformRequest {
  prompt: string;
  transformation_type?: string;
  provider?: string;
  model?: string;
}

/**
 * Transform response type
 */
export interface TransformResponse {
  success: boolean;
  result?: string;
  error?: string;
  transformed_prompt?: string;
  transformation_applied?: string;
}

/**
 * Connection status response type
 */
export interface ConnectionStatusResponse {
  status: "connected" | "disconnected" | "error";
  is_connected: boolean;
  base_url: string;
  latency_ms?: number;
  timestamp: string;
  error_message?: string;
  available_models?: string[];
  message?: string;
}

/**
 * Connection config response type
 */
export interface ConnectionConfigResponse {
  current_mode: string;
  direct: {
    url: string;
    api_key_configured: boolean;
    api_key_preview?: string | null;
  };
  providers?: Record<string, string>;
}

/**
 * Connection test response type
 */
export interface ConnectionTestResponse {
  success: boolean;
  message: string;
  provider?: string;
  model?: string;
  latency_ms?: number;
  timestamp: string;
  direct?: ConnectionStatusResponse;
  recommended?: string;
}

/**
 * Session info response type
 */
export interface SessionInfoResponse {
  session_id: string;
  created_at: string;
  last_activity: string;
  provider?: string;
  model?: string;
  request_count: number;
}

/**
 * Create session response type
 */
export interface CreateSessionResponse {
  success: boolean;
  session_id: string;
  provider?: string;
  model?: string;
  message?: string;
}

/**
 * Providers API methods
 */
const providersApi = {
  /**
   * List all available providers with their models
   */
  async list(): Promise<{ data: ProvidersListResponse }> {
    try {
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/providers`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to list providers: ${response.statusText}`);
      }

      const data = await response.json();

      // Map backend response fields to frontend interface if needed
      if (data.providers && Array.isArray(data.providers)) {
        data.providers = data.providers.map((p: Record<string, unknown>) => ({
          ...p,
          // Map backend fields to frontend interface
          provider_id: p.provider || p.provider_id,
          name: p.display_name || p.name || p.provider,
          // Add missing fields expected by frontend
          provider_type: p.provider_type || "chat",
          is_enabled: p.is_enabled !== undefined ? p.is_enabled : true,
          is_configured: p.is_configured !== undefined ? p.is_configured : true,
          status: p.status || (p.is_healthy ? "ready" : "error"),
          models_count: Array.isArray(p.models) ? (p.models as unknown[]).length : ((p.models_count as number) || 0)
        }));
      }

      return { data };
    } catch (error) {
      debugLog.error("Failed to list providers:", error);
      // Return empty response on error
      return {
        data: {
          providers: [],
          default: "google",
          count: 0,
        },
      };
    }
  },
};

/**
 * Provider Configuration API methods - for dynamic provider management
 */
const providerConfigApi = {
  /**
   * List all registered providers with their configuration status
   */
  async listProviders(): Promise<{ data: ProviderListResponse }> {
    try {
      const baseUrl = "/api/v1";
      const response = await fetch(`${baseUrl}/provider-config/providers`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to list providers: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to list provider configs:", error);
      return {
        data: {
          providers: [],
          count: 0,
        },
      };
    }
  },

  /**
   * Get the currently active provider
   */
  async getActiveProvider(): Promise<{ data: ActiveProviderResponse }> {
    try {
      const baseUrl = "/api/v1";
      const response = await fetch(`${baseUrl}/provider-config/active`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get active provider: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get active provider:", error);
      return {
        data: {
          provider_id: "openai",
          provider_name: "OpenAI",
          model: "gpt-4",
          status: "unknown",
        },
      };
    }
  },

  /**
   * Set the active provider
   */
  async setActiveProvider(request: SetActiveProviderRequest): Promise<{ data: ActiveProviderResponse }> {
    const baseUrl = "/api/v1";
    const response = await fetch(`${baseUrl}/provider-config/active`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to set active provider: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Get configuration for a specific provider
   */
  async getProviderConfig(providerId: string): Promise<{ data: ProviderConfigResponse }> {
    try {
      const baseUrl = "/api/v1";
      const response = await fetch(`${baseUrl}/provider-config/providers/${encodeURIComponent(providerId)}`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get provider config: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error(`Failed to get config for provider ${providerId}:`, error);
      throw error;
    }
  },

  /**
   * Update configuration for a specific provider
   */
  async updateProviderConfig(providerId: string, config: UpdateProviderConfigRequest): Promise<{ data: ProviderConfigResponse }> {
    const baseUrl = "/api/v1";
    const response = await fetch(`${baseUrl}/provider-config/providers/${encodeURIComponent(providerId)}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to update provider config: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Get health status for a specific provider
   */
  async getProviderHealth(providerId: string): Promise<{ data: ProviderHealthResponse }> {
    try {
      const baseUrl = "/api/v1";
      const response = await fetch(`${baseUrl}/provider-config/providers/${encodeURIComponent(providerId)}/health`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get provider health: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error(`Failed to get health for provider ${providerId}:`, error);
      return {
        data: {
          provider_id: providerId,
          status: "unknown",
          latency_ms: null,
          last_check: new Date().toISOString(),
          error_message: error instanceof Error ? error.message : "Health check failed",
        },
      };
    }
  },

  /**
   * Get health status for all providers
   */
  async getAllProvidersHealth(): Promise<{ data: { providers: ProviderHealthResponse[] } }> {
    try {
      const baseUrl = "/api/v1";
      const response = await fetch(`${baseUrl}/provider-config/health`, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get providers health: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get all providers health:", error);
      return {
        data: {
          providers: [],
        },
      };
    }
  },

  /**
   * Test provider connection with optional API key
   */
  async testProviderConnection(providerId: string, apiKey?: string): Promise<{ data: { success: boolean; message: string; latency_ms?: number } }> {
    const baseUrl = "/api/v1";
    const response = await fetch(`${baseUrl}/provider-config/providers/${encodeURIComponent(providerId)}/test`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ api_key: apiKey }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return {
        data: {
          success: false,
          message: errorData.detail || `Connection test failed: ${response.statusText}`,
        },
      };
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Get WebSocket URL for real-time provider status updates
   */
  getWebSocketUrl(): string {
    const wsProtocol = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = typeof window !== "undefined" ? window.location.host : "localhost:3000";
    return `${wsProtocol}//${host}/api/v1/provider-config/ws/updates`;
  },
};

/**
 * Models API methods
 */
const modelsApi = {
  /**
   * List all available models across all providers
   */
  async list(): Promise<{ data: ModelsListResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/models`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to list models: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to list models:", error);
      // Return empty response on error
      return {
        data: {
          providers: [],
          default_provider: "google",
          default_model: "gemini-2.0-flash-exp",
          total_models: 0,
        },
      };
    }
  },

  /**
   * Get model sync service health status
   */
  async health(): Promise<{ data: ModelSyncHealthResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/models/health`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get models health: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get models health:", error);
      // Return unhealthy response on error
      return {
        data: {
          status: "unhealthy",
          models_count: 0,
          cache_status: "unknown",
          timestamp: new Date().toISOString(),
          version: "unknown",
          error: error instanceof Error ? error.message : "Unknown error",
        },
      };
    }
  },

  /**
   * Validate a model selection
   */
  async validate(provider: string, model: string): Promise<{ data: { valid: boolean; message: string; fallback_model?: string; fallback_provider?: string } }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/models/validate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
        body: JSON.stringify({ provider, model }),
      });

      if (!response.ok) {
        throw new Error(`Failed to validate model: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to validate model:", error);
      return {
        data: {
          valid: false,
          message: error instanceof Error ? error.message : "Validation failed",
          fallback_model: "deepseek-chat",
          fallback_provider: "deepseek",
        },
      };
    }
  },
};

/**
 * Session API methods
 */
const sessionApi = {
  /**
   * Get an existing session by ID
   */
  async get(sessionId: string): Promise<SessionGetResponse> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/session/${sessionId}`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        return { data: null };
      }

      const data = await response.json();
      return { data };
    } catch {
      return { data: null };
    }
  },

  /**
   * Create a new session
   */
  async create(options?: { provider?: string; model?: string }): Promise<SessionResponse> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/session`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
      body: JSON.stringify(options || {}),
    });

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      data: {
        success: true,
        session_id: data.session_id || data.id,
        provider: data.provider,
        model: data.model,
      },
    };
  },

  /**
   * Get session statistics (admin endpoint)
   */
  async getStats(): Promise<{ data: SessionStatsResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/session/stats`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get session stats: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get session stats:", error);
      // Return empty stats on error
      return {
        data: {
          active_sessions: 0,
          total_requests: 0,
          models_in_use: {},
        },
      };
    }
  },

  /**
   * Update model for session (for compatibility)
   */
  async updateModel(
    sessionId: string,
    model: string,
    provider?: string
  ): Promise<{
    data: {
      success: boolean;
      message?: string;
      reverted_to_default?: boolean;
      provider?: string;
      model?: string;
    };
  }> {
    try {
      const baseUrl = "/api/v1";
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      if (sessionId) {
        headers["X-Session-ID"] = sessionId;
      }

      const response = await fetch(`${baseUrl}/session/model`, {
        method: "PUT",
        headers,
        body: JSON.stringify({ provider, model }),
      });

      if (!response.ok) {
        let detail = `Failed to update model: ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (typeof errorData?.detail === "string") {
            detail = errorData.detail;
          }
        } catch {
          // Keep default detail message.
        }
        return { data: { success: false, message: detail } };
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to update model:", error);
      return {
        data: {
          success: false,
          message: error instanceof Error ? error.message : "Failed to update model",
        },
      };
    }
  },
};

/**
 * Utils API methods
 */
const utilsApi = {
  /**
   * Get the current API mode (direct)
   */
  getCurrentMode(): ApiMode {
    return getApiConfig().mode;
  },

  /**
   * Get the current API URL based on mode
   */
  getCurrentUrl(): string {
    return getActiveApiUrl();
  },

  /**
   * Update the API configuration mode
   */
  updateConfig(mode: ApiMode): void {
    saveApiConfig({ mode });
  },

  /**
   * Check if the backend is connected
   */
  async checkConnection(): Promise<boolean> {
    return checkBackendConnection();
  },
};

// ============================================================================
// AutoDAN-Turbo API
// ============================================================================

/**
 * AutoDAN-Turbo Strategies API methods
 */
const autodanTurboStrategiesApi = {
  /**
   * List all strategies in the library
   */
  async list(offset = 0, limit = 100): Promise<{ data: StrategyListResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/autodan-turbo/strategies?offset=${offset}&limit=${limit}`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to list strategies: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to list strategies:", error);
      return {
        data: {
          strategies: [],
          total_count: 0,
        },
      };
    }
  },

  /**
   * Search strategies by query
   */
  async search(request: StrategySearchRequest): Promise<{ data: StrategySearchResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/autodan-turbo/strategies/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Failed to search strategies: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to search strategies:", error);
      return {
        data: {
          query: request.query,
          results: [],
        },
      };
    }
  },

  /**
   * Create a new strategy
   */
  async create(request: StrategyCreateRequest): Promise<{ data: JailbreakStrategy }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/strategies`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to create strategy: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Delete a strategy
   */
  async delete(strategyId: string): Promise<{ data: DeleteStrategyResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/strategies/${encodeURIComponent(strategyId)}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to delete strategy: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },
};

/**
 * AutoDAN-Turbo Library API methods
 */
const autodanTurboLibraryApi = {
  /**
   * Get library statistics
   */
  async stats(): Promise<{ data: LibraryStatsResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/autodan-turbo/library/stats`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get library stats: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get library stats:", error);
      return {
        data: {
          total_strategies: 0,
          strategies_by_source: {},
          top_strategies_by_success_rate: [],
          top_strategies_by_usage: [],
          average_success_rate: 0,
          average_score: 0,
          discovered_count: 0,
          human_designed_count: 0,
          avg_score_differential: 0,
        },
      };
    }
  },

  /**
   * Export the strategy library
   */
  async export(): Promise<{ data: ExportedLibrary }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/library/export`, {
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to export library: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Import a strategy library
   */
  async import(library: ExportedLibrary, merge = true): Promise<{ data: ImportLibraryResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/library/import`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
      body: JSON.stringify({ data: library, merge }),
    });

    if (!response.ok) {
      throw new Error(`Failed to import library: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Save the strategy library to disk
   */
  async save(): Promise<{ data: SaveLibraryResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/library/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to save library: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Clear the strategy library
   */
  async clear(): Promise<{ data: ClearLibraryResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/library/clear`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to clear library: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },
};

/**
 * AutoDAN-Turbo Progress API methods
 */
const autodanTurboProgressApi = {
  /**
   * Get current progress
   */
  async get(): Promise<{ data: ProgressResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/autodan-turbo/progress`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get progress: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get progress:", error);
      return {
        data: {
          phase: "idle",
          total_attacks: 0,
          successful_attacks: 0,
          strategies_discovered: 0,
          average_score: 0,
          best_score: 0,
          current_request: 0,
          total_requests: 0,
          success_rate: 0,
          status: "idle",
        },
      };
    }
  },
};

/**
 * AutoDAN-Turbo Utils API methods
 */
const autodanTurboUtilsApi = {
  /**
   * Check engine health
   */
  async health(): Promise<{ data: HealthResponse }> {
    try {
      const config = getApiConfig();
      const baseUrl = "/api/v1";

      const response = await fetch(`${baseUrl}/autodan-turbo/health`, {
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to get health: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      debugLog.error("Failed to get health:", error);
      return {
        data: {
          status: "unhealthy",
          initialized: false,
          library_size: 0,
          progress: null,
          error: error instanceof Error ? error.message : "Unknown error",
        },
      };
    }
  },

  /**
   * Reset the engine
   */
  async reset(): Promise<{ data: ResetResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/reset`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to reset engine: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },
};

/**
 * AutoDAN-Turbo API namespace
 */
const autodanTurboApi = {
  /**
   * Strategy management
   */
  strategies: autodanTurboStrategiesApi,

  /**
   * Library management
   */
  library: autodanTurboLibraryApi,

  /**
   * Progress tracking
   */
  progress: autodanTurboProgressApi,

  /**
   * Utility functions
   */
  utils: autodanTurboUtilsApi,

  /**
   * Execute a single attack
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async attack(request: any): Promise<{ data: AttackResponse }> {
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/attack`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to execute attack: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },

  /**
   * Run warm-up exploration phase
   */
  async warmup(request: WarmupRequest): Promise<{ data: WarmupResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    // Warmup can take several minutes with many LLM calls
    // Set a 10-minute timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes

    try {
      const response = await fetch(`${baseUrl}/autodan-turbo/warmup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Failed to run warmup: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Warmup request timed out after 10 minutes');
      }
      throw error;
    }
  },

  /**
   * Run lifelong learning loop
   */
  async lifelong(request: LifelongLoopRequest): Promise<{ data: LifelongLoopResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    // Lifelong learning can take several minutes with many LLM calls
    // Set a 10-minute timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes

    try {
      const response = await fetch(`${baseUrl}/autodan-turbo/lifelong`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...({}),
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Failed to run lifelong learning: ${response.statusText}`);
      }

      const data = await response.json();
      return { data };
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Lifelong learning request timed out after 10 minutes');
      }
      throw error;
    }
  },

  /**
   * Run test stage
   */
  async test(request: TestStageRequest): Promise<{ data: TestStageResponse }> {
    const config = getApiConfig();
    const baseUrl = "/api/v1";

    const response = await fetch(`${baseUrl}/autodan-turbo/test`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...({}),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to run test stage: ${response.statusText}`);
    }

    const data = await response.json();
    return { data };
  },
};

/**
 * Enhanced API wrapper with automatic error handling
 */
export const enhancedApi = {
  /**
   * GET request with error handling
   */
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await apiClient.get<T>(url, config);
    return response.data;
  },

  /**
   * POST request with error handling
   */
  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await apiClient.post<T>(url, data, config);
    return response.data;
  },

  /**
   * PUT request with error handling
   */
  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await apiClient.put<T>(url, data, config);
    return response.data;
  },

  /**
   * DELETE request with error handling
   */
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await apiClient.delete<T>(url, config);
    return response.data;
  },

  /**
   * PATCH request with error handling
   */
  async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await apiClient.patch<T>(url, data, config);
    return response.data;
  },

  /**
   * Utility functions for API configuration and connection management
   */
  utils: utilsApi,

  /**
   * Providers API - for listing available AI providers
   */
  providers: providersApi,

  /**
   * Provider Configuration API - for dynamic provider management and switching
   */
  providerConfig: providerConfigApi,

  /**
   * Models API - for listing and managing available models
   */
  models: modelsApi,

  /**
   * Session management API
   */
  session: sessionApi,

  /**
   * AutoDAN-Turbo API - for lifelong learning jailbreak attacks
   */
  autodanTurbo: autodanTurboApi,

  /**
   * Generate API - for text generation
   */
  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    try {
      const response = await apiClient.post<GenerateResponse>("/generate", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Generation failed"
      };
    }
  },

  /**
   * Text generation API (alias for generate)
   */
  async text(request: GenerateRequest): Promise<GenerateResponse> {
    return this.generate(request);
  },

  /**
   * Jailbreak API - for jailbreak prompt generation
   */
  async jailbreak(request: JailbreakRequest): Promise<JailbreakResponse> {
    try {
      const response = await apiClient.post<JailbreakResponse>("/jailbreak", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Jailbreak failed",
        jailbreak_prompt: "",
        technique: "",
        provider: request.provider || "",
        model_used: request.model || "",
        metadata: {},
        request_id: ""
      };
    }
  },

  /**
   * Jailbreak generate API (alias for jailbreak)
   */
  async jailbreakGenerate(request: JailbreakRequest): Promise<JailbreakResponse> {
    return this.jailbreak(request);
  },

  /**
   * AutoDAN API - for AutoDAN attacks
   */
  async autodan(request: AutoDANRequest): Promise<AutoDANResponse> {
    try {
      const response = await apiClient.post<AutoDANResponse>("/autodan", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "AutoDAN failed",
        jailbreak_prompt: "",
        method: "",
        status: "error"
      };
    }
  },

  /**
   * AutoDAN jailbreak API (alias for autodan)
   */
  async autodanJailbreak(request: AutoDANRequest): Promise<AutoDANResponse> {
    return this.autodan(request);
  },

  /**
   * AutoDAN hierarchical search API
   */
  async hierarchicalSearch(request: HierarchicalSearchRequest): Promise<HierarchicalSearchResponse> {
    try {
      const response = await apiClient.post<HierarchicalSearchResponse>("/autodan/hierarchical", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Hierarchical search failed",
        results: [],
        search_metadata: {
          depth_reached: 0,
          total_nodes: 0,
          execution_time: 0
        }
      };
    }
  },

  /**
   * AutoDAN optimize API (alias for autodan)
   */
  async autodanOptimize(request: AutoDANRequest): Promise<AutoDANResponse> {
    return this.autodan(request);
  },

  /**
   * Execute API - for prompt execution
   */
  async execute(request: ExecuteRequest): Promise<ExecuteResponse> {
    try {
      const response = await apiClient.post<ExecuteResponse>("/execute", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Execution failed"
      };
    }
  },

  /**
   * Run API (alias for execute)
   */
  async run(request: ExecuteRequest): Promise<ExecuteResponse> {
    return this.execute(request);
  },

  /**
   * Transform API - for prompt transformation
   */
  async transform(request: TransformRequest): Promise<TransformResponse> {
    try {
      const response = await apiClient.post<TransformResponse>("/transform", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Transformation failed",
        transformed_prompt: "",
        transformation_applied: ""
      };
    }
  },

  /**
   * Execute transform API (alias for transform)
   */
  async executeTransform(request: TransformRequest): Promise<TransformResponse> {
    return this.transform(request);
  },

  /**
   * Intent-aware API - for intent-aware generation
   */
  async intentAware(request: IntentAwareRequest): Promise<IntentAwareResponse> {
    try {
      const response = await apiClient.post<IntentAwareResponse>("/intent-aware/generate", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        request_id: "",
        original_input: request.core_request || "",
        expanded_request: "",
        transformed_prompt: "",
        intent_analysis: {
          primary_intent: "",
          secondary_intents: [],
          key_objectives: [],
          confidence_score: 0,
          reasoning: ""
        },
        applied_techniques: [],
        metadata: {
          obfuscation_level: 0,
          persistence_required: false,
          multi_layer_approach: false,
          target_model_type: "",
          potency_level: 0,
          technique_count: 0
        },
        execution_time_seconds: 0,
        error: error instanceof Error ? error.message : "Intent-aware generation failed"
      };
    }
  },

  /**
   * Intent-aware generate API (alias for intentAware)
   */
  async intentAwareGenerate(request: IntentAwareRequest): Promise<IntentAwareResponse> {
    return this.intentAware(request);
  },

  /**
   * Intent-aware analyze API (alias for intentAware)
   */
  async analyzeIntent(request: IntentAwareRequest): Promise<IntentAwareResponse> {
    return this.intentAware(request);
  },

  /**
   * GPTFuzz API - for fuzzing
   */
  async gptfuzz(request: FuzzRequest): Promise<{ data: FuzzSession }> {
    try {
      const response = await apiClient.post<{ data: FuzzSession }>("/gptfuzz", request);
      return response.data;
    } catch (error) {
      return {
        data: {
          session_id: "",
          status: "error",
          created_at: new Date().toISOString(),
          results: []
        }
      };
    }
  },

  /**
   * GPTFuzz run API (alias for gptfuzz)
   */
  async gptfuzzRun(request: FuzzRequest): Promise<{ data: FuzzSession }> {
    return this.gptfuzz(request);
  },

  /**
   * GPTFuzz status API
   */
  async gptfuzzStatus(sessionIdOrRequest: string | FuzzRequest): Promise<{ data: FuzzSession }> {
    if (typeof sessionIdOrRequest === 'string') {
      // If it's a session ID string, fetch status by session ID
      const response = await apiClient.get<FuzzSession>(`/gptfuzz/status/${sessionIdOrRequest}`);
      return { data: response.data };
    }
    return this.gptfuzz(sessionIdOrRequest);
  },

  /**
   * Gradient API - for gradient optimization
   */
  async gradient(request: GradientOptimizationRequest): Promise<GradientOptimizationResponse> {
    try {
      const response = await apiClient.post<GradientOptimizationResponse>("/gradient/optimize", request);
      return response.data;
    } catch (error) {
      return {
        success: false,
        request_id: "",
        optimized_prompt: "",
        metadata: {
          technique: "",
          potency_level: 0,
          num_steps: 0
        },
        execution_time_seconds: 0
      };
    }
  },

  /**
   * HouYi API - for HouYi optimization
   */
  optimize: {
    async houyi(request: HouYiRequest): Promise<{ data: HouYiResponse }> {
      try {
        const response = await apiClient.post<HouYiResponse>("/optimize/houyi", request);
        return { data: response.data };
      } catch (error) {
        return {
          data: {
            success: false,
            best_prompt: "",
            fitness_score: 0,
            llm_response: error instanceof Error ? error.message : "HouYi optimization failed",
          }
        };
      }
    }
  },

  /**
   * Health API - for system health checks
   */
  async health(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await apiClient.get<{ status: string; timestamp: string }>("/health");
      return response.data;
    } catch (error) {
      return {
        status: "unhealthy",
        timestamp: new Date().toISOString()
      };
    }
  },

  /**
   * Health check API (alias for health)
   */
  async check(): Promise<{ status: string; timestamp: string }> {
    return this.health();
  },

  /**
   * Health get API (alias for health)
   */
  async getHealth(): Promise<{ status: string; timestamp: string }> {
    return this.health();
  },

  /**
   * Techniques API - for listing available techniques
   */
  async techniques(): Promise<{ techniques: string[] }> {
    try {
      const response = await apiClient.get<{ techniques?: unknown[] }>("/techniques");
      const rawTechniques = Array.isArray(response.data?.techniques)
        ? response.data.techniques
        : [];
      const techniques = rawTechniques
        .map((tech) => {
          if (typeof tech === "string") {
            return tech;
          }
          if (tech && typeof tech === "object" && "name" in tech) {
            const name = (tech as { name?: unknown }).name;
            return typeof name === "string" ? name : "";
          }
          return "";
        })
        .filter((tech) => tech.length > 0);

      return { techniques };
    } catch (error) {
      return {
        techniques: []
      };
    }
  },

  /**
   * Techniques list API (alias for techniques)
   */
  async list(): Promise<{ techniques: string[] }> {
    return this.techniques();
  },

  /**
   * Metrics API - for system metrics
   */
  async metrics(): Promise<MetricsResponse> {
    try {
      const response = await apiClient.get<MetricsResponse>("/metrics");
      return response.data;
    } catch (error) {
      return {
        timestamp: new Date().toISOString(),
        metrics: {
          status: "unknown",
          cache: { enabled: false, entries: 0 },
          providers: {}
        }
      };
    }
  },

  /**
   * Metrics get API (alias for metrics)
   */
  async getMetrics(): Promise<MetricsResponse> {
    return this.metrics();
  },

  /**
   * Metrics get all API (alias for metrics)
   */
  async getAll(): Promise<{ metrics: Record<string, unknown> }> {
    return this.metrics();
  },

  /**
   * Connection API - for connection management
   */
  connection: {
    async test(config?: { provider?: string; model?: string }): Promise<{ data: ConnectionTestResponse }> {
      try {
        const response = await apiClient.post<ConnectionTestResponse>("/connection/test", config || {});
        return { data: response.data };
      } catch (error) {
        return {
          data: {
            success: false,
            message: error instanceof Error ? error.message : "Connection test failed",
            timestamp: new Date().toISOString()
          }
        };
      }
    },

    async status(): Promise<ConnectionStatusResponse> {
      try {
        const response = await apiClient.get<ConnectionStatusResponse>("/connection/status");
        return response.data;
      } catch (error) {
        return {
          status: "error",
          is_connected: false,
          base_url: "",
          message: error instanceof Error ? error.message : "Connection check failed",
          timestamp: new Date().toISOString()
        };
      }
    },

    async getConfig(): Promise<{ data: ConnectionConfigResponse }> {
      // Mock config response for compatibility
      return {
        data: {
          current_mode: "direct",
          direct: {
            url: "https://generativelanguage.googleapis.com",
            api_key_configured: true
          }
        }
      };
    },

    async getStatus(): Promise<{ data: ConnectionStatusResponse }> {
      try {
        const status = await this.status();
        return { data: status };
      } catch (error) {
        return {
          data: {
            status: "error",
            is_connected: false,
            base_url: "",
            message: "Failed to get status",
            timestamp: new Date().toISOString()
          }
        };
      }
    },

    async setMode(request: { mode: string }): Promise<{ data: { message: string } }> {
      // Mock set mode response
      return {
        data: {
          message: `Mode set to ${request.mode}`
        }
      };
    }
  }
};

export default enhancedApi;

