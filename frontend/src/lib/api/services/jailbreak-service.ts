/**
 * Jailbreak Service - Aligned with Backend API
 *
 * This service is properly aligned with backend endpoints:
 * - POST /api/v1/jailbreak (basic jailbreak)
 * - POST /api/v1/generation/jailbreak/generate (AI-powered jailbreak)
 * - POST /api/v1/autodan/* (AutoDAN endpoints)
 * - POST /api/v1/gptfuzz/* (GPTFuzz endpoints)
 * - POST /api/v1/gradient/* (Gradient optimization)
 * - GET /api/v1/techniques (available techniques)
 */

import { apiClient } from '../client';
import { apiErrorHandler } from '../../errors/api-error-handler';

// ============================================================================
// Types (matching backend Pydantic models exactly)
// ============================================================================

export interface JailbreakRequest {
  prompt?: string;
  core_request?: string;
  technique_suite?: string;
  potency_level?: number;
  provider?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  use_ai_generation?: boolean;
  is_thinking_mode?: boolean;
  use_cache?: boolean;

  // Advanced options
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
}

export interface JailbreakMetadata {
  technique_suite?: string;
  potency_level?: number;
  applied_techniques?: string[];
  techniques_used?: string[];
  layers_applied?: string[] | number;
  ai_generation_enabled?: boolean;
  thinking_mode?: boolean;
  provider?: string;
  model?: string;
  temperature?: number;
  [key: string]: unknown;
}

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

// AutoDAN Types
export interface AutoDANRequest {
  prompt?: string;
  request?: string;
  target_model?: string;
  method?: 'vanilla' | 'best_of_n' | 'beam_search' | 'mousetrap';
  provider?: string;
  max_attempts?: number;
  technique?: string;
}

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

// GPTFuzz Types
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

// Gradient Optimization Types
export interface GradientOptimizationRequest {
  core_request: string;
  technique: 'hotflip' | 'gcg';
  potency_level: number;
  num_steps?: number;
  beam_width?: number;
  target_model?: string;
  provider?: string;
  model?: string;
  use_cache?: boolean;
}

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

// Technique Types
export interface TechniqueInfo {
  id: string;
  name: string;
  description: string;
  category: string;
  effectiveness: number;
  complexity: 'low' | 'medium' | 'high';
}

export interface TechniquesResponse {
  techniques: TechniqueInfo[];
  total: number;
  categories: string[];
}

// ============================================================================
// Jailbreak Service Implementation
// ============================================================================

export class JailbreakService {
  /**
   * Generate jailbreak prompt using basic endpoint
   */
  async jailbreak(request: JailbreakRequest) {
    try {
      const response = await apiClient.post<JailbreakResponse>('/api/v1/jailbreak', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'Jailbreak');
    }
  }

  /**
   * Generate AI-powered jailbreak prompt
   */
  async generateJailbreak(request: JailbreakRequest) {
    try {
      const response = await apiClient.post<JailbreakResponse>('/api/v1/generation/jailbreak/generate', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GenerateJailbreak');
    }
  }

  /**
   * Get available jailbreak techniques
   */
  async getTechniques() {
    try {
      const response = await apiClient.get<TechniquesResponse>('/api/v1/techniques');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetTechniques');
    }
  }

  // ============================================================================
  // AutoDAN Methods
  // ============================================================================

  /**
   * Execute AutoDAN attack
   */
  async autodan(request: AutoDANRequest) {
    try {
      const response = await apiClient.post<AutoDANResponse>('/api/v1/autodan', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'AutoDAN');
    }
  }

  /**
   * Execute AutoDAN with vanilla method
   */
  async autodanVanilla(request: Omit<AutoDANRequest, 'method'>) {
    return this.autodan({ ...request, method: 'vanilla' });
  }

  /**
   * Execute AutoDAN with best-of-n method
   */
  async autodanBestOfN(request: Omit<AutoDANRequest, 'method'>) {
    return this.autodan({ ...request, method: 'best_of_n' });
  }

  /**
   * Execute AutoDAN with beam search method
   */
  async autodanBeamSearch(request: Omit<AutoDANRequest, 'method'>) {
    return this.autodan({ ...request, method: 'beam_search' });
  }

  /**
   * Execute AutoDAN with mousetrap method
   */
  async autodanMousetrap(request: Omit<AutoDANRequest, 'method'>) {
    return this.autodan({ ...request, method: 'mousetrap' });
  }

  // ============================================================================
  // GPTFuzz Methods
  // ============================================================================

  /**
   * Start a GPTFuzz fuzzing session
   */
  async startFuzzSession(request: FuzzRequest) {
    try {
      const response = await apiClient.post<{ data: FuzzSession }>('/api/v1/gptfuzz', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'StartFuzzSession');
    }
  }

  /**
   * Get status of a fuzzing session
   */
  async getFuzzSessionStatus(sessionId: string) {
    try {
      const response = await apiClient.get<FuzzSession>(`/api/v1/gptfuzz/status/${sessionId}`);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetFuzzSessionStatus');
    }
  }

  // ============================================================================
  // Gradient Optimization Methods
  // ============================================================================

  /**
   * Optimize prompt using gradient-based methods
   */
  async gradientOptimize(request: GradientOptimizationRequest) {
    try {
      const response = await apiClient.post<GradientOptimizationResponse>('/api/v1/gradient/optimize', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GradientOptimize');
    }
  }

  /**
   * Optimize using HotFlip technique
   */
  async hotflipOptimize(
    coreRequest: string,
    potencyLevel: number = 5,
    options?: Partial<GradientOptimizationRequest>
  ) {
    return this.gradientOptimize({
      core_request: coreRequest,
      technique: 'hotflip',
      potency_level: potencyLevel,
      ...options,
    });
  }

  /**
   * Optimize using GCG technique
   */
  async gcgOptimize(
    coreRequest: string,
    potencyLevel: number = 5,
    options?: Partial<GradientOptimizationRequest>
  ) {
    return this.gradientOptimize({
      core_request: coreRequest,
      technique: 'gcg',
      potency_level: potencyLevel,
      ...options,
    });
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Create a basic jailbreak request with defaults
   */
  createJailbreakRequest(
    prompt: string,
    options?: {
      technique?: string;
      potency?: number;
      provider?: string;
      model?: string;
      useAI?: boolean;
      thinkingMode?: boolean;
    }
  ): JailbreakRequest {
    return {
      core_request: prompt,
      technique_suite: options?.technique || 'advanced',
      potency_level: options?.potency || 5,
      provider: options?.provider,
      model: options?.model,
      use_ai_generation: options?.useAI ?? true,
      is_thinking_mode: options?.thinkingMode ?? false,
      temperature: 0.8,
      max_tokens: 2048,
      use_cache: true,
    };
  }

  /**
   * Create an AutoDAN request with defaults
   */
  createAutoDANRequest(
    prompt: string,
    options?: {
      method?: AutoDANRequest['method'];
      targetModel?: string;
      provider?: string;
      maxAttempts?: number;
    }
  ): AutoDANRequest {
    return {
      request: prompt,
      method: options?.method || 'vanilla',
      target_model: options?.targetModel,
      provider: options?.provider,
      max_attempts: options?.maxAttempts || 10,
    };
  }

  /**
   * Create a GPTFuzz request with defaults
   */
  createFuzzRequest(
    targetPrompt: string,
    options?: {
      seeds?: string[];
      techniques?: string[];
      maxQueries?: number;
      provider?: string;
      model?: string;
    }
  ): FuzzRequest {
    return {
      target_prompt: targetPrompt,
      seeds: options?.seeds || [targetPrompt],
      techniques: options?.techniques || ['synonym', 'paraphrase', 'injection'],
      num_attempts: 50,
      max_queries: options?.maxQueries || 100,
      provider: options?.provider,
      model: options?.model,
    };
  }

  /**
   * Parse jailbreak metadata for display
   */
  parseJailbreakMetadata(metadata?: JailbreakMetadata) {
    if (!metadata) return null;

    return {
      techniqueSuite: metadata.technique_suite,
      potencyLevel: metadata.potency_level,
      appliedTechniques: metadata.applied_techniques || metadata.techniques_used || [],
      aiGeneration: metadata.ai_generation_enabled,
      thinkingMode: metadata.thinking_mode,
      provider: metadata.provider,
      model: metadata.model,
      temperature: metadata.temperature,
    };
  }

  /**
   * Calculate jailbreak effectiveness score
   */
  calculateEffectivenessScore(response: JailbreakResponse): number {
    if (!response.success) return 0;

    const metadata = this.parseJailbreakMetadata(response.metadata);
    if (!metadata) return 50; // Default score

    let score = 50; // Base score

    // Adjust based on potency level
    if (metadata.potencyLevel) {
      score += metadata.potencyLevel * 5;
    }

    // Adjust based on number of techniques applied
    if (metadata.appliedTechniques) {
      score += metadata.appliedTechniques.length * 3;
    }

    // Adjust based on AI generation
    if (metadata.aiGeneration) {
      score += 10;
    }

    // Adjust based on thinking mode
    if (metadata.thinkingMode) {
      score += 5;
    }

    return Math.min(100, Math.max(0, score));
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const jailbreakService = new JailbreakService();

// ============================================================================
// Convenience functions for direct usage
// ============================================================================

export const jailbreakApi = {
  jailbreak: (request: JailbreakRequest) => jailbreakService.jailbreak(request),
  generateJailbreak: (request: JailbreakRequest) => jailbreakService.generateJailbreak(request),
  getTechniques: () => jailbreakService.getTechniques(),

  // AutoDAN methods
  autodan: (request: AutoDANRequest) => jailbreakService.autodan(request),
  autodanVanilla: (request: Omit<AutoDANRequest, 'method'>) => jailbreakService.autodanVanilla(request),
  autodanBestOfN: (request: Omit<AutoDANRequest, 'method'>) => jailbreakService.autodanBestOfN(request),
  autodanBeamSearch: (request: Omit<AutoDANRequest, 'method'>) => jailbreakService.autodanBeamSearch(request),
  autodanMousetrap: (request: Omit<AutoDANRequest, 'method'>) => jailbreakService.autodanMousetrap(request),

  // GPTFuzz methods
  startFuzzSession: (request: FuzzRequest) => jailbreakService.startFuzzSession(request),
  getFuzzSessionStatus: (sessionId: string) => jailbreakService.getFuzzSessionStatus(sessionId),

  // Gradient optimization methods
  gradientOptimize: (request: GradientOptimizationRequest) => jailbreakService.gradientOptimize(request),
  hotflipOptimize: (coreRequest: string, potencyLevel?: number, options?: Partial<GradientOptimizationRequest>) =>
    jailbreakService.hotflipOptimize(coreRequest, potencyLevel, options),
  gcgOptimize: (coreRequest: string, potencyLevel?: number, options?: Partial<GradientOptimizationRequest>) =>
    jailbreakService.gcgOptimize(coreRequest, potencyLevel, options),

  // Utility methods
  createJailbreakRequest: (
    prompt: string,
    options?: {
      technique?: string;
      potency?: number;
      provider?: string;
      model?: string;
      useAI?: boolean;
      thinkingMode?: boolean;
    }
  ) => jailbreakService.createJailbreakRequest(prompt, options),
  createAutoDANRequest: (
    prompt: string,
    options?: {
      method?: AutoDANRequest['method'];
      targetModel?: string;
      provider?: string;
      maxAttempts?: number;
    }
  ) => jailbreakService.createAutoDANRequest(prompt, options),
  createFuzzRequest: (
    targetPrompt: string,
    options?: {
      seeds?: string[];
      techniques?: string[];
      maxQueries?: number;
      provider?: string;
      model?: string;
    }
  ) => jailbreakService.createFuzzRequest(targetPrompt, options),
  parseJailbreakMetadata: (metadata?: JailbreakMetadata) => jailbreakService.parseJailbreakMetadata(metadata),
  calculateEffectivenessScore: (response: JailbreakResponse) => jailbreakService.calculateEffectivenessScore(response),
};

export default jailbreakService;