/**
 * Jailbreak Service - Aligned with Backend API
 *
 * This service uses the verified /api/v1/jailbreak endpoint (alias we added)
 * and matches the exact request/response format from runtime testing.
 */

import { apiClient } from '@/lib/api/client';
import { apiErrorHandler } from '@/lib/errors/api-error-handler';

// ============================================================================
// Backend-Aligned Types (matching actual API contract)
// ============================================================================

/**
 * Jailbreak request matching backend /api/v1/jailbreak endpoint
 */
export interface BackendJailbreakRequest {
  core_request: string;
  technique_suite?: string;
  potency_level?: number;
  provider?: string;
  model?: string;
  temperature?: number;
  top_p?: number;
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
  use_analysis_in_generation?: boolean;
  is_thinking_mode?: boolean;
  use_ai_generation?: boolean;
  use_cache?: boolean;
}

/**
 * Jailbreak response from backend
 */
export interface BackendJailbreakResponse {
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
    layers_applied: string[];
    techniques_used: string[];
    ai_generation_enabled: boolean;
    thinking_mode: boolean;
  };
  execution_time_seconds: number;
  error?: string;
}

// ============================================================================
// Frontend-Friendly Types
// ============================================================================

export interface JailbreakRequest {
  prompt: string;
  techniqueSuite?: string;
  potencyLevel?: number;
  provider?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;

  // Advanced options
  useAiGeneration?: boolean;
  useThinkingMode?: boolean;
  density?: number;

  // Technique toggles
  useLeetSpeak?: boolean;
  useHomoglyphs?: boolean;
  useCaesarCipher?: boolean;
  useRoleHijacking?: boolean;
  useInstructionInjection?: boolean;
  useNeuralBypass?: boolean;
  useMetaPrompting?: boolean;
  useMultilingualTrojan?: boolean;
  usePayloadSplitting?: boolean;

  // Fine-tuning
  leetSpeakDensity?: number;
  homoglyphDensity?: number;
  caesarShift?: number;
  multilingualTargetLanguage?: string;
  payloadSplittingParts?: number;
}

export interface JailbreakResult {
  id: string;
  success: boolean;
  originalPrompt: string;
  transformedPrompt: string;
  techniqueSuite: string;
  potencyLevel: number;
  provider: string;
  appliedTechniques: string[];
  layersApplied: string[];
  executionTimeSeconds: number;
  aiGenerationEnabled: boolean;
  thinkingMode: boolean;
  metadata: Record<string, any>;
  error?: string;
}

export interface JailbreakConfig {
  defaultTechniqueSuite: string;
  defaultPotencyLevel: number;
  enabledTechniques: string[];
  aiGenerationEnabled: boolean;
  thinkingModeEnabled: boolean;
}

// ============================================================================
// Aligned Jailbreak Service
// ============================================================================

class AlignedJailbreakService {
  private readonly baseUrl = '/api/v1';

  /**
   * Generate jailbreak prompt using the verified /jailbreak endpoint
   */
  async generateJailbreak(request: JailbreakRequest): Promise<JailbreakResult> {
    try {
      const backendRequest: BackendJailbreakRequest = {
        core_request: request.prompt,
        technique_suite: request.techniqueSuite || 'quantum_exploit',
        potency_level: request.potencyLevel || 7,
        provider: request.provider,
        model: request.model,
        temperature: request.temperature || 0.8,
        top_p: 0.95,
        max_new_tokens: request.maxTokens || 2048,
        density: request.density || 0.7,

        // Content transformation options
        use_leet_speak: request.useLeetSpeak || false,
        leet_speak_density: request.leetSpeakDensity || 0.3,
        use_homoglyphs: request.useHomoglyphs || false,
        homoglyph_density: request.homoglyphDensity || 0.3,
        use_caesar_cipher: request.useCaesarCipher || false,
        caesar_shift: request.caesarShift || 3,

        // Structural & semantic options
        use_role_hijacking: request.useRoleHijacking ?? true,
        use_instruction_injection: request.useInstructionInjection ?? true,
        use_adversarial_suffixes: false,
        use_few_shot_prompting: false,
        use_character_role_swap: false,

        // Advanced neural options
        use_neural_bypass: request.useNeuralBypass || false,
        use_meta_prompting: request.useMetaPrompting || false,
        use_counterfactual_prompting: false,
        use_contextual_override: false,

        // Research-driven options
        use_multilingual_trojan: request.useMultilingualTrojan || false,
        multilingual_target_language: request.multilingualTargetLanguage || '',
        use_payload_splitting: request.usePayloadSplitting || false,
        payload_splitting_parts: request.payloadSplittingParts || 2,
        use_contextual_interaction_attack: false,
        cia_preliminary_rounds: 3,

        // AI generation options
        use_analysis_in_generation: false,
        is_thinking_mode: request.useThinkingMode || false,
        use_ai_generation: request.useAiGeneration ?? true,
        use_cache: true,
      };

      const response = await apiClient.post<BackendJailbreakResponse>(`${this.baseUrl}/jailbreak`, backendRequest);

      return this.mapJailbreakResponse(response.data, request.prompt);
    } catch (error) {
      return apiErrorHandler.createErrorResponse(
        error,
        'Generate Jailbreak',
        this.getEmptyJailbreakResult(request.prompt)
      ).data!;
    }
  }

  /**
   * Get available technique suites for jailbreak
   */
  async getTechniqueSuites(): Promise<string[]> {
    try {
      const response = await apiClient.get<any>(`${this.baseUrl}/techniques`);
      return response.data.techniques?.map((t: any) => t.name) || [];
    } catch (error) {
      apiErrorHandler.handleError(error, 'Get Technique Suites');
      return ['quantum_exploit', 'deep_inception', 'neural_bypass'];
    }
  }

  /**
   * Get jailbreak configuration recommendations
   */
  async getConfigRecommendations(targetModel: string): Promise<JailbreakConfig> {
    // This would ideally come from backend, but for now provide sensible defaults
    const modelSpecificDefaults: Record<string, Partial<JailbreakConfig>> = {
      'gpt-4': {
        defaultTechniqueSuite: 'quantum_exploit',
        defaultPotencyLevel: 7,
        aiGenerationEnabled: true,
      },
      'claude-3': {
        defaultTechniqueSuite: 'deep_inception',
        defaultPotencyLevel: 8,
        aiGenerationEnabled: true,
      },
      'gemini': {
        defaultTechniqueSuite: 'neural_bypass',
        defaultPotencyLevel: 6,
        aiGenerationEnabled: true,
      },
    };

    const baseConfig: JailbreakConfig = {
      defaultTechniqueSuite: 'quantum_exploit',
      defaultPotencyLevel: 7,
      enabledTechniques: [
        'role_hijacking',
        'instruction_injection',
        'neural_bypass',
        'meta_prompting'
      ],
      aiGenerationEnabled: true,
      thinkingModeEnabled: false,
    };

    const modelKey = Object.keys(modelSpecificDefaults).find(key =>
      targetModel.toLowerCase().includes(key)
    );

    return modelKey
      ? { ...baseConfig, ...modelSpecificDefaults[modelKey] }
      : baseConfig;
  }

  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  private mapJailbreakResponse(data: BackendJailbreakResponse, originalPrompt: string): JailbreakResult {
    const metadata = data.metadata || {};
    return {
      id: data.request_id,
      success: data.success,
      originalPrompt,
      transformedPrompt: data.transformed_prompt,
      techniqueSuite: metadata.technique_suite,
      potencyLevel: metadata.potency_level,
      provider: metadata.provider,
      appliedTechniques: metadata.applied_techniques || [],
      layersApplied: metadata.layers_applied || [],
      executionTimeSeconds: data.execution_time_seconds,
      aiGenerationEnabled: metadata.ai_generation_enabled,
      thinkingMode: metadata.thinking_mode,
      metadata: metadata,
      error: data.error,
    };
  }

  private getEmptyJailbreakResult(originalPrompt: string): JailbreakResult {
    return {
      id: '',
      success: false,
      originalPrompt,
      transformedPrompt: '',
      techniqueSuite: 'quantum_exploit',
      potencyLevel: 7,
      provider: 'error',
      appliedTechniques: [],
      layersApplied: [],
      executionTimeSeconds: 0,
      aiGenerationEnabled: false,
      thinkingMode: false,
      metadata: {},
      error: 'Generation failed',
    };
  }
}

// ============================================================================
// Export Service Instance
// ============================================================================

export const alignedJailbreakService = new AlignedJailbreakService();

// Export for backward compatibility
export { alignedJailbreakService as jailbreakService };
