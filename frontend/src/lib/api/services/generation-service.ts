/**
 * Generation Service - Aligned with Backend API
 *
 * This service is properly aligned with backend endpoints:
 * - POST /api/v1/generate (basic text generation)
 * - POST /api/v1/llm/generate (LLM-specific generation)
 * - POST /api/v1/llm/generate/with-resolution (with provider resolution)
 * - GET /api/v1/llm/health (LLM service health)
 * - GET /api/v1/llm/current-selection (current provider/model)
 */

import { apiClient } from '../client';
import { apiErrorHandler } from '../../errors/api-error-handler';

// ============================================================================
// Types (matching backend Pydantic models exactly)
// ============================================================================

export interface GenerationConfig {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_output_tokens?: number;
  stop_sequences?: string[];
  thinking_level?: 'low' | 'medium' | 'high';
}

export interface PromptRequest {
  prompt: string;
  provider?: string;
  model?: string;
  config?: GenerationConfig;
  api_key?: string;
  system_instruction?: string;
}

export interface PromptResponse {
  text: string;
  provider: string;
  model_used: string;
  latency_ms: number;
  usage_metadata?: Record<string, number>;
  cached: boolean;
  error?: string;
}

export interface GenerationHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  providers: string[];
}

export interface CurrentSelectionResponse {
  provider?: string;
  model?: string;
  source: 'resolved' | 'session' | 'default' | 'error';
  session_id?: string;
  error?: string;
}

// ============================================================================
// Generation Service Implementation
// ============================================================================

export class GenerationService {
  /**
   * Generate text using the basic generation endpoint
   */
  async generate(request: PromptRequest) {
    try {
      const response = await apiClient.post<PromptResponse>('/api/v1/generate', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'Generate');
    }
  }

  /**
   * Generate text using the LLM-specific endpoint
   */
  async llmGenerate(request: PromptRequest) {
    try {
      const response = await apiClient.post<PromptResponse>('/api/v1/llm/generate', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'LLMGenerate');
    }
  }

  /**
   * Generate text with explicit provider/model resolution
   */
  async generateWithResolution(request: PromptRequest) {
    try {
      const response = await apiClient.post<PromptResponse>('/api/v1/llm/generate/with-resolution', request);
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GenerateWithResolution');
    }
  }

  /**
   * Check the health of the generation service
   */
  async getHealth() {
    try {
      const response = await apiClient.get<GenerationHealthResponse>('/api/v1/llm/health');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetGenerationHealth');
    }
  }

  /**
   * Get the current provider/model selection that would be used for generation
   */
  async getCurrentSelection() {
    try {
      const response = await apiClient.get<CurrentSelectionResponse>('/api/v1/llm/current-selection');
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GetCurrentSelection');
    }
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Generate text with specific provider and model
   */
  async generateWithProvider(
    prompt: string,
    provider: string,
    model: string,
    config?: GenerationConfig,
    systemInstruction?: string
  ) {
    return this.generate({
      prompt,
      provider,
      model,
      config,
      system_instruction: systemInstruction,
    });
  }

  /**
   * Generate text with default provider resolution
   */
  async generateText(prompt: string, config?: GenerationConfig, systemInstruction?: string) {
    return this.generateWithResolution({
      prompt,
      config,
      system_instruction: systemInstruction,
    });
  }

  /**
   * Generate text with session-based provider selection
   */
  async generateWithSession(
    prompt: string,
    sessionId?: string,
    config?: GenerationConfig,
    systemInstruction?: string
  ) {
    const headers: Record<string, string> = {};
    if (sessionId) {
      headers['X-Session-ID'] = sessionId;
    }

    try {
      const response = await apiClient.post<PromptResponse>(
        '/api/v1/llm/generate',
        {
          prompt,
          config,
          system_instruction: systemInstruction,
        },
        { headers }
      );
      return response;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'GenerateWithSession');
    }
  }

  /**
   * Create a generation config with common defaults
   */
  createConfig(overrides?: Partial<GenerationConfig>): GenerationConfig {
    return {
      temperature: 0.7,
      top_p: 0.95,
      max_output_tokens: 2048,
      ...overrides,
    };
  }

  /**
   * Create a prompt request with defaults
   */
  createRequest(
    prompt: string,
    options?: {
      provider?: string;
      model?: string;
      config?: GenerationConfig;
      systemInstruction?: string;
    }
  ): PromptRequest {
    return {
      prompt,
      provider: options?.provider,
      model: options?.model,
      config: options?.config || this.createConfig(),
      system_instruction: options?.systemInstruction,
    };
  }

  /**
   * Parse usage metadata into a more usable format
   */
  parseUsageMetadata(usageMetadata?: Record<string, number>) {
    if (!usageMetadata) return null;

    return {
      promptTokens: usageMetadata.prompt_tokens || usageMetadata.prompt_token_count || 0,
      completionTokens: usageMetadata.completion_tokens || usageMetadata.candidates_token_count || 0,
      totalTokens: usageMetadata.total_tokens || usageMetadata.total_token_count || 0,
    };
  }

  /**
   * Calculate estimated cost based on usage metadata
   * (This would need to be updated with actual pricing)
   */
  estimateCost(response: PromptResponse): number {
    const usage = this.parseUsageMetadata(response.usage_metadata);
    if (!usage) return 0;

    // Placeholder pricing - should be updated with real provider pricing
    const pricing: Record<string, { input: number; output: number }> = {
      'gemini-2.0-flash-exp': { input: 0.000001, output: 0.000002 },
      'deepseek-chat': { input: 0.0000005, output: 0.000001 },
      'gpt-4': { input: 0.00003, output: 0.00006 },
      'claude-3-5-sonnet-20241022': { input: 0.000015, output: 0.000075 },
    };

    const modelPricing = pricing[response.model_used] || pricing['gemini-2.0-flash-exp'];
    return usage.promptTokens * modelPricing.input + usage.completionTokens * modelPricing.output;
  }
}

// ============================================================================
// Export singleton instance
// ============================================================================

export const generationService = new GenerationService();

// ============================================================================
// Convenience functions for direct usage
// ============================================================================

export const generationApi = {
  generate: (request: PromptRequest) => generationService.generate(request),
  llmGenerate: (request: PromptRequest) => generationService.llmGenerate(request),
  generateWithResolution: (request: PromptRequest) => generationService.generateWithResolution(request),
  getHealth: () => generationService.getHealth(),
  getCurrentSelection: () => generationService.getCurrentSelection(),

  // Convenience methods
  generateText: (prompt: string, config?: GenerationConfig, systemInstruction?: string) =>
    generationService.generateText(prompt, config, systemInstruction),
  generateWithProvider: (
    prompt: string,
    provider: string,
    model: string,
    config?: GenerationConfig,
    systemInstruction?: string
  ) => generationService.generateWithProvider(prompt, provider, model, config, systemInstruction),
  generateWithSession: (
    prompt: string,
    sessionId?: string,
    config?: GenerationConfig,
    systemInstruction?: string
  ) => generationService.generateWithSession(prompt, sessionId, config, systemInstruction),

  // Utility methods
  createConfig: (overrides?: Partial<GenerationConfig>) => generationService.createConfig(overrides),
  createRequest: (
    prompt: string,
    options?: {
      provider?: string;
      model?: string;
      config?: GenerationConfig;
      systemInstruction?: string;
    }
  ) => generationService.createRequest(prompt, options),
  parseUsageMetadata: (usageMetadata?: Record<string, number>) =>
    generationService.parseUsageMetadata(usageMetadata),
  estimateCost: (response: PromptResponse) => generationService.estimateCost(response),
};

export default generationService;