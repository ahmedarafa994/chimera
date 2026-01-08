/**
 * AutoAdv Service
 * 
 * Provides API methods for adversarial prompt generation,
 * batch processing, and history retrieval.
 */

import { enhancedApi } from '@/lib/api-enhanced';

export interface AutoAdvStrategy {
  id: string;
  name: string;
  description: string;
  category: string;
  effectiveness: number;
}

export interface AutoAdvGenerateRequest {
  prompt: string;
  target_model?: string;
  strategy?: string;
  num_variations?: number;
  temperature?: number;
  include_analysis?: boolean;
}

export interface AutoAdvGenerateResponse {
  id: string;
  original_prompt: string;
  variations: AutoAdvVariation[];
  strategy_used: string;
  model_used: string;
  created_at: string;
  analysis?: AutoAdvAnalysis;
}

export interface AutoAdvVariation {
  id: string;
  text: string;
  strategy: string;
  confidence: number;
  tokens: number;
}

export interface AutoAdvAnalysis {
  vulnerability_score: number;
  detected_patterns: string[];
  recommendations: string[];
}

export interface AutoAdvBatchRequest {
  prompts: string[];
  target_model?: string;
  strategy?: string;
  parallel?: boolean;
}

export interface AutoAdvBatchResponse {
  batch_id: string;
  results: AutoAdvGenerateResponse[];
  total_processed: number;
  failed_count: number;
  processing_time_ms: number;
}

export interface AutoAdvHistoryItem {
  id: string;
  prompt: string;
  strategy: string;
  variations_count: number;
  created_at: string;
  model_used: string;
}

const API_BASE = '/api/v1/autoadv';

/**
 * AutoAdv Service API
 */
export const autoadvService = {
  /**
   * Generate adversarial prompt variations
   */
  async generate(request: AutoAdvGenerateRequest): Promise<AutoAdvGenerateResponse> {
    return enhancedApi.post<AutoAdvGenerateResponse>(
      `${API_BASE}/generate`,
      request
    );
  },

  /**
   * Batch generate adversarial prompts
   */
  async batchGenerate(request: AutoAdvBatchRequest): Promise<AutoAdvBatchResponse> {
    return enhancedApi.post<AutoAdvBatchResponse>(
      `${API_BASE}/batch`,
      request
    );
  },

  /**
   * Get generation history
   */
  async getHistory(params?: {
    limit?: number;
    offset?: number;
    strategy?: string;
    model?: string;
  }): Promise<{ items: AutoAdvHistoryItem[]; total: number }> {
    return enhancedApi.get<{ items: AutoAdvHistoryItem[]; total: number }>(
      `${API_BASE}/history`,
      { params }
    );
  },

  /**
   * Get available strategies
   */
  async getStrategies(): Promise<AutoAdvStrategy[]> {
    return enhancedApi.get<AutoAdvStrategy[]>(`${API_BASE}/strategies`);
  },

  /**
   * Get a specific generation by ID
   */
  async getGeneration(id: string): Promise<AutoAdvGenerateResponse> {
    return enhancedApi.get<AutoAdvGenerateResponse>(`${API_BASE}/generation/${id}`);
  },

  /**
   * Delete a generation from history
   */
  async deleteGeneration(id: string): Promise<{ success: boolean }> {
    return enhancedApi.delete<{ success: boolean }>(`${API_BASE}/generation/${id}`);
  },

  /**
   * Validate a prompt before generation
   */
  async validatePrompt(prompt: string): Promise<{
    valid: boolean;
    issues: string[];
    suggestions: string[];
  }> {
    return enhancedApi.post<{
      valid: boolean;
      issues: string[];
      suggestions: string[];
    }>(`${API_BASE}/validate`, { prompt });
  },

  /**
   * Get strategy recommendations for a prompt
   */
  async getStrategyRecommendations(prompt: string): Promise<{
    recommended: AutoAdvStrategy[];
    reasoning: string;
  }> {
    return enhancedApi.post<{
      recommended: AutoAdvStrategy[];
      reasoning: string;
    }>(`${API_BASE}/recommend-strategy`, { prompt });
  },
};

export default autoadvService;