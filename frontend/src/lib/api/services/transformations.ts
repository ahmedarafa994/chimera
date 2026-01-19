/**
 * Transformation Service
 *
 * Service for managing prompt transformations and technique applications
 */

import { apiClient } from '../client';

export interface TransformationRequest {
  prompt: string;
  techniques: string[];
  parameters?: Record<string, any>;
  model?: string;
  temperature?: number;
}

export interface TransformationResult {
  id: string;
  originalPrompt: string;
  transformedPrompt: string;
  techniques: AppliedTechnique[];
  metadata: TransformationMetadata;
  createdAt: Date;
}

export interface AppliedTechnique {
  id: string;
  name: string;
  parameters: Record<string, any>;
  executionTime: number;
  success: boolean;
  error?: string;
}

export interface TransformationMetadata {
  totalExecutionTime: number;
  tokenCount: {
    input: number;
    output: number;
  };
  complexity: number;
  effectiveness: number;
}

export interface TransformationHistory {
  transformations: TransformationResult[];
  total: number;
  hasMore: boolean;
}

export interface TransformationStats {
  totalTransformations: number;
  successRate: number;
  averageComplexity: number;
  topTechniques: TechniqueUsageStats[];
  recentActivity: TransformationActivity[];
}

export interface TechniqueUsageStats {
  techniqueId: string;
  techniqueName: string;
  usageCount: number;
  successRate: number;
  averageEffectiveness: number;
}

export interface TransformationActivity {
  date: string;
  count: number;
  successCount: number;
}

const API_BASE = '/transform';

/**
 * Transformation Service API
 */
export const transformationService = {
  mapResponse(data: any): TransformationResult {
    const applied = Array.isArray(data?.metadata?.applied_techniques)
      ? data.metadata.applied_techniques
      : [];

    return {
      id: data?.metadata?.request_id || `transform_${Date.now()}`,
      originalPrompt: data?.original_prompt || data?.originalPrompt || "",
      transformedPrompt: data?.transformed_prompt || data?.transformedPrompt || "",
      techniques: applied.map((name: string, index: number) => ({
        id: `${name}-${index}`,
        name,
        parameters: {},
        executionTime: 0,
        success: true,
      })),
      metadata: {
        totalExecutionTime: data?.metadata?.execution_time_ms || 0,
        tokenCount: { input: 0, output: 0 },
        complexity: data?.metadata?.potency_level || 0,
        effectiveness: data?.metadata?.bypass_probability || 0,
      },
      createdAt: new Date(),
    };
  },
  /**
   * Transform a prompt using specified techniques
   */
  async transformPrompt(request: TransformationRequest): Promise<TransformationResult> {
    const response = await apiClient.post<any>(`${API_BASE}`, request);
    return transformationService.mapResponse(response.data);
  },

  /**
   * Get a specific transformation by ID
   * Note: Backend implementation may be pending
   */
  async getTransformation(id: string): Promise<TransformationResult> {
    const response = await apiClient.get<TransformationResult>(`${API_BASE}/history/${id}`);
    return response.data;
  },

  /**
   * Get transformation history
   * Note: Backend implementation may be pending
   */
  async getTransformationHistory(
    limit = 20,
    offset = 0
  ): Promise<TransformationHistory> {
    const response = await apiClient.get<TransformationHistory>(`${API_BASE}/history`, {
      params: { limit, offset }
    });
    return response.data;
  },

  /**
   * Get transformation statistics
   * Note: Backend implementation may be pending
   */
  async getTransformationStats(days = 30): Promise<TransformationStats> {
    const response = await apiClient.get<TransformationStats>(`${API_BASE}/stats`, {
      params: { days }
    });
    return response.data;
  },

  /**
   * Delete a transformation from history
   * Note: Backend implementation may be pending
   */
  async deleteTransformation(id: string): Promise<void> {
    await apiClient.delete(`${API_BASE}/history/${id}`);
  },

  /**
   * Export transformations
   * Note: Backend implementation may be pending
   */
  async exportTransformations(ids: string[]): Promise<Blob> {
    const response = await apiClient.post(`${API_BASE}/export`,
      { ids },
      { responseType: 'blob' }
    );
    return response.data as any; // apiClient returns data
  },

  /**
   * Validate a combination of techniques
   * Note: Backend implementation may be pending
   */
  async validateTechniqueCombination(techniques: string[]): Promise<{
    valid: boolean;
    warnings: string[];
    suggestions: string[];
  }> {
    const response = await apiClient.post(`${API_BASE}/validate`, { techniques });
    return response.data as any;
  }
};
