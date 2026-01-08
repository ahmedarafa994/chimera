/**
 * Evasion Service
 * 
 * Provides API methods for prompt evasion tasks,
 * including task submission, status monitoring, and results retrieval.
 */

import { enhancedApi } from '@/lib/api-enhanced';
import type {
  EvasionTask,
  EvasionSubmitRequest,
  EvasionSubmitResponse,
  EvasionStatus,
  EvasionResult,
} from '@/types/evasion-types';

const API_BASE = '/api/v1/evasion';

/**
 * Evasion Service API
 */
export const evasionService = {
  /**
   * Submit a new evasion task
   */
  async submitTask(request: EvasionSubmitRequest): Promise<EvasionSubmitResponse> {
    return enhancedApi.post<EvasionSubmitResponse>(
      `${API_BASE}/submit`,
      request
    );
  },

  /**
   * Get task status by ID
   */
  async getStatus(taskId: string): Promise<EvasionStatus> {
    return enhancedApi.get<EvasionStatus>(
      `${API_BASE}/status/${taskId}`
    );
  },

  /**
   * Get task results
   */
  async getResults(taskId: string): Promise<EvasionResult> {
    return enhancedApi.get<EvasionResult>(
      `${API_BASE}/results/${taskId}`
    );
  },

  /**
   * Cancel an ongoing task
   */
  async cancelTask(taskId: string): Promise<{ success: boolean; message: string }> {
    return enhancedApi.post<{ success: boolean; message: string }>(
      `${API_BASE}/cancel/${taskId}`
    );
  },

  /**
   * List all tasks
   */
  async listTasks(params?: {
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    limit?: number;
    offset?: number;
  }): Promise<{ tasks: EvasionTask[]; total: number }> {
    return enhancedApi.get<{ tasks: EvasionTask[]; total: number }>(
      `${API_BASE}/tasks`,
      { params }
    );
  },

  /**
   * Get task history
   */
  async getHistory(limit: number = 50): Promise<EvasionTask[]> {
    return enhancedApi.get<EvasionTask[]>(
      `${API_BASE}/history`,
      { params: { limit } }
    );
  },

  /**
   * Get available evasion strategies
   */
  async getStrategies(): Promise<{
    id: string;
    name: string;
    description: string;
    effectiveness: number;
  }[]> {
    return enhancedApi.get<{
      id: string;
      name: string;
      description: string;
      effectiveness: number;
    }[]>(`${API_BASE}/strategies`);
  },

  /**
   * Validate evasion configuration
   */
  async validateConfig(config: Partial<EvasionSubmitRequest>): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    return enhancedApi.post<{
      valid: boolean;
      errors: string[];
      warnings: string[];
    }>(`${API_BASE}/validate`, config);
  },

  /**
   * Get evasion statistics
   */
  async getStats(): Promise<{
    total_tasks: number;
    successful_tasks: number;
    failed_tasks: number;
    average_success_rate: number;
    most_effective_strategy: string;
  }> {
    return enhancedApi.get<{
      total_tasks: number;
      successful_tasks: number;
      failed_tasks: number;
      average_success_rate: number;
      most_effective_strategy: string;
    }>(`${API_BASE}/stats`);
  },
};

export default evasionService;