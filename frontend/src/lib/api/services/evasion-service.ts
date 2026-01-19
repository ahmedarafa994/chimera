/**
 * Evasion Service
 *
 * Provides API methods for prompt evasion tasks,
 * including task submission, status monitoring, and results retrieval.
 */

import { apiClient } from '../client';
import type {
  EvasionTask,
  EvasionSubmitRequest,
  EvasionSubmitResponse,
  EvasionStatus,
  EvasionResult,
} from '@/types/evasion-types';

const API_BASE = '/evasion';

/**
 * Evasion Service API
 */
export const evasionService = {
  /**
   * Submit a new evasion task
   */
  async submitTask(request: EvasionSubmitRequest): Promise<EvasionSubmitResponse> {
    const response = await apiClient.post<EvasionSubmitResponse>(
      `${API_BASE}/submit`,
      request
    );
    return response.data;
  },

  /**
   * Get task status by ID
   */
  async getStatus(taskId: string): Promise<EvasionStatus> {
    const response = await apiClient.get<EvasionStatus>(
      `${API_BASE}/status/${taskId}`
    );
    return response.data;
  },

  /**
   * Get task results
   */
  async getResults(taskId: string): Promise<EvasionResult> {
    const response = await apiClient.get<EvasionResult>(
      `${API_BASE}/results/${taskId}`
    );
    return response.data;
  },

  /**
   * Cancel an ongoing task
   */
  async cancelTask(taskId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>(
      `${API_BASE}/cancel/${taskId}`
    );
    return response.data;
  },

  /**
   * List all tasks
   */
  async listTasks(params?: {
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    limit?: number;
    offset?: number;
  }): Promise<{ tasks: EvasionTask[]; total: number }> {
    const response = await apiClient.get<{ tasks: EvasionTask[]; total: number }>(
      `${API_BASE}/tasks`,
      { params }
    );
    return response.data;
  },

  /**
   * Get task history
   */
  async getHistory(limit: number = 50): Promise<EvasionTask[]> {
    const response = await apiClient.get<EvasionTask[]>(
      `${API_BASE}/history`,
      { params: { limit } }
    );
    return response.data;
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
    const response = await apiClient.get<any[]>(`${API_BASE}/strategies`);
    return response.data;
  },

  /**
   * Validate evasion configuration
   */
  async validateConfig(config: Partial<EvasionSubmitRequest>): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    const response = await apiClient.post<{
      valid: boolean;
      errors: string[];
      warnings: string[];
    }>(`${API_BASE}/validate`, config);
    return response.data;
  },

  /**
   * Get evasion statistics
   */
  async getStats(): Promise<any> {
    const response = await apiClient.get<any>(`${API_BASE}/stats`);
    return response.data;
  },
};

export default evasionService;
