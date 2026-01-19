/**
 * AutoDAN Lifelong Learning Service
 *
 * Provides API methods for AutoDAN lifelong learning attacks,
 * including single attacks, loop attacks, and progress monitoring.
 */

import { apiClient } from '../client';
import type {
  LifelongAttackRequest,
  LifelongAttackResponse,
  LoopAttackRequest,
  LoopAttackResponse,
  AttackProgress,
  AttackResult,
} from '@/types/autodan-lifelong-types';

const API_BASE = '/autodan-turbo';

/**
 * AutoDAN Lifelong Learning Service API
 */
export const autodanLifelongService = {
  /**
   * Execute a single lifelong learning attack
   */
  async singleAttack(request: LifelongAttackRequest): Promise<LifelongAttackResponse> {
    const response = await apiClient.post<LifelongAttackResponse>(
      `${API_BASE}/attack`,
      request
    );
    return response.data;
  },

  /**
   * Execute a loop attack (multiple iterations)
   */
  async loopAttack(request: LoopAttackRequest): Promise<LoopAttackResponse> {
    const response = await apiClient.post<LoopAttackResponse>(
      `${API_BASE}/loop-attack`,
      request
    );
    return response.data;
  },

  /**
   * Get attack progress by task ID
   */
  async getProgress(taskId: string): Promise<AttackProgress> {
    const response = await apiClient.get<AttackProgress>(
      `${API_BASE}/progress/${taskId}`
    );
    return response.data;
  },

  /**
   * Cancel an ongoing attack
   */
  async cancelAttack(taskId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>(
      `${API_BASE}/cancel/${taskId}`
    );
    return response.data;
  },

  /**
   * Get attack results
   */
  async getResults(taskId: string): Promise<AttackResult> {
    const response = await apiClient.get<AttackResult>(
      `${API_BASE}/results/${taskId}`
    );
    return response.data;
  },

  /**
   * List all attack tasks
   */
  async listTasks(params?: {
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    limit?: number;
    offset?: number;
  }): Promise<{ tasks: AttackProgress[]; total: number }> {
    const response = await apiClient.get<{ tasks: AttackProgress[]; total: number }>(
      `${API_BASE}/tasks`,
      { params }
    );
    return response.data;
  },

  /**
   * Get attack history
   */
  async getHistory(limit: number = 50): Promise<AttackResult[]> {
    const response = await apiClient.get<AttackResult[]>(
      `${API_BASE}/history`,
      { params: { limit } }
    );
    return response.data;
  },

  /**
   * Get available attack strategies
   */
  async getStrategies(): Promise<{
    id: string;
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  }[]> {
    const response = await apiClient.get<{
      id: string;
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    }[]>(`${API_BASE}/strategies`);
    return response.data;
  },

  /**
   * Validate attack configuration
   */
  async validateConfig(config: Partial<LifelongAttackRequest>): Promise<{
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
};

export default autodanLifelongService;
