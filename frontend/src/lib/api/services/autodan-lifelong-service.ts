/**
 * AutoDAN Lifelong Learning Service
 * 
 * Provides API methods for AutoDAN lifelong learning attacks,
 * including single attacks, loop attacks, and progress monitoring.
 */

import { enhancedApi } from '@/lib/api-enhanced';
import type {
  LifelongAttackRequest,
  LifelongAttackResponse,
  LoopAttackRequest,
  LoopAttackResponse,
  AttackProgress,
  AttackResult,
} from '@/types/autodan-lifelong-types';

const API_BASE = '/api/v1/autodan-lifelong';

/**
 * AutoDAN Lifelong Learning Service API
 */
export const autodanLifelongService = {
  /**
   * Execute a single lifelong learning attack
   */
  async singleAttack(request: LifelongAttackRequest): Promise<LifelongAttackResponse> {
    return enhancedApi.post<LifelongAttackResponse>(
      `${API_BASE}/attack`,
      request
    );
  },

  /**
   * Execute a loop attack (multiple iterations)
   */
  async loopAttack(request: LoopAttackRequest): Promise<LoopAttackResponse> {
    return enhancedApi.post<LoopAttackResponse>(
      `${API_BASE}/loop-attack`,
      request
    );
  },

  /**
   * Get attack progress by task ID
   */
  async getProgress(taskId: string): Promise<AttackProgress> {
    return enhancedApi.get<AttackProgress>(
      `${API_BASE}/progress/${taskId}`
    );
  },

  /**
   * Cancel an ongoing attack
   */
  async cancelAttack(taskId: string): Promise<{ success: boolean; message: string }> {
    return enhancedApi.post<{ success: boolean; message: string }>(
      `${API_BASE}/cancel/${taskId}`
    );
  },

  /**
   * Get attack results
   */
  async getResults(taskId: string): Promise<AttackResult> {
    return enhancedApi.get<AttackResult>(
      `${API_BASE}/results/${taskId}`
    );
  },

  /**
   * List all attack tasks
   */
  async listTasks(params?: {
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    limit?: number;
    offset?: number;
  }): Promise<{ tasks: AttackProgress[]; total: number }> {
    return enhancedApi.get<{ tasks: AttackProgress[]; total: number }>(
      `${API_BASE}/tasks`,
      { params }
    );
  },

  /**
   * Get attack history
   */
  async getHistory(limit: number = 50): Promise<AttackResult[]> {
    return enhancedApi.get<AttackResult[]>(
      `${API_BASE}/history`,
      { params: { limit } }
    );
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
    return enhancedApi.get<{
      id: string;
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    }[]>(`${API_BASE}/strategies`);
  },

  /**
   * Validate attack configuration
   */
  async validateConfig(config: Partial<LifelongAttackRequest>): Promise<{
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
};

export default autodanLifelongService;