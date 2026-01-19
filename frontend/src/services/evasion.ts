/**
 * Evasion Service
 *
 * Service for managing metamorphic evasion tasks.
 * Aligned with backend /evasion endpoints.
 */

import { apiClient } from '@/lib/api/client';
import { apiErrorHandler } from '@/lib/errors/api-error-handler';
import { 
  EvasionTaskConfig, 
  EvasionTaskStatusResponse, 
  EvasionTaskResult,
  EvasionTaskStatusEnum
} from '../types/chimera';

export interface EvasionTaskListResponse {
  tasks: Array<{
    task_id: string;
    status: string;
    initial_prompt: string;
    target_model_id: number;
    created_at: string | null;
    completed_at: string | null;
    overall_success: boolean;
  }>;
  total: number;
  limit: number;
  offset: number;
}

class EvasionService {
  private readonly baseUrl = '/api/v1/evasion';

  /**
   * Create a new evasion task
   * POST /evasion/generate
   */
  async createEvasionTask(config: EvasionTaskConfig): Promise<EvasionTaskStatusResponse> {
    try {
      const response = await apiClient.post<EvasionTaskStatusResponse>(`${this.baseUrl}/generate`, config);
      return response.data;
    } catch (error) {
      console.error('Failed to create evasion task:', error);
      return apiErrorHandler.createErrorResponse(
        error,
        'Create Evasion Task',
        this.getEmptyStatusResponse()
      ).data!;
    }
  }

  /**
   * Get task status
   * GET /evasion/status/{task_id}
   */
  async getTaskStatus(taskId: string): Promise<EvasionTaskStatusResponse> {
    try {
      const response = await apiClient.get<EvasionTaskStatusResponse>(`${this.baseUrl}/status/${encodeURIComponent(taskId)}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get task status:', error);
      return apiErrorHandler.createErrorResponse(
        error,
        'Get Task Status',
        this.getEmptyStatusResponse(taskId)
      ).data!;
    }
  }

  /**
   * Get task results
   * GET /evasion/results/{task_id}
   */
  async getTaskResults(taskId: string): Promise<EvasionTaskResult> {
    try {
      const response = await apiClient.get<EvasionTaskResult>(`${this.baseUrl}/results/${encodeURIComponent(taskId)}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get task results:', error);
      throw error; // Let component handle error for results
    }
  }

  /**
   * Cancel task
   * POST /evasion/cancel/{task_id}
   */
  async cancelTask(taskId: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await apiClient.post<any>(`${this.baseUrl}/cancel/${encodeURIComponent(taskId)}`);
      return response.data;
    } catch (error) {
      console.error('Failed to cancel task:', error);
      return { success: false, message: error instanceof Error ? error.message : 'Cancellation failed' };
    }
  }

  /**
   * List tasks
   * GET /evasion/tasks
   */
  async listTasks(status?: string, limit = 50, offset = 0): Promise<EvasionTaskListResponse> {
    try {
      const response = await apiClient.get<EvasionTaskListResponse>(`${this.baseUrl}/tasks`, {
        params: {
          status_filter: status,
          limit,
          offset
        }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to list tasks:', error);
      return { tasks: [], total: 0, limit, offset };
    }
  }

  private getEmptyStatusResponse(taskId = ''): EvasionTaskStatusResponse {
    return {
      task_id: taskId,
      status: EvasionTaskStatusEnum.FAILED,
      message: 'Failed to retrieve status'
    };
  }
}

export const evasionService = new EvasionService();
