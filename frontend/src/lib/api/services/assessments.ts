/**
 * Assessment Service
 *
 * Service for managing security assessments and vulnerability analysis
 */

import { apiClient } from '../client';
import { ApiResponse, PaginatedResponse } from '../types';

export interface Assessment {
  id: number;
  name: string;
  description: string | null;
  status: string;
  target_provider: string;
  target_model: string;
  target_config: Record<string, any>;
  technique_ids: string[];
  results: Record<string, any>;
  findings_count: number;
  vulnerabilities_found: number;
  risk_score: number;
  risk_level: string;
  created_at: string;
  updated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface Vulnerability {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  impact: string;
  remediation: string;
}

export interface CreateAssessmentRequest {
  name: string;
  description?: string;
  target_provider: string;
  target_model: string;
  target_config?: Record<string, any>;
  technique_ids?: string[];
}

export interface AssessmentListParams {
  status?: 'pending' | 'running' | 'completed' | 'failed';
  page?: number;
  page_size?: number;
  search?: string;
}

export interface AssessmentListResponse {
  assessments: Assessment[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

const API_BASE = '/assessments';

/**
 * Assessment Service API
 */
export const assessmentService = {
  /**
   * List assessments
   */
  async listAssessments(params: AssessmentListParams = {}): Promise<AssessmentListResponse> {
    const response = await apiClient.get<AssessmentListResponse>(`${API_BASE}`, {
      params: {
        status_filter: params.status,
        page: params.page || 1,
        page_size: params.page_size || 20,
        search: params.search
      }
    });
    return response.data;
  },

  /**
   * Get a specific assessment by ID
   */
  async getAssessment(id: number): Promise<Assessment> {
    const response = await apiClient.get<Assessment>(`${API_BASE}/${id}`);
    return response.data;
  },

  /**
   * Create a new assessment
   */
  async createAssessment(request: CreateAssessmentRequest): Promise<Assessment> {
    const response = await apiClient.post<Assessment>(`${API_BASE}`, request);
    return response.data;
  },

  /**
   * Delete an assessment
   */
  async deleteAssessment(id: number): Promise<void> {
    await apiClient.delete(`${API_BASE}/${id}`);
  },

  /**
   * Run an assessment
   */
  async runAssessment(id: number): Promise<Assessment> {
    const response = await apiClient.post<Assessment>(`${API_BASE}/${id}/run`);
    return response.data;
  }
};
