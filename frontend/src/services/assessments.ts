/**
 * Assessment Service
 *
 * Service for managing security assessments and vulnerability analysis
 */

import { apiClient } from '@/lib/api/client';
import { apiErrorHandler, ApiError } from '@/lib/errors/api-error-handler';

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

export interface AssessmentResults {
  overallScore: number;
  vulnerabilities: Vulnerability[];
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
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

class AssessmentService {
  private readonly baseUrl = '/assessments';

  async listAssessments(params: AssessmentListParams = {}): Promise<AssessmentListResponse> {
    try {
      const response = await apiClient.get<AssessmentListResponse>(`${this.baseUrl}/`, {
        params: {
          status_filter: params.status,
          page: params.page || 1,
          page_size: params.page_size || 20,
          search: params.search
        }
      });

      return response.data;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'Assessment.listAssessments');
    }
  }

  async getAssessment(id: number): Promise<Assessment> {
    try {
      const response = await apiClient.get<Assessment>(`${this.baseUrl}/${id}`);
      return response.data;
    } catch (error) {
      throw apiErrorHandler.handleError(error, `Assessment.getAssessment(${id})`);
    }
  }

  async createAssessment(request: CreateAssessmentRequest): Promise<Assessment> {
    try {
      const response = await apiClient.post<Assessment>(`${this.baseUrl}/`, request);
      return response.data;
    } catch (error) {
      throw apiErrorHandler.handleError(error, 'Assessment.createAssessment');
    }
  }

  async deleteAssessment(id: number): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/${id}`);
    } catch (error) {
      throw apiErrorHandler.handleError(error, `Assessment.deleteAssessment(${id})`);
    }
  }

  async runAssessment(id: number): Promise<Assessment> {
    try {
      const response = await apiClient.post<Assessment>(`${this.baseUrl}/${id}/run`);
      return response.data;
    } catch (error) {
      throw apiErrorHandler.handleError(error, `Assessment.runAssessment(${id})`);
    }
  }
}

export const assessmentService = new AssessmentService();
