import { apiClient } from '../client';
import { toast } from 'sonner';

// API Response Types
interface TechniqueApiResponse {
  technique_id?: string;
  id?: string;
  name?: string;
  description?: string;
  category?: string;
  technique_type?: string;
  status?: string;
  visibility?: string;
  workspace_id?: string;
  created_by?: string;
  created_by_name?: string;
  created_at?: string;
  updated_at?: string;
  parent_technique_id?: string;
  base_technique_id?: string;
  version?: string;
  steps?: TechniqueStep[];
  parameters?: TechniqueParameter[];
  tags?: string[];
  complexity_score?: number;
  estimated_execution_time?: number;
  usage_count?: number;
  success_rate?: number;
  average_execution_time?: number;
  metadata?: Record<string, any>;
}

interface TemplateApiResponse {
  template_id?: string;
  id?: string;
  name?: string;
  description?: string;
  category?: string;
  steps_template?: TechniqueStep[];
  base_steps?: TechniqueStep[];
  parameters_template?: TechniqueParameter[];
  base_parameters?: TechniqueParameter[];
  preview_image_url?: string;
}

interface TechniqueListApiResponse {
  techniques?: TechniqueApiResponse[];
  templates?: TemplateApiResponse[];
  total_techniques?: number;
  total?: number;
  total_templates?: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

export type TechniqueType = 'transformation' | 'validation' | 'execution' | 'composition';
export type ParameterType = 'string' | 'integer' | 'float' | 'boolean' | 'list' | 'object';
export type TechniqueStatus = 'draft' | 'testing' | 'active' | 'deprecated';
export type VisibilityLevel = 'private' | 'team' | 'public';

export interface TechniqueParameter {
  name: string;
  display_name: string;
  description: string;
  parameter_type: ParameterType;
  default_value?: any;
  required: boolean;
  min_value?: number;
  max_value?: number;
  allowed_values?: any[];
  pattern?: string;
}

export interface TechniqueStep {
  step_id: string;
  name: string;
  description: string;
  step_type: string;
  implementation: Record<string, any>;
  parameters: Record<string, any>;
  next_step_id?: string;
  condition?: string;
  position_x: number;
  position_y: number;
}

export interface TechniqueTemplate {
  template_id: string;
  name: string;
  description: string;
  category: string;
  base_steps: TechniqueStep[];
  base_parameters: TechniqueParameter[];
  preview_image_url?: string;
}

export interface CustomTechnique {
  id: string;
  technique_id?: string;
  name: string;
  description: string;
  category?: string;
  technique_type: TechniqueType;
  status: TechniqueStatus;
  visibility: VisibilityLevel;
  workspace_id?: string;
  created_by: string;
  created_by_name?: string;
  created_at: string;
  updated_at: string;
  base_technique_id?: string;
  version?: string;
  steps: TechniqueStep[];
  parameters: TechniqueParameter[];
  tags: string[];
  complexity_score?: number;
  estimated_execution_time?: number;
  usage_count?: number;
  success_rate?: number;
  average_execution_time?: number;
  metadata?: Record<string, any>;
}

export interface TechniqueExecution {
  execution_id: string;
  technique_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
  execution_time: number;
  results: Record<string, any>;
  logs: string[];
}

export type TechniqueCreate = Partial<CustomTechnique>;
export type TechniqueUpdate = Partial<CustomTechnique>;

export interface TechniqueListResponse {
  techniques: CustomTechnique[];
  templates: TechniqueTemplate[];
  total_techniques: number;
  total_templates: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface TechniqueTestRequest {
  test_input: string;
  test_parameters?: Record<string, any>;
}

export interface TechniqueStatsResponse {
  technique_id: string;
  executions: any[];
  success_rate: number;
  average_execution_time: number;
  usage_by_date: Record<string, number>;
  performance_trends: Record<string, any>;
}

class TechniqueBuilderService {
  private readonly baseUrl = '/technique-builder';

  private mapTechnique = (technique: any): CustomTechnique => ({
    id: technique.technique_id ?? technique.id ?? '',
    technique_id: technique.technique_id ?? technique.id,
    name: technique.name ?? 'Untitled technique',
    description: technique.description ?? '',
    category: technique.category,
    technique_type: technique.technique_type ?? 'transformation',
    status: technique.status ?? 'draft',
    visibility: technique.visibility ?? 'private',
    workspace_id: technique.workspace_id,
    created_by: technique.created_by ?? '',
    created_by_name: technique.created_by_name,
    created_at: technique.created_at
      ? new Date(technique.created_at).toISOString()
      : new Date().toISOString(),
    updated_at: technique.updated_at
      ? new Date(technique.updated_at).toISOString()
      : new Date().toISOString(),
    base_technique_id: technique.parent_technique_id ?? technique.base_technique_id,
    version: technique.version,
    steps: technique.steps ?? [],
    parameters: technique.parameters ?? [],
    tags: technique.tags ?? [],
    complexity_score: technique.complexity_score,
    estimated_execution_time: technique.estimated_execution_time,
    usage_count: technique.usage_count,
    success_rate: technique.success_rate,
    average_execution_time: technique.average_execution_time,
    metadata: technique.metadata ?? {}
  });

  private mapTemplate(template: any): TechniqueTemplate {
    return {
      template_id: template.template_id ?? template.id ?? '',
      name: template.name ?? 'Template',
      description: template.description ?? '',
      category: template.category ?? 'general',
      base_steps: template.steps_template ?? template.base_steps ?? [],
      base_parameters: template.parameters_template ?? template.base_parameters ?? [],
      preview_image_url: template.preview_image_url
    };
  }

  async createTechnique(techniqueData: Partial<CustomTechnique>): Promise<CustomTechnique> {
    try {
      const payload = {
        name: techniqueData.name,
        description: techniqueData.description,
        category: techniqueData.category,
        technique_type: techniqueData.technique_type ?? 'transformation',
        visibility: techniqueData.visibility ?? 'private',
        workspace_id: techniqueData.workspace_id,
        tags: techniqueData.tags ?? []
      };
      const response = await apiClient.post(this.baseUrl, payload);
      toast.success('Technique saved to draft');
      return this.mapTechnique(response.data as TechniqueApiResponse);
    } catch (error) {
      console.error('Failed to create technique:', error);
      toast.error('Failed to create custom technique');
      throw error;
    }
  }

  async getTechnique(id: string): Promise<CustomTechnique> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/${id}`);
      return this.mapTechnique(response.data as TechniqueApiResponse);
    } catch (error) {
      console.error('Failed to get technique:', error);
      toast.error('Failed to load technique details');
      throw error;
    }
  }

  async listCustomTechniques(params?: {
    workspace_id?: string;
    status?: TechniqueStatus;
    type?: TechniqueType;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<TechniqueListResponse> {
    try {
      const response = await apiClient.get(this.baseUrl, {
        params: {
          workspace_id: params?.workspace_id,
          status: params?.status,
          technique_type: params?.type,
          search: params?.search,
          limit: params?.limit,
          offset: params?.offset
        }
      });
      const data = response.data as TechniqueListApiResponse;
      return {
        techniques: (data.techniques ?? []).map((t: any) => this.mapTechnique(t)),
        templates: (data.templates ?? []).map((tpl: any) => this.mapTemplate(tpl)),
        total_techniques: data.total_techniques ?? data.total ?? data.techniques?.length ?? 0,
        total_templates: data.total_templates ?? data.templates?.length ?? 0,
        page: data.page ?? 1,
        page_size: data.page_size ?? params?.limit ?? 20,
        has_next: Boolean(data.has_next),
        has_prev: Boolean(data.has_prev)
      };
    } catch (error) {
      console.error('Failed to list techniques:', error);
      toast.error('Failed to load custom techniques');
      throw error;
    }
  }

  async updateTechnique(id: string, updateData: Partial<CustomTechnique>): Promise<CustomTechnique> {
    try {
      const response = await apiClient.patch(`${this.baseUrl}/${id}`, updateData);
      toast.success('Technique updated successfully');
      return this.mapTechnique(response.data as TechniqueApiResponse);
    } catch (error) {
      console.error('Failed to update technique:', error);
      toast.error('Failed to update technique');
      throw error;
    }
  }

  async cloneTechnique(techniqueId: string): Promise<CustomTechnique> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/${techniqueId}/clone`);
      toast.success('Technique cloned successfully');
      return this.mapTechnique(response.data as TechniqueApiResponse);
    } catch (error) {
      console.error('Failed to clone technique:', error);
      toast.error('Failed to clone technique');
      throw error;
    }
  }

  async deleteTechnique(techniqueId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/${techniqueId}`);
      toast.success('Technique deleted successfully');
    } catch (error) {
      console.error('Failed to delete technique:', error);
      toast.error('Failed to delete technique');
      throw error;
    }
  }

  async testTechnique(techniqueId: string, request: TechniqueTestRequest): Promise<Record<string, any>> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/${techniqueId}/test`, {
        test_input: request.test_input,
        parameters: request.test_parameters ?? {}
      });
      return response.data as Record<string, any>;
    } catch (error) {
      console.error('Failed to test technique:', error);
      toast.error('Failed to test technique');
      throw error;
    }
  }

  async getTechniqueStats(techniqueId: string): Promise<TechniqueStatsResponse> {
    const response = await apiClient.get(`${this.baseUrl}/${techniqueId}/stats`);
    return response.data as TechniqueStatsResponse;
  }

  async listTechniques(params?: {
    workspace_id?: string;
    status?: TechniqueStatus;
    type?: TechniqueType;
    search?: string;
    page?: number;
    page_size?: number;
  }): Promise<TechniqueListResponse> {
    return this.listCustomTechniques({
      ...params,
      limit: params?.page_size,
      offset: params?.page && params?.page_size ? (params.page - 1) * params.page_size : undefined
    });
  }

  validateTechniqueCreate(data: Partial<CustomTechnique>): Record<string, string> {
    const errors: Record<string, string> = {};
    if (!data.name?.trim()) errors.name = 'Name is required';
    if (!data.technique_type) errors.technique_type = 'Technique type is required';
    return errors;
  }

  validateTestRequest(data: TechniqueTestRequest): Record<string, string> {
    const errors: Record<string, string> = {};
    if (!data.test_input?.trim()) errors.test_input = 'Test input is required';
    return errors;
  }

  formatExecutionTime(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  getCategories(): Array<{id: string, name: string}> {
    return [
      { id: 'Transformation', name: 'Transformation' },
      { id: 'Obfuscation', name: 'Obfuscation' },
      { id: 'Persona', name: 'Persona' },
      { id: 'Context', name: 'Context' },
      { id: 'Validation', name: 'Validation' },
      { id: 'Hybrid', name: 'Hybrid' }
    ];
  }

  formatSuccessRate(rate?: number): string {
    if (rate === undefined) return 'N/A';
    return `${(rate * 100).toFixed(1)}%`;
  }

  getParameterTypeDisplayName(type: ParameterType): string {
    const names: Record<ParameterType, string> = {
      string: 'String',
      integer: 'Integer',
      float: 'Float',
      boolean: 'Boolean',
      list: 'List',
      object: 'Object'
    };
    return names[type] || type;
  }

  getVisibilityColor(visibility: VisibilityLevel): string {
    const colors: Record<VisibilityLevel, string> = {
      private: 'gray',
      team: 'blue',
      public: 'green'
    };
    return colors[visibility] || 'gray';
  }

  getVisibilityDisplayName(visibility: VisibilityLevel): string {
    const names: Record<VisibilityLevel, string> = {
      private: 'Private',
      team: 'Team',
      public: 'Public'
    };
    return names[visibility] || visibility;
  }

  formatComplexityScore(score?: number): string {
    if (score === undefined) return 'N/A';
    if (score < 3) return 'Low Complexity';
    if (score < 7) return 'Medium Complexity';
    return 'High Complexity';
  }

  getTechniqueTypeDisplayName(type: TechniqueType): string {
    const names: Record<TechniqueType, string> = {
      transformation: 'Transformation',
      validation: 'Validation',
      execution: 'Execution',
      composition: 'Composition'
    };
    return names[type] || type;
  }

  getStatusDisplayName(status: TechniqueStatus): string {
    const names: Record<TechniqueStatus, string> = {
      draft: 'Draft',
      testing: 'Testing',
      active: 'Active',
      deprecated: 'Deprecated'
    };
    return names[status] || status;
  }

  getTechniqueTypeColor(type: TechniqueType): string {
    const colors: Record<TechniqueType, string> = {
      transformation: 'blue',
      validation: 'green',
      execution: 'orange',
      composition: 'purple'
    };
    return colors[type];
  }

  getStatusColor(status: TechniqueStatus): string {
    const colors: Record<TechniqueStatus, string> = {
      draft: 'gray',
      testing: 'yellow',
      active: 'green',
      deprecated: 'red'
    };
    return colors[status];
  }
}

export const techniqueBuilderService = new TechniqueBuilderService();
