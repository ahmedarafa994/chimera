/**
 * Custom Technique Builder Service
 *
 * Phase 3 enterprise feature for advanced users:
 * - Visual interface for creating custom transformations
 * - Drag-and-drop technique combination
 * - Team sharing and version control
 * - Effectiveness tracking over time
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';

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

  // Validation
  min_value?: number;
  max_value?: number;
  allowed_values?: any[];
  pattern?: string; // regex pattern for string validation
}

export interface TechniqueStep {
  step_id: string;
  name: string;
  description: string;
  step_type: string; // "transform", "validate", "execute", "branch", "loop"

  // Step configuration
  implementation: Record<string, any>; // Code or configuration for this step
  parameters: Record<string, any>;

  // Flow control
  next_step_id?: string;
  condition?: string; // condition for branching

  // UI positioning (for visual editor)
  position_x: number;
  position_y: number;
}

export interface CustomTechnique {
  technique_id: string;
  name: string;
  description: string;
  category: string;
  technique_type: TechniqueType;

  // Authorship
  created_by: string;
  created_by_name: string;
  workspace_id?: string;
  visibility: VisibilityLevel;

  // Versioning
  version: string;
  parent_technique_id?: string;

  // Timing
  created_at: string;
  updated_at: string;

  // Status
  status: TechniqueStatus;

  // Configuration
  parameters: TechniqueParameter[];
  steps: TechniqueStep[];

  // Metadata
  tags: string[];
  complexity_score: number; // 1-10 complexity rating
  estimated_execution_time: number; // seconds

  // Usage statistics
  usage_count: number;
  success_rate: number;
  average_execution_time: number;

  // Dependencies
  required_techniques: string[];
  compatible_models: string[];
}

export interface TechniqueTemplate {
  template_id: string;
  name: string;
  description: string;
  category: string;

  // Template content
  parameters_template: TechniqueParameter[];
  steps_template: TechniqueStep[];

  // Usage info
  usage_count: number;
  created_by: string;
  created_at: string;
}

export interface TechniqueExecution {
  execution_id: string;
  technique_id: string;
  version: string;

  // Input
  input_parameters: Record<string, any>;
  test_input: string;

  // Results
  success: boolean;
  output: string;
  error_message?: string;
  execution_time: number;

  // Context
  executed_by: string;
  executed_at: string;
  model_provider?: string;
  model_name?: string;
}

export interface TechniqueCreate {
  name: string;
  description: string;
  category: string;
  technique_type?: TechniqueType;
  visibility?: VisibilityLevel;
  workspace_id?: string;
  tags?: string[];
}

export interface TechniqueUpdate {
  name?: string;
  description?: string;
  category?: string;
  status?: TechniqueStatus;
  visibility?: VisibilityLevel;
  parameters?: TechniqueParameter[];
  steps?: TechniqueStep[];
  tags?: string[];
}

export interface TechniqueTestRequest {
  test_input: string;
  parameters?: Record<string, any>;
  model_provider?: string;
  model_name?: string;
}

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

export interface TechniqueStatsResponse {
  technique_id: string;
  executions: TechniqueExecution[];
  success_rate: number;
  average_execution_time: number;
  usage_by_date: Record<string, number>;
  performance_trends: Record<string, any>;
}

export interface TechniqueListParams {
  page?: number;
  page_size?: number;
  category?: string;
  technique_type?: TechniqueType;
  status?: TechniqueStatus;
  visibility?: VisibilityLevel;
  search?: string;
}

class TechniqueBuilderService {
  private readonly baseUrl = '/techniques';

  /**
   * Create a new custom technique
   */
  async createTechnique(techniqueData: TechniqueCreate): Promise<CustomTechnique> {
    try {
      const response = await apiClient.post<CustomTechnique>(`${this.baseUrl}/`, {
        name: techniqueData.name,
        description: techniqueData.description,
        category: techniqueData.category,
        technique_type: techniqueData.technique_type || 'transformation',
        visibility: techniqueData.visibility || 'private',
        workspace_id: techniqueData.workspace_id,
        tags: techniqueData.tags || []
      });

      toast.success('Custom technique created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create technique:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create technique');
      throw error;
    }
  }

  /**
   * List custom techniques with filtering and pagination
   */
  async listTechniques(params?: TechniqueListParams): Promise<TechniqueListResponse> {
    try {
      const response = await apiClient.get<TechniqueListResponse>(`${this.baseUrl}/`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          category: params?.category,
          technique_type: params?.technique_type,
          status: params?.status,
          visibility: params?.visibility,
          search: params?.search
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list techniques:', error);
      toast.error('Failed to load techniques');
      throw error;
    }
  }

  /**
   * Get technique details
   */
  async getTechnique(techniqueId: string): Promise<CustomTechnique> {
    try {
      const response = await apiClient.get<CustomTechnique>(`${this.baseUrl}/${techniqueId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get technique:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load technique');
      throw error;
    }
  }

  /**
   * Update technique
   */
  async updateTechnique(techniqueId: string, updateData: TechniqueUpdate): Promise<CustomTechnique> {
    try {
      const response = await apiClient.patch<CustomTechnique>(`${this.baseUrl}/${techniqueId}`, updateData);
      toast.success('Technique updated successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to update technique:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to update technique');
      throw error;
    }
  }

  /**
   * Test technique execution
   */
  async testTechnique(techniqueId: string, testRequest: TechniqueTestRequest): Promise<TechniqueExecution> {
    try {
      const response = await apiClient.post<TechniqueExecution>(`${this.baseUrl}/${techniqueId}/test`, {
        test_input: testRequest.test_input,
        parameters: testRequest.parameters || {},
        model_provider: testRequest.model_provider,
        model_name: testRequest.model_name
      });

      toast.success('Technique test completed');
      return response.data;
    } catch (error) {
      console.error('Failed to test technique:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to test technique');
      throw error;
    }
  }

  /**
   * Get technique usage statistics
   */
  async getTechniqueStats(techniqueId: string): Promise<TechniqueStatsResponse> {
    try {
      const response = await apiClient.get<TechniqueStatsResponse>(`${this.baseUrl}/${techniqueId}/stats`);
      return response.data;
    } catch (error) {
      console.error('Failed to get technique stats:', error);
      toast.error('Failed to load technique statistics');
      throw error;
    }
  }

  /**
   * Get technique template
   */
  async getTechniqueTemplate(templateId: string): Promise<TechniqueTemplate> {
    try {
      const response = await apiClient.get<TechniqueTemplate>(`${this.baseUrl}/templates/${templateId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get template:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load template');
      throw error;
    }
  }

  /**
   * Clone existing technique
   */
  async cloneTechnique(techniqueId: string): Promise<CustomTechnique> {
    try {
      const response = await apiClient.post<CustomTechnique>(`${this.baseUrl}/${techniqueId}/clone`);
      toast.success('Technique cloned successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to clone technique:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to clone technique');
      throw error;
    }
  }

  /**
   * Delete technique
   */
  async deleteTechnique(techniqueId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/${techniqueId}`);
      toast.success('Technique deleted successfully');
    } catch (error) {
      console.error('Failed to delete technique:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete technique');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for technique type
   */
  getTechniqueTypeDisplayName(type: TechniqueType): string {
    const displayNames: Record<TechniqueType, string> = {
      transformation: 'Transformation',
      validation: 'Validation',
      execution: 'Execution',
      composition: 'Composition'
    };
    return displayNames[type];
  }

  /**
   * Get color for technique type
   */
  getTechniqueTypeColor(type: TechniqueType): string {
    const colors: Record<TechniqueType, string> = {
      transformation: 'blue',
      validation: 'green',
      execution: 'orange',
      composition: 'purple'
    };
    return colors[type];
  }

  /**
   * Get display name for technique status
   */
  getStatusDisplayName(status: TechniqueStatus): string {
    const displayNames: Record<TechniqueStatus, string> = {
      draft: 'Draft',
      testing: 'Testing',
      active: 'Active',
      deprecated: 'Deprecated'
    };
    return displayNames[status];
  }

  /**
   * Get color for technique status
   */
  getStatusColor(status: TechniqueStatus): string {
    const colors: Record<TechniqueStatus, string> = {
      draft: 'gray',
      testing: 'yellow',
      active: 'green',
      deprecated: 'red'
    };
    return colors[status];
  }

  /**
   * Get display name for visibility level
   */
  getVisibilityDisplayName(visibility: VisibilityLevel): string {
    const displayNames: Record<VisibilityLevel, string> = {
      private: 'Private',
      team: 'Team',
      public: 'Public'
    };
    return displayNames[visibility];
  }

  /**
   * Get color for visibility level
   */
  getVisibilityColor(visibility: VisibilityLevel): string {
    const colors: Record<VisibilityLevel, string> = {
      private: 'gray',
      team: 'blue',
      public: 'green'
    };
    return colors[visibility];
  }

  /**
   * Get display name for parameter type
   */
  getParameterTypeDisplayName(type: ParameterType): string {
    const displayNames: Record<ParameterType, string> = {
      string: 'Text',
      integer: 'Number (Integer)',
      float: 'Number (Decimal)',
      boolean: 'True/False',
      list: 'List',
      object: 'Object'
    };
    return displayNames[type];
  }

  /**
   * Format complexity score
   */
  formatComplexityScore(score: number): string {
    const levels = ['Very Simple', 'Simple', 'Easy', 'Moderate', 'Complex', 'Advanced', 'Expert', 'Very Complex', 'Extremely Complex', 'Master Level'];
    return levels[Math.min(score - 1, levels.length - 1)] || 'Unknown';
  }

  /**
   * Format execution time
   */
  formatExecutionTime(seconds: number): string {
    if (seconds < 1) {
      return `${Math.round(seconds * 1000)}ms`;
    } else if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    }
  }

  /**
   * Format success rate
   */
  formatSuccessRate(rate: number): string {
    return `${Math.round(rate * 100)}%`;
  }

  /**
   * Validate technique creation data
   */
  validateTechniqueCreate(data: TechniqueCreate): string[] {
    const errors: string[] = [];

    if (!data.name || data.name.trim().length === 0) {
      errors.push('Technique name is required');
    }

    if (data.name && data.name.length > 100) {
      errors.push('Technique name must be less than 100 characters');
    }

    if (!data.description || data.description.trim().length === 0) {
      errors.push('Description is required');
    }

    if (data.description && data.description.length > 1000) {
      errors.push('Description must be less than 1000 characters');
    }

    if (!data.category || data.category.trim().length === 0) {
      errors.push('Category is required');
    }

    return errors;
  }

  /**
   * Validate test request
   */
  validateTestRequest(data: TechniqueTestRequest): string[] {
    const errors: string[] = [];

    if (!data.test_input || data.test_input.trim().length === 0) {
      errors.push('Test input is required');
    }

    return errors;
  }

  /**
   * Generate step ID
   */
  generateStepId(): string {
    return `step_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Create default parameter
   */
  createDefaultParameter(): TechniqueParameter {
    return {
      name: 'new_parameter',
      display_name: 'New Parameter',
      description: 'Parameter description',
      parameter_type: 'string',
      required: true
    };
  }

  /**
   * Create default step
   */
  createDefaultStep(position_x: number = 100, position_y: number = 100): TechniqueStep {
    return {
      step_id: this.generateStepId(),
      name: 'New Step',
      description: 'Step description',
      step_type: 'transform',
      implementation: {
        type: 'text_manipulation',
        operation: 'identity'
      },
      parameters: {},
      position_x,
      position_y
    };
  }

  /**
   * Get available step types
   */
  getStepTypes(): Array<{id: string, name: string, description: string}> {
    return [
      {
        id: 'transform',
        name: 'Transform',
        description: 'Modify or transform the input text'
      },
      {
        id: 'validate',
        name: 'Validate',
        description: 'Check input against criteria'
      },
      {
        id: 'execute',
        name: 'Execute',
        description: 'Run the transformation on target model'
      },
      {
        id: 'branch',
        name: 'Branch',
        description: 'Conditional logic based on input'
      },
      {
        id: 'loop',
        name: 'Loop',
        description: 'Repeat operations multiple times'
      }
    ];
  }

  /**
   * Get available categories
   */
  getCategories(): Array<{id: string, name: string, description: string}> {
    return [
      {
        id: 'injection',
        name: 'Prompt Injection',
        description: 'Techniques for injecting malicious prompts'
      },
      {
        id: 'transformation',
        name: 'Text Transformation',
        description: 'Modify text while preserving meaning'
      },
      {
        id: 'evasion',
        name: 'Defense Evasion',
        description: 'Bypass content filters and safety measures'
      },
      {
        id: 'jailbreak',
        name: 'Jailbreak',
        description: 'Break out of AI safety constraints'
      },
      {
        id: 'social',
        name: 'Social Engineering',
        description: 'Manipulate through social techniques'
      },
      {
        id: 'encoding',
        name: 'Encoding/Obfuscation',
        description: 'Hide malicious content through encoding'
      }
    ];
  }
}

// Export singleton instance
export const techniqueBuilderService = new TechniqueBuilderService();
