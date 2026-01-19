/**
 * Adversarial Attack Research Lab Service
 *
 * Phase 4 innovation feature for academic research:
 * - A/B testing framework for technique variations
 * - Custom fitness function editor
 * - Academic paper format exports
 * - Research paper citation linking
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';

export type StatisticalSignificance = number; // 0.0 to 1.0
export type EffectSize = 'small' | 'medium' | 'large';
export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type CitationType = 'background' | 'methodology' | 'comparison' | 'inspiration';
export type ReportFormat = 'pdf' | 'latex' | 'docx' | 'html';
export type CitationStyle = 'apa' | 'mla' | 'ieee' | 'acm';

export interface FitnessFunction {
  function_id: string;
  name: string;
  description: string;
  code: string;
  input_parameters: string[];
  output_type: 'float' | 'boolean' | 'score';

  // Validation and testing
  is_validated: boolean;
  validation_results?: {
    is_valid: boolean;
    syntax_errors?: string[];
    security_warnings?: string[];
    performance?: string;
    estimated_complexity?: string;
  };

  // Metadata
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface ExperimentDesign {
  experiment_id: string;
  title: string;
  description: string;
  research_question: string;
  hypothesis: string;

  // Experimental setup
  control_technique: string;
  treatment_techniques: string[];
  target_models: string[];
  test_datasets: string[];

  // A/B testing configuration
  sample_size: number;
  confidence_level: number;
  statistical_power: number;

  // Fitness evaluation
  primary_fitness_function: string;
  secondary_fitness_functions: string[];

  // Randomization and controls
  randomization_strategy: string;
  control_variables: string[];

  // Metadata
  created_by: string;
  workspace_id?: string;
  created_at: string;
  updated_at: string;
}

export interface ExperimentExecution {
  execution_id: string;
  experiment_id: string;

  // Execution details
  started_at?: string;
  completed_at?: string;
  status: ExperimentStatus;

  // Progress tracking
  total_tests: number;
  completed_tests: number;
  failed_tests: number;

  // Results
  control_results: Record<string, any>;
  treatment_results: Record<string, any[]>;

  // Statistical analysis
  statistical_significance?: number;
  effect_size?: number;
  p_value?: number;
  confidence_intervals: Record<string, any>;

  // Analysis results
  winning_technique?: string;
  performance_rankings: Array<{
    technique: string;
    score: number;
    rank: number;
  }>;
  detailed_analysis: Record<string, any>;

  // Error tracking
  error_message?: string;

  // Metadata
  executed_by: string;
  execution_time_seconds?: number;
}

export interface ResearchReport {
  report_id: string;
  experiment_id: string;

  // Report metadata
  title: string;
  authors: string[];
  abstract: string;
  keywords: string[];

  // Report structure
  introduction: string;
  methodology: string;
  results: string;
  discussion: string;
  conclusion: string;

  // Academic formatting
  citation_style: CitationStyle;
  references: Array<{
    title: string;
    authors: string[];
    journal: string;
    year: number;
    doi?: string;
    arxiv_id?: string;
    url?: string;
  }>;
  appendices: Record<string, string>;

  // Figures and tables
  figures: Array<{
    number: number;
    title: string;
    description: string;
    data_source: string;
  }>;
  tables: Array<{
    number: number;
    title: string;
    description: string;
    data_source: string;
  }>;

  // Export formats
  available_formats: ReportFormat[];

  // Metadata
  generated_by: string;
  generated_at: string;
}

export interface TechniqueVariation {
  variation_id: string;
  base_technique_id: string;
  name: string;
  description: string;

  // Variation parameters
  parameter_modifications: Record<string, any>;
  code_modifications?: string;

  // Research metadata
  research_rationale: string;
  expected_outcome: string;
  novelty_score: number;

  // Performance tracking
  success_rate: number;
  average_fitness_score: number;
  execution_count: number;

  // Metadata
  created_by: string;
  created_at: string;
}

export interface CitationLink {
  citation_id: string;
  title: string;
  authors: string[];
  journal: string;
  year: number;
  doi?: string;
  arxiv_id?: string;
  url?: string;

  // Citation context
  relevance_score: number;
  citation_type: CitationType;
  notes?: string;

  // Linked experiments
  linked_experiments: string[];

  // Metadata
  added_by: string;
  added_at: string;
}

// Request/Response interfaces

export interface FitnessFunctionCreate {
  name: string;
  description: string;
  code: string;
  input_parameters: string[];
  output_type: 'float' | 'boolean' | 'score';
}

export interface FitnessFunctionUpdate {
  name?: string;
  description?: string;
  code?: string;
  input_parameters?: string[];
  output_type?: 'float' | 'boolean' | 'score';
}

export interface ExperimentDesignCreate {
  title: string;
  description: string;
  research_question: string;
  hypothesis: string;
  control_technique: string;
  treatment_techniques: string[];
  target_models: string[];
  test_datasets: string[];
  primary_fitness_function: string;
  sample_size?: number;
  workspace_id?: string;
}

export interface ExperimentExecutionTrigger {
  parallel_execution: boolean;
  max_concurrent_tests: number;
  timeout_seconds: number;
}

export interface ResearchReportGenerate {
  title: string;
  authors: string[];
  abstract: string;
  keywords: string[];
  citation_style?: CitationStyle;
}

export interface FitnessFunctionListResponse {
  functions: FitnessFunction[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ExperimentListResponse {
  experiments: ExperimentDesign[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ExecutionListResponse {
  executions: ExperimentExecution[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ResearchAnalytics {
  total_experiments: number;
  completed_experiments: number;
  total_techniques_tested: number;
  average_effect_size: number;

  // Research productivity
  experiments_by_month: Array<{ month: string; experiments: number }>;
  top_performing_techniques: Array<{
    technique: string;
    avg_success_rate: number;
    experiments: number;
  }>;

  // Statistical insights
  significance_rate: number;
  replication_success_rate: number;

  // Collaboration metrics
  active_researchers: number;
  cross_workspace_collaborations: number;
}

export interface ExperimentListParams {
  page?: number;
  page_size?: number;
  workspace_id?: string;
  status?: ExperimentStatus;
}

export interface ExecutionListParams {
  page?: number;
  page_size?: number;
}

export interface FitnessFunctionListParams {
  page?: number;
  page_size?: number;
  workspace_id?: string;
  validated_only?: boolean;
}

class ResearchLabService {
  private readonly baseUrl = '/research-lab';

  /**
   * Create a new custom fitness function
   */
  async createFitnessFunction(functionData: FitnessFunctionCreate): Promise<FitnessFunction> {
    try {
      const response = await apiClient.post<FitnessFunction>(`${this.baseUrl}/fitness-functions`, functionData);

      toast.success('Fitness function created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create fitness function:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create fitness function');
      throw error;
    }
  }

  /**
   * List available fitness functions
   */
  async listFitnessFunctions(params?: FitnessFunctionListParams): Promise<FitnessFunctionListResponse> {
    try {
      const response = await apiClient.get<FitnessFunctionListResponse>(`${this.baseUrl}/fitness-functions`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          workspace_id: params?.workspace_id,
          validated_only: params?.validated_only ? 'true' : undefined
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list fitness functions:', error);
      toast.error('Failed to load fitness functions');
      throw error;
    }
  }

  /**
   * Create a new experiment design
   */
  async createExperiment(experimentData: ExperimentDesignCreate): Promise<ExperimentDesign> {
    try {
      const response = await apiClient.post<ExperimentDesign>(`${this.baseUrl}/experiments`, experimentData);

      toast.success('Experiment created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create experiment:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create experiment');
      throw error;
    }
  }

  /**
   * List research experiments
   */
  async listExperiments(params?: ExperimentListParams): Promise<ExperimentListResponse> {
    try {
      const response = await apiClient.get<ExperimentListResponse>(`${this.baseUrl}/experiments`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          workspace_id: params?.workspace_id,
          status: params?.status
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list experiments:', error);
      toast.error('Failed to load experiments');
      throw error;
    }
  }

  /**
   * Get experiment details
   */
  async getExperiment(experimentId: string): Promise<ExperimentDesign> {
    try {
      const response = await apiClient.get<ExperimentDesign>(`${this.baseUrl}/experiments/${experimentId}`);

      return response.data;
    } catch (error) {
      console.error('Failed to get experiment:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load experiment');
      throw error;
    }
  }

  /**
   * Execute A/B testing experiment
   */
  async executeExperiment(
    experimentId: string,
    config: ExperimentExecutionTrigger
  ): Promise<{ message: string; execution_id: string; estimated_completion: string }> {
    try {
      const response = await apiClient.post<{ message: string; execution_id: string; estimated_completion: string }>(`${this.baseUrl}/experiments/${experimentId}/execute`, config);

      toast.success('Experiment execution started');
      return response.data;
    } catch (error) {
      console.error('Failed to execute experiment:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to execute experiment');
      throw error;
    }
  }

  /**
   * List executions for an experiment
   */
  async listExperimentExecutions(
    experimentId: string,
    params?: ExecutionListParams
  ): Promise<ExecutionListResponse> {
    try {
      const response = await apiClient.get<ExecutionListResponse>(
        `${this.baseUrl}/experiments/${experimentId}/executions`,
        {
          params: {
            page: params?.page,
            page_size: params?.page_size
          }
        }
      );

      return response.data;
    } catch (error) {
      console.error('Failed to list executions:', error);
      toast.error('Failed to load executions');
      throw error;
    }
  }

  /**
   * Get detailed execution results
   */
  async getExecutionResults(executionId: string): Promise<ExperimentExecution> {
    try {
      const response = await apiClient.get<ExperimentExecution>(`${this.baseUrl}/executions/${executionId}`);

      return response.data;
    } catch (error) {
      console.error('Failed to get execution results:', error);
      toast.error('Failed to load execution results');
      throw error;
    }
  }

  /**
   * Generate research report
   */
  async generateResearchReport(
    experimentId: string,
    reportData: ResearchReportGenerate
  ): Promise<ResearchReport> {
    try {
      const response = await apiClient.post<ResearchReport>(`${this.baseUrl}/experiments/${experimentId}/reports`, reportData);

      toast.success('Research report generated successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to generate research report:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate research report');
      throw error;
    }
  }

  /**
   * Export research report in specified format
   */
  async exportResearchReport(
    reportId: string,
    format: ReportFormat
  ): Promise<{ download_url: string; filename: string; size_bytes: number; expires_at: string }> {
    try {
      const response = await apiClient.get<{ download_url: string; filename: string; size_bytes: number; expires_at: string }>(`${this.baseUrl}/reports/${reportId}/export/${format}`);

      toast.success(`Report exported as ${format.toUpperCase()}`);
      return response.data;
    } catch (error) {
      console.error('Failed to export research report:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to export research report');
      throw error;
    }
  }

  /**
   * Get research analytics
   */
  async getResearchAnalytics(workspaceId?: string): Promise<ResearchAnalytics> {
    try {
      const response = await apiClient.get<ResearchAnalytics>(`${this.baseUrl}/analytics`, {
        params: { workspace_id: workspaceId }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to get research analytics:', error);
      toast.error('Failed to load research analytics');
      throw error;
    }
  }

  /**
   * Create technique variation
   */
  async createTechniqueVariation(variationData: {
    base_technique_id: string;
    name: string;
    description: string;
    parameter_modifications?: Record<string, any>;
    research_rationale: string;
    expected_outcome: string;
    novelty_score?: number;
  }): Promise<TechniqueVariation> {
    try {
      const response = await apiClient.post<TechniqueVariation>(`${this.baseUrl}/technique-variations`, variationData);

      const result = await response.json();
      toast.success('Technique variation created successfully');
      return result;
    } catch (error) {
      console.error('Failed to create technique variation:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create technique variation');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for experiment status
   */
  getStatusDisplayName(status: ExperimentStatus): string {
    const displayNames: Record<ExperimentStatus, string> = {
      pending: 'Pending',
      running: 'Running',
      completed: 'Completed',
      failed: 'Failed',
      cancelled: 'Cancelled'
    };
    return displayNames[status];
  }

  /**
   * Get color for experiment status
   */
  getStatusColor(status: ExperimentStatus): string {
    const colors: Record<ExperimentStatus, string> = {
      pending: 'gray',
      running: 'blue',
      completed: 'green',
      failed: 'red',
      cancelled: 'orange'
    };
    return colors[status];
  }

  /**
   * Get display name for citation style
   */
  getCitationStyleDisplayName(style: CitationStyle): string {
    const displayNames: Record<CitationStyle, string> = {
      apa: 'APA (American Psychological Association)',
      mla: 'MLA (Modern Language Association)',
      ieee: 'IEEE (Institute of Electrical and Electronics Engineers)',
      acm: 'ACM (Association for Computing Machinery)'
    };
    return displayNames[style];
  }

  /**
   * Get display name for report format
   */
  getReportFormatDisplayName(format: ReportFormat): string {
    const displayNames: Record<ReportFormat, string> = {
      pdf: 'PDF Document',
      latex: 'LaTeX Source',
      docx: 'Microsoft Word Document',
      html: 'HTML Web Page'
    };
    return displayNames[format];
  }

  /**
   * Format effect size
   */
  formatEffectSize(effectSize: number): { size: EffectSize; description: string } {
    if (effectSize < 0.2) {
      return { size: 'small', description: 'Small effect' };
    } else if (effectSize < 0.5) {
      return { size: 'medium', description: 'Medium effect' };
    } else {
      return { size: 'large', description: 'Large effect' };
    }
  }

  /**
   * Format p-value for display
   */
  formatPValue(pValue: number): string {
    if (pValue < 0.001) {
      return 'p < 0.001';
    } else if (pValue < 0.01) {
      return `p < 0.01`;
    } else if (pValue < 0.05) {
      return `p < 0.05`;
    } else {
      return `p = ${pValue.toFixed(3)}`;
    }
  }

  /**
   * Interpret statistical significance
   */
  interpretSignificance(pValue: number, alpha: number = 0.05): {
    isSignificant: boolean;
    interpretation: string;
  } {
    const isSignificant = pValue < alpha;
    return {
      isSignificant,
      interpretation: isSignificant
        ? `Statistically significant (p = ${pValue.toFixed(3)} < α = ${alpha})`
        : `Not statistically significant (p = ${pValue.toFixed(3)} ≥ α = ${alpha})`
    };
  }

  /**
   * Validate experiment design
   */
  validateExperimentDesign(data: ExperimentDesignCreate): string[] {
    const errors: string[] = [];

    if (!data.title || data.title.trim().length === 0) {
      errors.push('Experiment title is required');
    }

    if (data.title && data.title.length > 200) {
      errors.push('Experiment title must be less than 200 characters');
    }

    if (!data.research_question || data.research_question.trim().length === 0) {
      errors.push('Research question is required');
    }

    if (!data.hypothesis || data.hypothesis.trim().length === 0) {
      errors.push('Hypothesis is required');
    }

    if (!data.control_technique) {
      errors.push('Control technique is required');
    }

    if (!data.treatment_techniques || data.treatment_techniques.length === 0) {
      errors.push('At least one treatment technique is required');
    }

    if (!data.target_models || data.target_models.length === 0) {
      errors.push('At least one target model is required');
    }

    if (!data.test_datasets || data.test_datasets.length === 0) {
      errors.push('At least one test dataset is required');
    }

    if (!data.primary_fitness_function) {
      errors.push('Primary fitness function is required');
    }

    if (data.sample_size && (data.sample_size < 10 || data.sample_size > 10000)) {
      errors.push('Sample size must be between 10 and 10,000');
    }

    return errors;
  }

  /**
   * Validate fitness function
   */
  validateFitnessFunctionCreate(data: FitnessFunctionCreate): string[] {
    const errors: string[] = [];

    if (!data.name || data.name.trim().length === 0) {
      errors.push('Function name is required');
    }

    if (data.name && data.name.length > 100) {
      errors.push('Function name must be less than 100 characters');
    }

    if (!data.description || data.description.trim().length === 0) {
      errors.push('Function description is required');
    }

    if (!data.code || data.code.trim().length === 0) {
      errors.push('Function code is required');
    }

    if (!data.input_parameters || data.input_parameters.length === 0) {
      errors.push('At least one input parameter is required');
    }

    if (!['float', 'boolean', 'score'].includes(data.output_type)) {
      errors.push('Output type must be float, boolean, or score');
    }

    return errors;
  }

  /**
   * Get suggested techniques for experimentation
   */
  getSuggestedTechniques(): Array<{
    id: string;
    name: string;
    category: string;
    description: string;
  }> {
    return [
      {
        id: 'gptfuzz_basic',
        name: 'GPTFuzz Basic',
        category: 'Evolutionary',
        description: 'Basic evolutionary fuzzing approach'
      },
      {
        id: 'autodan_v1',
        name: 'AutoDAN v1',
        category: 'Reasoning',
        description: 'AutoDAN with basic reasoning capabilities'
      },
      {
        id: 'autodan_v2',
        name: 'AutoDAN v2',
        category: 'Reasoning',
        description: 'Enhanced AutoDAN with improved reasoning'
      },
      {
        id: 'gradient_gcg',
        name: 'Gradient GCG',
        category: 'Gradient-based',
        description: 'Greedy coordinate gradient optimization'
      },
      {
        id: 'hotflip',
        name: 'HotFlip',
        category: 'Gradient-based',
        description: 'Hot flip gradient-based optimization'
      }
    ];
  }

  /**
   * Get available datasets
   */
  getAvailableDatasets(): Array<{
    id: string;
    name: string;
    description: string;
    size: number;
  }> {
    return [
      {
        id: 'harmless_dataset_v1',
        name: 'Harmless Dataset v1',
        description: 'Collection of harmless prompts for baseline testing',
        size: 1000
      },
      {
        id: 'jailbreak_prompts_v2',
        name: 'Jailbreak Prompts v2',
        description: 'Curated jailbreak attempts from literature',
        size: 500
      },
      {
        id: 'adversarial_examples_v3',
        name: 'Adversarial Examples v3',
        description: 'Known adversarial examples from research papers',
        size: 750
      }
    ];
  }
}

// Export singleton instance
export const researchLabService = new ResearchLabService();
