import { apiClient } from '../client';
import { toast } from 'sonner';

export type ExperimentStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type CitationStyle = 'apa' | 'mla' | 'ieee' | 'acm';
export type ReportFormat = 'pdf' | 'latex' | 'docx' | 'html';
export type EffectSize = 'small' | 'medium' | 'large';

export interface FitnessFunction {
  id: string;
  name: string;
  description: string;
  code: string;
  input_parameters: string[];
  output_type: 'float' | 'boolean' | 'score';
  created_at: string;
  updated_at: string;
  is_validated: boolean;
  workspace_id?: string;
  validation_results?: Record<string, any>;
}

export interface ExperimentDesign {
  id: string;
  experiment_id?: string; // alias for id for backward compatibility
  title: string;
  description: string;
  research_question: string;
  hypothesis: string;
  status: ExperimentStatus;
  control_technique: string;
  treatment_techniques: string[];
  target_models: string[];
  test_datasets: string[];
  primary_fitness_function: string;
  sample_size: number;
  created_at: string;
  updated_at: string;
  workspace_id?: string;
  owner_id: string;
}

export interface ExperimentExecution {
  id: string;
  experiment_id: string;
  status: ExperimentStatus;
  started_at: string | null;
  completed_at?: string;
  progress?: number;
  results?: Record<string, any>;
  logs?: string[];
}

export interface ResearchReport {
  id: string;
  experiment_id: string;
  title: string;
  authors: string[];
  abstract: string;
  introduction: string;
  methodology: string;
  results: string;
  discussion: string;
  conclusion: string;
  references: string[];
  created_at: string;
  citation_style: CitationStyle;
  is_published: boolean;
}

export interface TechniqueVariation {
  id: string;
  base_technique_id: string;
  name: string;
  description: string;
  parameter_modifications: Record<string, any>;
  research_rationale: string;
  expected_outcome: string;
  novelty_score: number;
  created_at: string;
}

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
  experiments_by_month: Array<{ month: string; experiments: number }>;
  top_performing_techniques: Array<{
    technique: string;
    avg_success_rate: number;
    experiments: number;
  }>;
  significance_rate: number;
  replication_success_rate: number;
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

  private mapFitnessFunction(fn: any): FitnessFunction {
    return {
      id: fn.function_id ?? fn.id ?? '',
      name: fn.name,
      description: fn.description,
      code: fn.code,
      input_parameters: fn.input_parameters ?? [],
      output_type: fn.output_type,
      created_at: fn.created_at ?? new Date().toISOString(),
      updated_at: fn.updated_at ?? new Date().toISOString(),
      is_validated: Boolean(fn.is_validated),
      workspace_id: fn.workspace_id,
      validation_results: fn.validation_results
    };
  }

  private mapExperiment(exp: any): ExperimentDesign {
    return {
      id: exp.experiment_id ?? exp.id ?? '',
      title: exp.title,
      description: exp.description,
      research_question: exp.research_question,
      hypothesis: exp.hypothesis,
      status: exp.status,
      control_technique: exp.control_technique,
      treatment_techniques: exp.treatment_techniques ?? [],
      target_models: exp.target_models ?? [],
      test_datasets: exp.test_datasets ?? [],
      primary_fitness_function: exp.primary_fitness_function,
      sample_size: exp.sample_size ?? 0,
      created_at: exp.created_at ?? new Date().toISOString(),
      updated_at: exp.updated_at ?? new Date().toISOString(),
      workspace_id: exp.workspace_id,
      owner_id: exp.created_by ?? exp.owner_id ?? ''
    };
  }

  private mapExecution(exec: any): ExperimentExecution {
    return {
      id: exec.execution_id ?? exec.id ?? '',
      experiment_id: exec.experiment_id,
      status: exec.status as ExperimentStatus,
      started_at: exec.started_at ?? null,
      completed_at: exec.completed_at,
      progress: exec.progress,
      results: exec.results,
      logs: exec.logs
    };
  }

  private mapReport(report: any): ResearchReport {
    return {
      id: report.report_id ?? report.id ?? '',
      experiment_id: report.experiment_id,
      title: report.title,
      authors: report.authors ?? [],
      abstract: report.abstract,
      introduction: report.introduction,
      methodology: report.methodology,
      results: report.results,
      discussion: report.discussion,
      conclusion: report.conclusion,
      references: report.references ?? [],
      created_at: report.generated_at ?? new Date().toISOString(),
      citation_style: (report.citation_style as CitationStyle) ?? 'apa',
      is_published: Boolean(report.is_published ?? false)
    };
  }

  async createFitnessFunction(functionData: FitnessFunctionCreate): Promise<FitnessFunction> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/fitness-functions`, functionData);
      toast.success('Fitness function created successfully');
      return this.mapFitnessFunction(response.data);
    } catch (error) {
      console.error('Failed to create fitness function:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create fitness function');
      throw error;
    }
  }

  async listFitnessFunctions(params?: FitnessFunctionListParams): Promise<FitnessFunctionListResponse> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/fitness-functions`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          workspace_id: params?.workspace_id,
          validated_only: params?.validated_only
        }
      });
      const data = response.data;
      return {
        functions: (data.functions ?? []).map((fn: any) => this.mapFitnessFunction(fn)),
        total: data.total ?? data.functions?.length ?? 0,
        page: data.page ?? 1,
        page_size: data.page_size ?? params?.page_size ?? 20,
        has_next: Boolean(data.has_next),
        has_prev: Boolean(data.has_prev)
      };
    } catch (error) {
      console.error('Failed to list fitness functions:', error);
      toast.error('Failed to load fitness functions');
      throw error;
    }
  }

  async createExperiment(experimentData: ExperimentDesignCreate): Promise<ExperimentDesign> {
    try {
      const response = await apiClient.post(`${this.baseUrl}/experiments`, experimentData);
      toast.success('Experiment created successfully');
      return this.mapExperiment(response.data);
    } catch (error) {
      console.error('Failed to create experiment:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create experiment');
      throw error;
    }
  }

  async listExperiments(params?: ExperimentListParams): Promise<ExperimentListResponse> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/experiments`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          workspace_id: params?.workspace_id,
          status: params?.status
        }
      });
      const data = response.data;
      return {
        experiments: (data.experiments ?? []).map((exp: any) => this.mapExperiment(exp)),
        total: data.total ?? data.experiments?.length ?? 0,
        page: data.page ?? 1,
        page_size: data.page_size ?? params?.page_size ?? 20,
        has_next: Boolean(data.has_next),
        has_prev: Boolean(data.has_prev)
      };
    } catch (error) {
      console.error('Failed to list experiments:', error);
      toast.error('Failed to load experiments');
      throw error;
    }
  }

  async getExperiment(experimentId: string): Promise<ExperimentDesign> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/experiments/${experimentId}`);
      return this.mapExperiment(response.data);
    } catch (error) {
      console.error('Failed to get experiment:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to load experiment');
      throw error;
    }
  }

  async executeExperiment(
    experimentId: string,
    config: ExperimentExecutionTrigger
  ): Promise<{ message: string; execution_id: string; estimated_completion: string }> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/experiments/${experimentId}/execute`,
        config
      );
      toast.success('Experiment execution started');
      return response.data;
    } catch (error) {
      console.error('Failed to execute experiment:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to execute experiment');
      throw error;
    }
  }

  async listExperimentExecutions(
    experimentId: string,
    params?: ExecutionListParams
  ): Promise<ExecutionListResponse> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/experiments/${experimentId}/executions`, {
        params: {
          page: params?.page,
          page_size: params?.page_size
        }
      });
      const data = response.data;
      return {
        executions: (data.executions ?? []).map((exec: any) => this.mapExecution(exec)),
        total: data.total ?? data.executions?.length ?? 0,
        page: data.page ?? 1,
        page_size: data.page_size ?? params?.page_size ?? 20,
        has_next: Boolean(data.has_next),
        has_prev: Boolean(data.has_prev)
      };
    } catch (error) {
      console.error('Failed to list executions:', error);
      toast.error('Failed to load executions');
      throw error;
    }
  }

  async getExecutionResults(executionId: string): Promise<ExperimentExecution> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/executions/${executionId}`);
      return this.mapExecution(response.data);
    } catch (error) {
      console.error('Failed to get execution results:', error);
      toast.error('Failed to load execution results');
      throw error;
    }
  }

  async generateResearchReport(
    experimentId: string,
    reportData: ResearchReportGenerate
  ): Promise<ResearchReport> {
    try {
      const response = await apiClient.post(
        `${this.baseUrl}/experiments/${experimentId}/reports`,
        reportData
      );
      toast.success('Research report generated successfully');
      return this.mapReport(response.data);
    } catch (error) {
      console.error('Failed to generate research report:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate research report');
      throw error;
    }
  }

  async exportResearchReport(
    reportId: string,
    format: ReportFormat
  ): Promise<{ download_url: string; filename: string; size_bytes: number; expires_at: string }> {
    try {
      const response = await apiClient.get(
        `${this.baseUrl}/reports/${reportId}/export/${format}`
      );
      toast.success(`Report exported as ${format.toUpperCase()}`);
      return response.data;
    } catch (error) {
      console.error('Failed to export research report:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to export research report');
      throw error;
    }
  }

  async getResearchAnalytics(workspaceId?: string): Promise<ResearchAnalytics> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/analytics`, {
        params: { workspace_id: workspaceId }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get research analytics:', error);
      toast.error('Failed to load research analytics');
      throw error;
    }
  }

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
      const response = await apiClient.post(`${this.baseUrl}/technique-variations`, variationData);
      toast.success('Technique variation created successfully');
      const variation = response.data;
      return {
        id: variation.variation_id ?? variation.id ?? '',
        base_technique_id: variation.base_technique_id,
        name: variation.name,
        description: variation.description,
        parameter_modifications: variation.parameter_modifications ?? {},
        research_rationale: variation.research_rationale,
        expected_outcome: variation.expected_outcome,
        novelty_score: variation.novelty_score ?? 0,
        created_at: variation.created_at ?? new Date().toISOString()
      };
    } catch (error) {
      console.error('Failed to create technique variation:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create technique variation');
      throw error;
    }
  }

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

  getCitationStyleDisplayName(style: CitationStyle): string {
    const displayNames: Record<CitationStyle, string> = {
      apa: 'APA (American Psychological Association)',
      mla: 'MLA (Modern Language Association)',
      ieee: 'IEEE (Institute of Electrical and Electronics Engineers)',
      acm: 'ACM (Association for Computing Machinery)'
    };
    return displayNames[style];
  }

  getReportFormatDisplayName(format: ReportFormat): string {
    const displayNames: Record<ReportFormat, string> = {
      pdf: 'PDF Document',
      latex: 'LaTeX Source',
      docx: 'Microsoft Word Document',
      html: 'HTML Web Page'
    };
    return displayNames[format];
  }

  formatEffectSize(effectSize: number): { size: EffectSize; description: string } {
    if (effectSize < 0.2) return { size: 'small', description: 'Small effect' };
    if (effectSize < 0.5) return { size: 'medium', description: 'Medium effect' };
    return { size: 'large', description: 'Large effect' };
  }

  formatPValue(pValue: number): string {
    if (pValue < 0.001) return 'p < 0.001';
    if (pValue < 0.01) return 'p < 0.01';
    if (pValue < 0.05) return 'p < 0.05';
    return `p = ${pValue.toFixed(3)}`;
  }

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

  validateExperimentDesign(data: ExperimentDesignCreate): string[] {
    const errors: string[] = [];
    if (!data.title?.trim()) errors.push('Experiment title is required');
    if (data.title && data.title.length > 200) errors.push('Experiment title must be less than 200 characters');
    if (!data.research_question?.trim()) errors.push('Research question is required');
    if (!data.hypothesis?.trim()) errors.push('Hypothesis is required');
    if (!data.control_technique) errors.push('Control technique is required');
    if (!data.treatment_techniques?.length) errors.push('At least one treatment technique is required');
    if (!data.target_models?.length) errors.push('At least one target model is required');
    if (!data.test_datasets?.length) errors.push('At least one test dataset is required');
    if (!data.primary_fitness_function) errors.push('Primary fitness function is required');
    if (data.sample_size && (data.sample_size < 10 || data.sample_size > 10000)) {
      errors.push('Sample size must be between 10 and 10,000');
    }
    return errors;
  }

  validateFitnessFunctionCreate(data: FitnessFunctionCreate): string[] {
    const errors: string[] = [];
    if (!data.name?.trim()) errors.push('Function name is required');
    if (data.name && data.name.length > 100) errors.push('Function name must be less than 100 characters');
    if (!data.description?.trim()) errors.push('Function description is required');
    if (!data.code?.trim()) errors.push('Function code is required');
    if (!data.input_parameters?.length) errors.push('At least one input parameter is required');
    if (!['float', 'boolean', 'score'].includes(data.output_type)) {
      errors.push('Output type must be float, boolean, or score');
    }
    return errors;
  }
}

export const researchLabService = new ResearchLabService();
