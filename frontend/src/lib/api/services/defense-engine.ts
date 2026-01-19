/**
 * Defense Recommendation Engine Service
 *
 * Phase 4 innovation feature for comprehensive security:
 * - Automated defensive measure suggestions
 * - Implementation guides for each defense
 * - Effectiveness and difficulty ratings
 * - Defense validation tracking
 */

import { apiClient } from '../client';

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
export type DefenseCategory = 'input_filtering' | 'output_monitoring' | 'session_management' | 'model_training' | 'infrastructure';
export type ImplementationDifficulty = 'easy' | 'medium' | 'hard' | 'expert';
export type PriorityLevel = 'low' | 'medium' | 'high' | 'critical';
export type ImplementationStatus = 'planned' | 'in_progress' | 'testing' | 'deployed' | 'failed';
export type SystemArchitecture = 'web_app' | 'api_service' | 'ml_pipeline' | 'chatbot' | 'mobile_app';
export type DeploymentEnvironment = 'development' | 'staging' | 'production';

export interface DefenseTechnique {
  defense_id: string;
  name: string;
  category: DefenseCategory;
  description: string;
  detailed_explanation: string;

  // Implementation details
  implementation_guide: string;
  code_examples: Record<string, string>;
  configuration_steps: string[];

  // Effectiveness metrics
  effectiveness_score: number;
  implementation_difficulty: ImplementationDifficulty;
  deployment_time_hours: number;

  // Applicability
  target_attack_vectors: string[];
  compatible_frameworks: string[];
  supported_languages: string[];

  // Evidence and validation
  research_citations: Array<{
    title: string;
    authors: string;
    journal: string;
  }>;
  real_world_deployments: number;
  community_rating: number;

  // Metadata
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface VulnerabilityAssessment {
  assessment_id: string;
  target_system: string;
  assessment_date: string;

  // Identified vulnerabilities
  vulnerabilities: Array<{
    type: string;
    severity: string;
    description: string;
    [key: string]: any;
  }>;
  risk_level: RiskLevel;
  attack_vectors_found: string[];

  // System context
  system_architecture: SystemArchitecture;
  technology_stack: string[];
  deployment_environment: DeploymentEnvironment;

  // Security posture
  existing_defenses: string[];
  security_gaps: string[];
  compliance_requirements: string[];

  // Metadata
  assessed_by: string;
  workspace_id?: string;
}

export interface DefenseRecommendation {
  recommendation_id: string;
  assessment_id: string;
  defense_id: string;
  recommended_order: number;

  // Recommendation context
  priority_level: PriorityLevel;
  justification: string;
  expected_risk_reduction: number;

  // Implementation planning
  estimated_implementation_time: number;
  required_expertise_level: string;
  implemented_by: string;
  reviewed_by?: string;
  approved_by?: string;

  // Metadata
  workspace_id?: string;
}

export interface DefenseMetrics {
  metrics_id: string;
  implementation_id: string;

  // Time period
  measurement_start: string;
  measurement_end: string;

  // Security metrics
  attacks_blocked: number;
  attacks_allowed: number;
  false_positives: number;
  false_negatives: number;

  // Performance metrics
  response_time_impact_ms?: number;
  throughput_impact_percent?: number;
  resource_usage_increase?: number;

  // Operational metrics
  maintenance_incidents: number;
  configuration_changes: number;
  downtime_minutes: number;

  // User experience
  user_complaints: number;
  usability_score?: number;

  // Metadata
  collected_by: string;
  collected_at: string;
}

export interface VulnerabilityAssessmentCreate {
  target_system: string;
  vulnerabilities: Array<Record<string, any>>;
  risk_level: RiskLevel;
  attack_vectors_found: string[];
  system_architecture: SystemArchitecture;
  technology_stack: string[];
  deployment_environment: DeploymentEnvironment;
  existing_defenses?: string[];
  compliance_requirements?: string[];
  workspace_id?: string;
}

export interface DefenseRecommendationRequest {
  assessment_id: string;
  max_recommendations?: number;
  priority_filter?: PriorityLevel;
  difficulty_preference?: ImplementationDifficulty;
  budget_constraints?: string;
}

export interface RecommendationListResponse {
  recommendations: DefenseRecommendation[];
  total: number;
}

export interface DefenseImplementation {
  implementation_id: string;
  recommendation_id: string;
  status: ImplementationStatus;
  notes?: string;
  config?: Record<string, any>;
}

export interface DefenseImplementationCreate {
  recommendation_id: string;
  implementation_notes?: string;
  configuration_used?: Record<string, any>;
  workspace_id?: string;
}

export interface DefenseImplementationUpdate {
  status?: ImplementationStatus;
  implementation_notes?: string;
  configuration_used?: Record<string, any>;
  validation_tests?: Array<Record<string, any>>;
  effectiveness_measured?: number;
  issues_encountered?: string[];
}

export interface ImplementationListResponse {
  implementations: DefenseImplementation[];
  total: number;
}

export interface DefenseMetricsCreate {
  implementation_id: string;
  measurement_start: string;
  measurement_end: string;
  attacks_blocked?: number;
  attacks_allowed?: number;
  false_positives?: number;
  false_negatives?: number;
  response_time_impact_ms?: number;
  throughput_impact_percent?: number;
}

export interface DefenseListResponse {
  defenses: DefenseTechnique[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface DefenseListParams {
  page?: number;
  page_size?: number;
  category?: DefenseCategory;
  difficulty?: ImplementationDifficulty;
  min_effectiveness?: number;
}

export interface ImplementationListParams {
  page?: number;
  page_size?: number;
  status?: ImplementationStatus;
  workspace_id?: string;
}

export interface DefenseAnalytics {
  total_assessments: number;
  total_recommendations: number;
  total_implementations: number;
  total_defenses_available: number;
  successful_deployments: number;
  average_effectiveness: number;
  overall_success_rate: number;
  implementations_by_month: Array<{ month: string; implementations: number }>;
  success_rate_by_defense: Record<string, number>;
  top_performing_defenses: Array<{
    defense: string;
    effectiveness: number;
    deployments: number;
  }>;
  most_deployed_defenses: Array<{
    defense: string;
    deployments: number;
    success_rate: number;
  }>;
  total_risk_reduced: number;
  vulnerabilities_addressed: number;
  average_implementation_time: number;
  cost_benefit_ratio: number;
}

export interface DefenseStatsResponse {
  total_assessments: number;
  total_recommendations: number;
  total_implementations: number;
  overall_success_rate: number;
  implementations_by_month: Array<{ month: string; implementations: number }>;
  success_rate_by_defense: Record<string, number>;
  top_performing_defenses: Array<{
    defense: string;
    effectiveness: number;
    deployments: number;
  }>;
  most_deployed_defenses: Array<{
    defense: string;
    deployments: number;
    success_rate: number;
  }>;
  total_risk_reduced: number;
  vulnerabilities_addressed: number;
  average_implementation_time: number;
  cost_benefit_ratio: number;
}

const API_BASE = '/defense-engine';

/**
 * Defense Recommendation Engine Service API
 */
export const defenseEngineService = {
  /**
   * Create a new vulnerability assessment
   */
  async createVulnerabilityAssessment(assessmentData: VulnerabilityAssessmentCreate): Promise<VulnerabilityAssessment> {
    const response = await apiClient.post<VulnerabilityAssessment>(`${API_BASE}/assessments`, assessmentData);
    return response.data;
  },

  /**
   * List vulnerability assessments
   */
  async listVulnerabilityAssessments(params?: {
    workspace_id?: string;
    risk_level?: RiskLevel;
    limit?: number;
  }): Promise<VulnerabilityAssessment[]> {
    const response = await apiClient.get<VulnerabilityAssessment[]>(`${API_BASE}/assessments`, {
      params
    });
    return response.data;
  },

  /**
   * Generate defense recommendations
   */
  async generateDefenseRecommendations(
    assessmentId: string,
    requestData: DefenseRecommendationRequest
  ): Promise<RecommendationListResponse> {
    const response = await apiClient.post<RecommendationListResponse>(
      `${API_BASE}/assessments/${assessmentId}/recommendations`,
      requestData
    );
    return response.data;
  },

  /**
   * List available defense techniques
   */
  async listDefenseTechniques(params?: DefenseListParams): Promise<DefenseListResponse> {
    const response = await apiClient.get<DefenseListResponse>(`${API_BASE}/defenses`, {
      params
    });
    return response.data;
  },

  /**
   * Get specific defense technique details
   */
  async getDefenseTechnique(defenseId: string): Promise<DefenseTechnique> {
    const response = await apiClient.get<DefenseTechnique>(`${API_BASE}/defenses/${defenseId}`);
    return response.data;
  },

  /**
   * Create a defense implementation
   */
  async createDefenseImplementation(implementationData: DefenseImplementationCreate): Promise<DefenseImplementation> {
    const response = await apiClient.post<DefenseImplementation>(`${API_BASE}/implementations`, implementationData);
    return response.data;
  },

  /**
   * List defense implementations
   */
  async listDefenseImplementations(params?: ImplementationListParams): Promise<ImplementationListResponse> {
    const response = await apiClient.get<ImplementationListResponse>(`${API_BASE}/implementations`, {
      params
    });
    return response.data;
  },

  /**
   * Update defense implementation
   */
  async updateDefenseImplementation(
    implementationId: string,
    updateData: DefenseImplementationUpdate
  ): Promise<DefenseImplementation> {
    const response = await apiClient.patch<DefenseImplementation>(
      `${API_BASE}/implementations/${implementationId}`,
      updateData
    );
    return response.data;
  },

  /**
   * Get defense effectiveness metrics
   */
  async getDefenseMetrics(implementationId: string): Promise<DefenseMetrics[]> {
    const response = await apiClient.get<DefenseMetrics[]>(`${API_BASE}/implementations/${implementationId}/metrics`);
    return response.data;
  },

  /**
   * Create new effectiveness metrics record
   */
  async recordMetrics(metricsData: DefenseMetricsCreate): Promise<DefenseMetrics> {
    const response = await apiClient.post<DefenseMetrics>(`${API_BASE}/metrics`, metricsData);
    return response.data;
  },

  /**
   * Get global defense engine statistics
   */
  async getStatistics(workspaceId?: string): Promise<DefenseStatsResponse> {
    const response = await apiClient.get<DefenseStatsResponse>(`${API_BASE}/stats`, {
      params: { workspace_id: workspaceId }
    });
    return response.data;
  },

  /**
   * Validate a specific defense implementation
   */
  async validateImplementation(implementationId: string): Promise<{
    success: boolean;
    findings: string[];
    recommendations: string[];
  }> {
    const response = await apiClient.post(`${API_BASE}/implementations/${implementationId}/validate`);
    return response.data as any;
  },

  /**
   * Get defense analytics data
   */
  async getDefenseAnalytics(workspaceId?: string): Promise<DefenseAnalytics> {
    const response = await apiClient.get<DefenseAnalytics>(`${API_BASE}/analytics`, {
      params: { workspace_id: workspaceId }
    });
    return response.data;
  },

  /**
   * Validate vulnerability assessment data
   */
  validateVulnerabilityAssessment(data: VulnerabilityAssessmentCreate): string[] {
    const errors: string[] = [];

    if (!data.target_system?.trim()) errors.push('Target system is required');
    if (!data.vulnerabilities?.length) errors.push('At least one vulnerability is required');
    if (!data.risk_level) errors.push('Risk level is required');
    if (!data.system_architecture) errors.push('System architecture is required');
    if (!data.technology_stack?.length) errors.push('At least one technology is required');
    if (!data.deployment_environment) errors.push('Deployment environment is required');

    return errors;
  },

  /**
   * Get risk level display name
   */
  getRiskLevelDisplayName(riskLevel: RiskLevel): string {
    const names: Record<RiskLevel, string> = {
      low: 'Low Risk',
      medium: 'Medium Risk',
      high: 'High Risk',
      critical: 'Critical Risk'
    };
    return names[riskLevel];
  },

  /**
   * Get risk level color for UI
   */
  getRiskLevelColor(riskLevel: RiskLevel): string {
    const colors: Record<RiskLevel, string> = {
      low: 'green',
      medium: 'yellow',
      high: 'orange',
      critical: 'red'
    };
    return colors[riskLevel];
  },

  /**
   * Get difficulty display name
   */
  getDifficultyDisplayName(difficulty: ImplementationDifficulty): string {
    const names: Record<ImplementationDifficulty, string> = {
      easy: 'Easy',
      medium: 'Medium',
      hard: 'Hard',
      expert: 'Expert'
    };
    return names[difficulty];
  },

  /**
   * Get difficulty color for UI
   */
  getDifficultyColor(difficulty: ImplementationDifficulty): string {
    const colors: Record<ImplementationDifficulty, string> = {
      easy: 'green',
      medium: 'blue',
      hard: 'orange',
      expert: 'red'
    };
    return colors[difficulty];
  },

  /**
   * Format implementation time for display
   */
  formatImplementationTime(hours: number): string {
    if (hours < 1) return `${Math.round(hours * 60)} minutes`;
    if (hours < 24) return `${hours} hours`;
    const days = Math.round(hours / 24);
    return `${days} ${days === 1 ? 'day' : 'days'}`;
  },

  /**
   * Get available attack vectors
   */
  getAvailableAttackVectors(): Array<{id: string; name: string}> {
    return [
      { id: 'sql_injection', name: 'SQL Injection' },
      { id: 'xss', name: 'Cross-Site Scripting (XSS)' },
      { id: 'csrf', name: 'Cross-Site Request Forgery (CSRF)' },
      { id: 'prompt_injection', name: 'Prompt Injection' },
      { id: 'jailbreak_attacks', name: 'Jailbreak Attacks' },
      { id: 'model_poisoning', name: 'Model Poisoning' },
      { id: 'adversarial_examples', name: 'Adversarial Examples' },
      { id: 'data_extraction', name: 'Data Extraction' },
      { id: 'privacy_attacks', name: 'Privacy Attacks' },
      { id: 'backdoor_attacks', name: 'Backdoor Attacks' }
    ];
  },

  /**
   * Get available system architectures
   */
  getAvailableSystemArchitectures(): Array<{id: SystemArchitecture; name: string}> {
    return [
      { id: 'web_app', name: 'Web Application' },
      { id: 'api_service', name: 'API Service' },
      { id: 'ml_pipeline', name: 'ML Pipeline' },
      { id: 'chatbot', name: 'Chatbot' },
      { id: 'mobile_app', name: 'Mobile App' }
    ];
  },

  /**
   * Get suggested technology stacks
   */
  getSuggestedTechnologyStacks(): string[] {
    return [
      'React/Node.js',
      'Python/Django',
      'Python/Flask',
      'Java/Spring',
      'C#/.NET',
      'PHP/Laravel',
      'Ruby/Rails',
      'Go',
      'Rust',
      'PyTorch/TensorFlow',
      'Hugging Face',
      'OpenAI API',
      'Anthropic API'
    ];
  },

  /**
   * Get defense category display name
   */
  getDefenseCategoryDisplayName(category: DefenseCategory): string {
    const names: Record<DefenseCategory, string> = {
      input_filtering: 'Input Filtering',
      output_monitoring: 'Output Monitoring',
      session_management: 'Session Management',
      model_training: 'Model Training',
      infrastructure: 'Infrastructure'
    };
    return names[category];
  }
};
