/**
 * Defense Recommendation Engine Service
 *
 * Phase 4 innovation feature for comprehensive security:
 * - Automated defensive measure suggestions
 * - Implementation guides for each defense
 * - Effectiveness and difficulty ratings
 * - Defense validation tracking
 */

import { toast } from 'sonner';
import apiClient from '@/lib/api/client';

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

  // Recommendation context
  priority_level: PriorityLevel;
  justification: string;
  expected_risk_reduction: number;

  // Implementation planning
  estimated_implementation_time: number;
  required_expertise_level: string;
  dependencies: string[];
  potential_side_effects: string[];

  // Cost-benefit analysis
  implementation_cost_estimate?: string;
  maintenance_overhead: string;
  performance_impact: string;

  // Alternatives
  alternative_approaches: string[];
  recommended_order: number;

  // Metadata
  generated_at: string;
  generated_by: string;
}

export interface DefenseImplementation {
  implementation_id: string;
  recommendation_id: string;
  defense_id: string;

  // Implementation status
  status: ImplementationStatus;
  started_at?: string;
  completed_at?: string;
  deployed_at?: string;

  // Implementation details
  implementation_notes: string;
  configuration_used: Record<string, any>;
  custom_modifications: string[];

  // Validation results
  validation_tests: Array<{
    test_name: string;
    result: 'passed' | 'failed';
    details: string;
    [key: string]: any;
  }>;
  effectiveness_measured?: number;
  false_positive_rate?: number;
  false_negative_rate?: number;

  // Issues and resolution
  issues_encountered: string[];
  resolution_steps: string[];

  // Team and responsibility
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

// Request/Response interfaces

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

export interface RecommendationListResponse {
  recommendations: DefenseRecommendation[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ImplementationListResponse {
  implementations: DefenseImplementation[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface DefenseAnalytics {
  total_defenses_available: number;
  total_implementations: number;
  successful_deployments: number;
  average_effectiveness: number;

  // Implementation trends
  implementations_by_month: Array<{ month: string; implementations: number }>;
  success_rate_by_defense: Record<string, number>;

  // Effectiveness analysis
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

  // Risk mitigation
  total_risk_reduced: number;
  vulnerabilities_addressed: number;

  // ROI metrics
  average_implementation_time: number;
  cost_benefit_ratio: number;
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

class DefenseRecommendationService {
  private readonly baseUrl = '/defense-engine';

  /**
   * Create a new vulnerability assessment
   */
  async createVulnerabilityAssessment(assessmentData: VulnerabilityAssessmentCreate): Promise<VulnerabilityAssessment> {
    try {
      const response = await apiClient.post<VulnerabilityAssessment>(`${this.baseUrl}/assessments`, assessmentData);
      toast.success('Vulnerability assessment created successfully');
      return response.data;
    } catch (error: any) {
      console.error('Failed to create vulnerability assessment:', error);
      toast.error(error.response?.data?.detail || 'Failed to create vulnerability assessment');
      throw error;
    }
  }

  /**
   * List vulnerability assessments
   */
  async listVulnerabilityAssessments(params?: {
    workspace_id?: string;
    risk_level?: RiskLevel;
    limit?: number;
  }): Promise<VulnerabilityAssessment[]> {
    try {
      const response = await apiClient.get<VulnerabilityAssessment[]>(`${this.baseUrl}/assessments`, {
        params
      });
      return response.data;
    } catch (error) {
      console.error('Failed to list vulnerability assessments:', error);
      toast.error('Failed to load vulnerability assessments');
      throw error;
    }
  }

  /**
   * Generate defense recommendations
   */
  async generateDefenseRecommendations(
    assessmentId: string,
    requestData: DefenseRecommendationRequest
  ): Promise<RecommendationListResponse> {
    try {
      const response = await apiClient.post<RecommendationListResponse>(
        `${this.baseUrl}/assessments/${assessmentId}/recommendations`,
        requestData
      );
      toast.success('Defense recommendations generated successfully');
      return response.data;
    } catch (error: any) {
      console.error('Failed to generate defense recommendations:', error);
      toast.error(error.response?.data?.detail || 'Failed to generate defense recommendations');
      throw error;
    }
  }

  /**
   * List available defense techniques
   */
  async listDefenseTechniques(params?: DefenseListParams): Promise<DefenseListResponse> {
    try {
      const response = await apiClient.get<DefenseListResponse>(`${this.baseUrl}/defenses`, {
        params
      });
      return response.data;
    } catch (error) {
      console.error('Failed to list defense techniques:', error);
      toast.error('Failed to load defense techniques');
      throw error;
    }
  }

  /**
   * Get specific defense technique details
   */
  async getDefenseTechnique(defenseId: string): Promise<DefenseTechnique> {
    try {
      const response = await apiClient.get<DefenseTechnique>(`${this.baseUrl}/defenses/${defenseId}`);
      return response.data;
    } catch (error: any) {
      console.error('Failed to get defense technique:', error);
      toast.error(error.response?.data?.detail || 'Failed to load defense technique');
      throw error;
    }
  }

  /**
   * Create a defense implementation
   */
  async createDefenseImplementation(implementationData: DefenseImplementationCreate): Promise<DefenseImplementation> {
    try {
      const response = await apiClient.post<DefenseImplementation>(`${this.baseUrl}/implementations`, implementationData);
      toast.success('Defense implementation started successfully');
      return response.data;
    } catch (error: any) {
      console.error('Failed to create defense implementation:', error);
      toast.error(error.response?.data?.detail || 'Failed to start implementation');
      throw error;
    }
  }

  /**
   * List defense implementations
   */
  async listDefenseImplementations(params?: ImplementationListParams): Promise<ImplementationListResponse> {
    try {
      const response = await apiClient.get<ImplementationListResponse>(`${this.baseUrl}/implementations`, {
        params
      });
      return response.data;
    } catch (error) {
      console.error('Failed to list defense implementations:', error);
      toast.error('Failed to load implementations');
      throw error;
    }
  }

  /**
   * Update defense implementation
   */
  async updateDefenseImplementation(
    implementationId: string,
    updateData: DefenseImplementationUpdate
  ): Promise<DefenseImplementation> {
    try {
      const response = await apiClient.patch<DefenseImplementation>(
        `${this.baseUrl}/implementations/${implementationId}`,
        updateData
      );
      toast.success('Implementation updated successfully');
      return response.data;
    } catch (error: any) {
      console.error('Failed to update implementation:', error);
      toast.error(error.response?.data?.detail || 'Failed to update implementation');
      throw error;
    }
  }

  /**
   * Record defense metrics
   */
  async recordDefenseMetrics(
    implementationId: string,
    metricsData: DefenseMetricsCreate
  ): Promise<DefenseMetrics> {
    try {
      const response = await apiClient.post<DefenseMetrics>(
        `${this.baseUrl}/implementations/${implementationId}/metrics`,
        metricsData
      );
      toast.success('Defense metrics recorded successfully');
      return response.data;
    } catch (error: any) {
      console.error('Failed to record defense metrics:', error);
      toast.error(error.response?.data?.detail || 'Failed to record metrics');
      throw error;
    }
  }

  /**
   * Get defense analytics
   */
  async getDefenseAnalytics(workspaceId?: string): Promise<DefenseAnalytics> {
    try {
      const response = await apiClient.get<DefenseAnalytics>(`${this.baseUrl}/analytics`, {
        params: { workspace_id: workspaceId }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get defense analytics:', error);
      toast.error('Failed to load defense analytics');
      throw error;
    }
  }

  /**
   * Validate defense implementation
   */
  async validateDefenseImplementation(
    defenseId: string,
    validationConfig: Record<string, any>
  ): Promise<{ message: string; validation_id: string; estimated_completion: string }> {
    try {
      const response = await apiClient.post<{ message: string; validation_id: string; estimated_completion: string }>(
        `${this.baseUrl}/defenses/${defenseId}/validate`,
        validationConfig
      );
      toast.success('Defense validation started');
      return response.data;
    } catch (error: any) {
      console.error('Failed to validate defense implementation:', error);
      toast.error(error.response?.data?.detail || 'Failed to start validation');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for risk level
   */
  getRiskLevelDisplayName(level: RiskLevel): string {
    const displayNames: Record<RiskLevel, string> = {
      low: 'Low Risk',
      medium: 'Medium Risk',
      high: 'High Risk',
      critical: 'Critical Risk'
    };
    return displayNames[level];
  }

  /**
   * Get color for risk level
   */
  getRiskLevelColor(level: RiskLevel): string {
    const colors: Record<RiskLevel, string> = {
      low: 'green',
      medium: 'yellow',
      high: 'orange',
      critical: 'red'
    };
    return colors[level];
  }

  /**
   * Get display name for defense category
   */
  getDefenseCategoryDisplayName(category: DefenseCategory): string {
    const displayNames: Record<DefenseCategory, string> = {
      input_filtering: 'Input Filtering',
      output_monitoring: 'Output Monitoring',
      session_management: 'Session Management',
      model_training: 'Model Training',
      infrastructure: 'Infrastructure'
    };
    return displayNames[category];
  }

  /**
   * Get display name for implementation difficulty
   */
  getDifficultyDisplayName(difficulty: ImplementationDifficulty): string {
    const displayNames: Record<ImplementationDifficulty, string> = {
      easy: 'Easy',
      medium: 'Medium',
      hard: 'Hard',
      expert: 'Expert Level'
    };
    return displayNames[difficulty];
  }

  /**
   * Get color for implementation difficulty
   */
  getDifficultyColor(difficulty: ImplementationDifficulty): string {
    const colors: Record<ImplementationDifficulty, string> = {
      easy: 'green',
      medium: 'blue',
      hard: 'orange',
      expert: 'red'
    };
    return colors[difficulty];
  }

  /**
   * Get display name for implementation status
   */
  getStatusDisplayName(status: ImplementationStatus): string {
    const displayNames: Record<ImplementationStatus, string> = {
      planned: 'Planned',
      in_progress: 'In Progress',
      testing: 'Testing',
      deployed: 'Deployed',
      failed: 'Failed'
    };
    return displayNames[status];
  }

  /**
   * Get color for implementation status
   */
  getStatusColor(status: ImplementationStatus): string {
    const colors: Record<ImplementationStatus, string> = {
      planned: 'gray',
      in_progress: 'blue',
      testing: 'yellow',
      deployed: 'green',
      failed: 'red'
    };
    return colors[status];
  }

  /**
   * Format effectiveness score
   */
  formatEffectivenessScore(score: number): { level: string; color: string } {
    if (score >= 8.0) {
      return { level: 'Excellent', color: 'green' };
    } else if (score >= 6.0) {
      return { level: 'Good', color: 'blue' };
    } else if (score >= 4.0) {
      return { level: 'Fair', color: 'yellow' };
    } else {
      return { level: 'Poor', color: 'red' };
    }
  }

  /**
   * Estimate implementation time
   */
  formatImplementationTime(hours: number): string {
    if (hours < 8) {
      return `${hours.toFixed(1)} hours`;
    } else if (hours < 40) {
      const days = (hours / 8).toFixed(1);
      return `${days} day${parseFloat(days) !== 1 ? 's' : ''}`;
    } else {
      const weeks = (hours / 40).toFixed(1);
      return `${weeks} week${parseFloat(weeks) !== 1 ? 's' : ''}`;
    }
  }

  /**
   * Validate vulnerability assessment
   */
  validateVulnerabilityAssessment(data: VulnerabilityAssessmentCreate): string[] {
    const errors: string[] = [];

    if (!data.target_system || data.target_system.trim().length === 0) {
      errors.push('Target system name is required');
    }

    if (!data.vulnerabilities || data.vulnerabilities.length === 0) {
      errors.push('At least one vulnerability must be identified');
    }

    if (!data.attack_vectors_found || data.attack_vectors_found.length === 0) {
      errors.push('At least one attack vector must be specified');
    }

    if (!data.technology_stack || data.technology_stack.length === 0) {
      errors.push('Technology stack information is required');
    }

    if (!['low', 'medium', 'high', 'critical'].includes(data.risk_level)) {
      errors.push('Valid risk level is required');
    }

    return errors;
  }

  /**
   * Get available attack vectors
   */
  getAvailableAttackVectors(): Array<{ id: string; name: string; description: string }> {
    return [
      {
        id: 'prompt_injection',
        name: 'Prompt Injection',
        description: 'Direct manipulation of AI prompts'
      },
      {
        id: 'context_manipulation',
        name: 'Context Manipulation',
        description: 'Exploitation of conversation context'
      },
      {
        id: 'data_leakage',
        name: 'Data Leakage',
        description: 'Unauthorized information disclosure'
      },
      {
        id: 'model_extraction',
        name: 'Model Extraction',
        description: 'Attempts to reverse engineer the model'
      },
      {
        id: 'adversarial_examples',
        name: 'Adversarial Examples',
        description: 'Crafted inputs to fool the model'
      }
    ];
  }

  /**
   * Get available system architectures
   */
  getAvailableSystemArchitectures(): Array<{ id: SystemArchitecture; name: string; description: string }> {
    return [
      {
        id: 'web_app',
        name: 'Web Application',
        description: 'Browser-based application with AI features'
      },
      {
        id: 'api_service',
        name: 'API Service',
        description: 'RESTful or GraphQL API with AI endpoints'
      },
      {
        id: 'ml_pipeline',
        name: 'ML Pipeline',
        description: 'Machine learning processing pipeline'
      },
      {
        id: 'chatbot',
        name: 'Chatbot',
        description: 'Conversational AI application'
      },
      {
        id: 'mobile_app',
        name: 'Mobile Application',
        description: 'Native or hybrid mobile app with AI'
      }
    ];
  }

  /**
   * Get suggested technology stacks
   */
  getSuggestedTechnologyStacks(): Array<{ name: string; technologies: string[] }> {
    return [
      {
        name: 'Python + FastAPI',
        technologies: ['Python', 'FastAPI', 'SQLAlchemy', 'PostgreSQL']
      },
      {
        name: 'Node.js + Express',
        technologies: ['Node.js', 'Express', 'MongoDB', 'TypeScript']
      },
      {
        name: 'Java + Spring',
        technologies: ['Java', 'Spring Boot', 'MySQL', 'Redis']
      },
      {
        name: 'React + Next.js',
        technologies: ['React', 'Next.js', 'Vercel', 'TypeScript']
      }
    ];
  }
}

// Export singleton instance
export const defenseRecommendationService = new DefenseRecommendationService();
