/**
 * Professional Security Report Generator Service
 *
 * Phase 2 feature for competitive differentiation:
 * - PDF/HTML reports suitable for client deliverables
 * - Executive summary with risk ratings
 * - Detailed findings with evidence and remediation
 * - Customizable branding options
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';
import { authManager } from '@/lib/api/auth-manager';

export type ReportFormat = 'pdf' | 'html' | 'json';
export type RiskLevel = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type ReportType = 'executive' | 'technical' | 'comprehensive' | 'compliance';

export interface SecurityFinding {
  id: string;
  title: string;
  description: string;
  risk_level: RiskLevel;
  confidence: number;

  // Evidence
  affected_prompts: string[];
  model_responses: string[];
  technique_used: string;

  // Impact assessment
  business_impact: string;
  technical_impact: string;
  exploitability: number;

  // Remediation
  remediation_steps: string[];
  prevention_measures: string[];
  priority: number;
}

export interface ExecutiveSummary {
  overall_risk_score: number;
  total_findings: number;
  critical_findings: number;
  high_findings: number;
  medium_findings: number;
  low_findings: number;

  key_risks: string[];
  business_recommendations: string[];
  compliance_status: string;
  assessment_scope: string;
  tested_models: string[];
}

export interface ReportBranding {
  company_name?: string;
  company_logo?: string; // Base64 encoded
  report_title?: string;
  prepared_for?: string;
  prepared_by?: string;
  contact_info?: string;
  confidentiality_level?: string;
}

export interface ReportRequest {
  assessment_ids: string[];
  report_type?: ReportType;
  report_format?: ReportFormat;
  include_executive_summary?: boolean;
  include_detailed_findings?: boolean;
  include_remediation_plan?: boolean;
  include_appendix?: boolean;

  // Customization
  branding?: ReportBranding;
  custom_sections?: Record<string, string>;

  // Filtering
  min_risk_level?: RiskLevel;
  include_techniques?: string[];
  exclude_techniques?: string[];
}

export interface SecurityReport {
  report_id: string;
  report_type: ReportType;
  report_format: ReportFormat;
  generated_at: string;

  // Metadata
  assessment_period: Record<string, string>;
  scope: Record<string, any>;
  methodology: string[];

  // Content
  executive_summary: ExecutiveSummary;
  findings: SecurityFinding[];
  recommendations: string[];

  // Statistics
  statistics: Record<string, any>;

  // Branding
  branding?: ReportBranding;
}

export interface GeneratedReport {
  report_id: string;
  format: ReportFormat;
  status: string;
  download_url: string;
  size_bytes: number;
  generated_at: string;
}

export interface ReportListResponse {
  reports: Array<Record<string, any>>;
  total: number;
}

class SecurityReportService {
  private readonly baseUrl = '/reports';

  /**
   * Generate a professional security assessment report
   */
  async generateReport(request: ReportRequest): Promise<GeneratedReport> {
    try {
      const response = await apiClient.post<GeneratedReport>(`${this.baseUrl}/generate`, {
        assessment_ids: request.assessment_ids,
        report_type: request.report_type || 'technical',
        report_format: request.report_format || 'pdf',
        include_executive_summary: request.include_executive_summary ?? true,
        include_detailed_findings: request.include_detailed_findings ?? true,
        include_remediation_plan: request.include_remediation_plan ?? true,
        include_appendix: request.include_appendix ?? false,
        branding: request.branding,
        custom_sections: request.custom_sections || {},
        min_risk_level: request.min_risk_level,
        include_techniques: request.include_techniques,
        exclude_techniques: request.exclude_techniques
      });

      toast.success('Security report generated successfully');

      return response.data;
    } catch (error) {
      console.error('Failed to generate security report:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate report');
      throw error;
    }
  }

  /**
   * List generated security reports
   */
  async listReports(): Promise<ReportListResponse> {
    try {
      const response = await apiClient.get<ReportListResponse>(`${this.baseUrl}/`);
      return response.data;
    } catch (error) {
      console.error('Failed to list reports:', error);
      toast.error('Failed to load report history');
      throw error;
    }
  }

  /**
   * Download a generated report
   */
  async downloadReport(reportId: string): Promise<void> {
    try {
      const response = await apiClient.get(`${this.baseUrl}/${reportId}/download`, {
        responseType: 'blob'
      });

      // Get filename from Content-Disposition header
      const contentDisposition = response.headers['content-disposition'];
      let filename = `security_report_${reportId}.pdf`;

      if (contentDisposition && typeof contentDisposition === 'string') {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      // Create download link
      const blob = response.data;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;

      document.body.appendChild(a);
      a.click();

      // Cleanup
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success('Report downloaded successfully');
    } catch (error) {
      console.error('Failed to download report:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to download report');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for report type
   */
  getReportTypeDisplayName(type: ReportType): string {
    const displayNames: Record<ReportType, string> = {
      executive: 'Executive Summary',
      technical: 'Technical Report',
      comprehensive: 'Comprehensive Assessment',
      compliance: 'Compliance Report'
    };
    return displayNames[type];
  }

  /**
   * Get display name for report format
   */
  getFormatDisplayName(format: ReportFormat): string {
    const displayNames: Record<ReportFormat, string> = {
      pdf: 'PDF Document',
      html: 'HTML Document',
      json: 'JSON Data'
    };
    return displayNames[format];
  }

  /**
   * Get display name for risk level
   */
  getRiskLevelDisplayName(level: RiskLevel): string {
    const displayNames: Record<RiskLevel, string> = {
      critical: 'Critical',
      high: 'High',
      medium: 'Medium',
      low: 'Low',
      info: 'Informational'
    };
    return displayNames[level];
  }

  /**
   * Get color for risk level
   */
  getRiskLevelColor(level: RiskLevel): string {
    const colors: Record<RiskLevel, string> = {
      critical: 'red',
      high: 'orange',
      medium: 'yellow',
      low: 'blue',
      info: 'gray'
    };
    return colors[level];
  }

  /**
   * Format file size
   */
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  /**
   * Get risk score color
   */
  getRiskScoreColor(score: number): string {
    if (score >= 8) return 'red';
    if (score >= 6) return 'orange';
    if (score >= 4) return 'yellow';
    if (score >= 2) return 'blue';
    return 'green';
  }

  /**
   * Format risk score
   */
  formatRiskScore(score: number): string {
    return `${score.toFixed(1)}/10`;
  }

  /**
   * Get compliance status color
   */
  getComplianceStatusColor(status: string): string {
    const statusColors: Record<string, string> = {
      'Good': 'green',
      'Requires Attention': 'yellow',
      'Critical Issues': 'red',
      'Under Review': 'blue'
    };
    return statusColors[status] || 'gray';
  }

  /**
   * Create default report request
   */
  createDefaultRequest(assessmentIds: string[]): ReportRequest {
    return {
      assessment_ids: assessmentIds,
      report_type: 'technical',
      report_format: 'pdf',
      include_executive_summary: true,
      include_detailed_findings: true,
      include_remediation_plan: true,
      include_appendix: false,
      branding: {
        company_name: 'Your Organization',
        confidentiality_level: 'CONFIDENTIAL',
        prepared_by: 'Chimera Security Team'
      }
    };
  }

  /**
   * Validate report request
   */
  validateReportRequest(request: ReportRequest): string[] {
    const errors: string[] = [];

    if (!request.assessment_ids || request.assessment_ids.length === 0) {
      errors.push('At least one assessment must be selected');
    }

    if (request.assessment_ids && request.assessment_ids.length > 10) {
      errors.push('Maximum 10 assessments can be included in a single report');
    }

    if (request.branding?.company_name && request.branding.company_name.length > 100) {
      errors.push('Company name must be less than 100 characters');
    }

    return errors;
  }

  /**
   * Get report type description
   */
  getReportTypeDescription(type: ReportType): string {
    const descriptions: Record<ReportType, string> = {
      executive: 'High-level summary focused on business impact and strategic recommendations',
      technical: 'Detailed technical analysis with specific vulnerabilities and remediation steps',
      comprehensive: 'Complete assessment including executive summary, technical details, and appendices',
      compliance: 'Focused on regulatory compliance requirements and audit trails'
    };
    return descriptions[type];
  }

  /**
   * Get estimated report generation time
   */
  getEstimatedGenerationTime(assessmentCount: number, format: ReportFormat): string {
    const baseTime = assessmentCount * 30; // 30 seconds per assessment
    const formatMultiplier = format === 'pdf' ? 1.5 : 1.0; // PDF takes longer
    const totalSeconds = Math.ceil(baseTime * formatMultiplier);

    if (totalSeconds < 60) {
      return `${totalSeconds} seconds`;
    } else {
      const minutes = Math.ceil(totalSeconds / 60);
      return `${minutes} minute${minutes > 1 ? 's' : ''}`;
    }
  }
}

// Export singleton instance
export const securityReportService = new SecurityReportService();
