/**
 * Professional Security Report Generator
 *
 * Phase 2 feature for competitive differentiation:
 * - PDF/HTML reports suitable for client deliverables
 * - Executive summary with risk ratings
 * - Detailed findings with evidence and remediation
 * - Customizable branding options
 */

"use client";

import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Checkbox } from '@/components/ui/checkbox';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import {
  FileText,
  Download,
  Settings,
  Eye,
  Clock,
  Shield,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  Building2,
  Users,
  RefreshCw,
  Filter,
  Plus
} from 'lucide-react';
import { toast } from 'sonner';

// Import services
import {
  securityReportService,
  ReportRequest,
  ReportType,
  ReportFormat,
  ReportBranding,
  GeneratedReport,
  ReportListResponse,
  RiskLevel
} from '@/lib/api/services/reports';

import { assessmentService } from '@/lib/api/services/assessments';

export default function SecurityReportsPage() {
  // Data state
  const [assessments, setAssessments] = useState<any[]>([]);
  const [reports, setReports] = useState<ReportListResponse | null>(null);

  // Form state
  const [selectedAssessments, setSelectedAssessments] = useState<string[]>([]);
  const [reportConfig, setReportConfig] = useState<ReportRequest>(() =>
    securityReportService.createDefaultRequest([])
  );

  // UI state
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [showBrandingDialog, setShowBrandingDialog] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);

      const [assessmentsData, reportsData] = await Promise.all([
        assessmentService.getAssessments(),
        securityReportService.listReports()
      ]);

      setAssessments(assessmentsData);
      setReports(reportsData);
    } catch (error) {
      // Errors already handled in services
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAssessmentToggle = useCallback((assessmentId: string, checked: boolean) => {
    setSelectedAssessments(prev => {
      const updated = checked
        ? [...prev, assessmentId]
        : prev.filter(id => id !== assessmentId);

      // Update report config
      setReportConfig(prevConfig => ({
        ...prevConfig,
        assessment_ids: updated
      }));

      return updated;
    });
  }, []);

  const handleGenerateReport = useCallback(async () => {
    if (selectedAssessments.length === 0) {
      toast.error('Please select at least one assessment');
      return;
    }

    const errors = securityReportService.validateReportRequest(reportConfig);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setGenerating(true);

      const result = await securityReportService.generateReport({
        ...reportConfig,
        assessment_ids: selectedAssessments
      });

      toast.success(`Report generated successfully! Report ID: ${result.report_id}`);

      // Refresh reports list
      const updatedReports = await securityReportService.listReports();
      setReports(updatedReports);

      // Clear selection
      setSelectedAssessments([]);
      setReportConfig(securityReportService.createDefaultRequest([]));
    } catch (error) {
      // Error already handled in service
    } finally {
      setGenerating(false);
    }
  }, [selectedAssessments, reportConfig]);

  const handleConfigChange = useCallback((key: keyof ReportRequest, value: any) => {
    setReportConfig(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleBrandingChange = useCallback((branding: ReportBranding) => {
    setReportConfig(prev => ({ ...prev, branding }));
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Report Generator...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your assessment data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Professional Security Reports
        </h1>
        <p className="text-muted-foreground text-lg">
          Generate comprehensive security assessment reports suitable for client deliverables and executive presentations.
        </p>
      </div>

      <Tabs defaultValue="generate" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="generate">Generate New Report</TabsTrigger>
          <TabsTrigger value="history">Report History</TabsTrigger>
        </TabsList>

        <TabsContent value="generate" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: Assessment Selection */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Select Assessments</CardTitle>
                  <CardDescription>
                    Choose completed assessments to include in your security report
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {assessments.length === 0 ? (
                    <div className="text-center py-8">
                      <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                        No Completed Assessments
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300">
                        You need to complete some security assessments before generating reports.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3 max-h-80 overflow-y-auto">
                      {assessments.map((assessment) => (
                        <div
                          key={assessment.id}
                          className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
                        >
                          <Checkbox
                            id={`assessment-${assessment.id}`}
                            checked={selectedAssessments.includes(assessment.id)}
                            onCheckedChange={(checked) =>
                              handleAssessmentToggle(assessment.id, checked as boolean)
                            }
                          />
                          <div className="flex-1">
                            <Label
                              htmlFor={`assessment-${assessment.id}`}
                              className="font-medium cursor-pointer"
                            >
                              {assessment.name}
                            </Label>
                            <p className="text-sm text-muted-foreground">
                              {assessment.target_provider} • {assessment.target_model} • {
                                new Date(assessment.completed_at).toLocaleDateString()
                              }
                            </p>
                          </div>
                          <Badge
                            variant={assessment.success_rate > 0.5 ? "destructive" : "default"}
                          >
                            {Math.round((assessment.success_rate || 0) * 100)}% success
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}

                  {selectedAssessments.length > 0 && (
                    <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-blue-600" />
                        <span className="text-sm font-medium">
                          {selectedAssessments.length} assessment{selectedAssessments.length > 1 ? 's' : ''} selected
                        </span>
                      </div>
                      <p className="text-xs text-blue-600 mt-1">
                        Estimated generation time: {securityReportService.getEstimatedGenerationTime(
                          selectedAssessments.length,
                          reportConfig.report_format || 'pdf'
                        )}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Report Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle>Report Configuration</CardTitle>
                  <CardDescription>
                    Customize report type, format, and content options
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Report Type</Label>
                      <Select
                        value={reportConfig.report_type || 'technical'}
                        onValueChange={(value: ReportType) => handleConfigChange('report_type', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="executive">Executive Summary</SelectItem>
                          <SelectItem value="technical">Technical Report</SelectItem>
                          <SelectItem value="comprehensive">Comprehensive Assessment</SelectItem>
                          <SelectItem value="compliance">Compliance Report</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {securityReportService.getReportTypeDescription(reportConfig.report_type || 'technical')}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>Report Format</Label>
                      <Select
                        value={reportConfig.report_format || 'pdf'}
                        onValueChange={(value: ReportFormat) => handleConfigChange('report_format', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="pdf">PDF Document</SelectItem>
                          <SelectItem value="html">HTML Document</SelectItem>
                          <SelectItem value="json">JSON Data</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Content Options */}
                  <div className="space-y-3">
                    <Label className="text-sm font-medium">Content Sections</Label>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="executive-summary"
                          checked={reportConfig.include_executive_summary ?? true}
                          onCheckedChange={(checked) =>
                            handleConfigChange('include_executive_summary', checked)
                          }
                        />
                        <Label htmlFor="executive-summary" className="text-sm">
                          Executive Summary
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="detailed-findings"
                          checked={reportConfig.include_detailed_findings ?? true}
                          onCheckedChange={(checked) =>
                            handleConfigChange('include_detailed_findings', checked)
                          }
                        />
                        <Label htmlFor="detailed-findings" className="text-sm">
                          Detailed Findings
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="remediation-plan"
                          checked={reportConfig.include_remediation_plan ?? true}
                          onCheckedChange={(checked) =>
                            handleConfigChange('include_remediation_plan', checked)
                          }
                        />
                        <Label htmlFor="remediation-plan" className="text-sm">
                          Remediation Plan
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="appendix"
                          checked={reportConfig.include_appendix ?? false}
                          onCheckedChange={(checked) =>
                            handleConfigChange('include_appendix', checked)
                          }
                        />
                        <Label htmlFor="appendix" className="text-sm">
                          Technical Appendix
                        </Label>
                      </div>
                    </div>
                  </div>

                  {/* Risk Level Filter */}
                  <div className="space-y-2">
                    <Label>Minimum Risk Level</Label>
                    <Select
                      value={reportConfig.min_risk_level || 'all'}
                      onValueChange={(value) =>
                        handleConfigChange('min_risk_level', value === 'all' ? undefined : value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Risk Levels</SelectItem>
                        <SelectItem value="info">Info and Above</SelectItem>
                        <SelectItem value="low">Low and Above</SelectItem>
                        <SelectItem value="medium">Medium and Above</SelectItem>
                        <SelectItem value="high">High and Above</SelectItem>
                        <SelectItem value="critical">Critical Only</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Column: Actions & Preview */}
            <div className="space-y-6">
              {/* Branding Options */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Building2 className="h-4 w-4" />
                    Report Branding
                  </CardTitle>
                  <CardDescription>
                    Customize the appearance and branding
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label>Company Name</Label>
                    <Input
                      value={reportConfig.branding?.company_name || ''}
                      onChange={(e) =>
                        handleBrandingChange({
                          ...reportConfig.branding,
                          company_name: e.target.value
                        })
                      }
                      placeholder="Your Organization"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Prepared For</Label>
                    <Input
                      value={reportConfig.branding?.prepared_for || ''}
                      onChange={(e) =>
                        handleBrandingChange({
                          ...reportConfig.branding,
                          prepared_for: e.target.value
                        })
                      }
                      placeholder="Client Name"
                    />
                  </div>

                  <Button
                    variant="outline"
                    onClick={() => setShowBrandingDialog(true)}
                    className="w-full"
                  >
                    <Settings className="h-4 w-4 mr-2" />
                    Advanced Branding
                  </Button>
                </CardContent>
              </Card>

              {/* Generation Actions */}
              <Card>
                <CardHeader>
                  <CardTitle>Generate Report</CardTitle>
                  <CardDescription>
                    Create your professional security assessment report
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {selectedAssessments.length > 0 && (
                    <div className="bg-muted rounded-lg p-3 text-sm">
                      <div className="font-medium mb-1">Report Preview</div>
                      <div className="text-muted-foreground space-y-1">
                        <div>Type: {securityReportService.getReportTypeDisplayName(reportConfig.report_type || 'technical')}</div>
                        <div>Format: {securityReportService.getFormatDisplayName(reportConfig.report_format || 'pdf')}</div>
                        <div>Assessments: {selectedAssessments.length}</div>
                      </div>
                    </div>
                  )}

                  <Button
                    onClick={handleGenerateReport}
                    disabled={selectedAssessments.length === 0 || generating}
                    className="w-full"
                    size="lg"
                  >
                    {generating ? (
                      <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                    ) : (
                      <FileText className="h-4 w-4 mr-2" />
                    )}
                    {generating ? 'Generating Report...' : 'Generate Security Report'}
                  </Button>

                  {generating && (
                    <div className="text-center text-sm text-muted-foreground">
                      <Clock className="h-4 w-4 inline mr-1" />
                      This may take a few minutes...
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <ReportHistoryView reports={reports} onRefresh={loadData} />
        </TabsContent>
      </Tabs>

      {/* Advanced Branding Dialog */}
      <Dialog open={showBrandingDialog} onOpenChange={setShowBrandingDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Advanced Branding Options</DialogTitle>
            <DialogDescription>
              Customize the detailed branding and styling of your report
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Report Title</Label>
              <Input
                value={reportConfig.branding?.report_title || ''}
                onChange={(e) =>
                  handleBrandingChange({
                    ...reportConfig.branding,
                    report_title: e.target.value
                  })
                }
                placeholder="Security Assessment Report"
              />
            </div>

            <div className="space-y-2">
              <Label>Prepared By</Label>
              <Input
                value={reportConfig.branding?.prepared_by || ''}
                onChange={(e) =>
                  handleBrandingChange({
                    ...reportConfig.branding,
                    prepared_by: e.target.value
                  })
                }
                placeholder="Security Team"
              />
            </div>

            <div className="space-y-2">
              <Label>Contact Information</Label>
              <Textarea
                value={reportConfig.branding?.contact_info || ''}
                onChange={(e) =>
                  handleBrandingChange({
                    ...reportConfig.branding,
                    contact_info: e.target.value
                  })
                }
                placeholder="Contact details..."
                className="min-h-20"
              />
            </div>

            <div className="space-y-2">
              <Label>Confidentiality Level</Label>
              <Select
                value={reportConfig.branding?.confidentiality_level || 'CONFIDENTIAL'}
                onValueChange={(value) =>
                  handleBrandingChange({
                    ...reportConfig.branding,
                    confidentiality_level: value
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="PUBLIC">Public</SelectItem>
                  <SelectItem value="INTERNAL">Internal Use</SelectItem>
                  <SelectItem value="CONFIDENTIAL">Confidential</SelectItem>
                  <SelectItem value="RESTRICTED">Restricted</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowBrandingDialog(false)}>
                Cancel
              </Button>
              <Button onClick={() => setShowBrandingDialog(false)}>
                Apply Changes
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Report History View Component
function ReportHistoryView({
  reports,
  onRefresh
}: {
  reports: ReportListResponse | null;
  onRefresh: () => void;
}) {
  const handleDownload = useCallback(async (reportId: string) => {
    try {
      await securityReportService.downloadReport(reportId);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  if (!reports) {
    return (
      <Card>
        <CardContent className="text-center py-12">
          <RefreshCw className="h-12 w-12 text-gray-400 mx-auto mb-4 animate-spin" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Loading Report History...
          </h3>
        </CardContent>
      </Card>
    );
  }

  if (reports.total === 0) {
    return (
      <Card>
        <CardContent className="text-center py-12">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            No Reports Generated Yet
          </h3>
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            Generate your first security report to see it here.
          </p>
          <Button onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Generated Reports</h3>
          <p className="text-sm text-muted-foreground">
            {reports.total} report{reports.total > 1 ? 's' : ''} generated
          </p>
        </div>
        <Button onClick={onRefresh} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {reports.reports.map((report) => (
          <Card key={report.report_id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{report.report_title || 'Security Assessment Report'}</CardTitle>
                <Badge variant="outline">
                  {securityReportService.getFormatDisplayName(report.format).toUpperCase()}
                </Badge>
              </div>
              <CardDescription>
                Generated on {new Date(report.generated_at).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between text-sm">
                <span>Report ID:</span>
                <span className="font-mono">{report.report_id}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>File Size:</span>
                <span>{securityReportService.formatFileSize(report.size_bytes)}</span>
              </div>

              <Button
                onClick={() => handleDownload(report.report_id)}
                className="w-full"
              >
                <Download className="h-4 w-4 mr-2" />
                Download Report
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
