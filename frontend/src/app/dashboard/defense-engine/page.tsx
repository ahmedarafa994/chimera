/**
 * Defense Recommendation Engine Dashboard
 *
 * Phase 4 innovation feature for comprehensive security:
 * - Automated defensive measure suggestions
 * - Implementation guides for each defense
 * - Effectiveness and difficulty ratings
 * - Defense validation tracking
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
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import {
  Shield,
  Plus,
  Play,
  Settings,
  CheckCircle,
  XCircle,
  RefreshCw,
  Search,
  BarChart3,
  TrendingUp,
  Target,
  Lightbulb,
  Eye,
  Award,
  Bug,
  Database,
  Server,
  Globe,
  Smartphone,
  Monitor
} from 'lucide-react';
import { toast } from 'sonner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, RadialBarChart, RadialBar } from 'recharts';

// Import services
import {
  defenseEngineService,
  VulnerabilityAssessment,
  DefenseRecommendation,
  DefenseTechnique,
  DefenseImplementation,
  DefenseAnalytics,
  VulnerabilityAssessmentCreate,
  DefenseRecommendationRequest,
  DefenseImplementationCreate,
  DefenseImplementationUpdate,
  DefenseListResponse,
  RecommendationListResponse,
  ImplementationListResponse,
  RiskLevel,
  DefenseCategory,
  ImplementationDifficulty,
  PriorityLevel,
  ImplementationStatus,
  SystemArchitecture,
  DeploymentEnvironment
} from '@/lib/api/services/defense-engine';

export default function DefenseEnginePage() {
  // Data state
  const [analytics, setAnalytics] = useState<DefenseAnalytics | null>(null);
  const [assessments, setAssessments] = useState<VulnerabilityAssessment[]>([]);
  const [selectedAssessment, setSelectedAssessment] = useState<VulnerabilityAssessment | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationListResponse | null>(null);
  const [defenses, setDefenses] = useState<DefenseListResponse | null>(null);
  const [implementations, setImplementations] = useState<ImplementationListResponse | null>(null);

  // Form state
  const [newAssessmentData, setNewAssessmentData] = useState<VulnerabilityAssessmentCreate>({
    target_system: '',
    vulnerabilities: [],
    risk_level: 'medium',
    attack_vectors_found: [],
    system_architecture: 'web_app',
    technology_stack: [],
    deployment_environment: 'production',
    existing_defenses: [],
    compliance_requirements: []
  });

  const [recommendationRequest, setRecommendationRequest] = useState<DefenseRecommendationRequest>({
    assessment_id: '',
    max_recommendations: 5,
    priority_filter: undefined,
    difficulty_preference: undefined
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [showCreateAssessmentDialog, setShowCreateAssessmentDialog] = useState(false);
  const [showRecommendationsDialog, setShowRecommendationsDialog] = useState(false);
  const [showDefenseDetailsDialog, setShowDefenseDetailsDialog] = useState(false);
  const [selectedDefense, setSelectedDefense] = useState<DefenseTechnique | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [searchTerm, setSearchTerm] = useState('');
  const [riskLevelFilter, setRiskLevelFilter] = useState<RiskLevel | 'all'>('all');

  useEffect(() => {
    loadAnalytics();
    loadAssessments();
    loadDefenses();
    loadImplementations();
  }, []);

  const loadAnalytics = useCallback(async () => {
    try {
      const data = await defenseEngineService.getDefenseAnalytics();
      setAnalytics(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadAssessments = useCallback(async () => {
    try {
      setLoading(true);
      const params = riskLevelFilter !== 'all' ? { risk_level: riskLevelFilter } : undefined;
      const data = await defenseEngineService.listVulnerabilityAssessments(params);
      setAssessments(data);
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, [riskLevelFilter]);

  const loadDefenses = useCallback(async () => {
    try {
      const data = await defenseEngineService.listDefenseTechniques({ page: 1, page_size: 50 });
      setDefenses(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadImplementations = useCallback(async () => {
    try {
      const data = await defenseEngineService.listDefenseImplementations({ page: 1, page_size: 50 });
      setImplementations(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleCreateAssessment = useCallback(async () => {
    const errors = defenseEngineService.validateVulnerabilityAssessment(newAssessmentData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      const assessment = await defenseEngineService.createVulnerabilityAssessment(newAssessmentData);

      // Refresh data
      await Promise.all([loadAssessments(), loadAnalytics()]);

      // Select the new assessment
      setSelectedAssessment(assessment);

      // Reset form
      setNewAssessmentData({
        target_system: '',
        vulnerabilities: [],
        risk_level: 'medium',
        attack_vectors_found: [],
        system_architecture: 'web_app',
        technology_stack: [],
        deployment_environment: 'production',
        existing_defenses: [],
        compliance_requirements: []
      });
      setShowCreateAssessmentDialog(false);

      toast.success(`Vulnerability assessment "${assessment.target_system}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newAssessmentData, loadAssessments, loadAnalytics]);

  const handleGenerateRecommendations = useCallback(async () => {
    if (!selectedAssessment) return;

    try {
      setGenerating(true);
      const data = await defenseEngineService.generateDefenseRecommendations(
        selectedAssessment.assessment_id,
        { ...recommendationRequest, assessment_id: selectedAssessment.assessment_id }
      );

      setRecommendations(data);
      setShowRecommendationsDialog(true);
    } catch (error) {
      // Error already handled in service
    } finally {
      setGenerating(false);
    }
  }, [selectedAssessment, recommendationRequest]);

  const handleImplementDefense = useCallback(async (recommendation: DefenseRecommendation) => {
    try {
      const implementationData: DefenseImplementationCreate = {
        recommendation_id: recommendation.recommendation_id,
        implementation_notes: `Starting implementation of ${recommendation.defense_id}`,
        configuration_used: {}
      };

      await defenseEngineService.createDefenseImplementation(implementationData);

      // Refresh implementations
      await loadImplementations();
    } catch (error) {
      // Error already handled in service
    }
  }, [loadImplementations]);

  const filteredAssessments = assessments.filter(assessment => {
    const matchesSearch = assessment.target_system.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  });

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Defense Engine...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your defense recommendation data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Defense Recommendation Engine
        </h1>
        <p className="text-muted-foreground text-lg">
          Automated defensive measure suggestions with implementation guides, effectiveness ratings, and validation tracking.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="assessments">Assessments</TabsTrigger>
          <TabsTrigger value="defenses">Defenses</TabsTrigger>
          <TabsTrigger value="implementations">Implementations</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <DefenseDashboardView
            analytics={analytics}
            onRefresh={loadAnalytics}
          />
        </TabsContent>

        <TabsContent value="assessments" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-bold">Vulnerability Assessments</h2>
              <Badge variant="outline">
                {assessments.length} assessment{assessments.length !== 1 ? 's' : ''}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search assessments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-64"
                />
              </div>
              <Select value={riskLevelFilter} onValueChange={(value: RiskLevel | 'all') => setRiskLevelFilter(value)}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Risk Levels</SelectItem>
                  <SelectItem value="low">Low Risk</SelectItem>
                  <SelectItem value="medium">Medium Risk</SelectItem>
                  <SelectItem value="high">High Risk</SelectItem>
                  <SelectItem value="critical">Critical Risk</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={() => setShowCreateAssessmentDialog(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Assessment
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Assessment List */}
            <div className="lg:col-span-1">
              <AssessmentListView
                assessments={filteredAssessments}
                selectedAssessment={selectedAssessment}
                onSelectAssessment={setSelectedAssessment}
                onGenerateRecommendations={handleGenerateRecommendations}
                generating={generating}
                onRefresh={loadAssessments}
              />
            </div>

            {/* Assessment Details */}
            <div className="lg:col-span-2">
              {selectedAssessment ? (
                <AssessmentDetailsView
                  assessment={selectedAssessment}
                  recommendations={recommendations}
                  onGenerateRecommendations={handleGenerateRecommendations}
                  onImplementDefense={handleImplementDefense}
                  generating={generating}
                />
              ) : (
                <Card>
                  <CardContent className="text-center py-12">
                    <Shield className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      No Assessment Selected
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300">
                      Select a vulnerability assessment from the list to view details and generate defense recommendations.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="defenses" className="space-y-6">
          <DefenseCatalogView
            defenses={defenses}
            onSelectDefense={(defense: DefenseTechnique) => {
              setSelectedDefense(defense);
              setShowDefenseDetailsDialog(true);
            }}
            onRefresh={loadDefenses}
          />
        </TabsContent>

        <TabsContent value="implementations" className="space-y-6">
          <ImplementationsView
            implementations={implementations}
            onRefresh={loadImplementations}
          />
        </TabsContent>

        <TabsContent value="metrics" className="space-y-6">
          <MetricsView
            implementations={implementations}
          />
        </TabsContent>
      </Tabs>

      {/* Create Assessment Dialog */}
      <Dialog open={showCreateAssessmentDialog} onOpenChange={setShowCreateAssessmentDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Vulnerability Assessment</DialogTitle>
            <DialogDescription>
              Analyze your system&apos;s security posture to receive personalized defense recommendations.
            </DialogDescription>
          </DialogHeader>

          <CreateAssessmentForm
            data={newAssessmentData}
            onChange={setNewAssessmentData}
            onSubmit={handleCreateAssessment}
            onCancel={() => setShowCreateAssessmentDialog(false)}
            creating={creating}
          />
        </DialogContent>
      </Dialog>

      {/* Defense Details Dialog */}
      <Dialog open={showDefenseDetailsDialog} onOpenChange={setShowDefenseDetailsDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{selectedDefense?.name}</DialogTitle>
            <DialogDescription>
              Detailed implementation guide and effectiveness information
            </DialogDescription>
          </DialogHeader>

          {selectedDefense && (
            <DefenseDetailsForm
              defense={selectedDefense}
              onClose={() => setShowDefenseDetailsDialog(false)}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Dashboard View Component
function DefenseDashboardView({
  analytics,
  onRefresh
}: {
  analytics: DefenseAnalytics | null;
  onRefresh: () => void;
}) {
  if (!analytics) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            <div className="grid grid-cols-4 gap-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-24 bg-gray-200 rounded"></div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1'];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Defense Analytics Dashboard</h2>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <Shield className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold">{analytics.total_defenses_available}</div>
            <div className="text-sm text-muted-foreground">Available Defenses</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold">{analytics.successful_deployments}</div>
            <div className="text-sm text-muted-foreground">Successful Deployments</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <TrendingUp className="h-8 w-8 mx-auto mb-2 text-orange-600" />
            <div className="text-2xl font-bold">{analytics.average_effectiveness.toFixed(1)}%</div>
            <div className="text-sm text-muted-foreground">Avg Effectiveness</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Target className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold">{analytics.vulnerabilities_addressed}</div>
            <div className="text-sm text-muted-foreground">Vulnerabilities Addressed</div>
          </CardContent>
        </Card>
      </div>

      {/* Implementation Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Implementation Trend</CardTitle>
            <CardDescription>Defense implementations over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={analytics.implementations_by_month}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="implementations" stroke="#8884d8" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top Performing Defenses</CardTitle>
            <CardDescription>Effectiveness by defense type</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={analytics.top_performing_defenses}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="defense" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="effectiveness" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* ROI and Risk Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Risk Reduction</CardTitle>
            <CardDescription>Overall security improvement</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {analytics.total_risk_reduced.toFixed(1)}%
              </div>
              <p className="text-sm text-muted-foreground">
                Total risk reduction achieved
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Implementation Time</CardTitle>
            <CardDescription>Average deployment efficiency</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {analytics.average_implementation_time.toFixed(1)}h
              </div>
              <p className="text-sm text-muted-foreground">
                Average implementation time
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Cost-Benefit Ratio</CardTitle>
            <CardDescription>Return on security investment</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">
                {analytics.cost_benefit_ratio.toFixed(1)}:1
              </div>
              <p className="text-sm text-muted-foreground">
                Benefit to cost ratio
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Most Deployed Defenses */}
      <Card>
        <CardHeader>
          <CardTitle>Most Deployed Defenses</CardTitle>
          <CardDescription>Popular defense implementations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {analytics.most_deployed_defenses.map((defense, index) => (
              <div key={defense.defense} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="text-lg font-bold text-muted-foreground">
                    #{index + 1}
                  </div>
                  <div>
                    <h4 className="font-medium capitalize">{defense.defense.replace('_', ' ')}</h4>
                    <p className="text-sm text-muted-foreground">
                      {defense.deployments} deployments
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Progress value={defense.success_rate * 100} className="w-24" />
                  <span className="text-sm font-mono w-12">
                    {(defense.success_rate * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Placeholder components - in a real implementation these would be fully featured

function AssessmentListView({ assessments, selectedAssessment, onSelectAssessment, onGenerateRecommendations, generating, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Assessments
          </CardTitle>
          <Button size="sm" onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          {assessments.length} vulnerability assessment{assessments.length !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96">
          {assessments.length === 0 ? (
            <div className="text-center py-8 px-4">
              <Shield className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <h4 className="font-semibold mb-2">No Assessments</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Create your first vulnerability assessment to get defense recommendations.
              </p>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {assessments.map((assessment: VulnerabilityAssessment) => (
                <div
                  key={assessment.assessment_id}
                  className={`p-3 rounded-lg border transition-colors cursor-pointer ${
                    selectedAssessment?.assessment_id === assessment.assessment_id
                      ? 'bg-primary/10 border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                  onClick={() => onSelectAssessment(assessment)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium truncate">{assessment.target_system}</h4>
                    <Badge
                      variant="outline"
                      className={`bg-${defenseEngineService.getRiskLevelColor(assessment.risk_level)}-50 text-xs`}
                    >
                      {defenseEngineService.getRiskLevelDisplayName(assessment.risk_level)}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                    <span>{assessment.vulnerabilities.length} vulnerabilities</span>
                    <span>{assessment.attack_vectors_found.length} attack vectors</span>
                  </div>
                  <div className="flex items-center gap-1 mt-2">
                    <Button
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); onGenerateRecommendations(); }}
                      disabled={generating}
                    >
                      <Lightbulb className="h-3 w-3 mr-1" />
                      Recommendations
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function AssessmentDetailsView({ assessment, recommendations, onGenerateRecommendations, onImplementDefense, generating }: any) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>{assessment.target_system}</CardTitle>
              <CardDescription>
                {defenseEngineService.getRiskLevelDisplayName(assessment.risk_level)} •
                {assessment.vulnerabilities.length} vulnerabilities found
              </CardDescription>
            </div>
            <Button onClick={onGenerateRecommendations} disabled={generating}>
              {generating ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Lightbulb className="h-4 w-4 mr-2" />
              )}
              Generate Recommendations
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <Label className="font-medium">System Architecture</Label>
                <p className="capitalize">{assessment.system_architecture.replace('_', ' ')}</p>
              </div>
              <div>
                <Label className="font-medium">Deployment Environment</Label>
                <p className="capitalize">{assessment.deployment_environment}</p>
              </div>
            </div>

            <div>
              <Label className="font-medium">Technology Stack</Label>
              <div className="flex flex-wrap gap-1 mt-1">
                {assessment.technology_stack.map((tech: string) => (
                  <Badge key={tech} variant="secondary" className="text-xs">
                    {tech}
                  </Badge>
                ))}
              </div>
            </div>

            <div>
              <Label className="font-medium">Attack Vectors Found</Label>
              <div className="flex flex-wrap gap-1 mt-1">
                {assessment.attack_vectors_found.map((vector: string) => (
                  <Badge key={vector} variant="destructive" className="text-xs">
                    {vector.replace('_', ' ')}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {recommendations && (
        <Card>
          <CardHeader>
            <CardTitle>Defense Recommendations</CardTitle>
            <CardDescription>
              {recommendations.recommendations.length} recommended defense{recommendations.recommendations.length !== 1 ? 's' : ''}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recommendations.recommendations.map((recommendation: DefenseRecommendation) => (
                <div key={recommendation.recommendation_id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">Defense #{recommendation.recommended_order}</h4>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className={`bg-${defenseEngineService.getRiskLevelColor(recommendation.priority_level as RiskLevel)}-50`}>
                        {recommendation.priority_level} priority
                      </Badge>
                      <Badge variant="secondary">
                        {recommendation.expected_risk_reduction.toFixed(0)}% risk reduction
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">
                    {recommendation.justification}
                  </p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-3">
                    <span>Implementation time: {defenseEngineService.formatImplementationTime(recommendation.estimated_implementation_time)}</span>
                    <span>Expertise: {recommendation.required_expertise_level}</span>
                  </div>
                  <Button size="sm" onClick={() => onImplementDefense(recommendation)}>
                    <Play className="h-3 w-3 mr-1" />
                    Start Implementation
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function DefenseCatalogView({ defenses, onSelectDefense, onRefresh }: any) {
  if (!defenses) {
    return (
      <Card>
        <CardContent className="text-center py-8">
          <Shield className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Loading defense catalog...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Defense Catalog</h2>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {defenses.defenses.map((defense: DefenseTechnique) => (
          <Card key={defense.defense_id} className="cursor-pointer hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{defense.name}</CardTitle>
                <Badge variant="outline" className={`bg-${defenseEngineService.getDifficultyColor(defense.implementation_difficulty)}-50`}>
                  {defenseEngineService.getDifficultyDisplayName(defense.implementation_difficulty)}
                </Badge>
              </div>
              <CardDescription className="line-clamp-2">
                {defense.description}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span>Effectiveness:</span>
                  <div className="flex items-center gap-2">
                    <Progress value={defense.effectiveness_score * 10} className="w-16" />
                    <span className="font-medium">{defense.effectiveness_score.toFixed(1)}/10</span>
                  </div>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span>Deployment time:</span>
                  <span>{defenseEngineService.formatImplementationTime(defense.deployment_time_hours)}</span>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span>Deployments:</span>
                  <span>{defense.real_world_deployments}</span>
                </div>

                <Button
                  className="w-full mt-3"
                  variant="outline"
                  onClick={() => onSelectDefense(defense)}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  View Details
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}

function ImplementationsView({ implementations, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Defense Implementations</CardTitle>
        <CardDescription>Track and monitor your defense deployments</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Settings className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Implementation tracking will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

function MetricsView({ implementations }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Defense Metrics</CardTitle>
        <CardDescription>Performance and effectiveness measurements</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <BarChart3 className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Defense metrics will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

// Form Components
function CreateAssessmentForm({ data, onChange, onSubmit, onCancel, creating }: any) {
  const attackVectors = defenseEngineService.getAvailableAttackVectors();
  const architectures = defenseEngineService.getAvailableSystemArchitectures();
  const techStacks = defenseEngineService.getSuggestedTechnologyStacks();

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Target System</Label>
          <Input
            value={data.target_system}
            onChange={(e) => onChange({ ...data, target_system: e.target.value })}
            placeholder="AI Chat Application"
          />
        </div>
        <div className="space-y-2">
          <Label>Risk Level</Label>
          <Select
            value={data.risk_level}
            onValueChange={(value: RiskLevel) => onChange({ ...data, risk_level: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="low">Low Risk</SelectItem>
              <SelectItem value="medium">Medium Risk</SelectItem>
              <SelectItem value="high">High Risk</SelectItem>
              <SelectItem value="critical">Critical Risk</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>System Architecture</Label>
          <Select
            value={data.system_architecture}
            onValueChange={(value: SystemArchitecture) => onChange({ ...data, system_architecture: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {architectures.map(arch => (
                <SelectItem key={arch.id} value={arch.id}>
                  {arch.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label>Deployment Environment</Label>
          <Select
            value={data.deployment_environment}
            onValueChange={(value: DeploymentEnvironment) => onChange({ ...data, deployment_environment: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="development">Development</SelectItem>
              <SelectItem value="staging">Staging</SelectItem>
              <SelectItem value="production">Production</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <Label>Attack Vectors Found</Label>
        <div className="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto border rounded p-2">
          {attackVectors.map(vector => (
            <div key={vector.id} className="flex items-center space-x-2">
              <input
                type="checkbox"
                id={`vector-${vector.id}`}
                checked={data.attack_vectors_found.includes(vector.id)}
                onChange={(e) => {
                  const checked = e.target.checked;
                  onChange({
                    ...data,
                    attack_vectors_found: checked
                      ? [...data.attack_vectors_found, vector.id]
                      : data.attack_vectors_found.filter((v: string) => v !== vector.id)
                  });
                }}
              />
              <Label htmlFor={`vector-${vector.id}`} className="text-sm">
                {vector.name}
              </Label>
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-end space-x-2">
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button onClick={onSubmit} disabled={creating}>
          {creating ? (
            <RefreshCw className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Plus className="h-4 w-4 mr-2" />
          )}
          Create Assessment
        </Button>
      </div>
    </div>
  );
}

function DefenseDetailsForm({ defense, onClose }: any) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label className="font-medium">Category</Label>
          <Badge variant="outline" className="mt-1">
            {defenseEngineService.getDefenseCategoryDisplayName(defense.category as DefenseCategory)}
          </Badge>
        </div>
        <div>
          <Label className="font-medium">Effectiveness Score</Label>
          <div className="flex items-center gap-2 mt-1">
            <Progress value={defense.effectiveness_score * 10} className="w-32" />
            <span>{defense.effectiveness_score.toFixed(1)}/10</span>
          </div>
        </div>
      </div>

      <div>
        <Label className="font-medium">Detailed Explanation</Label>
        <p className="text-sm mt-1">{defense.detailed_explanation}</p>
      </div>

      <div>
        <Label className="font-medium">Implementation Guide</Label>
        <pre className="text-sm mt-1 p-3 bg-muted rounded whitespace-pre-wrap">
          {defense.implementation_guide}
        </pre>
      </div>

      <div>
        <Label className="font-medium">Code Examples</Label>
        <Tabs defaultValue="python" className="mt-2">
          <TabsList>
            {Object.keys(defense.code_examples).map(lang => (
              <TabsTrigger key={lang} value={lang} className="capitalize">
                {lang}
              </TabsTrigger>
            ))}
          </TabsList>
          {Object.entries(defense.code_examples).map(([lang, code]) => (
            <TabsContent key={lang} value={lang}>
              <pre className="text-sm p-3 bg-muted rounded overflow-x-auto">
                <code>{String(code)}</code>
              </pre>
            </TabsContent>
          ))}
        </Tabs>
      </div>

      <div className="flex justify-end">
        <Button onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  );
}