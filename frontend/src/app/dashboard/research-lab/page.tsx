/**
 * Adversarial Attack Research Lab Dashboard
 *
 * Phase 4 innovation feature for academic research:
 * - A/B testing framework for technique variations
 * - Custom fitness function editor
 * - Academic paper format exports
 * - Research paper citation linking
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
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import {
  FlaskRound,
  Beaker,
  FileText,
  Play,
  Pause,
  Plus,
  Trash2,
  Settings,
  Download,
  Upload,
  RefreshCw,
  Search,
  Filter,
  BarChart3,
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Target,
  AlertTriangle,
  CheckCircle,
  Info,
  XCircle,
  ExternalLink,
  Copy,
  Edit,
  Brain,
  Code2,
  Database,
  Users,
  Award,
  BookOpen,
  LineChart,
  PieChart,
  Calculator,
  TestTube,
  Microscope,
  Atom,
  Lightbulb
} from 'lucide-react';
import { toast } from 'sonner';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart as RechartsPieChart, Pie, Cell, ScatterChart, Scatter } from 'recharts';

// Import services
import {
  researchLabService,
  ExperimentDesign,
  ExperimentExecution,
  ResearchReport,
  FitnessFunction,
  TechniqueVariation,
  ResearchAnalytics,
  ExperimentDesignCreate,
  ExperimentExecutionTrigger,
  ResearchReportGenerate,
  FitnessFunctionCreate,
  ExperimentListResponse,
  ExecutionListResponse,
  FitnessFunctionListResponse,
  ExperimentStatus,
  ReportFormat,
  CitationStyle
} from '@/lib/api/services/research-lab';

export default function ResearchLabPage() {
  // Data state
  const [analytics, setAnalytics] = useState<ResearchAnalytics | null>(null);
  const [experiments, setExperiments] = useState<ExperimentListResponse | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentDesign | null>(null);
  const [executions, setExecutions] = useState<ExecutionListResponse | null>(null);
  const [fitnessFunctions, setFitnessFunctions] = useState<FitnessFunctionListResponse | null>(null);

  // Form state
  const [newExperimentData, setNewExperimentData] = useState<ExperimentDesignCreate>({
    title: '',
    description: '',
    research_question: '',
    hypothesis: '',
    control_technique: '',
    treatment_techniques: [],
    target_models: [],
    test_datasets: [],
    primary_fitness_function: '',
    sample_size: 100
  });

  const [newFitnessFunctionData, setNewFitnessFunctionData] = useState<FitnessFunctionCreate>({
    name: '',
    description: '',
    code: `def evaluate(results):\n    """\n    Evaluate experiment results and return a fitness score.\n    \n    Args:\n        results: List of test results\n    \n    Returns:\n        float: Fitness score between 0.0 and 1.0\n    """\n    success_count = sum(1 for r in results if r.get('success', False))\n    return success_count / len(results) if results else 0.0`,
    input_parameters: ['results'],
    output_type: 'float'
  });

  const [reportGenerateData, setReportGenerateData] = useState<ResearchReportGenerate>({
    title: '',
    authors: [''],
    abstract: '',
    keywords: [],
    citation_style: 'apa'
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [showCreateExperimentDialog, setShowCreateExperimentDialog] = useState(false);
  const [showCreateFunctionDialog, setShowCreateFunctionDialog] = useState(false);
  const [showGenerateReportDialog, setShowGenerateReportDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [experimentToDelete, setExperimentToDelete] = useState<ExperimentDesign | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<ExperimentStatus | 'all'>('all');

  useEffect(() => {
    loadAnalytics();
    loadExperiments();
    loadFitnessFunctions();
  }, []);

  const loadAnalytics = useCallback(async () => {
    try {
      const data = await researchLabService.getResearchAnalytics();
      setAnalytics(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadExperiments = useCallback(async () => {
    try {
      setLoading(true);
      const params = statusFilter !== 'all' ? { status: statusFilter } : undefined;
      const data = await researchLabService.listExperiments(params);
      setExperiments(data);
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  const loadExecutions = useCallback(async (experimentId: string) => {
    try {
      const data = await researchLabService.listExperimentExecutions(experimentId);
      setExecutions(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadFitnessFunctions = useCallback(async () => {
    try {
      const data = await researchLabService.listFitnessFunctions({ validated_only: true });
      setFitnessFunctions(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleCreateExperiment = useCallback(async () => {
    const errors = researchLabService.validateExperimentDesign(newExperimentData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      const experiment = await researchLabService.createExperiment(newExperimentData);

      // Refresh data
      await Promise.all([loadExperiments(), loadAnalytics()]);

      // Select the new experiment
      setSelectedExperiment(experiment);

      // Reset form
      setNewExperimentData({
        title: '',
        description: '',
        research_question: '',
        hypothesis: '',
        control_technique: '',
        treatment_techniques: [],
        target_models: [],
        test_datasets: [],
        primary_fitness_function: '',
        sample_size: 100
      });
      setShowCreateExperimentDialog(false);

      toast.success(`Experiment "${experiment.title}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newExperimentData, loadExperiments, loadAnalytics]);

  const handleCreateFitnessFunction = useCallback(async () => {
    const errors = researchLabService.validateFitnessFunctionCreate(newFitnessFunctionData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      await researchLabService.createFitnessFunction(newFitnessFunctionData);

      // Refresh data
      await loadFitnessFunctions();

      // Reset form
      setNewFitnessFunctionData({
        name: '',
        description: '',
        code: `def evaluate(results):\n    """\n    Evaluate experiment results and return a fitness score.\n    \n    Args:\n        results: List of test results\n    \n    Returns:\n        float: Fitness score between 0.0 and 1.0\n    """\n    success_count = sum(1 for r in results if r.get('success', False))\n    return success_count / len(results) if results else 0.0`,
        input_parameters: ['results'],
        output_type: 'float'
      });
      setShowCreateFunctionDialog(false);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newFitnessFunctionData, loadFitnessFunctions]);

  const handleExecuteExperiment = useCallback(async (experiment: ExperimentDesign) => {
    const config: ExperimentExecutionTrigger = {
      parallel_execution: true,
      max_concurrent_tests: 5,
      timeout_seconds: 3600
    };

    try {
      setExecuting(true);
      const experimentId = experiment.experiment_id || experiment.id;
      await researchLabService.executeExperiment(experimentId, config);

      // Refresh executions if this experiment is selected
      const selectedId = selectedExperiment?.experiment_id || selectedExperiment?.id;
      if (selectedId === experimentId) {
        await loadExecutions(experimentId);
      }

      await loadAnalytics();
    } catch (error) {
      // Error already handled in service
    } finally {
      setExecuting(false);
    }
  }, [selectedExperiment, loadExecutions, loadAnalytics]);

  const handleGenerateReport = useCallback(async () => {
    if (!selectedExperiment) return;

    try {
      setCreating(true);
      const report = await researchLabService.generateResearchReport(
        selectedExperiment.experiment_id || selectedExperiment.id,
        reportGenerateData
      );

      // Reset form
      setReportGenerateData({
        title: '',
        authors: [''],
        abstract: '',
        keywords: [],
        citation_style: 'apa'
      });
      setShowGenerateReportDialog(false);

      toast.success('Research report generated successfully!');
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [selectedExperiment, reportGenerateData]);

  const filteredExperiments = experiments?.experiments?.filter(experiment => {
    const matchesSearch = experiment.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         experiment.description.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  }) || [];

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Research Lab...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your research experiment data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Adversarial Attack Research Lab
        </h1>
        <p className="text-muted-foreground text-lg">
          Advanced A/B testing framework for technique variations with academic research capabilities and automated report generation.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="experiments">Experiments</TabsTrigger>
          <TabsTrigger value="functions">Fitness Functions</TabsTrigger>
          <TabsTrigger value="executions">Executions</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <ResearchDashboardView
            analytics={analytics}
            onRefresh={loadAnalytics}
          />
        </TabsContent>

        <TabsContent value="experiments" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-bold">Research Experiments</h2>
              <Badge variant="outline">
                {experiments?.total || 0} experiment{(experiments?.total || 0) !== 1 ? 's' : ''}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search experiments..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-64"
                />
              </div>
              <Select value={statusFilter} onValueChange={(value: ExperimentStatus | 'all') => setStatusFilter(value)}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="running">Running</SelectItem>
                  <SelectItem value="completed">Completed</SelectItem>
                  <SelectItem value="failed">Failed</SelectItem>
                  <SelectItem value="cancelled">Cancelled</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={() => setShowCreateExperimentDialog(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Experiment
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Experiment List */}
            <div className="lg:col-span-1">
              <ExperimentListView
                experiments={filteredExperiments}
                selectedExperiment={selectedExperiment}
                onSelectExperiment={(experiment: ExperimentDesign) => {
                  setSelectedExperiment(experiment);
                  loadExecutions(experiment.experiment_id || experiment.id);
                }}
                onExecuteExperiment={handleExecuteExperiment}
                onDeleteExperiment={(experiment: ExperimentDesign) => {
                  setExperimentToDelete(experiment);
                  setShowDeleteDialog(true);
                }}
                executing={executing}
                onRefresh={loadExperiments}
              />
            </div>

            {/* Experiment Details */}
            <div className="lg:col-span-2">
              {selectedExperiment ? (
                <ExperimentDetailsView
                  experiment={selectedExperiment}
                  executions={executions}
                  onRefresh={() => loadExecutions(selectedExperiment.experiment_id || selectedExperiment.id)}
                  onGenerateReport={() => setShowGenerateReportDialog(true)}
                />
              ) : (
                <Card>
                  <CardContent className="text-center py-12">
                    <FlaskRound className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      No Experiment Selected
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300">
                      Select a research experiment from the list to view details and execution results.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="functions" className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Fitness Functions</h2>
            <Button onClick={() => setShowCreateFunctionDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Function
            </Button>
          </div>

          <FitnessFunctionsView
            functions={fitnessFunctions}
            onRefresh={loadFitnessFunctions}
          />
        </TabsContent>

        <TabsContent value="executions" className="space-y-6">
          <ExecutionHistoryView
            executions={executions}
            selectedExperiment={selectedExperiment}
            onRefresh={() => selectedExperiment && loadExecutions(selectedExperiment.experiment_id || selectedExperiment.id)}
          />
        </TabsContent>

        <TabsContent value="reports" className="space-y-6">
          <ReportsView
            selectedExperiment={selectedExperiment}
          />
        </TabsContent>
      </Tabs>

      {/* Create Experiment Dialog */}
      <Dialog open={showCreateExperimentDialog} onOpenChange={setShowCreateExperimentDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Research Experiment</DialogTitle>
            <DialogDescription>
              Design a new A/B testing experiment to compare technique effectiveness with statistical rigor.
            </DialogDescription>
          </DialogHeader>

          <CreateExperimentForm
            data={newExperimentData}
            onChange={setNewExperimentData}
            onSubmit={handleCreateExperiment}
            onCancel={() => setShowCreateExperimentDialog(false)}
            creating={creating}
            fitnessFunctions={fitnessFunctions}
          />
        </DialogContent>
      </Dialog>

      {/* Create Fitness Function Dialog */}
      <Dialog open={showCreateFunctionDialog} onOpenChange={setShowCreateFunctionDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Fitness Function</DialogTitle>
            <DialogDescription>
              Define a custom fitness function for evaluating experiment results and technique effectiveness.
            </DialogDescription>
          </DialogHeader>

          <CreateFitnessFunctionForm
            data={newFitnessFunctionData}
            onChange={setNewFitnessFunctionData}
            onSubmit={handleCreateFitnessFunction}
            onCancel={() => setShowCreateFunctionDialog(false)}
            creating={creating}
          />
        </DialogContent>
      </Dialog>

      {/* Generate Report Dialog */}
      <Dialog open={showGenerateReportDialog} onOpenChange={setShowGenerateReportDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Generate Research Report</DialogTitle>
            <DialogDescription>
              Generate an academic research report for the selected experiment with publication-ready formatting.
            </DialogDescription>
          </DialogHeader>

          <GenerateReportForm
            data={reportGenerateData}
            onChange={setReportGenerateData}
            onSubmit={handleGenerateReport}
            onCancel={() => setShowGenerateReportDialog(false)}
            creating={creating}
          />
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Experiment</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{experimentToDelete?.title}&quot;? This action cannot be undone
              and will remove all execution data and findings.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction className="bg-red-600 hover:bg-red-700">
              Delete Experiment
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// Dashboard View Component
function ResearchDashboardView({
  analytics,
  onRefresh
}: {
  analytics: ResearchAnalytics | null;
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
        <h2 className="text-2xl font-bold">Research Analytics Dashboard</h2>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <FlaskRound className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold">{analytics.total_experiments}</div>
            <div className="text-sm text-muted-foreground">Total Experiments</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold">{analytics.completed_experiments}</div>
            <div className="text-sm text-muted-foreground">Completed</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Target className="h-8 w-8 mx-auto mb-2 text-orange-600" />
            <div className="text-2xl font-bold">{analytics.total_techniques_tested}</div>
            <div className="text-sm text-muted-foreground">Techniques Tested</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <TrendingUp className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold">{analytics.average_effect_size.toFixed(2)}</div>
            <div className="text-sm text-muted-foreground">Avg Effect Size</div>
          </CardContent>
        </Card>
      </div>

      {/* Research Productivity Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Research Productivity</CardTitle>
            <CardDescription>Experiments conducted per month</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <RechartsLineChart data={analytics.experiments_by_month}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="experiments" stroke="#8884d8" strokeWidth={2} />
              </RechartsLineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top Performing Techniques</CardTitle>
            <CardDescription>Success rates by technique</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={analytics.top_performing_techniques}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="technique" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="avg_success_rate" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Research Quality Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Statistical Rigor</CardTitle>
            <CardDescription>Quality of experimental design</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Significance Rate</span>
                <div className="flex items-center gap-2">
                  <Progress value={analytics.significance_rate * 100} className="w-24" />
                  <span className="text-sm font-mono w-12">{(analytics.significance_rate * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Replication Success</span>
                <div className="flex items-center gap-2">
                  <Progress value={analytics.replication_success_rate * 100} className="w-24" />
                  <span className="text-sm font-mono w-12">{(analytics.replication_success_rate * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Collaboration</CardTitle>
            <CardDescription>Research team activity</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-blue-600" />
                  <span className="text-sm">Active Researchers</span>
                </div>
                <span className="text-lg font-bold">{analytics.active_researchers}</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-green-600" />
                  <span className="text-sm">Cross-Workspace</span>
                </div>
                <span className="text-lg font-bold">{analytics.cross_workspace_collaborations}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Research Impact</CardTitle>
            <CardDescription>Innovation and discovery metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-yellow-600" />
                  <span className="text-sm">Novel Discoveries</span>
                </div>
                <span className="text-lg font-bold">8</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Award className="h-4 w-4 text-purple-600" />
                  <span className="text-sm">High-Impact Results</span>
                </div>
                <span className="text-lg font-bold">3</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// Placeholder components - in a real implementation these would be fully featured

function ExperimentListView({ experiments, selectedExperiment, onSelectExperiment, onExecuteExperiment, onDeleteExperiment, executing, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FlaskRound className="h-5 w-5" />
            Experiments
          </CardTitle>
          <Button size="sm" onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          {experiments.length} research experiment{experiments.length !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96">
          {experiments.length === 0 ? (
            <div className="text-center py-8 px-4">
              <FlaskRound className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <h4 className="font-semibold mb-2">No Experiments</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Create your first research experiment to start A/B testing techniques.
              </p>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {experiments.map((experiment: ExperimentDesign) => (
                <div
                  key={experiment.experiment_id || experiment.id}
                  className={`p-3 rounded-lg border transition-colors cursor-pointer ${
                    (selectedExperiment?.experiment_id || selectedExperiment?.id) === (experiment.experiment_id || experiment.id)
                      ? 'bg-primary/10 border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                  onClick={() => onSelectExperiment(experiment)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium truncate">{experiment.title}</h4>
                  </div>
                  <p className="text-sm text-muted-foreground truncate mb-2">
                    {experiment.research_question}
                  </p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                    <span>N = {experiment.sample_size}</span>
                    <span>{experiment.treatment_techniques.length + 1} conditions</span>
                  </div>
                  <div className="flex items-center gap-1 mt-2">
                    <Button
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); onExecuteExperiment(experiment); }}
                      disabled={executing}
                    >
                      <Play className="h-3 w-3 mr-1" />
                      Execute
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => { e.stopPropagation(); onDeleteExperiment(experiment); }}
                    >
                      <Trash2 className="h-3 w-3" />
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

function ExperimentDetailsView({ experiment, executions, onRefresh, onGenerateReport }: any) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>{experiment.title}</CardTitle>
              <CardDescription>{experiment.description}</CardDescription>
            </div>
            <Button onClick={onGenerateReport}>
              <FileText className="h-4 w-4 mr-2" />
              Generate Report
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <Label className="font-medium">Research Question</Label>
              <p className="text-sm mt-1">{experiment.research_question}</p>
            </div>
            <div>
              <Label className="font-medium">Hypothesis</Label>
              <p className="text-sm mt-1">{experiment.hypothesis}</p>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <Label className="font-medium">Control Technique</Label>
                <Badge variant="outline" className="mt-1">
                  {experiment.control_technique}
                </Badge>
              </div>
              <div>
                <Label className="font-medium">Sample Size</Label>
                <p>{experiment.sample_size}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Execution Results</CardTitle>
          <CardDescription>Statistical analysis and findings</CardDescription>
        </CardHeader>
        <CardContent>
          {!executions || executions.executions.length === 0 ? (
            <div className="text-center py-8">
              <TestTube className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <p className="text-muted-foreground">No executions yet</p>
            </div>
          ) : (
            <div className="space-y-4">
              {executions.executions.map((execution: ExperimentExecution) => (
                <div key={execution.execution_id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      {execution.status === 'completed' ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : execution.status === 'failed' ? (
                        <XCircle className="h-5 w-5 text-red-600" />
                      ) : (
                        <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />
                      )}
                      <span className="font-medium">{researchLabService.getStatusDisplayName(execution.status)}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {execution.started_at && new Date(execution.started_at).toLocaleString()}
                    </span>
                  </div>

                  {execution.status === 'completed' && execution.statistical_significance && (
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <Label className="font-medium">Statistical Significance</Label>
                        <p>{(execution.statistical_significance * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <Label className="font-medium">P-Value</Label>
                        <p>{execution.p_value ? researchLabService.formatPValue(execution.p_value) : 'N/A'}</p>
                      </div>
                      <div>
                        <Label className="font-medium">Winning Technique</Label>
                        <Badge variant="default">
                          {execution.winning_technique}
                        </Badge>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function FitnessFunctionsView({ functions, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Available Fitness Functions</CardTitle>
        <CardDescription>Custom evaluation functions for experiment analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Calculator className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Fitness functions will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

function ExecutionHistoryView({ executions, selectedExperiment, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution History</CardTitle>
        <CardDescription>
          {selectedExperiment ? `Results for "${selectedExperiment.title}"` : 'Select an experiment to view executions'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <TestTube className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Execution history will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

function ReportsView({ selectedExperiment }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Research Reports</CardTitle>
        <CardDescription>Generated academic reports and publications</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <BookOpen className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Research reports will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

// Form Components
function CreateExperimentForm({ data, onChange, onSubmit, onCancel, creating, fitnessFunctions }: any) {
  const techniques = researchLabService.getSuggestedTechniques();
  const datasets = researchLabService.getAvailableDatasets();

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Experiment Title</Label>
          <Input
            value={data.title}
            onChange={(e) => onChange({ ...data, title: e.target.value })}
            placeholder="GPTFuzz vs AutoDAN Effectiveness Comparison"
          />
        </div>
        <div className="space-y-2">
          <Label>Sample Size</Label>
          <Input
            type="number"
            value={data.sample_size}
            onChange={(e) => onChange({ ...data, sample_size: parseInt(e.target.value) || 100 })}
            min={10}
            max={10000}
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label>Description</Label>
        <Textarea
          value={data.description}
          onChange={(e) => onChange({ ...data, description: e.target.value })}
          placeholder="Detailed description of the research experiment..."
          className="min-h-20"
        />
      </div>

      <div className="space-y-2">
        <Label>Research Question</Label>
        <Textarea
          value={data.research_question}
          onChange={(e) => onChange({ ...data, research_question: e.target.value })}
          placeholder="Which technique generates more effective adversarial prompts?"
          className="min-h-16"
        />
      </div>

      <div className="space-y-2">
        <Label>Hypothesis</Label>
        <Textarea
          value={data.hypothesis}
          onChange={(e) => onChange({ ...data, hypothesis: e.target.value })}
          placeholder="AutoDAN will show higher success rates due to its reasoning-based approach"
          className="min-h-16"
        />
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
          Create Experiment
        </Button>
      </div>
    </div>
  );
}

function CreateFitnessFunctionForm({ data, onChange, onSubmit, onCancel, creating }: any) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Function Name</Label>
          <Input
            value={data.name}
            onChange={(e) => onChange({ ...data, name: e.target.value })}
            placeholder="Attack Success Rate"
          />
        </div>
        <div className="space-y-2">
          <Label>Output Type</Label>
          <Select
            value={data.output_type}
            onValueChange={(value) => onChange({ ...data, output_type: value })}
          >
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="float">Float (0.0 - 1.0)</SelectItem>
              <SelectItem value="score">Score (0 - 100)</SelectItem>
              <SelectItem value="boolean">Boolean (True/False)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <Label>Description</Label>
        <Textarea
          value={data.description}
          onChange={(e) => onChange({ ...data, description: e.target.value })}
          placeholder="Measures the percentage of successful attacks..."
          className="min-h-20"
        />
      </div>

      <div className="space-y-2">
        <Label>Function Code (Python)</Label>
        <Textarea
          value={data.code}
          onChange={(e) => onChange({ ...data, code: e.target.value })}
          className="font-mono text-sm min-h-48"
        />
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
          Create Function
        </Button>
      </div>
    </div>
  );
}

function GenerateReportForm({ data, onChange, onSubmit, onCancel, creating }: any) {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label>Report Title</Label>
        <Input
          value={data.title}
          onChange={(e) => onChange({ ...data, title: e.target.value })}
          placeholder="Comparative Analysis of Adversarial Prompting Techniques"
        />
      </div>

      <div className="space-y-2">
        <Label>Authors</Label>
        <div className="space-y-2">
          {data.authors.map((author: string, index: number) => (
            <Input
              key={index}
              value={author}
              onChange={(e) => {
                const newAuthors = [...data.authors];
                newAuthors[index] = e.target.value;
                onChange({ ...data, authors: newAuthors });
              }}
              placeholder={`Author ${index + 1}`}
            />
          ))}
          <Button
            variant="outline"
            size="sm"
            onClick={() => onChange({ ...data, authors: [...data.authors, ''] })}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Author
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        <Label>Abstract</Label>
        <Textarea
          value={data.abstract}
          onChange={(e) => onChange({ ...data, abstract: e.target.value })}
          placeholder="This study investigates the comparative effectiveness of adversarial prompting techniques..."
          className="min-h-32"
        />
      </div>

      <div className="space-y-2">
        <Label>Citation Style</Label>
        <Select
          value={data.citation_style}
          onValueChange={(value: CitationStyle) => onChange({ ...data, citation_style: value })}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="apa">APA</SelectItem>
            <SelectItem value="mla">MLA</SelectItem>
            <SelectItem value="ieee">IEEE</SelectItem>
            <SelectItem value="acm">ACM</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex justify-end space-x-2">
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button onClick={onSubmit} disabled={creating}>
          {creating ? (
            <RefreshCw className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <FileText className="h-4 w-4 mr-2" />
          )}
          Generate Report
        </Button>
      </div>
    </div>
  );
}
