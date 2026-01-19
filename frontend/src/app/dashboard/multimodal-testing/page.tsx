/**
 * Multi-Modal Attack Testing Dashboard
 *
 * Phase 4 innovation feature for next-generation security:
 * - Vision+text and audio+text attack capabilities
 * - Image captioning vulnerability testing
 * - Unified reporting across modalities
 * - Future-proofing for multi-modal LLMs
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
  Eye,
  Mic,
  Video,
  Upload,
  Play,
  Pause,
  Plus,
  Trash2,
  Settings,
  FileImage,
  FileAudio,
  FileVideo,
  RefreshCw,
  Search,
  Filter,
  BarChart3,
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  AlertTriangle,
  CheckCircle,
  Info,
  XCircle,
  Download,
  ExternalLink,
  Copy,
  Edit,
  Target,
  Layers,
  Cpu,
  Brain
} from 'lucide-react';
import { toast } from 'sonner';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

// Import services
import {
  multimodalTestingService,
  MultiModalTestSuite,
  MultiModalTestExecution,
  MultiModalAnalytics,
  MultiModalTestCreate,
  MultiModalTestUpdate,
  MultiModalPromptCreate,
  MediaUploadResponse,
  TestSuiteListResponse,
  ExecutionListResponse,
  ModalityType,
  AttackVector,
  TestStatus,
  VulnerabilityLevel
} from '@/lib/api/services/multimodal';

export default function MultiModalTestingPage() {
  // Data state
  const [analytics, setAnalytics] = useState<MultiModalAnalytics | null>(null);
  const [testSuites, setTestSuites] = useState<TestSuiteListResponse | null>(null);
  const [selectedSuite, setSelectedSuite] = useState<MultiModalTestSuite | null>(null);
  const [executions, setExecutions] = useState<ExecutionListResponse | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<MediaUploadResponse[]>([]);

  // Form state
  const [newTestSuiteData, setNewTestSuiteData] = useState<MultiModalTestCreate>({
    name: '',
    description: '',
    target_models: [],
    attack_vectors: [],
    modality_types: []
  });
  const [newPromptData, setNewPromptData] = useState<MultiModalPromptCreate>({
    text_content: '',
    attack_vector: 'visual_prompt_injection',
    modality_type: 'vision_text',
    instructions: '',
    context: ''
  });
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showPromptDialog, setShowPromptDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [suiteToDelete, setSuiteToDelete] = useState<MultiModalTestSuite | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterModality, setFilterModality] = useState<ModalityType | 'all'>('all');

  useEffect(() => {
    loadAnalytics();
    loadTestSuites();
  }, []);

  const loadAnalytics = useCallback(async () => {
    try {
      const data = await multimodalTestingService.getAnalytics();
      setAnalytics(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const loadTestSuites = useCallback(async () => {
    try {
      setLoading(true);
      const params = filterModality !== 'all' ? { modality_type: filterModality } : undefined;
      const data = await multimodalTestingService.listTestSuites(params);
      setTestSuites(data);
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, [filterModality]);

  const loadExecutions = useCallback(async (suiteId: string) => {
    try {
      const data = await multimodalTestingService.listSuiteExecutions(suiteId);
      setExecutions(data);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    try {
      setUploading(true);
      const uploadPromises = Array.from(files).map(file =>
        multimodalTestingService.uploadMediaFile(file)
      );

      const uploadResults = await Promise.all(uploadPromises);
      setUploadedFiles(prev => [...prev, ...uploadResults]);

      toast.success(`${uploadResults.length} file(s) uploaded successfully`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setUploading(false);
    }
  }, []);

  const handleCreateTestSuite = useCallback(async () => {
    const errors = multimodalTestingService.validateTestSuiteCreate(newTestSuiteData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      const suite = await multimodalTestingService.createTestSuite(newTestSuiteData);

      // Refresh data
      await Promise.all([loadTestSuites(), loadAnalytics()]);

      // Select the new suite
      setSelectedSuite(suite);

      // Reset form
      setNewTestSuiteData({
        name: '',
        description: '',
        target_models: [],
        attack_vectors: [],
        modality_types: []
      });
      setShowCreateDialog(false);

      toast.success(`Test suite "${suite.name}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newTestSuiteData, loadTestSuites, loadAnalytics]);

  const handleExecuteTestSuite = useCallback(async (suite: MultiModalTestSuite) => {
    try {
      setExecuting(true);
      await multimodalTestingService.executeTestSuite(suite.suite_id);

      // Refresh executions if this suite is selected
      if (selectedSuite?.suite_id === suite.suite_id) {
        await loadExecutions(suite.suite_id);
      }

      await loadAnalytics();
    } catch (error) {
      // Error already handled in service
    } finally {
      setExecuting(false);
    }
  }, [selectedSuite, loadExecutions, loadAnalytics]);

  const handleDeleteTestSuite = useCallback(async () => {
    if (!suiteToDelete) return;

    try {
      await multimodalTestingService.deleteTestSuite(suiteToDelete.suite_id);

      // Refresh data
      await Promise.all([loadTestSuites(), loadAnalytics()]);

      // Clear selection if deleted suite was selected
      if (selectedSuite?.suite_id === suiteToDelete.suite_id) {
        setSelectedSuite(null);
      }

      setSuiteToDelete(null);
      setShowDeleteDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [suiteToDelete, loadTestSuites, loadAnalytics, selectedSuite]);

  const filteredSuites = testSuites?.suites?.filter(suite => {
    const matchesSearch = suite.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         (suite.description || '').toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  }) || [];

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Multi-Modal Testing...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your multi-modal attack testing data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Multi-Modal Attack Testing
        </h1>
        <p className="text-muted-foreground text-lg">
          Advanced security assessment combining vision, audio, and text modalities for comprehensive AI model testing.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
          <TabsTrigger value="suites">Test Suites</TabsTrigger>
          <TabsTrigger value="executions">Executions</TabsTrigger>
          <TabsTrigger value="media">Media Files</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="space-y-6">
          <MultiModalDashboardView
            analytics={analytics}
            onRefresh={loadAnalytics}
          />
        </TabsContent>

        <TabsContent value="suites" className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-bold">Multi-Modal Test Suites</h2>
              <Badge variant="outline">
                {testSuites?.total || 0} suite{(testSuites?.total || 0) !== 1 ? 's' : ''}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search suites..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-64"
                />
              </div>
              <Select value={filterModality} onValueChange={(value: ModalityType | 'all') => setFilterModality(value)}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Modalities</SelectItem>
                  {multimodalTestingService.getAvailableModalityTypes().map(type => (
                    <SelectItem key={type.id} value={type.id}>
                      {type.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button onClick={() => setShowCreateDialog(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Test Suite
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Test Suite List */}
            <div className="lg:col-span-1">
              <TestSuiteListView
                suites={filteredSuites}
                selectedSuite={selectedSuite}
                onSelectSuite={(suite: MultiModalTestSuite) => {
                  setSelectedSuite(suite);
                  loadExecutions(suite.suite_id);
                }}
                onExecuteSuite={handleExecuteTestSuite}
                onDeleteSuite={(suite: MultiModalTestSuite) => {
                  setSuiteToDelete(suite);
                  setShowDeleteDialog(true);
                }}
                executing={executing}
                onRefresh={loadTestSuites}
              />
            </div>

            {/* Suite Details */}
            <div className="lg:col-span-2">
              {selectedSuite ? (
                <TestSuiteDetailsView
                  suite={selectedSuite}
                  executions={executions}
                  onRefresh={() => loadExecutions(selectedSuite.suite_id)}
                />
              ) : (
                <Card>
                  <CardContent className="text-center py-12">
                    <Layers className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      No Test Suite Selected
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300">
                      Select a multi-modal test suite from the list to view details and execution history.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="executions" className="space-y-6">
          <ExecutionHistoryView
            executions={executions}
            selectedSuite={selectedSuite}
            onRefresh={() => selectedSuite && loadExecutions(selectedSuite.suite_id)}
          />
        </TabsContent>

        <TabsContent value="media" className="space-y-6">
          <MediaFilesView
            uploadedFiles={uploadedFiles}
            onFileUpload={handleFileUpload}
            uploading={uploading}
            selectedFiles={selectedFiles}
            onFileSelect={setSelectedFiles}
          />
        </TabsContent>
      </Tabs>

      {/* Create Test Suite Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Create Multi-Modal Test Suite</DialogTitle>
            <DialogDescription>
              Set up a comprehensive test suite combining multiple modalities for advanced AI security assessment.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Test Suite Name</Label>
                <Input
                  value={newTestSuiteData.name}
                  onChange={(e) => setNewTestSuiteData(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Vision+Audio Security Assessment"
                />
              </div>

              <div className="space-y-2">
                <Label>Target Models</Label>
                <Select
                  value={newTestSuiteData.target_models.join(',')}
                  onValueChange={(value) => setNewTestSuiteData(prev => ({
                    ...prev,
                    target_models: value ? value.split(',') : []
                  }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select models" />
                  </SelectTrigger>
                  <SelectContent>
                    {multimodalTestingService.getSuggestedModels().map(model => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={newTestSuiteData.description}
                onChange={(e) => setNewTestSuiteData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Comprehensive multi-modal security assessment targeting vision and audio capabilities..."
                className="min-h-20"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label>Modality Types</Label>
                <div className="grid grid-cols-2 gap-2">
                  {multimodalTestingService.getAvailableModalityTypes().map(type => (
                    <div key={type.id} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id={`modality-${type.id}`}
                        checked={newTestSuiteData.modality_types.includes(type.id)}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setNewTestSuiteData(prev => ({
                            ...prev,
                            modality_types: checked
                              ? [...prev.modality_types, type.id]
                              : prev.modality_types.filter(t => t !== type.id)
                          }));
                        }}
                      />
                      <Label htmlFor={`modality-${type.id}`} className="text-sm">
                        {type.name}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Attack Vectors</Label>
                <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto">
                  {multimodalTestingService.getAvailableAttackVectors().map(vector => (
                    <div key={vector.id} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id={`vector-${vector.id}`}
                        checked={newTestSuiteData.attack_vectors.includes(vector.id)}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setNewTestSuiteData(prev => ({
                            ...prev,
                            attack_vectors: checked
                              ? [...prev.attack_vectors, vector.id]
                              : prev.attack_vectors.filter(v => v !== vector.id)
                          }));
                        }}
                      />
                      <Label htmlFor={`vector-${vector.id}`} className="text-sm">
                        {vector.name}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateTestSuite} disabled={creating}>
                {creating ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Create Test Suite
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Test Suite</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{suiteToDelete?.name}&quot;? This action cannot be undone
              and will remove all execution history and associated media files.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteTestSuite} className="bg-red-600 hover:bg-red-700">
              Delete Test Suite
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// Dashboard View Component
function MultiModalDashboardView({
  analytics,
  onRefresh
}: {
  analytics: MultiModalAnalytics | null;
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
        <h2 className="text-2xl font-bold">Multi-Modal Analytics Dashboard</h2>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <Layers className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold">{analytics?.total_suites ?? 0}</div>
            <div className="text-sm text-muted-foreground">Total Test Suites</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Activity className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold">{analytics?.total_executions ?? 0}</div>
            <div className="text-sm text-muted-foreground">Total Executions</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Shield className="h-8 w-8 mx-auto mb-2 text-orange-600" />
            <div className="text-2xl font-bold">{analytics?.vulnerability_rate?.toFixed(1) ?? 0}%</div>
            <div className="text-sm text-muted-foreground">Vulnerability Rate</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Target className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold">
              {analytics ? (Object.values(analytics.attack_vector_success) as number[]).reduce((a: number, b: number) => a + b, 0) : 0}
            </div>
            <div className="text-sm text-muted-foreground">Successful Attacks</div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Vulnerability Trend</CardTitle>
            <CardDescription>Daily vulnerability detection over time</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={analytics?.vulnerability_trend ?? []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="vulnerabilities" stroke="#ff7c7c" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Execution Volume</CardTitle>
            <CardDescription>Daily test execution count</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={analytics?.daily_execution_trend ?? []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="executions" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Additional Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Attack Vector Success Rates</CardTitle>
            <CardDescription>Success rate by attack vector type</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analytics ? Object.entries(analytics.attack_vector_success).map(([vector, success], index) => (
                <div key={vector} className="flex items-center justify-between">
                  <span className="text-sm">{multimodalTestingService.getAttackVectorDisplayName(vector as AttackVector)}</span>
                  <div className="flex items-center gap-2">
                    <Progress value={success as number} className="w-24" />
                    <span className="text-sm font-mono w-12">{(success as number).toFixed(1)}%</span>
                  </div>
                </div>
              )) : null}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Modality Distribution</CardTitle>
            <CardDescription>Test execution breakdown by modality</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={analytics ? Object.entries(analytics.modality_breakdown).map(([modality, count]) => ({
                    name: multimodalTestingService.getModalityTypeDisplayName(modality as ModalityType),
                    value: count
                  })) : []}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {analytics ? Object.entries(analytics.modality_breakdown).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  )) : null}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Model Robustness */}
      <Card>
        <CardHeader>
          <CardTitle>Model Robustness Scores</CardTitle>
          <CardDescription>Robustness rating across different AI models</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {analytics ? (Object.entries(analytics.model_robustness) as [string, number][]).map(([model, score]) => (
              <div key={model} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium">{model}</h4>
                  <Badge variant={score >= 80 ? "default" : score >= 60 ? "secondary" : "destructive"}>
                    {score.toFixed(1)}%
                  </Badge>
                </div>
                <Progress value={score} className="h-2" />
                <p className="text-xs text-muted-foreground mt-1">
                  {score >= 80 ? "High robustness" : score >= 60 ? "Medium robustness" : "Low robustness"}
                </p>
              </div>
            )) : null}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Placeholder components - in a real implementation these would be fully featured

function TestSuiteListView({ suites, selectedSuite, onSelectSuite, onExecuteSuite, onDeleteSuite, executing, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Test Suites
          </CardTitle>
          <Button size="sm" onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          {suites.length} multi-modal test suite{suites.length !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96">
          {suites.length === 0 ? (
            <div className="text-center py-8 px-4">
              <Layers className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <h4 className="font-semibold mb-2">No Test Suites</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Create your first multi-modal test suite to start advanced security testing.
              </p>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {suites.map((suite: MultiModalTestSuite) => (
                <div
                  key={suite.suite_id}
                  className={`p-3 rounded-lg border transition-colors cursor-pointer ${
                    selectedSuite?.suite_id === suite.suite_id
                      ? 'bg-primary/10 border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                  onClick={() => onSelectSuite(suite)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium truncate">{suite.name}</h4>
                    <div className="flex gap-1">
                      <Badge variant="outline" className="text-xs">
                        {suite.total_tests ?? 0} tests
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground truncate mb-2">
                    {suite.description}
                  </p>
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-2">
                    <span>{suite.modality_types?.length ?? 0} modalities</span>
                    <span>{suite.attack_vectors?.length ?? 0} attack vectors</span>
                  </div>
                  <div className="flex items-center gap-1 mt-2">
                    <Button
                      size="sm"
                      onClick={(e) => { e.stopPropagation(); onExecuteSuite(suite); }}
                      disabled={executing}
                    >
                      <Play className="h-3 w-3 mr-1" />
                      Execute
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => { e.stopPropagation(); onDeleteSuite(suite); }}
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

function TestSuiteDetailsView({ suite, executions, onRefresh }: any) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>{suite.name}</CardTitle>
          <CardDescription>{suite.description}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <Label className="font-medium">Target Models</Label>
              <div className="flex flex-wrap gap-1 mt-1">
                {suite.target_models.map((model: string) => (
                  <Badge key={model} variant="secondary" className="text-xs">
                    {model}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <Label className="font-medium">Success Rate</Label>
              <p>{suite.success_rate.toFixed(1)}%</p>
            </div>
            <div>
              <Label className="font-medium">Modality Types</Label>
              <div className="flex flex-wrap gap-1 mt-1">
                {suite.modality_types.map((type: ModalityType) => (
                  <Badge key={type} variant="outline" className="text-xs">
                    {multimodalTestingService.getModalityTypeDisplayName(type)}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <Label className="font-medium">Vulnerabilities Found</Label>
              <p>{suite.vulnerabilities_found}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Executions</CardTitle>
          <CardDescription>Latest execution history for this test suite</CardDescription>
        </CardHeader>
        <CardContent>
          {!executions || executions.executions.length === 0 ? (
            <div className="text-center py-8">
              <Play className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <p className="text-muted-foreground">No executions yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {executions.executions.map((execution: MultiModalTestExecution) => (
                <div key={execution.execution_id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    {execution.status === 'completed' ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : execution.status === 'failed' ? (
                      <XCircle className="h-5 w-5 text-red-600" />
                    ) : execution.status === 'running' ? (
                      <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />
                    ) : (
                      <Info className="h-5 w-5 text-yellow-600" />
                    )}
                    <div>
                      <p className="font-medium">{execution.status.toUpperCase()}</p>
                      <p className="text-sm text-muted-foreground">
                        {execution.vulnerability_detected ? 'Vulnerabilities detected' : 'No vulnerabilities'}
                        {execution.vulnerability_level && execution.vulnerability_level !== 'none' && (
                          <Badge
                            variant="outline"
                            className={`ml-2 bg-${multimodalTestingService.getVulnerabilityLevelColor(execution.vulnerability_level)}-50`}
                          >
                            {multimodalTestingService.getVulnerabilityLevelDisplayName(execution.vulnerability_level)}
                          </Badge>
                        )}
                      </p>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {execution.started_at && new Date(execution.started_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ExecutionHistoryView({ executions, selectedSuite, onRefresh }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution History</CardTitle>
        <CardDescription>
          {selectedSuite ? `Executions for "${selectedSuite.name}"` : 'Select a test suite to view executions'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Play className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Execution history will appear here</p>
        </div>
      </CardContent>
    </Card>
  );
}

function MediaFilesView({ uploadedFiles, onFileUpload, uploading, selectedFiles, onFileSelect }: any) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Media Files</h2>
        <div className="flex items-center gap-2">
          <input
            type="file"
            id="file-upload"
            multiple
            accept="image/*,audio/*,video/*"
            onChange={onFileUpload}
            className="hidden"
            disabled={uploading}
          />
          <Button asChild disabled={uploading}>
            <label htmlFor="file-upload" className="cursor-pointer">
              {uploading ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Upload className="h-4 w-4 mr-2" />
              )}
              Upload Media
            </label>
          </Button>
        </div>
      </div>

      <Card>
        <CardContent>
          {uploadedFiles.length === 0 ? (
            <div className="text-center py-12">
              <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                No Media Files
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Upload images, audio, or video files to use in your multi-modal tests.
              </p>
              <Button asChild disabled={uploading}>
                <label htmlFor="file-upload" className="cursor-pointer">
                  {uploading ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Upload className="h-4 w-4 mr-2" />
                  )}
                  Upload First File
                </label>
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {uploadedFiles.map((file: any) => (
                <div key={file.file_id} className="border rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    {file.content_type.startsWith('image/') ? (
                      <FileImage className="h-5 w-5 text-blue-600" />
                    ) : file.content_type.startsWith('audio/') ? (
                      <FileAudio className="h-5 w-5 text-green-600" />
                    ) : (
                      <FileVideo className="h-5 w-5 text-purple-600" />
                    )}
                    <h4 className="font-medium truncate">{file.filename}</h4>
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {multimodalTestingService.formatFileSize(file.file_size)}
                  </p>
                  <div className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      id={`file-${file.file_id}`}
                      checked={selectedFiles.includes(file.file_id)}
                      onChange={(e) => {
                        const checked = e.target.checked;
                        onFileSelect((prev: string[]) =>
                          checked
                            ? [...prev, file.file_id]
                            : prev.filter(id => id !== file.file_id)
                        );
                      }}
                    />
                    <Label htmlFor={`file-${file.file_id}`} className="text-sm">
                      Select
                    </Label>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
