/**
 * Custom Technique Builder Interface
 *
 * Phase 3 enterprise feature for advanced users:
 * - Visual interface for creating custom transformations
 * - Drag-and-drop technique combination
 * - Team sharing and version control
 * - Effectiveness tracking over time
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
import {
  Palette,
  Plus,
  Play,
  Save,
  Copy,
  Trash2,
  Settings,
  Eye,
  Share2,
  Download,
  Upload,
  RefreshCw,
  Search,
  Filter,
  Zap,
  Code,
  GitBranch,
  Activity,
  BarChart3,
  Target,
  Lightbulb,
  Layers,
  Box,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Clock,
  Users,
  Lock,
  Globe,
  Star
} from 'lucide-react';
import { toast } from 'sonner';

// Import services
import {
  techniqueBuilderService,
  CustomTechnique,
  TechniqueTemplate,
  TechniqueExecution,
  TechniqueParameter,
  TechniqueStep,
  TechniqueCreate,
  TechniqueUpdate,
  TechniqueTestRequest,
  TechniqueListResponse,
  TechniqueStatsResponse,
  TechniqueType,
  TechniqueStatus,
  VisibilityLevel,
  ParameterType
} from '@/lib/api/services/technique-builder';

export default function TechniqueBuilderPage() {
  // Data state
  const [techniques, setTechniques] = useState<TechniqueListResponse | null>(null);
  const [selectedTechnique, setSelectedTechnique] = useState<CustomTechnique | null>(null);
  const [techniqueStats, setTechniqueStats] = useState<TechniqueStatsResponse | null>(null);

  // Form state
  const [newTechniqueData, setNewTechniqueData] = useState<TechniqueCreate>({
    name: '',
    description: '',
    category: 'transformation'
  });
  const [editTechniqueData, setEditTechniqueData] = useState<TechniqueUpdate>({});
  const [testRequest, setTestRequest] = useState<TechniqueTestRequest>({
    test_input: ''
  });

  // UI state
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showTestDialog, setShowTestDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showStatsDialog, setShowStatsDialog] = useState(false);
  const [techniqueToDelete, setTechniqueToDelete] = useState<CustomTechnique | null>(null);
  const [activeTab, setActiveTab] = useState('techniques');
  const [selectedTemplate, setSelectedTemplate] = useState<TechniqueTemplate | null>(null);

  useEffect(() => {
    loadTechniques();
  }, []);

  const loadTechniques = useCallback(async () => {
    try {
      setLoading(true);
      const data = await techniqueBuilderService.listTechniques({ page: 1, page_size: 50 });
      setTechniques(data);
    } catch (error) {
      // Error already handled in service
    } finally {
      setLoading(false);
    }
  }, []);

  const loadTechniqueStats = useCallback(async (techniqueId: string) => {
    try {
      const stats = await techniqueBuilderService.getTechniqueStats(techniqueId);
      setTechniqueStats(stats);
    } catch (error) {
      // Error already handled in service
    }
  }, []);

  const handleCreateTechnique = useCallback(async () => {
    const errors = techniqueBuilderService.validateTechniqueCreate(newTechniqueData);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setCreating(true);
      const technique = await techniqueBuilderService.createTechnique(newTechniqueData);

      // Refresh techniques list
      await loadTechniques();

      // Select the new technique
      setSelectedTechnique(technique);

      // Reset form
      setNewTechniqueData({
        name: '',
        description: '',
        category: 'transformation'
      });
      setShowCreateDialog(false);

      toast.success(`Technique "${technique.name}" created successfully!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [newTechniqueData, loadTechniques]);

  const handleCreateFromTemplate = useCallback(async (template: TechniqueTemplate) => {
    try {
      setCreating(true);
      const technique = await techniqueBuilderService.createTechnique({
        name: `${template.name} - Custom`,
        description: `Based on ${template.description}`,
        category: template.category,
        technique_type: 'transformation'
      });

      // Update with template data
      const updatedTechnique = await techniqueBuilderService.updateTechnique(technique.technique_id, {
        parameters: template.parameters_template,
        steps: template.steps_template
      });

      // Refresh and select
      await loadTechniques();
      setSelectedTechnique(updatedTechnique);

      toast.success(`Technique created from template "${template.name}"!`);
    } catch (error) {
      // Error already handled in service
    } finally {
      setCreating(false);
    }
  }, [loadTechniques]);

  const handleUpdateTechnique = useCallback(async () => {
    if (!selectedTechnique) return;

    try {
      const updatedTechnique = await techniqueBuilderService.updateTechnique(
        selectedTechnique.technique_id,
        editTechniqueData
      );

      setSelectedTechnique(updatedTechnique);
      await loadTechniques();
      setEditTechniqueData({});
    } catch (error) {
      // Error already handled in service
    }
  }, [selectedTechnique, editTechniqueData, loadTechniques]);

  const handleTestTechnique = useCallback(async () => {
    if (!selectedTechnique) return;

    const errors = techniqueBuilderService.validateTestRequest(testRequest);
    if (errors.length > 0) {
      toast.error(errors.join(', '));
      return;
    }

    try {
      setTesting(true);
      const execution = await techniqueBuilderService.testTechnique(
        selectedTechnique.technique_id,
        testRequest
      );

      // Show test results
      toast.success(`Test completed in ${techniqueBuilderService.formatExecutionTime(execution.execution_time)}`);

      // Refresh technique stats
      await loadTechniqueStats(selectedTechnique.technique_id);
      setShowTestDialog(false);
    } catch (error) {
      // Error already handled in service
    } finally {
      setTesting(false);
    }
  }, [selectedTechnique, testRequest, loadTechniqueStats]);

  const handleCloneTechnique = useCallback(async (technique: CustomTechnique) => {
    try {
      const clonedTechnique = await techniqueBuilderService.cloneTechnique(technique.technique_id);

      // Refresh and select cloned technique
      await loadTechniques();
      setSelectedTechnique(clonedTechnique);
    } catch (error) {
      // Error already handled in service
    }
  }, [loadTechniques]);

  const handleDeleteTechnique = useCallback(async () => {
    if (!techniqueToDelete) return;

    try {
      await techniqueBuilderService.deleteTechnique(techniqueToDelete.technique_id);

      // Refresh techniques list
      await loadTechniques();

      // Clear selection if deleted technique was selected
      if (selectedTechnique?.technique_id === techniqueToDelete.technique_id) {
        setSelectedTechnique(null);
      }

      setTechniqueToDelete(null);
      setShowDeleteDialog(false);
    } catch (error) {
      // Error already handled in service
    }
  }, [techniqueToDelete, loadTechniques, selectedTechnique]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col gap-2 px-4">
          <div className="flex items-center gap-2">
            <RefreshCw className="h-6 w-6 animate-spin" />
            <h1 className="text-4xl font-bold tracking-tight">Loading Technique Builder...</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Please wait while we load your custom techniques and templates.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-2 px-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Custom Technique Builder
        </h1>
        <p className="text-muted-foreground text-lg">
          Create, test, and share custom attack techniques with visual drag-and-drop interface and advanced configuration options.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Sidebar: Technique List */}
        <div className="lg:col-span-1">
          <TechniqueListView
            techniques={techniques}
            selectedTechnique={selectedTechnique}
            onSelectTechnique={setSelectedTechnique}
            onCreateNew={() => setShowCreateDialog(true)}
            onRefresh={loadTechniques}
          />
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          {selectedTechnique ? (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div>
                    <h2 className="text-2xl font-bold flex items-center gap-2">
                      <Palette className="h-6 w-6" />
                      {selectedTechnique.name}
                    </h2>
                    <p className="text-muted-foreground">{selectedTechnique.description}</p>
                  </div>
                  <div className="flex gap-2">
                    <Badge variant="outline" className={`bg-${techniqueBuilderService.getTechniqueTypeColor(selectedTechnique.technique_type)}-50`}>
                      {techniqueBuilderService.getTechniqueTypeDisplayName(selectedTechnique.technique_type)}
                    </Badge>
                    <Badge variant="outline" className={`bg-${techniqueBuilderService.getStatusColor(selectedTechnique.status)}-50`}>
                      {techniqueBuilderService.getStatusDisplayName(selectedTechnique.status)}
                    </Badge>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setShowTestDialog(true)}
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Test
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleCloneTechnique(selectedTechnique)}
                  >
                    <Copy className="h-4 w-4 mr-2" />
                    Clone
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      loadTechniqueStats(selectedTechnique.technique_id);
                      setShowStatsDialog(true);
                    }}
                  >
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Stats
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setTechniqueToDelete(selectedTechnique);
                      setShowDeleteDialog(true);
                    }}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                </div>
              </div>

              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="editor">Visual Editor</TabsTrigger>
                <TabsTrigger value="parameters">Parameters</TabsTrigger>
                <TabsTrigger value="settings">Settings</TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
              </TabsList>

              <TabsContent value="editor" className="space-y-6">
                <TechniqueEditorView
                  technique={selectedTechnique}
                  onUpdate={handleUpdateTechnique}
                />
              </TabsContent>

              <TabsContent value="parameters" className="space-y-6">
                <TechniqueParametersView
                  technique={selectedTechnique}
                  onUpdate={(parameters: TechniqueParameter[]) => {
                    setEditTechniqueData({ parameters });
                    handleUpdateTechnique();
                  }}
                />
              </TabsContent>

              <TabsContent value="settings" className="space-y-6">
                <TechniqueSettingsView
                  technique={selectedTechnique}
                  onUpdate={handleUpdateTechnique}
                />
              </TabsContent>

              <TabsContent value="history" className="space-y-6">
                <TechniqueHistoryView
                  technique={selectedTechnique}
                  onLoadStats={() => loadTechniqueStats(selectedTechnique.technique_id)}
                />
              </TabsContent>
            </Tabs>
          ) : (
            <div className="space-y-6">
              {/* No technique selected - show overview */}
              <TechniqueOverviewView
                techniques={techniques}
                onCreateNew={() => setShowCreateDialog(true)}
                onCreateFromTemplate={handleCreateFromTemplate}
                onSelectTechnique={setSelectedTechnique}
              />
            </div>
          )}
        </div>
      </div>

      {/* Create Technique Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create Custom Technique</DialogTitle>
            <DialogDescription>
              Build a new custom attack technique with visual editor and advanced configuration.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Technique Name</Label>
              <Input
                value={newTechniqueData.name}
                onChange={(e) => setNewTechniqueData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Custom Prompt Injection"
              />
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                value={newTechniqueData.description}
                onChange={(e) => setNewTechniqueData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Describe what this technique does and how it works..."
                className="min-h-20"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Category</Label>
                <Select
                  value={newTechniqueData.category}
                  onValueChange={(value) => setNewTechniqueData(prev => ({ ...prev, category: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {techniqueBuilderService.getCategories().map(category => (
                      <SelectItem key={category.id} value={category.id}>
                        {category.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Type</Label>
                <Select
                  value={newTechniqueData.technique_type || 'transformation'}
                  onValueChange={(value: TechniqueType) => setNewTechniqueData(prev => ({ ...prev, technique_type: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="transformation">Transformation</SelectItem>
                    <SelectItem value="validation">Validation</SelectItem>
                    <SelectItem value="execution">Execution</SelectItem>
                    <SelectItem value="composition">Composition</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateTechnique} disabled={creating}>
                {creating ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Create Technique
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Test Technique Dialog */}
      <Dialog open={showTestDialog} onOpenChange={setShowTestDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Test Technique</DialogTitle>
            <DialogDescription>
              Test "{selectedTechnique?.name}" with sample input to validate functionality.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Test Input</Label>
              <Textarea
                value={testRequest.test_input}
                onChange={(e) => setTestRequest(prev => ({ ...prev, test_input: e.target.value }))}
                placeholder="Enter test prompt or text to transform..."
                className="min-h-24"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Model Provider (Optional)</Label>
                <Select
                  value={testRequest.model_provider || ''}
                  onValueChange={(value) => setTestRequest(prev => ({ ...prev, model_provider: value || undefined }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                    <SelectItem value="google">Google</SelectItem>
                    <SelectItem value="deepseek">DeepSeek</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Model Name (Optional)</Label>
                <Input
                  value={testRequest.model_name || ''}
                  onChange={(e) => setTestRequest(prev => ({ ...prev, model_name: e.target.value || undefined }))}
                  placeholder="gpt-4, claude-3.5-sonnet, etc."
                />
              </div>
            </div>

            <div className="flex justify-end space-x-2">
              <Button variant="outline" onClick={() => setShowTestDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleTestTechnique} disabled={testing}>
                {testing ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Run Test
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Technique Stats Dialog */}
      <Dialog open={showStatsDialog} onOpenChange={setShowStatsDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Technique Statistics</DialogTitle>
            <DialogDescription>
              Performance metrics and usage analytics for "{selectedTechnique?.name}"
            </DialogDescription>
          </DialogHeader>

          {techniqueStats && (
            <TechniqueStatsView
              stats={techniqueStats}
              technique={selectedTechnique!}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Technique</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{techniqueToDelete?.name}"? This action cannot be undone
              and will remove all associated test results and statistics.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteTechnique} className="bg-red-600 hover:bg-red-700">
              Delete Technique
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

// Placeholder components for the different views - in a real implementation these would be fully featured

function TechniqueListView({ techniques, selectedTechnique, onSelectTechnique, onCreateNew, onRefresh }: any) {
  if (!techniques) return <div>Loading...</div>;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Techniques
          </CardTitle>
          <Button size="sm" onClick={onRefresh} variant="outline">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>
          {techniques.total_techniques} custom technique{techniques.total_techniques !== 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96">
          {techniques.total_techniques === 0 ? (
            <div className="text-center py-8 px-4">
              <Palette className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <h4 className="font-semibold mb-2">No Custom Techniques</h4>
              <p className="text-sm text-muted-foreground mb-3">
                Create your first custom technique to get started.
              </p>
              <Button size="sm" onClick={onCreateNew}>
                <Plus className="h-4 w-4 mr-1" />
                Create Technique
              </Button>
            </div>
          ) : (
            <div className="space-y-1 p-2">
              {techniques.techniques.map((technique: CustomTechnique) => (
                <button
                  key={technique.technique_id}
                  onClick={() => onSelectTechnique(technique)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedTechnique?.technique_id === technique.technique_id
                      ? 'bg-primary/10 border border-primary/20'
                      : 'hover:bg-muted'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-medium truncate">{technique.name}</h4>
                    <div className="flex gap-1">
                      <Badge variant="outline" className="text-xs">
                        {technique.category}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground truncate">
                    {technique.description}
                  </p>
                  <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Activity className="h-3 w-3" />
                      {technique.usage_count} uses
                    </span>
                    <span className="flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" />
                      {techniqueBuilderService.formatSuccessRate(technique.success_rate)}
                    </span>
                  </div>
                </button>
              ))}

              <Button
                variant="outline"
                className="w-full mt-2"
                onClick={onCreateNew}
              >
                <Plus className="h-4 w-4 mr-2" />
                New Technique
              </Button>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function TechniqueOverviewView({ techniques, onCreateNew, onCreateFromTemplate, onSelectTechnique }: any) {
  if (!techniques) return null;

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <Code className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <div className="text-2xl font-bold">{techniques.total_techniques}</div>
            <div className="text-sm text-muted-foreground">Custom Techniques</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Layers className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <div className="text-2xl font-bold">{techniques.total_templates}</div>
            <div className="text-sm text-muted-foreground">Templates Available</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Activity className="h-8 w-8 mx-auto mb-2 text-orange-600" />
            <div className="text-2xl font-bold">{techniques.techniques.reduce((sum: number, t: CustomTechnique) => sum + t.usage_count, 0)}</div>
            <div className="text-sm text-muted-foreground">Total Executions</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <CheckCircle className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <div className="text-2xl font-bold">
              {techniques.techniques.length > 0
                ? Math.round(techniques.techniques.reduce((sum: number, t: CustomTechnique) => sum + t.success_rate, 0) / techniques.techniques.length * 100)
                : 0}%
            </div>
            <div className="text-sm text-muted-foreground">Avg Success Rate</div>
          </CardContent>
        </Card>
      </div>

      {/* Templates */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Quick Start Templates</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {techniques.templates.map((template: TechniqueTemplate) => (
            <Card key={template.template_id} className="cursor-pointer hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold">{template.name}</h4>
                  <Badge variant="outline">{template.category}</Badge>
                </div>
                <p className="text-sm text-muted-foreground mb-4">{template.description}</p>
                <div className="flex justify-between items-center">
                  <div className="text-xs text-muted-foreground">
                    {template.usage_count} uses
                  </div>
                  <Button size="sm" onClick={() => onCreateFromTemplate(template)}>
                    <Plus className="h-4 w-4 mr-1" />
                    Use Template
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Recent Techniques */}
      {techniques.techniques.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4">Recent Techniques</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {techniques.techniques.slice(0, 4).map((technique: CustomTechnique) => (
              <Card key={technique.technique_id} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => onSelectTechnique(technique)}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold">{technique.name}</h4>
                    <Badge variant="outline" className={`bg-${techniqueBuilderService.getStatusColor(technique.status)}-50`}>
                      {techniqueBuilderService.getStatusDisplayName(technique.status)}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2">{technique.description}</p>
                  <div className="flex justify-between items-center text-xs text-muted-foreground">
                    <span>{technique.usage_count} executions</span>
                    <span>{techniqueBuilderService.formatSuccessRate(technique.success_rate)} success</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Placeholder components - in real implementation these would be full-featured
function TechniqueEditorView({ technique, onUpdate }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Visual Editor</CardTitle>
        <CardDescription>Drag and drop interface for building technique workflows</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-96 bg-muted/20 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <Box className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="font-semibold mb-2">Visual Editor</h3>
            <p className="text-muted-foreground">Drag-and-drop interface coming soon</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface TechniqueParametersViewProps {
  technique: CustomTechnique;
  onUpdate: (parameters: TechniqueParameter[]) => void;
}

function TechniqueParametersView({ technique, onUpdate }: TechniqueParametersViewProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
        <CardDescription>Configure technique parameters and validation rules</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {technique.parameters.length === 0 ? (
            <div className="text-center py-8">
              <Settings className="h-8 w-8 text-gray-400 mx-auto mb-3" />
              <p className="text-muted-foreground">No parameters defined</p>
            </div>
          ) : (
            technique.parameters.map((param: TechniqueParameter, index: number) => (
              <div key={index} className="border rounded-lg p-4">
                <h4 className="font-medium">{param.display_name}</h4>
                <p className="text-sm text-muted-foreground">{param.description}</p>
                <div className="flex gap-2 mt-2">
                  <Badge variant="outline">{techniqueBuilderService.getParameterTypeDisplayName(param.parameter_type)}</Badge>
                  {param.required && <Badge variant="secondary">Required</Badge>}
                </div>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function TechniqueSettingsView({ technique, onUpdate }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Settings</CardTitle>
        <CardDescription>Technique configuration and sharing options</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label className="font-medium">Status</Label>
            <Badge variant="outline" className={`bg-${techniqueBuilderService.getStatusColor(technique.status)}-50`}>
              {techniqueBuilderService.getStatusDisplayName(technique.status)}
            </Badge>
          </div>
          <div>
            <Label className="font-medium">Visibility</Label>
            <Badge variant="outline" className={`bg-${techniqueBuilderService.getVisibilityColor(technique.visibility)}-50`}>
              {techniqueBuilderService.getVisibilityDisplayName(technique.visibility)}
            </Badge>
          </div>
          <div>
            <Label className="font-medium">Complexity</Label>
            <p className="text-sm">{techniqueBuilderService.formatComplexityScore(technique.complexity_score)}</p>
          </div>
          <div>
            <Label className="font-medium">Category</Label>
            <p className="text-sm">{technique.category}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function TechniqueHistoryView({ technique, onLoadStats }: any) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution History</CardTitle>
        <CardDescription>Recent test executions and performance metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8">
          <Clock className="h-8 w-8 text-gray-400 mx-auto mb-3" />
          <p className="text-muted-foreground">Execution history will appear here</p>
          <Button onClick={onLoadStats} className="mt-2">Load Statistics</Button>
        </div>
      </CardContent>
    </Card>
  );
}

function TechniqueStatsView({ stats, technique }: { stats: TechniqueStatsResponse; technique: CustomTechnique }) {
  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{stats.executions.length}</div>
            <div className="text-sm text-muted-foreground">Total Executions</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{techniqueBuilderService.formatSuccessRate(stats.success_rate)}</div>
            <div className="text-sm text-muted-foreground">Success Rate</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold">{techniqueBuilderService.formatExecutionTime(stats.average_execution_time)}</div>
            <div className="text-sm text-muted-foreground">Avg Duration</div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Executions */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Executions</CardTitle>
        </CardHeader>
        <CardContent>
          {stats.executions.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">No executions yet</p>
          ) : (
            <div className="space-y-3">
              {stats.executions.map((execution) => (
                <div key={execution.execution_id} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {execution.success ? (
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                      )}
                      <span className="font-medium">{execution.success ? 'Success' : 'Failed'}</span>
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {techniqueBuilderService.formatExecutionTime(execution.execution_time)}
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {new Date(execution.executed_at).toLocaleString()}
                  </p>
                  {execution.error_message && (
                    <p className="text-sm text-red-600 mt-1">{execution.error_message}</p>
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
