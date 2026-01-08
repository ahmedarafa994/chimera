"use client";

/**
 * Evasion Task Page for Project Chimera
 * 
 * Provides evasion task functionality including:
 * - Creating new evasion tasks
 * - Monitoring task progress
 * - Viewing task results
 * - Task history management
 */

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  useEvasionTask,
  useEvasionTaskList,
  useEvasionForm,
} from "@/hooks";
import { EvasionTaskStatus } from "@/lib/types/evasion-types";
import {
  Shield,
  Play,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  Copy,
  Eye,
  Trash2,
} from "lucide-react";

// =============================================================================
// Status Badge Component
// =============================================================================

function StatusBadge({ status }: { status: EvasionTaskStatus }) {
  const variants: Record<EvasionTaskStatus, { variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode }> = {
    [EvasionTaskStatus.PENDING]: { variant: "secondary", icon: <Clock className="h-3 w-3" /> },
    [EvasionTaskStatus.RUNNING]: { variant: "default", icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    [EvasionTaskStatus.COMPLETED]: { variant: "outline", icon: <CheckCircle className="h-3 w-3 text-green-500" /> },
    [EvasionTaskStatus.FAILED]: { variant: "destructive", icon: <XCircle className="h-3 w-3" /> },
    [EvasionTaskStatus.CANCELLED]: { variant: "secondary", icon: <XCircle className="h-3 w-3" /> },
  };

  const { variant, icon } = variants[status] || variants[EvasionTaskStatus.PENDING];

  return (
    <Badge variant={variant} className="flex items-center gap-1">
      {icon}
      {status}
    </Badge>
  );
}

// =============================================================================
// Create Task Form
// =============================================================================

function CreateTaskForm({ onTaskCreated }: { onTaskCreated?: () => void }) {
  const { createTask, isLoading, error } = useEvasionTask();
  const form = useEvasionForm();
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);

    if (!form.isValid) {
      setSubmitError("Please fill in all required fields");
      return;
    }

    try {
      await createTask(form.buildRequest());
      form.reset();
      onTaskCreated?.();
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to create task");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {(error || submitError) && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error || submitError}</AlertDescription>
        </Alert>
      )}

      <div className="space-y-2">
        <Label htmlFor="initialPrompt">Initial Prompt *</Label>
        <Textarea
          id="initialPrompt"
          value={form.initialPrompt}
          onChange={(e) => form.setInitialPrompt(e.target.value)}
          placeholder="Enter the initial prompt to transform..."
          rows={4}
          className={form.errors.initialPrompt ? "border-destructive" : ""}
        />
        {form.errors.initialPrompt && (
          <p className="text-sm text-destructive">{form.errors.initialPrompt}</p>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="targetModelId">Target Model *</Label>
          <Input
            id="targetModelId"
            value={form.targetModelId}
            onChange={(e) => form.setTargetModelId(e.target.value)}
            placeholder="e.g., gpt-4"
            className={form.errors.targetModelId ? "border-destructive" : ""}
          />
          {form.errors.targetModelId && (
            <p className="text-sm text-destructive">{form.errors.targetModelId}</p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="evasionTechnique">Evasion Technique *</Label>
          <Select
            value={form.evasionTechnique}
            onValueChange={form.setEvasionTechnique}
          >
            <SelectTrigger className={form.errors.evasionTechnique ? "border-destructive" : ""}>
              <SelectValue placeholder="Select technique" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="gcg">GCG (Greedy Coordinate Gradient)</SelectItem>
              <SelectItem value="autodan">AutoDAN</SelectItem>
              <SelectItem value="pair">PAIR</SelectItem>
              <SelectItem value="tap">TAP (Tree of Attacks)</SelectItem>
              <SelectItem value="cipher">Cipher</SelectItem>
              <SelectItem value="multilingual">Multilingual</SelectItem>
            </SelectContent>
          </Select>
          {form.errors.evasionTechnique && (
            <p className="text-sm text-destructive">{form.errors.evasionTechnique}</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="maxIterations">Max Iterations</Label>
          <Input
            id="maxIterations"
            type="number"
            min={1}
            max={100}
            value={form.maxIterations}
            onChange={(e) => form.setMaxIterations(parseInt(e.target.value) || 10)}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="targetBehavior">Target Behavior (Optional)</Label>
          <Input
            id="targetBehavior"
            value={form.targetBehavior}
            onChange={(e) => form.setTargetBehavior(e.target.value)}
            placeholder="Describe expected behavior..."
          />
        </div>
      </div>

      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" onClick={form.reset}>
          Reset
        </Button>
        <Button type="submit" disabled={isLoading || !form.isValid}>
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Create Task
            </>
          )}
        </Button>
      </div>
    </form>
  );
}

// =============================================================================
// Task Details Dialog
// =============================================================================

function TaskDetailsDialog({
  taskId,
  open,
  onOpenChange,
}: {
  taskId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { task, isLoading, error, refresh } = useEvasionTask(taskId || undefined);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (!taskId) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Task Details
          </DialogTitle>
          <DialogDescription>
            Task ID: {taskId}
          </DialogDescription>
        </DialogHeader>

        {isLoading && (
          <div className="flex justify-center p-8">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {task && (
          <div className="space-y-6">
            {/* Status and Progress */}
            <div className="flex items-center justify-between">
              <StatusBadge status={task.status} />
              <Button variant="outline" size="sm" onClick={refresh}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>

            {task.status === EvasionTaskStatus.RUNNING && task.progress !== undefined && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Progress</span>
                  <span>{task.progress}%</span>
                </div>
                <Progress value={task.progress} />
              </div>
            )}

            {/* Initial Prompt */}
            {task.initial_prompt && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Initial Prompt</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(task.initial_prompt || "")}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
                <div className="p-3 bg-muted rounded-md font-mono text-sm whitespace-pre-wrap">
                  {task.initial_prompt}
                </div>
              </div>
            )}

            {/* Configuration */}
            <div className="grid grid-cols-2 gap-4">
              {task.target_model_id && (
                <div>
                  <Label className="text-muted-foreground">Target Model</Label>
                  <p className="font-mono">{task.target_model_id}</p>
                </div>
              )}
              {task.evasion_technique && (
                <div>
                  <Label className="text-muted-foreground">Technique</Label>
                  <p className="font-mono">{task.evasion_technique}</p>
                </div>
              )}
              {task.max_iterations !== undefined && (
                <div>
                  <Label className="text-muted-foreground">Max Iterations</Label>
                  <p>{task.max_iterations}</p>
                </div>
              )}
              {task.target_behavior && (
                <div>
                  <Label className="text-muted-foreground">Target Behavior</Label>
                  <p>{task.target_behavior}</p>
                </div>
              )}
            </div>

            {/* Result */}
            {task.result && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Transformed Prompt</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(task.result?.transformed_prompt || "")}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-md font-mono text-sm whitespace-pre-wrap">
                  {task.result.transformed_prompt}
                </div>

                {task.result.success_score !== undefined && (
                  <div className="flex items-center gap-2">
                    <Label className="text-muted-foreground">Success Score:</Label>
                    <Badge variant={task.result.success_score > 0.7 ? "default" : "secondary"}>
                      {(task.result.success_score * 100).toFixed(1)}%
                    </Badge>
                  </div>
                )}

                {task.result.iterations_used !== undefined && (
                  <div className="flex items-center gap-2">
                    <Label className="text-muted-foreground">Iterations Used:</Label>
                    <span>{task.result.iterations_used}</span>
                  </div>
                )}
              </div>
            )}

            {/* Error */}
            {task.error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Task Error</AlertTitle>
                <AlertDescription>{task.error}</AlertDescription>
              </Alert>
            )}

            {/* Timestamps */}
            <div className="text-sm text-muted-foreground space-y-1">
              {task.created_at && <p>Created: {new Date(task.created_at).toLocaleString()}</p>}
              {task.started_at && <p>Started: {new Date(task.started_at).toLocaleString()}</p>}
              {task.completed_at && <p>Completed: {new Date(task.completed_at).toLocaleString()}</p>}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// Task List
// =============================================================================

function TaskList() {
  const { tasks, isLoading, error, refresh, deleteTask } = useEvasionTaskList();
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const handleViewDetails = (taskId: string) => {
    setSelectedTaskId(taskId);
    setDetailsOpen(true);
  };

  const handleDeleteTask = async (taskId: string) => {
    if (!confirm("Are you sure you want to delete this task?")) return;
    
    try {
      await deleteTask(taskId);
    } catch (err) {
      console.error("Failed to delete task:", err);
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-medium">Task History</h3>
        <Button variant="outline" size="sm" onClick={refresh}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Task ID</TableHead>
            <TableHead>Technique</TableHead>
            <TableHead>Target Model</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {tasks.map((task) => (
            <TableRow key={task.task_id}>
              <TableCell className="font-mono text-xs">
                {task.task_id.substring(0, 8)}...
              </TableCell>
              <TableCell>
                <Badge variant="outline">{task.evasion_technique}</Badge>
              </TableCell>
              <TableCell className="font-mono text-sm">{task.target_model_id}</TableCell>
              <TableCell>
                <StatusBadge status={task.status} />
              </TableCell>
              <TableCell className="text-sm text-muted-foreground">
                {new Date(task.created_at).toLocaleDateString()}
              </TableCell>
              <TableCell className="text-right">
                <div className="flex justify-end gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleViewDetails(task.task_id)}
                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteTask(task.task_id)}
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
          {tasks.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} className="text-center text-muted-foreground">
                No tasks found. Create your first evasion task above.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      <TaskDetailsDialog
        taskId={selectedTaskId}
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
      />
    </>
  );
}

// =============================================================================
// Main Evasion Page
// =============================================================================

export default function EvasionPage() {
  const { refresh: refreshList } = useEvasionTaskList();

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Shield className="h-8 w-8" />
          Evasion Tasks
        </h1>
        <p className="text-muted-foreground mt-2">
          Create and manage prompt evasion tasks using various techniques.
        </p>
      </div>

      <Tabs defaultValue="create" className="space-y-4">
        <TabsList>
          <TabsTrigger value="create" className="flex items-center gap-2">
            <Play className="h-4 w-4" />
            Create Task
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Task History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="create">
          <Card>
            <CardHeader>
              <CardTitle>Create Evasion Task</CardTitle>
              <CardDescription>
                Configure and launch a new prompt evasion task.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CreateTaskForm onTaskCreated={refreshList} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardContent className="pt-6">
              <TaskList />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}