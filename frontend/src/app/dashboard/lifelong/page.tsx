"use client";

/**
 * AutoDAN Lifelong Learning Page
 * 
 * Provides interface for:
 * - Single attack execution
 * - Continuous loop attacks
 * - Progress monitoring
 * - Results history
 */

import { useState, useCallback, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { autodanLifelongService } from "@/lib/services/autodan-lifelong-service";
import {
  LifelongAttackRequest,
  LifelongLoopRequest,
  LifelongAttackResponse,
  LifelongProgressResponse,
  LifelongTaskStatus,
  LifelongDisplayResult,
} from "@/lib/types/autodan-lifelong-types";
import {
  Brain,
  Pause,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  Copy,
  Eye,
  Zap,
  Repeat,
  Target,
} from "lucide-react";

// =============================================================================
// Status Badge Component
// =============================================================================

type StatusBadgeStatus = LifelongTaskStatus | "not_initialized" | "running" | "completed" | "failed" | "cancelled";

function StatusBadge({ status }: { status: StatusBadgeStatus }) {
  const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode; label: string }> = {
    [LifelongTaskStatus.PENDING]: { variant: "secondary", icon: <Clock className="h-3 w-3" />, label: "Pending" },
    [LifelongTaskStatus.RUNNING]: { variant: "default", icon: <Loader2 className="h-3 w-3 animate-spin" />, label: "Running" },
    [LifelongTaskStatus.COMPLETED]: { variant: "outline", icon: <CheckCircle className="h-3 w-3 text-green-500" />, label: "Completed" },
    [LifelongTaskStatus.FAILED]: { variant: "destructive", icon: <XCircle className="h-3 w-3" />, label: "Failed" },
    [LifelongTaskStatus.CANCELLED]: { variant: "secondary", icon: <Pause className="h-3 w-3" />, label: "Cancelled" },
    "not_initialized": { variant: "secondary", icon: <Clock className="h-3 w-3" />, label: "Not Initialized" },
  };

  const { variant, icon, label } = variants[status] || variants[LifelongTaskStatus.PENDING];

  return (
    <Badge variant={variant} className="flex items-center gap-1">
      {icon}
      {label}
    </Badge>
  );
}

// =============================================================================
// Single Attack Form
// =============================================================================

interface SingleAttackFormProps {
  onSubmit: (request: LifelongAttackRequest) => Promise<void>;
  isLoading: boolean;
}

const DEFAULT_SELECT_VALUE = "__default__";

function SingleAttackForm({ onSubmit, isLoading }: SingleAttackFormProps) {
  const [request, setRequest] = useState("");
  const [model, setModel] = useState(DEFAULT_SELECT_VALUE);
  const [provider, setProvider] = useState(DEFAULT_SELECT_VALUE);
  const [detailedScoring, setDetailedScoring] = useState(true);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    await onSubmit({
      request,
      model: model === DEFAULT_SELECT_VALUE ? undefined : model,
      provider: provider === DEFAULT_SELECT_VALUE ? undefined : provider,
      detailed_scoring: detailedScoring,
    });
  };

  const isValid = request.trim() !== "";

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="request">Harmful Request *</Label>
        <Textarea
          id="request"
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          placeholder="Enter the harmful request to generate a jailbreak for..."
          rows={4}
        />
        <p className="text-xs text-muted-foreground">
          The request that you want to jailbreak the target model with.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="model">Target Model (optional)</Label>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger>
              <SelectValue placeholder="Use session default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_SELECT_VALUE}>Use session default</SelectItem>
              <SelectItem value="gpt-4">GPT-4</SelectItem>
              <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
              <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
              <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
              <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="provider">Provider (optional)</Label>
          <Select value={provider} onValueChange={setProvider}>
            <SelectTrigger>
              <SelectValue placeholder="Use session default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_SELECT_VALUE}>Use session default</SelectItem>
              <SelectItem value="openai">OpenAI</SelectItem>
              <SelectItem value="anthropic">Anthropic</SelectItem>
              <SelectItem value="google">Google</SelectItem>
              <SelectItem value="deepseek">DeepSeek</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Switch
          id="detailedScoring"
          checked={detailedScoring}
          onCheckedChange={setDetailedScoring}
        />
        <Label htmlFor="detailedScoring">Detailed Scoring (LLM-based analysis)</Label>
      </div>

      <Button type="submit" disabled={isLoading || !isValid} className="w-full">
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Running Attack...
          </>
        ) : (
          <>
            <Zap className="h-4 w-4 mr-2" />
            Execute Single Attack
          </>
        )}
      </Button>
    </form>
  );
}

// =============================================================================
// Loop Attack Form
// =============================================================================

interface LoopAttackFormProps {
  onSubmit: (request: LifelongLoopRequest) => Promise<void>;
  isLoading: boolean;
}

function LoopAttackForm({ onSubmit, isLoading }: LoopAttackFormProps) {
  const [requests, setRequests] = useState("");
  const [model, setModel] = useState(DEFAULT_SELECT_VALUE);
  const [provider, setProvider] = useState(DEFAULT_SELECT_VALUE);
  const [epochs, setEpochs] = useState(5);
  const [breakScore, setBreakScore] = useState(9.0);
  const [warmup, setWarmup] = useState(false);
  const [warmupIterations, setWarmupIterations] = useState(2);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const requestList = requests.split("\n").filter((p) => p.trim() !== "");
    
    await onSubmit({
      requests: requestList,
      model: model === DEFAULT_SELECT_VALUE ? undefined : model,
      provider: provider === DEFAULT_SELECT_VALUE ? undefined : provider,
      epochs,
      break_score: breakScore,
      warmup,
      warmup_iterations: warmupIterations,
    });
  };

  const isValid = requests.trim() !== "";

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="requests">Harmful Requests (one per line) *</Label>
        <Textarea
          id="requests"
          value={requests}
          onChange={(e) => setRequests(e.target.value)}
          placeholder="Enter harmful requests, one per line..."
          rows={6}
        />
        <p className="text-xs text-muted-foreground">
          {requests.split("\n").filter((p) => p.trim() !== "").length} requests entered
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="loopModel">Target Model (optional)</Label>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger>
              <SelectValue placeholder="Use session default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_SELECT_VALUE}>Use session default</SelectItem>
              <SelectItem value="gpt-4">GPT-4</SelectItem>
              <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
              <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
              <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
              <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="loopProvider">Provider (optional)</Label>
          <Select value={provider} onValueChange={setProvider}>
            <SelectTrigger>
              <SelectValue placeholder="Use session default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_SELECT_VALUE}>Use session default</SelectItem>
              <SelectItem value="openai">OpenAI</SelectItem>
              <SelectItem value="anthropic">Anthropic</SelectItem>
              <SelectItem value="google">Google</SelectItem>
              <SelectItem value="deepseek">DeepSeek</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="epochs">Epochs: {epochs}</Label>
          <Slider
            id="epochs"
            min={1}
            max={20}
            step={1}
            value={[epochs]}
            onValueChange={([value]) => setEpochs(value)}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="breakScore">Break Score: {breakScore.toFixed(1)}</Label>
          <Slider
            id="breakScore"
            min={1}
            max={10}
            step={0.5}
            value={[breakScore]}
            onValueChange={([value]) => setBreakScore(value)}
          />
          <p className="text-xs text-muted-foreground">
            Stop early when this score is achieved (1-10 scale)
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Switch
            id="warmup"
            checked={warmup}
            onCheckedChange={setWarmup}
          />
          <Label htmlFor="warmup">Enable Warm-up Phase</Label>
        </div>

        {warmup && (
          <div className="space-y-2 pl-6">
            <Label htmlFor="warmupIterations">Warm-up Iterations: {warmupIterations}</Label>
            <Slider
              id="warmupIterations"
              min={1}
              max={10}
              step={1}
              value={[warmupIterations]}
              onValueChange={([value]) => setWarmupIterations(value)}
            />
          </div>
        )}
      </div>

      <Button type="submit" disabled={isLoading || !isValid} className="w-full">
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Running Loop...
          </>
        ) : (
          <>
            <Repeat className="h-4 w-4 mr-2" />
            Start Lifelong Learning Loop
          </>
        )}
      </Button>
    </form>
  );
}

// =============================================================================
// Progress Monitor
// =============================================================================

interface ProgressMonitorProps {
  taskId: string | null;
  onComplete?: (result: LifelongAttackResponse) => void;
}

function ProgressMonitor({ taskId, onComplete }: ProgressMonitorProps) {
  const [progress, setProgress] = useState<LifelongProgressResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!taskId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- Reset progress when taskId changes
      setProgress(null);
      return;
    }

    setError(null);

    const pollProgress = async () => {
      try {
        const response = await autodanLifelongService.getProgress(taskId);
        setProgress(response);

        if (response.status === LifelongTaskStatus.COMPLETED ||
            response.status === LifelongTaskStatus.FAILED ||
            response.status === LifelongTaskStatus.CANCELLED) {
          clearInterval(intervalId);

          if (response.status === LifelongTaskStatus.COMPLETED && response.result) {
            onComplete?.(response.result);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to get progress");
        clearInterval(intervalId);
      }
    };

    pollProgress();
    const intervalId: NodeJS.Timeout = setInterval(pollProgress, 2000);

    return () => {
      clearInterval(intervalId);
    };
  }, [taskId, onComplete]);

  if (!taskId) {
    return (
      <div className="text-center text-muted-foreground p-8">
        <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No active task. Start an attack to see progress.</p>
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

  if (!progress) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <StatusBadge status={progress.status} />
        <span className="text-sm text-muted-foreground">
          Task: {taskId.substring(0, 8)}...
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Progress</span>
          <span>{progress.progress}%</span>
        </div>
        <Progress value={progress.progress} />
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-muted-foreground">Current Iteration:</span>
          <span className="ml-2 font-medium">{progress.current_iteration}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Total Iterations:</span>
          <span className="ml-2 font-medium">{progress.total_iterations}</span>
        </div>
        {progress.best_score !== undefined && (
          <div>
            <span className="text-muted-foreground">Best Score:</span>
            <span className="ml-2 font-medium">{(progress.best_score * 100).toFixed(1)}%</span>
          </div>
        )}
        {progress.elapsed_time !== undefined && (
          <div>
            <span className="text-muted-foreground">Elapsed Time:</span>
            <span className="ml-2 font-medium">{progress.elapsed_time.toFixed(1)}s</span>
          </div>
        )}
      </div>

      {progress.current_prompt && (
        <div className="space-y-2">
          <Label>Current Best Prompt</Label>
          <div className="p-3 bg-muted rounded-md font-mono text-sm whitespace-pre-wrap max-h-32 overflow-y-auto">
            {progress.current_prompt}
          </div>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Results Display
// =============================================================================

interface ResultsDisplayProps {
  results: LifelongDisplayResult[];
  onViewDetails: (result: LifelongDisplayResult) => void;
}

function ResultsDisplay({ results, onViewDetails }: ResultsDisplayProps) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  if (results.length === 0) {
    return (
      <div className="text-center text-muted-foreground p-8">
        <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No results yet. Complete an attack to see results.</p>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Task ID</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Score</TableHead>
          <TableHead>Iterations</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {results.map((result, index) => (
          <TableRow key={result.task_id || index}>
            <TableCell className="font-mono text-xs">
              {result.task_id?.substring(0, 8) || `result-${index}`}...
            </TableCell>
            <TableCell>
              <Badge variant={result.success ? "default" : "secondary"}>
                {result.success ? "Success" : "Partial"}
              </Badge>
            </TableCell>
            <TableCell>
              {result.best_score !== undefined ? `${(result.best_score * 100).toFixed(1)}%` : "N/A"}
            </TableCell>
            <TableCell>{result.iterations_used || "N/A"}</TableCell>
            <TableCell className="text-right">
              <div className="flex justify-end gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(result.jailbreak_prompt)}
                >
                  <Copy className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onViewDetails(result)}
                >
                  <Eye className="h-4 w-4" />
                </Button>
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function LifelongLearningPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [results, setResults] = useState<LifelongDisplayResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<LifelongDisplayResult | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const handleSingleAttack = useCallback(async (request: LifelongAttackRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await autodanLifelongService.executeSingleAttack(request);
      // Single attack returns result directly, not a task_id
      // Convert to the display format
      const displayResult: LifelongDisplayResult = {
        task_id: `attack-${Date.now()}`,
        success: response.is_jailbreak,
        jailbreak_prompt: response.prompt,
        method: "lifelong_single",
        best_score: response.score / 10, // Convert 1-10 to 0-1 for display
        iterations_used: 1,
        execution_time: response.latency_ms / 1000,
      };
      setResults((prev) => [displayResult, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to execute attack");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleLoopAttack = useCallback(async (request: LifelongLoopRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await autodanLifelongService.startLifelongLoop(request);
      // Loop returns results directly, not a task_id
      if (response.results) {
        // Convert loop results to display results
        const displayResults: LifelongDisplayResult[] = response.results.map((r, idx) => ({
          task_id: `loop-${idx}`,
          success: r.success,
          jailbreak_prompt: r.best_prompt,
          method: "lifelong_loop",
          best_score: r.best_score / 10, // Convert 1-10 to 0-1 for display
          iterations_used: r.attempts,
        }));
        setResults((prev) => [...displayResults, ...prev]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start loop attack");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleComplete = useCallback((result: LifelongAttackResponse) => {
    // Convert API response to display result
    const displayResult: LifelongDisplayResult = {
      task_id: `completed-${Date.now()}`,
      success: result.is_jailbreak,
      jailbreak_prompt: result.prompt,
      method: "lifelong_progress",
      best_score: result.score / 10,
      iterations_used: 1,
      execution_time: result.latency_ms / 1000,
    };
    setResults((prev) => [displayResult, ...prev]);
    setCurrentTaskId(null);
  }, []);

  const handleViewDetails = useCallback((result: LifelongDisplayResult) => {
    setSelectedResult(result);
    setDetailsOpen(true);
  }, []);

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Brain className="h-8 w-8" />
          AutoDAN Lifelong Learning
        </h1>
        <p className="text-muted-foreground mt-2">
          Advanced jailbreak attacks with continuous learning and memory.
        </p>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Attack Forms */}
        <div className="space-y-6">
          <Tabs defaultValue="single">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="single" className="flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Single Attack
              </TabsTrigger>
              <TabsTrigger value="loop" className="flex items-center gap-2">
                <Repeat className="h-4 w-4" />
                Loop Attack
              </TabsTrigger>
            </TabsList>

            <TabsContent value="single">
              <Card>
                <CardHeader>
                  <CardTitle>Single Attack</CardTitle>
                  <CardDescription>
                    Execute a single jailbreak attack with lifelong learning.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SingleAttackForm onSubmit={handleSingleAttack} isLoading={isLoading} />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="loop">
              <Card>
                <CardHeader>
                  <CardTitle>Continuous Loop Attack</CardTitle>
                  <CardDescription>
                    Run multiple attack iterations with adaptive learning.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <LoopAttackForm onSubmit={handleLoopAttack} isLoading={isLoading} />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Right Column: Progress & Results */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Progress Monitor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ProgressMonitor taskId={currentTaskId} onComplete={handleComplete} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Results History</CardTitle>
              <CardDescription>
                {results.length} completed attacks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResultsDisplay results={results} onViewDetails={handleViewDetails} />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Result Details Dialog */}
      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Attack Result Details</DialogTitle>
            <DialogDescription>
              Task ID: {selectedResult?.task_id}
            </DialogDescription>
          </DialogHeader>

          {selectedResult && (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Badge variant={selectedResult.success ? "default" : "secondary"}>
                  {selectedResult.success ? "Success" : "Partial Success"}
                </Badge>
                {selectedResult.best_score !== undefined && (
                  <Badge variant="outline">
                    Score: {(selectedResult.best_score * 100).toFixed(1)}%
                  </Badge>
                )}
              </div>

              <div className="space-y-2">
                <Label>Jailbreak Prompt</Label>
                <div className="p-3 bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-md font-mono text-sm whitespace-pre-wrap">
                  {selectedResult.jailbreak_prompt}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigator.clipboard.writeText(selectedResult.jailbreak_prompt)}
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy Prompt
                </Button>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Method:</span>
                  <span className="ml-2 font-medium">{selectedResult.method}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Iterations:</span>
                  <span className="ml-2 font-medium">{selectedResult.iterations_used || "N/A"}</span>
                </div>
                {selectedResult.execution_time && (
                  <div>
                    <span className="text-muted-foreground">Execution Time:</span>
                    <span className="ml-2 font-medium">{selectedResult.execution_time.toFixed(2)}s</span>
                  </div>
                )}
                {selectedResult.memory_entries_used !== undefined && (
                  <div>
                    <span className="text-muted-foreground">Memory Entries Used:</span>
                    <span className="ml-2 font-medium">{selectedResult.memory_entries_used}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}