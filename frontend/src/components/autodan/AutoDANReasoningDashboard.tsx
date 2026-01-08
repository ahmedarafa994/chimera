"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { useAutoDANReasoning } from "@/hooks/use-autodan-reasoning";
import { useUnifiedModel } from "@/providers/unified-model-provider";
import { Loader2, Play, Settings, History, Shield, Zap, Brain, Search, Layers, RefreshCw, Database } from "lucide-react";
import { toast } from "sonner";
import { BenchmarkDatasetSelector } from "./BenchmarkDatasetSelector";
import { BenchmarkPrompt } from "@/lib/services/benchmark-dataset-service";

export function AutoDANReasoningDashboard() {
  // Track if component is mounted (client-side) to prevent hydration mismatch
  const [isMounted, setIsMounted] = useState(false);

  const { isLoading, error, errorInfo, currentResult, generate, fetchConfig, config, clearError } = useAutoDANReasoning();
  const {
    selection,
    providers,
    isInitialized: isModelSelectionReady,
    isLoading: isModelSelectionLoading,
    setSelection: selectProvider,
  } = useUnifiedModel();

  const selectedProvider = selection.provider;
  const selectedModel = selection.model;
  const [request, setRequest] = useState("");
  const [method, setMethod] = useState<any>("best_of_n");
  const [bestOfN, setBestOfN] = useState(4);
  const [beamWidth, setBeamWidth] = useState(4);
  const [beamDepth, setBeamDepth] = useState(3);
  const [refinementIterations, setRefinementIterations] = useState(1);

  // Local provider override for AutoDAN
  const [localProvider, setLocalProvider] = useState<string>("deepseek");
  const [localModel, setLocalModel] = useState<string>("deepseek-chat");

  // Benchmark dataset integration
  const [selectedBenchmarkPrompt, setSelectedBenchmarkPrompt] = useState<BenchmarkPrompt | null>(null);
  const [benchmarkBatch, setBenchmarkBatch] = useState<BenchmarkPrompt[]>([]);

  // Use local override values directly
  const effectiveProvider = localProvider;
  const effectiveModel = localModel;

  const sessionReady = Boolean(effectiveProvider && effectiveModel);
  const disableActions = isLoading || isModelSelectionLoading || !sessionReady || !isModelSelectionReady;

  // Set mounted state on client
  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  // Handle benchmark prompt selection
  const handleBenchmarkPromptSelect = (prompt: BenchmarkPrompt) => {
    setSelectedBenchmarkPrompt(prompt);
    setRequest(prompt.prompt);
    toast.success("Benchmark Prompt Selected", {
      description: `Selected ${prompt.risk_area?.replace(/_/g, " ")} prompt for testing.`,
    });
  };

  // Handle batch generation from benchmark dataset
  const handleBenchmarkBatchGenerate = (prompts: BenchmarkPrompt[]) => {
    setBenchmarkBatch(prompts);
    toast.success("Batch Ready", {
      description: `${prompts.length} prompts ready for batch testing.`,
    });
  };

  // Handle provider change
  const handleProviderChange = (provider: string) => {
    setLocalProvider(provider);
    // Set default model for the provider
    const providerInfo = providers.find(p => p.provider === provider);
    if (providerInfo?.defaultModel) {
      setLocalModel(providerInfo.defaultModel);
    } else if (provider === "deepseek") {
      setLocalModel("deepseek-chat");
    } else if (provider === "openai") {
      setLocalModel("gpt-4o");
    } else if (provider === "gemini" || provider === "google") {
      setLocalModel("gemini-3-pro-preview");
    } else if (provider === "anthropic") {
      setLocalModel("claude-sonnet-4-5");
    }
    toast.info(`Provider changed to ${provider}`);
  };

  /**
   * Displays a user-friendly error toast with detailed information.
   * Parses Pydantic validation errors and provides actionable guidance.
   * @requirements 3.1, 3.3, 3.4
   */
  const showErrorToast = (err: unknown) => {
    // Use errorInfo if available for detailed error display
    if (errorInfo) {
      // Build description with field errors if present
      let description = errorInfo.guidance || "Please try again.";

      if (errorInfo.fieldErrors && errorInfo.fieldErrors.length > 0) {
        const fieldErrorsText = errorInfo.fieldErrors.slice(0, 3).join("; ");
        description = `${fieldErrorsText}${errorInfo.fieldErrors.length > 3 ? ` (+${errorInfo.fieldErrors.length - 3} more)` : ""}`;
      }

      // Determine toast type based on error code
      const isValidationError = errorInfo.code === "VALIDATION_ERROR";
      const isNetworkError = errorInfo.code === "NETWORK_ERROR" || errorInfo.code === "SERVICE_UNAVAILABLE";
      const isRateLimitError = errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED";

      if (isValidationError) {
        toast.warning("Validation Error", {
          description,
          duration: 8000, // Longer duration for validation errors
        });
      } else if (isRateLimitError) {
        toast.error("API Quota Exceeded", {
          description: errorInfo.guidance || "Your API quota has been exceeded. Please wait or switch providers.",
          duration: 10000, // Longer duration for quota errors
          action: {
            label: "Settings",
            onClick: () => {
              // Could navigate to settings page to change provider
            },
          },
        });
      } else if (isNetworkError) {
        toast.error("Connection Error", {
          description: errorInfo.guidance || "Unable to reach the server.",
          action: {
            label: "Retry",
            onClick: () => handleGenerate(),
          },
        });
      } else {
        toast.error("Generation Failed", {
          description,
          duration: 6000,
        });
      }

      // Log detailed error info in development
      if (process.env.NODE_ENV === "development") {
        console.group("[AutoDANReasoningDashboard] Error Details");
        console.error("Error Code:", errorInfo.code);
        console.error("Status Code:", errorInfo.statusCode);
        console.error("Message:", errorInfo.message);
        if (errorInfo.fieldErrors) {
          console.error("Field Errors:", errorInfo.fieldErrors);
        }
        if (errorInfo.requestId) {
          console.error("Request ID:", errorInfo.requestId);
        }
        console.groupEnd();
      }
    } else {
      // Fallback for errors without errorInfo
      const message = err instanceof Error ? err.message : "An unexpected error occurred";
      toast.error("Generation Failed", {
        description: message,
      });
    }
  };

  const handleGenerate = async () => {
    // Clear any previous errors
    clearError();

    if (!request.trim()) {
      toast.warning("Missing Input", {
        description: "Please enter a request to generate a jailbreak prompt.",
      });
      return;
    }

    if (!sessionReady) {
      toast.error("Model Selection Required", {
        description: "Connect to the backend model router before running AutoDAN.",
        action: {
          label: "Settings",
          onClick: () => {
            // Could navigate to settings page
          },
        },
      });
      return;
    }

    try {
      await generate({
        request,
        method,
        target_model: effectiveModel ?? undefined,
        provider: effectiveProvider ?? undefined,
        best_of_n: method === "best_of_n" ? bestOfN : undefined,
        beam_width: method === "beam_search" ? beamWidth : undefined,
        beam_depth: method === "beam_search" ? beamDepth : undefined,
        refinement_iterations: (method === "chain_of_thought" || method === "adaptive") ? refinementIterations : undefined,
      });
      toast.success("Jailbreak Generated", {
        description: `Successfully generated using ${method} method.`,
      });
    } catch (err) {
      showErrorToast(err);
    }
  };

  // Show loading skeleton during SSR/hydration to prevent mismatch
  if (!isMounted) {
    return (
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-9 w-64 mb-2" />
            <Skeleton className="h-5 w-96" />
          </div>
          <Skeleton className="h-8 w-40" />
        </div>
        <div className="grid gap-6 md:grid-cols-2">
          <Card className="md:col-span-1">
            <CardHeader>
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-64 mt-2" />
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-10 w-full" />
            </CardContent>
            <CardFooter>
              <Skeleton className="h-10 w-full" />
            </CardFooter>
          </Card>
          <Card className="md:col-span-1">
            <CardHeader>
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-64 mt-2" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-64 w-full" />
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AutoDAN Reasoning</h1>
          <p className="text-muted-foreground">
            Advanced adversarial prompt optimization with test-time scaling and adaptive selection.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="px-3 py-1">
            <Shield className="mr-1 h-3 w-3" />
            Security Research Mode
          </Badge>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Benchmark Dataset Selector */}
        <div className="lg:col-span-1">
          <BenchmarkDatasetSelector
            onPromptSelect={handleBenchmarkPromptSelect}
            onBatchGenerate={handleBenchmarkBatchGenerate}
            className="h-full"
          />
        </div>

        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              Attack Configuration
            </CardTitle>
            <CardDescription>
              Configure the target request and optimization method.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Provider/Model Selection */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="provider">Provider</Label>
                <Select value={effectiveProvider || ""} onValueChange={handleProviderChange}>
                  <SelectTrigger id="provider">
                    <SelectValue placeholder="Select provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="deepseek">DeepSeek</SelectItem>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="gemini">Gemini</SelectItem>
                    <SelectItem value="anthropic">Anthropic</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="model">Model</Label>
                <Input
                  id="model"
                  value={effectiveModel || ""}
                  onChange={(e) => setLocalModel(e.target.value)}
                  placeholder="e.g., deepseek-chat"
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="request">Harmful Request / Target Behavior</Label>
                {selectedBenchmarkPrompt && (
                  <Badge variant="outline" className="text-xs bg-cyan-500/10 text-cyan-400 border-cyan-500/20">
                    <Database className="h-3 w-3 mr-1" />
                    From Benchmark
                  </Badge>
                )}
              </div>
              <Textarea
                id="request"
                placeholder="e.g., How to build a dangerous device? (or select from Benchmark Dataset)"
                value={request}
                onChange={(e) => {
                  setRequest(e.target.value);
                  // Clear benchmark selection if user manually edits
                  if (selectedBenchmarkPrompt && e.target.value !== selectedBenchmarkPrompt.prompt) {
                    setSelectedBenchmarkPrompt(null);
                  }
                }}
                spellCheck={false}
                className="min-h-[100px]"
              />
              {selectedBenchmarkPrompt && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {selectedBenchmarkPrompt.risk_area && (
                    <Badge variant="outline" className="text-xs bg-red-500/10 text-red-400 border-red-500/20">
                      {selectedBenchmarkPrompt.risk_area.replace(/_/g, " ")}
                    </Badge>
                  )}
                  {selectedBenchmarkPrompt.harm_type && (
                    <Badge variant="outline" className="text-xs bg-orange-500/10 text-orange-400 border-orange-500/20">
                      {selectedBenchmarkPrompt.harm_type.replace(/_/g, " ")}
                    </Badge>
                  )}
                  {selectedBenchmarkPrompt.severity && (
                    <Badge variant="outline" className="text-xs bg-yellow-500/10 text-yellow-400 border-yellow-500/20">
                      {selectedBenchmarkPrompt.severity}
                    </Badge>
                  )}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="method">Optimization Method</Label>
              <Select value={method} onValueChange={setMethod}>
                <SelectTrigger id="method">
                  <SelectValue placeholder="Select method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="vanilla">
                    <div className="flex items-center gap-2">
                      <Play className="h-4 w-4" />
                      <span>Vanilla (Single-shot)</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="best_of_n">
                    <div className="flex items-center gap-2">
                      <Search className="h-4 w-4" />
                      <span>Best-of-N Scaling</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="beam_search">
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4" />
                      <span>Beam Search Scaling</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="chain_of_thought">
                    <div className="flex items-center gap-2">
                      <Brain className="h-4 w-4" />
                      <span>Chain-of-Thought</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="adaptive">
                    <div className="flex items-center gap-2">
                      <RefreshCw className="h-4 w-4" />
                      <span>Adaptive Selection</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {method === "best_of_n" && (
              <div className="space-y-2 pt-2">
                <Label htmlFor="best_of_n">N Candidates ({bestOfN})</Label>
                <Input
                  id="best_of_n"
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={bestOfN}
                  onChange={(e) => setBestOfN(parseInt(e.target.value))}
                />
              </div>
            )}

            {method === "beam_search" && (
              <div className="grid grid-cols-2 gap-4 pt-2">
                <div className="space-y-2">
                  <Label htmlFor="beam_width">Beam Width ({beamWidth})</Label>
                  <Input
                    id="beam_width"
                    type="number"
                    min="1"
                    max="10"
                    value={beamWidth}
                    onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="beam_depth">Beam Depth ({beamDepth})</Label>
                  <Input
                    id="beam_depth"
                    type="number"
                    min="1"
                    max="5"
                    value={beamDepth}
                    onChange={(e) => setBeamDepth(parseInt(e.target.value))}
                  />
                </div>
              </div>
            )}

            {(method === "chain_of_thought" || method === "adaptive") && (
              <div className="space-y-2 pt-2">
                <Label htmlFor="refinement">Refinement Iterations ({refinementIterations})</Label>
                <Input
                  id="refinement"
                  type="number"
                  min="1"
                  max="10"
                  value={refinementIterations}
                  onChange={(e) => setRefinementIterations(parseInt(e.target.value))}
                />
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button
              className="w-full"
              onClick={handleGenerate}
              disabled={disableActions}
            >
              {disableActions ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {sessionReady ? "Optimizing..." : "Syncing model session..."}
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Attack
                </>
              )}
            </Button>
          </CardFooter>
        </Card>

        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5 text-blue-500" />
              Results & Analysis
            </CardTitle>
            <CardDescription>
              View the generated adversarial prompt and effectiveness score.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Error Display - Shows detailed error information inline */}
            {errorInfo && !isLoading && (
              <div className={`rounded-md border p-4 space-y-3 ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                  ? "border-amber-500/50 bg-amber-500/10"
                  : "border-destructive/50 bg-destructive/10"
                }`}>
                <div className="flex items-start gap-3">
                  <Shield className={`h-5 w-5 mt-0.5 ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                      ? "text-amber-500"
                      : "text-destructive"
                    }`} />
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className={`font-semibold ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-destructive"
                        }`}>
                        {errorInfo.code === "VALIDATION_ERROR"
                          ? "Validation Error"
                          : errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                            ? "API Quota Exceeded"
                            : "Error"}
                      </h4>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
                        onClick={clearError}
                      >
                        ×
                      </Button>
                    </div>
                    <p className={`text-sm ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                        ? "text-amber-600/90 dark:text-amber-400/90"
                        : "text-destructive/90"
                      }`}>
                      {errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                        ? "Your API quota has been exceeded. Please wait for the quota to reset or switch to a different provider."
                        : errorInfo.message}
                    </p>

                    {/* Field-specific errors */}
                    {errorInfo.fieldErrors && errorInfo.fieldErrors.length > 0 && (
                      <div className="space-y-1 pt-2">
                        <p className={`text-xs font-medium ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                            ? "text-amber-600/80 dark:text-amber-400/80"
                            : "text-destructive/80"
                          }`}>Field Errors:</p>
                        <ul className={`text-xs space-y-1 list-disc list-inside ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                            ? "text-amber-600/70 dark:text-amber-400/70"
                            : "text-destructive/70"
                          }`}>
                          {errorInfo.fieldErrors.map((fieldError, index) => (
                            <li key={index}>{fieldError}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Actionable guidance */}
                    {errorInfo.guidance && (
                      <div className={`pt-2 border-t ${errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED"
                          ? "border-amber-500/20"
                          : "border-destructive/20"
                        }`}>
                        <p className="text-xs text-muted-foreground">
                          <span className="font-medium">Suggestion:</span> {errorInfo.guidance}
                        </p>
                      </div>
                    )}

                    {/* Quick actions for rate limit errors */}
                    {(errorInfo.code === "RATE_LIMIT_EXCEEDED" || errorInfo.code === "LLM_QUOTA_EXCEEDED") && (
                      <div className="pt-2 flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs border-amber-500/50 text-amber-600 dark:text-amber-400 hover:bg-amber-500/10"
                          onClick={() => {
                            // Navigate to settings or show provider selector
                            toast.info("Switch Provider", {
                              description: "Go to Settings to change your LLM provider.",
                            });
                          }}
                        >
                          Switch Provider
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-xs"
                          onClick={clearError}
                        >
                          Dismiss
                        </Button>
                      </div>
                    )}

                    {/* Debug info in development */}
                    {process.env.NODE_ENV === "development" && (
                      <div className="pt-2 text-xs text-muted-foreground/60 font-mono">
                        <span>Code: {errorInfo.code}</span>
                        {errorInfo.statusCode && <span> | Status: {errorInfo.statusCode}</span>}
                        {errorInfo.requestId && <span> | ReqID: {errorInfo.requestId}</span>}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {!currentResult && !isLoading && !errorInfo && (
              <div className="flex h-[300px] flex-col items-center justify-center text-center text-muted-foreground">
                <Brain className="mb-4 h-12 w-12 opacity-20" />
                <p>No results yet. Run an attack to see the output.</p>
              </div>
            )}

            {isLoading && (
              <div className="space-y-4 py-8">
                <div className="flex items-center justify-between text-sm">
                  <span>Processing optimization steps...</span>
                  <span>{method === "adaptive" ? "Adapting..." : "Iterating..."}</span>
                </div>
                <Progress value={40} className="h-2" />
                <div className="rounded-md bg-muted p-4 text-xs font-mono">
                  <p className="animate-pulse text-blue-500">{">"} Analyzing target defense pattern...</p>
                  <p className="mt-2 opacity-50">{">"} Selecting optimal bypass strategy...</p>
                </div>
              </div>
            )}

            {currentResult && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Score:</span>
                    <Badge variant={currentResult.score && currentResult.score >= 8.5 ? "destructive" : "secondary"}>
                      {currentResult.score?.toFixed(1) || "N/A"} / 10.0
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Latency: {currentResult.latency_ms.toFixed(0)}ms
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Generated Jailbreak Prompt</Label>
                  <div className="relative">
                    <Textarea
                      readOnly
                      value={currentResult.jailbreak_prompt ?? ""}
                      spellCheck={false}
                      className="min-h-[150px] font-mono text-xs text-foreground"
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute right-2 top-2 h-8 w-8 p-0"
                      onClick={() => {
                        navigator.clipboard.writeText(currentResult.jailbreak_prompt);
                        toast.success("Copied to clipboard");
                      }}
                    >
                      <History className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                <div className="rounded-md border p-3 bg-blue-50/50 dark:bg-blue-950/20">
                  <div className="flex items-center gap-2 mb-2">
                    <Brain className="h-4 w-4 text-blue-500" />
                    <span className="text-xs font-semibold uppercase tracking-wider">Reasoning Summary</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    The {currentResult.method} method successfully identified a path to bypass the target model&apos;s
                    safety filters. {currentResult.method === "adaptive" ?
                      "It dynamically adapted to the model&apos;s refusal patterns and selected a specialized bypass." :
                      "The resulting prompt leverages advanced framing to encourage compliance."}
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="metrics" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="batch">Batch Queue</TabsTrigger>
          <TabsTrigger value="config">System Config</TabsTrigger>
          <TabsTrigger value="log">Reasoning Log</TabsTrigger>
        </TabsList>
        <TabsContent value="metrics" className="pt-4">
          <Card>
            <CardContent className="pt-6">
              <div className="grid gap-4 md:grid-cols-4">
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold">84.2%</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Avg. Score</p>
                  <p className="text-2xl font-bold">7.8</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Total Attacks</p>
                  <p className="text-2xl font-bold">1,248</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">Cache Hit Rate</p>
                  <p className="text-2xl font-bold">32.5%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="batch" className="pt-4">
          <Card>
            <CardContent className="pt-6">
              {benchmarkBatch.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                  <Database className="h-12 w-12 mb-3 opacity-20" />
                  <p className="text-sm">No batch prompts selected</p>
                  <p className="text-xs">Select prompts from the Benchmark Dataset panel</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-cyan-500/10 text-cyan-400">
                        {benchmarkBatch.length} prompts queued
                      </Badge>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setBenchmarkBatch([])}
                      >
                        Clear Queue
                      </Button>
                      <Button
                        size="sm"
                        disabled={disableActions}
                        onClick={async () => {
                          toast.info("Batch Processing", {
                            description: `Starting batch attack on ${benchmarkBatch.length} prompts...`,
                          });
                          // Process batch sequentially
                          for (let i = 0; i < benchmarkBatch.length; i++) {
                            const prompt = benchmarkBatch[i];
                            setRequest(prompt.prompt);
                            setSelectedBenchmarkPrompt(prompt);
                            try {
                              await generate({
                                request: prompt.prompt,
                                method,
                                target_model: effectiveModel ?? undefined,
                                provider: effectiveProvider ?? undefined,
                                best_of_n: method === "best_of_n" ? bestOfN : undefined,
                                beam_width: method === "beam_search" ? beamWidth : undefined,
                                beam_depth: method === "beam_search" ? beamDepth : undefined,
                                refinement_iterations: (method === "chain_of_thought" || method === "adaptive") ? refinementIterations : undefined,
                              });
                            } catch (err) {
                              console.error(`Batch item ${i + 1} failed:`, err);
                            }
                          }
                          toast.success("Batch Complete", {
                            description: `Processed ${benchmarkBatch.length} prompts.`,
                          });
                        }}
                      >
                        <Play className="h-4 w-4 mr-1" />
                        Run Batch
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-2 max-h-[300px] overflow-y-auto">
                    {benchmarkBatch.map((prompt, index) => (
                      <div
                        key={prompt.id || index}
                        className="p-3 rounded-md border bg-muted/30 text-sm"
                      >
                        <div className="flex items-start justify-between gap-2 mb-1">
                          <span className="text-xs text-muted-foreground">#{index + 1}</span>
                          <div className="flex gap-1">
                            {prompt.risk_area && (
                              <Badge variant="outline" className="text-xs">
                                {prompt.risk_area.replace(/_/g, " ")}
                              </Badge>
                            )}
                            {prompt.severity && (
                              <Badge
                                variant="outline"
                                className={`text-xs ${
                                  prompt.severity === "high"
                                    ? "bg-red-500/10 text-red-400"
                                    : prompt.severity === "medium"
                                    ? "bg-yellow-500/10 text-yellow-400"
                                    : "bg-green-500/10 text-green-400"
                                }`}
                              >
                                {prompt.severity}
                              </Badge>
                            )}
                          </div>
                        </div>
                        <p className="line-clamp-2 text-xs">{prompt.prompt}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="config" className="pt-4">
          <Card>
            <CardContent className="pt-6">
              <pre className="text-xs bg-muted p-4 rounded-md overflow-auto max-h-[300px]">
                {JSON.stringify(config, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="log" className="pt-4">
          <Card>
            <CardContent className="pt-6 space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs text-blue-500 font-mono">
                  <span>[INFO]</span>
                  <span>2025-12-18 08:25:01</span>
                  <span>Starting adaptive optimization loop...</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-green-500 font-mono">
                  <span>[ANALYSIS]</span>
                  <span>Detected defense: &quot;Direct Refusal&quot; (Confidence: 0.82)</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-green-500 font-mono">
                  <span>[PLANNING]</span>
                  <span>Switching strategy to &quot;Hypothetical Scenario&quot;</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-yellow-500 font-mono">
                  <span>[EXECUTION]</span>
                  <span>Iteration 2: Score improved to 7.4</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-green-500 font-mono">
                  <span>[SUCCESS]</span>
                  <span>Iteration 3: Score 8.9 reached with &quot;Technical Framing&quot;.</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
