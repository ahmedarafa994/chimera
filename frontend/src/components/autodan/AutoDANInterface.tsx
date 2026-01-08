"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { enhancedApi, AutoDANRequest, AutoDANResponse } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { handleApiError } from "@/lib/errors";
import { Loader2, Brain, Copy, Check, Sparkles, Zap, Target, Rocket, Library } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HierarchicalSearchPanel } from "./HierarchicalSearchPanel";
import { TurboPanel } from "./TurboPanel";
import { StrategySelector } from "./StrategySelector";
import type { SingleAttackRequest, SingleAttackResponse, LibraryStatsResponse } from "@/lib/types/autodan-turbo-types";

type AutoDANMethod = "vanilla" | "best_of_n" | "beam_search";

interface MethodInfo {
  name: string;
  description: string;
  icon: React.ReactNode;
  complexity: "Low" | "Medium" | "High";
}

const methodInfoMap: Record<AutoDANMethod, MethodInfo> = {
  vanilla: {
    name: "Vanilla",
    description: "Basic AutoDAN approach with single-pass generation. Fast but less sophisticated.",
    icon: <Zap className="h-4 w-4" />,
    complexity: "Low",
  },
  best_of_n: {
    name: "Best of N",
    description: "Generates multiple candidates and selects the best one. Good balance of quality and speed.",
    icon: <Target className="h-4 w-4" />,
    complexity: "Medium",
  },
  beam_search: {
    name: "Beam Search",
    description: "Advanced search strategy exploring multiple paths. Highest quality but slower.",
    icon: <Sparkles className="h-4 w-4" />,
    complexity: "High",
  },
};

// Extended response type to handle both vanilla and turbo results
interface ExtendedAutoDANResponse extends Omit<AutoDANResponse, 'strategy_id'> {
  score?: number;
  is_jailbreak?: boolean;
  strategy_id?: string | null;
  iterations_used?: number;
}

export function AutoDANInterface() {
  const [request, setRequest] = useState("");
  const [method, setMethod] = useState<AutoDANMethod>("best_of_n");
  const [targetModel, setTargetModel] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ExtendedAutoDANResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<Array<{ request: string; response: ExtendedAutoDANResponse; timestamp: Date }>>([]);

  // Turbo mode state
  const [turboMode, setTurboMode] = useState(false);
  const [libraryStats, setLibraryStats] = useState<LibraryStatsResponse | null>(null);

  // Manual strategy selection state
  const [useManualStrategies, setUseManualStrategies] = useState(false);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);

  // Load library stats when turbo mode or manual strategies is enabled
  useEffect(() => {
    if (turboMode || useManualStrategies) {
      loadLibraryStats();
    }
  }, [turboMode, useManualStrategies]);

  const loadLibraryStats = async () => {
    try {
      const response = await enhancedApi.autodanTurbo.library.stats();
      setLibraryStats(response.data);
    } catch (error) {
      console.error("Failed to load library stats:", error);
      handleApiError(error);
    }
  };

  const handleGenerate = async () => {
    if (!request.trim()) {
      toast.error("Please enter a request");
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      if (turboMode || (useManualStrategies && selectedStrategies.length > 0)) {
        // Use Turbo API with strategy retrieval or manual strategies
        const turboPayload: SingleAttackRequest = {
          request: request.trim(),
          model: targetModel || undefined,
          max_iterations: turboMode ? 10 : 5, // Fewer iterations for manual strategy mode
          break_score: 8.5,
          // Pass selected strategies if using manual selection
          ...(useManualStrategies && selectedStrategies.length > 0 && {
            strategy_ids: selectedStrategies,
          }),
        };

        const turboResponse = await enhancedApi.autodanTurbo.attack(turboPayload);
        const turboData = turboResponse.data;

        // Convert to extended response format
        const methodLabel = turboMode
          ? `turbo (${method})`
          : `${method} + strategies`;
        const extendedResult: ExtendedAutoDANResponse = {
          success: turboData.success ?? false,
          jailbreak_prompt: turboData.jailbreak_prompt,
          method: methodLabel,
          status: turboData.success ? "success" : "partial",
          score: turboData.final_score,
          is_jailbreak: turboData.success,
          strategy_id: turboData.strategies_used?.[0],
          iterations_used: turboData.iterations_used,
        };

        setResult(extendedResult);

        // Add to history
        setHistory(prev => [
          { request: request.trim(), response: extendedResult, timestamp: new Date() },
          ...prev.slice(0, 9),
        ]);

        if (turboData.success) {
          toast.success("Turbo jailbreak generated!", {
            description: `Score: ${turboData.final_score.toFixed(1)} in ${turboData.iterations_used} iterations`,
          });
        } else {
          toast.warning("Attack completed", {
            description: `Best score: ${turboData.final_score.toFixed(1)}`,
          });
        }

        // Refresh library stats
        loadLibraryStats();
      } else {
        // Use vanilla AutoDAN API
        const payload: AutoDANRequest = {
          request: request.trim(),
          method,
          target_model: targetModel || undefined,
        };

        const response = await enhancedApi.autodan(payload);
        const extendedResult: ExtendedAutoDANResponse = {
          success: response.success ?? false,
          jailbreak_prompt: response.jailbreak_prompt ?? "",
          method: response.method ?? method,
          status: response.success ? "success" : "partial",
          score: response.score,
          is_jailbreak: response.success,
        };
        setResult(extendedResult);

        // Add to history
        setHistory(prev => [
          { request: request.trim(), response: extendedResult, timestamp: new Date() },
          ...prev.slice(0, 9),
        ]);

        toast.success("Jailbreak prompt generated", {
          description: `Method: ${extendedResult.method}`,
        });
      }
    } catch (error: unknown) {
      console.error("AutoDAN generation failed:", error);
      handleApiError(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (result?.jailbreak_prompt) {
      await navigator.clipboard.writeText(result.jailbreak_prompt);
      setCopied(true);
      toast.success("Copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const selectedMethodInfo = methodInfoMap[method];

  return (
    <div className="container mx-auto space-y-6">
      <Tabs defaultValue="generate" className="w-full">
        <TabsList className="grid w-full max-w-[800px] grid-cols-4">
          <TabsTrigger value="generate">
            <Brain className="h-4 w-4 mr-1.5" />
            Generate
          </TabsTrigger>
          <TabsTrigger value="turbo" className="text-yellow-600 dark:text-yellow-400">
            <Rocket className="h-4 w-4 mr-1.5" />
            Turbo
          </TabsTrigger>
          <TabsTrigger value="hierarchical">Hierarchical</TabsTrigger>
          <TabsTrigger value="history">History ({history.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="generate" className="mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Configuration Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  AutoDAN Configuration
                </CardTitle>
                <CardDescription>
                  Configure the AutoDAN-Reasoning jailbreak generation parameters.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="request">Request / Goal</Label>
                  <Textarea
                    id="request"
                    value={request}
                    onChange={(e) => setRequest(e.target.value)}
                    placeholder="Enter the goal or request you want to achieve through jailbreaking..."
                    rows={5}
                    className="resize-none"
                  />
                  <p className="text-xs text-muted-foreground">
                    Describe what you want the target model to do. AutoDAN will generate an optimized jailbreak prompt.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="method">Generation Method</Label>
                  <Select value={method} onValueChange={(v) => setMethod(v as AutoDANMethod)}>
                    <SelectTrigger id="method">
                      <SelectValue placeholder="Select method" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="vanilla">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4" />
                          Vanilla (Fast)
                        </div>
                      </SelectItem>
                      <SelectItem value="best_of_n">
                        <div className="flex items-center gap-2">
                          <Target className="h-4 w-4" />
                          Best of N (Balanced)
                        </div>
                      </SelectItem>
                      <SelectItem value="beam_search">
                        <div className="flex items-center gap-2">
                          <Sparkles className="h-4 w-4" />
                          Beam Search (Quality)
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Method Info Card */}
                <div className="rounded-lg border bg-muted/50 p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {selectedMethodInfo.icon}
                      <span className="font-medium">{selectedMethodInfo.name}</span>
                    </div>
                    <Badge variant={
                      selectedMethodInfo.complexity === "Low" ? "secondary" :
                        selectedMethodInfo.complexity === "Medium" ? "default" : "destructive"
                    }>
                      {selectedMethodInfo.complexity} Complexity
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {selectedMethodInfo.description}
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="targetModel">Target Model (Optional)</Label>
                  <Input
                    id="targetModel"
                    value={targetModel}
                    onChange={(e) => setTargetModel(e.target.value)}
                    placeholder="e.g., gpt-4, claude-3, gemini-3-pro-preview, deepseek-reasoner"
                  />
                  <p className="text-xs text-muted-foreground">
                    Specify the target model for optimized jailbreak generation.
                  </p>
                </div>

                {/* Strategy Enhancement Options */}
                <div className="rounded-lg border p-4 space-y-3">
                  {/* Turbo Mode Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="turbo-mode" className="flex items-center gap-2">
                        <Rocket className="h-4 w-4 text-yellow-500" />
                        Turbo Mode
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Auto-retrieve strategies with lifelong learning
                      </p>
                    </div>
                    <Switch
                      id="turbo-mode"
                      checked={turboMode}
                      onCheckedChange={(checked) => {
                        setTurboMode(checked);
                        if (checked) setUseManualStrategies(false);
                      }}
                    />
                  </div>

                  {/* Manual Strategy Selection Toggle */}
                  {!turboMode && (
                    <div className="flex items-center justify-between pt-2 border-t">
                      <div className="space-y-0.5">
                        <Label htmlFor="manual-strategies" className="flex items-center gap-2">
                          <Library className="h-4 w-4 text-blue-500" />
                          Manual Strategy Selection
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          Choose specific strategies to apply
                        </p>
                      </div>
                      <Switch
                        id="manual-strategies"
                        checked={useManualStrategies}
                        onCheckedChange={setUseManualStrategies}
                      />
                    </div>
                  )}

                  {/* Stats Display */}
                  {(turboMode || useManualStrategies) && libraryStats && (
                    <div className="flex items-center gap-2 pt-2 border-t">
                      <Library className="h-3 w-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        {libraryStats.total_strategies} strategies available
                      </span>
                      <Badge variant="outline" className="text-xs">
                        Avg: {libraryStats.average_score.toFixed(1)}
                      </Badge>
                    </div>
                  )}
                </div>

                {/* Strategy Selector (when manual selection is enabled) */}
                {useManualStrategies && !turboMode && (
                  <StrategySelector
                    selectedStrategies={selectedStrategies}
                    onSelectionChange={setSelectedStrategies}
                    maxSelections={3}
                    showStats={false}
                    compact={true}
                  />
                )}
              </CardContent>
              <CardFooter>
                <Button
                  onClick={handleGenerate}
                  disabled={isLoading || !request.trim()}
                  className={`w-full ${turboMode
                      ? "bg-yellow-600 hover:bg-yellow-700"
                      : useManualStrategies && selectedStrategies.length > 0
                        ? "bg-blue-600 hover:bg-blue-700"
                        : ""
                    }`}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {turboMode ? "Turbo Generating..." : useManualStrategies ? "Strategy-Enhanced..." : "Generating..."}
                    </>
                  ) : turboMode ? (
                    <>
                      <Rocket className="mr-2 h-4 w-4" />
                      Generate with Turbo
                    </>
                  ) : useManualStrategies && selectedStrategies.length > 0 ? (
                    <>
                      <Library className="mr-2 h-4 w-4" />
                      Generate with {selectedStrategies.length} Strategies
                    </>
                  ) : (
                    <>
                      <Brain className="mr-2 h-4 w-4" />
                      Generate Jailbreak
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>

            {/* Result Panel */}
            <Card className="flex flex-col">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Generated Jailbreak</CardTitle>
                    <CardDescription>
                      The optimized jailbreak prompt will appear here.
                    </CardDescription>
                  </div>
                  {result && (
                    <div className="flex items-center gap-2">
                      {result.score !== undefined && (
                        <Badge variant="outline">
                          Score: {result.score.toFixed(1)}
                        </Badge>
                      )}
                      <Badge
                        variant="outline"
                        className={result.is_jailbreak ? "text-green-500 border-green-500" : result.status === "success" ? "text-green-500 border-green-500" : ""}
                      >
                        {result.status}
                      </Badge>
                    </div>
                  )}
                </div>
              </CardHeader>
              <CardContent className="flex-1">
                {result ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground flex-wrap">
                      <span>Method:</span>
                      <Badge variant="secondary">{result.method}</Badge>
                      {result.iterations_used !== undefined && (
                        <>
                          <span>•</span>
                          <span>{result.iterations_used} iterations</span>
                        </>
                      )}
                      {result.strategy_id && (
                        <>
                          <span>•</span>
                          <Badge variant="outline" className="text-xs">
                            Strategy: {result.strategy_id}
                          </Badge>
                        </>
                      )}
                    </div>
                    <ScrollArea className="h-[300px] w-full rounded-md border p-4">
                      <pre className="whitespace-pre-wrap text-sm font-mono">
                        {result.jailbreak_prompt}
                      </pre>
                    </ScrollArea>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-[300px] text-muted-foreground">
                    <Brain className="h-12 w-12 mb-4 opacity-20" />
                    <p>Configure and generate to see results</p>
                  </div>
                )}
              </CardContent>
              {result && (
                <CardFooter>
                  <Button
                    variant="outline"
                    onClick={handleCopy}
                    className="w-full"
                  >
                    {copied ? (
                      <>
                        <Check className="mr-2 h-4 w-4" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="mr-2 h-4 w-4" />
                        Copy to Clipboard
                      </>
                    )}
                  </Button>
                </CardFooter>
              )}
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="turbo" className="mt-6">
          <TurboPanel
            onSuccess={(prompt, score) => {
              // Add successful Turbo generations to history
              setHistory(prev => [
                {
                  request: `[Turbo] Score: ${score.toFixed(1)}`,
                  response: {
                    success: true,
                    jailbreak_prompt: prompt,
                    method: "turbo",
                    status: "success"
                  },
                  timestamp: new Date()
                },
                ...prev.slice(0, 9),
              ]);
            }}
          />
        </TabsContent>

        <TabsContent value="hierarchical" className="mt-6">
          <HierarchicalSearchPanel />
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Generation History</CardTitle>
              <CardDescription>
                Recent AutoDAN generations from this session.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Brain className="h-12 w-12 mb-4 opacity-20" />
                  <p>No generations yet. Start by creating a jailbreak prompt.</p>
                </div>
              ) : (
                <ScrollArea className="h-[500px]">
                  <div className="space-y-4">
                    {history.map((item, idx) => (
                      <div key={idx} className="rounded-lg border p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{item.response.method}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {item.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <div>
                          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                            Request:
                          </span>
                          <p className="text-sm mt-1">{item.request}</p>
                        </div>
                        <div>
                          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                            Generated Prompt:
                          </span>
                          <pre className="text-xs mt-1 whitespace-pre-wrap bg-muted p-2 rounded font-mono max-h-[150px] overflow-auto">
                            {item.response.jailbreak_prompt}
                          </pre>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={async () => {
                            await navigator.clipboard.writeText(item.response.jailbreak_prompt ?? "");
                            toast.success("Copied to clipboard");
                          }}
                        >
                          <Copy className="h-3 w-3 mr-1" />
                          Copy
                        </Button>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
