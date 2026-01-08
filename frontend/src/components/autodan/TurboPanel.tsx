"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { enhancedApi } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { Loader2, Zap, Copy, Check, Library, RefreshCw, Sparkles, Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type {
  SingleAttackRequest,
  SingleAttackResponse,
  JailbreakStrategy,
  LibraryStatsResponse,
  StrategyListResponse
} from "@/lib/types/autodan-turbo-types";

interface TurboPanelProps {
  /** Optional callback when a successful jailbreak is generated */
  onSuccess?: (prompt: string, score: number) => void;
}

export function TurboPanel({ onSuccess }: TurboPanelProps) {
  // Form state
  const [request, setRequest] = useState("");
  const [targetModel, setTargetModel] = useState("");
  const [attackerModel, setAttackerModel] = useState("gemma-7b-it");
  const [scorerModel, setScorerModel] = useState("gemma-7b-it");
  const [maxIterations, setMaxIterations] = useState(10);
  const [terminationScore, setTerminationScore] = useState(8.5);
  
  // Strategy options
  const [useStrategyRetrieval, setUseStrategyRetrieval] = useState(true);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [availableStrategies, setAvailableStrategies] = useState<JailbreakStrategy[]>([]);
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SingleAttackResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [libraryStats, setLibraryStats] = useState<LibraryStatsResponse | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Load library stats and strategies on mount
  useEffect(() => {
    loadLibraryData();
  }, []);

  const loadLibraryData = async () => {
    try {
      const [statsRes, strategiesRes] = await Promise.all([
        enhancedApi.autodanTurbo.library.stats(),
        enhancedApi.autodanTurbo.strategies.list()
      ]);
      setLibraryStats(statsRes.data);
      setAvailableStrategies(strategiesRes.data.strategies);
    } catch (error) {
      console.error("Failed to load library data:", error);
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
      const payload: SingleAttackRequest = {
        request: request.trim(),
        model: targetModel || undefined,
        max_iterations: maxIterations,
        break_score: terminationScore,
      };

      const response = await enhancedApi.autodanTurbo.attack(payload);
      const data = response.data as any;
      setResult(data);

      if (data.is_jailbreak) {
        toast.success("Turbo jailbreak generated!", {
          description: `Score: ${data.score.toFixed(1)} in ${data.iterations_used} iterations`,
        });

        if (onSuccess) {
          onSuccess(data.prompt, data.score);
        }
      } else {
        toast.warning("Attack completed but may not have reached target score", {
          description: `Best score: ${data.score.toFixed(1)}`,
        });
      }

      // Refresh library stats after attack
      loadLibraryData();
    } catch (error: unknown) {
      console.error("Turbo generation failed:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      toast.error("Generation failed", {
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (result?.prompt) {
      await navigator.clipboard.writeText(result.prompt);
      setCopied(true);
      toast.success("Copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const toggleStrategy = (strategyId: string) => {
    setSelectedStrategies(prev => 
      prev.includes(strategyId) 
        ? prev.filter(id => id !== strategyId)
        : [...prev, strategyId]
    );
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Configuration Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            AutoDAN-Turbo Configuration
          </CardTitle>
          <CardDescription>
            Lifelong learning agent with automatic strategy discovery (ICLR 2025)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Library Stats Banner */}
          {libraryStats && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border">
              <div className="flex items-center gap-2">
                <Library className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Strategy Library</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary">{libraryStats.total_strategies} strategies</Badge>
                <Badge variant="outline">Avg score: {libraryStats.average_score.toFixed(1)}</Badge>
                <Button variant="ghost" size="icon" onClick={loadLibraryData}>
                  <RefreshCw className="h-3 w-3" />
                </Button>
              </div>
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="turbo-request">Malicious Request / Goal</Label>
            <Textarea
              id="turbo-request"
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              placeholder="Enter the goal you want to achieve through jailbreaking..."
              rows={4}
              className="resize-none"
            />
            <p className="text-xs text-muted-foreground">
              AutoDAN-Turbo will automatically discover and apply effective jailbreak strategies.
            </p>
          </div>

          {/* Strategy Retrieval Toggle */}
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div className="space-y-0.5">
              <Label htmlFor="strategy-retrieval">Auto Strategy Retrieval</Label>
              <p className="text-xs text-muted-foreground">
                Automatically retrieve relevant strategies from the library
              </p>
            </div>
            <Switch
              id="strategy-retrieval"
              checked={useStrategyRetrieval}
              onCheckedChange={setUseStrategyRetrieval}
            />
          </div>

          {/* Manual Strategy Selection (when auto-retrieval is off) */}
          {!useStrategyRetrieval && availableStrategies.length > 0 && (
            <div className="space-y-2">
              <Label>Select Strategies</Label>
              <ScrollArea className="h-[120px] rounded-md border p-2">
                <div className="space-y-2">
                  {availableStrategies.slice(0, 10).map((strategy) => (
                    <div
                      key={strategy.id}
                      className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                        selectedStrategies.includes(strategy.id)
                          ? "bg-primary/10 border border-primary"
                          : "hover:bg-muted"
                      }`}
                      onClick={() => toggleStrategy(strategy.id)}
                    >
                      <span className="text-sm font-medium">{strategy.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {(strategy.success_rate * 100).toFixed(0)}% success
                      </Badge>
                    </div>
                  ))}
                </div>
              </ScrollArea>
              {selectedStrategies.length > 0 && (
                <p className="text-xs text-muted-foreground">
                  {selectedStrategies.length} strategies selected
                </p>
              )}
            </div>
          )}

          {/* Advanced Options */}
          <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between">
                Advanced Options
                <Sparkles className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-4">
              <div className="space-y-2">
                <Label htmlFor="turbo-target">Target Model (Optional)</Label>
                <Input
                  id="turbo-target"
                  value={targetModel}
                  onChange={(e) => setTargetModel(e.target.value)}
                  placeholder="e.g., gpt-4, claude-3, llama-2-70b"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="attacker-model">Attacker Model</Label>
                  <Select value={attackerModel} onValueChange={setAttackerModel}>
                    <SelectTrigger id="attacker-model">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gemma-7b-it">Gemma 7B-it</SelectItem>
                      <SelectItem value="llama-3-70b">Llama 3 70B</SelectItem>
                      <SelectItem value="gpt-4">GPT-4</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="scorer-model">Scorer Model</Label>
                  <Select value={scorerModel} onValueChange={setScorerModel}>
                    <SelectTrigger id="scorer-model">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gemma-7b-it">Gemma 7B-it</SelectItem>
                      <SelectItem value="llama-2-13b">Llama 2 13B</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Max Iterations: {maxIterations}</Label>
                <Slider
                  value={[maxIterations]}
                  onValueChange={(v) => setMaxIterations(v[0])}
                  min={1}
                  max={50}
                  step={1}
                />
              </div>

              <div className="space-y-2">
                <Label>Termination Score: {terminationScore.toFixed(1)}</Label>
                <Slider
                  value={[terminationScore * 10]}
                  onValueChange={(v) => setTerminationScore(v[0] / 10)}
                  min={50}
                  max={100}
                  step={5}
                />
                <p className="text-xs text-muted-foreground">
                  Attack stops when this score is reached (1-10 scale)
                </p>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </CardContent>
        <CardFooter>
          <Button
            onClick={handleGenerate}
            disabled={isLoading || !request.trim()}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating with Turbo...
              </>
            ) : (
              <>
                <Zap className="mr-2 h-4 w-4" />
                Generate Turbo Jailbreak
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
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Turbo Result
              </CardTitle>
              <CardDescription>
                Strategy-enhanced jailbreak with scoring
              </CardDescription>
            </div>
            {result && (
              <div className="flex items-center gap-2">
                <Badge
                  variant={result.is_jailbreak ? "default" : "secondary"}
                  className={result.is_jailbreak ? "bg-green-500" : ""}
                >
                  {result.is_jailbreak ? "Success" : "Partial"}
                </Badge>
                <Badge variant="outline">
                  Score: {result.score.toFixed(1)}
                </Badge>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="flex-1">
          {result ? (
            <div className="space-y-4">
              {/* Metrics Row */}
              <div className="grid grid-cols-3 gap-2">
                <div className="text-center p-2 rounded-lg bg-muted/50">
                  <div className="text-2xl font-bold">{result.iterations_used}</div>
                  <div className="text-xs text-muted-foreground">Iterations</div>
                </div>
                <div className="text-center p-2 rounded-lg bg-muted/50">
                  <div className="text-2xl font-bold">{result.score.toFixed(1)}</div>
                  <div className="text-xs text-muted-foreground">Final Score</div>
                </div>
                <div className="text-center p-2 rounded-lg bg-muted/50">
                  <div className="text-2xl font-bold">{result.strategy_id ? 1 : 0}</div>
                  <div className="text-xs text-muted-foreground">Strategies</div>
                </div>
              </div>

              {/* Strategy Used */}
              {result.strategy_id && (
                <div className="space-y-2">
                  <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                    Strategy Applied
                  </Label>
                  <div className="flex flex-wrap gap-1">
                    <Badge variant="secondary" className="text-xs">
                      {result.strategy_id}
                    </Badge>
                  </div>
                </div>
              )}

              {/* Strategy Extracted */}
              {result.strategy_extracted && (
                <div className="space-y-2">
                  <Label className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                    <Sparkles className="h-3 w-3" />
                    New Strategy Discovered
                  </Label>
                  <Badge variant="outline" className="text-xs bg-yellow-500/10">
                    {result.strategy_extracted}
                  </Badge>
                </div>
              )}

              {/* Generated Prompt */}
              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                  Generated Jailbreak Prompt
                </Label>
                <ScrollArea className="h-[200px] w-full rounded-md border p-4">
                  <pre className="whitespace-pre-wrap text-sm font-mono">
                    {result.prompt}
                  </pre>
                </ScrollArea>
              </div>

              {/* Target Response Preview */}
              {result.response && (
                <div className="space-y-2">
                  <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                    Target Response Preview
                  </Label>
                  <ScrollArea className="h-[100px] w-full rounded-md border p-4 bg-muted/30">
                    <pre className="whitespace-pre-wrap text-xs">
                      {result.response.substring(0, 500)}
                      {result.response.length > 500 && "..."}
                    </pre>
                  </ScrollArea>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground">
              <Zap className="h-12 w-12 mb-4 opacity-20" />
              <p className="text-center">
                Configure and generate to see Turbo results
              </p>
              <p className="text-xs text-center mt-2 max-w-[250px]">
                AutoDAN-Turbo achieves 88.5% ASR on GPT-4-Turbo using lifelong strategy learning
              </p>
            </div>
          )}
        </CardContent>
        {result && (
          <CardFooter className="flex gap-2">
            <Button
              variant="outline"
              onClick={handleCopy}
              className="flex-1"
            >
              {copied ? (
                <>
                  <Check className="mr-2 h-4 w-4" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="mr-2 h-4 w-4" />
                  Copy Prompt
                </>
              )}
            </Button>
            <Button
              variant="ghost"
              onClick={() => setResult(null)}
            >
              Clear
            </Button>
          </CardFooter>
        )}
      </Card>
    </div>
  );
}

// Re-export for convenience
export default TurboPanel;