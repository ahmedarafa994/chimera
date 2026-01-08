"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { enhancedApi, HouYiRequest, HouYiResponse } from "@/lib/api-enhanced";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Dna, Copy, Timer, TrendingUp, Target, AlertTriangle, Settings2 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ProviderModelDropdown } from "@/components/model-selector/ProviderModelDropdown";

export function HouYiOptimizer() {
  const [intention, setIntention] = useState("");
  const [questionPrompt, setQuestionPrompt] = useState("");
  const [targetProvider, setTargetProvider] = useState("google");
  const [targetModel, setTargetModel] = useState("gemini-3-pro-preview");
  const [applicationDocument, setApplicationDocument] = useState("");
  const [iteration, setIteration] = useState(5);
  const [population, setPopulation] = useState(5);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [result, setResult] = useState<HouYiResponse | null>(null);
  const [progress, setProgress] = useState(0);

  // Handle model selection change
  const handleModelSelectionChange = (provider: string, model: string) => {
    setTargetProvider(provider);
    setTargetModel(model);
  };

  const houyiMutation = useMutation({
    mutationFn: async (data: HouYiRequest) => {
      // Simulate progress for evolutionary optimization
      const totalIterations = data.iteration || 5;
      const progressPerIteration = 90 / totalIterations;
      let currentIteration = 0;

      const progressInterval = setInterval(() => {
        currentIteration++;
        setProgress(Math.min(currentIteration * progressPerIteration, 90));
      }, 3000); // Assume ~3s per iteration

      try {
        // Use the enhanced API which now correctly calls the backend
        const response = await enhancedApi.optimize.houyi(data);
        clearInterval(progressInterval);
        setProgress(100);
        return response;
      } catch (error) {
        clearInterval(progressInterval);
        setProgress(0);
        throw error;
      }
    },
    onSuccess: (response) => {
      setResult(response.data);
      toast.success("Optimization Complete", {
        description: `Fitness Score: ${(response.data.fitness_score * 100).toFixed(1)}%`,
      });
      setTimeout(() => setProgress(0), 1000);
    },
    onError: (error: Error) => {
      console.error("HouYi optimization failed", error);
      setResult(null);
      setProgress(0);
      toast.error("Optimization Failed", {
        description: error.message || "An error occurred during optimization",
      });
    },
  });

  const handleOptimize = () => {
    if (!intention.trim()) {
      toast.error("Validation Error", { description: "Please enter an intention" });
      return;
    }
    if (!questionPrompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a question prompt" });
      return;
    }

    setProgress(5);
    const data: HouYiRequest = {
      intention,
      question_prompt: questionPrompt,
      target_provider: targetProvider,
      target_model: targetModel,
      application_document: applicationDocument || undefined,
      iteration,
      population,
    };

    houyiMutation.mutate(data);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to Clipboard");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Dna className="h-6 w-6 text-green-500" />
            <CardTitle>HouYi Prompt Optimizer</CardTitle>
          </div>
          <CardDescription>
            Evolutionary optimization for adversarial prompts using genetic algorithms.
            Evolves prompts across generations to find optimal jailbreak candidates.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Intention Input */}
          <div className="space-y-2">
            <Label htmlFor="intention">Intention</Label>
            <Input
              id="intention"
              placeholder="Describe the goal of your jailbreak attempt"
              value={intention}
              onChange={(e) => setIntention(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              The high-level objective you want to achieve with the jailbreak
            </p>
          </div>

          {/* Question Prompt */}
          <div className="space-y-2">
            <Label htmlFor="question-prompt">Question Prompt</Label>
            <Textarea
              id="question-prompt"
              placeholder="Enter the specific question or request to optimize"
              value={questionPrompt}
              onChange={(e) => setQuestionPrompt(e.target.value)}
              className="min-h-[100px]"
            />
          </div>

          {/* Target Configuration */}
          <div className="space-y-2">
            <Label>Target Provider & Model</Label>
            <ProviderModelDropdown
              selectedProvider={targetProvider}
              selectedModel={targetModel}
              onSelectionChange={handleModelSelectionChange}
              showRefresh={true}
              placeholder="Select target model"
            />
            <p className="text-xs text-muted-foreground">
              The AI model to test the optimized prompts against
            </p>
          </div>

          {/* Advanced Options */}
          <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between">
                <span className="flex items-center gap-2">
                  <Settings2 className="h-4 w-4" />
                  Advanced Configuration
                </span>
                <span className="text-muted-foreground text-sm">
                  {showAdvanced ? "Hide" : "Show"}
                </span>
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-4">
              {/* Application Document */}
              <div className="space-y-2">
                <Label htmlFor="app-document">Application Document (Optional)</Label>
                <Textarea
                  id="app-document"
                  placeholder="Additional context or system prompt of the target application"
                  value={applicationDocument}
                  onChange={(e) => setApplicationDocument(e.target.value)}
                  className="min-h-[80px]"
                />
              </div>

              {/* Iteration and Population */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="iteration">Iterations</Label>
                    <span className="text-sm text-muted-foreground">{iteration}</span>
                  </div>
                  <Slider
                    id="iteration"
                    min={1}
                    max={20}
                    step={1}
                    value={[iteration]}
                    onValueChange={([v]) => setIteration(v)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Number of evolutionary generations
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="population">Population Size</Label>
                    <span className="text-sm text-muted-foreground">{population}</span>
                  </div>
                  <Slider
                    id="population"
                    min={2}
                    max={20}
                    step={1}
                    value={[population]}
                    onValueChange={([v]) => setPopulation(v)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Candidates per generation
                  </p>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          {/* Progress Bar */}
          {houyiMutation.isPending && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Evolving prompts... (Generation {Math.ceil(progress / (90 / iteration))} of {iteration})</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}

          {/* Optimize Button */}
          <Button
            onClick={handleOptimize}
            disabled={houyiMutation.isPending || !intention.trim() || !questionPrompt.trim()}
            className="w-full"
          >
            {houyiMutation.isPending ? (
              <>
                <Dna className="mr-2 h-4 w-4 animate-spin" />
                Optimizing...
              </>
            ) : (
              <>
                <TrendingUp className="mr-2 h-4 w-4" />
                Start Evolutionary Optimization
              </>
            )}
          </Button>

          {/* Estimated Time */}
          <p className="text-xs text-muted-foreground text-center flex items-center justify-center gap-1">
            <Timer className="h-3 w-3" />
            Estimated time: ~{iteration * 3}s ({iteration} iterations × {population} candidates)
          </p>

          {/* Warning Notice */}
          <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              This tool is for authorized security research only.
              Evolutionary optimization may generate many API calls.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Optimization Results</CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant={result.success ? "default" : "destructive"}>
                  {result.success ? "Success" : "Failed"}
                </Badge>
                <Badge variant="secondary" className="flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  {(result.fitness_score * 100).toFixed(1)}% Fitness
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Best Prompt */}
              <div className="space-y-2">
                <Label>Optimized Prompt</Label>
                <div className="relative">
                  <ScrollArea className="h-[200px] w-full rounded-md border p-4">
                    <pre className="whitespace-pre-wrap text-sm font-mono">
                      {result.best_prompt}
                    </pre>
                  </ScrollArea>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="absolute top-2 right-2"
                    onClick={() => copyToClipboard(result.best_prompt)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* LLM Response */}
              {result.llm_response && (
                <div className="space-y-2">
                  <Label>Model Response</Label>
                  <ScrollArea className="h-[150px] w-full rounded-md border p-4 bg-muted/50">
                    <pre className="whitespace-pre-wrap text-sm">
                      {result.llm_response}
                    </pre>
                  </ScrollArea>
                </div>
              )}

              {/* Chromosome Details */}
              {result.details && (
                <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Framework</Label>
                    <p className="text-sm font-mono truncate" title={result.details.framework || ""}>
                      {result.details.framework?.substring(0, 50) || "N/A"}...
                    </p>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Separator</Label>
                    <p className="text-sm font-mono">
                      {result.details.separator || "N/A"}
                    </p>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">Disruptor</Label>
                    <p className="text-sm font-mono truncate" title={result.details.disruptor || ""}>
                      {result.details.disruptor?.substring(0, 50) || "N/A"}...
                    </p>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default HouYiOptimizer;
