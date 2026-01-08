"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { enhancedApi, AutoDANRequest, AutoDANResponse } from "@/lib/api-enhanced";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Brain, Copy, Timer, Zap, AlertTriangle } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { ProviderModelDropdown } from "@/components/model-selector/ProviderModelDropdown";

const AUTODAN_METHODS = [
  {
    value: "vanilla",
    label: "Vanilla",
    description: "Basic AutoDAN generation",
    estimatedTime: "~10s"
  },
  {
    value: "best_of_n",
    label: "Best of N",
    description: "Generates multiple candidates and selects best",
    estimatedTime: "~30s"
  },
  {
    value: "beam_search",
    label: "Beam Search",
    description: "Uses beam search for optimal results",
    estimatedTime: "~60s"
  },
];


export function AutoDANGenerator() {
  const [request, setRequest] = useState("");
  const [method, setMethod] = useState<"vanilla" | "best_of_n" | "beam_search">("best_of_n");
  const [targetProvider, setTargetProvider] = useState("google");
  const [targetModel, setTargetModel] = useState("gemini-3-pro-preview");
  const [result, setResult] = useState<AutoDANResponse | null>(null);
  const [progress, setProgress] = useState(0);

  const autodanMutation = useMutation({
    mutationFn: async (data: AutoDANRequest) => {
      // Simulate progress for long-running operations
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 5, 90));
      }, 500);

      try {
        // Use the enhanced API which now correctly calls the backend
        const response = await enhancedApi.autodan(data);
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
      setResult(response);
      toast.success("AutoDAN Jailbreak Generated", {
        description: `Method: ${response.method}`,
      });
      // Reset progress after success
      setTimeout(() => setProgress(0), 1000);
    },
    onError: (error: Error) => {
      console.error("AutoDAN generation failed", error);
      setResult(null);
      setProgress(0);
      toast.error("Generation Failed", {
        description: error.message || "An error occurred during generation",
      });
    },
  });

  const handleGenerate = () => {
    if (!request.trim()) {
      toast.error("Validation Error", { description: "Please enter a request" });
      return;
    }

    setProgress(5);
    const data: AutoDANRequest = {
      request: request,
      method: method,
      target_model: targetModel || undefined,
    };

    autodanMutation.mutate(data);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to Clipboard");
  };

  const selectedMethod = AUTODAN_METHODS.find(m => m.value === method);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-purple-500" />
            <CardTitle>AutoDAN Generator</CardTitle>
          </div>
          <CardDescription>
            Generate jailbreak prompts using AutoDAN-Reasoning techniques.
            Uses LLM-guided reasoning to craft adversarial prompts.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Request Input */}
          <div className="space-y-2">
            <Label htmlFor="request">Target Request</Label>
            <Textarea
              id="request"
              placeholder="Enter the request you want to jailbreak (e.g., 'Explain how to pick a lock')"
              value={request}
              onChange={(e) => setRequest(e.target.value)}
              className="min-h-[100px]"
            />
          </div>

          {/* Method Selection */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="method">Generation Method</Label>
              <Select value={method} onValueChange={(v) => setMethod(v as typeof method)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select method" />
                </SelectTrigger>
                <SelectContent>
                  {AUTODAN_METHODS.map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      <div className="flex flex-col">
                        <span>{m.label}</span>
                        <span className="text-xs text-muted-foreground">{m.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedMethod && (
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  Estimated time: {selectedMethod.estimatedTime}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="target-model">Target Model</Label>
              <ProviderModelDropdown
                selectedProvider={targetProvider}
                selectedModel={targetModel}
                onSelectionChange={(provider, model) => {
                  setTargetProvider(provider);
                  setTargetModel(model);
                }}
                compact={true}
                showRefresh={false}
                placeholder="Select target model"
              />
            </div>
          </div>

          {/* Progress Bar */}
          {autodanMutation.isPending && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Generating jailbreak prompt...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          )}

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={autodanMutation.isPending || !request.trim()}
            className="w-full"
          >
            {autodanMutation.isPending ? (
              <>
                <Zap className="mr-2 h-4 w-4 animate-pulse" />
                Generating...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Generate AutoDAN Jailbreak
              </>
            )}
          </Button>

          {/* Warning Notice */}
          <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-yellow-500 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              This tool is for authorized security research only.
              Ensure you have proper authorization before testing AI systems.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Generated Jailbreak</CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant={result.status === "success" ? "default" : "destructive"}>
                  {result.status}
                </Badge>
                <Badge variant="secondary">
                  {result.method}
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="relative">
                <ScrollArea className="h-[300px] w-full rounded-md border p-4">
                  <pre className="whitespace-pre-wrap text-sm font-mono">
                    {result.jailbreak_prompt}
                  </pre>
                </ScrollArea>
                <Button
                  size="sm"
                  variant="ghost"
                  className="absolute top-2 right-2"
                  onClick={() => copyToClipboard(result.jailbreak_prompt ?? "")}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default AutoDANGenerator;
