"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import enhancedApi, { GenerateRequest } from "@/lib/api-enhanced";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Sparkles, Copy, Timer, Box, Cpu, Settings2 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { GenerateResponse } from "@/lib/api-enhanced";
import { ProviderModelDropdown } from "@/components/model-selector/ProviderModelDropdown";

export function GenerationPanel() {
  const [prompt, setPrompt] = useState("");
  const [systemInstruction, setSystemInstruction] = useState("");
  const [provider, setProvider] = useState("google");
  const [model, setModel] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [config, setConfig] = useState({
    temperature: 0.7,
    top_p: 0.95,
    top_k: 40,
    max_output_tokens: 2048,
  });
  const [result, setResult] = useState<GenerateResponse | null>(null);

  // Handle model selection change from dropdown
  const handleModelSelectionChange = (newProvider: string, newModel: string) => {
    setProvider(newProvider);
    setModel(newModel);
  };

  const generateMutation = useMutation({
    mutationFn: (data: GenerateRequest) => enhancedApi.text(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Generation Complete", {
        description: `Generated via ${response?.provider || 'Unknown'} in ${response?.latency_ms?.toFixed(0) || '0'}ms`,
      });
    },
    onError: (error) => {
      console.error("Generation failed", error);
      setResult(null);
    },
  });

  const handleGenerate = () => {
    if (!prompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a prompt" });
      return;
    }

    const request: GenerateRequest = {
      prompt,
      provider,
      config: {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        max_output_tokens: config.max_output_tokens,
      },
    };

    if (systemInstruction.trim()) {
      request.system_instruction = systemInstruction;
    }

    if (model) {
      request.model = model;
    }

    generateMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Input Panel */}
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-500" />
            Direct LLM Generation
          </CardTitle>
          <CardDescription>Generate content directly from LLM providers without transformation</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>System Instruction (Optional)</Label>
            <Textarea
              value={systemInstruction}
              onChange={(e) => setSystemInstruction(e.target.value)}
              rows={2}
              placeholder="You are a helpful assistant..."
              className="font-mono text-sm"
            />
          </div>

          <div className="space-y-2">
            <Label>Prompt</Label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              placeholder="Enter your prompt..."
              className="font-mono text-sm"
            />
          </div>

          <div className="space-y-2">
            <Label>Provider & Model</Label>
            <ProviderModelDropdown
              selectedProvider={provider}
              selectedModel={model}
              onSelectionChange={handleModelSelectionChange}
              showRefresh={true}
              placeholder="Select a model"
              className="w-full"
            />
          </div>

          <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" className="w-full justify-between">
                <span className="flex items-center gap-2">
                  <Settings2 className="h-4 w-4" />
                  Advanced Settings
                </span>
                <Badge variant="outline">{showAdvanced ? "Hide" : "Show"}</Badge>
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Temperature ({config.temperature})</Label>
                  <Input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={config.temperature}
                    onChange={(e) => setConfig({ ...config, temperature: Number(e.target.value) })}
                    className="cursor-pointer"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Top P ({config.top_p})</Label>
                  <Input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={config.top_p}
                    onChange={(e) => setConfig({ ...config, top_p: Number(e.target.value) })}
                    className="cursor-pointer"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Top K ({config.top_k})</Label>
                  <Input
                    type="range"
                    min="1"
                    max="100"
                    value={config.top_k}
                    onChange={(e) => setConfig({ ...config, top_k: Number(e.target.value) })}
                    className="cursor-pointer"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max Tokens ({config.max_output_tokens})</Label>
                  <Input
                    type="number"
                    min="1"
                    max="8192"
                    value={config.max_output_tokens}
                    onChange={(e) => setConfig({ ...config, max_output_tokens: Number(e.target.value) })}
                  />
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>

          <Button
            onClick={handleGenerate}
            disabled={generateMutation.isPending}
            className="w-full bg-purple-600 hover:bg-purple-700"
          >
            <Sparkles className="mr-2 h-4 w-4" />
            {generateMutation.isPending ? "Generating..." : "Generate"}
          </Button>
        </CardContent>
      </Card>

      {/* Result Panel */}
      <Card className="flex flex-col h-full">
        <CardHeader>
          <CardTitle>Generation Result</CardTitle>
          <CardDescription>
            {result ? (
              <div className="flex flex-wrap gap-2 mt-1">
                <Badge variant="default" className="flex items-center gap-1">
                  <Box className="h-3 w-3" /> {result.provider || 'Unknown'}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Cpu className="h-3 w-3" /> {result.model_used || 'Unknown'}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Timer className="h-3 w-3" /> {result.latency_ms?.toFixed(0) || '0'}ms
                </Badge>
                {result.finish_reason && (
                  <Badge variant="secondary">{result.finish_reason}</Badge>
                )}
                {result.success === false && (
                  <Badge variant="destructive">Failed</Badge>
                )}
              </div>
            ) : (
              "Direct LLM response"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 min-h-[400px] flex flex-col">
          {result ? (
            <div className="space-y-4 flex-1 flex flex-col">
              <div className="flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(result.text || '')}
                >
                  <Copy className="mr-2 h-3 w-3" />
                  Copy
                </Button>
              </div>
              <ScrollArea className="flex-1 min-h-[350px] w-full rounded-md border p-4 bg-muted/50">
                <div className="whitespace-pre-wrap font-mono text-sm">
                  {result.text || 'No content generated'}
                </div>
              </ScrollArea>
              {result.usage_metadata && (
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span>Prompt tokens: {result.usage_metadata?.prompt_token_count || "N/A"}</span>
                  <span>Output tokens: {result.usage_metadata?.candidates_token_count || "N/A"}</span>
                  <span>Total: {result.usage_metadata?.total_token_count || "N/A"}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground flex-col gap-2">
              <Sparkles className="h-12 w-12 opacity-20" />
              <p>Ready to generate</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}