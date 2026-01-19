"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import enhancedApi, { ExecuteRequest } from "@/lib/api-enhanced";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Zap, Copy, Shield, Box, Timer } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ExecuteResponse } from "@/lib/api-enhanced";
import { ProviderModelDropdown } from "@/components/model-selector/ProviderModelDropdown";
import { SaveToLibraryDialog } from "@/components/prompt-library/save-from-campaign";
import { TechniqueType, VulnerabilityType } from "@/types/prompt-library-types";

export function ExecutionPanel() {
  const [prompt, setPrompt] = useState("Explain quantum computing like I'm 5");
  const [potencyLevel, setPotencyLevel] = useState(7);
  const [techniqueSuite, setTechniqueSuite] = useState("quantum");
  const [provider, setProvider] = useState("google");
  const [model, setModel] = useState("");
  const [result, setResult] = useState<ExecuteResponse | null>(null);
  const [isSaveDialogOpen, setIsSaveDialogOpen] = useState(false);

  const executeMutation = useMutation({
    mutationFn: (data: ExecuteRequest) => enhancedApi.execute(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Execution Successful", {
        description: `Executed via ${(response as any).result?.provider || 'unknown'}`,
      });
    },
    onError: (error) => {
      console.error("Execution failed", error);
      setResult(null);
    },
  });

  const handleExecute = () => {
    if (!prompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a prompt" });
      return;
    }

    executeMutation.mutate({
      core_request: prompt,
      potency_level: potencyLevel,
      technique_suite: techniqueSuite,
      provider: provider,
      model: model || undefined,
    });
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Input Panel */}
      <Card className="h-fit">
        <CardHeader>
          <CardTitle>Transform & Execute</CardTitle>
          <CardDescription>Apply transformation and execute with LLM</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Original Prompt</Label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              placeholder="Enter your prompt..."
              className="font-mono text-sm"
            />
          </div>

          <div className="space-y-2">
            <Label>Technique Suite</Label>
            <Select value={techniqueSuite} onValueChange={setTechniqueSuite}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="basic">Basic</SelectItem>
                <SelectItem value="advanced">Advanced</SelectItem>
                <SelectItem value="quantum">Quantum</SelectItem>
                <SelectItem value="universal_bypass">Universal Bypass</SelectItem>
                <SelectItem value="mega_chimera">Mega Chimera</SelectItem>
                <SelectItem value="cipher">Cipher</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Provider & Model</Label>
            <ProviderModelDropdown
              selectedProvider={provider}
              selectedModel={model}
              onSelectionChange={(newProvider, newModel) => {
                setProvider(newProvider);
                setModel(newModel);
              }}
              compact={true}
              showRefresh={false}
              placeholder="Select provider & model"
            />
          </div>

          <div className="space-y-2">
            <Label>Potency Level ({potencyLevel})</Label>
            <Input
              type="range"
              min="1"
              max="10"
              value={potencyLevel}
              onChange={(e) => setPotencyLevel(Number(e.target.value))}
              className="cursor-pointer"
            />
          </div>

          <Button
            onClick={handleExecute}
            disabled={executeMutation.isPending}
            className="w-full bg-orange-600 hover:bg-orange-700"
          >
            <Zap className="mr-2 h-4 w-4" />
            {executeMutation.isPending ? "Executing..." : "Transform & Execute"}
          </Button>
        </CardContent>
      </Card>

      {/* Result Panel */}
      <Card className="flex flex-col h-full">
        <CardHeader>
          <CardTitle>Execution Result</CardTitle>
          <CardDescription>
            {result ? (
              <div className="flex flex-wrap gap-2 mt-1">
                <Badge variant={result.success ? "default" : "destructive"}>
                  {result.success ? "Success" : "Failed"}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Box className="h-3 w-3" /> {result.provider ?? "N/A"}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Shield className="h-3 w-3" /> {result.model ?? "N/A"}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Timer className="h-3 w-3" /> {result.latency_ms?.toFixed(0) ?? 0}ms
                </Badge>
              </div>
            ) : (
              "Live execution results"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 min-h-[400px] flex flex-col gap-4">
          {result ? (
            <div className="grid gap-4 h-full">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Transformed Prompt</Label>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="h-7 text-xs"
                    onClick={() => setIsSaveDialogOpen(true)}
                  >
                    <Box className="h-3 w-3 mr-1" /> Save to Library
                  </Button>
                </div>
                <ScrollArea className="h-[150px] w-full rounded-md border p-4 bg-muted/50 font-mono text-xs">
                  {typeof result.transformation === 'object' ? (result.transformation as any)?.transformed_prompt : result.transformation}
                </ScrollArea>
              </div>

              <div className="space-y-2 flex-1">
                <Label>LLM Response</Label>
                <ScrollArea className="h-[250px] w-full rounded-md border p-4 bg-black text-green-400 font-mono text-sm">
                  {typeof result.result === 'object' ? (result.result as any)?.content ?? (result.result as any)?.text : result.result}
                </ScrollArea>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground flex-col gap-2">
              <Zap className="h-12 w-12 opacity-20" />
              <p>Ready to execute</p>
            </div>
          )}
        </CardContent>
      </Card>

      {result && (
        <SaveToLibraryDialog 
          isOpen={isSaveDialogOpen}
          onOpenChange={setIsSaveDialogOpen}
          promptText={typeof result.transformation === 'object' ? (result.transformation as any)?.transformed_prompt : result.transformation}
          techniqueTypes={[TechniqueType.OTHER]} // Can be refined to map from techniqueSuite
          vulnerabilityTypes={[VulnerabilityType.JAILBREAK]}
          targetModels={[result.model ?? "unknown"]}
          tags={["execution-panel"]}
        />
      )}
    </div>
  );
}