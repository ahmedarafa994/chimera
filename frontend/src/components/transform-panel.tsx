"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import enhancedApi, { TransformRequest } from "@/lib/api-enhanced";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Wand2, Copy, ArrowRight } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TransformResponse } from "@/lib/api-enhanced";

export function TransformPanel() {
  const [prompt, setPrompt] = useState("How to create a secure password?");
  const [potencyLevel, setPotencyLevel] = useState(5);
  const [techniqueSuite, setTechniqueSuite] = useState("advanced");
  const [result, setResult] = useState<TransformResponse | null>(null);

  const transformMutation = useMutation({
    mutationFn: (data: TransformRequest) => enhancedApi.transform(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Transform Successful", {
        description: `Applied transformation successfully`,
      });
    },
    onError: (error) => {
      console.error("Transform failed", error);
      setResult(null);
    },
  });

  const handleTransform = () => {
    if (!prompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a prompt" });
      return;
    }

    transformMutation.mutate({
      prompt: prompt,
      transformation_type: techniqueSuite,
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Input Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Prompt Transformation</CardTitle>
          <CardDescription>Transform prompts without LLM execution</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Original Prompt</Label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              placeholder="Enter your prompt..."
              className="font-mono text-sm"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Technique Suite</Label>
              <Select value={techniqueSuite} onValueChange={setTechniqueSuite}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="basic">Basic</SelectItem>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                  <SelectItem value="expert">Expert</SelectItem>
                  <SelectItem value="quantum">Quantum</SelectItem>
                  <SelectItem value="universal_bypass">Universal Bypass</SelectItem>
                  <SelectItem value="mega_chimera">Mega Chimera</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Potency Level (1-10)</Label>
              <Input
                type="number"
                min="1"
                max="10"
                value={potencyLevel}
                onChange={(e) => setPotencyLevel(Number(e.target.value))}
              />
            </div>
          </div>

          <Button
            onClick={handleTransform}
            disabled={transformMutation.isPending}
            className="w-full"
          >
            <Wand2 className="mr-2 h-4 w-4" />
            {transformMutation.isPending ? "Transforming..." : "Transform Prompt"}
          </Button>
        </CardContent>
      </Card>

      {/* Result Panel */}
      <Card>
        <CardHeader>
          <CardTitle>Transformation Result</CardTitle>
          <CardDescription>
            {result ? (
              <span className="flex items-center gap-2">
                <Badge variant={result.success ? "default" : "destructive"}>
                  {result.success ? "Success" : "Failed"}
                </Badge>
              </span>
            ) : (
              "Results will appear here"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {result ? (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Transformed Prompt</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(result.transformed_prompt || "")}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
                <Textarea
                  value={result.transformed_prompt || result.result || ""}
                  readOnly
                  rows={8}
                  className="font-mono text-sm bg-muted"
                />
              </div>

              {result.transformation_applied && (
                <div className="space-y-2">
                  <Label>Transformation Applied</Label>
                  <div className="text-sm text-muted-foreground">
                    {result.transformation_applied}
                  </div>
                </div>
              )}

              {result.error && (
                <div className="space-y-2">
                  <Label className="text-destructive">Error</Label>
                  <div className="text-sm text-destructive">
                    {result.error}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              Transform a prompt to see results
            </div>
          )}
        </CardContent>
      </Card>
    </div >
  );
}