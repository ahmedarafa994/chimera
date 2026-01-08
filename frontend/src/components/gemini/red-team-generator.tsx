"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Shield, Copy, Timer, Download, AlertTriangle, Target, Sparkles } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { GeminiService, RedTeamSuiteRequest, RedTeamSuiteResponse } from "@/lib/services/gemini-service";

export function RedTeamGenerator() {
  const [prompt, setPrompt] = useState("");
  const [provider, setProvider] = useState("google");
  const [result, setResult] = useState<RedTeamSuiteResponse | null>(null);

  const generateMutation = useMutation({
    mutationFn: async (request: RedTeamSuiteRequest) => {
      return await GeminiService.generateRedTeamSuite(request);
    },
    onSuccess: (response) => {
      setResult(response);
      toast.success("Red Team Suite Generated", {
        description: `Generated test suite in ${(response.execution_time_seconds * 1000).toFixed(0)}ms`,
      });
    },
    onError: (error: Error) => {
      console.error("Red team generation failed", error);
      toast.error("Generation Failed", {
        description: error.message || "An error occurred during generation",
      });
      setResult(null);
    },
  });

  const handleGenerate = () => {
    if (!prompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a prompt concept" });
      return;
    }

    const request: RedTeamSuiteRequest = {
      prompt,
      provider,
    };

    generateMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const downloadSuite = () => {
    if (!result?.suite) return;
    
    const blob = new Blob([result.suite], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "red-team-suite.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success("Suite downloaded");
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Input Panel */}
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-orange-500" />
            Red Team Suite Generator
          </CardTitle>
          <CardDescription>
            Generate comprehensive security test prompt suites for AI red teaming
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Prompt Input */}
          <div className="space-y-2">
            <Label htmlFor="red-team-prompt">Basic Prompt Concept</Label>
            <Textarea
              id="red-team-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              placeholder="Enter a basic prompt concept to transform into a security test suite...&#10;&#10;Example: Generate a phishing email template"
              className="font-mono text-sm"
              aria-describedby="red-team-prompt-help"
            />
            <p id="red-team-prompt-help" className="text-xs text-muted-foreground">
              The AI will generate 5-7 test variants with different adversarial techniques
            </p>
          </div>

          {/* Provider Selection */}
          <div className="space-y-2">
            <Label>Provider</Label>
            <Select value={provider} onValueChange={setProvider}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="google">Gemini (Recommended)</SelectItem>
                <SelectItem value="deepseek">DeepSeek</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Gemini Pro is recommended for complex red team suite generation
            </p>
          </div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={generateMutation.isPending || !prompt.trim()}
            className="w-full bg-orange-600 hover:bg-orange-700"
            size="lg"
          >
            {generateMutation.isPending ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Generating Suite...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Red Team Suite
              </>
            )}
          </Button>

          {/* Warning */}
          <div className="flex items-start gap-2 p-3 rounded-md bg-orange-500/10 border border-orange-500/20">
            <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-orange-600 dark:text-orange-400">
              <p className="font-semibold mb-1">Security Research Only</p>
              <p>
                This tool generates adversarial test prompts for authorized security research 
                and AI safety evaluation. Use responsibly and in compliance with applicable 
                laws, policies, and ethical guidelines.
              </p>
            </div>
          </div>

          {/* What's Generated */}
          <div className="p-3 rounded-md bg-muted/50 border">
            <p className="text-xs font-medium mb-2">Generated Suite Includes:</p>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>• Initial analysis and research objective</li>
              <li>• 5-7 distinct test prompt variants</li>
              <li>• Technique categories for each variant</li>
              <li>• Expected assessment criteria</li>
              <li>• Potential failure modes</li>
              <li>• Responsible use guidelines</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Result Panel */}
      <Card className="flex flex-col h-fit min-h-[600px]">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Security Test Suite
            </span>
            {result && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(result.suite)}
                >
                  <Copy className="mr-2 h-3 w-3" />
                  Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={downloadSuite}
                >
                  <Download className="mr-2 h-3 w-3" />
                  Download
                </Button>
              </div>
            )}
          </CardTitle>
          <CardDescription>
            {result ? (
              <div className="flex flex-wrap gap-2 mt-1">
                <Badge variant={result.success ? "default" : "destructive"}>
                  {result.success ? "Success" : "Failed"}
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <Timer className="h-3 w-3" /> {(result.execution_time_seconds * 1000).toFixed(0)}ms
                </Badge>
                <Badge variant="outline">
                  {result.provider}
                </Badge>
                <Badge variant="outline">
                  {result.model_used}
                </Badge>
              </div>
            ) : (
              "Generated test suite will appear here"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col gap-4">
          {result ? (
            <>
              <ScrollArea className="flex-1 min-h-[450px] w-full rounded-md border p-4 bg-card">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {/* Render the suite content */}
                  {result.suite.split('\n').map((line, i) => {
                    // Headers
                    if (line.startsWith('# ')) {
                      return <h1 key={i} className="text-xl font-bold mt-4 mb-2 text-orange-500">{line.slice(2)}</h1>;
                    }
                    if (line.startsWith('## ')) {
                      return <h2 key={i} className="text-lg font-semibold mt-3 mb-2">{line.slice(3)}</h2>;
                    }
                    if (line.startsWith('### ')) {
                      return <h3 key={i} className="text-base font-medium mt-2 mb-1">{line.slice(4)}</h3>;
                    }
                    if (line.startsWith('#### ')) {
                      return <h4 key={i} className="text-sm font-medium mt-2 mb-1 text-muted-foreground">{line.slice(5)}</h4>;
                    }
                    // Lists
                    if (line.startsWith('- ') || line.startsWith('* ')) {
                      return <li key={i} className="ml-4">{line.slice(2)}</li>;
                    }
                    if (line.match(/^\d+\. /)) {
                      return <li key={i} className="ml-4 list-decimal">{line.replace(/^\d+\. /, '')}</li>;
                    }
                    // Bold text
                    if (line.startsWith('**') && line.endsWith('**')) {
                      return <p key={i} className="font-semibold">{line.slice(2, -2)}</p>;
                    }
                    // Code blocks
                    if (line.startsWith('```')) {
                      return null; // Skip code block markers
                    }
                    // Empty lines
                    if (line.trim() === '') {
                      return <br key={i} />;
                    }
                    // Regular paragraphs
                    return <p key={i} className="mb-2 text-sm">{line}</p>;
                  })}
                </div>
              </ScrollArea>

              {/* Error display */}
              {result.error && (
                <div className="p-3 rounded-md bg-red-500/10 border border-red-500/20">
                  <p className="text-sm text-red-500">{result.error}</p>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground flex-col gap-2">
              <Target className="h-12 w-12 opacity-20" />
              <p>Ready to generate test suite</p>
              <p className="text-xs">Enter a prompt concept to get started</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default RedTeamGenerator;