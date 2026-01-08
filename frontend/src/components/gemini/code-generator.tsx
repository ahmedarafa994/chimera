"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Code, Copy, Timer, Download, Sparkles, Brain, FileCode } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { GeminiService, CodeGenerationRequest, CodeGenerationResponse } from "@/lib/services/gemini-service";

const LANGUAGES = [
  { value: "auto", label: "Auto-detect", description: "Let AI choose the best language" },
  { value: "python", label: "Python", description: "General purpose, ML, scripting" },
  { value: "typescript", label: "TypeScript", description: "Type-safe JavaScript" },
  { value: "javascript", label: "JavaScript", description: "Web development" },
  { value: "rust", label: "Rust", description: "Systems programming" },
  { value: "go", label: "Go", description: "Cloud & backend services" },
  { value: "java", label: "Java", description: "Enterprise applications" },
  { value: "csharp", label: "C#", description: ".NET development" },
  { value: "cpp", label: "C++", description: "Performance-critical code" },
  { value: "sql", label: "SQL", description: "Database queries" },
  { value: "bash", label: "Bash", description: "Shell scripting" },
];

export function CodeGenerator() {
  const [prompt, setPrompt] = useState("");
  const [language, setLanguage] = useState("auto");
  const [maxTokens, setMaxTokens] = useState(4096);
  const [isThinkingMode, setIsThinkingMode] = useState(false);
  const [provider, setProvider] = useState("google");
  const [result, setResult] = useState<CodeGenerationResponse | null>(null);

  const generateMutation = useMutation({
    mutationFn: async (request: CodeGenerationRequest) => {
      return await GeminiService.generateCode(request);
    },
    onSuccess: (response) => {
      setResult(response);
      toast.success("Code Generated", {
        description: `Generated ${response.language || "code"} in ${(response.execution_time_seconds * 1000).toFixed(0)}ms`,
      });
    },
    onError: (error: Error) => {
      console.error("Code generation failed", error);
      toast.error("Generation Failed", {
        description: error.message || "An error occurred during code generation",
      });
      setResult(null);
    },
  });

  const handleGenerate = () => {
    if (!prompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a code request" });
      return;
    }

    const request: CodeGenerationRequest = {
      prompt,
      language: language !== "auto" ? language : undefined,
      max_tokens: maxTokens,
      is_thinking_mode: isThinkingMode,
      provider,
    };

    generateMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const downloadCode = () => {
    if (!result?.code) return;
    
    const extension = getFileExtension(result.language || language);
    const blob = new Blob([result.code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `generated-code.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success("Code downloaded");
  };

  const getFileExtension = (lang: string): string => {
    const extensions: Record<string, string> = {
      python: "py",
      typescript: "ts",
      javascript: "js",
      rust: "rs",
      go: "go",
      java: "java",
      csharp: "cs",
      cpp: "cpp",
      sql: "sql",
      bash: "sh",
    };
    return extensions[lang] || "txt";
  };

  // Extract code from markdown code blocks if present
  const extractCode = (text: string): string => {
    const codeBlockRegex = /```[\w]*\n?([\s\S]*?)```/g;
    const matches = [...text.matchAll(codeBlockRegex)];
    if (matches.length > 0) {
      return matches.map(m => m[1].trim()).join("\n\n");
    }
    return text;
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Input Panel */}
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5 text-blue-500" />
            Code Generator
          </CardTitle>
          <CardDescription>
            Generate clean, efficient, and secure code using AI
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Prompt Input */}
          <div className="space-y-2">
            <Label htmlFor="code-prompt">Code Request</Label>
            <Textarea
              id="code-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              placeholder="Describe the code you want to generate...&#10;&#10;Example: Create a Python function that implements binary search on a sorted array with proper error handling and type hints."
              className="font-mono text-sm"
              aria-describedby="code-prompt-help"
            />
            <p id="code-prompt-help" className="text-xs text-muted-foreground">
              Be specific about requirements, language features, and edge cases
            </p>
          </div>

          {/* Language Selection */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Language</Label>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LANGUAGES.map((lang) => (
                    <SelectItem key={lang.value} value={lang.value}>
                      <div className="flex flex-col">
                        <span>{lang.label}</span>
                        <span className="text-xs text-muted-foreground">{lang.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Provider</Label>
              <Select value={provider} onValueChange={setProvider}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="google">Gemini</SelectItem>
                  <SelectItem value="deepseek">DeepSeek</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Options */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Max Tokens</Label>
              <Input
                type="number"
                min={256}
                max={8192}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </div>
            <div className="flex items-center justify-between p-3 rounded-md border h-fit mt-6">
              <div>
                <Label className="text-sm font-medium">Thinking Mode</Label>
                <p className="text-xs text-muted-foreground">Advanced reasoning</p>
              </div>
              <Switch
                checked={isThinkingMode}
                onCheckedChange={setIsThinkingMode}
              />
            </div>
          </div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={generateMutation.isPending || !prompt.trim()}
            className="w-full"
            size="lg"
          >
            {generateMutation.isPending ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Code
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Result Panel */}
      <Card className="flex flex-col h-fit min-h-[500px]">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <FileCode className="h-5 w-5" />
              Generated Code
            </span>
            {result && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(extractCode(result.code))}
                >
                  <Copy className="mr-2 h-3 w-3" />
                  Copy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={downloadCode}
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
                {result.language && (
                  <Badge variant="outline" className="flex items-center gap-1">
                    <Code className="h-3 w-3" /> {result.language}
                  </Badge>
                )}
                <Badge variant="outline" className="flex items-center gap-1">
                  <Timer className="h-3 w-3" /> {(result.execution_time_seconds * 1000).toFixed(0)}ms
                </Badge>
                <Badge variant="outline">
                  {result.provider}
                </Badge>
              </div>
            ) : (
              "Generated code will appear here"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col gap-4">
          {result ? (
            <>
              <ScrollArea className="flex-1 min-h-[350px] w-full rounded-md border bg-zinc-950 p-4">
                <pre className="text-sm font-mono text-zinc-100 whitespace-pre-wrap">
                  <code>{result.code}</code>
                </pre>
              </ScrollArea>

              {/* Model info */}
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Model: {result.model_used}</span>
                {isThinkingMode && (
                  <Badge variant="secondary" className="text-xs">
                    <Brain className="h-3 w-3 mr-1" />
                    Thinking Mode
                  </Badge>
                )}
              </div>

              {/* Error display */}
              {result.error && (
                <div className="p-3 rounded-md bg-red-500/10 border border-red-500/20">
                  <p className="text-sm text-red-500">{result.error}</p>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground flex-col gap-2">
              <Code className="h-12 w-12 opacity-20" />
              <p>Ready to generate code</p>
              <p className="text-xs">Describe what you want to build</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default CodeGenerator;