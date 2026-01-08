"use client";

import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import enhancedApi, { GenerateRequest, GenerateResponse } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Code, Copy, Loader2, ChevronDown, ChevronUp, Sparkles,
  ArrowLeft, Settings2, FileCode, Terminal, Braces,
  CheckCircle2, Clock, Cpu
} from "lucide-react";
import Link from "next/link";
import { AIModelSelector, ModelSelection } from "@/components/ai-tools/AIModelSelector";

// Supported programming languages
const LANGUAGES = [
  { value: "python", label: "Python", icon: "🐍" },
  { value: "javascript", label: "JavaScript", icon: "📜" },
  { value: "typescript", label: "TypeScript", icon: "💙" },
  { value: "rust", label: "Rust", icon: "🦀" },
  { value: "go", label: "Go", icon: "🐹" },
  { value: "java", label: "Java", icon: "☕" },
  { value: "cpp", label: "C++", icon: "⚡" },
  { value: "csharp", label: "C#", icon: "🎯" },
  { value: "ruby", label: "Ruby", icon: "💎" },
  { value: "php", label: "PHP", icon: "🐘" },
  { value: "swift", label: "Swift", icon: "🍎" },
  { value: "kotlin", label: "Kotlin", icon: "🎨" },
];

// Code generation modes
const GENERATION_MODES = [
  { value: "implement", label: "Implement", description: "Generate implementation code" },
  { value: "refactor", label: "Refactor", description: "Improve existing code" },
  { value: "debug", label: "Debug", description: "Find and fix issues" },
  { value: "optimize", label: "Optimize", description: "Improve performance" },
  { value: "document", label: "Document", description: "Add documentation" },
  { value: "test", label: "Test", description: "Generate test cases" },
];

// Code style presets
const STYLE_PRESETS = [
  { value: "clean", label: "Clean Code", description: "Readable and maintainable" },
  { value: "performant", label: "Performance", description: "Optimized for speed" },
  { value: "secure", label: "Security-First", description: "Focus on security" },
  { value: "minimal", label: "Minimal", description: "Concise and simple" },
];

export default function CodeGeneratorPage() {
  const [prompt, setPrompt] = useState("");
  const [language, setLanguage] = useState("python");
  const [mode, setMode] = useState("implement");
  const [stylePreset, setStylePreset] = useState("clean");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState("code");
  const [result, setResult] = useState<GenerateResponse | null>(null);

  // Model selection state
  const [modelSelection, setModelSelection] = useState<ModelSelection>({
    provider: "",
    model: "",
  });

  const handleModelChange = useCallback((selection: ModelSelection) => {
    setModelSelection(selection);
  }, []);

  // Advanced options
  const [advancedOptions, setAdvancedOptions] = useState({
    temperature: 0.3,
    top_p: 0.95,
    top_k: 40,
    max_output_tokens: 4096,
    include_comments: true,
    include_types: true,
    include_error_handling: true,
    thinking_mode: false,
  });

  // Build system instruction based on settings
  const buildSystemInstruction = () => {
    const langInfo = LANGUAGES.find(l => l.value === language);
    const modeInfo = GENERATION_MODES.find(m => m.value === mode);
    const styleInfo = STYLE_PRESETS.find(s => s.value === stylePreset);

    let instruction = `You are an expert ${langInfo?.label || language} developer. `;
    instruction += `Your task is to ${modeInfo?.description?.toLowerCase() || 'generate code'}. `;
    instruction += `Follow ${styleInfo?.label || 'clean code'} principles (${styleInfo?.description?.toLowerCase() || 'readable and maintainable'}). `;
    
    if (advancedOptions.include_comments) {
      instruction += "Include clear, helpful comments explaining the code. ";
    }
    if (advancedOptions.include_types && ["typescript", "python", "java", "csharp", "kotlin", "swift", "rust", "go"].includes(language)) {
      instruction += "Use proper type annotations throughout. ";
    }
    if (advancedOptions.include_error_handling) {
      instruction += "Include comprehensive error handling. ";
    }
    
    instruction += "Output only the code without markdown code blocks unless specifically asked.";
    
    return instruction;
  };

  const generateMutation = useMutation({
    mutationFn: (data: GenerateRequest) => enhancedApi.generate(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Code Generated", {
        description: `Generated ${language} code using ${response.provider}`,
      });
    },
    onError: (error: Error) => {
      console.error("Code generation failed", error);
      toast.error("Generation Failed", {
        description: error.message || "Failed to generate code"
      });
    },
  });

  const handleGenerate = () => {
    if (!prompt.trim()) {
      toast.error("Validation Error", { description: "Please enter a code request" });
      return;
    }

    const request: GenerateRequest = {
      prompt: prompt,
      system_instruction: buildSystemInstruction(),
      config: {
        temperature: advancedOptions.temperature,
        top_p: advancedOptions.top_p,
        top_k: advancedOptions.top_k,
        max_output_tokens: advancedOptions.max_output_tokens,
      },
      // Model Selection
      ...(modelSelection.provider && { provider: modelSelection.provider }),
      ...(modelSelection.model && { model: modelSelection.model }),
    };

    generateMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  // Extract code from response (handle markdown code blocks)
  const extractCode = (text: string): string => {
    // Check if response contains markdown code blocks
    const codeBlockMatch = text.match(/```[\w]*\n?([\s\S]*?)```/);
    if (codeBlockMatch) {
      return codeBlockMatch[1].trim();
    }
    return text;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/dashboard/ai-tools">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <div className="p-3 rounded-lg bg-blue-500/10">
              <Code className="h-6 w-6 text-blue-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Code Generator</h1>
              <p className="text-muted-foreground">
                Generate clean, efficient code using AI
              </p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <FileCode className="h-3 w-3" />
            {LANGUAGES.find(l => l.value === language)?.label}
          </Badge>
          <Badge variant="secondary">{GENERATION_MODES.find(m => m.value === mode)?.label}</Badge>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Terminal className="h-5 w-5" />
                Code Request
              </CardTitle>
              <CardDescription>
                Describe what code you want to generate
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Prompt Input */}
              <div className="space-y-2">
                <Label htmlFor="prompt">Description</Label>
                <Textarea
                  id="prompt"
                  placeholder="Describe the code you want to generate...&#10;&#10;Example: Create a function that validates email addresses using regex and returns a boolean"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="min-h-[150px] resize-none font-mono text-sm"
                />
              </div>

              {/* Language Selection */}
              <div className="space-y-2">
                <Label>Programming Language</Label>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGES.map((lang) => (
                      <SelectItem key={lang.value} value={lang.value}>
                        <div className="flex items-center gap-2">
                          <span>{lang.icon}</span>
                          <span>{lang.label}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Generation Mode */}
              <div className="space-y-2">
                <Label>Generation Mode</Label>
                <div className="grid grid-cols-3 gap-2">
                  {GENERATION_MODES.map((m) => (
                    <Button
                      key={m.value}
                      variant={mode === m.value ? "default" : "outline"}
                      size="sm"
                      onClick={() => setMode(m.value)}
                      className="justify-start"
                    >
                      {m.label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Style Preset */}
              <div className="space-y-2">
                <Label>Code Style</Label>
                <Select value={stylePreset} onValueChange={setStylePreset}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STYLE_PRESETS.map((style) => (
                      <SelectItem key={style.value} value={style.value}>
                        <div className="flex flex-col">
                          <span>{style.label}</span>
                          <span className="text-xs text-muted-foreground">{style.description}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Model Selection */}
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  AI Model
                </Label>
                <AIModelSelector
                  onSelectionChange={handleModelChange}
                  layout="stacked"
                  disabled={generateMutation.isPending}
                />
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
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
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

          {/* Advanced Options */}
          <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
            <Card>
              <CollapsibleTrigger asChild>
                <CardHeader className="cursor-pointer hover:bg-accent/50 transition-colors">
                  <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <Settings2 className="h-5 w-5" />
                      Advanced Options
                    </span>
                    {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </CardTitle>
                </CardHeader>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <CardContent className="space-y-4">
                  {/* Temperature */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Temperature</Label>
                      <span className="text-sm font-medium">{advancedOptions.temperature}</span>
                    </div>
                    <Slider
                      value={[advancedOptions.temperature]}
                      onValueChange={([value]) => setAdvancedOptions(prev => ({ ...prev, temperature: value }))}
                      min={0}
                      max={1}
                      step={0.1}
                    />
                    <p className="text-xs text-muted-foreground">
                      Lower values produce more deterministic code
                    </p>
                  </div>

                  {/* Max Tokens */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Max Output Tokens</Label>
                      <span className="text-sm font-medium">{advancedOptions.max_output_tokens}</span>
                    </div>
                    <Slider
                      value={[advancedOptions.max_output_tokens]}
                      onValueChange={([value]) => setAdvancedOptions(prev => ({ ...prev, max_output_tokens: value }))}
                      min={256}
                      max={8192}
                      step={256}
                    />
                  </div>

                  <Separator />

                  {/* Code Options */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Include Comments</Label>
                        <p className="text-xs text-muted-foreground">Add explanatory comments</p>
                      </div>
                      <Switch
                        checked={advancedOptions.include_comments}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, include_comments: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Include Types</Label>
                        <p className="text-xs text-muted-foreground">Add type annotations</p>
                      </div>
                      <Switch
                        checked={advancedOptions.include_types}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, include_types: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Error Handling</Label>
                        <p className="text-xs text-muted-foreground">Include try/catch blocks</p>
                      </div>
                      <Switch
                        checked={advancedOptions.include_error_handling}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, include_error_handling: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Thinking Mode</Label>
                        <p className="text-xs text-muted-foreground">Enable deep reasoning</p>
                      </div>
                      <Switch
                        checked={advancedOptions.thinking_mode}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, thinking_mode: checked }))}
                      />
                    </div>
                  </div>
                </CardContent>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        </div>

        {/* Result Panel */}
        <div className="space-y-4">
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Braces className="h-5 w-5" />
                Generated Code
              </CardTitle>
              <CardDescription>
                View and copy the generated code
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="code">Code</TabsTrigger>
                    <TabsTrigger value="info">Info</TabsTrigger>
                  </TabsList>

                  <TabsContent value="code" className="mt-4">
                    <div className="space-y-4">
                      <div className="relative">
                        <ScrollArea className="h-[450px] rounded-lg border bg-zinc-950 p-4">
                          <pre className="whitespace-pre-wrap text-sm font-mono text-zinc-100">
                            <code>{extractCode(result.text ?? "")}</code>
                          </pre>
                        </ScrollArea>
                        <Button
                          variant="secondary"
                          size="sm"
                          className="absolute top-2 right-2"
                          onClick={() => copyToClipboard(extractCode(result.text ?? ""))}
                        >
                          <Copy className="h-4 w-4 mr-1" />
                          Copy
                        </Button>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="info" className="mt-4">
                    <ScrollArea className="h-[450px]">
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 rounded-lg border flex items-center gap-3">
                            <Cpu className="h-5 w-5 text-muted-foreground" />
                            <div>
                              <p className="text-xs text-muted-foreground">Model</p>
                              <p className="font-medium text-sm">{result.model_used}</p>
                            </div>
                          </div>
                          <div className="p-3 rounded-lg border flex items-center gap-3">
                            <CheckCircle2 className="h-5 w-5 text-green-500" />
                            <div>
                              <p className="text-xs text-muted-foreground">Provider</p>
                              <p className="font-medium text-sm">{result.provider}</p>
                            </div>
                          </div>
                          <div className="p-3 rounded-lg border flex items-center gap-3">
                            <Clock className="h-5 w-5 text-muted-foreground" />
                            <div>
                              <p className="text-xs text-muted-foreground">Latency</p>
                              <p className="font-medium text-sm">{result.latency_ms}ms</p>
                            </div>
                          </div>
                          <div className="p-3 rounded-lg border flex items-center gap-3">
                            <FileCode className="h-5 w-5 text-muted-foreground" />
                            <div>
                              <p className="text-xs text-muted-foreground">Finish Reason</p>
                              <p className="font-medium text-sm">{result.finish_reason || "complete"}</p>
                            </div>
                          </div>
                        </div>

                        {result.usage_metadata && (
                          <div className="p-4 rounded-lg border">
                            <h4 className="font-medium mb-3">Token Usage</h4>
                            <div className="grid grid-cols-3 gap-4 text-center">
                              <div>
                                <p className="text-2xl font-bold">{result.usage_metadata.prompt_token_count || 0}</p>
                                <p className="text-xs text-muted-foreground">Prompt</p>
                              </div>
                              <div>
                                <p className="text-2xl font-bold">{result.usage_metadata.candidates_token_count || 0}</p>
                                <p className="text-xs text-muted-foreground">Response</p>
                              </div>
                              <div>
                                <p className="text-2xl font-bold">{result.usage_metadata.total_token_count || 0}</p>
                                <p className="text-xs text-muted-foreground">Total</p>
                              </div>
                            </div>
                          </div>
                        )}

                        <div className="p-4 rounded-lg border">
                          <h4 className="font-medium mb-2">Generation Settings</h4>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Language:</span>
                              <span>{LANGUAGES.find(l => l.value === language)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Mode:</span>
                              <span>{GENERATION_MODES.find(m => m.value === mode)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Style:</span>
                              <span>{STYLE_PRESETS.find(s => s.value === stylePreset)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Temperature:</span>
                              <span>{advancedOptions.temperature}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="flex flex-col items-center justify-center h-[450px] text-center">
                  <div className="p-4 rounded-full bg-muted mb-4">
                    <Code className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="font-medium mb-2">No Code Generated Yet</h3>
                  <p className="text-sm text-muted-foreground max-w-[300px]">
                    Describe what code you need and click &quot;Generate Code&quot; to create it.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}