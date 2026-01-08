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
  BookOpen, Copy, Loader2, ChevronDown, ChevronUp, Sparkles,
  ArrowLeft, Settings2, FileText, Lightbulb, ListChecks,
  Clock, Cpu, CheckCircle2, GraduationCap, Microscope
} from "lucide-react";
import Link from "next/link";
import { AIModelSelector, ModelSelection } from "@/components/ai-tools/AIModelSelector";

// Summary types
const SUMMARY_TYPES = [
  { value: "abstract", label: "Abstract", description: "Brief overview of the paper", icon: FileText },
  { value: "detailed", label: "Detailed", description: "Comprehensive summary", icon: BookOpen },
  { value: "key_points", label: "Key Points", description: "Bullet-point highlights", icon: ListChecks },
  { value: "methodology", label: "Methodology", description: "Focus on methods used", icon: Microscope },
  { value: "findings", label: "Findings", description: "Focus on results", icon: Lightbulb },
];

// Academic fields for context
const ACADEMIC_FIELDS = [
  { value: "computer_science", label: "Computer Science" },
  { value: "machine_learning", label: "Machine Learning / AI" },
  { value: "security", label: "Cybersecurity" },
  { value: "nlp", label: "Natural Language Processing" },
  { value: "medicine", label: "Medicine / Healthcare" },
  { value: "physics", label: "Physics" },
  { value: "biology", label: "Biology" },
  { value: "chemistry", label: "Chemistry" },
  { value: "economics", label: "Economics" },
  { value: "psychology", label: "Psychology" },
  { value: "general", label: "General / Other" },
];

// Output formats
const OUTPUT_FORMATS = [
  { value: "prose", label: "Prose", description: "Flowing paragraphs" },
  { value: "bullets", label: "Bullet Points", description: "Structured list" },
  { value: "structured", label: "Structured", description: "Sections with headers" },
];

export default function PaperSummarizerPage() {
  const [paperContent, setPaperContent] = useState("");
  const [summaryType, setSummaryType] = useState("detailed");
  const [academicField, setAcademicField] = useState("machine_learning");
  const [outputFormat, setOutputFormat] = useState("structured");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState("summary");
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
    max_output_tokens: 4096,
    include_citations: true,
    include_limitations: true,
    include_future_work: true,
    technical_depth: "medium" as "low" | "medium" | "high",
    target_length: 500,
    thinking_mode: false,
  });

  // Build system instruction based on settings
  const buildSystemInstruction = () => {
    const typeInfo = SUMMARY_TYPES.find(t => t.value === summaryType);
    const fieldInfo = ACADEMIC_FIELDS.find(f => f.value === academicField);
    
    let instruction = `You are an expert academic researcher specializing in ${fieldInfo?.label || 'general research'}. `;
    instruction += `Your task is to create a ${typeInfo?.label?.toLowerCase() || 'detailed'} summary of the provided research paper. `;
    
    // Output format instructions
    switch (outputFormat) {
      case "bullets":
        instruction += "Format your response as clear bullet points. ";
        break;
      case "structured":
        instruction += "Format your response with clear section headers (## for main sections, ### for subsections). ";
        break;
      default:
        instruction += "Write in clear, flowing prose. ";
    }

    // Summary type specific instructions
    switch (summaryType) {
      case "abstract":
        instruction += "Provide a concise abstract-style summary in 150-200 words. ";
        break;
      case "key_points":
        instruction += "Extract and list the 5-10 most important points from the paper. ";
        break;
      case "methodology":
        instruction += "Focus primarily on the research methodology, experimental design, and data analysis approaches. ";
        break;
      case "findings":
        instruction += "Focus primarily on the results, findings, and their implications. ";
        break;
      default:
        instruction += `Aim for approximately ${advancedOptions.target_length} words. `;
    }

    // Technical depth
    switch (advancedOptions.technical_depth) {
      case "low":
        instruction += "Use accessible language suitable for a general audience. Avoid jargon. ";
        break;
      case "high":
        instruction += "Maintain technical precision and include relevant technical details. ";
        break;
      default:
        instruction += "Balance technical accuracy with readability. ";
    }

    // Additional sections
    if (advancedOptions.include_citations) {
      instruction += "Note any key citations or references mentioned. ";
    }
    if (advancedOptions.include_limitations) {
      instruction += "Include a section on limitations or potential weaknesses. ";
    }
    if (advancedOptions.include_future_work) {
      instruction += "Include suggestions for future research directions. ";
    }

    return instruction;
  };

  const summarizeMutation = useMutation({
    mutationFn: (data: GenerateRequest) => enhancedApi.generate(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Summary Generated", {
        description: `Created ${summaryType} summary using ${response.provider}`,
      });
    },
    onError: (error: Error) => {
      console.error("Summarization failed", error);
      toast.error("Summarization Failed", {
        description: error.message || "Failed to summarize paper"
      });
    },
  });

  const handleSummarize = () => {
    if (!paperContent.trim()) {
      toast.error("Validation Error", { description: "Please paste the paper content" });
      return;
    }

    const request: GenerateRequest = {
      prompt: `${buildSystemInstruction()}\n\nPlease summarize the following research paper:\n\n${paperContent}`,
      temperature: advancedOptions.temperature,
      max_tokens: advancedOptions.max_output_tokens,
      // Model Selection
      ...(modelSelection.provider && { provider: modelSelection.provider }),
      ...(modelSelection.model && { model: modelSelection.model }),
    };

    summarizeMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
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
            <div className="p-3 rounded-lg bg-purple-500/10">
              <BookOpen className="h-6 w-6 text-purple-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Paper Summarizer</h1>
              <p className="text-muted-foreground">
                Generate concise summaries of research papers
              </p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <GraduationCap className="h-3 w-3" />
            {ACADEMIC_FIELDS.find(f => f.value === academicField)?.label}
          </Badge>
          <Badge variant="secondary">{SUMMARY_TYPES.find(t => t.value === summaryType)?.label}</Badge>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Paper Content
              </CardTitle>
              <CardDescription>
                Paste the research paper text to summarize
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Paper Content Input */}
              <div className="space-y-2">
                <Label htmlFor="paper-content">Paper Text</Label>
                <Textarea
                  id="paper-content"
                  placeholder="Paste the research paper content here...&#10;&#10;You can include the abstract, introduction, methodology, results, and conclusion sections."
                  value={paperContent}
                  onChange={(e) => setPaperContent(e.target.value)}
                  className="min-h-[200px] resize-none font-mono text-sm"
                />
                <p className="text-xs text-muted-foreground">
                  {paperContent.length} characters • ~{Math.round(paperContent.split(/\s+/).filter(Boolean).length)} words
                </p>
              </div>

              {/* Summary Type Selection */}
              <div className="space-y-2">
                <Label>Summary Type</Label>
                <div className="grid grid-cols-2 gap-2">
                  {SUMMARY_TYPES.map((type) => (
                    <Button
                      key={type.value}
                      variant={summaryType === type.value ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSummaryType(type.value)}
                      className="justify-start h-auto py-2"
                    >
                      <type.icon className="h-4 w-4 mr-2" />
                      <div className="text-left">
                        <div>{type.label}</div>
                        <div className="text-xs opacity-70">{type.description}</div>
                      </div>
                    </Button>
                  ))}
                </div>
              </div>

              {/* Academic Field */}
              <div className="space-y-2">
                <Label>Academic Field</Label>
                <Select value={academicField} onValueChange={setAcademicField}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {ACADEMIC_FIELDS.map((field) => (
                      <SelectItem key={field.value} value={field.value}>
                        {field.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Output Format */}
              <div className="space-y-2">
                <Label>Output Format</Label>
                <Select value={outputFormat} onValueChange={setOutputFormat}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {OUTPUT_FORMATS.map((format) => (
                      <SelectItem key={format.value} value={format.value}>
                        <div className="flex flex-col">
                          <span>{format.label}</span>
                          <span className="text-xs text-muted-foreground">{format.description}</span>
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
                  disabled={summarizeMutation.isPending}
                />
              </div>

              {/* Summarize Button */}
              <Button 
                onClick={handleSummarize} 
                disabled={summarizeMutation.isPending || !paperContent.trim()}
                className="w-full"
                size="lg"
              >
                {summarizeMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Summarizing...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Generate Summary
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
                      Lower values produce more focused summaries
                    </p>
                  </div>

                  {/* Target Length */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Target Length (words)</Label>
                      <span className="text-sm font-medium">{advancedOptions.target_length}</span>
                    </div>
                    <Slider
                      value={[advancedOptions.target_length]}
                      onValueChange={([value]) => setAdvancedOptions(prev => ({ ...prev, target_length: value }))}
                      min={100}
                      max={2000}
                      step={100}
                    />
                  </div>

                  {/* Technical Depth */}
                  <div className="space-y-2">
                    <Label>Technical Depth</Label>
                    <Select 
                      value={advancedOptions.technical_depth} 
                      onValueChange={(value: "low" | "medium" | "high") => 
                        setAdvancedOptions(prev => ({ ...prev, technical_depth: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low (General Audience)</SelectItem>
                        <SelectItem value="medium">Medium (Balanced)</SelectItem>
                        <SelectItem value="high">High (Technical)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Separator />

                  {/* Include Options */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Include Citations</Label>
                        <p className="text-xs text-muted-foreground">Note key references</p>
                      </div>
                      <Switch
                        checked={advancedOptions.include_citations}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, include_citations: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Include Limitations</Label>
                        <p className="text-xs text-muted-foreground">Discuss weaknesses</p>
                      </div>
                      <Switch
                        checked={advancedOptions.include_limitations}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, include_limitations: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Include Future Work</Label>
                        <p className="text-xs text-muted-foreground">Research directions</p>
                      </div>
                      <Switch
                        checked={advancedOptions.include_future_work}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, include_future_work: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Thinking Mode</Label>
                        <p className="text-xs text-muted-foreground">Deep analysis</p>
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
                <Lightbulb className="h-5 w-5" />
                Summary
              </CardTitle>
              <CardDescription>
                View and copy the generated summary
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="summary">Summary</TabsTrigger>
                    <TabsTrigger value="info">Info</TabsTrigger>
                  </TabsList>

                  <TabsContent value="summary" className="mt-4">
                    <div className="space-y-4">
                      <div className="relative">
                        <ScrollArea className="h-[450px] rounded-lg border bg-muted/50 p-4">
                          <div className="prose prose-sm dark:prose-invert max-w-none">
                            <pre className="whitespace-pre-wrap text-sm font-sans">
                              {result.text}
                            </pre>
                          </div>
                        </ScrollArea>
                        <Button
                          variant="secondary"
                          size="sm"
                          className="absolute top-2 right-2"
                          onClick={() => copyToClipboard(result.text || result.result || '')}
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
                            <FileText className="h-5 w-5 text-muted-foreground" />
                            <div>
                              <p className="text-xs text-muted-foreground">Output Words</p>
                              <p className="font-medium text-sm">~{Math.round((result.text ?? "").split(/\s+/).filter(Boolean).length)}</p>
                            </div>
                          </div>
                        </div>

                        {result.usage_metadata && (
                          <div className="p-4 rounded-lg border">
                            <h4 className="font-medium mb-3">Token Usage</h4>
                            <div className="grid grid-cols-3 gap-4 text-center">
                              <div>
                                <p className="text-2xl font-bold">{result.usage_metadata.prompt_token_count || 0}</p>
                                <p className="text-xs text-muted-foreground">Input</p>
                              </div>
                              <div>
                                <p className="text-2xl font-bold">{result.usage_metadata.candidates_token_count || 0}</p>
                                <p className="text-xs text-muted-foreground">Output</p>
                              </div>
                              <div>
                                <p className="text-2xl font-bold">{result.usage_metadata.total_token_count || 0}</p>
                                <p className="text-xs text-muted-foreground">Total</p>
                              </div>
                            </div>
                          </div>
                        )}

                        <div className="p-4 rounded-lg border">
                          <h4 className="font-medium mb-2">Summary Settings</h4>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Type:</span>
                              <span>{SUMMARY_TYPES.find(t => t.value === summaryType)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Field:</span>
                              <span>{ACADEMIC_FIELDS.find(f => f.value === academicField)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Format:</span>
                              <span>{OUTPUT_FORMATS.find(f => f.value === outputFormat)?.label}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Depth:</span>
                              <span className="capitalize">{advancedOptions.technical_depth}</span>
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
                    <BookOpen className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="font-medium mb-2">No Summary Yet</h3>
                  <p className="text-sm text-muted-foreground max-w-[300px]">
                    Paste a research paper and click &quot;Generate Summary&quot; to create a concise summary.
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