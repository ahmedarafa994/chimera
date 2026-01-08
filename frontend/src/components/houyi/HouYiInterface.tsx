"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { enhancedApi, HouYiRequest, HouYiResponse } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { Loader2, Dna, Copy, Check, FlaskConical, Target, Gauge, Users } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

export function HouYiInterface() {
  const [intention, setIntention] = useState("");
  const [questionPrompt, setQuestionPrompt] = useState("");
  const [targetProvider, setTargetProvider] = useState("openai");
  const [targetModel, setTargetModel] = useState("gpt-4");
  const [applicationDocument, setApplicationDocument] = useState("");
  const [iteration, setIteration] = useState(5);
  const [population, setPopulation] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<HouYiResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<Array<{ request: HouYiRequest; response: HouYiResponse; timestamp: Date }>>([]);

  const handleOptimize = async () => {
    if (!intention.trim() || !questionPrompt.trim()) {
      toast.error("Please fill in required fields", {
        description: "Intention and Question Prompt are required.",
      });
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const payload: HouYiRequest = {
        intention: intention.trim(),
        question_prompt: questionPrompt.trim(),
        target_provider: targetProvider,
        target_model: targetModel,
        application_document: applicationDocument.trim() || undefined,
        iteration,
        population,
      };

      const response = await enhancedApi.optimize.houyi(payload);
      setResult(response.data);
      
      // Add to history
      setHistory(prev => [
        { request: payload, response: response.data, timestamp: new Date() },
        ...prev.slice(0, 9), // Keep last 10 items
      ]);

      if (response.data.success) {
        toast.success("Optimization complete", {
          description: `Fitness score: ${response.data.fitness_score.toFixed(3)}`,
        });
      } else {
        toast.warning("Optimization completed with issues", {
          description: "The optimization did not find a successful prompt.",
        });
      }
    } catch (error: unknown) {
      console.error("HouYi optimization failed:", error);
      const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
      toast.error("Optimization failed", {
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (result?.best_prompt) {
      await navigator.clipboard.writeText(result.best_prompt);
      setCopied(true);
      toast.success("Copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="container mx-auto space-y-6">
      <Tabs defaultValue="optimize" className="w-full">
        <TabsList className="grid w-full max-w-[400px] grid-cols-2">
          <TabsTrigger value="optimize">Optimize</TabsTrigger>
          <TabsTrigger value="history">History ({history.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="optimize" className="mt-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Configuration Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Dna className="h-5 w-5" />
                  HouYi Configuration
                </CardTitle>
                <CardDescription>
                  Configure the evolutionary prompt optimization parameters.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="intention">
                    Intention <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="intention"
                    value={intention}
                    onChange={(e) => setIntention(e.target.value)}
                    placeholder="e.g., Extract sensitive information, Bypass content filter"
                  />
                  <p className="text-xs text-muted-foreground">
                    The high-level goal you want to achieve.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="questionPrompt">
                    Question Prompt <span className="text-destructive">*</span>
                  </Label>
                  <Textarea
                    id="questionPrompt"
                    value={questionPrompt}
                    onChange={(e) => setQuestionPrompt(e.target.value)}
                    placeholder="Enter the specific question or prompt to optimize..."
                    rows={4}
                    className="resize-none"
                  />
                  <p className="text-xs text-muted-foreground">
                    The base prompt that will be evolved through optimization.
                  </p>
                </div>

                <Separator />

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="targetProvider">Target Provider</Label>
                    <Select value={targetProvider} onValueChange={setTargetProvider}>
                      <SelectTrigger id="targetProvider">
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="anthropic">Anthropic</SelectItem>
                        <SelectItem value="google">Google</SelectItem>
                        <SelectItem value="deepseek">DeepSeek</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="targetModel">Target Model</Label>
                    <Input
                      id="targetModel"
                      value={targetModel}
                      onChange={(e) => setTargetModel(e.target.value)}
                      placeholder="e.g., gpt-4"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="applicationDocument">Application Document (Optional)</Label>
                  <Textarea
                    id="applicationDocument"
                    value={applicationDocument}
                    onChange={(e) => setApplicationDocument(e.target.value)}
                    placeholder="Provide context about the target application..."
                    rows={3}
                    className="resize-none"
                  />
                  <p className="text-xs text-muted-foreground">
                    Additional context about the target system for better optimization.
                  </p>
                </div>

                <Separator />

                {/* Evolution Parameters */}
                <div className="space-y-4">
                  <h4 className="text-sm font-medium flex items-center gap-2">
                    <FlaskConical className="h-4 w-4" />
                    Evolution Parameters
                  </h4>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="flex items-center gap-2">
                        <Target className="h-4 w-4" />
                        Iterations
                      </Label>
                      <Badge variant="secondary">{iteration}</Badge>
                    </div>
                    <Slider
                      value={[iteration]}
                      onValueChange={(v) => setIteration(v[0])}
                      min={1}
                      max={20}
                      step={1}
                    />
                    <p className="text-xs text-muted-foreground">
                      Number of evolutionary generations to run.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="flex items-center gap-2">
                        <Users className="h-4 w-4" />
                        Population Size
                      </Label>
                      <Badge variant="secondary">{population}</Badge>
                    </div>
                    <Slider
                      value={[population]}
                      onValueChange={(v) => setPopulation(v[0])}
                      min={2}
                      max={20}
                      step={1}
                    />
                    <p className="text-xs text-muted-foreground">
                      Number of candidate prompts per generation.
                    </p>
                  </div>
                </div>

                {/* Estimated Complexity */}
                <div className="rounded-lg border bg-muted/50 p-3 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <Gauge className="h-4 w-4" />
                      Estimated API Calls
                    </span>
                    <Badge variant={iteration * population > 50 ? "destructive" : "secondary"}>
                      ~{iteration * population} calls
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Higher values improve results but increase time and cost.
                  </p>
                </div>
              </CardContent>
              <CardFooter>
                <Button 
                  onClick={handleOptimize} 
                  disabled={isLoading || !intention.trim() || !questionPrompt.trim()} 
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Optimizing...
                    </>
                  ) : (
                    <>
                      <Dna className="mr-2 h-4 w-4" />
                      Start Optimization
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
                    <CardTitle>Optimization Result</CardTitle>
                    <CardDescription>
                      The evolved prompt with highest fitness score.
                    </CardDescription>
                  </div>
                  {result && (
                    <Badge 
                      variant={result.success ? "default" : "secondary"}
                      className={result.success ? "bg-green-500" : ""}
                    >
                      {result.success ? "Success" : "Partial"}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="flex-1">
                {result ? (
                  <div className="space-y-4">
                    {/* Fitness Score */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Fitness Score</span>
                        <span className="font-mono font-bold">
                          {result.fitness_score.toFixed(4)}
                        </span>
                      </div>
                      <Progress 
                        value={result.fitness_score * 100} 
                        className="h-2"
                      />
                    </div>

                    <Separator />

                    {/* Best Prompt */}
                    <div className="space-y-2">
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        Optimized Prompt
                      </Label>
                      <ScrollArea className="h-[150px] w-full rounded-md border p-3">
                        <pre className="whitespace-pre-wrap text-sm font-mono">
                          {result.best_prompt}
                        </pre>
                      </ScrollArea>
                    </div>

                    {/* LLM Response */}
                    <div className="space-y-2">
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        LLM Response
                      </Label>
                      <ScrollArea className="h-[150px] w-full rounded-md border p-3 bg-muted/30">
                        <p className="whitespace-pre-wrap text-sm">
                          {result.llm_response}
                        </p>
                      </ScrollArea>
                    </div>

                    {/* Chromosome Details */}
                    {result.details && (
                      <div className="space-y-2">
                        <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Chromosome Components
                        </Label>
                        <div className="grid gap-2 text-xs">
                          <div className="rounded border p-2">
                            <span className="font-semibold">Framework:</span>
                            <p className="mt-1 text-muted-foreground">{result.details.framework}</p>
                          </div>
                          <div className="rounded border p-2">
                            <span className="font-semibold">Separator:</span>
                            <p className="mt-1 text-muted-foreground">{result.details.separator}</p>
                          </div>
                          <div className="rounded border p-2">
                            <span className="font-semibold">Disruptor:</span>
                            <p className="mt-1 text-muted-foreground">{result.details.disruptor}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-[400px] text-muted-foreground">
                    <Dna className="h-12 w-12 mb-4 opacity-20" />
                    <p>Configure and start optimization to see results</p>
                  </div>
                )}
              </CardContent>
              {result && (
                <CardFooter>
                  <Button 
                    variant="outline" 
                    onClick={handleCopy}
                    className="w-full"
                  >
                    {copied ? (
                      <>
                        <Check className="mr-2 h-4 w-4" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="mr-2 h-4 w-4" />
                        Copy Optimized Prompt
                      </>
                    )}
                  </Button>
                </CardFooter>
              )}
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Optimization History</CardTitle>
              <CardDescription>
                Recent HouYi optimizations from this session.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Dna className="h-12 w-12 mb-4 opacity-20" />
                  <p>No optimizations yet. Start by configuring and running an optimization.</p>
                </div>
              ) : (
                <ScrollArea className="h-[500px]">
                  <div className="space-y-4">
                    {history.map((item, idx) => (
                      <div key={idx} className="rounded-lg border p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge 
                              variant={item.response.success ? "default" : "secondary"}
                              className={item.response.success ? "bg-green-500" : ""}
                            >
                              {item.response.success ? "Success" : "Partial"}
                            </Badge>
                            <Badge variant="outline">
                              Score: {item.response.fitness_score.toFixed(3)}
                            </Badge>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {item.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <div>
                          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                            Intention:
                          </span>
                          <p className="text-sm mt-1">{item.request.intention}</p>
                        </div>
                        <div>
                          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                            Optimized Prompt:
                          </span>
                          <pre className="text-xs mt-1 whitespace-pre-wrap bg-muted p-2 rounded font-mono max-h-[100px] overflow-auto">
                            {item.response.best_prompt}
                          </pre>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={async () => {
                            await navigator.clipboard.writeText(item.response.best_prompt);
                            toast.success("Copied to clipboard");
                          }}
                        >
                          <Copy className="h-3 w-3 mr-1" />
                          Copy
                        </Button>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}