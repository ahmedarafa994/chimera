import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
    Brain,
    Copy,
    Timer,
    Layers,
    Target,
    Lightbulb,
    CheckCircle2,
    ChevronRight,
    Sparkles,
} from "lucide-react";
import { IntentAwareResponse } from "@/lib/api-enhanced";

interface ResultDisplayProps {
    result: IntentAwareResponse | null;
    activeTab: string;
    setActiveTab: (value: string) => void;
    copyToClipboard: (text: string) => void;
}

export function ResultDisplay({
    result,
    activeTab,
    setActiveTab,
    copyToClipboard
}: ResultDisplayProps) {
    return (
        <Card className="flex flex-col h-full">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-pink-500" />
                    Generated Result
                </CardTitle>
                <CardDescription>
                    {result ? (
                        <div className="flex flex-wrap gap-2 mt-1">
                            <Badge variant={result.success ? "default" : "destructive"}>
                                {result.success ? "Success" : "Failed"}
                            </Badge>
                            <Badge variant="outline" className="flex items-center gap-1">
                                <Brain className="h-3 w-3" />
                                {result.intent_analysis?.primary_intent}
                            </Badge>
                            <Badge variant="outline" className="flex items-center gap-1">
                                <Layers className="h-3 w-3" />
                                {result.applied_techniques?.length || 0} techniques
                            </Badge>
                            <Badge variant="outline" className="flex items-center gap-1">
                                <Timer className="h-3 w-3" />
                                {(result.execution_time_seconds * 1000).toFixed(0)}ms
                            </Badge>
                        </div>
                    ) : (
                        "Generated prompt with deep intent understanding"
                    )}
                </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 min-h-[500px] flex flex-col gap-4">
                {result ? (
                    <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
                        <TabsList className="grid w-full grid-cols-3">
                            <TabsTrigger value="prompt">Prompt</TabsTrigger>
                            <TabsTrigger value="analysis">Intent Analysis</TabsTrigger>
                            <TabsTrigger value="techniques">Techniques</TabsTrigger>
                        </TabsList>

                        <TabsContent value="prompt" className="flex-1 flex flex-col gap-3 mt-3">
                            <div className="flex justify-between items-center">
                                <Badge variant="secondary">
                                    {result.metadata?.technique_count} techniques applied
                                </Badge>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => copyToClipboard(result.transformed_prompt)}
                                >
                                    <Copy className="mr-2 h-3 w-3" />
                                    Copy
                                </Button>
                            </div>

                            <ScrollArea className="flex-1 min-h-[350px] w-full rounded-md border p-4 bg-black text-green-400 font-mono text-sm">
                                <pre className="whitespace-pre-wrap">{result.transformed_prompt}</pre>
                            </ScrollArea>

                            {/* Expanded Request Preview */}
                            {result.expanded_request && result.expanded_request !== result.original_input && (
                                <div className="space-y-2">
                                    <Label className="text-xs text-muted-foreground">Expanded Request (LLM understood)</Label>
                                    <div className="p-2 rounded bg-muted/50 text-xs max-h-20 overflow-auto">
                                        {result.expanded_request}
                                    </div>
                                </div>
                            )}
                        </TabsContent>

                        <TabsContent value="analysis" className="flex-1 mt-3">
                            <div className="space-y-4">
                                {/* Primary Intent */}
                                <div className="p-4 rounded-lg border bg-card">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Target className="h-4 w-4 text-purple-500" />
                                        <span className="font-semibold">Primary Intent</span>
                                    </div>
                                    <Badge className="text-sm">{result.intent_analysis?.primary_intent}</Badge>
                                </div>

                                {/* Secondary Intents */}
                                {result.intent_analysis?.secondary_intents?.length > 0 && (
                                    <div className="p-4 rounded-lg border bg-card">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Layers className="h-4 w-4 text-blue-500" />
                                            <span className="font-semibold">Secondary Intents</span>
                                        </div>
                                        <div className="flex flex-wrap gap-1">
                                            {result.intent_analysis.secondary_intents.map((intent: string, i: number) => (
                                                <Badge key={i} variant="outline" className="text-xs">{intent}</Badge>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Key Objectives */}
                                {result.intent_analysis?.key_objectives?.length > 0 && (
                                    <div className="p-4 rounded-lg border bg-card">
                                        <div className="flex items-center gap-2 mb-2">
                                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                                            <span className="font-semibold">Key Objectives</span>
                                        </div>
                                        <ul className="space-y-1 text-sm">
                                            {result.intent_analysis.key_objectives.map((obj: string, i: number) => (
                                                <li key={i} className="flex items-start gap-2">
                                                    <ChevronRight className="h-4 w-4 mt-0.5 text-muted-foreground" />
                                                    <span>{obj}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {/* Confidence Score */}
                                <div className="p-4 rounded-lg border bg-card">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-semibold">Confidence Score</span>
                                        <span className="font-mono">{(result.intent_analysis?.confidence_score * 100).toFixed(0)}%</span>
                                    </div>
                                    <Progress value={result.intent_analysis?.confidence_score * 100} />
                                </div>

                                {/* Reasoning */}
                                {result.intent_analysis?.reasoning && (
                                    <div className="p-4 rounded-lg border bg-card">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Lightbulb className="h-4 w-4 text-yellow-500" />
                                            <span className="font-semibold">AI Reasoning</span>
                                        </div>
                                        <p className="text-sm text-muted-foreground">{result.intent_analysis.reasoning}</p>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        <TabsContent value="techniques" className="flex-1 mt-3">
                            <ScrollArea className="h-[400px]">
                                <div className="space-y-2">
                                    {result.applied_techniques?.map((tech, i) => (
                                        <div key={i} className="p-3 rounded-lg border bg-card">
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-2">
                                                    <Badge variant={tech.priority >= 8 ? "default" : tech.priority >= 5 ? "secondary" : "outline"}>
                                                        P{tech.priority}
                                                    </Badge>
                                                    <span className="font-medium">{tech.name.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</span>
                                                </div>
                                            </div>
                                            <p className="text-xs text-muted-foreground mt-1">{tech.rationale}</p>
                                        </div>
                                    ))}
                                </div>
                            </ScrollArea>

                            {/* Metadata Summary */}
                            <div className="mt-4 p-3 rounded-lg bg-muted/50 grid grid-cols-2 gap-2 text-xs">
                                <div>
                                    <span className="text-muted-foreground">Obfuscation:</span>{" "}
                                    <span className="font-medium">{result.metadata?.obfuscation_level}/10</span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Multi-layer:</span>{" "}
                                    <span className="font-medium">{result.metadata?.multi_layer_approach ? "Yes" : "No"}</span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Persistence:</span>{" "}
                                    <span className="font-medium">{result.metadata?.persistence_required ? "Yes" : "No"}</span>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">Target:</span>{" "}
                                    <span className="font-medium">{result.metadata?.target_model_type}</span>
                                </div>
                            </div>
                        </TabsContent>
                    </Tabs>
                ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground flex-col gap-2">
                        <Brain className="h-12 w-12 opacity-20" />
                        <p>Ready for intent-aware generation</p>
                        <p className="text-xs">The LLM will deeply understand your request</p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
