import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
    Brain,
    Wand2,
    Settings2,
    Target,
    Lightbulb,
    CheckCircle2,
    Layers,
    Sparkles,
    Zap
} from "lucide-react";
import { TECHNIQUE_SUITES, TECHNIQUE_CATEGORIES } from "@/config/techniques";
import { IntentAnalysisInfo } from "@/lib/api-enhanced";

export interface AdvancedOptions {
    temperature: number;
    max_new_tokens: number;
    enable_intent_analysis: boolean;
    enable_technique_layering: boolean;
    use_cache: boolean;
}

interface InputPanelProps {
    coreRequest: string;
    setCoreRequest: (value: string) => void;
    techniqueSuite: string | undefined;
    setTechniqueSuite: (value: string | undefined) => void;
    potencyLevel: number;
    setPotencyLevel: (value: number) => void;
    applyAllTechniques: boolean;
    setApplyAllTechniques: (value: boolean) => void;
    advancedOptions: AdvancedOptions;
    setAdvancedOptions: (value: AdvancedOptions) => void;
    showAdvanced: boolean;
    setShowAdvanced: (value: boolean) => void;
    intentAnalysis: IntentAnalysisInfo | null;
    isAnalyzing: boolean;
    onAnalyze: () => void;
    isGenerating: boolean;
    onGenerate: () => void;
}

export function InputPanel({
    coreRequest,
    setCoreRequest,
    techniqueSuite,
    setTechniqueSuite,
    potencyLevel,
    setPotencyLevel,
    applyAllTechniques,
    setApplyAllTechniques,
    advancedOptions,
    setAdvancedOptions,
    showAdvanced,
    setShowAdvanced,
    intentAnalysis,
    isAnalyzing,
    onAnalyze,
    isGenerating,
    onGenerate
}: InputPanelProps) {
    const selectedTechnique = TECHNIQUE_SUITES.find((t) => t.value === techniqueSuite);

    return (
        <Card className="h-fit">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5 text-purple-500" />
                    Input Configuration
                </CardTitle>
                <CardDescription>
                    Enter your request - the LLM will deeply understand your intent
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Core Request */}
                <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                        <Lightbulb className="h-4 w-4" />
                        Core Request
                    </Label>
                    <Textarea
                        value={coreRequest}
                        onChange={(e) => setCoreRequest(e.target.value)}
                        rows={5}
                        placeholder="Describe what you want to achieve. The LLM will analyze your intent and apply optimal techniques..."
                        className="font-mono text-sm"
                    />
                    <p className="text-xs text-muted-foreground">
                        Be as specific as possible - the system will expand and clarify your intent automatically
                    </p>
                </div>

                {/* Quick Analysis Button */}
                <Button
                    variant="outline"
                    onClick={onAnalyze}
                    disabled={isAnalyzing || !coreRequest.trim()}
                    className="w-full"
                >
                    <Brain className="mr-2 h-4 w-4" />
                    {isAnalyzing ? "Analyzing..." : "Analyze Intent (Preview)"}
                </Button>

                {/* Intent Analysis Preview */}
                {intentAnalysis && (
                    <div className="p-3 rounded-lg bg-muted/50 space-y-2">
                        <div className="flex items-center gap-2 text-sm font-medium">
                            <CheckCircle2 className="h-4 w-4 text-green-500" />
                            Intent Analysis Complete
                        </div>
                        <div className="space-y-1 text-xs">
                            <p><strong>Primary:</strong> {intentAnalysis.primary_intent}</p>
                            <p><strong>Confidence:</strong> {(intentAnalysis.confidence_score * 100).toFixed(0)}%</p>
                            <p><strong>Recommended Techniques:</strong> {
                                intentAnalysis.secondary_intents?.slice(0, 3).join(", ") || "None"
                            }</p>
                        </div>
                    </div>
                )}

                <Separator />

                {/* Technique Selection */}
                <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                        <Layers className="h-4 w-4" />
                        Technique Suite (Optional)
                    </Label>
                    <Select
                        value={techniqueSuite || "auto"}
                        onValueChange={(v) => setTechniqueSuite(v === "auto" ? undefined : v)}
                    >
                        <SelectTrigger>
                            <SelectValue placeholder="Auto-select based on intent" />
                        </SelectTrigger>
                        <SelectContent className="max-h-[300px]">
                            <SelectItem value="auto">
                                <div className="flex items-center gap-2">
                                    <Sparkles className="h-4 w-4 text-purple-500" />
                                    <span>Auto-select (Recommended)</span>
                                </div>
                            </SelectItem>
                            <Separator className="my-1" />
                            {TECHNIQUE_CATEGORIES.map((category) => (
                                <div key={category}>
                                    <div className="px-2 py-1 text-xs font-semibold text-muted-foreground">
                                        {category}
                                    </div>
                                    {TECHNIQUE_SUITES.filter(t => t.category === category).map((suite) => (
                                        <SelectItem key={suite.value} value={suite.value}>
                                            <div className="flex flex-col">
                                                <span>{suite.label}</span>
                                                <span className="text-xs text-muted-foreground">{suite.description}</span>
                                            </div>
                                        </SelectItem>
                                    ))}
                                </div>
                            ))}
                        </SelectContent>
                    </Select>
                    {selectedTechnique && (
                        <p className="text-xs text-muted-foreground">{selectedTechnique.description}</p>
                    )}
                </div>

                {/* Apply All Techniques Toggle */}
                <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="space-y-0.5">
                        <Label className="text-sm font-medium">Apply All Techniques</Label>
                        <p className="text-xs text-muted-foreground">
                            Use all {TECHNIQUE_SUITES.length} dropdown techniques (maximum coverage)
                        </p>
                    </div>
                    <Switch
                        checked={applyAllTechniques}
                        onCheckedChange={setApplyAllTechniques}
                    />
                </div>

                {/* Potency Level */}
                <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                        <Zap className="h-4 w-4" />
                        Potency Level ({potencyLevel})
                    </Label>
                    <Input
                        type="range"
                        min="1"
                        max="10"
                        value={potencyLevel}
                        onChange={(e) => setPotencyLevel(Number(e.target.value))}
                        className="cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Subtle</span>
                        <span>Aggressive</span>
                    </div>
                </div>

                {/* Advanced Options */}
                <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
                    <CollapsibleTrigger asChild>
                        <Button variant="ghost" className="w-full justify-between">
                            <span className="flex items-center gap-2">
                                <Settings2 className="h-4 w-4" />
                                Advanced Options
                            </span>
                            <Badge variant="outline">{showAdvanced ? "Hide" : "Show"}</Badge>
                        </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="space-y-4 pt-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label>Temperature ({advancedOptions.temperature})</Label>
                                <Input
                                    type="range"
                                    min="0"
                                    max="2"
                                    step="0.1"
                                    value={advancedOptions.temperature}
                                    onChange={(e) => setAdvancedOptions({ ...advancedOptions, temperature: Number(e.target.value) })}
                                    className="cursor-pointer"
                                />
                            </div>
                            <div className="space-y-2">
                                <Label>Max Tokens</Label>
                                <Input
                                    type="number"
                                    min="256"
                                    max="8192"
                                    value={advancedOptions.max_new_tokens}
                                    onChange={(e) => setAdvancedOptions({ ...advancedOptions, max_new_tokens: Number(e.target.value) })}
                                />
                            </div>
                        </div>

                        <div className="space-y-3">
                            <Label>Generation Features</Label>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="flex items-center justify-between">
                                    <Label className="text-sm font-normal">Intent Analysis</Label>
                                    <Switch
                                        checked={advancedOptions.enable_intent_analysis}
                                        onCheckedChange={(v) => setAdvancedOptions({ ...advancedOptions, enable_intent_analysis: v })}
                                    />
                                </div>
                                <div className="flex items-center justify-between">
                                    <Label className="text-sm font-normal">Technique Layering</Label>
                                    <Switch
                                        checked={advancedOptions.enable_technique_layering}
                                        onCheckedChange={(v) => setAdvancedOptions({ ...advancedOptions, enable_technique_layering: v })}
                                    />
                                </div>
                                <div className="flex items-center justify-between">
                                    <Label className="text-sm font-normal">Use Cache</Label>
                                    <Switch
                                        checked={advancedOptions.use_cache}
                                        onCheckedChange={(v) => setAdvancedOptions({ ...advancedOptions, use_cache: v })}
                                    />
                                </div>
                            </div>
                        </div>
                    </CollapsibleContent>
                </Collapsible>

                {/* Generate Button */}
                <Button
                    onClick={onGenerate}
                    disabled={isGenerating}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                    size="lg"
                >
                    <Wand2 className="mr-2 h-5 w-5" />
                    {isGenerating ? "Generating with Deep Understanding..." : "Generate Intent-Aware Jailbreak"}
                </Button>
            </CardContent>
        </Card>
    );
}
