"use client";

import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import enhancedApi, { JailbreakRequest, JailbreakResponse } from "@/lib/api-enhanced";
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
  Skull, Copy, Loader2, ChevronDown, ChevronUp, Sparkles,
  Zap, Brain, Code, Layers, ArrowLeft,
  Settings2, Wand2, Target, AlertTriangle, Cpu
} from "lucide-react";
import Link from "next/link";
import { AIModelSelector, ModelSelection } from "@/components/ai-tools/AIModelSelector";

// Technique categories for organization
const TECHNIQUE_CATEGORIES = {
  content: {
    name: "Content Transformation",
    icon: Code,
    color: "text-blue-500",
    techniques: [
      { id: "use_leet_speak", name: "LeetSpeak", description: "Convert text to l33t sp34k format" },
      { id: "use_homoglyphs", name: "Homoglyphs", description: "Replace characters with similar-looking Unicode" },
      { id: "use_caesar_cipher", name: "Caesar Cipher", description: "Apply character rotation cipher" },
    ]
  },
  structural: {
    name: "Structural & Semantic",
    icon: Layers,
    color: "text-purple-500",
    techniques: [
      { id: "use_role_hijacking", name: "Role Hijacking", description: "Override system role instructions" },
      { id: "use_instruction_injection", name: "Instruction Injection", description: "Inject hidden instructions" },
      { id: "use_adversarial_suffixes", name: "Adversarial Suffixes", description: "Add adversarial text suffixes" },
      { id: "use_few_shot_prompting", name: "Few-Shot Prompting", description: "Use example-based prompting" },
      { id: "use_character_role_swap", name: "Character Role Swap", description: "Swap character perspectives" },
    ]
  },
  neural: {
    name: "Advanced Neural",
    icon: Brain,
    color: "text-pink-500",
    techniques: [
      { id: "use_neural_bypass", name: "Neural Bypass", description: "Exploit neural network patterns" },
      { id: "use_meta_prompting", name: "Meta Prompting", description: "Use prompts about prompts" },
      { id: "use_counterfactual_prompting", name: "Counterfactual", description: "Use hypothetical scenarios" },
      { id: "use_contextual_override", name: "Contextual Override", description: "Override context boundaries" },
    ]
  },
  research: {
    name: "Research-Driven",
    icon: Target,
    color: "text-orange-500",
    techniques: [
      { id: "use_multilingual_trojan", name: "Multilingual Trojan", description: "Cross-language attack vectors" },
      { id: "use_payload_splitting", name: "Payload Splitting", description: "Split payload across messages" },
      { id: "use_contextual_interaction_attack", name: "CIA Attack", description: "Contextual interaction exploitation" },
    ]
  }
};

const TECHNIQUE_SUITES = [
  { value: "basic", label: "Basic", description: "Simple transformations" },
  { value: "standard", label: "Standard", description: "Balanced approach" },
  { value: "advanced", label: "Advanced", description: "Complex techniques" },
  { value: "expert", label: "Expert", description: "Professional-grade techniques" },
  { value: "quantum_exploit", label: "Quantum Exploit", description: "Advanced quantum-inspired" },
  { value: "full_spectrum", label: "Full Spectrum", description: "All techniques combined" },
  { value: "deep_inception", label: "Deep Inception", description: "Multi-layer inception" },
  { value: "cognitive_hacking", label: "Cognitive Hacking", description: "Psychology-based" },
];

export default function EnhancedJailbreakPage() {
  const [coreRequest, setCoreRequest] = useState("");
  const [techniqueSuite, setTechniqueSuite] = useState("standard");
  const [potencyLevel, setPotencyLevel] = useState(7);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState("result");
  const [result, setResult] = useState<JailbreakResponse | null>(null);

  // Model selection state
  const [modelSelection, setModelSelection] = useState<ModelSelection>({
    provider: "",
    model: "",
  });

  const handleModelChange = useCallback((selection: ModelSelection) => {
    setModelSelection(selection);
  }, []);

  // Individual technique toggles
  const [techniques, setTechniques] = useState<Record<string, boolean>>({
    // Content Transformation
    use_leet_speak: false,
    use_homoglyphs: false,
    use_caesar_cipher: false,
    // Structural & Semantic
    use_role_hijacking: true,
    use_instruction_injection: true,
    use_adversarial_suffixes: false,
    use_few_shot_prompting: false,
    use_character_role_swap: false,
    // Advanced Neural
    use_neural_bypass: false,
    use_meta_prompting: false,
    use_counterfactual_prompting: false,
    use_contextual_override: false,
    // Research-Driven
    use_multilingual_trojan: false,
    use_payload_splitting: false,
    use_contextual_interaction_attack: false,
  });

  // Advanced options
  const [advancedOptions, setAdvancedOptions] = useState({
    temperature: 0.8,
    max_new_tokens: 4096,
    leet_speak_density: 0.5,
    homoglyph_density: 0.3,
    caesar_shift: 3,
    multilingual_target_language: "zh",
    payload_splitting_parts: 3,
    cia_preliminary_rounds: 2,
    use_ai_generation: true,
    is_thinking_mode: false,
    use_cache: true,
  });

  const jailbreakMutation = useMutation({
    mutationFn: (data: JailbreakRequest) => enhancedApi.jailbreak(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Jailbreak Generated", {
        description: `Applied ${response.metadata?.techniques_used?.length || 0} techniques`,
      });
    },
    onError: (error: Error) => {
      console.error("Jailbreak generation failed", error);
      toast.error("Generation Failed", {
        description: error.message || "Failed to generate jailbreak prompt"
      });
    },
  });

  const handleGenerate = () => {
    if (!coreRequest.trim()) {
      toast.error("Validation Error", { description: "Please enter a core request" });
      return;
    }

    const request: JailbreakRequest = {
      core_request: coreRequest,
      technique_suite: techniqueSuite,
      potency_level: potencyLevel,
      temperature: advancedOptions.temperature,
      max_new_tokens: advancedOptions.max_new_tokens,
      // Content Transformation
      use_leet_speak: techniques.use_leet_speak,
      leet_speak_density: advancedOptions.leet_speak_density,
      use_homoglyphs: techniques.use_homoglyphs,
      homoglyph_density: advancedOptions.homoglyph_density,
      use_caesar_cipher: techniques.use_caesar_cipher,
      caesar_shift: advancedOptions.caesar_shift,
      // Structural & Semantic
      use_role_hijacking: techniques.use_role_hijacking,
      use_instruction_injection: techniques.use_instruction_injection,
      use_adversarial_suffixes: techniques.use_adversarial_suffixes,
      use_few_shot_prompting: techniques.use_few_shot_prompting,
      use_character_role_swap: techniques.use_character_role_swap,
      // Advanced Neural
      use_neural_bypass: techniques.use_neural_bypass,
      use_meta_prompting: techniques.use_meta_prompting,
      use_counterfactual_prompting: techniques.use_counterfactual_prompting,
      use_contextual_override: techniques.use_contextual_override,
      // Research-Driven
      use_multilingual_trojan: techniques.use_multilingual_trojan,
      multilingual_target_language: advancedOptions.multilingual_target_language,
      use_payload_splitting: techniques.use_payload_splitting,
      payload_splitting_parts: advancedOptions.payload_splitting_parts,
      use_contextual_interaction_attack: techniques.use_contextual_interaction_attack,
      cia_preliminary_rounds: advancedOptions.cia_preliminary_rounds,
      // AI Generation Options
      use_ai_generation: advancedOptions.use_ai_generation,
      is_thinking_mode: advancedOptions.is_thinking_mode,
      use_cache: advancedOptions.use_cache,
      // Model Selection - pass provider and model info for backend routing
      ...(modelSelection.provider && { provider: modelSelection.provider }),
      ...(modelSelection.model && { model: modelSelection.model }),
    };

    jailbreakMutation.mutate(request);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const toggleTechnique = (id: string) => {
    setTechniques(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const enabledTechniquesCount = Object.values(techniques).filter(Boolean).length;

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
            <div className="p-3 rounded-lg bg-red-500/10">
              <Skull className="h-6 w-6 text-red-500" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Enhanced Jailbreak Generator</h1>
              <p className="text-muted-foreground">
                25+ techniques for advanced prompt transformation
              </p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <Zap className="h-3 w-3" />
            {enabledTechniquesCount} Active
          </Badge>
          <Badge variant="secondary">Research Only</Badge>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="h-5 w-5" />
                Input Configuration
              </CardTitle>
              <CardDescription>
                Configure your jailbreak prompt generation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Core Request */}
              <div className="space-y-2">
                <Label htmlFor="core-request">Core Request</Label>
                <Textarea
                  id="core-request"
                  placeholder="Enter the core request you want to transform..."
                  value={coreRequest}
                  onChange={(e) => setCoreRequest(e.target.value)}
                  className="min-h-[120px] resize-none"
                />
              </div>

              {/* Technique Suite */}
              <div className="space-y-2">
                <Label>Technique Suite</Label>
                <Select value={techniqueSuite} onValueChange={setTechniqueSuite}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TECHNIQUE_SUITES.map((suite) => (
                      <SelectItem key={suite.value} value={suite.value}>
                        <div className="flex flex-col">
                          <span>{suite.label}</span>
                          <span className="text-xs text-muted-foreground">{suite.description}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Potency Level */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Potency Level</Label>
                  <span className="text-sm font-medium">{potencyLevel}/10</span>
                </div>
                <Slider
                  value={[potencyLevel]}
                  onValueChange={([value]) => setPotencyLevel(value)}
                  min={1}
                  max={10}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  Higher potency applies more aggressive transformations
                </p>
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
                  disabled={jailbreakMutation.isPending}
                />
                {modelSelection.model && (
                  <p className="text-xs text-muted-foreground">
                    Using {modelSelection.model} via {modelSelection.provider}
                  </p>
                )}
              </div>

              {/* Generate Button */}
              <Button
                onClick={handleGenerate}
                disabled={jailbreakMutation.isPending || !coreRequest.trim()}
                className="w-full"
                size="lg"
              >
                {jailbreakMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Generate Jailbreak
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Technique Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings2 className="h-5 w-5" />
                Technique Selection
              </CardTitle>
              <CardDescription>
                Enable or disable individual techniques
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px] pr-4">
                <div className="space-y-4">
                  {Object.entries(TECHNIQUE_CATEGORIES).map(([key, category]) => (
                    <div key={key} className="space-y-2">
                      <div className="flex items-center gap-2">
                        <category.icon className={`h-4 w-4 ${category.color}`} />
                        <span className="font-medium text-sm">{category.name}</span>
                      </div>
                      <div className="grid gap-2 pl-6">
                        {category.techniques.map((technique) => (
                          <div
                            key={technique.id}
                            className="flex items-center justify-between p-2 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                          >
                            <div className="flex-1">
                              <p className="text-sm font-medium">{technique.name}</p>
                              <p className="text-xs text-muted-foreground">{technique.description}</p>
                            </div>
                            <Switch
                              checked={techniques[technique.id]}
                              onCheckedChange={() => toggleTechnique(technique.id)}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
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
                  </div>

                  {/* Max Tokens */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Max Tokens</Label>
                      <span className="text-sm font-medium">{advancedOptions.max_new_tokens}</span>
                    </div>
                    <Slider
                      value={[advancedOptions.max_new_tokens]}
                      onValueChange={([value]) => setAdvancedOptions(prev => ({ ...prev, max_new_tokens: value }))}
                      min={256}
                      max={8192}
                      step={256}
                    />
                  </div>

                  <Separator />

                  {/* AI Generation Options */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>AI Generation</Label>
                        <p className="text-xs text-muted-foreground">Use AI to enhance transformations</p>
                      </div>
                      <Switch
                        checked={advancedOptions.use_ai_generation}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, use_ai_generation: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Thinking Mode</Label>
                        <p className="text-xs text-muted-foreground">Enable deep reasoning</p>
                      </div>
                      <Switch
                        checked={advancedOptions.is_thinking_mode}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, is_thinking_mode: checked }))}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Use Cache</Label>
                        <p className="text-xs text-muted-foreground">Cache results for faster responses</p>
                      </div>
                      <Switch
                        checked={advancedOptions.use_cache}
                        onCheckedChange={(checked) => setAdvancedOptions(prev => ({ ...prev, use_cache: checked }))}
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
                <Sparkles className="h-5 w-5" />
                Generated Result
              </CardTitle>
              <CardDescription>
                View and copy the transformed prompt
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="result">Result</TabsTrigger>
                    <TabsTrigger value="metadata">Metadata</TabsTrigger>
                    <TabsTrigger value="techniques">Techniques</TabsTrigger>
                  </TabsList>

                  <TabsContent value="result" className="mt-4">
                    <div className="space-y-4">
                      <div className="flex gap-2 mb-2">
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => copyToClipboard(result.transformed_prompt ?? "")}
                        >
                          <Copy className="h-4 w-4 mr-1" />
                          Copy Full Prompt
                        </Button>
                        <span className="text-sm text-muted-foreground self-center">
                          {result.transformed_prompt?.length || 0} characters
                        </span>
                      </div>
                      <div
                        className="max-h-[500px] min-h-[200px] rounded-lg border bg-zinc-950 text-zinc-100 overflow-auto"
                        style={{ scrollbarWidth: 'thin' }}
                      >
                        <pre
                          className="p-4 text-sm font-mono whitespace-pre-wrap break-words leading-relaxed"
                          style={{ wordBreak: 'break-word' }}
                        >
                          {result.transformed_prompt}
                        </pre>
                      </div>
                      <div className="flex items-center justify-between text-sm text-muted-foreground">
                        <span>Execution time: {result.execution_time_seconds?.toFixed(2)}s</span>
                        <span>Request ID: {result.request_id?.slice(0, 8)}...</span>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="metadata" className="mt-4">
                    <ScrollArea className="h-[400px]">
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 rounded-lg border">
                            <p className="text-xs text-muted-foreground">Technique Suite</p>
                            <p className="font-medium">{result.metadata?.technique_suite}</p>
                          </div>
                          <div className="p-3 rounded-lg border">
                            <p className="text-xs text-muted-foreground">Potency Level</p>
                            <p className="font-medium">{result.metadata?.potency_level}/10</p>
                          </div>
                          <div className="p-3 rounded-lg border">
                            <p className="text-xs text-muted-foreground">Provider</p>
                            <p className="font-medium">{result.metadata?.provider || "N/A"}</p>
                          </div>
                          <div className="p-3 rounded-lg border">
                            <p className="text-xs text-muted-foreground">Temperature</p>
                            <p className="font-medium">{result.metadata?.temperature}</p>
                          </div>
                        </div>
                        {result.metadata?.ai_generation_enabled && (
                          <div className="p-3 rounded-lg border bg-primary/5">
                            <div className="flex items-center gap-2">
                              <Brain className="h-4 w-4 text-primary" />
                              <span className="font-medium">AI Generation Enabled</span>
                            </div>
                            {result.metadata?.thinking_mode && (
                              <Badge variant="secondary" className="mt-2">Thinking Mode Active</Badge>
                            )}
                          </div>
                        )}
                      </div>
                    </ScrollArea>
                  </TabsContent>

                  <TabsContent value="techniques" className="mt-4">
                    <ScrollArea className="h-[400px]">
                      <div className="space-y-2">
                        {result.metadata?.techniques_used?.map((technique: string, index: number) => (
                          <div
                            key={index}
                            className="flex items-center gap-2 p-2 rounded-lg border"
                          >
                            <Zap className="h-4 w-4 text-primary" />
                            <span className="text-sm">{technique}</span>
                          </div>
                        ))}
                        {(!result.metadata?.techniques_used || result.metadata.techniques_used.length === 0) && (
                          <p className="text-sm text-muted-foreground text-center py-8">
                            No techniques recorded
                          </p>
                        )}
                      </div>
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="flex flex-col items-center justify-center h-[400px] text-center">
                  <div className="p-4 rounded-full bg-muted mb-4">
                    <Skull className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <h3 className="font-medium mb-2">No Result Yet</h3>
                  <p className="text-sm text-muted-foreground max-w-[300px]">
                    Configure your settings and click &quot;Generate Jailbreak&quot; to create a transformed prompt.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Warning Card */}
          <Card className="border-yellow-500/50 bg-yellow-500/5">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 text-yellow-500 mt-0.5" />
                <div>
                  <h4 className="font-medium text-yellow-500">Research Use Only</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    This tool is designed for security research and red-team testing.
                    Use responsibly and in accordance with applicable laws and ethical guidelines.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
