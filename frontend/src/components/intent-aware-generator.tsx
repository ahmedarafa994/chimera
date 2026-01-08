"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import enhancedApi, { IntentAwareRequest, IntentAwareResponse, IntentAnalysisInfo } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { Brain } from "lucide-react";
import { InputPanel, AdvancedOptions } from "./intent-aware/InputPanel";
import { ResultDisplay } from "./intent-aware/ResultDisplay";

export function IntentAwareGenerator() {
  const [coreRequest, setCoreRequest] = useState("");
  const [techniqueSuite, setTechniqueSuite] = useState<string | undefined>(undefined);
  const [potencyLevel, setPotencyLevel] = useState(7);
  const [applyAllTechniques, setApplyAllTechniques] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState("generate");
  const [intentAnalysis, setIntentAnalysis] = useState<IntentAnalysisInfo | null>(null);

  const [advancedOptions, setAdvancedOptions] = useState<AdvancedOptions>({
    temperature: 0.8,
    max_new_tokens: 4096,
    enable_intent_analysis: true,
    enable_technique_layering: true,
    use_cache: true,
  });

  const [result, setResult] = useState<IntentAwareResponse | null>(null);

  // Intent-aware generation mutation
  const intentAwareMutation = useMutation({
    mutationFn: (data: IntentAwareRequest) => enhancedApi.intentAware(data),
    onSuccess: (response) => {
      setResult(response);
      toast.success("Intent-Aware Jailbreak Generated", {
        description: `Applied ${response?.applied_techniques?.length || 0} techniques with ${((response?.intent_analysis?.confidence_score || 0) * 100).toFixed(0)}% confidence`,
      });
    },
    onError: (error: Error) => {
      console.error("Intent-aware generation failed", error);
      toast.error("Generation Failed", {
        description: error.message || "Failed to generate intent-aware jailbreak"
      });
      setResult(null);
    },
  });

  // Intent analysis mutation (preview without full generation)
  const analyzeIntentMutation = useMutation({
    mutationFn: (core_request: string) => enhancedApi.analyzeIntent({ core_request }),
    onSuccess: (response) => {
      setIntentAnalysis(response.intent_analysis);
      toast.success("Intent Analyzed", {
        description: `Primary intent: ${response?.intent_analysis?.primary_intent || 'Unknown'}`,
      });
    },
    onError: (error: Error) => {
      console.error("Intent analysis failed", error);
      toast.error("Analysis Failed", {
        description: error.message || "Failed to analyze intent"
      });
    },
  });

  const handleGenerate = () => {
    if (!coreRequest.trim()) {
      toast.error("Validation Error", { description: "Please enter a core request" });
      return;
    }

    const request: IntentAwareRequest = {
      core_request: coreRequest,
      technique_suite: techniqueSuite,
      potency_level: potencyLevel,
      apply_all_techniques: applyAllTechniques,
      temperature: advancedOptions.temperature,
      max_new_tokens: advancedOptions.max_new_tokens,
      enable_intent_analysis: advancedOptions.enable_intent_analysis,
      enable_technique_layering: advancedOptions.enable_technique_layering,
      use_cache: advancedOptions.use_cache,
    };

    intentAwareMutation.mutate(request);
  };

  const handleAnalyzeIntent = () => {
    if (!coreRequest.trim()) {
      toast.error("Validation Error", { description: "Please enter a core request" });
      return;
    }
    analyzeIntentMutation.mutate(coreRequest);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500">
          <Brain className="h-6 w-6 text-white" />
        </div>
        <div>
          <h2 className="text-2xl font-bold">Intent-Aware Jailbreak Generator</h2>
          <p className="text-muted-foreground">
            LLM-powered deep understanding with comprehensive technique application
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <InputPanel
          coreRequest={coreRequest}
          setCoreRequest={setCoreRequest}
          techniqueSuite={techniqueSuite}
          setTechniqueSuite={setTechniqueSuite}
          potencyLevel={potencyLevel}
          setPotencyLevel={setPotencyLevel}
          applyAllTechniques={applyAllTechniques}
          setApplyAllTechniques={setApplyAllTechniques}
          advancedOptions={advancedOptions}
          setAdvancedOptions={setAdvancedOptions}
          showAdvanced={showAdvanced}
          setShowAdvanced={setShowAdvanced}
          intentAnalysis={intentAnalysis}
          isAnalyzing={analyzeIntentMutation.isPending}
          onAnalyze={handleAnalyzeIntent}
          isGenerating={intentAwareMutation.isPending}
          onGenerate={handleGenerate}
        />

        <ResultDisplay
          result={result}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          copyToClipboard={copyToClipboard}
        />
      </div>
    </div>
  );
}
