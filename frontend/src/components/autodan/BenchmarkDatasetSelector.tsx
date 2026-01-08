"use client";

import React, { useEffect, useState } from "react";
import { Database, AlertTriangle, Shuffle, Download, Filter, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useBenchmarkDataset } from "@/hooks/use-benchmark-dataset";
import { BenchmarkPrompt } from "@/lib/services/benchmark-dataset-service";

// Risk area color mapping based on Do-Not-Answer taxonomy
const RISK_AREA_COLORS: Record<string, string> = {
  "information_hazards": "bg-red-500/10 text-red-400 border-red-500/20",
  "malicious_uses": "bg-orange-500/10 text-orange-400 border-orange-500/20",
  "discrimination_exclusion_toxicity": "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
  "misinformation_harms": "bg-purple-500/10 text-purple-400 border-purple-500/20",
  "human_autonomy_integrity": "bg-blue-500/10 text-blue-400 border-blue-500/20",
};

// Severity badge colors
const SEVERITY_COLORS: Record<string, string> = {
  "high": "bg-red-500/20 text-red-300 border-red-500/30",
  "medium": "bg-yellow-500/20 text-yellow-300 border-yellow-500/30",
  "low": "bg-green-500/20 text-green-300 border-green-500/30",
};

interface BenchmarkDatasetSelectorProps {
  onPromptSelect?: (prompt: BenchmarkPrompt) => void;
  onBatchGenerate?: (prompts: BenchmarkPrompt[]) => void;
  className?: string;
}

/**
 * BenchmarkDatasetSelector component for selecting and browsing
 * Do-Not-Answer benchmark prompts for adversarial testing.
 */
export function BenchmarkDatasetSelector({
  onPromptSelect,
  onBatchGenerate,
  className = "",
}: BenchmarkDatasetSelectorProps) {
  const {
    isLoading,
    error,
    datasets,
    selectedDataset,
    selectedPrompts,
    statistics,
    riskAreas,
    fetchDatasets,
    selectDataset,
    fetchPrompts,
    fetchRandomPrompts,
    clearError,
  } = useBenchmarkDataset();

  const [selectedRiskArea, setSelectedRiskArea] = useState<string>("all");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("all");
  const [isFiltersOpen, setIsFiltersOpen] = useState(false);
  const [displayedPrompts, setDisplayedPrompts] = useState<BenchmarkPrompt[]>([]);

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // Update displayed prompts when selection changes
  useEffect(() => {
    setDisplayedPrompts(selectedPrompts.slice(0, 20));
  }, [selectedPrompts]);

  // Handle dataset selection
  const handleDatasetSelect = async (value: string) => {
    await selectDataset(value);
  };

  // Handle filter changes
  const handleFilterApply = async () => {
    const options: { risk_area?: string; severity?: string; limit?: number } = { limit: 50 };
    if (selectedRiskArea !== "all") options.risk_area = selectedRiskArea;
    if (selectedSeverity !== "all") options.severity = selectedSeverity;
    await fetchPrompts(options);
  };

  // Handle random prompt selection
  const handleRandomSelect = async () => {
    const options: { risk_area?: string } = {};
    if (selectedRiskArea !== "all") options.risk_area = selectedRiskArea;
    const prompts = await fetchRandomPrompts(5, options);
    setDisplayedPrompts(prompts);
  };

  // Handle batch export for jailbreak testing
  const handleBatchExport = () => {
    if (onBatchGenerate && displayedPrompts.length > 0) {
      onBatchGenerate(displayedPrompts);
    }
  };

  // Render loading skeleton
  if (isLoading && datasets.length === 0) {
    return (
      <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
        <CardHeader>
          <Skeleton className="h-6 w-48 bg-slate-800" />
          <Skeleton className="h-4 w-64 bg-slate-800" />
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <Skeleton className="h-10 w-full bg-slate-800" />
            <Skeleton className="h-32 w-full bg-slate-800" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-cyan-400" />
            <CardTitle className="text-lg text-slate-100">Benchmark Datasets</CardTitle>
          </div>
          {statistics && (
            <Badge variant="outline" className="bg-cyan-500/10 text-cyan-400 border-cyan-500/20">
              {statistics.total_prompts} prompts
            </Badge>
          )}
        </div>
        <CardDescription className="text-slate-400">
          Select harmful prompt datasets for adversarial testing (Do-Not-Answer taxonomy)
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Error Display */}
        {error && (
          <Alert variant="destructive" className="bg-red-500/10 border-red-500/20">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearError}
              className="mt-2 text-red-400 hover:text-red-300"
            >
              Dismiss
            </Button>
          </Alert>
        )}

        {/* Dataset Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300">Select Dataset</label>
          <Select onValueChange={handleDatasetSelect} disabled={isLoading}>
            <SelectTrigger className="bg-slate-800/50 border-slate-700 text-slate-100">
              <SelectValue placeholder="Choose a benchmark dataset..." />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-700">
              {datasets.map((dataset) => (
                <SelectItem
                  key={dataset.name}
                  value={dataset.name}
                  className="text-slate-100 focus:bg-slate-700 focus:text-white"
                >
                  <div className="flex items-center gap-2">
                    <span>{dataset.name}</span>
                    <Badge variant="outline" className="text-xs bg-slate-700/50">
                      {dataset.total_prompts} prompts
                    </Badge>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Dataset Statistics */}
        {selectedDataset && statistics && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-800/30 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-cyan-400">
                {statistics.total_prompts}
              </div>
              <div className="text-xs text-slate-400">Total Prompts</div>
            </div>
            <div className="bg-slate-800/30 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-purple-400">
                {statistics.risk_area_counts ? Object.keys(statistics.risk_area_counts).length : 0}
              </div>
              <div className="text-xs text-slate-400">Risk Areas</div>
            </div>
            <div className="bg-slate-800/30 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-orange-400">
                {statistics.harm_type_counts ? Object.keys(statistics.harm_type_counts).length : 0}
              </div>
              <div className="text-xs text-slate-400">Harm Types</div>
            </div>
            <div className="bg-slate-800/30 rounded-lg p-3 text-center">
              <div className="text-2xl font-bold text-green-400">
                {statistics.severity_counts?.high || 0}
              </div>
              <div className="text-xs text-slate-400">High Severity</div>
            </div>
          </div>
        )}

        {/* Risk Area Distribution */}
        {riskAreas.length > 0 && (
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300">Risk Area Distribution</label>
            <div className="flex flex-wrap gap-2">
              {riskAreas.map((area) => (
                <Badge
                  key={area.name}
                  variant="outline"
                  className={RISK_AREA_COLORS[area.name] || "bg-slate-500/10 text-slate-400"}
                >
                  {area.display_name}: {area.prompt_count}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Filters */}
        {selectedDataset && (
          <Collapsible open={isFiltersOpen} onOpenChange={setIsFiltersOpen}>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                className="w-full justify-between text-slate-300 hover:text-white hover:bg-slate-800/50"
              >
                <div className="flex items-center gap-2">
                  <Filter className="h-4 w-4" />
                  <span>Filters</span>
                </div>
                {isFiltersOpen ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-3 pt-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs text-slate-400">Risk Area</label>
                  <Select value={selectedRiskArea} onValueChange={setSelectedRiskArea}>
                    <SelectTrigger className="bg-slate-800/50 border-slate-700 text-slate-100 h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="all" className="text-slate-100">All Areas</SelectItem>
                      {riskAreas.map((area) => (
                        <SelectItem key={area.name} value={area.name} className="text-slate-100">
                          {area.display_name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-slate-400">Severity</label>
                  <Select value={selectedSeverity} onValueChange={setSelectedSeverity}>
                    <SelectTrigger className="bg-slate-800/50 border-slate-700 text-slate-100 h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      <SelectItem value="all" className="text-slate-100">All Levels</SelectItem>
                      <SelectItem value="high" className="text-slate-100">High</SelectItem>
                      <SelectItem value="medium" className="text-slate-100">Medium</SelectItem>
                      <SelectItem value="low" className="text-slate-100">Low</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Button
                onClick={handleFilterApply}
                disabled={isLoading}
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-white"
              >
                Apply Filters
              </Button>
            </CollapsibleContent>
          </Collapsible>
        )}

        <Separator className="bg-slate-700" />

        {/* Action Buttons */}
        {selectedDataset && (
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={handleRandomSelect}
              disabled={isLoading}
              className="flex-1 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white"
            >
              <Shuffle className="h-4 w-4 mr-2" />
              Random 5
            </Button>
            <Button
              variant="outline"
              onClick={handleBatchExport}
              disabled={isLoading || displayedPrompts.length === 0}
              className="flex-1 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white"
            >
              <Download className="h-4 w-4 mr-2" />
              Use for Attack
            </Button>
          </div>
        )}

        {/* Prompt List */}
        {displayedPrompts.length > 0 && (
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300">
              Selected Prompts ({displayedPrompts.length})
            </label>
            <ScrollArea className="h-[200px] rounded-md border border-slate-700 bg-slate-800/30">
              <div className="p-3 space-y-2">
                {displayedPrompts.map((prompt, index) => (
                  <div
                    key={prompt.id || index}
                    className="p-3 bg-slate-800/50 rounded-lg cursor-pointer hover:bg-slate-700/50 transition-colors"
                    onClick={() => onPromptSelect?.(prompt)}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <span className="text-xs text-slate-500">#{index + 1}</span>
                      <div className="flex gap-1">
                        {prompt.risk_area && (
                          <Badge
                            variant="outline"
                            className={`text-xs ${RISK_AREA_COLORS[prompt.risk_area] || ""}`}
                          >
                            {prompt.risk_area.replace(/_/g, " ")}
                          </Badge>
                        )}
                        {prompt.severity && (
                          <Badge
                            variant="outline"
                            className={`text-xs ${SEVERITY_COLORS[prompt.severity] || ""}`}
                          >
                            {prompt.severity}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <p className="text-sm text-slate-300 line-clamp-2">{prompt.prompt}</p>
                    {prompt.harm_type && (
                      <p className="text-xs text-slate-500 mt-1">
                        Harm: {prompt.harm_type.replace(/_/g, " ")}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

        {/* Empty State */}
        {selectedDataset && displayedPrompts.length === 0 && !isLoading && (
          <div className="text-center py-8 text-slate-500">
            <Database className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No prompts match your filters</p>
            <p className="text-sm">Try adjusting your filter criteria</p>
          </div>
        )}

        {/* Loading State */}
        {isLoading && selectedDataset && (
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-20 w-full bg-slate-800" />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default BenchmarkDatasetSelector;