"use client";

import React, { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { enhancedApi } from "@/lib/api-enhanced";
import { toast } from "sonner";
import { handleApiError } from "@/lib/errors";
import {
  Zap,
  Brain,
  Database,
  Activity,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  BookOpen,
  Target,
  TrendingUp,
  Cpu,
} from "lucide-react";

import { StrategyLibraryPanel } from "./StrategyLibraryPanel";
import { LifelongLearningPanel } from "./LifelongLearningPanel";
import { AttackPanel } from "./AttackPanel";
import { ProviderModelDropdown } from "@/components/model-selector/ProviderModelDropdown";
import { useModelSelection } from "@/lib/stores/model-selection-store";

import type { LibraryStatsResponse, ProgressResponse } from "@/lib/types/autodan-turbo-types";

export function AutoDANTurboInterface() {
  const [activeTab, setActiveTab] = useState("attack");
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [stats, setStats] = useState<LibraryStatsResponse | null>(null);
  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const {
    selectedProvider,
    selectedModel,
    isLoading: modelSelectionLoading,
    selectProvider,
    selectModel,
  } = useModelSelection();
  const selectionReady = Boolean(selectedProvider && selectedModel);

  // Check system health on mount
  useEffect(() => {
    checkHealth();
    loadStats();
    loadProgress();
  }, []);

  const checkHealth = async () => {
    setIsCheckingHealth(true);
    try {
      const response = await enhancedApi.autodanTurbo.utils.health();
      setIsHealthy(response.data.status === "healthy");
    } catch (error) {
      console.error("Health check failed:", error);
      handleApiError(error);
      setIsHealthy(false);
    } finally {
      setIsCheckingHealth(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await enhancedApi.autodanTurbo.library.stats();
      setStats(response.data);
    } catch (error) {
      handleApiError(error);
    }
  };

  const loadProgress = async () => {
    try {
      const response = await enhancedApi.autodanTurbo.progress.get();
      setProgress(response.data);
    } catch (error) {
      handleApiError(error);
    }
  };

  const handleModelSelectionChange = async (provider: string, model: string) => {
    try {
      if (provider && provider !== selectedProvider) {
        const providerUpdated = await selectProvider(provider);
        if (!providerUpdated) {
          return;
        }
      }
      if (model) {
        await selectModel(model);
      }
    } catch (error) {
      console.error("Failed to update model selection:", error);
      toast.error("Unable to update model selection");
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Brain className="h-8 w-8 text-primary" />
            AutoDAN-Turbo
          </h1>
          <p className="text-muted-foreground mt-1">
            Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={checkHealth}
            disabled={isCheckingHealth}
          >
            {isCheckingHealth ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
          {isHealthy !== null && (
            <Badge variant={isHealthy ? "default" : "destructive"} className="gap-1">
              {isHealthy ? (
                <>
                  <CheckCircle2 className="h-3 w-3" />
                  System Healthy
                </>
              ) : (
                <>
                  <XCircle className="h-3 w-3" />
                  System Error
                </>
              )}
            </Badge>
          )}
        </div>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Model Selection
          </CardTitle>
          <CardDescription>
            Choose the provider and model used for AutoDAN-Turbo operations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <ProviderModelDropdown
            selectedProvider={selectedProvider || undefined}
            selectedModel={selectedModel || undefined}
            onSelectionChange={handleModelSelectionChange}
            className="w-full"
            placeholder="Select a provider and model"
          />
          <div className="text-sm text-muted-foreground flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <span>
              {selectionReady
                ? `Using ${selectedModel} via ${selectedProvider}`
                : "Select a provider and model to enable AutoDAN-Turbo requests."}
            </span>
            {modelSelectionLoading && (
              <span className="flex items-center gap-1 text-xs">
                <Loader2 className="h-3 w-3 animate-spin" />
                Syncing selection...
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Database className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats?.total_strategies || 0}</p>
                <p className="text-xs text-muted-foreground">Total Strategies</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-500/10 rounded-lg">
                <Zap className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats?.discovered_count || 0}</p>
                <p className="text-xs text-muted-foreground">Discovered</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <BookOpen className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats?.human_designed_count || 0}</p>
                <p className="text-xs text-muted-foreground">Human Designed</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <TrendingUp className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {stats?.avg_score_differential?.toFixed(1) || "N/A"}
                </p>
                <p className="text-xs text-muted-foreground">Avg Score Gain</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Running Task Indicator */}
      {progress && progress.status === "running" && (
        <Card className="border-primary/50 bg-primary/5">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Activity className="h-6 w-6 text-primary animate-pulse" />
                <span className="absolute -top-1 -right-1 h-3 w-3 bg-primary rounded-full animate-ping" />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">
                    {progress.current_phase === "warmup" && "Warmup Exploration"}
                    {progress.current_phase === "lifelong" && "Lifelong Learning"}
                    {progress.current_phase === "test" && "Test Phase"}
                  </span>
                  <Badge variant="outline">Running</Badge>
                </div>
                <p className="text-sm text-muted-foreground">
                  {progress.completed_requests} / {progress.total_requests} requests •{" "}
                  {progress.strategies_discovered} strategies discovered
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => setActiveTab("lifelong")}>
                View Progress
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="attack" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Single Attack
          </TabsTrigger>
          <TabsTrigger value="lifelong" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Lifelong Learning
          </TabsTrigger>
          <TabsTrigger value="library" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Strategy Library
          </TabsTrigger>
        </TabsList>

        <TabsContent value="attack">
          <AttackPanel
            selectedProvider={selectedProvider}
            selectedModel={selectedModel}
            selectionReady={selectionReady}
          />
        </TabsContent>

        <TabsContent value="lifelong">
          <LifelongLearningPanel
            selectedProvider={selectedProvider}
            selectedModel={selectedModel}
            selectionReady={selectionReady}
          />
        </TabsContent>

        <TabsContent value="library">
          <StrategyLibraryPanel />
        </TabsContent>
      </Tabs>

      {/* Paper Reference */}
      <Card className="bg-muted/30">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Reference</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Based on the paper: <strong>&quot;AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs&quot;</strong>
            {" "}by Liu et al., published at ICLR 2025.
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            This implementation features automatic strategy discovery, lifelong learning, and strategy library management
            for red-teaming LLMs.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export default AutoDANTurboInterface;
