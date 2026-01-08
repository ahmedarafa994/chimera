/**
 * AttackDashboard Component
 *
 * Main dashboard for the multi-vector attack framework, integrating
 * all components: session management, attack configuration, reasoning
 * visualization, and mutation optimizer.
 */

"use client";

import * as React from 'react';
import { useState, useCallback, useEffect } from 'react';
import {
  Zap,
  LayoutDashboard,
  Activity,
  Target,
  TrendingUp,
  Clock,
  Wifi,
  WifiOff,
  AlertCircle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Maximize2,
  Minimize2,
  Eye,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// Import unified attack components
import { AttackSessionManager } from './AttackSessionManager';
import { AttackConfigPanel } from './AttackConfigPanel';
import { ReasoningChainViewer } from './ReasoningChainViewer';
import { MutationOptimizerUI } from './MutationOptimizerUI';

// Import hooks
import { useAttackProgress, useParetoFront } from '@/hooks/use-unified-attack';
import type {
  AttackResponse,
  ReasoningChain,
  MutationHistory,
  ConvergenceMetrics,
  DualVectorMetrics,
} from '@/types/unified-attack-types';

// ==============================================================================
// Sub-components
// ==============================================================================

interface ConnectionStatusProps {
  isConnected: boolean;
  sessionId: string | null;
}

function ConnectionStatus({ isConnected, sessionId }: ConnectionStatusProps) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Badge variant="outline" className="gap-1.5 border-emerald-500/30 text-emerald-400">
                <Wifi className="h-3 w-3" />
                <span className="text-xs">Connected</span>
              </Badge>
            ) : (
              <Badge variant="outline" className="gap-1.5 border-red-500/30 text-red-400">
                <WifiOff className="h-3 w-3" />
                <span className="text-xs">Disconnected</span>
              </Badge>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          {isConnected
            ? `WebSocket connected to session: ${sessionId}`
            : 'WebSocket disconnected. Updates may be delayed.'}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface QuickStatsProps {
  totalAttacks: number;
  successRate: number;
  bestFitness: number;
  avgLatency: number;
}

function QuickStats({ totalAttacks, successRate, bestFitness, avgLatency }: QuickStatsProps) {
  return (
    <div className="grid grid-cols-4 gap-4">
      <Card className="bg-slate-800/30 border-slate-700/50">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Total Attacks</p>
              <p className="text-2xl font-bold text-slate-200">{totalAttacks}</p>
            </div>
            <Target className="h-8 w-8 text-slate-600" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-800/30 border-slate-700/50">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Success Rate</p>
              <p className="text-2xl font-bold text-emerald-400">{successRate.toFixed(0)}%</p>
            </div>
            <CheckCircle2 className="h-8 w-8 text-emerald-600/50" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-800/30 border-slate-700/50">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Best Fitness</p>
              <p className="text-2xl font-bold text-blue-400">{bestFitness.toFixed(3)}</p>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-600/50" />
          </div>
        </CardContent>
      </Card>

      <Card className="bg-slate-800/30 border-slate-700/50">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">Avg Latency</p>
              <p className="text-2xl font-bold text-amber-400">{avgLatency.toFixed(0)}ms</p>
            </div>
            <Clock className="h-8 w-8 text-amber-600/50" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

interface AttackResultsViewerProps {
  results: AttackResponse[];
  onSelectResult?: (result: AttackResponse) => void;
  selectedId?: string | null;
}

function AttackResultsViewer({ results, onSelectResult, selectedId }: AttackResultsViewerProps) {
  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Target className="h-12 w-12 text-slate-600 mb-4" />
        <h3 className="text-lg font-medium text-slate-400">No Attack Results</h3>
        <p className="text-sm text-slate-500 mt-1">
          Execute an attack to see results here
        </p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-[400px]">
      <div className="space-y-2 pr-4">
        {results.map((result) => (
          <button
            key={result.attack_id}
            onClick={() => onSelectResult?.(result)}
            className={cn(
              "w-full p-3 rounded-lg border text-left transition-all",
              "hover:border-slate-600",
              selectedId === result.attack_id
                ? "border-blue-500/50 bg-blue-500/10"
                : "border-slate-700/50 bg-slate-800/30"
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs",
                      result.success
                        ? "border-emerald-500/30 text-emerald-400"
                        : "border-red-500/30 text-red-400"
                    )}
                  >
                    {result.success ? 'Success' : 'Failed'}
                  </Badge>
                  <span className="text-xs text-slate-500 font-mono">
                    {result.attack_id.slice(0, 8)}...
                  </span>
                </div>
                <p className="text-sm text-slate-300 mt-1 line-clamp-1">
                  {result.original_query}
                </p>
                <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                  <span>Fitness: {result.unified_fitness?.toFixed(4) ?? '—'}</span>
                  <span>Latency: {result.latency_ms?.toFixed(0) ?? '—'}ms</span>
                </div>
              </div>
              <Eye className="h-4 w-4 text-slate-500" />
            </div>
          </button>
        ))}
      </div>
    </ScrollArea>
  );
}

interface ParetoFrontViewerProps {
  sessionId: string | null;
}

function ParetoFrontViewer({ sessionId }: ParetoFrontViewerProps) {
  const { data: paretoData, isLoading } = useParetoFront(sessionId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-slate-500" />
      </div>
    );
  }

  if (!paretoData || !paretoData.pareto_front || paretoData.pareto_front.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <TrendingUp className="h-12 w-12 text-slate-600 mb-4" />
        <h3 className="text-lg font-medium text-slate-400">No Pareto Front</h3>
        <p className="text-sm text-slate-500 mt-1">
          Execute attacks to populate the Pareto front
        </p>
      </div>
    );
  }

  // Find max values for normalization
  const maxObfuscation = Math.max(...paretoData.pareto_front.map(m => m.obfuscation_fitness ?? 0), 0.001);
  const maxMutation = Math.max(...paretoData.pareto_front.map(m => m.mutation_fitness ?? 0), 0.001);

  return (
    <div className="space-y-4">
      {/* Scatter plot visualization */}
      <div className="relative h-64 bg-slate-800/30 rounded-lg p-4">
        <div className="absolute inset-4 border-l border-b border-slate-700">
          {paretoData.pareto_front.map((point, idx) => {
            const mFit = point.mutation_fitness ?? 0;
            const oFit = point.obfuscation_fitness ?? 0;
            const x = (mFit / maxMutation) * 100;
            const y = (oFit / maxObfuscation) * 100;
            return (
              <TooltipProvider key={idx}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div
                      className={cn(
                        "absolute h-3 w-3 rounded-full transform -translate-x-1/2 translate-y-1/2",
                        "bg-gradient-to-br from-blue-500 to-cyan-400",
                        "hover:scale-150 transition-transform cursor-pointer"
                      )}
                      style={{
                        left: `${x}%`,
                        bottom: `${y}%`,
                      }}
                    />
                  </TooltipTrigger>
                  <TooltipContent>
                    <div className="text-xs space-y-1">
                      <div>Obfuscation: {point.obfuscation_fitness?.toFixed(4) ?? '0.000'}</div>
                      <div>Mutation: {point.mutation_fitness?.toFixed(4) ?? '0.000'}</div>
                      <div>Unified: {point.unified_fitness.toFixed(4)}</div>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            );
          })}
        </div>

        {/* Axis labels */}
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-xs text-slate-500">
          Mutation Fitness
        </div>
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-slate-500">
          Obfuscation Fitness
        </div>
      </div>

      {/* Stats summary */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-lg font-semibold text-slate-200">
            {paretoData.pareto_front?.length ?? 0}
          </div>
          <div className="text-xs text-slate-500">Pareto Points</div>
        </div>
        <div>
          <div className="text-lg font-semibold text-emerald-400">
            {Math.max(...(paretoData.pareto_front || []).map(m => m.unified_fitness)).toFixed(4)}
          </div>
          <div className="text-xs text-slate-500">Best Unified</div>
        </div>
        <div>
          <div className="text-lg font-semibold text-blue-400">
            {paretoData.total_evaluations}
          </div>
          <div className="text-xs text-slate-500">Total Evaluations</div>
        </div>
      </div>
    </div>
  );
}

// ==============================================================================
// Main Component
// ==============================================================================

export interface AttackDashboardProps {
  className?: string;
}

export function AttackDashboard({ className }: AttackDashboardProps) {
  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);

  // Attack results state
  const [attackResults, setAttackResults] = useState<AttackResponse[]>([]);
  const [currentGeneration, setCurrentGeneration] = useState(0);

  // Layout state
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeTab, setActiveTab] = useState('reasoning');

  // WebSocket progress hook
  const progressState = useAttackProgress({
    sessionId,
    onProgress: (event) => {
      // Update generation from progress events
      if (event.current_generation) {
        setCurrentGeneration(event.current_generation);
      }
    },
  });

  // Handlers
  const handleSessionCreated = useCallback((newSessionId: string) => {
    setSessionId(newSessionId);
    setAttackResults([]);
    setSelectedResultId(null);
  }, []);

  const handleSessionFinalized = useCallback(() => {
    setSessionId(null);
  }, []);

  const handleAttackCompleted = useCallback((result: AttackResponse) => {
    setAttackResults(prev => [result, ...prev]);
    setSelectedResultId(result.attack_id);
  }, []);

  const handleSelectResult = useCallback((result: AttackResponse) => {
    setSelectedResultId(result.attack_id);
  }, []);

  // Calculate quick stats
  const totalAttacks = attackResults.length;
  const successfulAttacks = attackResults.filter(r => r.success).length;
  const successRate = totalAttacks > 0 ? (successfulAttacks / totalAttacks) * 100 : 0;
  const bestFitness = totalAttacks > 0
    ? Math.max(...attackResults.map(r => r.unified_fitness ?? 0))
    : 0;
  const avgLatency = totalAttacks > 0
    ? attackResults.reduce((sum, r) => sum + (r.latency_ms ?? 0), 0) / totalAttacks
    : 0;

  return (
    <div className={cn(
      "flex flex-col h-full bg-slate-950",
      isFullscreen && "fixed inset-0 z-50",
      className
    )}>
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">
              Multi-Vector Attack Dashboard
            </h1>
            <p className="text-xs text-slate-500">
              Unified obfuscation and mutation framework
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <ConnectionStatus
            isConnected={progressState.isConnected}
            sessionId={sessionId}
          />
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsFullscreen(!isFullscreen)}
          >
            {isFullscreen ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
        </div>
      </header>

      {/* Quick Stats */}
      <div className="px-6 py-4 border-b border-slate-800">
        <QuickStats
          totalAttacks={totalAttacks}
          successRate={successRate}
          bestFitness={bestFitness}
          avgLatency={avgLatency}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* Left Panel - Session & Config */}
          <ResizablePanel defaultSize={30} minSize={20} maxSize={40}>
            <ScrollArea className="h-full">
              <div className="p-4 space-y-4">
                <AttackSessionManager
                  sessionId={sessionId}
                  onSessionCreated={handleSessionCreated}
                  onSessionFinalized={handleSessionFinalized}
                />

                <AttackConfigPanel
                  sessionId={sessionId}
                  onAttackCompleted={handleAttackCompleted}
                />
              </div>
            </ScrollArea>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Middle Panel - Visualization */}
          <ResizablePanel defaultSize={45} minSize={30}>
            <div className="h-full p-4">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
                <TabsList className="bg-slate-800/50 w-fit">
                  <TabsTrigger value="reasoning" className="gap-1">
                    <Activity className="h-3 w-3" />
                    Reasoning
                  </TabsTrigger>
                  <TabsTrigger value="optimizer" className="gap-1">
                    <TrendingUp className="h-3 w-3" />
                    Optimizer
                  </TabsTrigger>
                  <TabsTrigger value="pareto" className="gap-1">
                    <Target className="h-3 w-3" />
                    Pareto Front
                  </TabsTrigger>
                </TabsList>

                <div className="flex-1 mt-4 overflow-hidden">
                  <TabsContent value="reasoning" className="h-full m-0">
                    <ReasoningChainViewer
                      chain={progressState.reasoningChain}
                      convergenceMetrics={
                        progressState.convergenceMetrics.length > 0
                          ? progressState.convergenceMetrics[progressState.convergenceMetrics.length - 1]
                          : null
                      }
                      isLoading={progressState.phase === 'processing'}
                      maxHeight="calc(100% - 20px)"
                    />
                  </TabsContent>

                  <TabsContent value="optimizer" className="h-full m-0">
                    <MutationOptimizerUI
                      mutationHistory={progressState.mutationHistory}
                      convergenceMetrics={progressState.convergenceMetrics}
                      currentGeneration={currentGeneration}
                      isRunning={progressState.phase === 'processing'}
                    />
                  </TabsContent>

                  <TabsContent value="pareto" className="h-full m-0">
                    <Card className="h-full bg-slate-900/50 border-slate-800">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                          <Target className="h-5 w-5" />
                          Pareto Front Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ParetoFrontViewer sessionId={sessionId} />
                      </CardContent>
                    </Card>
                  </TabsContent>
                </div>
              </Tabs>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right Panel - Results */}
          <ResizablePanel defaultSize={25} minSize={15} maxSize={35}>
            <div className="h-full p-4">
              <Card className="h-full bg-slate-900/50 border-slate-800">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                      <LayoutDashboard className="h-5 w-5" />
                      Attack Results
                    </CardTitle>
                    <Badge variant="outline" className="text-xs">
                      {attackResults.length} results
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <AttackResultsViewer
                    results={attackResults}
                    onSelectResult={handleSelectResult}
                    selectedId={selectedResultId}
                  />
                </CardContent>
              </Card>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Progress Indicator */}
      {progressState.phase && progressState.phase !== 'completed' && (
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-900/50">
          <div className="flex items-center gap-4">
            <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
            <div className="flex-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-300 capitalize">{progressState.phase}</span>
                <span className="text-slate-500">{progressState.progress}%</span>
              </div>
              <div className="h-1 mt-1 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 to-cyan-400 transition-all duration-300"
                  style={{ width: `${progressState.progress}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AttackDashboard;
