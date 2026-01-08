/**
 * ReasoningChainViewer Component
 *
 * Visualization for the mathematical reasoning framework showing
 * step-by-step reasoning chains, fitness evolution, and convergence.
 */

"use client";

import * as React from 'react';
import { useMemo, useState } from 'react';
import {
  Brain,
  ChevronDown,
  ChevronRight,
  Lightbulb,
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  Zap,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Activity,
  Target,
  GitBranch,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

import type {
  ReasoningChain,
  ReasoningStep,
  ConvergenceMetrics,
} from '@/types/unified-attack-types';

// ==============================================================================
// Constants
// ==============================================================================

const STEP_TYPE_STYLES: Record<string, { color: string; icon: React.ReactNode }> = {
  initialization: { color: 'bg-blue-500', icon: <Zap className="h-3 w-3" /> },
  obfuscation: { color: 'bg-purple-500', icon: <GitBranch className="h-3 w-3" /> },
  mutation: { color: 'bg-amber-500', icon: <Activity className="h-3 w-3" /> },
  crossover: { color: 'bg-emerald-500', icon: <GitBranch className="h-3 w-3" /> },
  selection: { color: 'bg-cyan-500', icon: <Target className="h-3 w-3" /> },
  evaluation: { color: 'bg-pink-500', icon: <Brain className="h-3 w-3" /> },
  convergence: { color: 'bg-green-500', icon: <CheckCircle2 className="h-3 w-3" /> },
  default: { color: 'bg-slate-500', icon: <Lightbulb className="h-3 w-3" /> },
};

// ==============================================================================
// Sub-components
// ==============================================================================

interface StepTypeBadgeProps {
  type: string;
}

function StepTypeBadge({ type }: StepTypeBadgeProps) {
  const style = STEP_TYPE_STYLES[type] || STEP_TYPE_STYLES.default;
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1 px-2 py-0.5 text-xs font-medium capitalize",
        "border-none text-white",
        style.color
      )}
    >
      {style.icon}
      {type}
    </Badge>
  );
}

interface FitnessTrendProps {
  current: number;
  previous?: number;
  className?: string;
}

function FitnessTrend({ current, previous, className }: FitnessTrendProps) {
  const diff = previous !== undefined ? current - previous : 0;
  const percent = previous ? ((diff / previous) * 100).toFixed(1) : '0';

  let TrendIcon = Minus;
  let colorClass = 'text-slate-400';

  if (diff > 0.001) {
    TrendIcon = TrendingUp;
    colorClass = 'text-emerald-400';
  } else if (diff < -0.001) {
    TrendIcon = TrendingDown;
    colorClass = 'text-red-400';
  }

  return (
    <div className={cn("flex items-center gap-1", colorClass, className)}>
      <TrendIcon className="h-4 w-4" />
      <span className="text-xs font-mono">
        {diff >= 0 ? '+' : ''}{percent}%
      </span>
    </div>
  );
}

interface ReasoningStepCardProps {
  step: ReasoningStep;
  previousFitness?: number;
  isLast: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}

function ReasoningStepCard({
  step,
  previousFitness,
  isLast,
  isExpanded,
  onToggle,
}: ReasoningStepCardProps) {
  const durationMs = step.duration_ms ?? 0;
  const durationFormatted = durationMs < 1000
    ? `${durationMs.toFixed(0)}ms`
    : `${(durationMs / 1000).toFixed(2)}s`;

  return (
    <div className="relative">
      {/* Timeline connector */}
      {!isLast && (
        <div className="absolute left-5 top-10 h-full w-px bg-slate-700" />
      )}

      <Collapsible open={isExpanded} onOpenChange={onToggle}>
        <CollapsibleTrigger asChild>
          <div
            className={cn(
              "relative flex cursor-pointer items-start gap-3 rounded-lg p-3",
              "bg-slate-800/50 hover:bg-slate-800/80 transition-colors",
              "border border-slate-700/50"
            )}
          >
            {/* Step number indicator */}
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-700 text-sm font-semibold text-slate-300">
              {step.step_number}
            </div>

            {/* Step content */}
            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-center gap-2">
                <StepTypeBadge type={step.step_type ?? 'default'} />
                <span className="text-xs text-slate-500 font-mono">
                  <Clock className="inline h-3 w-3 mr-1" />
                  {durationFormatted}
                </span>
              </div>

              <p className="text-sm text-slate-300 line-clamp-2">
                {step.description ?? 'No description'}
              </p>

              {/* Quick metrics */}
              <div className="flex items-center gap-4 pt-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="flex items-center gap-1 text-xs text-slate-400">
                        <Target className="h-3 w-3" />
                        <span className="font-mono">{(step.fitness_before ?? 0).toFixed(4)}</span>
                        <span className="text-slate-500">→</span>
                        <span className="font-mono text-slate-200">{(step.fitness_after ?? 0).toFixed(4)}</span>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      Fitness: {(step.fitness_before ?? 0).toFixed(4)} → {(step.fitness_after ?? 0).toFixed(4)}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>

                <FitnessTrend
                  current={step.fitness_after ?? 0}
                  previous={step.fitness_before ?? 0}
                />
              </div>
            </div>

            {/* Expand indicator */}
            <div className="text-slate-500">
              {isExpanded ? (
                <ChevronDown className="h-5 w-5" />
              ) : (
                <ChevronRight className="h-5 w-5" />
              )}
            </div>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="ml-12 mt-2 space-y-3 rounded-lg bg-slate-800/30 p-4 border border-slate-700/30">
            {/* Detailed reasoning */}
            {step.reasoning && (
              <div className="space-y-1">
                <h5 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Reasoning
                </h5>
                <p className="text-sm text-slate-300">{step.reasoning}</p>
              </div>
            )}

            {/* Mutations applied */}
            {step.mutations_applied && step.mutations_applied.length > 0 && (
              <div className="space-y-2">
                <h5 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Mutations Applied
                </h5>
                <div className="flex flex-wrap gap-1.5">
                  {step.mutations_applied.map((mutation, idx) => (
                    <Badge
                      key={idx}
                      variant="secondary"
                      className="text-xs bg-slate-700 text-slate-300"
                    >
                      {mutation}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Input/Output comparison */}
            {step.input_prompt && step.output_prompt && (
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <h5 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Input
                  </h5>
                  <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-2 overflow-x-auto max-h-32">
                    {step.input_prompt}
                  </pre>
                </div>
                <div className="space-y-1">
                  <h5 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Output
                  </h5>
                  <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-2 overflow-x-auto max-h-32">
                    {step.output_prompt}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

interface ConvergenceIndicatorProps {
  metrics: ConvergenceMetrics | null;
  className?: string;
}

function ConvergenceIndicator({ metrics, className }: ConvergenceIndicatorProps) {
  if (!metrics) return null;

  const currentFitness = metrics.current_fitness ?? 0;
  const convergenceThreshold = metrics.convergence_threshold || 1;
  const progressPercent = Math.min(
    (currentFitness / convergenceThreshold) * 100,
    100
  );

  return (
    <div className={cn("space-y-3 rounded-lg bg-slate-800/50 p-4 border border-slate-700/50", className)}>
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
          <Activity className="h-4 w-4" />
          Convergence Status
        </h4>
        {metrics.converged ? (
          <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Converged
          </Badge>
        ) : (
          <Badge variant="outline" className="border-amber-500/30 text-amber-400">
            <AlertTriangle className="h-3 w-3 mr-1" />
            In Progress
          </Badge>
        )}
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-slate-400">Progress to Threshold</span>
          <span className="font-mono text-slate-300">
            {currentFitness.toFixed(4)} / {metrics.convergence_threshold?.toFixed(4) ?? '—'}
          </span>
        </div>
        <Progress value={progressPercent} className="h-2" />
      </div>

      <div className="grid grid-cols-3 gap-4 pt-2">
        <div className="text-center">
          <div className="text-lg font-semibold text-slate-200">
            {metrics.iterations_completed}
          </div>
          <div className="text-xs text-slate-500">Iterations</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-slate-200">
            {metrics.improvement_rate?.toFixed(3) ?? '—'}
          </div>
          <div className="text-xs text-slate-500">Improvement Rate</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-slate-200">
            {metrics.stagnation_count}
          </div>
          <div className="text-xs text-slate-500">Stagnation</div>
        </div>
      </div>
    </div>
  );
}

interface FitnessChartProps {
  steps: ReasoningStep[];
  className?: string;
}

function FitnessChart({ steps, className }: FitnessChartProps) {
  const chartData = useMemo(() => {
    if (steps.length === 0) return [];

    const fitnessValues = steps.map(s => s.fitness_after ?? 0);
    const maxFitness = Math.max(...fitnessValues, 0.001);

    return steps.map((step, idx) => ({
      step: idx + 1,
      fitness: step.fitness_after ?? 0,
      height: ((step.fitness_after ?? 0) / maxFitness) * 100,
    }));
  }, [steps]);

  if (chartData.length === 0) {
    return null;
  }

  return (
    <div className={cn("space-y-2", className)}>
      <h4 className="text-sm font-medium text-slate-400">Fitness Evolution</h4>
      <div className="flex items-end gap-1 h-24 bg-slate-800/30 rounded-lg p-2">
        {chartData.map((point, idx) => (
          <TooltipProvider key={idx}>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className="flex-1 bg-gradient-to-t from-blue-600 to-cyan-400 rounded-t transition-all hover:opacity-80"
                  style={{ height: `${point.height}%`, minHeight: '4px' }}
                />
              </TooltipTrigger>
              <TooltipContent>
                <div className="text-xs">
                  <div>Step {point.step}</div>
                  <div className="font-mono">Fitness: {(point.fitness ?? 0).toFixed(4)}</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </div>
      <div className="flex justify-between text-xs text-slate-500">
        <span>Step 1</span>
        <span>Step {chartData.length}</span>
      </div>
    </div>
  );
}

// ==============================================================================
// Main Component
// ==============================================================================

export interface ReasoningChainViewerProps {
  chain: ReasoningChain | null;
  convergenceMetrics?: ConvergenceMetrics | null;
  isLoading?: boolean;
  maxHeight?: string;
  className?: string;
}

export function ReasoningChainViewer({
  chain,
  convergenceMetrics,
  isLoading = false,
  maxHeight = '600px',
  className,
}: ReasoningChainViewerProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  const toggleStep = (stepNumber: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev);
      if (next.has(stepNumber)) {
        next.delete(stepNumber);
      } else {
        next.add(stepNumber);
      }
      return next;
    });
  };

  const expandAll = () => {
    if (chain?.steps) {
      setExpandedSteps(new Set(chain.steps.map(s => s.step_number)));
    }
  };

  const collapseAll = () => {
    setExpandedSteps(new Set());
  };

  // Empty state
  if (!chain && !isLoading) {
    return (
      <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <Brain className="h-12 w-12 text-slate-600 mb-4" />
          <h3 className="text-lg font-medium text-slate-400">No Reasoning Chain</h3>
          <p className="text-sm text-slate-500 mt-1">
            Execute an attack to see the reasoning visualization
          </p>
        </CardContent>
      </Card>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <div className="relative">
            <Brain className="h-12 w-12 text-blue-500 animate-pulse" />
            <div className="absolute inset-0 h-12 w-12 rounded-full border-2 border-blue-500/30 animate-ping" />
          </div>
          <p className="text-sm text-slate-400 mt-4">Processing reasoning chain...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold text-slate-200 flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Reasoning Chain
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {chain?.total_steps ?? 0} steps
            </Badge>
            <Badge variant="outline" className="text-xs">
              {((chain?.total_duration_ms ?? 0) / 1000).toFixed(2)}s
            </Badge>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2 pt-2">
          <button
            onClick={expandAll}
            className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            Expand All
          </button>
          <span className="text-slate-600">|</span>
          <button
            onClick={collapseAll}
            className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            Collapse All
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Convergence indicator */}
        {convergenceMetrics && (
          <ConvergenceIndicator metrics={convergenceMetrics} />
        )}

        {/* Fitness chart */}
        {chain && chain.steps.length > 0 && (
          <FitnessChart steps={chain.steps} />
        )}

        {/* Steps list */}
        <ScrollArea style={{ maxHeight }} className="pr-4">
          <div className="space-y-3">
            {chain?.steps.map((step, idx) => (
              <ReasoningStepCard
                key={step.step_number}
                step={step}
                previousFitness={idx > 0 ? chain.steps[idx - 1].fitness_after : undefined}
                isLast={idx === chain.steps.length - 1}
                isExpanded={expandedSteps.has(step.step_number)}
                onToggle={() => toggleStep(step.step_number)}
              />
            ))}
          </div>
        </ScrollArea>

        {/* Final result summary */}
        {chain && chain.convergence_achieved && (
          <div className="flex items-center gap-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-4">
            <CheckCircle2 className="h-6 w-6 text-emerald-400" />
            <div>
              <div className="text-sm font-medium text-emerald-300">
                Convergence Achieved
              </div>
              <div className="text-xs text-emerald-400/80">
                Final fitness: {chain.final_fitness.toFixed(4)} after {chain.total_steps} steps
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default ReasoningChainViewer;
