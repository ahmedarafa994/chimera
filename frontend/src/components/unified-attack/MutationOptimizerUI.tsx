/**
 * MutationOptimizerUI Component
 *
 * Displays mutation history, operator statistics, and optimization
 * metrics for the genetic algorithm-based attack optimization.
 */

"use client";

import * as React from 'react';
import { useMemo, useState } from 'react';
import {
  Dna,
  Shuffle,
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Zap,
  Clock,
  BarChart3,
  Layers,
  RefreshCw,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronUp,
  Activity,
  Sparkles,
} from 'lucide-react';

import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import type {
  MutationHistory,
  ConvergenceMetrics,
} from '@/types/unified-attack-types';

// ==============================================================================
// Types
// ==============================================================================

interface MutationOperator {
  name: string;
  applications: number;
  successRate: number;
  avgFitnessGain: number;
  color: string;
}

interface GenerationStats {
  generation: number;
  bestFitness: number;
  avgFitness: number;
  diversity: number;
  timestamp: number;
}

// ==============================================================================
// Constants
// ==============================================================================

const OPERATOR_COLORS: Record<string, string> = {
  synonym_substitution: 'bg-blue-500',
  token_injection: 'bg-purple-500',
  phrase_reordering: 'bg-amber-500',
  context_expansion: 'bg-emerald-500',
  semantic_shift: 'bg-cyan-500',
  format_mutation: 'bg-pink-500',
  crossover: 'bg-orange-500',
  random_insertion: 'bg-red-500',
  character_swap: 'bg-indigo-500',
  default: 'bg-slate-500',
};

// ==============================================================================
// Sub-components
// ==============================================================================

interface FitnessChangeIndicatorProps {
  before: number;
  after: number;
  size?: 'sm' | 'md';
}

function FitnessChangeIndicator({ before, after, size = 'sm' }: FitnessChangeIndicatorProps) {
  const change = after - before;
  const percentChange = before > 0 ? ((change / before) * 100).toFixed(1) : '0';

  const iconSize = size === 'sm' ? 'h-3 w-3' : 'h-4 w-4';
  const textSize = size === 'sm' ? 'text-xs' : 'text-sm';

  if (change > 0.001) {
    return (
      <span className={cn("flex items-center gap-0.5 text-emerald-400", textSize)}>
        <TrendingUp className={iconSize} />
        +{percentChange}%
      </span>
    );
  } else if (change < -0.001) {
    return (
      <span className={cn("flex items-center gap-0.5 text-red-400", textSize)}>
        <TrendingDown className={iconSize} />
        {percentChange}%
      </span>
    );
  }
  return (
    <span className={cn("flex items-center gap-0.5 text-slate-400", textSize)}>
      <Minus className={iconSize} />
      0%
    </span>
  );
}

interface OperatorBarProps {
  operator: MutationOperator;
  maxApplications: number;
}

function OperatorBar({ operator, maxApplications }: OperatorBarProps) {
  const barWidth = maxApplications > 0 ? (operator.applications / maxApplications) * 100 : 0;
  const operatorColor = OPERATOR_COLORS[operator.name] || OPERATOR_COLORS.default;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={cn("h-2.5 w-2.5 rounded-full", operatorColor)} />
          <span className="text-sm text-slate-300 capitalize">
            {operator.name.replace(/_/g, ' ')}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-slate-400">
            {operator.applications} uses
          </span>
          <Badge
            variant="outline"
            className={cn(
              "text-xs px-1.5 py-0",
              operator.successRate > 0.7
                ? 'border-emerald-500/30 text-emerald-400'
                : operator.successRate > 0.4
                  ? 'border-amber-500/30 text-amber-400'
                  : 'border-red-500/30 text-red-400'
            )}
          >
            {(operator.successRate * 100).toFixed(0)}%
          </Badge>
        </div>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all duration-500", operatorColor)}
          style={{ width: `${barWidth}%` }}
        />
      </div>
    </div>
  );
}

interface MutationHistoryTableProps {
  history: MutationHistory[];
  maxRows?: number;
}

function MutationHistoryTable({ history, maxRows = 20 }: MutationHistoryTableProps) {
  const [expanded, setExpanded] = useState(false);
  const displayedHistory = expanded ? history : history.slice(-maxRows);

  if (history.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500 text-sm">
        No mutations recorded yet
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <ScrollArea className="h-[300px]">
        <Table>
          <TableHeader>
            <TableRow className="border-slate-700/50 hover:bg-transparent">
              <TableHead className="text-slate-400 w-16">#</TableHead>
              <TableHead className="text-slate-400">Operator</TableHead>
              <TableHead className="text-slate-400">Fitness Change</TableHead>
              <TableHead className="text-slate-400 text-right">Duration</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {displayedHistory.map((mutation, idx) => {
              const operatorKey = mutation.operator || 'unknown';
              const operatorColor = OPERATOR_COLORS[operatorKey as keyof typeof OPERATOR_COLORS] || OPERATOR_COLORS.default;
              return (
                <TableRow key={idx} className="border-slate-700/30">
                  <TableCell className="text-slate-500 font-mono text-xs">
                    {mutation.generation}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className={cn("h-2 w-2 rounded-full", operatorColor)} />
                      <span className="text-sm text-slate-300 capitalize">
                        {(mutation.operator || 'unknown').replace(/_/g, ' ')}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-slate-400">
                        {mutation.fitness_before.toFixed(4)}
                      </span>
                      <span className="text-slate-600">→</span>
                      <span className="text-xs font-mono text-slate-200">
                        {mutation.fitness_after.toFixed(4)}
                      </span>
                      <FitnessChangeIndicator
                        before={mutation.fitness_before}
                        after={mutation.fitness_after}
                      />
                    </div>
                  </TableCell>
                  <TableCell className="text-right text-xs font-mono text-slate-400">
                    {mutation.duration_ms?.toFixed(0) ?? 0}ms
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </ScrollArea>

      {history.length > maxRows && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded(!expanded)}
          className="w-full text-slate-400 hover:text-slate-200"
        >
          {expanded ? (
            <>
              <ChevronUp className="h-4 w-4 mr-1" />
              Show Less
            </>
          ) : (
            <>
              <ChevronDown className="h-4 w-4 mr-1" />
              Show All ({history.length})
            </>
          )}
        </Button>
      )}
    </div>
  );
}

interface GenerationChartProps {
  generations: GenerationStats[];
}

function GenerationChart({ generations }: GenerationChartProps) {
  const maxFitness = useMemo(() => {
    return Math.max(...generations.map(g => g.bestFitness), 0.001);
  }, [generations]);

  if (generations.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500 text-sm">
        No generation data available
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Chart */}
      <div className="flex items-end gap-0.5 h-32 bg-slate-800/30 rounded-lg p-2">
        {generations.map((gen, idx) => (
          <TooltipProvider key={idx}>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex-1 flex flex-col gap-0.5">
                  {/* Best fitness bar */}
                  <div
                    className="w-full bg-gradient-to-t from-emerald-600 to-emerald-400 rounded-t opacity-90 hover:opacity-100 transition-opacity"
                    style={{
                      height: `${(gen.bestFitness / maxFitness) * 100}%`,
                      minHeight: '2px',
                    }}
                  />
                  {/* Average fitness bar */}
                  <div
                    className="w-full bg-slate-600 rounded-t"
                    style={{
                      height: `${(gen.avgFitness / maxFitness) * 50}%`,
                      minHeight: '1px',
                    }}
                  />
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <div className="text-xs space-y-1">
                  <div>Generation {gen.generation}</div>
                  <div>Best: {gen.bestFitness.toFixed(4)}</div>
                  <div>Avg: {gen.avgFitness.toFixed(4)}</div>
                  <div>Diversity: {(gen.diversity * 100).toFixed(1)}%</div>
                </div>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-xs text-slate-400">
        <div className="flex items-center gap-1.5">
          <div className="h-2 w-4 bg-emerald-500 rounded" />
          <span>Best Fitness</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-2 w-4 bg-slate-600 rounded" />
          <span>Average</span>
        </div>
      </div>
    </div>
  );
}

interface PopulationDiversityProps {
  diversity: number;
  populationSize: number;
  eliteCount: number;
}

function PopulationDiversity({ diversity, populationSize, eliteCount }: PopulationDiversityProps) {
  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="text-center p-3 rounded-lg bg-slate-800/30">
        <div className="text-2xl font-semibold text-slate-200">{populationSize}</div>
        <div className="text-xs text-slate-500">Population Size</div>
      </div>
      <div className="text-center p-3 rounded-lg bg-slate-800/30">
        <div className="text-2xl font-semibold text-amber-400">{eliteCount}</div>
        <div className="text-xs text-slate-500">Elite Count</div>
      </div>
      <div className="text-center p-3 rounded-lg bg-slate-800/30">
        <div className="text-2xl font-semibold text-cyan-400">
          {(diversity * 100).toFixed(0)}%
        </div>
        <div className="text-xs text-slate-500">Diversity</div>
      </div>
    </div>
  );
}

// ==============================================================================
// Main Component
// ==============================================================================

export interface MutationOptimizerUIProps {
  mutationHistory: MutationHistory[];
  convergenceMetrics?: ConvergenceMetrics[];
  currentGeneration?: number;
  populationSize?: number;
  eliteCount?: number;
  isRunning?: boolean;
  className?: string;
}

export function MutationOptimizerUI({
  mutationHistory,
  convergenceMetrics = [],
  currentGeneration = 0,
  populationSize = 50,
  eliteCount = 5,
  isRunning = false,
  className,
}: MutationOptimizerUIProps) {
  const [activeTab, setActiveTab] = useState('operators');

  // Compute operator statistics from mutation history
  const operatorStats = useMemo<MutationOperator[]>(() => {
    const stats: Record<string, { count: number; successes: number; totalGain: number }> = {};

    mutationHistory.forEach(mutation => {
      const op = mutation.operator || 'unknown';
      if (!stats[op]) {
        stats[op] = { count: 0, successes: 0, totalGain: 0 };
      }
      stats[op].count++;
      const gain = (mutation.fitness_after ?? 0) - (mutation.fitness_before ?? 0);
      stats[op].totalGain += gain;
      if (gain > 0) {
        stats[op].successes++;
      }
    });

    return Object.entries(stats)
      .map(([name, data]) => ({
        name,
        applications: data.count,
        successRate: data.count > 0 ? data.successes / data.count : 0,
        avgFitnessGain: data.count > 0 ? data.totalGain / data.count : 0,
        color: OPERATOR_COLORS[name] || OPERATOR_COLORS.default,
      }))
      .sort((a, b) => b.applications - a.applications);
  }, [mutationHistory]);

  // Compute generation statistics
  const generationStats = useMemo<GenerationStats[]>(() => {
    const byGeneration: Record<number, MutationHistory[]> = {};
    mutationHistory.forEach(m => {
      if (!byGeneration[m.generation]) {
        byGeneration[m.generation] = [];
      }
      byGeneration[m.generation].push(m);
    });

    return Object.entries(byGeneration)
      .map(([gen, mutations]) => {
        const fitnessValues = mutations.map(m => m.fitness_after);
        return {
          generation: parseInt(gen),
          bestFitness: Math.max(...fitnessValues),
          avgFitness: fitnessValues.reduce((a, b) => a + b, 0) / fitnessValues.length,
          diversity: 0.8 - (parseInt(gen) * 0.02), // Placeholder diversity calculation
        };
      })
      .sort((a, b) => a.generation - b.generation);
  }, [mutationHistory]);

  // Get latest convergence metrics
  const latestConvergence = convergenceMetrics[convergenceMetrics.length - 1];
  const currentDiversity = latestConvergence?.improvement_rate ?? 0.8;

  const maxApplications = Math.max(...operatorStats.map(o => o.applications), 1);

  // Summary stats
  const totalMutations = mutationHistory.length;
  const successfulMutations = mutationHistory.filter(
    m => m.fitness_after > m.fitness_before
  ).length;
  const avgImprovement = totalMutations > 0
    ? mutationHistory.reduce((sum, m) => sum + (m.fitness_after - m.fitness_before), 0) / totalMutations
    : 0;

  return (
    <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold text-slate-200 flex items-center gap-2">
            <Dna className="h-5 w-5" />
            Mutation Optimizer
          </CardTitle>
          <div className="flex items-center gap-2">
            {isRunning && (
              <Badge variant="outline" className="gap-1 border-emerald-500/30 text-emerald-400">
                <Activity className="h-3 w-3 animate-pulse" />
                Running
              </Badge>
            )}
            <Badge variant="outline" className="text-xs">
              Gen {currentGeneration}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Quick Stats */}
        <div className="grid grid-cols-4 gap-3">
          <div className="text-center p-2 rounded-lg bg-slate-800/40">
            <div className="text-xl font-semibold text-slate-200">{totalMutations}</div>
            <div className="text-xs text-slate-500">Total</div>
          </div>
          <div className="text-center p-2 rounded-lg bg-slate-800/40">
            <div className="text-xl font-semibold text-emerald-400">{successfulMutations}</div>
            <div className="text-xs text-slate-500">Successful</div>
          </div>
          <div className="text-center p-2 rounded-lg bg-slate-800/40">
            <div className="text-xl font-semibold text-slate-200">
              {totalMutations > 0 ? ((successfulMutations / totalMutations) * 100).toFixed(0) : 0}%
            </div>
            <div className="text-xs text-slate-500">Success Rate</div>
          </div>
          <div className="text-center p-2 rounded-lg bg-slate-800/40">
            <div className={cn(
              "text-xl font-semibold",
              avgImprovement > 0 ? 'text-emerald-400' : avgImprovement < 0 ? 'text-red-400' : 'text-slate-400'
            )}>
              {avgImprovement > 0 ? '+' : ''}{avgImprovement.toFixed(4)}
            </div>
            <div className="text-xs text-slate-500">Avg Δ Fitness</div>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full bg-slate-800/50">
            <TabsTrigger value="operators" className="flex-1 gap-1">
              <Shuffle className="h-3 w-3" />
              Operators
            </TabsTrigger>
            <TabsTrigger value="history" className="flex-1 gap-1">
              <Clock className="h-3 w-3" />
              History
            </TabsTrigger>
            <TabsTrigger value="evolution" className="flex-1 gap-1">
              <BarChart3 className="h-3 w-3" />
              Evolution
            </TabsTrigger>
          </TabsList>

          <TabsContent value="operators" className="mt-4 space-y-3">
            {operatorStats.length > 0 ? (
              operatorStats.map((operator) => (
                <OperatorBar
                  key={operator.name}
                  operator={operator}
                  maxApplications={maxApplications}
                />
              ))
            ) : (
              <div className="text-center py-8 text-slate-500 text-sm">
                No operator data available
              </div>
            )}
          </TabsContent>

          <TabsContent value="history" className="mt-4">
            <MutationHistoryTable history={mutationHistory} />
          </TabsContent>

          <TabsContent value="evolution" className="mt-4 space-y-4">
            <GenerationChart generations={generationStats} />
            <PopulationDiversity
              diversity={currentDiversity}
              populationSize={populationSize}
              eliteCount={eliteCount}
            />
          </TabsContent>
        </Tabs>

        {/* Convergence Progress */}
        {latestConvergence && (
          <div className="pt-2 border-t border-slate-700/50 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-400 flex items-center gap-1">
                <Target className="h-4 w-4" />
                Convergence Progress
              </span>
              <span className="font-mono text-slate-300">
                {(latestConvergence.current_fitness ?? 0).toFixed(4)} / {latestConvergence.convergence_threshold?.toFixed(4) ?? '—'}
              </span>
            </div>
            <Progress
              value={
                latestConvergence.convergence_threshold
                  ? Math.min(((latestConvergence.current_fitness ?? 0) / latestConvergence.convergence_threshold) * 100, 100)
                  : 0
              }
              className="h-2"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default MutationOptimizerUI;
