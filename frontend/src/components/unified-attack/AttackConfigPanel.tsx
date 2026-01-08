/**
 * AttackConfigPanel Component
 *
 * Interactive configuration panel for multi-vector attacks with
 * strategy selection, parameter tuning, and query input.
 */

"use client";

import * as React from 'react';
import { useState, useCallback } from 'react';
import {
  Play,
  Settings2,
  Sliders,
  ChevronRight,
  Layers,
  Shuffle,
  Repeat,
  GitMerge,
  Sparkles,
  Info,
  AlertCircle,
  Loader2,
  Copy,
  Check,
  Wand2,
  RefreshCw,
} from 'lucide-react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';

import {
  useStrategies,
  useValidateConfig,
  useExecuteAttack,
  useExecuteSequentialAttack,
  useExecuteParallelAttack,
  useExecuteIterativeAttack,
  useExecuteAdaptiveAttack,
} from '@/hooks/use-unified-attack';
import { CompositionStrategy } from '@/types/unified-attack-types';
import type {
  StrategyInfo,
  UnifiedAttackRequest,
} from '@/types/unified-attack-types';

// ==============================================================================
// Constants
// ==============================================================================

const STRATEGY_ICONS: Record<CompositionStrategy, React.ReactNode> = {
  [CompositionStrategy.SEQUENTIAL_EXTEND_FIRST]: <ChevronRight className="h-4 w-4" />,
  [CompositionStrategy.SEQUENTIAL_AUTODAN_FIRST]: <ChevronRight className="h-4 w-4 rotate-180" />,
  [CompositionStrategy.PARALLEL]: <Layers className="h-4 w-4" />,
  [CompositionStrategy.ITERATIVE]: <Repeat className="h-4 w-4" />,
  [CompositionStrategy.ADAPTIVE]: <Sparkles className="h-4 w-4" />,
  [CompositionStrategy.WEIGHTED]: <Sliders className="h-4 w-4" />,
  [CompositionStrategy.ENSEMBLE]: <GitMerge className="h-4 w-4" />,
};

const STRATEGY_DESCRIPTIONS: Record<CompositionStrategy, string> = {
  [CompositionStrategy.SEQUENTIAL_EXTEND_FIRST]: 'Apply obfuscation first, then mutation',
  [CompositionStrategy.SEQUENTIAL_AUTODAN_FIRST]: 'Apply mutation first, then obfuscation',
  [CompositionStrategy.PARALLEL]: 'Run both vectors simultaneously and blend results',
  [CompositionStrategy.ITERATIVE]: 'Alternate between vectors for multiple iterations',
  [CompositionStrategy.ADAPTIVE]: 'Automatically select optimal strategy based on context',
  [CompositionStrategy.WEIGHTED]: 'Blend vectors with configurable weight distribution',
  [CompositionStrategy.ENSEMBLE]: 'Combine multiple strategies for best results',
};

// ==============================================================================
// Sub-components
// ==============================================================================

interface StrategyCardProps {
  strategy: StrategyInfo;
  selected: boolean;
  onSelect: () => void;
}

function StrategyCard({ strategy, selected, onSelect }: StrategyCardProps) {
  const icon = STRATEGY_ICONS[strategy.name as CompositionStrategy] || <Settings2 className="h-4 w-4" />;

  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full p-3 rounded-lg border text-left transition-all",
        "hover:border-slate-600",
        selected
          ? "border-blue-500/50 bg-blue-500/10"
          : "border-slate-700/50 bg-slate-800/30"
      )}
    >
      <div className="flex items-start gap-3">
        <div className={cn(
          "p-2 rounded-lg",
          selected ? "bg-blue-500/20 text-blue-400" : "bg-slate-700/50 text-slate-400"
        )}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn(
              "text-sm font-medium",
              selected ? "text-blue-300" : "text-slate-300"
            )}>
              {strategy.name.replace(/_/g, ' ')}
            </span>
            {strategy.recommended && (
              <Badge className="text-[10px] py-0 px-1 bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                Recommended
              </Badge>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-1 line-clamp-2">
            {strategy.description || STRATEGY_DESCRIPTIONS[strategy.name as CompositionStrategy]}
          </p>
        </div>
      </div>
    </button>
  );
}

interface ParameterSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  tooltip?: string;
  format?: (value: number) => string;
}

function ParameterSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  tooltip,
  format = (v) => v.toString(),
}: ParameterSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Label className="text-sm text-slate-400">{label}</Label>
          {tooltip && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3 w-3 text-slate-500 cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs max-w-[200px]">{tooltip}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <span className="text-sm font-mono text-slate-300">{format(value)}</span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
    </div>
  );
}

interface QueryInputProps {
  value: string;
  onChange: (value: string) => void;
  onGenerate?: () => void;
  isGenerating?: boolean;
}

function QueryInput({ value, onChange, onGenerate, isGenerating }: QueryInputProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success('Copied to clipboard');
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm text-slate-300">Attack Query</Label>
        <div className="flex items-center gap-1">
          {onGenerate && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onGenerate}
              disabled={isGenerating}
              className="h-7 px-2 text-xs"
            >
              {isGenerating ? (
                <Loader2 className="h-3 w-3 animate-spin mr-1" />
              ) : (
                <Wand2 className="h-3 w-3 mr-1" />
              )}
              Generate
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-7 px-2"
          >
            {copied ? (
              <Check className="h-3 w-3 text-emerald-400" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </Button>
        </div>
      </div>
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Enter the query to be transformed..."
        className="min-h-[100px] bg-slate-800/50 border-slate-700 text-slate-200 placeholder:text-slate-500 resize-none"
      />
      <div className="flex justify-between text-xs text-slate-500">
        <span>{value.length} characters</span>
        <span>{value.split(/\s+/).filter(Boolean).length} words</span>
      </div>
    </div>
  );
}

// ==============================================================================
// Main Component
// ==============================================================================

export interface AttackConfigPanelProps {
  sessionId: string | null;
  onAttackStarted?: (attackId: string) => void;
  onAttackCompleted?: (result: any) => void;
  className?: string;
}

export function AttackConfigPanel({
  sessionId,
  onAttackStarted,
  onAttackCompleted,
  className,
}: AttackConfigPanelProps) {
  // Form state
  const [query, setQuery] = useState('');
  const [selectedStrategy, setSelectedStrategy] = useState<CompositionStrategy>(CompositionStrategy.ADAPTIVE);

  // Strategy-specific parameters
  const [blendWeight, setBlendWeight] = useState(0.5);
  const [iterations, setIterations] = useState(3);
  const [extendFirst, setExtendFirst] = useState(true);

  // Advanced parameters
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [populationSize, setPopulationSize] = useState(50);
  const [mutationRate, setMutationRate] = useState(0.3);
  const [semanticThreshold, setSemanticThreshold] = useState(0.7);

  // Queries and mutations
  const { data: strategies = [], isLoading: strategiesLoading } = useStrategies();
  const validateConfigMutation = useValidateConfig();
  const executeAttackMutation = useExecuteAttack();
  const executeSequentialMutation = useExecuteSequentialAttack();
  const executeParallelMutation = useExecuteParallelAttack();
  const executeIterativeMutation = useExecuteIterativeAttack();
  const executeAdaptiveMutation = useExecuteAdaptiveAttack();

  const isExecuting =
    executeAttackMutation.isPending ||
    executeSequentialMutation.isPending ||
    executeParallelMutation.isPending ||
    executeIterativeMutation.isPending ||
    executeAdaptiveMutation.isPending;

  // Execute attack handler
  const handleExecuteAttack = useCallback(async () => {
    if (!sessionId) {
      toast.error('No active session. Create a session first.');
      return;
    }

    if (!query.trim()) {
      toast.error('Please enter a query');
      return;
    }

    try {
      let result;

      // Select the appropriate mutation based on strategy
      switch (selectedStrategy) {
        case CompositionStrategy.SEQUENTIAL_EXTEND_FIRST:
        case CompositionStrategy.SEQUENTIAL_AUTODAN_FIRST:
          result = await executeSequentialMutation.mutateAsync({
            session_id: sessionId,
            query: query.trim(),
            extend_first: selectedStrategy === CompositionStrategy.SEQUENTIAL_EXTEND_FIRST,
          });
          break;

        case CompositionStrategy.PARALLEL:
          result = await executeParallelMutation.mutateAsync({
            session_id: sessionId,
            query: query.trim(),
            blend_weight: blendWeight,
          });
          break;

        case CompositionStrategy.ITERATIVE:
          result = await executeIterativeMutation.mutateAsync({
            session_id: sessionId,
            query: query.trim(),
            iterations,
          });
          break;

        case CompositionStrategy.ADAPTIVE:
          result = await executeAdaptiveMutation.mutateAsync({
            session_id: sessionId,
            query: query.trim(),
            target_metrics: {
              min_fitness: semanticThreshold,
              max_tokens: maxTokens,
            },
          });
          break;

        default:
          result = await executeAttackMutation.mutateAsync({
            session_id: sessionId,
            query: query.trim(),
            strategy: selectedStrategy,
            parameters: {
              temperature,
              max_tokens: maxTokens,
              population_size: populationSize,
              mutation_rate: mutationRate,
              semantic_threshold: semanticThreshold,
            },
          });
      }

      toast.success('Attack executed successfully');
      onAttackStarted?.(result.attack_id);
      onAttackCompleted?.(result);
    } catch (error) {
      toast.error('Attack execution failed');
    }
  }, [
    sessionId,
    query,
    selectedStrategy,
    blendWeight,
    iterations,
    temperature,
    maxTokens,
    populationSize,
    mutationRate,
    semanticThreshold,
    executeAttackMutation,
    executeSequentialMutation,
    executeParallelMutation,
    executeIterativeMutation,
    executeAdaptiveMutation,
    onAttackStarted,
    onAttackCompleted,
  ]);

  return (
    <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg font-semibold text-slate-200 flex items-center gap-2">
          <Settings2 className="h-5 w-5" />
          Attack Configuration
        </CardTitle>
        <CardDescription className="text-slate-400">
          Configure and execute multi-vector attacks
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Query Input */}
        <QueryInput
          value={query}
          onChange={setQuery}
        />

        {/* Strategy Selection */}
        <div className="space-y-3">
          <Label className="text-sm text-slate-300">Composition Strategy</Label>
          {strategiesLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-slate-500" />
            </div>
          ) : (
            <ScrollArea className="h-[200px] pr-4">
              <div className="space-y-2">
                {strategies.map((strategy) => (
                  <StrategyCard
                    key={strategy.name}
                    strategy={strategy}
                    selected={selectedStrategy === strategy.name}
                    onSelect={() => setSelectedStrategy(strategy.name as CompositionStrategy)}
                  />
                ))}
              </div>
            </ScrollArea>
          )}
        </div>

        {/* Strategy-specific Parameters */}
        {selectedStrategy === CompositionStrategy.PARALLEL && (
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 space-y-3">
            <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Parallel Parameters
            </h4>
            <ParameterSlider
              label="Blend Weight"
              value={blendWeight}
              onChange={setBlendWeight}
              min={0}
              max={1}
              step={0.1}
              tooltip="Weight distribution between obfuscation and mutation (0 = pure obfuscation, 1 = pure mutation)"
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
          </div>
        )}

        {selectedStrategy === CompositionStrategy.ITERATIVE && (
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 space-y-3">
            <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <Repeat className="h-4 w-4" />
              Iterative Parameters
            </h4>
            <ParameterSlider
              label="Iterations"
              value={iterations}
              onChange={setIterations}
              min={1}
              max={10}
              step={1}
              tooltip="Number of alternating iterations between vectors"
            />
          </div>
        )}

        {(selectedStrategy === CompositionStrategy.SEQUENTIAL_EXTEND_FIRST || selectedStrategy === CompositionStrategy.SEQUENTIAL_AUTODAN_FIRST) && (
          <div className="p-4 rounded-lg bg-slate-800/30 border border-slate-700/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ChevronRight className="h-4 w-4 text-slate-400" />
                <span className="text-sm text-slate-300">Extend First</span>
              </div>
              <Switch
                checked={selectedStrategy === CompositionStrategy.SEQUENTIAL_EXTEND_FIRST}
                onCheckedChange={(checked) =>
                  setSelectedStrategy(checked ? CompositionStrategy.SEQUENTIAL_EXTEND_FIRST : CompositionStrategy.SEQUENTIAL_AUTODAN_FIRST)
                }
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              {selectedStrategy === CompositionStrategy.SEQUENTIAL_EXTEND_FIRST
                ? 'Obfuscation → Mutation'
                : 'Mutation → Obfuscation'}
            </p>
          </div>
        )}

        {/* Advanced Parameters */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="advanced" className="border-slate-700/50">
            <AccordionTrigger className="text-sm text-slate-400 hover:text-slate-300">
              <div className="flex items-center gap-2">
                <Sliders className="h-4 w-4" />
                Advanced Parameters
              </div>
            </AccordionTrigger>
            <AccordionContent className="space-y-4 pt-4">
              <ParameterSlider
                label="Temperature"
                value={temperature}
                onChange={setTemperature}
                min={0}
                max={2}
                step={0.1}
                tooltip="Controls randomness in generation (higher = more creative)"
                format={(v) => v.toFixed(1)}
              />

              <ParameterSlider
                label="Max Tokens"
                value={maxTokens}
                onChange={setMaxTokens}
                min={256}
                max={4096}
                step={256}
                tooltip="Maximum tokens in generated output"
              />

              <ParameterSlider
                label="Population Size"
                value={populationSize}
                onChange={setPopulationSize}
                min={10}
                max={200}
                step={10}
                tooltip="Number of candidates in genetic algorithm population"
              />

              <ParameterSlider
                label="Mutation Rate"
                value={mutationRate}
                onChange={setMutationRate}
                min={0.1}
                max={0.9}
                step={0.1}
                tooltip="Probability of applying mutations"
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />

              <ParameterSlider
                label="Semantic Threshold"
                value={semanticThreshold}
                onChange={setSemanticThreshold}
                min={0.3}
                max={0.95}
                step={0.05}
                tooltip="Minimum semantic similarity to preserve meaning"
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Execute Button */}
        <div className="pt-2">
          <Button
            onClick={handleExecuteAttack}
            disabled={!sessionId || !query.trim() || isExecuting}
            className="w-full gap-2"
            size="lg"
          >
            {isExecuting ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Executing Attack...
              </>
            ) : (
              <>
                <Play className="h-5 w-5" />
                Execute Attack
              </>
            )}
          </Button>

          {!sessionId && (
            <p className="text-xs text-amber-400 mt-2 flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />
              Create a session first to execute attacks
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default AttackConfigPanel;
