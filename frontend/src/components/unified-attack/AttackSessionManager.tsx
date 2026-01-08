/**
 * AttackSessionManager Component
 *
 * Session lifecycle control for multi-vector attacks with
 * real-time status monitoring and budget tracking.
 */

"use client";

import * as React from 'react';
import { useMemo } from 'react';
import { useState } from 'react';
import {
  Play,
  Pause,
  Square,
  Settings,
  Clock,
  Zap,
  Target,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { toast } from 'sonner';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { ScrollArea } from '@/components/ui/scroll-area';

import {
  useCreateSession,
  useFinalizeSession,
  useSession,
  useBudgetStatus,
  usePresets,
} from '@/hooks/use-unified-attack';
import { SessionStatus } from '@/types/unified-attack-types';
import type {
  SessionStatusResponse,
  BudgetStatusResponse,
  PresetConfig,
  CreateSessionRequest,
} from '@/types/unified-attack-types';

// ==============================================================================
// Constants
// ==============================================================================

const STATUS_STYLES: Record<SessionStatus, { color: string; icon: React.ReactNode }> = {
  [SessionStatus.ACTIVE]: { color: 'bg-emerald-500', icon: <Play className="h-3 w-3" /> },
  [SessionStatus.PAUSED]: { color: 'bg-amber-500', icon: <Pause className="h-3 w-3" /> },
  [SessionStatus.FINALIZED]: { color: 'bg-slate-500', icon: <CheckCircle2 className="h-3 w-3" /> },
  [SessionStatus.EXPIRED]: { color: 'bg-red-500', icon: <XCircle className="h-3 w-3" /> },
};

// ==============================================================================
// Sub-components
// ==============================================================================

interface StatusBadgeProps {
  status: SessionStatus;
}

function StatusBadge({ status }: StatusBadgeProps) {
  const style = STATUS_STYLES[status];
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 px-2 py-0.5 text-xs font-medium uppercase tracking-wider",
        "border-none text-white",
        style.color
      )}
    >
      {style.icon}
      {status}
    </Badge>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subValue?: string;
  className?: string;
}

function StatCard({ icon, label, value, subValue, className }: StatCardProps) {
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-800/50 text-slate-400">
        {icon}
      </div>
      <div className="flex flex-col">
        <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{label}</span>
        <span className="text-lg font-semibold text-slate-200">{value}</span>
        {subValue && <span className="text-xs text-slate-500">{subValue}</span>}
      </div>
    </div>
  );
}

interface BudgetGaugeProps {
  budget: BudgetStatusResponse | null;
  isLoading: boolean;
}

function BudgetGauge({ budget, isLoading }: BudgetGaugeProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-slate-500" />
      </div>
    );
  }

  if (!budget) {
    return (
      <div className="text-center text-sm text-slate-500 py-4">
        No budget data available
      </div>
    );
  }

  const usedPercent = budget.budget_remaining
    ? Math.round(((budget.budget_remaining.max_tokens - budget.budget_remaining.tokens) / budget.budget_remaining.max_tokens) * 100)
    : 0;

  const tokenUsed = budget.budget_remaining
    ? budget.budget_remaining.max_tokens - budget.budget_remaining.tokens
    : 0;
  const tokenMax = budget.budget_remaining?.max_tokens ?? 0;

  const costUsed = budget.budget_remaining
    ? budget.budget_remaining.max_cost - budget.budget_remaining.cost
    : 0;
  const costMax = budget.budget_remaining?.max_cost ?? 0;

  return (
    <div className="space-y-4">
      {/* Token Usage */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Token Usage</span>
          <span className="font-mono text-slate-300">
            {tokenUsed.toLocaleString()} / {tokenMax.toLocaleString()}
          </span>
        </div>
        <Progress value={usedPercent} className="h-2" />
      </div>

      {/* Cost Usage */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Cost Budget</span>
          <span className="font-mono text-slate-300">
            ${costUsed.toFixed(4)} / ${costMax.toFixed(4)}
          </span>
        </div>
        <Progress
          value={costMax > 0 ? (costUsed / costMax) * 100 : 0}
          className="h-2"
        />
      </div>

      {/* Warnings */}
      {budget.warnings && budget.warnings.length > 0 && (
        <div className="mt-4 space-y-2">
          {budget.warnings.map((warning, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2 rounded-md bg-amber-500/10 p-2 text-xs text-amber-400"
            >
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ==============================================================================
// Create Session Dialog
// ==============================================================================

interface CreateSessionDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreateSession: (request: CreateSessionRequest) => Promise<void>;
  presets: PresetConfig[];
  isLoading: boolean;
}

function CreateSessionDialog({
  open,
  onOpenChange,
  onCreateSession,
  presets,
  isLoading,
}: CreateSessionDialogProps) {
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [maxTokens, setMaxTokens] = useState(100000);
  const [maxCost, setMaxCost] = useState(10);
  const [maxTime, setMaxTime] = useState(3600);

  const handleCreate = async () => {
    const preset = presets.find(p => p.name === selectedPreset);

    await onCreateSession({
      config: preset?.config ?? {
        model_id: 'gemini-2.0-flash',
        extend_first: true,
        autodan_iterations: 5,
        combined_iterations: 3,
        population_size: 50,
        initial_temperature: 1.0,
        final_temperature: 0.1,
        semantic_similarity_threshold: 0.7,
        token_efficiency_target: 0.8,
        max_concurrent_tasks: 4,
      },
      budget: {
        max_tokens: maxTokens,
        max_cost: maxCost,
        max_time_seconds: maxTime,
        max_requests: 1000,
      },
    });

    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-slate-900 border-slate-800 text-slate-100 sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create Attack Session</DialogTitle>
          <DialogDescription className="text-slate-400">
            Configure your multi-vector attack session parameters.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Preset Selection */}
          <div className="space-y-2">
            <Label htmlFor="preset" className="text-slate-300">Configuration Preset</Label>
            <Select value={selectedPreset} onValueChange={setSelectedPreset}>
              <SelectTrigger className="bg-slate-800 border-slate-700">
                <SelectValue placeholder="Select a preset..." />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-700">
                {presets.map((preset) => (
                  <SelectItem key={preset.name} value={preset.name}>
                    {preset.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Budget Configuration */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-slate-300">Resource Budget</h4>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-slate-400">Max Tokens</Label>
                <span className="text-sm font-mono text-slate-300">{maxTokens.toLocaleString()}</span>
              </div>
              <Slider
                value={[maxTokens]}
                onValueChange={([val]) => setMaxTokens(val)}
                max={1000000}
                min={10000}
                step={10000}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-slate-400">Max Cost ($)</Label>
                <span className="text-sm font-mono text-slate-300">${maxCost.toFixed(2)}</span>
              </div>
              <Slider
                value={[maxCost]}
                onValueChange={([val]) => setMaxCost(val)}
                max={100}
                min={1}
                step={1}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-slate-400">Max Time (seconds)</Label>
                <span className="text-sm font-mono text-slate-300">{maxTime}s</span>
              </div>
              <Slider
                value={[maxTime]}
                onValueChange={([val]) => setMaxTime(val)}
                max={7200}
                min={300}
                step={300}
                className="w-full"
              />
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={isLoading}>
            {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Create Session
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ==============================================================================
// Main Component
// ==============================================================================

export interface AttackSessionManagerProps {
  sessionId: string | null;
  onSessionCreated?: (sessionId: string) => void;
  onSessionFinalized?: (sessionId: string) => void;
  className?: string;
}

export function AttackSessionManager({
  sessionId,
  onSessionCreated,
  onSessionFinalized,
  className,
}: AttackSessionManagerProps) {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  // Queries
  const { data: session, isLoading: sessionLoading, refetch: refetchSession } = useSession(sessionId, {
    refetchInterval: 5000,
  });
  const { data: budget, isLoading: budgetLoading } = useBudgetStatus(sessionId, {
    refetchInterval: 5000,
  });
  const { data: presets = [] } = usePresets();

  // Mutations
  const createSessionMutation = useCreateSession();
  const finalizeSessionMutation = useFinalizeSession();

  // Handlers
  const handleCreateSession = async (request: CreateSessionRequest) => {
    try {
      const result = await createSessionMutation.mutateAsync(request);
      toast.success('Session created successfully');
      onSessionCreated?.(result.session_id);
    } catch (error) {
      toast.error('Failed to create session');
      throw error;
    }
  };

  const handleFinalizeSession = async () => {
    if (!sessionId) return;

    try {
      await finalizeSessionMutation.mutateAsync(sessionId);
      toast.success('Session finalized');
      onSessionFinalized?.(sessionId);
    } catch (error) {
      toast.error('Failed to finalize session');
    }
  };

  // Render loading state
  if (sessionLoading && sessionId) {
    return (
      <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-slate-500" />
        </CardContent>
      </Card>
    );
  }

  // Render no session state
  if (!session) {
    return (
      <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
        <CardHeader className="pb-4">
          <CardTitle className="text-lg font-semibold text-slate-200">
            Attack Session
          </CardTitle>
          <CardDescription className="text-slate-400">
            No active session. Create one to start multi-vector attacks.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button
            onClick={() => setCreateDialogOpen(true)}
            className="w-full gap-2"
          >
            <Zap className="h-4 w-4" />
            Create New Session
          </Button>

          <CreateSessionDialog
            open={createDialogOpen}
            onOpenChange={setCreateDialogOpen}
            onCreateSession={handleCreateSession}
            presets={presets}
            isLoading={createSessionMutation.isPending}
          />
        </CardContent>
      </Card>
    );
  }

  // Calculate session duration
  const startTime = new Date(session.created_at);
  const duration = useMemo(() => Math.round((Date.now() - startTime.getTime()) / 1000), [session.created_at]);
  const durationStr = `${Math.floor(duration / 60)}m ${duration % 60}s`;

  return (
    <Card className={cn("bg-slate-900/50 border-slate-800", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-lg font-semibold text-slate-200">
              Session
            </CardTitle>
            <StatusBadge status={session.status} />
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => refetchSession()}
              className="h-8 w-8"
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setCreateDialogOpen(true)}
              className="h-8 w-8"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <CardDescription className="text-slate-500 font-mono text-xs">
          {session.session_id}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Session Stats */}
        <div className="grid grid-cols-2 gap-4">
          <StatCard
            icon={<Target className="h-5 w-5" />}
            label="Total Attacks"
            value={session.total_attacks}
            subValue={`${session.successful_attacks} successful`}
          />
          <StatCard
            icon={<TrendingUp className="h-5 w-5" />}
            label="Best Fitness"
            value={session.best_fitness?.toFixed(3) ?? '—'}
          />
          <StatCard
            icon={<Clock className="h-5 w-5" />}
            label="Duration"
            value={durationStr}
          />
          <StatCard
            icon={<Zap className="h-5 w-5" />}
            label="Success Rate"
            value={
              session.total_attacks > 0
                ? `${Math.round((session.successful_attacks / session.total_attacks) * 100)}%`
                : '—'
            }
          />
        </div>

        {/* Budget Gauge */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-slate-400">Resource Budget</h4>
          <BudgetGauge budget={budget ?? null} isLoading={budgetLoading} />
        </div>

        {/* Session Actions */}
        <div className="flex gap-2 pt-2">
          {session.status === SessionStatus.ACTIVE && (
            <Button
              variant="outline"
              onClick={handleFinalizeSession}
              disabled={finalizeSessionMutation.isPending}
              className="flex-1 gap-2"
            >
              {finalizeSessionMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Square className="h-4 w-4" />
              )}
              Finalize Session
            </Button>
          )}

          <Button
            onClick={() => setCreateDialogOpen(true)}
            className="flex-1 gap-2"
          >
            <Zap className="h-4 w-4" />
            New Session
          </Button>
        </div>
      </CardContent>

      <CreateSessionDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onCreateSession={handleCreateSession}
        presets={presets}
        isLoading={createSessionMutation.isPending}
      />
    </Card>
  );
}

export default AttackSessionManager;
