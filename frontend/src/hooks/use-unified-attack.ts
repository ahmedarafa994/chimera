/**
 * Unified Attack Hooks
 * 
 * React hooks for the multi-vector attack framework with
 * TanStack Query integration, WebSocket support, and state management.
 */

"use client";

import { useState, useCallback, useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { unifiedAttackApi } from '@/lib/api/unified-attack-api';
import type {
  // Types
  CompositionStrategy,
  SessionStatus,
  // Request types
  CreateSessionRequest,
  UnifiedAttackRequest,
  BatchAttackRequest,
  SequentialAttackRequest,
  ParallelAttackRequest,
  IterativeAttackRequest,
  AdaptiveAttackRequest,
  EvaluationRequest,
  AllocationRequest,
  BenchmarkRequest,
  ValidationRequest,
  // Response types
  SessionResponse,
  SessionStatusResponse,
  SessionSummaryResponse,
  AttackResponse,
  BatchAttackResponse,
  StrategyInfo,
  PresetConfig,
  ValidationResult,
  EvaluationResponse,
  ParetoFrontResponse,
  ResourceUsageResponse,
  BudgetStatusResponse,
  AllocationResponse,
  BenchmarkResponse,
  BenchmarkDataset,
  // Event types
  UnifiedAttackEvent,
  AttackProgressEvent,
  ReasoningStep,
  MutationHistory,
  ConvergenceMetrics,
  ReasoningChain,
} from '@/types/unified-attack-types';

// ==============================================================================
// Query Keys
// ==============================================================================

export const unifiedAttackKeys = {
  all: ['unified-attack'] as const,
  sessions: () => [...unifiedAttackKeys.all, 'sessions'] as const,
  session: (id: string) => [...unifiedAttackKeys.sessions(), id] as const,
  strategies: () => [...unifiedAttackKeys.all, 'strategies'] as const,
  presets: () => [...unifiedAttackKeys.all, 'presets'] as const,
  resources: (sessionId: string) => [...unifiedAttackKeys.all, 'resources', sessionId] as const,
  budget: (sessionId: string) => [...unifiedAttackKeys.all, 'budget', sessionId] as const,
  pareto: (sessionId: string) => [...unifiedAttackKeys.all, 'pareto', sessionId] as const,
  benchmarkDatasets: () => [...unifiedAttackKeys.all, 'benchmark-datasets'] as const,
};

// ==============================================================================
// Session Management Hooks
// ==============================================================================

/**
 * Hook to create a new attack session
 */
export function useCreateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateSessionRequest) => unifiedAttackApi.createSession(request),
    onSuccess: (data) => {
      queryClient.setQueryData(unifiedAttackKeys.session(data.session_id), data);
    },
  });
}

/**
 * Hook to get session status
 */
export function useSession(sessionId: string | null, options?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: unifiedAttackKeys.session(sessionId ?? ''),
    queryFn: () => unifiedAttackApi.getSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: options?.refetchInterval,
  });
}

/**
 * Hook to finalize session
 */
export function useFinalizeSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) => unifiedAttackApi.finalizeSession(sessionId),
    onSuccess: (data, sessionId) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(sessionId) });
    },
  });
}

// ==============================================================================
// Attack Execution Hooks
// ==============================================================================

/**
 * Hook to execute a unified attack
 */
export function useExecuteAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: UnifiedAttackRequest) => unifiedAttackApi.executeAttack(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(data.session_id) });
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.resources(data.session_id) });
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.budget(data.session_id) });
    },
  });
}

/**
 * Hook to execute batch attacks
 */
export function useExecuteBatchAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BatchAttackRequest) => unifiedAttackApi.executeBatchAttack(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(data.session_id) });
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.resources(data.session_id) });
    },
  });
}

/**
 * Hook to execute sequential attack
 */
export function useExecuteSequentialAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: SequentialAttackRequest) => unifiedAttackApi.executeSequentialAttack(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(data.session_id) });
    },
  });
}

/**
 * Hook to execute parallel attack
 */
export function useExecuteParallelAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ParallelAttackRequest) => unifiedAttackApi.executeParallelAttack(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(data.session_id) });
    },
  });
}

/**
 * Hook to execute iterative attack
 */
export function useExecuteIterativeAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: IterativeAttackRequest) => unifiedAttackApi.executeIterativeAttack(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(data.session_id) });
    },
  });
}

/**
 * Hook to execute adaptive attack
 */
export function useExecuteAdaptiveAttack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: AdaptiveAttackRequest) => unifiedAttackApi.executeAdaptiveAttack(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.session(data.session_id) });
    },
  });
}

// ==============================================================================
// Configuration Hooks
// ==============================================================================

/**
 * Hook to get available strategies
 */
export function useStrategies() {
  return useQuery({
    queryKey: unifiedAttackKeys.strategies(),
    queryFn: () => unifiedAttackApi.getStrategies(),
    staleTime: 1000 * 60 * 60, // 1 hour
  });
}

/**
 * Hook to get attack presets
 */
export function usePresets() {
  return useQuery({
    queryKey: unifiedAttackKeys.presets(),
    queryFn: () => unifiedAttackApi.getPresets(),
    staleTime: 1000 * 60 * 60, // 1 hour
  });
}

/**
 * Hook to validate configuration
 */
export function useValidateConfig() {
  return useMutation({
    mutationFn: (request: ValidationRequest) => unifiedAttackApi.validateConfig(request),
  });
}

// ==============================================================================
// Evaluation Hooks
// ==============================================================================

/**
 * Hook to evaluate attack
 */
export function useEvaluateAttack() {
  return useMutation({
    mutationFn: (request: EvaluationRequest) => unifiedAttackApi.evaluateAttack(request),
  });
}

/**
 * Hook to get Pareto front
 */
export function useParetoFront(sessionId: string | null) {
  return useQuery({
    queryKey: unifiedAttackKeys.pareto(sessionId ?? ''),
    queryFn: () => unifiedAttackApi.getParetoFront(sessionId!),
    enabled: !!sessionId,
  });
}

// ==============================================================================
// Resource Tracking Hooks
// ==============================================================================

/**
 * Hook to get resource usage
 */
export function useResourceUsage(sessionId: string | null, options?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: unifiedAttackKeys.resources(sessionId ?? ''),
    queryFn: () => unifiedAttackApi.getResourceUsage(sessionId!),
    enabled: !!sessionId,
    refetchInterval: options?.refetchInterval,
  });
}

/**
 * Hook to get budget status
 */
export function useBudgetStatus(sessionId: string | null, options?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: unifiedAttackKeys.budget(sessionId ?? ''),
    queryFn: () => unifiedAttackApi.getBudgetStatus(sessionId!),
    enabled: !!sessionId,
    refetchInterval: options?.refetchInterval,
  });
}

/**
 * Hook to allocate resources
 */
export function useAllocateResources() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: AllocationRequest) => unifiedAttackApi.allocateResources(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: unifiedAttackKeys.resources(data.session_id) });
    },
  });
}

// ==============================================================================
// Benchmark Hooks
// ==============================================================================

/**
 * Hook to run benchmark
 */
export function useRunBenchmark() {
  return useMutation({
    mutationFn: (request: BenchmarkRequest) => unifiedAttackApi.runBenchmark(request),
  });
}

/**
 * Hook to get benchmark datasets
 */
export function useBenchmarkDatasets() {
  return useQuery({
    queryKey: unifiedAttackKeys.benchmarkDatasets(),
    queryFn: () => unifiedAttackApi.getBenchmarkDatasets(),
    staleTime: 1000 * 60 * 60, // 1 hour
  });
}

// ==============================================================================
// WebSocket Progress Hook
// ==============================================================================

export interface UseAttackProgressOptions {
  sessionId: string | null;
  onProgress?: (event: AttackProgressEvent) => void;
  onReasoningStep?: (step: ReasoningStep) => void;
  onMutation?: (mutation: MutationHistory) => void;
  onConvergence?: (metrics: ConvergenceMetrics) => void;
  onError?: (error: Error) => void;
}

export interface UseAttackProgressState {
  isConnected: boolean;
  currentAttackId: string | null;
  progress: number;
  phase: string | null;
  reasoningChain: ReasoningChain | null;
  mutationHistory: MutationHistory[];
  convergenceMetrics: ConvergenceMetrics[];
}

/**
 * Hook for WebSocket-based attack progress streaming
 */
export function useAttackProgress(options: UseAttackProgressOptions): UseAttackProgressState {
  const { sessionId, onProgress, onReasoningStep, onMutation, onConvergence, onError } = options;
  
  const [isConnected, setIsConnected] = useState(false);
  const [currentAttackId, setCurrentAttackId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<string | null>(null);
  const [reasoningChain, setReasoningChain] = useState<ReasoningChain | null>(null);
  const [mutationHistory, setMutationHistory] = useState<MutationHistory[]>([]);
  const [convergenceMetrics, setConvergenceMetrics] = useState<ConvergenceMetrics[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!sessionId) return;

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/unified-attack/${sessionId}`;
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log('[UnifiedAttack WS] Connected');
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('[UnifiedAttack WS] Disconnected');
        
        // Attempt reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('[UnifiedAttack WS] Error:', error);
        onError?.(new Error('WebSocket connection error'));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as UnifiedAttackEvent;
          
          switch (data.event_type) {
            case 'attack_progress': {
              const progressEvent = data as AttackProgressEvent;
              setCurrentAttackId(progressEvent.attack_id);
              setProgress(progressEvent.progress_percent);
              setPhase(progressEvent.phase);
              onProgress?.(progressEvent);
              
              // Reset state on new attack
              if (progressEvent.phase === 'started') {
                setReasoningChain(null);
                setMutationHistory([]);
                setConvergenceMetrics([]);
              }
              break;
            }
            
            case 'reasoning_step': {
              const stepEvent = data;
              const step = stepEvent.step;
              onReasoningStep?.(step);
              
              setReasoningChain((prev) => {
                if (!prev) {
                  return {
                    chain_id: `chain-${stepEvent.attack_id}`,
                    attack_id: stepEvent.attack_id,
                    steps: [step],
                    total_steps: 1,
                    total_duration_ms: step.duration_ms,
                    convergence_achieved: false,
                    final_fitness: 0,
                  };
                }
                return {
                  ...prev,
                  steps: [...prev.steps, step],
                  total_steps: prev.total_steps + 1,
                  total_duration_ms: prev.total_duration_ms + step.duration_ms,
                };
              });
              break;
            }
            
            case 'mutation': {
              const mutationEvent = data;
              const mutation = mutationEvent.mutation;
              onMutation?.(mutation);
              setMutationHistory((prev) => [...prev, mutation]);
              break;
            }
            
            case 'convergence': {
              const convEvent = data;
              const metrics = convEvent.metrics;
              onConvergence?.(metrics);
              setConvergenceMetrics((prev) => [...prev, metrics]);
              break;
            }
          }
        } catch (error) {
          console.error('[UnifiedAttack WS] Parse error:', error);
        }
      };
    } catch (error) {
      console.error('[UnifiedAttack WS] Connection error:', error);
      onError?.(error as Error);
    }
  }, [sessionId, onProgress, onReasoningStep, onMutation, onConvergence, onError]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    isConnected,
    currentAttackId,
    progress,
    phase,
    reasoningChain,
    mutationHistory,
    convergenceMetrics,
  };
}

// ==============================================================================
// Combined Session Manager Hook
// ==============================================================================

export interface UseAttackSessionManagerOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export interface UseAttackSessionManagerReturn {
  // Session state
  session: SessionStatusResponse | null;
  sessionLoading: boolean;
  sessionError: Error | null;
  
  // Operations
  createSession: ReturnType<typeof useCreateSession>['mutateAsync'];
  finalizeSession: ReturnType<typeof useFinalizeSession>['mutateAsync'];
  executeAttack: ReturnType<typeof useExecuteAttack>['mutateAsync'];
  
  // Attack state
  currentAttack: AttackResponse | null;
  attackLoading: boolean;
  attackError: Error | null;
  
  // Progress (from WebSocket)
  progress: UseAttackProgressState;
  
  // Resources
  budgetStatus: BudgetStatusResponse | null;
  resourceUsage: ResourceUsageResponse | null;
  
  // Strategies & Presets
  strategies: StrategyInfo[];
  presets: PresetConfig[];
  
  // Actions
  setSessionId: (id: string | null) => void;
}

/**
 * Combined hook for managing attack sessions with all related state
 */
export function useAttackSessionManager(
  options: UseAttackSessionManagerOptions = {}
): UseAttackSessionManagerReturn {
  const { autoRefresh = true, refreshInterval = 5000 } = options;
  
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentAttack, setCurrentAttack] = useState<AttackResponse | null>(null);

  // Session query
  const sessionQuery = useSession(sessionId, {
    refetchInterval: autoRefresh ? refreshInterval : undefined,
  });

  // Mutations
  const createSessionMutation = useCreateSession();
  const finalizeSessionMutation = useFinalizeSession();
  const executeAttackMutation = useExecuteAttack();

  // Resources
  const budgetQuery = useBudgetStatus(sessionId, {
    refetchInterval: autoRefresh ? refreshInterval : undefined,
  });
  const resourceQuery = useResourceUsage(sessionId, {
    refetchInterval: autoRefresh ? refreshInterval : undefined,
  });

  // Strategies & Presets
  const strategiesQuery = useStrategies();
  const presetsQuery = usePresets();

  // WebSocket progress
  const progress = useAttackProgress({
    sessionId,
    onProgress: (event) => {
      if (event.phase === 'completed') {
        // Refetch session on attack completion
        sessionQuery.refetch();
      }
    },
  });

  // Wrapper for createSession that sets the session ID
  const handleCreateSession = useCallback(async (request: CreateSessionRequest) => {
    const result = await createSessionMutation.mutateAsync(request);
    setSessionId(result.session_id);
    return result;
  }, [createSessionMutation]);

  // Wrapper for executeAttack that sets current attack
  const handleExecuteAttack = useCallback(async (request: UnifiedAttackRequest) => {
    const result = await executeAttackMutation.mutateAsync(request);
    setCurrentAttack(result);
    return result;
  }, [executeAttackMutation]);

  return {
    // Session state
    session: sessionQuery.data ?? null,
    sessionLoading: sessionQuery.isLoading,
    sessionError: sessionQuery.error as Error | null,
    
    // Operations
    createSession: handleCreateSession,
    finalizeSession: finalizeSessionMutation.mutateAsync,
    executeAttack: handleExecuteAttack,
    
    // Attack state
    currentAttack,
    attackLoading: executeAttackMutation.isPending,
    attackError: executeAttackMutation.error as Error | null,
    
    // Progress
    progress,
    
    // Resources
    budgetStatus: budgetQuery.data ?? null,
    resourceUsage: resourceQuery.data ?? null,
    
    // Strategies & Presets
    strategies: strategiesQuery.data ?? [],
    presets: presetsQuery.data ?? [],
    
    // Actions
    setSessionId,
  };
}