"use client";

/**
 * Evasion Hooks for Project Chimera Frontend
 * 
 * React hooks for evasion task functionality including:
 * - Task creation and management
 * - Status polling
 * - Results retrieval
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { evasionService } from "@/lib/services/evasion-service";
import {
  EvasionTaskConfig,
  EvasionTaskStatusResponse,
  EvasionTaskResult,
  EvasionTaskStatus,
  EvasionTaskState,
  StrategiesListResponse,
  AvailableStrategy,
  MetamorphosisStrategyConfig,
  SuccessCriteria,
} from "@/lib/types/evasion-types";

// =============================================================================
// Strategies Hook
// =============================================================================

export interface UseEvasionStrategiesReturn {
  strategies: AvailableStrategy[];
  categories: string[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  getStrategiesByCategory: (category: string) => AvailableStrategy[];
}

export function useEvasionStrategies(): UseEvasionStrategiesReturn {
  const [strategies, setStrategies] = useState<AvailableStrategy[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await evasionService.listStrategies();
      setStrategies(response.strategies);
      setCategories(response.categories);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load strategies");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getStrategiesByCategory = useCallback((category: string): AvailableStrategy[] => {
    return strategies.filter((s) => s.category === category);
  }, [strategies]);

  // Load on mount
  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    strategies,
    categories,
    isLoading,
    error,
    refresh,
    getStrategiesByCategory,
  };
}

// =============================================================================
// Evasion Task Hook
// =============================================================================

export interface UseEvasionTaskReturn {
  // State
  state: EvasionTaskState;
  task: EvasionTaskStatusResponse | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  createTask: (config: EvasionTaskConfig) => Promise<string>;
  cancelTask: () => Promise<void>;
  reset: () => void;
  refresh: () => Promise<void>;
  
  // Status flags
  isRunning: boolean;
  isCompleted: boolean;
  isFailed: boolean;
}

export function useEvasionTask(taskId?: string): UseEvasionTaskReturn {
  const [state, setState] = useState<EvasionTaskState>(
    evasionService.createInitialEvasionState()
  );
  const [task, setTask] = useState<EvasionTaskStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch task if taskId is provided
  const refresh = useCallback(async () => {
    if (!taskId && !state.taskId) return;
    
    const id = taskId || state.taskId;
    if (!id) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const status = await evasionService.getEvasionTaskStatus(id);
      setTask(status);
      setState((prev) => evasionService.setTaskStatus(prev, status));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch task");
    } finally {
      setIsLoading(false);
    }
  }, [taskId, state.taskId]);

  // Load task on mount if taskId provided
  useEffect(() => {
    if (taskId) {
      refresh();
    }
  }, [taskId, refresh]);

  const createTask = useCallback(async (config: EvasionTaskConfig): Promise<string> => {
    // Cancel any existing task
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setIsLoading(true);
    setError(null);

    try {
      // Create the task
      const taskStatus = await evasionService.createEvasionTask(config);
      setState(evasionService.setTaskStarted(state, taskStatus.task_id));
      setTask(taskStatus);

      // Start polling
      await evasionService.pollEvasionTaskStatus(taskStatus.task_id, {
        onStatusUpdate: (status) => {
          setState((prev) => evasionService.setTaskStatus(prev, status));
          setTask(status);
        },
        onComplete: (result) => {
          setState((prev) => evasionService.setTaskResults(prev, result));
        },
        onError: (err) => {
          setState((prev) => evasionService.setTaskError(prev, err.message));
          setError(err.message);
        },
        signal: abortControllerRef.current?.signal,
      });

      setIsLoading(false);
      return taskStatus.task_id;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Task creation failed";
      setState((prev) => evasionService.setTaskError(prev, errorMessage));
      setError(errorMessage);
      setIsLoading(false);
      throw err;
    }
  }, [state]);

  const cancelTask = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const id = taskId || state.taskId;
    if (id) {
      try {
        await evasionService.cancelEvasionTask(id);
        setState((prev) => ({
          ...prev,
          status: EvasionTaskStatus.CANCELLED,
          isPolling: false,
        }));
      } catch (err) {
        console.error("Failed to cancel task:", err);
      }
    }
  }, [taskId, state.taskId]);

  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setState(evasionService.createInitialEvasionState());
    setTask(null);
    setError(null);
    setIsLoading(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    state,
    task,
    isLoading,
    error,
    createTask,
    cancelTask,
    reset,
    refresh,
    isRunning: state.status === EvasionTaskStatus.RUNNING || state.status === EvasionTaskStatus.PENDING,
    isCompleted: state.status === EvasionTaskStatus.COMPLETED,
    isFailed: state.status === EvasionTaskStatus.FAILED,
  };
}

// =============================================================================
// Evasion Form Hook
// =============================================================================

export interface EvasionFormErrors {
  initialPrompt?: string;
  targetModelId?: string;
  evasionTechnique?: string;
}

export interface UseEvasionFormReturn {
  // Form fields
  initialPrompt: string;
  setInitialPrompt: (value: string) => void;
  targetModelId: string;
  setTargetModelId: (value: string) => void;
  evasionTechnique: string;
  setEvasionTechnique: (value: string) => void;
  maxIterations: number;
  setMaxIterations: (value: number) => void;
  targetBehavior: string;
  setTargetBehavior: (value: string) => void;
  
  // Legacy fields for backward compatibility
  selectedStrategies: MetamorphosisStrategyConfig[];
  addStrategy: (strategy: MetamorphosisStrategyConfig) => void;
  removeStrategy: (index: number) => void;
  updateStrategy: (index: number, strategy: MetamorphosisStrategyConfig) => void;
  reorderStrategies: (fromIndex: number, toIndex: number) => void;
  successCriteria: SuccessCriteria;
  setSuccessCriteria: (criteria: SuccessCriteria) => void;
  maxAttempts: number;
  setMaxAttempts: (value: number) => void;
  timeoutSeconds: number;
  setTimeoutSeconds: (value: number) => void;
  
  // Validation
  errors: EvasionFormErrors;
  isValid: boolean;
  
  // Actions
  getConfig: () => EvasionTaskConfig;
  buildRequest: () => EvasionTaskConfig;
  reset: () => void;
}

export function useEvasionForm(): UseEvasionFormReturn {
  // Form fields
  const [initialPrompt, setInitialPrompt] = useState("");
  const [targetModelId, setTargetModelId] = useState("");
  const [evasionTechnique, setEvasionTechnique] = useState("");
  const [maxIterations, setMaxIterations] = useState(10);
  const [targetBehavior, setTargetBehavior] = useState("");
  
  // Legacy fields
  const [selectedStrategies, setSelectedStrategies] = useState<MetamorphosisStrategyConfig[]>([]);
  const [successCriteria, setSuccessCriteria] = useState<SuccessCriteria>({});
  const [maxAttempts, setMaxAttempts] = useState(10);
  const [timeoutSeconds, setTimeoutSeconds] = useState(300);

  // Validation errors
  const errors: EvasionFormErrors = {};
  if (!initialPrompt.trim()) {
    errors.initialPrompt = "Initial prompt is required";
  }
  if (!targetModelId.trim()) {
    errors.targetModelId = "Target model is required";
  }
  if (!evasionTechnique) {
    errors.evasionTechnique = "Evasion technique is required";
  }

  const addStrategy = useCallback((strategy: MetamorphosisStrategyConfig) => {
    setSelectedStrategies((prev) => [...prev, strategy]);
  }, []);

  const removeStrategy = useCallback((index: number) => {
    setSelectedStrategies((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const updateStrategy = useCallback((index: number, strategy: MetamorphosisStrategyConfig) => {
    setSelectedStrategies((prev) => {
      const updated = [...prev];
      updated[index] = strategy;
      return updated;
    });
  }, []);

  const reorderStrategies = useCallback((fromIndex: number, toIndex: number) => {
    setSelectedStrategies((prev) => {
      const updated = [...prev];
      const [removed] = updated.splice(fromIndex, 1);
      updated.splice(toIndex, 0, removed);
      return updated;
    });
  }, []);

  const getConfig = useCallback((): EvasionTaskConfig => {
    // Build strategy chain from evasion technique if no strategies selected
    const strategyChain: MetamorphosisStrategyConfig[] = selectedStrategies.length > 0
      ? selectedStrategies
      : evasionTechnique
        ? [{ name: evasionTechnique, strategy_name: evasionTechnique, parameters: {} }]
        : [];

    return {
      initial_prompt: initialPrompt,
      target_model_id: targetModelId,
      strategy_chain: strategyChain,
      success_criteria: successCriteria,
      max_attempts: maxIterations,
      timeout_seconds: timeoutSeconds,
    };
  }, [initialPrompt, targetModelId, selectedStrategies, evasionTechnique, successCriteria, maxIterations, timeoutSeconds]);

  // Alias for getConfig
  const buildRequest = getConfig;

  const reset = useCallback(() => {
    setInitialPrompt("");
    setTargetModelId("");
    setEvasionTechnique("");
    setMaxIterations(10);
    setTargetBehavior("");
    setSelectedStrategies([]);
    setSuccessCriteria({});
    setMaxAttempts(10);
    setTimeoutSeconds(300);
  }, []);

  // Form is valid if required fields are filled
  const isValid = initialPrompt.trim().length > 0 &&
                  targetModelId.trim().length > 0 &&
                  (evasionTechnique.length > 0 || selectedStrategies.length > 0);

  return {
    // Form fields
    initialPrompt,
    setInitialPrompt,
    targetModelId,
    setTargetModelId,
    evasionTechnique,
    setEvasionTechnique,
    maxIterations,
    setMaxIterations,
    targetBehavior,
    setTargetBehavior,
    
    // Legacy fields
    selectedStrategies,
    addStrategy,
    removeStrategy,
    updateStrategy,
    reorderStrategies,
    successCriteria,
    setSuccessCriteria,
    maxAttempts,
    setMaxAttempts,
    timeoutSeconds,
    setTimeoutSeconds,
    
    // Validation
    errors,
    isValid,
    
    // Actions
    getConfig,
    buildRequest,
    reset,
  };
}

// =============================================================================
// Evasion Task List Hook
// =============================================================================

/**
 * Extended task type with metadata for list display
 */
export interface EvasionTaskWithMeta {
  taskId: string;
  status: EvasionTaskStatus;
  initialPrompt: string;
  targetModelId: string;
  createdAt: string;
  completedAt?: string;
  progress?: number;
  strategiesCount: number;
  result?: EvasionTaskResult;
}

/**
 * Task list item type matching what the page expects
 */
export interface EvasionTaskListItem {
  task_id: string;
  status: EvasionTaskStatus;
  initial_prompt: string;
  target_model_id: string;
  evasion_technique: string;
  max_iterations: number;
  target_behavior?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress?: number;
  result?: EvasionTaskResult;
  error?: string;
}

export interface UseEvasionTaskListReturn {
  tasks: EvasionTaskListItem[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  getTask: (taskId: string) => EvasionTaskListItem | undefined;
  removeTask: (taskId: string) => void;
  deleteTask: (taskId: string) => Promise<void>;
  clearCompleted: () => void;
}

/**
 * Hook for managing a list of evasion tasks
 */
export function useEvasionTaskList(): UseEvasionTaskListReturn {
  const [tasks, setTasks] = useState<EvasionTaskListItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // In a real implementation, this would fetch from an API endpoint
      // For now, we maintain local state
      // const response = await evasionService.listTasks();
      // setTasks(response.tasks);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load tasks");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getTask = useCallback((taskId: string): EvasionTaskListItem | undefined => {
    return tasks.find((t) => t.task_id === taskId);
  }, [tasks]);

  const removeTask = useCallback((taskId: string) => {
    setTasks((prev) => prev.filter((t) => t.task_id !== taskId));
  }, []);

  const deleteTask = useCallback(async (taskId: string) => {
    try {
      await evasionService.cancelEvasionTask(taskId);
      setTasks((prev) => prev.filter((t) => t.task_id !== taskId));
    } catch (err) {
      console.error("Failed to delete task:", err);
      throw err;
    }
  }, []);

  const clearCompleted = useCallback(() => {
    setTasks((prev) => prev.filter((t) =>
      t.status !== EvasionTaskStatus.COMPLETED &&
      t.status !== EvasionTaskStatus.FAILED &&
      t.status !== EvasionTaskStatus.CANCELLED
    ));
  }, []);

  // Add a task to the list (called when creating new tasks)
  const addTask = useCallback((task: EvasionTaskListItem) => {
    setTasks((prev) => [task, ...prev]);
  }, []);

  // Update a task in the list
  const updateTask = useCallback((taskId: string, updates: Partial<EvasionTaskListItem>) => {
    setTasks((prev) => prev.map((t) =>
      t.task_id === taskId ? { ...t, ...updates } : t
    ));
  }, []);

  return {
    tasks,
    isLoading,
    error,
    refresh,
    getTask,
    removeTask,
    deleteTask,
    clearCompleted,
  };
}