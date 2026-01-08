/**
 * Evasion Service for Project Chimera Frontend
 *
 * Provides API methods for metamorphic evasion functionality including:
 * - Task creation
 * - Status tracking
 * - Results retrieval
 * - Strategy management
 */

import { enhancedApi } from "../api-enhanced";
import {
  EvasionTaskConfig,
  EvasionTaskStatusResponse,
  EvasionTaskResult,
  EvasionTaskStatus,
  EvasionTaskState,
  EvasionPollingConfig,
  DEFAULT_POLLING_CONFIG,
  StrategiesListResponse,
  AvailableStrategy,
} from "../types/evasion-types";

// =============================================================================
// Configuration
// =============================================================================

// Evasion task endpoints (under /api/v1/evasion)
const EVASION_BASE_PATH = "/api/v1/evasion";

// Strategies endpoint is at /api/v1/strategies (separate from evasion)
const STRATEGIES_BASE_PATH = "/api/v1/strategies";

// =============================================================================
// API Methods
// =============================================================================

/**
 * Create and start a new evasion task
 */
export async function createEvasionTask(
  config: EvasionTaskConfig
): Promise<EvasionTaskStatusResponse> {
  const response = await enhancedApi.post<EvasionTaskStatusResponse>(
    `${EVASION_BASE_PATH}/generate`,
    config
  );
  return response;
}

/**
 * Get the status of an evasion task
 */
export async function getEvasionTaskStatus(
  taskId: string
): Promise<EvasionTaskStatusResponse> {
  const response = await enhancedApi.get<EvasionTaskStatusResponse>(
    `${EVASION_BASE_PATH}/status/${encodeURIComponent(taskId)}`
  );
  return response;
}

/**
 * Get the results of a completed evasion task
 */
export async function getEvasionTaskResults(
  taskId: string
): Promise<EvasionTaskResult> {
  const response = await enhancedApi.get<EvasionTaskResult>(
    `${EVASION_BASE_PATH}/results/${encodeURIComponent(taskId)}`
  );
  return response;
}

/**
 * Cancel a running evasion task
 * Note: This endpoint may not be implemented on the backend yet.
 * The function will attempt to cancel but may fail gracefully.
 */
export async function cancelEvasionTask(
  taskId: string
): Promise<{ success: boolean; message: string }> {
  try {
    // Try the cancel endpoint if it exists
    const response = await enhancedApi.post<{ success: boolean; message: string }>(
      `${EVASION_BASE_PATH}/cancel/${encodeURIComponent(taskId)}`,
      {}
    );
    return response;
  } catch (error) {
    // If cancel endpoint doesn't exist, return a graceful failure
    console.warn(`Cancel endpoint not available for task ${taskId}:`, error);
    return {
      success: false,
      message: "Task cancellation not supported by backend. Task may continue running.",
    };
  }
}

/**
 * List available metamorphosis strategies
 * Note: Strategies are at /api/v1/strategies/, not under /evasion
 */
export async function listStrategies(): Promise<StrategiesListResponse> {
  try {
    // Backend returns array of MetamorphosisStrategyInfo directly
    const strategies = await enhancedApi.get<AvailableStrategy[]>(
      `${STRATEGIES_BASE_PATH}/`
    );

    // Transform to StrategiesListResponse format expected by frontend
    const categories = [...new Set(strategies.map(s => s.category || "general"))];

    return {
      strategies,
      categories,
    };
  } catch (error) {
    console.error("Failed to fetch strategies:", error);
    // Return empty response on error
    return {
      strategies: [],
      categories: [],
    };
  }
}

/**
 * Get details for a specific strategy by name
 */
export async function getStrategyByName(
  strategyName: string
): Promise<AvailableStrategy | null> {
  try {
    const response = await enhancedApi.get<AvailableStrategy>(
      `${STRATEGIES_BASE_PATH}/${encodeURIComponent(strategyName)}`
    );
    return response;
  } catch (error) {
    console.error(`Failed to fetch strategy ${strategyName}:`, error);
    return null;
  }
}

// =============================================================================
// Polling Utilities
// =============================================================================

export interface EvasionPollingOptions {
  /** Polling configuration */
  config?: Partial<EvasionPollingConfig>;
  /** Callback for status updates */
  onStatusUpdate?: (status: EvasionTaskStatusResponse) => void;
  /** Callback for completion */
  onComplete?: (result: EvasionTaskResult) => void;
  /** Callback for errors */
  onError?: (error: Error) => void;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
}

/**
 * Poll for evasion task status until completion
 */
export async function pollEvasionTaskStatus(
  taskId: string,
  options: EvasionPollingOptions = {}
): Promise<EvasionTaskResult> {
  const {
    config = {},
    onStatusUpdate,
    onComplete,
    onError,
    signal,
  } = options;

  const pollingConfig: EvasionPollingConfig = {
    ...DEFAULT_POLLING_CONFIG,
    ...config,
  };

  const startTime = Date.now();

  return new Promise((resolve, reject) => {
    const poll = async () => {
      // Check for abort
      if (signal?.aborted) {
        const error = new Error("Polling aborted");
        onError?.(error);
        reject(error);
        return;
      }

      // Check for timeout
      if (Date.now() - startTime > pollingConfig.maxDurationMs) {
        const error = new Error("Polling timeout exceeded");
        onError?.(error);
        reject(error);
        return;
      }

      try {
        const status = await getEvasionTaskStatus(taskId);

        // Notify status callback
        onStatusUpdate?.(status);

        // Check if completed or failed
        if (
          status.status === EvasionTaskStatus.COMPLETED ||
          status.status === EvasionTaskStatus.FAILED ||
          status.status === EvasionTaskStatus.CANCELLED
        ) {
          if (pollingConfig.stopOnComplete) {
            try {
              const result = await getEvasionTaskResults(taskId);
              onComplete?.(result);
              resolve(result);
            } catch (_resultError) {
              // If we can't get results, create a minimal result from status
              const minimalResult: EvasionTaskResult = {
                task_id: taskId,
                status: status.status,
                initial_prompt: "",
                target_model_id: "",
                strategy_chain: [],
                success_criteria: "",
                final_status: status.message || status.status,
                results: [],
                overall_success: status.status === EvasionTaskStatus.COMPLETED,
                completed_at: new Date().toISOString(),
                failed_reason: status.status === EvasionTaskStatus.FAILED
                  ? status.message || "Unknown error"
                  : null,
              };
              onComplete?.(minimalResult);
              resolve(minimalResult);
            }
            return;
          }
        }

        // Schedule next poll
        setTimeout(poll, pollingConfig.intervalMs);
      } catch (error) {
        onError?.(error as Error);
        reject(error);
      }
    };

    // Start polling
    poll();
  });
}

// =============================================================================
// State Management Helpers
// =============================================================================

/**
 * Create initial evasion task state
 */
export function createInitialEvasionState(): EvasionTaskState {
  return {
    taskId: null,
    status: null,
    progress: 0,
    currentStep: "",
    results: null,
    error: null,
    isPolling: false,
  };
}

/**
 * Update state with task started
 */
export function setTaskStarted(
  state: EvasionTaskState,
  taskId: string
): EvasionTaskState {
  return {
    ...state,
    taskId,
    status: EvasionTaskStatus.PENDING,
    progress: 0,
    currentStep: "Task created",
    error: null,
    isPolling: true,
  };
}

/**
 * Update state with status update
 */
export function setTaskStatus(
  state: EvasionTaskState,
  status: EvasionTaskStatusResponse
): EvasionTaskState {
  return {
    ...state,
    status: status.status,
    progress: status.progress || state.progress,
    currentStep: status.current_step || status.message || state.currentStep,
  };
}

/**
 * Update state with results
 */
export function setTaskResults(
  state: EvasionTaskState,
  results: EvasionTaskResult
): EvasionTaskState {
  return {
    ...state,
    status: results.status,
    progress: 100,
    currentStep: results.final_status,
    results,
    isPolling: false,
  };
}

/**
 * Update state with error
 */
export function setTaskError(
  state: EvasionTaskState,
  error: string
): EvasionTaskState {
  return {
    ...state,
    status: EvasionTaskStatus.FAILED,
    error,
    isPolling: false,
  };
}

/**
 * Reset state
 */
export function resetTaskState(): EvasionTaskState {
  return createInitialEvasionState();
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Execute an evasion task and wait for results
 */
export async function executeEvasionTask(
  config: EvasionTaskConfig,
  options: EvasionPollingOptions = {}
): Promise<EvasionTaskResult> {
  // Create the task
  const taskStatus = await createEvasionTask(config);

  // Poll for completion
  const result = await pollEvasionTaskStatus(taskStatus.task_id, options);

  return result;
}

// =============================================================================
// Service Object
// =============================================================================

export const evasionService = {
  // API methods
  createEvasionTask,
  getEvasionTaskStatus,
  getEvasionTaskResults,
  cancelEvasionTask,
  listStrategies,
  getStrategyByName,

  // Polling
  pollEvasionTaskStatus,

  // Convenience
  executeEvasionTask,

  // State helpers
  createInitialEvasionState,
  setTaskStarted,
  setTaskStatus,
  setTaskResults,
  setTaskError,
  resetTaskState,
};

export default evasionService;
