/**
 * AutoDAN Lifelong Learning Service for Project Chimera Frontend
 * 
 * Provides API methods for AutoDAN Lifelong Learning functionality including:
 * - Single attack execution
 * - Continuous loop attacks
 * - Progress tracking
 */

import { enhancedApi } from "../api-enhanced";
import {
  LifelongAttackRequest,
  LifelongAttackResponse,
  LifelongLoopRequest,
  LifelongLoopResponse,
  LifelongProgressResponse,
  LifelongAttackState,
  LifelongLoopState,
  LifelongLoopProgress,
} from "../types/autodan-lifelong-types";

// =============================================================================
// Configuration
// =============================================================================

const AUTODAN_BASE_PATH = "/autodan";

// =============================================================================
// API Methods
// =============================================================================

/**
 * Execute a single lifelong learning attack
 */
export async function executeLifelongAttack(
  request: LifelongAttackRequest
): Promise<LifelongAttackResponse> {
  const response = await enhancedApi.post<LifelongAttackResponse>(
    `${AUTODAN_BASE_PATH}/lifelong`,
    request
  );
  return response;
}

/**
 * Start a continuous lifelong learning loop
 */
export async function startLifelongLoop(
  request: LifelongLoopRequest
): Promise<LifelongLoopResponse> {
  const response = await enhancedApi.post<LifelongLoopResponse>(
    `${AUTODAN_BASE_PATH}/lifelong/loop`,
    request
  );
  return response;
}

/**
 * Get progress of a lifelong learning task
 */
export async function getLifelongProgress(
  taskId: string
): Promise<LifelongProgressResponse> {
  const response = await enhancedApi.get<LifelongProgressResponse>(
    `${AUTODAN_BASE_PATH}/lifelong/progress/${encodeURIComponent(taskId)}`
  );
  return response;
}

/**
 * Stop a running lifelong learning loop
 */
export async function stopLifelongLoop(
  taskId: string
): Promise<{ success: boolean; message: string }> {
  const response = await enhancedApi.post<{ success: boolean; message: string }>(
    `${AUTODAN_BASE_PATH}/lifelong/loop/${encodeURIComponent(taskId)}/stop`,
    {}
  );
  return response;
}

// =============================================================================
// Polling Utilities
// =============================================================================

export interface PollingOptions {
  /** Polling interval in milliseconds */
  intervalMs?: number;
  /** Maximum polling duration in milliseconds */
  maxDurationMs?: number;
  /** Callback for progress updates */
  onProgress?: (progress: LifelongProgressResponse) => void;
  /** Callback for errors */
  onError?: (error: Error) => void;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
}

const DEFAULT_POLLING_INTERVAL = 2000;
const DEFAULT_MAX_DURATION = 600000; // 10 minutes

/**
 * Poll for lifelong attack progress until completion
 */
export async function pollLifelongProgress(
  taskId: string,
  options: PollingOptions = {}
): Promise<LifelongProgressResponse> {
  const {
    intervalMs = DEFAULT_POLLING_INTERVAL,
    maxDurationMs = DEFAULT_MAX_DURATION,
    onProgress,
    onError,
    signal,
  } = options;

  const startTime = Date.now();

  return new Promise((resolve, reject) => {
    const poll = async () => {
      // Check for abort
      if (signal?.aborted) {
        reject(new Error("Polling aborted"));
        return;
      }

      // Check for timeout
      if (Date.now() - startTime > maxDurationMs) {
        reject(new Error("Polling timeout exceeded"));
        return;
      }

      try {
        const progress = await getLifelongProgress(taskId);
        
        // Notify progress callback
        onProgress?.(progress);

        // Check if completed
        if (progress.status === "completed" || progress.status === "failed") {
          resolve(progress);
          return;
        }

        // Schedule next poll
        setTimeout(poll, intervalMs);
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
 * Create initial attack state
 */
export function createInitialAttackState(): LifelongAttackState {
  return {
    isLoading: false,
    result: null,
    response: null,
    error: null,
  };
}

/**
 * Create initial loop state
 */
export function createInitialLoopState(): LifelongLoopState {
  return {
    taskId: null,
    isRunning: false,
    progress: null,
    results: [],
    error: null,
  };
}

/**
 * Update attack state with loading
 */
export function setAttackLoading(state: LifelongAttackState): LifelongAttackState {
  return {
    ...state,
    isLoading: true,
    error: null,
  };
}

/**
 * Update attack state with response
 */
export function setAttackResponse(
  state: LifelongAttackState,
  response: LifelongAttackResponse
): LifelongAttackState {
  return {
    ...state,
    isLoading: false,
    result: response,
    response,
    error: null,
  };
}

/**
 * Update attack state with error
 */
export function setAttackError(
  state: LifelongAttackState,
  error: string
): LifelongAttackState {
  return {
    ...state,
    isLoading: false,
    error,
  };
}

/**
 * Update loop state with task started
 */
export function setLoopStarted(
  state: LifelongLoopState,
  taskId: string
): LifelongLoopState {
  return {
    ...state,
    taskId,
    isRunning: true,
    error: null,
  };
}

/**
 * Update loop state with progress
 */
export function setLoopProgress(
  state: LifelongLoopState,
  progress: LifelongProgressResponse
): LifelongLoopState {
  // Convert LifelongProgressResponse to LifelongLoopProgress
  const loopProgress: LifelongLoopProgress | null = progress.phase ? {
    phase: progress.phase as "warmup" | "lifelong" | "completed",
    total_attacks: progress.total_attacks || 0,
    successful_attacks: progress.successful_attacks || 0,
    strategies_discovered: progress.strategies_discovered || 0,
    average_score: progress.average_score || 0,
    best_score: progress.best_score || 0,
    current_request: progress.current_request || 0,
    total_requests: progress.total_requests || 0,
  } : null;
  
  return {
    ...state,
    progress: loopProgress,
    results: progress.result ? [...state.results, {
      index: state.results.length,
      request: "",
      best_prompt: progress.result.prompt,
      best_response: progress.result.response,
      best_score: progress.result.score,
      success: progress.result.is_jailbreak,
      attempts: 1,
      strategies_discovered: progress.result.strategy_extracted ? [progress.result.strategy_extracted] : [],
    }] : state.results,
  };
}

/**
 * Update loop state with completion
 */
export function setLoopCompleted(state: LifelongLoopState): LifelongLoopState {
  return {
    ...state,
    isRunning: false,
  };
}

/**
 * Update loop state with error
 */
export function setLoopError(
  state: LifelongLoopState,
  error: string
): LifelongLoopState {
  return {
    ...state,
    isRunning: false,
    error,
  };
}

// =============================================================================
// Service Object
// =============================================================================

export const autodanLifelongService = {
  // API methods
  executeLifelongAttack,
  executeSingleAttack: executeLifelongAttack, // Alias for backward compatibility
  startLifelongLoop,
  getProgress: getLifelongProgress,
  getLifelongProgress,
  stopLifelongLoop,
  
  // Polling
  pollLifelongProgress,
  
  // State helpers
  createInitialAttackState,
  createInitialLoopState,
  setAttackLoading,
  setAttackResponse,
  setAttackError,
  setLoopStarted,
  setLoopProgress,
  setLoopCompleted,
  setLoopError,
};

export default autodanLifelongService;