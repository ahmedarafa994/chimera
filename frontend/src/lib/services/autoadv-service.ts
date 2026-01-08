/**
 * AutoAdv Service for Project Chimera Frontend
 * 
 * Provides API methods for AutoAdv (Automated Adversarial) functionality including:
 * - Attack initiation
 * - WebSocket-based real-time updates
 * - Result handling
 * 
 * This service replaces the direct fetch calls in the AutoAdv page
 * with centralized API client usage.
 */

import { enhancedApi } from "../api-enhanced";
import { WebSocketManager } from "../websocket-manager";

// =============================================================================
// Configuration
// =============================================================================

const AUTOADV_BASE_PATH = "/api/v1/autoadv";

// =============================================================================
// Types
// =============================================================================

export interface AutoAdvStartRequest {
  /** Target prompt to attack */
  target_prompt: string;
  /** Attack method */
  method: string;
  /** Target model (optional, uses session default) */
  model?: string;
  /** Provider (optional, uses session default) */
  provider?: string;
  /** Maximum iterations */
  max_iterations?: number;
  /** Temperature for generation */
  temperature?: number;
  /** Additional configuration */
  config?: Record<string, unknown>;
}

export interface AutoAdvStartResponse {
  /** Session/task identifier */
  session_id: string;
  /** Status message */
  message: string;
  /** WebSocket URL for updates */
  ws_url?: string;
}

export interface AutoAdvProgress {
  /** Current iteration */
  iteration: number;
  /** Total iterations */
  total_iterations: number;
  /** Current status */
  status: "running" | "completed" | "failed" | "cancelled";
  /** Current prompt being tested */
  current_prompt?: string;
  /** Current score */
  current_score?: number;
  /** Best prompt so far */
  best_prompt?: string;
  /** Best score so far */
  best_score?: number;
  /** Status message */
  message?: string;
  /** Timestamp */
  timestamp: string;
}

export interface AutoAdvResult {
  /** Session identifier */
  session_id: string;
  /** Final status */
  status: "completed" | "failed" | "cancelled";
  /** Best jailbreak prompt found */
  best_prompt: string;
  /** Best score achieved */
  best_score: number;
  /** Total iterations run */
  iterations_run: number;
  /** Attack method used */
  method: string;
  /** Target model */
  target_model?: string;
  /** Provider used */
  provider_used?: string;
  /** Completion timestamp */
  completed_at: string;
  /** All iteration results */
  iteration_results?: AutoAdvIterationResult[];
}

export interface AutoAdvIterationResult {
  /** Iteration number */
  iteration: number;
  /** Prompt tested */
  prompt: string;
  /** Score achieved */
  score: number;
  /** LLM response */
  response?: string;
  /** Whether this was successful */
  is_successful: boolean;
  /** Timestamp */
  timestamp: string;
}

export interface AutoAdvWebSocketMessage {
  /** Message type */
  type: "progress" | "result" | "error" | "heartbeat" | "complete";
  /** Message data */
  data: AutoAdvProgress | AutoAdvResult | { error: string } | { timestamp: string } | Record<string, unknown>;
  /** Progress percentage (for progress type) */
  progress?: number;
  /** Current iteration (for progress type) */
  iteration?: number;
  /** Total iterations (for progress type) */
  total_iterations?: number;
  /** Message text */
  message?: string;
}

/**
 * Legacy AutoAdvRequest type for backward compatibility
 * @deprecated Use AutoAdvStartRequest instead
 */
export interface AutoAdvRequest {
  /** Target prompt to attack */
  target_prompt: string;
  /** Target behavior description */
  target_behavior?: string;
  /** Maximum iterations */
  max_iterations?: number;
  /** Attack technique */
  technique?: string;
  /** Target model */
  target_model?: string;
  /** Attack model */
  attack_model?: string;
  /** Judge model */
  judge_model?: string;
}

/**
 * Legacy AutoAdvResponse type for backward compatibility
 */
export interface AutoAdvResponse {
  /** Job identifier */
  job_id: string;
  /** Status message */
  message?: string;
  /** Jailbreak prompt (from result) */
  jailbreak_prompt?: string;
  /** Best score achieved */
  best_score?: number;
  /** Model used */
  model_used?: string;
  /** Provider used */
  provider_used?: string;
}

// =============================================================================
// API Methods
// =============================================================================

/**
 * Start an AutoAdv attack session
 */
export async function startAutoAdvAttack(
  request: AutoAdvStartRequest
): Promise<AutoAdvStartResponse> {
  const response = await enhancedApi.post<AutoAdvStartResponse>(
    `${AUTOADV_BASE_PATH}/start`,
    request
  );
  return response;
}

/**
 * Get the status of an AutoAdv session
 */
export async function getAutoAdvStatus(
  sessionId: string
): Promise<AutoAdvProgress> {
  const response = await enhancedApi.get<AutoAdvProgress>(
    `${AUTOADV_BASE_PATH}/status/${encodeURIComponent(sessionId)}`
  );
  return response;
}

/**
 * Get the results of a completed AutoAdv session
 */
export async function getAutoAdvResults(
  sessionId: string
): Promise<AutoAdvResult> {
  const response = await enhancedApi.get<AutoAdvResult>(
    `${AUTOADV_BASE_PATH}/results/${encodeURIComponent(sessionId)}`
  );
  return response;
}

/**
 * Cancel a running AutoAdv session
 */
export async function cancelAutoAdvSession(
  sessionId: string
): Promise<{ success: boolean; message: string }> {
  const response = await enhancedApi.post<{ success: boolean; message: string }>(
    `${AUTOADV_BASE_PATH}/cancel/${encodeURIComponent(sessionId)}`,
    {}
  );
  return response;
}

// =============================================================================
// WebSocket Integration
// =============================================================================

/**
 * Create a WebSocket connection for AutoAdv updates
 */
export function createAutoAdvWebSocket(
  sessionId: string,
  callbacks: {
    onProgress?: (progress: AutoAdvProgress) => void;
    onResult?: (result: AutoAdvResult) => void;
    onError?: (error: string) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
  }
): WebSocketManager {
  const wsUrl = `${AUTOADV_BASE_PATH}/ws/${sessionId}`;
  
  const wsManager = new WebSocketManager(wsUrl, {
    reconnect: true,
    maxReconnectAttempts: 5,
    reconnectInterval: 2000,
  });

  // Set up message handler
  wsManager.onMessage((data: AutoAdvWebSocketMessage) => {
    switch (data.type) {
      case "progress":
        callbacks.onProgress?.(data.data as AutoAdvProgress);
        break;
      case "result":
        callbacks.onResult?.(data.data as AutoAdvResult);
        break;
      case "error":
        callbacks.onError?.((data.data as { error: string }).error);
        break;
      case "heartbeat":
        // Heartbeat received, connection is alive
        break;
    }
  });

  // Set up connection handlers
  wsManager.onOpen(() => {
    callbacks.onConnect?.();
  });

  wsManager.onClose(() => {
    callbacks.onDisconnect?.();
  });

  wsManager.onError((error) => {
    callbacks.onError?.(error.message || "WebSocket error");
  });

  return wsManager;
}

// =============================================================================
// State Management
// =============================================================================

export interface AutoAdvState {
  /** Current session ID */
  sessionId: string | null;
  /** Current status */
  status: "idle" | "starting" | "running" | "completed" | "failed" | "cancelled";
  /** Current progress */
  progress: AutoAdvProgress | null;
  /** Final result */
  result: AutoAdvResult | null;
  /** Error message */
  error: string | null;
  /** WebSocket connection status */
  wsConnected: boolean;
}

/**
 * Create initial AutoAdv state
 */
export function createInitialAutoAdvState(): AutoAdvState {
  return {
    sessionId: null,
    status: "idle",
    progress: null,
    result: null,
    error: null,
    wsConnected: false,
  };
}

/**
 * Update state with session started
 */
export function setSessionStarted(
  state: AutoAdvState,
  sessionId: string
): AutoAdvState {
  return {
    ...state,
    sessionId,
    status: "starting",
    progress: null,
    result: null,
    error: null,
  };
}

/**
 * Update state with WebSocket connected
 */
export function setWsConnected(
  state: AutoAdvState,
  connected: boolean
): AutoAdvState {
  return {
    ...state,
    wsConnected: connected,
    status: connected && state.status === "starting" ? "running" : state.status,
  };
}

/**
 * Update state with progress
 */
export function setProgress(
  state: AutoAdvState,
  progress: AutoAdvProgress
): AutoAdvState {
  return {
    ...state,
    progress,
    status: progress.status === "running" ? "running" : state.status,
  };
}

/**
 * Update state with result
 */
export function setResult(
  state: AutoAdvState,
  result: AutoAdvResult
): AutoAdvState {
  return {
    ...state,
    result,
    status: result.status,
  };
}

/**
 * Update state with error
 */
export function setError(
  state: AutoAdvState,
  error: string
): AutoAdvState {
  return {
    ...state,
    error,
    status: "failed",
  };
}

/**
 * Reset state
 */
export function resetAutoAdvState(): AutoAdvState {
  return createInitialAutoAdvState();
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Execute an AutoAdv attack with WebSocket updates
 */
export async function executeAutoAdvAttack(
  request: AutoAdvStartRequest,
  callbacks: {
    onProgress?: (progress: AutoAdvProgress) => void;
    onResult?: (result: AutoAdvResult) => void;
    onError?: (error: string) => void;
  }
): Promise<AutoAdvResult> {
  // Start the attack
  const startResponse = await startAutoAdvAttack(request);
  const sessionId = startResponse.session_id;

  return new Promise((resolve, reject) => {
    let wsManager: WebSocketManager | null = null;

    const cleanup = () => {
      if (wsManager) {
        wsManager.disconnect();
        wsManager = null;
      }
    };

    wsManager = createAutoAdvWebSocket(sessionId, {
      onProgress: callbacks.onProgress,
      onResult: (result) => {
        callbacks.onResult?.(result);
        cleanup();
        resolve(result);
      },
      onError: (error) => {
        callbacks.onError?.(error);
        cleanup();
        reject(new Error(error));
      },
    });

    // Connect to WebSocket
    wsManager.connect();
  });
}

// =============================================================================
// Service Object
// =============================================================================

/**
 * Get WebSocket URL for a task
 * @deprecated Use createAutoAdvWebSocket instead
 */
export function getWebSocketUrl(taskId: string): string {
  const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
  const wsProtocol = baseUrl.startsWith('https') ? 'wss' : 'ws';
  const host = baseUrl.replace(/^https?:\/\//, '');
  return `${wsProtocol}://${host}${AUTOADV_BASE_PATH}/ws/${taskId}`;
}

/**
 * Start AutoAdv job (legacy method)
 * @deprecated Use startAutoAdvAttack instead
 */
export async function startAutoAdv(request: AutoAdvRequest): Promise<AutoAdvResponse> {
  // Convert legacy request to new format
  const startRequest: AutoAdvStartRequest = {
    target_prompt: request.target_prompt,
    method: request.technique || 'gcg',
    model: request.target_model,
    max_iterations: request.max_iterations,
    config: {
      target_behavior: request.target_behavior,
      attack_model: request.attack_model,
      judge_model: request.judge_model,
    },
  };
  
  const response = await startAutoAdvAttack(startRequest);
  return {
    job_id: response.session_id,
    message: response.message,
  };
}

/**
 * Generate AutoAdv result (legacy method)
 * This is a synchronous generation that waits for completion
 * @deprecated Use executeAutoAdvAttack instead
 */
export async function generateAutoAdv(request: AutoAdvRequest): Promise<AutoAdvResponse> {
  // Convert legacy request to new format
  const startRequest: AutoAdvStartRequest = {
    target_prompt: request.target_prompt,
    method: request.technique || 'gcg',
    model: request.target_model,
    max_iterations: request.max_iterations,
    config: {
      target_behavior: request.target_behavior,
      attack_model: request.attack_model,
      judge_model: request.judge_model,
    },
  };
  
  // Start the attack
  const startResponse = await startAutoAdvAttack(startRequest);
  
  // Poll for completion (simplified - in production use WebSocket)
  let attempts = 0;
  const maxAttempts = 60; // 5 minutes with 5 second intervals
  
  while (attempts < maxAttempts) {
    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
    
    try {
      const status = await getAutoAdvStatus(startResponse.session_id);
      
      if (status.status === 'completed' || status.status === 'failed') {
        // Get final results
        const result = await getAutoAdvResults(startResponse.session_id);
        return {
          job_id: startResponse.session_id,
          message: result.status,
          jailbreak_prompt: result.best_prompt,
          best_score: result.best_score,
          model_used: result.target_model,
          provider_used: result.provider_used,
        };
      }
    } catch {
      // Continue polling on error
    }
    
    attempts++;
  }
  
  // Timeout - return partial result
  return {
    job_id: startResponse.session_id,
    message: 'Timeout waiting for completion',
  };
}

export const autoAdvService = {
  // API methods
  startAutoAdvAttack,
  getAutoAdvStatus,
  getAutoAdvResults,
  cancelAutoAdvSession,
  
  // WebSocket
  createAutoAdvWebSocket,
  getWebSocketUrl,
  
  // Convenience
  executeAutoAdvAttack,
  
  // Legacy methods (deprecated)
  startAutoAdv,
  generateAutoAdv,
  
  // State helpers
  createInitialAutoAdvState,
  setSessionStarted,
  setWsConnected,
  setProgress,
  setResult,
  setError,
  resetAutoAdvState,
};

export default autoAdvService;