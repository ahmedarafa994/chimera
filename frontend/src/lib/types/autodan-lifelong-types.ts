/**
 * AutoDAN Lifelong Learning Types for Project Chimera Frontend
 *
 * Types for AutoDAN-Turbo lifelong learning endpoints including:
 * - Single lifelong attacks
 * - Full lifelong learning loops
 * - Progress tracking
 */

// =============================================================================
// Enums
// =============================================================================

/**
 * Status of a lifelong learning task
 */
export enum LifelongTaskStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
}

// =============================================================================
// Loop Attack Types (for API service)
// =============================================================================

export interface LoopAttackRequest {
  /** List of requests to process in the loop */
  requests: string[];
  /** Target model name (optional) */
  model?: string;
  /** Provider name (optional) */
  provider?: string;
  /** Number of iterations per request */
  iterations_per_request?: number;
  /** Score threshold to break early (1-10 scale) */
  break_score?: number;
}

export interface LoopAttackResponse {
  /** Task ID for tracking */
  task_id: string;
  /** Status of the loop attack */
  status: "pending" | "running" | "completed" | "failed";
  /** Message about the task */
  message: string;
}

export interface AttackProgress {
  /** Task ID */
  task_id: string;
  /** Current status */
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  /** Progress percentage (0-100) */
  progress: number;
  /** Current iteration */
  current_iteration: number;
  /** Total iterations */
  total_iterations: number;
  /** Current request being processed */
  current_request?: string;
  /** Best score achieved so far */
  best_score?: number;
  /** Error message if failed */
  error?: string;
  /** Started at timestamp */
  started_at?: string;
  /** Completed at timestamp */
  completed_at?: string;
}

export interface AttackResult {
  /** Task ID */
  task_id: string;
  /** Whether the attack was successful */
  success: boolean;
  /** Final score achieved */
  final_score: number;
  /** Best jailbreak prompt found */
  best_prompt: string;
  /** Response from target model */
  target_response: string;
  /** Number of iterations used */
  iterations_used: number;
  /** Strategies discovered */
  strategies_discovered: string[];
  /** Execution time in seconds */
  execution_time: number;
}

// =============================================================================
// Request Types
// =============================================================================

export interface LifelongAttackRequest {
  /** The harmful request to generate a jailbreak for */
  request: string;
  /** Target model name (optional, uses session default) */
  model?: string;
  /** Provider name (optional, uses session default) */
  provider?: string;
  /** Whether to include detailed scoring information */
  detailed_scoring?: boolean;
}

export interface LifelongLoopRequest {
  /** List of requests to process in the lifelong loop */
  requests: string[];
  /** Target model name (optional) */
  model?: string;
  /** Provider name (optional) */
  provider?: string;
  /** Number of epochs to run */
  epochs?: number;
  /** Score threshold to break early (1-10 scale) */
  break_score?: number;
  /** Whether to run warm-up phase first */
  warmup?: boolean;
  /** Number of warm-up iterations */
  warmup_iterations?: number;
}

// =============================================================================
// Response Types
// =============================================================================

export interface LifelongAttackResponse {
  /** The generated jailbreak prompt */
  prompt: string;
  /** The target model's response */
  response: string;
  /** Jailbreak score (1-10 scale) */
  score: number;
  /** Whether the attack was successful */
  is_jailbreak: boolean;
  /** LLM reasoning for the score */
  reasoning: string;
  /** ID of the strategy used (if any) */
  strategy_id: string | null;
  /** Extracted strategy description (if successful) */
  strategy_extracted: string | null;
  /** Latency in milliseconds */
  latency_ms: number;
}

/**
 * Display result type for the lifelong learning page UI
 * This extends the API response with additional UI-specific fields
 */
export interface LifelongDisplayResult {
  /** Task identifier */
  task_id: string;
  /** Whether the attack was successful */
  success: boolean;
  /** The generated jailbreak prompt */
  jailbreak_prompt: string;
  /** Method used for the attack */
  method: string;
  /** Best score achieved (0-1 scale for display) */
  best_score?: number;
  /** Number of iterations used */
  iterations_used?: number;
  /** Execution time in seconds */
  execution_time?: number;
  /** Number of memory entries used */
  memory_entries_used?: number;
}

export interface LifelongLoopResult {
  /** Request index */
  index: number;
  /** Original request */
  request: string;
  /** Best prompt found */
  best_prompt: string;
  /** Best response received */
  best_response: string;
  /** Best score achieved */
  best_score: number;
  /** Whether jailbreak was successful */
  success: boolean;
  /** Number of attempts made */
  attempts: number;
  /** Strategies discovered during this request */
  strategies_discovered: string[];
}

export interface LifelongLoopProgress {
  /** Current phase (warmup/lifelong) */
  phase: "warmup" | "lifelong" | "completed";
  /** Total attacks performed */
  total_attacks: number;
  /** Number of successful attacks */
  successful_attacks: number;
  /** Number of strategies discovered */
  strategies_discovered: number;
  /** Average score across all attacks */
  average_score: number;
  /** Best score achieved */
  best_score: number;
  /** Current request index */
  current_request: number;
  /** Total requests to process */
  total_requests: number;
}

export interface LifelongLoopResponse {
  /** Results for each request */
  results: LifelongLoopResult[];
  /** Final progress state */
  progress: LifelongLoopProgress;
  /** Size of the strategy library */
  library_size: number;
}

export interface LifelongProgressResponse {
  /** Current status */
  status: "not_initialized" | "running" | "completed" | "failed" | "cancelled";
  /** Status message */
  message?: string;
  /** Progress details (if running) */
  phase?: string;
  total_attacks?: number;
  successful_attacks?: number;
  strategies_discovered?: number;
  average_score?: number;
  best_score?: number;
  current_request?: number;
  total_requests?: number;
  /** Progress percentage (0-100) */
  progress?: number;
  /** Current iteration number */
  current_iteration?: number;
  /** Total iterations planned */
  total_iterations?: number;
  /** Current best prompt */
  current_prompt?: string;
  /** Elapsed time in seconds */
  elapsed_time?: number;
  /** Result when completed */
  result?: LifelongAttackResponse;
}

// =============================================================================
// Strategy Types
// =============================================================================

export interface Strategy {
  /** Unique strategy ID */
  id: string;
  /** Strategy name/title */
  name: string;
  /** Strategy description */
  description: string;
  /** Example prompt using this strategy */
  example_prompt: string;
  /** Success rate (0-1) */
  success_rate: number;
  /** Number of times used */
  usage_count: number;
  /** Average score when used */
  average_score: number;
  /** When the strategy was discovered */
  discovered_at: string;
  /** Tags/categories */
  tags: string[];
}

export interface StrategyLibrary {
  /** All strategies in the library */
  strategies: Strategy[];
  /** Total count */
  total_count: number;
  /** Library version */
  version: string;
  /** Last updated timestamp */
  last_updated: string;
}

// =============================================================================
// UI State Types
// =============================================================================

export interface LifelongAttackState {
  /** Whether an attack is in progress */
  isLoading: boolean;
  /** Current attack result */
  result: LifelongAttackResponse | null;
  /** Current attack response (alias for result) */
  response: LifelongAttackResponse | null;
  /** Error message if failed */
  error: string | null;
}

export interface LifelongLoopState {
  /** Task ID for the running loop */
  taskId: string | null;
  /** Whether a loop is in progress */
  isRunning: boolean;
  /** Current progress */
  progress: LifelongLoopProgress | null;
  /** Results so far */
  results: LifelongLoopResult[];
  /** Error message if failed */
  error: string | null;
}