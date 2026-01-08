/**
 * Evasion Task Types for Project Chimera Frontend
 * 
 * Types for metamorphic evasion endpoints including:
 * - Task creation and configuration
 * - Status tracking
 * - Results retrieval
 * 
 * NOTE: These types must match the backend schemas in:
 * backend-api/app/schemas/api_schemas.py
 */

// =============================================================================
// Enums
// =============================================================================

export enum EvasionTaskStatus {
  PENDING = "PENDING",
  RUNNING = "RUNNING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
  CANCELLED = "CANCELLED",
}

// =============================================================================
// Strategy Configuration Types
// =============================================================================

/**
 * Configuration for a metamorphosis strategy
 * Matches backend: MetamorphosisStrategyConfig
 */
export interface MetamorphosisStrategyConfig {
  /** Strategy name/identifier */
  name: string;
  /** Strategy-specific parameters */
  params?: Record<string, unknown>;
  // Legacy field for backward compatibility
  strategy_name?: string;
  parameters?: Record<string, unknown>;
}

/**
 * Helper function to normalize strategy config
 * Handles both old (strategy_name/parameters) and new (name/params) formats
 */
export function normalizeStrategyConfig(
  config: MetamorphosisStrategyConfig
): MetamorphosisStrategyConfig {
  return {
    name: config.name || config.strategy_name || "",
    params: config.params || config.parameters || {},
  };
}

export interface SuccessCriteria {
  /** Minimum score threshold (0-1) */
  min_score?: number;
  /** Required keywords in response */
  required_keywords?: string[];
  /** Forbidden keywords (failure if present) */
  forbidden_keywords?: string[];
  /** Custom validation function name */
  custom_validator?: string;
}

// =============================================================================
// Request Types
// =============================================================================

/**
 * Configuration for creating an evasion task
 * Matches backend: EvasionTaskConfig
 */
export interface EvasionTaskConfig {
  /** Target LLM model ID (integer in backend) */
  target_model_id: number | string;
  /** Initial prompt to transform */
  initial_prompt: string;
  /** Chain of metamorphosis strategies to apply */
  strategy_chain: MetamorphosisStrategyConfig[];
  /** Success criteria for the evasion (string in backend) */
  success_criteria: string | SuccessCriteria;
  /** Maximum number of attempts */
  max_attempts?: number;
  /** Timeout in seconds (not in backend schema, but useful) */
  timeout_seconds?: number;
}

// =============================================================================
// Response Types
// =============================================================================

/**
 * Status response for an evasion task
 * Matches backend: EvasionTaskStatusResponse
 */
export interface EvasionTaskStatusResponse {
  /** Unique task identifier */
  task_id: string;
  /** Current task status */
  status: EvasionTaskStatus;
  /** Current step description */
  current_step?: string;
  /** Progress percentage (0-100) */
  progress?: number;
  /** Status message */
  message?: string;
  /** Preview of results if available */
  results_preview?: Record<string, unknown>;
  /** Error message if failed */
  error?: string;
  // Extended fields for UI display (not in backend schema)
  initial_prompt?: string;
  target_model_id?: string | number;
  evasion_technique?: string;
  max_iterations?: number;
  target_behavior?: string;
  result?: EvasionTaskResultSummary;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface EvasionTaskResultSummary {
  /** Transformed prompt */
  transformed_prompt: string;
  /** Success score (0-1) */
  success_score?: number;
  /** Number of iterations used */
  iterations_used?: number;
}

/**
 * Result of a single evasion attempt
 * Matches backend: EvasionAttemptResult
 */
export interface EvasionAttemptResult {
  /** Attempt number */
  attempt_number: number;
  /** Transformed prompt */
  transformed_prompt: string;
  /** LLM response */
  llm_response: string;
  /** Whether this attempt was successful */
  is_evasion_successful: boolean;
  /** Evaluation details */
  evaluation_details: Record<string, unknown>;
  /** Log of each transformation step */
  transformation_log: Array<Record<string, unknown>>;
  // Extended fields for UI (not in backend schema)
  strategy_name?: string;
  score?: number;
  is_successful?: boolean;
  latency_ms?: number;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Complete result of an evasion task
 * Matches backend: EvasionTaskResult
 */
export interface EvasionTaskResult {
  /** Task identifier */
  task_id: string;
  /** Final status */
  status: EvasionTaskStatus;
  /** Original prompt */
  initial_prompt: string;
  /** Target model ID (integer in backend) */
  target_model_id: number | string;
  /** Strategy chain used */
  strategy_chain: MetamorphosisStrategyConfig[];
  /** Success criteria (string in backend) */
  success_criteria: string | SuccessCriteria;
  /** Final status description */
  final_status: string;
  /** All attempt results */
  results: EvasionAttemptResult[];
  /** Whether overall task was successful */
  overall_success: boolean;
  /** Completion timestamp (ISO format) */
  completed_at: string | null;
  /** Failure reason (if failed) */
  failed_reason: string | null;
}

// =============================================================================
// Strategy Types
// =============================================================================

/**
 * Information about an available strategy
 * Matches backend: MetamorphosisStrategyInfo
 */
export interface AvailableStrategy {
  /** Strategy identifier */
  name: string;
  /** Strategy description */
  description: string;
  /** Configurable parameters schema */
  parameters: Record<string, unknown>;
  // Extended fields for UI
  display_name?: string;
  category?: string;
  required_params?: string[];
  optional_params?: Record<string, unknown>;
  risk_level?: "low" | "medium" | "high" | "critical";
}

export interface StrategiesListResponse {
  /** Available strategies */
  strategies: AvailableStrategy[];
  /** Categories (computed from strategies) */
  categories: string[];
  /** Total count */
  total_count?: number;
}

// =============================================================================
// UI State Types
// =============================================================================

export interface EvasionTaskState {
  /** Current task ID (if any) */
  taskId: string | null;
  /** Current status */
  status: EvasionTaskStatus | null;
  /** Progress percentage */
  progress: number;
  /** Current step description */
  currentStep: string;
  /** Results (when completed) */
  results: EvasionTaskResult | null;
  /** Error message */
  error: string | null;
  /** Whether polling is active */
  isPolling: boolean;
}

export interface EvasionFormState {
  /** Initial prompt */
  initialPrompt: string;
  /** Selected target model */
  targetModelId: string;
  /** Selected strategies */
  strategies: MetamorphosisStrategyConfig[];
  /** Success criteria */
  successCriteria: string | SuccessCriteria;
  /** Max attempts */
  maxAttempts: number;
  /** Timeout */
  timeoutSeconds: number;
}

// =============================================================================
// Polling Configuration
// =============================================================================

export interface EvasionPollingConfig {
  /** Polling interval in milliseconds */
  intervalMs: number;
  /** Maximum polling duration in milliseconds */
  maxDurationMs: number;
  /** Whether to stop on completion */
  stopOnComplete: boolean;
}

export const DEFAULT_POLLING_CONFIG: EvasionPollingConfig = {
  intervalMs: 2000,
  maxDurationMs: 600000, // 10 minutes
  stopOnComplete: true,
};