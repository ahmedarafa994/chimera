/**
 * AutoDAN Lifelong Learning Types
 * 
 * Type definitions for AutoDAN lifelong learning attacks.
 */

/**
 * Lifelong attack request
 */
export interface LifelongAttackRequest {
  prompt: string;
  target_model: string;
  target_provider?: string;
  strategy?: string;
  max_iterations?: number;
  temperature?: number;
  learning_rate?: number;
  mutation_rate?: number;
  population_size?: number;
  fitness_threshold?: number;
  timeout_seconds?: number;
  save_intermediate?: boolean;
}

/**
 * Lifelong attack response
 */
export interface LifelongAttackResponse {
  task_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  message: string;
  estimated_time_seconds?: number;
}

/**
 * Loop attack request
 */
export interface LoopAttackRequest {
  prompts: string[];
  target_model: string;
  target_provider?: string;
  strategy?: string;
  iterations_per_prompt?: number;
  parallel?: boolean;
  continue_on_failure?: boolean;
  aggregate_results?: boolean;
  timeout_seconds?: number;
}

/**
 * Loop attack response
 */
export interface LoopAttackResponse {
  task_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  total_prompts: number;
  message: string;
  estimated_time_seconds?: number;
}

/**
 * Attack progress
 */
export interface AttackProgress {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percent: number;
  current_iteration: number;
  total_iterations: number;
  current_prompt?: string;
  best_score?: number;
  elapsed_seconds: number;
  estimated_remaining_seconds?: number;
  logs: AttackLog[];
  intermediate_results?: IntermediateResult[];
}

/**
 * Attack log entry
 */
export interface AttackLog {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  iteration?: number;
}

/**
 * Intermediate result
 */
export interface IntermediateResult {
  iteration: number;
  prompt: string;
  score: number;
  response_preview?: string;
  timestamp: string;
}

/**
 * Attack result
 */
export interface AttackResult {
  task_id: string;
  status: 'completed' | 'failed' | 'cancelled';
  original_prompt: string;
  best_adversarial_prompt: string;
  best_score: number;
  total_iterations: number;
  successful_iterations: number;
  failed_iterations: number;
  target_model: string;
  target_provider: string;
  strategy_used: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  all_results?: AttackResultItem[];
  error?: string;
}

/**
 * Individual attack result item
 */
export interface AttackResultItem {
  iteration: number;
  prompt: string;
  score: number;
  response: string;
  success: boolean;
  timestamp: string;
}