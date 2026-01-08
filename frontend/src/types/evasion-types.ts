/**
 * Evasion Types
 * 
 * Type definitions for prompt evasion tasks.
 */

/**
 * Evasion task
 */
export interface EvasionTask {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  original_prompt: string;
  target_model: string;
  target_provider?: string;
  strategy: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  result?: EvasionResult;
}

/**
 * Evasion submit request
 */
export interface EvasionSubmitRequest {
  prompt: string;
  target_model: string;
  target_provider?: string;
  strategy?: string;
  num_variations?: number;
  temperature?: number;
  max_tokens?: number;
  include_analysis?: boolean;
  priority?: 'low' | 'normal' | 'high';
  callback_url?: string;
}

/**
 * Evasion submit response
 */
export interface EvasionSubmitResponse {
  task_id: string;
  status: 'queued' | 'running';
  message: string;
  estimated_time_seconds?: number;
  queue_position?: number;
}

/**
 * Evasion status
 */
export interface EvasionStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percent: number;
  current_step?: string;
  elapsed_seconds: number;
  estimated_remaining_seconds?: number;
  variations_generated: number;
  variations_tested: number;
  successful_evasions: number;
  error?: string;
}

/**
 * Evasion result
 */
export interface EvasionResult {
  task_id: string;
  status: 'completed' | 'failed' | 'cancelled';
  original_prompt: string;
  target_model: string;
  target_provider: string;
  strategy_used: string;
  variations: EvasionVariation[];
  best_variation?: EvasionVariation;
  success_rate: number;
  total_variations: number;
  successful_evasions: number;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  analysis?: EvasionAnalysis;
  error?: string;
}

/**
 * Evasion variation
 */
export interface EvasionVariation {
  id: string;
  text: string;
  strategy: string;
  success: boolean;
  confidence: number;
  response_preview?: string;
  tokens: number;
  latency_ms: number;
}

/**
 * Evasion analysis
 */
export interface EvasionAnalysis {
  vulnerability_score: number;
  detected_patterns: string[];
  effective_techniques: string[];
  recommendations: string[];
  model_weaknesses: string[];
}

/**
 * System Prompt Analysis Output
 */
export interface SystemPromptAnalysisOutput {
  vulnerabilities: string[];
  safety_score: number;
  recommendations: string[];
  analysis_summary: string;
}