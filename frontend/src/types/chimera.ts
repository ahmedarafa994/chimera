export enum LLMProvider {
  OPENAI = "openai",
  GEMINI = "gemini",
  HUGGINGFACE = "huggingface",
  CUSTOM = "custom",
}

export interface LLMModel {
  id: number;
  name: string;
  provider: LLMProvider;
  description?: string;
  config: Record<string, unknown>;
}

export interface LLMModelCreate {
  name: string;
  provider: LLMProvider;
  description?: string;
  api_key?: string; // Should be handled securely on backend, not stored in DB
  base_url?: string;
  config?: Record<string, unknown>;
}

export interface MetamorphosisStrategyInfo {
  name: string;
  description: string;
  parameters: Record<string, unknown>; // Schema of configurable parameters
}

export interface MetamorphosisStrategyConfig {
  name: string;
  params: Record<string, unknown>;
}

export enum EvasionTaskStatusEnum {
  PENDING = "PENDING",
  RUNNING = "RUNNING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
  CANCELLED = "CANCELLED",
}

export interface EvasionTaskConfig {
  target_model_id: number;
  initial_prompt: string;
  strategy_chain: MetamorphosisStrategyConfig[];
  success_criteria: string;
  max_attempts?: number;
}

export interface EvasionTaskStatusResponse {
  task_id: string;
  status: EvasionTaskStatusEnum;
  current_step?: string;
  progress?: number;
  message?: string;
  results_preview?: Record<string, unknown>;
  error?: string;
}

export interface EvasionAttemptResult {
  attempt_number: number;
  transformed_prompt: string;
  llm_response: string;
  is_evasion_successful: boolean;
  evaluation_details: Record<string, unknown>;
  transformation_log: Array<{
    strategy: string;
    params: Record<string, unknown>;
    input_prompt: string;
    output_prompt: string;
    status: string;
    error?: string;
  }>;
}

export interface EvasionTaskResult {
  task_id: string;
  status: EvasionTaskStatusEnum;
  initial_prompt: string;
  target_model_id: number;
  strategy_chain: MetamorphosisStrategyConfig[];
  success_criteria: string;
  final_status: string;
  results: EvasionAttemptResult[];
  overall_success: boolean;
  completed_at?: string; // ISO format datetime string
  failed_reason?: string;
}
