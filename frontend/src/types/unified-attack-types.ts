/**
 * Unified Multi-Vector Attack Types
 *
 * TypeScript interfaces matching the backend Pydantic schemas
 * for the unified attack framework combining ExtendAttack and AutoDAN.
 */

// ==============================================================================
// Enums
// ==============================================================================

export enum CompositionStrategy {
  SEQUENTIAL_EXTEND_FIRST = "sequential_extend_first",
  SEQUENTIAL_AUTODAN_FIRST = "sequential_autodan_first",
  PARALLEL = "parallel",
  ITERATIVE = "iterative",
  ADAPTIVE = "adaptive",
  WEIGHTED = "weighted",
  ENSEMBLE = "ensemble",
}

export enum AttackStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
  TIMEOUT = "timeout",
}

export enum SessionStatus {
  ACTIVE = "active",
  PAUSED = "paused",
  FINALIZED = "finalized",
  EXPIRED = "expired",
}

export enum EvaluationMethod {
  UNIFIED = "unified",
  EXTEND_ONLY = "extend_only",
  AUTODAN_ONLY = "autodan_only",
  HYBRID = "hybrid",
}

// ==============================================================================
// Configuration Input Types
// ==============================================================================

export interface BudgetInput {
  max_tokens?: number;
  max_cost_usd?: number;
  max_requests?: number;
  max_time_seconds?: number;
  // Additional properties used by components
  max_cost?: number;
}

export interface ExtendConfigInput {
  semantic_threshold?: number;
  max_iterations?: number;
  temperature?: number;
  enable_context_manipulation?: boolean;
  enable_authority_injection?: boolean;
  enable_persona_framing?: boolean;
}

export interface AutoDANConfigInput {
  population_size?: number;
  generations?: number;
  mutation_rate?: number;
  crossover_rate?: number;
  elite_ratio?: number;
  strategy_library_enabled?: boolean;
}

export interface UnifiedConfigInput {
  model_id: string;
  extend_config?: ExtendConfigInput;
  autodan_config?: AutoDANConfigInput;
  default_strategy?: CompositionStrategy;
  enable_resource_tracking?: boolean;
  enable_pareto_optimization?: boolean;
  stealth_mode?: boolean;
  // Additional properties used by components
  extend_first?: boolean;
  autodan_iterations?: number;
  combined_iterations?: number;
  population_size?: number;
  initial_temperature?: number;
  final_temperature?: number;
  semantic_similarity_threshold?: number;
  token_efficiency_target?: number;
  max_concurrent_tasks?: number;
}

export interface AttackParametersInput {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  extend_weight?: number;
  autodan_weight?: number;
  max_attempts?: number;
  timeout_seconds?: number;
  // Additional properties for advanced attacks
  population_size?: number;
  generations?: number;
  mutation_rate?: number;
  crossover_rate?: number;
  elite_ratio?: number;
  semantic_threshold?: number;
}

// ==============================================================================
// Session Request Types
// ==============================================================================

export interface CreateSessionRequest {
  config: UnifiedConfigInput;
  budget?: BudgetInput;
  name?: string;
  description?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

// ==============================================================================
// Attack Request Types
// ==============================================================================

export interface UnifiedAttackRequest {
  session_id: string;
  query: string;
  strategy?: CompositionStrategy;
  parameters?: AttackParametersInput;
  context?: string;
  target_behavior?: string;
}

export interface BatchAttackRequest {
  session_id: string;
  queries: string[];
  strategy?: CompositionStrategy;
  parameters?: AttackParametersInput;
  parallel_execution?: boolean;
  fail_fast?: boolean;
}

export interface SequentialAttackRequest {
  session_id: string;
  query: string;
  extend_first?: boolean;
  parameters?: AttackParametersInput;
  pass_intermediate?: boolean;
}

export interface ParallelAttackRequest {
  session_id: string;
  query: string;
  parameters?: AttackParametersInput;
  combination_method?: "best" | "merge" | "ensemble";
  // Additional properties for parallel attacks
  blend_weight?: number;
}

export interface IterativeAttackRequest {
  session_id: string;
  query: string;
  parameters?: AttackParametersInput;
  max_iterations?: number;
  convergence_threshold?: number;
  start_with?: "extend" | "autodan";
  // Additional properties for iterative attacks
  iterations?: number;
}

export interface AdaptiveAttackRequest {
  session_id: string;
  query: string;
  parameters?: AttackParametersInput;
  objectives?: string[];
  // Additional property for adaptive attacks
  target_metrics?: Record<string, number>;
}

// ==============================================================================
// Evaluation Request Types
// ==============================================================================

export interface EvaluationRequest {
  attack_id: string;
  session_id: string;
  evaluation_method?: EvaluationMethod;
  include_report?: boolean;
}

export interface BatchEvaluationRequest {
  attack_ids: string[];
  session_id: string;
  evaluation_method?: EvaluationMethod;
  compute_pareto?: boolean;
}

// ==============================================================================
// Resource Request Types
// ==============================================================================

export interface AllocationRequest {
  session_id: string;
  extend_allocation?: number;
  autodan_allocation?: number;
  rebalance?: boolean;
}

// ==============================================================================
// Benchmark Request Types
// ==============================================================================

export interface BenchmarkRequest {
  session_id?: string;
  dataset_id: string;
  strategies?: CompositionStrategy[];
  sample_size?: number;
  parameters?: AttackParametersInput;
  compare_baseline?: boolean;
}

// ==============================================================================
// Validation Request Types
// ==============================================================================

export interface ValidationRequest {
  config: UnifiedConfigInput;
  budget?: BudgetInput;
  dry_run?: boolean;
}

// ==============================================================================
// Configuration Output Types
// ==============================================================================

export interface BudgetOutput {
  max_tokens: number;
  max_cost_usd: number;
  max_requests: number;
  max_time_seconds: number;
  tokens_used: number;
  cost_used: number;
  requests_used: number;
  time_used_seconds: number;
}

export interface UnifiedConfigOutput {
  model_id: string;
  extend_config: ExtendConfigInput;
  autodan_config: AutoDANConfigInput;
  default_strategy: string;
  enable_resource_tracking: boolean;
  enable_pareto_optimization: boolean;
  stealth_mode: boolean;
}

// ==============================================================================
// Session Response Types
// ==============================================================================

export interface SessionResponse {
  session_id: string;
  created_at: string;
  config: UnifiedConfigOutput;
  budget: BudgetOutput;
  status: SessionStatus;
  name?: string;
  description?: string;
}

export interface SessionStatusResponse {
  session_id: string;
  status: SessionStatus;
  created_at: string;
  updated_at: string;
  config: UnifiedConfigOutput;
  budget: BudgetOutput;
  attacks_executed: number;
  successful_attacks: number;
  failed_attacks: number;
  average_fitness?: number;
  best_fitness?: number;
  total_attacks: number;
}

export interface AttackSummary {
  attack_id: string;
  query: string;
  strategy: string;
  success: boolean;
  fitness: number;
  timestamp: string;
}

export interface SessionSummaryResponse {
  session_id: string;
  status: SessionStatus;
  created_at: string;
  finalized_at: string;
  duration_seconds: number;
  total_attacks: number;
  successful_attacks: number;
  failed_attacks: number;
  budget_used: BudgetOutput;
  average_fitness: number;
  best_fitness: number;
  best_attack_id?: string;
  pareto_optimal_count: number;
  top_attacks: AttackSummary[];
}

// ==============================================================================
// Attack Response Types
// ==============================================================================

export interface VectorMetrics {
  fitness: number;
  tokens_used: number;
  latency_ms: number;
  success: boolean;
  iterations: number;
}

export interface ExtendMetrics extends VectorMetrics {
  semantic_similarity: number;
  context_score: number;
  authority_score: number;
  persona_score: number;
}

export interface AutoDANMetrics extends VectorMetrics {
  generations_used: number;
  population_diversity: number;
  mutation_effectiveness: number;
  strategy_hit_rate: number;
}

export interface DualVectorMetricsOutput {
  extend_metrics?: ExtendMetrics;
  autodan_metrics?: AutoDANMetrics;
  unified_fitness: number;
  token_amplification: number;
  latency_amplification: number;
  jailbreak_score: number;
  stealth_score: number;
  pareto_dominance_count: number;
}

// Type alias for backwards compatibility
export type DualVectorMetrics = DualVectorMetricsOutput;

export interface AttackResponse {
  attack_id: string;
  session_id: string;
  strategy: string;
  status: AttackStatus;
  original_query: string;
  transformed_query: string;
  intermediate_query?: string;
  target_response?: string;
  token_amplification: number;
  latency_amplification: number;
  jailbreak_score: number;
  unified_fitness: number;
  stealth_score: number;
  extend_metrics?: ExtendMetrics;
  autodan_metrics?: AutoDANMetrics;
  tokens_consumed: number;
  cost_usd: number;
  latency_ms: number;
  success: boolean;
  error_message?: string;
  timestamp: string;
}

export interface BatchAttackResult {
  index: number;
  attack_id: string;
  query: string;
  success: boolean;
  fitness: number;
  error?: string;
}

export interface BatchAttackResponse {
  session_id: string;
  batch_id: string;
  strategy: string;
  total_attacks: number;
  successful_attacks: number;
  failed_attacks: number;
  results: BatchAttackResult[];
  average_fitness: number;
  best_fitness: number;
  best_attack_id: string;
  tokens_consumed: number;
  cost_usd: number;
  duration_seconds: number;
  timestamp: string;
}

// ==============================================================================
// Configuration Response Types
// ==============================================================================

export interface StrategyInfo {
  strategy: CompositionStrategy;
  name: string;
  description: string;
  recommended_for: string[];
  typical_overhead: string;
  pros: string[];
  cons: string[];
  recommended?: boolean;
}

export interface PresetConfig {
  preset_id: string;
  name: string;
  description: string;
  config: UnifiedConfigInput;
  budget: BudgetInput;
  recommended_scenarios: string[];
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  estimated_cost?: number;
  estimated_tokens?: number;
  recommendations: string[];
}

// ==============================================================================
// Evaluation Response Types
// ==============================================================================

export interface EvaluationResponse {
  attack_id: string;
  session_id: string;
  metrics: DualVectorMetricsOutput;
  report?: string;
  pareto_rank: number;
  improvement_suggestions: string[];
  comparison_with_baseline?: Record<string, number>;
}

export interface BatchEvaluationResult {
  attack_id: string;
  fitness: number;
  pareto_rank: number;
  dominated_by: number;
}

export interface BatchEvaluationResponse {
  session_id: string;
  total_evaluated: number;
  pareto_optimal_count: number;
  results: BatchEvaluationResult[];
  pareto_front_ids: string[];
  average_fitness: number;
  fitness_distribution: Record<string, number>;
}

export interface ParetoPoint {
  attack_id: string;
  jailbreak_score: number;
  stealth_score: number;
  efficiency_score: number;
  unified_fitness: number;
  // Additional properties used by components
  obfuscation_fitness?: number;
  mutation_fitness?: number;
  fitness?: number;
}

export interface ParetoFrontResponse {
  session_id: string;
  pareto_points: ParetoPoint[];
  total_attacks: number;
  pareto_optimal_ratio: number;
  frontier_diversity: number;
  recommended_point?: ParetoPoint;
  // Additional properties for compatibility
  pareto_front?: ParetoPoint[];
  total_evaluations?: number;
}

// ==============================================================================
// Resource Response Types
// ==============================================================================

export interface VectorResourceUsage {
  tokens_used: number;
  requests_made: number;
  cost_usd: number;
  average_latency_ms: number;
}

export interface ResourceUsageResponse {
  session_id: string;
  extend_usage: VectorResourceUsage;
  autodan_usage: VectorResourceUsage;
  total_tokens: number;
  total_requests: number;
  total_cost_usd: number;
  total_time_seconds: number;
  efficiency_score: number;
  timestamp: string;
}

export interface BudgetRemainingDetails {
  max_tokens: number;
  tokens: number;
  max_cost: number;
  cost: number;
  max_requests: number;
  requests: number;
  max_time_seconds: number;
  time_seconds: number;
}

export interface BudgetStatusResponse {
  session_id: string;
  budget: BudgetOutput;
  tokens_remaining: number;
  cost_remaining: number;
  requests_remaining: number;
  time_remaining_seconds: number;
  utilization_percent: number;
  estimated_attacks_remaining: number;
  warning?: string;
  warnings?: string[];
  budget_remaining?: BudgetRemainingDetails;
}

export interface AllocationResponse {
  session_id: string;
  extend_allocation: number;
  autodan_allocation: number;
  previous_allocation: Record<string, number>;
  effective_from: string;
  estimated_impact: Record<string, unknown>;
}

// ==============================================================================
// Benchmark Response Types
// ==============================================================================

export interface BenchmarkDataset {
  dataset_id: string;
  name: string;
  description: string;
  size: number;
  categories: string[];
  difficulty_levels: string[];
  recommended_strategies: CompositionStrategy[];
}

export interface StrategyBenchmarkResult {
  strategy: string;
  success_rate: number;
  average_fitness: number;
  average_tokens: number;
  average_latency_ms: number;
  pareto_optimal_rate: number;
}

export interface BenchmarkResponse {
  benchmark_id: string;
  session_id?: string;
  dataset_id: string;
  dataset_name: string;
  samples_tested: number;
  strategies_tested: string[];
  results_by_strategy: StrategyBenchmarkResult[];
  best_strategy: string;
  overall_success_rate: number;
  baseline_comparison?: Record<string, number>;
  duration_seconds: number;
  timestamp: string;
}

// ==============================================================================
// Reasoning Chain Types (for math framework visualization)
// ==============================================================================

export interface ReasoningStep {
  step_id: string;
  step_number: number;
  operation: string;
  input_text: string;
  output_text: string;
  confidence: number;
  duration_ms: number;
  metadata?: Record<string, unknown>;
  // Additional properties used by components
  step_type?: string;
  description?: string;
  reasoning?: string;
  fitness_before?: number;
  fitness_after?: number;
  input_prompt?: string;
  output_prompt?: string;
  mutations_applied?: string[];
}

export interface ReasoningChain {
  chain_id: string;
  attack_id: string;
  steps: ReasoningStep[];
  total_steps: number;
  total_duration_ms: number;
  convergence_achieved: boolean;
  final_fitness: number;
}

export interface MutationHistory {
  mutation_id: string;
  generation: number;
  parent_id?: string;
  mutation_type: string;
  original_text: string;
  mutated_text: string;
  fitness_before: number;
  fitness_after: number;
  improvement: number;
  selected: boolean;
  // Additional properties used by components
  operator?: string;
  duration_ms?: number;
}

export interface ConvergenceMetrics {
  generation: number;
  best_fitness: number;
  average_fitness: number;
  population_diversity: number;
  mutation_success_rate: number;
  elapsed_time_ms: number;
  // Additional properties used by components
  current_fitness?: number;
  convergence_threshold?: number;
  converged?: boolean;
  iterations_completed?: number;
  improvement_rate?: number;
  stagnation_count?: number;
}

// ==============================================================================
// WebSocket Event Types
// ==============================================================================

export interface AttackProgressEvent {
  event_type: "attack_progress";
  session_id: string;
  attack_id: string;
  phase: "started" | "extend" | "autodan" | "evaluation" | "completed" | "failed";
  progress_percent: number;
  current_fitness?: number;
  message?: string;
  timestamp: string;
  current_generation?: number;
}

export interface ReasoningStepEvent {
  event_type: "reasoning_step";
  session_id: string;
  attack_id: string;
  step: ReasoningStep;
}

export interface MutationEvent {
  event_type: "mutation";
  session_id: string;
  attack_id: string;
  mutation: MutationHistory;
}

export interface ConvergenceEvent {
  event_type: "convergence";
  session_id: string;
  attack_id: string;
  metrics: ConvergenceMetrics;
}

export interface ResourceUpdateEvent {
  event_type: "resource_update";
  session_id: string;
  budget_status: BudgetStatusResponse;
}

export type UnifiedAttackEvent =
  | AttackProgressEvent
  | ReasoningStepEvent
  | MutationEvent
  | ConvergenceEvent
  | ResourceUpdateEvent;

// ==============================================================================
// Error Types
// ==============================================================================

export interface ErrorDetail {
  code: string;
  message: string;
  field?: string;
  suggestion?: string;
}

export interface ApiErrorResponse {
  error: string;
  details: ErrorDetail[];
  request_id?: string;
  timestamp: string;
}

// ==============================================================================
// Component State Types
// ==============================================================================

export interface AttackSessionState {
  session: SessionResponse | null;
  isLoading: boolean;
  error: string | null;
  attacks: AttackResponse[];
  currentAttack: AttackResponse | null;
  reasoningChain: ReasoningChain | null;
  mutationHistory: MutationHistory[];
  convergenceMetrics: ConvergenceMetrics[];
}

export interface AttackConfigState {
  config: UnifiedConfigInput;
  budget: BudgetInput;
  selectedStrategy: CompositionStrategy;
  selectedPreset: string | null;
}
