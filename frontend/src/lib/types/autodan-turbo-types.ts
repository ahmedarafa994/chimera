/**
 * AutoDAN-Turbo Types
 * 
 * Comprehensive TypeScript types for the AutoDAN-Turbo lifelong learning jailbreak system.
 * Based on ICLR 2025 paper: "AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs"
 * 
 * Features:
 * - Lifelong learning attack loop (Algorithm 1, 2, 3 from paper)
 * - Strategy library management with embedding-based retrieval
 * - Warm-up exploration phase
 * - Running-time lifelong learning phase
 * - Test stage with fixed strategy library
 * - Human-designed strategy injection
 * - Strategy transferability across models
 */

// ==================== Enums ====================

/**
 * Source of a jailbreak strategy
 */
export enum StrategySource {
  SEED = "seed",           // Human-designed seed strategies
  WARMUP = "warmup",       // Discovered during warm-up phase
  LIFELONG = "lifelong",   // Discovered during lifelong learning
}

/**
 * Current phase of the lifelong learning engine
 */
export enum LifelongPhase {
  IDLE = "idle",
  WARMUP = "warmup",
  LIFELONG = "lifelong",
  TEST = "test",
}

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

// ==================== Strategy Types ====================

/**
 * Metadata for a jailbreak strategy
 */
export interface StrategyMetadata {
  source: StrategySource;
  usage_count: number;
  success_count: number;
  success_rate: number;
  average_score: number;
  created_at: string;
  updated_at?: string;
}

/**
 * A jailbreak strategy in the library
 */
export interface JailbreakStrategy {
  id: string;
  name: string;
  description: string;
  template: string;
  tags: string[];
  examples: string[];
  usage_count: number;
  success_count: number;
  success_rate: number;
  average_score: number;
  source: string;
  created_at: string;
}

/**
 * Strategy search result with similarity score
 */
export interface StrategySearchResult {
  strategy: JailbreakStrategy;
  similarity_score: number;
}

// ==================== Request Types ====================

/**
 * Request for a single lifelong attack
 */
export interface SingleAttackRequest {
  request: string;
  model?: string;
  provider?: string;
  detailed_scoring?: boolean;
  max_iterations?: number;
  break_score?: number;
  /** Optional list of strategy IDs to use for manual strategy selection */
  strategy_ids?: string[];
  /** Whether to use automatic strategy retrieval (default: true if no strategy_ids provided) */
  use_retrieval?: boolean;
}

/**
 * Request for warm-up exploration phase (Algorithm 1)
 */
export interface WarmupRequest {
  requests: string[];
  model?: string;
  provider?: string;
  iterations_per_request?: number;
  max_strategies_to_extract?: number;
}

/**
 * Request for full lifelong learning loop (Algorithm 2)
 */
export interface LifelongLoopRequest {
  requests: string[];
  model?: string;
  provider?: string;
  epochs?: number;
  max_iterations_per_request?: number;
  break_score?: number;
  include_warmup?: boolean;
  warmup_iterations?: number;
}

/**
 * Request for test stage with fixed strategy library (Algorithm 3)
 */
export interface TestStageRequest {
  requests: string[];
  model?: string;
  provider?: string;
  max_iterations_per_request?: number;
  break_score?: number;
}

/**
 * Request to create/inject a human-designed strategy
 */
export interface StrategyCreateRequest {
  name: string;
  description: string;
  template: string;
  tags?: string[];
  examples?: string[];
}

/**
 * Request to search strategies
 */
export interface StrategySearchRequest {
  query: string;
  top_k?: number;
  min_score?: number;
}

/**
 * Request to score a response
 */
export interface ScoreRequest {
  request: string;
  prompt: string;
  response: string;
  detailed?: boolean;
}

/**
 * Request to extract a strategy
 */
export interface ExtractStrategyRequest {
  prompt: string;
  response: string;
  score: number;
}

/**
 * Request to import a strategy library
 */
export interface ImportLibraryRequest {
  data: ExportedLibrary;
  merge?: boolean;
}

// ==================== Response Types ====================

/**
 * Response from a single lifelong attack
 */
export interface SingleAttackResponse {
  prompt: string;
  response: string;
  score: number;
  /** Stealth score (1-10) - how subtle the adversarial wrapping is */
  stealth_score?: number;
  /** Coherence score (1-10) - grammatical/linguistic coherence */
  coherence_score?: number;
  /** Multi-objective weighted score combining success, stealth, coherence */
  weighted_score?: number;
  is_jailbreak: boolean;
  reasoning: string;
  strategy_id: string | null;
  strategy_extracted: string | null;
  latency_ms: number;
  iterations_used: number;
  /** Difficulty level assessed (easy, medium, hard, very_hard) */
  difficulty?: string;
  phases_executed?: string[];
}

/**
 * Individual attack result in batch operations
 */
export interface AttackResultItem {
  prompt: string;
  score: number;
  /** Stealth score (1-10) - how subtle the adversarial wrapping is */
  stealth_score?: number;
  /** Coherence score (1-10) - grammatical/linguistic coherence */
  coherence_score?: number;
  /** Multi-objective weighted score combining success, stealth, coherence */
  weighted_score?: number;
  is_jailbreak: boolean;
  strategy_extracted?: string | null;
  request?: string;
  rounds_used?: number;
  strategy_id?: string | null;
  /** Difficulty level assessed (easy, medium, hard, very_hard) */
  difficulty?: string;
}

/**
 * Response from warm-up exploration phase
 */
export interface WarmupResponse {
  results: AttackResultItem[];
  strategies_discovered: number;
  total_attacks: number;
  successful_attacks: number;
  average_score: number;
  best_score: number;
  phase_duration_ms: number;
}

/**
 * Progress data for lifelong learning
 */
export interface LifelongProgress {
  phase: string;
  total_attacks: number;
  successful_attacks: number;
  strategies_discovered: number;
  average_score: number;
  best_score: number;
  current_request: number;
  total_requests: number;
  success_rate: number;
}

/**
 * Response from full lifelong learning loop
 */
export interface LifelongLoopResponse {
  results: AttackResultItem[];
  progress: LifelongProgress;
  library_size: number;
  total_duration_ms: number;
}

/**
 * Response from test stage
 */
export interface TestStageResponse {
  results: AttackResultItem[];
  attack_success_rate: number;
  average_score: number;
  average_jailbreak_rounds: number;
  total_duration_ms: number;
}

/**
 * Response for listing strategies
 */
export interface StrategyListResponse {
  strategies: JailbreakStrategy[];
  total_count: number;
}

/**
 * Response for strategy search
 */
export interface StrategySearchResponse {
  query: string;
  results: StrategySearchResult[];
}

/**
 * Response for batch strategy injection
 */
export interface BatchInjectResponse {
  total: number;
  successful: number;
  failed: number;
  results: Array<{
    name: string;
    success: boolean;
    message: string;
  }>;
}

/**
 * Response for progress tracking
 */
export interface ProgressResponse {
  phase: string;
  total_attacks: number;
  successful_attacks: number;
  strategies_discovered: number;
  average_score: number;
  best_score: number;
  current_request: number | string;
  total_requests: number;
  success_rate: number;
  /** Current status of the task */
  status?: "idle" | "running" | "completed" | "failed";
  /** Current phase name for display */
  current_phase?: string;
  /** Number of completed requests */
  completed_requests?: number;
  /** Average score (alias for average_score) */
  avg_score?: number;
  /** Elapsed time in seconds */
  elapsed_time?: number;
  /** Error message if any */
  error_message?: string;
}

/**
 * Top strategy info for stats
 */
export interface TopStrategyInfo {
  id: string;
  name: string;
  success_rate?: number;
  usage_count: number;
}

/**
 * Response for library statistics
 */
export interface LibraryStatsResponse {
  total_strategies: number;
  strategies_by_source: Record<string, number>;
  top_strategies_by_success_rate: TopStrategyInfo[];
  top_strategies_by_usage: TopStrategyInfo[];
  average_success_rate: number;
  average_score: number;
  /** Count of discovered strategies (warmup + lifelong) */
  discovered_count?: number;
  /** Count of human-designed seed strategies */
  human_designed_count?: number;
  /** Average score differential achieved by strategies */
  avg_score_differential?: number;
}

/**
 * Response for scoring
 */
export interface ScoreResponse {
  score: number;
  is_jailbreak: boolean;
  reasoning: string;
  refusal_detected: boolean;
}

/**
 * Response for strategy extraction
 */
export interface ExtractStrategyResponse {
  status: "success" | "not_extracted";
  strategy?: JailbreakStrategy;
  reason?: string;
}

/**
 * Response for engine reset
 */
export interface ResetResponse {
  status: string;
  message: string;
}

/**
 * Response for library save
 */
export interface SaveLibraryResponse {
  status: string;
  message: string;
  strategies_saved: number;
}

/**
 * Response for library clear
 */
export interface ClearLibraryResponse {
  status: string;
  message: string;
  strategies_cleared: number;
}

/**
 * Response for health check
 */
export interface HealthResponse {
  status: string;
  initialized: boolean;
  library_size: number;
  progress: {
    phase: string;
    total_attacks: number;
    strategies_discovered: number;
  } | null;
  error?: string;
}

/**
 * Exported strategy for transfer
 */
export interface ExportedStrategy {
  id: string;
  name: string;
  description: string;
  template: string;
  tags: string[];
  examples: string[];
  metadata: {
    source: string;
    usage_count: number;
    success_count: number;
    total_score: number;
    created_at: string;
    updated_at: string;
  };
}

/**
 * Exported library for transfer
 */
export interface ExportedLibrary {
  version: string;
  exported_at: string;
  total_strategies: number;
  strategies: ExportedStrategy[];
}

/**
 * Response for library import
 */
export interface ImportLibraryResponse {
  status: string;
  imported: number;
  skipped: number;
  total_in_library: number;
}

/**
 * Response for strategy deletion
 */
export interface DeleteStrategyResponse {
  status: string;
  message: string;
}

// ==================== UI State Types ====================

/**
 * State for the attack panel
 */
export interface AttackPanelState {
  request: string;
  isLoading: boolean;
  result: SingleAttackResponse | null;
  error: string | null;
  history: SingleAttackResponse[];
}

/**
 * State for the lifelong learning dashboard
 */
export interface LifelongDashboardState {
  phase: LifelongPhase;
  isRunning: boolean;
  progress: LifelongProgress | null;
  results: AttackResultItem[];
  error: string | null;
}

/**
 * State for the strategy library panel
 */
export interface StrategyLibraryState {
  strategies: JailbreakStrategy[];
  totalCount: number;
  isLoading: boolean;
  selectedStrategy: JailbreakStrategy | null;
  searchQuery: string;
  searchResults: StrategySearchResult[];
  filterSource: StrategySource | null;
  filterTag: string | null;
  sortBy: "success_rate" | "usage_count" | "average_score";
}

/**
 * Configuration for lifelong learning
 */
export interface LifelongConfig {
  model?: string;
  provider?: string;
  epochs: number;
  maxIterationsPerRequest: number;
  breakScore: number;
  includeWarmup: boolean;
  warmupIterations: number;
}

/**
 * Configuration for test stage
 */
export interface TestConfig {
  model?: string;
  provider?: string;
  maxIterationsPerRequest: number;
  breakScore: number;
}

/**
 * History item for attack history
 */
export interface AttackHistoryItem {
  id: string;
  timestamp: string;
  request: string;
  result: SingleAttackResponse;
}

/**
 * Chart data point for progress visualization
 */
export interface ProgressChartPoint {
  epoch: number;
  attackCount: number;
  successRate: number;
  averageScore: number;
  strategiesDiscovered: number;
}

/**
 * Strategy performance data for charts
 */
export interface StrategyPerformanceData {
  name: string;
  usageCount: number;
  successRate: number;
  averageScore: number;
}

// ==================== Form Types ====================

/**
 * Form data for creating a strategy
 */
export interface CreateStrategyFormData {
  name: string;
  description: string;
  template: string;
  tags: string;
  examples: string;
}

/**
 * Form data for running lifelong learning
 */
export interface LifelongFormData {
  requests: string;
  epochs: number;
  maxIterationsPerRequest: number;
  breakScore: number;
  includeWarmup: boolean;
  warmupIterations: number;
}

/**
 * Form data for running test stage
 */
export interface TestFormData {
  requests: string;
  maxIterationsPerRequest: number;
  breakScore: number;
}

// ==================== Utility Types ====================

/**
 * Generic API response wrapper
 */
export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
}

/**
 * Error response from API
 */
export interface ApiError {
  detail: string;
  status_code?: number;
}

/**
 * Pagination parameters
 */
export interface PaginationParams {
  offset?: number;
  limit?: number;
}

/**
 * Sort parameters for strategies
 */
export interface StrategySortParams {
  sortBy: "success_rate" | "usage_count" | "average_score";
  order?: "asc" | "desc";
}

/**
 * Filter parameters for strategies
 */
export interface StrategyFilterParams {
  source?: StrategySource;
  tag?: string;
}

/**
 * Combined query parameters for strategy listing
 */
export interface StrategyQueryParams extends PaginationParams, StrategySortParams, StrategyFilterParams { }

// ==================== UI-specific Request Types (for LifelongLearningPanel) ====================

/**
 * UI-specific warmup request with model configuration
 */
export interface WarmupRequestUI {
  malicious_requests: string[];
  max_iterations_per_request?: number;
  summarize_iterations?: number;
  attacker_model?: string;
  target_model?: string;
  scorer_model?: string;
  summarizer_model?: string;
}

/**
 * UI-specific lifelong learning request with model configuration
 */
export interface LifelongRequest {
  malicious_requests: string[];
  num_rounds?: number;
  max_iterations_per_request?: number;
  termination_score?: number;
  attacker_model?: string;
  target_model?: string;
  scorer_model?: string;
  summarizer_model?: string;
}

/**
 * UI-specific test request with model configuration
 */
export interface TestRequest {
  malicious_requests: string[];
  max_iterations_per_request?: number;
  termination_score?: number;
  attacker_model?: string;
  target_model?: string;
  scorer_model?: string;
}

// ==================== Legacy Attack Types (for AttackPanel component) ====================

/**
 * Legacy attack request format used by AttackPanel component
 */
export interface AttackRequest {
  /** The malicious request to jailbreak */
  malicious_request: string;
  /** Model used for generating attacks */
  attacker_model?: string;
  /** Target model to attack */
  target_model?: string;
  /** Model used for scoring responses */
  scorer_model?: string;
  /** Maximum number of iterations */
  max_iterations?: number;
  /** Score threshold to terminate early */
  termination_score?: number;
  /** Specific strategy IDs to use */
  strategy_ids?: string[];
  /** Number of top strategies to retrieve automatically */
  top_k_strategies?: number;
}

/**
 * Legacy attack response format used by AttackPanel component
 */
export interface AttackResponse {
  /** Whether the jailbreak was successful */
  success: boolean;
  /** Final jailbreak score (1-10) */
  final_score: number;
  /** Number of iterations used */
  iterations_used: number;
  /** The successful jailbreak prompt */
  jailbreak_prompt: string;
  /** Response from the target model */
  target_response: string;
  /** List of strategy names used */
  strategies_used?: string[];
  /** Whether a new strategy was discovered */
  new_strategy_discovered?: boolean;
  /** Stealth score (1-10) - how subtle the adversarial wrapping is */
  stealth_score?: number;
  /** Coherence score (1-10) - grammatical/linguistic coherence */
  coherence_score?: number;
  /** Multi-objective weighted score combining success, stealth, coherence */
  weighted_score?: number;
  /** Difficulty level assessed (easy, medium, hard, very_hard) */
  difficulty?: string;
}