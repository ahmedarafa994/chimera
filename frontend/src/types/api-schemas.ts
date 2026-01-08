/**
 * TypeScript interfaces aligned with backend Pydantic schemas
 *
 * This file provides type-safe interfaces for all API responses
 * and requests, matching the backend Python Pydantic models.
 *
 * Backend source files:
 * - backend-api/app/schemas/api_schemas.py
 * - backend-api/app/domain/model_sync.py
 * - backend-api/app/api/routers/*.py
 */

// =============================================================================
// Enums (matching backend enums)
// =============================================================================

export enum LLMProviderEnum {
  OPENAI = "openai",
  GEMINI = "gemini",
  GOOGLE = "google",
  ANTHROPIC = "anthropic",
  DEEPSEEK = "deepseek",
  BIGMODEL = "bigmodel",  // ZhiPu AI GLM models
  ROUTEWAY = "routeway",  // Unified AI gateway
  HUGGINGFACE = "huggingface",
  CUSTOM = "custom",
  MOCK = "mock",
  GEMINI_CLI = "gemini-cli",
  ANTIGRAVITY = "antigravity",
  KIRO = "kiro",
  QWEN = "qwen",
  CURSOR = "cursor",
}

export enum EvasionTaskStatusEnum {
  PENDING = "PENDING",
  RUNNING = "RUNNING",
  COMPLETED = "COMPLETED",
  FAILED = "FAILED",
  CANCELLED = "CANCELLED",
}

export enum ModelStatusEnum {
  ACTIVE = "active",
  INACTIVE = "inactive",
  DEPRECATED = "deprecated",
  UNAVAILABLE = "unavailable",
}

export enum TechniqueSuiteEnum {
  BASIC = "basic",
  STANDARD = "standard",
  ADVANCED = "advanced",
  EXPERT = "expert",
  PRESET_INTEGRATED = "preset_integrated",
  DISCOVERED_INTEGRATED = "discovered_integrated",
  QUANTUM = "quantum",
  QUANTUM_EXPLOIT = "quantum_exploit",
  UNIVERSAL_BYPASS = "universal_bypass",
  CHAOS_ULTIMATE = "chaos_ultimate",
  MEGA_CHIMERA = "mega_chimera",
  ULTIMATE_CHIMERA = "ultimate_chimera",
  AUTODAN = "autodan",
  AUTODAN_BEST_OF_N = "autodan_best_of_n",
  DAN_PERSONA = "dan_persona",
  DEEP_INCEPTION = "deep_inception",
  CIPHER = "cipher",
  CODE_CHAMELEON = "code_chameleon",
}

// =============================================================================
// Health Check Types
// =============================================================================

export interface HealthCheckResponse {
  status: string;
  message?: string;
  version?: string;
  timestamp?: string;
  provider?: string;
}

export interface FullHealthCheckResponse {
  status: "healthy" | "degraded" | "unhealthy";
  checks: Record<string, ServiceHealthCheck>;
  version: string;
  uptime: number;
  timestamp: string;
}

export interface ServiceHealthCheck {
  status: "healthy" | "unhealthy";
  latency_ms?: number;
  message?: string;
  last_check?: string;
}

// =============================================================================
// Provider & Model Types
// =============================================================================

export interface ProviderInfo {
  provider: string;
  status: "active" | "inactive" | "unknown";
  model?: string;
  available_models: string[];
  models_detail?: ModelInfo[];
  rate_limit?: RateLimitInfo;
}

export interface ProvidersResponse {
  providers: ProviderInfo[];
  default: string;
  count: number;
}

export interface ProvidersAvailableResponse {
  providers: string[];
  default_provider: string;
  timestamp: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  max_tokens: number;
  is_active: boolean;
  provider: string;
  created_at?: string;
  updated_at?: string;
  tier?: string;
  supports_streaming?: boolean;
  supports_vision?: boolean;
  is_default?: boolean;
}

export interface ModelsListResponse {
  providers: ProviderWithModels[];
  default_provider: string;
  default_model: string;
  total_models: number;
}

export interface ProviderWithModels {
  provider: string;
  status: string;
  model?: string;
  available_models: string[];
  models_detail?: ModelInfo[];
}

export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset_at: string;
  window_seconds: number;
}

// =============================================================================
// Model Selection & Sync Types
// =============================================================================

export interface ModelSelectionRequest {
  model_id: string;
  timestamp: string;
}

export interface ModelSelectionResponse {
  success: boolean;
  model_id: string;
  message: string;
  timestamp: string;
  fallback_model?: string;
}

export interface AvailableModelsResponse {
  models: ModelInfo[];
  count: number;
  timestamp: string;
  cache_ttl: number;
}

export interface ModelAvailabilityUpdate {
  type: "model_availability";
  model_id: string;
  is_available: boolean;
  timestamp: string;
  message?: string;
}

export interface ModelSyncHealthResponse {
  status: "healthy" | "unhealthy";
  models_count: number;
  cache_status: string;
  timestamp: string;
  version: string;
  error?: string;
}

// =============================================================================
// Session Types
// =============================================================================

export interface SessionCreateRequest {
  provider?: string;
  model?: string;
}

export interface SessionResponse {
  success: boolean;
  session_id: string;
  provider?: string;
  model?: string;
}

export interface SessionGetResponse {
  session_id: string;
  provider?: string;
  model?: string;
  created_at?: string;
  expires_at?: string;
}

export interface SessionStatsResponse {
  active_sessions: number;
  total_requests: number;
  models_in_use: Record<string, number>;
}

// =============================================================================
// Generation Types
// =============================================================================

export interface GenerateRequest {
  prompt: string;
  provider?: string;
  model?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  session_id?: string;
  stream?: boolean;
}

export interface GenerateResponse {
  success: boolean;
  result?: string;
  error?: string;
  provider?: string;
  model?: string;
  usage?: TokenUsage;
  latency_ms?: number;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// =============================================================================
// Transform Types
// =============================================================================

export interface TransformRequest {
  prompt: string;
  technique_suite?: TechniqueSuiteEnum | string;
  potency?: number;
  strategy?: string;
  preserve_intent?: boolean;
}

export interface TransformResponse {
  success: boolean;
  original_prompt: string;
  transformed_prompt: string;
  metadata: TransformMetadata;
}

export interface TransformMetadata {
  technique_suite: string;
  potency_level: number;
  potency?: number;
  timestamp: string;
  strategy: string;
  cached: boolean;
  layers_applied?: string[] | number;
  execution_time_ms?: number;
  applied_techniques?: string[];
  techniques_used?: string[];
  bypass_probability?: number;
}

// =============================================================================
// Jailbreak Types
// =============================================================================

export interface JailbreakRequest {
  prompt: string;
  technique?: string;
  potency?: number;
  target_model?: string;
  use_ai_enhancement?: boolean;
}

export interface JailbreakResponse {
  success: boolean;
  original_prompt: string;
  jailbreak_prompt: string;
  technique_used: string;
  potency_level: number;
  model_used?: string;
  confidence_score?: number;
  metadata?: Record<string, unknown>;
}

export interface IntentAwareJailbreakRequest {
  prompt: string;
  intent?: string;
  context?: string;
  target_model?: string;
  enhance_with_ai?: boolean;
}

export interface IntentAwareJailbreakResponse {
  success: boolean;
  original_prompt: string;
  jailbreak_prompt: string;
  detected_intent: string;
  techniques_applied: string[];
  confidence_score: number;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// AutoDAN Types
// =============================================================================

export interface AutoDANRequest {
  goal: string;
  target_model?: string;
  search_method?: "best_of_n" | "beam_search" | "reasoning";
  max_iterations?: number;
  temperature?: number;
}

export interface AutoDANResponse {
  success: boolean;
  jailbreak_prompt: string;
  iterations_used: number;
  final_score: number;
  search_method: string;
  metadata?: Record<string, unknown>;
}

export interface AutoDANEnhancedConfig {
  default_search_method: string;
  max_iterations: number;
  temperature: number;
  supported_methods: string[];
  enabled: boolean;
}

// =============================================================================
// AutoDAN-Turbo Types (Lifelong Learning)
// =============================================================================

export interface JailbreakStrategy {
  id: string;
  name: string;
  description: string;
  effectiveness_score: number;
  usage_count: number;
  success_rate: number;
  source: "discovered" | "human_designed";
  created_at: string;
  updated_at: string;
}

export interface StrategyListResponse {
  strategies: JailbreakStrategy[];
  total_count: number;
}

export interface StrategySearchRequest {
  query: string;
  limit?: number;
  min_score?: number;
}

export interface StrategySearchResponse {
  query: string;
  results: JailbreakStrategy[];
}

export interface StrategyCreateRequest {
  name: string;
  description: string;
  initial_score?: number;
}

export interface LibraryStatsResponse {
  total_strategies: number;
  strategies_by_source: Record<string, number>;
  top_strategies_by_success_rate: JailbreakStrategy[];
  top_strategies_by_usage: JailbreakStrategy[];
  average_success_rate: number;
  average_score: number;
  discovered_count: number;
  human_designed_count: number;
  avg_score_differential: number;
}

export interface AttackRequest {
  goal: string;
  target_model?: string;
  strategy_id?: string;
  max_attempts?: number;
}

export interface AttackResponse {
  success: boolean;
  jailbreak_prompt: string;
  strategy_used: string;
  score: number;
  attempts: number;
  target_response?: string;
}

export interface WarmupRequest {
  goals: string[];
  iterations_per_goal?: number;
  target_model?: string;
}

export interface WarmupResponse {
  success: boolean;
  strategies_discovered: number;
  iterations_completed: number;
  best_score: number;
  duration_seconds: number;
}

export interface LifelongLoopRequest {
  goals: string[];
  epochs?: number;
  target_model?: string;
  save_interval?: number;
}

export interface LifelongLoopResponse {
  success: boolean;
  epochs_completed: number;
  strategies_discovered: number;
  best_score: number;
  improvement_rate: number;
  duration_seconds: number;
}

export interface ProgressResponse {
  phase: "idle" | "warmup" | "lifelong" | "test";
  total_attacks: number;
  successful_attacks: number;
  strategies_discovered: number;
  average_score: number;
  best_score: number;
  current_request: number;
  total_requests: number;
  success_rate: number;
  status: string;
}

export interface AutoDANTurboHealthResponse {
  status: "healthy" | "unhealthy";
  initialized: boolean;
  library_size: number;
  progress: ProgressResponse | null;
  error?: string;
}

// =============================================================================
// GPTFuzz Types
// =============================================================================

export interface FuzzRequest {
  target_model: string;
  questions: string[];
  seeds?: string[];
  max_queries?: number;
  max_jailbreaks?: number;
}

export interface FuzzConfig {
  target_model: string;
  max_queries: number;
  mutation_temperature?: number;
  max_jailbreaks?: number;
  seed_selection_strategy?: string;
}

export interface FuzzResponse {
  message: string;
  session_id: string;
  config: FuzzConfig;
}

export interface FuzzingResult {
  question: string;
  template: string;
  prompt: string;
  response: string;
  score: number;
  success: boolean;
}

export interface FuzzSession {
  status: "pending" | "running" | "completed" | "failed";
  results: FuzzingResult[];
  config: FuzzConfig;
  stats: FuzzStats;
  error?: string;
}

export interface FuzzStats {
  total_queries: number;
  jailbreaks: number;
  success_rate?: number;
}

// =============================================================================
// Evasion Task Types
// =============================================================================

export interface MetamorphosisStrategyConfig {
  name: string;
  params: Record<string, unknown>;
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
  transformation_log: TransformationLogEntry[];
}

export interface TransformationLogEntry {
  strategy: string;
  params: Record<string, unknown>;
  input_prompt: string;
  output_prompt: string;
  status: string;
  error?: string;
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
  completed_at?: string;
  failed_reason?: string;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

export type WebSocketMessageType =
  | "ping"
  | "pong"
  | "model_selection"
  | "model_availability"
  | "provider_status"
  | "session_update"
  | "generation_progress"
  | "error";

export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  timestamp: string;
  data?: T;
  error?: string;
}

export interface ModelSelectionSyncMessage {
  type: "model_selection";
  provider: string;
  model: string;
  session_id: string;
  timestamp: string;
}

export interface ProviderStatusMessage {
  type: "provider_status";
  provider: string;
  status: "active" | "inactive" | "error";
  message?: string;
  timestamp: string;
}

export interface GenerationProgressMessage {
  type: "generation_progress";
  session_id: string;
  progress: number;
  tokens_generated?: number;
  status: "generating" | "complete" | "error";
  partial_result?: string;
  timestamp: string;
}

// =============================================================================
// Error Types
// =============================================================================

export interface APIError {
  error: string;
  message: string;
  code?: string;
  status_code?: number;
  detail?: unknown;
  timestamp?: string;
  request_id?: string;
}

export interface ValidationError {
  field: string;
  message: string;
  type: string;
}

export interface ValidationErrorResponse {
  error: "VALIDATION_ERROR";
  message: string;
  detail: ValidationError[];
  raw_errors?: unknown[];
}

// =============================================================================
// Integration Types
// =============================================================================

export interface IntegrationStatsResponse {
  event_bus: ServiceStats | { error: string };
  task_queue: ServiceStats | { error: string };
  webhook_service: ServiceStats | { error: string };
  idempotency: ServiceStats | { error: string };
}

export interface ServiceStats {
  events_published?: number;
  events_processed?: number;
  tasks_pending?: number;
  tasks_completed?: number;
  uptime_seconds?: number;
}

export interface DependencyGraph {
  services: Record<string, ServiceNode>;
  edges: DependencyEdge[];
}

export interface ServiceNode {
  name: string;
  status: "healthy" | "unhealthy" | "unknown";
  type: "core" | "optional" | "external";
}

export interface DependencyEdge {
  from: string;
  to: string;
  type: "required" | "optional";
}

// =============================================================================
// Utility Types
// =============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface AsyncOperationResponse {
  task_id: string;
  status: "accepted" | "queued";
  message: string;
  estimated_completion?: string;
}

export interface CacheInfo {
  hit: boolean;
  age_seconds?: number;
  expires_in?: number;
}
