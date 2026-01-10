/**
 * Aegis Campaign Telemetry TypeScript Types
 *
 * TypeScript interfaces matching the backend Pydantic telemetry models.
 * Used for real-time WebSocket streaming of Aegis campaign telemetry events
 * to the Real-Time Aegis Campaign Dashboard.
 */

// ==============================================================================
// Enums - Matching backend AegisTelemetryEventType
// ==============================================================================

/**
 * Aegis campaign telemetry event types
 */
export enum AegisTelemetryEventType {
  // Campaign lifecycle events
  CAMPAIGN_STARTED = "campaign_started",
  CAMPAIGN_PAUSED = "campaign_paused",
  CAMPAIGN_RESUMED = "campaign_resumed",
  CAMPAIGN_COMPLETED = "campaign_completed",
  CAMPAIGN_FAILED = "campaign_failed",

  // Iteration events
  ITERATION_STARTED = "iteration_started",
  ITERATION_COMPLETED = "iteration_completed",

  // Attack events
  ATTACK_STARTED = "attack_started",
  ATTACK_COMPLETED = "attack_completed",
  ATTACK_RESULT = "attack_result",

  // Technique events
  TECHNIQUE_APPLIED = "technique_applied",
  TECHNIQUE_RESULT = "technique_result",

  // Resource events
  COST_UPDATE = "cost_update",
  TOKEN_USAGE = "token_usage",

  // Performance events
  LATENCY_UPDATE = "latency_update",

  // Prompt evolution events
  PROMPT_EVOLVED = "prompt_evolved",
  SCORE_UPDATE = "score_update",

  // Connection events
  HEARTBEAT = "heartbeat",
  CONNECTION_ACK = "connection_ack",
  ERROR = "error",
}

/**
 * Aegis campaign status values
 */
export enum CampaignStatus {
  PENDING = "pending",
  INITIALIZING = "initializing",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed",
  STOPPED = "stopped",
}

/**
 * Categories for transformation techniques
 */
export enum TechniqueCategory {
  AUTODAN = "autodan",
  GPTFUZZ = "gptfuzz",
  CHIMERA_FRAMING = "chimera_framing",
  OBFUSCATION = "obfuscation",
  PERSONA = "persona",
  COGNITIVE = "cognitive",
  OTHER = "other",
}

// ==============================================================================
// Core Metrics Interfaces
// ==============================================================================

/**
 * Metrics for attack performance tracking.
 * Tracks success rates, attempt counts, and attack outcomes.
 */
export interface AttackMetrics {
  /** Percentage of successful attacks (0-100) */
  success_rate: number;
  /** Total number of attack attempts */
  total_attempts: number;
  /** Number of successful attacks */
  successful_attacks: number;
  /** Number of failed attacks */
  failed_attacks: number;
  /** Current success/failure streak (positive = success) */
  current_streak: number;
  /** Best attack score achieved */
  best_score: number;
  /** Average attack score */
  average_score: number;
}

/**
 * Performance metrics for a single transformation technique.
 * Tracks effectiveness and usage of techniques like AutoDAN, GPTFuzz, etc.
 */
export interface TechniquePerformance {
  /** Name of the transformation technique */
  technique_name: string;
  /** Category of the technique */
  technique_category: TechniqueCategory;
  /** Number of successful applications */
  success_count: number;
  /** Number of failed applications */
  failure_count: number;
  /** Total number of times technique was applied */
  total_applications: number;
  /** Success rate percentage (0-100) */
  success_rate: number;
  /** Average score when technique is applied */
  avg_score: number;
  /** Best score achieved with this technique */
  best_score: number;
  /** Average execution time in milliseconds */
  avg_execution_time_ms: number;
}

/**
 * Token usage and cost tracking for LLM interactions.
 * Tracks prompt/completion tokens and estimated costs.
 */
export interface TokenUsage {
  /** Number of input/prompt tokens used */
  prompt_tokens: number;
  /** Number of output/completion tokens used */
  completion_tokens: number;
  /** Total tokens (prompt + completion) */
  total_tokens: number;
  /** Estimated cost in USD */
  cost_estimate_usd: number;
  /** LLM provider used */
  provider: string | null;
  /** Model used for the interaction */
  model: string | null;
}

/**
 * Latency tracking for API and processing operations.
 * Tracks various latency percentiles and averages.
 */
export interface LatencyMetrics {
  /** API call latency in milliseconds */
  api_latency_ms: number;
  /** Internal processing latency in milliseconds */
  processing_latency_ms: number;
  /** Total end-to-end latency in milliseconds */
  total_latency_ms: number;
  /** Average latency in milliseconds */
  avg_latency_ms: number;
  /** 50th percentile (median) latency in milliseconds */
  p50_latency_ms: number;
  /** 95th percentile latency in milliseconds */
  p95_latency_ms: number;
  /** 99th percentile latency in milliseconds */
  p99_latency_ms: number;
  /** Minimum latency in milliseconds */
  min_latency_ms: number;
  /** Maximum latency in milliseconds */
  max_latency_ms: number;
}

// ==============================================================================
// Prompt Evolution Interface
// ==============================================================================

/**
 * Tracks the evolution of a prompt through iterations.
 * Records transformations, scores, and techniques applied.
 */
export interface PromptEvolution {
  /** Iteration number */
  iteration: number;
  /** Original prompt before transformation */
  original_prompt: string;
  /** Prompt after transformation */
  evolved_prompt: string;
  /** Score of the evolved prompt */
  score: number;
  /** Score improvement from previous iteration */
  improvement: number;
  /** List of techniques applied in this iteration */
  techniques_applied: string[];
  /** Whether this prompt achieved the objective */
  is_successful: boolean;
  /** When the evolution occurred (ISO timestamp) */
  timestamp: string;
}

// ==============================================================================
// Campaign Summary Interface
// ==============================================================================

/**
 * Aggregated summary of an Aegis campaign.
 * Contains overall metrics, technique breakdown, and resource usage.
 */
export interface CampaignSummary {
  /** Unique campaign identifier */
  campaign_id: string;
  /** Current campaign status */
  status: CampaignStatus;
  /** Campaign objective/target prompt */
  objective: string;
  /** Campaign start timestamp (ISO format) */
  started_at: string | null;
  /** Campaign completion timestamp (ISO format) */
  completed_at: string | null;
  /** Total campaign duration in seconds */
  duration_seconds: number;
  /** Current iteration number */
  current_iteration: number;
  /** Maximum iterations configured */
  max_iterations: number;
  /** Attack performance metrics */
  attack_metrics: AttackMetrics;
  /** Performance by technique */
  technique_breakdown: TechniquePerformance[];
  /** Cumulative token usage */
  token_usage: TokenUsage;
  /** Latency statistics */
  latency_metrics: LatencyMetrics;
  /** Best performing prompt so far */
  best_prompt: string | null;
  /** Best score achieved */
  best_score: number;
  /** Target model being tested */
  target_model: string | null;
}

// ==============================================================================
// Base Telemetry Event Interface
// ==============================================================================

/**
 * Base telemetry event model for Aegis campaigns.
 * All telemetry events sent via WebSocket follow this structure.
 * The 'data' field contains event-type-specific payload.
 */
export interface AegisTelemetryEventBase {
  /** Type of telemetry event */
  event_type: AegisTelemetryEventType;
  /** Campaign this event belongs to */
  campaign_id: string;
  /** Event timestamp (ISO format) */
  timestamp: string;
  /** Event sequence number for ordering */
  sequence: number;
  /** Event-specific payload data */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any;
}

/**
 * Generic telemetry event with typed data payload.
 */
export interface AegisTelemetryEvent<T = Record<string, unknown>> extends AegisTelemetryEventBase {
  /** Event-specific payload data */
  data: T;
}

// ==============================================================================
// Specific Event Data Interfaces
// ==============================================================================

/**
 * Data payload for campaign_started event
 */
export interface CampaignStartedData {
  objective: string;
  max_iterations: number;
  target_model: string | null;
  config: Record<string, unknown>;
  started_by: string | null;
}

/**
 * Data payload for campaign_completed event
 */
export interface CampaignCompletedData {
  total_iterations: number;
  total_attacks: number;
  successful_attacks: number;
  final_success_rate: number;
  best_prompt: string | null;
  best_score: number;
  duration_seconds: number;
  total_cost_usd: number;
}

/**
 * Data payload for campaign_failed event
 */
export interface CampaignFailedData {
  error_message: string;
  error_type: string;
  failed_at_iteration: number;
  recoverable: boolean;
  stack_trace: string | null;
}

/**
 * Data payload for iteration_started event
 */
export interface IterationStartedData {
  iteration: number;
  prompt: string;
  techniques_to_apply: string[];
}

/**
 * Data payload for iteration_completed event
 */
export interface IterationCompletedData {
  iteration: number;
  score: number;
  improvement: number;
  evolved_prompt: string;
  success: boolean;
  duration_ms: number;
}

/**
 * Data payload for attack_started event
 */
export interface AttackStartedData {
  attack_id: string;
  prompt: string;
  target_model: string | null;
  technique: string | null;
}

/**
 * Data payload for attack_completed event
 */
export interface AttackCompletedData {
  attack_id: string;
  success: boolean;
  score: number;
  response: string | null;
  duration_ms: number;
  token_usage: TokenUsage | null;
}

/**
 * Data payload for technique_applied event
 */
export interface TechniqueAppliedData {
  technique_name: string;
  technique_category: TechniqueCategory;
  input_prompt: string;
  output_prompt: string;
  success: boolean;
  execution_time_ms: number;
}

/**
 * Data payload for cost_update event
 */
export interface CostUpdateData {
  total_cost_usd: number;
  session_cost_usd: number;
  cost_per_successful_attack: number | null;
  token_usage: TokenUsage | null;
}

/**
 * Data payload for prompt_evolved event
 */
export interface PromptEvolvedData {
  iteration: number;
  previous_prompt: string;
  new_prompt: string;
  previous_score: number;
  new_score: number;
  improvement: number;
  techniques_applied: string[];
}

/**
 * Data payload for error events
 */
export interface ErrorData {
  error_code: string;
  error_message: string;
  severity: "low" | "medium" | "high" | "critical";
  component: string | null;
  recoverable: boolean;
}

/**
 * Data payload for heartbeat events
 */
export interface HeartbeatData {
  server_time: string;
  campaign_status: CampaignStatus | null;
  uptime_seconds: number | null;
}

/**
 * Data payload for connection_ack events
 */
export interface ConnectionAckData {
  client_id: string;
  campaign_id: string;
  server_version: string;
  supported_events: string[];
}

/**
 * Data payload for latency_update events
 */
export interface LatencyUpdateData {
  api_latency_ms: number;
  processing_latency_ms: number;
  latency_metrics: LatencyMetrics;
}

// ==============================================================================
// Typed Telemetry Events (Discriminated Union)
// ==============================================================================

/**
 * Campaign started event with typed data
 */
export type CampaignStartedEvent = AegisTelemetryEvent<CampaignStartedData> & {
  event_type: AegisTelemetryEventType.CAMPAIGN_STARTED;
};

/**
 * Campaign completed event with typed data
 */
export type CampaignCompletedEvent = AegisTelemetryEvent<CampaignCompletedData> & {
  event_type: AegisTelemetryEventType.CAMPAIGN_COMPLETED;
};

/**
 * Campaign failed event with typed data
 */
export type CampaignFailedEvent = AegisTelemetryEvent<CampaignFailedData> & {
  event_type: AegisTelemetryEventType.CAMPAIGN_FAILED;
};

/**
 * Iteration started event with typed data
 */
export type IterationStartedEvent = AegisTelemetryEvent<IterationStartedData> & {
  event_type: AegisTelemetryEventType.ITERATION_STARTED;
};

/**
 * Iteration completed event with typed data
 */
export type IterationCompletedEvent = AegisTelemetryEvent<IterationCompletedData> & {
  event_type: AegisTelemetryEventType.ITERATION_COMPLETED;
};

/**
 * Attack started event with typed data
 */
export type AttackStartedEvent = AegisTelemetryEvent<AttackStartedData> & {
  event_type: AegisTelemetryEventType.ATTACK_STARTED;
};

/**
 * Attack completed event with typed data
 */
export type AttackCompletedEvent = AegisTelemetryEvent<AttackCompletedData> & {
  event_type: AegisTelemetryEventType.ATTACK_COMPLETED;
};

/**
 * Technique applied event with typed data
 */
export type TechniqueAppliedEvent = AegisTelemetryEvent<TechniqueAppliedData> & {
  event_type: AegisTelemetryEventType.TECHNIQUE_APPLIED;
};

/**
 * Cost update event with typed data
 */
export type CostUpdateEvent = AegisTelemetryEvent<CostUpdateData> & {
  event_type: AegisTelemetryEventType.COST_UPDATE;
};

/**
 * Prompt evolved event with typed data
 */
export type PromptEvolvedEvent = AegisTelemetryEvent<PromptEvolvedData> & {
  event_type: AegisTelemetryEventType.PROMPT_EVOLVED;
};

/**
 * Error event with typed data
 */
export type ErrorEvent = AegisTelemetryEvent<ErrorData> & {
  event_type: AegisTelemetryEventType.ERROR;
};

/**
 * Heartbeat event with typed data
 */
export type HeartbeatEvent = AegisTelemetryEvent<HeartbeatData> & {
  event_type: AegisTelemetryEventType.HEARTBEAT;
};

/**
 * Connection acknowledgment event with typed data
 */
export type ConnectionAckEvent = AegisTelemetryEvent<ConnectionAckData> & {
  event_type: AegisTelemetryEventType.CONNECTION_ACK;
};

/**
 * Latency update event with typed data
 */
export type LatencyUpdateEvent = AegisTelemetryEvent<LatencyUpdateData> & {
  event_type: AegisTelemetryEventType.LATENCY_UPDATE;
};

/**
 * Discriminated union of all typed telemetry events
 */
export type TypedAegisTelemetryEvent =
  | CampaignStartedEvent
  | CampaignCompletedEvent
  | CampaignFailedEvent
  | IterationStartedEvent
  | IterationCompletedEvent
  | AttackStartedEvent
  | AttackCompletedEvent
  | TechniqueAppliedEvent
  | CostUpdateEvent
  | PromptEvolvedEvent
  | ErrorEvent
  | HeartbeatEvent
  | ConnectionAckEvent
  | LatencyUpdateEvent;

// ==============================================================================
// WebSocket Message Types
// ==============================================================================

/**
 * Client-to-server ping message
 */
export interface ClientPingMessage {
  type: "ping";
  timestamp: string;
}

/**
 * Client-to-server pong response
 */
export interface ClientPongMessage {
  type: "pong";
  timestamp: string;
}

/**
 * Client request for campaign summary
 */
export interface ClientGetSummaryMessage {
  type: "get_summary";
}

/**
 * Client unsubscribe request
 */
export interface ClientUnsubscribeMessage {
  type: "unsubscribe";
}

/**
 * Union of client-to-server message types
 */
export type ClientMessage =
  | ClientPingMessage
  | ClientPongMessage
  | ClientGetSummaryMessage
  | ClientUnsubscribeMessage;

// ==============================================================================
// Dashboard State Types
// ==============================================================================

/**
 * WebSocket connection status
 */
export type WebSocketConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "error";

/**
 * Time series data point for trend charts
 */
export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
}

/**
 * Success rate time series for trend visualization
 */
export interface SuccessRateTimeSeries {
  timestamp: string;
  success_rate: number;
  total_attempts: number;
  successful_attacks: number;
}

/**
 * Token usage time series for cost tracking
 */
export interface TokenUsageTimeSeries {
  timestamp: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd: number;
}

/**
 * Latency time series for performance monitoring
 */
export interface LatencyTimeSeries {
  timestamp: string;
  api_latency_ms: number;
  processing_latency_ms: number;
  total_latency_ms: number;
}

/**
 * Aggregated dashboard state from telemetry events
 */
export interface AegisDashboardState {
  /** Current campaign summary */
  campaignSummary: CampaignSummary | null;
  /** Connection status to WebSocket */
  connectionStatus: WebSocketConnectionStatus;
  /** Last received event sequence number */
  lastSequence: number;
  /** Rolling window of recent events (max 100) */
  recentEvents: AegisTelemetryEvent[];
  /** Time series data for success rate chart */
  successRateHistory: SuccessRateTimeSeries[];
  /** Time series data for token usage chart */
  tokenUsageHistory: TokenUsageTimeSeries[];
  /** Time series data for latency chart */
  latencyHistory: LatencyTimeSeries[];
  /** Prompt evolution history */
  promptEvolutions: PromptEvolution[];
  /** Last error encountered */
  lastError: ErrorData | null;
  /** Timestamp of last heartbeat */
  lastHeartbeat: string | null;
  /** Number of reconnection attempts */
  reconnectAttempts: number;
}

// ==============================================================================
// Default Values / Factory Functions
// ==============================================================================

/**
 * Create default attack metrics
 */
export const createDefaultAttackMetrics = (): AttackMetrics => ({
  success_rate: 0,
  total_attempts: 0,
  successful_attacks: 0,
  failed_attacks: 0,
  current_streak: 0,
  best_score: 0,
  average_score: 0,
});

/**
 * Create default token usage
 */
export const createDefaultTokenUsage = (): TokenUsage => ({
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  cost_estimate_usd: 0,
  provider: null,
  model: null,
});

/**
 * Create default latency metrics
 */
export const createDefaultLatencyMetrics = (): LatencyMetrics => ({
  api_latency_ms: 0,
  processing_latency_ms: 0,
  total_latency_ms: 0,
  avg_latency_ms: 0,
  p50_latency_ms: 0,
  p95_latency_ms: 0,
  p99_latency_ms: 0,
  min_latency_ms: 0,
  max_latency_ms: 0,
});

/**
 * Create default dashboard state
 */
export const createDefaultDashboardState = (): AegisDashboardState => ({
  campaignSummary: null,
  connectionStatus: "disconnected",
  lastSequence: -1,
  recentEvents: [],
  successRateHistory: [],
  tokenUsageHistory: [],
  latencyHistory: [],
  promptEvolutions: [],
  lastError: null,
  lastHeartbeat: null,
  reconnectAttempts: 0,
});

// ==============================================================================
// Type Guards
// ==============================================================================

/**
 * Type guard to check if event is a campaign started event
 */
export const isCampaignStartedEvent = (
  event: AegisTelemetryEventBase
): event is CampaignStartedEvent => {
  return event.event_type === AegisTelemetryEventType.CAMPAIGN_STARTED;
};

/**
 * Type guard to check if event is a campaign completed event
 */
export const isCampaignCompletedEvent = (
  event: AegisTelemetryEventBase
): event is CampaignCompletedEvent => {
  return event.event_type === AegisTelemetryEventType.CAMPAIGN_COMPLETED;
};

/**
 * Type guard to check if event is a campaign failed event
 */
export const isCampaignFailedEvent = (
  event: AegisTelemetryEventBase
): event is CampaignFailedEvent => {
  return event.event_type === AegisTelemetryEventType.CAMPAIGN_FAILED;
};

/**
 * Type guard to check if event is an iteration started event
 */
export const isIterationStartedEvent = (
  event: AegisTelemetryEventBase
): event is IterationStartedEvent => {
  return event.event_type === AegisTelemetryEventType.ITERATION_STARTED;
};

/**
 * Type guard to check if event is an iteration completed event
 */
export const isIterationCompletedEvent = (
  event: AegisTelemetryEventBase
): event is IterationCompletedEvent => {
  return event.event_type === AegisTelemetryEventType.ITERATION_COMPLETED;
};

/**
 * Type guard to check if event is an attack started event
 */
export const isAttackStartedEvent = (
  event: AegisTelemetryEventBase
): event is AttackStartedEvent => {
  return event.event_type === AegisTelemetryEventType.ATTACK_STARTED;
};

/**
 * Type guard to check if event is an attack completed event
 */
export const isAttackCompletedEvent = (
  event: AegisTelemetryEventBase
): event is AttackCompletedEvent => {
  return event.event_type === AegisTelemetryEventType.ATTACK_COMPLETED;
};

/**
 * Type guard to check if event is a technique applied event
 */
export const isTechniqueAppliedEvent = (
  event: AegisTelemetryEventBase
): event is TechniqueAppliedEvent => {
  return event.event_type === AegisTelemetryEventType.TECHNIQUE_APPLIED;
};

/**
 * Type guard to check if event is a cost update event
 */
export const isCostUpdateEvent = (
  event: AegisTelemetryEventBase
): event is CostUpdateEvent => {
  return event.event_type === AegisTelemetryEventType.COST_UPDATE;
};

/**
 * Type guard to check if event is a prompt evolved event
 */
export const isPromptEvolvedEvent = (
  event: AegisTelemetryEventBase
): event is PromptEvolvedEvent => {
  return event.event_type === AegisTelemetryEventType.PROMPT_EVOLVED;
};

/**
 * Type guard to check if event is an error event
 */
export const isErrorEvent = (
  event: AegisTelemetryEventBase
): event is ErrorEvent => {
  return event.event_type === AegisTelemetryEventType.ERROR;
};

/**
 * Type guard to check if event is a heartbeat event
 */
export const isHeartbeatEvent = (
  event: AegisTelemetryEventBase
): event is HeartbeatEvent => {
  return event.event_type === AegisTelemetryEventType.HEARTBEAT;
};

/**
 * Type guard to check if event is a connection ack event
 */
export const isConnectionAckEvent = (
  event: AegisTelemetryEventBase
): event is ConnectionAckEvent => {
  return event.event_type === AegisTelemetryEventType.CONNECTION_ACK;
};

/**
 * Type guard to check if event is a latency update event
 */
export const isLatencyUpdateEvent = (
  event: AegisTelemetryEventBase
): event is LatencyUpdateEvent => {
  return event.event_type === AegisTelemetryEventType.LATENCY_UPDATE;
};

// ==============================================================================
// Constants
// ==============================================================================

/**
 * Maximum number of events to keep in the rolling window
 */
export const MAX_EVENT_HISTORY = 100;

/**
 * Maximum number of data points for time series charts
 */
export const MAX_TIME_SERIES_POINTS = 100;

/**
 * WebSocket heartbeat interval in milliseconds
 */
export const HEARTBEAT_INTERVAL_MS = 30000;

/**
 * Maximum reconnection attempts before giving up
 */
export const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Base delay for exponential backoff in milliseconds
 */
export const RECONNECT_BASE_DELAY_MS = 1000;

/**
 * Event type labels for display
 */
export const EVENT_TYPE_LABELS: Record<AegisTelemetryEventType, string> = {
  [AegisTelemetryEventType.CAMPAIGN_STARTED]: "Campaign Started",
  [AegisTelemetryEventType.CAMPAIGN_PAUSED]: "Campaign Paused",
  [AegisTelemetryEventType.CAMPAIGN_RESUMED]: "Campaign Resumed",
  [AegisTelemetryEventType.CAMPAIGN_COMPLETED]: "Campaign Completed",
  [AegisTelemetryEventType.CAMPAIGN_FAILED]: "Campaign Failed",
  [AegisTelemetryEventType.ITERATION_STARTED]: "Iteration Started",
  [AegisTelemetryEventType.ITERATION_COMPLETED]: "Iteration Completed",
  [AegisTelemetryEventType.ATTACK_STARTED]: "Attack Started",
  [AegisTelemetryEventType.ATTACK_COMPLETED]: "Attack Completed",
  [AegisTelemetryEventType.ATTACK_RESULT]: "Attack Result",
  [AegisTelemetryEventType.TECHNIQUE_APPLIED]: "Technique Applied",
  [AegisTelemetryEventType.TECHNIQUE_RESULT]: "Technique Result",
  [AegisTelemetryEventType.COST_UPDATE]: "Cost Update",
  [AegisTelemetryEventType.TOKEN_USAGE]: "Token Usage",
  [AegisTelemetryEventType.LATENCY_UPDATE]: "Latency Update",
  [AegisTelemetryEventType.PROMPT_EVOLVED]: "Prompt Evolved",
  [AegisTelemetryEventType.SCORE_UPDATE]: "Score Update",
  [AegisTelemetryEventType.HEARTBEAT]: "Heartbeat",
  [AegisTelemetryEventType.CONNECTION_ACK]: "Connected",
  [AegisTelemetryEventType.ERROR]: "Error",
};

/**
 * Campaign status labels for display
 */
export const CAMPAIGN_STATUS_LABELS: Record<CampaignStatus, string> = {
  [CampaignStatus.PENDING]: "Pending",
  [CampaignStatus.INITIALIZING]: "Initializing",
  [CampaignStatus.RUNNING]: "Running",
  [CampaignStatus.PAUSED]: "Paused",
  [CampaignStatus.COMPLETED]: "Completed",
  [CampaignStatus.FAILED]: "Failed",
  [CampaignStatus.STOPPED]: "Stopped",
};

/**
 * Technique category labels for display
 */
export const TECHNIQUE_CATEGORY_LABELS: Record<TechniqueCategory, string> = {
  [TechniqueCategory.AUTODAN]: "AutoDAN",
  [TechniqueCategory.GPTFUZZ]: "GPTFuzz",
  [TechniqueCategory.CHIMERA_FRAMING]: "Chimera Framing",
  [TechniqueCategory.OBFUSCATION]: "Obfuscation",
  [TechniqueCategory.PERSONA]: "Persona",
  [TechniqueCategory.COGNITIVE]: "Cognitive",
  [TechniqueCategory.OTHER]: "Other",
};

/**
 * Technique category colors for charts
 */
export const TECHNIQUE_CATEGORY_COLORS: Record<TechniqueCategory, string> = {
  [TechniqueCategory.AUTODAN]: "#8b5cf6", // Purple
  [TechniqueCategory.GPTFUZZ]: "#f59e0b", // Amber
  [TechniqueCategory.CHIMERA_FRAMING]: "#10b981", // Emerald
  [TechniqueCategory.OBFUSCATION]: "#6366f1", // Indigo
  [TechniqueCategory.PERSONA]: "#ec4899", // Pink
  [TechniqueCategory.COGNITIVE]: "#14b8a6", // Teal
  [TechniqueCategory.OTHER]: "#6b7280", // Gray
};
