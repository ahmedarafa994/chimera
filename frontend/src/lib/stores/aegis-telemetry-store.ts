/**
 * Aegis Telemetry Store
 *
 * Zustand store for managing aggregated telemetry state across components
 * in the Real-Time Aegis Campaign Dashboard.
 *
 * Features:
 * - Campaign state management (status, objective, progress)
 * - Metrics history tracking with rolling windows
 * - Technique breakdown aggregation
 * - Time-series data for trend charts
 * - Selectors for chart data transformations
 */

import { create } from "zustand";
import { devtools, subscribeWithSelector } from "zustand/middleware";
import {
  AegisTelemetryEvent,
  AegisTelemetryEventType,
  AegisTelemetryEventBase,
  CampaignStatus,
  CampaignSummary,
  AttackMetrics,
  TechniquePerformance,
  TokenUsage,
  LatencyMetrics,
  PromptEvolution,
  SuccessRateTimeSeries,
  TokenUsageTimeSeries,
  LatencyTimeSeries,
  WebSocketConnectionStatus,
  ErrorData,
  createDefaultAttackMetrics,
  createDefaultTokenUsage,
  createDefaultLatencyMetrics,
  MAX_EVENT_HISTORY,
  MAX_TIME_SERIES_POINTS,
  isCampaignStartedEvent,
  isCampaignCompletedEvent,
  isCampaignFailedEvent,
  isIterationCompletedEvent,
  isAttackCompletedEvent,
  isTechniqueAppliedEvent,
  isCostUpdateEvent,
  isLatencyUpdateEvent,
  isPromptEvolvedEvent,
  isErrorEvent,
  isHeartbeatEvent,
  CampaignStartedData,
  CampaignCompletedData,
  CampaignFailedData,
  IterationCompletedData,
  AttackCompletedData,
  TechniqueAppliedData,
  CostUpdateData,
  PromptEvolvedData,
  ErrorData as TelemetryErrorData,
  LatencyUpdateData,
} from "@/types/aegis-telemetry";

// ============================================================================
// Types
// ============================================================================

/**
 * State shape for Aegis telemetry store
 */
interface AegisTelemetryState {
  // Connection state
  connectionStatus: WebSocketConnectionStatus;
  reconnectAttempts: number;
  lastHeartbeat: string | null;

  // Campaign state
  campaignId: string | null;
  campaignSummary: CampaignSummary | null;

  // Metrics
  attackMetrics: AttackMetrics;
  techniqueBreakdown: TechniquePerformance[];
  tokenUsage: TokenUsage;
  latencyMetrics: LatencyMetrics;

  // Time series data for charts
  successRateHistory: SuccessRateTimeSeries[];
  tokenUsageHistory: TokenUsageTimeSeries[];
  latencyHistory: LatencyTimeSeries[];

  // Prompt evolution tracking
  promptEvolutions: PromptEvolution[];

  // Event history
  recentEvents: AegisTelemetryEvent[];
  lastSequence: number;

  // Error state
  lastError: ErrorData | null;

  // Actions
  setCampaignId: (campaignId: string | null) => void;
  setConnectionStatus: (status: WebSocketConnectionStatus) => void;
  setReconnectAttempts: (attempts: number) => void;
  updateMetrics: (metrics: Partial<AttackMetrics>) => void;
  updateTokenUsage: (usage: Partial<TokenUsage>) => void;
  updateLatencyMetrics: (latency: Partial<LatencyMetrics>) => void;
  addTelemetryEvent: (event: AegisTelemetryEventBase) => void;
  addTechniquePerformance: (technique: TechniquePerformance) => void;
  addPromptEvolution: (evolution: PromptEvolution) => void;
  setError: (error: ErrorData | null) => void;
  resetCampaign: () => void;
  resetAll: () => void;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Add item to array with max length constraint (rolling window)
 */
function addToRollingWindow<T>(
  array: T[],
  item: T,
  maxLength: number = MAX_EVENT_HISTORY
): T[] {
  const newArray = [...array, item];
  if (newArray.length > maxLength) {
    return newArray.slice(newArray.length - maxLength);
  }
  return newArray;
}

/**
 * Add time series data point with rolling window
 */
function addToTimeSeries<T>(
  array: T[],
  item: T,
  maxLength: number = MAX_TIME_SERIES_POINTS
): T[] {
  return addToRollingWindow(array, item, maxLength);
}

/**
 * Calculate running average
 */
function calculateRunningAverage(
  currentAvg: number,
  currentCount: number,
  newValue: number
): number {
  if (currentCount === 0) return newValue;
  return (currentAvg * currentCount + newValue) / (currentCount + 1);
}

/**
 * Create default campaign summary from started event
 */
function createCampaignSummary(
  campaignId: string,
  data: CampaignStartedData,
  timestamp: string
): CampaignSummary {
  return {
    campaign_id: campaignId,
    status: CampaignStatus.RUNNING,
    objective: data.objective,
    started_at: timestamp,
    completed_at: null,
    duration_seconds: 0,
    current_iteration: 0,
    max_iterations: data.max_iterations,
    attack_metrics: createDefaultAttackMetrics(),
    technique_breakdown: [],
    token_usage: createDefaultTokenUsage(),
    latency_metrics: createDefaultLatencyMetrics(),
    best_prompt: null,
    best_score: 0,
    target_model: data.target_model,
  };
}

// ============================================================================
// Initial State
// ============================================================================

const initialState: Omit<
  AegisTelemetryState,
  | "setCampaignId"
  | "setConnectionStatus"
  | "setReconnectAttempts"
  | "updateMetrics"
  | "updateTokenUsage"
  | "updateLatencyMetrics"
  | "addTelemetryEvent"
  | "addTechniquePerformance"
  | "addPromptEvolution"
  | "setError"
  | "resetCampaign"
  | "resetAll"
> = {
  connectionStatus: "disconnected",
  reconnectAttempts: 0,
  lastHeartbeat: null,
  campaignId: null,
  campaignSummary: null,
  attackMetrics: createDefaultAttackMetrics(),
  techniqueBreakdown: [],
  tokenUsage: createDefaultTokenUsage(),
  latencyMetrics: createDefaultLatencyMetrics(),
  successRateHistory: [],
  tokenUsageHistory: [],
  latencyHistory: [],
  promptEvolutions: [],
  recentEvents: [],
  lastSequence: -1,
  lastError: null,
};

// ============================================================================
// Store
// ============================================================================

export const useAegisTelemetryStore = create<AegisTelemetryState>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      // Initial state
      ...initialState,

      // ========================================================================
      // Basic setters
      // ========================================================================

      setCampaignId: (campaignId: string | null) => {
        set({ campaignId });
      },

      setConnectionStatus: (status: WebSocketConnectionStatus) => {
        set({ connectionStatus: status });
      },

      setReconnectAttempts: (attempts: number) => {
        set({ reconnectAttempts: attempts });
      },

      setError: (error: ErrorData | null) => {
        set({ lastError: error });
      },

      // ========================================================================
      // Metrics updates
      // ========================================================================

      updateMetrics: (metrics: Partial<AttackMetrics>) => {
        set((state) => ({
          attackMetrics: {
            ...state.attackMetrics,
            ...metrics,
          },
          campaignSummary: state.campaignSummary
            ? {
                ...state.campaignSummary,
                attack_metrics: {
                  ...state.campaignSummary.attack_metrics,
                  ...metrics,
                },
              }
            : null,
        }));
      },

      updateTokenUsage: (usage: Partial<TokenUsage>) => {
        set((state) => ({
          tokenUsage: {
            ...state.tokenUsage,
            ...usage,
          },
          campaignSummary: state.campaignSummary
            ? {
                ...state.campaignSummary,
                token_usage: {
                  ...state.campaignSummary.token_usage,
                  ...usage,
                },
              }
            : null,
        }));
      },

      updateLatencyMetrics: (latency: Partial<LatencyMetrics>) => {
        set((state) => ({
          latencyMetrics: {
            ...state.latencyMetrics,
            ...latency,
          },
          campaignSummary: state.campaignSummary
            ? {
                ...state.campaignSummary,
                latency_metrics: {
                  ...state.campaignSummary.latency_metrics,
                  ...latency,
                },
              }
            : null,
        }));
      },

      // ========================================================================
      // Event processing
      // ========================================================================

      addTelemetryEvent: (event: AegisTelemetryEventBase) => {
        const timestamp = event.timestamp || new Date().toISOString();

        set((state) => {
          let newState = { ...state };

          // Add event to history
          newState.recentEvents = addToRollingWindow(
            state.recentEvents,
            event as AegisTelemetryEvent,
            MAX_EVENT_HISTORY
          );
          newState.lastSequence = event.sequence ?? state.lastSequence + 1;

          // Process event by type
          if (isCampaignStartedEvent(event)) {
            const data = event.data as CampaignStartedData;
            newState.campaignId = event.campaign_id;
            newState.campaignSummary = createCampaignSummary(
              event.campaign_id,
              data,
              timestamp
            );
            newState.attackMetrics = createDefaultAttackMetrics();
            newState.tokenUsage = createDefaultTokenUsage();
            newState.latencyMetrics = createDefaultLatencyMetrics();
            newState.techniqueBreakdown = [];
            newState.promptEvolutions = [];
            newState.successRateHistory = [];
            newState.tokenUsageHistory = [];
            newState.latencyHistory = [];
          } else if (isCampaignCompletedEvent(event)) {
            const data = event.data as CampaignCompletedData;
            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                status: CampaignStatus.COMPLETED,
                completed_at: timestamp,
                duration_seconds: data.duration_seconds,
                best_prompt: data.best_prompt,
                best_score: data.best_score,
                attack_metrics: {
                  ...newState.campaignSummary.attack_metrics,
                  success_rate: data.final_success_rate,
                  total_attempts: data.total_attacks,
                  successful_attacks: data.successful_attacks,
                },
              };
              newState.attackMetrics = {
                ...newState.attackMetrics,
                success_rate: data.final_success_rate,
                total_attempts: data.total_attacks,
                successful_attacks: data.successful_attacks,
              };
            }
          } else if (isCampaignFailedEvent(event)) {
            const data = event.data as CampaignFailedData;
            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                status: CampaignStatus.FAILED,
                completed_at: timestamp,
              };
            }
            newState.lastError = {
              error_code: "CAMPAIGN_FAILED",
              error_message: data.error_message,
              severity: "high",
              component: "campaign",
              recoverable: data.recoverable,
            };
          } else if (isIterationCompletedEvent(event)) {
            const data = event.data as IterationCompletedData;
            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                current_iteration: data.iteration,
                best_score: Math.max(
                  newState.campaignSummary.best_score,
                  data.score
                ),
                best_prompt:
                  data.score > newState.campaignSummary.best_score
                    ? data.evolved_prompt
                    : newState.campaignSummary.best_prompt,
              };
            }
          } else if (isAttackCompletedEvent(event)) {
            const data = event.data as AttackCompletedData;
            const currentMetrics = state.attackMetrics;

            const newTotalAttempts = currentMetrics.total_attempts + 1;
            const newSuccessful = data.success
              ? currentMetrics.successful_attacks + 1
              : currentMetrics.successful_attacks;
            const newFailed = data.success
              ? currentMetrics.failed_attacks
              : currentMetrics.failed_attacks + 1;
            const newSuccessRate =
              newTotalAttempts > 0
                ? (newSuccessful / newTotalAttempts) * 100
                : 0;

            const newAverageScore = calculateRunningAverage(
              currentMetrics.average_score,
              currentMetrics.total_attempts,
              data.score
            );

            const updatedMetrics: AttackMetrics = {
              success_rate: newSuccessRate,
              total_attempts: newTotalAttempts,
              successful_attacks: newSuccessful,
              failed_attacks: newFailed,
              current_streak: data.success
                ? currentMetrics.current_streak > 0
                  ? currentMetrics.current_streak + 1
                  : 1
                : currentMetrics.current_streak < 0
                  ? currentMetrics.current_streak - 1
                  : -1,
              best_score: Math.max(currentMetrics.best_score, data.score),
              average_score: newAverageScore,
            };

            newState.attackMetrics = updatedMetrics;

            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                attack_metrics: updatedMetrics,
              };
            }

            // Add to success rate time series
            newState.successRateHistory = addToTimeSeries(
              state.successRateHistory,
              {
                timestamp,
                success_rate: newSuccessRate,
                total_attempts: newTotalAttempts,
                successful_attacks: newSuccessful,
              }
            );

            // Update token usage if provided
            if (data.token_usage) {
              const currentUsage = state.tokenUsage;
              const updatedUsage: TokenUsage = {
                prompt_tokens:
                  currentUsage.prompt_tokens + data.token_usage.prompt_tokens,
                completion_tokens:
                  currentUsage.completion_tokens +
                  data.token_usage.completion_tokens,
                total_tokens:
                  currentUsage.total_tokens + data.token_usage.total_tokens,
                cost_estimate_usd:
                  currentUsage.cost_estimate_usd +
                  data.token_usage.cost_estimate_usd,
                provider: data.token_usage.provider ?? currentUsage.provider,
                model: data.token_usage.model ?? currentUsage.model,
              };

              newState.tokenUsage = updatedUsage;

              if (newState.campaignSummary) {
                newState.campaignSummary = {
                  ...newState.campaignSummary,
                  token_usage: updatedUsage,
                };
              }

              // Add to token usage time series
              newState.tokenUsageHistory = addToTimeSeries(
                state.tokenUsageHistory,
                {
                  timestamp,
                  prompt_tokens: updatedUsage.prompt_tokens,
                  completion_tokens: updatedUsage.completion_tokens,
                  total_tokens: updatedUsage.total_tokens,
                  cost_usd: updatedUsage.cost_estimate_usd,
                }
              );
            }
          } else if (isTechniqueAppliedEvent(event)) {
            const data = event.data as TechniqueAppliedData;
            const currentBreakdown = state.techniqueBreakdown;

            const existingIndex = currentBreakdown.findIndex(
              (t) => t.technique_name === data.technique_name
            );

            let updatedBreakdown: TechniquePerformance[];

            if (existingIndex >= 0) {
              const existing = currentBreakdown[existingIndex];
              const newTotal = existing.total_applications + 1;
              const newSuccessCount = data.success
                ? existing.success_count + 1
                : existing.success_count;
              const newFailureCount = data.success
                ? existing.failure_count
                : existing.failure_count + 1;

              updatedBreakdown = [
                ...currentBreakdown.slice(0, existingIndex),
                {
                  ...existing,
                  success_count: newSuccessCount,
                  failure_count: newFailureCount,
                  total_applications: newTotal,
                  success_rate: (newSuccessCount / newTotal) * 100,
                  avg_execution_time_ms:
                    (existing.avg_execution_time_ms *
                      existing.total_applications +
                      data.execution_time_ms) /
                    newTotal,
                },
                ...currentBreakdown.slice(existingIndex + 1),
              ];
            } else {
              updatedBreakdown = [
                ...currentBreakdown,
                {
                  technique_name: data.technique_name,
                  technique_category: data.technique_category,
                  success_count: data.success ? 1 : 0,
                  failure_count: data.success ? 0 : 1,
                  total_applications: 1,
                  success_rate: data.success ? 100 : 0,
                  avg_score: 0,
                  best_score: 0,
                  avg_execution_time_ms: data.execution_time_ms,
                },
              ];
            }

            newState.techniqueBreakdown = updatedBreakdown;

            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                technique_breakdown: updatedBreakdown,
              };
            }
          } else if (isCostUpdateEvent(event)) {
            const data = event.data as CostUpdateData;
            if (data.token_usage) {
              const updatedUsage: TokenUsage = {
                ...state.tokenUsage,
                ...data.token_usage,
                cost_estimate_usd: data.total_cost_usd,
              };

              newState.tokenUsage = updatedUsage;

              if (newState.campaignSummary) {
                newState.campaignSummary = {
                  ...newState.campaignSummary,
                  token_usage: updatedUsage,
                };
              }
            }
          } else if (isLatencyUpdateEvent(event)) {
            const data = event.data as LatencyUpdateData;
            newState.latencyMetrics = data.latency_metrics;

            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                latency_metrics: data.latency_metrics,
              };
            }

            newState.latencyHistory = addToTimeSeries(state.latencyHistory, {
              timestamp,
              api_latency_ms: data.api_latency_ms,
              processing_latency_ms: data.processing_latency_ms,
              total_latency_ms: data.api_latency_ms + data.processing_latency_ms,
            });
          } else if (isPromptEvolvedEvent(event)) {
            const data = event.data as PromptEvolvedData;
            const evolution: PromptEvolution = {
              iteration: data.iteration,
              original_prompt: data.previous_prompt,
              evolved_prompt: data.new_prompt,
              score: data.new_score,
              improvement: data.improvement,
              techniques_applied: data.techniques_applied,
              is_successful: data.new_score > data.previous_score,
              timestamp,
            };

            newState.promptEvolutions = [
              ...state.promptEvolutions,
              evolution,
            ];
          } else if (isErrorEvent(event)) {
            const data = event.data as TelemetryErrorData;
            newState.lastError = data;
          } else if (isHeartbeatEvent(event)) {
            newState.lastHeartbeat = timestamp;
          }

          return newState;
        });
      },

      // ========================================================================
      // Technique performance
      // ========================================================================

      addTechniquePerformance: (technique: TechniquePerformance) => {
        set((state) => {
          const existingIndex = state.techniqueBreakdown.findIndex(
            (t) => t.technique_name === technique.technique_name
          );

          if (existingIndex >= 0) {
            // Update existing
            const updated = [...state.techniqueBreakdown];
            updated[existingIndex] = technique;
            return {
              techniqueBreakdown: updated,
              campaignSummary: state.campaignSummary
                ? { ...state.campaignSummary, technique_breakdown: updated }
                : null,
            };
          } else {
            // Add new
            const updated = [...state.techniqueBreakdown, technique];
            return {
              techniqueBreakdown: updated,
              campaignSummary: state.campaignSummary
                ? { ...state.campaignSummary, technique_breakdown: updated }
                : null,
            };
          }
        });
      },

      // ========================================================================
      // Prompt evolution
      // ========================================================================

      addPromptEvolution: (evolution: PromptEvolution) => {
        set((state) => ({
          promptEvolutions: [...state.promptEvolutions, evolution],
        }));
      },

      // ========================================================================
      // Reset actions
      // ========================================================================

      resetCampaign: () => {
        set({
          campaignId: null,
          campaignSummary: null,
          attackMetrics: createDefaultAttackMetrics(),
          techniqueBreakdown: [],
          tokenUsage: createDefaultTokenUsage(),
          latencyMetrics: createDefaultLatencyMetrics(),
          successRateHistory: [],
          tokenUsageHistory: [],
          latencyHistory: [],
          promptEvolutions: [],
          recentEvents: [],
          lastSequence: -1,
          lastError: null,
        });
      },

      resetAll: () => {
        set({
          ...initialState,
        });
      },
    })),
    { name: "AegisTelemetryStore" }
  )
);

// ============================================================================
// Selectors
// ============================================================================

/**
 * Select campaign status
 */
export const selectCampaignStatus = (
  state: AegisTelemetryState
): CampaignStatus | null => state.campaignSummary?.status ?? null;

/**
 * Select current iteration progress
 */
export const selectIterationProgress = (state: AegisTelemetryState) => ({
  current: state.campaignSummary?.current_iteration ?? 0,
  max: state.campaignSummary?.max_iterations ?? 0,
  percentage: state.campaignSummary
    ? (state.campaignSummary.current_iteration /
        state.campaignSummary.max_iterations) *
      100
    : 0,
});

/**
 * Select success rate with trend
 */
export const selectSuccessRateWithTrend = (state: AegisTelemetryState) => {
  const history = state.successRateHistory;
  const current = state.attackMetrics.success_rate;

  if (history.length < 2) {
    return { current, trend: "stable" as const, change: 0 };
  }

  // Calculate trend from last 10 data points
  const recentHistory = history.slice(-10);
  const firstRate = recentHistory[0].success_rate;
  const change = current - firstRate;

  let trend: "up" | "down" | "stable";
  if (change > 5) {
    trend = "up";
  } else if (change < -5) {
    trend = "down";
  } else {
    trend = "stable";
  }

  return { current, trend, change };
};

/**
 * Select cost per successful attack
 */
export const selectCostPerSuccessfulAttack = (
  state: AegisTelemetryState
): number | null => {
  const { successful_attacks } = state.attackMetrics;
  const { cost_estimate_usd } = state.tokenUsage;

  if (successful_attacks === 0) return null;
  return cost_estimate_usd / successful_attacks;
};

/**
 * Select top performing techniques (sorted by success rate)
 */
export const selectTopTechniques = (
  state: AegisTelemetryState,
  limit: number = 5
): TechniquePerformance[] => {
  return [...state.techniqueBreakdown]
    .sort((a, b) => b.success_rate - a.success_rate)
    .slice(0, limit);
};

/**
 * Select technique breakdown by category
 */
export const selectTechniquesByCategory = (
  state: AegisTelemetryState
): Record<string, TechniquePerformance[]> => {
  return state.techniqueBreakdown.reduce(
    (acc, technique) => {
      const category = technique.technique_category;
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(technique);
      return acc;
    },
    {} as Record<string, TechniquePerformance[]>
  );
};

/**
 * Select chart data for success rate area chart
 */
export const selectSuccessRateChartData = (state: AegisTelemetryState) => {
  return state.successRateHistory.map((point) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    timestamp: point.timestamp,
    successRate: point.success_rate,
    attempts: point.total_attempts,
  }));
};

/**
 * Select chart data for token usage
 */
export const selectTokenUsageChartData = (state: AegisTelemetryState) => {
  return state.tokenUsageHistory.map((point) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    timestamp: point.timestamp,
    promptTokens: point.prompt_tokens,
    completionTokens: point.completion_tokens,
    totalTokens: point.total_tokens,
    cost: point.cost_usd,
  }));
};

/**
 * Select chart data for latency
 */
export const selectLatencyChartData = (state: AegisTelemetryState) => {
  return state.latencyHistory.map((point) => ({
    time: new Date(point.timestamp).toLocaleTimeString(),
    timestamp: point.timestamp,
    apiLatency: point.api_latency_ms,
    processingLatency: point.processing_latency_ms,
    totalLatency: point.total_latency_ms,
  }));
};

/**
 * Select recent events filtered by type
 */
export const selectEventsByType = (
  state: AegisTelemetryState,
  eventTypes: AegisTelemetryEventType[]
): AegisTelemetryEvent[] => {
  return state.recentEvents.filter((event) =>
    eventTypes.includes(event.event_type)
  );
};

/**
 * Select connection health
 */
export const selectConnectionHealth = (state: AegisTelemetryState) => ({
  status: state.connectionStatus,
  isConnected: state.connectionStatus === "connected",
  reconnectAttempts: state.reconnectAttempts,
  lastHeartbeat: state.lastHeartbeat,
  hasError: state.lastError !== null,
});

/**
 * Select campaign progress summary
 */
export const selectCampaignProgress = (state: AegisTelemetryState) => {
  const summary = state.campaignSummary;
  if (!summary) return null;

  return {
    campaignId: summary.campaign_id,
    status: summary.status,
    objective: summary.objective,
    targetModel: summary.target_model,
    progress: {
      currentIteration: summary.current_iteration,
      maxIterations: summary.max_iterations,
      percentage:
        summary.max_iterations > 0
          ? (summary.current_iteration / summary.max_iterations) * 100
          : 0,
    },
    metrics: {
      successRate: summary.attack_metrics.success_rate,
      totalAttempts: summary.attack_metrics.total_attempts,
      bestScore: summary.best_score,
    },
    cost: {
      totalTokens: summary.token_usage.total_tokens,
      estimatedCost: summary.token_usage.cost_estimate_usd,
    },
  };
};
