/**
 * useAegisTelemetry Hook
 *
 * React hook that manages WebSocket connection to Aegis telemetry endpoint
 * with automatic reconnection, state management, and event aggregation.
 *
 * Features:
 * - Automatic reconnection with exponential backoff (max 5 attempts)
 * - Heartbeat response handling
 * - State aggregation: cumulative success rate, technique breakdown, cost totals
 * - Updates within 500ms of event receipt (per spec requirement)
 * - Memory-efficient event history (rolling window of 100 events)
 */

"use client";

import { useState, useEffect, useRef, useCallback, useMemo, useTransition } from "react";
import {
  AegisTelemetryEventType,
  AegisTelemetryEvent,
  AegisTelemetryEventBase,
  WebSocketConnectionStatus,
  AegisDashboardState,
  CampaignSummary,
  AttackMetrics,
  TokenUsage,
  LatencyMetrics,
  TechniquePerformance,
  PromptEvolution,
  SuccessRateTimeSeries,
  TokenUsageTimeSeries,
  LatencyTimeSeries,
  CampaignStatus,
  createDefaultDashboardState,
  createDefaultAttackMetrics,
  createDefaultTokenUsage,
  createDefaultLatencyMetrics,
  MAX_EVENT_HISTORY,
  MAX_TIME_SERIES_POINTS,
  MAX_RECONNECT_ATTEMPTS,
  RECONNECT_BASE_DELAY_MS,
  HEARTBEAT_INTERVAL_MS,
  isCampaignStartedEvent,
  isCampaignCompletedEvent,
  isCampaignFailedEvent,
  isIterationCompletedEvent,
  isAttackCompletedEvent,
  isTechniqueAppliedEvent,
  isCostUpdateEvent,
  isPromptEvolvedEvent,
  isErrorEvent,
  isHeartbeatEvent,
  isConnectionAckEvent,
  isLatencyUpdateEvent,
  CampaignStartedData,
  CampaignCompletedData,
  CampaignFailedData,
  IterationCompletedData,
  AttackCompletedData,
  TechniqueAppliedData,
  CostUpdateData,
  PromptEvolvedData,
  ErrorData,
  LatencyUpdateData,
} from "@/types/aegis-telemetry";
import { getApiConfig } from "@/lib/api-config";
import { aegisPerformanceMonitor, MAX_EVENT_PROCESSING_MS } from "./useAegisPerformanceMonitor";

// ============================================================================
// Types
// ============================================================================

/**
 * Options for configuring the useAegisTelemetry hook
 */
export interface UseAegisTelemetryOptions {
  /** Enable automatic connection on mount (default: true) */
  autoConnect?: boolean;
  /** Custom WebSocket URL (overrides default) */
  wsUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Client ID for reconnection support */
  clientId?: string;
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Return type for the useAegisTelemetry hook
 */
export interface UseAegisTelemetryReturn {
  /** Current WebSocket connection status */
  connectionStatus: WebSocketConnectionStatus;
  /** Complete dashboard state */
  dashboardState: AegisDashboardState;
  /** Current attack metrics */
  metrics: AttackMetrics;
  /** Campaign summary (if available) */
  campaignSummary: CampaignSummary | null;
  /** Technique performance breakdown */
  techniqueBreakdown: TechniquePerformance[];
  /** Token usage and cost */
  tokenUsage: TokenUsage;
  /** Latency metrics */
  latencyMetrics: LatencyMetrics;
  /** Recent events (rolling window) */
  recentEvents: AegisTelemetryEvent[];
  /** Success rate time series for charts */
  successRateHistory: SuccessRateTimeSeries[];
  /** Token usage time series for charts */
  tokenUsageHistory: TokenUsageTimeSeries[];
  /** Latency time series for charts */
  latencyHistory: LatencyTimeSeries[];
  /** Prompt evolution history */
  promptEvolutions: PromptEvolution[];
  /** Last error encountered */
  error: ErrorData | null;
  /** Number of reconnection attempts */
  reconnectAttempts: number;
  /** Whether connected */
  isConnected: boolean;
  /** Manual reconnect function */
  reconnect: () => void;
  /** Disconnect function */
  disconnect: () => void;
  /** Request campaign summary */
  requestSummary: () => void;
  /** Send ping to keep connection alive */
  sendPing: () => void;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate exponential backoff delay with jitter
 */
function calculateBackoffDelay(attempt: number): number {
  const baseDelay = RECONNECT_BASE_DELAY_MS;
  const maxDelay = 30000; // 30 seconds max
  const exponentialDelay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);
  // Add jitter (±20%)
  const jitter = exponentialDelay * 0.2 * (Math.random() * 2 - 1);
  return Math.round(exponentialDelay + jitter);
}

/**
 * Build WebSocket URL for Aegis telemetry
 */
function buildWebSocketUrl(
  campaignId: string,
  options: UseAegisTelemetryOptions
): string {
  if (options.wsUrl) {
    return options.wsUrl;
  }

  const config = getApiConfig();
  const baseUrl = config.backendApiUrl || "http://localhost:8001/api/v1";

  // Convert HTTP(S) to WS(S)
  const wsBaseUrl = baseUrl
    .replace("http://", "ws://")
    .replace("https://", "wss://");

  // Build query parameters
  const params = new URLSearchParams();
  if (options.clientId) {
    params.set("client_id", options.clientId);
  }
  if (options.apiKey) {
    params.set("api_key", options.apiKey);
  }

  const queryString = params.toString();
  const url = `${wsBaseUrl}/ws/aegis/telemetry/${campaignId}`;
  return queryString ? `${url}?${queryString}` : url;
}

/**
 * Add data point to time series with rolling window
 */
function addToTimeSeries<T>(array: T[], item: T, maxLength: number): T[] {
  const newArray = [...array, item];
  if (newArray.length > maxLength) {
    return newArray.slice(newArray.length - maxLength);
  }
  return newArray;
}

/**
 * Add event to history with rolling window
 */
function addToEventHistory(
  events: AegisTelemetryEvent[],
  event: AegisTelemetryEvent
): AegisTelemetryEvent[] {
  const newEvents = [...events, event];
  if (newEvents.length > MAX_EVENT_HISTORY) {
    return newEvents.slice(newEvents.length - MAX_EVENT_HISTORY);
  }
  return newEvents;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Custom React hook for managing Aegis telemetry WebSocket connection
 *
 * @param campaignId - The ID of the campaign to subscribe to
 * @param options - Configuration options
 * @returns Telemetry state and control functions
 *
 * @example
 * ```tsx
 * const {
 *   connectionStatus,
 *   metrics,
 *   techniqueBreakdown,
 *   reconnect,
 * } = useAegisTelemetry("campaign-123");
 *
 * if (connectionStatus === "connected") {
 *   console.log(`Success rate: ${metrics.success_rate}%`);
 * }
 * ```
 */
export function useAegisTelemetry(
  campaignId: string,
  options: UseAegisTelemetryOptions = {}
): UseAegisTelemetryReturn {
  const { autoConnect = true, debug = false } = options;

  // ============================================================================
  // State
  // ============================================================================

  const [dashboardState, setDashboardState] = useState<AegisDashboardState>(
    createDefaultDashboardState
  );

  // ============================================================================
  // Refs
  // ============================================================================

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isUnmountedRef = useRef(false);
  const lastEventTimeRef = useRef<number>(0);

  // ============================================================================
  // Debug Logger
  // ============================================================================

  const log = useCallback(
    (...args: unknown[]) => {
      if (debug) {
        // eslint-disable-next-line no-console
        console.log("[useAegisTelemetry]", ...args);
      }
    },
    [debug]
  );

  // ============================================================================
  // Event Handlers
  // ============================================================================

  /**
   * Process incoming telemetry event and update state
   * Performance tracked to ensure <500ms processing requirement
   */
  const processEvent = useCallback(
    (event: AegisTelemetryEventBase) => {
      // Start performance measurement
      const endMeasure = aegisPerformanceMonitor.startMeasure(event.event_type);

      const now = Date.now();
      const eventLatency = now - lastEventTimeRef.current;
      lastEventTimeRef.current = now;

      log("Processing event:", event.event_type, "Latency:", eventLatency, "ms");

      setDashboardState((prev) => {
        const timestamp = event.timestamp || new Date().toISOString();
        let newState = { ...prev };

        // Add to event history (rolling window)
        newState.recentEvents = addToEventHistory(
          prev.recentEvents,
          event as AegisTelemetryEvent
        );
        newState.lastSequence = event.sequence ?? prev.lastSequence + 1;

        // Process event by type
        if (isCampaignStartedEvent(event)) {
          const data = event.data as CampaignStartedData;
          newState.campaignSummary = {
            campaign_id: event.campaign_id,
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
          const currentMetrics = newState.campaignSummary?.attack_metrics ?? createDefaultAttackMetrics();

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

          // Calculate running average score
          const totalScore =
            currentMetrics.average_score * currentMetrics.total_attempts +
            data.score;
          const newAverageScore = totalScore / newTotalAttempts;

          const updatedMetrics: AttackMetrics = {
            success_rate: newSuccessRate,
            total_attempts: newTotalAttempts,
            successful_attacks: newSuccessful,
            failed_attacks: newFailed,
            current_streak: data.success
              ? (currentMetrics.current_streak > 0
                  ? currentMetrics.current_streak + 1
                  : 1)
              : (currentMetrics.current_streak < 0
                  ? currentMetrics.current_streak - 1
                  : -1),
            best_score: Math.max(currentMetrics.best_score, data.score),
            average_score: newAverageScore,
          };

          if (newState.campaignSummary) {
            newState.campaignSummary = {
              ...newState.campaignSummary,
              attack_metrics: updatedMetrics,
            };
          }

          // Add to success rate time series
          newState.successRateHistory = addToTimeSeries(
            prev.successRateHistory,
            {
              timestamp,
              success_rate: newSuccessRate,
              total_attempts: newTotalAttempts,
              successful_attacks: newSuccessful,
            },
            MAX_TIME_SERIES_POINTS
          );

          // Update token usage if provided
          if (data.token_usage) {
            const currentUsage = newState.campaignSummary?.token_usage ?? createDefaultTokenUsage();
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

            if (newState.campaignSummary) {
              newState.campaignSummary = {
                ...newState.campaignSummary,
                token_usage: updatedUsage,
              };
            }

            // Add to token usage time series
            newState.tokenUsageHistory = addToTimeSeries(
              prev.tokenUsageHistory,
              {
                timestamp,
                prompt_tokens: updatedUsage.prompt_tokens,
                completion_tokens: updatedUsage.completion_tokens,
                total_tokens: updatedUsage.total_tokens,
                cost_usd: updatedUsage.cost_estimate_usd,
              },
              MAX_TIME_SERIES_POINTS
            );
          }
        } else if (isTechniqueAppliedEvent(event)) {
          const data = event.data as TechniqueAppliedData;
          const currentBreakdown = newState.campaignSummary?.technique_breakdown ?? [];

          // Find or create technique entry
          const existingIndex = currentBreakdown.findIndex(
            (t) => t.technique_name === data.technique_name
          );

          let updatedBreakdown: TechniquePerformance[];

          if (existingIndex >= 0) {
            // Update existing technique
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
            // Create new technique entry
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

          if (newState.campaignSummary) {
            newState.campaignSummary = {
              ...newState.campaignSummary,
              technique_breakdown: updatedBreakdown,
            };
          }
        } else if (isCostUpdateEvent(event)) {
          const data = event.data as CostUpdateData;
          if (newState.campaignSummary && data.token_usage) {
            newState.campaignSummary = {
              ...newState.campaignSummary,
              token_usage: {
                ...newState.campaignSummary.token_usage,
                ...data.token_usage,
                cost_estimate_usd: data.total_cost_usd,
              },
            };
          }
        } else if (isLatencyUpdateEvent(event)) {
          const data = event.data as LatencyUpdateData;
          if (newState.campaignSummary) {
            newState.campaignSummary = {
              ...newState.campaignSummary,
              latency_metrics: data.latency_metrics,
            };
          }

          // Add to latency time series
          newState.latencyHistory = addToTimeSeries(
            prev.latencyHistory,
            {
              timestamp,
              api_latency_ms: data.api_latency_ms,
              processing_latency_ms: data.processing_latency_ms,
              total_latency_ms: data.api_latency_ms + data.processing_latency_ms,
            },
            MAX_TIME_SERIES_POINTS
          );
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

          newState.promptEvolutions = [...prev.promptEvolutions, evolution];
        } else if (isErrorEvent(event)) {
          const data = event.data as ErrorData;
          newState.lastError = data;
        } else if (isHeartbeatEvent(event)) {
          newState.lastHeartbeat = timestamp;
        } else if (isConnectionAckEvent(event)) {
          log("Connection acknowledged:", event.data);
        }

        return newState;
      });

      // End performance measurement after state update is queued
      endMeasure();
    },
    [log]
  );

  // ============================================================================
  // WebSocket Management
  // ============================================================================

  /**
   * Send message to WebSocket
   */
  const sendMessage = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  /**
   * Send ping to keep connection alive
   */
  const sendPing = useCallback(() => {
    sendMessage({
      type: "ping",
      timestamp: new Date().toISOString(),
    });
  }, [sendMessage]);

  /**
   * Request campaign summary
   */
  const requestSummary = useCallback(() => {
    sendMessage({ type: "get_summary" });
  }, [sendMessage]);

  /**
   * Set up heartbeat response handling
   */
  const startHeartbeatTimeout = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }

    // If we don't receive a heartbeat within the expected interval + buffer,
    // consider the connection stale
    heartbeatTimeoutRef.current = setTimeout(() => {
      log("Heartbeat timeout - connection may be stale");
      // The server should send heartbeats - if none received, let onclose handle it
    }, HEARTBEAT_INTERVAL_MS * 2);
  }, [log]);

  /**
   * Clean up connection resources
   */
  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  /**
   * Close WebSocket connection
   */
  const disconnect = useCallback(() => {
    cleanup();
    reconnectAttemptsRef.current = 0;

    if (wsRef.current) {
      wsRef.current.close(1000, "User disconnect");
      wsRef.current = null;
    }

    setDashboardState((prev) => ({
      ...prev,
      connectionStatus: "disconnected",
      reconnectAttempts: 0,
    }));

    log("Disconnected");
  }, [cleanup, log]);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    if (isUnmountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    cleanup();

    const url = buildWebSocketUrl(campaignId, options);
    log("Connecting to:", url);

    setDashboardState((prev) => ({
      ...prev,
      connectionStatus:
        reconnectAttemptsRef.current > 0 ? "reconnecting" : "connecting",
      reconnectAttempts: reconnectAttemptsRef.current,
    }));

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (isUnmountedRef.current) return;

        log("Connected");
        reconnectAttemptsRef.current = 0;

        setDashboardState((prev) => ({
          ...prev,
          connectionStatus: "connected",
          reconnectAttempts: 0,
          lastError: null,
        }));

        // Start heartbeat monitoring
        startHeartbeatTimeout();

        // Send initial ping
        sendPing();
      };

      ws.onmessage = (event) => {
        if (isUnmountedRef.current) return;

        try {
          const data = JSON.parse(event.data);
          processEvent(data);

          // Reset heartbeat timeout on any message
          startHeartbeatTimeout();

          // Respond to server pings
          if (data.event_type === AegisTelemetryEventType.HEARTBEAT) {
            sendMessage({
              type: "pong",
              timestamp: new Date().toISOString(),
            });
          }
        } catch (err) {
          log("Failed to parse message:", err);
        }
      };

      ws.onerror = (event) => {
        log("WebSocket error:", event);

        setDashboardState((prev) => ({
          ...prev,
          connectionStatus: "error",
          lastError: {
            error_code: "WEBSOCKET_ERROR",
            error_message: "WebSocket connection error",
            severity: "high",
            component: "websocket",
            recoverable: true,
          },
        }));
      };

      ws.onclose = (event) => {
        if (isUnmountedRef.current) return;

        log("Connection closed:", event.code, event.reason);
        cleanup();
        wsRef.current = null;

        // Check if we should reconnect
        if (
          event.code !== 1000 && // Normal closure
          reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
        ) {
          reconnectAttemptsRef.current++;
          const delay = calculateBackoffDelay(reconnectAttemptsRef.current);

          log(
            `Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`
          );

          setDashboardState((prev) => ({
            ...prev,
            connectionStatus: "reconnecting",
            reconnectAttempts: reconnectAttemptsRef.current,
          }));

          reconnectTimeoutRef.current = setTimeout(() => {
            if (!isUnmountedRef.current) {
              connect();
            }
          }, delay);
        } else {
          setDashboardState((prev) => ({
            ...prev,
            connectionStatus: "disconnected",
            reconnectAttempts: reconnectAttemptsRef.current,
          }));
        }
      };
    } catch (err) {
      log("Failed to create WebSocket:", err);

      setDashboardState((prev) => ({
        ...prev,
        connectionStatus: "error",
        lastError: {
          error_code: "CONNECTION_FAILED",
          error_message:
            err instanceof Error ? err.message : "Failed to connect",
          severity: "high",
          component: "websocket",
          recoverable: true,
        },
      }));
    }
  }, [
    campaignId,
    options,
    cleanup,
    log,
    processEvent,
    sendMessage,
    sendPing,
    startHeartbeatTimeout,
  ]);

  /**
   * Manual reconnect function
   */
  const reconnect = useCallback(() => {
    log("Manual reconnect requested");
    reconnectAttemptsRef.current = 0;
    disconnect();

    // Use timeout to ensure cleanup completes
    setTimeout(() => {
      if (!isUnmountedRef.current) {
        connect();
      }
    }, 100);
  }, [connect, disconnect, log]);

  // ============================================================================
  // Effects
  // ============================================================================

  // Auto-connect on mount
  useEffect(() => {
    isUnmountedRef.current = false;

    if (autoConnect && campaignId) {
      connect();
    }

    return () => {
      isUnmountedRef.current = true;
      disconnect();
    };
  }, [autoConnect, campaignId, connect, disconnect]);

  // ============================================================================
  // Memoized Return Values
  // ============================================================================

  const metrics = useMemo(
    () =>
      dashboardState.campaignSummary?.attack_metrics ??
      createDefaultAttackMetrics(),
    [dashboardState.campaignSummary?.attack_metrics]
  );

  const techniqueBreakdown = useMemo(
    () => dashboardState.campaignSummary?.technique_breakdown ?? [],
    [dashboardState.campaignSummary?.technique_breakdown]
  );

  const tokenUsage = useMemo(
    () =>
      dashboardState.campaignSummary?.token_usage ?? createDefaultTokenUsage(),
    [dashboardState.campaignSummary?.token_usage]
  );

  const latencyMetrics = useMemo(
    () =>
      dashboardState.campaignSummary?.latency_metrics ??
      createDefaultLatencyMetrics(),
    [dashboardState.campaignSummary?.latency_metrics]
  );

  const isConnected = dashboardState.connectionStatus === "connected";

  return {
    connectionStatus: dashboardState.connectionStatus,
    dashboardState,
    metrics,
    campaignSummary: dashboardState.campaignSummary,
    techniqueBreakdown,
    tokenUsage,
    latencyMetrics,
    recentEvents: dashboardState.recentEvents,
    successRateHistory: dashboardState.successRateHistory,
    tokenUsageHistory: dashboardState.tokenUsageHistory,
    latencyHistory: dashboardState.latencyHistory,
    promptEvolutions: dashboardState.promptEvolutions,
    error: dashboardState.lastError,
    reconnectAttempts: dashboardState.reconnectAttempts,
    isConnected,
    reconnect,
    disconnect,
    requestSummary,
    sendPing,
  };
}

// ============================================================================
// Export hook type for use in components
// ============================================================================

export type UseAegisTelemetry = typeof useAegisTelemetry;
