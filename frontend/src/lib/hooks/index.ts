/**
 * Hooks Index
 *
 * Centralized exports for all custom React hooks.
 */

// Existing hooks
export { useWebSocket } from "./useWebSocket";
export type { WebSocketOptions, WebSocketHook } from "./useWebSocket";

// Concurrent hooks
export * from "./concurrent-hooks";

// Debounced request hooks
export * from "./use-debounced-request";

// Aegis Campaign Telemetry hook
export { useAegisTelemetry } from "./useAegisTelemetry";
export type {
  UseAegisTelemetryOptions,
  UseAegisTelemetryReturn,
  UseAegisTelemetry,
} from "./useAegisTelemetry";

// Aegis Performance Optimization hooks
export {
  useChartDataDebounce,
  CHART_UPDATE_DEBOUNCE_MS,
  MAX_CHART_RENDER_POINTS,
} from "./useChartDataDebounce";
export type { UseChartDataDebounceReturn } from "./useChartDataDebounce";

export {
  usePerformanceMonitor,
  aegisPerformanceMonitor,
  AegisPerformanceMonitor,
  MAX_EVENT_PROCESSING_MS,
} from "./useAegisPerformanceMonitor";
export type {
  PerformanceSample,
  PerformanceMetrics,
  PerformanceMonitorConfig,
} from "./useAegisPerformanceMonitor";
