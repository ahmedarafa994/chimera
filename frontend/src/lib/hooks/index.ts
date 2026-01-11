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
