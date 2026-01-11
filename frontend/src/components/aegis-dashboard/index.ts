/**
 * Aegis Dashboard Components
 *
 * This module exports all components for the Real-Time Aegis Campaign Dashboard.
 * Components include WebSocket connection status, metrics cards, charts, and
 * event feeds for monitoring Aegis campaign telemetry in real-time.
 */

// Connection status indicator
export { ConnectionStatus } from "./ConnectionStatus";
export type { ConnectionStatusProps } from "./ConnectionStatus";

// Attack success rate metric card
export { SuccessRateCard } from "./SuccessRateCard";
export type { SuccessRateCardProps } from "./SuccessRateCard";

// Success rate trend chart (area chart over time)
export { SuccessRateTrendChart } from "./SuccessRateTrendChart";
export type { SuccessRateTrendChartProps } from "./SuccessRateTrendChart";
