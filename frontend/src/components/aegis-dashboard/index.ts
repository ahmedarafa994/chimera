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

// Technique performance breakdown (bar chart with metrics)
export { TechniqueBreakdown } from "./TechniqueBreakdown";
export type { TechniqueBreakdownProps } from "./TechniqueBreakdown";

// Token usage and cost tracking card
export { TokenUsageCard } from "./TokenUsageCard";
export type { TokenUsageCardProps } from "./TokenUsageCard";

// Latency metrics card (API and processing latency)
export { LatencyCard } from "./LatencyCard";
export type { LatencyCardProps } from "./LatencyCard";

// Live event feed (scrolling real-time events)
export { LiveEventFeed } from "./LiveEventFeed";
export type { LiveEventFeedProps } from "./LiveEventFeed";

// Prompt evolution timeline (prompt transformation visualization)
export { PromptEvolutionTimeline } from "./PromptEvolutionTimeline";
export type { PromptEvolutionTimelineProps } from "./PromptEvolutionTimeline";
