/**
 * Health Monitoring Dashboard Components
 *
 * A comprehensive set of UI components for monitoring LLM provider health,
 * displaying real-time metrics, and visualizing historical performance data.
 *
 * Features:
 * - Provider health cards with operational/degraded/down status indicators
 * - Real-time latency and error rate visualization
 * - Uptime tracking with multiple time window displays
 * - Circuit breaker status monitoring
 * - Historical metrics charts with time range selection
 *
 * @example
 * ```tsx
 * import {
 *   ProviderHealthCard,
 *   HealthMetricsChart,
 *   UptimeIndicator,
 *   UptimeBadges,
 *   CircularUptime,
 * } from "@/components/health-dashboard";
 *
 * // Display a provider health card
 * <ProviderHealthCard
 *   metrics={{
 *     provider_id: "openai",
 *     provider_name: "OpenAI",
 *     status: "operational",
 *     latency_ms: 245,
 *     error_rate: 0.1,
 *     uptime_percent: 99.95,
 *     ...
 *   }}
 *   onRefresh={handleRefresh}
 *   onViewDetails={handleViewDetails}
 * />
 *
 * // Display health metrics chart
 * <HealthMetricsChart
 *   data={historyData}
 *   providerName="OpenAI"
 *   onTimeRangeChange={handleTimeRangeChange}
 * />
 *
 * // Display uptime indicator
 * <UptimeIndicator
 *   uptimePercent={99.95}
 *   size="md"
 *   showLabel
 * />
 *
 * // Display uptime across multiple time windows
 * <UptimeBadges
 *   lastHour={99.99}
 *   last24Hours={99.95}
 *   last7Days={99.9}
 *   last30Days={99.85}
 * />
 * ```
 */

// Provider Health Card - Status card with operational/degraded/down indicators
export { ProviderHealthCard } from "./ProviderHealthCard";
export type {
  ProviderHealthCardProps,
  ProviderHealthMetrics,
  ProviderStatus,
} from "./ProviderHealthCard";

// Health Metrics Chart - Latency and error rate line charts
export { HealthMetricsChart } from "./HealthMetricsChart";
export type {
  HealthMetricsChartProps,
  HealthHistoryEntry,
  TimeRange,
} from "./HealthMetricsChart";

// Uptime Indicator - Visual uptime percentage display
export {
  UptimeIndicator,
  UptimeBadges,
  CircularUptime,
  UptimeDots,
} from "./UptimeIndicator";
export type {
  UptimeIndicatorProps,
  UptimeBadgesProps,
  CircularUptimeProps,
  UptimeDotsProps,
} from "./UptimeIndicator";

// Full Dashboard Component - Combines all health components
export { ProviderHealthDashboard } from "./ProviderHealthDashboard";
export type {
  ProviderHealthDashboardProps,
  HealthSummary,
  HealthAlert,
} from "./ProviderHealthDashboard";
