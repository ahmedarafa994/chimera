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
 * - Quota usage tracking with progress bars and alerts
 * - Rate limit visualization with gauges showing RPM/TPM vs caps
 *
 * @example
 * ```tsx
 * import {
 *   ProviderHealthCard,
 *   HealthMetricsChart,
 *   UptimeIndicator,
 *   UptimeBadges,
 *   CircularUptime,
 *   QuotaUsageCard,
 *   QuotaDashboard,
 *   RateLimitGauge,
 *   RateLimitDashboard,
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
 *
 * // Display quota usage card
 * <QuotaUsageCard
 *   quota={{
 *     provider_id: "openai",
 *     provider_name: "OpenAI",
 *     usage_percent: 75.5,
 *     tokens_used: 1500000,
 *     tokens_limit: 2000000,
 *     ...
 *   }}
 *   history={quotaHistory}
 *   onRefresh={handleRefresh}
 * />
 *
 * // Display rate limit gauge
 * <RateLimitGauge
 *   metrics={{
 *     provider_id: "openai",
 *     provider_name: "OpenAI",
 *     requests_per_minute: 45,
 *     effective_rpm_limit: 60,
 *     rpm_usage_percent: 75,
 *     ...
 *   }}
 *   onRefresh={handleRefresh}
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

// Quota Usage Card - Progress bar showing quota usage with alerts
export { QuotaUsageCard, QuotaDashboard } from "./QuotaUsageCard";
export type {
  QuotaUsageCardProps,
  QuotaDashboardProps,
  QuotaStatus,
  QuotaHistoryEntry,
  QuotaPeriod,
  QuotaAlertLevel,
} from "./QuotaUsageCard";

// Rate Limit Gauge - Visual gauge showing requests/minute vs cap
export { RateLimitGauge, RateLimitDashboard } from "./RateLimitGauge";
export type {
  RateLimitGaugeProps,
  RateLimitDashboardProps,
  RateLimitMetrics,
  RateLimitLevel,
} from "./RateLimitGauge";
