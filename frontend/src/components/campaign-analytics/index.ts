/**
 * Campaign Analytics Components
 *
 * This module exports all components for the Campaign Telemetry Analytics Panel.
 * Import from this barrel file for clean, consistent imports:
 *
 * @example
 * ```tsx
 * import {
 *   StatisticsCard,
 *   CampaignSelector,
 *   StatisticsSummaryPanel,
 *   CampaignOverviewCard,
 * } from '@/components/campaign-analytics';
 * ```
 */

// =============================================================================
// StatisticsCard Components
// =============================================================================

export {
  StatisticsCard,
  StatisticsCardSkeleton,
  DistributionStatCard,
  SuccessRateCard,
  LatencyCard,
  TokenUsageCard,
  CostCard,
  type TrendDirection,
  type StatisticMetricType,
  type ValueFormat,
  type ColorVariant,
  type StatisticsCardProps,
  type DistributionStatCardProps,
} from "./StatisticsCard";

// =============================================================================
// CampaignSelector Components
// =============================================================================

export {
  CampaignSelector,
  CampaignSelectorSingle,
  CampaignSelectorMulti,
  CampaignComparisonSelector,
  type SelectionMode,
  type CampaignSelectorProps,
} from "./CampaignSelector";

// =============================================================================
// StatisticsSummaryPanel Components
// =============================================================================

export {
  StatisticsSummaryPanel,
  StatisticsSummaryPanelSkeleton,
  StatisticsSummaryPanelError,
  StatisticsSummaryPanelEmpty,
  CompactStatisticsSummary,
  InlineStatisticsSummary,
  type StatisticsSummaryPanelProps,
} from "./StatisticsSummaryPanel";

// =============================================================================
// CampaignOverviewCard Components
// =============================================================================

export {
  CampaignOverviewCard,
  CampaignOverviewCardSkeleton,
  CampaignOverviewCardError,
  CampaignOverviewCardEmpty,
  CompactCampaignOverviewCard,
  CampaignListCard,
  type CampaignQuickAction,
  type CampaignOverviewCardProps,
} from "./CampaignOverviewCard";

// =============================================================================
// Chart Components
// =============================================================================

export {
  SuccessRateTimeSeriesChart,
  SuccessRateTimeSeriesChartSkeleton,
  SuccessRateTimeSeriesChartError,
  SuccessRateTimeSeriesChartEmpty,
  SimpleSuccessRateChart,
  ComparisonSuccessRateChart,
  type ChartDataPoint,
  type ChartSeries,
  type ZoomState,
  type SuccessRateTimeSeriesChartProps,
} from "./charts/SuccessRateTimeSeriesChart";
