/**
 * Comparison Components Index
 *
 * This module exports all comparison-related components for the Campaign Analytics feature.
 * These components enable side-by-side comparison of up to 4 campaigns with normalized metrics.
 *
 * @example
 * ```tsx
 * import {
 *   CampaignComparisonPanel,
 *   ComparisonTable,
 *   NormalizedMetricsChart,
 *   DeltaIndicator,
 * } from '@/components/campaign-analytics/comparison';
 * ```
 */

// =============================================================================
// ComparisonTable Components
// =============================================================================

export {
  ComparisonTable,
  ComparisonTableSkeleton,
  ComparisonTableError,
  ComparisonTableEmpty,
  SimpleComparisonTable,
  CompactComparisonTable,
  SummaryComparisonTable,
  type MetricDirection,
  type MetricFormat,
  type MetricRowConfig,
  type MetricGroupConfig,
  type ComparisonTableProps,
} from "./ComparisonTable";

// =============================================================================
// NormalizedMetricsChart Components
// =============================================================================

export {
  NormalizedMetricsChart,
  NormalizedMetricsChartSkeleton,
  NormalizedMetricsChartError,
  NormalizedMetricsChartEmpty,
  SimpleNormalizedMetricsChart,
  CompactNormalizedMetricsChart,
  DetailedNormalizedMetricsChart,
  type NormalizedMetric,
  type RadarDataPoint,
  type CampaignSeries,
  type NormalizedMetricsChartProps,
} from "./NormalizedMetricsChart";

// =============================================================================
// DeltaIndicator Components
// =============================================================================

export {
  DeltaIndicator,
  PercentageDeltaIndicator,
  AbsoluteDeltaIndicator,
  SuccessRateDelta,
  LatencyDelta,
  CostDelta,
  TokenDelta,
  InlineDelta,
  LargeDelta,
  DeltaComparison,
  DeltaBadge,
  type DeltaMode,
  type DeltaDirection,
  type DeltaSize,
  type DeltaIconStyle,
  type DeltaColorPreset,
  type DeltaIndicatorProps,
  type DeltaComparisonProps,
  type DeltaBadgeProps,
} from "./DeltaIndicator";

// =============================================================================
// CampaignComparisonPanel Components
// =============================================================================

export {
  CampaignComparisonPanel,
  SimpleComparisonPanel,
  CompactComparisonPanel,
  DetailedComparisonPanel,
  type ComparisonViewMode,
  type CampaignComparisonPanelProps,
} from "./CampaignComparisonPanel";
