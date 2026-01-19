/**
 * Campaign Analytics Charts Index
 *
 * Exports all chart components for campaign analytics visualization.
 */

// =============================================================================
// Time Series Charts
// =============================================================================

export {
  SuccessRateTimeSeriesChart,
  SuccessRateTimeSeriesChartSkeleton,
  SuccessRateTimeSeriesChartError,
  SuccessRateTimeSeriesChartEmpty,
  SimpleSuccessRateChart,
  ComparisonSuccessRateChart,
} from "./SuccessRateTimeSeriesChart";

export type {
  SuccessRateTimeSeriesChartProps,
  ChartDataPoint,
  ChartSeries,
  ZoomState,
} from "./SuccessRateTimeSeriesChart";

// =============================================================================
// Technique Effectiveness Charts
// =============================================================================

export {
  TechniqueEffectivenessChart,
  TechniqueEffectivenessChartSkeleton,
  TechniqueEffectivenessChartEmpty,
  SimpleTechniqueChart,
  TechniqueRadarChart,
  CompactTechniqueChart,
} from "./TechniqueEffectivenessChart";

export type {
  TechniqueEffectivenessChartProps,
  TechniqueEffectivenessData,
} from "./TechniqueEffectivenessChart";

// =============================================================================
// Provider Comparison Charts
// =============================================================================

export {
  ProviderComparisonChart,
  ProviderComparisonChartSkeleton,
  ProviderComparisonChartError,
  ProviderComparisonChartEmpty,
  SimpleProviderChart,
  ProviderLatencyChart,
  ProviderCostChart,
  CompactProviderChart,
} from "./ProviderComparisonChart";

export type {
  ProviderComparisonChartProps,
  ProviderChartDataPoint,
  DisplayMode,
  ProviderSortField,
} from "./ProviderComparisonChart";

// =============================================================================
// Prompt Evolution Charts
// =============================================================================

export {
  PromptEvolutionChart,
  PromptEvolutionChartSkeleton,
  PromptEvolutionChartError,
  PromptEvolutionChartEmpty,
  SimplePromptEvolutionChart,
  CompactPromptEvolutionChart,
  DetailedPromptEvolutionChart,
} from "./PromptEvolutionChart";

export type {
  PromptEvolutionChartProps,
  PromptEvolutionDataPoint,
  TrendLinePoint,
  PromptEvolutionFilter,
  CorrelationResult,
} from "./PromptEvolutionChart";

// =============================================================================
// Latency Distribution Charts
// =============================================================================

export {
  LatencyDistributionChart,
  LatencyDistributionChartSkeleton,
  LatencyDistributionChartError,
  LatencyDistributionChartEmpty,
  SimpleLatencyChart,
  CompactLatencyChart,
  DetailedLatencyChart,
  PercentileLatencyChart,
} from "./LatencyDistributionChart";

export type {
  LatencyDistributionChartProps,
  HistogramBin,
  BoxPlotStats,
  LatencyDataPoint,
  LatencyFilter,
  ChartViewMode as LatencyChartViewMode,
} from "./LatencyDistributionChart";
