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

export {
  TechniqueEffectivenessChart,
  TechniqueEffectivenessChartSkeleton,
  TechniqueEffectivenessChartError,
  TechniqueEffectivenessChartEmpty,
  SimpleTechniqueChart,
  TechniqueRadarChart,
  CompactTechniqueChart,
  type TechniqueEffectivenessChartProps,
  type TechniqueChartDataPoint,
  type ChartViewMode,
  type SortField,
  type SortDirection,
  type SuccessTier,
} from "./charts/TechniqueEffectivenessChart";

export {
  ProviderComparisonChart,
  ProviderComparisonChartSkeleton,
  ProviderComparisonChartError,
  ProviderComparisonChartEmpty,
  SimpleProviderChart,
  ProviderLatencyChart,
  ProviderCostChart,
  CompactProviderChart,
  type ProviderComparisonChartProps,
  type ProviderChartDataPoint,
  type DisplayMode,
  type ProviderSortField,
} from "./charts/ProviderComparisonChart";

export {
  PromptEvolutionChart,
  PromptEvolutionChartSkeleton,
  PromptEvolutionChartError,
  PromptEvolutionChartEmpty,
  SimplePromptEvolutionChart,
  CompactPromptEvolutionChart,
  DetailedPromptEvolutionChart,
  type PromptEvolutionChartProps,
  type PromptEvolutionDataPoint,
  type TrendLinePoint,
  type PromptEvolutionFilter,
  type CorrelationResult,
} from "./charts/PromptEvolutionChart";

export {
  LatencyDistributionChart,
  LatencyDistributionChartSkeleton,
  LatencyDistributionChartError,
  LatencyDistributionChartEmpty,
  SimpleLatencyChart,
  CompactLatencyChart,
  DetailedLatencyChart,
  PercentileLatencyChart,
  type LatencyDistributionChartProps,
  type HistogramBin,
  type BoxPlotStats,
  type LatencyDataPoint,
  type LatencyFilter,
} from "./charts/LatencyDistributionChart";

// =============================================================================
// Comparison Components
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
} from "./comparison/ComparisonTable";

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
} from "./comparison/NormalizedMetricsChart";

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
} from "./comparison/DeltaIndicator";

export {
  CampaignComparisonPanel,
  SimpleComparisonPanel,
  CompactComparisonPanel,
  DetailedComparisonPanel,
  type ComparisonViewMode,
  type CampaignComparisonPanelProps,
} from "./comparison/CampaignComparisonPanel";

// =============================================================================
// Export Components
// =============================================================================

export {
  ExportButton,
  ChartExportButton,
  DataExportButton,
  CompactExportButton,
  FullExportButton,
  ExportButtonSkeleton,
  ExportButtonDisabled,
  type ExportFormat,
  type ExportResult,
  type ChartExportOptions,
  type CSVExportOptions,
  type ExportButtonProps,
} from "./ExportButton";

export {
  ExportPanel,
  ExportPanelSheet,
  ExportPanelCompact,
  ExportPanelTrigger,
  ExportPanelSkeleton,
  ExportPanelEmpty,
  ExportPanelDisabled,
  type ExportableChart,
  type ExportableData,
  type BulkExportFormat,
  type ChartExportConfig,
  type CSVExportConfig,
  type ExportConfig,
  type ExportProgress,
  type BulkExportResult,
  type ExportPanelProps,
} from "./ExportPanel";

// =============================================================================
// Filter Components
// =============================================================================

export {
  FilterBar,
  FilterBarSkeleton,
  FilterBarEmpty,
  CompactFilterBar,
  InlineFilterBar,
  createDefaultFilterState,
  filterStateToParams,
  type DateRange,
  type FilterState,
  type FilterBarProps,
} from "./FilterBar";

// =============================================================================
// DateRangePicker Components
// =============================================================================

export {
  DateRangePicker,
  SimpleDateRangePicker,
  CompactDateRangePicker,
  CalendarOnlyDateRangePicker,
  AnalyticsDateRangePicker,
  DateRangePickerSkeleton,
  DateRangePickerEmpty,
  type DateRangeValue,
  type DateRangePreset,
  type DateRangePickerSize,
  type DateRangePickerAlign,
  type PresetConfig,
  type DateRangePickerProps,
} from "./DateRangePicker";
