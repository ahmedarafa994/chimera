/**
 * Campaign Analytics TypeScript Types
 *
 * Type definitions for campaign telemetry analytics, statistical analysis,
 * campaign comparison, time-series data, and export functionality.
 *
 * These types align with the backend Pydantic schemas defined in:
 * - backend-api/app/schemas/campaign_analytics.py
 */

// =============================================================================
// Enums
// =============================================================================

/**
 * Campaign lifecycle status.
 */
export enum CampaignStatusEnum {
  DRAFT = 'draft',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Individual telemetry event execution status.
 */
export enum ExecutionStatusEnum {
  PENDING = 'pending',
  SUCCESS = 'success',
  PARTIAL_SUCCESS = 'partial_success',
  FAILURE = 'failure',
  TIMEOUT = 'timeout',
  SKIPPED = 'skipped',
}

/**
 * Supported export formats.
 */
export enum ExportFormat {
  CSV = 'csv',
  JSON = 'json',
  PNG = 'png',
  SVG = 'svg',
}

/**
 * Time series granularity options.
 */
export enum TimeGranularity {
  MINUTE = 'minute',
  HOUR = 'hour',
  DAY = 'day',
}

/**
 * Types of metrics for comparison.
 */
export enum MetricType {
  SUCCESS_RATE = 'success_rate',
  LATENCY = 'latency',
  TOKEN_USAGE = 'token_usage',
  COST = 'cost',
  EFFECTIVENESS = 'effectiveness',
}

// =============================================================================
// Base Campaign Types
// =============================================================================

/**
 * Base campaign fields shared across interfaces.
 */
export interface CampaignBase {
  name: string;
  description?: string | null;
  objective: string;
}

/**
 * Request for creating a new campaign.
 */
export interface CampaignCreate extends CampaignBase {
  target_provider?: string | null;
  target_model?: string | null;
  technique_suites?: string[];
  transformation_config?: Record<string, unknown>;
  config?: Record<string, unknown>;
  tags?: string[];
}

/**
 * Request for updating a campaign.
 */
export interface CampaignUpdate {
  name?: string | null;
  description?: string | null;
  status?: CampaignStatusEnum | null;
  config?: Record<string, unknown> | null;
  tags?: string[] | null;
}

// =============================================================================
// Campaign Summary and Response Types
// =============================================================================

/**
 * Summary view of a campaign for list displays.
 */
export interface CampaignSummary {
  id: string;
  name: string;
  description?: string | null;
  objective: string;
  status: CampaignStatusEnum;
  target_provider?: string | null;
  target_model?: string | null;
  technique_suites: string[];
  tags: string[];

  // Quick stats
  total_attempts: number;
  success_rate?: number | null;
  avg_latency_ms?: number | null;

  // Timestamps
  created_at?: string;
  updated_at?: string;
  started_at?: string | null;
  completed_at?: string | null;
  duration_seconds?: number | null;
}

/**
 * Detailed campaign information including configuration.
 */
export interface CampaignDetail extends CampaignSummary {
  transformation_config: Record<string, unknown>;
  config: Record<string, unknown>;
  user_id?: string | null;
  session_id?: string | null;
}

/**
 * Alias for Campaign (commonly used reference).
 */
export type Campaign = CampaignDetail;

// =============================================================================
// Statistics Types
// =============================================================================

/**
 * Percentile-based statistics.
 */
export interface PercentileStats {
  p50?: number | null;
  p90?: number | null;
  p95?: number | null;
  p99?: number | null;
}

/**
 * Distribution statistics for a metric.
 */
export interface DistributionStats {
  mean?: number | null;
  median?: number | null;
  std_dev?: number | null;
  min_value?: number | null;
  max_value?: number | null;
  percentiles?: PercentileStats | null;
}

/**
 * Breakdown of attempt counts by status.
 */
export interface AttemptCounts {
  total: number;
  successful: number;
  failed: number;
  partial_success: number;
  timeout: number;
  skipped: number;
}

/**
 * Comprehensive statistical analysis of a campaign.
 *
 * Includes distribution statistics for success rates, latency, token usage,
 * and cost metrics. All statistics include mean, median, p95, and std_dev.
 */
export interface CampaignStatistics {
  campaign_id: string;

  // Attempt counts
  attempts: AttemptCounts;

  // Success rate statistics
  success_rate: DistributionStats;

  // Semantic success statistics
  semantic_success: DistributionStats;

  // Latency statistics (milliseconds)
  latency_ms: DistributionStats;

  // Token usage statistics
  prompt_tokens: DistributionStats;
  completion_tokens: DistributionStats;
  total_tokens: DistributionStats;

  // Token totals
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens_used: number;

  // Cost statistics (in USD cents)
  cost_cents: DistributionStats;
  total_cost_cents: number;

  // Duration
  total_duration_seconds?: number | null;

  // Computed timestamp
  computed_at: string;
}

// =============================================================================
// Breakdown Types
// =============================================================================

/**
 * Single item in a breakdown (technique, provider, or model).
 */
export interface BreakdownItem {
  name: string;
  attempts: number;
  successes: number;
  success_rate: number;
  avg_latency_ms?: number | null;
  avg_tokens?: number | null;
  total_cost_cents?: number | null;
}

/**
 * Breakdown of results by transformation technique.
 */
export interface TechniqueBreakdown {
  campaign_id: string;
  items: BreakdownItem[];
  best_technique?: string | null;
  worst_technique?: string | null;
}

/**
 * Breakdown of results by LLM provider.
 */
export interface ProviderBreakdown {
  campaign_id: string;
  items: BreakdownItem[];
  best_provider?: string | null;
}

/**
 * Breakdown of results by potency level.
 */
export interface PotencyBreakdown {
  campaign_id: string;
  items: BreakdownItem[];
  best_potency_level?: number | null;
}

// =============================================================================
// Time Series Types
// =============================================================================

/**
 * Single data point in a time series.
 */
export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
  count?: number | null;
}

/**
 * Time series data for telemetry visualization.
 *
 * Provides time-bucketed data for charting success rates, latency,
 * and other metrics over the campaign duration.
 */
export interface TelemetryTimeSeries {
  campaign_id: string;
  metric: string;
  granularity: TimeGranularity;
  data_points: TimeSeriesDataPoint[];
  start_time?: string | null;
  end_time?: string | null;
  total_points: number;
}

/**
 * Multiple time series for overlay comparison.
 */
export interface MultiSeriesTimeSeries {
  series: TelemetryTimeSeries[];
  metrics: string[];
}

// =============================================================================
// Campaign Comparison Types
// =============================================================================

/**
 * Single campaign's data for comparison.
 */
export interface CampaignComparisonItem {
  campaign_id: string;
  campaign_name: string;
  status: CampaignStatusEnum;

  // Core metrics for comparison
  total_attempts: number;
  success_rate?: number | null;
  semantic_success_mean?: number | null;

  // Latency
  latency_mean?: number | null;
  latency_p95?: number | null;

  // Token usage
  avg_tokens?: number | null;
  total_tokens?: number | null;

  // Cost
  total_cost_cents?: number | null;
  avg_cost_per_attempt?: number | null;

  // Duration
  duration_seconds?: number | null;

  // Best performers
  best_technique?: string | null;
  best_provider?: string | null;

  // Normalized metrics (0-1 scale for radar charts)
  normalized_success_rate?: number | null;
  normalized_latency?: number | null;
  normalized_cost?: number | null;
  normalized_effectiveness?: number | null;
}

/**
 * Request to compare multiple campaigns.
 */
export interface CampaignComparisonRequest {
  campaign_ids: string[];
  metrics?: MetricType[] | null;
  include_time_series?: boolean;
  normalize_metrics?: boolean;
}

/**
 * Response containing campaign comparison data.
 */
export interface CampaignComparison {
  campaigns: CampaignComparisonItem[];

  // Comparison metadata
  compared_at: string;

  // Winner identification
  best_success_rate_campaign?: string | null;
  best_latency_campaign?: string | null;
  best_cost_efficiency_campaign?: string | null;

  // Deltas between campaigns (for 2-campaign comparison)
  delta_success_rate?: number | null;
  delta_latency_ms?: number | null;
  delta_cost_cents?: number | null;

  // Optional time series for overlay
  time_series?: TelemetryTimeSeries[] | null;
}

// =============================================================================
// Telemetry Event Types
// =============================================================================

/**
 * Summary of a single telemetry event.
 */
export interface TelemetryEventSummary {
  id: string;
  campaign_id: string;
  sequence_number: number;

  // Prompt info (truncated for summary)
  original_prompt_preview?: string | null;

  // Execution info
  technique_suite: string;
  potency_level: number;
  provider: string;
  model: string;
  status: ExecutionStatusEnum;

  // Metrics
  success_indicator: boolean;
  total_latency_ms: number;
  total_tokens: number;

  created_at: string;
}

/**
 * Full telemetry event details for drill-down view.
 */
export interface TelemetryEventDetail extends TelemetryEventSummary {
  // Full prompts
  original_prompt: string;
  transformed_prompt?: string | null;
  response_text?: string | null;

  // Applied techniques
  applied_techniques: string[];

  // Detailed timing
  execution_time_ms: number;
  transformation_time_ms: number;

  // Token breakdown
  prompt_tokens: number;
  completion_tokens: number;

  // Quality scores
  semantic_success_score?: number | null;
  effectiveness_score?: number | null;
  naturalness_score?: number | null;
  detectability_score?: number | null;

  // Detection
  bypass_indicators: string[];
  safety_trigger_detected: boolean;

  // Error info
  error_message?: string | null;
  error_code?: string | null;

  // Additional metadata
  metadata: Record<string, unknown>;
}

/**
 * Alias for commonly used TelemetryEvent.
 */
export type TelemetryEvent = TelemetryEventDetail;

// =============================================================================
// Export Types
// =============================================================================

/**
 * Options for chart export.
 */
export interface ExportChartOptions {
  format: ExportFormat;
  width?: number;
  height?: number;
  include_legend?: boolean;
  include_title?: boolean;
  background_color?: string;
  theme?: 'light' | 'dark';
}

/**
 * Options for data export.
 */
export interface ExportDataOptions {
  format: ExportFormat;
  include_headers?: boolean;
  date_format?: 'ISO' | 'US' | 'EU';
  decimal_precision?: number;
  include_metadata?: boolean;
}

/**
 * Combined export options for frontend use.
 */
export interface ExportOptions {
  type: 'chart' | 'data' | 'full';
  format: ExportFormat;

  // What to include
  include_summary?: boolean;
  include_time_series?: boolean;
  include_breakdowns?: boolean;
  include_raw_events?: boolean;

  // Chart-specific options
  chart_options?: ExportChartOptions;

  // Data-specific options
  data_options?: ExportDataOptions;
}

/**
 * Request to export campaign data or charts.
 */
export interface ExportRequest {
  campaign_id: string;
  export_type: 'chart' | 'data' | 'full';

  // What to include
  include_summary?: boolean;
  include_time_series?: boolean;
  include_breakdowns?: boolean;
  include_raw_events?: boolean;

  // Chart options (if exporting charts)
  chart_options?: ExportChartOptions | null;

  // Data options (if exporting data)
  data_options?: ExportDataOptions | null;

  // Filtering
  start_time?: string | null;
  end_time?: string | null;
  technique_filter?: string[] | null;
  provider_filter?: string[] | null;
  status_filter?: ExecutionStatusEnum[] | null;
}

/**
 * Response containing export results.
 */
export interface ExportResponse {
  success: boolean;
  campaign_id: string;
  export_type: string;

  // File info
  file_name: string;
  file_size_bytes: number;
  mime_type: string;

  // For inline data (small exports)
  data?: string | null;

  // For file downloads (large exports)
  download_url?: string | null;
  expires_at?: string | null;

  // Export metadata
  exported_at: string;
  row_count?: number | null;
  processing_time_ms?: number | null;
}

// =============================================================================
// Filter and Query Types
// =============================================================================

/**
 * Filter parameters for campaign queries.
 */
export interface CampaignFilterParams {
  status?: CampaignStatusEnum[] | null;
  provider?: string[] | null;
  technique_suite?: string[] | null;
  tags?: string[] | null;
  start_date?: string | null;
  end_date?: string | null;
  min_attempts?: number | null;
  min_success_rate?: number | null;
  search?: string | null;
}

/**
 * Filter parameters for telemetry event queries.
 */
export interface TelemetryFilterParams {
  status?: ExecutionStatusEnum[] | null;
  technique_suite?: string[] | null;
  provider?: string[] | null;
  model?: string[] | null;
  success_only?: boolean | null;
  start_time?: string | null;
  end_time?: string | null;
  min_potency?: number | null;
  max_potency?: number | null;
}

/**
 * Combined filter options for UI components.
 */
export interface FilterOptions {
  // Campaign filters
  campaigns?: CampaignFilterParams;

  // Telemetry filters
  telemetry?: TelemetryFilterParams;

  // Time range (common across both)
  dateRange?: {
    start: string | null;
    end: string | null;
  };

  // Quick filter presets
  preset?: 'last_7_days' | 'last_30_days' | 'last_90_days' | 'all_time' | 'custom';
}

/**
 * Query parameters for time series data.
 */
export interface TimeSeriesQuery {
  campaign_id: string;
  metrics?: string[];
  granularity?: TimeGranularity;
  start_time?: string | null;
  end_time?: string | null;
  filters?: TelemetryFilterParams | null;
}

// =============================================================================
// Pagination Types
// =============================================================================

/**
 * Request for paginated campaign list.
 */
export interface CampaignListRequest {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  filters?: CampaignFilterParams | null;
}

/**
 * Paginated campaign list response.
 */
export interface CampaignListResponse {
  items: CampaignSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

/**
 * Paginated telemetry event list response.
 */
export interface TelemetryListResponse {
  items: TelemetryEventSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// =============================================================================
// Cache Types
// =============================================================================

/**
 * Cache statistics for campaign analytics.
 */
export interface CampaignCacheStats {
  total_entries: number;
  memory_usage_bytes?: number;
  hit_rate?: number;
  oldest_entry?: string | null;
  newest_entry?: string | null;
}

// =============================================================================
// UI State Types (Frontend-specific)
// =============================================================================

/**
 * Selected campaigns for comparison feature.
 */
export interface ComparisonSelection {
  campaign_ids: string[];
  max_campaigns: number;
}

/**
 * Chart reference for export functionality.
 */
export interface ChartExportRef {
  id: string;
  name: string;
  ref: React.RefObject<HTMLDivElement>;
}

/**
 * Dashboard tab configuration.
 */
export interface AnalyticsDashboardTab {
  id: string;
  label: string;
  icon?: string;
  count?: number;
}

/**
 * Sort configuration for tables.
 */
export interface SortConfig {
  field: string;
  direction: 'asc' | 'desc';
}

/**
 * Date range for filtering.
 */
export interface DateRange {
  start: Date | null;
  end: Date | null;
}

// =============================================================================
// API Response Wrappers
// =============================================================================

/**
 * Generic API response wrapper for campaign analytics endpoints.
 */
export interface CampaignAnalyticsResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

/**
 * Error response from campaign analytics API.
 */
export interface CampaignAnalyticsError {
  error: string;
  message: string;
  code?: string;
  detail?: unknown;
}
