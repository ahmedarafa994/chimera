/**
 * Types Index
 *
 * Centralized exports for all type definitions.
 */

// Admin types
export type {
  AdminStats,
  ImportDataRequest,
  ImportDataResponse,
  CacheStats,
  SystemHealth,
} from './admin-types';

// AutoDAN Lifelong Learning types
export type {
  LifelongAttackRequest,
  LifelongAttackResponse,
  LoopAttackRequest,
  LoopAttackResponse,
  AttackProgress,
  AttackLog,
  IntermediateResult,
  AttackResult,
  AttackResultItem,
} from './autodan-lifelong-types';

// Evasion types
export type {
  EvasionTask,
  EvasionSubmitRequest,
  EvasionSubmitResponse,
  EvasionStatus,
  EvasionResult,
  EvasionVariation,
  EvasionAnalysis,
} from './evasion-types';

// Provider management types
export type {
  ProviderStatus,
  RateLimitInfo,
  FallbackSuggestion,
  CircuitBreakerState,
  ProviderHealth,
  ProviderConfig,
  ModelInfo,
} from './provider-management-types';

// Re-export existing types
export * from './chimera';
export * from './schemas';

// Campaign Analytics types
export {
  CampaignStatusEnum,
  ExecutionStatusEnum,
  ExportFormat,
  TimeGranularity,
  MetricType,
} from './campaign-analytics';

export type {
  // Base campaign types
  CampaignBase,
  CampaignCreate,
  CampaignUpdate,
  // Campaign response types
  CampaignSummary,
  CampaignDetail,
  Campaign,
  // Statistics types
  PercentileStats,
  DistributionStats,
  AttemptCounts,
  CampaignStatistics,
  // Breakdown types
  BreakdownItem,
  TechniqueBreakdown,
  ProviderBreakdown,
  PotencyBreakdown,
  // Time series types
  TimeSeriesDataPoint,
  TelemetryTimeSeries,
  MultiSeriesTimeSeries,
  // Comparison types
  CampaignComparisonItem,
  CampaignComparisonRequest,
  CampaignComparison,
  // Telemetry event types
  TelemetryEventSummary,
  TelemetryEventDetail,
  TelemetryEvent,
  // Export types
  ExportChartOptions,
  ExportDataOptions,
  ExportOptions,
  ExportRequest,
  ExportResponse,
  // Filter types
  CampaignFilterParams,
  TelemetryFilterParams,
  FilterOptions,
  TimeSeriesQuery,
  // Pagination types
  CampaignListRequest,
  CampaignListResponse,
  TelemetryListResponse,
  // Cache types
  CampaignCacheStats,
  // UI state types
  ComparisonSelection,
  ChartExportRef,
  AnalyticsDashboardTab,
  SortConfig,
  DateRange,
  // API response types
  CampaignAnalyticsResponse,
  CampaignAnalyticsError,
} from './campaign-analytics';

// Unified Provider System types (NEW)
export type {
  ProviderInfo as UnifiedProviderInfo,
  ModelInfo as UnifiedModelInfo,
  ProviderStatus as UnifiedProviderStatus,
  ProviderCapability,
  ModelCapability,
  ModelPricing,
  SelectionState,
  FullSelection,
  SelectionScope,
  ProvidersListResponse,
  ModelsListResponse,
  CurrentSelectionResponse,
  SaveSelectionRequest,
  RefreshCatalogResponse,
  CatalogStatsResponse,
  WSMessageType,
  SelectionChangedMessage,
  ProviderStatusMessage,
  ModelValidationMessage,
  WSMessage,
  CascadingProviderModelSelectorProps,
  ProviderStatusBadgeProps,
  ModelCapabilityBadgesProps,
  ConnectionStatus,
  UseUnifiedProviderSelectionReturn,
  CapabilityConfig,
  StatusConfig,
} from './unified-providers';

export { CAPABILITY_CONFIG, STATUS_CONFIG } from './unified-providers';

// Aegis Campaign Telemetry types (Real-Time Dashboard)
export type {
  AegisTelemetryEvent,
  AegisTelemetryEventBase,
  AttackMetrics,
  TechniquePerformance,
  TokenUsage,
  LatencyMetrics,
  PromptEvolution,
  CampaignStartedData,
  CampaignCompletedData,
  CampaignFailedData,
  IterationStartedData,
  IterationCompletedData,
  AttackStartedData,
  AttackCompletedData,
  TechniqueAppliedData,
  CostUpdateData,
  PromptEvolvedData,
  ErrorData,
  HeartbeatData,
  ConnectionAckData,
  LatencyUpdateData,
  CampaignStartedEvent,
  CampaignCompletedEvent,
  CampaignFailedEvent,
  IterationStartedEvent,
  IterationCompletedEvent,
  AttackStartedEvent,
  AttackCompletedEvent,
  TechniqueAppliedEvent,
  CostUpdateEvent,
  PromptEvolvedEvent,
  ErrorEvent,
  HeartbeatEvent,
  ConnectionAckEvent,
  LatencyUpdateEvent,
  TypedAegisTelemetryEvent,
  ClientMessage,
  ClientPingMessage,
  ClientPongMessage,
  ClientGetSummaryMessage,
  ClientUnsubscribeMessage,
  WebSocketConnectionStatus,
  SuccessRateTimeSeries,
  TokenUsageTimeSeries,
  LatencyTimeSeries,
  AegisDashboardState,
} from './aegis-telemetry';

export {
  AegisTelemetryEventType,
  CampaignStatus,
  TechniqueCategory,
  createDefaultAttackMetrics,
  createDefaultTokenUsage,
  createDefaultLatencyMetrics,
  createDefaultDashboardState,
  isCampaignStartedEvent,
  isCampaignCompletedEvent,
  isCampaignFailedEvent,
  isIterationStartedEvent,
  isIterationCompletedEvent,
  isAttackStartedEvent,
  isAttackCompletedEvent,
  isTechniqueAppliedEvent,
  isCostUpdateEvent,
  isPromptEvolvedEvent,
  isErrorEvent,
  isHeartbeatEvent,
  isConnectionAckEvent,
  isLatencyUpdateEvent,
  MAX_EVENT_HISTORY,
  MAX_TIME_SERIES_POINTS,
  HEARTBEAT_INTERVAL_MS,
  MAX_RECONNECT_ATTEMPTS,
  RECONNECT_BASE_DELAY_MS,
  EVENT_TYPE_LABELS,
  CAMPAIGN_STATUS_LABELS,
  TECHNIQUE_CATEGORY_LABELS,
  TECHNIQUE_CATEGORY_COLORS,
} from './aegis-telemetry';

// Prompt Library types
export type {
  TemplateMetadata,
  TemplateVersion,
  TemplateRating,
  RatingStatistics,
  PromptTemplate,
  TemplateSearchFilters,
  TemplateSearchRequest,
  CreateTemplateRequest,
  UpdateTemplateRequest,
  SearchTemplatesRequest,
  RateTemplateRequest,
  UpdateRatingRequest,
  CreateVersionRequest,
  SaveFromCampaignRequest,
  TemplateResponse,
  TemplateListResponse,
  TemplateVersionResponse,
  TemplateVersionListResponse,
  RatingResponse,
  RatingListResponse,
  RatingStatisticsResponse,
  TopRatedTemplatesResponse,
  TemplateDeleteResponse,
  TemplateStatsResponse,
  VersionDiff,
  VersionChange,
  VersionComparisonResponse,
  VersionTimelineEntry,
  VersionTimelineResponse,
  TechniqueTypeOption,
  VulnerabilityTypeOption,
  TemplateSummary,
} from './prompt-library-types';

export {
  TechniqueType,
  VulnerabilityType,
  SharingLevel,
  TemplateStatus,
  TemplateSortField,
  SortOrder,
  TECHNIQUE_TYPE_CATEGORIES,
  VULNERABILITY_TYPE_CATEGORIES,
  formatTechniqueType,
  formatVulnerabilityType,
  formatSharingLevel,
  formatTemplateStatus,
  DEFAULT_TEMPLATE_METADATA,
  DEFAULT_RATING_STATISTICS,
} from './prompt-library-types';
