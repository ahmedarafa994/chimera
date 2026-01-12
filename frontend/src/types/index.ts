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
