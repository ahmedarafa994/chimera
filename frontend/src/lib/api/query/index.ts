/**
 * TanStack Query API Module
 * Centralized export of all query hooks and utilities
 */

// Query client configuration
export {
  queryKeys,
  STALE_TIMES,
  createQueryClient,
  getQueryClient,
} from "./query-client";

// Provider queries
export {
  useProviders,
  useActiveProvider,
  useProviderConfig,
  useProviderHealth,
  useAllProvidersHealth,
  useSetActiveProvider,
  useUpdateProviderConfig,
  useTestProviderConnection,
} from "./providers-queries";

// AutoDAN queries
export {
  useStrategies,
  useStrategySearch,
  useLibraryStats,
  useProgress,
  useAutodanHealth,
  useCreateStrategy,
  useDeleteStrategy,
  useAttack,
  useWarmup,
  useLifelong,
  useExportLibrary,
  useImportLibrary,
  useSaveLibrary,
  useClearLibrary,
  useResetEngine,
} from "./autodan-queries";

// Session queries
export {
  useSession,
  useSessionStats,
  useCreateSession,
  useUpdateSessionModel,
  useDeleteSession,
  useCurrentSession,
  getStoredSessionId,
  storeSessionId,
  clearStoredSessionId,
} from "./session-queries";
export type {
  CreateSessionRequest,
  UpdateSessionModelRequest,
  UpdateSessionModelResponse,
} from "./session-queries";

// System queries
export {
  useHealth,
  useMetrics,
  useConnectionStatus,
  useModels,
  useModelsByProvider,
  useModelsHealth,
  useTestConnection,
  useValidateModel,
  usePrefetchSystemData,
  useIsBackendConnected,
} from "./system-queries";
export type {
  HealthResponse,
  ModelValidationRequest,
  ModelValidationResponse,
} from "./system-queries";

// Prompt Library queries
export {
  // Query keys
  promptLibraryKeys,
  // Query hooks
  usePromptTemplates,
  usePromptTemplate,
  useSearchTemplates,
  useInfiniteSearchTemplates,
  useTopRatedTemplates,
  useLibraryStatistics,
  useTemplateVersions,
  useTemplateVersion,
  useTemplateRatings,
  useRatingStatistics,
  // Mutation hooks
  useCreateTemplate,
  useUpdateTemplate,
  useDeleteTemplate,
  useRateTemplate,
  useUpdateRating,
  useDeleteRating,
  useCreateVersion,
  useRestoreVersion,
  useSaveFromCampaign,
  // Utility hooks
  usePrefetchTemplate,
  usePrefetchTemplateVersions,
  useInvalidatePromptLibrary,
} from "./prompt-library";

// Campaign Analytics queries
export {
  // Query key factory
  campaignQueryKeys,
  // Query hooks
  useCampaigns,
  useCampaign,
  useCampaignSummary,
  useCampaignStatistics,
  useTechniqueBreakdown,
  useProviderBreakdown,
  usePotencyBreakdown,
  useCampaignTimeSeries,
  useCampaignComparison,
  useTelemetryEvents,
  useTelemetryEventDetail,
  useCampaignCacheStats,
  // Mutation hooks
  useCampaignExport,
  useInvalidateCampaignCache,
  // Prefetch utilities
  usePrefetchCampaign,
  usePrefetchCampaignList,
  // Cache invalidation utilities
  useInvalidateAllCampaigns,
  useInvalidateCampaign,
  // Utility functions
  getCampaignCSVExportUrl,
} from "./campaign-queries";
export type {
  CampaignListParams,
  TimeSeriesParams,
  TelemetryEventListParams,
  ExportCSVParams,
} from "./campaign-queries";
