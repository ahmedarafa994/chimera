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
