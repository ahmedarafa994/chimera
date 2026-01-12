/**
 * Hooks Index for Project Chimera Frontend
 *
 * Central export point for all custom React hooks.
 */

// Admin hooks
export {
  useAdminAuth,
  useFeatureFlags,
  useTenants,
  useUsageAnalytics,
  type UseAdminAuthReturn,
  type UseFeatureFlagsReturn,
  type UseTenantsReturn,
  type UseUsageAnalyticsReturn,
} from "./use-admin";

// AutoAdv hooks
export {
  useAutoAdv,
  useAutoAdvWebSocket,
  useAutoAdvForm,
  useAutoAdvWithProgress,
  type UseAutoAdvReturn,
  type UseAutoAdvWebSocketReturn,
  type UseAutoAdvFormReturn,
  type UseAutoAdvWithProgressReturn,
  type AutoAdvHistoryItem,
} from "./use-autoadv";

// Chat hooks
export {
  useChat,
  useCompletion,
  useStreamingCompletion,
  type UseChatReturn,
  type UseCompletionReturn,
  type UseStreamingCompletionReturn,
  type ChatMessageWithMeta,
  type SendMessageOptions,
  type CompletionOptions,
} from "./use-chat";

// Evasion hooks
export {
  useEvasionStrategies,
  useEvasionTask,
  useEvasionTaskList,
  useEvasionForm,
  type UseEvasionStrategiesReturn,
  type UseEvasionTaskReturn,
  type UseEvasionTaskListReturn,
  type UseEvasionFormReturn,
  type EvasionTaskWithMeta,
} from "./use-evasion";

// Provider management hooks
export {
  useProviders,
  useModels,
  useModelSelection,
  useProviderHealth,
  useProviderSync,
  useModelUpdates,
  type UseProvidersReturn,
  type UseModelsReturn,
  type UseModelSelectionReturn,
  type UseProviderHealthReturn,
  type UseProviderSyncReturn,
  type UseModelUpdatesReturn,
} from "./use-provider-management";

// Enhanced model selection hook (self-contained with WebSocket, optimistic updates, rate limiting)
export {
  useEnhancedModelSelection,
  type Provider as EnhancedProvider,
  type Model as EnhancedModel,
  type ModelSelection,
  type UseEnhancedModelSelectionOptions,
  type UseEnhancedModelSelectionReturn,
} from "./use-enhanced-model-selection";

// Activity feed hooks
export {
  useActivityFeed,
  type UseActivityFeedOptions,
} from "./use-activity-feed";

// Unified Provider System (NEW - Phase 2)
export {
  useUnifiedProviders,
  useUnifiedModels,
  useAllUnifiedModels,
  useCurrentSelection,
  useSaveSelection,
  useClearSelection,
  type UnifiedProvider,
  type UnifiedModel,
  type CurrentSelection,
  type SaveSelectionRequest,
  type SelectionScope,
  type UnifiedProvidersResponse,
  type UnifiedModelsResponse,
} from "@/lib/api/query/unified-provider-queries";

export {
  useModelSelectionSync,
  type UseModelSelectionSyncOptions,
  type UseModelSelectionSyncReturn,
  type SelectionChangedData,
  type ProviderStatusData,
  type ModelValidationData,
  type WebSocketMessageType,
  type WebSocketMessage,
} from "./useModelSelectionSync";

// Unified Provider Selection Hook (NEW - Cascading Provider/Model Selection)
export {
  useUnifiedProviderSelection,
  UNIFIED_PROVIDER_QUERY_KEYS,
  unifiedProviderQueryKeys,
} from "./useUnifiedProviderSelection";

// Streaming Generation Hook (NEW - Unified Streaming with Selection)
export {
  useStreamingGeneration,
  type StreamChunk,
  type StreamResult,
  type StreamError,
  type StreamingOptions,
  type GenerateParams,
  type ChatParams,
  type ChatMessage as StreamChatMessage,
  type StreamState,
  type StreamMetadata,
  type UseStreamingGenerationReturn,
} from "./useStreamingGeneration";

// WebSocket Streaming Hook (NEW - Real-time WebSocket streaming)
export {
  useWebSocketStreaming,
  type WebSocketStreamingOptions,
  type WebSocketStreamChunk,
  type WebSocketStreamResult,
  type UseWebSocketStreamingReturn,
} from "./useWebSocketStreaming";

// Re-export types from unified providers
export type {
  ProviderInfo,
  ModelInfo,
  ProviderStatus,
  ModelCapability,
  ConnectionStatus,
  UseUnifiedProviderSelectionReturn,
} from "@/types/unified-providers";

// API Key Management Hook (NEW - Secure API Key Storage)
export { useApiKeys, default as useApiKeysDefault } from "./useApiKeys";

// Provider Health Dashboard Hooks (NEW - Real-time Health Monitoring)
export {
  useProviderHealthDashboard,
  useHealthHistory,
  useProviderHealthDetail,
  default as useProviderHealthDashboardDefault,
  type UseProviderHealthOptions,
  type UseProviderHealthResult,
  type UseHealthHistoryOptions,
  type UseHealthHistoryResult,
  type ProviderHealthDetail,
  type UseProviderHealthDetailResult,
} from "./useProviderHealth";
