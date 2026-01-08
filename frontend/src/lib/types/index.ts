/**
 * Types Index for Project Chimera Frontend
 *
 * Central export point for all type definitions
 */

// Admin Types
export * from "./admin-types";

// AutoDAN Lifelong Types (legacy)
export * from "./autodan-lifelong-types";

// AutoDAN-Turbo Types (ICLR 2025 paper implementation)
// Note: Some types are duplicated in autodan-lifelong-types, so we selectively export
export {
  // Enums
  StrategySource,
  LifelongPhase,
  // Note: LifelongTaskStatus is already exported from autodan-lifelong-types
} from "./autodan-turbo-types";

// Export types using 'export type' for isolatedModules compatibility
export type {
  // Strategy Types
  StrategyMetadata,
  JailbreakStrategy,
  StrategySearchResult,
  
  // Request Types
  SingleAttackRequest,
  WarmupRequest,
  // Note: LifelongLoopRequest is already exported from autodan-lifelong-types
  TestStageRequest,
  StrategyCreateRequest,
  StrategySearchRequest,
  ScoreRequest,
  ExtractStrategyRequest,
  ImportLibraryRequest,
  
  // Response Types
  SingleAttackResponse,
  AttackResultItem,
  WarmupResponse,
  LifelongProgress,
  // Note: LifelongLoopResponse is already exported from autodan-lifelong-types
  TestStageResponse,
  StrategyListResponse,
  StrategySearchResponse,
  BatchInjectResponse,
  ProgressResponse,
  TopStrategyInfo,
  LibraryStatsResponse,
  ScoreResponse,
  ExtractStrategyResponse,
  ResetResponse,
  SaveLibraryResponse,
  ClearLibraryResponse,
  HealthResponse,
  ExportedStrategy,
  ExportedLibrary,
  ImportLibraryResponse,
  DeleteStrategyResponse,
  
  // UI State Types
  AttackPanelState,
  LifelongDashboardState,
  StrategyLibraryState,
  LifelongConfig,
  TestConfig,
  AttackHistoryItem,
  ProgressChartPoint,
  StrategyPerformanceData,
  
  // Form Types
  CreateStrategyFormData,
  LifelongFormData,
  TestFormData,
  
  // Utility Types
  ApiResponse,
  ApiError,
  PaginationParams,
  StrategySortParams,
  StrategyFilterParams,
  StrategyQueryParams,
  
  // UI-specific Request Types
  WarmupRequestUI,
  LifelongRequest,
  TestRequest,
  
  // Legacy Attack Types
  AttackRequest,
  AttackResponse,
} from "./autodan-turbo-types";

// Evasion Types
export * from "./evasion-types";

// Provider Management Types
export * from "./provider-management-types";