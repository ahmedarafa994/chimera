/**
 * API Services Index - Aligned with Backend
 *
 * This module exports both new aligned services and existing services.
 * Gradually migrate from existing services to the new aligned ones.
 */

// ============================================================================
// NEW ALIGNED SERVICES (Recommended)
// ============================================================================

// Import all aligned services
export { providerService, providerApi } from './provider-service';
export { generationService, generationApi } from './generation-service';
export { jailbreakService, jailbreakApi } from './jailbreak-service';
export { healthMonitoringService, healthApi } from './health-service';

// Unified API client
export { default as chimeraApi } from './index';

// ============================================================================
// EXISTING SERVICES (Legacy - Migrate to aligned versions)
// ============================================================================

export { providersApi } from './providers';
export { sessionsApi } from './sessions';
export { chatService } from './chat-service';
export { autoadvService } from './autoadv-service';
export { autodanLifelongService } from './autodan-lifelong-service';
export { evasionService } from './evasion-service';
export { providerManagementService } from './provider-management-service';
export { adminService } from './admin-service';
export * from './jobs';
export { assessmentsService } from './assessments';
export { transformationsService } from './transformations';
export { reportsService } from './reports';
export { workspaceService } from './workspaces';
export { defenseEngineService } from './defense-engine';
export { documentationService } from './documentation';
export { researchLabService } from './research-lab';
export { attackSessionService } from './attack-sessions';
export { techniqueLibraryService } from './technique-library';
export { techniqueBuilderService } from './technique-builder';
export { scheduledTestingService } from './scheduled-testing';
export { multimodalTestingService } from './multimodal';

// ============================================================================
// TYPE EXPORTS
// ============================================================================

// New aligned types
export type {
  // Provider types
  ProviderInfo,
  ProviderModel,
  ProvidersListResponse,
  ProviderModelsResponse,
  SelectProviderRequest,
  SelectProviderResponse,
  CurrentSelectionResponse,
  ProviderHealthResponse,
  RateLimitInfoResponse,
} from './provider-service';

export type {
  // Generation types
  GenerationConfig,
  PromptRequest,
  PromptResponse,
  GenerationHealthResponse,
} from './generation-service';

export type {
  // Jailbreak types
  JailbreakRequest,
  JailbreakResponse,
  JailbreakMetadata,
  AutoDANRequest,
  AutoDANResponse,
  FuzzRequest,
  FuzzSession,
  FuzzResult,
  GradientOptimizationRequest,
  GradientOptimizationResponse,
  TechniqueInfo,
  TechniquesResponse,
} from './jailbreak-service';

export type {
  // Health types
  BasicHealthResponse,
  ReadinessResponse,
  FullHealthResponse,
  IntegrationHealthResponse,
  MetricsResponse,
  IntegrationStatsResponse,
} from './health-service';

// Legacy types (for backward compatibility)
export type { ProvidersResponse, SetDefaultProviderRequest } from './providers';
export type { CreateSessionRequest, SendMessageRequest, UpdateSessionRequest } from './sessions';
export type { ChatCompletionRequest, ChatCompletionResponse, StreamChunk } from './chat-service';
export type { EvasionTask, EvasionSubmitRequest, EvasionResult } from '@/types/evasion-types';

// ============================================================================
// MIGRATION HELPERS
// ============================================================================

/**
 * Migration guide from old API to new aligned API
 */
export const MIGRATION_GUIDE = {
  // Provider management
  'enhancedApi.providers.list()': 'providerApi.getProviders()',
  'enhancedApi.providerConfig.getActiveProvider()': 'providerApi.getCurrentSelection()',
  'enhancedApi.providerConfig.setActiveProvider()': 'providerApi.selectProvider()',

  // Generation
  'enhancedApi.generate()': 'generationApi.generate()',
  'enhancedApi.text()': 'generationApi.generateText()',

  // Jailbreak
  'enhancedApi.jailbreak()': 'jailbreakApi.jailbreak()',
  'enhancedApi.autodan()': 'jailbreakApi.autodan()',
  'enhancedApi.gptfuzz()': 'jailbreakApi.startFuzzSession()',

  // Health
  'enhancedApi.health()': 'healthApi.getBasicHealth()',
  'enhancedApi.metrics()': 'healthApi.getMetrics()',

  // Chat (already aligned)
  'chatService.createChatCompletion()': 'chatService.createChatCompletion()',
} as const;

/**
 * Constants for service status
 */
export const SERVICE_STATUS = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  UNHEALTHY: 'unhealthy',
  UNKNOWN: 'unknown',
} as const;

export const PROVIDER_STATUS = {
  READY: 'ready',
  ERROR: 'error',
  CONNECTING: 'connecting',
  DISCONNECTED: 'disconnected',
} as const;
