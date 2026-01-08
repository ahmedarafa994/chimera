/**
 * API Module Index
 * Central export for all API-related functionality
 */

// Core client
export { apiClient } from './client';

// Managers
export { authManager } from './auth-manager';
export { apiCache } from './cache-manager';
export { wsManager, createStream, StreamConnection } from './websocket-manager';

// Utilities
export { logger } from './logger';
export { retryWithBackoff, createRetryStrategy, RetryStrategy } from './retry-strategy';
export {
  errorHandler,
  formatValidationErrors,
  isNetworkError,
  getRetryDelay
} from './error-handler';

// Services
export { providersApi, sessionsApi } from './services';

// Stores
export { useProvidersStore, useSessionStore } from './stores';

// Hooks
export {
  useApi,
  useMutation,
  useQuery,
  usePaginatedQuery,
  useOptimisticMutation,
  useWebSocket,
  useStream,
  useModelSelectionSync,
} from './hooks';

// Validation
export {
  // Schemas
  promptRequestSchema,
  jailbreakRequestSchema,
  enhancedJailbreakRequestSchema,
  autodanConfigSchema,
  createSessionRequestSchema,
  sendMessageRequestSchema,
  loginRequestSchema,
  transformationRequestSchema,
  // Helpers
  validateRequest,
  getValidationErrors,
} from './validation';

// Types
export type {
  // Base types
  ApiResponse,
  ApiError,
  PaginatedResponse,
  // Provider types
  Provider,
  ProviderModel,
  // Prompt types
  PromptRequest,
  PromptResponse,
  // Jailbreak types
  JailbreakRequest,
  JailbreakResponse,
  JailbreakTechnique,
  // AutoDAN types
  AutoDANConfig,
  AutoDANRequest,
  AutoDANResponse,
  // Session types
  Session,
  Message,
  // WebSocket types
  WebSocketMessage,
  StreamEvent,
  // Model types
  ModelSelectionRequest,
  ModelSelectionResponse,
  // Transformation types
  TransformationRequest,
  TransformationResponse,
  TransformationType,
  // Auth types
  AuthTokens,
  User,
  // Health types
  HealthStatus,
  ServiceStatus,
} from './types';

// Hook types
export type {
  UseApiState,
  UseApiOptions,
  UseApiResult,
  UseWebSocketOptions,
  UseWebSocketResult,
  UseStreamOptions,
  UseStreamResult,
} from './hooks';

// WebSocket types
export type {
  ConnectionState,
  WebSocketConfig,
  MessageHandler,
  StreamHandler,
  StateHandler
} from './websocket-manager';

// Service types
export type {
  ProvidersResponse,
  ProviderModelsResponse,
  SetDefaultProviderRequest,
  CreateSessionRequest,
  SendMessageRequest,
  UpdateSessionRequest,
} from './services';
