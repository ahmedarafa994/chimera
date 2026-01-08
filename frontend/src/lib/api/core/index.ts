/**
 * API Core Module
 *
 * Central export point for all core API functionality.
 *
 * @module lib/api/core
 */

// Configuration
export {
  configManager,
  getApiConfig,
  updateApiConfig,
  getActiveApiUrl,
  getActiveApiKey,
  getCurrentApiMode,
  getCurrentEnvironment,
  getApiHeaders,
  ENDPOINTS,
  type ApiClientConfig as APIConfig,
  type ProviderConfig,
  type AIProvider as APIProvider,
  type ApiMode as APIMode,
  type Environment,
  type EndpointConfig,
  type RequestHeaders,
} from './config';

// Errors
export {
  APIError,
  NetworkError,
  TimeoutError,
  AuthenticationError,
  AuthorizationError,
  ValidationError,
  NotFoundError,
  RateLimitError,
  ServiceUnavailableError,
  LLMProviderError,
  mapBackendError,
  isRetryableError,
  isAPIError,
} from '../../errors';

// Retry Logic
export {
  withRetry,
  calculateBackoffDelay,
  DEFAULT_RETRY_CONFIG,
  type RetryConfig,
  type RetryResult,
} from '../../resilience/retry';

// Circuit Breaker
export {
  CircuitBreaker,
  CircuitBreakerRegistry,
  circuitBreakerRegistry,
  withCircuitBreaker,
  CircuitState,
  type CircuitBreakerConfig,
  type CircuitBreakerStats,
} from '../../resilience/circuit-breaker';

// Request Deduplication
export {
  RequestDeduplicator,
  RequestBatcher,
  requestDeduplicator,
  debouncedRequest,
  throttledRequest,
  deduplicated,
  generateRequestKey,
  type BatchConfig,
  type BatchResult,
  type DeduplicationConfig,
} from './request-deduplication';

// Monitoring
export {
  metricsCollector,
  healthCheckMonitor,
  apiLogger,
  trackRequest,
  getAPIHealth,
  exportMetrics,
  generateRequestId,
  type RequestMetrics,
  type AggregatedMetrics,
  type HealthCheckResult,
  type AlertThreshold,
  type MonitoringConfig,
  type LogLevel,
} from './monitoring';

// Authentication
export {
  tokenManager,
  apiKeyManager,
  requestSigner,
  authHeaderBuilder,
  setAuthToken,
  getAuthToken,
  clearAuth,
  isAuthenticated,
  setProviderAPIKey,
  getProviderAPIKey,
  configureTokenRefresh,
  configureRequestSigning,
  buildAuthHeaders,
  type TokenInfo,
  type APIKeyInfo,
  type AuthHeaders,
  type RefreshTokenResult,
  type TokenRefreshFn,
  type AuthConfig,
} from './auth';

// API Client
export {
  APIClient,
  apiClient,
  get,
  post,
  put,
  patch,
  del,
  healthCheck,
  getModels,
  getProviders,
  generateJailbreak,
  chatCompletion,
  getTechniques,
  getMetrics,
  type RequestOptions,
  type APIClientConfig,
} from './client';

// Types
export * from './types';
