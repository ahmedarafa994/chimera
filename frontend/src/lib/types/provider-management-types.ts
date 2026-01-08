/**
 * Provider Management Types for Project Chimera Frontend
 * 
 * Types for provider management endpoints including:
 * - Provider listing and selection
 * - Model discovery
 * - Rate limiting
 * - Health monitoring
 * - WebSocket sync
 */

// =============================================================================
// Enums
// =============================================================================

export enum ProviderStatus {
  AVAILABLE = "available",
  DEGRADED = "degraded",
  UNAVAILABLE = "unavailable",
  RATE_LIMITED = "rate_limited",
  UNKNOWN = "unknown",
}

export enum CircuitBreakerState {
  CLOSED = "closed",
  OPEN = "open",
  HALF_OPEN = "half_open",
}

// =============================================================================
// Provider Types
// =============================================================================

export interface ProviderInfo {
  /** Provider identifier */
  provider_id: string;
  /** Display name */
  name: string;
  /** Provider description */
  description?: string;
  /** Current status */
  status: ProviderStatus;
  /** Whether provider is enabled */
  is_enabled: boolean;
  /** Supported model types */
  supported_model_types: string[];
  /** Base URL (if applicable) */
  base_url?: string;
  /** API version */
  api_version?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

export interface AvailableProvidersResponse {
  /** List of available providers */
  providers: ProviderInfo[];
  /** Total count */
  total_count: number;
  /** Timestamp */
  timestamp: string;
}

// =============================================================================
// Model Types
// =============================================================================

export interface ModelInfo {
  /** Model identifier */
  model_id: string;
  /** Display name */
  name: string;
  /** Model description */
  description?: string;
  /** Provider ID */
  provider_id: string;
  /** Model type (chat, completion, embedding, etc.) */
  model_type: string;
  /** Context window size */
  context_window?: number;
  /** Maximum output tokens */
  max_output_tokens?: number;
  /** Whether model supports streaming */
  supports_streaming: boolean;
  /** Whether model supports function calling */
  supports_function_calling: boolean;
  /** Pricing info */
  pricing?: ModelPricing;
  /** Additional capabilities */
  capabilities?: string[];
  /** Model metadata */
  metadata?: Record<string, unknown>;
}

export interface ModelPricing {
  /** Input price per 1K tokens */
  input_per_1k_tokens?: number;
  /** Output price per 1K tokens */
  output_per_1k_tokens?: number;
  /** Currency */
  currency: string;
}

export interface ProviderModelsResponse {
  /** Provider ID */
  provider_id: string;
  /** List of models */
  models: ModelInfo[];
  /** Total count */
  total_count: number;
  /** Last updated */
  last_updated: string;
}

// =============================================================================
// Selection Types
// =============================================================================

export interface ModelSelectionRequest {
  /** Provider ID */
  provider_id: string;
  /** Model ID */
  model_id: string;
  /** Session ID (optional, auto-generated if not provided) */
  session_id?: string;
  /** Selection reason/context */
  reason?: string;
}

export interface ModelSelectionResponse {
  /** Whether selection was successful */
  success: boolean;
  /** Session ID */
  session_id: string;
  /** Selected provider */
  provider_id: string;
  /** Selected model */
  model_id: string;
  /** Selection timestamp */
  selected_at: string;
  /** Message */
  message?: string;
}

export interface CurrentSelectionResponse {
  /** Whether there's an active selection */
  has_selection: boolean;
  /** Session ID */
  session_id?: string;
  /** Current provider */
  provider_id?: string;
  /** Current model */
  model_id?: string;
  /** Selection timestamp */
  selected_at?: string;
  /** Provider info */
  provider_info?: ProviderInfo;
  /** Model info */
  model_info?: ModelInfo;
}

// =============================================================================
// Rate Limiting Types
// =============================================================================

export interface RateLimitInfo {
  /** Provider ID */
  provider_id: string;
  /** Model ID */
  model_id: string;
  /** Whether currently rate limited */
  is_rate_limited: boolean;
  /** Requests remaining */
  requests_remaining?: number;
  /** Tokens remaining */
  tokens_remaining?: number;
  /** Reset timestamp */
  reset_at?: string;
  /** Retry after (seconds) */
  retry_after_seconds?: number;
  /** Rate limit tier */
  tier?: string;
}

export interface RateLimitResponse {
  /** Rate limit info */
  rate_limit: RateLimitInfo;
  /** Fallback suggestions */
  fallback_suggestions?: FallbackSuggestion[];
}

export interface FallbackSuggestion {
  /** Suggested provider */
  provider_id: string;
  /** Suggested model */
  model_id: string;
  /** Reason for suggestion */
  reason: string;
  /** Estimated availability */
  estimated_availability: string;
  /** Compatibility score (0-1) */
  compatibility_score: number;
}

// =============================================================================
// Health Types
// =============================================================================

export interface ProviderHealthStatus {
  /** Provider ID */
  provider_id: string;
  /** Overall health status */
  status: ProviderStatus;
  /** Circuit breaker state */
  circuit_breaker_state: CircuitBreakerState;
  /** Last successful request */
  last_success?: string;
  /** Last failure */
  last_failure?: string;
  /** Failure count */
  failure_count: number;
  /** Success rate (0-1) */
  success_rate: number;
  /** Average latency (ms) */
  avg_latency_ms?: number;
  /** Health check timestamp */
  checked_at: string;
}

export interface AllProvidersHealthResponse {
  /** Health status for all providers */
  providers: ProviderHealthStatus[];
  /** Overall system health */
  system_health: "healthy" | "degraded" | "unhealthy";
  /** Timestamp */
  timestamp: string;
}

// =============================================================================
// WebSocket Types
// =============================================================================

export interface ProviderSelectionSyncMessage {
  /** Message type */
  type: "selection_update" | "selection_cleared" | "sync_request" | "sync_response";
  /** Session ID */
  session_id: string;
  /** Provider ID */
  provider_id?: string;
  /** Model ID */
  model_id?: string;
  /** Timestamp */
  timestamp: string;
  /** Additional data */
  data?: Record<string, unknown>;
}

export interface ModelUpdateMessage {
  /** Message type */
  type: "model_added" | "model_removed" | "model_updated" | "provider_status_changed";
  /** Provider ID */
  provider_id: string;
  /** Model ID (if applicable) */
  model_id?: string;
  /** Update data */
  data: Record<string, unknown>;
  /** Timestamp */
  timestamp: string;
}

// =============================================================================
// UI State Types
// =============================================================================

export interface ProviderManagementState {
  /** Available providers */
  providers: ProviderInfo[];
  /** Models by provider */
  modelsByProvider: Record<string, ModelInfo[]>;
  /** Current selection */
  currentSelection: CurrentSelectionResponse | null;
  /** Health status */
  healthStatus: AllProvidersHealthResponse | null;
  /** Rate limit info */
  rateLimitInfo: RateLimitInfo | null;
  /** Loading states */
  loading: {
    providers: boolean;
    models: boolean;
    selection: boolean;
    health: boolean;
  };
  /** Error states */
  errors: {
    providers: string | null;
    models: string | null;
    selection: string | null;
    health: string | null;
  };
  /** WebSocket connection status */
  wsConnected: boolean;
}

// =============================================================================
// Session Management
// =============================================================================

export interface SessionConfig {
  /** Session ID */
  sessionId: string;
  /** Session creation time */
  createdAt: string;
  /** Session expiry time */
  expiresAt?: string;
  /** Auto-refresh enabled */
  autoRefresh: boolean;
}

export const SESSION_HEADER = "X-Session-ID";
export const SESSION_COOKIE = "chimera_session_id";

/**
 * Generate a new session ID
 */
export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}

/**
 * Get session ID from storage or generate new one
 */
export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") {
    return generateSessionId();
  }
  
  let sessionId: string | null = localStorage.getItem(SESSION_COOKIE);
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem(SESSION_COOKIE, sessionId);
  } else {
    // Handle case where session ID was stored as JSON-stringified value
    if (sessionId.startsWith('"') && sessionId.endsWith('"')) {
      try {
        const parsed = JSON.parse(sessionId) as string;
        sessionId = parsed;
        // Re-save without quotes
        localStorage.setItem(SESSION_COOKIE, sessionId);
      } catch {
        // If parsing fails, use as-is
      }
    }
  }
  return sessionId;
}