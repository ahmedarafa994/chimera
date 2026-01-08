/**
 * TypeScript types for the Unified Provider System
 *
 * These types align with the backend API endpoints at:
 * - GET /api/v1/unified-providers/providers
 * - GET /api/v1/unified-providers/providers/{id}/models
 * - POST /api/v1/unified-providers/selection
 *
 * @module unified-providers
 */

// =============================================================================
// Provider Types
// =============================================================================

/**
 * Provider health status
 */
export type ProviderStatus = "healthy" | "degraded" | "unhealthy" | "unknown";

/**
 * Provider capability types
 */
export type ProviderCapability =
  | "chat"
  | "completion"
  | "embeddings"
  | "vision"
  | "function_calling"
  | "streaming"
  | "code"
  | "reasoning";

/**
 * Provider information from the unified API
 */
export interface ProviderInfo {
  /** Unique provider identifier (e.g., "openai", "anthropic") */
  provider_id: string;
  /** Human-readable provider name */
  display_name: string;
  /** Current health status */
  status: ProviderStatus;
  /** Whether the provider is enabled in config */
  is_enabled: boolean;
  /** Whether an API key is configured */
  has_api_key: boolean;
  /** Whether the provider is ready to use */
  is_available: boolean;
  /** Number of available models */
  model_count: number;
  /** List of provider capabilities */
  capabilities: ProviderCapability[];
  /** Default model ID for this provider */
  default_model?: string;
  /** Health score (0-1) for degraded providers */
  health_score?: number;
  /** Optional description */
  description?: string;
}

// =============================================================================
// Model Types
// =============================================================================

/**
 * Model capability types
 */
export type ModelCapability =
  | "streaming"
  | "vision"
  | "function_calling"
  | "json_mode"
  | "code"
  | "reasoning"
  | "embeddings"
  | "tool_use";

/**
 * Model pricing information
 */
export interface ModelPricing {
  /** Price per 1K input tokens in USD */
  input: number;
  /** Price per 1K output tokens in USD */
  output: number;
}

/**
 * Model information from the unified API
 */
export interface ModelInfo {
  /** Unique model identifier */
  model_id: string;
  /** Human-readable model name */
  name: string;
  /** Provider this model belongs to */
  provider_id: string;
  /** Maximum context length in tokens */
  context_length: number;
  /** Maximum output tokens */
  max_output_tokens?: number;
  /** List of model capabilities */
  capabilities: ModelCapability[];
  /** Pricing information */
  pricing?: ModelPricing;
  /** Whether this is the default model for the provider */
  is_default: boolean;
  /** Whether the model is currently available */
  is_available: boolean;
  /** Whether the model supports streaming */
  supports_streaming: boolean;
  /** Whether the model supports vision/images */
  supports_vision: boolean;
  /** Whether the model supports function calling */
  supports_function_calling: boolean;
  /** Optional description */
  description?: string;
}

// =============================================================================
// Selection Types
// =============================================================================

/**
 * Selection scope indicating where the selection came from
 */
export type SelectionScope = "REQUEST" | "SESSION" | "GLOBAL";

/**
 * Current selection state
 */
export interface SelectionState {
  /** Selected provider ID */
  provider: string | null;
  /** Selected model ID */
  model: string | null;
  /** Unix timestamp of last update */
  timestamp: number;
}

/**
 * Full selection with metadata
 */
export interface FullSelection extends SelectionState {
  /** Source of this selection */
  scope: SelectionScope;
  /** Session ID if session-scoped */
  session_id?: string;
  /** User ID if user-scoped */
  user_id?: string;
}

// =============================================================================
// API Response Types
// =============================================================================

/**
 * Response from GET /api/v1/unified-providers/providers
 */
export interface ProvidersListResponse {
  providers: ProviderInfo[];
  total: number;
}

/**
 * Response from GET /api/v1/unified-providers/providers/{id}/models
 */
export interface ModelsListResponse {
  models: ModelInfo[];
  total: number;
  provider_id: string;
}

/**
 * Response from GET /api/v1/unified-providers/selection
 */
export interface CurrentSelectionResponse {
  provider_id: string;
  model_id: string;
  scope: SelectionScope;
  session_id?: string;
  user_id?: string;
  created_at?: string;
  updated_at?: string;
  /** Version for optimistic concurrency control */
  version?: number;
  /** Selection metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Request body for POST /api/v1/unified-providers/selection
 */
export interface SaveSelectionRequest {
  provider_id: string;
  model_id: string;
  session_id?: string;
}

/**
 * Response from POST /api/v1/unified-providers/refresh
 */
export interface RefreshCatalogResponse {
  success: boolean;
  message: string;
  providers_refreshed: string[];
  total_models: number;
}

/**
 * Response from GET /api/v1/unified-providers/catalog/stats
 */
export interface CatalogStatsResponse {
  total_providers: number;
  total_models: number;
  providers_with_models: Record<string, number>;
  last_refresh: string | null;
  cache_ttl_seconds: number;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

/**
 * WebSocket message types for real-time updates
 */
export type WSMessageType =
  | "SELECTION_CHANGED"
  | "PROVIDER_STATUS"
  | "MODEL_VALIDATION"
  | "PING"
  | "PONG"
  | "ERROR"
  // New message types for optimistic concurrency
  | "current_selection"
  | "selection_changed"
  | "update_result";

/**
 * Selection changed WebSocket message (legacy format)
 */
export interface SelectionChangedMessage {
  type: "SELECTION_CHANGED";
  data: {
    provider_id: string;
    model_id: string;
    scope: SelectionScope;
    session_id?: string;
    user_id?: string;
    /** Version for optimistic concurrency */
    version?: number;
  };
  timestamp: string;
}

/**
 * Current selection WebSocket message (new format)
 */
export interface CurrentSelectionWSMessage {
  type: "current_selection";
  session_id: string;
  provider: string;
  model: string;
  version: number;
  timestamp: string;
}

/**
 * Selection changed WebSocket message (new format)
 */
export interface SelectionChangedWSMessage {
  type: "selection_changed";
  session_id: string;
  provider: string;
  model: string;
  version: number;
  timestamp: string;
}

/**
 * Update result WebSocket message
 */
export interface UpdateResultWSMessage {
  type: "update_result";
  success: boolean;
  provider: string;
  model: string;
  version: number;
  conflict: boolean;
  timestamp: string;
}

/**
 * Provider status WebSocket message
 */
export interface ProviderStatusMessage {
  type: "PROVIDER_STATUS";
  data: {
    provider_id: string;
    status: ProviderStatus;
    is_available: boolean;
    health_score?: number;
    message?: string;
  };
  timestamp: string;
}

/**
 * Model validation WebSocket message
 */
export interface ModelValidationMessage {
  type: "MODEL_VALIDATION";
  data: {
    provider_id: string;
    model_id: string;
    is_valid: boolean;
    errors?: string[];
  };
  timestamp: string;
}

/**
 * Union type for all WebSocket messages
 */
export type WSMessage =
  | SelectionChangedMessage
  | ProviderStatusMessage
  | ModelValidationMessage
  | CurrentSelectionWSMessage
  | SelectionChangedWSMessage
  | UpdateResultWSMessage
  | { type: "PING"; timestamp: string }
  | { type: "PONG"; timestamp: string }
  | { type: "ERROR"; data: { message: string }; timestamp: string };

// =============================================================================
// Component Prop Types
// =============================================================================

/**
 * Props for CascadingProviderModelSelector component
 */
export interface CascadingProviderModelSelectorProps {
  /** Callback when selection changes */
  onSelectionChange?: (provider: string, model: string) => void;
  /** Default provider to select */
  defaultProvider?: string;
  /** Default model to select */
  defaultModel?: string;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Whether to show provider status indicators */
  showStatus?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Whether to show model capabilities */
  showCapabilities?: boolean;
  /** Session ID for session-scoped selection */
  sessionId?: string;
  /** Placeholder text for provider dropdown */
  providerPlaceholder?: string;
  /** Placeholder text for model dropdown */
  modelPlaceholder?: string;
}

/**
 * Props for ProviderStatusBadge component
 */
export interface ProviderStatusBadgeProps {
  /** Current status */
  status: ProviderStatus;
  /** Optional health score (0-1) */
  healthScore?: number;
  /** Whether to show as a compact badge */
  compact?: boolean;
  /** Last health check timestamp */
  lastChecked?: string;
  /** Error rate percentage */
  errorRate?: number;
  /** Average latency in ms */
  latency?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Props for ModelCapabilityBadges component
 */
export interface ModelCapabilityBadgesProps {
  /** List of capabilities to display */
  capabilities: ModelCapability[];
  /** Maximum number of badges to show before collapsing */
  maxVisible?: number;
  /** Size variant */
  size?: "sm" | "md";
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// Hook Return Types
// =============================================================================

/**
 * Connection status for real-time sync
 */
export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

/**
 * Return type for useUnifiedProviderSelection hook
 */
export interface UseUnifiedProviderSelectionReturn {
  // State
  providers: ProviderInfo[];
  models: ModelInfo[];
  selectedProvider: string | null;
  selectedModel: string | null;

  // Loading states
  isLoadingProviders: boolean;
  isLoadingModels: boolean;
  isSaving: boolean;
  /** Whether a sync operation is in progress */
  isSyncing?: boolean;

  // Error states
  providersError: Error | null;
  modelsError: Error | null;
  /** Error from last sync attempt */
  syncError?: Error | null;

  // Actions
  selectProvider: (providerId: string) => Promise<void>;
  selectModel: (modelId: string) => Promise<void>;
  refreshProviders: () => Promise<void>;
  refreshModels: () => Promise<void>;
  clearSelection: () => Promise<void>;
  /** Force sync with server to get latest state */
  forceSync?: () => Promise<void>;

  // Status
  providerStatus: Record<string, ProviderStatus>;
  connectionStatus: ConnectionStatus;

  // Version tracking
  /** Current selection version for optimistic concurrency */
  selectionVersion?: number;

  // Helpers
  getProviderById: (id: string) => ProviderInfo | undefined;
  getModelById: (id: string) => ModelInfo | undefined;
  getModelsForProvider: (providerId: string) => ModelInfo[];
}

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Capability display configuration
 */
export interface CapabilityConfig {
  /** Display label */
  label: string;
  /** Icon name or component */
  icon: string;
  /** Badge color variant */
  color: "default" | "primary" | "secondary" | "success" | "warning" | "destructive";
  /** Description for tooltip */
  description: string;
}

/**
 * Map of capability to display configuration
 */
export const CAPABILITY_CONFIG: Record<ModelCapability, CapabilityConfig> = {
  streaming: {
    label: "Streaming",
    icon: "zap",
    color: "primary",
    description: "Supports real-time token streaming",
  },
  vision: {
    label: "Vision",
    icon: "eye",
    color: "secondary",
    description: "Can process images and visual content",
  },
  function_calling: {
    label: "Functions",
    icon: "code",
    color: "success",
    description: "Supports function/tool calling",
  },
  json_mode: {
    label: "JSON",
    icon: "braces",
    color: "default",
    description: "Structured JSON output mode",
  },
  code: {
    label: "Code",
    icon: "terminal",
    color: "warning",
    description: "Optimized for code generation",
  },
  reasoning: {
    label: "Reasoning",
    icon: "brain",
    color: "primary",
    description: "Advanced reasoning capabilities",
  },
  embeddings: {
    label: "Embeddings",
    icon: "layers",
    color: "secondary",
    description: "Generates text embeddings",
  },
  tool_use: {
    label: "Tools",
    icon: "wrench",
    color: "success",
    description: "Can use external tools",
  },
};

/**
 * Status display configuration
 */
export interface StatusConfig {
  /** Display label */
  label: string;
  /** Badge color variant */
  color: "default" | "secondary" | "destructive" | "outline";
  /** Description for tooltip */
  description: string;
}

/**
 * Map of status to display configuration
 */
export const STATUS_CONFIG: Record<ProviderStatus, StatusConfig> = {
  healthy: {
    label: "Healthy",
    color: "default",
    description: "Provider is operating normally",
  },
  degraded: {
    label: "Degraded",
    color: "secondary",
    description: "Provider is experiencing issues but still functional",
  },
  unhealthy: {
    label: "Unavailable",
    color: "destructive",
    description: "Provider is currently unavailable",
  },
  unknown: {
    label: "Unknown",
    color: "outline",
    description: "Provider status could not be determined",
  },
};
