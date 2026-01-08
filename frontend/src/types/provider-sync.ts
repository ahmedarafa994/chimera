/**
 * Provider Synchronization Types
 *
 * Type-safe interfaces for provider/model synchronization between
 * frontend and backend. These mirror the backend domain models.
 */

// =============================================================================
// Enums
// =============================================================================

export enum SyncEventType {
  FULL_SYNC = 'full_sync',
  INCREMENTAL_UPDATE = 'incremental_update',
  PROVIDER_ADDED = 'provider_added',
  PROVIDER_UPDATED = 'provider_updated',
  PROVIDER_REMOVED = 'provider_removed',
  PROVIDER_STATUS_CHANGED = 'provider_status_changed',
  MODEL_ADDED = 'model_added',
  MODEL_UPDATED = 'model_updated',
  MODEL_DEPRECATED = 'model_deprecated',
  MODEL_REMOVED = 'model_removed',
  ACTIVE_PROVIDER_CHANGED = 'active_provider_changed',
  ACTIVE_MODEL_CHANGED = 'active_model_changed',
  HEARTBEAT = 'heartbeat',
  ERROR = 'error',
  INITIAL_STATE = 'initial_state',
  STATE_CHANGED = 'state_changed',
}

export enum SyncStatus {
  SYNCED = 'synced',
  SYNCING = 'syncing',
  STALE = 'stale',
  ERROR = 'error',
  DISCONNECTED = 'disconnected',
}

export enum ModelDeprecationStatus {
  ACTIVE = 'active',
  DEPRECATED = 'deprecated',
  SUNSET = 'sunset',
  REMOVED = 'removed',
}

export enum ProviderStatus {
  AVAILABLE = 'available',
  UNAVAILABLE = 'unavailable',
  DEGRADED = 'degraded',
  RATE_LIMITED = 'rate_limited',
  MAINTENANCE = 'maintenance',
  UNKNOWN = 'unknown',
}

export enum ProviderType {
  OPENAI = 'openai',
  ANTHROPIC = 'anthropic',
  GOOGLE = 'google',
  DEEPSEEK = 'deepseek',
  BIGMODEL = 'bigmodel',  // ZhiPu AI GLM models
  ROUTEWAY = 'routeway',  // Unified AI gateway
  LOCAL = 'local',
  CUSTOM = 'custom',
}

export enum ModelTier {
  FREE = 'free',
  STANDARD = 'standard',
  PREMIUM = 'premium',
  ENTERPRISE = 'enterprise',
}

// =============================================================================
// Model Specifications
// =============================================================================

export interface ModelPricing {
  input_per_1k_tokens: number;
  output_per_1k_tokens: number;
  currency: string;
  effective_date?: string;
}

export interface ModelSpecification {
  id: string;
  name: string;
  provider_id: string;
  description?: string;
  context_window: number;
  max_input_tokens: number;
  max_output_tokens: number;
  supports_streaming: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supports_json_mode: boolean;
  is_default: boolean;
  is_available: boolean;
  tier: ModelTier;
  pricing?: ModelPricing;
  deprecation_status: ModelDeprecationStatus;
  deprecation_date?: string;
  sunset_date?: string;
  replacement_model_id?: string;
  version?: string;
  release_date?: string;
  training_cutoff?: string;
  tags: string[];
}

// =============================================================================
// Provider Information
// =============================================================================

export interface ProviderCapabilities {
  supports_streaming: boolean;
  supports_function_calling: boolean;
  supports_vision: boolean;
  supports_embeddings: boolean;
  supports_fine_tuning: boolean;
  max_context_window: number;
  rate_limit_rpm?: number;
  rate_limit_tpm?: number;
}

export interface ProviderHealthInfo {
  status: ProviderStatus;
  latency_ms?: number;
  last_check?: string;
  error_message?: string;
  consecutive_failures: number;
  uptime_percentage?: number;
}

export interface ProviderSyncInfo {
  id: string;
  type: ProviderType;
  display_name: string;
  description?: string;
  enabled: boolean;
  is_configured: boolean;
  is_default: boolean;
  is_fallback: boolean;
  priority: number;
  capabilities: ProviderCapabilities;
  health?: ProviderHealthInfo;
  models: ModelSpecification[];
  model_count: number;
  default_model_id?: string;
  base_url?: string;
  api_version?: string;
  last_updated?: string;
}

// =============================================================================
// Sync Metadata
// =============================================================================

export interface SyncMetadata {
  version: number;
  last_sync_time: string;
  server_time: string;
  checksum: string;
  provider_count: number;
  model_count: number;
  sync_duration_ms?: number;
}

// =============================================================================
// Sync State
// =============================================================================

export interface SyncState {
  providers: ProviderSyncInfo[];
  all_models: ModelSpecification[];
  active_provider_id?: string;
  active_model_id?: string;
  default_provider_id?: string;
  default_model_id?: string;
  metadata: SyncMetadata;
  provider_count: number;
  model_count: number;
}

// =============================================================================
// Sync Events
// =============================================================================

export interface SyncEvent<T = unknown> {
  type: SyncEventType;
  timestamp: string;
  version: number;
  data?: T;
  provider_id?: string;
  model_id?: string;
  error?: string;
}

// Specific event data types
export interface ProviderAddedEventData {
  provider: ProviderSyncInfo;
}

export interface ProviderUpdatedEventData {
  provider: ProviderSyncInfo;
  changed_fields: string[];
}

export interface ProviderRemovedEventData {
  provider_id: string;
}

export interface ProviderStatusChangedEventData {
  provider_id: string;
  old_status: ProviderStatus;
  new_status: ProviderStatus;
  health?: ProviderHealthInfo;
}

export interface ModelDeprecatedEventData {
  model: ModelSpecification;
  deprecation_date: string;
  sunset_date?: string;
  replacement_model_id?: string;
}

export interface ActiveProviderChangedEventData {
  previous_provider_id?: string;
}

export interface HeartbeatEventData {
  server_time: string;
  connected_clients: number;
}

export interface ErrorEventData {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

// =============================================================================
// Sync Request/Response
// =============================================================================

export type SyncType = 'full' | 'incremental';

export interface SyncRequest {
  client_version?: number;
  last_sync_time?: string;
  sync_type: SyncType;
  include_deprecated?: boolean;
  provider_ids?: string[];
}

export interface SyncResponse {
  success: boolean;
  sync_type: SyncType;
  state?: SyncState;
  events?: SyncEvent[];
  metadata?: SyncMetadata;
  error?: string;
  retry_after_seconds?: number;
}

// =============================================================================
// Availability Information
// =============================================================================

export interface ProviderAvailabilityInfo {
  provider_id: string;
  is_available: boolean;
  status: ProviderStatus;
  health?: ProviderHealthInfo;
  available_models: number;
  total_models: number;
  fallback_provider_id?: string;
  fallback_provider_name?: string;
  estimated_recovery_time?: string;
  maintenance_message?: string;
}

export interface ModelAvailabilityInfo {
  model_id: string;
  provider_id: string;
  is_available: boolean;
  deprecation_status: ModelDeprecationStatus;
  deprecation_warning?: string;
  sunset_date?: string;
  replacement_model_id?: string;
  replacement_model_name?: string;
  alternative_models: string[];
}

// =============================================================================
// Client State
// =============================================================================

export interface ProviderSyncClientState {
  status: SyncStatus;
  lastSyncTime?: Date;
  version: number;
  error?: string;
  isConnected: boolean;
  reconnectAttempts: number;
  providers: Map<string, ProviderSyncInfo>;
  models: Map<string, ModelSpecification>;
  activeProviderId?: string;
  activeModelId?: string;
}

// =============================================================================
// WebSocket Messages
// =============================================================================

export type WebSocketMessageType =
  | 'ping'
  | 'pong'
  | 'sync_request'
  | 'sync_response'
  | 'get_version'
  | 'version';

export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType | SyncEventType;
  data?: T;
  timestamp?: string;
  version?: number;
}

// =============================================================================
// Utility Types
// =============================================================================

export type ProviderMap = Map<string, ProviderSyncInfo>;
export type ModelMap = Map<string, ModelSpecification>;

export interface SyncEventHandler<T = unknown> {
  (event: SyncEvent<T>): void | Promise<void>;
}

export interface SyncEventHandlers {
  onFullSync?: SyncEventHandler<SyncState>;
  onProviderAdded?: SyncEventHandler<ProviderAddedEventData>;
  onProviderUpdated?: SyncEventHandler<ProviderUpdatedEventData>;
  onProviderRemoved?: SyncEventHandler<ProviderRemovedEventData>;
  onProviderStatusChanged?: SyncEventHandler<ProviderStatusChangedEventData>;
  onModelDeprecated?: SyncEventHandler<ModelDeprecatedEventData>;
  onActiveProviderChanged?: SyncEventHandler<ActiveProviderChangedEventData>;
  onStateChanged?: SyncEventHandler<ProviderSyncClientState>;
  onHeartbeat?: SyncEventHandler<HeartbeatEventData>;
  onError?: SyncEventHandler<ErrorEventData>;
  onConnectionChange?: (connected: boolean) => void;
}

// =============================================================================
// Configuration
// =============================================================================

export interface ProviderSyncConfig {
  /** Base URL for the sync API */
  apiBaseUrl: string;
  /** WebSocket URL for real-time updates */
  wsUrl: string;
  /** Enable WebSocket connection */
  enableWebSocket: boolean;
  /** Polling interval in ms when WebSocket is unavailable */
  pollingInterval: number;
  /** Maximum reconnection attempts */
  maxReconnectAttempts: number;
  /** Base delay for exponential backoff in ms */
  reconnectBaseDelay: number;
  /** Maximum delay for exponential backoff in ms */
  reconnectMaxDelay: number;
  /** Heartbeat interval in ms */
  heartbeatInterval: number;
  /** Sync timeout in ms */
  syncTimeout: number;
  /** Include deprecated models in sync */
  includeDeprecated: boolean;
  /** Enable local caching */
  enableCache: boolean;
  /** Cache TTL in ms */
  cacheTtl: number;
}

export const DEFAULT_SYNC_CONFIG: ProviderSyncConfig = {
  // Use environment variable or fallback to direct backend URL
  // Note: These are overridden at runtime by getProviderSyncService()
  apiBaseUrl: typeof window !== 'undefined'
    ? `${process.env.NEXT_PUBLIC_CHIMERA_API_URL || 'http://localhost:8001/api/v1'}/provider-sync`
    : '/api/v1/provider-sync',
  wsUrl: typeof window !== 'undefined'
    ? `${(process.env.NEXT_PUBLIC_CHIMERA_API_URL || 'http://localhost:8001/api/v1').replace(/^http/, 'ws')}/provider-sync/ws`
    : '/api/v1/provider-sync/ws',
  enableWebSocket: true,
  pollingInterval: 30000, // 30 seconds
  maxReconnectAttempts: 5,
  reconnectBaseDelay: 1000, // 1 second
  reconnectMaxDelay: 30000, // 30 seconds
  heartbeatInterval: 25000, // 25 seconds
  syncTimeout: 10000, // 10 seconds
  includeDeprecated: false,
  enableCache: true,
  cacheTtl: 300000, // 5 minutes
};
