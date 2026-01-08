/**
 * Real-time Synchronization Types
 * 
 * Type definitions for WebSocket, SSE, and state synchronization
 */

// ============================================================================
// Connection Types
// ============================================================================

export type ConnectionState = 
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'reconnecting'
  | 'error';

export interface ConnectionInfo {
  state: ConnectionState;
  url: string;
  connectedAt?: Date;
  disconnectedAt?: Date;
  reconnectAttempts: number;
  latency?: number;
  error?: Error;
}

// ============================================================================
// WebSocket Types
// ============================================================================

export interface WebSocketConfig {
  /** WebSocket URL */
  url: string;
  /** Protocols to use */
  protocols?: string | string[];
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Maximum reconnection attempts */
  maxReconnectAttempts?: number;
  /** Base delay between reconnection attempts (ms) */
  reconnectDelay?: number;
  /** Maximum reconnection delay (ms) */
  maxReconnectDelay?: number;
  /** Heartbeat interval (ms), 0 to disable */
  heartbeatInterval?: number;
  /** Heartbeat timeout (ms) */
  heartbeatTimeout?: number;
  /** Message queue size when disconnected */
  queueSize?: number;
  /** Enable debug logging */
  debug?: boolean;
}

export interface WebSocketMessage<T = unknown> {
  id: string;
  type: string;
  payload: T;
  timestamp: number;
  correlationId?: string;
}

export interface WebSocketRequest<T = unknown> extends WebSocketMessage<T> {
  expectResponse?: boolean;
  timeout?: number;
}

export interface WebSocketResponse<T = unknown> extends WebSocketMessage<T> {
  success: boolean;
  error?: string;
  correlationId: string;
}

export type WebSocketEventType = 
  | 'open'
  | 'close'
  | 'error'
  | 'message'
  | 'reconnecting'
  | 'reconnected'
  | 'heartbeat';

export interface WebSocketEvent {
  type: WebSocketEventType;
  timestamp: number;
  data?: unknown;
  error?: Error;
}

export type WebSocketMessageHandler<T = unknown> = (message: WebSocketMessage<T>) => void;
export type WebSocketEventHandler = (event: WebSocketEvent) => void;

// ============================================================================
// Server-Sent Events Types
// ============================================================================

export interface SSEConfig {
  /** SSE endpoint URL */
  url: string;
  /** Custom headers (via EventSource polyfill) */
  headers?: Record<string, string>;
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Maximum reconnection attempts */
  maxReconnectAttempts?: number;
  /** Reconnection delay (ms) */
  reconnectDelay?: number;
  /** Enable debug logging */
  debug?: boolean;
}

export interface SSEMessage<T = unknown> {
  id?: string;
  event: string;
  data: T;
  timestamp: number;
}

export type SSEEventHandler<T = unknown> = (message: SSEMessage<T>) => void;

// ============================================================================
// State Synchronization Types
// ============================================================================

export interface SyncState<T = unknown> {
  data: T;
  version: number;
  lastSyncedAt: Date;
  isDirty: boolean;
  syncStatus: 'idle' | 'syncing' | 'error';
  error?: Error;
}

export interface SyncOperation<T = unknown> {
  id: string;
  type: 'create' | 'update' | 'delete' | 'patch';
  path: string;
  data?: T;
  previousData?: T;
  timestamp: number;
  status: 'pending' | 'syncing' | 'synced' | 'failed' | 'rolled_back';
  retryCount: number;
  error?: Error;
}

export interface SyncConfig {
  /** Debounce delay for batching operations (ms) */
  debounceDelay?: number;
  /** Maximum batch size */
  maxBatchSize?: number;
  /** Maximum retry attempts */
  maxRetries?: number;
  /** Retry delay (ms) */
  retryDelay?: number;
  /** Enable optimistic updates */
  optimisticUpdates?: boolean;
  /** Conflict resolution strategy */
  conflictResolution?: 'client-wins' | 'server-wins' | 'merge' | 'manual';
}

export interface ConflictInfo<T = unknown> {
  operationId: string;
  clientData: T;
  serverData: T;
  timestamp: number;
}

export type ConflictResolver<T = unknown> = (conflict: ConflictInfo<T>) => T | Promise<T>;

// ============================================================================
// Optimistic Update Types
// ============================================================================

export interface OptimisticUpdate<T = unknown> {
  id: string;
  timestamp: number;
  optimisticData: T;
  originalData: T;
  mutation: () => Promise<T>;
  rollback: () => void;
  status: 'pending' | 'committed' | 'rolled_back';
}

export interface MutationOptions<T = unknown, V = unknown> {
  /** Optimistic data to apply immediately */
  optimisticData?: T | ((current: T) => T);
  /** Rollback function if mutation fails */
  onRollback?: (error: Error, variables: V) => void;
  /** Success callback */
  onSuccess?: (data: T, variables: V) => void;
  /** Error callback */
  onError?: (error: Error, variables: V) => void;
  /** Retry configuration */
  retry?: number | boolean;
  /** Retry delay (ms) */
  retryDelay?: number;
}

// ============================================================================
// Event Bus Types
// ============================================================================

export interface EventBusConfig {
  /** Maximum listeners per event */
  maxListeners?: number;
  /** Enable debug logging */
  debug?: boolean;
  /** Event history size */
  historySize?: number;
}

export interface BusEvent<T = unknown> {
  type: string;
  payload: T;
  timestamp: number;
  source?: string;
}

export type EventHandler<T = unknown> = (event: BusEvent<T>) => void;
export type EventFilter<T = unknown> = (event: BusEvent<T>) => boolean;

export interface Subscription {
  unsubscribe: () => void;
}

// ============================================================================
// Channel Types (for pub/sub patterns)
// ============================================================================

export interface Channel<T = unknown> {
  name: string;
  subscribe: (handler: EventHandler<T>) => Subscription;
  publish: (payload: T) => void;
  getLastMessage: () => BusEvent<T> | undefined;
  getHistory: () => BusEvent<T>[];
}

export interface ChannelConfig {
  /** Persist last message */
  persistLast?: boolean;
  /** History size */
  historySize?: number;
  /** Replay history on subscribe */
  replayOnSubscribe?: boolean;
}

// ============================================================================
// Presence Types (for collaborative features)
// ============================================================================

export interface PresenceUser {
  id: string;
  name?: string;
  avatar?: string;
  status: 'online' | 'away' | 'busy' | 'offline';
  lastSeen: Date;
  metadata?: Record<string, unknown>;
}

export interface PresenceState {
  users: Map<string, PresenceUser>;
  currentUser?: PresenceUser;
}

export interface PresenceConfig {
  /** Heartbeat interval (ms) */
  heartbeatInterval?: number;
  /** Away timeout (ms) */
  awayTimeout?: number;
  /** Offline timeout (ms) */
  offlineTimeout?: number;
}

// ============================================================================
// Retry Types
// ============================================================================

export interface RetryConfig {
  /** Maximum retry attempts */
  maxAttempts: number;
  /** Base delay between retries (ms) */
  baseDelay: number;
  /** Maximum delay between retries (ms) */
  maxDelay: number;
  /** Backoff multiplier */
  backoffMultiplier: number;
  /** Jitter factor (0-1) */
  jitter: number;
  /** Retryable error codes */
  retryableErrors?: string[];
  /** Should retry function */
  shouldRetry?: (error: Error, attempt: number) => boolean;
}

export interface RetryState {
  attempt: number;
  nextDelay: number;
  totalDelay: number;
  lastError?: Error;
}

// ============================================================================
// Circuit Breaker Types
// ============================================================================

export type CircuitState = 'closed' | 'open' | 'half-open';

export interface CircuitBreakerConfig {
  /** Failure threshold to open circuit */
  failureThreshold: number;
  /** Success threshold to close circuit (in half-open state) */
  successThreshold: number;
  /** Time to wait before half-open (ms) */
  resetTimeout: number;
  /** Time window for failure counting (ms) */
  failureWindow: number;
}

export interface CircuitBreakerState {
  state: CircuitState;
  failures: number;
  successes: number;
  lastFailure?: Date;
  lastSuccess?: Date;
  nextAttempt?: Date;
}