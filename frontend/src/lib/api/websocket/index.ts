/**
 * WebSocket Index
 * 
 * Centralized exports for all WebSocket connections.
 */

// Provider Sync
export {
  createProviderSyncConnection,
  type ProviderSyncMessage,
  type ProviderStatusUpdate,
  type RateLimitAlert,
  type ProviderSyncOptions,
  type ProviderSyncConnection,
} from './provider-sync';

// Model Updates
export {
  createModelUpdatesConnection,
  type ModelUpdateMessage,
  type ModelAvailabilityUpdate,
  type ModelDeprecationWarning,
  type NewModelNotification,
  type ModelUpdatesOptions,
  type ModelUpdatesConnection,
} from './model-updates';