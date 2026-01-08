/**
 * WebSocket Index for Project Chimera Frontend
 * 
 * Central export point for all WebSocket integrations
 */

// Provider Sync WebSocket
export {
  ProviderSyncManager,
  getProviderSyncManager,
  destroyProviderSyncManager,
  createProviderSyncHookState,
} from "./provider-sync";
export type {
  ProviderSyncCallbacks,
  ProviderSyncOptions,
  UseProviderSyncConfig,
} from "./provider-sync";

// Model Updates WebSocket
export {
  ModelUpdatesManager,
  getModelUpdatesManager,
  destroyModelUpdatesManager,
  subscribeToProviderUpdates,
  subscribeToAllModelUpdates,
  createModelUpdatesHookState,
} from "./model-updates";
export type {
  ModelUpdateCallbacks,
  ModelUpdatesOptions,
  ModelUpdatesHookState,
} from "./model-updates";