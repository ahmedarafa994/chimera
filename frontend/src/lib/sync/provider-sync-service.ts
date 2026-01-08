/**
 * Provider Sync Service
 *
 * Handles synchronization of provider and model configurations between
 * frontend and backend with:
 * - Version-based conflict resolution
 * - WebSocket real-time updates
 * - Polling fallback
 * - Local caching
 * - Retry logic with exponential backoff
 */

import {
  SyncEventType,
  SyncStatus,
  SyncState,
  SyncEvent,
  SyncRequest,
  SyncResponse,
  SyncMetadata,
  ProviderAddedEventData,
  ProviderUpdatedEventData,
  ProviderRemovedEventData,
  ProviderStatusChangedEventData,
  ModelDeprecatedEventData,
  ActiveProviderChangedEventData,
  HeartbeatEventData,
  ErrorEventData,
  ProviderSyncInfo,
  ModelSpecification,
  ProviderSyncConfig,
  ProviderSyncClientState,
  SyncEventHandlers,
  DEFAULT_SYNC_CONFIG,
  WebSocketMessage,
  ProviderAvailabilityInfo,
  ModelAvailabilityInfo,
} from '@/types/provider-sync';

// =============================================================================
// Cache Manager
// =============================================================================

class SyncCacheManager {
  private readonly CACHE_KEY = 'chimera_provider_sync_cache';
  private readonly VERSION_KEY = 'chimera_provider_sync_version';
  private readonly SELECTION_KEY = 'chimera_provider_selection';

  save(state: SyncState): void {
    try {
      localStorage.setItem(this.CACHE_KEY, JSON.stringify(state));
      localStorage.setItem(this.VERSION_KEY, state.metadata.version.toString());
    } catch (error) {
      console.warn('[SyncCache] Failed to save cache:', error);
    }
  }

  load(): SyncState | null {
    try {
      const cached = localStorage.getItem(this.CACHE_KEY);
      if (!cached) return null;

      const state = JSON.parse(cached) as SyncState;

      // Check if cache is expired (based on last sync time)
      const lastSync = new Date(state.metadata.last_sync_time);
      const now = new Date();
      const ttl = DEFAULT_SYNC_CONFIG.cacheTtl;

      if (now.getTime() - lastSync.getTime() > ttl) {
        this.clear();
        return null;
      }

      return state;
    } catch (error) {
      console.warn('[SyncCache] Failed to load cache:', error);
      return null;
    }
  }

  getVersion(): number {
    try {
      const version = localStorage.getItem(this.VERSION_KEY);
      return version ? parseInt(version, 10) : 0;
    } catch {
      return 0;
    }
  }

  saveSelection(providerId: string | undefined, modelId: string | undefined): void {
    try {
      localStorage.setItem(this.SELECTION_KEY, JSON.stringify({ providerId, modelId }));
    } catch (error) {
      console.warn('[SyncCache] Failed to save selection:', error);
    }
  }

  loadSelection(): { providerId?: string; modelId?: string } | null {
    try {
      const cached = localStorage.getItem(this.SELECTION_KEY);
      if (!cached) return null;
      return JSON.parse(cached);
    } catch {
      return null;
    }
  }

  clear(): void {
    try {
      localStorage.removeItem(this.CACHE_KEY);
      localStorage.removeItem(this.VERSION_KEY);
    } catch (error) {
      console.warn('[SyncCache] Failed to clear cache:', error);
    }
  }
}

// =============================================================================
// Cross-Tab Broadcast Channel
// =============================================================================

interface BroadcastMessage {
  type: 'selection_changed' | 'sync_updated' | 'force_sync';
  payload: {
    providerId?: string;
    modelId?: string;
    version?: number;
    timestamp: number;
    sourceTabId: string;
  };
}

// =============================================================================
// Provider Sync Service
// =============================================================================

export class ProviderSyncService {
  private config: ProviderSyncConfig;
  private cache: SyncCacheManager;
  private ws: WebSocket | null = null;
  private state: ProviderSyncClientState;
  private handlers: SyncEventHandlers = {};
  private pollingInterval: ReturnType<typeof setInterval> | null = null;
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private isDestroyed = false;
  private broadcastChannel: BroadcastChannel | null = null;
  private tabId: string;

  constructor(config: Partial<ProviderSyncConfig> = {}) {
    this.config = { ...DEFAULT_SYNC_CONFIG, ...config };
    this.cache = new SyncCacheManager();
    this.state = this.createInitialState();
    this.tabId = Math.random().toString(36).substring(2, 9);

    // Initialize BroadcastChannel for cross-tab sync
    this.initBroadcastChannel();
  }

  /**
   * Initialize BroadcastChannel for cross-tab synchronization
   */
  private initBroadcastChannel(): void {
    if (typeof window === 'undefined' || !('BroadcastChannel' in window)) {
      console.log('[ProviderSync] BroadcastChannel not available');
      return;
    }

    try {
      this.broadcastChannel = new BroadcastChannel('chimera_provider_sync');

      this.broadcastChannel.onmessage = (event: MessageEvent<BroadcastMessage>) => {
        const message = event.data;

        // Ignore messages from this tab
        if (message.payload.sourceTabId === this.tabId) {
          return;
        }

        console.log('[ProviderSync] Received cross-tab message:', message.type);

        switch (message.type) {
          case 'selection_changed':
            // Apply selection from another tab
            if (message.payload.providerId !== undefined) {
              this.state.activeProviderId = message.payload.providerId;
            }
            if (message.payload.modelId !== undefined) {
              this.state.activeModelId = message.payload.modelId;
            }
            this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());
            break;

          case 'sync_updated':
            // Another tab performed a sync, refresh our data
            if (message.payload.version && message.payload.version > this.state.version) {
              this.performFullSync().catch(err =>
                console.warn('[ProviderSync] Cross-tab sync refresh failed:', err)
              );
            }
            break;

          case 'force_sync':
            // Another tab requested a force sync
            this.performFullSync().catch(err =>
              console.warn('[ProviderSync] Cross-tab force sync failed:', err)
            );
            break;
        }
      };

      this.broadcastChannel.onmessageerror = (error) => {
        console.warn('[ProviderSync] BroadcastChannel message error:', error);
      };

      console.log('[ProviderSync] BroadcastChannel initialized, tabId:', this.tabId);
    } catch (error) {
      console.warn('[ProviderSync] Failed to initialize BroadcastChannel:', error);
    }
  }

  /**
   * Broadcast a message to other tabs
   */
  private broadcast(message: Omit<BroadcastMessage, 'payload'> & { payload: Omit<BroadcastMessage['payload'], 'sourceTabId' | 'timestamp'> }): void {
    if (!this.broadcastChannel) return;

    try {
      this.broadcastChannel.postMessage({
        ...message,
        payload: {
          ...message.payload,
          sourceTabId: this.tabId,
          timestamp: Date.now(),
        },
      } as BroadcastMessage);
    } catch (error) {
      console.warn('[ProviderSync] Failed to broadcast message:', error);
    }
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Initialize the sync service
   */
  async initialize(): Promise<void> {
    console.log('[ProviderSync] Initializing...');

    // Load cached state first for immediate UI
    if (this.config.enableCache) {
      const cached = this.cache.load();
      if (cached) {
        console.log('[ProviderSync] Loaded cached state, version:', cached.metadata.version);
        this.applyState(cached);
        this.state.status = SyncStatus.STALE;
      }

      // Load persisted selection
      const selection = this.cache.loadSelection();
      if (selection) {
        console.log('[ProviderSync] Loaded cached selection:', selection);
        this.state.activeProviderId = selection.providerId;
        this.state.activeModelId = selection.modelId;
      }
    }

    // Perform initial sync (don't fail initialization if sync fails)
    try {
      await this.performFullSync();
    } catch (error) {
      console.warn('[ProviderSync] Initial sync failed, will retry via polling:', error);
      this.state.status = SyncStatus.ERROR;
      this.state.error = 'Backend unavailable - using cached data if available';
    }

    // Start WebSocket connection or polling
    if (this.config.enableWebSocket) {
      this.connectWebSocket();
    } else {
      this.startPolling();
    }
  }

  /**
   * Destroy the sync service and cleanup resources
   */
  destroy(): void {
    console.log('[ProviderSync] Destroying...');
    this.isDestroyed = true;
    this.disconnectWebSocket();
    this.stopPolling();
    this.stopHeartbeat();

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Cleanup BroadcastChannel
    if (this.broadcastChannel) {
      this.broadcastChannel.close();
      this.broadcastChannel = null;
    }
  }

  /**
   * Register event handlers
   */
  setHandlers(handlers: SyncEventHandlers): void {
    this.handlers = handlers;
  }

  /**
   * Get current sync state
   */
  getState(): ProviderSyncClientState {
    return { ...this.state };
  }

  /**
   * Get all providers
   */
  getProviders(): ProviderSyncInfo[] {
    return Array.from(this.state.providers.values());
  }

  /**
   * Get a specific provider
   */
  getProvider(id: string): ProviderSyncInfo | undefined {
    return this.state.providers.get(id);
  }

  /**
   * Get all models
   */
  getModels(): ModelSpecification[] {
    return Array.from(this.state.models.values());
  }

  /**
   * Get models for a specific provider
   */
  getProviderModels(providerId: string): ModelSpecification[] {
    return Array.from(this.state.models.values())
      .filter(m => m.provider_id === providerId);
  }

  /**
   * Get a specific model
   */
  getModel(id: string): ModelSpecification | undefined {
    return this.state.models.get(id);
  }

  /**
   * Get active provider
   */
  getActiveProvider(): ProviderSyncInfo | undefined {
    if (!this.state.activeProviderId) return undefined;
    return this.state.providers.get(this.state.activeProviderId);
  }

  /**
   * Get active model
   */
  getActiveModel(): ModelSpecification | undefined {
    if (!this.state.activeModelId) return undefined;
    return this.state.models.get(this.state.activeModelId);
  }

  /**
   * Force a full sync
   */
  async forceSync(): Promise<void> {
    await this.performFullSync();
  }

  /**
   * Get provider availability info
   */
  async getProviderAvailability(providerId: string): Promise<ProviderAvailabilityInfo | null> {
    try {
      const response = await fetch(
        `${this.config.apiBaseUrl}/providers/${providerId}/availability`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('[ProviderSync] Failed to get provider availability:', error);
      return null;
    }
  }

  /**
   * Get model availability info
   */
  async getModelAvailability(
    modelId: string,
    providerId?: string
  ): Promise<ModelAvailabilityInfo | null> {
    try {
      const params = new URLSearchParams();
      if (providerId) params.set('provider_id', providerId);

      const response = await fetch(
        `${this.config.apiBaseUrl}/models/${modelId}/availability?${params}`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('[ProviderSync] Failed to get model availability:', error);
      return null;
    }
  }

  /**
   * Select a provider with optimistic update
   */
  async selectProvider(
    providerId: string,
    modelId?: string,
    options?: { persist?: boolean }
  ): Promise<{
    success: boolean;
    providerId: string;
    modelId?: string;
    fallbackApplied?: boolean;
    fallbackReason?: string;
    error?: string;
  }> {
    const previousProviderId = this.state.activeProviderId;
    const previousModelId = this.state.activeModelId;

    // Optimistic update
    this.state.activeProviderId = providerId;
    if (modelId) {
      this.state.activeModelId = modelId;
    }

    // Notify handlers immediately for responsive UI
    this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/select/provider`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider_id: providerId,
          model_id: modelId,
          persist: options?.persist ?? true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        // Update to actual values from server (may differ if fallback applied)
        this.state.activeProviderId = result.provider_id;
        this.state.activeModelId = result.model_id;

        // Save selection to local storage for persistence
        this.cache.saveSelection(result.provider_id, result.model_id);

        // Broadcast selection change to other tabs
        this.broadcast({
          type: 'selection_changed',
          payload: {
            providerId: result.provider_id,
            modelId: result.model_id,
          },
        });

        // Notify handlers
        this.handlers.onActiveProviderChanged?.({
          type: SyncEventType.ACTIVE_PROVIDER_CHANGED,
          timestamp: new Date().toISOString(),
          version: this.state.version,
          provider_id: result.provider_id,
          data: {
            previous_provider_id: previousProviderId,
          },
        });

        return {
          success: true,
          providerId: result.provider_id,
          modelId: result.model_id,
          fallbackApplied: result.fallback_applied,
          fallbackReason: result.fallback_reason,
        };
      } else {
        // Rollback optimistic update
        this.state.activeProviderId = previousProviderId;
        this.state.activeModelId = previousModelId;
        this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());

        return {
          success: false,
          providerId,
          error: result.message || 'Selection failed',
        };
      }
    } catch (error) {
      // Rollback optimistic update on error
      this.state.activeProviderId = previousProviderId;
      this.state.activeModelId = previousModelId;
      this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());

      console.error('[ProviderSync] Failed to select provider:', error);
      return {
        success: false,
        providerId,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Select a model within the current provider with optimistic update
   */
  async selectModel(
    modelId: string,
    options?: { persist?: boolean }
  ): Promise<{
    success: boolean;
    modelId: string;
    providerId?: string;
    error?: string;
  }> {
    const previousModelId = this.state.activeModelId;

    // Optimistic update
    this.state.activeModelId = modelId;

    // Notify handlers immediately for responsive UI
    this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/select/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: modelId,
          persist: options?.persist ?? true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        this.state.activeModelId = result.model_id;

        // Save selection to local storage for persistence
        this.cache.saveSelection(this.state.activeProviderId, result.model_id);

        // Broadcast selection change to other tabs
        this.broadcast({
          type: 'selection_changed',
          payload: {
            providerId: this.state.activeProviderId,
            modelId: result.model_id,
          },
        });

        return {
          success: true,
          modelId: result.model_id,
          providerId: result.provider_id,
        };
      } else {
        // Rollback optimistic update
        this.state.activeModelId = previousModelId;
        this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());

        return {
          success: false,
          modelId,
          error: result.message || 'Selection failed',
        };
      }
    } catch (error) {
      // Rollback optimistic update on error
      this.state.activeModelId = previousModelId;
      this.notifyHandlers(SyncEventType.STATE_CHANGED, this.getState());

      console.error('[ProviderSync] Failed to select model:', error);
      return {
        success: false,
        modelId,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Get current active selection
   */
  async getActiveSelection(): Promise<{
    providerId?: string;
    providerName?: string;
    modelId?: string;
    modelName?: string;
    isFallback?: boolean;
  } | null> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/active`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const result = await response.json();

      // Update local state
      if (result.provider_id) {
        this.state.activeProviderId = result.provider_id;
      }
      if (result.model_id) {
        this.state.activeModelId = result.model_id;
      }

      return {
        providerId: result.provider_id,
        providerName: result.provider_name,
        modelId: result.model_id,
        modelName: result.model_name,
        isFallback: result.is_fallback,
      };
    } catch (error) {
      console.error('[ProviderSync] Failed to get active selection:', error);
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Sync Operations
  // ---------------------------------------------------------------------------

  private async performFullSync(): Promise<void> {
    if (this.isDestroyed) return;

    console.log('[ProviderSync] Performing full sync...');
    this.state.status = SyncStatus.SYNCING;

    try {
      const request: SyncRequest = {
        sync_type: 'full',
        include_deprecated: this.config.includeDeprecated,
      };

      const response = await this.sendSyncRequest(request);

      if (response.success && response.state) {
        this.applyState(response.state);
        this.state.status = SyncStatus.SYNCED;
        this.state.lastSyncTime = new Date();
        this.state.error = undefined;

        // Cache the state
        if (this.config.enableCache) {
          this.cache.save(response.state);
        }

        // Notify handlers
        this.handlers.onFullSync?.({
          type: SyncEventType.FULL_SYNC,
          timestamp: new Date().toISOString(),
          version: response.state.metadata.version,
          data: response.state,
        });

        console.log('[ProviderSync] Full sync completed, version:', response.state.metadata.version);
      } else {
        throw new Error(response.error || 'Sync failed');
      }
    } catch (error) {
      console.error('[ProviderSync] Full sync failed:', error);
      this.state.status = SyncStatus.ERROR;
      this.state.error = error instanceof Error ? error.message : 'Unknown error';

      this.handlers.onError?.({
        type: SyncEventType.ERROR,
        timestamp: new Date().toISOString(),
        version: this.state.version,
        data: {
          code: 'SYNC_FAILED',
          message: this.state.error,
        },
      });
    }
  }

  private async performIncrementalSync(): Promise<void> {
    if (this.isDestroyed) return;

    console.log('[ProviderSync] Performing incremental sync...');

    try {
      const request: SyncRequest = {
        sync_type: 'incremental',
        client_version: this.state.version,
        last_sync_time: this.state.lastSyncTime?.toISOString(),
        include_deprecated: this.config.includeDeprecated,
      };

      const response = await this.sendSyncRequest(request);

      if (response.success) {
        if (response.sync_type === 'full' && response.state) {
          // Server requested full sync (version mismatch)
          this.applyState(response.state);

          if (this.config.enableCache) {
            this.cache.save(response.state);
          }
        } else if (response.events) {
          // Apply incremental events
          for (const event of response.events) {
            this.handleSyncEvent(event);
          }
        }

        this.state.status = SyncStatus.SYNCED;
        this.state.lastSyncTime = new Date();
        this.state.error = undefined;
      }
    } catch (error) {
      console.error('[ProviderSync] Incremental sync failed:', error);
      // Don't change status to error for incremental sync failures
      // Just log and continue with polling
    }
  }

  private async sendSyncRequest(request: SyncRequest): Promise<SyncResponse> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.syncTimeout);

    try {
      console.log('[ProviderSync] Sending sync request to:', `${this.config.apiBaseUrl}/sync`);
      const response = await fetch(`${this.config.apiBaseUrl}/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!response.ok) {
        // Log more details for debugging
        const errorText = await response.text().catch(() => 'Unable to read error body');
        console.error('[ProviderSync] Sync request failed:', {
          status: response.status,
          statusText: response.statusText,
          url: `${this.config.apiBaseUrl}/sync`,
          errorBody: errorText.substring(0, 200),
        });
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      // Handle network errors more gracefully
      if (error instanceof TypeError && error.message.includes('fetch')) {
        console.warn('[ProviderSync] Network error - backend may be unavailable');
        throw new Error('Network error: Backend unavailable');
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  // ---------------------------------------------------------------------------
  // State Management
  // ---------------------------------------------------------------------------

  private createInitialState(): ProviderSyncClientState {
    return {
      status: SyncStatus.DISCONNECTED,
      version: 0,
      isConnected: false,
      reconnectAttempts: 0,
      providers: new Map(),
      models: new Map(),
    };
  }

  private applyState(state: SyncState): void {
    // Update version
    this.state.version = state.metadata.version;

    // Update providers
    this.state.providers.clear();
    for (const provider of state.providers) {
      this.state.providers.set(provider.id, provider);
    }

    // Update models
    this.state.models.clear();
    for (const model of state.all_models) {
      this.state.models.set(model.id, model);
    }

    // Update active selections
    this.state.activeProviderId = state.active_provider_id;
    this.state.activeModelId = state.active_model_id;
  }

  // ---------------------------------------------------------------------------
  // Event Handling
  // ---------------------------------------------------------------------------

  private notifyHandlers(type: SyncEventType, state: ProviderSyncClientState): void {
    if (type !== SyncEventType.STATE_CHANGED) {
      return;
    }

    this.handlers.onStateChanged?.({
      type,
      timestamp: new Date().toISOString(),
      version: this.state.version,
      data: state,
    });
  }

  private handleSyncEvent(event: SyncEvent): void {
    console.log('[ProviderSync] Handling event:', event.type);

    // Update version if newer
    if (event.version > this.state.version) {
      this.state.version = event.version;
    }

    switch (event.type) {
      case SyncEventType.FULL_SYNC:
      case SyncEventType.INITIAL_STATE:
        if (event.data) {
          this.applyState(event.data as SyncState);
          this.handlers.onFullSync?.(event as SyncEvent<SyncState>);
        }
        break;

      case SyncEventType.PROVIDER_ADDED:
        if (event.data) {
          const { provider } = event.data as { provider: ProviderSyncInfo };
          this.state.providers.set(provider.id, provider);
          // Add provider's models
          for (const model of provider.models) {
            this.state.models.set(model.id, model);
          }
          this.handlers.onProviderAdded?.(event as SyncEvent<ProviderAddedEventData>);
        }
        break;

      case SyncEventType.PROVIDER_UPDATED:
        if (event.data) {
          const { provider } = event.data as { provider: ProviderSyncInfo };
          this.state.providers.set(provider.id, provider);
          // Update provider's models
          for (const model of provider.models) {
            this.state.models.set(model.id, model);
          }
          this.handlers.onProviderUpdated?.(event as SyncEvent<ProviderUpdatedEventData>);
        }
        break;

      case SyncEventType.PROVIDER_REMOVED:
        if (event.provider_id) {
          const provider = this.state.providers.get(event.provider_id);
          if (provider) {
            // Remove provider's models
            for (const model of provider.models) {
              this.state.models.delete(model.id);
            }
          }
          this.state.providers.delete(event.provider_id);
          this.handlers.onProviderRemoved?.(event as SyncEvent<ProviderRemovedEventData>);
        }
        break;

      case SyncEventType.PROVIDER_STATUS_CHANGED:
        if (event.provider_id && event.data) {
          const provider = this.state.providers.get(event.provider_id);
          if (provider) {
            const { new_status, health } = event.data as {
              new_status: string;
              health?: ProviderSyncInfo['health'];
            };
            provider.health = health;
          }
          this.handlers.onProviderStatusChanged?.(event as SyncEvent<ProviderStatusChangedEventData>);
        }
        break;

      case SyncEventType.MODEL_DEPRECATED:
        if (event.model_id && event.data) {
          const model = this.state.models.get(event.model_id);
          if (model) {
            const { deprecation_date, sunset_date, replacement_model_id } = event.data as {
              deprecation_date: string;
              sunset_date?: string;
              replacement_model_id?: string;
            };
            model.deprecation_date = deprecation_date;
            model.sunset_date = sunset_date;
            model.replacement_model_id = replacement_model_id;
          }
          this.handlers.onModelDeprecated?.(event as SyncEvent<ModelDeprecatedEventData>);
        }
        break;

      case SyncEventType.ACTIVE_PROVIDER_CHANGED:
        if (event.provider_id) {
          this.state.activeProviderId = event.provider_id;
        }
        this.handlers.onActiveProviderChanged?.(event as SyncEvent<ActiveProviderChangedEventData>);
        break;

      case SyncEventType.ACTIVE_MODEL_CHANGED:
        if (event.data) {
          const { new_model_id } = event.data as { new_model_id: string };
          this.state.activeModelId = new_model_id;
        }
        break;

      case SyncEventType.HEARTBEAT:
        this.handlers.onHeartbeat?.(event as SyncEvent<HeartbeatEventData>);
        break;

      case SyncEventType.ERROR:
        this.handlers.onError?.(event as SyncEvent<ErrorEventData>);
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // WebSocket Management
  // ---------------------------------------------------------------------------

  private connectWebSocket(): void {
    if (this.isDestroyed || this.ws?.readyState === WebSocket.OPEN) return;

    console.log('[ProviderSync] Connecting WebSocket...');

    try {
      // Build WebSocket URL
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = this.config.wsUrl.startsWith('ws')
        ? this.config.wsUrl
        : `${wsProtocol}//${window.location.host}${this.config.wsUrl}`;

      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[ProviderSync] WebSocket connected');
        this.state.isConnected = true;
        this.state.reconnectAttempts = 0;
        this.state.status = SyncStatus.SYNCED;
        this.stopPolling();
        this.startHeartbeat();
        this.handlers.onConnectionChange?.(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;

          if (message.type === 'pong') {
            // Heartbeat response, ignore
            return;
          }

          // Handle as sync event
          this.handleSyncEvent({
            type: message.type as SyncEventType,
            timestamp: message.timestamp || new Date().toISOString(),
            version: message.version || this.state.version,
            data: message.data,
          });
        } catch (error) {
          console.error('[ProviderSync] Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('[ProviderSync] WebSocket closed:', event.code, event.reason);
        this.state.isConnected = false;
        this.stopHeartbeat();
        this.handlers.onConnectionChange?.(false);

        if (!this.isDestroyed) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = () => {
        console.error('[ProviderSync] WebSocket error');
        // Error details are not available for security reasons
        // The onclose handler will be called after this
      };
    } catch (error) {
      console.error('[ProviderSync] Failed to create WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  private disconnectWebSocket(): void {
    if (this.ws) {
      this.ws.onclose = null; // Prevent reconnect
      this.ws.close();
      this.ws = null;
    }
    this.state.isConnected = false;
  }

  private scheduleReconnect(): void {
    if (this.isDestroyed) return;

    this.state.reconnectAttempts++;

    if (this.state.reconnectAttempts > this.config.maxReconnectAttempts) {
      console.log('[ProviderSync] Max reconnect attempts reached, falling back to polling');
      this.state.status = SyncStatus.DISCONNECTED;
      this.startPolling();
      return;
    }

    // Exponential backoff
    const delay = Math.min(
      this.config.reconnectBaseDelay * Math.pow(2, this.state.reconnectAttempts - 1),
      this.config.reconnectMaxDelay
    );

    console.log(`[ProviderSync] Reconnecting in ${delay}ms (attempt ${this.state.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.connectWebSocket();
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Polling Fallback
  // ---------------------------------------------------------------------------

  private startPolling(): void {
    if (this.pollingInterval) return;

    console.log('[ProviderSync] Starting polling fallback');

    this.pollingInterval = setInterval(() => {
      this.performIncrementalSync();
    }, this.config.pollingInterval);
  }

  private stopPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let syncServiceInstance: ProviderSyncService | null = null;

export function getProviderSyncService(
  config?: Partial<ProviderSyncConfig>
): ProviderSyncService {
  if (!syncServiceInstance) {
    // Build config from environment - use direct backend URL
    // NEXT_PUBLIC_CHIMERA_API_URL should be set to http://localhost:8001/api/v1
    const envUrl = process.env.NEXT_PUBLIC_CHIMERA_API_URL;
    const apiBaseUrl = envUrl || 'http://localhost:8001/api/v1';
    const wsBaseUrl = apiBaseUrl.replace(/^http/, 'ws');

    // Check if apiBaseUrl already contains /api/v1
    const hasApiV1 = apiBaseUrl.includes('/api/v1');
    const syncPath = hasApiV1 ? '/provider-sync' : '/api/v1/provider-sync';

    const finalApiUrl = `${apiBaseUrl}${syncPath}`;
    const finalWsUrl = `${wsBaseUrl}${syncPath}/ws`;

    console.log('[ProviderSync] Initializing with config:', {
      envUrl,
      apiBaseUrl,
      finalApiUrl,
      finalWsUrl,
    });

    syncServiceInstance = new ProviderSyncService({
      apiBaseUrl: finalApiUrl,
      wsUrl: finalWsUrl,
      ...config,
    });
  }
  return syncServiceInstance;
}

export function destroyProviderSyncService(): void {
  if (syncServiceInstance) {
    syncServiceInstance.destroy();
    syncServiceInstance = null;
  }
}
