# Provider Synchronization System

## Overview

The Provider Synchronization System enables real-time synchronization of AI provider configurations and model specifications between the backend and frontend. It provides:

- **Full and incremental sync** - Initial full state load with subsequent incremental updates
- **Real-time updates via WebSocket** - Instant notifications when providers/models change
- **Polling fallback** - Automatic fallback to polling when WebSocket is unavailable
- **Version-based conflict resolution** - Ensures data consistency across clients
- **Local caching** - Offline support and faster initial loads
- **Type-safe interfaces** - Shared types between backend and frontend

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ ProviderSync    │  │ useProviderSync │  │ SyncStatus      │ │
│  │ Context         │  │ Hooks           │  │ Indicator       │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                 │
│                    ┌───────────▼───────────┐                    │
│                    │ ProviderSyncService   │                    │
│                    │ - WebSocket client    │                    │
│                    │ - Polling fallback    │                    │
│                    │ - Local cache         │                    │
│                    └───────────┬───────────┘                    │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    HTTP / WebSocket     │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                         Backend                                  │
├────────────────────────────────┼────────────────────────────────┤
│                    ┌───────────▼───────────┐                    │
│                    │ provider_sync.py      │                    │
│                    │ - REST endpoints      │                    │
│                    │ - WebSocket endpoint  │                    │
│                    └───────────┬───────────┘                    │
│                                │                                 │
│           ┌────────────────────┼────────────────────┐           │
│           │                    │                    │           │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐ │
│  │ ProviderSync    │  │ ProviderMgmt    │  │ sync_models.py  │ │
│  │ Service         │  │ Service         │  │ Domain Models   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Backend Components

### Domain Models (`backend-api/app/domain/sync_models.py`)

Defines all sync-related data structures:

```python
# Event types for sync operations
class SyncEventType(str, Enum):
    FULL_SYNC = "full_sync"
    INCREMENTAL_UPDATE = "incremental_update"
    PROVIDER_ADDED = "provider_added"
    PROVIDER_UPDATED = "provider_updated"
    PROVIDER_REMOVED = "provider_removed"
    PROVIDER_STATUS_CHANGED = "provider_status_changed"
    MODEL_DEPRECATED = "model_deprecated"
    # ... more events

# Sync status
class SyncStatus(str, Enum):
    SYNCED = "synced"
    SYNCING = "syncing"
    STALE = "stale"
    ERROR = "error"
    DISCONNECTED = "disconnected"

# Model specification with full details
class ModelSpecification(BaseModel):
    id: str
    name: str
    provider_id: str
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    supports_streaming: bool
    supports_vision: bool
    deprecation_status: ModelDeprecationStatus
    # ... more fields

# Complete sync state
class SyncState(BaseModel):
    providers: list[ProviderSyncInfo]
    all_models: list[ModelSpecification]
    active_provider_id: Optional[str]
    metadata: SyncMetadata
```

### Sync Service (`backend-api/app/services/provider_sync_service.py`)

Handles sync logic and event broadcasting:

```python
class ProviderSyncService:
    async def get_full_sync_state(self, providers, active_provider_id, default_provider_id) -> SyncState:
        """Build complete sync state for initial load"""
        
    async def handle_sync_request(self, request, providers, ...) -> SyncResponse:
        """Handle full or incremental sync requests"""
        
    async def notify_provider_added(self, provider):
        """Broadcast provider added event"""
        
    async def notify_model_deprecated(self, model, deprecation_date, ...):
        """Broadcast model deprecation event"""
```

### API Router (`backend-api/app/api/routes/provider_sync.py`)

REST and WebSocket endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/provider-sync/state` | GET | Get current sync state |
| `/provider-sync/sync` | POST | Perform sync operation |
| `/provider-sync/version` | GET | Get current sync version |
| `/provider-sync/providers/{id}/availability` | GET | Get provider availability |
| `/provider-sync/models/{id}/availability` | GET | Get model availability |
| `/provider-sync/models` | GET | List all models with filters |
| `/provider-sync/ws` | WebSocket | Real-time sync updates |

## Frontend Components

### Types (`frontend/src/types/provider-sync.ts`)

TypeScript interfaces mirroring backend models:

```typescript
// Enums
export enum SyncEventType { ... }
export enum SyncStatus { ... }
export enum ModelDeprecationStatus { ... }

// Interfaces
export interface ModelSpecification { ... }
export interface ProviderSyncInfo { ... }
export interface SyncState { ... }
export interface SyncEvent<T = unknown> { ... }

// Configuration
export interface ProviderSyncConfig {
  apiBaseUrl: string;
  wsUrl: string;
  enableWebSocket: boolean;
  pollingInterval: number;
  maxReconnectAttempts: number;
  // ...
}
```

### Sync Service (`frontend/src/lib/sync/provider-sync-service.ts`)

Core synchronization logic:

```typescript
class ProviderSyncService {
  // Initialize and start syncing
  async initialize(): Promise<void>
  
  // Get current state
  getState(): ProviderSyncClientState
  getProviders(): ProviderSyncInfo[]
  getModels(): ModelSpecification[]
  
  // Force manual sync
  async forceSync(): Promise<void>
  
  // Cleanup
  destroy(): void
}

// Singleton access
export function getProviderSyncService(config?): ProviderSyncService
```

### React Context (`frontend/src/contexts/ProviderSyncContext.tsx`)

React integration with hooks:

```tsx
// Provider component
<ProviderSyncProvider config={...} onSyncEvent={...}>
  {children}
</ProviderSyncProvider>

// Main hook
const {
  status,
  isConnected,
  providers,
  models,
  activeProvider,
  forceSync,
} = useProviderSync();

// Selector hooks
const providers = useProviders({ enabledOnly: true });
const models = useModels({ providerId: 'openai' });
const { isSynced, hasError } = useSyncStatus();
```

### React Query Hooks (`frontend/src/hooks/useProviderSync.ts`)

TanStack Query integration:

```typescript
// Get sync state with caching
const { data, isLoading } = useSyncState();

// Get specific provider
const { provider } = useProvider('openai');

// Get models with filtering
const { models } = useModelsFromState({ 
  providerId: 'openai',
  excludeDeprecated: true 
});

// WebSocket connection
const { isConnected } = useSyncWebSocket({
  onEvent: (event) => console.log(event),
});

// Combined hook with WebSocket
const { providers, models, isConnected, forceSync } = useProviderSyncWithWebSocket();
```

### UI Components (`frontend/src/components/providers/SyncStatusIndicator.tsx`)

Visual indicators:

```tsx
// Sync status indicator
<SyncStatusIndicator
  status={SyncStatus.SYNCED}
  isConnected={true}
  lastSyncTime={new Date()}
  version={42}
  onSync={handleSync}
  showDetails
/>

// Model deprecation warning
<ModelDeprecationWarning
  modelId="gpt-3.5-turbo"
  modelName="GPT-3.5 Turbo"
  deprecationDate="2024-06-01"
  sunsetDate="2024-12-01"
  replacementModelId="gpt-4-turbo"
  onSelectReplacement={handleSelect}
/>

// Provider unavailable warning
<ProviderUnavailableWarning
  providerId="openai"
  providerName="OpenAI"
  status="unavailable"
  fallbackProviderId="anthropic"
  onSelectFallback={handleFallback}
/>
```

## Sync Flow

### Initial Sync

1. Frontend calls `GET /provider-sync/state`
2. Backend builds complete `SyncState` with all providers and models
3. Frontend caches state locally and updates UI
4. WebSocket connection established for real-time updates

### Incremental Sync

1. Frontend sends `POST /provider-sync/sync` with `client_version`
2. Backend compares versions:
   - If versions match: returns empty response (no changes)
   - If versions differ slightly: returns incremental events
   - If versions differ significantly: returns full state
3. Frontend applies changes and updates version

### Real-time Updates

1. Backend detects change (provider added, model deprecated, etc.)
2. Backend broadcasts `SyncEvent` to all WebSocket clients
3. Frontend receives event and updates local state
4. React Query cache invalidated for affected queries

### Fallback Polling

1. WebSocket connection fails or is unavailable
2. Service automatically switches to polling mode
3. Polls `GET /provider-sync/version` every 30 seconds
4. If version changed, performs incremental sync
5. Attempts WebSocket reconnection with exponential backoff

## Configuration

### Backend Configuration

```python
# In settings or environment
SYNC_HEARTBEAT_INTERVAL = 25  # seconds
SYNC_VERSION_CHECK_INTERVAL = 30  # seconds
```

### Frontend Configuration

```typescript
const config: ProviderSyncConfig = {
  apiBaseUrl: '/api/provider-sync',
  wsUrl: '/api/provider-sync/ws',
  enableWebSocket: true,
  pollingInterval: 30000,  // 30 seconds
  maxReconnectAttempts: 5,
  reconnectBaseDelay: 1000,  // 1 second
  reconnectMaxDelay: 30000,  // 30 seconds
  heartbeatInterval: 25000,  // 25 seconds
  syncTimeout: 10000,  // 10 seconds
  includeDeprecated: false,
  enableCache: true,
  cacheTtl: 300000,  // 5 minutes
};
```

## Error Handling

### Network Errors

- Automatic retry with exponential backoff
- Graceful degradation to polling
- User notification via `SyncStatusIndicator`

### Version Conflicts

- Server always wins (authoritative)
- Client re-syncs on version mismatch
- No data loss due to full state availability

### Provider Unavailability

- Health checks detect unavailable providers
- Fallback provider suggestions
- UI warnings with action buttons

### Model Deprecation

- Deprecation warnings with dates
- Replacement model suggestions
- Urgent warnings for imminent sunset

## Usage Examples

### Basic Usage

```tsx
// In your app layout
import { ProviderSyncProvider } from '@/contexts/ProviderSyncContext';

function App() {
  return (
    <ProviderSyncProvider>
      <YourApp />
    </ProviderSyncProvider>
  );
}

// In a component
import { useProviderSync } from '@/contexts/ProviderSyncContext';

function ProviderList() {
  const { providers, isLoading, error } = useProviderSync();
  
  if (isLoading) return <Spinner />;
  if (error) return <Error message={error} />;
  
  return (
    <ul>
      {providers.map(p => (
        <li key={p.id}>{p.display_name}</li>
      ))}
    </ul>
  );
}
```

### With React Query

```tsx
import { useProviderSyncWithWebSocket } from '@/hooks/useProviderSync';

function ModelSelector() {
  const { models, isConnected, forceSync } = useProviderSyncWithWebSocket();
  
  return (
    <div>
      <SyncStatusIndicator isConnected={isConnected} onSync={forceSync} />
      <Select>
        {models.map(m => (
          <SelectItem key={m.id} value={m.id}>
            {m.name}
          </SelectItem>
        ))}
      </Select>
    </div>
  );
}
```

### Handling Deprecation

```tsx
import { useModel, useModelAvailability } from '@/hooks/useProviderSync';
import { ModelDeprecationWarning } from '@/components/providers/SyncStatusIndicator';

function ModelInfo({ modelId }: { modelId: string }) {
  const { model } = useModel(modelId);
  const { data: availability } = useModelAvailability(modelId);
  
  if (!model) return null;
  
  return (
    <div>
      <h3>{model.name}</h3>
      
      {model.deprecation_status !== 'active' && (
        <ModelDeprecationWarning
          modelId={model.id}
          modelName={model.name}
          deprecationDate={model.deprecation_date}
          sunsetDate={model.sunset_date}
          replacementModelId={availability?.replacement_model_id}
          replacementModelName={availability?.replacement_model_name}
        />
      )}
    </div>
  );
}
```

## Testing

### Backend Tests

```bash
# Run sync-related tests
poetry run pytest backend-api/tests/test_provider_sync.py -v
```

### Frontend Tests

```bash
# Run sync hook tests
cd frontend && npx vitest run src/hooks/useProviderSync.test.ts
```

## Troubleshooting

### WebSocket Not Connecting

1. Check CORS configuration allows WebSocket upgrade
2. Verify WebSocket URL is correct (ws:// vs wss://)
3. Check browser console for connection errors
4. Ensure backend WebSocket endpoint is registered

### Sync State Stale

1. Check network connectivity
2. Verify polling is active (check console logs)
3. Force sync using `forceSync()` function
4. Clear local cache and reload

### Version Mismatch Loops

1. Check server logs for sync errors
2. Verify backend version incrementing correctly
3. Clear frontend cache
4. Restart backend service

## API Reference

See the OpenAPI documentation at `/docs` for complete API reference.