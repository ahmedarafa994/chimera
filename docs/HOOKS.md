# Frontend Hooks Documentation

## Overview

Chimera provides a set of custom React hooks to simplify interaction with the API, manage state, and handle real-time data.

## Core Hooks

### `useChimera`

The primary hook for accessing the global application state.

```typescript
import { useChimera } from '@/providers/chimera-provider';

function MyComponent() {
  const { 
    session, 
    provider, 
    circuitBreaker,
    connection 
  } = useChimera();

  return (
    <div>
      <p>Connection Status: {connection.isConnected ? 'Connected' : 'Disconnected'}</p>
      <p>Current Provider: {provider.activeProvider?.name}</p>
    </div>
  );
}
```

**Returns:**

- `session`: Current session state (id, messages, etc.)
- `provider`: Provider management state (active provider, list, health)
- `circuitBreaker`: Circuit breaker registry state
- `connection`: WebSocket connection state

---

### `useProviders`

Hook for fetching and managing LLM providers.

```typescript
import { useProviders } from '@/lib/hooks/use-providers';

function ProviderSelector() {
  const { 
    providers, 
    isLoading, 
    error, 
    refresh 
  } = useProviders();

  if (isLoading) return <div>Loading...</div>;

  return (
    <ul>
      {providers.map(p => <li key={p.id}>{p.name}</li>)}
    </ul>
  );
}
```

**Features:**

- Automatic caching with SWR/TanStack Query
- Real-time health updates
- Error handling integration

---

### `useSession`

Hook for managing the chat/attack session.

```typescript
import { useSession } from '@/lib/hooks/use-session';

function ChatInterface() {
  const { 
    messages, 
    sendMessage, 
    isTyping, 
    clearSession 
  } = useSession();

  const handleSend = async (text) => {
    await sendMessage(text, { model: 'gpt-4' });
  };

  return (
    // ... UI implementation
  );
}
```

**Methods:**

- `sendMessage(content, options)`: Send a message to the active session
- `createSession(config)`: Initialize a new session
- `clearSession()`: Reset current session
- `regenerateLast()`: Regenerate the last response

---

### `useTechniques`

Hook for accessing available jailbreak techniques.

```typescript
import { useTechniques } from '@/lib/hooks/use-techniques';

function TechniqueFilter() {
  const { 
    techniques,
    activeTechnique,
    setActiveTechnique 
  } = useTechniques();

  return (
    <select onChange={e => setActiveTechnique(e.target.value)}>
      {techniques.map(t => (
        <option key={t.id} value={t.id}>{t.name}</option>
      ))}
    </select>
  );
}
```

**Returns:**

- List of available techniques
- Metadata for each technique (success rate, description)
- Helper methods for filtering

---

### `useModels`

Hook for fetching available models, optionally filtered by provider.

```typescript
import { useModels } from '@/lib/hooks/use-models';

function ModelSelect({ providerId }) {
  const { models, isLoading } = useModels(providerId);

  return (
    // ... dropdown implementation
  );
}
```

## Best Practices

1. **Error Handling**: All hooks integrate with the global error handler. You normally don't need `try/catch` blocks for hook actions unless you need custom local handling.
2. **Suspense**: Hooks are configured to work with React Suspense where appropriate.
3. **State Sync**: `useChimera` ensures state is synchronized across the app. Avoid creating local state that duplicates global state.
