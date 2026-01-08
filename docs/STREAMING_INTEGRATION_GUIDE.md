# Streaming Integration Guide

This guide documents the unified streaming system for Project Chimera, covering architecture, API endpoints, frontend hooks, and best practices.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Backend Services](#backend-services)
3. [API Endpoints Reference](#api-endpoints-reference)
4. [Frontend Hooks](#frontend-hooks)
5. [Stream Formats](#stream-formats)
6. [Error Handling](#error-handling)
7. [Metrics and Monitoring](#metrics-and-monitoring)
8. [Best Practices](#best-practices)

---

## Architecture Overview

The streaming system integrates with the Unified Provider/Model Selection System to ensure consistent provider and model resolution across all streaming operations.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend                                    │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │ useStreamingGeneration│◄───│ useUnifiedProviderSelection      │  │
│  │ useWebSocketStreaming │    └──────────────────────────────────┘  │
│  └──────────┬───────────┘                                           │
│             │ SSE/WebSocket                                         │
└─────────────┼───────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────────┐
│                          Backend API                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Streaming Endpoints                        │   │
│  │  POST /stream/generate     POST /stream/chat                 │   │
│  │  POST /stream/generate/jsonl                                 │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │              UnifiedStreamingService                          │   │
│  │  - Provider/model resolution                                 │   │
│  │  - Stream lifecycle management                               │   │
│  │  - Metrics collection                                        │   │
│  └──────────────────────────────┬───────────────────────────────┘   │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────────┐   │
│  │              StreamingContext                                 │   │
│  │  - Selection locking                                         │   │
│  │  - Session tracking                                          │   │
│  │  - Concurrent stream management                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Provider Resolution Hierarchy**: Explicit → Session → User → Global
2. **Selection Locking**: Prevents selection changes during active streams
3. **Stream Tracing**: Headers for debugging and observability
4. **Multi-Format Support**: SSE, JSON Lines, WebSocket

---

## Backend Services

### UnifiedStreamingService

Location: [`backend-api/app/services/unified_streaming_service.py`](../backend-api/app/services/unified_streaming_service.py)

The core service that handles all streaming operations with unified provider selection.

```python
from app.services.unified_streaming_service import get_unified_streaming_service

service = get_unified_streaming_service()

# Stream generation
async for chunk in service.stream_generate(
    prompt="Explain quantum computing",
    session_id="session-123",
    explicit_provider=None,  # Uses session/global selection
    explicit_model=None,
    temperature=0.7,
    max_tokens=1000,
):
    print(chunk.text)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `stream_generate()` | Async generator for text generation |
| `stream_chat()` | Async generator for chat completion |
| `get_active_streams()` | List currently active streams |
| `get_metrics_summary()` | Get aggregate streaming metrics |
| `cancel_stream()` | Cancel an active stream by ID |

### StreamingContext

Location: [`backend-api/app/services/stream_context.py`](../backend-api/app/services/stream_context.py)

Manages stream lifecycle and selection locking.

```python
from app.services.stream_context import get_streaming_context

context = get_streaming_context()

# Create a locked stream session
async with context.locked_stream(
    session_id="session-123",
    provider="openai",
    model="gpt-4o",
) as session:
    # Selection is locked for this session
    # Other streams can read but not modify selection
    session.record_chunk(100)  # Record bytes sent
```

### StreamResponseFormatter

Location: [`backend-api/app/services/stream_formatters.py`](../backend-api/app/services/stream_formatters.py)

Handles formatting of stream chunks for different output formats.

```python
from app.services.stream_formatters import (
    StreamResponseFormatter,
    StreamFormat,
    StreamingHeadersBuilder,
)

formatter = StreamResponseFormatter()

# Format as SSE
sse_chunk = formatter.format_sse(
    text="Hello",
    chunk_index=0,
    stream_id="stream-123",
    provider="openai",
    model="gpt-4o",
    is_final=False,
)

# Build response headers
headers = StreamingHeadersBuilder.build_sse_headers(
    provider="openai",
    model="gpt-4o",
    session_id="session-123",
    stream_id="stream-456",
)
```

### StreamMetricsService

Location: [`backend-api/app/services/stream_metrics_service.py`](../backend-api/app/services/stream_metrics_service.py)

Collects and aggregates streaming performance metrics.

```python
from app.services.stream_metrics_service import get_stream_metrics_service

metrics = get_stream_metrics_service()

# Get provider-specific metrics
provider_stats = metrics.get_provider_metrics("openai")

# Get recent errors
errors = metrics.get_recent_errors(limit=10)
```

---

## API Endpoints Reference

### POST /api/v1/stream/generate

Stream text generation using SSE format.

**Request:**

```json
{
  "prompt": "Explain quantum computing in simple terms",
  "system_instruction": "You are a helpful science teacher",
  "temperature": 0.7,
  "max_tokens": 1000,
  "provider": null,
  "model": null
}
```

**Headers:**

| Header | Description |
|--------|-------------|
| `X-Session-Id` | Session ID for session-scoped selection |
| `X-User-Id` | User ID for user-scoped selection |

**Response Headers:**

| Header | Description |
|--------|-------------|
| `X-Stream-Provider` | Resolved provider |
| `X-Stream-Model` | Resolved model |
| `X-Stream-Session-Id` | Session ID used |
| `X-Stream-Id` | Unique stream identifier |
| `X-Stream-Started-At` | ISO timestamp of stream start |
| `X-Stream-Resolution-Source` | Where selection came from |
| `X-Stream-Resolution-Priority` | Priority level (lower = higher priority) |

**Response (SSE):**

```
event: chunk
id: 0
data: {"text":"Quantum","chunk_index":0,"stream_id":"stream_abc123","provider":"openai","model":"gpt-4o","is_final":false,"timestamp":"2024-01-01T00:00:00Z"}

event: chunk
id: 1
data: {"text":" computing","chunk_index":1,"stream_id":"stream_abc123","provider":"openai","model":"gpt-4o","is_final":false,"timestamp":"2024-01-01T00:00:00Z"}

event: done
data: {"stream_id":"stream_abc123","total_chunks":10,"finish_reason":"stop"}
```

### POST /api/v1/stream/chat

Stream chat completion using SSE format.

**Request:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "provider": null,
  "model": null
}
```

### POST /api/v1/stream/generate/jsonl

Stream text generation using JSON Lines format.

**Response:**

```json
{"text":"Quantum","chunk_index":0,"stream_id":"stream_abc123","provider":"openai","model":"gpt-4o","is_final":false}
{"text":" computing","chunk_index":1,"stream_id":"stream_abc123","provider":"openai","model":"gpt-4o","is_final":false}
{"text":" is...","chunk_index":2,"stream_id":"stream_abc123","provider":"openai","model":"gpt-4o","is_final":true,"finish_reason":"stop"}
```

### GET /api/v1/stream/active

Get currently active streams.

**Response:**

```json
{
  "active_streams": [
    {
      "stream_id": "stream_abc123",
      "session_id": "session-456",
      "provider": "openai",
      "model": "gpt-4o",
      "started_at": "2024-01-01T00:00:00Z",
      "chunks_sent": 15
    }
  ],
  "count": 1
}
```

### GET /api/v1/stream/metrics

Get streaming metrics summary.

**Response:**

```json
{
  "total_streams": 1000,
  "active_streams": 5,
  "by_provider": {
    "openai": 600,
    "anthropic": 300,
    "google": 100
  },
  "error_rate": 0.02,
  "avg_duration_ms": 2500,
  "avg_first_chunk_latency_ms": 150
}
```

### DELETE /api/v1/stream/{stream_id}

Cancel an active stream.

**Response:**

```json
{
  "status": "cancelled",
  "stream_id": "stream_abc123"
}
```

---

## Frontend Hooks

### useStreamingGeneration

Location: [`frontend/src/hooks/useStreamingGeneration.ts`](../frontend/src/hooks/useStreamingGeneration.ts)

Primary hook for SSE-based streaming.

```tsx
import { useStreamingGeneration } from '@/hooks';

function MyComponent() {
  const {
    state,           // 'idle' | 'streaming' | 'completed' | 'error' | 'cancelled'
    isStreaming,     // boolean
    currentText,     // accumulated text
    streamId,        // current stream ID
    error,           // StreamError | null
    chunkIndex,      // number of chunks received
    streamMetadata,  // metadata from response headers
    streamGenerate,  // start generation
    streamChat,      // start chat
    abortStream,     // cancel current stream
    reset,           // reset state
  } = useStreamingGeneration({
    sessionId: 'my-session',     // optional
    provider: 'openai',          // optional override
    model: 'gpt-4o',             // optional override
    timeout: 300000,             // 5 minutes
    useJsonl: false,             // use SSE format
  });

  const handleGenerate = async () => {
    await streamGenerate(
      { prompt: 'Hello, world!', temperature: 0.7 },
      (chunk) => console.log('Chunk:', chunk.text),
      (result) => console.log('Done:', result.fullText),
      (error) => console.error('Error:', error.message)
    );
  };

  return (
    <div>
      <button onClick={handleGenerate} disabled={isStreaming}>
        Generate
      </button>
      <button onClick={abortStream} disabled={!isStreaming}>
        Cancel
      </button>
      <div>{currentText}</div>
    </div>
  );
}
```

**Type Definitions:**

```typescript
interface StreamChunk {
  text: string;
  chunk_index: number;
  stream_id: string;
  provider: string;
  model: string;
  is_final: boolean;
  finish_reason?: string;
  token_count?: number;
  timestamp: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface StreamResult {
  fullText: string;
  streamId: string;
  provider: string;
  model: string;
  totalChunks: number;
  totalTokens?: number;
  durationMs: number;
  finishReason?: string;
  usage?: StreamChunk['usage'];
  metadata?: StreamMetadata;
}

interface StreamMetadata {
  provider: string | null;
  model: string | null;
  sessionId: string | null;
  streamId: string | null;
  startedAt: string | null;
  resolutionSource: string | null;
  resolutionPriority: number | null;
}
```

### useWebSocketStreaming

Location: [`frontend/src/hooks/useWebSocketStreaming.ts`](../frontend/src/hooks/useWebSocketStreaming.ts)

Hook for WebSocket-based streaming with bi-directional communication.

```tsx
import { useWebSocketStreaming } from '@/hooks';

function MyComponent() {
  const {
    isConnected,
    isStreaming,
    currentText,
    error,
    startGeneration,
    stopGeneration,
    disconnect,
  } = useWebSocketStreaming({
    url: '/api/v1/ws/stream',
    sessionId: 'my-session',
    onChunk: (chunk) => console.log('Chunk:', chunk),
    onComplete: (result) => console.log('Done:', result),
    onError: (error) => console.error('Error:', error),
  });

  return (
    <div>
      {isConnected ? 'Connected' : 'Disconnected'}
      <button onClick={() => startGeneration({ prompt: 'Hello' })}>
        Start
      </button>
    </div>
  );
}
```

---

## Stream Formats

### Server-Sent Events (SSE)

Default format for unidirectional streaming.

**Event Types:**

| Event | Description |
|-------|-------------|
| `chunk` | Text chunk with metadata |
| `done` | Stream completion with summary |
| `error` | Error occurred |
| `ping` | Keep-alive (`:` comment) |

**Example:**

```
event: chunk
id: 0
data: {"text":"Hello","chunk_index":0,...}

event: done
data: {"stream_id":"...","total_chunks":5}
```

### JSON Lines (JSONL)

Alternative format for simpler parsing.

**Example:**

```
{"text":"Hello","chunk_index":0,"is_final":false,...}
{"text":" world","chunk_index":1,"is_final":true,...}
```

### WebSocket

Bi-directional format for real-time interaction.

**Message Types:**

| Type | Direction | Description |
|------|-----------|-------------|
| `start` | Client → Server | Start streaming |
| `chunk` | Server → Client | Text chunk |
| `done` | Server → Client | Stream complete |
| `cancel` | Client → Server | Cancel stream |
| `error` | Server → Client | Error occurred |

---

## Error Handling

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `PROVIDER_NOT_AVAILABLE` | 503 | Provider not configured or unavailable |
| `MODEL_NOT_FOUND` | 404 | Model not found for provider |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit hit (includes Retry-After) |
| `STREAM_INTERRUPTED` | 500 | Unexpected stream interruption |
| `CANCELLED` | N/A | Client cancelled stream |
| `STREAM_ERROR` | 500 | Generic stream error |

### Frontend Error Handling

```tsx
const { streamGenerate, error } = useStreamingGeneration();

// Handle errors in callback
await streamGenerate(
  { prompt: 'Hello' },
  null,
  null,
  (error) => {
    switch (error.code) {
      case 'PROVIDER_NOT_AVAILABLE':
        showNotification('Provider is not available');
        break;
      case 'RATE_LIMIT_EXCEEDED':
        showNotification('Rate limit hit, please wait');
        break;
      case 'CANCELLED':
        // User cancelled, no notification needed
        break;
      default:
        showNotification(`Error: ${error.message}`);
    }
  }
);

// Or check error state
if (error) {
  console.error(`Stream error (${error.code}): ${error.message}`);
}
```

### Backend Error Handling

```python
from app.api.v1.endpoints.streaming import (
    ProviderNotAvailableError,
    ModelNotFoundError,
    RateLimitExceededError,
)

try:
    async for chunk in streaming_service.stream_generate(...):
        yield chunk
except ProviderNotAvailableError as e:
    logger.error(f"Provider {e.provider} not available")
    raise HTTPException(
        status_code=503,
        detail={"error": "PROVIDER_NOT_AVAILABLE", "provider": e.provider}
    )
```

---

## Metrics and Monitoring

### Available Metrics

| Metric | Description |
|--------|-------------|
| `total_streams` | Total streams processed |
| `active_streams` | Currently active streams |
| `by_provider` | Stream counts by provider |
| `error_rate` | Percentage of failed streams |
| `avg_duration_ms` | Average stream duration |
| `avg_first_chunk_latency_ms` | Time to first chunk |

### Alert Callbacks

```python
from app.services.stream_metrics_service import get_stream_metrics_service

metrics = get_stream_metrics_service()

# Register alert callbacks
metrics.on_high_error_rate(
    threshold=0.1,
    callback=lambda provider, rate: send_alert(f"High error rate for {provider}: {rate}")
)

metrics.on_high_latency(
    threshold_ms=5000,
    callback=lambda provider, latency: send_alert(f"High latency for {provider}: {latency}ms")
)
```

### Monitoring Dashboard Integration

```typescript
// Fetch metrics for dashboard
const response = await fetch('/api/v1/stream/metrics');
const metrics = await response.json();

// Display in dashboard
updateDashboard({
  totalStreams: metrics.total_streams,
  activeStreams: metrics.active_streams,
  errorRate: metrics.error_rate,
  avgLatency: metrics.avg_first_chunk_latency_ms,
});
```

---

## Best Practices

### 1. Use Session IDs Consistently

```tsx
// Always pass session ID for consistent selection
const { streamGenerate } = useStreamingGeneration({
  sessionId: userSession.id,
});
```

### 2. Handle Cancellation Gracefully

```tsx
useEffect(() => {
  return () => {
    // Clean up on component unmount
    abortStream();
  };
}, [abortStream]);
```

### 3. Show Progress Feedback

```tsx
<div>
  {isStreaming && (
    <div>
      <Spinner /> Generating... ({chunkIndex} chunks)
    </div>
  )}
  <div className="whitespace-pre-wrap">
    {currentText}
    {isStreaming && <Cursor />}
  </div>
</div>
```

### 4. Retry on Transient Errors

```tsx
const handleGenerate = async (retryCount = 0) => {
  try {
    await streamGenerate({ prompt });
  } catch (error) {
    if (error.code === 'RATE_LIMIT_EXCEEDED' && retryCount < 3) {
      await delay(error.retryAfter * 1000);
      return handleGenerate(retryCount + 1);
    }
    throw error;
  }
};
```

### 5. Monitor and Log

```tsx
streamGenerate(
  { prompt },
  (chunk) => {
    // Log for debugging
    console.debug(`Chunk ${chunk.chunk_index}: ${chunk.text.length} chars`);
  },
  (result) => {
    // Log performance metrics
    console.info(`Stream complete: ${result.totalChunks} chunks in ${result.durationMs}ms`);
    analytics.track('stream_complete', {
      provider: result.provider,
      model: result.model,
      duration: result.durationMs,
      tokens: result.totalTokens,
    });
  }
);
```

---

## See Also

- [Unified Provider Selection Guide](./UNIFIED_PROVIDER_SELECTION.md)
- [API Reference](./API_REFERENCE.md)
- [Hooks Reference](./HOOKS_REFERENCE.md)

---

*Last updated: January 2026*
