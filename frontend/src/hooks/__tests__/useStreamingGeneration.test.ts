/**
 * Unit tests for useStreamingGeneration hook
 *
 * Tests the streaming generation hook that integrates with the
 * unified provider/model selection system.
 *
 * @module hooks/__tests__/useStreamingGeneration.test
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useStreamingGeneration } from '../useStreamingGeneration';
import type {
  StreamChunk,
  StreamResult,
  StreamError,
  StreamMetadata,
} from '../useStreamingGeneration';

// =============================================================================
// Mock Setup
// =============================================================================

// Mock the unified provider selection hook
vi.mock('../useUnifiedProviderSelection', () => ({
  useUnifiedProviderSelection: () => ({
    selectedProvider: 'openai',
    selectedModel: 'gpt-4',
    providers: [],
    models: [],
    isLoadingProviders: false,
    isLoadingModels: false,
    isSaving: false,
    isSyncing: false,
    providersError: null,
    modelsError: null,
    syncError: null,
    selectProvider: vi.fn(),
    selectModel: vi.fn(),
    refreshProviders: vi.fn(),
    refreshModels: vi.fn(),
    clearSelection: vi.fn(),
    forceSync: vi.fn(),
    providerStatus: {},
    connectionStatus: 'connected',
    selectionVersion: 1,
    getProviderById: vi.fn(),
    getModelById: vi.fn(),
    getModelsForProvider: vi.fn(),
  }),
}));

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a wrapper component with QueryClientProvider
 */
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity,
      },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(
      QueryClientProvider,
      { client: queryClient },
      children
    );
  };
}

/**
 * Create a mock SSE stream
 */
function createMockSSEStream(chunks: Array<Partial<StreamChunk>>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;

  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        const chunk = chunks[index];
        const sseEvent = `event: chunk\ndata: ${JSON.stringify({
          text: chunk.text || '',
          chunk_index: chunk.chunk_index ?? index,
          stream_id: chunk.stream_id || 'test-stream-123',
          provider: chunk.provider || 'openai',
          model: chunk.model || 'gpt-4',
          is_final: chunk.is_final || index === chunks.length - 1,
          finish_reason: chunk.finish_reason,
          timestamp: chunk.timestamp || new Date().toISOString(),
          usage: chunk.usage,
        })}\n\n`;

        controller.enqueue(encoder.encode(sseEvent));
        index++;
      } else {
        // Send done event
        controller.enqueue(encoder.encode('event: done\ndata: {}\n\n'));
        controller.close();
      }
    },
  });
}

/**
 * Create a mock JSON Lines stream
 */
function createMockJsonlStream(chunks: Array<Partial<StreamChunk>>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;

  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        const chunk = chunks[index];
        const jsonLine = JSON.stringify({
          text: chunk.text || '',
          chunk_index: chunk.chunk_index ?? index,
          stream_id: chunk.stream_id || 'test-stream-123',
          provider: chunk.provider || 'openai',
          model: chunk.model || 'gpt-4',
          is_final: chunk.is_final || index === chunks.length - 1,
          finish_reason: chunk.finish_reason,
          timestamp: chunk.timestamp || new Date().toISOString(),
          usage: chunk.usage,
        }) + '\n';

        controller.enqueue(encoder.encode(jsonLine));
        index++;
      } else {
        controller.close();
      }
    },
  });
}

/**
 * Create mock response headers with stream metadata
 */
function createMockHeaders(metadata?: Partial<StreamMetadata>): Headers {
  const headers = new Headers({
    'Content-Type': 'text/event-stream',
  });

  if (metadata?.provider) {
    headers.set('X-Stream-Provider', metadata.provider);
  }
  if (metadata?.model) {
    headers.set('X-Stream-Model', metadata.model);
  }
  if (metadata?.sessionId) {
    headers.set('X-Stream-Session-Id', metadata.sessionId);
  }
  if (metadata?.streamId) {
    headers.set('X-Stream-Id', metadata.streamId);
  }
  if (metadata?.startedAt) {
    headers.set('X-Stream-Started-At', metadata.startedAt);
  }
  if (metadata?.resolutionSource) {
    headers.set('X-Stream-Resolution-Source', metadata.resolutionSource);
  }
  if (metadata?.resolutionPriority !== null && metadata?.resolutionPriority !== undefined) {
    headers.set('X-Stream-Resolution-Priority', String(metadata.resolutionPriority));
  }

  return headers;
}

// =============================================================================
// Tests
// =============================================================================

describe('useStreamingGeneration', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  // ---------------------------------------------------------------------------
  // Basic State Tests
  // ---------------------------------------------------------------------------

  describe('initial state', () => {
    it('should have correct initial state', () => {
      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      expect(result.current.state).toBe('idle');
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.currentText).toBe('');
      expect(result.current.streamId).toBeNull();
      expect(result.current.error).toBeNull();
      expect(result.current.chunkIndex).toBe(0);
      expect(result.current.streamMetadata).toBeNull();
    });

    it('should expose all expected methods', () => {
      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      expect(typeof result.current.streamGenerate).toBe('function');
      expect(typeof result.current.streamChat).toBe('function');
      expect(typeof result.current.abortStream).toBe('function');
      expect(typeof result.current.reset).toBe('function');
    });
  });

  // ---------------------------------------------------------------------------
  // SSE Parsing Tests
  // ---------------------------------------------------------------------------

  describe('SSE parsing', () => {
    it('should parse SSE events correctly', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Hello', chunk_index: 0 },
        { text: ' World', chunk_index: 1 },
        { text: '!', chunk_index: 2, is_final: true, finish_reason: 'stop' },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({
          provider: 'openai',
          model: 'gpt-4',
          streamId: 'test-stream-123',
        }),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const receivedChunks: StreamChunk[] = [];
      const onChunk = vi.fn((chunk: StreamChunk) => {
        receivedChunks.push(chunk);
      });
      const onComplete = vi.fn();

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test prompt' },
          onChunk,
          onComplete
        );
      });

      expect(receivedChunks.length).toBe(3);
      expect(receivedChunks[0].text).toBe('Hello');
      expect(receivedChunks[1].text).toBe(' World');
      expect(receivedChunks[2].text).toBe('!');
      expect(onComplete).toHaveBeenCalled();
    });

    it('should accumulate text correctly', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Hello', chunk_index: 0 },
        { text: ' World', chunk_index: 1 },
        { text: '!', chunk_index: 2, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      let finalResult: StreamResult | undefined;
      const onComplete = vi.fn((res: StreamResult) => {
        finalResult = res;
      });

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test prompt' },
          undefined,
          onComplete
        );
      });

      expect(result.current.currentText).toBe('Hello World!');
      expect(finalResult?.fullText).toBe('Hello World!');
    });
  });

  // ---------------------------------------------------------------------------
  // Header Extraction Tests
  // ---------------------------------------------------------------------------

  describe('header extraction', () => {
    it('should extract stream metadata from headers', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const metadata: StreamMetadata = {
        provider: 'anthropic',
        model: 'claude-3',
        sessionId: 'session-456',
        streamId: 'stream-789',
        startedAt: '2024-01-01T00:00:00Z',
        resolutionSource: 'session',
        resolutionPriority: 2,
      };

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders(metadata),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.streamGenerate({ prompt: 'Test' });
      });

      expect(result.current.streamMetadata).toEqual(metadata);
    });

    it('should handle missing headers gracefully', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: new Headers({ 'Content-Type': 'text/event-stream' }),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.streamGenerate({ prompt: 'Test' });
      });

      expect(result.current.streamMetadata).toEqual({
        provider: null,
        model: null,
        sessionId: null,
        streamId: null,
        startedAt: null,
        resolutionSource: null,
        resolutionPriority: null,
      });
    });
  });

  // ---------------------------------------------------------------------------
  // Abort Tests
  // ---------------------------------------------------------------------------

  describe('abort functionality', () => {
    it('should abort stream correctly', async () => {
      // Create a stream that never ends
      const slowStream = new ReadableStream({
        async pull(controller) {
          // Wait forever - will be aborted
          await new Promise(() => {});
        },
      });

      const mockResponse = new Response(slowStream, {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const onError = vi.fn();

      // Start streaming (don't await - it won't complete)
      act(() => {
        result.current.streamGenerate({ prompt: 'Test' }, undefined, undefined, onError);
      });

      // Wait for streaming to start
      await waitFor(() => {
        expect(result.current.isStreaming).toBe(true);
      });

      // Abort the stream
      act(() => {
        result.current.abortStream();
      });

      // Wait for abort to complete
      await waitFor(() => {
        expect(result.current.state).toBe('cancelled');
      });

      expect(result.current.error?.code).toBe('CANCELLED');
    });

    it('should call onError callback when aborted', async () => {
      const slowStream = new ReadableStream({
        async pull() {
          await new Promise(() => {});
        },
      });

      const mockResponse = new Response(slowStream, {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const onError = vi.fn();

      act(() => {
        result.current.streamGenerate({ prompt: 'Test' }, undefined, undefined, onError);
      });

      await waitFor(() => {
        expect(result.current.isStreaming).toBe(true);
      });

      act(() => {
        result.current.abortStream();
      });

      await waitFor(() => {
        expect(onError).toHaveBeenCalled();
      });

      expect(onError.mock.calls[0][0].code).toBe('CANCELLED');
    });
  });

  // ---------------------------------------------------------------------------
  // Completion Tests
  // ---------------------------------------------------------------------------

  describe('completion callback', () => {
    it('should call onComplete with full result', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Hello', chunk_index: 0 },
        {
          text: ' World',
          chunk_index: 1,
          is_final: true,
          finish_reason: 'stop',
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({
          provider: 'openai',
          model: 'gpt-4',
          streamId: 'stream-123',
        }),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      let completionResult: StreamResult | undefined;
      const onComplete = vi.fn((res: StreamResult) => {
        completionResult = res;
      });

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test prompt' },
          undefined,
          onComplete
        );
      });

      expect(onComplete).toHaveBeenCalledTimes(1);
      expect(completionResult).toBeDefined();
      expect(completionResult?.fullText).toBe('Hello World');
      expect(completionResult?.totalChunks).toBe(2);
      expect(completionResult?.finishReason).toBe('stop');
      expect(completionResult?.usage?.total_tokens).toBe(15);
      expect(completionResult?.durationMs).toBeGreaterThan(0);
    });

    it('should include metadata in result', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const metadata: StreamMetadata = {
        provider: 'anthropic',
        model: 'claude-3',
        sessionId: 'session-123',
        streamId: 'stream-456',
        startedAt: '2024-01-01T00:00:00Z',
        resolutionSource: 'explicit',
        resolutionPriority: 1,
      };

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders(metadata),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      let completionResult: StreamResult | undefined;
      const onComplete = vi.fn((res: StreamResult) => {
        completionResult = res;
      });

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test' },
          undefined,
          onComplete
        );
      });

      expect(completionResult?.metadata).toEqual(metadata);
      expect(completionResult?.provider).toBe('anthropic');
      expect(completionResult?.model).toBe('claude-3');
    });
  });

  // ---------------------------------------------------------------------------
  // Error Handling Tests
  // ---------------------------------------------------------------------------

  describe('error handling', () => {
    it('should handle HTTP errors', async () => {
      const errorResponse = new Response(
        JSON.stringify({ detail: 'Provider not available' }),
        {
          status: 503,
          statusText: 'Service Unavailable',
        }
      );

      global.fetch = vi.fn().mockResolvedValue(errorResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const onError = vi.fn();

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test' },
          undefined,
          undefined,
          onError
        );
      });

      expect(result.current.state).toBe('error');
      expect(result.current.error?.message).toContain('Provider not available');
      expect(onError).toHaveBeenCalled();
    });

    it('should handle SSE error events', async () => {
      const encoder = new TextEncoder();
      const errorStream = new ReadableStream({
        pull(controller) {
          controller.enqueue(
            encoder.encode(
              'event: error\ndata: {"error": "Model rate limited"}\n\n'
            )
          );
          controller.close();
        },
      });

      const mockResponse = new Response(errorStream, {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const onError = vi.fn();

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test' },
          undefined,
          undefined,
          onError
        );
      });

      expect(result.current.state).toBe('error');
      expect(result.current.error?.message).toBe('Model rate limited');
      expect(onError).toHaveBeenCalled();
    });

    it('should handle network errors', async () => {
      global.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const onError = vi.fn();

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test' },
          undefined,
          undefined,
          onError
        );
      });

      expect(result.current.state).toBe('error');
      expect(result.current.error?.message).toBe('Failed to fetch');
      expect(onError).toHaveBeenCalled();
    });
  });

  // ---------------------------------------------------------------------------
  // Chat Streaming Tests
  // ---------------------------------------------------------------------------

  describe('streamChat', () => {
    it('should stream chat completions', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'I am', chunk_index: 0 },
        { text: ' an AI', chunk_index: 1, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      const onComplete = vi.fn();

      await act(async () => {
        await result.current.streamChat(
          {
            messages: [
              { role: 'user', content: 'Who are you?' },
            ],
          },
          undefined,
          onComplete
        );
      });

      expect(result.current.currentText).toBe('I am an AI');
      expect(onComplete).toHaveBeenCalled();

      // Verify the fetch call includes messages
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/stream/chat'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"messages"'),
        })
      );
    });
  });

  // ---------------------------------------------------------------------------
  // JSON Lines Format Tests
  // ---------------------------------------------------------------------------

  describe('JSON Lines format', () => {
    it('should handle JSONL format correctly', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Line 1', chunk_index: 0 },
        { text: ' Line 2', chunk_index: 1, is_final: true },
      ];

      const mockResponse = new Response(createMockJsonlStream(chunks), {
        status: 200,
        headers: new Headers({ 'Content-Type': 'application/x-ndjson' }),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(
        () => useStreamingGeneration({ useJsonl: true }),
        { wrapper: createWrapper() }
      );

      const onComplete = vi.fn();

      await act(async () => {
        await result.current.streamGenerate(
          { prompt: 'Test' },
          undefined,
          onComplete
        );
      });

      expect(result.current.currentText).toBe('Line 1 Line 2');
      expect(onComplete).toHaveBeenCalled();

      // Verify JSONL endpoint is used
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/stream/generate/jsonl'),
        expect.any(Object)
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Reset Tests
  // ---------------------------------------------------------------------------

  describe('reset functionality', () => {
    it('should reset state to initial values', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({ streamId: 'stream-123' }),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      // First, complete a stream
      await act(async () => {
        await result.current.streamGenerate({ prompt: 'Test' });
      });

      expect(result.current.currentText).toBe('Test');
      expect(result.current.state).toBe('completed');

      // Then reset
      act(() => {
        result.current.reset();
      });

      expect(result.current.state).toBe('idle');
      expect(result.current.currentText).toBe('');
      expect(result.current.streamId).toBeNull();
      expect(result.current.error).toBeNull();
      expect(result.current.chunkIndex).toBe(0);
      expect(result.current.streamMetadata).toBeNull();
    });
  });

  // ---------------------------------------------------------------------------
  // Provider/Model Override Tests
  // ---------------------------------------------------------------------------

  describe('provider and model overrides', () => {
    it('should use explicit provider/model from params', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(() => useStreamingGeneration(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.streamGenerate({
          prompt: 'Test',
          provider: 'anthropic',
          model: 'claude-3',
        });
      });

      // Verify the request includes the overridden provider/model
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"provider":"anthropic"'),
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"model":"claude-3"'),
        })
      );
    });

    it('should use default provider/model from options', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(
        () =>
          useStreamingGeneration({
            provider: 'google',
            model: 'gemini-pro',
          }),
        { wrapper: createWrapper() }
      );

      await act(async () => {
        await result.current.streamGenerate({ prompt: 'Test' });
      });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"provider":"google"'),
        })
      );
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"model":"gemini-pro"'),
        })
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Session ID Tests
  // ---------------------------------------------------------------------------

  describe('session handling', () => {
    it('should include session ID in request headers', async () => {
      const chunks: Array<Partial<StreamChunk>> = [
        { text: 'Test', chunk_index: 0, is_final: true },
      ];

      const mockResponse = new Response(createMockSSEStream(chunks), {
        status: 200,
        headers: createMockHeaders({}),
      });

      global.fetch = vi.fn().mockResolvedValue(mockResponse);

      const { result } = renderHook(
        () => useStreamingGeneration({ sessionId: 'session-xyz' }),
        { wrapper: createWrapper() }
      );

      await act(async () => {
        await result.current.streamGenerate({ prompt: 'Test' });
      });

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-Session-Id': 'session-xyz',
          }),
        })
      );
    });
  });
});
