/**
 * StreamingDemo Component
 *
 * A demonstration component showing how to use the streaming hooks
 * with the unified provider/model selection system.
 *
 * Features demonstrated:
 * - useStreamingGeneration hook usage
 * - Real-time streaming text display
 * - Stream metadata display
 * - Abort functionality
 * - Error handling
 *
 * @module components/examples/StreamingDemo
 */

"use client";

import React, { useCallback, useState } from "react";
import { useStreamingGeneration, useUnifiedProviderSelection } from "@/hooks";
import type { StreamChunk, StreamResult, StreamError } from "@/hooks";

// =============================================================================
// Component
// =============================================================================

export function StreamingDemo() {
  // State for input
  const [prompt, setPrompt] = useState<string>("");
  const [systemInstruction, setSystemInstruction] = useState<string>("");

  // Provider selection
  const {
    providers,
    models,
    selectedProvider,
    selectedModel,
    selectProvider,
    selectModel,
    isLoadingProviders,
    isLoadingModels,
  } = useUnifiedProviderSelection();

  // Streaming hook
  const {
    state,
    isStreaming,
    currentText,
    streamId,
    error,
    chunkIndex,
    streamMetadata,
    streamGenerate,
    abortStream,
    reset,
  } = useStreamingGeneration();

  // Handlers
  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;

    await streamGenerate(
      {
        prompt,
        systemInstruction: systemInstruction || undefined,
        temperature: 0.7,
        maxTokens: 1000,
      },
      // onChunk callback
      (chunk: StreamChunk) => {
        console.log("Received chunk:", chunk.chunk_index, chunk.text);
      },
      // onComplete callback
      (result: StreamResult) => {
        console.log("Stream completed:", {
          totalChunks: result.totalChunks,
          totalTokens: result.totalTokens,
          durationMs: result.durationMs,
        });
      },
      // onError callback
      (error: StreamError) => {
        console.error("Stream error:", error.message);
      }
    );
  }, [prompt, systemInstruction, streamGenerate]);

  const handleAbort = useCallback(() => {
    abortStream();
  }, [abortStream]);

  const handleReset = useCallback(() => {
    reset();
    setPrompt("");
    setSystemInstruction("");
  }, [reset]);

  const handleProviderChange = useCallback(
    async (e: React.ChangeEvent<HTMLSelectElement>) => {
      await selectProvider(e.target.value);
    },
    [selectProvider]
  );

  const handleModelChange = useCallback(
    async (e: React.ChangeEvent<HTMLSelectElement>) => {
      await selectModel(e.target.value);
    },
    [selectModel]
  );

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold">Streaming Demo</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Demonstrates the useStreamingGeneration hook with unified provider selection
        </p>
      </div>

      {/* Provider/Model Selection */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Provider</label>
          <select
            value={selectedProvider || ""}
            onChange={handleProviderChange}
            disabled={isLoadingProviders || isStreaming}
            className="w-full p-2 border rounded-md disabled:opacity-50"
          >
            <option value="">Select a provider...</option>
            {providers.map((provider) => (
              <option key={provider.provider_id} value={provider.provider_id}>
                {provider.display_name} ({provider.status})
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Model</label>
          <select
            value={selectedModel || ""}
            onChange={handleModelChange}
            disabled={isLoadingModels || isStreaming || !selectedProvider}
            className="w-full p-2 border rounded-md disabled:opacity-50"
          >
            <option value="">Select a model...</option>
            {models.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* System Instruction */}
      <div>
        <label className="block text-sm font-medium mb-1">
          System Instruction (optional)
        </label>
        <input
          type="text"
          value={systemInstruction}
          onChange={(e) => setSystemInstruction(e.target.value)}
          placeholder="You are a helpful assistant..."
          disabled={isStreaming}
          className="w-full p-2 border rounded-md disabled:opacity-50"
        />
      </div>

      {/* Prompt Input */}
      <div>
        <label className="block text-sm font-medium mb-1">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt..."
          disabled={isStreaming}
          rows={4}
          className="w-full p-2 border rounded-md resize-vertical disabled:opacity-50"
        />
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handleGenerate}
          disabled={isStreaming || !prompt.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isStreaming ? "Generating..." : "Generate"}
        </button>

        <button
          onClick={handleAbort}
          disabled={!isStreaming}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Cancel
        </button>

        <button
          onClick={handleReset}
          disabled={isStreaming}
          className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reset
        </button>
      </div>

      {/* Status Display */}
      <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-md space-y-2">
        <div className="flex items-center gap-4 text-sm">
          <span className="font-medium">State:</span>
          <span
            className={`px-2 py-1 rounded ${
              state === "streaming"
                ? "bg-blue-100 text-blue-800"
                : state === "completed"
                  ? "bg-green-100 text-green-800"
                  : state === "error"
                    ? "bg-red-100 text-red-800"
                    : state === "cancelled"
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-gray-200 text-gray-800"
            }`}
          >
            {state}
          </span>
        </div>

        {streamId && (
          <div className="text-sm">
            <span className="font-medium">Stream ID:</span> {streamId}
          </div>
        )}

        {chunkIndex > 0 && (
          <div className="text-sm">
            <span className="font-medium">Chunks received:</span> {chunkIndex + 1}
          </div>
        )}
      </div>

      {/* Stream Metadata */}
      {streamMetadata && (
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md">
          <h3 className="font-medium mb-2">Stream Metadata</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-gray-600 dark:text-gray-400">Provider:</span>{" "}
              {streamMetadata.provider || "N/A"}
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Model:</span>{" "}
              {streamMetadata.model || "N/A"}
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Resolution Source:</span>{" "}
              {streamMetadata.resolutionSource || "N/A"}
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Resolution Priority:</span>{" "}
              {streamMetadata.resolutionPriority ?? "N/A"}
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Session ID:</span>{" "}
              {streamMetadata.sessionId || "N/A"}
            </div>
            <div>
              <span className="text-gray-600 dark:text-gray-400">Started At:</span>{" "}
              {streamMetadata.startedAt || "N/A"}
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <h3 className="font-medium text-red-800 dark:text-red-200 mb-1">Error</h3>
          <p className="text-red-700 dark:text-red-300">{error.message}</p>
          {error.code && (
            <p className="text-sm text-red-600 dark:text-red-400">
              Code: {error.code}
            </p>
          )}
        </div>
      )}

      {/* Generated Output */}
      <div className="p-4 bg-white dark:bg-gray-900 border rounded-md min-h-[200px]">
        <h3 className="font-medium mb-2">Generated Output</h3>
        <div className="whitespace-pre-wrap font-mono text-sm">
          {currentText || (
            <span className="text-gray-400 italic">
              Generated text will appear here...
            </span>
          )}
          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-blue-500 ml-1 animate-pulse" />
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Exports
// =============================================================================

export default StreamingDemo;
