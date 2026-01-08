"use client";

/**
 * Chat Hooks for Project Chimera Frontend
 * 
 * React hooks for chat functionality including:
 * - Chat completions
 * - Streaming responses
 * - Message history
 * - Model selection integration
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { chatService, ChatMessage, ChatCompletionRequest } from "@/lib/services/chat-service";
import { useModelSelection } from "./use-provider-management";

// =============================================================================
// Types
// =============================================================================

export interface ChatMessageWithMeta extends ChatMessage {
  id: string;
  timestamp: Date;
  isStreaming?: boolean;
  error?: string;
}

export interface UseChatReturn {
  messages: ChatMessageWithMeta[];
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  sendMessage: (content: string, options?: SendMessageOptions) => Promise<void>;
  clearMessages: () => void;
  retryLastMessage: () => Promise<void>;
  stopStreaming: () => void;
}

export interface SendMessageOptions {
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
}

// =============================================================================
// Chat Hook
// =============================================================================

export function useChat(initialSystemPrompt?: string): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessageWithMeta[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { currentSelection } = useModelSelection();

  // Generate unique message ID
  const generateId = useCallback(() => {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Add message to history
  const addMessage = useCallback((message: ChatMessage, meta?: Partial<ChatMessageWithMeta>) => {
    const newMessage: ChatMessageWithMeta = {
      ...message,
      id: generateId(),
      timestamp: new Date(),
      ...meta,
    };
    setMessages((prev) => [...prev, newMessage]);
    return newMessage.id;
  }, [generateId]);

  // Update message by ID
  const updateMessage = useCallback((id: string, updates: Partial<ChatMessageWithMeta>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...updates } : msg))
    );
  }, []);

  // Send message
  const sendMessage = useCallback(async (content: string, options: SendMessageOptions = {}) => {
    if (!content.trim()) return;

    setError(null);
    setIsLoading(true);

    // Add user message
    addMessage({ role: "user", content });

    // Build message history for API
    const apiMessages: ChatMessage[] = [];
    
    // Add system prompt if provided
    const systemPrompt = options.systemPrompt || initialSystemPrompt;
    if (systemPrompt) {
      apiMessages.push({ role: "system", content: systemPrompt });
    }

    // Add conversation history
    messages.forEach((msg) => {
      if (msg.role !== "system") {
        apiMessages.push({ role: msg.role, content: msg.content });
      }
    });

    // Add current message
    apiMessages.push({ role: "user", content });

    // Build request
    const request: ChatCompletionRequest = {
      messages: apiMessages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stream: options.stream ?? true,
    };

    // Add model selection if available
    if (currentSelection?.provider_id && currentSelection?.model_id) {
      request.provider = currentSelection.provider_id;
      request.model = currentSelection.model_id;
    }

    try {
      if (options.stream !== false) {
        // Streaming response
        setIsStreaming(true);
        const assistantMsgId = addMessage(
          { role: "assistant", content: "" },
          { isStreaming: true }
        );

        abortControllerRef.current = new AbortController();

        let accumulatedContent = "";
        for await (const chunk of chatService.createStreamingChatCompletion(request)) {
          const chunkContent = chunk.choices[0]?.delta?.content || "";
          accumulatedContent += chunkContent;
          updateMessage(assistantMsgId, {
            content: accumulatedContent,
            isStreaming: true,
          });
        }

        updateMessage(assistantMsgId, { isStreaming: false });
      } else {
        // Non-streaming response
        const response = await chatService.createChatCompletion(request);
        const assistantContent = response.choices[0]?.message?.content || "";
        addMessage({ role: "assistant", content: assistantContent });
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        // Streaming was cancelled
        return;
      }
      const errorMessage = err instanceof Error ? err.message : "Failed to send message";
      setError(errorMessage);
      addMessage(
        { role: "assistant", content: "Sorry, an error occurred." },
        { error: errorMessage }
      );
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [messages, currentSelection, initialSystemPrompt, addMessage, updateMessage]);

  // Clear all messages
  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  // Retry last user message
  const retryLastMessage = useCallback(async () => {
    // Find last user message
    const lastUserMessage = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUserMessage) return;

    // Remove last assistant message if it exists
    setMessages((prev) => {
      const lastIndex = prev.length - 1;
      if (prev[lastIndex]?.role === "assistant") {
        return prev.slice(0, lastIndex);
      }
      return prev;
    });

    // Resend
    await sendMessage(lastUserMessage.content);
  }, [messages, sendMessage]);

  // Stop streaming
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    messages,
    isLoading,
    isStreaming,
    error,
    sendMessage,
    clearMessages,
    retryLastMessage,
    stopStreaming,
  };
}

// =============================================================================
// Simple Completion Hook
// =============================================================================

export interface UseCompletionReturn {
  completion: string;
  isLoading: boolean;
  error: string | null;
  complete: (prompt: string, options?: CompletionOptions) => Promise<string>;
  reset: () => void;
}

export interface CompletionOptions {
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
}

export function useCompletion(): UseCompletionReturn {
  const [completion, setCompletion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { currentSelection } = useModelSelection();

  const complete = useCallback(async (prompt: string, options: CompletionOptions = {}): Promise<string> => {
    setIsLoading(true);
    setError(null);
    setCompletion("");

    const messages: ChatMessage[] = [];
    
    if (options.systemPrompt) {
      messages.push({ role: "system", content: options.systemPrompt });
    }
    messages.push({ role: "user", content: prompt });

    const request: ChatCompletionRequest = {
      messages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stream: false,
    };

    if (currentSelection?.provider_id && currentSelection?.model_id) {
      request.provider = currentSelection.provider_id;
      request.model = currentSelection.model_id;
    }

    try {
      const response = await chatService.createChatCompletion(request);
      const content = response.choices[0]?.message?.content || "";
      setCompletion(content);
      return content;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Completion failed";
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentSelection]);

  const reset = useCallback(() => {
    setCompletion("");
    setError(null);
  }, []);

  return {
    completion,
    isLoading,
    error,
    complete,
    reset,
  };
}

// =============================================================================
// Streaming Completion Hook
// =============================================================================

export interface UseStreamingCompletionReturn {
  completion: string;
  isStreaming: boolean;
  error: string | null;
  stream: (prompt: string, options?: CompletionOptions) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

export function useStreamingCompletion(): UseStreamingCompletionReturn {
  const [completion, setCompletion] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const { currentSelection } = useModelSelection();

  const stream = useCallback(async (prompt: string, options: CompletionOptions = {}) => {
    setIsStreaming(true);
    setError(null);
    setCompletion("");

    const messages: ChatMessage[] = [];
    
    if (options.systemPrompt) {
      messages.push({ role: "system", content: options.systemPrompt });
    }
    messages.push({ role: "user", content: prompt });

    const request: ChatCompletionRequest = {
      messages,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stream: true,
    };

    if (currentSelection?.provider_id && currentSelection?.model_id) {
      request.provider = currentSelection.provider_id;
      request.model = currentSelection.model_id;
    }

    abortControllerRef.current = new AbortController();

    try {
      let accumulatedContent = "";
      for await (const chunk of chatService.createStreamingChatCompletion(request)) {
        const chunkContent = chunk.choices[0]?.delta?.content || "";
        accumulatedContent += chunkContent;
        setCompletion(accumulatedContent);
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      const errorMessage = err instanceof Error ? err.message : "Streaming failed";
      setError(errorMessage);
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [currentSelection]);

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const reset = useCallback(() => {
    setCompletion("");
    setError(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    completion,
    isStreaming,
    error,
    stream,
    stop,
    reset,
  };
}