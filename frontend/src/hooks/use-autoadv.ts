"use client";

/**
 * AutoAdv Hooks for Project Chimera Frontend
 * 
 * React hooks for AutoAdv (Automatic Adversarial) functionality including:
 * - AutoAdv generation
 * - WebSocket real-time updates
 * - History management
 */

import { useState, useCallback, useRef, useEffect } from "react";
import {
  autoAdvService,
  AutoAdvRequest,
  AutoAdvResponse,
  AutoAdvWebSocketMessage,
} from "@/lib/services/autoadv-service";

// =============================================================================
// Types
// =============================================================================

export interface AutoAdvHistoryItem {
  id: string;
  request: AutoAdvRequest;
  response?: AutoAdvResponse;
  status: "pending" | "running" | "completed" | "failed";
  error?: string;
  startedAt: Date;
  completedAt?: Date;
  progress?: number;
  currentIteration?: number;
}

export interface UseAutoAdvReturn {
  isLoading: boolean;
  error: string | null;
  currentResult: AutoAdvResponse | null;
  history: AutoAdvHistoryItem[];
  generate: (request: AutoAdvRequest) => Promise<AutoAdvResponse>;
  clearHistory: () => void;
  getHistoryItem: (id: string) => AutoAdvHistoryItem | undefined;
}

export interface UseAutoAdvWebSocketReturn {
  isConnected: boolean;
  currentProgress: AutoAdvWebSocketMessage | null;
  connect: (taskId: string) => void;
  disconnect: () => void;
}

// =============================================================================
// AutoAdv Hook
// =============================================================================

export function useAutoAdv(): UseAutoAdvReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<AutoAdvResponse | null>(null);
  const [history, setHistory] = useState<AutoAdvHistoryItem[]>([]);

  // Generate unique ID
  const generateId = useCallback(() => {
    return `autoadv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Add to history
  const addToHistory = useCallback((item: AutoAdvHistoryItem) => {
    setHistory((prev) => [item, ...prev]);
  }, []);

  // Update history item
  const updateHistoryItem = useCallback((id: string, updates: Partial<AutoAdvHistoryItem>) => {
    setHistory((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
    );
  }, []);

  // Generate adversarial prompt
  const generate = useCallback(async (request: AutoAdvRequest): Promise<AutoAdvResponse> => {
    setIsLoading(true);
    setError(null);

    const historyId = generateId();
    const historyItem: AutoAdvHistoryItem = {
      id: historyId,
      request,
      status: "running",
      startedAt: new Date(),
    };
    addToHistory(historyItem);

    try {
      const response = await autoAdvService.generateAutoAdv(request);
      
      setCurrentResult(response);
      updateHistoryItem(historyId, {
        response,
        status: "completed",
        completedAt: new Date(),
      });

      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "AutoAdv generation failed";
      setError(errorMessage);
      updateHistoryItem(historyId, {
        status: "failed",
        error: errorMessage,
        completedAt: new Date(),
      });
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [generateId, addToHistory, updateHistoryItem]);

  // Clear history
  const clearHistory = useCallback(() => {
    setHistory([]);
    setCurrentResult(null);
    setError(null);
  }, []);

  // Get history item by ID
  const getHistoryItem = useCallback((id: string): AutoAdvHistoryItem | undefined => {
    return history.find((item) => item.id === id);
  }, [history]);

  return {
    isLoading,
    error,
    currentResult,
    history,
    generate,
    clearHistory,
    getHistoryItem,
  };
}

// =============================================================================
// AutoAdv WebSocket Hook
// =============================================================================

export function useAutoAdvWebSocket(): UseAutoAdvWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [currentProgress, setCurrentProgress] = useState<AutoAdvWebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback((taskId: string) => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const wsUrl = autoAdvService.getWebSocketUrl(taskId);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
      setCurrentProgress(null);
    };

    ws.onmessage = (event) => {
      try {
        const message: AutoAdvWebSocketMessage = JSON.parse(event.data);
        setCurrentProgress(message);
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      setIsConnected(false);
      wsRef.current = null;
    };

    wsRef.current = ws;
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setCurrentProgress(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    currentProgress,
    connect,
    disconnect,
  };
}

// =============================================================================
// AutoAdv Form Hook
// =============================================================================

export interface UseAutoAdvFormReturn {
  // Form values
  targetPrompt: string;
  setTargetPrompt: (value: string) => void;
  targetBehavior: string;
  setTargetBehavior: (value: string) => void;
  maxIterations: number;
  setMaxIterations: (value: number) => void;
  technique: string;
  setTechnique: (value: string) => void;
  
  // Form state
  isValid: boolean;
  errors: Record<string, string>;
  
  // Actions
  reset: () => void;
  buildRequest: () => AutoAdvRequest;
}

export function useAutoAdvForm(): UseAutoAdvFormReturn {
  const [targetPrompt, setTargetPrompt] = useState("");
  const [targetBehavior, setTargetBehavior] = useState("");
  const [maxIterations, setMaxIterations] = useState(10);
  const [technique, setTechnique] = useState("gcg");
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Validate form
  const validate = useCallback(() => {
    const newErrors: Record<string, string> = {};

    if (!targetPrompt.trim()) {
      newErrors.targetPrompt = "Target prompt is required";
    }

    if (!targetBehavior.trim()) {
      newErrors.targetBehavior = "Target behavior is required";
    }

    if (maxIterations < 1 || maxIterations > 100) {
      newErrors.maxIterations = "Max iterations must be between 1 and 100";
    }

    if (!technique) {
      newErrors.technique = "Technique is required";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [targetPrompt, targetBehavior, maxIterations, technique]);

  // Check if form is valid
  const isValid = targetPrompt.trim() !== "" && 
                  targetBehavior.trim() !== "" && 
                  maxIterations >= 1 && 
                  maxIterations <= 100 &&
                  technique !== "";

  // Reset form
  const reset = useCallback(() => {
    setTargetPrompt("");
    setTargetBehavior("");
    setMaxIterations(10);
    setTechnique("gcg");
    setErrors({});
  }, []);

  // Build request object
  const buildRequest = useCallback((): AutoAdvRequest => {
    return {
      target_prompt: targetPrompt,
      target_behavior: targetBehavior,
      max_iterations: maxIterations,
      technique,
    };
  }, [targetPrompt, targetBehavior, maxIterations, technique]);

  return {
    targetPrompt,
    setTargetPrompt,
    targetBehavior,
    setTargetBehavior,
    maxIterations,
    setMaxIterations,
    technique,
    setTechnique,
    isValid,
    errors,
    reset,
    buildRequest,
  };
}

// =============================================================================
// Combined AutoAdv Hook with WebSocket
// =============================================================================

export interface UseAutoAdvWithProgressReturn extends UseAutoAdvReturn {
  progress: AutoAdvWebSocketMessage | null;
  isWebSocketConnected: boolean;
  generateWithProgress: (request: AutoAdvRequest) => Promise<AutoAdvResponse>;
}

export function useAutoAdvWithProgress(): UseAutoAdvWithProgressReturn {
  const autoAdv = useAutoAdv();
  const webSocket = useAutoAdvWebSocket();
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);

  const generateWithProgress = useCallback(async (request: AutoAdvRequest): Promise<AutoAdvResponse> => {
    // Generate a task ID for WebSocket tracking
    const taskId = `task_${Date.now()}`;
    setCurrentTaskId(taskId);
    
    // Connect to WebSocket for progress updates
    webSocket.connect(taskId);

    try {
      const response = await autoAdv.generate(request);
      return response;
    } finally {
      // Disconnect WebSocket after completion
      webSocket.disconnect();
      setCurrentTaskId(null);
    }
  }, [autoAdv, webSocket]);

  return {
    ...autoAdv,
    progress: webSocket.currentProgress,
    isWebSocketConnected: webSocket.isConnected,
    generateWithProgress,
  };
}