"use client";

import { useState, useCallback } from "react";
import {
  autodanReasoningService,
  JailbreakRequest,
  JailbreakResponse,
  AutoDANConfig,
} from "@/lib/services/autodan-reasoning-service";
import { isAPIError, type APIError } from "@/lib/errors";

/**
 * Structured error information for display in UI.
 * Provides field-specific errors and actionable guidance.
 * @requirements 3.1, 3.3, 3.4
 */
export interface ErrorInfo {
  /** Main error message */
  message: string;
  /** Error code for categorization */
  code: string;
  /** HTTP status code if available */
  statusCode?: number;
  /** Field-specific validation errors */
  fieldErrors?: string[];
  /** Actionable guidance for the user */
  guidance?: string;
  /** Timestamp of when the error occurred */
  timestamp: string;
  /** Request ID for debugging */
  requestId?: string;
}

export interface UseAutoDANReasoningReturn {
  isLoading: boolean;
  /** Simple error message for backward compatibility */
  error: string | null;
  /** Detailed error information for enhanced display */
  errorInfo: ErrorInfo | null;
  currentResult: JailbreakResponse | null;
  config: AutoDANConfig | null;
  generate: (request: JailbreakRequest) => Promise<JailbreakResponse>;
  fetchConfig: () => Promise<void>;
  updateConfig: (updates: Partial<AutoDANConfig>) => Promise<void>;
  /** Clear the current error state */
  clearError: () => void;
}

/**
 * Parses an error into structured ErrorInfo for display.
 * Handles APIError, ValidationError, and generic errors.
 * @requirements 3.1, 3.3, 3.4
 */
function parseErrorInfo(err: unknown): ErrorInfo {
  const timestamp = new Date().toISOString();
  
  // Handle APIError instances (from our error system)
  if (isAPIError(err)) {
    const apiError = err as APIError;
    const fieldErrors: string[] = [];
    let guidance: string | undefined;
    
    // Extract field-specific errors from details
    if (apiError.details) {
      if (Array.isArray(apiError.details.validation_errors)) {
        fieldErrors.push(...(apiError.details.validation_errors as string[]));
      }
      if (Array.isArray(apiError.details.errors)) {
        fieldErrors.push(...(apiError.details.errors as string[]));
      }
    }
    
    // Provide actionable guidance based on error code
    switch (apiError.errorCode) {
      case "VALIDATION_ERROR":
        guidance = "Please check that all required fields are filled in correctly and values are within valid ranges.";
        break;
      case "AUTHENTICATION_ERROR":
        guidance = "Please check your API key configuration in the settings.";
        break;
      case "AUTHORIZATION_ERROR":
        guidance = "You don't have permission to perform this action.";
        break;
      case "NETWORK_ERROR":
        guidance = "Unable to reach the server. Please check your connection and ensure the backend is running.";
        break;
      case "RATE_LIMIT_EXCEEDED":
        guidance = "Too many requests. Please wait a moment before trying again.";
        break;
      case "SERVICE_UNAVAILABLE":
        guidance = "The service is temporarily unavailable. Please try again later.";
        break;
      case "GATEWAY_TIMEOUT":
        guidance = "The request timed out. Try reducing the complexity of your request.";
        break;
      case "LLM_CONTENT_BLOCKED":
        guidance = "The content was blocked by safety filters. Try rephrasing your request.";
        break;
      case "LLM_QUOTA_EXCEEDED":
        guidance = "API quota exceeded. Please check your provider's usage limits.";
        break;
      default:
        guidance = "An unexpected error occurred. Please try again.";
    }
    
    return {
      message: apiError.message,
      code: apiError.errorCode,
      statusCode: apiError.statusCode,
      fieldErrors: fieldErrors.length > 0 ? fieldErrors : undefined,
      guidance,
      timestamp,
      requestId: apiError.requestId,
    };
  }
  
  // Handle generic Error instances
  if (err instanceof Error) {
    return {
      message: err.message,
      code: "UNKNOWN_ERROR",
      guidance: "An unexpected error occurred. Please try again.",
      timestamp,
    };
  }
  
  // Handle unknown error types
  return {
    message: String(err) || "An unknown error occurred",
    code: "UNKNOWN_ERROR",
    guidance: "An unexpected error occurred. Please try again.",
    timestamp,
  };
}

/**
 * Hook for AutoDAN Reasoning functionality with enhanced error handling.
 * Provides structured error information for better UI display.
 * @requirements 3.1, 3.3, 3.4
 */
export function useAutoDANReasoning(): UseAutoDANReasoningReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorInfo, setErrorInfo] = useState<ErrorInfo | null>(null);
  const [currentResult, setCurrentResult] = useState<JailbreakResponse | null>(null);
  const [config, setConfig] = useState<AutoDANConfig | null>(null);

  const clearError = useCallback(() => {
    setError(null);
    setErrorInfo(null);
  }, []);

  const generate = useCallback(async (request: JailbreakRequest): Promise<JailbreakResponse> => {
    setIsLoading(true);
    clearError();
    
    try {
      const response = await autodanReasoningService.generateJailbreak(request);
      setCurrentResult(response);
      return response;
    } catch (err) {
      // Parse error into structured format
      const parsedError = parseErrorInfo(err);
      setError(parsedError.message);
      setErrorInfo(parsedError);
      
      // Log error details in development
      if (process.env.NODE_ENV === "development") {
        console.error("[useAutoDANReasoning] Generation failed:", parsedError);
      }
      
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [clearError]);

  const fetchConfig = useCallback(async () => {
    try {
      const data = await autodanReasoningService.getConfiguration();
      setConfig(data);
    } catch (err) {
      const parsedError = parseErrorInfo(err);
      console.error("[useAutoDANReasoning] Failed to fetch config:", parsedError);
      // Don't set error state for config fetch failures - it's not critical
    }
  }, []);

  const updateConfig = useCallback(async (updates: Partial<AutoDANConfig>) => {
    try {
      await autodanReasoningService.updateConfiguration(updates);
      await fetchConfig();
    } catch (err) {
      const parsedError = parseErrorInfo(err);
      console.error("[useAutoDANReasoning] Failed to update config:", parsedError);
      // Set error state for config update failures
      setError(parsedError.message);
      setErrorInfo(parsedError);
      throw err;
    }
  }, [fetchConfig]);

  return {
    isLoading,
    error,
    errorInfo,
    currentResult,
    config,
    generate,
    fetchConfig,
    updateConfig,
    clearError,
  };
}
