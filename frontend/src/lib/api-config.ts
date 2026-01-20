// API Configuration Manager
// Supports Direct (Client-to-Provider) and Proxy (Client-to-Backend) modes
//
// SECURITY NOTE: API keys should be stored in environment variables, not localStorage.
// localStorage is used only for non-sensitive configuration like URLs.
//
// ============================================================================
// DEPRECATION NOTICE
// ============================================================================
// This module is deprecated and will be removed in a future release.
//
// Please migrate to the new Zustand-based configuration store:
//
// BEFORE (deprecated):
//   import { getApiConfig, saveApiConfig, getApiHeaders } from '@/lib/api-config';
//   const config = getApiConfig();
//   const headers = getApiHeaders();
//
// AFTER (recommended):
//   import { useConfigStore, useApiConfig } from '@/lib/api/stores';
//
//   // In React components:
//   const { apiMode, activeUrl, headers, isConfigured } = useApiConfig();
//
//   // Or use the store directly:
//   const config = useConfigStore.getState();
//   const headers = config.getApiHeaders();
//
// Benefits of the new approach:
// - Reactive state updates via Zustand
// - Type-safe configuration
// - Better DevTools integration
// - Centralized state management
// - Optimized selectors to prevent unnecessary re-renders
//
// Migration guide: See docs/adr/ADR-001-frontend-api-tanstack-query-migration.md
// ============================================================================

export type ApiMode = "direct" | "proxy";

export interface ApiConfig {
  mode: ApiMode;
  backendApiUrl: string;
  backendApiKey: string;
  geminiApiKey: string;
  geminiApiUrl: string;
  deepseekApiKey: string;
  deepseekApiUrl: string;
  aiProvider: "gemini" | "deepseek";
}

// Non-sensitive config that can be stored in localStorage
interface StorableConfig {
  mode: ApiMode;
  backendApiUrl: string;
  geminiApiUrl: string;
  deepseekApiUrl: string;
  aiProvider: "gemini" | "deepseek";
}

// Default configuration from environment variables
const defaultConfig: ApiConfig = {
  mode: (process.env.NEXT_PUBLIC_API_MODE as ApiMode) || "direct",
  backendApiUrl: process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8002/api/v1",
  backendApiKey: process.env.NEXT_PUBLIC_CHIMERA_API_KEY || "",
  geminiApiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY || "",
  geminiApiUrl: process.env.NEXT_PUBLIC_GEMINI_API_URL || "https://generativelanguage.googleapis.com/v1beta/openai/",
  deepseekApiKey: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || "",
  deepseekApiUrl: process.env.NEXT_PUBLIC_DEEPSEEK_API_URL || "https://api.deepseek.com/v1",
  aiProvider: (process.env.NEXT_PUBLIC_AI_PROVIDER as "gemini" | "deepseek") || "gemini",
};

// Validate required configuration
if (typeof window !== "undefined") {
  // Only validate in browser environment
  if (defaultConfig.mode === "direct") {
    if (!defaultConfig.geminiApiKey && defaultConfig.aiProvider === "gemini") {
      console.error("CRITICAL: NEXT_PUBLIC_GEMINI_API_KEY is missing. Direct Gemini mode will fail.");
    }
    if (!defaultConfig.deepseekApiKey && defaultConfig.aiProvider === "deepseek") {
      console.error("CRITICAL: NEXT_PUBLIC_DEEPSEEK_API_KEY is missing. Direct DeepSeek mode will fail.");
    }
  }
}

// Storage key for persisting configuration
const STORAGE_KEY = "chimera_api_config";

// Get configuration from localStorage or use defaults
// Note: Environment variables always take precedence over localStorage
// API keys are NEVER stored in localStorage - only retrieved from environment variables
export function getApiConfig(): ApiConfig {
  if (typeof window === "undefined") {
    return defaultConfig;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Partial<StorableConfig>;

      const envGeminiUrl = process.env.NEXT_PUBLIC_GEMINI_API_URL;
      const envDeepseekUrl = process.env.NEXT_PUBLIC_DEEPSEEK_API_URL;
      const envBackendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL;
      const envAiProvider = process.env.NEXT_PUBLIC_AI_PROVIDER as "gemini" | "deepseek";

      // Build config from stored non-sensitive values
      const storableConfig: StorableConfig = {
        mode: parsed.mode || defaultConfig.mode,
        backendApiUrl: envBackendUrl || parsed.backendApiUrl || defaultConfig.backendApiUrl,
        geminiApiUrl: envGeminiUrl || parsed.geminiApiUrl || defaultConfig.geminiApiUrl,
        deepseekApiUrl: envDeepseekUrl || parsed.deepseekApiUrl || defaultConfig.deepseekApiUrl,
        aiProvider: envAiProvider || parsed.aiProvider || defaultConfig.aiProvider,
      };

      // Update localStorage with only non-sensitive values
      localStorage.setItem(STORAGE_KEY, JSON.stringify(storableConfig));

      // Return full config with API keys from environment only
      return {
        ...storableConfig,
        backendApiKey: defaultConfig.backendApiKey, // Always from env
        geminiApiKey: defaultConfig.geminiApiKey, // Always from env
        deepseekApiKey: defaultConfig.deepseekApiKey, // Always from env
      };
    }
  } catch (e) {
    console.warn("Failed to load API config from storage:", e);
  }

  return defaultConfig;
}

// Save configuration to localStorage
// SECURITY: Only non-sensitive configuration is stored in localStorage
// API keys are never persisted to localStorage
export function saveApiConfig(config: Partial<ApiConfig>): ApiConfig {
  const currentConfig = getApiConfig();
  const newConfig = { ...currentConfig, ...config };

  if (typeof window !== "undefined") {
    try {
      // Only store non-sensitive configuration
      const storableConfig: StorableConfig = {
        mode: newConfig.mode,
        backendApiUrl: newConfig.backendApiUrl,
        geminiApiUrl: newConfig.geminiApiUrl,
        deepseekApiUrl: newConfig.deepseekApiUrl,
        aiProvider: newConfig.aiProvider,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(storableConfig));
    } catch (e) {
      console.warn("Failed to save API config to storage:", e);
    }
  }

  return newConfig;
}

// Get the active API URL based on current mode
export function getActiveApiUrl(): string {
  const config = getApiConfig();

  if (config.mode === "proxy") {
    return config.backendApiUrl;
  }

  if (config.aiProvider === "deepseek") {
    return config.deepseekApiUrl;
  }
  return config.geminiApiUrl;
}

// Get headers for API requests based on mode
export function getApiHeaders(): Record<string, string> {
  const config = getApiConfig();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (config.mode === "proxy") {
    if (config.backendApiKey) {
      headers["X-API-Key"] = config.backendApiKey;
    }

    // Add Bearer token for authenticated requests
    if (typeof window !== "undefined") {
      try {
        const encodedToken = localStorage.getItem("chimera_access_token");
        if (encodedToken) {
          // Decode: AuthProvider uses btoa(encodeURIComponent(value))
          const accessToken = decodeURIComponent(atob(encodedToken));
          if (accessToken) {
            headers["Authorization"] = `Bearer ${accessToken}`;
          }
        }
      } catch {
        // Ignore decoding errors
      }
    }
  } else {
    // Direct mode
    if (config.aiProvider === "gemini" && config.geminiApiKey) {
      headers["x-goog-api-key"] = config.geminiApiKey;
    } else if (config.aiProvider === "deepseek" && config.deepseekApiKey) {
      headers["Authorization"] = `Bearer ${config.deepseekApiKey}`;
    }
  }

  // Include session ID if available (for model selection sync)
  if (typeof window !== "undefined") {
    const rawSessionId = localStorage.getItem("chimera_session_id");
    if (rawSessionId) {
      try {
        // Parse JSON-stringified session ID
        const sessionId = JSON.parse(rawSessionId);
        if (typeof sessionId === "string" && sessionId) {
          headers["X-Session-ID"] = sessionId;
        }
      } catch {
        // If not valid JSON, use as-is after removing any quotes
        const cleanSessionId = rawSessionId.replace(/^"|"$/g, "");
        if (cleanSessionId) {
          headers["X-Session-ID"] = cleanSessionId;
        }
      }
    }
  }

  return headers;
}

// Check if direct API is configured
export function isDirectApiConfigured(): boolean {
  const config = getApiConfig();

  if (config.mode === "proxy") {
    return !!config.backendApiUrl;
  }

  if (config.aiProvider === "gemini") {
    return !!config.geminiApiKey;
  }
  if (config.aiProvider === "deepseek") {
    return !!config.deepseekApiKey;
  }
  return false;
}

/**
 * Get the currently active API URL.
 * Alias for getActiveApiUrl() for backwards compatibility with api.ts
 */
export function getCurrentApiUrl(): string {
  return getActiveApiUrl();
}

// Mode display names
export const API_MODE_LABELS: Record<ApiMode, string> = {
  direct: "Direct API Mode",
  proxy: "Backend Proxy Mode",
};

// Mode descriptions
export const API_MODE_DESCRIPTIONS: Record<ApiMode, string> = {
  direct: "Connect directly to the configured AI Provider (Gemini or DeepSeek) using your API key",
  proxy: "Connect via the Chimera Backend API (Recommended for enhanced features)",
};
