/**
 * Config Store
 * Unified Zustand store for API configuration and global app settings
 * 
 * Centralizes all configuration that was previously scattered across localStorage:
 * - API mode (direct/proxy)
 * - Provider settings
 * - Session IDs
 * - UI preferences
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { logger } from '../logger';

// ============================================================================
// Types
// ============================================================================

export type ApiMode = 'direct' | 'proxy';
export type AIProvider = 'gemini' | 'deepseek' | 'openai' | 'anthropic';

interface ApiUrls {
  backend: string;
  gemini: string;
  deepseek: string;
}

interface ConfigState {
  // API Configuration
  apiMode: ApiMode;
  aiProvider: AIProvider;
  apiUrls: ApiUrls;
  
  // Session
  sessionId: string | null;
  
  // UI Preferences
  theme: 'light' | 'dark' | 'system';
  sidebarCollapsed: boolean;
  
  // Connection state
  isConnected: boolean;
  lastConnectionCheck: number | null;
  
  // Actions
  setApiMode: (mode: ApiMode) => void;
  setAiProvider: (provider: AIProvider) => void;
  setApiUrl: (type: keyof ApiUrls, url: string) => void;
  setSessionId: (sessionId: string | null) => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  setConnectionStatus: (connected: boolean) => void;
  
  // Computed getters
  getActiveApiUrl: () => string;
  getApiHeaders: () => Record<string, string>;
  isConfigured: () => boolean;
  
  // Reset
  resetConfig: () => void;
}

// ============================================================================
// Default Configuration
// ============================================================================

const getDefaultConfig = () => {
  // Only access environment variables on client side
  if (typeof window === 'undefined') {
    return {
      apiMode: 'proxy' as ApiMode,
      aiProvider: 'gemini' as AIProvider,
      apiUrls: {
        backend: 'http://localhost:8001/api/v1',
        gemini: 'https://generativelanguage.googleapis.com/v1beta/openai/',
        deepseek: 'https://api.deepseek.com/v1',
      },
    };
  }
  
  return {
    apiMode: (process.env.NEXT_PUBLIC_API_MODE as ApiMode) || 'proxy',
    aiProvider: (process.env.NEXT_PUBLIC_AI_PROVIDER as AIProvider) || 'gemini',
    apiUrls: {
      backend: process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8001/api/v1',
      gemini: process.env.NEXT_PUBLIC_GEMINI_API_URL || 
        'https://generativelanguage.googleapis.com/v1beta/openai/',
      deepseek: process.env.NEXT_PUBLIC_DEEPSEEK_API_URL || 'https://api.deepseek.com/v1',
    },
  };
};

// ============================================================================
// Store
// ============================================================================

export const useConfigStore = create<ConfigState>()(
  devtools(
    persist(
      (set, get) => {
        const defaults = getDefaultConfig();
        
        return {
          // Initial state
          apiMode: defaults.apiMode,
          aiProvider: defaults.aiProvider,
          apiUrls: defaults.apiUrls,
          sessionId: null,
          theme: 'system',
          sidebarCollapsed: false,
          isConnected: false,
          lastConnectionCheck: null,
          
          // API Mode
          setApiMode: (mode: ApiMode) => {
            set({ apiMode: mode });
            logger.logInfo('API mode changed', { mode });
          },
          
          // AI Provider
          setAiProvider: (provider: AIProvider) => {
            set({ aiProvider: provider });
            logger.logInfo('AI provider changed', { provider });
          },
          
          // API URLs
          setApiUrl: (type: keyof ApiUrls, url: string) => {
            set((state) => ({
              apiUrls: { ...state.apiUrls, [type]: url },
            }));
            logger.logInfo('API URL updated', { type, url });
          },
          
          // Session ID
          setSessionId: (sessionId: string | null) => {
            set({ sessionId });
            
            // Also update legacy localStorage for backward compatibility
            if (typeof window !== 'undefined') {
              if (sessionId) {
                localStorage.setItem('chimera_session_id', JSON.stringify(sessionId));
              } else {
                localStorage.removeItem('chimera_session_id');
              }
            }
            
            logger.logInfo('Session ID updated', { sessionId: sessionId ? 'set' : 'cleared' });
          },
          
          // Theme
          setTheme: (theme: 'light' | 'dark' | 'system') => {
            set({ theme });
            logger.logInfo('Theme changed', { theme });
          },
          
          // Sidebar
          setSidebarCollapsed: (collapsed: boolean) => {
            set({ sidebarCollapsed: collapsed });
          },
          
          // Connection status
          setConnectionStatus: (connected: boolean) => {
            set({ isConnected: connected, lastConnectionCheck: Date.now() });
          },
          
          // Get active API URL based on mode
          getActiveApiUrl: () => {
            const { apiMode, aiProvider, apiUrls } = get();
            
            if (apiMode === 'proxy') {
              return apiUrls.backend;
            }
            
            switch (aiProvider) {
              case 'deepseek':
                return apiUrls.deepseek;
              case 'gemini':
              default:
                return apiUrls.gemini;
            }
          },
          
          // Get API headers based on mode
          getApiHeaders: () => {
            const { apiMode, aiProvider, sessionId } = get();
            const headers: Record<string, string> = {
              'Content-Type': 'application/json',
            };
            
            // API keys are always from environment variables (never stored in state)
            if (typeof window !== 'undefined') {
              if (apiMode === 'proxy') {
                const backendKey = process.env.NEXT_PUBLIC_CHIMERA_API_KEY;
                if (backendKey) {
                  headers['X-API-Key'] = backendKey;
                }
              } else {
                // Direct mode
                if (aiProvider === 'gemini') {
                  const geminiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY;
                  if (geminiKey) {
                    headers['x-goog-api-key'] = geminiKey;
                  }
                } else if (aiProvider === 'deepseek') {
                  const deepseekKey = process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY;
                  if (deepseekKey) {
                    headers['Authorization'] = `Bearer ${deepseekKey}`;
                  }
                }
              }
              
              // Include session ID if available
              if (sessionId) {
                headers['X-Session-ID'] = sessionId;
              }
            }
            
            return headers;
          },
          
          // Check if properly configured
          isConfigured: () => {
            const { apiMode, aiProvider, apiUrls } = get();
            
            if (apiMode === 'proxy') {
              return !!apiUrls.backend;
            }
            
            // Direct mode - check for API keys in env
            if (typeof window !== 'undefined') {
              if (aiProvider === 'gemini') {
                return !!process.env.NEXT_PUBLIC_GEMINI_API_KEY;
              }
              if (aiProvider === 'deepseek') {
                return !!process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY;
              }
            }
            
            return false;
          },
          
          // Reset to defaults
          resetConfig: () => {
            const defaults = getDefaultConfig();
            set({
              apiMode: defaults.apiMode,
              aiProvider: defaults.aiProvider,
              apiUrls: defaults.apiUrls,
              sessionId: null,
              isConnected: false,
              lastConnectionCheck: null,
            });
            
            // Clear legacy localStorage
            if (typeof window !== 'undefined') {
              localStorage.removeItem('chimera_session_id');
              localStorage.removeItem('chimera_api_config');
            }
            
            logger.logInfo('Config reset to defaults');
          },
        };
      },
      {
        name: 'chimera-config-storage',
        // Only persist non-sensitive configuration
        partialize: (state) => ({
          apiMode: state.apiMode,
          aiProvider: state.aiProvider,
          apiUrls: state.apiUrls,
          sessionId: state.sessionId,
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
        }),
      }
    ),
    { name: 'ConfigStore' }
  )
);

// ============================================================================
// Selectors (for optimized re-renders)
// ============================================================================

export const selectApiMode = (state: ConfigState) => state.apiMode;
export const selectAiProvider = (state: ConfigState) => state.aiProvider;
export const selectSessionId = (state: ConfigState) => state.sessionId;
export const selectTheme = (state: ConfigState) => state.theme;
export const selectIsConnected = (state: ConfigState) => state.isConnected;

// ============================================================================
// Hooks for common patterns
// ============================================================================

/**
 * Hook to get current API configuration
 */
export function useApiConfig() {
  const apiMode = useConfigStore(selectApiMode);
  const aiProvider = useConfigStore(selectAiProvider);
  const getActiveApiUrl = useConfigStore((s) => s.getActiveApiUrl);
  const getApiHeaders = useConfigStore((s) => s.getApiHeaders);
  const isConfigured = useConfigStore((s) => s.isConfigured);
  
  return {
    apiMode,
    aiProvider,
    activeUrl: getActiveApiUrl(),
    headers: getApiHeaders(),
    isConfigured: isConfigured(),
  };
}

/**
 * Hook to manage session
 */
export function useSession() {
  const sessionId = useConfigStore(selectSessionId);
  const setSessionId = useConfigStore((s) => s.setSessionId);
  
  return {
    sessionId,
    setSessionId,
    hasSession: !!sessionId,
  };
}

/**
 * Hook for theme management
 */
export function useTheme() {
  const theme = useConfigStore(selectTheme);
  const setTheme = useConfigStore((s) => s.setTheme);
  
  return { theme, setTheme };
}