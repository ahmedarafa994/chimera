/**
 * API Configuration Manager
 *
 * Centralized configuration for API client with environment-based settings,
 * multiple endpoint support, and runtime configuration updates.
 *
 * Direct connection mode only - connects directly to AI providers.
 *
 * @module lib/api/core/config
 */

// ============================================================================
// Types
// ============================================================================

export type Environment = 'development' | 'staging' | 'production';
export type ApiMode = 'direct' | 'proxy';
export type AIProvider = 'gemini' | 'deepseek' | 'openai' | 'anthropic' | 'bigmodel' | 'routeway';

export interface EndpointConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

export interface ProviderConfig {
  apiKey: string;
  apiUrl: string;
  defaultModel?: string;
  timeout?: number;
}

export interface ProxyConfig {
  /** Backend API URL for proxy mode */
  backendUrl: string;
  /** API key for backend authentication */
  apiKey: string;
}

export interface ResilienceConfig {
  /** Enable circuit breaker (auto-disabled in proxy mode for production) */
  enableCircuitBreaker: boolean;
  /** Enable retry logic (minimal in proxy mode) */
  enableRetry: boolean;
  /** Max retry attempts (reduced in proxy mode) */
  maxRetries: number;
}

export interface ApiClientConfig {
  /** Current environment */
  environment: Environment;
  /** API mode: 'proxy' (recommended for production) or 'direct' (dev only) */
  mode: ApiMode;
  /** Active AI provider for direct mode */
  aiProvider: AIProvider;

  /** Proxy configuration (for proxy mode) */
  proxy: ProxyConfig;

  /** Direct provider configurations */
  providers: {
    gemini: ProviderConfig;
    deepseek: ProviderConfig;
    openai: ProviderConfig;
    anthropic: ProviderConfig;
    bigmodel: ProviderConfig;
    routeway: ProviderConfig;
  };

  /** Resilience configuration (adjusted based on mode) */
  resilience: ResilienceConfig;

  /** Request configuration */
  request: {
    /** Default timeout in milliseconds */
    defaultTimeout: number;
    /** Extended timeout for LLM operations */
    llmTimeout: number;
    /** Maximum retry attempts */
    maxRetries: number;
    /** Base delay for exponential backoff (ms) */
    retryBaseDelay: number;
    /** Maximum retry delay (ms) */
    retryMaxDelay: number;
  };

  /** Cache configuration */
  cache: {
    enabled: boolean;
    defaultTTL: number;
    maxEntries: number;
  };

  /** Circuit breaker configuration */
  circuitBreaker: {
    enabled: boolean;
    failureThreshold: number;
    resetTimeout: number;
    halfOpenRequests: number;
  };

  /** Logging configuration */
  logging: {
    enabled: boolean;
    level: 'debug' | 'info' | 'warn' | 'error';
    includeTimings: boolean;
    includeHeaders: boolean;
  };

  /** User preferences */
  preferences: {
    theme: 'light' | 'dark' | 'system';
    autoScroll: boolean;
    compactMode: boolean;
    showDebugTools: boolean;
  };
}

// ============================================================================
// Environment Detection
// ============================================================================

function detectEnvironment(): Environment {
  if (typeof window === 'undefined') {
    return (process.env.NODE_ENV as Environment) || 'development';
  }

  const hostname = window.location.hostname;

  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'development';
  }

  if (hostname.includes('staging') || hostname.includes('stage')) {
    return 'staging';
  }

  return 'production';
}

// ============================================================================
// Default Configuration
// ============================================================================

/**
 * Determine default API mode based on environment
 * - Production: proxy mode (backend handles resilience)
 * - Development: direct mode allowed (for testing direct provider calls)
 */
function getDefaultApiMode(): ApiMode {
  const env = detectEnvironment();
  // In production, force proxy mode for security (API keys should not be in client)
  if (env === 'production') {
    return 'proxy';
  }
  // In dev/staging, check if backend URL is configured, prefer proxy
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (backendUrl) {
    return 'proxy';
  }
  return 'direct';
}

/**
 * Get resilience config based on API mode and environment
 *
 * In PROXY mode:
 * - Circuit breaker is DISABLED (backend handles this)
 * - Retries are MINIMAL (network errors only)
 * - This prevents "double-wrapping" and thundering herd problems
 *
 * In DIRECT mode:
 * - Circuit breaker is ENABLED (frontend must handle provider failures)
 * - Full retry logic enabled
 */
function getResilienceConfig(mode: ApiMode, env: Environment): ResilienceConfig {
  if (mode === 'proxy') {
    // Proxy mode: let the backend handle resilience
    return {
      enableCircuitBreaker: false, // Backend has its own circuit breaker
      enableRetry: true,           // Only retry network errors
      maxRetries: 1,               // Minimal retries (backend will retry upstream)
    };
  }

  // Direct mode: frontend must handle resilience
  return {
    enableCircuitBreaker: true,
    enableRetry: true,
    maxRetries: env === 'production' ? 3 : 2,
  };
}

const DEFAULT_CONFIG: ApiClientConfig = (() => {
  const env = detectEnvironment();
  const mode = getDefaultApiMode();

  return {
    environment: env,
    mode,
    aiProvider: 'gemini',

    proxy: {
      backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001',
      apiKey: process.env.NEXT_PUBLIC_BACKEND_API_KEY || '',
    },

    providers: {
      gemini: {
        apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY || '',
        apiUrl: process.env.NEXT_PUBLIC_GEMINI_API_URL || 'https://generativelanguage.googleapis.com/v1beta/openai/',
        defaultModel: 'gemini-2.0-flash-exp',
        timeout: 0, // No timeout
      },
      deepseek: {
        apiKey: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY || '',
        apiUrl: process.env.NEXT_PUBLIC_DEEPSEEK_API_URL || 'https://api.deepseek.com/v1',
        defaultModel: 'deepseek-chat',
        timeout: 0, // No timeout
      },
      openai: {
        apiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY || '',
        apiUrl: process.env.NEXT_PUBLIC_OPENAI_API_URL || 'https://api.openai.com/v1',
        defaultModel: 'gpt-4-turbo-preview',
        timeout: 0, // No timeout
      },
      anthropic: {
        apiKey: process.env.NEXT_PUBLIC_ANTHROPIC_API_KEY || '',
        apiUrl: process.env.NEXT_PUBLIC_ANTHROPIC_API_URL || 'https://api.anthropic.com/v1',
        defaultModel: 'claude-3-opus-20240229',
        timeout: 0, // No timeout
      },
      bigmodel: {
        apiKey: process.env.NEXT_PUBLIC_BIGMODEL_API_KEY || '',
        apiUrl: process.env.NEXT_PUBLIC_BIGMODEL_API_URL || 'https://open.bigmodel.cn/api/paas/v4',
        defaultModel: 'glm-4.7',
        timeout: 0, // No timeout
      },
      routeway: {
        apiKey: process.env.NEXT_PUBLIC_ROUTEWAY_API_KEY || '',
        apiUrl: process.env.NEXT_PUBLIC_ROUTEWAY_API_URL || 'https://api.routeway.ai/v1',
        defaultModel: 'gpt-4o-mini',
        timeout: 0, // No timeout
      },
    },

    resilience: getResilienceConfig(mode, env),

    request: {
      defaultTimeout: 0, // No timeout
      llmTimeout: 0, // No timeout for LLM operations
      maxRetries: 3,
      retryBaseDelay: 1000, // 1 second
      retryMaxDelay: 30000, // 30 seconds
    },

    cache: {
      enabled: true,
      defaultTTL: 5 * 60 * 1000, // 5 minutes
      maxEntries: 100,
    },

    circuitBreaker: {
      enabled: mode !== 'proxy', // Disabled in proxy mode
      failureThreshold: 5,
      resetTimeout: 30000, // 30 seconds
      halfOpenRequests: 3,
    },

    logging: {
      enabled: process.env.NODE_ENV === 'development',
      level: 'info',
      includeTimings: true,
      includeHeaders: false,
    },

    preferences: {
      theme: 'system',
      autoScroll: true,
      compactMode: false,
      showDebugTools: process.env.NODE_ENV === 'development',
    },
  };
})();

// ============================================================================
// Environment-Specific Overrides
// ============================================================================

const ENVIRONMENT_OVERRIDES: Record<Environment, Partial<ApiClientConfig>> = {
  development: {
    logging: {
      enabled: true,
      level: 'debug',
      includeTimings: true,
      includeHeaders: true,
    },
  },
  staging: {
    logging: {
      enabled: true,
      level: 'info',
      includeTimings: true,
      includeHeaders: false,
    },
    request: {
      ...DEFAULT_CONFIG.request,
      maxRetries: 2,
    },
  },
  production: {
    logging: {
      enabled: false,
      level: 'error',
      includeTimings: false,
      includeHeaders: false,
    },
    request: {
      ...DEFAULT_CONFIG.request,
      maxRetries: 3,
    },
  },
};

// ============================================================================
// Header Types (needed by ConfigManager)
// ============================================================================

export interface RequestHeaders {
  'Content-Type': string;
  Authorization?: string;
  'X-API-Key'?: string;
  'X-Session-ID'?: string;
  'x-goog-api-key'?: string;
  [key: string]: string | undefined;
}

// ============================================================================
// Configuration Manager
// ============================================================================

const STORAGE_KEY = 'chimera_api_config_v2';

// Non-sensitive config that can be stored
interface StorableConfig {
  mode: ApiMode;
  aiProvider: AIProvider;
  cache: {
    enabled: boolean;
    defaultTTL: number;
  };
  proxy?: {
    backendUrl: string;
  };
  preferences?: ApiClientConfig['preferences'];
}

class ConfigManager {
  private config: ApiClientConfig;
  private listeners: Set<(config: ApiClientConfig) => void> = new Set();

  constructor() {
    this.config = this.loadConfig();
  }

  private loadConfig(): ApiClientConfig {
    const env = detectEnvironment();
    const envOverrides = ENVIRONMENT_OVERRIDES[env] || {};

    // Start with defaults + environment overrides
    let config: ApiClientConfig = {
      ...DEFAULT_CONFIG,
      ...envOverrides,
      environment: env,
    };

    // Load stored non-sensitive config
    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored) as Partial<StorableConfig>;
          // Validate mode - only allow direct mode in development
          let mode = parsed.mode || config.mode;
          if (mode === 'direct' && config.environment === 'production') {
            console.warn('Direct mode not allowed in production, forcing proxy mode');
            mode = 'proxy';
          }

          config = {
            ...config,
            mode,
            aiProvider: parsed.aiProvider || config.aiProvider,
            cache: {
              ...config.cache,
              ...parsed.cache,
            },
            proxy: {
              ...config.proxy,
              backendUrl: parsed.proxy?.backendUrl || config.proxy.backendUrl,
            },
            preferences: {
              ...config.preferences,
              ...parsed.preferences,
            },
            // Recalculate resilience config based on mode
            resilience: getResilienceConfig(mode, config.environment),
            circuitBreaker: {
              ...config.circuitBreaker,
              enabled: mode !== 'proxy',
            },
          };
        }
      } catch (e) {
        console.warn('Failed to load API config from storage:', e);
      }
    }

    return config;
  }

  private saveConfig(): void {
    if (typeof window === 'undefined') return;

    try {
      const storable: StorableConfig = {
        mode: this.config.mode,
        aiProvider: this.config.aiProvider,
        cache: {
          enabled: this.config.cache.enabled,
          defaultTTL: this.config.cache.defaultTTL,
        },
        proxy: {
          backendUrl: this.config.proxy.backendUrl,
        },
        preferences: this.config.preferences,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(storable));
    } catch (e) {
      console.warn('Failed to save API config to storage:', e);
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<ApiClientConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(updates: Partial<ApiClientConfig>): void {
    this.config = {
      ...this.config,
      ...updates,
      providers: {
        ...this.config.providers,
        ...updates.providers,
      },
      request: {
        ...this.config.request,
        ...updates.request,
      },
      cache: {
        ...this.config.cache,
        ...updates.cache,
      },
      circuitBreaker: {
        ...this.config.circuitBreaker,
        ...updates.circuitBreaker,
      },
      logging: {
        ...this.config.logging,
        ...updates.logging,
      },
    };

    this.saveConfig();
    this.notifyListeners();
  }

  /**
   * Set API mode
   * @throws Error if trying to set direct mode in production
   */
  setMode(mode: ApiMode): void {
    if (mode === 'direct' && this.config.environment === 'production') {
      throw new Error(
        'Direct mode is not allowed in production for security reasons. ' +
        'API keys should not be exposed to the client. Use proxy mode instead.'
      );
    }

    // Update mode and recalculate resilience config
    const resilience = getResilienceConfig(mode, this.config.environment);
    this.updateConfig({
      mode,
      resilience,
      circuitBreaker: {
        ...this.config.circuitBreaker,
        enabled: mode !== 'proxy',
      },
    });
  }

  /**
   * Set AI provider for direct mode
   */
  setAIProvider(provider: AIProvider): void {
    this.updateConfig({ aiProvider: provider });
  }

  /**
   * Get active base URL based on current mode
   */
  getActiveBaseUrl(): string {
    if (this.config.mode === 'proxy') {
      return this.config.proxy.backendUrl;
    }
    const provider = this.config.providers[this.config.aiProvider];
    return provider.apiUrl;
  }

  /**
   * Check if currently in proxy mode
   */
  isProxyMode(): boolean {
    return this.config.mode === 'proxy';
  }

  /**
   * Check if circuit breaker should be enabled based on current mode
   */
  shouldEnableCircuitBreaker(): boolean {
    return this.config.resilience.enableCircuitBreaker;
  }

  /**
   * Get resilience configuration for current mode
   */
  getResilienceConfig(): ResilienceConfig {
    return { ...this.config.resilience };
  }

  /**
   * Get active API key based on current mode
   */
  getActiveApiKey(): string {
    const provider = this.config.providers[this.config.aiProvider];
    return provider.apiKey;
  }

  /**
   * Get timeout for a specific operation type
   */
  getTimeout(operationType: 'default' | 'llm' = 'default'): number {
    return operationType === 'llm'
      ? this.config.request.llmTimeout
      : this.config.request.defaultTimeout;
  }

  /**
   * Check if a provider is configured
   */
  isProviderConfigured(provider: AIProvider): boolean {
    return !!this.config.providers[provider]?.apiKey;
  }

  /**
   * Get provider configuration
   */
  getProviderConfig(provider: AIProvider): ProviderConfig {
    return { ...this.config.providers[provider] };
  }

  /**
   * Subscribe to configuration changes
   */
  subscribe(listener: (config: ApiClientConfig) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    const config = this.getConfig();
    this.listeners.forEach(listener => listener(config));
  }

  /**
   * Reset to default configuration
   */
  reset(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
    this.config = this.loadConfig();
    this.notifyListeners();
  }

  /**
   * Validate current configuration
   */
  validate(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    const provider = this.config.providers[this.config.aiProvider];
    if (!provider.apiKey) {
      errors.push(`${this.config.aiProvider} API key is not configured`);
    }

    if (this.config.request.maxRetries < 0) {
      errors.push('Max retries must be non-negative');
    }

    if (this.config.request.defaultTimeout < 1000) {
      errors.push('Default timeout must be at least 1000ms');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Get API headers based on current configuration
   */
  getApiHeaders(): RequestHeaders {
    const headers: RequestHeaders = {
      'Content-Type': 'application/json',
    };

    // Add authentication headers based on mode
    const provider = this.config.providers[this.config.aiProvider];
    if (provider.apiKey) {
      if (this.config.aiProvider === 'gemini') {
        headers['x-goog-api-key'] = provider.apiKey;
      } else {
        headers['Authorization'] = `Bearer ${provider.apiKey}`;
      }
    }

    // Add session ID if available
    if (typeof window !== 'undefined') {
      const sessionId = localStorage.getItem('chimera_session_id');
      if (sessionId) {
        headers['X-Session-ID'] = sessionId.replace(/^"|"$/g, '');
      }
    }

    return headers;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

export const configManager = new ConfigManager();

// ============================================================================
// Convenience Exports
// ============================================================================

export function getApiConfig(): Readonly<ApiClientConfig> {
  return configManager.getConfig();
}

export function updateApiConfig(updates: Partial<ApiClientConfig>): void {
  configManager.updateConfig(updates);
}

export function getActiveApiUrl(): string {
  return configManager.getActiveBaseUrl();
}

export function getActiveApiKey(): string {
  return configManager.getActiveApiKey();
}

export function getCurrentApiMode(): ApiMode {
  return configManager.getConfig().mode;
}

export function getCurrentEnvironment(): Environment {
  return configManager.getConfig().environment;
}

// ============================================================================
// Header Generation (Convenience Function)
// ============================================================================

export function getApiHeaders(): RequestHeaders {
  const config = configManager.getConfig();
  const headers: RequestHeaders = {
    'Content-Type': 'application/json',
  };

  // Add authentication headers for direct mode
  const provider = config.providers[config.aiProvider];
  if (provider.apiKey) {
    if (config.aiProvider === 'gemini') {
      headers['x-goog-api-key'] = provider.apiKey;
    } else {
      headers['Authorization'] = `Bearer ${provider.apiKey}`;
    }
  }

  // Add session ID if available
  if (typeof window !== 'undefined') {
    const sessionId = localStorage.getItem('chimera_session_id');
    if (sessionId) {
      headers['X-Session-ID'] = sessionId.replace(/^"|"$/g, '');
    }
  }

  return headers;
}

// ============================================================================
// Endpoint Configuration
//
// IMPORTANT: These endpoints must match the backend routes exactly.
// Backend routes are defined in:
// - backend-api/app/api/v1/router.py (main router)
// - backend-api/app/api/api_routes.py (additional routes)
// - backend-api/app/main.py (root-level routes)
// ============================================================================

export const ENDPOINTS = {
  // ==========================================================================
  // Health & Status (root-level and /api/v1)
  // ==========================================================================
  HEALTH: '/health',                           // GET - Basic health check
  HEALTH_LIVE: '/health/live',                 // GET - Kubernetes liveness probe
  HEALTH_READY: '/health/ready',               // GET - Kubernetes readiness probe
  HEALTH_FULL: '/health/full',                 // GET - Full health check
  HEALTH_INTEGRATION: '/health/integration',   // GET - Integration health
  METRICS: '/metrics',                         // GET - System metrics
  METRICS_DETAILED: '/metrics/detailed',       // GET - Detailed metrics

  // ==========================================================================
  // Session Management (/api/v1/session)
  // ==========================================================================
  SESSION: '/session',                         // POST - Create session, GET - Get session, DELETE - Delete session
  SESSION_MODEL: '/session/model',             // PUT - Update model selection
  SESSION_STATS: '/session/stats',             // GET - Session statistics
  SESSION_CURRENT_MODEL: '/session/current-model', // GET - Get current model for routing

  // ==========================================================================
  // Providers (/api/v1/providers)
  // ==========================================================================
  PROVIDERS: '/providers',                     // GET - List providers (legacy)
  PROVIDERS_AVAILABLE: '/providers/available', // GET - Get available providers with health
  PROVIDERS_SELECT: '/providers/select',       // POST - Select provider/model
  PROVIDERS_CURRENT: '/providers/current',     // GET - Get current selection
  PROVIDERS_HEALTH: '/providers/health',       // GET - Provider health status
  PROVIDERS_RATE_LIMIT: '/providers/rate-limit', // GET - Check rate limit
  PROVIDERS_MODELS: '/providers/{provider}/models', // GET - Get models for provider

  // ==========================================================================
  // Models (/api/v1/models)
  // ==========================================================================
  MODELS: '/models',                           // GET - Get master model list
  MODELS_VALIDATE: '/models/validate',         // POST - Validate model selection
  MODELS_AVAILABLE: '/models/available',       // GET - Get available models (cached)
  MODELS_HEALTH: '/models/health',             // GET - Model sync service health
  MODELS_SESSION_MODEL: '/models/session/model', // POST - Select model for session
  MODELS_SESSION_CURRENT: '/models/session/current', // GET - Get current session model
  MODELS_VALIDATE_ID: '/models/validate/{model_id}', // POST - Validate specific model
  MODELS_AVAILABILITY: '/models/availability/{model_id}', // POST - Update model availability

  // ==========================================================================
  // Model Selection (/api/v1/model-selection)
  // ==========================================================================
  MODEL_SELECTION: '/model-selection',         // GET/POST/DELETE - Model selection management

  // ==========================================================================
  // Core Operations (/api/v1)
  // ==========================================================================
  TRANSFORM: '/transform',                     // POST - Transform prompt
  EXECUTE: '/execute',                         // POST - Transform and execute
  GENERATE: '/generate',                       // POST - Generate text with LLM

  // ==========================================================================
  // Jailbreak Generation (/api/v1/generation)
  // ==========================================================================
  JAILBREAK_GENERATE: '/generation/jailbreak/generate', // POST - AI-powered jailbreak generation
  CODE_GENERATE: '/generation/code/generate',  // POST - Code generation
  REDTEAM_GENERATE: '/generation/redteam/generate-suite', // POST - Red team suite generation
  VALIDATE_PROMPT: '/generation/validate/prompt', // POST - Validate prompt
  GENERATION_TECHNIQUES: '/generation/techniques/available', // GET - Available techniques
  GENERATION_STATISTICS: '/generation/statistics', // GET - Generation statistics
  GENERATION_HEALTH: '/generation/health',     // GET - Generation service health
  GENERATION_RESET: '/generation/reset',       // POST - Reset generation service
  GENERATION_CONFIG: '/generation/config',     // GET - Generation configuration

  // ==========================================================================
  // Intent-Aware Generation (/api/v1/intent-aware)
  // ==========================================================================
  INTENT_AWARE_GENERATE: '/intent-aware/generate', // POST - Intent-aware jailbreak generation
  INTENT_AWARE_ANALYZE: '/intent-aware/analyze-intent', // POST - Analyze intent
  INTENT_AWARE_TECHNIQUES: '/intent-aware/techniques', // GET - Available techniques

  // ==========================================================================
  // AutoDAN (/api/v1/autodan)
  // ==========================================================================
  AUTODAN_JAILBREAK: '/autodan/jailbreak',     // POST - AutoDAN jailbreak generation

  // ==========================================================================
  // GPTFuzz (/api/v1/gptfuzz)
  // ==========================================================================
  GPTFUZZ_RUN: '/gptfuzz/run',                 // POST - Start fuzzing session
  GPTFUZZ_STATUS: '/gptfuzz/status/{session_id}', // GET - Get fuzzing status

  // ==========================================================================
  // AutoAdv (/api/v1/autoadv)
  // ==========================================================================
  AUTOADV_START: '/autoadv/start',             // POST - Start AutoAdv session

  // ==========================================================================
  // HouYi Optimization (/api/v1/optimize)
  // ==========================================================================
  HOUYI_OPTIMIZE: '/optimize/optimize',        // POST - HouYi optimization

  // ==========================================================================
  // Gradient Optimization (/api/v1/gradient)
  // ==========================================================================
  GRADIENT_OPTIMIZE: '/gradient/optimize',     // POST - Gradient-based optimization

  // ==========================================================================
  // Jailbreak Operation (/api/v1/jailbreak)
  // ==========================================================================
  JAILBREAK_DIRECT_GENERATE: '/jailbreak/generate',   // POST - Direct jailbreak generation
  JAILBREAK_BATCH: '/jailbreak/batch',         // POST - Batch generation
  JAILBREAK_RUN_EXECUTE: '/jailbreak/execute',     // POST - Generate and execute
  JAILBREAK_STRATEGIES: '/jailbreak/strategies', // GET - Available strategies
  JAILBREAK_STATS: '/jailbreak/statistics', // GET - Operation stats
  JAILBREAK_PROMPT_VALIDATE: '/jailbreak/validate-prompt', // POST - Validate prompt
  JAILBREAK_PROMPT_SEARCH: '/jailbreak/search',       // POST - Search prompts
  JAILBREAK_STATUS: '/jailbreak/health',       // GET - Health check
  JAILBREAK_AUDIT: '/jailbreak/audit/logs',    // GET - Audit logs
  JAILBREAK_SESSION: '/jailbreak/session/{id}', // GET/DELETE - Session management

  // ==========================================================================
  // Techniques (/api/v1/techniques)
  // ==========================================================================
  TECHNIQUES: '/techniques',                   // GET - List transformation techniques
  TECHNIQUES_DETAIL: '/techniques/{technique_name}', // GET - Get technique details
  JAILBREAK_TECHNIQUE_DETAIL: '/jailbreak/techniques/{technique_id}', // GET - Get technique details
  JAILBREAK_AUDIT_LOGS: '/jailbreak/audit/logs', // GET - Audit logs

  // ==========================================================================
  // Connection Management (/api/v1/connection)
  // ==========================================================================
  CONNECTION_CONFIG: '/connection/config',     // GET - Get connection config
  CONNECTION_STATUS: '/connection/status',     // GET - Get connection status
  CONNECTION_MODE: '/connection/mode',         // POST - Switch connection mode
  CONNECTION_TEST: '/connection/test',         // POST - Test connections
  CONNECTION_HEALTH: '/connection/health',     // GET - Connection health
  CONNECTION_ENDPOINTS: '/connection/endpoints', // GET - List configured endpoints
  CONNECTION_VALIDATE: '/connection/endpoints/validate', // POST - Validate endpoint
  CONNECTION_TEST_PROVIDER: '/connection/endpoints/test/{provider}', // POST - Test provider endpoint

  // ==========================================================================
  // Evasion Tasks (/api/v1/evasion)
  // ==========================================================================
  EVASION_GENERATE: '/evasion/generate',       // POST - Create evasion task
  EVASION_STATUS: '/evasion/status/{task_id}', // GET - Get task status
  EVASION_RESULTS: '/evasion/results/{task_id}', // GET - Get task results

  // ==========================================================================
  // Target Models (/api/v1/target-models)
  // ==========================================================================
  TARGET_MODELS: '/target-models/',            // GET - List models, POST - Create model
  TARGET_MODEL_DETAIL: '/target-models/{model_id}', // GET - Get model, DELETE - Delete model

  // ==========================================================================
  // Strategies (/api/v1/strategies)
  // ==========================================================================
  STRATEGIES: '/strategies/',                  // GET - List metamorphosis strategies
  STRATEGY_DETAIL: '/strategies/{strategy_name}', // GET - Get strategy details

  // ==========================================================================
  // Metamorph (/api/v1/metamorph)
  // ==========================================================================
  METAMORPH_STATUS: '/metamorph/status',       // GET - MetamorphService status
  METAMORPH_SUITES: '/metamorph/suites',       // GET - List transformation suites
  METAMORPH_TRANSFORM: '/metamorph/transform', // POST - Transform prompt
  METAMORPH_DATASETS: '/metamorph/datasets',   // GET - List datasets
  METAMORPH_DATASET_SAMPLE: '/metamorph/datasets/{name}/sample', // GET - Get dataset samples

  // ==========================================================================
  // Admin (/api/v1/admin)
  // ==========================================================================
  ADMIN_FEATURE_FLAGS: '/admin/feature-flags', // GET - List feature flags
  ADMIN_FEATURE_FLAGS_STATS: '/admin/feature-flags/stats', // GET - Feature flag stats
  ADMIN_FEATURE_FLAGS_TOGGLE: '/admin/feature-flags/toggle', // POST - Toggle feature
  ADMIN_FEATURE_FLAGS_RELOAD: '/admin/feature-flags/reload', // POST - Reload config
  ADMIN_FEATURE_FLAG_DETAIL: '/admin/feature-flags/{technique_name}', // GET - Get technique config
  ADMIN_TENANTS: '/admin/tenants',             // GET - List tenants, POST - Create tenant
  ADMIN_TENANT_DETAIL: '/admin/tenants/{tenant_id}', // GET - Get tenant, DELETE - Delete tenant
  ADMIN_TENANTS_STATS: '/admin/tenants/stats/summary', // GET - Tenant statistics
  ADMIN_USAGE_GLOBAL: '/admin/usage/global',   // GET - Global usage stats
  ADMIN_USAGE_TENANT: '/admin/usage/tenant/{tenant_id}', // GET - Tenant usage
  ADMIN_USAGE_TECHNIQUES: '/admin/usage/techniques/top', // GET - Top techniques
  ADMIN_USAGE_QUOTA: '/admin/usage/quota/{tenant_id}', // GET - Quota status

  // ==========================================================================
  // Chat (/api/v1/chat)
  // ==========================================================================
  CHAT: '/chat',                               // General chat endpoint
  CHAT_COMPLETIONS: '/chat/completions',       // POST - Chat completions

  // ==========================================================================
  // Integration Stats (root-level)
  // ==========================================================================
  INTEGRATION_STATS: '/integration/stats',     // GET - Integration service stats
} as const;

export type EndpointKey = keyof typeof ENDPOINTS;

/**
 * Helper function to build endpoint URL with path parameters
 */
export function buildEndpointUrl(
  endpoint: string,
  params?: Record<string, string | number>
): string {
  if (!params) return endpoint;

  let url = endpoint;
  for (const [key, value] of Object.entries(params)) {
    url = url.replace(`{${key}}`, encodeURIComponent(String(value)));
  }
  return url;
}
