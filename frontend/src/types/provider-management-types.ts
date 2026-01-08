/**
 * Provider Management Types
 * 
 * Type definitions for LLM provider management.
 */

/**
 * Provider status
 */
export interface ProviderStatus {
  id: string;
  name: string;
  status: 'available' | 'degraded' | 'unavailable' | 'rate_limited';
  is_configured: boolean;
  is_enabled: boolean;
  latency_ms?: number;
  error_rate?: number;
  last_check: string;
  models_available: number;
  rate_limit_info?: RateLimitInfo;
  circuit_breaker?: CircuitBreakerState;
}

/**
 * Rate limit information
 */
export interface RateLimitInfo {
  provider_id: string;
  model_id?: string;
  requests_per_minute: number;
  requests_used: number;
  requests_remaining: number;
  tokens_per_minute: number;
  tokens_used: number;
  tokens_remaining: number;
  reset_at: string;
  is_limited: boolean;
  limit_percent: number;
}

/**
 * Fallback suggestion
 */
export interface FallbackSuggestion {
  provider_id: string;
  provider_name: string;
  model_id: string;
  model_name: string;
  reason: string;
  compatibility_score: number;
  estimated_latency_ms: number;
  cost_multiplier: number;
  capabilities: string[];
}

/**
 * Circuit breaker state
 */
export interface CircuitBreakerState {
  provider_id: string;
  state: 'closed' | 'open' | 'half_open';
  failure_count: number;
  failure_threshold: number;
  success_count: number;
  success_threshold: number;
  last_failure?: string;
  last_success?: string;
  opened_at?: string;
  closes_at?: string;
  half_open_requests: number;
}

/**
 * Provider health
 */
export interface ProviderHealth {
  provider_id: string;
  provider_name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime_percent: number;
  average_latency_ms: number;
  error_rate_percent: number;
  requests_last_hour: number;
  errors_last_hour: number;
  last_error?: {
    message: string;
    timestamp: string;
    code?: string;
  };
  metrics: {
    timestamp: string;
    latency_ms: number;
    success: boolean;
  }[];
}

/**
 * Provider configuration
 */
export interface ProviderConfig {
  id: string;
  name: string;
  api_base_url: string;
  default_model: string;
  rate_limit_rpm: number;
  rate_limit_tpm: number;
  timeout_seconds: number;
  retry_attempts: number;
  retry_delay_ms: number;
  headers?: Record<string, string>;
}

/**
 * Model information
 */
export interface ModelInfo {
  id: string;
  name: string;
  provider_id: string;
  provider_name: string;
  context_length: number;
  max_output_tokens: number;
  capabilities: string[];
  pricing?: {
    input_per_1k_tokens: number;
    output_per_1k_tokens: number;
  };
  is_available: boolean;
  deprecation_date?: string;
}