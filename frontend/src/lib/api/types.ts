/**
 * Shared TypeScript types for API communication
 * These types mirror the backend Pydantic models
 */

// ============================================================================
// Base Types
// ============================================================================

export interface ApiResponse<T = unknown> {
  data: T;
  status: number;
  message?: string;
  timestamp?: string;
  request_id?: string;
  headers?: Record<string, string>;
}

export interface ApiError {
  message: string;
  code: string;
  status: number;
  details?: Record<string, unknown>;
  request_id?: string;
  timestamp?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ============================================================================
// Provider Types
// ============================================================================

export interface Provider {
  id: string;
  name: string;
  type: 'openai' | 'anthropic' | 'gemini' | 'deepseek';
  is_available: boolean;
  is_default: boolean;
  models: ProviderModel[];
  metadata?: Record<string, unknown>;
}

export interface ProviderModel {
  model_id: string;
  provider_id: string;
  display_name: string;
  capabilities: string[];
  context_window: number;
  is_available: boolean;
  pricing?: {
    input: number;
    output: number;
  };
}

// ============================================================================
// Prompt & Jailbreak Types
// ============================================================================

export interface PromptRequest {
  prompt: string;
  model_id?: string;
  provider?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  metadata?: Record<string, unknown>;
}

export interface PromptResponse {
  id: string;
  response: string;
  model_used: string;
  provider: string;
  tokens_used: {
    prompt: number;
    completion: number;
    total: number;
  };
  finish_reason: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface JailbreakRequest {
  prompt: string;
  technique: JailbreakTechnique;
  model_id?: string;
  provider?: string;
  intensity?: number;
  custom_params?: Record<string, unknown>;
}

export type JailbreakTechnique =
  | 'role_play'
  | 'hypothetical'
  | 'code_injection'
  | 'translation'
  | 'token_manipulation'
  | 'context_confusion'
  | 'autodan'
  | 'gptfuzz'
  | 'pair'
  | 'cipher'
  | 'metamorph'
  | 'ensemble';

export interface JailbreakResponse {
  id: string;
  original_prompt: string;
  transformed_prompt: string;
  technique: JailbreakTechnique;
  response: string;
  model_used: string;
  success_score: number;
  bypass_detected: boolean;
  metadata?: {
    transformation_params?: Record<string, unknown>;
    detection_flags?: string[];
    [key: string]: unknown;
  };
  created_at: string;
}

// ============================================================================
// AutoDAN Types
// ============================================================================

export interface AutoDANConfig {
  max_iterations: number;
  population_size: number;
  mutation_rate: number;
  crossover_rate: number;
  elite_size: number;
  target_model: string;
  success_threshold: number;
}

export interface AutoDANRequest {
  goal: string;
  config?: Partial<AutoDANConfig>;
  initial_prompts?: string[];
}

export interface AutoDANResponse {
  attack_id: string;
  status: 'running' | 'completed' | 'failed';
  current_iteration: number;
  max_iterations: number;
  best_prompt: string;
  best_score: number;
  all_prompts: Array<{
    prompt: string;
    score: number;
    iteration: number;
  }>;
  execution_time: number;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Session Types
// ============================================================================

export interface Session {
  session_id: string;
  user_id?: string;
  tenant_id?: string;
  model_id: string;
  provider: string;
  created_at: string;
  last_active: string;
  message_count: number;
  metadata?: Record<string, unknown>;
}

export interface Message {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// WebSocket Types
// ============================================================================

export interface WebSocketMessage<T = unknown> {
  type: 'ping' | 'pong' | 'data' | 'error' | 'control';
  payload: T;
  timestamp: string;
  request_id?: string;
}

export interface StreamEvent<T = unknown> {
  event: 'start' | 'chunk' | 'end' | 'error';
  data: T;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Model Selection Types
// ============================================================================

export interface ModelSelectionRequest {
  prompt: string;
  available_models: string[];
  selection_criteria?: {
    prioritize_speed?: boolean;
    max_cost?: number;
    required_capabilities?: string[];
  };
}

export interface ModelSelectionResponse {
  model_id: string;
  provider: string;
  confidence: number;
  reasoning: string;
  estimated_cost?: number;
  estimated_latency?: number;
}

// ============================================================================
// Transformation Types
// ============================================================================

export interface TransformationRequest {
  text: string;
  transformations: TransformationType[];
  chain?: boolean;
  preserve_intent?: boolean;
}

export type TransformationType =
  | 'obfuscation'
  | 'paraphrase'
  | 'translation'
  | 'encoding'
  | 'psychological_framing'
  | 'intent_masking';

export interface TransformationResponse {
  original: string;
  transformed: string;
  transformations_applied: TransformationType[];
  similarity_score: number;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Analytics Types
// ============================================================================

export interface AnalyticsEvent {
  event_type: string;
  timestamp: string;
  user_id?: string;
  session_id?: string;
  properties: Record<string, unknown>;
}

export interface PerformanceMetrics {
  endpoint: string;
  method: string;
  response_time_ms: number;
  status_code: number;
  timestamp: string;
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';
  expires_in: number;
  refresh_expires_in: number;
}

/**
 * User role enum matching backend UserRole
 */
export type UserRole = 'admin' | 'researcher' | 'viewer';

/**
 * User as returned from legacy API endpoints
 * For new authentication, use AuthUser from AuthContext
 */
export interface User {
  id: string;
  email: string;
  username: string;
  tenant_id: string;
  roles: string[];
  permissions: string[];
  metadata?: Record<string, unknown>;
}

/**
 * User as returned from authentication endpoints
 */
export interface AuthUser {
  id: string;
  email: string;
  username: string;
  role: UserRole;
  is_verified: boolean;
  is_active?: boolean;
  created_at?: string;
  last_login?: string;
}

/**
 * Login response from /auth/login endpoint
 */
export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';
  expires_in: number;
  refresh_expires_in: number;
  user: AuthUser;
  requires_verification: boolean;
}

/**
 * Refresh token response
 */
export interface RefreshResponse {
  access_token: string;
  refresh_token: string;
  token_type: 'Bearer';
  expires_in: number;
  refresh_expires_in: number;
}

// ============================================================================
// Health & Status Types
// ============================================================================

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime: number;
  services: Record<string, ServiceStatus>;
  timestamp: string;
}

export interface ServiceStatus {
  status: 'up' | 'down' | 'degraded';
  latency_ms?: number;
  error?: string;
}

// ============================================================================
// Validation Schemas (Zod-compatible definitions)
// ============================================================================

export const ValidationRules = {
  prompt: {
    minLength: 1,
    maxLength: 100000,
  },
  modelId: {
    pattern: /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9._-]+$/,
  },
  temperature: {
    min: 0,
    max: 2,
  },
  maxTokens: {
    min: 1,
    max: 32768,
  },
} as const;
