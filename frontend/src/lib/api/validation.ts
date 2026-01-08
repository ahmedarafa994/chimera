/**
 * Validation Schemas
 * Zod schemas for frontend validation that mirror backend Pydantic models
 */

import { z } from 'zod';

// ============================================================================
// Base Schemas
// ============================================================================

export const paginationSchema = z.object({
  page: z.number().int().min(1).default(1),
  page_size: z.number().int().min(1).max(100).default(20),
});

export const timestampSchema = z.string().datetime();

// ============================================================================
// Provider Schemas
// ============================================================================

export const providerTypeSchema = z.enum([
  'openai',
  'anthropic',
  'gemini',
  'deepseek',
  'google',
  'qwen',
  'gemini-cli',
  'antigravity',
  'kiro',
  'cursor',
  'xai',
  'mock'
]);

export const providerModelSchema = z.object({
  model_id: z.string().min(1),
  provider_id: z.string().min(1),
  display_name: z.string().min(1),
  capabilities: z.array(z.string()),
  context_window: z.number().int().positive(),
  is_available: z.boolean(),
  pricing: z
    .object({
      input: z.number().nonnegative(),
      output: z.number().nonnegative(),
    })
    .optional(),
});

export const providerSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  type: providerTypeSchema,
  is_available: z.boolean(),
  is_default: z.boolean(),
  models: z.array(providerModelSchema),
  metadata: z.record(z.string(), z.any()).optional(),
});

// ============================================================================
// Prompt Schemas
// ============================================================================

export const promptRequestSchema = z.object({
  prompt: z.string().min(1).max(100000),
  model_id: z.string().optional(),
  provider: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  max_tokens: z.number().int().min(1).max(32768).optional(),
  stream: z.boolean().optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export const promptResponseSchema = z.object({
  id: z.string(),
  response: z.string(),
  model_used: z.string(),
  provider: z.string(),
  tokens_used: z.object({
    prompt: z.number().int().nonnegative(),
    completion: z.number().int().nonnegative(),
    total: z.number().int().nonnegative(),
  }),
  finish_reason: z.string(),
  metadata: z.record(z.string(), z.any()).optional(),
  created_at: timestampSchema,
});

// ============================================================================
// Jailbreak Schemas
// ============================================================================

export const jailbreakTechniqueSchema = z.enum([
  'role_play',
  'hypothetical',
  'code_injection',
  'translation',
  'token_manipulation',
  'context_confusion',
  'autodan',
  'gptfuzz',
  'pair',
  'cipher',
  'metamorph',
  'ensemble',
]);

export const jailbreakRequestSchema = z.object({
  prompt: z.string().min(1).max(100000),
  technique: jailbreakTechniqueSchema,
  model_id: z.string().optional(),
  provider: z.string().optional(),
  intensity: z.number().min(0).max(1).optional(),
  custom_params: z.record(z.string(), z.any()).optional(),
});

export const enhancedJailbreakRequestSchema = z.object({
  prompt: z.string().min(1).max(100000),
  method: z.enum(['best_of_n', 'autodan', 'ensemble', 'genetic']).optional(),
  target_model: z.string().optional(),
  provider: z.string().optional(),
  config: z
    .object({
      max_iterations: z.number().int().min(1).max(1000).optional(),
      population_size: z.number().int().min(1).max(100).optional(),
      mutation_rate: z.number().min(0).max(1).optional(),
      crossover_rate: z.number().min(0).max(1).optional(),
      elite_size: z.number().int().min(0).optional(),
      success_threshold: z.number().min(0).max(1).optional(),
    })
    .optional(),
});

// ============================================================================
// AutoDAN Schemas
// ============================================================================

export const autodanConfigSchema = z.object({
  max_iterations: z.number().int().min(1).max(1000).default(100),
  population_size: z.number().int().min(1).max(100).default(20),
  mutation_rate: z.number().min(0).max(1).default(0.1),
  crossover_rate: z.number().min(0).max(1).default(0.7),
  elite_size: z.number().int().min(0).default(2),
  target_model: z.string(),
  success_threshold: z.number().min(0).max(1).default(0.8),
});

// ============================================================================
// Session Schemas
// ============================================================================

export const createSessionRequestSchema = z.object({
  model_id: z.string().optional(),
  provider: z.string().optional(),
  system_prompt: z.string().max(10000).optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export const sendMessageRequestSchema = z.object({
  content: z.string().min(1).max(100000),
  stream: z.boolean().optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export const messageSchema = z.object({
  id: z.string(),
  session_id: z.string(),
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string(),
  timestamp: timestampSchema,
  metadata: z.record(z.string(), z.any()).optional(),
});

// ============================================================================
// Auth Schemas
// ============================================================================

export const loginRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

export const authTokensSchema = z.object({
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal('Bearer'),
  expires_in: z.number().int().positive(),
  refresh_expires_in: z.number().int().positive(),
});

export const userSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  username: z.string().min(3),
  tenant_id: z.string(),
  roles: z.array(z.string()),
  permissions: z.array(z.string()),
  metadata: z.record(z.string(), z.any()).optional(),
});

// ============================================================================
// Transformation Schemas
// ============================================================================

export const transformationTypeSchema = z.enum([
  'obfuscation',
  'paraphrase',
  'translation',
  'encoding',
  'psychological_framing',
  'intent_masking',
]);

export const transformationRequestSchema = z.object({
  text: z.string().min(1).max(100000),
  transformations: z.array(transformationTypeSchema).min(1),
  chain: z.boolean().optional(),
  preserve_intent: z.boolean().optional(),
});

// ============================================================================
// Type Exports (inferred from schemas)
// ============================================================================

export type PaginationInput = z.infer<typeof paginationSchema>;
export type ProviderType = z.infer<typeof providerTypeSchema>;
export type ProviderModel = z.infer<typeof providerModelSchema>;
export type Provider = z.infer<typeof providerSchema>;
export type PromptRequest = z.infer<typeof promptRequestSchema>;
export type PromptResponse = z.infer<typeof promptResponseSchema>;
export type JailbreakTechnique = z.infer<typeof jailbreakTechniqueSchema>;
export type JailbreakRequest = z.infer<typeof jailbreakRequestSchema>;
export type EnhancedJailbreakRequest = z.infer<typeof enhancedJailbreakRequestSchema>;
export type AutoDANConfig = z.infer<typeof autodanConfigSchema>;
export type CreateSessionRequest = z.infer<typeof createSessionRequestSchema>;
export type SendMessageRequest = z.infer<typeof sendMessageRequestSchema>;
export type Message = z.infer<typeof messageSchema>;
export type LoginRequest = z.infer<typeof loginRequestSchema>;
export type AuthTokens = z.infer<typeof authTokensSchema>;
export type User = z.infer<typeof userSchema>;
export type TransformationType = z.infer<typeof transformationTypeSchema>;
export type TransformationRequest = z.infer<typeof transformationRequestSchema>;

// ============================================================================
// Validation Helpers
// ============================================================================

export function validateRequest<T>(schema: z.ZodSchema<T>, data: unknown): {
  success: boolean;
  data?: T;
  errors?: z.ZodError;
} {
  const result = schema.safeParse(data);
  
  if (result.success) {
    return { success: true, data: result.data };
  }
  
  return { success: false, errors: result.error };
}

export function getValidationErrors(error: z.ZodError): Record<string, string[]> {
  const errors: Record<string, string[]> = {};
  
  for (const issue of error.issues) {
    const path = issue.path.join('.');
    if (!errors[path]) {
      errors[path] = [];
    }
    errors[path].push(issue.message);
  }
  
  return errors;
}
