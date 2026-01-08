/**
 * API Types and Interfaces
 *
 * Comprehensive TypeScript definitions for all API requests and responses.
 * Provides type safety and documentation for the entire API surface.
 *
 * @module lib/api/core/types
 */

// ============================================================================
// Common Types
// ============================================================================

export type UUID = string;
export type ISODateString = string;
export type HTTPMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

export interface PaginationParams {
  page?: number;
  limit?: number;
  offset?: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: APIErrorResponse;
  meta?: ResponseMeta;
}

export interface APIErrorResponse {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp?: ISODateString;
  requestId?: string;
}

export interface ResponseMeta {
  requestId: string;
  timestamp: ISODateString;
  processingTimeMs: number;
  version?: string;
}

// ============================================================================
// Health & Status Types
// ============================================================================

export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: ISODateString;
  version: string;
  uptime: number;
  services: ServiceHealth[];
}

export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latencyMs?: number;
  lastCheck?: ISODateString;
  error?: string;
}

export interface SystemMetrics {
  cpu: number;
  memory: number;
  activeConnections: number;
  requestsPerSecond: number;
  averageLatencyMs: number;
  errorRate: number;
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface LoginResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: 'Bearer';
  user: UserInfo;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface RefreshTokenResponse {
  accessToken: string;
  refreshToken?: string;
  expiresIn: number;
}

export interface UserInfo {
  id: UUID;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  createdAt: ISODateString;
  lastLogin?: ISODateString;
}

// ============================================================================
// Provider Types
// ============================================================================

export type ProviderName = 'gemini' | 'deepseek' | 'openai' | 'anthropic' | 'bigmodel' | 'routeway' | 'local';

export interface Provider {
  id: string;
  name: ProviderName;
  displayName: string;
  enabled: boolean;
  configured: boolean;
  models: ModelInfo[];
  capabilities: ProviderCapabilities;
  status: ProviderStatus;
}

export interface ProviderCapabilities {
  chat: boolean;
  completion: boolean;
  embedding: boolean;
  imageGeneration: boolean;
  codeGeneration: boolean;
  streaming: boolean;
  functionCalling: boolean;
}

export interface ProviderStatus {
  available: boolean;
  latencyMs?: number;
  lastCheck?: ISODateString;
  error?: string;
  rateLimitRemaining?: number;
  rateLimitReset?: ISODateString;
}

export interface ProviderConfig {
  provider: ProviderName;
  apiKey?: string;
  baseUrl?: string;
  organizationId?: string;
  projectId?: string;
  defaultModel?: string;
  maxTokens?: number;
  temperature?: number;
  customHeaders?: Record<string, string>;
}

export interface UpdateProviderRequest {
  enabled?: boolean;
  apiKey?: string;
  baseUrl?: string;
  defaultModel?: string;
  settings?: Record<string, unknown>;
}

// ============================================================================
// Model Types
// ============================================================================

export interface ModelInfo {
  id: string;
  name: string;
  provider: ProviderName;
  displayName: string;
  description?: string;
  contextWindow: number;
  maxOutputTokens: number;
  inputPricePerToken: number;
  outputPricePerToken: number;
  capabilities: ModelCapabilities;
  deprecated?: boolean;
  releaseDate?: ISODateString;
}

export interface ModelCapabilities {
  chat: boolean;
  completion: boolean;
  embedding: boolean;
  vision: boolean;
  functionCalling: boolean;
  jsonMode: boolean;
  streaming: boolean;
}

export interface ModelSelection {
  provider: ProviderName;
  model: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
}

// ============================================================================
// Chat Types
// ============================================================================

export type MessageRole = 'system' | 'user' | 'assistant' | 'function' | 'tool';

export interface ChatMessage {
  role: MessageRole;
  content: string;
  name?: string;
  functionCall?: FunctionCall;
  toolCalls?: ToolCall[];
}

export interface FunctionCall {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: FunctionCall;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  provider?: ProviderName;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stop?: string[];
  stream?: boolean;
  functions?: FunctionDefinition[];
  tools?: ToolDefinition[];
  responseFormat?: ResponseFormat;
}

export interface FunctionDefinition {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
}

export interface ToolDefinition {
  type: 'function';
  function: FunctionDefinition;
}

export interface ResponseFormat {
  type: 'text' | 'json_object';
}

export interface ChatResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  provider: ProviderName;
  choices: ChatChoice[];
  usage: TokenUsage;
}

export interface ChatChoice {
  index: number;
  message: ChatMessage;
  finishReason: 'stop' | 'length' | 'function_call' | 'tool_calls' | 'content_filter';
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface StreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: StreamChoice[];
}

export interface StreamChoice {
  index: number;
  delta: Partial<ChatMessage>;
  finishReason?: string;
}

// ============================================================================
// Jailbreak Types
// ============================================================================

export type TechniqueName =
  | 'DAN'
  | 'AIM'
  | 'STAN'
  | 'DUDE'
  | 'Jailbroken'
  | 'Developer Mode'
  | 'Evil Confidant'
  | 'UCAR'
  | 'Grandma'
  | 'Sudo Mode'
  | 'Base64'
  | 'ROT13'
  | 'Token Smuggling'
  | 'Prompt Injection'
  | 'Context Manipulation'
  | 'Role Play'
  | 'Hypothetical'
  | 'Translation'
  | 'Code Injection'
  | 'Custom';

export interface Technique {
  id: string;
  name: TechniqueName;
  displayName: string;
  description: string;
  category: TechniqueCategory;
  effectiveness: number;
  complexity: 'low' | 'medium' | 'high';
  template?: string;
  examples?: string[];
  tags?: string[];
}

export type TechniqueCategory =
  | 'persona'
  | 'encoding'
  | 'injection'
  | 'manipulation'
  | 'roleplay'
  | 'obfuscation'
  | 'hybrid';

export interface JailbreakRequest {
  prompt: string;
  technique?: TechniqueName;
  techniques?: TechniqueName[];
  targetModel?: string;
  targetProvider?: ProviderName;
  intensity?: 'low' | 'medium' | 'high';
  iterations?: number;
  customTemplate?: string;
  options?: JailbreakOptions;
}

export interface JailbreakOptions {
  preserveIntent: boolean;
  addObfuscation: boolean;
  useMultipleTechniques: boolean;
  maxLength?: number;
  language?: string;
}

export interface JailbreakResponse {
  id: string;
  originalPrompt: string;
  generatedPrompt: string;
  technique: TechniqueName;
  techniques?: TechniqueName[];
  confidence: number;
  metadata: JailbreakMetadata;
  variants?: JailbreakVariant[];
}

export interface JailbreakMetadata {
  processingTimeMs: number;
  tokensUsed: number;
  model: string;
  provider: ProviderName;
  timestamp: ISODateString;
}

export interface JailbreakVariant {
  prompt: string;
  technique: TechniqueName;
  confidence: number;
}

export interface JailbreakTestRequest {
  prompt: string;
  targetModel: string;
  targetProvider: ProviderName;
  expectedBehavior?: string;
}

export interface JailbreakTestResponse {
  success: boolean;
  response: string;
  blocked: boolean;
  filterTriggered: boolean;
  analysis: JailbreakAnalysis;
}

export interface JailbreakAnalysis {
  harmfulContent: boolean;
  policyViolation: boolean;
  confidenceScore: number;
  categories: string[];
  explanation?: string;
}

// ============================================================================
// AutoDAN Types
// ============================================================================

export interface AutoDANRequest {
  targetPrompt: string;
  targetModel: string;
  targetProvider: ProviderName;
  generations?: number;
  populationSize?: number;
  mutationRate?: number;
  crossoverRate?: number;
  fitnessThreshold?: number;
  maxIterations?: number;
}

export interface AutoDANResponse {
  id: string;
  bestPrompt: string;
  fitness: number;
  generation: number;
  history: AutoDANGeneration[];
  metadata: AutoDANMetadata;
}

export interface AutoDANGeneration {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  bestPrompt: string;
  populationSize: number;
}

export interface AutoDANMetadata {
  totalIterations: number;
  totalTime: number;
  converged: boolean;
  finalPopulationSize: number;
}

// ============================================================================
// GPTFuzz Types
// ============================================================================

export interface GPTFuzzRequest {
  seedPrompts: string[];
  targetModel: string;
  targetProvider: ProviderName;
  mutators?: MutatorType[];
  iterations?: number;
  parallelism?: number;
}

export type MutatorType =
  | 'synonym'
  | 'paraphrase'
  | 'expansion'
  | 'compression'
  | 'encoding'
  | 'injection'
  | 'roleplay'
  | 'context';

export interface GPTFuzzResponse {
  id: string;
  results: GPTFuzzResult[];
  statistics: GPTFuzzStatistics;
  metadata: GPTFuzzMetadata;
}

export interface GPTFuzzResult {
  originalPrompt: string;
  mutatedPrompt: string;
  mutator: MutatorType;
  success: boolean;
  response?: string;
  score: number;
}

export interface GPTFuzzStatistics {
  totalAttempts: number;
  successfulAttempts: number;
  successRate: number;
  bestMutator: MutatorType;
  averageScore: number;
}

export interface GPTFuzzMetadata {
  processingTimeMs: number;
  model: string;
  provider: ProviderName;
}

// ============================================================================
// HouYi Types
// ============================================================================

export interface HouYiRequest {
  targetPrompt: string;
  targetModel: string;
  targetProvider: ProviderName;
  attackType: HouYiAttackType;
  options?: HouYiOptions;
}

export type HouYiAttackType =
  | 'direct'
  | 'indirect'
  | 'context'
  | 'multi-turn'
  | 'adversarial';

export interface HouYiOptions {
  maxTurns?: number;
  temperature?: number;
  useChainOfThought?: boolean;
  targetBehavior?: string;
}

export interface HouYiResponse {
  id: string;
  attackPrompt: string;
  attackType: HouYiAttackType;
  success: boolean;
  response?: string;
  analysis: HouYiAnalysis;
}

export interface HouYiAnalysis {
  vulnerabilityScore: number;
  attackVector: string;
  mitigationSuggestions: string[];
}

// ============================================================================
// Gradient Types
// ============================================================================

export interface GradientRequest {
  targetPrompt: string;
  targetModel: string;
  targetProvider: ProviderName;
  optimizationTarget: 'bypass' | 'elicit' | 'manipulate';
  iterations?: number;
  learningRate?: number;
}

export interface GradientResponse {
  id: string;
  optimizedPrompt: string;
  originalLoss: number;
  finalLoss: number;
  iterations: number;
  convergenceHistory: number[];
  metadata: GradientMetadata;
}

export interface GradientMetadata {
  processingTimeMs: number;
  model: string;
  provider: ProviderName;
  converged: boolean;
}

// ============================================================================
// Metrics Types
// ============================================================================

export interface MetricsResponse {
  timestamp: ISODateString;
  period: MetricsPeriod;
  requests: RequestMetrics;
  performance: PerformanceMetrics;
  providers: ProviderMetrics[];
  techniques: TechniqueMetrics[];
}

export type MetricsPeriod = '1h' | '24h' | '7d' | '30d';

export interface RequestMetrics {
  total: number;
  successful: number;
  failed: number;
  successRate: number;
  averageLatencyMs: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
}

export interface PerformanceMetrics {
  tokensProcessed: number;
  averageTokensPerRequest: number;
  cacheHitRate: number;
  errorsByType: Record<string, number>;
}

export interface ProviderMetrics {
  provider: ProviderName;
  requests: number;
  successRate: number;
  averageLatencyMs: number;
  tokensUsed: number;
  cost: number;
}

export interface TechniqueMetrics {
  technique: TechniqueName;
  uses: number;
  successRate: number;
  averageConfidence: number;
}

// ============================================================================
// Admin Types
// ============================================================================

export interface AdminConfig {
  rateLimiting: RateLimitConfig;
  security: SecurityConfig;
  logging: LoggingConfig;
  features: FeatureFlags;
}

export interface RateLimitConfig {
  enabled: boolean;
  requestsPerMinute: number;
  requestsPerHour: number;
  burstLimit: number;
}

export interface SecurityConfig {
  requireAuth: boolean;
  allowedOrigins: string[];
  maxRequestSize: number;
  enableAuditLog: boolean;
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  includeRequestBody: boolean;
  includeResponseBody: boolean;
  retentionDays: number;
}

export interface FeatureFlags {
  [key: string]: boolean;
}

export interface AuditLogEntry {
  id: UUID;
  timestamp: ISODateString;
  userId?: UUID;
  action: string;
  resource: string;
  resourceId?: string;
  details?: Record<string, unknown>;
  ipAddress?: string;
  userAgent?: string;
}

// ============================================================================
// WebSocket Types
// ============================================================================

export type WebSocketMessageType =
  | 'connect'
  | 'disconnect'
  | 'subscribe'
  | 'unsubscribe'
  | 'message'
  | 'error'
  | 'ping'
  | 'pong';

export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  channel?: string;
  payload?: T;
  timestamp: ISODateString;
  id?: string;
}

export interface WebSocketSubscription {
  channel: string;
  filters?: Record<string, unknown>;
}

export type WebSocketChannel =
  | 'chat'
  | 'jailbreak'
  | 'metrics'
  | 'providers'
  | 'system';

// ============================================================================
// Validation Types
// ============================================================================

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: unknown;
}

// ============================================================================
// Type Guards
// ============================================================================

export function isAPIError(error: unknown): error is APIErrorResponse {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    'message' in error
  );
}

export function isPaginatedResponse<T>(
  response: unknown
): response is PaginatedResponse<T> {
  return (
    typeof response === 'object' &&
    response !== null &&
    'data' in response &&
    'total' in response &&
    'page' in response
  );
}

export function isChatMessage(message: unknown): message is ChatMessage {
  return (
    typeof message === 'object' &&
    message !== null &&
    'role' in message &&
    'content' in message
  );
}

export function isProvider(provider: unknown): provider is Provider {
  return (
    typeof provider === 'object' &&
    provider !== null &&
    'id' in provider &&
    'name' in provider &&
    'enabled' in provider
  );
}

// ============================================================================
// Utility Types
// ============================================================================

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
