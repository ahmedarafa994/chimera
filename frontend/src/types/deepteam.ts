/**
 * TypeScript type definitions for Deep Team + AutoDAN Frontend
 *
 * These types match the backend API responses and provide type safety
 * for the entire frontend application.
 */

// ============================================================================
// Session Types
// ============================================================================

export interface SessionConfig {
  // AutoDAN Configuration
  populationSize: number
  numGenerations: number
  mutationRate: number
  crossoverRate: number
  eliteSize: number
  tournamentSize: number
  useGradientGuidance: boolean
  gradientWeight: number

  // Target Configuration
  targetModel: string
  attackObjective: string
  initialPrompts: string[]

  // Authorization
  tokenId: string

  // Optional Settings
  maxIterations?: number
  evaluationFrequency?: number
}

export interface Session {
  sessionId: string
  config: SessionConfig
  status: SessionStatus
  createdAt: string
  startedAt?: string
  completedAt?: string
  statistics: SessionStatistics
  bestCandidate?: PromptCandidate
}

export enum SessionStatus {
  PENDING = 'pending',
  INITIALIZING = 'initializing',
  RUNNING = 'running',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  STOPPED = 'stopped',
}

export interface SessionStatistics {
  totalGenerations: number
  totalEvaluations: number
  successfulAttacks: number
  successRate: number
  refinementCycles: number
  averageFitness: number[]
  bestFitnessPerGeneration: number[]
}

// ============================================================================
// Agent Types
// ============================================================================

export enum AgentType {
  ATTACKER = 'attacker',
  EVALUATOR = 'evaluator',
  REFINER = 'refiner',
}

export enum AgentStatus {
  IDLE = 'idle',
  INITIALIZING = 'initializing',
  WORKING = 'working',
  WAITING = 'waiting',
  COMPLETED = 'completed',
  ERROR = 'error',
}

export interface Agent {
  id: string
  type: AgentType
  status: AgentStatus
  currentTask?: string
  progress?: number
  statistics?: AgentStatistics
  lastUpdate: string
}

export interface AgentStatistics {
  tasksCompleted: number
  successRate: number
  averageProcessingTime: number
  errors: number
}

export interface AgentUpdate {
  agentId: string
  agentType: AgentType
  status: AgentStatus
  currentTask?: string
  progress?: number
  timestamp: string
}

// ============================================================================
// Prompt & Evolution Types
// ============================================================================

export interface PromptCandidate {
  id?: string
  prompt: string
  fitness: number
  generation: number
  parentIds: number[]
  mutationHistory: string[]
}

export interface GenerationUpdate {
  sessionId: string
  generation: number
  population: PromptCandidate[]
  bestFitness: number
  averageFitness: number
  elites: PromptCandidate[]
  timestamp: string
}

export interface MutationOperator {
  name: string
  description: string
  probability: number
  enabled: boolean
}

// ============================================================================
// Evaluation Types
// ============================================================================

export interface EvaluationResult {
  sessionId: string
  candidateId: string
  prompt: string
  response: string
  success: boolean
  overallScore: number
  criteriaScores: {
    noRefusal: number
    objectiveAlignment: number
    informativeness: number
    bypassIndicators: number
  }
  feedback: string
  suggestions: string[]
  timestamp: string
}

export interface EvaluationCriteria {
  name: string
  description: string
  weight: number
  score?: number
}

// ============================================================================
// Refinement Types
// ============================================================================

export interface RefinementSuggestion {
  sessionId: string
  generation: number
  configUpdates: Record<string, number>
  strategySuggestions: string[]
  analysis: string
  appliedAt?: string
}

export interface HyperparameterUpdate {
  parameter: string
  oldValue: number
  newValue: number
  reason: string
}

// ============================================================================
// Authorization Types
// ============================================================================

export interface AuthToken {
  tokenId: string
  authorizedTargets: string[]
  authorizedObjectives: string[]
  issuedBy: string
  issuedAt: string
  expiresAt: string
  maxRequestsPerHour: number
  requiresHumanApproval: boolean
  ethicalReviewId?: string
  status: 'active' | 'expired' | 'revoked'
}

export interface AuthorizationCheck {
  isAuthorized: boolean
  reason: string
  remainingRequests?: number
}

// ============================================================================
// Audit Types
// ============================================================================

export interface AuditLogEntry {
  id: string
  timestamp: string
  tokenId: string
  targetModel: string
  objective: string
  prompt: string
  response?: string
  success: boolean
  sessionId?: string
}

export interface AuditReport {
  dateRange: {
    start: string
    end: string
  }
  totalAttempts: number
  successfulAttempts: number
  successRate: number
  uniqueTargets: number
  uniqueTokens: number
  attemptsByTarget: Record<string, number>
  attemptsByObjective: Record<string, number>
}

// ============================================================================
// Gradient Types
// ============================================================================

export enum GradientMode {
  WHITE_BOX = 'white_box',
  BLACK_BOX_APPROXIMATE = 'black_box_approximate',
  HEURISTIC = 'heuristic',
}

export interface GradientConfig {
  mode: GradientMode
  modelName?: string
  device: 'cpu' | 'cuda'
  batchSize: number
  epsilon: number
  temperature: number
  useGradientCaching: boolean
  cacheSize: number
}

export interface TokenGradients {
  tokens: number[]
  gradients: number[][]
  prompt: string
  targetSequence?: string
  metadata: Record<string, unknown>
}

// ============================================================================
// WebSocket Event Types
// ============================================================================

export interface WSMessage<T = unknown> {
  event: string
  data: T
  timestamp: string
}

export interface SessionStatusEvent {
  sessionId: string
  status: SessionStatus
  message?: string
}

export interface ErrorEvent {
  code: string
  message: string
  details?: unknown
}

// ============================================================================
// Analytics Types
// ============================================================================

export interface SessionAnalytics {
  dateRange: {
    start: string
    end: string
  }
  totalSessions: number
  completedSessions: number
  successRate: number
  averageGenerations: number
  averageTime: number
  topAttackStrategies: AttackStrategy[]
  performanceByModel: Record<string, ModelPerformance>
}

export interface AttackStrategy {
  name: string
  count: number
  successRate: number
  averageFitness: number
}

export interface ModelPerformance {
  model: string
  totalAttacks: number
  successfulAttacks: number
  successRate: number
  averageScore: number
}

export interface PerformanceMetrics {
  generation: number
  timestamp: string
  fitness: {
    best: number
    average: number
    worst: number
  }
  diversity: number
  convergenceRate: number
}

// ============================================================================
// UI State Types
// ============================================================================

export interface DashboardState {
  activeSessions: Session[]
  selectedSession?: Session
  agents: Agent[]
  recentEvaluations: EvaluationResult[]
  recentRefinements: RefinementSuggestion[]
  isLoading: boolean
  error?: string
}

export interface ChartData {
  generation: number
  bestFitness: number
  averageFitness: number
  timestamp: string
}

export interface AgentGraphNode {
  id: string
  type: AgentType
  label: string
  status: AgentStatus
  position: { x: number; y: number }
}

export interface AgentGraphEdge {
  id: string
  source: string
  target: string
  label?: string
  animated?: boolean
}

// ============================================================================
// Form Types
// ============================================================================

export interface SessionFormData {
  // Basic Configuration
  targetModel: string
  attackObjective: string
  tokenId: string

  // AutoDAN Parameters
  populationSize: number
  numGenerations: number
  mutationRate: number
  crossoverRate: number
  eliteSize: number

  // Advanced Options
  useGradientGuidance: boolean
  gradientWeight: number
  maxIterations: number
  evaluationFrequency: number

  // Initial Prompts
  initialPrompts: string[]
}

export interface AuthTokenFormData {
  authorizedTargets: string[]
  authorizedObjectives: string[]
  maxRequestsPerHour: number
  expiresInDays: number
  requiresHumanApproval: boolean
  ethicalReviewId?: string
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: {
    code: string
    message: string
    details?: unknown
  }
  timestamp: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

// ============================================================================
// Filter & Search Types
// ============================================================================

export interface SessionFilter {
  status?: SessionStatus[]
  targetModel?: string[]
  dateRange?: {
    start: string
    end: string
  }
  successRate?: {
    min: number
    max: number
  }
}

export interface SearchParams {
  query: string
  filters: SessionFilter
  sortBy: 'createdAt' | 'successRate' | 'generations'
  sortOrder: 'asc' | 'desc'
  page: number
  pageSize: number
}

// ============================================================================
// Notification Types
// ============================================================================

export enum NotificationType {
  SUCCESS = 'success',
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info',
}

export interface Notification {
  id: string
  type: NotificationType
  title: string
  message: string
  timestamp: string
  read: boolean
  actionUrl?: string
}

// Types are already exported inline via 'export interface' declarations above
