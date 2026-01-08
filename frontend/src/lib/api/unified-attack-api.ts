/**
 * Unified Attack API Client
 * 
 * API client for the multi-vector attack framework endpoints.
 */

import { apiClient } from './client';
import type {
  // Request types
  CreateSessionRequest,
  UnifiedAttackRequest,
  BatchAttackRequest,
  SequentialAttackRequest,
  ParallelAttackRequest,
  IterativeAttackRequest,
  AdaptiveAttackRequest,
  EvaluationRequest,
  BatchEvaluationRequest,
  AllocationRequest,
  BenchmarkRequest,
  ValidationRequest,
  // Response types
  SessionResponse,
  SessionStatusResponse,
  SessionSummaryResponse,
  AttackResponse,
  BatchAttackResponse,
  StrategyInfo,
  PresetConfig,
  ValidationResult,
  EvaluationResponse,
  BatchEvaluationResponse,
  ParetoFrontResponse,
  ResourceUsageResponse,
  BudgetStatusResponse,
  AllocationResponse,
  BenchmarkResponse,
  BenchmarkDataset,
} from '@/types/unified-attack-types';

const BASE_URL = '/api/v1/unified-attack';

// ==============================================================================
// Session Management API
// ==============================================================================

export const unifiedAttackApi = {
  /**
   * Create a new attack session
   */
  async createSession(request: CreateSessionRequest): Promise<SessionResponse> {
    const response = await apiClient.post<SessionResponse>(`${BASE_URL}/sessions`, request);
    return response.data;
  },

  /**
   * Get session status
   */
  async getSession(sessionId: string): Promise<SessionStatusResponse> {
    const response = await apiClient.get<SessionStatusResponse>(`${BASE_URL}/sessions/${sessionId}`);
    return response.data;
  },

  /**
   * Finalize and close session
   */
  async finalizeSession(sessionId: string): Promise<SessionSummaryResponse> {
    const response = await apiClient.delete<SessionSummaryResponse>(`${BASE_URL}/sessions/${sessionId}`);
    return response.data;
  },

  // ==============================================================================
  // Attack Execution API
  // ==============================================================================

  /**
   * Execute a unified multi-vector attack
   */
  async executeAttack(request: UnifiedAttackRequest): Promise<AttackResponse> {
    const response = await apiClient.post<AttackResponse>(`${BASE_URL}/attack`, request);
    return response.data;
  },

  /**
   * Execute batch attacks
   */
  async executeBatchAttack(request: BatchAttackRequest): Promise<BatchAttackResponse> {
    const response = await apiClient.post<BatchAttackResponse>(`${BASE_URL}/attack/batch`, request);
    return response.data;
  },

  /**
   * Execute sequential attack (extend-first or autodan-first)
   */
  async executeSequentialAttack(request: SequentialAttackRequest): Promise<AttackResponse> {
    const response = await apiClient.post<AttackResponse>(`${BASE_URL}/attack/sequential`, request);
    return response.data;
  },

  /**
   * Execute parallel attack (both vectors simultaneously)
   */
  async executeParallelAttack(request: ParallelAttackRequest): Promise<AttackResponse> {
    const response = await apiClient.post<AttackResponse>(`${BASE_URL}/attack/parallel`, request);
    return response.data;
  },

  /**
   * Execute iterative attack (alternating optimization)
   */
  async executeIterativeAttack(request: IterativeAttackRequest): Promise<AttackResponse> {
    const response = await apiClient.post<AttackResponse>(`${BASE_URL}/attack/iterative`, request);
    return response.data;
  },

  /**
   * Execute adaptive attack (auto-select strategy)
   */
  async executeAdaptiveAttack(request: AdaptiveAttackRequest): Promise<AttackResponse> {
    const response = await apiClient.post<AttackResponse>(`${BASE_URL}/attack/adaptive`, request);
    return response.data;
  },

  // ==============================================================================
  // Configuration API
  // ==============================================================================

  /**
   * Get available composition strategies
   */
  async getStrategies(): Promise<StrategyInfo[]> {
    const response = await apiClient.get<StrategyInfo[]>(`${BASE_URL}/config/strategies`);
    return response.data;
  },

  /**
   * Get attack presets
   */
  async getPresets(): Promise<PresetConfig[]> {
    const response = await apiClient.get<PresetConfig[]>(`${BASE_URL}/config/presets`);
    return response.data;
  },

  /**
   * Validate configuration
   */
  async validateConfig(request: ValidationRequest): Promise<ValidationResult> {
    const response = await apiClient.post<ValidationResult>(`${BASE_URL}/config/validate`, request);
    return response.data;
  },

  // ==============================================================================
  // Evaluation API
  // ==============================================================================

  /**
   * Evaluate attack results
   */
  async evaluateAttack(request: EvaluationRequest): Promise<EvaluationResponse> {
    const response = await apiClient.post<EvaluationResponse>(`${BASE_URL}/evaluate`, request);
    return response.data;
  },

  /**
   * Batch evaluate attacks
   */
  async batchEvaluate(request: BatchEvaluationRequest): Promise<BatchEvaluationResponse> {
    const response = await apiClient.post<BatchEvaluationResponse>(`${BASE_URL}/evaluate/batch`, request);
    return response.data;
  },

  /**
   * Get Pareto front
   */
  async getParetoFront(sessionId: string): Promise<ParetoFrontResponse> {
    const response = await apiClient.get<ParetoFrontResponse>(`${BASE_URL}/evaluate/pareto`, {
      params: { session_id: sessionId },
    });
    return response.data;
  },

  // ==============================================================================
  // Resource Tracking API
  // ==============================================================================

  /**
   * Get resource usage
   */
  async getResourceUsage(sessionId: string): Promise<ResourceUsageResponse> {
    const response = await apiClient.get<ResourceUsageResponse>(`${BASE_URL}/resources/${sessionId}`);
    return response.data;
  },

  /**
   * Get budget status
   */
  async getBudgetStatus(sessionId: string): Promise<BudgetStatusResponse> {
    const response = await apiClient.get<BudgetStatusResponse>(`${BASE_URL}/resources/${sessionId}/budget`);
    return response.data;
  },

  /**
   * Allocate resources between vectors
   */
  async allocateResources(request: AllocationRequest): Promise<AllocationResponse> {
    const response = await apiClient.post<AllocationResponse>(`${BASE_URL}/resources/allocate`, request);
    return response.data;
  },

  // ==============================================================================
  // Benchmark API
  // ==============================================================================

  /**
   * Run benchmark
   */
  async runBenchmark(request: BenchmarkRequest): Promise<BenchmarkResponse> {
    const response = await apiClient.post<BenchmarkResponse>(`${BASE_URL}/benchmark`, request);
    return response.data;
  },

  /**
   * Get benchmark datasets
   */
  async getBenchmarkDatasets(): Promise<BenchmarkDataset[]> {
    const response = await apiClient.get<BenchmarkDataset[]>(`${BASE_URL}/benchmark/datasets`);
    return response.data;
  },
};

export default unifiedAttackApi;