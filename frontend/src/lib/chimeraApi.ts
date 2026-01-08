/**
 * Chimera API Client
 * 
 * Provides typed API methods for the Chimera Evasion Platform including:
 * - Target LLM model management
 * - Metamorphosis strategy retrieval
 * - Evasion task creation and monitoring
 */

import axios from 'axios';
import { LLMModel, LLMModelCreate, MetamorphosisStrategyInfo, EvasionTaskConfig, EvasionTaskStatusResponse, EvasionTaskResult } from '../types/chimera';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api/v1';
const API_KEY = process.env.NEXT_PUBLIC_CHIMERA_API_KEY || '';

// Create axios instance with default configuration
const chimeraApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY, 'Authorization': `Bearer ${API_KEY}` } : {}),
  },
});

// =============================================================================
// Target Models API
// Backend endpoints: /api/v1/target-models/
// =============================================================================

/**
 * Get all registered LLM models
 * GET /api/v1/target-models/
 */
export const getLLMModels = async (): Promise<LLMModel[]> => {
  const response = await chimeraApi.get('/target-models/');
  return response.data;
};

/**
 * Register a new target LLM model
 * POST /api/v1/target-models/
 */
export const registerLLMModel = async (modelData: LLMModelCreate): Promise<LLMModel> => {
  const response = await chimeraApi.post('/target-models/', modelData);
  return response.data;
};

/**
 * Get a specific LLM model by ID
 * GET /api/v1/target-models/{model_id}
 */
export const getLLMModelById = async (modelId: number): Promise<LLMModel> => {
  const response = await chimeraApi.get(`/target-models/${modelId}`);
  return response.data;
};

/**
 * Delete an LLM model by ID
 * DELETE /api/v1/target-models/{model_id}
 */
export const deleteLLMModel = async (modelId: number): Promise<void> => {
  await chimeraApi.delete(`/target-models/${modelId}`);
};

// =============================================================================
// Strategies API
// Backend endpoints: /api/v1/strategies/
// =============================================================================

/**
 * Get all available metamorphosis strategies
 * GET /api/v1/strategies/
 */
export const getMetamorphosisStrategies = async (): Promise<MetamorphosisStrategyInfo[]> => {
  const response = await chimeraApi.get('/strategies/');
  return response.data;
};

/**
 * Get a specific strategy by name
 * GET /api/v1/strategies/{strategy_name}
 */
export const getStrategyByName = async (strategyName: string): Promise<MetamorphosisStrategyInfo> => {
  const response = await chimeraApi.get(`/strategies/${encodeURIComponent(strategyName)}`);
  return response.data;
};

// =============================================================================
// Evasion Tasks API
// Backend endpoints: /api/v1/evasion/
// =============================================================================

/**
 * Create a new evasion task
 * POST /api/v1/evasion/generate
 */
export const createEvasionTask = async (taskConfig: EvasionTaskConfig): Promise<EvasionTaskStatusResponse> => {
  const response = await chimeraApi.post('/evasion/generate', taskConfig);
  return response.data;
};

/**
 * Get the status of an evasion task
 * GET /api/v1/evasion/status/{task_id}
 */
export const getEvasionTaskStatus = async (taskId: string): Promise<EvasionTaskStatusResponse> => {
  const response = await chimeraApi.get(`/evasion/status/${encodeURIComponent(taskId)}`);
  return response.data;
};

/**
 * Get the results of a completed evasion task
 * GET /api/v1/evasion/results/{task_id}
 */
export const getEvasionTaskResults = async (taskId: string): Promise<EvasionTaskResult> => {
  const response = await chimeraApi.get(`/evasion/results/${encodeURIComponent(taskId)}`);
  return response.data;
};

// =============================================================================
// Export the axios instance for custom requests
// =============================================================================

export { chimeraApi };
export default chimeraApi;