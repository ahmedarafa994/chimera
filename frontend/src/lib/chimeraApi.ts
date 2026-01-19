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
// Backend endpoints: /api/v1/models (Unified Model Sync)
// =============================================================================

/**
 * Get all registered LLM models
 * GET /api/v1/models
 */
export const getLLMModels = async (): Promise<LLMModel[]> => {
  try {
    const response = await chimeraApi.get('/models');
    
    // Transform /models response (ModelsListResponse) to LLMModel[]
    if (response.data && response.data.providers) {
      const models: LLMModel[] = [];
      for (const provider of response.data.providers) {
        if (provider.models_detail) {
          for (const model of provider.models_detail) {
            models.push({
              id: model.id,
              name: model.name,
              provider: provider.provider,
              description: model.description,
              config: {
                max_tokens: model.max_tokens,
                supports_streaming: model.supports_streaming,
                is_default: model.is_default
              }
            });
          }
        }
      }
      return models;
    }
    return [];
  } catch (error) {
    console.error("Failed to fetch models:", error);
    return [];
  }
};

/**
 * Register a new target LLM model
 * NOT SUPPORTED in current backend version.
 */
export const registerLLMModel = async (modelData: LLMModelCreate): Promise<LLMModel> => {
  console.warn("Dynamic model registration is not supported by the backend.");
  throw new Error("Dynamic model registration is not supported.");
};

/**
 * Get a specific LLM model by ID
 * GET /api/v1/models/validate/{model_id} (Approximation)
 * Note: There is no direct "get model by ID" endpoint that returns full config.
 */
export const getLLMModelById = async (modelId: number | string): Promise<LLMModel> => {
  // We have to fetch all and find it, or use validate
  const models = await getLLMModels();
  const model = models.find(m => m.id === modelId || m.id === String(modelId));
  if (!model) {
    throw new Error(`Model with ID ${modelId} not found`);
  }
  return model;
};

/**
 * Delete an LLM model by ID
 * NOT SUPPORTED
 */
export const deleteLLMModel = async (modelId: number | string): Promise<void> => {
  console.warn("Model deletion is not supported by the backend.");
  throw new Error("Model deletion is not supported.");
};

// =============================================================================
// Strategies API
// Backend endpoints: /api/v1/metamorph/suites
// =============================================================================

/**
 * Get all available metamorphosis strategies
 * GET /api/v1/metamorph/suites
 */
export const getMetamorphosisStrategies = async (): Promise<MetamorphosisStrategyInfo[]> => {
  try {
    const response = await chimeraApi.get('/metamorph/suites');
    // Map SuiteInfo to MetamorphosisStrategyInfo
    if (Array.isArray(response.data)) {
      return response.data.map((suite: any) => ({
        name: suite.name,
        description: `Suite with ${suite.transformers_count} transformers, ${suite.framers_count} framers, ${suite.obfuscators_count} obfuscators.`,
        parameters: {} // Backend doesn't expose params yet
      }));
    }
    return [];
  } catch (error) {
    console.error("Failed to fetch strategies:", error);
    return [];
  }
};

/**
 * Get a specific strategy by name
 * Client-side filtering of getMetamorphosisStrategies
 */
export const getStrategyByName = async (strategyName: string): Promise<MetamorphosisStrategyInfo> => {
  const strategies = await getMetamorphosisStrategies();
  const strategy = strategies.find(s => s.name === strategyName);
  if (!strategy) {
    // Fallback if not found in list (maybe name mismatch)
    return {
      name: strategyName,
      description: "Strategy details not available",
      parameters: {}
    };
  }
  return strategy;
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