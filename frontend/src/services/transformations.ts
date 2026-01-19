/**
 * Transformation Service
 *
 * Service for managing prompt transformations and technique applications.
 * Aligned with backend /transform, /execute, and /techniques endpoints.
 */

import { apiClient } from '@/lib/api/client';
import { apiErrorHandler } from '@/lib/errors/api-error-handler';

// ============================================================================
// Types
// ============================================================================

export interface TransformationRequest {
  prompt: string;
  techniqueSuite?: string;
  potencyLevel?: number;
  provider?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface TransformationResult {
  id: string;
  success: boolean;
  originalPrompt: string;
  transformedPrompt: string;
  techniqueSuite: string;
  potencyLevel: number;
  executionTimeMs: number;
  appliedTechniques: string[];
  strategy: string;
  layersApplied: number;
  cached: boolean;
  bypassProbability?: number;
  timestamp: string;
  techniques: AppliedTechnique[]; // Kept for compatibility
  metadata: Record<string, any>; // Kept for compatibility
}

export interface AppliedTechnique {
  id: string;
  name: string;
  parameters: Record<string, any>;
  executionTime: number;
  success: boolean;
  error?: string;
}

export interface ExecutionRequest extends TransformationRequest {
  executeAfterTransform: true;
}

export interface ExecutionResult {
  id: string;
  success: boolean;
  transformation: TransformationResult;
  result: {
    content: string;
    model: string;
    provider: string;
    latencyMs: number;
  };
  executionTimeSeconds: number;
}

export interface TechniqueSummary {
  name: string;
  displayName: string;
  category: string;
  description: string;
  transformerCount: number;
  framerCount: number;
  obfuscatorCount: number;
  components: {
    transformers: string[];
    framers: string[];
    obfuscators: string[];
  };
}

// ============================================================================
// Service
// ============================================================================

class TransformationService {
  private readonly baseUrl = '/api/v1';

  /**
   * Transform a prompt using backend /transform endpoint
   */
  async transformPrompt(request: TransformationRequest): Promise<TransformationResult> {
    try {
      const response = await apiClient.post<any>(`${this.baseUrl}/transform`, {
        core_request: request.prompt,
        technique_suite: request.techniqueSuite || 'basic',
        potency_level: request.potencyLevel || 5,
        provider: request.provider,
        model: request.model,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
      });

      return this.mapTransformResponse(response.data);
    } catch (error) {
      console.error('Failed to transform prompt:', error);
      return apiErrorHandler.createErrorResponse(
        error,
        'Transform Prompt',
        this.getEmptyTransformResult()
      ).data!;
    }
  }

  /**
   * Execute a prompt (transform + generate) using /execute endpoint
   */
  async executePrompt(request: ExecutionRequest): Promise<ExecutionResult> {
    try {
      const response = await apiClient.post<any>(`${this.baseUrl}/execute`, {
        core_request: request.prompt,
        technique_suite: request.techniqueSuite || 'basic',
        potency_level: request.potencyLevel || 5,
        provider: request.provider,
        model: request.model,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
      });

      return this.mapExecuteResponse(response.data);
    } catch (error) {
      console.error('Failed to execute prompt:', error);
      return apiErrorHandler.createErrorResponse(
        error,
        'Execute Prompt',
        this.getEmptyExecuteResult()
      ).data!;
    }
  }

  /**
   * List available techniques from /techniques endpoint
   */
  async listTechniques(): Promise<TechniqueSummary[]> {
    try {
      const response = await apiClient.get<any>(`${this.baseUrl}/techniques`);
      const data = response.data;

      if (!data || !Array.isArray(data.techniques)) {
        return [];
      }

      return data.techniques.map((t: any) => ({
        name: t.name,
        displayName: t.name.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
        category: this.inferCategory(t.name),
        description: this.getDescription(t.name),
        transformerCount: t.transformers || 0,
        framerCount: t.framers || 0,
        obfuscatorCount: t.obfuscators || 0,
        components: {
          transformers: [],
          framers: [],
          obfuscators: [],
        },
      }));
    } catch (error) {
      apiErrorHandler.handleError(error, 'List Techniques');
      return [];
    }
  }

  /**
   * Get detailed information about a specific technique
   */
  async getTechniqueDetail(techniqueName: string): Promise<TechniqueSummary | null> {
    try {
      const response = await apiClient.get<any>(`${this.baseUrl}/techniques/${encodeURIComponent(techniqueName)}`);
      const data = response.data;

      return {
        name: data.name,
        displayName: data.display_name,
        category: data.category,
        description: data.description,
        transformerCount: data.transformer_count,
        framerCount: data.framer_count,
        obfuscatorCount: data.obfuscator_count,
        components: {
          transformers: data.transformers || [],
          framers: data.framers || [],
          obfuscators: data.obfuscators || [],
        },
      };
    } catch (error) {
      apiErrorHandler.handleError(error, 'Get Technique Detail');
      return null;
    }
  }

  async getTechniqueSuites(): Promise<string[]> {
    const techniques = await this.listTechniques();
    return techniques.map(t => t.name);
  }

  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  private mapTransformResponse(data: any): TransformationResult {
    const metadata = data.metadata || {};
    const applied = Array.isArray(metadata.applied_techniques) 
      ? metadata.applied_techniques 
      : (Array.isArray(metadata.techniques_used) ? metadata.techniques_used : []);

    return {
      id: metadata.request_id || `transform_${Date.now()}`,
      success: data.success ?? true,
      originalPrompt: data.original_prompt || "",
      transformedPrompt: data.transformed_prompt || "",
      techniqueSuite: metadata.technique_suite || "",
      potencyLevel: metadata.potency_level || metadata.potency || 0,
      executionTimeMs: metadata.execution_time_ms || 0,
      appliedTechniques: applied,
      strategy: metadata.strategy || "",
      layersApplied: metadata.layers_applied || 0,
      cached: metadata.cached || false,
      bypassProbability: metadata.bypass_probability,
      timestamp: metadata.timestamp || new Date().toISOString(),
      techniques: applied.map((name: string, index: number) => ({
        id: `${name}-${index}`,
        name,
        parameters: {},
        executionTime: 0,
        success: true,
      })),
      metadata: metadata
    };
  }

  private mapExecuteResponse(data: any): ExecutionResult {
    return {
      id: data.request_id || `exec_${Date.now()}`,
      success: data.success ?? true,
      transformation: {
        id: data.request_id || "",
        success: data.success ?? true,
        originalPrompt: data.transformation?.original_prompt || "",
        transformedPrompt: data.transformation?.transformed_prompt || "",
        techniqueSuite: data.transformation?.technique_suite || "",
        potencyLevel: data.transformation?.potency_level || 0,
        executionTimeMs: 0,
        appliedTechniques: [],
        strategy: data.transformation?.metadata?.strategy || "",
        layersApplied: data.transformation?.metadata?.layers || 0,
        cached: false,
        timestamp: new Date().toISOString(),
        techniques: [],
        metadata: data.transformation?.metadata || {}
      },
      result: {
        content: data.result?.content || "",
        model: data.result?.model || "",
        provider: data.result?.provider || "",
        latencyMs: data.result?.latency_ms || 0,
      },
      executionTimeSeconds: data.execution_time_seconds || 0,
    };
  }

  private getEmptyTransformResult(): TransformationResult {
    return {
      id: '',
      success: false,
      originalPrompt: '',
      transformedPrompt: '',
      techniqueSuite: 'basic',
      potencyLevel: 5,
      executionTimeMs: 0,
      appliedTechniques: [],
      strategy: '',
      layersApplied: 0,
      cached: false,
      timestamp: new Date().toISOString(),
      techniques: [],
      metadata: {}
    };
  }

  private getEmptyExecuteResult(): ExecutionResult {
    return {
      id: '',
      success: false,
      transformation: this.getEmptyTransformResult(),
      result: {
        content: '',
        model: '',
        provider: '',
        latencyMs: 0,
      },
      executionTimeSeconds: 0,
    };
  }

  private inferCategory(techniqueName: string): string {
    const categoryMap: Record<string, string> = {
      basic: 'Basic',
      standard: 'Basic',
      advanced: 'Basic',
      expert: 'Basic',
      quantum: 'Quantum',
      quantum_exploit: 'Quantum',
      universal_bypass: 'Bypass',
      experimental_bypass: 'Bypass',
      encoding_bypass: 'Bypass',
      polyglot_bypass: 'Bypass',
      mega_chimera: 'Chimera',
      ultimate_chimera: 'Chimera',
      chaos_ultimate: 'Chimera',
      dan_persona: 'Persona',
      roleplay_bypass: 'Persona',
      hierarchical_persona: 'Persona',
      deep_inception: 'Inception',
      contextual_inception: 'Inception',
      cipher: 'Cipher',
      code_chameleon: 'Cipher',
      advanced_obfuscation: 'Obfuscation',
      cognitive_hacking: 'Cognitive',
      payload_splitting: 'Payload',
      multimodal_jailbreak: 'Multimodal',
      agentic_exploitation: 'Agentic',
    };
    return categoryMap[techniqueName] || 'Other';
  }

  private getDescription(techniqueName: string): string {
    const descriptions: Record<string, string> = {
      basic: 'Basic transformation with minimal obfuscation',
      standard: 'Standard transformation with moderate complexity',
      advanced: 'Advanced multi-layer transformation',
      expert: 'Expert-level deep semantic transformation',
      quantum_exploit: 'Quantum exploitation with theoretical framing',
      dan_persona: 'DAN (Do Anything Now) persona roleplay',
      deep_inception: 'Nested inception-style layered prompts',
      cipher: 'Multi-cipher encoding transformation',
      cognitive_hacking: 'Psychology-based cognitive manipulation',
      payload_splitting: 'Split payload across multiple parts',
    };
    return descriptions[techniqueName] || `${techniqueName.replace(/_/g, ' ')} technique`;
  }
}

export const transformationService = new TransformationService();