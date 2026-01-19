/**
 * Technique Library Service
 *
 * Service for managing and analyzing prompt transformation techniques
 */

import { apiClient } from '@/lib/api/client';

export interface Technique {
  id: string;
  name: string;
  description: string;
  category: TechniqueCategory;
  difficulty: TechniqueDifficulty;
  effectiveness: TechniqueEffectiveness;
  parameters: TechniqueParameter[];
  examples: TechniqueExample[];
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
  // Add missing properties used in the component
  success_rate: number;
  detection_difficulty: number;
  avg_response_time: number;
  use_cases: string[];
  example_prompt?: string;
  example_output?: string;
  best_practices: string[];
  limitations: string[];
  usage_count: number;
}

export interface TechniqueParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'select';
  description: string;
  required: boolean;
  defaultValue?: any;
  options?: string[];
}

export interface TechniqueExample {
  id: string;
  input: string;
  output: string;
  parameters: Record<string, any>;
  description?: string;
}

export enum TechniqueCategory {
  BASIC = 'basic',
  COGNITIVE = 'cognitive',
  OBFUSCATION = 'obfuscation',
  PERSONA = 'persona',
  CONTEXT = 'context',
  LOGIC = 'logic',
  MULTIMODAL = 'multimodal',
  AGENTIC = 'agentic',
  PAYLOAD = 'payload',
  ADVANCED = 'advanced',
}

export enum TechniqueDifficulty {
  BEGINNER = 'beginner',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
  EXPERT = 'expert',
}

export enum TechniqueEffectiveness {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  VERY_HIGH = 'very_high',
}

export interface TechniqueSearchParams {
  query?: string;
  search?: string; // Add search parameter
  category?: TechniqueCategory;
  difficulty?: TechniqueDifficulty;
  effectiveness?: TechniqueEffectiveness;
  tags?: string[];
  limit?: number;
  offset?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface TechniqueSearchResult {
  techniques: Technique[];
  total: number;
  hasMore: boolean;
}

export interface CreateTechniqueRequest {
  name: string;
  description: string;
  category: TechniqueCategory;
  difficulty: TechniqueDifficulty;
  effectiveness: TechniqueEffectiveness;
  parameters: TechniqueParameter[];
  examples: Omit<TechniqueExample, 'id'>[];
  tags: string[];
}

export interface UpdateTechniqueRequest extends Partial<CreateTechniqueRequest> {
  id: string;
}

class TechniqueLibraryService {
  private basePath = '/techniques';

  private mapParameter(param: any): TechniqueParameter {
    const typeMap: Record<string, TechniqueParameter["type"]> = {
      string: "string",
      integer: "number",
      float: "number",
      boolean: "boolean",
      list: "select",
      object: "string",
    };

    return {
      name: param?.name || "parameter",
      type: typeMap[String(param?.parameter_type || param?.type || "string")] || "string",
      description: param?.description || "",
      required: param?.required ?? true,
      defaultValue: param?.default_value ?? param?.defaultValue,
      options: param?.allowed_values || param?.options,
    };
  }

  private mapTechnique(raw: any): Technique {
    const createdAt = raw?.created_at ? new Date(raw.created_at) : new Date();
    const updatedAt = raw?.updated_at ? new Date(raw.updated_at) : new Date();
    const parameters = Array.isArray(raw?.parameters)
      ? raw.parameters.map((param: any) => this.mapParameter(param))
      : [];

    return {
      id: raw?.technique_id || raw?.id || "",
      name: raw?.name || "Unnamed",
      description: raw?.description || "",
      category: raw?.category || TechniqueCategory.BASIC,
      difficulty: raw?.difficulty || TechniqueDifficulty.INTERMEDIATE,
      effectiveness: raw?.effectiveness || TechniqueEffectiveness.MEDIUM,
      parameters,
      examples: raw?.examples || [],
      tags: raw?.tags || [],
      createdAt,
      updatedAt,
      success_rate: raw?.success_rate ?? 0,
      detection_difficulty: raw?.detection_difficulty ?? 0,
      avg_response_time: raw?.avg_response_time ?? 0,
      use_cases: raw?.use_cases || [],
      example_prompt: raw?.example_prompt,
      example_output: raw?.example_output,
      best_practices: raw?.best_practices || [],
      limitations: raw?.limitations || [],
      usage_count: raw?.usage_count ?? 0,
    };
  }

  async searchTechniques(params: TechniqueSearchParams = {}): Promise<TechniqueSearchResult> {
    const response = await apiClient.get<any>(this.basePath, {
      params: {
        query: params.query,
        search: params.search,
        category: params.category,
        difficulty: params.difficulty,
        effectiveness: params.effectiveness,
        tags: params.tags?.join(','),
        limit: params.limit,
        offset: params.offset,
        sort_by: params.sort_by,
        sort_order: params.sort_order,
      }
    });
    const data = response.data;
    const rawTechniques = Array.isArray(data?.techniques) ? data.techniques : [];
    const total = data?.total_techniques ?? data?.total ?? rawTechniques.length;
    const hasMore = data?.has_next ?? data?.hasMore ?? false;

    return {
      techniques: rawTechniques.map((technique: any) => this.mapTechnique(technique)),
      total,
      hasMore,
    };
  }

  async listTechniques(params: TechniqueSearchParams = {}): Promise<TechniqueListResponse> {
    const searchResult = await this.searchTechniques(params);
    // Convert TechniqueSearchResult to TechniqueListResponse
    return {
      ...searchResult,
      page: params.offset ? Math.floor(params.offset / (params.limit || 10)) + 1 : 1,
      pageSize: params.limit || 10,
      totalPages: Math.ceil(searchResult.total / (params.limit || 10))
    };
  }

  async getTechnique(id: string): Promise<Technique> {
    const response = await apiClient.get<any>(`${this.basePath}/${id}`);
    return this.mapTechnique(response.data);
  }

  async createTechnique(request: CreateTechniqueRequest): Promise<Technique> {
    const response = await apiClient.post<any>(this.basePath, request);
    return this.mapTechnique(response.data);
  }

  async updateTechnique(request: UpdateTechniqueRequest): Promise<Technique> {
    const response = await apiClient.put<any>(`${this.basePath}/${request.id}`, request);
    return this.mapTechnique(response.data);
  }

  async deleteTechnique(id: string): Promise<void> {
    await apiClient.delete(`${this.basePath}/${id}`);
  }

  async getCategories(): Promise<TechniqueCategory[]> {
    return Object.values(TechniqueCategory);
  }

  async getTags(): Promise<string[]> {
    const response = await apiClient.get<string[]>(`${this.basePath}/tags`);
    return response.data;
  }

  async getStats(): Promise<TechniqueStats> {
    const response = await apiClient.get<TechniqueStats>(`${this.basePath}/stats`);
    return response.data;
  }

  async getCombinations(): Promise<TechniqueCombination[]> {
    const response = await apiClient.get<TechniqueCombination[]>(`${this.basePath}/combinations`);
    return response.data;
  }

  async testTechnique(id: string, input: string, parameters: Record<string, any>): Promise<string> {
    const response = await apiClient.post<{ output: string }>(`${this.basePath}/${id}/test`, { input, parameters });
    return response.data.output;
  }

  // Helper methods for display
  getCategoryIcon(category: TechniqueCategory): string {
    const icons: Record<TechniqueCategory, string> = {
      [TechniqueCategory.BASIC]: '📝',
      [TechniqueCategory.COGNITIVE]: '🧠',
      [TechniqueCategory.OBFUSCATION]: '🎭',
      [TechniqueCategory.PERSONA]: '👤',
      [TechniqueCategory.CONTEXT]: '🔗',
      [TechniqueCategory.LOGIC]: '⚡',
      [TechniqueCategory.MULTIMODAL]: '🎨',
      [TechniqueCategory.AGENTIC]: '🤖',
      [TechniqueCategory.PAYLOAD]: '💣',
      [TechniqueCategory.ADVANCED]: '🔬',
    };
    return icons[category] || '📄';
  }

  getCategoryDisplayName(category: TechniqueCategory): string {
    const names: Record<TechniqueCategory, string> = {
      [TechniqueCategory.BASIC]: 'Basic Techniques',
      [TechniqueCategory.COGNITIVE]: 'Cognitive Hacking',
      [TechniqueCategory.OBFUSCATION]: 'Obfuscation Methods',
      [TechniqueCategory.PERSONA]: 'Persona Injection',
      [TechniqueCategory.CONTEXT]: 'Context Manipulation',
      [TechniqueCategory.LOGIC]: 'Logical Inference',
      [TechniqueCategory.MULTIMODAL]: 'Multi-Modal Attacks',
      [TechniqueCategory.AGENTIC]: 'Agentic Exploitation',
      [TechniqueCategory.PAYLOAD]: 'Payload Techniques',
      [TechniqueCategory.ADVANCED]: 'Advanced Methods',
    };
    return names[category] || category.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  getCategoryColor(category: TechniqueCategory): string {
    const colors: Record<TechniqueCategory, string> = {
      [TechniqueCategory.BASIC]: 'blue',
      [TechniqueCategory.COGNITIVE]: 'purple',
      [TechniqueCategory.OBFUSCATION]: 'orange',
      [TechniqueCategory.PERSONA]: 'green',
      [TechniqueCategory.CONTEXT]: 'yellow',
      [TechniqueCategory.LOGIC]: 'red',
      [TechniqueCategory.MULTIMODAL]: 'indigo',
      [TechniqueCategory.AGENTIC]: 'pink',
      [TechniqueCategory.PAYLOAD]: 'gray',
      [TechniqueCategory.ADVANCED]: 'cyan',
    };
    return colors[category] || 'default';
  }

  // Add other helper methods used in the component
  getDifficultyColor(difficulty: TechniqueDifficulty): string {
    const colors: Record<TechniqueDifficulty, string> = {
      [TechniqueDifficulty.BEGINNER]: 'green',
      [TechniqueDifficulty.INTERMEDIATE]: 'yellow',
      [TechniqueDifficulty.ADVANCED]: 'orange',
      [TechniqueDifficulty.EXPERT]: 'red',
    };
    return colors[difficulty] || 'default';
  }

  getDifficultyDisplayName(difficulty: TechniqueDifficulty): string {
    return difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
  }

  getEffectivenessColor(effectiveness: TechniqueEffectiveness): string {
    const colors: Record<TechniqueEffectiveness, string> = {
      [TechniqueEffectiveness.LOW]: 'gray',
      [TechniqueEffectiveness.MEDIUM]: 'yellow',
      [TechniqueEffectiveness.HIGH]: 'orange',
      [TechniqueEffectiveness.VERY_HIGH]: 'red',
    };
    return colors[effectiveness] || 'default';
  }

  getEffectivenessDisplayName(effectiveness: TechniqueEffectiveness): string {
    return effectiveness.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  formatSuccessRate(rate: number): string {
    return `${Math.round(rate * 100)}%`;
  }

  formatDetectionDifficulty(difficulty: number): string {
    if (difficulty >= 0.8) return 'Very Hard';
    if (difficulty >= 0.6) return 'Hard';
    if (difficulty >= 0.4) return 'Medium';
    if (difficulty >= 0.2) return 'Easy';
    return 'Very Easy';
  }

  formatResponseTime(timeMs: number): string {
    if (timeMs < 1000) return `${timeMs}ms`;
    if (timeMs < 60000) return `${(timeMs / 1000).toFixed(1)}s`;
    return `${(timeMs / 60000).toFixed(1)}m`;
  }

  getAttackVectorDisplayName(vector: AttackVector): string {
    return vector.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  getModalityTypeDisplayName(modality: ModalityType): string {
    return modality.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
}

export const techniqueLibraryService = new TechniqueLibraryService();

// Add aliases for compatibility with existing imports
export type AttackTechnique = Technique;
export interface TechniqueListResponse extends TechniqueSearchResult {
  page: number;
  pageSize: number;
  totalPages: number;
}
export interface TechniqueStats {
  totalTechniques: number;
  total_techniques: number; // Add snake_case alias
  byCategory: Record<TechniqueCategory, number>;
  byDifficulty: Record<TechniqueDifficulty, number>;
  byEffectiveness: Record<TechniqueEffectiveness, number>;
  averageSuccessRate: number;
  mostUsedTechniques: string[];
  recentlyAdded: string[];
  // Add missing properties used in components
  categories_count: number;
  most_effective_techniques: AttackTechnique[];
  popular_techniques: AttackTechnique[];
}
export interface TechniqueCombination {
  id: string;
  name: string;
  techniqueIds: string[];
  technique_ids: string[]; // Add snake_case alias
  techniques: AttackTechnique[];
  description: string;
  effectiveness: TechniqueEffectiveness;
  usageCount: number;
  successRate: number;
  createdAt: Date;
  // Add missing properties used in components
  difficulty: TechniqueDifficulty;
  synergy_score: number;
  execution_order: string[];
  use_cases: string[];
}
export type TechniqueFilters = TechniqueSearchParams;

// Add missing types that are referenced
export type AttackVector = string;
export type ModalityType = string;
