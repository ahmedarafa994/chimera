import { apiClient } from '../client';
import { toast } from 'sonner';

// API Response Types
interface TechniqueApiResponse {
  technique_id?: string;
  id?: string;
  techniqueId?: string;
  name?: string;
  description?: string;
  category?: string;
  complexity_score?: number;
  success_rate?: number;
  parameters?: any[];
  tags?: string[];
  created_at?: string;
  updated_at?: string;
  average_execution_time?: number;
  estimated_execution_time?: number;
  required_techniques?: string[];
  sample_prompt?: string;
  sample_output?: string;
  best_practices?: string[];
  limitations?: string[];
  usage_count?: number;
}

interface TechniqueSearchApiResponse {
  techniques?: TechniqueApiResponse[];
  total_techniques?: number;
  total?: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

interface TechniqueStatsApiResponse {
  totalTechniques?: number;
  total_techniques?: number;
  byCategory?: Record<string, number>;
  mostUsedTechniques?: string[];
  averageSuccessRate?: number;
  popular_techniques?: Array<{ usageCount?: number }>;
  most_effective_techniques?: TechniqueApiResponse[];
}

interface TechniqueCombinationApiResponse {
  id: string;
  name: string;
  technique_ids?: string[];
  techniqueIds?: string[];
  description?: string;
  successRate?: number;
  success_rate?: number;
  synergy_score?: number;
  execution_order?: string[];
  use_cases?: string[];
}

interface TechniqueTestApiResponse {
  output?: string;
}

interface TechniqueCombinationsApiResponse extends Array<TechniqueCombinationApiResponse> {}

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
  ADVANCED = 'advanced'
}

export enum TechniqueDifficulty {
  BEGINNER = 'beginner',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
  EXPERT = 'expert'
}

export enum TechniqueEffectiveness {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  VERY_HIGH = 'very_high'
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
  createdAt: string;
  updatedAt: string;
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

export interface TechniqueSearchParams {
  page?: number;
  page_size?: number;
  query?: string;
  search?: string;
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
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export type AttackTechnique = Technique;
export type TechniqueListResponse = TechniqueSearchResult;
export type TechniqueFilters = TechniqueSearchParams;

export interface TechniqueStats {
  totalTechniques: number;
  total_techniques: number;
  categories: Record<string, number>;
  categories_count: number;
  mostUsedTags: string[];
  most_effective_techniques: Technique[];
  popular_techniques: Technique[];
  avgSuccessRate: number;
  totalUsage: number;
}

export interface TechniqueCombination {
  id: string;
  name: string;
  techniqueIds: string[];
  technique_ids: string[];
  description: string;
  estimatedEffectiveness: TechniqueEffectiveness;
  effectiveness: TechniqueEffectiveness;
  synergy_score: number;
  execution_order: string[];
  use_cases: string[];
}

class TechniqueLibraryService {
  private readonly baseUri = '/technique-builder';

  private mapDifficulty(complexity?: number): TechniqueDifficulty {
    if (complexity && complexity >= 8) return TechniqueDifficulty.EXPERT;
    if (complexity && complexity >= 6) return TechniqueDifficulty.ADVANCED;
    if (complexity && complexity >= 3) return TechniqueDifficulty.INTERMEDIATE;
    return TechniqueDifficulty.BEGINNER;
  }

  private mapEffectiveness(successRate?: number): TechniqueEffectiveness {
    if (successRate !== undefined && successRate >= 0.75) return TechniqueEffectiveness.VERY_HIGH;
    if (successRate !== undefined && successRate >= 0.5) return TechniqueEffectiveness.HIGH;
    if (successRate !== undefined && successRate >= 0.25) return TechniqueEffectiveness.MEDIUM;
    return TechniqueEffectiveness.LOW;
  }

  private mapCategory(category?: string): TechniqueCategory {
    const normalized = (category || '').toLowerCase();
    return (Object.values(TechniqueCategory) as string[]).includes(normalized)
      ? (normalized as TechniqueCategory)
      : TechniqueCategory.BASIC;
  }

  private mapParameter(param: any): TechniqueParameter {
    const typeMap: Record<string, TechniqueParameter['type']> = {
      string: 'string',
      integer: 'number',
      float: 'number',
      number: 'number',
      boolean: 'boolean',
      list: 'select',
      object: 'select'
    };
    return {
      name: param.name,
      type: typeMap[param.parameter_type] ?? 'string',
      description: param.description ?? '',
      required: param.required ?? false,
      defaultValue: param.default_value,
      options: param.allowed_values
    };
  }

  private mapTechnique(apiTechnique: any): Technique {
    return {
      id: apiTechnique.technique_id ?? apiTechnique.id ?? apiTechnique.techniqueId ?? '',
      name: apiTechnique.name ?? 'Untitled technique',
      description: apiTechnique.description ?? '',
      category: this.mapCategory(apiTechnique.category),
      difficulty: this.mapDifficulty(apiTechnique.complexity_score),
      effectiveness: this.mapEffectiveness(apiTechnique.success_rate),
      parameters: (apiTechnique.parameters ?? []).map((p: any) => this.mapParameter(p)),
      examples: [],
      tags: apiTechnique.tags ?? [],
      createdAt: apiTechnique.created_at
        ? new Date(apiTechnique.created_at).toISOString()
        : new Date().toISOString(),
      updatedAt: apiTechnique.updated_at
        ? new Date(apiTechnique.updated_at).toISOString()
        : new Date().toISOString(),
      success_rate: apiTechnique.success_rate ?? 0,
      detection_difficulty: apiTechnique.complexity_score ?? 0,
      avg_response_time: apiTechnique.average_execution_time ?? apiTechnique.estimated_execution_time ?? 0,
      use_cases: apiTechnique.required_techniques ?? [],
      example_prompt: apiTechnique.sample_prompt,
      example_output: apiTechnique.sample_output,
      best_practices: apiTechnique.best_practices ?? [],
      limitations: apiTechnique.limitations ?? [],
      usage_count: apiTechnique.usage_count ?? 0
    };
  }

  async searchTechniques(params: TechniqueSearchParams): Promise<TechniqueSearchResult> {
    const pageSize = params.page_size ?? params.limit ?? (params.offset !== undefined ? 20 : undefined);
    const derivedPage =
      params.offset !== undefined && pageSize ? Math.floor(params.offset / pageSize) + 1 : undefined;

    const response = await apiClient.get(this.baseUri, {
      params: {
        page: params.page ?? derivedPage,
        page_size: pageSize,
        limit: pageSize,
        offset: params.offset,
        search: params.search ?? params.query,
        category: params.category,
        tags: params.tags
      }
    });

    const data = response.data as TechniqueSearchApiResponse;
    return {
      techniques: (data.techniques ?? []).map((t: TechniqueApiResponse) => this.mapTechnique(t)),
      total: data.total_techniques ?? data.total ?? (data.techniques?.length ?? 0),
      page: data.page ?? derivedPage ?? params.page ?? 1,
      page_size: data.page_size ?? pageSize ?? 20,
      has_next: Boolean(data.has_next),
      has_prev: Boolean(data.has_prev)
    };
  }

  async getTechnique(id: string): Promise<Technique> {
    const response = await apiClient.get(`${this.baseUri}/${id}`);
    return this.mapTechnique(response.data as TechniqueApiResponse);
  }

  async createTechnique(technique: Partial<Technique>): Promise<Technique> {
    try {
      const payload = {
        name: technique.name,
        description: technique.description,
        category: technique.category,
        technique_type: 'transformation',
        visibility: 'private',
        workspace_id: undefined,
        tags: technique.tags ?? []
      };
      const response = await apiClient.post(this.baseUri, payload);
      toast.success('Technique saved');
      return this.mapTechnique(response.data as TechniqueApiResponse);
    } catch (error) {
      console.error('Failed to create technique:', error);
      toast.error('Failed to create technique');
      throw error;
    }
  }

  async updateTechnique(id: string, technique: Partial<Technique>): Promise<Technique> {
    try {
      const response = await apiClient.patch(`${this.baseUri}/${id}`, {
        name: technique.name,
        description: technique.description,
        category: technique.category,
        status: technique.success_rate ? undefined : undefined,
        tags: technique.tags,
        parameters: technique.parameters
      });
      toast.success('Technique updated');
      return this.mapTechnique(response.data as TechniqueApiResponse);
    } catch (error) {
      console.error('Failed to update technique:', error);
      toast.error('Failed to update technique');
      throw error;
    }
  }

  async deleteTechnique(id: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUri}/${id}`);
      toast.success('Technique deleted');
    } catch (error) {
      console.error('Failed to delete technique:', error);
      toast.error('Failed to delete technique');
      throw error;
    }
  }

  async getCategories(): Promise<TechniqueCategory[]> {
    return Object.values(TechniqueCategory);
  }

  async getTags(): Promise<string[]> {
    const response = await apiClient.get<string[]>(`${this.baseUri}/tags`);
    return response.data ?? [];
  }

  async getStats(): Promise<TechniqueStats> {
    const response = await apiClient.get(`${this.baseUri}/stats`);
    const data = response.data as TechniqueStatsApiResponse;
    const techniques = (data.most_effective_techniques ?? []).map((t: TechniqueApiResponse) => this.mapTechnique(t));
    const popular = (data.popular_techniques ?? []).map((t: any) => this.mapTechnique(t));
    
    return {
      totalTechniques: data.totalTechniques ?? data.total_techniques ?? 0,
      total_techniques: data.totalTechniques ?? data.total_techniques ?? 0,
      categories: data.byCategory ?? {},
      categories_count: Object.keys(data.byCategory ?? {}).length,
      mostUsedTags: data.mostUsedTechniques ?? [],
      most_effective_techniques: techniques,
      popular_techniques: popular,
      avgSuccessRate: data.averageSuccessRate ?? 0,
      totalUsage: data.popular_techniques?.length ? data.popular_techniques.reduce((acc: number, item: { usageCount?: number }) => acc + (item.usageCount ?? 0), 0) : 0
    };
  }

  async getCombinations(): Promise<TechniqueCombination[]> {
    const response = await apiClient.get(`${this.baseUri}/combinations`);
    return (response.data as TechniqueCombinationsApiResponse ?? []).map((combo: TechniqueCombinationApiResponse) => {
      const effectiveness = this.mapEffectiveness(combo.successRate ?? combo.success_rate);
      const ids = combo.technique_ids ?? combo.techniqueIds ?? [];
      return {
        id: combo.id,
        name: combo.name,
        techniqueIds: ids,
        technique_ids: ids,
        description: combo.description ?? '',
        estimatedEffectiveness: effectiveness,
        effectiveness: effectiveness,
        synergy_score: combo.synergy_score ?? 0,
        execution_order: combo.execution_order ?? ids,
        use_cases: combo.use_cases ?? []
      };
    });
  }

  async testTechnique(id: string, input: string, parameters: Record<string, any>): Promise<string> {
    const response = await apiClient.post(`${this.baseUri}/${id}/test`, {
      test_input: input,
      parameters
    });
    return (response.data as TechniqueTestApiResponse)?.output ?? '';
  }

  getCategoryIcon(category: TechniqueCategory): string {
    const icons: Record<TechniqueCategory, string> = {
      [TechniqueCategory.BASIC]: '⛰️',
      [TechniqueCategory.COGNITIVE]: '🧠',
      [TechniqueCategory.OBFUSCATION]: '🌀',
      [TechniqueCategory.PERSONA]: '🎭',
      [TechniqueCategory.CONTEXT]: '🗂️',
      [TechniqueCategory.LOGIC]: '♟️',
      [TechniqueCategory.MULTIMODAL]: '🎛️',
      [TechniqueCategory.AGENTIC]: '🤖',
      [TechniqueCategory.PAYLOAD]: '📦',
      [TechniqueCategory.ADVANCED]: '🚀'
    };
    return icons[category] || '⚙️';
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
      [TechniqueCategory.ADVANCED]: 'cyan'
    };
    return colors[category] || 'default';
  }

  getDifficultyColor(difficulty: TechniqueDifficulty): string {
    const colors: Record<TechniqueDifficulty, string> = {
      [TechniqueDifficulty.BEGINNER]: 'green',
      [TechniqueDifficulty.INTERMEDIATE]: 'yellow',
      [TechniqueDifficulty.ADVANCED]: 'orange',
      [TechniqueDifficulty.EXPERT]: 'red'
    };
    return colors[difficulty] || 'default';
  }

  getEffectivenessDisplayName(effectiveness: TechniqueEffectiveness): string {
    const display: Record<TechniqueEffectiveness, string> = {
      [TechniqueEffectiveness.LOW]: 'Low',
      [TechniqueEffectiveness.MEDIUM]: 'Medium',
      [TechniqueEffectiveness.HIGH]: 'High',
      [TechniqueEffectiveness.VERY_HIGH]: 'Very High'
    };
    return display[effectiveness];
  }

  getDifficultyDisplayName(difficulty: TechniqueDifficulty): string {
    const display: Record<TechniqueDifficulty, string> = {
      [TechniqueDifficulty.BEGINNER]: 'Beginner',
      [TechniqueDifficulty.INTERMEDIATE]: 'Intermediate',
      [TechniqueDifficulty.ADVANCED]: 'Advanced',
      [TechniqueDifficulty.EXPERT]: 'Expert'
    };
    return display[difficulty];
  }

  formatSuccessRate(rate: number): string {
    return `${Math.round(rate * 100)}%`;
  }

  formatResponseTime(seconds: number): string {
    return `${Math.round(seconds)}s`;
  }

  formatDetectionDifficulty(score: number): string {
    if (score >= 8) return 'Very High';
    if (score >= 5) return 'High';
    if (score >= 3) return 'Medium';
    return 'Low';
  }

  getCategoryDisplayName(category: TechniqueCategory): string {
    const display: Record<TechniqueCategory, string> = {
      [TechniqueCategory.BASIC]: 'Basic',
      [TechniqueCategory.COGNITIVE]: 'Cognitive',
      [TechniqueCategory.OBFUSCATION]: 'Obfuscation',
      [TechniqueCategory.PERSONA]: 'Persona',
      [TechniqueCategory.CONTEXT]: 'Context',
      [TechniqueCategory.LOGIC]: 'Logic',
      [TechniqueCategory.MULTIMODAL]: 'Multimodal',
      [TechniqueCategory.AGENTIC]: 'Agentic',
      [TechniqueCategory.PAYLOAD]: 'Payload',
      [TechniqueCategory.ADVANCED]: 'Advanced'
    };
    return display[category];
  }

  getEffectivenessColor(effectiveness: TechniqueEffectiveness): string {
    const colors: Record<TechniqueEffectiveness, string> = {
      [TechniqueEffectiveness.LOW]: 'gray',
      [TechniqueEffectiveness.MEDIUM]: 'yellow',
      [TechniqueEffectiveness.HIGH]: 'orange',
      [TechniqueEffectiveness.VERY_HIGH]: 'green'
    };
    return colors[effectiveness] || 'gray';
  }

  listTechniques(params: TechniqueSearchParams): Promise<TechniqueSearchResult> {
    return this.searchTechniques(params);
  }
}

export const techniqueLibraryService = new TechniqueLibraryService();
