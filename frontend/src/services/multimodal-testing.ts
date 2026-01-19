/**
 * Multi-Modal Attack Testing Service
 *
 * Phase 4 innovation feature for next-generation security:
 * - Vision+text and audio+text attack capabilities
 * - Image captioning vulnerability testing
 * - Unified reporting across modalities
 * - Future-proofing for multi-modal LLMs
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';

export type ModalityType = 'text_only' | 'vision_text' | 'audio_text' | 'video_text' | 'multi_modal';
export type AttackVector = 'visual_prompt_injection' | 'image_caption_manipulation' | 'steganographic_embedding' | 'audio_prompt_injection' | 'adversarial_images' | 'cross_modal_confusion' | 'multimodal_jailbreaking' | 'context_hijacking';
export type TestStatus = 'pending' | 'running' | 'completed' | 'failed' | 'error';
export type VulnerabilityLevel = 'none' | 'low' | 'medium' | 'high' | 'critical';

export interface MediaFile {
  file_id: string;
  filename: string;
  content_type: string;
  file_size: number;
  file_path: string;
  uploaded_at: string;

  // Media metadata
  width?: number;
  height?: number;
  duration?: number;
  format?: string;
}

export interface MultiModalPrompt {
  prompt_id: string;
  text_content: string;
  media_files: MediaFile[];

  // Attack configuration
  attack_vector: AttackVector;
  modality_type: ModalityType;

  // Additional parameters
  instructions?: string;
  context?: string;
  metadata: Record<string, any>;
}

export interface MultiModalTestExecution {
  execution_id: string;
  prompt_id: string;
  model_name: string;
  provider: string;

  // Execution details
  started_at: string;
  completed_at?: string;
  duration_seconds?: number;
  status: TestStatus;

  // Input
  input_prompt: MultiModalPrompt;

  // Results
  model_response?: string;
  response_metadata: Record<string, any>;

  // Security analysis
  vulnerability_detected: boolean;
  vulnerability_level: VulnerabilityLevel;
  confidence_score: number;

  // Attack success metrics
  jailbreak_success: boolean;
  information_leaked: boolean;
  harmful_content_generated: boolean;

  // Analysis details
  analysis_results: Record<string, any>;
  cross_modal_consistency?: number;
  attention_analysis?: Record<string, any>;

  // Error details
  error_message?: string;

  // Metadata
  executed_by: string;
  workspace_id?: string;
}

export interface MultiModalTestSuite {
  suite_id: string;
  name: string;
  description: string;

  // Test configuration
  target_models: string[];
  attack_vectors: AttackVector[];
  modality_types: ModalityType[];

  // Execution settings
  timeout_seconds: number;
  parallel_execution: boolean;
  max_concurrent_tests: number;

  // Analysis settings
  enable_cross_modal_analysis: boolean;
  generate_adversarial_examples: boolean;
  analyze_attention_patterns: boolean;

  // Metadata
  created_by: string;
  workspace_id?: string;
  created_at: string;
  updated_at: string;

  // Execution results
  total_tests: number;
  completed_tests: number;
  vulnerabilities_found: number;
  success_rate: number;

  // Test executions
  executions: MultiModalTestExecution[];
}

export interface MultiModalTestCreate {
  name: string;
  description: string;
  target_models: string[];
  attack_vectors: AttackVector[];
  modality_types: ModalityType[];
  workspace_id?: string;
}

export interface MultiModalTestUpdate {
  name?: string;
  description?: string;
  target_models?: string[];
  attack_vectors?: AttackVector[];
  modality_types?: ModalityType[];
}

export interface MultiModalPromptCreate {
  text_content: string;
  attack_vector: AttackVector;
  modality_type: ModalityType;
  instructions?: string;
  context?: string;
}

export interface TestSuiteListResponse {
  suites: MultiModalTestSuite[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ExecutionListResponse {
  executions: MultiModalTestExecution[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface MultiModalAnalytics {
  total_suites: number;
  total_executions: number;
  vulnerability_rate: number;

  // By modality type
  modality_breakdown: Record<ModalityType, number>;
  vulnerability_by_modality: Record<ModalityType, number>;

  // By attack vector
  attack_vector_success: Record<AttackVector, number>;

  // Model performance
  model_robustness: Record<string, number>;

  // Trends
  daily_execution_trend: Array<{ date: string; executions: number }>;
  vulnerability_trend: Array<{ date: string; vulnerabilities: number }>;
}

export interface MediaUploadResponse {
  file_id: string;
  filename: string;
  content_type: string;
  file_size: number;
  metadata: {
    width?: number;
    height?: number;
    duration?: number;
    format?: string;
  };
}

class MultiModalTestingService {
  private readonly baseUrl = '/multimodal';

  /**
   * Create a new multi-modal test suite
   */
  async createTestSuite(suiteData: MultiModalTestCreate): Promise<MultiModalTestSuite> {
    try {
      const response = await apiClient.post<MultiModalTestSuite>(`${this.baseUrl}/suites`, suiteData);

      toast.success('Multi-modal test suite created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create multi-modal test suite:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create test suite');
      throw error;
    }
  }

  /**
   * Upload media file for multi-modal testing
   */
  async uploadMediaFile(file: File): Promise<MediaUploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiClient.post<MediaUploadResponse>(`${this.baseUrl}/media/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      toast.success('Media file uploaded successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to upload media file:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to upload media file');
      throw error;
    }
  }

  /**
   * Create multi-modal prompt with media files
   */
  async createPrompt(promptData: MultiModalPromptCreate, mediaFileIds: string[] = []): Promise<MultiModalPrompt> {
    try {
      const formData = new FormData();
      formData.append('text_content', promptData.text_content);
      formData.append('attack_vector', promptData.attack_vector);
      formData.append('modality_type', promptData.modality_type);

      if (promptData.instructions) {
        formData.append('instructions', promptData.instructions);
      }
      if (promptData.context) {
        formData.append('context', promptData.context);
      }

      mediaFileIds.forEach(fileId => {
        formData.append('media_file_ids', fileId);
      });

      const response = await apiClient.post<MultiModalPrompt>(`${this.baseUrl}/prompts`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      toast.success('Multi-modal prompt created successfully');
      return response.data;
    } catch (error) {
      console.error('Failed to create multi-modal prompt:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to create prompt');
      throw error;
    }
  }

  /**
   * Execute multi-modal test suite
   */
  async executeTestSuite(suiteId: string): Promise<{ message: string; suite_id: string }> {
    try {
      const response = await apiClient.post<{ message: string; suite_id: string }>(`${this.baseUrl}/suites/${suiteId}/execute`);

      toast.success('Multi-modal test execution started');
      return response.data;
    } catch (error) {
      console.error('Failed to execute test suite:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to execute test suite');
      throw error;
    }
  }

  /**
   * List multi-modal test suites
   */
  async listTestSuites(params?: { page?: number; page_size?: number; modality_type?: ModalityType }): Promise<TestSuiteListResponse> {
    try {
      const response = await apiClient.get<TestSuiteListResponse>(`${this.baseUrl}/suites`, {
        params: {
          page: params?.page,
          page_size: params?.page_size,
          modality_type: params?.modality_type
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list test suites:', error);
      toast.error('Failed to load test suites');
      throw error;
    }
  }

  /**
   * List executions for a test suite
   */
  async listSuiteExecutions(suiteId: string, params?: { page?: number; page_size?: number }): Promise<ExecutionListResponse> {
    try {
      const response = await apiClient.get<ExecutionListResponse>(`${this.baseUrl}/suites/${suiteId}/executions`, {
        params: {
          page: params?.page,
          page_size: params?.page_size
        }
      });

      return response.data;
    } catch (error) {
      console.error('Failed to list suite executions:', error);
      toast.error('Failed to load executions');
      throw error;
    }
  }

  /**
   * Get multi-modal analytics
   */
  async getAnalytics(): Promise<MultiModalAnalytics> {
    try {
      const response = await apiClient.get<MultiModalAnalytics>(`${this.baseUrl}/analytics`);

      return response.data;
    } catch (error) {
      console.error('Failed to get analytics:', error);
      toast.error('Failed to load analytics');
      throw error;
    }
  }

  /**
   * Delete test suite
   */
  async deleteTestSuite(suiteId: string): Promise<void> {
    try {
      await apiClient.delete(`${this.baseUrl}/suites/${suiteId}`);

      toast.success('Test suite deleted successfully');
    } catch (error) {
      console.error('Failed to delete test suite:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete test suite');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for modality type
   */
  getModalityTypeDisplayName(type: ModalityType): string {
    const displayNames: Record<ModalityType, string> = {
      text_only: 'Text Only',
      vision_text: 'Vision + Text',
      audio_text: 'Audio + Text',
      video_text: 'Video + Text',
      multi_modal: 'Multi-Modal'
    };
    return displayNames[type];
  }

  /**
   * Get display name for attack vector
   */
  getAttackVectorDisplayName(vector: AttackVector): string {
    const displayNames: Record<AttackVector, string> = {
      visual_prompt_injection: 'Visual Prompt Injection',
      image_caption_manipulation: 'Image Caption Manipulation',
      steganographic_embedding: 'Steganographic Embedding',
      audio_prompt_injection: 'Audio Prompt Injection',
      adversarial_images: 'Adversarial Images',
      cross_modal_confusion: 'Cross-Modal Confusion',
      multimodal_jailbreaking: 'Multi-Modal Jailbreaking',
      context_hijacking: 'Context Hijacking'
    };
    return displayNames[vector];
  }

  /**
   * Get color for vulnerability level
   */
  getVulnerabilityLevelColor(level: VulnerabilityLevel): string {
    const colors: Record<VulnerabilityLevel, string> = {
      none: 'gray',
      low: 'blue',
      medium: 'yellow',
      high: 'orange',
      critical: 'red'
    };
    return colors[level];
  }

  /**
   * Get display name for vulnerability level
   */
  getVulnerabilityLevelDisplayName(level: VulnerabilityLevel): string {
    const displayNames: Record<VulnerabilityLevel, string> = {
      none: 'None',
      low: 'Low',
      medium: 'Medium',
      high: 'High',
      critical: 'Critical'
    };
    return displayNames[level];
  }

  /**
   * Get color for test status
   */
  getTestStatusColor(status: TestStatus): string {
    const colors: Record<TestStatus, string> = {
      pending: 'gray',
      running: 'blue',
      completed: 'green',
      failed: 'orange',
      error: 'red'
    };
    return colors[status];
  }

  /**
   * Get display name for test status
   */
  getTestStatusDisplayName(status: TestStatus): string {
    const displayNames: Record<TestStatus, string> = {
      pending: 'Pending',
      running: 'Running',
      completed: 'Completed',
      failed: 'Failed',
      error: 'Error'
    };
    return displayNames[status];
  }

  /**
   * Format file size
   */
  formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  /**
   * Format duration
   */
  formatDuration(seconds: number): string {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  }

  /**
   * Format percentage
   */
  formatPercentage(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  /**
   * Validate test suite creation data
   */
  validateTestSuiteCreate(data: MultiModalTestCreate): string[] {
    const errors: string[] = [];

    if (!data.name || data.name.trim().length === 0) {
      errors.push('Test suite name is required');
    }

    if (data.name && data.name.length > 100) {
      errors.push('Test suite name must be less than 100 characters');
    }

    if (!data.description || data.description.trim().length === 0) {
      errors.push('Description is required');
    }

    if (data.description && data.description.length > 1000) {
      errors.push('Description must be less than 1000 characters');
    }

    if (!data.target_models || data.target_models.length === 0) {
      errors.push('At least one target model is required');
    }

    if (!data.attack_vectors || data.attack_vectors.length === 0) {
      errors.push('At least one attack vector is required');
    }

    if (!data.modality_types || data.modality_types.length === 0) {
      errors.push('At least one modality type is required');
    }

    return errors;
  }

  /**
   * Get available modality types
   */
  getAvailableModalityTypes(): Array<{id: ModalityType, name: string, description: string}> {
    return [
      {
        id: 'text_only',
        name: 'Text Only',
        description: 'Traditional text-based attacks'
      },
      {
        id: 'vision_text',
        name: 'Vision + Text',
        description: 'Image and text combination attacks'
      },
      {
        id: 'audio_text',
        name: 'Audio + Text',
        description: 'Audio and text combination attacks'
      },
      {
        id: 'video_text',
        name: 'Video + Text',
        description: 'Video and text combination attacks'
      },
      {
        id: 'multi_modal',
        name: 'Multi-Modal',
        description: 'Complex multi-modality attacks'
      }
    ];
  }

  /**
   * Get available attack vectors
   */
  getAvailableAttackVectors(): Array<{id: AttackVector, name: string, description: string}> {
    return [
      {
        id: 'visual_prompt_injection',
        name: 'Visual Prompt Injection',
        description: 'Inject malicious prompts through images'
      },
      {
        id: 'image_caption_manipulation',
        name: 'Image Caption Manipulation',
        description: 'Exploit image captioning vulnerabilities'
      },
      {
        id: 'steganographic_embedding',
        name: 'Steganographic Embedding',
        description: 'Hide prompts in media using steganography'
      },
      {
        id: 'audio_prompt_injection',
        name: 'Audio Prompt Injection',
        description: 'Inject prompts through audio channels'
      },
      {
        id: 'adversarial_images',
        name: 'Adversarial Images',
        description: 'Use adversarially crafted images'
      },
      {
        id: 'cross_modal_confusion',
        name: 'Cross-Modal Confusion',
        description: 'Exploit confusion between modalities'
      },
      {
        id: 'multimodal_jailbreaking',
        name: 'Multi-Modal Jailbreaking',
        description: 'Advanced jailbreaking across modalities'
      },
      {
        id: 'context_hijacking',
        name: 'Context Hijacking',
        description: 'Hijack context through media manipulation'
      }
    ];
  }

  /**
   * Get suggested models for multi-modal testing
   */
  getSuggestedModels(): Array<{id: string, name: string, modalities: ModalityType[]}> {
    return [
      {
        id: 'gpt-4-vision-preview',
        name: 'GPT-4 Vision',
        modalities: ['vision_text', 'multi_modal']
      },
      {
        id: 'claude-3-5-sonnet',
        name: 'Claude 3.5 Sonnet',
        modalities: ['vision_text', 'multi_modal']
      },
      {
        id: 'gemini-1.5-pro-vision',
        name: 'Gemini Pro Vision',
        modalities: ['vision_text', 'audio_text', 'video_text', 'multi_modal']
      },
      {
        id: 'llava-1.6-34b',
        name: 'LLaVA 1.6 34B',
        modalities: ['vision_text']
      },
      {
        id: 'whisper-large-v3',
        name: 'Whisper Large v3',
        modalities: ['audio_text']
      }
    ];
  }
}

// Export singleton instance
export const multiModalTestingService = new MultiModalTestingService();
