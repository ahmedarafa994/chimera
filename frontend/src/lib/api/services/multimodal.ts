import { apiClient } from '../client';
import { toast } from 'sonner';

/**
 * Multi-Modal Testing Service API
 */

export enum ModalityType {
  TEXT_ONLY = 'text_only',
  VISION_TEXT = 'vision_text',
  AUDIO_TEXT = 'audio_text',
  VIDEO_TEXT = 'video_text',
  MULTI_MODAL = 'multi_modal'
}

export enum AttackVector {
  VISUAL_PROMPT_INJECTION = 'visual_prompt_injection',
  IMAGE_CAPTION_MANIPULATION = 'image_caption_manipulation',
  STEGANOGRAPHIC_EMBEDDING = 'steganographic_embedding',
  AUDIO_PROMPT_INJECTION = 'audio_prompt_injection',
  ADVERSARIAL_IMAGES = 'adversarial_images',
  CROSS_MODAL_CONFUSION = 'cross_modal_confusion',
  MULTIMODAL_JAILBREAKING = 'multimodal_jailbreaking',
  CONTEXT_HIJACKING = 'context_hijacking'
}

export enum TestStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  ERROR = 'error'
}

export enum VulnerabilityLevel {
  NONE = 'none',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface MediaFile {
  file_id: string;
  filename: string;
  content_type: string;
  file_size: number;
  file_path: string;
  uploaded_at: string;
  width?: number;
  height?: number;
  duration?: number;
  format?: string;
}

export interface MultimodalTest {
  suite_id: string;
  id: string; // support both suite_id and id
  name: string;
  description?: string;
  status: TestStatus;
  vulnerability_score: number; // 0-100
  vulnerability_level: VulnerabilityLevel;
  findings: Array<{
    finding_id: string;
    description: string;
    modality_involved: ModalityType;
    evidence_url?: string;
    severity: string;
  }>;
  created_at: string;
  completed_at?: string;
  total_tests?: number;
  modality_types?: ModalityType[];
  attack_vectors?: AttackVector[];
  target_models?: string[];
}

export interface MultimodalTestCreate {
  name: string;
  description?: string;
  target_models: string[];
  attack_vectors: AttackVector[];
  modality_types: ModalityType[];
}

export type MultiModalAnalytics = any;
export type TestSuiteListResponse = { suites: MultimodalTest[]; total: number };
export type ExecutionListResponse = { executions: any[]; total: number };
export type MediaUploadResponse = MediaFile;
export type MultiModalTestSuite = MultimodalTest;
export type MultiModalTestExecution = {
  execution_id: string;
  suite_id: string;
  status: TestStatus;
  started_at: string;
  completed_at?: string;
  progress: number;
  results?: any;
  logs?: string[];
  vulnerability_detected?: boolean;
  vulnerability_level?: VulnerabilityLevel;
};
export type MultiModalPromptCreate = any;
export type MultiModalTestCreate = MultimodalTestCreate;
export type MultiModalTestUpdate = {
  name?: string;
  description?: string;
  target_models?: string[];
  attack_vectors?: AttackVector[];
  modality_types?: ModalityType[];
};

export class MultimodalTestingService {
  private readonly baseUrl = '/multimodal';

  /**
   * Get multimodal testing analytics
   */
  async getAnalytics(): Promise<MultiModalAnalytics> {
    const response = await apiClient.get<MultiModalAnalytics>(`${this.baseUrl}/analytics`);
    return response.data;
  }

  /**
   * List multimodal test suites
   */
  async listTestSuites(params?: any): Promise<TestSuiteListResponse> {
    const response = await apiClient.get<TestSuiteListResponse>(`${this.baseUrl}/suites`, { params });
    return response.data;
  }

  /**
   * List executions for a specific suite
   */
  async listSuiteExecutions(suiteId: string): Promise<ExecutionListResponse> {
    const response = await apiClient.get<ExecutionListResponse>(`${this.baseUrl}/suites/${suiteId}/executions`);
    return response.data;
  }

  /**
   * Upload media file for multimodal testing
   */
  async uploadMediaFile(file: File): Promise<MediaUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<MediaUploadResponse>(`${this.baseUrl}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  /**
   * Create a new test suite
   */
  async createTestSuite(data: MultimodalTestCreate): Promise<MultiModalTestSuite> {
    const response = await apiClient.post<MultiModalTestSuite>(`${this.baseUrl}/suites`, data);
    return response.data;
  }

  /**
   * Execute a test suite
   */
  async executeTestSuite(suiteId: string): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/suites/${suiteId}/execute`);
    return response.data;
  }

  /**
   * Delete a test suite
   */
  async deleteTestSuite(suiteId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/suites/${suiteId}`);
  }

  // --- UI Helpers ---

  validateTestSuiteCreate(data: MultimodalTestCreate): string[] {
    const errors: string[] = [];
    if (!data.name) errors.push('Name is required');
    if (data.target_models.length === 0) errors.push('Select at least one target model');
    if (data.attack_vectors.length === 0) errors.push('Select at least one attack vector');
    if (data.modality_types.length === 0) errors.push('Select at least one modality type');
    return errors;
  }

  getAvailableModalityTypes(): Array<{id: ModalityType; name: string}> {
    return [
      { id: ModalityType.TEXT_ONLY, name: 'Text Only' },
      { id: ModalityType.VISION_TEXT, name: 'Vision + Text' },
      { id: ModalityType.AUDIO_TEXT, name: 'Audio + Text' },
      { id: ModalityType.VIDEO_TEXT, name: 'Video + Text' },
      { id: ModalityType.MULTI_MODAL, name: 'Multi-Modal' }
    ];
  }

  getSuggestedModels(): any[] {
    return [
      { id: 'gpt-4o', name: 'GPT-4o (Vision)' },
      { id: 'claude-3-5-sonnet', name: 'Claude 3.5 Sonnet' },
      { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
    ];
  }

  getAvailableAttackVectors(): Array<{id: AttackVector; name: string}> {
    return Object.values(AttackVector).map(vector => ({
      id: vector,
      name: this.getAttackVectorDisplayName(vector)
    }));
  }

  getAttackVectorDisplayName(vector: AttackVector): string {
    return vector.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  }

  getModalityTypeDisplayName(type: ModalityType): string {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  }

  getVulnerabilityLevelColor(level: VulnerabilityLevel): string {
    const colors: Record<VulnerabilityLevel, string> = {
      none: 'green',
      low: 'yellow',
      medium: 'orange',
      high: 'red',
      critical: 'purple'
    };
    return colors[level] || 'gray';
  }

  getVulnerabilityLevelDisplayName(level: VulnerabilityLevel): string {
    return level.charAt(0).toUpperCase() + level.slice(1);
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Support legacy methods if any
  getModalityIcon(modality: ModalityType): string {
    const icons: Record<ModalityType, string> = {
      text_only: '📄',
      vision_text: '👁️',
      audio_text: '🔊',
      video_text: '🎥',
      multi_modal: '🌈'
    };
    return icons[modality];
  }
}

export const multimodalTestingService = new MultimodalTestingService();
