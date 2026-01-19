/**
 * Interactive Help & Documentation Hub Service
 *
 * Phase 2 feature for competitive differentiation:
 * - In-app searchable documentation
 * - Contextual help and best practices
 * - Ethical guidelines and legal considerations
 * - Video tutorials and getting started guides
 */

import { toast } from 'sonner';
import { apiClient } from '@/lib/api/client';

export type DocumentationType =
  | 'tutorial'
  | 'guide'
  | 'reference'
  | 'faq'
  | 'ethical_guidelines'
  | 'legal'
  | 'best_practices'
  | 'troubleshooting';

export type DocumentationCategory =
  | 'getting_started'
  | 'attack_techniques'
  | 'providers'
  | 'reporting'
  | 'collaboration'
  | 'advanced'
  | 'ethics_legal'
  | 'api';

export interface DocumentationSection {
  title: string;
  anchor: string;
}

export interface DocumentationItem {
  id: string;
  title: string;
  content: string;
  type: DocumentationType;
  category: DocumentationCategory;

  // Metadata
  tags: string[];
  difficulty_level: 'beginner' | 'intermediate' | 'advanced';
  estimated_read_time: number; // minutes

  // Navigation
  parent_id?: string;
  order: number;

  // Content structure
  sections: DocumentationSection[];
  related_articles: string[];

  // Media
  video_url?: string;
  images: string[];

  // Tracking
  views: number;
  last_updated: string;
  author: string;
}

export interface VideoTutorial {
  id: string;
  title: string;
  description: string;
  video_url: string;
  thumbnail_url?: string;
  duration: number; // seconds
  category: DocumentationCategory;
  tags: string[];
  transcript?: string;
  created_at: string;
  views: number;
}

export interface SearchRequest {
  query: string;
  category?: DocumentationCategory;
  type?: DocumentationType;
  difficulty_level?: string;
  tags?: string[];
}

export interface SearchResult {
  item: DocumentationItem;
  relevance_score: number;
  matching_sections: string[];
  highlight_text: string;
}

export interface DocumentationListResponse {
  items: DocumentationItem[];
  categories: Array<{
    id: string;
    name: string;
    count: number;
  }>;
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface HelpContext {
  page_url: string;
  user_action?: string;
  feature?: string;
  error_code?: string;
}

export interface ContextualHelpResponse {
  suggested_item?: DocumentationItem;
  related_items: DocumentationItem[];
  quick_actions: Array<{
    label: string;
    action: string;
    target: string;
  }>;
}

export interface QuickStartGuide {
  welcome_message: string;
  progress: {
    api_keys_configured: boolean;
    first_assessment_completed: boolean;
    first_report_generated: boolean;
  };
  next_steps: Array<{
    title: string;
    description: string;
    action: string;
    target: string;
    estimated_time: string;
    priority: 'high' | 'medium' | 'low';
  }>;
  featured_content: DocumentationItem[];
}

export interface DocumentationListParams {
  category?: DocumentationCategory;
  type?: DocumentationType;
  difficulty?: string;
  page?: number;
  page_size?: number;
}

class DocumentationService {
  private readonly baseUrl = '/docs';

  /**
   * List documentation items with filtering and pagination
   */
  async listDocumentation(params?: DocumentationListParams): Promise<DocumentationListResponse> {
    try {
      const response = await apiClient.get<DocumentationListResponse>(`${this.baseUrl}/`, {
        params
      });
      return response.data;
    } catch (error) {
      console.error('Failed to list documentation:', error);
      toast.error('Failed to load documentation');
      throw error;
    }
  }

  /**
   * Get specific documentation item
   */
  async getDocumentationItem(itemId: string): Promise<DocumentationItem> {
    try {
      const response = await apiClient.get<DocumentationItem>(`${this.baseUrl}/${itemId}`);
      return response.data;
    } catch (error: any) {
      console.error('Failed to get documentation item:', error);
      toast.error(error.response?.data?.detail || 'Failed to load documentation item');
      throw error;
    }
  }

  /**
   * Search documentation content
   */
  async searchDocumentation(searchRequest: SearchRequest): Promise<SearchResult[]> {
    try {
      const response = await apiClient.post<SearchResult[]>(`${this.baseUrl}/search`, searchRequest);
      return response.data;
    } catch (error) {
      console.error('Failed to search documentation:', error);
      toast.error('Failed to search documentation');
      throw error;
    }
  }

  /**
   * Get documentation by category
   */
  async getDocumentationByCategory(category: DocumentationCategory): Promise<DocumentationItem[]> {
    try {
      const response = await apiClient.get<DocumentationItem[]>(`${this.baseUrl}/categories/${category}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get documentation by category:', error);
      toast.error('Failed to load category documentation');
      throw error;
    }
  }

  /**
   * Get contextual help based on current page/action
   */
  async getContextualHelp(context: HelpContext): Promise<ContextualHelpResponse> {
    try {
      const response = await apiClient.post<ContextualHelpResponse>(`${this.baseUrl}/context-help`, context);
      return response.data;
    } catch (error) {
      console.error('Failed to get contextual help:', error);
      toast.error('Failed to load contextual help');
      throw error;
    }
  }

  /**
   * List video tutorials
   */
  async listVideoTutorials(category?: DocumentationCategory): Promise<VideoTutorial[]> {
    try {
      const response = await apiClient.get<VideoTutorial[]>(`${this.baseUrl}/videos/`, {
        params: { category }
      });
      return response.data;
    } catch (error) {
      console.error('Failed to list video tutorials:', error);
      toast.error('Failed to load video tutorials');
      throw error;
    }
  }

  /**
   * Get quick start guide
   */
  async getQuickStartGuide(): Promise<QuickStartGuide> {
    try {
      const response = await apiClient.get<QuickStartGuide>(`${this.baseUrl}/quick-start`);
      return response.data;
    } catch (error) {
      console.error('Failed to get quick start guide:', error);
      toast.error('Failed to load quick start guide');
      throw error;
    }
  }

  // Utility functions for UI display

  /**
   * Get display name for documentation type
   */
  getTypeDisplayName(type: DocumentationType): string {
    const displayNames: Record<DocumentationType, string> = {
      tutorial: 'Tutorial',
      guide: 'Guide',
      reference: 'Reference',
      faq: 'FAQ',
      ethical_guidelines: 'Ethical Guidelines',
      legal: 'Legal Information',
      best_practices: 'Best Practices',
      troubleshooting: 'Troubleshooting'
    };
    return displayNames[type];
  }

  /**
   * Get display name for documentation category
   */
  getCategoryDisplayName(category: DocumentationCategory): string {
    const displayNames: Record<DocumentationCategory, string> = {
      getting_started: 'Getting Started',
      attack_techniques: 'Attack Techniques',
      providers: 'Providers',
      reporting: 'Reporting',
      collaboration: 'Collaboration',
      advanced: 'Advanced',
      ethics_legal: 'Ethics & Legal',
      api: 'API'
    };
    return displayNames[category];
  }

  /**
   * Get color for difficulty level
   */
  getDifficultyColor(level: string): string {
    const colors: Record<string, string> = {
      beginner: 'green',
      intermediate: 'yellow',
      advanced: 'red'
    };
    return colors[level] || 'gray';
  }

  /**
   * Format reading time
   */
  formatReadingTime(minutes: number): string {
    if (minutes < 1) {
      return 'Less than 1 minute';
    } else if (minutes === 1) {
      return '1 minute read';
    } else {
      return `${minutes} minute read`;
    }
  }

  /**
   * Format video duration
   */
  formatVideoDuration(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    } else {
      return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
  }

  /**
   * Generate table of contents from sections
   */
  generateTableOfContents(sections: DocumentationSection[]): string {
    if (sections.length === 0) return '';

    return sections.map(section =>
      `- [${section.title}](#${section.anchor})`
    ).join('\n');
  }

  /**
   * Extract sections from content
   */
  extractSections(content: string): DocumentationSection[] {
    const sections: DocumentationSection[] = [];
    const lines = content.split('\n');

    for (const line of lines) {
      const match = line.match(/^#{1,6}\s+(.+)$/);
      if (match) {
        const title = match[1];
        const anchor = title
          .toLowerCase()
          .replace(/[^a-z0-9\s-]/g, '')
          .replace(/\s+/g, '-')
          .trim();

        sections.push({ title, anchor });
      }
    }

    return sections;
  }

  /**
   * Validate search request
   */
  validateSearchRequest(request: SearchRequest): string[] {
    const errors: string[] = [];

    if (!request.query || request.query.trim().length === 0) {
      errors.push('Search query is required');
    }

    if (request.query && request.query.length > 200) {
      errors.push('Search query must be less than 200 characters');
    }

    return errors;
  }

  /**
   * Get category icon
   */
  getCategoryIcon(category: DocumentationCategory): string {
    const icons: Record<DocumentationCategory, string> = {
      getting_started: '🚀',
      attack_techniques: '⚔️',
      providers: '🔗',
      reporting: '📊',
      collaboration: '👥',
      advanced: '🎓',
      ethics_legal: '⚖️',
      api: '🔧'
    };
    return icons[category] || '📄';
  }

  /**
   * Get type badge color
   */
  getTypeBadgeColor(type: DocumentationType): string {
    const colors: Record<DocumentationType, string> = {
      tutorial: 'blue',
      guide: 'green',
      reference: 'purple',
      faq: 'orange',
      ethical_guidelines: 'red',
      legal: 'gray',
      best_practices: 'yellow',
      troubleshooting: 'pink'
    };
    return colors[type] || 'gray';
  }

  /**
   * Create contextual help request from current page
   */
  createHelpContext(pathname: string, feature?: string, error?: string): HelpContext {
    return {
      page_url: pathname,
      feature,
      error_code: error
    };
  }

  /**
   * Process markdown content for display
   */
  processMarkdownContent(content: string): string {
    // Simple markdown processing for display
    // In production, would use a proper markdown parser
    return content
      .replace(/^# (.+)$/gm, '<h1>$1</h1>')
      .replace(/^## (.+)$/gm, '<h2>$1</h2>')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^\s*-\s+(.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  }
}

// Export singleton instance
export const documentationService = new DocumentationService();
