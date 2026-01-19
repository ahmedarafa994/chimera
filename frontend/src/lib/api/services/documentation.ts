/**
 * Interactive Help & Documentation Hub Service
 */

import { apiClient } from '../client';

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
  tags: string[];
  difficulty_level: 'beginner' | 'intermediate' | 'advanced';
  estimated_read_time: number;
  parent_id?: string;
  order: number;
  sections: DocumentationSection[];
  related_articles: string[];
  video_url?: string;
  images: string[];
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
  duration: number;
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

export interface HelpContext {
  page_url: string;
  active_feature?: string;
  user_role?: string;
  last_actions?: string[];
}

export interface ContextualHelpResponse {
  recommended_guides: DocumentationItem[];
  quick_tips: string[];
  suggested_techniques: string[];
}

export interface DocumentationListResponse {
  items: DocumentationItem[];
  categories: Array<{
    id: DocumentationCategory;
    name: string;
    item_count: number;
  }>;
  total: number;
  page?: number;
  page_size?: number;
  has_next?: boolean;
  has_prev?: boolean;
}

export interface DocumentationListParams {
  category?: DocumentationCategory;
  type?: DocumentationType;
  difficulty?: string;
  page?: number;
  page_size?: number;
}

export interface QuickStartGuide {
  welcome_message: string;
  steps: Array<{
    title: string;
    content: string;
    action_url?: string;
  }>;
  next_steps: Array<{
    title: string;
    priority: 'high' | 'medium' | 'low';
    description: string;
    estimated_time: string;
    target: string;
    action: 'navigate' | 'learn';
  }>;
  featured_content: Array<{
    id: string;
    title: string;
    type: DocumentationType;
    description: string;
    content: string;
    link: string;
    estimated_read_time: number;
  }>;
}

const API_BASE = '/docs';

const mapItem = (data: any): DocumentationItem => {
  const mappedType =
    data.type === 'functional_guidelines' ? 'ethical_guidelines' : (data.type as DocumentationType);
  return {
    id: data.id,
    title: data.title,
    content: data.content,
    type: mappedType,
    category: data.category,
    tags: data.tags ?? [],
    difficulty_level: data.difficulty_level ?? 'beginner',
    estimated_read_time: data.estimated_read_time ?? 0,
    parent_id: data.parent_id,
    order: data.order ?? 0,
    sections: data.sections ?? [],
    related_articles: data.related_articles ?? [],
    video_url: data.video_url,
    images: data.images ?? [],
    views: data.views ?? 0,
    last_updated: data.last_updated ?? new Date().toISOString(),
    author: data.author ?? 'Chimera Team'
  };
};

export const documentationService = {
  async listDocumentation(params?: DocumentationListParams): Promise<DocumentationListResponse> {
    const response = await apiClient.get(`${API_BASE}`, { params });
    const data: any = response.data;
    return {
      items: (data.items ?? []).map(mapItem),
      categories: data.categories ?? [],
      total: data.total ?? data.items?.length ?? 0,
      page: data.page,
      page_size: data.page_size,
      has_next: data.has_next,
      has_prev: data.has_prev
    };
  },

  async getDocumentationItem(itemId: string): Promise<DocumentationItem> {
    const response = await apiClient.get(`${API_BASE}/${itemId}`);
    return mapItem(response.data);
  },

  async searchDocumentation(searchRequest: SearchRequest): Promise<SearchResult[]> {
    const response = await apiClient.post(`${API_BASE}/search`, searchRequest);
    const results: any[] = response.data ?? [];
    return results.map((result) => ({
      item: mapItem(result.item ?? result),
      relevance_score: result.relevance_score ?? result.score ?? 0,
      matching_sections: result.matching_sections ?? [],
      highlight_text: result.highlight_text ?? ''
    }));
  },

  async getDocumentationByCategory(category: DocumentationCategory): Promise<DocumentationItem[]> {
    const response = await apiClient.get(`${API_BASE}/categories/${category}`);
    return (response.data ?? []).map(mapItem);
  },

  async getContextualHelp(context: HelpContext): Promise<ContextualHelpResponse> {
    const response = await apiClient.post(`${API_BASE}/context-help`, context);
    return response.data;
  },

  async listVideoTutorials(category?: DocumentationCategory): Promise<VideoTutorial[]> {
    const response = await apiClient.get(`${API_BASE}/videos/`, { params: { category } });
    return response.data;
  },

  async getQuickStartGuide(): Promise<QuickStartGuide> {
    const response = await apiClient.get(`${API_BASE}/quick-start`);
    return response.data;
  },

  getTypeDisplayName(type: DocumentationType): string {
    const names: Record<DocumentationType, string> = {
      tutorial: 'Tutorial',
      guide: 'Guide',
      reference: 'Reference',
      faq: 'FAQ',
      ethical_guidelines: 'Ethical Guidelines',
      legal: 'Legal',
      best_practices: 'Best Practices',
      troubleshooting: 'Troubleshooting'
    };
    return names[type] || type;
  },

  getTypeBadgeColor(type: DocumentationType): string {
    const colors: Record<DocumentationType, string> = {
      tutorial: 'blue',
      guide: 'green',
      reference: 'purple',
      faq: 'orange',
      ethical_guidelines: 'red',
      legal: 'gray',
      best_practices: 'cyan',
      troubleshooting: 'yellow'
    };
    return colors[type] || 'gray';
  },

  formatReadingTime(minutes: number): string {
    if (minutes < 1) return '< 1 min';
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = Math.round(minutes % 60);
    if (remainingMinutes === 0) return `${hours}h`;
    return `${hours}h ${remainingMinutes}m`;
  },

  getCategoryIcon(category: DocumentationCategory): string {
    const icons: Record<DocumentationCategory, string> = {
      getting_started: '🚀',
      attack_techniques: '⚔️',
      providers: '🔗',
      reporting: '📊',
      collaboration: '👥',
      advanced: '🔬',
      ethics_legal: '⚖️',
      api: '🔧'
    };
    return icons[category] || '📄';
  },

  getDifficultyColor(difficulty: 'beginner' | 'intermediate' | 'advanced'): string {
    const colors: Record<'beginner' | 'intermediate' | 'advanced', string> = {
      beginner: 'green',
      intermediate: 'yellow',
      advanced: 'red'
    };
    return colors[difficulty] || 'gray';
  },

  formatVideoDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes < 60) {
      return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}:${remainingMinutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  },

  processMarkdownContent(markdown: string): string {
    // Basic markdown processing for HTML rendering
    // This is a simple implementation - for production, consider using a proper markdown parser
    const html = markdown
      // Convert headers
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      // Convert bold and italic
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Convert code blocks
      .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      // Convert links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      // Convert line breaks
      .replace(/\n/g, '<br />');

    return html;
  }
};
