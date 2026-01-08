/**
 * Admin Service
 * 
 * Provides API methods for admin operations including data import,
 * statistics retrieval, and cache management.
 */

import { enhancedApi } from '@/lib/api-enhanced';
import type {
  AdminStats,
  ImportDataRequest,
  ImportDataResponse,
  CacheStats,
  SystemHealth,
} from '@/types/admin-types';

const API_BASE = '/api/v1/admin';

/**
 * Get admin API key from environment or storage
 */
function getAdminApiKey(): string {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('admin_api_key') || process.env.NEXT_PUBLIC_ADMIN_API_KEY || '';
  }
  return process.env.NEXT_PUBLIC_ADMIN_API_KEY || '';
}

/**
 * Create authorization headers for admin endpoints
 */
function getAuthHeaders(): Record<string, string> {
  const apiKey = getAdminApiKey();
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

/**
 * Admin Service API
 */
export const adminService = {
  /**
   * Import data from various sources
   */
  async importData(request: ImportDataRequest): Promise<ImportDataResponse> {
    return enhancedApi.post<ImportDataResponse>(
      `${API_BASE}/import-data`,
      request,
      { headers: getAuthHeaders() }
    );
  },

  /**
   * Get system statistics
   */
  async getStats(): Promise<AdminStats> {
    return enhancedApi.get<AdminStats>(
      `${API_BASE}/stats`,
      { headers: getAuthHeaders() }
    );
  },

  /**
   * Clear cache (all or specific type)
   */
  async clearCache(cacheType?: 'all' | 'models' | 'providers' | 'sessions'): Promise<{ success: boolean; message: string }> {
    return enhancedApi.post<{ success: boolean; message: string }>(
      `${API_BASE}/clear-cache`,
      { cache_type: cacheType || 'all' },
      { headers: getAuthHeaders() }
    );
  },

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<CacheStats> {
    return enhancedApi.get<CacheStats>(
      `${API_BASE}/cache-stats`,
      { headers: getAuthHeaders() }
    );
  },

  /**
   * Get system health status
   */
  async getSystemHealth(): Promise<SystemHealth> {
    return enhancedApi.get<SystemHealth>(
      `${API_BASE}/health`,
      { headers: getAuthHeaders() }
    );
  },

  /**
   * Get import history
   */
  async getImportHistory(limit: number = 50): Promise<ImportDataResponse[]> {
    return enhancedApi.get<ImportDataResponse[]>(
      `${API_BASE}/import-history`,
      { 
        headers: getAuthHeaders(),
        params: { limit }
      }
    );
  },

  /**
   * Validate admin API key
   */
  async validateApiKey(apiKey: string): Promise<{ valid: boolean; permissions: string[] }> {
    return enhancedApi.post<{ valid: boolean; permissions: string[] }>(
      `${API_BASE}/validate-key`,
      { api_key: apiKey }
    );
  },

  /**
   * Set admin API key in local storage
   */
  setApiKey(apiKey: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('admin_api_key', apiKey);
    }
  },

  /**
   * Clear admin API key from local storage
   */
  clearApiKey(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('admin_api_key');
    }
  },
};

export default adminService;