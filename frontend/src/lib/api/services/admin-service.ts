/**
 * Admin Service
 *
 * Provides API methods for admin operations including data import,
 * statistics retrieval, and cache management.
 */

import { apiClient } from '../client';
import type {
  AdminStats,
  ImportDataRequest,
  ImportDataResponse,
  CacheStats,
  SystemHealth,
} from '@/types/admin-types';

const API_BASE = '/admin';

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
  return apiKey ? { 'X-Admin-API-Key': apiKey } : {};
}

/**
 * Admin Service API
 */
export const adminService = {
  /**
   * Import data from various sources
   */
  async importData(request: ImportDataRequest): Promise<ImportDataResponse> {
    const response = await apiClient.post<ImportDataResponse>(
      `${API_BASE}/import-data`,
      request,
      { headers: getAuthHeaders() }
    );
    return response.data;
  },

  /**
   * Get system statistics
   */
  async getStats(): Promise<AdminStats> {
    const response = await apiClient.get<AdminStats>(
      `${API_BASE}/stats`,
      { headers: getAuthHeaders() }
    );
    return response.data;
  },

  /**
   * Clear cache (all or specific type)
   */
  async clearCache(cacheType?: 'all' | 'models' | 'providers' | 'sessions'): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post<{ success: boolean; message: string }>(
      `${API_BASE}/clear-cache`,
      { cache_type: cacheType || 'all' },
      { headers: getAuthHeaders() }
    );
    return response.data;
  },

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<CacheStats> {
    const response = await apiClient.get<CacheStats>(
      `${API_BASE}/cache-stats`,
      { headers: getAuthHeaders() }
    );
    return response.data;
  },

  /**
   * Get system health status
   */
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await apiClient.get<SystemHealth>(
      `${API_BASE}/health`,
      { headers: getAuthHeaders() }
    );
    return response.data;
  },

  /**
   * Get import history
   */
  async getImportHistory(limit: number = 50): Promise<ImportDataResponse[]> {
    const response = await apiClient.get<ImportDataResponse[]>(
      `${API_BASE}/import-history`,
      {
        headers: getAuthHeaders(),
        params: { limit }
      }
    );
    return response.data;
  },

  /**
   * Validate admin API key
   */
  async validateApiKey(apiKey: string): Promise<{ valid: boolean; permissions: string[] }> {
    const response = await apiClient.post<{ valid: boolean; permissions: string[] }>(
      `${API_BASE}/validate-key`,
      { api_key: apiKey }
    );
    return response.data;
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
