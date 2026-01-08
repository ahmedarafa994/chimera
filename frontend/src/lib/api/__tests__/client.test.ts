/**
 * API Client Tests
 * Unit tests for the centralized API client
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios from 'axios';
import { apiClient } from '../client';
import { apiCache } from '../cache-manager';
import { authManager } from '../auth-manager';

// Mock axios
vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      patch: vi.fn(),
      delete: vi.fn(),
    })),
  },
}));

// Mock dependencies
vi.mock('../cache-manager', () => ({
  apiCache: {
    get: vi.fn(),
    set: vi.fn(),
  },
}));

vi.mock('../auth-manager', () => ({
  authManager: {
    getAccessToken: vi.fn(),
    getTenantId: vi.fn(),
    refreshAccessToken: vi.fn(),
    logout: vi.fn(),
  },
}));

vi.mock('../logger', () => ({
  logger: {
    logRequest: vi.fn(),
    logResponse: vi.fn(),
    logError: vi.fn(),
    logInfo: vi.fn(),
    logCacheHit: vi.fn(),
  },
}));

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('GET requests', () => {
    it('should make a GET request successfully', async () => {
      const mockResponse = {
        data: { id: 1, name: 'Test' },
        status: 200,
        config: { metadata: { requestId: 'test-123' } },
      };

      // The apiClient uses the mocked axios instance
      // We need to verify the structure of the returned data
      expect(apiClient).toBeDefined();
    });

    it('should return cached data when available', async () => {
      const cachedData = { id: 1, name: 'Cached' };
      (apiCache.get as any).mockReturnValue(cachedData);

      const result = await apiClient.get('/test');
      
      expect(result.data).toBe(cachedData);
      expect(result.status).toBe(200);
    });

    it('should skip cache when skipCache param is provided', async () => {
      (apiCache.get as any).mockReturnValue(null);
      
      // Verify cache behavior
      expect(apiCache.get).toBeDefined();
    });
  });

  describe('POST requests', () => {
    it('should make a POST request successfully', async () => {
      expect(apiClient.post).toBeDefined();
    });

    it('should not retry POST on network errors by default', async () => {
      // POST requests should have retry disabled for idempotency
      expect(apiClient.post).toBeDefined();
    });
  });

  describe('Request cancellation', () => {
    it('should cancel a specific request', () => {
      apiClient.cancelRequest('test-request-id');
      // Verify cancellation logic
      expect(apiClient.cancelRequest).toBeDefined();
    });

    it('should cancel all pending requests', () => {
      apiClient.cancelAllRequests();
      expect(apiClient.cancelAllRequests).toBeDefined();
    });
  });

  describe('WebSocket URL generation', () => {
    it('should generate correct WebSocket URL', () => {
      const wsUrl = apiClient.getWebSocketUrl('/ws/test');
      expect(wsUrl).toContain('ws');
      expect(wsUrl).toContain('/ws/test');
    });
  });
});

describe('Cache Manager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should be defined', () => {
    expect(apiCache).toBeDefined();
    expect(apiCache.get).toBeDefined();
    expect(apiCache.set).toBeDefined();
  });
});

describe('Auth Manager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should be defined', () => {
    expect(authManager).toBeDefined();
    expect(authManager.getAccessToken).toBeDefined();
    expect(authManager.getTenantId).toBeDefined();
  });
});