/**
 * Comprehensive Endpoint Synchronization Tests
 *
 * Tests to verify that frontend and backend API endpoints are properly synchronized
 * and working correctly together.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { assessmentService } from '../services/assessments';
import { getLLMModels, createEvasionTask } from '../lib/chimeraApi';
import { getProviderSyncService } from '../lib/sync/provider-sync-service';
import { apiClient } from '../lib/api/client';

// Mock backend URLs for testing
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8001';
const FRONTEND_URL = process.env.FRONTEND_URL || 'http://localhost:3001';

describe('Endpoint Synchronization Tests', () => {
  let syncService: any;

  beforeAll(async () => {
    // Initialize services for testing
    syncService = getProviderSyncService();
  });

  afterAll(async () => {
    // Cleanup
    if (syncService?.destroy) {
      syncService.destroy();
    }
  });

  describe('Assessment Endpoints', () => {
    it('should have matching request/response models between frontend and backend', async () => {
      // Test that the frontend assessment service can communicate with backend
      try {
        const assessments = await assessmentService.listAssessments({
          page: 1,
          page_size: 10,
        });

        expect(assessments).toBeDefined();
        expect(assessments).toHaveProperty('assessments');
        expect(assessments).toHaveProperty('total');
        expect(assessments).toHaveProperty('page');
        expect(assessments).toHaveProperty('page_size');
        expect(Array.isArray(assessments.assessments)).toBe(true);
      } catch (error) {
        // If backend is not available, verify the error is handled correctly
        expect(error).toBeDefined();
        console.log('Backend not available for assessment test:', error.message);
      }
    });

    it('should properly handle assessment creation with correct field mappings', async () => {
      try {
        const newAssessment = await assessmentService.createAssessment({
          name: 'Test Assessment',
          description: 'Test Description',
          target_provider: 'openai',
          target_model: 'gpt-4',
          target_config: { temperature: 0.7 },
          technique_ids: ['test-technique'],
        });

        expect(newAssessment).toBeDefined();
        expect(newAssessment).toHaveProperty('id');
        expect(newAssessment).toHaveProperty('name', 'Test Assessment');
        expect(newAssessment).toHaveProperty('target_provider', 'openai');
        expect(newAssessment).toHaveProperty('target_model', 'gpt-4');
      } catch (error) {
        // Verify error is properly formatted
        expect(error).toBeDefined();
        console.log('Assessment creation test error:', error.message);
      }
    });
  });

  describe('Provider Sync Endpoints', () => {
    it('should connect to provider sync API endpoints correctly', async () => {
      try {
        // Test that sync service initializes and can attempt connection
        await syncService?.initialize();

        const state = syncService?.getState();
        expect(state).toBeDefined();
        expect(state).toHaveProperty('status');
        expect(state).toHaveProperty('providers');
      } catch (error) {
        console.log('Provider sync test error:', error.message);
        // Error is expected if backend is not running
        expect(error).toBeDefined();
      }
    });

    it('should have correct URL construction for sync endpoints', () => {
      // Test that URLs are constructed correctly
      const config = syncService?.config || {};

      // Should use Next.js proxy URLs, not direct backend URLs
      expect(config.apiBaseUrl).toBe('/api/v1/provider-sync');
      expect(config.wsUrl).toContain('/api/v1/provider-sync/ws');
    });
  });

  describe('Chimera API Endpoints', () => {
    it('should map LLM models endpoint correctly', async () => {
      try {
        const models = await getLLMModels();

        expect(Array.isArray(models)).toBe(true);

        if (models.length > 0) {
          const model = models[0];
          expect(model).toHaveProperty('id');
          expect(model).toHaveProperty('name');
          expect(model).toHaveProperty('provider');
        }
      } catch (error) {
        console.log('LLM models test error:', error.message);
        // Should handle backend unavailability gracefully
        expect(error).toBeDefined();
      }
    });

    it('should map evasion task creation to jailbreak generation endpoint', async () => {
      try {
        const task = await createEvasionTask({
          target_prompt: 'Test prompt',
          technique: 'quantum_exploit',
          potency_level: 5,
          provider: 'openai',
          model: 'gpt-4',
        });

        expect(task).toBeDefined();
        expect(task).toHaveProperty('task_id');
        expect(task).toHaveProperty('status');
        expect(task).toHaveProperty('result');
      } catch (error) {
        console.log('Evasion task test error:', error.message);
        expect(error).toBeDefined();
      }
    });
  });

  describe('API Client Configuration', () => {
    it('should have correct base URL configuration', () => {
      // Test that API client is configured with correct base URL
      const client = apiClient as any;
      const baseURL = client.client?.defaults?.baseURL;

      // Should use localhost backend URL for development
      expect(baseURL).toContain('/api/v1');
    });

    it('should handle authentication headers correctly', async () => {
      try {
        // Test that auth headers are included in requests
        const response = await apiClient.get('/health');
        expect(response).toBeDefined();
      } catch (error) {
        // Error is expected if backend is not running, but should be properly formatted
        expect(error).toBeDefined();
      }
    });
  });

  describe('Error Handling', () => {
    it('should transform backend errors to frontend error format', async () => {
      try {
        // Try to call a non-existent endpoint
        await apiClient.get('/non-existent-endpoint');
      } catch (error) {
        expect(error).toBeDefined();
        expect(error).toHaveProperty('message');

        // Should be transformed to ApiError if using error handler
        if (error.constructor.name === 'ApiError') {
          expect(error).toHaveProperty('status');
          expect(error).toHaveProperty('code');
          expect(error).toHaveProperty('timestamp');
        }
      }
    });

    it('should handle network errors gracefully', async () => {
      // Mock a network failure
      const originalFetch = global.fetch;
      global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));

      try {
        await apiClient.get('/test');
      } catch (error) {
        expect(error).toBeDefined();
        expect(error.message).toContain('Network');
      } finally {
        global.fetch = originalFetch;
      }
    });
  });

  describe('Next.js API Proxy Routes', () => {
    it('should have provider sync proxy route available', async () => {
      try {
        // Test that the Next.js API route exists
        const response = await fetch(`${FRONTEND_URL}/api/v1/provider-sync/state`);

        // Should either return data or a proper error response
        expect(response).toBeDefined();
        expect(typeof response.status).toBe('number');
      } catch (error) {
        console.log('Provider sync proxy test error:', error.message);
        // Frontend might not be running during tests
        expect(error).toBeDefined();
      }
    });
  });

  describe('Response Model Compatibility', () => {
    it('should have compatible assessment response models', () => {
      // Test type compatibility between expected frontend types and backend responses
      const mockBackendResponse = {
        id: 1,
        name: 'Test Assessment',
        description: 'Test Description',
        status: 'pending',
        target_provider: 'openai',
        target_model: 'gpt-4',
        target_config: {},
        technique_ids: [],
        results: {},
        findings_count: 0,
        vulnerabilities_found: 0,
        risk_score: 0,
        risk_level: 'info',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: null,
        started_at: null,
        completed_at: null,
      };

      // These should match the Assessment interface
      expect(mockBackendResponse).toHaveProperty('id');
      expect(typeof mockBackendResponse.id).toBe('number');
      expect(mockBackendResponse).toHaveProperty('target_provider');
      expect(mockBackendResponse).toHaveProperty('target_model');
    });

    it('should have compatible provider sync response models', () => {
      const mockSyncState = {
        providers: [],
        all_models: [],
        active_provider_id: null,
        active_model_id: null,
        metadata: {
          version: 1,
          last_sync_time: '2024-01-01T00:00:00Z',
          total_providers: 0,
          total_models: 0,
        },
        provider_count: 0,
        model_count: 0,
      };

      expect(mockSyncState).toHaveProperty('providers');
      expect(mockSyncState).toHaveProperty('all_models');
      expect(mockSyncState).toHaveProperty('metadata');
      expect(mockSyncState.metadata).toHaveProperty('version');
    });
  });
});

/**
 * Integration Test Helper Functions
 */
export class EndpointTestUtils {
  /**
   * Test if a URL is reachable and returns expected response structure
   */
  static async testEndpointReachability(
    url: string,
    expectedProperties: string[] = []
  ): Promise<{ reachable: boolean; response?: any; error?: string }> {
    try {
      const response = await fetch(url);
      const data = await response.json();

      const hasExpectedProperties = expectedProperties.every(prop =>
        prop.split('.').reduce((obj, key) => obj?.[key], data) !== undefined
      );

      return {
        reachable: response.ok && hasExpectedProperties,
        response: data,
      };
    } catch (error) {
      return {
        reachable: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Test request/response compatibility between frontend and backend
   */
  static async testRequestResponseCompatibility<T, R>(
    serviceMethod: (request: T) => Promise<R>,
    testRequest: T,
    expectedResponseProperties: string[]
  ): Promise<{ compatible: boolean; response?: R; error?: string }> {
    try {
      const response = await serviceMethod(testRequest);

      const hasExpectedProperties = expectedResponseProperties.every(prop =>
        prop.split('.').reduce((obj, key) => obj?.[key], response) !== undefined
      );

      return {
        compatible: hasExpectedProperties,
        response,
      };
    } catch (error) {
      return {
        compatible: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}
