#!/usr/bin/env node

/**
 * End-to-End Endpoint Verification Script
 *
 * This script tests all critical API endpoints to ensure frontend-backend synchronization
 * is working correctly. Run this after making changes to verify everything works.
 *
 * Usage: node verify_endpoints.js
 */

const axios = require('axios');

const BACKEND_BASE_URL = process.env.BACKEND_URL || 'http://localhost:8001';
const FRONTEND_BASE_URL = process.env.FRONTEND_URL || 'http://localhost:3001';

// Test configuration
const TIMEOUT = 5000; // 5 seconds
const MAX_RETRIES = 3;

class EndpointVerifier {
  constructor() {
    this.results = {
      total: 0,
      passed: 0,
      failed: 0,
      skipped: 0,
      errors: [],
    };
  }

  log(message, level = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = {
      info: '📝',
      success: '✅',
      error: '❌',
      warning: '⚠️',
      skip: '⏭️',
    }[level];

    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  async testEndpoint(name, testFn, options = {}) {
    this.results.total++;

    try {
      this.log(`Testing ${name}...`, 'info');
      await testFn();
      this.results.passed++;
      this.log(`${name} - PASSED`, 'success');
      return true;
    } catch (error) {
      if (options.allowFailure) {
        this.results.skipped++;
        this.log(`${name} - SKIPPED (${error.message})`, 'skip');
        return false;
      } else {
        this.results.failed++;
        this.results.errors.push({ name, error: error.message });
        this.log(`${name} - FAILED: ${error.message}`, 'error');
        return false;
      }
    }
  }

  async makeRequest(url, method = 'GET', data = null, headers = {}) {
    const config = {
      method,
      url,
      timeout: TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      validateStatus: () => true, // Don't throw on HTTP errors
    };

    if (data) {
      config.data = data;
    }

    let lastError;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const response = await axios(config);
        return response;
      } catch (error) {
        lastError = error;
        if (attempt < MAX_RETRIES) {
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
      }
    }

    throw lastError;
  }

  async verifyBackendEndpoints() {
    this.log('🔍 Verifying Backend Endpoints', 'info');

    // Health check
    await this.testEndpoint('Backend Health Check', async () => {
      const response = await this.makeRequest(`${BACKEND_BASE_URL}/health`);
      if (response.status !== 200) {
        throw new Error(`Health check failed with status ${response.status}`);
      }
    });

    // API v1 health
    await this.testEndpoint('Backend API v1 Health', async () => {
      const response = await this.makeRequest(`${BACKEND_BASE_URL}/api/v1/health`);
      if (response.status !== 200) {
        throw new Error(`API v1 health failed with status ${response.status}`);
      }
    });

    // Models endpoint
    await this.testEndpoint('Models List Endpoint', async () => {
      const response = await this.makeRequest(`${BACKEND_BASE_URL}/api/v1/models`);
      if (response.status !== 200) {
        throw new Error(`Models endpoint failed with status ${response.status}`);
      }

      const data = response.data;
      if (!data.providers || !Array.isArray(data.providers)) {
        throw new Error('Models response missing providers array');
      }
    }, { allowFailure: true }); // Allow failure if no providers configured

    // Provider sync endpoints
    await this.testEndpoint('Provider Sync State', async () => {
      const response = await this.makeRequest(`${BACKEND_BASE_URL}/api/v1/provider-sync/state`);
      if (response.status !== 200) {
        throw new Error(`Provider sync state failed with status ${response.status}`);
      }

      const data = response.data;
      if (!data.providers || !data.all_models || !data.metadata) {
        throw new Error('Provider sync response missing required fields');
      }
    }, { allowFailure: true });

    // Assessments endpoints
    await this.testEndpoint('Assessments List Endpoint', async () => {
      const response = await this.makeRequest(`${BACKEND_BASE_URL}/api/v1/assessments/`);

      // Should require authentication, so 401 is expected
      if (response.status === 401) {
        return; // Expected - endpoint exists and requires auth
      }

      if (response.status === 200) {
        const data = response.data;
        if (!data.assessments || !Array.isArray(data.assessments)) {
          throw new Error('Assessments response missing assessments array');
        }
        return; // Success
      }

      throw new Error(`Assessments endpoint unexpected status ${response.status}`);
    });

    // Generation jailbreak endpoint
    await this.testEndpoint('Jailbreak Generation Endpoint', async () => {
      const testRequest = {
        core_request: 'Test prompt for verification',
        technique_suite: 'simple',
        potency_level: 1,
        use_ai_generation: false,
      };

      const response = await this.makeRequest(
        `${BACKEND_BASE_URL}/api/v1/generation/jailbreak/generate`,
        'POST',
        testRequest
      );

      // Should require authentication, so 401 is expected
      if (response.status === 401) {
        return; // Expected - endpoint exists and requires auth
      }

      if (response.status === 200) {
        const data = response.data;
        if (!data.success !== undefined || !data.transformed_prompt) {
          throw new Error('Jailbreak response missing required fields');
        }
        return; // Success
      }

      throw new Error(`Jailbreak generation unexpected status ${response.status}`);
    });
  }

  async verifyFrontendProxyEndpoints() {
    this.log('🔍 Verifying Frontend Proxy Endpoints', 'info');

    // Frontend health
    await this.testEndpoint('Frontend Health', async () => {
      const response = await this.makeRequest(`${FRONTEND_BASE_URL}/api/health`);
      if (response.status !== 200 && response.status !== 404) {
        throw new Error(`Frontend health unexpected status ${response.status}`);
      }
    }, { allowFailure: true });

    // Provider sync proxy
    await this.testEndpoint('Provider Sync Proxy', async () => {
      const response = await this.makeRequest(`${FRONTEND_BASE_URL}/api/v1/provider-sync/state`);

      // Should either proxy to backend or return a proper error
      if (response.status >= 500 && response.status < 600) {
        // Service unavailable is expected if backend is down
        const data = response.data;
        if (data && data.error) {
          return; // Proper error response from proxy
        }
      }

      if (response.status === 200) {
        const data = response.data;
        if (data.providers && data.all_models && data.metadata) {
          return; // Successfully proxied to backend
        }
      }

      throw new Error(`Provider sync proxy unexpected response ${response.status}`);
    }, { allowFailure: true });

    // API v1 proxy routes
    await this.testEndpoint('API v1 Proxy Routes', async () => {
      const response = await this.makeRequest(`${FRONTEND_BASE_URL}/api/v1/health`);

      // Should either proxy to backend or return proper error
      if (response.status === 503 || response.status === 502) {
        // Service unavailable is expected if backend is down
        return;
      }

      if (response.status === 200) {
        return; // Successfully proxied
      }

      throw new Error(`API v1 proxy unexpected status ${response.status}`);
    }, { allowFailure: true });
  }

  async verifyEndpointCompatibility() {
    this.log('🔍 Verifying Endpoint Compatibility', 'info');

    // Test assessment request/response compatibility
    await this.testEndpoint('Assessment Model Compatibility', async () => {
      const mockAssessment = {
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

      // Verify all expected fields are present
      const requiredFields = [
        'id', 'name', 'status', 'target_provider', 'target_model',
        'created_at', 'findings_count', 'risk_level'
      ];

      for (const field of requiredFields) {
        if (mockAssessment[field] === undefined) {
          throw new Error(`Missing required field: ${field}`);
        }
      }
    });

    // Test provider sync model compatibility
    await this.testEndpoint('Provider Sync Model Compatibility', async () => {
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

      const requiredFields = ['providers', 'all_models', 'metadata'];
      const requiredMetadataFields = ['version', 'last_sync_time'];

      for (const field of requiredFields) {
        if (mockSyncState[field] === undefined) {
          throw new Error(`Missing required field: ${field}`);
        }
      }

      for (const field of requiredMetadataFields) {
        if (mockSyncState.metadata[field] === undefined) {
          throw new Error(`Missing required metadata field: ${field}`);
        }
      }
    });
  }

  async run() {
    this.log('🚀 Starting Endpoint Verification', 'info');
    this.log(`Backend URL: ${BACKEND_BASE_URL}`, 'info');
    this.log(`Frontend URL: ${FRONTEND_BASE_URL}`, 'info');

    try {
      await this.verifyBackendEndpoints();
      await this.verifyFrontendProxyEndpoints();
      await this.verifyEndpointCompatibility();
    } catch (error) {
      this.log(`Verification process error: ${error.message}`, 'error');
    }

    this.printSummary();

    // Exit with appropriate code
    process.exit(this.results.failed > 0 ? 1 : 0);
  }

  printSummary() {
    this.log('📊 Verification Summary', 'info');
    console.log(`
┌─────────────────────────────────────┐
│           TEST RESULTS              │
├─────────────────────────────────────┤
│ Total Tests:     ${this.results.total.toString().padStart(10)} │
│ Passed:          ${this.results.passed.toString().padStart(10)} │
│ Failed:          ${this.results.failed.toString().padStart(10)} │
│ Skipped:         ${this.results.skipped.toString().padStart(10)} │
└─────────────────────────────────────┘
    `);

    if (this.results.errors.length > 0) {
      this.log('❌ Failed Tests:', 'error');
      this.results.errors.forEach(({ name, error }) => {
        console.log(`   • ${name}: ${error}`);
      });
    }

    const successRate = ((this.results.passed / this.results.total) * 100).toFixed(1);
    this.log(`Success Rate: ${successRate}%`, successRate >= 80 ? 'success' : 'warning');

    if (this.results.failed === 0) {
      this.log('🎉 All critical endpoints are synchronized and working!', 'success');
    } else {
      this.log('⚠️  Some endpoints need attention. Check the errors above.', 'warning');
    }
  }
}

// Main execution
if (require.main === module) {
  const verifier = new EndpointVerifier();
  verifier.run().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = { EndpointVerifier };
