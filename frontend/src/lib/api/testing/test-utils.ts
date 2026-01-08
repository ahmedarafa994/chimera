/**
 * API Testing Utilities
 *
 * Provides utilities for testing API integrations including
 * test helpers, assertions, and integration test setup.
 *
 * @module lib/api/testing/test-utils
 */

import { APIClient, RequestOptions } from '../core/client';
import { MockServer, createMockAdapter, MockRequestHandler, mockResponse, mockError } from './mocks';
import { APIError } from '../../errors';
import { configManager } from '../core/config';

// ============================================================================
// Types
// ============================================================================

export interface TestContext {
  client: APIClient;
  mockServer: MockServer;
  cleanup: () => void;
}

export interface RequestAssertion {
  method?: string;
  url?: string | RegExp;
  headers?: Record<string, string>;
  body?: unknown;
  called?: boolean;
  callCount?: number;
}

export interface ResponseAssertion<T = unknown> {
  status?: number;
  data?: T | ((data: T) => boolean);
  error?: string | RegExp;
}

// ============================================================================
// Test Setup
// ============================================================================

/**
 * Create a test context with mock server
 */
export function createTestContext(): TestContext {
  const mockServer = new MockServer();
  const client = new APIClient({
    enableDeduplication: false,
    enableCircuitBreaker: false,
    enableRetry: false,
    enableCaching: false,
    enableMonitoring: false,
  });

  // Override axios adapter
  const axiosInstance = client.getAxiosInstance();
  axiosInstance.defaults.adapter = createMockAdapter(mockServer);

  return {
    client,
    mockServer,
    cleanup: () => {
      mockServer.reset();
      client.clearCache();
    },
  };
}

/**
 * Setup test environment
 */
export function setupTestEnvironment(): void {
  // Set test environment - use Object.defineProperty to work around read-only
  if (typeof process !== 'undefined' && process.env) {
    Object.defineProperty(process.env, 'NODE_ENV', {
      value: 'test',
      writable: true,
      configurable: true,
    });
  }

  // Configure for testing
  configManager.updateConfig({
    logging: {
      enabled: false,
      level: 'error',
      includeTimings: false,
      includeHeaders: false,
    },
  });
}

/**
 * Teardown test environment
 */
export function teardownTestEnvironment(): void {
  // Reset configuration
  configManager.updateConfig({
    logging: {
      enabled: true,
      level: 'info',
      includeTimings: true,
      includeHeaders: false,
    },
  });
}

// ============================================================================
// Request Tracking
// ============================================================================

export class RequestTracker {
  private requests: Array<{
    method: string;
    url: string;
    headers: Record<string, string>;
    body: unknown;
    timestamp: number;
  }> = [];

  /**
   * Track a request
   */
  track(method: string, url: string, headers: Record<string, string>, body: unknown): void {
    this.requests.push({
      method,
      url,
      headers,
      body,
      timestamp: Date.now(),
    });
  }

  /**
   * Get all tracked requests
   */
  getRequests(): typeof this.requests {
    return [...this.requests];
  }

  /**
   * Get requests matching criteria
   */
  getMatchingRequests(criteria: {
    method?: string;
    url?: string | RegExp;
  }): typeof this.requests {
    return this.requests.filter(req => {
      if (criteria.method && req.method !== criteria.method) {
        return false;
      }
      if (criteria.url) {
        if (criteria.url instanceof RegExp) {
          if (!criteria.url.test(req.url)) {
            return false;
          }
        } else if (req.url !== criteria.url) {
          return false;
        }
      }
      return true;
    });
  }

  /**
   * Assert request was made
   */
  assertCalled(criteria: RequestAssertion): void {
    const matching = this.getMatchingRequests({
      method: criteria.method,
      url: criteria.url,
    });

    if (criteria.called === false) {
      if (matching.length > 0) {
        throw new Error(`Expected no requests matching criteria, but found ${matching.length}`);
      }
      return;
    }

    if (matching.length === 0) {
      throw new Error(`Expected request matching criteria, but none found`);
    }

    if (criteria.callCount !== undefined && matching.length !== criteria.callCount) {
      throw new Error(`Expected ${criteria.callCount} requests, but found ${matching.length}`);
    }

    if (criteria.headers) {
      const lastRequest = matching[matching.length - 1];
      for (const [key, value] of Object.entries(criteria.headers)) {
        if (lastRequest.headers[key] !== value) {
          throw new Error(`Expected header ${key}=${value}, but got ${lastRequest.headers[key]}`);
        }
      }
    }

    if (criteria.body !== undefined) {
      const lastRequest = matching[matching.length - 1];
      if (JSON.stringify(lastRequest.body) !== JSON.stringify(criteria.body)) {
        throw new Error(`Request body mismatch`);
      }
    }
  }

  /**
   * Clear tracked requests
   */
  clear(): void {
    this.requests = [];
  }
}

// ============================================================================
// Response Assertions
// ============================================================================

/**
 * Assert response matches expectations
 */
export function assertResponse<T>(
  response: T,
  assertion: ResponseAssertion<T>
): void {
  if (assertion.data !== undefined) {
    if (typeof assertion.data === 'function' && assertion.data instanceof Function) {
      const validator = assertion.data as (data: T) => boolean;
      if (!validator(response)) {
        throw new Error('Response data assertion failed');
      }
    } else {
      if (JSON.stringify(response) !== JSON.stringify(assertion.data)) {
        throw new Error('Response data mismatch');
      }
    }
  }
}

/**
 * Assert error matches expectations
 */
export function assertError(
  error: unknown,
  assertion: { code?: string; message?: string | RegExp; status?: number }
): void {
  if (!(error instanceof APIError)) {
    throw new Error(`Expected APIError, got ${typeof error}`);
  }

  if (assertion.code && (error as any).errorCode !== assertion.code) {
    throw new Error(`Expected error code ${assertion.code}, got ${(error as any).errorCode}`);
  }

  if (assertion.message) {
    if (assertion.message instanceof RegExp) {
      if (!assertion.message.test((error as any).message)) {
        throw new Error(`Error message does not match pattern`);
      }
    } else if ((error as any).message !== assertion.message) {
      throw new Error(`Expected error message "${assertion.message}", got "${(error as any).message}"`);
    }
  }

  if (assertion.status && (error as any).statusCode !== assertion.status) {
    throw new Error(`Expected status ${assertion.status}, got ${(error as any).statusCode}`);
  }
}

// ============================================================================
// Mock Helpers
// ============================================================================

/**
 * Create a mock handler that returns success
 */
export function mockSuccess<T>(
  method: string,
  url: string,
  data: T,
  options: { delay?: number; headers?: Record<string, string> } = {}
): MockRequestHandler {
  return {
    method,
    url,
    response: {
      ...mockResponse(data),
      delay: options.delay,
      headers: {
        'content-type': 'application/json',
        ...options.headers,
      },
    },
  };
}

/**
 * Create a mock handler that returns an error
 */
export function mockFailure(
  method: string,
  url: string,
  code: string,
  message: string,
  status: number = 400
): MockRequestHandler {
  return {
    method,
    url,
    response: mockError(code, message, status),
  };
}

/**
 * Create a mock handler that times out
 */
export function mockTimeout(
  method: string,
  url: string,
  delayMs: number = 30000
): MockRequestHandler {
  return {
    method,
    url,
    response: {
      data: null,
      status: 0,
      statusText: 'Timeout',
      headers: {},
      delay: delayMs,
    },
  };
}

/**
 * Create a mock handler that returns different responses
 */
export function mockSequence<T>(
  method: string,
  url: string,
  responses: Array<{ data?: T; error?: { code: string; message: string }; status?: number }>
): MockRequestHandler {
  let callIndex = 0;

  return {
    method,
    url,
    response: () => {
      const responseConfig = responses[Math.min(callIndex++, responses.length - 1)];

      if (responseConfig.error) {
        return mockError(
          responseConfig.error.code,
          responseConfig.error.message,
          responseConfig.status || 400
        );
      }

      return mockResponse(responseConfig.data, { status: responseConfig.status });
    },
  };
}

// ============================================================================
// Integration Test Helpers
// ============================================================================

/**
 * Test API endpoint availability
 */
export async function testEndpointAvailability(
  client: APIClient,
  endpoints: string[]
): Promise<Map<string, boolean>> {
  const results = new Map<string, boolean>();

  for (const endpoint of endpoints) {
    try {
      await client.get(endpoint, { skipRetry: true, skipCircuitBreaker: true });
      results.set(endpoint, true);
    } catch {
      results.set(endpoint, false);
    }
  }

  return results;
}

/**
 * Test retry behavior
 */
export async function testRetryBehavior(
  client: APIClient,
  mockServer: MockServer,
  endpoint: string,
  failCount: number
): Promise<{ succeeded: boolean; attempts: number }> {
  let attempts = 0;

  mockServer.register({
    method: 'GET',
    url: endpoint,
    response: () => {
      attempts++;
      if (attempts <= failCount) {
        return mockError('server_error', 'Temporary failure', 500);
      }
      return mockResponse({ success: true });
    },
  });

  try {
    await client.get(endpoint, {
      retryConfig: {
        maxRetries: failCount + 1,
        baseDelay: 10,
        maxDelay: 100,
        backoffMultiplier: 2,
        retryableStatusCodes: [500],
      },
    });
    return { succeeded: true, attempts };
  } catch {
    return { succeeded: false, attempts };
  }
}

/**
 * Test circuit breaker behavior
 */
export async function testCircuitBreakerBehavior(
  client: APIClient,
  mockServer: MockServer,
  endpoint: string,
  failureThreshold: number
): Promise<{ circuitOpened: boolean; failureCount: number }> {
  let failureCount = 0;

  mockServer.register({
    method: 'GET',
    url: endpoint,
    response: mockError('server_error', 'Service unavailable', 503),
  });

  // Trigger failures
  for (let i = 0; i < failureThreshold + 2; i++) {
    try {
      await client.get(endpoint, {
        skipRetry: true,
        circuitBreakerKey: 'test-circuit',
      });
    } catch (error) {
      failureCount++;
      if (error instanceof APIError && error.errorCode === 'CIRCUIT_BREAKER_OPEN') {
        return { circuitOpened: true, failureCount };
      }
    }
  }

  return { circuitOpened: false, failureCount };
}

/**
 * Test caching behavior
 */
export async function testCachingBehavior(
  client: APIClient,
  mockServer: MockServer,
  endpoint: string
): Promise<{ cacheHit: boolean; requestCount: number }> {
  let requestCount = 0;

  mockServer.register({
    method: 'GET',
    url: endpoint,
    response: () => {
      requestCount++;
      return mockResponse({ data: 'test', timestamp: Date.now() });
    },
  });

  // First request
  await client.get(endpoint, { cacheTTL: 60000 });

  // Second request (should be cached)
  await client.get(endpoint, { cacheTTL: 60000 });

  return {
    cacheHit: requestCount === 1,
    requestCount,
  };
}

// ============================================================================
// Performance Testing
// ============================================================================

/**
 * Measure request latency
 */
export async function measureLatency(
  client: APIClient,
  endpoint: string,
  iterations: number = 10
): Promise<{
  min: number;
  max: number;
  avg: number;
  p50: number;
  p95: number;
  p99: number;
}> {
  const latencies: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = Date.now();
    try {
      await client.get(endpoint, { skipCache: true });
    } catch {
      // Include failed requests in latency measurement
    }
    latencies.push(Date.now() - start);
  }

  latencies.sort((a, b) => a - b);

  const percentile = (p: number) => {
    const index = Math.ceil((p / 100) * latencies.length) - 1;
    return latencies[Math.max(0, index)];
  };

  return {
    min: Math.min(...latencies),
    max: Math.max(...latencies),
    avg: latencies.reduce((a, b) => a + b, 0) / latencies.length,
    p50: percentile(50),
    p95: percentile(95),
    p99: percentile(99),
  };
}

/**
 * Test concurrent request handling
 */
export async function testConcurrency(
  client: APIClient,
  endpoint: string,
  concurrency: number
): Promise<{
  successful: number;
  failed: number;
  totalTime: number;
}> {
  const start = Date.now();
  const results = await Promise.allSettled(
    Array(concurrency)
      .fill(null)
      .map(() => client.get(endpoint, { skipDeduplication: true }))
  );

  return {
    successful: results.filter(r => r.status === 'fulfilled').length,
    failed: results.filter(r => r.status === 'rejected').length,
    totalTime: Date.now() - start,
  };
}

// ============================================================================
// Contract Testing
// ============================================================================

export interface ContractTest {
  name: string;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  requestBody?: unknown;
  expectedStatus: number;
  responseSchema?: (response: unknown) => boolean;
}

/**
 * Run contract tests
 */
export async function runContractTests(
  client: APIClient,
  tests: ContractTest[]
): Promise<Array<{ name: string; passed: boolean; error?: string }>> {
  const results: Array<{ name: string; passed: boolean; error?: string }> = [];

  for (const test of tests) {
    try {
      let response: unknown;

      switch (test.method) {
        case 'GET':
          response = await client.get(test.endpoint);
          break;
        case 'POST':
          response = await client.post(test.endpoint, test.requestBody);
          break;
        case 'PUT':
          response = await client.put(test.endpoint, test.requestBody);
          break;
        case 'DELETE':
          response = await client.delete(test.endpoint);
          break;
      }

      if (test.responseSchema && !test.responseSchema(response)) {
        results.push({
          name: test.name,
          passed: false,
          error: 'Response schema validation failed',
        });
        continue;
      }

      results.push({ name: test.name, passed: true });
    } catch (error) {
      results.push({
        name: test.name,
        passed: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return results;
}

// ============================================================================
// Exports
// ============================================================================

export const testUtils = {
  createTestContext,
  setupTestEnvironment,
  teardownTestEnvironment,
  RequestTracker,
  assertResponse,
  assertError,
  mockSuccess,
  mockFailure,
  mockTimeout,
  mockSequence,
  testEndpointAvailability,
  testRetryBehavior,
  testCircuitBreakerBehavior,
  testCachingBehavior,
  measureLatency,
  testConcurrency,
  runContractTests,
};
