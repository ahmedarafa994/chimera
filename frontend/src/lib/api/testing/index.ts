/**
 * API Testing Module
 * 
 * Export point for all testing utilities.
 * 
 * @module lib/api/testing
 */

// Mocks
export {
  MockServer,
  mockServer,
  createMockAdapter,
  fixtures,
  mockResponse,
  mockError,
  delayedResponse,
  waitFor,
  type MockResponse,
  type MockRequestHandler,
  type MockServerConfig,
} from './mocks';

// Test Utilities
export {
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
  testUtils,
  type TestContext,
  type RequestAssertion,
  type ResponseAssertion,
  type ContractTest,
} from './test-utils';