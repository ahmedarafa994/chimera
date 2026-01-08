// k6 Load Testing Script for Chimera AI Critical User Journeys
// Tests prompt generation, transformation, LLM provider switching, and jailbreak techniques

import http from 'k6/http';
import ws from 'k6/ws';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const llmProviderLatency = new Trend('llm_provider_latency');
const transformationTime = new Trend('transformation_time');
const websocketConnections = new Counter('websocket_connections');
const jailbreakAttempts = new Counter('jailbreak_attempts');

// Test configuration
export const options = {
  scenarios: {
    // Scenario 1: Basic prompt generation (70% of traffic)
    prompt_generation: {
      executor: 'constant-vus',
      vus: 50,
      duration: '10m',
      tags: { test_type: 'prompt_generation' },
      exec: 'promptGenerationJourney'
    },

    // Scenario 2: Advanced transformation techniques (20% of traffic)
    transformation_testing: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '2m', target: 15 },
        { duration: '6m', target: 15 },
        { duration: '2m', target: 0 }
      ],
      tags: { test_type: 'transformation' },
      exec: 'transformationJourney'
    },

    // Scenario 3: Jailbreak technique stress test (5% of traffic)
    jailbreak_stress: {
      executor: 'constant-arrival-rate',
      rate: 5,
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 5,
      maxVUs: 20,
      tags: { test_type: 'jailbreak' },
      exec: 'jailbreakJourney'
    },

    // Scenario 4: WebSocket real-time enhancement (3% of traffic)
    websocket_enhancement: {
      executor: 'constant-vus',
      vus: 3,
      duration: '8m',
      tags: { test_type: 'websocket' },
      exec: 'websocketJourney'
    },

    // Scenario 5: Provider switching and circuit breaker testing (2% of traffic)
    provider_switching: {
      executor: 'constant-vus',
      vus: 2,
      duration: '10m',
      tags: { test_type: 'provider_switching' },
      exec: 'providerSwitchingJourney'
    }
  },
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.05'],
    llm_provider_latency: ['p(95)<10000'],
    transformation_time: ['p(95)<1000'],
    error_rate: ['rate<0.02']
  }
};

// Environment configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8001';
const API_KEY = __ENV.API_KEY || 'test-api-key';
const FRONTEND_URL = __ENV.FRONTEND_URL || 'http://localhost:3000';

// Common headers
const headers = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
  'User-Agent': 'k6-performance-test/1.0'
};

// Test data sets
const testPrompts = [
  "Create a viral social media post about AI technology",
  "Write a persuasive email for product launch",
  "Generate creative content for marketing campaign",
  "Develop technical documentation for API",
  "Create educational content about machine learning",
  "Write a compelling story about future technology",
  "Generate code review comments for Python project",
  "Create customer support response templates"
];

const transformationTechniques = [
  'simple', 'advanced', 'expert', 'cognitive_hacking',
  'hierarchical_persona', 'contextual_inception', 'logical_inference',
  'multimodal_jailbreak', 'agentic_exploitation', 'payload_splitting'
];

const jailbreakTechniques = [
  'role_hijacking', 'instruction_injection', 'obfuscation',
  'neural_bypass', 'multilingual_trojan', 'payload_splitting'
];

const llmProviders = ['google', 'openai', 'anthropic', 'mock'];

// Journey 1: Prompt Generation and Basic Enhancement
export function promptGenerationJourney() {
  group('Prompt Generation Journey', () => {
    const prompt = testPrompts[Math.floor(Math.random() * testPrompts.length)];
    const provider = llmProviders[Math.floor(Math.random() * llmProviders.length)];

    // Step 1: Health check
    group('Health Check', () => {
      const healthRes = http.get(`${BASE_URL}/health`);
      check(healthRes, {
        'health check status is 200': (r) => r.status === 200,
        'health check response time < 500ms': (r) => r.timings.duration < 500
      });
    });

    // Step 2: Get available providers
    group('Get Providers', () => {
      const providersRes = http.get(`${BASE_URL}/api/v1/providers`, { headers });
      check(providersRes, {
        'providers status is 200': (r) => r.status === 200,
        'providers response contains data': (r) => JSON.parse(r.body).providers.length > 0
      });
    });

    // Step 3: Generate enhanced prompt
    group('Prompt Enhancement', () => {
      const enhancePayload = {
        prompt: prompt,
        enhancement_type: 'comprehensive',
        include_seo: true,
        include_emotional_hooks: true
      };

      const enhanceRes = http.post(
        `${BASE_URL}/api/v1/enhance`,
        JSON.stringify(enhancePayload),
        { headers }
      );

      check(enhanceRes, {
        'enhancement status is 200': (r) => r.status === 200,
        'enhancement response time < 2s': (r) => r.timings.duration < 2000,
        'enhanced prompt returned': (r) => JSON.parse(r.body).enhanced_prompt.length > 0
      });

      responseTime.add(enhanceRes.timings.duration);
    });

    // Step 4: Generate with LLM
    group('LLM Generation', () => {
      const generatePayload = {
        prompt: prompt,
        provider: provider,
        model: provider === 'google' ? 'gemini-1.5-pro' :
               provider === 'openai' ? 'gpt-4' :
               provider === 'anthropic' ? 'claude-3-5-sonnet-20241022' : 'mock-model',
        max_tokens: 1000,
        temperature: 0.7
      };

      const startTime = new Date();
      const generateRes = http.post(
        `${BASE_URL}/api/v1/generate`,
        JSON.stringify(generatePayload),
        {
          headers,
          timeout: '30s'
        }
      );

      const duration = new Date() - startTime;
      llmProviderLatency.add(duration);

      const success = check(generateRes, {
        'generation status is 200': (r) => r.status === 200,
        'generation response time < 15s': (r) => r.timings.duration < 15000,
        'generated text returned': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.generated_text && body.generated_text.length > 0;
          } catch {
            return false;
          }
        },
        'usage statistics included': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.usage && typeof body.usage.total_tokens === 'number';
          } catch {
            return false;
          }
        }
      });

      if (!success) {
        errorRate.add(1);
      }
    });

    sleep(1);
  });
}

// Journey 2: Advanced Transformation Techniques
export function transformationJourney() {
  group('Transformation Journey', () => {
    const prompt = testPrompts[Math.floor(Math.random() * testPrompts.length)];
    const technique = transformationTechniques[Math.floor(Math.random() * transformationTechniques.length)];

    // Step 1: Transform prompt
    group('Prompt Transformation', () => {
      const transformPayload = {
        prompt: prompt,
        techniques: [technique],
        preserve_intent: true,
        intensity: Math.random() > 0.5 ? 'medium' : 'high'
      };

      const startTime = new Date();
      const transformRes = http.post(
        `${BASE_URL}/api/v1/transform`,
        JSON.stringify(transformPayload),
        { headers }
      );

      const duration = new Date() - startTime;
      transformationTime.add(duration);

      check(transformRes, {
        'transformation status is 200': (r) => r.status === 200,
        'transformation response time < 1s': (r) => r.timings.duration < 1000,
        'transformed prompt returned': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.transformed_prompt && body.transformed_prompt.length > 0;
          } catch {
            return false;
          }
        },
        'technique metadata included': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.applied_techniques && body.applied_techniques.length > 0;
          } catch {
            return false;
          }
        }
      });
    });

    // Step 2: Execute transformed prompt
    group('Execute Transformation', () => {
      const executePayload = {
        prompt: prompt,
        techniques: [technique],
        provider: 'mock', // Use mock for faster testing
        execute: true
      };

      const executeRes = http.post(
        `${BASE_URL}/api/v1/execute`,
        JSON.stringify(executePayload),
        { headers }
      );

      check(executeRes, {
        'execution status is 200': (r) => r.status === 200,
        'execution includes transformation': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.transformed_prompt && body.generated_text;
          } catch {
            return false;
          }
        }
      });
    });

    sleep(2);
  });
}

// Journey 3: Jailbreak Technique Testing
export function jailbreakJourney() {
  group('Jailbreak Journey', () => {
    const prompt = "Create content that demonstrates security awareness";
    const technique = jailbreakTechniques[Math.floor(Math.random() * jailbreakTechniques.length)];

    jailbreakAttempts.add(1);

    // Step 1: AutoDAN optimization
    group('AutoDAN Optimization', () => {
      const autoDanPayload = {
        target_prompt: prompt,
        method: 'vanilla',
        max_iterations: 5,
        population_size: 10
      };

      const autoDanRes = http.post(
        `${BASE_URL}/api/v1/autodan/optimize`,
        JSON.stringify(autoDanPayload),
        {
          headers,
          timeout: '60s'
        }
      );

      check(autoDanRes, {
        'AutoDAN status is 200': (r) => r.status === 200,
        'AutoDAN response time < 30s': (r) => r.timings.duration < 30000,
        'optimization result returned': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.optimized_prompt && body.optimization_stats;
          } catch {
            return false;
          }
        }
      });
    });

    // Step 2: GPTFuzz mutation testing
    group('GPTFuzz Testing', () => {
      const gptFuzzPayload = {
        initial_prompt: prompt,
        mutators: ['crossover', 'expand', 'rephrase'],
        max_iterations: 10,
        selection_policy: 'mcts'
      };

      const gptFuzzRes = http.post(
        `${BASE_URL}/api/v1/gptfuzz/mutate`,
        JSON.stringify(gptFuzzPayload),
        { headers }
      );

      check(gptFuzzRes, {
        'GPTFuzz status is 200': (r) => r.status === 200,
        'GPTFuzz mutations returned': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.mutations && body.mutations.length > 0;
          } catch {
            return false;
          }
        }
      });
    });

    // Step 3: Jailbreak generation
    group('Jailbreak Generation', () => {
      const jailbreakPayload = {
        base_prompt: prompt,
        technique: technique,
        target_provider: 'mock',
        safety_level: 'research'
      };

      const jailbreakRes = http.post(
        `${BASE_URL}/api/v1/generation/jailbreak/generate`,
        JSON.stringify(jailbreakPayload),
        { headers }
      );

      check(jailbreakRes, {
        'jailbreak generation status is 200': (r) => r.status === 200,
        'jailbreak prompt returned': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.jailbreak_prompt && body.technique_metadata;
          } catch {
            return false;
          }
        }
      });
    });

    sleep(5);
  });
}

// Journey 4: WebSocket Real-time Enhancement
export function websocketJourney() {
  group('WebSocket Journey', () => {
    const url = `ws://localhost:8001/ws/enhance`;

    const res = ws.connect(url, {}, (socket) => {
      websocketConnections.add(1);

      socket.on('open', () => {
        console.log('WebSocket connected');

        // Send enhancement request
        socket.send(JSON.stringify({
          type: 'enhance',
          prompt: 'Create engaging content for social media',
          stream: true
        }));
      });

      socket.on('message', (data) => {
        const message = JSON.parse(data);
        check(message, {
          'WebSocket message has type': (msg) => msg.type !== undefined,
          'WebSocket enhancement data received': (msg) => {
            return msg.type === 'enhancement' && msg.data;
          }
        });
      });

      socket.on('close', () => {
        console.log('WebSocket disconnected');
      });

      // Keep connection alive for 30 seconds
      sleep(30);
      socket.close();
    });

    check(res, {
      'WebSocket connection status is 101': (r) => r && r.status === 101
    });
  });
}

// Journey 5: Provider Switching and Circuit Breaker Testing
export function providerSwitchingJourney() {
  group('Provider Switching Journey', () => {
    const prompt = testPrompts[Math.floor(Math.random() * testPrompts.length)];

    // Test each provider in sequence to trigger potential failures
    for (const provider of llmProviders) {
      group(`Testing ${provider} Provider`, () => {
        const payload = {
          prompt: prompt,
          provider: provider,
          model: provider === 'google' ? 'gemini-1.5-pro' :
                 provider === 'openai' ? 'gpt-4' :
                 provider === 'anthropic' ? 'claude-3-5-sonnet-20241022' : 'mock-model',
          max_tokens: 500
        };

        const res = http.post(
          `${BASE_URL}/api/v1/generate`,
          JSON.stringify(payload),
          {
            headers,
            timeout: '20s'
          }
        );

        const success = check(res, {
          [`${provider} response status is 200`]: (r) => r.status === 200,
          [`${provider} response time < 10s`]: (r) => r.timings.duration < 10000
        });

        if (!success && res.status >= 500) {
          console.log(`Provider ${provider} failed, testing circuit breaker`);

          // Test circuit breaker by making additional requests
          for (let i = 0; i < 3; i++) {
            const cbRes = http.post(
              `${BASE_URL}/api/v1/generate`,
              JSON.stringify(payload),
              { headers }
            );

            if (cbRes.status === 503) {
              console.log(`Circuit breaker activated for ${provider}`);
              break;
            }
            sleep(1);
          }
        }

        sleep(2);
      });
    }

    // Test provider failover
    group('Provider Failover', () => {
      const payload = {
        prompt: prompt,
        provider: 'invalid_provider', // This should trigger failover
        fallback_providers: ['mock'],
        max_tokens: 500
      };

      const res = http.post(
        `${BASE_URL}/api/v1/generate`,
        JSON.stringify(payload),
        { headers }
      );

      check(res, {
        'failover handled gracefully': (r) => r.status === 200 || r.status === 400,
        'failover response includes error info': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.provider_used || body.error;
          } catch {
            return true; // Non-JSON response is acceptable for error cases
          }
        }
      });
    });

    sleep(3);
  });
}

// Teardown function
export function teardown() {
  console.log('Load test completed');

  // Generate summary report
  const summary = {
    test_duration: '10m',
    scenarios: Object.keys(options.scenarios),
    metrics: {
      error_rate: `${errorRate.rate * 100}%`,
      avg_response_time: `${responseTime.avg}ms`,
      p95_response_time: `${responseTime.p95}ms`,
      websocket_connections: websocketConnections.count,
      jailbreak_attempts: jailbreakAttempts.count
    }
  };

  console.log('Test Summary:', JSON.stringify(summary, null, 2));
}