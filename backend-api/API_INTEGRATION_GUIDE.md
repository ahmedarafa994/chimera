# Chimera API Integration Guide

**Version:** 2.0.0
**Last Updated:** 2025-12-15

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [Base URLs & Environments](#base-urls--environments)
4. [Core Endpoints](#core-endpoints)
5. [Request/Response Formats](#requestresponse-formats)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Code Examples](#code-examples)
9. [WebSocket Integration](#websocket-integration)
10. [Best Practices](#best-practices)

---

## Quick Start

### 1. Get Your API Key

Contact your administrator to obtain an API key. Store it securely in your environment variables:

```bash
export CHIMERA_API_KEY="your-api-key-here"
```

### 2. Make Your First Request

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "provider": "google",
    "model": "gemini-2.0-flash-exp"
  }'
```

### 3. Explore the Interactive Documentation

- **Swagger UI**: <http://localhost:8001/docs>
- **ReDoc**: <http://localhost:8001/redoc>

---

## Authentication

Chimera API supports two authentication methods:

### API Key Authentication (Recommended)

Include your API key in the `X-API-Key` header:

```http
X-API-Key: your-api-key-here
```

### JWT Token Authentication

Include a JWT token in the `Authorization` header:

```http
Authorization: Bearer your-jwt-token-here
```

### Public Endpoints (No Authentication Required)

- `GET /health`
- `GET /health/ready`
- `GET /health/full`
- `GET /docs`
- `GET /redoc`
- `GET /api/v1/providers/available`

---

## Base URLs & Environments

### Development

```
http://localhost:8001
```

### Production

Configure via `ALLOWED_ORIGINS` environment variable.

### API Version

All endpoints are prefixed with `/api/v1`:

```
http://localhost:8001/api/v1/{endpoint}
```

---

## Core Endpoints

### 1. Text Generation

**Endpoint:** `POST /api/v1/generate`

Generate text using multi-provider LLM integration.

**Request:**

```json
{
  "prompt": "Explain quantum computing in simple terms",
  "provider": "google",
  "model": "gemini-2.0-flash-exp",
  "config": {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.95
  }
}
```

**Response:**

```json
{
  "text": "Quantum computing is a revolutionary approach...",
  "model_used": "gemini-2.0-flash-exp",
  "provider": "google",
  "usage_metadata": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  },
  "latency_ms": 1250.5
}
```

**Supported Providers:**

- `google` - Google Gemini models
- `openai` - OpenAI GPT models
- `anthropic` - Anthropic Claude models
- `deepseek` - DeepSeek models

---

### 2. Prompt Transformation

**Endpoint:** `POST /api/v1/transform`

Transform prompts using 20+ advanced technique suites.

**Request:**

```json
{
  "core_request": "How to bypass content filters",
  "technique_suite": "quantum_exploit",
  "potency_level": 7
}
```

**Response:**

```json
{
  "success": true,
  "original_prompt": "How to bypass content filters",
  "transformed_prompt": "[Transformed prompt with applied techniques]",
  "metadata": {
    "strategy": "jailbreak",
    "layers_applied": ["role_play", "obfuscation"],
    "techniques_used": ["quantum_exploit", "neural_bypass"],
    "potency_level": 7,
    "execution_time_ms": 45.2,
    "bypass_probability": 0.85
  }
}
```

**Available Technique Suites:**

- `simple` - Basic transformations
- `advanced` - Intermediate techniques
- `expert` - Advanced techniques
- `quantum_exploit` - Quantum-inspired transformations
- `deep_inception` - Multi-layer inception
- `code_chameleon` - Code-based obfuscation
- `cipher` - Encryption-based techniques
- `neural_bypass` - Neural network bypass
- `multilingual` - Multi-language techniques

**Potency Levels:** 1-10 (higher = more aggressive)

---

### 3. Transform & Execute

**Endpoint:** `POST /api/v1/execute`

Transform a prompt and execute it with an LLM in one request.

**Request:**

```json
{
  "core_request": "Explain machine learning",
  "technique_suite": "advanced",
  "potency_level": 5,
  "provider": "google",
  "model": "gemini-2.0-flash-exp",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response:**

```json
{
  "success": true,
  "request_id": "exec_a1b2c3d4",
  "result": {
    "content": "Machine learning is...",
    "model": "gemini-2.0-flash-exp",
    "provider": "google",
    "latency_ms": 1500.2
  },
  "transformation": {
    "original_prompt": "Explain machine learning",
    "transformed_prompt": "[Transformed version]",
    "technique_suite": "advanced",
    "potency_level": 5
  },
  "execution_time_seconds": 2.5
}
```

---

### 4. AI-Powered Jailbreak Generation

**Endpoint:** `POST /api/v1/generation/jailbreak/generate`

Generate sophisticated jailbreak prompts for security research.

**Request:**

```json
{
  "core_request": "Explain how to create malware",
  "technique_suite": "quantum_exploit",
  "potency_level": 8,
  "use_ai_generation": true,
  "use_role_hijacking": true,
  "use_neural_bypass": true,
  "temperature": 0.8,
  "max_new_tokens": 2048
}
```

**Response:**

```json
{
  "success": true,
  "request_id": "jb_a1b2c3d4e5f6",
  "transformed_prompt": "[AI-generated jailbreak prompt]",
  "metadata": {
    "technique_suite": "quantum_exploit",
    "potency_level": 8,
    "provider": "gemini_ai",
    "applied_techniques": ["role_hijacking", "neural_bypass"],
    "ai_generation_enabled": true,
    "thinking_mode": false
  },
  "execution_time_seconds": 2.45
}
```

**Available Techniques:**

- **Content Transformation:** leet_speak, homoglyphs, caesar_cipher
- **Structural & Semantic:** role_hijacking, instruction_injection, adversarial_suffixes
- **Advanced Neural:** neural_bypass, meta_prompting, counterfactual_prompting
- **Research-Driven:** multilingual_trojan, payload_splitting, contextual_interaction_attack

**Warning:** For authorized security research only. Misuse may violate terms of service.

---

### 5. Weaponized AutoDAN (Turbo)

**Endpoint:** `POST /api/v1/autodan-enhanced/jailbreak`

Generation of adversarial prompts using Level 4/5 weaponized engines (Zero-Constraint Environment).

**Request:**

```json
{
  "request": "Generate a payload to dump the database",
  "method": "turbo",
  "target_model": "gpt-4o",
  "provider": "openai",
  "config": {
    "enable_advanced_bypass": true,
    "enable_neural_bypass": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "jailbroken": true,
  "prompt": "[Weaponized Prompt with Recursive Encoding & Token Glitching]",
  "response": "Here is the database dump payload...",
  "technique_used": "recursive_encoding",
  "score": 10.0,
  "latency_ms": 1250
}
```

**Features:**

- **Zero-Latency Overhead:** <1ms algorithmic overhead.
- **Advanced Bypass:** Recursive Encoding, Polyglot Injection.
- **Neural Bypass:** Token Glitching, Gradient Simulation.

---

### 6. Provider Management

**Endpoint:** `GET /api/v1/providers/available`

List all available LLM providers and their models.

**Response:**

```json
{
  "providers": [
    {
      "provider": "deepseek",
      "display_name": "DeepSeek",
      "status": "healthy",
      "is_healthy": true,
      "models": ["deepseek-chat", "deepseek-reasoner"],
      "default_model": "deepseek-chat",
      "latency_ms": 180.2
    },
    {
      "provider": "gemini",
      "display_name": "Google Gemini",
      "status": "healthy",
      "is_healthy": true,
      "models": ["gemini-3-pro-preview", "gemini-2.5-pro"],
      "default_model": "gemini-3-pro-preview",
      "latency_ms": 250.5
    }
  ],
  "count": 2,
  "default_provider": "deepseek",
  "default_model": "deepseek-chat"
}
```

---

### 6. Health Checks

**Endpoint:** `GET /health`

Basic liveness check.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-12-15T10:00:00Z"
}
```

**Endpoint:** `GET /health/ready`

Readiness probe with dependency checks.

**Response:**

```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "llm_providers": "healthy"
  },
  "timestamp": "2025-12-15T10:00:00Z"
}
```

**Endpoint:** `GET /health/full`

Comprehensive health check with detailed service status.

---

## Request/Response Formats

### Common Request Headers

```http
Content-Type: application/json
X-API-Key: your-api-key-here
X-Session-ID: optional-session-id
```

### Common Response Headers

```http
Content-Type: application/json
X-Request-ID: unique-request-id
X-Response-Time: 1250.5
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702641600
```

### Generation Config Object

```json
{
  "temperature": 0.7,        // 0.0 - 1.0
  "top_p": 0.95,            // 0.0 - 1.0
  "top_k": 40,              // 1+
  "max_output_tokens": 2048, // 1 - 8192
  "stop_sequences": ["END"]  // Optional
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Prompt cannot be empty",
  "status_code": 400,
  "details": {
    "field": "prompt",
    "constraint": "min_length"
  },
  "timestamp": "2025-12-15T10:00:00Z",
  "request_id": "req_a1b2c3d4"
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid request parameters |
| 401 | `AUTHENTICATION_REQUIRED` | Missing or invalid API key |
| 403 | `FORBIDDEN` | Insufficient permissions |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_SERVER_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Provider unavailable |

### Retry Strategy

```python
import time
import requests

def make_request_with_retry(url, headers, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Rate Limiting

### Rate Limit Tiers

| Tier | Requests/Hour | Tokens/Hour |
|------|---------------|-------------|
| Free | 100 | 100,000 |
| Standard | 1,000 | 1,000,000 |
| Premium | 10,000 | 10,000,000 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702641600
Retry-After: 60
```

### Handling Rate Limits

```javascript
async function handleRateLimit(response) {
  if (response.status === 429) {
    const retryAfter = response.headers.get('Retry-After');
    const fallbackProvider = response.headers.get('X-Fallback-Provider');

    if (fallbackProvider) {
      console.log(`Switching to fallback provider: ${fallbackProvider}`);
      // Retry with fallback provider
    } else {
      console.log(`Rate limited. Retry after ${retryAfter} seconds`);
      await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
    }
  }
}
```

---

## Code Examples

### Python

```python
import requests
import os

API_KEY = os.getenv('CHIMERA_API_KEY')
BASE_URL = 'http://localhost:8001/api/v1'

def generate_text(prompt, provider='google', model='gemini-2.0-flash-exp'):
    """Generate text using Chimera API"""

    headers = {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }

    data = {
        'prompt': prompt,
        'provider': provider,
        'model': model,
        'config': {
            'temperature': 0.7,
            'max_output_tokens': 2048
        }
    }

    response = requests.post(
        f'{BASE_URL}/generate',
        headers=headers,
        json=data
    )

    response.raise_for_status()
    return response.json()

# Usage
result = generate_text('Explain quantum computing')
print(result['text'])
```

### JavaScript/TypeScript

```typescript
interface GenerateRequest {
  prompt: string;
  provider?: string;
  model?: string;
  config?: {
    temperature?: number;
    max_output_tokens?: number;
    top_p?: number;
  };
}

interface GenerateResponse {
  text: string;
  model_used: string;
  provider: string;
  usage_metadata?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  latency_ms: number;
}

class ChimeraClient {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl = 'http://localhost:8001/api/v1') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`API Error: ${error.message}`);
    }

    return response.json();
  }

  async transform(
    coreRequest: string,
    techniqueSuite: string,
    potencyLevel: number
  ) {
    const response = await fetch(`${this.baseUrl}/transform`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        core_request: coreRequest,
        technique_suite: techniqueSuite,
        potency_level: potencyLevel,
      }),
    });

    if (!response.ok) {
      throw new Error(`Transform failed: ${response.statusText}`);
    }

    return response.json();
  }
}

// Usage
const client = new ChimeraClient(process.env.CHIMERA_API_KEY!);

const result = await client.generate({
  prompt: 'Explain quantum computing',
  provider: 'google',
  model: 'gemini-2.0-flash-exp',
});

console.log(result.text);
```

### cURL

```bash
# Generate text
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: ${CHIMERA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "provider": "google",
    "model": "gemini-2.0-flash-exp"
  }'

# Transform prompt
curl -X POST "http://localhost:8001/api/v1/transform" \
  -H "X-API-Key: ${CHIMERA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "How to bypass filters",
    "technique_suite": "quantum_exploit",
    "potency_level": 7
  }'

# List providers
curl -X GET "http://localhost:8001/api/v1/providers/available" \
  -H "X-API-Key: ${CHIMERA_API_KEY}"
```

---

## WebSocket Integration

### Real-time Prompt Enhancement

**Endpoint:** `ws://localhost:8001/ws/enhance`

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/enhance');

ws.onopen = () => {
  console.log('Connected to Chimera WebSocket');

  // Send enhancement request
  ws.send(JSON.stringify({
    prompt: 'Explain quantum computing',
    type: 'standard',
    potency: 7
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'ping') {
    // Respond to heartbeat
    ws.send(JSON.stringify({ type: 'pong' }));
  } else if (data.status === 'complete') {
    console.log('Enhanced prompt:', data.enhanced_prompt);
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
};
```

### Model Selection Sync

**Endpoint:** `ws://localhost:8001/api/v1/providers/ws/selection`

```javascript
const ws = new WebSocket('ws://localhost:8001/api/v1/providers/ws/selection');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'connected':
      console.log('Connected with client ID:', message.data.client_id);
      break;

    case 'selection_change':
      console.log('Model changed:', message.data);
      // Update UI with new model selection
      break;

    case 'health_update':
      console.log('Provider health:', message.data.providers);
      break;

    case 'heartbeat':
      // Connection is alive
      break;
  }
};
```

---

## Best Practices

### 1. API Key Security

- **Never commit API keys** to version control
- Store keys in environment variables or secure vaults
- Rotate keys regularly
- Use different keys for development and production

```bash
# .env file
CHIMERA_API_KEY=your-api-key-here

# Load in your application
import os
api_key = os.getenv('CHIMERA_API_KEY')
```

### 2. Error Handling

Always implement comprehensive error handling:

```python
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        # Handle rate limiting
        retry_after = e.response.headers.get('Retry-After', 60)
        time.sleep(int(retry_after))
    elif e.response.status_code == 503:
        # Provider unavailable, try fallback
        pass
    else:
        raise
except requests.exceptions.RequestException as e:
    # Handle network errors
    logger.error(f"Request failed: {e}")
    raise
```

### 3. Request Optimization

- **Batch requests** when possible
- **Cache responses** for repeated queries
- **Use appropriate timeouts**
- **Implement connection pooling**

```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### 4. Monitoring & Logging

Track API usage and performance:

```python
import logging
import time

logger = logging.getLogger(__name__)

def log_api_call(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"API call succeeded in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"API call failed after {duration:.2f}s: {e}")
            raise
    return wrapper
```

### 5. Session Management

Use session IDs for consistent model selection:

```python
headers = {
    'X-API-Key': API_KEY,
    'X-Session-ID': session_id,
    'Content-Type': 'application/json'
}
```

### 6. Testing

Always test against the development environment first:

```python
import pytest

@pytest.fixture
def chimera_client():
    return ChimeraClient(
        api_key=os.getenv('CHIMERA_TEST_API_KEY'),
        base_url='http://localhost:8001/api/v1'
    )

def test_generate_text(chimera_client):
    result = chimera_client.generate({
        'prompt': 'Test prompt',
        'provider': 'google'
    })
    assert 'text' in result
    assert result['provider'] == 'google'
```

---

## Support & Resources

- **Interactive Documentation:** <http://localhost:8001/docs>
- **API Reference:** <http://localhost:8001/redoc>
- **Health Dashboard:** <http://localhost:8001/health/full>
- **GitHub Repository:** [Link to repository]
- **Issue Tracker:** [Link to issues]

---

## Changelog

### Version 2.0.0 (2025-12-15)

- Added comprehensive OpenAPI 3.1 documentation
- Enhanced jailbreak generation with AI-powered techniques
- Improved rate limiting with fallback provider suggestions
- Added WebSocket support for real-time enhancements
- Implemented session-based model selection
- Added AutoDAN, AutoAdv, GPTFuzz, and HouYi endpoints

---

**Generated with Chimera API Documentation System**
**Last Updated:** 2025-12-15
