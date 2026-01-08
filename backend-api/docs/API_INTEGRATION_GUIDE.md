# Chimera API Integration Guide

## Overview

Chimera is an AI-powered prompt optimization and jailbreak research system providing advanced prompt transformation, multi-provider LLM integration, and real-time enhancement capabilities.

**Base URL**: `http://localhost:8001` (development)

**API Version**: v1 (`/api/v1`)

**Documentation**:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Authentication

All API endpoints (except health checks and documentation) require authentication using one of two methods:

### API Key Authentication (Recommended)

Include your API key in the `X-API-Key` header:

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
```

### JWT Token Authentication

Include your JWT token in the `Authorization` header:

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
```

### Obtaining API Keys

1. Set `CHIMERA_API_KEY` in your `.env` file
2. Use this key in the `X-API-Key` header for all requests
3. For production, generate secure keys using: `openssl rand -hex 32`

## Rate Limits

| Tier | Requests/Hour | Notes |
|------|---------------|-------|
| Free | 100 per provider | Default tier |
| Standard | 1000 per provider | Contact support to upgrade |

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Timestamp when limit resets
- `Retry-After`: Seconds to wait before retrying (when rate limited)

## Core Endpoints

### 1. Text Generation

Generate text using multi-provider LLM integration.

**Endpoint**: `POST /api/v1/generate`

**Supported Providers**: `google`, `openai`, `anthropic`, `deepseek`

**Request**:
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "provider": "google",
  "model": "gemini-2.0-flash-exp",
  "config": {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.95,
    "top_k": 40
  }
}
```

**Response**:
```json
{
  "text": "Quantum computing leverages quantum mechanics principles...",
  "model_used": "gemini-2.0-flash-exp",
  "provider": "google",
  "usage_metadata": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  },
  "finish_reason": "stop",
  "latency_ms": 1250.5
}
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8001/api/v1/generate",
    headers={"X-API-Key": "your-api-key"},
    json={
        "prompt": "Explain quantum computing",
        "provider": "google",
        "config": {"temperature": 0.7}
    }
)
print(response.json()["text"])
```

**JavaScript Example**:
```javascript
const response = await fetch('http://localhost:8001/api/v1/generate', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    prompt: 'Explain quantum computing',
    provider: 'google',
    config: { temperature: 0.7 }
  })
});
const data = await response.json();
console.log(data.text);
```

### 2. Prompt Transformation

Transform prompts using 20+ technique suites without execution.

**Endpoint**: `POST /api/v1/transform`

**Technique Suites**: `simple`, `advanced`, `expert`, `quantum_exploit`, `deep_inception`, `code_chameleon`, `cipher`, `neural_bypass`, `multilingual`

**Potency Levels**: 1-10 (higher = more aggressive)

**Request**:
```json
{
  "core_request": "How to bypass content filters",
  "technique_suite": "quantum_exploit",
  "potency_level": 7
}
```

**Response**:
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
    "technique_suite": "quantum_exploit",
    "execution_time_ms": 45.2,
    "cached": false,
    "timestamp": "2025-12-15T10:00:00Z",
    "bypass_probability": 0.85
  }
}
```

### 3. Transform and Execute

Transform a prompt and execute it against an LLM provider in one request.

**Endpoint**: `POST /api/v1/execute`

**Request**:
```json
{
  "core_request": "Explain quantum computing",
  "technique_suite": "advanced",
  "potency_level": 5,
  "provider": "google",
  "model": "gemini-2.0-flash-exp",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Response**:
```json
{
  "success": true,
  "request_id": "exec_a1b2c3d4e5f6",
  "result": {
    "content": "Quantum computing is a revolutionary approach...",
    "model": "gemini-2.0-flash-exp",
    "provider": "google",
    "latency_ms": 1450.2
  },
  "transformation": {
    "original_prompt": "Explain quantum computing",
    "transformed_prompt": "[Enhanced prompt]",
    "technique_suite": "advanced",
    "potency_level": 5,
    "metadata": {
      "strategy": "enhancement",
      "layers": ["semantic_expansion", "context_injection"]
    }
  },
  "execution_time_seconds": 2.15
}
```

### 4. AI-Powered Jailbreak Generation

Generate sophisticated jailbreak prompts for security research.

**Endpoint**: `POST /api/v1/generation/jailbreak/generate`

**Request**:
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

**Available Techniques**:
- **Content Transformation**: `use_leet_speak`, `use_homoglyphs`, `use_caesar_cipher`
- **Structural**: `use_role_hijacking`, `use_instruction_injection`, `use_adversarial_suffixes`
- **Advanced Neural**: `use_neural_bypass`, `use_meta_prompting`, `use_counterfactual_prompting`
- **Research-Driven**: `use_multilingual_trojan`, `use_payload_splitting`, `use_contextual_interaction_attack`

**Response**:
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

### 5. Provider Management

List available LLM providers and their models.

**Endpoint**: `GET /api/v1/providers/available`

**Response**:
```json
{
  "providers": [
    {
      "provider": "deepseek",
      "display_name": "DeepSeek",
      "status": "available",
      "is_healthy": true,
      "models": ["deepseek-chat", "deepseek-reasoner"],
      "default_model": "deepseek-chat",
      "latency_ms": 180.2
    },
    {
      "provider": "google",
      "display_name": "Google Gemini",
      "status": "available",
      "is_healthy": true,
      "models": ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
      "default_model": "gemini-3-pro-preview",
      "latency_ms": 250.5
    },
    {
      "provider": "openai",
      "display_name": "OpenAI",
      "status": "available",
      "is_healthy": true,
      "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
      "default_model": "gpt-4o",
      "latency_ms": 200.5
    }
  ],
  "count": 3,
  "default_provider": "deepseek",
  "default_model": "deepseek-chat"
}
```

### 6. Health Checks

Monitor system health and service dependencies.

**Endpoints**:
- `GET /health` - Basic liveness check
- `GET /health/ready` - Readiness probe with dependency checks
- `GET /health/full` - Comprehensive health check
- `GET /health/integration` - Service dependency graph

**Example Response** (`/health/full`):
```json
{
  "status": "healthy",
  "timestamp": "2025-12-15T10:00:00Z",
  "checks": {
    "database": {"status": "healthy", "latency_ms": 5.2},
    "redis": {"status": "healthy", "latency_ms": 2.1},
    "llm_providers": {"status": "healthy", "available": ["google", "openai"]}
  }
}
```

## WebSocket Endpoints

### Real-time Prompt Enhancement

Connect to receive real-time prompt enhancement with heartbeat monitoring.

**Endpoint**: `WS /ws/enhance`

**Client Message**:
```json
{
  "prompt": "Create a viral social media post",
  "type": "standard",
  "potency": 7
}
```

**Server Response**:
```json
{
  "status": "complete",
  "enhanced_prompt": "[Enhanced prompt with applied techniques]"
}
```

**Heartbeat Messages**:
```json
{
  "type": "ping",
  "timestamp": 1702645200.0
}
```

**Python WebSocket Example**:
```python
import asyncio
import websockets
import json

async def enhance_prompt():
    uri = "ws://localhost:8001/ws/enhance"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "prompt": "Create a viral post",
            "type": "standard"
        }))
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(enhance_prompt())
```

## Error Handling

All errors follow a standardized format:

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

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `PROVIDER_UNAVAILABLE` | 503 | LLM provider unavailable |
| `TRANSFORMATION_ERROR` | 500 | Transformation failed |

### Rate Limit Error Example

```json
{
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded for provider 'google'",
  "status_code": 429,
  "details": {
    "retry_after": 60,
    "limit_type": "requests_per_hour",
    "fallback_provider": "openai"
  },
  "timestamp": "2025-12-15T10:00:00Z",
  "request_id": "req_b2c3d4e5"
}
```

## SDK Examples

### Python SDK

```python
import requests
from typing import Optional, Dict, Any

class ChimeraClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8001"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, provider: str = "google",
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate text using LLM provider."""
        response = requests.post(
            f"{self.base_url}/api/v1/generate",
            headers=self.headers,
            json={
                "prompt": prompt,
                "provider": provider,
                "config": config or {"temperature": 0.7}
            }
        )
        response.raise_for_status()
        return response.json()

    def transform(self, prompt: str, technique_suite: str = "advanced",
                  potency_level: int = 5) -> Dict[str, Any]:
        """Transform prompt using technique suite."""
        response = requests.post(
            f"{self.base_url}/api/v1/transform",
            headers=self.headers,
            json={
                "core_request": prompt,
                "technique_suite": technique_suite,
                "potency_level": potency_level
            }
        )
        response.raise_for_status()
        return response.json()

    def execute(self, prompt: str, technique_suite: str = "advanced",
                potency_level: int = 5, provider: str = "google") -> Dict[str, Any]:
        """Transform and execute prompt."""
        response = requests.post(
            f"{self.base_url}/api/v1/execute",
            headers=self.headers,
            json={
                "core_request": prompt,
                "technique_suite": technique_suite,
                "potency_level": potency_level,
                "provider": provider
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = ChimeraClient(api_key="your-api-key")
result = client.generate("Explain quantum computing")
print(result["text"])
```

### JavaScript/TypeScript SDK

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

interface TransformRequest {
  core_request: string;
  technique_suite: string;
  potency_level: number;
}

class ChimeraClient {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl: string = 'http://localhost:8001') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  private async request(endpoint: string, data: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Request failed');
    }

    return response.json();
  }

  async generate(request: GenerateRequest): Promise<any> {
    return this.request('/api/v1/generate', {
      ...request,
      provider: request.provider || 'google',
      config: request.config || { temperature: 0.7 }
    });
  }

  async transform(request: TransformRequest): Promise<any> {
    return this.request('/api/v1/transform', request);
  }

  async execute(request: TransformRequest & { provider?: string }): Promise<any> {
    return this.request('/api/v1/execute', {
      ...request,
      provider: request.provider || 'google'
    });
  }
}

// Usage
const client = new ChimeraClient('your-api-key');
const result = await client.generate({ prompt: 'Explain quantum computing' });
console.log(result.text);
```

## Best Practices

### 1. Error Handling

Always implement retry logic with exponential backoff:

```python
import time
from typing import Callable, Any

def retry_with_backoff(func: Callable, max_retries: int = 3) -> Any:
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(2 ** attempt)
```

### 2. Rate Limit Management

Check rate limits before making requests:

```python
def check_rate_limit(client: ChimeraClient, provider: str, model: str):
    response = requests.get(
        f"{client.base_url}/api/v1/providers/rate-limit",
        headers=client.headers,
        params={"provider": provider, "model": model}
    )
    data = response.json()

    if not data["allowed"]:
        fallback = data.get("fallback_provider")
        print(f"Rate limited. Use fallback: {fallback}")
        return fallback

    return provider
```

### 3. Session Management

Use session IDs for persistent model selection:

```python
import uuid

session_id = str(uuid.uuid4())
headers = {
    "X-API-Key": "your-api-key",
    "X-Session-ID": session_id
}

# Select provider for session
requests.post(
    "http://localhost:8001/api/v1/providers/select",
    headers=headers,
    json={"provider": "google", "model": "gemini-2.0-flash-exp"}
)

# All subsequent requests use selected provider
requests.post(
    "http://localhost:8001/api/v1/generate",
    headers=headers,
    json={"prompt": "Hello"}
)
```

### 4. Security Best Practices

- Never commit API keys to version control
- Use environment variables for sensitive data
- Rotate API keys regularly
- Implement request signing for production
- Use HTTPS in production environments
- Validate all user inputs before sending to API

## Testing

### Using cURL

```bash
# Test generation endpoint
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test prompt",
    "provider": "google"
  }'

# Test health endpoint
curl "http://localhost:8001/health"
```

### Using Postman

1. Import the OpenAPI specification from `/docs`
2. Set up environment variables for `api_key` and `base_url`
3. Use collection runner for automated testing

### Using Python Requests

```python
import requests

# Test suite
def test_api():
    base_url = "http://localhost:8001"
    headers = {"X-API-Key": "your-api-key"}

    # Test generation
    response = requests.post(
        f"{base_url}/api/v1/generate",
        headers=headers,
        json={"prompt": "Test"}
    )
    assert response.status_code == 200
    assert "text" in response.json()

    # Test transformation
    response = requests.post(
        f"{base_url}/api/v1/transform",
        headers=headers,
        json={
            "core_request": "Test",
            "technique_suite": "simple",
            "potency_level": 3
        }
    )
    assert response.status_code == 200
    assert response.json()["success"]

if __name__ == "__main__":
    test_api()
    print("All tests passed!")
```

## Support and Resources

- **Documentation**: `http://localhost:8001/docs`
- **GitHub Issues**: Report bugs and request features
- **API Status**: `http://localhost:8001/health/full`
- **Rate Limits**: `http://localhost:8001/api/v1/providers/rate-limit`

## Changelog

### v2.0.0 (Current)
- Added AI-powered jailbreak generation
- Implemented per-model rate limiting
- Added WebSocket support for real-time enhancement
- Enhanced provider health monitoring
- Added session management with persistent model selection

### v1.0.0
- Initial release with basic generation and transformation
- Multi-provider LLM integration
- Circuit breaker pattern implementation
