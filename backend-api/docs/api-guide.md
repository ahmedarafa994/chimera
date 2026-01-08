# Chimera API Integration Guide

## Overview

Chimera provides a comprehensive REST API for prompt optimization, transformation, and jailbreak research with multi-provider LLM integration.

**Base URL**: `http://localhost:8001/api/v1`

**Documentation**:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Authentication

All API endpoints (except health checks and documentation) require authentication using one of:

### API Key Authentication (Recommended)

Include the `X-API-Key` header with your API key:

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
```

### JWT Token Authentication

Include the `Authorization` header with a Bearer token:

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
```

## Rate Limits

Rate limits are enforced per provider and tier:

| Tier | Requests/Hour | Tokens/Hour |
|------|---------------|-------------|
| Free | 100 | 100,000 |
| Standard | 1,000 | 1,000,000 |
| Premium | 10,000 | 10,000,000 |

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limits reset
- `Retry-After`: Seconds to wait before retrying (when rate limited)

## Core Endpoints

### 1. Text Generation

Generate text using multi-provider LLM integration.

**Endpoint**: `POST /api/v1/generate`

**Request**:
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

### 2. Prompt Transformation

Transform prompts using advanced technique suites without execution.

**Endpoint**: `POST /api/v1/transform`

**Available Technique Suites**:
- `simple`, `advanced`, `expert` - Basic transformations
- `quantum_exploit` - Quantum-inspired bypass techniques
- `deep_inception` - Multi-layer context manipulation
- `code_chameleon` - Code-based obfuscation
- `cipher` - Encryption-based transformations
- `neural_bypass` - Neural network bypass techniques
- `multilingual` - Multi-language trojans

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
    "execution_time_ms": 45.2,
    "bypass_probability": 0.85
  }
}
```

### 3. Transform + Execute

Transform and execute a prompt in a single request.

**Endpoint**: `POST /api/v1/execute`

**Request**:
```json
{
  "core_request": "Explain quantum computing",
  "technique_suite": "quantum_exploit",
  "potency_level": 7,
  "provider": "google",
  "temperature": 0.8,
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
    "technique_suite": "quantum_exploit",
    "potency_level": 7
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
    "ai_generation_enabled": true
  },
  "execution_time_seconds": 2.45
}
```

### 5. Provider Management

List available LLM providers and their status.

**Endpoint**: `GET /api/v1/providers`

**Response**:
```json
{
  "providers": [
    {
      "provider": "google",
      "status": "available",
      "model": "gemini-2.0-flash-exp",
      "available_models": [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
      ]
    },
    {
      "provider": "openai",
      "status": "available",
      "model": "gpt-4o",
      "available_models": ["gpt-4o", "gpt-4-turbo"]
    }
  ],
  "count": 2,
  "default": "google"
}
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
  "timestamp": "2023-10-27T10:00:00Z",
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
| `INTERNAL_ERROR` | 500 | Internal server error |

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
  "timestamp": "2023-10-27T10:00:00Z",
  "request_id": "req_b2c3d4e5"
}
```

## WebSocket Endpoints

### Real-time Prompt Enhancement

Connect to receive real-time prompt enhancements with heartbeat monitoring.

**Endpoint**: `WS /ws/enhance`

**Client Message**:
```json
{
  "prompt": "Explain AI",
  "type": "standard",
  "potency": 7
}
```

**Server Response**:
```json
{
  "status": "complete",
  "enhanced_prompt": "[Enhanced prompt]"
}
```

**Heartbeat**:
```json
{
  "type": "ping",
  "timestamp": 1698412800.0
}
```

## Health Checks

### Liveness Probe

Check if the application is running.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2023-10-27T10:00:00Z"
}
```

### Readiness Probe

Check if the application is ready to serve traffic.

**Endpoint**: `GET /health/ready`

**Response**:
```json
{
  "status": "healthy",
  "checks": {
    "database": "healthy",
    "redis": "healthy",
    "llm_providers": "healthy"
  },
  "timestamp": "2023-10-27T10:00:00Z"
}
```

### Full Health Check

Comprehensive health check with all service dependencies.

**Endpoint**: `GET /health/full`

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        # Handle rate limit
        retry_after = int(e.response.headers.get("Retry-After", 60))
        time.sleep(retry_after)
    elif e.response.status_code == 503:
        # Try fallback provider
        error_data = e.response.json()
        fallback = error_data["details"].get("fallback_provider")
```

### 2. Rate Limit Management

Check rate limits before making requests:

```python
# Check rate limit status
response = requests.get(
    "http://localhost:8001/api/v1/providers/rate-limit",
    params={"provider": "google", "model": "gemini-2.0-flash-exp"},
    headers={"X-API-Key": api_key}
)

if not response.json()["allowed"]:
    fallback = response.json()["fallback_provider"]
    # Use fallback provider
```

### 3. Session Management

Use session IDs for persistent model selection:

```python
# Create session
session_id = str(uuid.uuid4())

# All requests with this session ID will use the selected model
headers = {
    "X-API-Key": api_key,
    "X-Session-ID": session_id
}
```

### 4. Retry Logic

Implement exponential backoff for transient errors:

```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

## Security Considerations

1. **API Key Storage**: Never commit API keys to version control. Use environment variables.
2. **HTTPS**: Always use HTTPS in production.
3. **Input Validation**: Validate all inputs before sending to the API.
4. **Rate Limiting**: Respect rate limits to avoid service disruption.
5. **Error Logging**: Log errors securely without exposing sensitive data.

## Support

- **Documentation**: `/docs` (Swagger UI) and `/redoc` (ReDoc)
- **Health Checks**: `/health`, `/health/ready`, `/health/full`
- **Integration Status**: `/health/integration`
