# AI Provider API Reference

Complete API reference documentation for the AI provider integration layer, including all endpoints, request/response schemas, and example usage.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Headers](#common-headers)
4. [Configuration Endpoints](#configuration-endpoints)
5. [Provider Management Endpoints](#provider-management-endpoints)
6. [Health Check Endpoints](#health-check-endpoints)
7. [Generation Endpoints](#generation-endpoints)
8. [Cost Tracking Endpoints](#cost-tracking-endpoints)
9. [Rate Limit Endpoints](#rate-limit-endpoints)
10. [WebSocket Endpoints](#websocket-endpoints)
11. [Error Responses](#error-responses)
12. [SDK Examples](#sdk-examples)

---

## Overview

### Base URL

```
http://localhost:8000/api/v1
```

### API Versioning

The API uses URL-based versioning. Current version: `v1`

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **Streaming**: `text/event-stream`

### Response Format

All responses follow a consistent JSON structure:

```json
{
  "status": "success",
  "data": { ... },
  "meta": {
    "provider": "gemini",
    "model": "gemini-2.0-flash-exp",
    "cost": 0.000150,
    "latency_ms": 234
  }
}
```

Error responses:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "status_code": 400
}
```

---

## Authentication

### API Key Authentication

Include your API key in the request header:

```bash
curl -H "X-API-Key: your-api-key" https://api.example.com/api/v1/generate
```

### Bearer Token Authentication

```bash
curl -H "Authorization: Bearer your-token" https://api.example.com/api/v1/generate
```

---

## Common Headers

### Request Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Content-Type` | string | Yes | Must be `application/json` |
| `X-API-Key` | string | Conditional | API key for authentication |
| `Authorization` | string | Conditional | Bearer token authentication |
| `X-Provider` | string | No | Override default provider |
| `X-Model` | string | No | Override default model |
| `X-Failover-Chain` | string | No | Named failover chain to use |
| `X-Request-ID` | string | No | Custom request ID for tracing |

### Response Headers

| Header | Type | Description |
|--------|------|-------------|
| `X-Provider-Used` | string | Actual provider used for request |
| `X-Model-Used` | string | Actual model used for request |
| `X-Request-Cost` | string | Cost of request (e.g., `$0.000150`) |
| `X-RateLimit-Remaining` | integer | Remaining requests in current window |
| `X-RateLimit-Reset` | integer | Unix timestamp when limit resets |
| `X-Request-ID` | string | Request ID for tracing |

---

## Configuration Endpoints

### Get Current Configuration

Retrieves the current AI provider configuration.

```http
GET /config
```

**Response:**

```json
{
  "version": "1.0",
  "global": {
    "default_provider": "gemini",
    "default_model": "gemini-2.0-flash-exp",
    "request_timeout": 30,
    "max_retries": 3,
    "enable_caching": true,
    "cache_ttl": 3600
  },
  "providers": {
    "gemini": {
      "enabled": true,
      "capabilities": {
        "supports_streaming": true,
        "supports_vision": true,
        "supports_function_calling": true
      }
    }
  },
  "failover_chains": {
    "default": ["gemini", "openai", "anthropic"],
    "premium": ["openai", "anthropic", "gemini"]
  }
}
```

**Example:**

```bash
curl -X GET http://localhost:8000/api/v1/config \
  -H "X-API-Key: your-api-key"
```

---

### Get Configuration Status

Returns the status of the configuration system.

```http
GET /config/status
```

**Response:**

```json
{
  "loaded": true,
  "config_path": "app/config/providers.yaml",
  "version": "1.0",
  "last_reload": "2024-01-15T10:30:00Z",
  "providers_count": 8,
  "enabled_providers": ["gemini", "openai", "anthropic", "deepseek"],
  "validation_status": "valid"
}
```

---

### Reload Configuration

Hot-reload the configuration without restarting the application.

```http
POST /config/reload
```

**Request Body:**

```json
{
  "validate_only": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `validate_only` | boolean | No | Only validate, don't apply changes |

**Response:**

```json
{
  "status": "success",
  "message": "Configuration reloaded successfully",
  "changes": {
    "providers_added": [],
    "providers_removed": [],
    "settings_changed": ["global.request_timeout"]
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/config/reload \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"validate_only": false}'
```

---

### Validate Configuration

Validate a configuration file without applying it.

```http
POST /config/validate
```

**Request Body:**

```json
{
  "config_path": "app/config/providers.yaml"
}
```

**Response:**

```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    "Provider 'cursor' has no models configured"
  ]
}
```

---

## Provider Management Endpoints

### List Providers

Get a list of all configured providers.

```http
GET /providers
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled_only` | boolean | Only return enabled providers |
| `include_models` | boolean | Include model configurations |

**Response:**

```json
{
  "providers": [
    {
      "id": "gemini",
      "display_name": "Google Gemini",
      "enabled": true,
      "healthy": true,
      "models_count": 3,
      "default_model": "gemini-2.0-flash-exp",
      "capabilities": {
        "supports_streaming": true,
        "supports_vision": true,
        "supports_function_calling": true,
        "supports_json_mode": true,
        "supports_embeddings": true
      }
    },
    {
      "id": "openai",
      "display_name": "OpenAI",
      "enabled": true,
      "healthy": true,
      "models_count": 5,
      "default_model": "gpt-4o",
      "capabilities": {
        "supports_streaming": true,
        "supports_vision": true,
        "supports_function_calling": true,
        "supports_json_mode": true,
        "supports_embeddings": true
      }
    }
  ],
  "total": 8,
  "enabled": 6
}
```

**Example:**

```bash
curl -X GET "http://localhost:8000/api/v1/providers?enabled_only=true&include_models=true" \
  -H "X-API-Key: your-api-key"
```

---

### Get Provider Details

Get detailed information about a specific provider.

```http
GET /providers/{provider_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider_id` | string | Provider identifier (e.g., `gemini`, `openai`) |

**Response:**

```json
{
  "id": "openai",
  "display_name": "OpenAI",
  "enabled": true,
  "base_url": "https://api.openai.com/v1",
  "api_version": null,
  "capabilities": {
    "supports_streaming": true,
    "supports_vision": true,
    "supports_function_calling": true,
    "supports_json_mode": true,
    "supports_system_prompt": true,
    "supports_token_counting": true,
    "supports_embeddings": true
  },
  "rate_limits": {
    "requests_per_minute": 500,
    "tokens_per_minute": 150000,
    "requests_per_day": 10000
  },
  "models": [
    {
      "id": "gpt-4o",
      "display_name": "GPT-4o",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "is_default": true,
      "pricing": {
        "input_per_1k": 0.005,
        "output_per_1k": 0.015
      }
    }
  ],
  "health": {
    "status": "healthy",
    "last_check": "2024-01-15T10:30:00Z",
    "latency_ms": 150,
    "error_rate": 0.01
  }
}
```

---

### Enable/Disable Provider

Enable or disable a provider at runtime.

```http
PATCH /providers/{provider_id}
```

**Request Body:**

```json
{
  "enabled": false
}
```

**Response:**

```json
{
  "status": "success",
  "provider_id": "openai",
  "enabled": false,
  "message": "Provider disabled successfully"
}
```

---

### Get Provider Models

List all models for a specific provider.

```http
GET /providers/{provider_id}/models
```

**Response:**

```json
{
  "provider_id": "openai",
  "models": [
    {
      "id": "gpt-4o",
      "display_name": "GPT-4o",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "is_default": true,
      "supports_vision": true,
      "supports_function_calling": true,
      "pricing": {
        "input_per_1k": 0.005,
        "output_per_1k": 0.015
      }
    },
    {
      "id": "gpt-4-turbo",
      "display_name": "GPT-4 Turbo",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "is_default": false,
      "supports_vision": true,
      "supports_function_calling": true,
      "pricing": {
        "input_per_1k": 0.01,
        "output_per_1k": 0.03
      }
    }
  ],
  "total": 5
}
```

---

### Get Failover Chains

List all configured failover chains.

```http
GET /providers/failover-chains
```

**Response:**

```json
{
  "chains": {
    "default": {
      "providers": ["gemini", "openai", "anthropic"],
      "description": "Default failover chain"
    },
    "premium": {
      "providers": ["openai", "anthropic", "gemini"],
      "description": "High-quality output chain"
    },
    "cost_optimized": {
      "providers": ["deepseek", "qwen", "gemini"],
      "description": "Cost-effective chain"
    }
  }
}
```

---

## Health Check Endpoints

### System Health

Check overall system health.

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "config": "healthy",
    "providers": "healthy",
    "database": "healthy",
    "cache": "healthy"
  }
}
```

---

### Provider Health

Check health of all providers.

```http
GET /health/providers
```

**Response:**

```json
{
  "overall_status": "degraded",
  "providers": {
    "gemini": {
      "status": "healthy",
      "latency_ms": 120,
      "last_check": "2024-01-15T10:30:00Z",
      "circuit_breaker": "closed"
    },
    "openai": {
      "status": "healthy",
      "latency_ms": 200,
      "last_check": "2024-01-15T10:30:00Z",
      "circuit_breaker": "closed"
    },
    "anthropic": {
      "status": "unhealthy",
      "error": "Connection timeout",
      "last_check": "2024-01-15T10:29:00Z",
      "circuit_breaker": "open"
    }
  },
  "healthy_count": 6,
  "unhealthy_count": 2
}
```

---

### Single Provider Health

Check health of a specific provider.

```http
GET /health/providers/{provider_id}
```

**Response:**

```json
{
  "provider_id": "openai",
  "status": "healthy",
  "latency_ms": 180,
  "last_check": "2024-01-15T10:30:00Z",
  "circuit_breaker": {
    "state": "closed",
    "failure_count": 0,
    "success_count": 150
  },
  "rate_limit": {
    "remaining": 450,
    "limit": 500,
    "reset_at": "2024-01-15T10:31:00Z"
  }
}
```

---

### Circuit Breaker Status

Get circuit breaker status for all providers.

```http
GET /health/circuit-breakers
```

**Response:**

```json
{
  "circuit_breakers": {
    "gemini": {
      "state": "closed",
      "failure_count": 0,
      "success_count": 200,
      "last_failure": null
    },
    "openai": {
      "state": "closed",
      "failure_count": 2,
      "success_count": 198,
      "last_failure": "2024-01-15T09:15:00Z"
    },
    "anthropic": {
      "state": "open",
      "failure_count": 5,
      "success_count": 0,
      "last_failure": "2024-01-15T10:25:00Z",
      "opens_at": "2024-01-15T10:26:00Z"
    }
  }
}
```

---

## Generation Endpoints

### Generate Text

Generate text using the configured AI provider.

```http
POST /generate
```

**Request Body:**

```json
{
  "prompt": "Write a haiku about coding",
  "system_prompt": "You are a creative poet",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop_sequences": ["\n\n"],
  "stream": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The input prompt |
| `system_prompt` | string | No | - | System instruction |
| `max_tokens` | integer | No | 1024 | Maximum output tokens |
| `temperature` | float | No | 0.7 | Sampling temperature (0-2) |
| `top_p` | float | No | 1.0 | Top-p sampling |
| `stop_sequences` | array | No | [] | Stop sequences |
| `stream` | boolean | No | false | Enable streaming response |

**Response:**

```json
{
  "text": "Lines of code flow down\nBugs emerge like morning mist\nDebug, rinse, repeat",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 22,
    "total_tokens": 37
  },
  "provider": "gemini",
  "model": "gemini-2.0-flash-exp",
  "cost": 0.0,
  "latency_ms": 450
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "Write a haiku about coding",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

### Generate with Specific Provider

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "X-Provider: openai" \
  -H "X-Model: gpt-4o" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 500
  }'
```

---

### Generate with Failover Chain

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "X-Failover-Chain: premium" \
  -d '{
    "prompt": "Write professional documentation",
    "max_tokens": 1000
  }'
```

---

### Streaming Generation

Generate text with streaming response.

```http
POST /generate/stream
```

**Request Body:**

```json
{
  "prompt": "Write a story about a robot",
  "max_tokens": 500,
  "temperature": 0.8
}
```

**Response:** `text/event-stream`

```
data: {"delta": "Once", "index": 0}

data: {"delta": " upon", "index": 1}

data: {"delta": " a", "index": 2}

data: {"delta": " time", "index": 3}

data: [DONE]
```

**Example:**

```bash
curl -X POST http://localhost:8000/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -N \
  -d '{
    "prompt": "Write a story about a robot",
    "max_tokens": 500
  }'
```

---

### Chat Completion

Multi-turn chat completion endpoint.

```http
POST /chat
```

**Request Body:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "What's the weather like?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7
}
```

**Response:**

```json
{
  "message": {
    "role": "assistant",
    "content": "I don't have access to real-time weather data. Could you tell me your location? I can suggest some weather websites or apps you could check."
  },
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 32,
    "total_tokens": 77
  },
  "provider": "gemini",
  "model": "gemini-2.0-flash-exp"
}
```

---

### Generate Embeddings

Generate embeddings for text.

```http
POST /embeddings
```

**Request Body:**

```json
{
  "input": "The quick brown fox jumps over the lazy dog",
  "model": "text-embedding-3-small"
}
```

Or batch input:

```json
{
  "input": [
    "First sentence",
    "Second sentence",
    "Third sentence"
  ]
}
```

**Response:**

```json
{
  "embeddings": [
    {
      "index": 0,
      "embedding": [0.001, -0.023, 0.145, ...],
      "dimensions": 1536
    }
  ],
  "usage": {
    "total_tokens": 9
  },
  "provider": "openai",
  "model": "text-embedding-3-small"
}
```

---

## Cost Tracking Endpoints

### Get Cost Summary

Get aggregated cost information.

```http
GET /costs/summary
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `since` | datetime | Start datetime (ISO 8601) |
| `until` | datetime | End datetime (ISO 8601) |

**Response:**

```json
{
  "total_cost": 12.50,
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  },
  "by_provider": {
    "openai": 8.75,
    "anthropic": 3.25,
    "gemini": 0.50
  },
  "by_model": {
    "openai:gpt-4o": 6.50,
    "openai:gpt-3.5-turbo": 2.25,
    "anthropic:claude-3-sonnet": 3.25,
    "gemini:gemini-2.0-flash-exp": 0.50
  },
  "request_count": 1250,
  "token_count": {
    "input": 450000,
    "output": 125000
  }
}
```

---

### Get Daily Costs

Get daily cost breakdown.

```http
GET /costs/daily
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | integer | 7 | Number of days to retrieve |

**Response:**

```json
{
  "daily_costs": {
    "2024-01-15": 1.25,
    "2024-01-14": 0.95,
    "2024-01-13": 1.50,
    "2024-01-12": 2.10,
    "2024-01-11": 1.75,
    "2024-01-10": 1.30,
    "2024-01-09": 1.65
  },
  "total": 10.50,
  "average_daily": 1.50
}
```

---

### Get Cost by Provider

Get costs broken down by provider.

```http
GET /costs/by-provider
```

**Response:**

```json
{
  "costs": {
    "openai": {
      "total": 8.75,
      "request_count": 500,
      "input_tokens": 250000,
      "output_tokens": 75000
    },
    "anthropic": {
      "total": 3.25,
      "request_count": 200,
      "input_tokens": 100000,
      "output_tokens": 30000
    }
  }
}
```

---

### Get Cost Alerts

Get active cost alerts.

```http
GET /costs/alerts
```

**Response:**

```json
{
  "alerts": [
    {
      "type": "daily_budget_warning",
      "provider": null,
      "threshold": 50.00,
      "current_value": 42.50,
      "timestamp": "2024-01-15T14:30:00Z",
      "message": "Daily cost $42.50 has reached 85% of budget $50.00"
    }
  ]
}
```

---

### Get Recent Cost Entries

Get recent cost entries for detailed analysis.

```http
GET /costs/entries
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Maximum entries to return |

**Response:**

```json
{
  "entries": [
    {
      "provider": "openai",
      "model": "gpt-4o",
      "input_tokens": 500,
      "output_tokens": 150,
      "cost": 0.0047,
      "timestamp": "2024-01-15T10:30:00Z",
      "request_id": "req_abc123"
    }
  ],
  "total": 100
}
```

---

## Rate Limit Endpoints

### Get Rate Limit Status

Get current rate limit status.

```http
GET /rate-limits
```

**Response:**

```json
{
  "limits": {
    "gemini": {
      "requests_per_minute": {
        "limit": 60,
        "remaining": 45,
        "reset_at": "2024-01-15T10:31:00Z"
      },
      "tokens_per_minute": {
        "limit": 1000000,
        "remaining": 850000,
        "reset_at": "2024-01-15T10:31:00Z"
      }
    },
    "openai": {
      "requests_per_minute": {
        "limit": 500,
        "remaining": 480,
        "reset_at": "2024-01-15T10:31:00Z"
      },
      "requests_per_day": {
        "limit": 10000,
        "remaining": 9500,
        "reset_at": "2024-01-16T00:00:00Z"
      }
    }
  }
}
```

---

### Get Provider Rate Limit

Get rate limit status for a specific provider.

```http
GET /rate-limits/{provider_id}
```

**Response:**

```json
{
  "provider_id": "openai",
  "requests_per_minute": {
    "limit": 500,
    "remaining": 480,
    "used": 20,
    "reset_at": "2024-01-15T10:31:00Z"
  },
  "tokens_per_minute": {
    "limit": 150000,
    "remaining": 145000,
    "used": 5000,
    "reset_at": "2024-01-15T10:31:00Z"
  },
  "requests_per_day": {
    "limit": 10000,
    "remaining": 9500,
    "used": 500,
    "reset_at": "2024-01-16T00:00:00Z"
  }
}
```

---

### Reset Rate Limits (Admin)

Reset rate limits for a provider (admin only).

```http
POST /rate-limits/{provider_id}/reset
```

**Response:**

```json
{
  "status": "success",
  "provider_id": "openai",
  "message": "Rate limits reset successfully"
}
```

---

## WebSocket Endpoints

### Real-time Provider Updates

Subscribe to real-time provider health and status updates.

```
WS /ws/providers
```

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/providers');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Provider update:', data);
};
```

**Message Types:**

```json
{
  "type": "health_update",
  "provider": "openai",
  "status": "healthy",
  "latency_ms": 150,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

```json
{
  "type": "circuit_breaker",
  "provider": "anthropic",
  "state": "open",
  "reason": "Consecutive failures exceeded threshold",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

```json
{
  "type": "rate_limit_warning",
  "provider": "openai",
  "remaining": 10,
  "limit": 500,
  "reset_at": "2024-01-15T10:31:00Z"
}
```

---

### Streaming Generation WebSocket

Real-time streaming generation via WebSocket.

```
WS /ws/generate
```

**Send:**

```json
{
  "action": "generate",
  "prompt": "Write a story",
  "max_tokens": 500,
  "temperature": 0.7
}
```

**Receive:**

```json
{
  "type": "token",
  "content": "Once",
  "index": 0
}
```

```json
{
  "type": "token",
  "content": " upon",
  "index": 1
}
```

```json
{
  "type": "done",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 150,
    "total_tokens": 160
  }
}
```

---

## Error Responses

### Error Format

All errors follow this format:

```json
{
  "error": "ErrorType",
  "detail": "Detailed error message",
  "status_code": 400,
  "request_id": "req_abc123",
  "provider": "openai",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | `ValidationError` | Invalid request parameters |
| 401 | `AuthenticationError` | Invalid or missing API key |
| 403 | `PermissionDenied` | Insufficient permissions |
| 404 | `NotFound` | Resource not found |
| 429 | `RateLimitExceeded` | Rate limit exceeded |
| 500 | `InternalError` | Server error |
| 502 | `ProviderError` | Upstream provider error |
| 503 | `ServiceUnavailable` | Service temporarily unavailable |

### Error Examples

**Validation Error (400):**

```json
{
  "error": "ValidationError",
  "detail": "max_tokens must be between 1 and 4096",
  "status_code": 400
}
```

**Rate Limit Exceeded (429):**

```json
{
  "error": "RateLimitExceeded",
  "detail": "Rate limit exceeded: 500/minute",
  "status_code": 429,
  "provider": "openai",
  "retry_after": 45,
  "remaining": {
    "requests_remaining_minute": 0,
    "reset_at_minute": 1705315860
  }
}
```

**Provider Unavailable (503):**

```json
{
  "error": "ServiceUnavailable",
  "detail": "All providers in failover chain are unavailable",
  "status_code": 503,
  "providers_tried": ["gemini", "openai", "anthropic"],
  "errors": {
    "gemini": "Connection timeout",
    "openai": "Circuit breaker open",
    "anthropic": "Rate limit exceeded"
  }
}
```

---

## SDK Examples

### Python

```python
import httpx

class ChimeraClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-API-Key": api_key}
        )
    
    async def generate(
        self,
        prompt: str,
        provider: str = None,
        model: str = None,
        **kwargs
    ):
        headers = {}
        if provider:
            headers["X-Provider"] = provider
        if model:
            headers["X-Model"] = model
        
        response = await self.client.post(
            "/api/v1/generate",
            json={"prompt": prompt, **kwargs},
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    async def stream_generate(self, prompt: str, **kwargs):
        async with self.client.stream(
            "POST",
            "/api/v1/generate/stream",
            json={"prompt": prompt, **kwargs}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]

# Usage
async def main():
    client = ChimeraClient(
        "http://localhost:8000",
        "your-api-key"
    )
    
    # Simple generation
    result = await client.generate(
        "Write a haiku about coding",
        temperature=0.7
    )
    print(result["text"])
    
    # With specific provider
    result = await client.generate(
        "Explain quantum physics",
        provider="openai",
        model="gpt-4o",
        max_tokens=500
    )
    print(result["text"])
    
    # Streaming
    async for chunk in client.stream_generate("Write a story"):
        print(chunk, end="", flush=True)
```

---

### JavaScript/TypeScript

```typescript
interface GenerateOptions {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  provider?: string;
  model?: string;
}

interface GenerateResponse {
  text: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  provider: string;
  model: string;
  cost: number;
}

class ChimeraClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  async generate(options: GenerateOptions): Promise<GenerateResponse> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-API-Key': this.apiKey,
    };

    if (options.provider) {
      headers['X-Provider'] = options.provider;
    }
    if (options.model) {
      headers['X-Model'] = options.model;
    }

    const response = await fetch(`${this.baseUrl}/api/v1/generate`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        prompt: options.prompt,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Generation failed');
    }

    return response.json();
  }

  async *streamGenerate(options: GenerateOptions): AsyncGenerator<string> {
    const response = await fetch(`${this.baseUrl}/api/v1/generate/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
      },
      body: JSON.stringify({
        prompt: options.prompt,
        max_tokens: options.maxTokens,
        temperature: options.temperature,
      }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) throw new Error('No response body');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data !== '[DONE]') {
            const parsed = JSON.parse(data);
            yield parsed.delta;
          }
        }
      }
    }
  }
}

// Usage
const client = new ChimeraClient('http://localhost:8000', 'your-api-key');

// Simple generation
const result = await client.generate({
  prompt: 'Write a haiku about coding',
  temperature: 0.7,
});
console.log(result.text);

// Streaming
for await (const chunk of client.streamGenerate({ prompt: 'Write a story' })) {
  process.stdout.write(chunk);
}
```

---

### cURL Examples

**Basic Generation:**

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'
```

**With Provider Override:**

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -H "X-Provider: openai" \
  -H "X-Model: gpt-4o" \
  -d '{"prompt": "Explain AI", "max_tokens": 500}'
```

**Streaming:**

```bash
curl -X POST http://localhost:8000/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -N \
  -d '{"prompt": "Write a story", "max_tokens": 500}'
```

**Get Provider Health:**

```bash
curl http://localhost:8000/api/v1/health/providers \
  -H "X-API-Key: your-api-key"
```

**Get Cost Summary:**

```bash
curl "http://localhost:8000/api/v1/costs/summary?since=2024-01-01T00:00:00Z" \
  -H "X-API-Key: your-api-key"
```

---

## Next Steps

- [Integration Guide](./ai_provider_integration_guide.md) - Complete implementation guide
- [Migration Guide](./ai_provider_migration_guide.md) - Migrating from legacy configuration
- [Configuration Map](./ai_provider_configuration_map.md) - Complete configuration matrix
