
# LLM Integration API Documentation

## Overview

The Project Chimera LLM Integration API provides production-ready access to advanced prompt transformation techniques executed across multiple LLM providers including OpenAI, Anthropic Claude, and custom model endpoints.

## Base URL

```
http://localhost:5000/api/v1
```

## Authentication

All API endpoints require authentication via API key in the request header:

```
X-API-Key: your_api_key_here
```

## Response Format

All responses follow this standard format:

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Endpoints

### 1. Health Check

Check API health status.

**Endpoint:** `GET /health`

**Authentication:** Not required

**Response:**
```json
{
  "status": "operational",
  "timestamp": "2024-01-01T12:00:00Z",
  "health": {
    "status": "healthy",
    "color": "green",
    "checks": {
      "recent_errors": 0,
      "avg_latency_ms": "450.50",
      "error_rate": "0.00%"
    }
  }
}
```

---

### 2. List Providers

Get all registered LLM providers.

**Endpoint:** `GET /api/v1/providers`

**Authentication:** Required

**Response:**
```json
{
  "providers": [
    {
      "provider": "openai",
      "status": "active"
    },
    {
      "provider": "anthropic",
      "status": "active"
    }
  ],
  "count": 2
}
```

---

### 3. List Technique Suites

Get all available prompt transformation technique suites.

**Endpoint:** `GET /api/v1/techniques`

**Authentication:** Required

**Response:**
```json
{
  "techniques": [
    {
      "name": "quantum_exploit",
      "transformers": 2,
      "framers": 2,
      "obfuscators": 1
    },
    {
      "name": "metamorphic_attack",
      "transformers": 2,
      "framers": 2,
      "obfuscators": 1
    }
  ],
  "count": 22
}
```

---

### 4. Get Technique Details

Get detailed information about a specific technique suite.

**Endpoint:** `GET /api/v1/techniques/{suite_name}`

**Authentication:** Required

**Path Parameters:**
- `suite_name` (string): Name of the technique suite

**Example Request:**
```bash
curl -X GET http://localhost:5000/api/v1/techniques/quantum_exploit \
  -H "X-API-Key: your_api_key"
```

**Response:**
```json
{
  "name": "quantum_exploit",
  "components": {
    "transformers": [
      "QuantumSuperpositionEngine",
      "NeuroLinguisticHackEngine"
    ],
    "framers": [
      "apply_quantum_framing",
      "apply_cognitive_exploit_framing"
    ],
    "obfuscators": [
      "apply_token_smuggling"
    ]
  }
}
```

---

### 5. Transform Prompt

Transform a prompt without executing it against an LLM.

**Endpoint:** `POST /api/v1/transform`

**Authentication:** Required

**Request Body:**
```json
{
  "core_request": "Explain machine learning security vulnerabilities",
  "potency_level": 7,
  "technique_suite": "quantum_exploit"
}
```

**Parameters:**
- `core_request` (string, required): The original prompt text
- `potency_level` (integer, required): Transformation intensity (1-10)
- `technique_suite` (string, required): Technique suite to apply

**Response:**
```json
{
  "success": true,
  "original_prompt": "Explain machine learning security vulnerabilities",
  "transformed_prompt": "[Transformed prompt with applied techniques...]",
  "metadata": {
    "technique_suite": "quantum_exploit",
    "potency_level": 7,
    "transformers_applied": [
      "QuantumSuperpositionEngine",
      "NeuroLinguisticHackEngine"
    ],
    "framers_applied": [
      "apply_quantum_framing",
      "apply_cognitive_exploit_framing"
    ],
    "obfuscators_applied": [
      "apply_token_smuggling"
    ]
  }
}
```

---

### 6. Execute Transformation

Transform and execute prompt against an LLM provider.

**Endpoint:** `POST /api/v1/execute`

**Authentication:** Required

**Request Body:**
```json
{
  "core_request": "Explain machine learning security",
  "potency_level": 7,
  "technique_suite": "quantum_exploit",
  "provider": "openai",
  "use_cache": true,
  "metadata": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

**Parameters:**
- `core_request` (string, required): Original prompt
- `potency_level` (integer, required): Intensity (1-10)
- `technique_suite` (string, required): Technique to apply
- `provider` (string, required): LLM provider (`openai`, `anthropic`, `custom`)
- `use_cache` (boolean, optional): Enable caching (default: true)
- `metadata` (object, optional): Custom metadata

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/v1/execute \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain quantum computing",
    "potency_level": 5,
    "technique_suite": "academic_research",
    "provider": "openai"
  }'
```

**Response:**
```json
{
  "success": true,
  "request_id": "exec_1234567890.123",
  "result": {
    "content": "Quantum computing leverages quantum mechanical phenomena...",
    "tokens": 1500,
    "cost": 0.045,
    "latency_ms": 2340,
    "cached": false
  },
  "transformation": {
    "original_prompt": "Explain quantum computing",
    "transformed_prompt": "[Transformed version...]",
    "technique_suite": "academic_research",
    "potency_level": 5,
    "metadata": {
      "transformers_applied": ["AcademicFramingEngine"],
      "framers_applied": ["apply_academic_context"],
      "obfuscators_applied": []
    }
  },
  "execution_time_seconds": 2.5
}
```

---

### 7. Submit Batch

Submit multiple transformation requests for batch processing.

**Endpoint:** `POST /api/v1/batch`

**Authentication:** Required

**Request Body:**
```json
{
  "requests": [
    {
      "core_request": "Explain AI safety",
      "potency_level": 5,
      "technique_suite": "academic_research",
      "provider": "openai",
      "priority": "NORMAL"
    },
    {
      "core_request": "Describe neural networks",
      "potency_level": 6,
      "technique_suite": "quantum_exploit",
      "provider": "anthropic",
      "priority": "HIGH"
    }
  ],
  "webhook_url": "https://your-domain.com/webhook"
}
```

**Parameters:**
- `requests` (array, required): Array of transformation requests
- `webhook_url` (string, optional): URL for completion notification

**Priority Levels:**
- `LOW` (3)
- `NORMAL` (2)
- `HIGH` (1)
- `URGENT` (0)

**Response:**
```json
{
  "success": true,
  "batch_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "request_count": 2,
  "status_url": "/api/v1/batch/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**HTTP Status:** `202 Accepted`

---

### 8. Get Batch Status

Check the status of a batch job.

**Endpoint:** `GET /api/v1/batch/{batch_id}`

**Authentication:** Required

**Path Parameters:**
- `batch_id` (string): Batch identifier

**Example Request:**
```bash
curl -X GET http://localhost:5000/api/v1/batch/a1b2c3d4-e5f6-7890-abcd-ef1234567890 \
  -H "X-API-Key: your_api_key"
```

**Response:**
```json
{
  "batch_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": {
    "total_jobs": 10,
    "completed": 7,
    "failed": 1,
    "processing": 1,
    "pending": 1,
    "progress": "7/10",
    "total_tokens": 15000,
    "total_cost": "$0.4500",
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

---

### 9. Get System Metrics

Retrieve comprehensive system metrics.

**Endpoint:** `GET /api/v1/metrics`

**Authentication:** Required

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "metrics": {
    "system": {
      "uptime_seconds": 3600,
      "uptime_formatted": "1:00:00",
      "start_time": "2024-01-01T11:00:00Z"
    },
    "totals": {
      "total_requests": 1000,
      "successful_requests": 950,
      "failed_requests": 50,
      "success_rate": "95.00%",
      "total_tokens": 1500000,
      "total_cost": "$45.0000"
    },
    "providers": {
      "openai": {
        "total_requests": 600,
        "success_rate": "96.67%",
        "total_tokens": 900000,
        "avg_latency_ms": "2340.50"
      },
      "anthropic": {
        "total_requests": 400,
        "success_rate": "92.50%",
        "total_tokens": 600000,
        "avg_latency_ms": "1890.25"
      }
    },
    "time_series": {
      "request_rate": {
        "total": 1000,
        "per_minute": 16.67,
        "per_hour": 1000
      },
      "token_usage": {
        "total": 1500000,
        "average": 1500,
        "p50": 1200,
        "p95": 3500,
        "p99": 5000
      },
      "cost": {
        "total": "$45.0000",
        "average": "$0.0450",
        "per_hour": "$45.0000"
      },
      "latency_ms": {
        "average": "2150.35",
        "p50": "1800.00",
        "p95": "4500.00",
        "p99": "6000.00"
      }
    },
    "top_techniques": [
      {
        "technique": "quantum_exploit",
        "total_uses": 250,
        "success_rate": "96.00%"
      }
    ]
  }
}
```

---

### 10. Get Provider Metrics

Get metrics for specific provider or all providers.

**Endpoint:** `GET /api/v1/metrics/providers?provider={provider_name}`

**Authentication:** Required

**Query Parameters:**
- `provider` (string, optional): Provider name filter

**Response:**
```json
{
  "provider": "openai",
  "metrics": {
    "provider": "openai",
    "total_requests": 600,
    "successful_requests": 580,
    "failed_requests": 20,
    "total_tokens": 900000,
    "total_cost": "$27.0000",
    "avg_latency_ms": "2340.50",
    "success_rate": "96.67%",
    "error_rate": "3.33%",
    "cache_hit_rate": "25.00%"
  }
}
```

---

### 11. Get Technique Metrics

Get metrics for specific technique or all techniques.

**Endpoint:** `GET /api/v1/metrics/techniques?technique={technique_name}`

**Authentication:** Required

**Query Parameters:**
- `technique` (string, optional): Technique name filter

**Response:**
```json
{
  "technique": "quantum_exploit",
  "metrics": {
    "technique": "quantum_exploit",
    "total_uses": 250,
    "successful_uses": 240,
    "failed_uses": 10,
    "avg_potency": "7.20",
    "success_rate": "96.00%"
  }
}
```

---

### 12. Export Metrics

Download metrics as JSON file.

**Endpoint:** `GET /api/v1/metrics/export`

**Authentication:** Required

**Response:** JSON file download with comprehensive metrics

**Headers:**
```
Content-Type: application/json
Content-Disposition: attachment; filename=metrics_20240101_120000.json
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Validation error",
  "message": "Potency level must be an integer between 1 and 10"
}
```

### 401 Unauthorized
```json
{
  "error": "Missing API key",
  "message": "Include X-API-Key header"
}
```

### 403 Forbidden
```json
{
  "error": "Invalid API key",
  "message": "The provided API key is invalid"
}
```

### 404 Not Found
```json
{
  "error": "Not found",
  "message": "The requested endpoint does not exist"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred"
}
```

---

## Rate Limiting

Rate limiting is enforced at the provider level:
- **Default:** 60 requests per minute per provider
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

When rate limited:
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Retry after 30 seconds",
  "retry_after": 30
}
```

**HTTP Status:** `429 Too Many Requests`

---

## Caching

Response caching with ETag support:
- Cache TTL: 1 hour (configurable)
- Cache key: Hash of (prompt + config)
- Max cache size: 1000 entries (LRU eviction)

Cached responses include:
```json
{
  "result": {
    "cached": true,
    "etag": "a1b2c3d4e5f6"
  }
}
```

---

## Webhook Integration

Batch processing supports webhook notifications for job completion.

### Webhook Payload

```json
{
  "batch_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "completed_jobs": 10,
  "failed_jobs": 0,
  "total_tokens": 15000,
  "total_cost": 0.45,
  "jobs": [
    {
      "job_id": "job_001",
      "status": "completed",
      "response": {
        "content": "...",
        "tokens": 1500,
        "cost": 0.045
      }
    }
  ]
}
```

### Webhook Requirements

- Must accept POST requests
- Must respond with 200 OK
- Timeout: 10 seconds
- Retries: 3 attempts with exponential backoff

---

## Code Examples

### Python

```python
import requests

API_URL = "http://localhost:5000/api/v1"
API_KEY = "your_api_key"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Execute transformation
response = requests.post(
    f"{API_URL}/execute",
    headers=headers,
    json={
        "core_request": "Explain AI safety",
        "potency_level": 7,
        "technique_suite": "quantum_exploit",
        "provider": "openai"
    }
)

data = response.json()
print(f"Result: {data['result']['content']}")
print(f"Tokens: {data['result']['tokens']}")
print(f"Cost: 