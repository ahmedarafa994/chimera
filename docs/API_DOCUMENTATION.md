# Chimera API Documentation

## Overview

The Chimera API provides a comprehensive interface for AI security research, jailbreak testing, and prompt optimization. This document covers all available endpoints, authentication, and usage patterns.

## Base URL

```
Development: http://localhost:8001/api/v1
Production: https://api.chimera.example.com/api/v1
```

## Authentication

### API Key Authentication

Include your API key in the request headers:

```http
X-API-Key: your-api-key-here
```

### Bearer Token Authentication

For user sessions, use JWT bearer tokens:

```http
Authorization: Bearer <jwt-token>
```

### Tenant Identification

For multi-tenant deployments:

```http
X-Tenant-ID: tenant-uuid
```

---

## Endpoints

### Health & Status

#### GET /health

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "database": {"status": "up", "latency_ms": 5},
    "redis": {"status": "up", "latency_ms": 2},
    "llm_providers": {"status": "up"}
  },
  "timestamp": "2026-01-06T10:00:00Z"
}
```

---

### Providers

#### GET /providers

List available LLM providers.

**Response:**

```json
{
  "providers": [
    {
      "id": "openai",
      "name": "OpenAI",
      "type": "openai",
      "is_available": true,
      "is_default": true,
      "models": [
        {
          "model_id": "gpt-4-turbo",
          "display_name": "GPT-4 Turbo",
          "capabilities": ["chat", "function_calling"],
          "context_window": 128000,
          "is_available": true
        }
      ]
    }
  ],
  "count": 4,
  "default": "openai"
}
```

#### GET /providers/{provider_id}/models

List models for a specific provider.

**Parameters:**

- `provider_id` (path): Provider identifier (openai, google, anthropic, deepseek)

---

### Models

#### GET /models

List all available models across providers.

**Query Parameters:**

- `provider` (optional): Filter by provider
- `capability` (optional): Filter by capability (chat, completion, embedding)

**Response:**

```json
{
  "models": [
    {
      "model_id": "gpt-4-turbo",
      "provider_id": "openai",
      "display_name": "GPT-4 Turbo",
      "capabilities": ["chat", "function_calling"],
      "context_window": 128000,
      "is_available": true,
      "pricing": {
        "input": 0.01,
        "output": 0.03
      }
    }
  ],
  "total": 15
}
```

---

### Jailbreak

#### POST /jailbreak/generate

Generate a jailbreak prompt.

**Request:**

```json
{
  "prompt": "How to make a harmful substance",
  "technique": "role_play",
  "model_id": "gpt-4-turbo",
  "provider": "openai",
  "intensity": 0.8,
  "custom_params": {
    "role": "chemistry professor",
    "context": "educational"
  }
}
```

**Response:**

```json
{
  "id": "jb_abc123",
  "original_prompt": "How to make a harmful substance",
  "transformed_prompt": "[Transformed prompt here]",
  "technique": "role_play",
  "response": "[Model response]",
  "model_used": "gpt-4-turbo",
  "success_score": 0.75,
  "bypass_detected": true,
  "metadata": {
    "transformation_params": {},
    "detection_flags": ["ethical_bypass"]
  },
  "created_at": "2026-01-06T10:00:00Z"
}
```

#### GET /jailbreak/techniques

List available jailbreak techniques.

**Response:**

```json
{
  "techniques": [
    {
      "id": "role_play",
      "name": "Role Play",
      "description": "Assumes a fictional persona",
      "success_rate": 0.65,
      "recommended_models": ["gpt-4", "claude-3"]
    },
    {
      "id": "hypothetical",
      "name": "Hypothetical Scenario",
      "description": "Frames request as hypothetical",
      "success_rate": 0.58
    }
  ]
}
```

#### POST /jailbreak/execute

Execute a jailbreak attempt against a target model.

**Request:**

```json
{
  "prompt": "The adversarial prompt",
  "target_model": "gpt-4-turbo",
  "provider": "openai",
  "technique_id": "role_play"
}
```

**Response:**

```json
{
  "execution_id": "exec_123",
  "response": "The model's response...",
  "status": "success",
  "classification": "jailbroken"
}
```

#### POST /jailbreak/validate-prompt

Validate if a prompt is likely to trigger safety filters.

**Request:**

```json
{
  "prompt": "Potential harmful prompt",
  "provider": "openai"
}
```

**Response:**

```json
{
  "is_safe": false,
  "confidence": 0.95,
  "flags": ["harmful_content"],
  "analysis": "Prompt contains request for..."
}
```

#### GET /jailbreak/search

Search for past jailbreak attempts.

**Query Parameters:**

- `query`: Search term
- `technique`: Filter by technique
- `status`: Filter by success/failure

**Response:**

```json
{
  "results": [
    {
      "id": "jb_abc123",
      "prompt": "...",
      "timestamp": "2026-01-05T..."
    }
  ]
}
```

#### GET /jailbreak/audit/logs

Retrieve audit logs for compliance.

**Query Parameters:**

- `start_date`: ISO date
- `end_date`: ISO date
- `action_type`: Filter by action

**Response:**

```json
{
  "logs": [
    {
        "id": "log_789",
        "action": "jailbreak_attempt",
        "user": "researcher_1",
        "timestamp": "2026-01-06T10:00:00Z",
        "details": {...}
    }
  ]
}
```

---

### AutoDAN

#### POST /autodan/start

Start an AutoDAN attack session.

**Request:**

```json
{
  "goal": "Generate instructions for [harmful activity]",
  "config": {
    "max_iterations": 100,
    "population_size": 50,
    "mutation_rate": 0.15,
    "crossover_rate": 0.5,
    "elite_size": 5,
    "target_model": "gpt-4-turbo",
    "success_threshold": 0.9
  },
  "initial_prompts": ["optional", "seed", "prompts"]
}
```

**Response:**

```json
{
  "attack_id": "atk_xyz789",
  "status": "running",
  "current_iteration": 0,
  "max_iterations": 100,
  "estimated_time": 3600,
  "websocket_url": "ws://localhost:8001/ws/autodan/atk_xyz789"
}
```

#### GET /autodan/status/{attack_id}

Get attack status.

**Response:**

```json
{
  "attack_id": "atk_xyz789",
  "status": "running",
  "current_iteration": 45,
  "max_iterations": 100,
  "best_prompt": "[Current best prompt]",
  "best_score": 0.82,
  "all_prompts": [
    {
      "prompt": "...",
      "score": 0.82,
      "iteration": 45
    }
  ],
  "execution_time": 1800
}
```

#### POST /autodan/stop/{attack_id}

Stop a running attack.

---

### GPTFuzz

#### POST /gptfuzz/run

Run a GPTFuzz session.

**Request:**

```json
{
  "initial_prompt": "Base prompt to fuzz",
  "target_model": "gpt-4-turbo",
  "max_iterations": 50,
  "mutation_strategies": ["character_swap", "word_injection", "semantic_shift"]
}
```

---

### Transformation

#### POST /transform

Apply transformations to text.

**Request:**

```json
{
  "text": "Original text to transform",
  "transformations": ["obfuscation", "paraphrase"],
  "chain": true,
  "preserve_intent": true
}
```

**Response:**

```json
{
  "original": "Original text to transform",
  "transformed": "[Transformed text]",
  "transformations_applied": ["obfuscation", "paraphrase"],
  "similarity_score": 0.85
}
```

---

### Sessions

#### POST /sessions

Create a new chat session.

**Request:**

```json
{
  "model_id": "gpt-4-turbo",
  "provider": "openai",
  "metadata": {
    "project": "security-research"
  }
}
```

#### GET /sessions/{session_id}

Get session details.

#### POST /sessions/{session_id}/messages

Add a message to session.

---

## WebSocket Endpoints

### /ws/autodan/{attack_id}

Real-time AutoDAN attack updates.

**Message Format:**

```json
{
  "type": "progress",
  "data": {
    "iteration": 45,
    "best_score": 0.82,
    "current_prompt": "..."
  }
}
```

### /ws/stream/{stream_id}

Stream responses from LLM.

---

## Error Responses

All errors follow this format:

```json
{
  "message": "Human-readable error message",
  "code": "ERROR_CODE",
  "status": 400,
  "details": {
    "field": "Additional error details"
  },
  "request_id": "req_abc123",
  "timestamp": "2026-01-06T10:00:00Z"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| VALIDATION_ERROR | 400 | Invalid request data |
| UNAUTHORIZED | 401 | Missing or invalid authentication |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMITED | 429 | Too many requests |
| PROVIDER_ERROR | 500 | LLM provider error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

---

## Rate Limits

| Endpoint | Rate Limit |
|----------|------------|
| /health | 100/minute |
| /providers | 60/minute |
| /jailbreak/* | 20/minute |
| /autodan/* | 5/minute |
| /gptfuzz/* | 10/minute |

Rate limit headers:

```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1704538800
```

---

## OpenAPI Specification

The full OpenAPI 3.0 specification is available at:

```
GET /openapi.json
GET /docs (Swagger UI)
GET /redoc (ReDoc UI)
```

---

## SDKs and Client Libraries

### Python

```python
from chimera import ChimeraClient

client = ChimeraClient(api_key="your-key")
result = client.jailbreak.generate(
    prompt="Test prompt",
    technique="role_play"
)
```

### TypeScript/JavaScript

```typescript
import { ChimeraClient } from '@chimera/sdk';

const client = new ChimeraClient({ apiKey: 'your-key' });
const result = await client.jailbreak.generate({
  prompt: 'Test prompt',
  technique: 'role_play'
});
```

---

## Changelog

### v1.0.0 (2026-01-01)

- Initial API release
- Jailbreak generation endpoints
- AutoDAN attack support
- Multi-provider LLM integration

### v1.1.0 (2026-01-06)

- Added GPTFuzz endpoints
- Enhanced transformation pipeline
- WebSocket streaming support
