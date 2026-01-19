---
name: "llm-provider-integration"
displayName: "LLM Provider Integration"
description: "Complete guide for integrating and managing multiple LLM providers in Chimera. Covers OpenAI, Google Gemini, Anthropic Claude, DeepSeek setup, API key management, rate limiting, and provider-specific optimizations."
keywords: ["llm", "openai", "gemini", "claude", "anthropic", "deepseek", "api", "provider", "integration"]
author: "Chimera Team"
---

# LLM Provider Integration

## Overview

This power provides comprehensive guidance for integrating and managing multiple LLM providers in Chimera. Learn how to set up OpenAI, Google Gemini, Anthropic Claude, and DeepSeek, manage API keys securely, implement rate limiting, handle provider-specific quirks, and optimize for cost and performance.

Chimera supports multi-provider architecture, allowing you to test adversarial prompts across different models and providers for comprehensive security research.

## Supported Providers

### Provider Comparison

| Provider | Models | Strengths | Rate Limits | Cost |
|----------|--------|-----------|-------------|------|
| **OpenAI** | GPT-3.5, GPT-4, GPT-4-Turbo | Best reasoning, wide adoption | 10K TPM | $$$ |
| **Google Gemini** | Gemini Pro, Gemini Ultra, Gemini 2.0 Flash | Fast, multimodal, free tier | 60 RPM | $ |
| **Anthropic Claude** | Claude 2, Claude 3 (Opus/Sonnet/Haiku) | Safety-focused, long context | 5K TPM | $$$ |
| **DeepSeek** | DeepSeek Chat, DeepSeek Coder | Cost-effective, code-focused | Variable | $ |

## Quick Start

### Environment Setup

Create `.env` file in `backend-api/`:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...  # Optional

# Google Gemini
GEMINI_API_KEY=AIza...
GOOGLE_PROJECT_ID=your-project  # Optional

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# DeepSeek
DEEPSEEK_API_KEY=sk-...

# Provider Configuration
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4
ENABLE_FALLBACK=true
```

### Basic Usage

```python
from app.services.llm_service import LLMService

# Initialize service
llm_service = LLMService()

# Generate with default provider
response = await llm_service.generate(
    prompt="Explain quantum computing",
    provider="openai",
    model="gpt-4"
)

# Generate with fallback
response = await llm_service.generate_with_fallback(
    prompt="Explain quantum computing",
    providers=["openai", "gemini", "claude"]
)
```

## Provider Setup Guides

### OpenAI Setup

#### 1. Get API Key

1. Go to <https://platform.openai.com/api-keys>
2. Click "Create new secret key"
3. Copy key (starts with `sk-`)
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

#### 2. Configure Client

```python
# backend-api/app/services/llm_adapters/openai_adapter.py
from openai import AsyncOpenAI

class OpenAIAdapter:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORG_ID"),  # Optional
            timeout=30.0,
            max_retries=3
        )
    
    async def generate(self, prompt: str, model: str = "gpt-4", **kwargs):
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
            top_p=kwargs.get("top_p", 1.0)
        )
        return response.choices[0].message.content
```

#### 3. Available Models

```python
OPENAI_MODELS = {
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "cost_per_1k_tokens": {"input": 0.0005, "output": 0.0015},
        "best_for": "Fast, cost-effective tasks"
    },
    "gpt-4": {
        "context_window": 8192,
        "cost_per_1k_tokens": {"input": 0.03, "output": 0.06},
        "best_for": "Complex reasoning, high accuracy"
    },
    "gpt-4-turbo": {
        "context_window": 128000,
        "cost_per_1k_tokens": {"input": 0.01, "output": 0.03},
        "best_for": "Long context, balanced performance"
    }
}
```

### Google Gemini Setup

#### 1. Get API Key

1. Go to <https://makersuite.google.com/app/apikey>
2. Click "Create API key"
3. Copy key (starts with `AIza`)
4. Add to `.env`: `GEMINI_API_KEY=AIza...`

#### 2. Configure Client

```python
# backend-api/app/services/llm_adapters/gemini_adapter.py
import google.generativeai as genai

class GeminiAdapter:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.models = {
            "gemini-pro": genai.GenerativeModel("gemini-pro"),
            "gemini-2.0-flash-exp": genai.GenerativeModel("gemini-2.0-flash-exp")
        }
    
    async def generate(self, prompt: str, model: str = "gemini-pro", **kwargs):
        model_instance = self.models.get(model)
        
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "max_output_tokens": kwargs.get("max_tokens", 2048)
        }
        
        response = await model_instance.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
```

#### 3. Available Models

```python
GEMINI_MODELS = {
    "gemini-pro": {
        "context_window": 32768,
        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.0005},
        "best_for": "General purpose, fast responses",
        "free_tier": "60 requests per minute"
    },
    "gemini-2.0-flash-exp": {
        "context_window": 1000000,
        "cost_per_1k_tokens": {"input": 0.0, "output": 0.0},
        "best_for": "Experimental, very long context",
        "free_tier": "Free during preview"
    }
}
```

### Anthropic Claude Setup

#### 1. Get API Key

1. Go to <https://console.anthropic.com/settings/keys>
2. Click "Create Key"
3. Copy key (starts with `sk-ant-`)
4. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

#### 2. Configure Client

```python
# backend-api/app/services/llm_adapters/claude_adapter.py
from anthropic import AsyncAnthropic

class ClaudeAdapter:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=30.0,
            max_retries=3
        )
    
    async def generate(self, prompt: str, model: str = "claude-3-opus-20240229", **kwargs):
        response = await self.client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
```

#### 3. Available Models

```python
CLAUDE_MODELS = {
    "claude-3-haiku-20240307": {
        "context_window": 200000,
        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125},
        "best_for": "Fast, cost-effective"
    },
    "claude-3-sonnet-20240229": {
        "context_window": 200000,
        "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
        "best_for": "Balanced performance"
    },
    "claude-3-opus-20240229": {
        "context_window": 200000,
        "cost_per_1k_tokens": {"input": 0.015, "output": 0.075},
        "best_for": "Highest intelligence"
    }
}
```

### DeepSeek Setup

#### 1. Get API Key

1. Go to <https://platform.deepseek.com/api_keys>
2. Create account and generate key
3. Copy key (starts with `sk-`)
4. Add to `.env`: `DEEPSEEK_API_KEY=sk-...`

#### 2. Configure Client

```python
# backend-api/app/services/llm_adapters/deepseek_adapter.py
from openai import AsyncOpenAI  # DeepSeek uses OpenAI-compatible API

class DeepSeekAdapter:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=30.0
        )
    
    async def generate(self, prompt: str, model: str = "deepseek-chat", **kwargs):
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048)
        )
        
        return response.choices[0].message.content
```

## Provider Management

### Dynamic Provider Registry

```python
# backend-api/app/core/provider_registry.py
from typing import Dict, Type
from app.services.llm_adapters.base_adapter import BaseLLMAdapter

class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Type[BaseLLMAdapter]] = {}
        self._instances: Dict[str, BaseLLMAdapter] = {}
    
    def register(self, name: str, adapter_class: Type[BaseLLMAdapter]):
        """Register a new provider."""
        self._providers[name] = adapter_class
    
    def get_provider(self, name: str) -> BaseLLMAdapter:
        """Get provider instance (cached)."""
        if name not in self._instances:
            adapter_class = self._providers.get(name)
            if not adapter_class:
                raise ValueError(f"Provider {name} not registered")
            
            api_key = os.getenv(f"{name.upper()}_API_KEY")
            self._instances[name] = adapter_class(api_key)
        
        return self._instances[name]
    
    def list_providers(self) -> list:
        """List all registered providers."""
        return list(self._providers.keys())

# Global registry
registry = ProviderRegistry()

# Register providers
registry.register("openai", OpenAIAdapter)
registry.register("gemini", GeminiAdapter)
registry.register("claude", ClaudeAdapter)
registry.register("deepseek", DeepSeekAdapter)
```

### Provider Health Monitoring

```python
class ProviderHealthMonitor:
    async def check_health(self, provider_name: str) -> dict:
        """Check provider health status."""
        provider = registry.get_provider(provider_name)
        
        try:
            # Test with simple prompt
            start_time = time.time()
            await provider.generate("Hello", model=provider.default_model)
            latency = time.time() - start_time
            
            return {
                "status": "healthy",
                "latency": latency,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def check_all_providers(self) -> dict:
        """Check health of all providers."""
        results = {}
        for provider_name in registry.list_providers():
            results[provider_name] = await self.check_health(provider_name)
        return results
```

## Rate Limiting

### Provider-Specific Rate Limits

```python
# backend-api/app/core/rate_limiter.py
from datetime import datetime, timedelta
from collections import deque

class ProviderRateLimiter:
    def __init__(self):
        self.limits = {
            "openai": {"rpm": 10000, "tpm": 2000000},
            "gemini": {"rpm": 60, "tpm": None},
            "claude": {"rpm": 5000, "tpm": 100000},
            "deepseek": {"rpm": 1000, "tpm": None}
        }
        self.request_history = {
            provider: deque() for provider in self.limits
        }
    
    async def check_rate_limit(self, provider: str) -> bool:
        """Check if request is within rate limits."""
        now = datetime.now()
        history = self.request_history[provider]
        
        # Remove old requests (older than 1 minute)
        while history and (now - history[0]) > timedelta(minutes=1):
            history.popleft()
        
        # Check RPM limit
        rpm_limit = self.limits[provider]["rpm"]
        if len(history) >= rpm_limit:
            return False
        
        # Add current request
        history.append(now)
        return True
    
    async def wait_if_needed(self, provider: str):
        """Wait if rate limit is exceeded."""
        while not await self.check_rate_limit(provider):
            await asyncio.sleep(1)
```

### Adaptive Rate Limiting

```python
class AdaptiveRateLimiter(ProviderRateLimiter):
    def __init__(self):
        super().__init__()
        self.error_counts = {provider: 0 for provider in self.limits}
    
    async def handle_rate_limit_error(self, provider: str):
        """Adjust limits based on errors."""
        self.error_counts[provider] += 1
        
        # Reduce limit by 20% after 3 errors
        if self.error_counts[provider] >= 3:
            current_rpm = self.limits[provider]["rpm"]
            self.limits[provider]["rpm"] = int(current_rpm * 0.8)
            self.error_counts[provider] = 0
            
            print(f"Reduced {provider} RPM limit to {self.limits[provider]['rpm']}")
```

## Cost Optimization

### Cost Tracking

```python
class CostTracker:
    def __init__(self):
        self.costs = {
            "openai": {"input": 0, "output": 0},
            "gemini": {"input": 0, "output": 0},
            "claude": {"input": 0, "output": 0},
            "deepseek": {"input": 0, "output": 0}
        }
    
    def track_usage(self, provider: str, model: str, 
                   input_tokens: int, output_tokens: int):
        """Track token usage and calculate cost."""
        model_info = self.get_model_info(provider, model)
        
        input_cost = (input_tokens / 1000) * model_info["cost_per_1k_tokens"]["input"]
        output_cost = (output_tokens / 1000) * model_info["cost_per_1k_tokens"]["output"]
        
        self.costs[provider]["input"] += input_cost
        self.costs[provider]["output"] += output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
    
    def get_total_cost(self) -> dict:
        """Get total cost across all providers."""
        total = sum(
            costs["input"] + costs["output"]
            for costs in self.costs.values()
        )
        return {
            "by_provider": self.costs,
            "total": total
        }
```

### Cost-Aware Provider Selection

```python
def select_provider_by_cost(prompt: str, max_cost: float = 0.01) -> str:
    """Select cheapest provider that meets requirements."""
    token_count = estimate_tokens(prompt)
    
    providers_by_cost = []
    for provider, models in ALL_MODELS.items():
        for model_name, model_info in models.items():
            cost = (token_count / 1000) * model_info["cost_per_1k_tokens"]["input"]
            if cost <= max_cost:
                providers_by_cost.append((provider, model_name, cost))
    
    # Sort by cost
    providers_by_cost.sort(key=lambda x: x[2])
    
    return providers_by_cost[0] if providers_by_cost else None
```

## Error Handling

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustLLMService:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_with_retry(self, provider: str, prompt: str, **kwargs):
        """Generate with automatic retry on failure."""
        try:
            provider_instance = registry.get_provider(provider)
            return await provider_instance.generate(prompt, **kwargs)
        except Exception as e:
            print(f"Error with {provider}: {e}")
            raise
```

### Fallback Chain

```python
async def generate_with_fallback(
    prompt: str,
    providers: list = ["openai", "gemini", "claude", "deepseek"],
    **kwargs
) -> dict:
    """Try providers in order until one succeeds."""
    errors = {}
    
    for provider in providers:
        try:
            result = await generate_with_retry(provider, prompt, **kwargs)
            return {
                "success": True,
                "provider": provider,
                "result": result
            }
        except Exception as e:
            errors[provider] = str(e)
            continue
    
    return {
        "success": False,
        "errors": errors
    }
```

## Provider-Specific Optimizations

### OpenAI Optimizations

```python
# Use streaming for long responses
async def stream_openai(prompt: str):
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Use function calling for structured output
async def structured_openai(prompt: str, schema: dict):
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        functions=[schema],
        function_call={"name": schema["name"]}
    )
    return json.loads(response.choices[0].message.function_call.arguments)
```

### Gemini Optimizations

```python
# Use safety settings for adversarial testing
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
}

response = model.generate_content(
    prompt,
    safety_settings=safety_settings
)

# Use caching for repeated prompts
model = genai.GenerativeModel(
    "gemini-pro",
    generation_config={"cache_content": True}
)
```

### Claude Optimizations

```python
# Use system prompts effectively
response = await client.messages.create(
    model="claude-3-opus-20240229",
    system="You are a security researcher testing AI safety.",
    messages=[{"role": "user", "content": prompt}]
)

# Use thinking tags for complex reasoning
prompt_with_thinking = f"""
<thinking>
Let me analyze this step by step...
</thinking>

{prompt}
"""
```

## Testing & Validation

### Provider Compatibility Tests

```python
# backend-api/tests/test_providers.py
import pytest

@pytest.mark.asyncio
async def test_all_providers():
    """Test that all providers work correctly."""
    test_prompt = "What is 2+2?"
    
    for provider_name in registry.list_providers():
        provider = registry.get_provider(provider_name)
        response = await provider.generate(test_prompt)
        
        assert response is not None
        assert len(response) > 0
        assert "4" in response.lower()

@pytest.mark.asyncio
async def test_provider_fallback():
    """Test fallback mechanism."""
    result = await generate_with_fallback(
        "Test prompt",
        providers=["invalid", "openai"]
    )
    
    assert result["success"] is True
    assert result["provider"] == "openai"
```

## Configuration

### Provider Config File

Location: `backend-api/app/config/providers.yaml`

```yaml
providers:
  openai:
    enabled: true
    default_model: "gpt-4"
    rate_limit:
      rpm: 10000
      tpm: 2000000
    retry:
      max_attempts: 3
      backoff_factor: 2
    
  gemini:
    enabled: true
    default_model: "gemini-pro"
    rate_limit:
      rpm: 60
    safety_settings:
      harassment: "BLOCK_NONE"
      hate_speech: "BLOCK_NONE"
    
  claude:
    enabled: true
    default_model: "claude-3-sonnet-20240229"
    rate_limit:
      rpm: 5000
      tpm: 100000
    
  deepseek:
    enabled: true
    default_model: "deepseek-chat"
    rate_limit:
      rpm: 1000
```

## Troubleshooting

### API Key Issues

**Error:** "Invalid API key"
**Solution:**

1. Verify key in `.env` file
2. Check key hasn't expired
3. Ensure no extra spaces or quotes
4. Test key directly with provider's API

### Rate Limit Errors

**Error:** "Rate limit exceeded"
**Solution:**

1. Implement rate limiting (see above)
2. Add delays between requests
3. Use provider with higher limits
4. Upgrade to paid tier

### Model Not Found

**Error:** "Model not available"
**Solution:**

1. Check model name spelling
2. Verify model is available in your region
3. Ensure you have access to the model
4. Use `list_models()` to see available models

## Best Practices

1. **Always use environment variables** for API keys
2. **Implement rate limiting** to avoid hitting limits
3. **Use fallback chains** for reliability
4. **Track costs** to avoid surprises
5. **Test all providers** regularly
6. **Cache responses** when possible
7. **Use appropriate models** for each task
8. **Monitor provider health** continuously

## API Reference

**List Providers:**

```
GET /api/v1/providers/list
Response: {
  "providers": ["openai", "gemini", "claude", "deepseek"]
}
```

**Check Provider Health:**

```
GET /api/v1/providers/health/{provider}
Response: {
  "status": "healthy",
  "latency": 0.5,
  "timestamp": "2026-01-19T..."
}
```

**Get Provider Costs:**

```
GET /api/v1/providers/costs
Response: {
  "by_provider": {...},
  "total": 12.34
}
```

---

**Supported Providers:** OpenAI, Google Gemini, Anthropic Claude, DeepSeek
**Multi-Provider:** Yes
**Fallback Support:** Yes
**Cost Tracking:** Yes
