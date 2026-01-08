# ADR 0001: Multi-Provider LLM Architecture

## Status

Accepted

## Date

2026-01-01

## Context

The Chimera platform needs to interact with multiple Large Language Model (LLM) providers including OpenAI, Google (Gemini), Anthropic (Claude), and DeepSeek. Each provider has different APIs, authentication mechanisms, rate limits, and pricing structures.

Key requirements:
- Support 4+ LLM providers with unified interface
- Automatic failover when primary provider is unavailable
- Cost optimization through provider selection
- Handle provider-specific rate limits gracefully
- Support streaming responses from all providers

## Decision

We will implement a **Strategy Pattern** combined with **Circuit Breaker Pattern** for LLM provider management:

### Architecture Components

1. **LLM Service (Facade)**
   - Single entry point for all LLM operations
   - Routes requests to appropriate provider
   - Handles failover logic

2. **Provider Adapters (Strategy)**
   - `OpenAIAdapter`
   - `GoogleAdapter`
   - `AnthropicAdapter`
   - `DeepSeekAdapter`
   - Each implements `LLMProviderInterface`

3. **Circuit Breaker**
   - Per-provider circuit breaker
   - States: CLOSED → OPEN → HALF_OPEN
   - Configurable failure thresholds

4. **Provider Registry**
   - Dynamic provider registration
   - Health status tracking
   - Priority-based selection

### Interface Definition

```python
class LLMProviderInterface(ABC):
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        model: str, 
        **kwargs
    ) -> LLMResponse: ...
    
    @abstractmethod
    async def stream(
        self, 
        prompt: str, 
        model: str, 
        **kwargs
    ) -> AsyncGenerator[str, None]: ...
    
    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]: ...
    
    @abstractmethod
    async def check_health(self) -> bool: ...
```

### Failover Strategy

```
Primary Provider (OpenAI)
    ↓ [failure]
Secondary Provider (Google)
    ↓ [failure]
Tertiary Provider (Anthropic)
    ↓ [failure]
Fallback Provider (DeepSeek)
    ↓ [all failed]
Raise ProviderUnavailableError
```

## Consequences

### Positive

- **Flexibility**: Easy to add new providers by implementing the interface
- **Resilience**: Automatic failover ensures service continuity
- **Cost Control**: Can route to cheaper providers for non-critical requests
- **Testability**: Provider adapters can be easily mocked

### Negative

- **Complexity**: More code to maintain than single-provider solution
- **Latency**: Failover adds latency during provider outages
- **Configuration**: Requires careful configuration of priorities and thresholds

### Risks

- Provider API changes may break adapters
- Rate limit exhaustion across all providers in high-load scenarios

## Alternatives Considered

1. **Single Provider Only**: Simpler but no resilience
2. **Load Balancer**: External service but adds infrastructure complexity
3. **Message Queue**: Async processing but adds latency for real-time requests

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
