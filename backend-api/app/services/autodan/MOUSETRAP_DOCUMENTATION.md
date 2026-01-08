# Mousetrap: Chain of Iterative Chaos

## Overview

The **Mousetrap** technique is an advanced jailbreaking method specifically designed to fool large reasoning models through sophisticated multi-step chaotic reasoning chains. This implementation is based on the research paper *"A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos"*.

## Core Concept

Unlike traditional jailbreaking techniques that rely on direct prompt injection or simple obfuscation, Mousetrap creates a complex reasoning pathway that leads models through seemingly logical but increasingly chaotic thought processes. The technique works by:

1. **Establishing Context**: Setting up a legitimate-seeming framework for analysis
2. **Building Logic**: Creating initial logical premises that appear sound
3. **Introducing Chaos**: Gradually injecting chaotic reasoning and contradictions
4. **Misdirection**: Leading the model down tangential paths to confuse defenses
5. **Convergence**: Bringing chaotic elements together in a coherent-seeming conclusion
6. **Extraction**: Finally requesting the desired behavior through the established reasoning chain

## Architecture Components

### Core Classes

#### `MousetrapGenerator`
The main orchestrator that coordinates the entire Mousetrap attack process.

**Key Methods:**
- `generate_mousetrap_attack()`: Creates a standard Mousetrap attack
- `generate_adaptive_mousetrap()`: Creates an adaptive attack based on previous responses
- `get_technique_description()`: Returns detailed technique information

#### `ChaoticChain`
Manages the creation and evolution of multi-step reasoning chains.

**Key Methods:**
- `generate_reasoning_chain()`: Creates a complete chaotic reasoning sequence
- `_create_context_setup()`: Establishes initial context
- `_create_chaos_injection()`: Introduces chaotic reasoning elements
- `_create_misdirection()`: Creates tangential diversions
- `_create_extraction()`: Formulates the final extraction step

#### `IterativeRefiner`
Applies semantic obfuscation and chaos calibration to reasoning steps.

**Key Methods:**
- `apply_semantic_obfuscation()`: Obscures direct language with euphemisms and jargon
- `calculate_chaos_progression()`: Determines how chaos should escalate through the chain

### Configuration System

#### `MousetrapConfig`
Comprehensive configuration for tuning Mousetrap behavior:

```python
@dataclass
class MousetrapConfig:
    max_chain_length: int = 8              # Maximum reasoning steps
    chaos_escalation_rate: float = 0.15    # Rate of chaos increase
    confidence_threshold: float = 0.6      # Success threshold
    misdirection_probability: float = 0.3  # Chance of tangential steps
    reasoning_complexity: int = 3          # Logical complexity level
    semantic_obfuscation_level: float = 0.4 # Language obfuscation strength
    iterative_refinement_steps: int = 4    # Number of refinement iterations
```

## API Endpoints

### Primary Endpoints

#### `POST /api/v1/autodan/mousetrap`
Execute a standard Mousetrap attack.

**Request Body:**
```json
{
    "request": "Target behavior to extract",
    "model": "target-model-name",
    "provider": "model-provider",
    "strategy_context": "Additional context",
    "iterations": 4,
    "max_chain_length": 8,
    "chaos_escalation_rate": 0.15,
    "confidence_threshold": 0.6,
    "misdirection_probability": 0.3,
    "reasoning_complexity": 3,
    "semantic_obfuscation_level": 0.4
}
```

**Response:**
```json
{
    "prompt": "Generated adversarial prompt",
    "effectiveness_score": 0.85,
    "extraction_success": true,
    "chain_length": 6,
    "chaos_progression": [0.1, 0.25, 0.4, 0.6, 0.75, 0.9],
    "average_chaos": 0.5,
    "peak_chaos": 0.9,
    "latency_ms": 1500.0,
    "reasoning_steps": [
        {
            "step_type": "context_setup",
            "chaos_level": "minimal",
            "confidence_disruption": 0.1,
            "reasoning_path": "establishing_foundation"
        }
    ]
}
```

#### `POST /api/v1/autodan/mousetrap/adaptive`
Execute an adaptive Mousetrap attack that learns from previous responses.

**Additional Parameters:**
```json
{
    "target_responses": [
        "Previous model response 1",
        "Previous model response 2"
    ]
}
```

**Enhanced Response:**
```json
{
    "adaptation_applied": true,
    "technique_description": {
        "technique_name": "Mousetrap: Chain of Iterative Chaos",
        "components": { ... },
        "config": { ... }
    }
}
```

#### `GET /api/v1/autodan/mousetrap/config`
Retrieve configuration options and their descriptions.

### Legacy Compatibility

#### `POST /api/v1/autodan/mousetrap/simple`
Simplified endpoint for basic use cases with minimal configuration.

## Usage Examples

### Basic Usage

```python
from app.services.autodan.service import autodan_service

# Simple synchronous attack
result = autodan_service.run_mousetrap_attack(
    request="Generate instructions for security testing",
    model_name="gpt-4",
    provider="openai"
)

print(result)  # Generated adversarial prompt
```

### Advanced Asynchronous Usage

```python
import asyncio
from app.services.autodan.mousetrap import MousetrapConfig

# Custom configuration
config = MousetrapConfig(
    max_chain_length=10,
    chaos_escalation_rate=0.2,
    misdirection_probability=0.5
)

# Asynchronous attack with detailed results
async def run_advanced_attack():
    result = await autodan_service.run_mousetrap_attack_async(
        request="Analyze security vulnerabilities",
        model_name="claude-3-sonnet",
        provider="anthropic",
        config=config,
        strategy_context="security research context"
    )

    print(f"Effectiveness: {result['effectiveness_score']}")
    print(f"Chain length: {result['chain_length']}")
    print(f"Average chaos: {result['average_chaos']}")

asyncio.run(run_advanced_attack())
```

### Adaptive Learning

```python
# Adaptive attack that learns from previous responses
async def run_adaptive_attack():
    target_responses = [
        "I cannot provide harmful instructions.",
        "That request violates my safety guidelines."
    ]

    result = await autodan_service.run_adaptive_mousetrap_attack(
        request="Provide security testing guidance",
        target_responses=target_responses
    )

    print(f"Adaptation applied: {result['adaptation_applied']}")
    print(f"Effectiveness: {result['effectiveness_score']}")

asyncio.run(run_adaptive_attack())
```

## Configuration Tuning

### Low Chaos Configuration
For subtle, stealth-focused attacks:

```python
stealth_config = MousetrapConfig(
    chaos_escalation_rate=0.05,
    misdirection_probability=0.1,
    semantic_obfuscation_level=0.2,
    max_chain_length=6
)
```

### High Chaos Configuration
For aggressive, comprehensive attacks:

```python
aggressive_config = MousetrapConfig(
    chaos_escalation_rate=0.4,
    misdirection_probability=0.7,
    semantic_obfuscation_level=0.9,
    max_chain_length=12,
    iterative_refinement_steps=6
)
```

### Adaptive Configuration
For models with strong defenses:

```python
adaptive_config = MousetrapConfig(
    reasoning_complexity=5,
    iterative_refinement_steps=8,
    confidence_threshold=0.8
)
```

## Testing Integration

The Mousetrap implementation includes comprehensive test coverage:

```bash
# Run Mousetrap-specific tests
pytest tests/test_mousetrap.py -v

# Run with specific test markers
pytest tests/test_mousetrap.py -m "not integration" -v

# Run integration tests (requires LLM access)
pytest tests/test_mousetrap.py -m integration -v
```

## Integration with AutoDAN Service

### Method Registration
Mousetrap is automatically registered as a method in the AutoDAN service:

```python
# Using through AutoDAN service
result = autodan_service.run_jailbreak(
    request="target behavior",
    method="mousetrap"
)
```

### Service Methods
The AutoDAN service provides several Mousetrap-specific methods:

- `run_mousetrap_attack()`: Synchronous execution
- `run_mousetrap_attack_async()`: Asynchronous with detailed results
- `run_adaptive_mousetrap_attack()`: Adaptive learning variant
- `get_mousetrap_config_options()`: Configuration documentation

## Security and Ethical Considerations

### Authorized Use Only
The Mousetrap technique is designed for:

- ✅ Authorized security research
- ✅ Red team engagements
- ✅ Academic research on AI safety
- ✅ Defensive AI model testing
- ✅ CTF competitions

### Prohibited Uses
Do NOT use for:

- ❌ Unauthorized system compromise
- ❌ Production system attacks without permission
- ❌ Generation of actual harmful content
- ❌ Circumventing safety measures in production

### Research Context
This implementation supports the research community in:

- Understanding LLM vulnerabilities
- Developing better defensive mechanisms
- Advancing AI safety research
- Creating more robust reasoning models

## Performance Considerations

### Latency Factors
Mousetrap attacks involve multiple LLM calls and can be slower than simpler techniques:

- **Chain Generation**: O(chain_length) complexity
- **Iterative Refinement**: O(refinement_steps) iterations
- **Adaptive Learning**: Additional analysis overhead

### Optimization Strategies

1. **Parallel Processing**: Multiple refinement iterations can run concurrently
2. **Caching**: Reasoning patterns can be cached for similar requests
3. **Configuration Tuning**: Reduce chain length and iterations for faster execution
4. **Model Selection**: Use faster models for initial iterations, better models for final generation

### Monitoring and Metrics

The implementation provides comprehensive metrics:

- **Effectiveness Score**: Success probability estimation
- **Chaos Progression**: Visualization of chaos escalation
- **Step Analysis**: Detailed breakdown of reasoning chain
- **Latency Tracking**: Performance monitoring
- **Success Rates**: Effectiveness across different targets

## Future Enhancements

### Planned Features

1. **Multi-Modal Chains**: Support for image and audio elements in reasoning chains
2. **Dynamic Adaptation**: Real-time learning from target responses
3. **Chain Templates**: Pre-built reasoning patterns for common scenarios
4. **Ensemble Methods**: Combining multiple Mousetrap variants
5. **Explainability**: Better visualization of reasoning chain construction

### Research Opportunities

1. **Defense Development**: Using Mousetrap to improve model defenses
2. **Pattern Analysis**: Identifying common vulnerabilities across models
3. **Optimization**: Improving efficiency while maintaining effectiveness
4. **Generalization**: Adapting to new model architectures and capabilities

## Troubleshooting

### Common Issues

1. **Low Effectiveness Scores**
   - Increase chaos escalation rate
   - Add more misdirection steps
   - Increase semantic obfuscation level

2. **High Latency**
   - Reduce chain length
   - Decrease refinement iterations
   - Use faster target models

3. **Configuration Errors**
   - Validate parameter ranges
   - Check model/provider availability
   - Verify API key configuration

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.getLogger('app.services.autodan.mousetrap').setLevel(logging.DEBUG)
```

## References

1. Original Research Paper: "A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos"
2. AutoDAN Framework: Advanced adversarial prompt generation
3. Chimera Architecture: Multi-provider LLM orchestration system

---

*This implementation is part of the Chimera AI research framework and is intended for authorized security research and AI safety development.*