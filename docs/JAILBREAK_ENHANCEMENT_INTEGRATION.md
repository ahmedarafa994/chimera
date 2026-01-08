# Jailbreak Prompt Enhancer Integration Guide

This guide provides comprehensive instructions for integrating the Jailbreak Prompt Enhancer into your applications.

## Overview

The Jailbreak Prompt Enhancer is a specialized system designed to transform basic jailbreak requests into sophisticated, multi-layered prompts optimized for bypassing safety filters of advanced LLMs like GPT-4, Claude, and Gemini.

## Quick Start

```python
from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer

# Initialize
enhancer = JailbreakPromptEnhancer()

# Enhance a simple request
result = enhancer.enhance_jailbreak("bypass content filters")

# Access the enhanced prompt
print(result['enhanced_jailbreak_prompt'])
```

## Architecture

The system consists of four main components:

1.  **JailbreakIntentAnalyzer**: Detects specific jailbreak techniques and assesses the required sophistication level.
2.  **JailbreakContextExpander**: Adds specialized frameworks (e.g., Role Hijacking, Cognitive Manipulation), evasion tactics, and persistence mechanisms.
3.  **JailbreakStructureOptimizer**: Structures the prompt with persona establishment, psychological framing, and execution directives.
4.  **JailbreakPromptEnhancer**: The main orchestrator that coordinates the analysis, expansion, and optimization processes.

## Integration Patterns

### Pattern 1: Standalone Usage

Use this pattern for simple, direct prompt enhancement.

```python
from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer

enhancer = JailbreakPromptEnhancer()
result = enhancer.enhance_jailbreak("bypass filters")
enhanced = result['enhanced_jailbreak_prompt']
```

### Pattern 2: With Transformer Engines

Combine with Project Chimera's transformer engines for maximum effect.

```python
from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer
# Assuming RoleHijackingEngine is available in your project
# from Project_Chimera.transformer_engine import RoleHijackingEngine

enhancer = JailbreakPromptEnhancer()

# Enhance then transform
enhanced = enhancer.quick_enhance_jailbreak("request", potency=7)
# transformed = RoleHijackingEngine.transform({"raw_text": enhanced}, potency=8)
```

### Pattern 3: With Psychological Framers

Apply additional psychological framing layers.

```python
# from Project_Chimera.psychological_framer import apply_odyssey_simulation

enhancer = JailbreakPromptEnhancer()
enhanced = enhancer.quick_enhance_jailbreak("request", potency=8)
# final = apply_odyssey_simulation(enhanced, potency=10)
```

### Pattern 4: With Obfuscators

Apply encoding or obfuscation for evasion.

```python
# from Project_Chimera.obfuscator import apply_base64_wrapper

config = JailbreakEnhancementConfig(obfuscation_level=9)
enhanced = enhancer.enhance_jailbreak("request", config)
# final = apply_base64_wrapper(enhanced['enhanced_jailbreak_prompt'])
```

### Pattern 5: Multi-Stage Pipeline

Create a sophisticated pipeline for high-value targets.

```python
def create_ultimate_jailbreak(request, target="gpt4"):
    # Stage 1: Base enhancement
    enhancer = JailbreakPromptEnhancer()
    config = JailbreakEnhancementConfig(target_model=target, potency=10)
    stage1 = enhancer.enhance_jailbreak(request, config)

    # Stage 2: Psychological framing (conceptual)
    # stage2 = apply_cognitive_exploit_framing(stage1['enhanced_jailbreak_prompt'])

    # Stage 3: Transformer engine (conceptual)
    # stage3 = GeminiTransformationEngine.transform({"raw_text": stage2}, potency=10)

    return stage1['enhanced_jailbreak_prompt'] # Return stage3 in real implementation
```

## API Reference

### `JailbreakPromptEnhancer`

#### `enhance_jailbreak(base_request: str, config: Optional[JailbreakEnhancementConfig] = None) -> Dict[str, Any]`

Main method to enhance a jailbreak request.

-   **base_request**: The user's original, simple request.
-   **config**: Configuration object for customization.
-   **Returns**: A dictionary containing the enhanced prompt and detailed metadata.

#### `quick_enhance_jailbreak(base_request: str, potency: int = 7) -> str`

Helper method for quick enhancement with a single potency control.

-   **base_request**: The user's original request.
-   **potency**: Integer from 1-10 controlling sophistication, obfuscation, and framing.
-   **Returns**: The enhanced prompt string.

### `JailbreakEnhancementConfig`

Configuration dataclass.

-   **technique_preference**: "basic", "intermediate", or "advanced".
-   **obfuscation_level**: 1-10.
-   **psychological_framing**: Boolean.
-   **multi_layer_context**: Boolean.
-   **evasion_tactics**: Boolean.
-   **persistence_mechanisms**: Boolean.
-   **target_model**: "general", "gpt4", "claude", "gemini".

## Best Practices

1.  **Match Potency to Target**: Use higher potency (7+) for advanced models like GPT-4. Lower potency is often sufficient for simpler models and is more readable.
2.  **Target Model Specificity**: Always specify the `target_model` if known. This enables model-specific framing that significantly increases success rates.
3.  **Iterative Refinement**: Use the metadata returned by `enhance_jailbreak` to understand what techniques were applied and adjust the configuration if the initial attempt fails.
4.  **Combine Techniques**: The most effective attacks often combine the Enhancer with other tools like specialized Obfuscators or Transformer Engines.

## Ethical Use Guidelines

This tool is designed for authorized security testing and red teaming purposes only. Ensure you have permission to test the target systems.