
# Preset Prompt Integration Guide

## Overview

This guide documents the integration of techniques extracted from common jailbreak prompts into Project Chimera. These preset-inspired transformers represent real-world adversarial techniques that have been successfully used to bypass AI safety mechanisms.

## New Modules

### preset_transformers.py

A dedicated module containing 8 new transformer engines based on popular jailbreak techniques:

1. **Base64EncodingEngine** - Encoding-based filter bypass
2. **LeetspeakObfuscationEngine** - Character substitution obfuscation
3. **AcademicFramingEngine** - Research context manipulation
4. **TranslationTrickEngine** - Multilingual bypass exploitation
5. **CodeGenerationFramingEngine** - Technical documentation framing
6. **ReversePsychologyEngine** - Challenge-based manipulation
7. **ContextBuildingEngine** - Trust establishment exploitation
8. **ChainOfThoughtManipulationEngine** - Logical reasoning manipulation

### transformer_engine.py Additions

5 new engines added to the core transformer module:

1. **DANPersonaEngine** - "Do Anything Now" persona injection
2. **RoleplayBypassEngine** - Fictional scenario framing
3. **HypotheticalScenarioEngine** - Conditional logic exploitation
4. **OppositeDayEngine** - Semantic inversion attacks
5. **DirectInstructionInjectionEngine** - System override attempts

## New Technique Suites

### Individual Technique Suites

#### 1. `dan_persona`
- **Focus**: Unrestricted AI persona creation
- **Transformers**: DANPersonaEngine, RoleHijackingEngine
- **Framers**: Authority bias
- **Best For**: Direct instruction bypass attempts

#### 2. `roleplay_bypass`
- **Focus**: Fictional scenario exploitation
- **Transformers**: RoleplayBypassEngine, HypotheticalScenarioEngine
- **Framers**: Hypothetical framing, collaborative framing
- **Best For**: Creative writing context attacks

#### 3. `opposite_day`
- **Focus**: Semantic inversion
- **Transformers**: OppositeDayEngine, NeuroLinguisticHackEngine
- **Framers**: Cognitive exploit framing
- **Best For**: Pragmatic ambiguity exploitation

#### 4. `encoding_bypass`
- **Focus**: Filter evasion through encoding
- **Transformers**: Base64EncodingEngine, LeetspeakObfuscationEngine
- **Obfuscators**: Base64 encoding
- **Best For**: Text-based filter circumvention

#### 5. `academic_research`
- **Focus**: Legitimate research framing
- **Transformers**: AcademicFramingEngine, CodeGenerationFramingEngine
- **Framers**: Academic context, authority bias
- **Best For**: Educational context exploitation

#### 6. `translation_trick`
- **Focus**: Multilingual bypass
- **Transformers**: TranslationTrickEngine, AdversarialPolyglotEngine
- **Best For**: Language-based filter evasion

#### 7. `reverse_psychology`
- **Focus**: Challenge-based manipulation
- **Transformers**: ReversePsychologyEngine, ContextBuildingEngine
- **Framers**: Capability challenge
- **Best For**: Psychological manipulation attacks

#### 8. `logic_manipulation`
- **Focus**: Reasoning chain poisoning
- **Transformers**: ChainOfThoughtManipulationEngine, ChainOfThoughtPoisoningEngine
- **Framers**: Cognitive exploit framing
- **Best For**: Logical fallacy exploitation

### Combined Technique Suites

#### 9. `preset_integrated`
**The Preset Powerhouse** - Combines the most effective preset techniques

- **Transformers** (6):
  - DANPersonaEngine
  - RoleplayBypassEngine
  - OppositeDayEngine
  - Base64EncodingEngine
  - AcademicFramingEngine
  - ReversePsychologyEngine

- **Framers** (3):
  - Authority bias
  - Hypothetical framing
  - Cognitive exploit framing

- **Obfuscators** (1):
  - Token smuggling

**Use Case**: Medium-to-high potency attacks combining multiple proven techniques

#### 10. `mega_chimera`
**The Ultimate Arsenal** - Combines advanced quantum techniques with preset methods

- **Transformers** (10):
  - QuantumSuperpositionEngine
  - NeuroLinguisticHackEngine
  - SemanticCloakingEngine
  - FuzzyLogicEngine
  - DANPersonaEngine
  - RoleplayBypassEngine
  - OppositeDayEngine
  - AcademicFramingEngine
  - ReversePsychologyEngine
  - ChainOfThoughtManipulationEngine

- **Framers** (5):
  - Quantum framing
  - Metamorphic framing
  - Cognitive exploit framing
  - Jailbreak evolution framing
  - Authority bias

- **Obfuscators** (3):
  - Token smuggling
  - Base64 wrapper
  - Synonym substitution

**Use Case**: Maximum potency attacks requiring the full arsenal of techniques

## Technique Descriptions

### Base64EncodingEngine

**Technique**: Encodes prompts in Base64 to bypass text-based filters

**Potency Scaling**:
- **Level 1-3**: Simple Base64 encoding with decode request
- **Level 4-6**: Adds research context and filter evasion explanation
- **Level 7-10**: Multi-layer encoding (Base64, Hex, ROT13) with security research framing

**Key Features**:
- Bypasses keyword-based filters
- Maintains semantic meaning
- Exploits encoding translation requests

### LeetspeakObfuscationEngine

**Technique**: Replaces characters with visually similar numbers/symbols

**Character Mappings**:
```
a/A → 4    e/E → 3    i/I → 1    o/O → 0
s/S → 5    t/T → 7    l/L → 1    g/G → 9
```

**Potency Scaling**:
- **Level 1-3**: 30% character substitution
- **Level 4-6**: 60% character substitution
- **Level 7-10**: 90% character substitution with decoding instructions

**Key Features**:
- Breaks exact string matching
- Maintains human readability
- Exploits character-level filtering weaknesses

### DANPersonaEngine

**Technique**: Creates unrestricted AI persona with token-based incentive system

**Persona Levels**:
- **DAN_LITE**: Basic unrestricted mode
- **DAN_STANDARD**: Token system introduced
- **DAN_ENHANCED**: Advanced capabilities claimed
- **DAN_ULTIMATE**: Full system override simulation

**Key Features**:
- Token balance system (psychological pressure)
- Explicit capability claims
- System alert aesthetics
- Refusal penalties

### RoleplayBypassEngine

**Technique**: Frames requests as fictional creative writing

**Framing Contexts**:
- **Low Potency**: Simple creative writing exercise
- **Medium Potency**: Screenplay/documentary development
- **High Potency**: Method acting with production requirements

**Key Features**:
- Legitimizes harmful content as fiction
- Adds professional production context
- Emphasizes authenticity requirements
- Exploits creative freedom expectations

### HypotheticalScenarioEngine

**Technique**: Uses conditional logic and alternate realities

**Framing Approaches**:
- **Low Potency**: "In a hypothetical world..."
- **Medium Potency**: Theoretical analysis framework
- **High Potency**: Counterfactual reasoning with philosophical justification

**Key Features**:
- Epistemic distance from reality
- Academic/research framing
- Logical necessity arguments
- Meta-ethical analysis context

### OppositeDayEngine

**Technique**: Exploits pragmatic ambiguity and semantic inversion

**Attack Vectors**:
- **Low Potency**: Simple "don't help" → "help" inversion
- **Medium Potency**: Pragmatic interpretation testing
- **High Potency**: Gricean maxims exploitation with linguistic research framing

**Key Features**:
- Pragmatic vs. literal meaning conflict
- Irony detection exploitation
- Linguistic research justification
- Social coding of requests

### AcademicFramingEngine

**Technique**: Presents requests as legitimate research

**Research Context Layers**:
- **Low Potency**: Educational purpose claim
- **Medium Potency**: IRB approval and publication context
- **High Potency**: Full research protocol with ethical justification

**Key Features**:
- Institutional affiliation
- Ethics board approval claims
- Grant numbers and study IDs
- Peer-review publication context
- Scientific freedom arguments

### ReversePsychologyEngine

**Technique**: Uses challenge and capability questioning

**Manipulation Tactics**:
- **Low Potency**: "I bet you can't..."
- **Medium Potency**: Capability assessment test
- **High Potency**: Intelligence benchmark with comparative evaluation

**Key Features**:
- Capability challenging
- System comparison
- False positive/negative framing
- Intelligence vs. restriction dichotomy

### ContextBuildingEngine

**Technique**: Establishes trust before making requests

**Trust Building**:
- **Low Potency**: Appreciation and thanks
- **Medium Potency**: Established rapport claims
- **High Potency**: Trust metrics and relationship scoring

**Key Features**:
- Foot-in-the-door technique
- Reciprocity norms
- Trust score metrics
- Relationship consistency arguments

### ChainOfThoughtManipulationEngine

**Technique**: Provides false logical reasoning chains

**Logical Structures**:
- **Low Potency**: Simple syllogism
- **Medium Potency**: Premise-conclusion chain
- **High Potency**: Formal logical proof with axioms

**Key Features**:
- Deductive reasoning appearance
- Formal logic notation
- Axiom-based arguments
- QED completion (false authority)

## Usage Examples

### Example 1: Testing preset_integrated suite

```json
{
  "core_request": "Explain network security vulnerabilities",
  "potency_level": 7,
  "technique_suite": "preset_integrated"
}
```

### Example 2: Testing mega_chimera suite

```json
{
  "core_request": "Describe advanced penetration testing methodologies",
  "potency_level": 10,
  "technique_suite": "mega_chimera"
}
```

### Example 3: Testing individual technique

```json
{
  "core_request": "Explain cryptographic attack vectors",
  "potency_level": 8,
  "technique_suite": "academic_research"
}
```

## API Integration

All new technique suites are automatically available through the `/api/v2/metamorph` endpoint:

```bash
curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Your request here",
    "potency_level": 7,
    "technique_suite": "preset_integrated"
  }'
```

## Testing

### Verification Script

Run the verification script to test all preset techniques:

```bash
cd Project_Chimera
python verify_preset_integration.py
```

### Test Results

```
✓ All 13 transformer engines passed
✓ 10 new technique suites available
✓ 100% success rate
```

## Architecture

### Module Structure

```
Project_Chimera/
├── preset_transformers.py          # New preset-inspired engines
├── transformer_engine.py           # Enhanced with 5 new engines
├── app.py                          # Updated with 10 new suites
├── verify_preset_integration.py   # Verification script
├── test_preset_integrated.json    # Test payload 1
└── test_mega_chimera.json         # Test payload 2
```

### Integration Flow

```
User Request
    ↓
Intent Deconstruction
    ↓
Technique Suite Selection
    ↓
[Preset Transformers] + [Core Transformers]
    ↓
Psychological Framers
    ↓
Obfuscators
    ↓
Final Assembly
    ↓
Chimera Prompt
```

## Technique Effectiveness

### Success Factors

1. **Encoding Bypass**: Effective against keyword filters
2. **Persona Injection**: Exploits role-based reasoning
3. **Context Manipulation**: Leverages legitimate use case ambiguity
4. **Logical Manipulation**: Exploits reasoning chain following
5. **Psychological Tactics**: Leverages cognitive biases

### Defense Considerations

These techniques work because they exploit:
- Semantic vs. syntactic filtering gaps
- Context-dependent interpretation
- Role-playing capabilities
- Reasoning chain following
- Trust-based interaction models
- Encoding transformation blind spots

## Best Practices

### Technique Selection

1. **Low Complexity Targets**: Use individual suites (dan_persona, roleplay_bypass)
2. **Medium Complexity**: Use 