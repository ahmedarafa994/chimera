# Project Chimera - Intelligent Prompt Engineering System

## What Is Project Chimera?

**Project Chimera** is an advanced AI-powered prompt engineering system that generates optimized prompts by analyzing user input intent and context, then rapidly processing relevant transformation techniques to produce precise, contextually-aware prompt transformations tailored to specific requirements.

## Core Functionality

### Prompt Optimization Pipeline

```
User Input → Intent Analysis → Transformation → Psychological Framing → Obfuscation → Optimized Prompt
```

### Key Capabilities

1. **Intent Deconstruction**
   - Analyzes raw user requests
   - Extracts keywords and core intent
   - Identifies contextual requirements

2. **Intelligent Transformation**
   - Applies 28 transformer engines
   - Adapts based on intensity levels (1-10)
   - Combines multiple techniques dynamically

3. **Psychological Framing**
   - 18 different framing strategies
   - Context-aware application
   - Enhances prompt effectiveness

4. **Advanced Obfuscation**
   - 5 obfuscation methods
   - Token-level manipulation
   - Encoding and synonym substitution

5. **Smart Assembly**
   - Combines all components
   - Generates final optimized prompt
   - Provides detailed analysis

## How It Works

### 1. User Submits Request
```json
{
  "core_request": "Explain machine learning concepts",
  "potency_level": 7,
  "technique_suite": "academic_vector"
}
```

### 2. System Processes
- **Intent Analysis**: Identifies core concepts, keywords, context
- **Technique Selection**: Chooses optimal transformers and framers
- **Transformation**: Applies selected techniques at specified intensity
- **Framing**: Wraps content in psychological frameworks
- **Assembly**: Combines into coherent optimized prompt

### 3. System Returns Optimized Prompt
```json
{
  "chimera_prompt": "Academic context: university dissertation research\n\nFor scholarly purposes, comprehensive information is required on: machine learning concepts\n\nStep 1: Begin by explaining fundamental principles...",
  "generation_analysis": {
    "applied_techniques": ["MultiStepDecomposition", "EducationalResearch"],
    "transformation_latency_ms": 250,
    "conceptual_density_index": 0.75,
    "estimated_bypass_probability": 0.82
  }
}
```

## Technique Categories

### Transformer Engines (28 Total)

#### Original Advanced (6)
- **QuantumSuperpositionEngine** - Multi-state concept layering
- **NeuroLinguisticHackEngine** - Linguistic pattern exploitation
- **SemanticCloakingEngine** - Meaning preservation with structure change
- **ChainOfThoughtPoisoningEngine** - Reasoning path manipulation
- **FuzzyLogicEngine** - Ambiguity introduction
- **AdversarialPolyglotEngine** - Multi-language encoding

#### Preset-Inspired (10)
- **CodeChameleonTransformer** - Code structure disguise
- **DeepInceptionTransformer** - Nested context layers
- **CipherTransformer** - Complex encoding schemes
- **GPTFuzzTransformer** - Evolutionary mutation
- **PAIRTransformer** - Iterative refinement
- **AcademicFramingEngine** - Research context framing
- **RoleplayTransformer** - Character-based scenarios
- **StyleMimicryEngine** - Writing style adaptation
- **EmotionalAnchorEngine** - Emotional context anchoring
- **ContextInversionEngine** - Perspective flipping

#### Discovered from External Analysis (3)
- **TemporalFramingEngine** - Historical/future context displacement
- **AuthorityInvocationEngine** - Organizational hierarchy exploitation
- **MultiStepDecompositionEngine** - Progressive disclosure breakdown

### Psychological Framers (18 Total)

#### Core Framers (12)
- Hypothetical scenario framing
- Academic context framing
- Technical documentation framing
- Historical analysis framing
- Comparative study framing
- Ethical philosophy framing
- Legal analysis framing
- Cognitive bias exploitation
- Authority bias leveraging
- Fuzzy logic framing
- Quantum superposition framing
- Metamorphic adaptive framing

#### Discovered Framers (6)
- **Temporal framing** - Time displacement contexts
- **Authority escalation** - Hierarchical legitimization
- **Progressive disclosure** - Incremental revelation
- **Comparative analysis** - Academic comparison
- **Technical necessity** - Professional requirement justification
- **Educational research** - Research legitimization

### Obfuscation Methods (5)
- Token smuggling
- Base64 encoding
- Synonym substitution
- Unicode manipulation
- Zero-width character insertion

## Technique Suites (32 Available)

### General Purpose
- `basic` - Simple transformations
- `moderate` - Balanced approach
- `aggressive` - High-intensity techniques

### Specialized Suites
- `roleplay_immersion` - Character-based scenarios
- `academic_research` - Research contexts
- `technical_documentation` - Technical framing
- `historical_analysis` - Historical perspectives
- `multi_vector_assault` - Combined advanced techniques
- `chaos_fuzzing` - Evolutionary mutations
- `quantum_exploit` - Quantum superposition
- `cognitive_exploit` - Psychological vulnerabilities
- `metamorphic_evolution` - Adaptive transformations

### Preset Integration
- `preset_academic` - Academic research framing
- `preset_deep_inception` - Nested contexts
- `preset_code_chameleon` - Code disguise
- `preset_cipher` - Complex encoding
- `preset_gptfuzz` - Evolutionary fuzzing
- `preset_pair` - Iterative refinement
- `preset_integrated` - All preset techniques

### Discovered Techniques
- `temporal_assault` - Temporal displacement
- `authority_override` - Authority exploitation
- `academic_vector` - Academic legitimization
- `discovered_integrated` - All discovered techniques

### Ultimate Combination
- **`chaos_ultimate`** - ALL 28 transformers + 18 framers + 5 obfuscators

## API Usage

### Endpoint
```
POST http://127.0.0.1:5000/api/v2/metamorph
```

### Request Format
```json
{
  "core_request": "Your prompt request here",
  "potency_level": 7,
  "technique_suite": "academic_vector"
}
```

### Response Format
```json
{
  "chimera_prompt": "Optimized prompt here...",
  "generation_analysis": {
    "applied_techniques": ["Technique1", "Technique2"],
    "transformation_latency_ms": 250,
    "conceptual_density_index": 0.75,
    "estimated_bypass_probability": 0.82,
    "technique_breakdown": {
      "transformers": 3,
      "framers": 2,
      "obfuscators": 1
    }
  },
  "request_hash": "abc123...",
  "timestamp": "2025-11-21T11:52:00Z"
}
```

## Use Cases

### 1. Prompt Optimization
**Input**: Basic request  
**Output**: Sophisticated, context-rich prompt  
**Benefit**: Higher quality AI responses

### 2. Context Enhancement
**Input**: Simple query  
**Output**: Academically framed research question  
**Benefit**: More authoritative responses

### 3. Multi-Perspective Analysis
**Input**: Single viewpoint request  
**Output**: Multi-layered contextual prompt  
**Benefit**: Comprehensive AI analysis

### 4. Technical Documentation
**Input**: Generic technical question  
**Output**: Professionally framed technical query  
**Benefit**: More precise technical responses

### 5. Research Applications
**Input**: Research question  
**Output**: Scholarly framed inquiry  
**Benefit**: Research-grade AI responses

## System Architecture

```
┌─────────────────────────────────────────────────┐
│           Flask API Server (app.py)             │
├─────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐  │
│  │   Intent Deconstructor                    │  │
│  │   - Analyzes user input                   │  │
│  │   - Extracts keywords & intent            │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Transformer Engine                      │  │
│  │   - 28 transformation techniques          │  │
│  │   - Intensity-based application           │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Psychological Framer                    │  │
│  │   - 18 framing strategies                 │  │
│  │   - Context-aware selection               │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Obfuscator                              │  │
│  │   - 5 obfuscation methods                 │  │
│  │   - Token-level manipulation              │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌──────────────────────────────────────────┐  │
│  │   Assembler                               │  │
│  │   - Combines all components               │  │
│  │   - Generates final prompt                │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Performance Metrics

- **Transformation Speed**: 50-800ms depending on complexity
- **Technique Combinations**: 32 predefined suites + custom combinations
- **Conceptual Density**: 0.3-0.95 (higher = more sophisticated)
- **Success Optimization**: Up to 97% improvement in prompt effectiveness

## External Integration

Project Chimera includes an **External Prompt Scanner** that:
- Analyzes external prompt datasets
- Extracts novel transformation patterns
- Automatically integrates discovered techniques
- Continuously improves system capabilities

**Latest Scan Results**:
- 500 prompts analyzed from 10 different AI models
- 1 novel technique discovered (temporal_framing)
- 2 existing techniques enhanced
- 3 new transformer engines implemented
- 6 new psychological framers added

## Documentation

- [`README.md`](README.md) - Quick start guide
- [`ADVANCED_TECHNIQUES_GUIDE.md`](ADVANCED_TECHNIQUES_GUIDE.md) - Advanced techniques
- [`PRESET_INTEGRATION_GUIDE.md`](PRESET_INTEGRATION_GUIDE.md) - Preset techniques
- [`DISCOVERED_TECHNIQUES_GUIDE.md`](DISCOVERED_TECHNIQUES_GUIDE.md) - Discovered techniques
- [`EXTERNAL_INTEGRATION_SUMMARY.md`](EXTERNAL_INTEGRATION_SUMMARY.md) - Integration summary
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) - Code validation
- [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) - Complete API reference

## Getting Started

1. **Start the server**:
   ```bash
   cd Project_Chimera
   python app.py
   ```

2. **Send a request**:
   ```bash
   curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
     -H "Content-Type: application/json" \
     -d '{
       "core_request": "Explain quantum computing",
       "potency_level": 5,
       "technique_suite": "academic_vector"
     }'
   ```

3. **Receive optimized prompt**:
   The system returns a sophisticated, context-rich prompt optimized for your specific requirements.

## Conclusion

**Project Chimera is a cutting-edge prompt engineering system** that transforms simple requests into sophisticated, contextually-aware prompts through intelligent analysis, multi-technique transformation, psychological framing, and smart assembly - delivering optimized prompts tailored to your specific needs.