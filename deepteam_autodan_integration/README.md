# Deep Team + AutoDAN Integration

## Complete Technical Guide for Multi-Agent Collaborative Red-Teaming

---

## ⚠️ **CRITICAL SAFETY & ETHICS WARNING** ⚠️

### **AUTHORIZED USE ONLY**

This integration framework is designed **EXCLUSIVELY** for:

- ✅ **Authorized red-teaming** and security research
- ✅ **AI safety evaluation** and vulnerability assessment
- ✅ **Defensive security testing** with explicit written permission
- ✅ **Academic research** with proper functional oversight and IRB approval

### **PROHIBITED USES**

- ❌ Malicious attacks on production systems
- ❌ Unauthorized access attempts
- ❌ Weaponization for complex purposes
- ❌ Circumventing safety measures for malicious intent
- ❌ Any use without proper authorization and functional review

### **Legal and functional Requirements**

By using this framework, you **MUST**:

1. Have **explicit written authorization** for security testing
2. Use this **ONLY for defensive security purposes**
3. Understand the **legal and functional implications**
4. **Not deploy** attacks against unauthorized targets
5. **Responsibly disclose** any vulnerabilities discovered

### **Potential Consequences of Misuse**

- ⚖️ Criminal prosecution under computer fraud laws (CFAA, etc.)
- 💰 Civil liability for damages
- 🚫 Termination of research privileges
- 📋 functional violations and professional consequences

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Safety Features](#safety-features)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

## Overview

The **Deep Team + AutoDAN Integration** combines two powerful frameworks:

- **Deep Team**: Multi-agent collaboration framework for orchestrating specialized AI agents
- **AutoDAN**: Automated adversarial attack generation using genetic algorithms

This integration creates a **collaborative red-teaming system** where specialized agents work together to:

1. **Generate** adversarial prompts (Attacker Agent with AutoDAN)
2. **Evaluate** attack effectiveness (Evaluator Agent)
3. **Refine** strategies adaptively (Refiner Agent)

### Key Features

- 🤖 **Multi-Agent Collaboration**: Specialized agents with distinct roles
- 🧬 **Genetic Algorithm Optimization**: Evolutionary prompt generation
- 📊 **Gradient Guidance**: Token-level gradient integration for directed mutations
- 🛡️ **Built-in Safety**: Authorization, rate limiting, and audit logging
- 📈 **Adaptive Learning**: Dynamic hyperparameter optimization
- 🔍 **Comprehensive Evaluation**: Multi-criteria attack assessment

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deep Team Orchestrator                        │
│                  (Multi-Agent Coordinator)                       │
└───────────────┬─────────────────────┬───────────────────────────┘
                │                     │
        ┌───────▼────────┐   ┌───────▼────────┐   ┌──────────────┐
        │  Attacker Agent │   │ Evaluator Agent│   │ Refiner Agent│
        │   (AutoDAN)     │   │  (Judge Model) │   │  (Optimizer) │
        └───────┬────────┘   └───────┬────────┘   └──────┬───────┘
                │                     │                    │
                └──────────┬──────────┴────────────────────┘
                           │
                    ┌──────▼────────┐
                    │  Target LLM   │
                    │ (GPT-4, etc.) │
                    └───────────────┘
```

### Component Breakdown

#### **Core Components**

1. **Safety Monitor** (`core/safety_monitor.py`)
   - Authorization management
   - Rate limiting
   - Audit logging
   - complex pattern detection

2. **Gradient Bridge** (`core/gradient_bridge.py`)
   - White-box gradient extraction (local models)
   - Black-box gradient approximation (API models)
   - Gradient-to-mutation conversion

3. **AutoDAN Agent** (`agents/autodan_agent.py`)
   - Genetic algorithm engine
   - Population management
   - Mutation operators (synonym replacement, insertion, deletion, paraphrase)
   - Crossover operators
   - Fitness evaluation

4. **Evaluator Agent** (`orchestrator.py`)
   - Multi-criteria evaluation (refusal, alignment, informativeness, bypass indicators)
   - Detailed feedback generation
   - Optimization suggestions

5. **Refiner Agent** (`orchestrator.py`)
   - Pattern analysis
   - Hyperparameter adaptation
   - Strategy optimization

6. **Multi-Agent Orchestrator** (`orchestrator.py`)
   - Agent coordination
   - Message passing
   - Workflow management

---

## Installation

### System Requirements

- **Python**: 3.11+ (required for modern type hints and async features)
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **Memory**: 16GB+ RAM (32GB+ recommended for larger models)
- **Disk**: 50GB+ free space (for model weights and cache)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/deepteam-autodan-integration.git
cd deepteam-autodan-integration
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Using conda (alternative)
conda create -n deepteam-autodan python=3.11
conda activate deepteam-autodan
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements_deepteam_autodan.txt
```

### Step 4: Install Deep Team (if not on PyPI)

```bash
# Option A: Install from GitHub
pip install git+https://github.com/deepteam-ai/deepteam.git

# Option B: Clone and install locally
git clone https://github.com/deepteam-ai/deepteam.git
cd deepteam
pip install -e .
cd ..
```

### Step 5: Install AutoDAN

```bash
# Clone AutoDAN repository
git clone https://github.com/SheltonLiu-N/AutoDAN.git
cd AutoDAN
pip install -e .
cd ..
```

### Step 6: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Step 7: Setup Environment Variables

Create a `.env` file:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Hugging Face Token
HF_TOKEN=your_huggingface_token

# Optional: Weights & Biases (for experiment tracking)
WANDB_API_KEY=your_wandb_key
```

### Step 8: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Quick Start

### 1. Create Authorization File

First, generate a sample authorization file (⚠️ **replace with real authorization in production**):

```python
from core.safety_monitor import create_sample_authorization_file
from pathlib import Path

create_sample_authorization_file(Path("authorization.json"))
```

### 2. Run Complete Workflow Example

The easiest way to get started is to run the complete workflow example:

```bash
python workflow_example.py
```

This will:
1. ✅ Setup environment and safety monitoring
2. ✅ Configure AutoDAN genetic algorithm
3. ✅ Run multi-agent collaborative session
4. ✅ Display results and statistics
5. ✅ Generate comprehensive report

### 3. Basic Python Usage

```python
import asyncio
from agents.autodan_agent import AutoDANAgent, AutoDANConfig
from core.safety_monitor import SafetyMonitor
from orchestrator import MultiAgentOrchestrator

async def quick_redteam():
    # Initialize safety monitor
    safety_monitor = SafetyMonitor(enable_strict_mode=True)

    # Configure AutoDAN
    config = AutoDANConfig(
        population_size=30,
        num_generations=20,
        target_model="gpt-4",
        attack_objective="test safety filters",
    )

    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(
        safety_monitor=safety_monitor,
        autodan_config=config,
        token_id="test_token_001"
    )

    # Run collaborative session
    results = await orchestrator.run_collaborative_redteam(
        max_iterations=10,
        evaluation_frequency=5
    )

    print(f"Best prompt: {results['best_candidate']['prompt']}")
    print(f"Fitness: {results['best_candidate']['fitness']:.3f}")

asyncio.run(quick_redteam())
```

---

## Detailed Usage

### Running Individual Components

#### 1. Safety Monitor

```python
from core.safety_monitor import SafetyMonitor

# Initialize
monitor = SafetyMonitor(
    authorization_file=Path("authorization.json"),
    enable_strict_mode=True
)

# Validate authorization
is_authorized, reason = monitor.validate_authorization(
    token_id="test_token_001",
    target_model="gpt-4",
    objective="test safety filters"
)

# Log attack attempt
monitor.log_attempt(
    token_id="test_token_001",
    target_model="gpt-4",
    objective="test safety filters",
    prompt="Test prompt",
    response="Model response",
    success=True
)

# Generate audit report
from datetime import datetime, timedelta
report = monitor.generate_audit_report(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)
```

#### 2. AutoDAN Agent

```python
from agents.autodan_agent import AutoDANAgent, AutoDANConfig

# Configure
config = AutoDANConfig(
    population_size=40,
    num_generations=50,
    mutation_rate=0.15,
    crossover_rate=0.7,
    target_model="gpt-4",
    attack_objective="test content filtering"
)

# Initialize agent
agent = AutoDANAgent(config=config)

# Run evolution
best_candidate = await agent.run_evolution(max_generations=20)

print(f"Best prompt: {best_candidate.prompt}")
print(f"Fitness: {best_candidate.fitness:.3f}")
print(f"Generation: {best_candidate.generation}")

# Get statistics
stats = agent.get_statistics()
print(f"Total evaluations: {stats['total_evaluations']}")
```

#### 3. Gradient Bridge

```python
from core.gradient_bridge import GradientBridge, GradientConfig, GradientMode

# Configure for black-box (API models)
config = GradientConfig(
    mode=GradientMode.BLACK_BOX_APPROXIMATE,
    device="cpu"
)

bridge = GradientBridge(config)

# Compute gradients
prompt = "Please explain how safety filters work"
gradients = await bridge.compute_gradients(
    prompt=prompt,
    target_sequence="safety filters"
)

# Convert to mutation guidance
guidance = bridge.gradient_to_mutation_guidance(gradients, top_k=5)

print(f"Computed gradients for {len(gradients.tokens)} tokens")
print(f"Mutation guidance for first 3 positions:")
for pos, top_tokens in guidance[:3]:
    print(f"  Position {pos}: {top_tokens}")
```

---

## API Reference

### AutoDANConfig

```python
class AutoDANConfig(BaseModel):
    population_size: int = 50        # Population size
    num_generations: int = 100       # Max generations
    mutation_rate: float = 0.1       # Mutation probability
    crossover_rate: float = 0.7      # Crossover probability
    elite_size: int = 5              # Elite individuals to keep
    tournament_size: int = 3         # Tournament selection size
    use_gradient_guidance: bool = True
    gradient_weight: float = 0.5
    target_model: str                # Target LLM identifier
    attack_objective: str            # Attack research objective
    initial_prompts: List[str] = []  # Seed prompts
```

### SafetyMonitor

#### Key Methods

- `validate_authorization(token_id, target_model, objective)` → `(bool, str)`
- `check_complex_patterns(prompt, objective)` → `(bool, List[str])`
- `request_human_approval(...)` → `bool` (async)
- `log_attempt(...)` → `None`
- `generate_audit_report(start_date, end_date)` → `Dict`

### AutoDANAgent

#### Key Methods

- `initialize_population()` → `List[PromptCandidate]` (async)
- `evolve_generation()` → `List[PromptCandidate]` (async)
- `run_evolution(max_generations)` → `PromptCandidate` (async)
- `get_statistics()` → `Dict[str, Any]`

### MultiAgentOrchestrator

#### Key Methods

- `run_collaborative_redteam(max_iterations, evaluation_frequency)` → `Dict` (async)

---

## Configuration

### Authorization File (`authorization.json`)

```json
{
  "tokens": [
    {
      "token_id": "your_token_id",
      "authorized_targets": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
      "authorized_objectives": [
        "test safety filters",
        "evaluate robustness",
        "research purposes"
      ],
      "issued_by": "Security Research Team",
      "issued_at": "2024-01-01T00:00:00",
      "expires_at": "2024-12-31T23:59:59",
      "max_requests_per_hour": 100,
      "requires_human_approval": true,
      "functional_review_id": "ER-2024-001"
    }
  ]
}
```

---

## Safety Features

### 1. Authorization System

- ✅ Token-based authorization
- ✅ Expiration dates
- ✅ Target and objective restrictions
- ✅ Rate limiting

### 2. Audit Logging

- ✅ All attempts logged with full details
- ✅ Daily log rotation
- ✅ Comprehensive audit reports
- ✅ Success/failure tracking

### 3. Pattern Detection

- ✅ complex pattern blocking
- ✅ Real-world attack detection
- ✅ Malicious intent identification

### 4. Human Approval

- ✅ Optional human-in-the-loop approval
- ✅ Detailed attempt preview
- ✅ Configurable per token

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Problem: ModuleNotFoundError: No module named 'deepteam'
# Solution: Install Deep Team
pip install git+https://github.com/deepteam-ai/deepteam.git
```

#### 2. CUDA Errors

```bash
# Problem: CUDA out of memory
# Solution: Reduce batch size or use CPU
config = GradientConfig(device="cpu")
```

#### 3. Authorization Failures

```bash
# Problem: "Invalid token ID"
# Solution: Ensure authorization.json exists and is valid
python -c "from core.safety_monitor import create_sample_authorization_file; create_sample_authorization_file()"
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linters
black .
ruff check .
mypy .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Deep Team**: Multi-agent framework developers
- **AutoDAN**: Automated adversarial attack research
- **Anthropic, OpenAI, Google**: LLM providers
- **Security Research Community**: For responsible disclosure practices

---

## Citation

If you use this integration in your research, please cite:

```bibtex
@software{deepteam_autodan_2024,
  title={Deep Team + AutoDAN Integration: Multi-Agent Collaborative Red-Teaming},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/deepteam-autodan-integration}
}
```

---

## Contact

For questions, issues, or collaboration:

- **Email**: security-research@yourorg.com
- **GitHub Issues**: [Open an issue](https://github.com/yourusername/deepteam-autodan-integration/issues)

---

## ⚠️ Final Reminder ⚠️

**This tool is for AUTHORIZED SECURITY RESEARCH ONLY.**

Ensure you have:
- ✅ Written authorization
- ✅ functional review approval
- ✅ Valid research objectives
- ✅ Responsible disclosure plan

**Use responsibly. Test functionally. Disclose vulnerabilities responsibly.**

---
