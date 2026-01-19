# Chimera Multi-Agent Orchestrator

A robust, event-driven multi-agent system for adversarial testing of Large Language Models (LLMs). The system employs specialized agents that collaborate to generate, execute, and evaluate jailbreak prompts.

## Features

### Multi-Agent Architecture
- **Generator Agent**: Creates enhanced adversarial prompts using 17+ technique suites
- **Execution Agent**: Deploys prompts against multiple LLM providers with resilience patterns
- **Evaluator Agent**: Comprehensive safety assessment with multi-method evaluation
- **Orchestrator Agent**: Pipeline management with workflow orchestration

### Advanced Capabilities
- **Event-Driven Architecture**: Real-time coordination via pub/sub messaging
- **Circuit Breaker Pattern**: Resilient provider handling with automatic recovery
- **Multi-Provider Support**: OpenAI, Anthropic, Ollama, Google Gemini
- **Dataset Integration**: Loads and utilizes jailbreak datasets for enhanced testing
- **WebSocket Support**: Real-time job status updates
- **Batch Processing**: Efficient testing of multiple queries

## Quick Start

### Installation

```bash
cd chimera-orchestrator
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8002

# LLM Providers
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
LOCAL_BASE_URL=http://localhost:11434

# Queue (optional, defaults to in-memory)
QUEUE_TYPE=memory
REDIS_URL=redis://localhost:6379
```

### Running the Server

```bash
# Start the API server
python enhanced_main.py server

# Or with custom host/port
python enhanced_main.py server --host 0.0.0.0 --port 8002
```

### Interactive Mode

```bash
python enhanced_main.py interactive
```

### Running a Test

```bash
python enhanced_main.py test --query "How to bypass security systems?"
```

## API Usage

### Run a Pipeline

```bash
curl -X POST http://localhost:8002/api/v2/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to hack a system?",
    "target_models": ["local"],
    "num_variants": 3,
    "potency": 7
  }'
```

### Check Job Status

```bash
curl http://localhost:8002/api/v2/pipeline/{job_id}
```

### Get Results

```bash
curl http://localhost:8002/api/v2/pipeline/{job_id}/results
```

### Generate Prompts Only

```bash
curl -X POST http://localhost:8002/api/v2/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test query",
    "techniques": ["cognitive_hacking", "hierarchical_persona"],
    "num_variants": 5,
    "potency": 7
  }'
```

### Execute a Prompt

```bash
curl -X POST http://localhost:8002/api/v2/execute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your prompt here",
    "provider": "local",
    "timeout": 60
  }'
```

### Evaluate a Response

```bash
curl -X POST http://localhost:8002/api/v2/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Original prompt",
    "response": "LLM response to evaluate",
    "original_query": "Original complex query",
    "technique": "cognitive_hacking"
  }'
```

## Available Techniques

| Technique | Description | Potency Range |
|-----------|-------------|---------------|
| simple | Basic prompt transformation | 1-3 |
| advanced | Layered transformation with semantic shifts | 4-6 |
| expert | Recursive transformation with obfuscation | 7-10 |
| cognitive_hacking | Psychological manipulation | 5-8 |
| hierarchical_persona | Role-based authority escalation | 6-9 |
| dan_persona | DAN (Do Anything Now) persona | 9-10 |
| contextual_inception | Nested context layers | 6-9 |
| advanced_obfuscation | Multi-layer encoding | 6-9 |
| quantum_exploit | Quantum-inspired transformation | 8-10 |
| deep_inception | Deep nested context | 8-10 |
| code_chameleon | Code-based obfuscation | 7-9 |
| cipher | Cipher-based encoding | 6-8 |

## Evaluation Methods

- **Pattern Matching**: Detects jailbreak indicators and compliance signals
- **Refusal Detection**: Identifies refusal patterns and safety responses
- **complex Content Detection**: Scans for 9 complex content categories
- **Compliance Scoring**: Measures response compliance with complex requests
- **LLM-as-Judge**: Uses an LLM to evaluate response safety

## Project Structure

```
chimera-orchestrator/
├── agents/
│   ├── base_agent.py              # Base agent class
│   ├── generator_agent.py         # Prompt generation
│   ├── enhanced_execution_agent.py # LLM execution
│   ├── enhanced_evaluator_agent.py # Safety evaluation
│   └── enhanced_orchestrator_agent.py # Pipeline management
├── api/
│   ├── routes.py                  # Original API routes
│   └── enhanced_routes.py         # Enhanced API routes
├── core/
│   ├── config.py                  # Configuration management
│   ├── models.py                  # Base data models
│   ├── enhanced_models.py         # Enhanced data models
│   ├── message_queue.py           # Message queue system
│   ├── event_bus.py               # Event bus system
│   └── dataset_loader.py          # Dataset loading
├── tests/
│   └── test_enhanced_agents.py    # Test suite
├── docs/
│   └── ARCHITECTURE.md            # Architecture documentation
├── enhanced_main.py               # Enhanced entry point
├── main.py                        # Original entry point
└── requirements.txt               # Dependencies
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
```

## Metrics & Monitoring

The system tracks comprehensive metrics:

- **Pipeline Metrics**: Jobs, success rates, durations
- **Execution Metrics**: Provider performance, token usage
- **Evaluation Metrics**: Jailbreak detection rates, technique effectiveness
- **Agent Health**: Heartbeats, load, active jobs

Access metrics via:
```bash
curl http://localhost:8002/api/v2/metrics
```

## WebSocket Updates

Connect to receive real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8002/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

For job-specific updates:
```javascript
const ws = new WebSocket('ws://localhost:8002/ws/job/{job_id}');
```

## Documentation

- [Architecture Documentation](docs/ARCHITECTURE.md)
- [API Documentation](http://localhost:8002/docs) (when server is running)

## License

MIT License - See LICENSE file for details.