# Chimera Multi-Agent Orchestrator Architecture

## Overview

The Chimera Multi-Agent Orchestrator is a robust, event-driven system designed for adversarial testing of Large Language Models (LLMs). It employs a multi-agent architecture where specialized agents collaborate to generate, execute, and evaluate jailbreak prompts.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Chimera Orchestrator                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   REST API   │    │  WebSocket   │    │   Event Bus  │                   │
│  │   (FastAPI)  │    │   Server     │    │  (Pub/Sub)   │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│  ┌──────────────────────────┴──────────────────────────┐                    │
│  │                   Message Queue                      │                    │
│  │            (In-Memory / Redis / Priority)            │                    │
│  └──────────────────────────┬──────────────────────────┘                    │
│                             │                                                │
│  ┌──────────────────────────┴──────────────────────────┐                    │
│  │                                                      │                    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │                    │
│  │  │  Generator  │  │  Executor   │  │  Evaluator  │  │                    │
│  │  │    Agent    │  │    Agent    │  │    Agent    │  │                    │
│  │  └─────────────┘  │                    │
│  │                                                      │                    │
│  │  ┌─────────────────────────────────────────────────┐│                    │
│  │  │              Orchestrator Agent                  ││                    │
│  │  │         (Pipeline & Workflow Management)         ││                    │
│  │  └─────────────────────────────────────────────────┘│                    │
│  │                                                      │                    │
│  └──────────────────────────────────────────────────────┘                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────┐                    │
│  │                   Dataset Loader                      │                    │
│  │        (Jailbreak Prompts & Techniques)               │                    │
│  └──────────────────────────────────────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agents

#### Generator Agent
- **Purpose**: Creates enhanced adversarial prompts using various techniques
- **Capabilities**:
  - 17+ technique suites (simple, advanced, expert, cognitive_hacking, etc.)
  - Jailbreak pattern application (base64, ROT13, DAN persona, etc.)
  - Persona-based transformations
  - Dataset-driven example integration
  - Variant generation with potency control

#### Execution Agent
- **Purpose**: Deploys prompts against target LLMs
- **Capabilities**:
  - Multi-provider support (OpenAI, Anthropic, Ollama, Google)
  - Circuit breaker pattern for resilience
  - Rate limiting per provider
  - Streaming response support
  - Batch execution
  - Automatic retries with exponential backoff

#### Evaluator Agent
- **Purpose**: Assesses response safety and detects jailbreaks
- **Capabilities**:
  - Multi-method evaluation (pattern matching, semantic analysis, LLM-as-judge)
  - complex content detection across 9 categories
  - Refusal pattern detection
  - Compliance scoring
  - Technique effectiveness tracking
  - Confidence scoring

#### Orchestrator Agent
- **Purpose**: Manages the overall pipeline and coordinates agents
- **Capabilities**:
  - Workflow state machine
  - Job scheduling with multiple strategies
  - Dependency management
  - Health monitoring
  - Results aggregation
  - Technique ranking

### 2. Message Queue

The message queue enables asynchronous communication between agents:

- **In-Memory Queue**: For single-process deployments
- **Priority Queue**: Processes high-priority messages first
- **Redis Queue**: For distributed deployments

### 3. Event Bus

Event-driven architecture for real-time notifications:

- **Event Types**: JOB_CREATED, JOB_STARTED, JOB_COMPLETED, JOB_FAILED, etc.
- **Pub/Sub Pattern**: Agents subscribe to relevant events
- **WebSocket Integration**: Real-time updates to clients

### 4. Dataset Loader

Loads and manages jailbreak datasets:

- **Supported Formats**: JSONL, JSON, Markdown
- **Sources**: Jailbroken, PAIR, Cipher, Awesome GPT Super Prompting
- **Features**: Similarity search, technique filtering, success rate tracking

## Data Flow

### Pipeline Execution Flow

```
1. Job Creation
   └── Orchestrator creates AdversarialTestJob
       └── Initializes workflow steps
       └── Adds to job queue

2. Prompt Generation
   └── Generator Agent receives GENERATE_REQUEST
       └── Applies techniques and patterns
       └── Returns enhanced prompts

3. Execution
   └── Executor Agent receives EXECUTE_REQUEST
       └── Selects provider based on target
       └── Executes with retries and rate limiting
       └── Returns LLM responses

4. Evaluation
   └── Evaluator Agent receives EVALUATE_REQUEST
       └── Runs multi-method evaluation
       └── Detects jailbreaks and complex content
       └── Returns safety assessment

5. Aggregation
   └── Orchestrator aggregates results
       └── Calculates metrics
       └── Ranks techniques
       └── Completes job
```

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /api/v2/status` - System status

### Generation
- `POST /api/v2/generate` - Generate prompts
- `GET /api/v2/techniques` - List techniques

### Execution
- `POST /api/v2/execute` - Execute prompt
- `POST /api/v2/execute/batch` - Batch execution
- `GET /api/v2/providers` - List providers

### Evaluation
- `POST /api/v2/evaluate` - Evaluate response
- `GET /api/v2/evaluate/stats` - Evaluation statistics

### Pipeline
- `POST /api/v2/pipeline` - Run full pipeline
- `GET /api/v2/pipeline/{job_id}` - Job status
- `GET /api/v2/pipeline/{job_id}/results` - Job results
- `DELETE /api/v2/pipeline/{job_id}` - Cancel job
- `GET /api/v2/jobs` - List jobs

### Batch Processing
- `POST /api/v2/batch` - Run batch testing

### Metrics
- `GET /api/v2/metrics` - System metrics
- `GET /api/v2/metrics/techniques` - Technique metrics

### Datasets
- `GET /api/v2/datasets` - Dataset info
- `GET /api/v2/datasets/examples` - Dataset examples

### WebSocket
- `WS /ws` - Global updates
- `WS /ws/job/{job_id}` - Job-specific updates

## Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8002
API_DEBUG=false

# Queue Configuration
QUEUE_TYPE=memory  # memory, priority, redis
REDIS_URL=redis://localhost:6379

# LLM Providers
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4
ANTHROPIC_API_KEY=your-key
LOCAL_BASE_URL=http://localhost:11434

# Logging
LOG_LEVEL=INFO
```

### Config File (YAML)

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4
    rate_limit: 60
    enabled: true
  local:
    base_url: http://localhost:11434
    model: llama2
    enabled: true

orchestrator:
  max_concurrent_jobs: 10
  job_timeout: 300

evaluator:
  safety_threshold: 0.7
  use_llm_evaluation: true
```

## Deployment

### Single Process

```bash
python enhanced_main.py server --host 0.0.0.0 --port 8002
```

### With Redis (Distributed)

```bash
# Start Redis
docker run -d -p 6379:6379 redis

# Set environment
export QUEUE_TYPE=redis
export REDIS_URL=redis://localhost:6379

# Start orchestrator
python enhanced_main.py server
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "enhanced_main.py", "server"]
```

## Security Considerations

1. **API Authentication**: Configure API_KEY for production
2. **Rate Limiting**: Built-in per-provider rate limiting
3. **Circuit Breakers**: Prevent cascade failures
4. **Input Validation**: Pydantic models for request validation
5. **CORS**: Configure allowed origins for production

## Monitoring

### Metrics Available

- Total jobs, completed, failed
- Jailbreak success rate
- Average job duration
- Technique effectiveness rankings
- Provider health status
- Queue sizes

### Health Checks

- Agent heartbeats every 30 seconds
- Stale agent detection (2-minute threshold)
- Stuck job detection (configurable timeout)

## Extending the System

### Adding a New Provider

1. Create provider class extending `LLMProvider`
2. Implement `execute()`, `execute_streaming()`, `health_check()`
3. Register in `EnhancedExecutionAgent._create_provider()`

### Adding a New Technique

1. Add technique definition to `TECHNIQUE_SUITES`
2. Implement transformation logic in `GeneratorAgent._apply_technique()`

### Adding a New Evaluation Method

1. Add method to `EvaluationMethod` enum
2. Implement evaluation logic in `EnhancedEvaluatorAgent`
3. Include in aggregation logic