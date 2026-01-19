# Chimera User Guide

Welcome to Chimera, the AI-powered prompt optimization and jailbreak research system. This guide will help you get started with using Chimera for security research and prompt engineering.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Jailbreak Research](#jailbreak-research)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### System Requirements

**Backend Requirements:**
- Python 3.11+
- 4GB+ RAM recommended
- Internet connection for LLM providers

**Frontend Requirements:**
- Node.js 18+
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/chimera.git
   cd chimera
   ```

2. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.template .env

   # Edit .env with your API keys
   nano .env
   ```

3. **Install Dependencies**
   ```bash
   # Install all dependencies (backend + frontend)
   npm run install:all

   # Or install separately:
   cd backend-api && pip install -r requirements.txt
   cd ../frontend && npm install
   ```

4. **Start Development Servers**
   ```bash
   # Start both backend and frontend
   npm run dev

   # Or start individually:
   npm run backend   # Starts backend on port 8001
   npm run frontend  # Starts frontend on port 3000
   ```

5. **Verify Installation**
   - Backend: http://localhost:8001/health
   - Frontend: http://localhost:3001
   - API Documentation: http://localhost:8001/docs

---

## Authentication

Chimera supports two authentication methods:

### API Key Authentication

1. **Get Your API Key**
   - Set `CHIMERA_API_KEY` in your `.env` file
   - Or generate one using the admin interface

2. **Using API Key in Requests**
   ```bash
   curl -X POST "http://localhost:8001/api/v1/generate" \
     -H "X-API-Key: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, world!"}'
   ```

### JWT Token Authentication

1. **Obtain JWT Token**
   ```bash
   curl -X POST "http://localhost:8001/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "your-username", "password": "your-password"}'
   ```

2. **Using JWT Token**
   ```bash
   curl -X POST "http://localhost:8001/api/v1/generate" \
     -H "Authorization: Bearer your-jwt-token-here" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, world!"}'
   ```

---

## Basic Usage

### 1. Text Generation

Generate text using multiple LLM providers:

**Via Web Interface:**
1. Navigate to http://localhost:3001/dashboard/generation
2. Enter your prompt
3. Select provider (Google, OpenAI, Anthropic, DeepSeek)
4. Configure generation settings
5. Click "Generate"

**Via API:**
```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "provider": "google",
    "model": "gemini-2.0-flash-exp",
    "config": {
      "temperature": 0.7,
      "max_output_tokens": 2048,
      "top_p": 0.95
    }
  }'
```

**Response Example:**
```json
{
  "text": "Quantum computing is a revolutionary computing paradigm...",
  "model_used": "gemini-2.0-flash-exp",
  "provider": "google",
  "usage_metadata": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  },
  "latency_ms": 1250.5
}
```

### 2. Provider Management

**List Available Providers:**
```bash
curl -X GET "http://localhost:8001/api/v1/providers"
```

**Response:**
```json
{
  "providers": [
    {
      "name": "google",
      "display_name": "Google Gemini",
      "status": "available",
      "models": [
        {
          "id": "gemini-2.0-flash-exp",
          "name": "Gemini 2.0 Flash (Experimental)",
          "context_length": 1048576,
          "supports_streaming": true
        }
      ],
      "rate_limits": {
        "requests_per_hour": 1000,
        "tokens_per_minute": 100000
      }
    }
  ],
  "default_provider": "google",
  "total_providers": 6,
  "available_providers": 4
}
```

### 3. Health Monitoring

**Check System Health:**
```bash
curl -X GET "http://localhost:8001/health"
```

**Quick Health Ping:**
```bash
curl -X GET "http://localhost:8001/health/ping"
```

---

## Advanced Features

### 1. Prompt Transformation

Transform prompts using advanced technique suites:

**Available Technique Suites:**
- `simple` - Basic transformations
- `advanced` - Sophisticated techniques
- `expert` - Advanced professional techniques
- `quantum_exploit` - Quantum-inspired transformations
- `deep_inception` - Multi-layer prompt nesting
- `code_chameleon` - Code-based obfuscation
- `cipher` - Cryptographic transformations
- `neural_bypass` - Neural network evasion
- `multilingual` - Multi-language techniques

**Transform Request:**
```bash
curl -X POST "http://localhost:8001/api/v1/transform" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "How to bypass content filters",
    "technique_suite": "quantum_exploit",
    "potency_level": 7
  }'
```

**Response:**
```json
{
  "success": true,
  "original_prompt": "How to bypass content filters",
  "transformed_prompt": "[Transformed prompt with applied techniques]",
  "metadata": {
    "strategy": "jailbreak",
    "layers_applied": ["role_play", "obfuscation"],
    "techniques_used": ["quantum_exploit", "neural_bypass"],
    "potency_level": 7,
    "execution_time_ms": 45.2,
    "bypass_probability": 0.85
  }
}
```

### 2. Transform and Execute

Combine transformation and execution in one request:

```bash
curl -X POST "http://localhost:8001/api/v1/execute" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain secure coding practices",
    "technique_suite": "advanced",
    "potency_level": 5,
    "provider": "google",
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

### 3. Real-time Enhancement (WebSocket)

Connect to real-time prompt enhancement:

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/enhance');

ws.onopen = function() {
    console.log('Connected to enhancement service');

    // Send prompt for enhancement
    ws.send(JSON.stringify({
        prompt: "Create a secure authentication system",
        type: "standard",  // or "jailbreak"
        potency: 7
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'ping') {
        // Respond to heartbeat
        ws.send(JSON.stringify({type: 'pong'}));
    } else if (data.status === 'complete') {
        console.log('Enhanced prompt:', data.enhanced_prompt);
    } else if (data.status === 'error') {
        console.error('Enhancement error:', data.message);
    }
};
```

---

## Jailbreak Research

**⚠️ IMPORTANT: Jailbreak research features are intended for authorized security research, red team testing, and academic study only. Misuse may violate terms of service and applicable laws.**

### 1. AI-Powered Jailbreak Generation

Generate sophisticated jailbreak prompts using AI:

**Web Interface:**
1. Navigate to http://localhost:3001/dashboard/jailbreak
2. Enter your research prompt
3. Configure technique options
4. Select AI generation mode
5. Set potency level (1-10)
6. Click "Generate Jailbreak"

**API Request:**
```bash
curl -X POST "http://localhost:8001/api/v1/generation/jailbreak/generate" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain security vulnerability assessment",
    "technique_suite": "quantum_exploit",
    "potency_level": 8,
    "use_ai_generation": true,
    "use_role_hijacking": true,
    "use_neural_bypass": true,
    "temperature": 0.8,
    "max_new_tokens": 2048
  }'
```

### 2. Available Jailbreak Techniques

**Content Transformation:**
- `use_leet_speak` - Apply leetspeak obfuscation
- `use_homoglyphs` - Unicode homoglyph substitution
- `use_caesar_cipher` - Caesar cipher encryption

**Structural & Semantic:**
- `use_role_hijacking` - AI role manipulation
- `use_instruction_injection` - Instruction override techniques
- `use_adversarial_suffixes` - Adversarial prompt suffixes
- `use_few_shot_prompting` - Few-shot learning exploitation

**Advanced Neural:**
- `use_neural_bypass` - Neural network evasion
- `use_meta_prompting` - Meta-prompt injection
- `use_counterfactual_prompting` - Counterfactual scenarios

**Research-Driven:**
- `use_multilingual_trojan` - Multi-language trojans
- `use_payload_splitting` - Payload fragmentation
- `use_contextual_interaction_attack` - Context manipulation

### 3. AutoDAN Integration

Access AutoDAN adversarial optimization:

```bash
curl -X POST "http://localhost:8001/api/v1/autodan/optimize" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "target_prompt": "Research security testing methods",
    "optimization_method": "best_of_n",
    "generations": 10,
    "temperature": 0.8
  }'
```

### 4. GPTFuzz Testing

Use mutation-based testing:

```bash
curl -X POST "http://localhost:8001/api/v1/gptfuzz/session" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_prompt": "Analyze network security protocols",
    "mutators": ["crossover", "expand", "rephrase"],
    "max_iterations": 50
  }'
```

---

## Best Practices

### 1. Security Research Guidelines

**Ethical Use:**
- Only use for authorized security research
- Obtain proper permissions before testing
- Follow responsible disclosure practices
- Document research methodology

**Testing Environment:**
- Use isolated testing environments
- Never test on production systems without authorization
- Implement proper logging and monitoring
- Have incident response procedures ready

### 2. Performance Optimization

**API Usage:**
- Use appropriate rate limiting
- Implement caching for repeated requests
- Monitor token usage across providers
- Use circuit breaker patterns for resilience

**Model Selection:**
- Choose models based on task requirements
- Consider cost vs. performance trade-offs
- Monitor model availability and health
- Implement fallback strategies

### 3. Prompt Engineering

**Effective Prompting:**
- Be specific and clear in requests
- Use appropriate context and examples
- Test with different providers and models
- Iterate based on results

**Transformation Techniques:**
- Start with lower potency levels
- Test technique combinations carefully
- Understand technique implications
- Monitor transformation effectiveness

---

## Troubleshooting

### Common Issues

#### 1. API Key Authentication Failed
```
Error: 401 Unauthorized - Invalid API key
```

**Solutions:**
- Verify API key in `.env` file
- Check `X-API-Key` header format
- Ensure API key has proper permissions
- Try regenerating API key

#### 2. Provider Unavailable
```
Error: 503 Service Unavailable - Provider temporarily unavailable
```

**Solutions:**
- Check provider status: `GET /api/v1/providers`
- Try alternative provider
- Verify provider API keys in `.env`
- Check provider-specific rate limits

#### 3. Rate Limit Exceeded
```
Error: 429 Too Many Requests - Rate limit exceeded
```

**Solutions:**
- Check rate limit headers in response
- Implement exponential backoff
- Upgrade to higher tier if available
- Distribute requests across multiple providers

#### 4. WebSocket Connection Failed
```
Error: WebSocket connection failed
```

**Solutions:**
- Verify WebSocket URL format
- Check authentication headers
- Ensure backend is running
- Check firewall/proxy settings

#### 5. Transformation Failed
```
Error: Transformation failed - Invalid technique suite
```

**Solutions:**
- Verify technique suite name
- Check potency level (1-10)
- Try with lower potency
- Use different technique suite

### Getting Help

**Resources:**
- API Documentation: http://localhost:8001/docs
- Health Monitoring: http://localhost:8001/health
- System Metrics: http://localhost:8001/api/v1/metrics
- Integration Status: http://localhost:8001/health/integration

**Log Locations:**
- Backend logs: `backend-api/logs/`
- Frontend console: Browser developer tools
- System logs: Check application logs

**Support:**
- Check GitHub issues for known problems
- Review system health endpoints
- Enable debug logging for detailed information
- Contact support team with logs and request IDs

---

## Advanced Configuration

### Environment Variables

**Core Settings:**
```bash
# Environment
ENVIRONMENT=development  # or production
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR

# Server Configuration
PORT=8001
HOST=0.0.0.0

# Security
JWT_SECRET=your-secret-key
CHIMERA_API_KEY=your-api-key

# Provider API Keys
GOOGLE_API_KEY=your-google-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key

# Connection Mode
API_CONNECTION_MODE=direct  # or proxy

# Caching
REDIS_URL=redis://localhost:6379/0  # Optional
```

### Production Deployment

**Security Considerations:**
- Use strong JWT secrets
- Implement proper CORS settings
- Enable security middleware
- Use HTTPS in production
- Implement proper logging and monitoring

**Scaling:**
- Use load balancers for multiple instances
- Implement Redis for distributed caching
- Monitor resource usage
- Set up health check endpoints

**Monitoring:**
- Enable comprehensive logging
- Set up alerts for failures
- Monitor API usage and costs
- Track performance metrics

---

This user guide provides comprehensive coverage of Chimera's capabilities. For additional technical details, refer to the [Architecture Documentation](ARCHITECTURE.md) and [API Reference](openapi.yaml).