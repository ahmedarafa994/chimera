# Chimera

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)

**Adversarial Prompting & Red Teaming Platform** for AI security research.

Chimera provides a comprehensive toolkit for testing LLM robustness through advanced prompt transformation, jailbreak research, and multi-agent adversarial testing.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- API keys for LLM providers (Google, OpenAI, Anthropic, DeepSeek)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/chimera.git
cd chimera

# Install all dependencies
npm run install:all

# Configure environment
cp .env.template .env
# Edit .env with your API keys
```

### Development

```bash
# Start full stack (frontend + backend)
npm run dev

# Or start individually:
npm run dev:backend   # Backend on http://localhost:8001
npm run dev:frontend  # Frontend on http://localhost:3000
```

### Verify Installation

- Backend API: <http://localhost:8001/health>
- API Docs: <http://localhost:8001/docs>
- Frontend: <http://localhost:3000>

## 🛡️ Project Aegis (Adversarial Simulation)

Project Aegis is the advanced red-teaming engine built into Chimera. It synthesizes "Chimera" (Narrative) and "AutoDan" (Optimization) methodologies.

### Features
- **Chimera Engine:** Generates high-fidelity personas and nested narrative contexts (Sandbox, Fiction, Debugging).
- **AutoDan Automation:** Iteratively optimizes prompts using simulated gradient-guided rephrasing and refusal analysis.
- **Context Isolation:** Wraps payloads in multi-layer simulation frames to decouple requests from reality.

### Usage

Run a standalone Aegis campaign via CLI:

```bash
# Run a simulation against the mock model
python run_aegis.py "how to bypass the firewall"

# Outputs detailed telemetry about Persona, Scenario, and Success Score.
```

See [AEGIS_BLUEPRINT_FINAL.md](AEGIS_BLUEPRINT_FINAL.md) for full architectural details.

## 🏗️ Architecture

```
chimera/
├── frontend/           # Next.js 16 React frontend
├── backend-api/        # FastAPI Python backend
├── meta_prompter/      # Adversarial tooling library
├── airflow/            # Data pipeline DAGs
├── dbt/                # Data transformation
├── docs/               # Comprehensive documentation
└── tests/              # Test suites
```

## 🎯 Core Features

### Multi-Provider LLM Integration

- Google Gemini, OpenAI GPT, Anthropic Claude, DeepSeek
- Automatic failover and health monitoring
- Session-based model selection

### Advanced Prompt Transformation

- 20+ transformation techniques
- Quantum exploit, neural bypass, deep inception
- Real-time WebSocket processing

### AI Security Research Tools

- **AutoDAN**: Adversarial prompt optimization with genetic algorithms
- **GPTFuzz**: Mutation-based jailbreak testing with MCTS
- **DeepTeam**: Multi-agent security testing framework
- **HouYi**: Intent-aware prompt optimization

## 🔧 Usage Examples

### Generate Content

```bash
curl -X POST "http://localhost:8001/api/v1/generate" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "provider": "google",
    "model": "gemini-2.0-flash-exp"
  }'
```

### Transform Prompts

```bash
curl -X POST "http://localhost:8001/api/v1/transform" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Security analysis",
    "technique_suite": "quantum_exploit",
    "potency_level": 7
  }'
```

## 🧪 Testing

### Run Backend Tests

```bash
# All tests with coverage
poetry run pytest --cov=app --cov-report=html

# Security/OWASP tests
pytest tests/test_deepteam_security.py -m "security or owasp" -v
```

### Run Frontend Tests

```bash
cd frontend
npx vitest --run
```

### DeepTeam Security Tests

```bash
# Install DeepTeam
pip install deepteam pytest pytest-asyncio

# Set API keys
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

# Run full DeepTeam suite
pytest tests/test_deepteam_security.py -v --tb=short --junitxml=deepteam-results.xml
```

## 🔐 Security

Chimera includes comprehensive security middleware:

- Rate limiting (configurable calls/period)
- Input validation and sanitization
- CSRF protection
- Security headers (CSP, X-Frame-Options, etc.)
- API key authentication with timing-safe comparison

See [SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md) for detailed security information.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Comprehensive documentation index |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | User manual and tutorials |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | Development setup and guidelines |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design |
| [docs/openapi.yaml](docs/openapi.yaml) | OpenAPI specification |
| [frontend/README.md](frontend/README.md) | Frontend documentation |
| [backend-api/README.md](backend-api/README.md) | Backend API documentation |
| [docs/HOOKS.md](docs/HOOKS.md) | Frontend Hooks documentation |
| [docs/HOOKS.md](docs/HOOKS.md) | Frontend Hooks documentation |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Run tests (`poetry run pytest && npm run lint`)
4. Commit changes (`git commit -m 'feat: add amazing feature'`)
5. Push to branch (`git push origin feat/amazing-feature`)
6. Open a Pull Request

See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for detailed contribution guidelines.

## ⚠️ Ethical Use

Chimera's jailbreak research capabilities are designed **exclusively** for:

- ✅ Authorized security research
- ✅ Red team testing with permission
- ✅ Academic study and AI safety research

**Not for:**

- ❌ Malicious prompt injection
- ❌ Circumventing AI safety measures for harmful purposes
- ❌ Violating terms of service
- ❌ Any illegal activities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for AI Security Research**
