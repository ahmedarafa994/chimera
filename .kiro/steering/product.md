# Product Overview

Chimera is an adversarial prompting and red teaming platform designed for AI security research. It provides sophisticated prompt transformation, jailbreak generation, and LLM testing capabilities.

## Core Capabilities

- **Multi-Provider LLM Integration**: Seamless integration with Google Gemini, OpenAI, Anthropic Claude, and DeepSeek
- **Advanced Prompt Transformation**: 20+ technique suites including quantum exploit, deep inception, code chameleon, and neural bypass
- **AI-Powered Jailbreak Generation**: Sophisticated jailbreak prompt generation using advanced obfuscation and bypass techniques
- **Research Tools**: AutoDAN, AutoDAN-Turbo (ICLR 2025), AutoAdv, GPTFuzz, and HouYi optimization engines
- **Real-time Enhancement**: WebSocket support for live prompt enhancement with heartbeat monitoring
- **Session Management**: Persistent session tracking with model selection and caching

## Architecture

Chimera is a full-stack application with:
- **Backend**: FastAPI-based Python API (`backend-api/`) providing REST and WebSocket endpoints
- **Frontend**: Next.js React application (`frontend/`) with Tailwind CSS and shadcn/ui components
- **Orchestration**: Multi-agent orchestrator (`chimera-orchestrator/`) for coordinating adversarial techniques
- **Meta Prompter**: Shared prompt enhancement library (`meta_prompter/`) used across agents

## Security & Research Focus

This platform is designed for authorized security research and red teaming. All jailbreak and adversarial techniques are intended for testing AI safety measures, not for malicious use.
