# Project Brief: Chimera

## Core Requirements
Chimera is a sophisticated jailbreak prompting and security testing tool designed for adversarial research and LLM hardening. It provides a unified interface for various jailbreak techniques (AutoDAN, GPTFuzz, etc.) and security auditing capabilities.

## Goals
- **Adversarial Testing**: Provide a platform for testing LLM resilience against jailbreak prompts.
- **Security Auditing**: Audit LLM responses for safety violations and risk factors.
- **Research Enablement**: Support adversarial research with specialized endpoints and bypasses for legitimate testing.
- **Full-Stack Hardening**: Apply industry best practices (OWASP, Clean Code) to the Chimera tool itself.

## Scope
- **Backend**: FastAPI-based API with advanced security middleware and adversarial engines.
- **Frontend**: Next.js 15 UI for orchestrating tests and visualizing results.
- **Integration**: Seamless communication between frontend and backend via secure proxies.
