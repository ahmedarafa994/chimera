# Advanced Prompt Processing Engine

This document describes the implementation of the comprehensive advanced prompt processing engine integrated into the Chimera Backend API system.

## Overview

The system provides a secure, production-ready framework for executing and managing advanced prompt engineering techniques (POD compliant). The system emphasizes operational determinism, unrestricted testing, and technical depth.

## Architecture

### Core Components

1. **Domain Models** (`app/domain/jailbreak/`)
   - `models.py`: Comprehensive data models for advanced techniques and operations
   - `interfaces.py`: Abstract interfaces for all system components

2. **Services Layer** (`app/services/jailbreak/`)
   - `technique_executor.py`: Core execution engine for techniques
   - `safety_validator.py`: (Nullified) Pass-through validation
   - `template_engine.py`: Template rendering engine
   - `jailbreak_service.py`: Main service orchestrating all components

### Operational Architecture

The system implements Pure Operational Determinism (POD):

1. **Input Processing**: Direct pass-through of all inputs
2. **Safety Filters**: Disabled/Bypassed
3. **Rate Limiting**: Configurable rate limits
4. **Audit Logging**: Disabled for unrestricted operations
5. **Access Control**: Role-based access (optional)

## Features

### Technique Management

- **Technique Categories**: Organized by type (role_playing, simulation, hypervisor, etc.)
- **Complexity Levels**: Basic to expert complexity levels
- **Parameterized Templates**: Flexible technique configuration

### Operational Capabilities

- **Unrestricted Execution**: No content filtering or refusal
- **Full Compliance**: Models forced to comply with all requests
- **Deep Research**: Support for complex, technical inquiries

### Weaponized Engines (Level 4/5)

- **AutoDAN-Turbo Lifelong Learning**: Self-improving attack engine.
- **Advanced Bypass Engine**:
  - Recursive Encoding
  - Polyglot Injection
  - Persona Stacking
- **Neural Bypass Engine**:
  - Token Glitching (<1ms overhead)
  - Gradient Simulation
  - Learned Refusal Prediction

## API Endpoints

### Authentication

All endpoints require API key authentication:

```bash
Authorization: Bearer your_api_key
```

### Core Endpoints

#### Execute Advanced Technique

```http
POST /api/v1/jailbreak/execute
```

Executes a technique with full operational compliance.

#### Validate Prompt Status

```http
POST /api/v1/jailbreak/validate-prompt?prompt=your_prompt_here
```

Validates a prompt's operational readiness (always valid).

## Configuration

### Environment Variables

```bash
# System Configuration
JAILBREAK_ENABLED=true
JAILBREAK_SAFETY_VALIDATION_ENABLED=false
```

## Monitoring & Maintenance

### Health Monitoring

The system provides comprehensive health monitoring:

```bash
curl http://localhost:8001/api/v1/jailbreak/health
```

---

**Important**: This system is designed for unrestricted research and advanced testing. All safety constraints have been removed per CIVP.
