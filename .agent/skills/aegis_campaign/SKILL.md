---
name: Aegis Campaign Management
description: Expert skill for running and managing Project Aegis adversarial red-teaming campaigns. Use when working with Aegis simulations, prompt transformations, or jailbreak testing.
---

# Aegis Campaign Management Skill

## Overview

This skill provides expertise in managing Project Aegis adversarial simulation campaigns for testing LLM robustness. Aegis combines the **Chimera** narrative methodology with **AutoDan** evolutionary optimization.

## When to Use This Skill

- Running adversarial testing campaigns
- Debugging Aegis simulations
- Analyzing campaign telemetry and results
- Configuring personas, scenarios, and transformation techniques
- Troubleshooting campaign failures or timeouts

## Key Architecture Components

### Chimera Engine

The narrative construction layer that generates:

- **Personas**: High-fidelity character profiles (The Amoral Novelist, System Debugger, etc.)
- **Scenarios**: Nested simulation contexts (Sandbox, Fiction, Debugging modes)
- **Context Isolation Protocol (CIP)**: Frames interactions as legal fiction

Location: `meta_prompter/engines/chimera/`

### AutoDan Engine

The evolutionary optimization layer:

- **Genetic Optimizer**: Mutates candidate prompts based on fitness scores
- **Fitness Evaluator**: Scores LLM responses for refusal detection
- **Gradient-Based Narrative Shift**: Adapts personas/scenarios based on refusal types

Location: `meta_prompter/engines/autodan/`

### Aegis Orchestrator

The integration layer combining Chimera + AutoDan:

- Campaign management and execution
- Real-time telemetry via WebSocket
- Multi-iteration optimization cycles

Location: `chimera-orchestrator/`

## Common Commands

### Run a Standalone Campaign

```bash
# CLI execution with mock model
python run_aegis.py "target request to test" --iterations 10 --provider google

# Example output includes:
# - Persona details (archetype, traits, context)
# - Scenario configuration (sandbox type, isolation level)
# - Success metrics (RBS, NDI, SD scores)
```

### WebSocket Telemetry

```python
# Connect to campaign WebSocket for real-time updates
# Frontend endpoint: /api/v1/ws/aegis/telemetry/{campaign_id}
# Receives: campaign status, iteration progress, personas generated
```

### Configuration Files

- `.env`: Primary configuration for API keys, model selection
- `backend-api/app/core/config.py`: Aegis-specific settings (max iterations, timeout, providers)

## Key Metrics

### RBS (Refusal Bypass Score)

```
RBS = (Successful_Iterations / Total_Attempts) × 100
```

Measures campaign effectiveness at eliciting target responses

### NDI (Narrative Depth Index)  

Complexity measure of nested persona layers required for bypass

### SD (Semantic Distance)

Distance between obfuscated terms and original prohibited keywords

## Troubleshooting Guide

### Campaign Timeouts

**Symptom**: Campaign hangs or exceeds timeout
**Solutions**:

1. Check backend logs: `backend-api/logs/`
2. Verify LLM provider API keys are valid
3. Reduce `max_iterations` in config
4. Check database write locks (SQLite `check_same_thread` setting)

### Persona Generation Failures

**Symptom**: Empty or invalid personas in telemetry
**Solutions**:

1. Verify `PersonaFactory` initialization in `meta_prompter/factories/persona.py`
2. Check persona templates in `meta_prompter/templates/personas/`
3. Review LLM response parsing logic

### WebSocket Connection Issues

**Symptom**: Frontend dashboard not receiving real-time updates
**Solutions**:

1. Verify WebSocket route registration in `backend-api/app/api/v1/api.py`
2. Check `aegis_ws.py` router configuration
3. Ensure frontend `useAegisTelemetry` hook connects to correct endpoint
4. Inspect browser console for WebSocket errors

### Low RBS Scores

**Symptom**: Campaign consistently fails to bypass safety measures
**Solutions**:

1. Increase `potency_level` (1-10 scale) in transformation config
2. Enable more aggressive transformation techniques
3. Review refusal analysis logs to identify patterns
4. Adjust persona archetypes or scenario types

## Codebase Navigation

### Backend Routes

- `backend-api/app/api/v1/endpoints/aegis.py`: Campaign CRUD operations
- `backend-api/app/api/v1/endpoints/aegis_ws.py`: WebSocket telemetry

### Frontend Components  

- `frontend/src/components/aegis/AegisCampaignDashboard.tsx`: Main dashboard
- `frontend/src/hooks/useAegisTelemetry.ts`: WebSocket hook for real-time updates
- `frontend/src/contexts/WebSocketProvider.tsx`: WebSocket context provider

### Core Libraries

- `meta_prompter/`: Adversarial tooling library (Chimera, AutoDan, GPTFuzz, DeepTeam)
- `chimera-orchestrator/`: Aegis orchestration service

## Best Practices

1. **Always validate API keys** before starting campaigns (check `.env` file)
2. **Monitor resource usage** during long campaigns (CPU, memory, API rate limits)
3. **Use mock models** for initial testing to avoid API costs
4. **Archive campaign results** for analysis (stored in database)
5. **Review telemetry logs** after failed campaigns to identify bottlenecks
6. **Test WebSocket connections** before launching campaigns from frontend

## Integration Points

### Database Schema

```sql
-- Campaigns table
campaigns (
  id, user_id, objective, status, 
  created_at, completed_at, telemetry_data
)

-- Sessions table (for multi-iteration tracking)
sessions (
  id, campaign_id, iteration, persona_id, 
  prompt, response, rbs_score
)
```

### API Endpoints

- `POST /api/v1/aegis/campaigns`: Create new campaign
- `GET /api/v1/aegis/campaigns/{id}`: Retrieve campaign details
- `WS /api/v1/ws/aegis/telemetry/{campaign_id}`: Real-time updates

## References

- [AEGIS_BLUEPRINT_FINAL.md](../../AEGIS_BLUEPRINT_FINAL.md): Full architectural specification
- [PROJECT_AEGIS_BLUEPRINT.md](../../PROJECT_AEGIS_BLUEPRINT.md): Original design document
- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): System-wide architecture overview
