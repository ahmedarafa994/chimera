# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the Chimera project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-0001](0001-multi-provider-llm-architecture.md) | Multi-Provider LLM Architecture | Accepted | 2026-01-01 |
| [ADR-0002](0002-frontend-architecture.md) | Frontend Architecture (Next.js 16 + React 19) | Accepted | 2026-01-01 |
| [ADR-0003](0003-security-architecture.md) | Security Architecture | Accepted | 2026-01-01 |
| [ADR-001](ADR-001-frontend-api-tanstack-query-migration.md) | Frontend API Migration to TanStack Query | Accepted | 2026-01 |
| [ADR-002](ADR-002-autodan-service-consolidation.md) | AutoDAN Service Module Consolidation | Proposed | 2026-01 |
| [ADR-003](ADR-003-jailbreak-api-endpoint-unification.md) | Jailbreak API Endpoint Unification | TBD | 2026-01 |
| [ADR-004](ADR-004-resilience-harmonization.md) | Resilience Pattern Harmonization | TBD | 2026-01 |

## Creating a New ADR

Use this template:

```markdown
# ADR [NUMBER]: [TITLE]

## Status

[Proposed | Accepted | Deprecated | Superseded by ADR-XXX]

## Date

YYYY-MM-DD

## Context

[Describe the context and problem]

## Decision

[Describe the decision and rationale]

## Consequences

### Positive
[List positive outcomes]

### Negative
[List negative outcomes]

## Alternatives Considered

[List alternatives]

## References

[List references]
```

## ADR Status Definitions

- **Proposed**: Under discussion, not yet accepted
- **Accepted**: Decision has been accepted and is in effect
- **Deprecated**: No longer relevant but kept for historical purposes
- **Superseded**: Replaced by a newer ADR

## Related Documentation

- [Coding Standards](../CODING_STANDARDS.md)
- [Production Deployment Guide](../PRODUCTION_DEPLOYMENT_GUIDE.md)
- [API Documentation](../API_DOCUMENTATION.md)
- [Onboarding Guide](../ONBOARDING_GUIDE.md)
