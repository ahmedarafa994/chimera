# Track Spec: Refine Error Handling and Telemetry in Aegis Engine

## Overview
This track focuses on improving the robustness and observability of the Aegis adversarial simulation engine. Currently, campaign execution lacks structured error reporting and deep telemetry, making it difficult to debug failed persona synthesis or payload transformations.

## Objectives
- Implement a structured exception hierarchy for Aegis-specific errors.
- Enhance telemetry logging to capture granular details of persona synthesis and narrative framing.
- Improve real-time feedback in the CLI/API for long-running transformation tasks.

## Technical Details
- **Error Handling:** Define custom exceptions in `backend-api/app/core/errors.py` (or equivalent).
- **Telemetry:** Integrate with existing logging or Prometheus metrics to track transformation success rates and latency.
- **UI/CLI Impact:** Ensure the Aegis campaign telemetry is exposed via the `/api/v1/telemetry` endpoint or printed clearly in `run_aegis.py`.

## Success Criteria
- [ ] Custom Aegis exceptions are implemented and caught during campaign execution.
- [ ] Telemetry logs capture persona details and transformation potency levels.
- [ ] Unit tests verify error handling paths with >80% coverage.
