# Requirements Document

## Introduction

This specification defines the architectural refactoring of the `backend-api` workspace to eliminate technical debt, consolidate fragmented server implementations, and establish a production-grade FastAPI-based backend. The current state includes multiple redundant server files (Node.js, raw HTTP, Flask-style), inconsistent port configurations, and duplicated code that must be unified into a clean, modular architecture.

## Glossary

- **Backend_API**: The FastAPI-based production backend service located in `backend-api/`
- **Redundant_Files**: Legacy server implementations that duplicate functionality (server.js, simple_server.py, minimal_server.py, working_server.py, etc.)
- **Entry_Point**: The single unified application startup file (`run.py` or `app/main.py`)
- **Technique_Suite**: A named collection of transformation strategies (transformers, framers, obfuscators)
- **LLM_Provider**: An external AI service integration (OpenAI, Anthropic, Google, etc.)

## Requirements

### Requirement 1: Eliminate Redundant Server Implementations

**User Story:** As a developer, I want a single authoritative server implementation, so that I can maintain and extend the codebase without confusion about which file to modify.

#### Acceptance Criteria

1. WHEN the refactoring is complete, THE Backend_API SHALL have exactly one server entry point at `run.py` that starts the FastAPI application.
2. THE Backend_API SHALL NOT contain any Node.js server files (server.js).
3. THE Backend_API SHALL NOT contain legacy HTTP server implementations (simple_server.py, minimal_server.py, working_server.py, minimal_flask_server.py).
4. THE Backend_API SHALL NOT contain duplicate or experimental server files (chimera_server.py, real_ai_server.py, debug_server.py, appmain.py).
5. WHEN a developer runs `python run.py`, THE Backend_API SHALL start on a standardized port (9250) as defined in configuration.

### Requirement 2: Consolidate Configuration and Environment

**User Story:** As a developer, I want a single source of truth for configuration, so that environment settings are consistent and easy to manage.

#### Acceptance Criteria

1. THE Backend_API SHALL use a single `.env` file for environment configuration.
2. THE Backend_API SHALL remove redundant environment template files, keeping only `.env.example` as the canonical template.
3. THE Backend_API SHALL define the default port (9250) in `app/core/config.py` using pydantic-settings.
4. WHEN the README.md is updated, THE Backend_API SHALL accurately reflect the FastAPI framework and correct startup commands.

### Requirement 3: Remove Debug and Verification Scripts

**User Story:** As a developer, I want a clean workspace without temporary debug scripts, so that the codebase only contains production-relevant code.

#### Acceptance Criteria

1. THE Backend_API SHALL NOT contain debug scripts (debug_env.py, debug_providers.py, debug_server.py).
2. THE Backend_API SHALL NOT contain one-off verification scripts (verify_*.py files).
3. THE Backend_API SHALL NOT contain test runner scripts outside the tests/ directory (run_test.py, test_*.py in root).
4. THE Backend_API SHALL NOT contain log files or output artifacts (error_log.txt, output.txt, traceback.txt, test_server.log, security_validation_report.json).
5. THE Backend_API SHALL retain only the `tests/` directory for all test code.

### Requirement 4: Standardize Project Structure

**User Story:** As a developer, I want a well-organized modular structure, so that I can easily navigate and extend the codebase.

#### Acceptance Criteria

1. THE Backend_API SHALL organize all API routes under `app/api/` with versioned subdirectories (v1/, v2/).
2. THE Backend_API SHALL organize all business logic under `app/services/`.
3. THE Backend_API SHALL organize all external integrations under `app/infrastructure/`.
4. THE Backend_API SHALL organize all middleware under `app/middleware/`.
5. THE Backend_API SHALL organize all domain models under `app/domain/`.
6. THE Backend_API SHALL organize all Pydantic schemas in `app/schemas.py` or `app/schemas/` directory.

### Requirement 5: Update Documentation

**User Story:** As a developer, I want accurate documentation, so that I can quickly understand how to run and develop the backend.

#### Acceptance Criteria

1. WHEN the README.md is updated, THE Backend_API SHALL document FastAPI as the framework (not Flask).
2. WHEN the README.md is updated, THE Backend_API SHALL document the correct startup command (`python run.py` or `uvicorn app.main:app`).
3. WHEN the README.md is updated, THE Backend_API SHALL document the standardized port (9250).
4. WHEN the README.md is updated, THE Backend_API SHALL list all available API endpoints with their authentication requirements.
5. THE Backend_API SHALL maintain the JAILBREAK_SYSTEM.md and MULTI_MODEL_SETUP.md documentation files.

### Requirement 6: Ensure Type Safety and Error Handling

**User Story:** As a developer, I want consistent type annotations and error handling, so that the codebase is maintainable and bugs are caught early.

#### Acceptance Criteria

1. THE Backend_API SHALL use Pydantic models for all request/response validation.
2. THE Backend_API SHALL implement centralized error handling in `app/core/errors.py`.
3. THE Backend_API SHALL use Python type hints throughout all modules.
4. WHEN an API error occurs, THE Backend_API SHALL return a consistent JSON error response with status code, message, and timestamp.
