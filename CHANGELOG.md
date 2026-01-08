# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-01-06

### Added

- **Frontend**: Lazy loading for heavyweight dashboard components (Jailbreak, AutoDAN) to improve initial load time.
- **Frontend**: New Hooks documentation (`docs/HOOKS.md`).
- **Tests**: Comprehensive E2E test suite covering Data Flow, Error Handling, WebSocket, and Session Management.
- **API**: New endpoints for Jailbreak Execution (`/jailbreak/execute`) and validation (`/jailbreak/validate-prompt`).

### Changed

- **Documentation**: Updated `API_DOCUMENTATION.md` with new endpoints and examples.
- **Performance**: Optimized frontend bundle size via code splitting.

### Fixed

- **Integration**: Resolved mismatches between frontend types and backend schemas (Phase 4.4).
- **Reliability**: Enhanced error handling and circuit breaker implementation in frontend.
