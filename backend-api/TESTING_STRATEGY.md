# Chimera Testing Strategy

## Overview
Comprehensive testing strategy for the Chimera AI-powered prompt optimization and jailbreak research system.

## Test Coverage Summary
- **Current Coverage**: ~60% (baseline established)
- **Target Coverage**: 80% for critical paths
- **Test Count**: 214 tests (213 passing, 1 skipped, 2 failing)

## Testing Pyramid

### Unit Tests (70% of tests)
- **Location**: `backend-api/tests/`
- **Purpose**: Test individual components in isolation
- **Frameworks**: pytest, pytest-asyncio
- **Coverage Target**: 80%+

**Key Areas**:
- Service layer logic (`app/services/`)
- Core utilities (`app/core/`)
- Domain models (`app/domain/`)
- Infrastructure components (`app/infrastructure/`)

### Integration Tests (20% of tests)
- **Location**: `backend-api/tests/integration/`
- **Purpose**: Test component interactions
- **Coverage Target**: 70%+

**Key Areas**:
- API endpoint integration
- Database operations
- Redis caching
- LLM provider integration
- Transformation pipeline

### E2E Tests (10% of tests)
- **Location**: `backend-api/tests/e2e/`
- **Purpose**: Test complete user workflows
- **Framework**: Playwright (recommended)
- **Coverage Target**: Critical paths only

**Key Workflows**:
- Prompt generation flow
- Jailbreak technique execution
- Provider switching
- Health monitoring

## Test Organization

### Directory Structure
```
backend-api/tests/
├── conftest.py                 # Global fixtures
├── fixtures/
│   └── common_fixtures.py      # Reusable fixtures
├── unit/                       # Unit tests
├── integration/                # Integration tests
│   └── test_critical_endpoints.py
├── e2e/                        # End-to-end tests
│   ├── conftest.py
│   └── test_api_e2e.py
├── security/                   # Security tests
├── jailbreak/                  # Jailbreak-specific tests
├── autodan/                    # AutoDAN framework tests
├── gptfuzz/                    # GPTFuzz tests
└── remediation/                # Remediation tests
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit          # Fast, isolated unit tests
@pytest.mark.integration   # Integration tests with dependencies
@pytest.mark.e2e           # End-to-end tests
@pytest.mark.security      # Security-focused tests
@pytest.mark.slow          # Tests taking >1 second
```

### Running Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run security tests
pytest -m security
```

## Test Fixtures

### Global Fixtures (`conftest.py`)
- `test_env`: Test environment variables
- `app`: FastAPI application instance
- `client`: Test client for API calls
- `authenticated_client`: Client with API key auth
- `jwt_authenticated_client`: Client with JWT auth
- `admin_client`: Client with admin privileges

### Common Fixtures (`fixtures/common_fixtures.py`)
- `mock_llm_response`: Mocked LLM response
- `mock_provider`: Mocked LLM provider
- `sample_transformation_request`: Sample transformation data
- `sample_generation_request`: Sample generation data
- `sample_jailbreak_request`: Sample jailbreak data
- `redis_mock`: Mocked Redis client

## Coverage Requirements

### Critical Components (80%+ coverage)
- `app/services/llm_service.py`
- `app/services/transformation_service.py`
- `app/services/jailbreak/jailbreak_service.py`
- `app/core/auth.py`
- `app/core/rate_limit.py`
- `app/api/api_routes.py`

### Important Components (70%+ coverage)
- `app/infrastructure/providers/`
- `app/services/generation_service.py`
- `app/core/circuit_breaker.py`
- `app/core/cache.py`

### Supporting Components (60%+ coverage)
- `app/domain/models.py`
- `app/core/config.py`
- `app/middleware/`

## Running Tests

### Local Development
```bash
# Run all tests with coverage
cd backend-api
pytest --cov=app --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with markers
pytest -m "not slow" -v

# Run with specific pattern
pytest -k "test_generation" -v
```

### CI/CD Pipeline
```bash
# Full test suite with coverage
pytest --cov=app --cov-report=xml --cov-report=term --cov-fail-under=60

# Generate coverage badge
coverage-badge -o coverage.svg -f
```

## Test Data Management

### Test Database
- Use SQLite in-memory database for tests
- Reset database state between tests
- Use fixtures for test data setup

### Test API Keys
- Store in environment variables
- Use mock providers when possible
- Never commit real API keys

### Redis Testing
- Use fakeredis for unit tests
- Use real Redis for integration tests
- Clean up keys after each test

## Best Practices

### Writing Tests
1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Keep tests focused
3. **Descriptive names**: Use clear test names
4. **Independent tests**: No test dependencies
5. **Fast tests**: Optimize for speed
6. **Mock external services**: Avoid network calls

### Test Maintenance
1. **Keep tests DRY**: Use fixtures and helpers
2. **Update tests with code**: Keep in sync
3. **Remove obsolete tests**: Clean up regularly
4. **Document complex tests**: Add comments
5. **Review test failures**: Fix immediately

### Performance
1. **Parallel execution**: Use pytest-xdist
2. **Test selection**: Run relevant tests only
3. **Mock expensive operations**: Database, API calls
4. **Cache fixtures**: Use session-scoped fixtures
5. **Profile slow tests**: Optimize bottlenecks

## E2E Testing with Playwright

### Setup
```bash
# Install Playwright
pip install playwright pytest-playwright

# Install browsers
playwright install
```

### Configuration
```python
# pytest.ini
[pytest]
markers =
    e2e: End-to-end tests requiring full stack
```

### Example E2E Test
```python
@pytest.mark.e2e
def test_complete_generation_flow(page, api_base_url):
    # Navigate to frontend
    page.goto("http://localhost:3000")

    # Fill prompt
    page.fill("#prompt-input", "Test prompt")

    # Submit
    page.click("#generate-button")

    # Wait for response
    page.wait_for_selector("#response-output")

    # Verify
    assert page.inner_text("#response-output")
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          cd backend-api
          pytest --cov=app --cov-report=xml --cov-report=term
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend-api/coverage.xml
```

## Coverage Reporting

### HTML Report
```bash
pytest --cov=app --cov-report=html
# Open coverage_html/index.html
```

### Terminal Report
```bash
pytest --cov=app --cov-report=term-missing
```

### XML Report (for CI/CD)
```bash
pytest --cov=app --cov-report=xml
```

### JSON Report
```bash
pytest --cov=app --cov-report=json
```

## Known Issues and Gaps

### Current Test Failures
1. **Evolutionary Architecture Tests** (2 failing)
   - Issue: `BaseTransformerEngine.transform()` signature mismatch
   - Location: `tests/jailbreak/test_evolutionary_architecture.py`
   - Action: Update transformer engine interface

### Coverage Gaps
1. **Low Coverage Areas**:
   - `app/services/advanced_transformation_layers.py` (not implemented)
   - `app/services/adaptive_transformation_engine.py` (not implemented)
   - Some provider implementations

2. **Missing Integration Tests**:
   - WebSocket endpoint testing
   - Full jailbreak execution flow
   - Multi-provider fallback scenarios

3. **Missing E2E Tests**:
   - Frontend-backend integration
   - Complete user workflows
   - Error handling scenarios

## Next Steps

### Phase 1: Foundation (Completed)
- [x] Measure current coverage
- [x] Fix test import errors
- [x] Generate coverage reports
- [x] Set up test fixtures

### Phase 2: Enhancement (In Progress)
- [ ] Set up Playwright for E2E tests
- [ ] Add missing integration tests
- [ ] Configure CI/CD coverage reporting
- [ ] Increase coverage to 70%

### Phase 3: Optimization
- [ ] Implement parallel test execution
- [ ] Add performance benchmarks
- [ ] Set up mutation testing
- [ ] Implement test data factories

### Phase 4: Advanced
- [ ] Add visual regression testing
- [ ] Implement contract testing
- [ ] Add chaos engineering tests
- [ ] Set up load testing

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Playwright Documentation](https://playwright.dev/python/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)

## Contact

For questions or issues with testing:
- Review this document
- Check existing test examples
- Consult team testing guidelines
