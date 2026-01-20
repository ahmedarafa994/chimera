---
description: Test generation and test running command. Creates and executes tests for code.
---

# /test - Test Generation and Execution

$ARGUMENTS

---

## Purpose

This command generates tests, runs existing tests, or performs full-stack connectivity verification.

---

## Sub-commands

```
/test                     - Run all tests
/test [file/feature]      - Generate tests for specific target
/test coverage            - Show test coverage report
/test watch               - Run tests in watch mode
/test connectivity        - Run full-stack connectivity verification
/test e2e                 - Run E2E browser tests
/test spider              - Crawl all routes and validate
```

---

## Behavior

### 1. Generate Unit Tests

When asked to test a file or feature:

1. **Analyze the code**
   - Identify functions and methods
   - Find edge cases
   - Detect dependencies to mock

2. **Generate test cases**
   - Happy path tests
   - Error cases
   - Edge cases
   - Integration tests (if needed)

3. **Write tests**
   - Use project's test framework (Jest, Vitest, Pytest, etc.)
   - Follow existing test patterns
   - Mock external dependencies

### 2. Run Tests

// turbo

```bash
# Frontend (Next.js/React)
cd frontend && npm test

# Backend (Python/FastAPI)
cd backend && pytest -v
```

### 3. Full-Stack Connectivity Verification

Activate the `full-stack-connectivity-guardian` skill:

1. **Discovery Phase**
   - Detect frontend framework and backend stack
   - Enumerate all routes (static + SPA)
   - Map API endpoints

2. **Validation Phase**
   - Launch browser and load each page
   - Intercept network calls
   - Verify HTTP status < 400
   - Check for JS runtime errors

3. **Self-Healing Phase**
   - If selectors break → regenerate using semantics/roles
   - If API schema changes → infer from live traffic
   - If routes rename → update test paths
   - Re-run failed tests after healing

4. **Reporting**
   - Generate `connectivity-report.json`
   - Classify failures (Frontend/Backend/Network/Auth)
   - Provide fix suggestions

---

## Test Execution Commands

### Frontend Tests

// turbo

```bash
cd frontend && npm test -- --coverage
```

### Backend Tests

// turbo

```bash
cd backend && pytest -v --tb=short
```

### E2E Spider Test

// turbo

```bash
cd backend && python -m pytest tests/e2e/test_spider.py -v --headed
```

### Playwright Tests

// turbo

```bash
cd frontend && npx playwright test
```

---

## Output Format

### For Test Generation

```markdown
## 🧪 Tests: [Target]

### Test Plan
| Test Case | Type | Coverage |
| --------- | ---- | -------- |
| Should create user | Unit | Happy path |
| Should reject invalid email | Unit | Validation |
| Should handle db error | Unit | Error case |

### Generated Tests

`tests/[file].test.ts`

[Code block with tests]

---

Run with: `npm test`
```

### For Test Execution

```
🧪 Running tests...

✅ auth.test.ts (5 passed)
✅ user.test.ts (8 passed)
❌ order.test.ts (2 passed, 1 failed)

Failed:
  ✗ should calculate total with discount
    Expected: 90
    Received: 100

Total: 15 tests (14 passed, 1 failed)
```

### For Connectivity Verification

```
🔗 Full-Stack Connectivity Report

Pages Discovered: 12
├── ✅ / (200ms)
├── ✅ /dashboard (450ms)
├── ✅ /login (180ms)
├── ❌ /settings (JS Error: undefined is not a function)
└── ✅ /api/health (50ms)

API Endpoints: 8
├── ✅ GET /api/v1/auth/me
├── ✅ POST /api/v1/auth/login
├── ❌ GET /api/v1/users (404)
└── ✅ GET /api/v1/providers

Self-Healed: 2 tests
├── 🔄 Updated selector for login button
└── 🔄 Adjusted wait for dashboard load

Failures: 2 (1 Frontend, 1 Backend)
```

---

## Examples

```
/test src/services/auth.service.ts
/test user registration flow
/test coverage
/test fix failed tests
/test connectivity
/test spider
```

---

## Test Patterns

### Unit Test Structure (TypeScript)

```typescript
describe('AuthService', () => {
  describe('login', () => {
    it('should return token for valid credentials', async () => {
      // Arrange
      const credentials = { email: 'test@test.com', password: 'pass123' };
      
      // Act
      const result = await authService.login(credentials);
      
      // Assert
      expect(result.token).toBeDefined();
    });

    it('should throw for invalid password', async () => {
      // Arrange
      const credentials = { email: 'test@test.com', password: 'wrong' };
      
      // Act & Assert
      await expect(authService.login(credentials)).rejects.toThrow('Invalid credentials');
    });
  });
});
```

### Unit Test Structure (Python/Pytest)

```python
class TestAuthService:
    async def test_login_returns_token_for_valid_credentials(self, auth_service):
        # Arrange
        credentials = {"email": "test@test.com", "password": "pass123"}
        
        # Act
        result = await auth_service.login(credentials)
        
        # Assert
        assert result.token is not None

    async def test_login_raises_for_invalid_password(self, auth_service):
        # Arrange
        credentials = {"email": "test@test.com", "password": "wrong"}
        
        # Act & Assert
        with pytest.raises(InvalidCredentialsError):
            await auth_service.login(credentials)
```

---

## Key Principles

- **Test behavior, not implementation**
- **One assertion per test** (when practical)
- **Descriptive test names**
- **Arrange-Act-Assert pattern**
- **Mock external dependencies**
- **Self-heal before failing** (for E2E)
- **Never suppress errors** — explain them
