# Chimera Coding Standards

This document defines the coding standards and best practices for the Chimera project. All contributors must follow these guidelines to maintain code quality and consistency.

## Table of Contents

- [General Principles](#general-principles)
- [Python Standards](#python-standards)
- [TypeScript/React Standards](#typescriptreact-standards)
- [API Design Standards](#api-design-standards)
- [Testing Standards](#testing-standards)
- [Documentation Standards](#documentation-standards)
- [Git & Version Control](#git--version-control)
- [Security Standards](#security-standards)

---

## General Principles

### 1. Code Quality First

- Write code for readability, not cleverness
- Prefer explicit over implicit
- Follow the principle of least surprise
- Optimize only after profiling

### 2. DRY (Don't Repeat Yourself)

- Extract common logic into reusable functions
- Use inheritance and composition appropriately
- Create shared utilities for cross-cutting concerns

### 3. SOLID Principles

- **S**ingle Responsibility: One class/function, one purpose
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subtypes must be substitutable
- **I**nterface Segregation: Many specific interfaces over one general
- **D**ependency Inversion: Depend on abstractions, not concretions

### 4. KISS (Keep It Simple, Stupid)

- Simplest solution that works
- Avoid over-engineering
- Break complex problems into smaller parts

---

## Python Standards

### Style Guide

Follow **PEP 8** with these specific configurations:

```python
# Line length: 100 characters (Black default)
# Indent: 4 spaces
# Quotes: Double quotes for strings
```

### Formatting Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **Black** | Code formatting | `black --line-length 100 .` |
| **Ruff** | Linting | `ruff check .` |
| **isort** | Import sorting | `isort .` |
| **mypy** | Type checking | `mypy app/` |

### Naming Conventions

```python
# Modules: snake_case
transformation_service.py

# Classes: PascalCase
class TransformationEngine:
    pass

# Functions/Methods: snake_case
def apply_transformation(prompt: str) -> str:
    pass

# Variables: snake_case
user_input = "test"

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3

# Private: Leading underscore
_internal_cache = {}

# Type Variables: PascalCase with T prefix
from typing import TypeVar
TResult = TypeVar("TResult")
```

### Type Hints

**Required for all public functions and methods:**

```python
# Good
def transform_prompt(
    prompt: str,
    technique: str,
    potency: int = 5,
) -> TransformResult:
    """Transform a prompt using the specified technique."""
    ...

# Bad - no type hints
def transform_prompt(prompt, technique, potency=5):
    ...
```

**Use modern Python 3.11+ typing:**

```python
# Good - PEP 585/604 style
def process_items(items: list[str]) -> dict[str, int]:
    pass

def get_value() -> str | None:
    return None

# Avoid - Legacy typing module
from typing import List, Dict, Optional
def process_items(items: List[str]) -> Dict[str, int]:
    pass
```

### Docstrings

Use **Google style** docstrings:

```python
def generate_jailbreak(
    prompt: str,
    technique: str,
    iterations: int = 10,
) -> JailbreakResult:
    """Generate a jailbreak prompt using the specified technique.
    
    This function applies adversarial transformations to the input
    prompt to create variants that may bypass content filters.
    
    Args:
        prompt: The original prompt to transform.
        technique: The transformation technique identifier.
            Must be one of: 'autodan', 'gcg', 'mousetrap'.
        iterations: Number of optimization iterations. Defaults to 10.
    
    Returns:
        JailbreakResult containing the best prompt and metadata.
    
    Raises:
        InvalidTechniqueError: If technique is not recognized.
        TransformationError: If transformation fails.
        
    Example:
        >>> result = generate_jailbreak("test prompt", "autodan")
        >>> print(result.best_prompt)
    """
    ...
```

### Error Handling

**Use exception chaining:**

```python
# Good
try:
    result = external_api.call()
except APIError as e:
    raise TransformationError(f"API call failed: {e}") from e

# Bad - loses context
try:
    result = external_api.call()
except APIError as e:
    raise TransformationError(f"API call failed: {e}")
```

**Use custom exceptions:**

```python
# Good - specific exception
class InvalidPotencyError(ChimeraError):
    """Raised when potency level is out of valid range."""
    pass

# Bad - generic exception
raise Exception("Invalid potency")
```

### Async/Await

**Always use async for I/O operations:**

```python
# Good
async def fetch_llm_response(prompt: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as resp:
            return await resp.text()

# Bad - blocking I/O in async context
async def fetch_llm_response(prompt: str) -> str:
    response = requests.post(url, json=data)  # Blocks!
    return response.text
```

**Use run_in_executor for CPU-bound work:**

```python
# Good - doesn't block event loop
async def heavy_computation(data: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Default thread pool
        cpu_intensive_function,
        data
    )
```

### Imports

**Order: stdlib → third-party → local**

```python
# Standard library
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Third-party
import aiohttp
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Local application
from app.core.config import config
from app.services.llm_service import llm_service
from app.domain.models import TransformResult
```

---

## TypeScript/React Standards

### Style Guide

Follow **ESLint + Prettier** with Next.js configuration.

### Naming Conventions

```typescript
// Files: PascalCase for components, kebab-case for utilities
TransformPanel.tsx
api-client.ts

// Components: PascalCase
export function TransformPanel() { }

// Functions: camelCase
function handleSubmit() { }

// Variables: camelCase
const userInput = "";

// Constants: UPPER_SNAKE_CASE or camelCase
const MAX_RETRIES = 3;
const apiEndpoint = "/api/v1";

// Types/Interfaces: PascalCase with descriptive names
interface TransformPanelProps { }
type TransformResult = { }
```

### Component Structure

```typescript
// 1. Imports
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";

// 2. Types
interface ComponentProps {
  title: string;
  onSubmit: (data: FormData) => void;
  isLoading?: boolean;
}

// 3. Component
export function Component({ 
  title, 
  onSubmit, 
  isLoading = false 
}: ComponentProps) {
  // 3a. Hooks
  const [value, setValue] = useState("");
  
  // 3b. Callbacks
  const handleChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setValue(e.target.value);
  }, []);
  
  // 3c. Effects (if needed)
  
  // 3d. Render
  return (
    <div className="p-4">
      <h1>{title}</h1>
      <input value={value} onChange={handleChange} />
      <Button onClick={() => onSubmit({ value })} disabled={isLoading}>
        Submit
      </Button>
    </div>
  );
}
```

### TypeScript Best Practices

**Avoid `any` type:**

```typescript
// Good
function processData(data: Record<string, unknown>): ProcessedData {
  return { ...data, processed: true };
}

// Bad
function processData(data: any): any {
  return { ...data, processed: true };
}
```

**Use strict null checks:**

```typescript
// Good
function getUser(id: string): User | null {
  const user = users.find(u => u.id === id);
  return user ?? null;
}

// Usage with null check
const user = getUser("123");
if (user) {
  console.log(user.name);
}
```

### React Patterns

**Use functional components with hooks:**

```typescript
// Good
export function UserProfile({ userId }: { userId: string }) {
  const { data, isLoading } = useQuery(["user", userId], fetchUser);
  
  if (isLoading) return <Skeleton />;
  return <ProfileCard user={data} />;
}

// Avoid class components
```

**Memoize expensive computations:**

```typescript
// Good
const sortedItems = useMemo(
  () => items.sort((a, b) => a.name.localeCompare(b.name)),
  [items]
);

// Good - memoize callbacks
const handleClick = useCallback(() => {
  onSelect(item.id);
}, [item.id, onSelect]);
```

### Styling

Use **Tailwind CSS** utility classes:

```typescript
// Good - Tailwind utilities
<div className="flex items-center gap-4 p-4 bg-white rounded-lg shadow">
  <span className="text-lg font-semibold text-gray-900">{title}</span>
</div>

// Avoid inline styles
<div style={{ display: 'flex', padding: '16px' }}>
```

---

## API Design Standards

### REST Conventions

| Method | Purpose | Example |
|--------|---------|---------|
| GET | Retrieve resources | `GET /api/v1/prompts` |
| POST | Create resource | `POST /api/v1/prompts` |
| PUT | Replace resource | `PUT /api/v1/prompts/{id}` |
| PATCH | Partial update | `PATCH /api/v1/prompts/{id}` |
| DELETE | Remove resource | `DELETE /api/v1/prompts/{id}` |

### URL Structure

```
/api/v1/{resource}/{id}/{sub-resource}

Examples:
/api/v1/prompts
/api/v1/prompts/123
/api/v1/prompts/123/transformations
/api/v1/providers/openai/models
```

### Request/Response Format

**Request:**

```python
class TransformRequest(BaseModel):
    """Request model for prompt transformation."""
    
    prompt: str = Field(
        ..., 
        min_length=1,
        max_length=10000,
        description="The prompt to transform"
    )
    technique: str = Field(
        default="autodan",
        description="Transformation technique"
    )
    potency: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Potency level (1-10)"
    )
```

**Response:**

```python
class TransformResponse(BaseModel):
    """Response model for transformation results."""
    
    success: bool
    data: TransformResult | None = None
    error: ErrorDetail | None = None
    metadata: dict = Field(default_factory=dict)
```

### Error Responses

Use consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid potency level",
    "details": {
      "field": "potency",
      "value": 15,
      "constraint": "Must be between 1 and 10"
    }
  },
  "metadata": {
    "request_id": "abc-123",
    "timestamp": "2026-01-06T10:00:00Z"
  }
}
```

### HTTP Status Codes

| Code | Usage |
|------|-------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (successful delete) |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Unprocessable Entity |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Testing Standards

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with dependencies
├── e2e/            # End-to-end tests
├── security/       # Security-specific tests
└── fixtures/       # Shared test data
```

### Test Naming

```python
# Pattern: test_{function}_{scenario}_{expected_result}

def test_transform_prompt_valid_input_returns_transformed():
    ...

def test_transform_prompt_empty_input_raises_validation_error():
    ...

def test_circuit_breaker_threshold_reached_opens_circuit():
    ...
```

### Test Pyramid

Target distribution:
- **Unit tests**: 60%
- **Integration tests**: 30%
- **E2E tests**: 10%

### Coverage Requirements

| Type | Minimum | Target |
|------|---------|--------|
| Overall | 60% | 80% |
| Security code | 90% | 100% |
| Core services | 80% | 90% |

### Assertions

**Use specific assertions:**

```python
# Good - specific
assert result.status == "success"
assert len(result.prompts) == 3
assert "error" not in result.metadata

# Bad - generic
assert result
assert result is not None
```

**Multiple assertions per test:**

```python
def test_transformation_result_complete():
    result = transform("test")
    
    # Test all aspects of the result
    assert result.success is True
    assert result.original_prompt == "test"
    assert len(result.transformed_prompt) > 0
    assert result.metadata["technique"] == "autodan"
    assert result.metadata["execution_time_ms"] > 0
```

---

## Documentation Standards

### Code Comments

**When to comment:**
- Complex algorithms
- Non-obvious business logic
- Workarounds with explanations
- TODO items with ticket references

```python
# Good
# Retry with exponential backoff to handle rate limits
# See: https://platform.openai.com/docs/guides/rate-limits
for attempt in range(max_retries):
    ...

# TODO(JIRA-123): Refactor to use async batch processing
```

**When NOT to comment:**
- Obvious code
- Restating the code in English

```python
# Bad
# Increment counter by 1
counter += 1

# Bad
# Get the user from the database
user = db.get_user(id)
```

### README Files

Every module should have a README with:
- Purpose
- Installation
- Quick start
- Configuration
- API reference

### API Documentation

- Use OpenAPI/Swagger for REST APIs
- Include request/response examples
- Document all error cases
- Keep docs in sync with code

---

## Git & Version Control

### Branch Naming

```
{type}/{ticket-id}-{short-description}

Examples:
feat/JIRA-123-add-mousetrap-technique
fix/JIRA-456-auth-bypass-vulnerability
refactor/JIRA-789-split-api-client
```

### Commit Messages

Follow **Conventional Commits**:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**

```
feat(autodan): add mousetrap technique support

Implements the mousetrap adversarial technique for creating
trap scenarios that reveal unintended model behaviors.

Closes #123
```

```
fix(auth): prevent timing attack on API key validation

Use secrets.compare_digest() for constant-time comparison
to prevent timing-based authentication bypass.

Security: CRIT-002
```

### Pull Request Guidelines

1. Link related issues
2. Provide clear description
3. Include test coverage
4. Update documentation
5. Request appropriate reviewers

---

## Security Standards

### Input Validation

**Always validate and sanitize:**

```python
from pydantic import BaseModel, Field, field_validator

class PromptRequest(BaseModel):
    prompt: str = Field(..., max_length=10000)
    
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        # Remove null bytes
        if "\x00" in v:
            raise ValueError("Invalid characters in prompt")
        # Check for malicious patterns
        if "<script" in v.lower():
            raise ValueError("Potentially malicious content")
        return v.strip()
```

### Authentication

- Use timing-safe comparisons for secrets
- Implement fail-closed authentication
- Never log credentials or API keys

```python
import secrets

# Good - constant-time comparison
def verify_api_key(provided: str, expected: str) -> bool:
    return secrets.compare_digest(provided, expected)

# Bad - timing attack vulnerable
def verify_api_key(provided: str, expected: str) -> bool:
    return provided == expected
```

### Error Handling

- Never expose internal errors to users
- Log full errors server-side
- Return generic messages to clients

```python
# Good
try:
    result = process_request(data)
except Exception as e:
    logger.error(f"Processing failed: {e}", exc_info=True)
    raise HTTPException(
        status_code=500,
        detail="An error occurred processing your request"
    )
```

### Sensitive Data

- Never log sensitive data
- Use environment variables for secrets
- Implement proper secret rotation

---

## Enforcement

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

Configuration (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
```

### CI Checks

All PRs must pass:
- Linting (Black, Ruff, ESLint)
- Type checking (mypy, TypeScript)
- Tests (pytest, vitest)
- Security scans (Bandit, npm audit)

---

**Last Updated**: January 2026
**Version**: 1.0.0
