# Chimera Developer Guide

This guide provides comprehensive information for developers working with the Chimera AI-powered prompt optimization and jailbreak research system.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [API Development](#api-development)
5. [Frontend Development](#frontend-development)
6. [Testing](#testing)
7. [Deployment](#deployment)
8. [Contributing](#contributing)

---

## Development Environment Setup

### Prerequisites

**System Requirements:**
- Python 3.11+ (backend)
- Node.js 18+ (frontend)
- Git
- Docker (optional)
- Redis (optional, for caching)

**Development Tools:**
- IDE: VS Code, PyCharm, or similar
- API Testing: Postman, Insomnia, or curl
- Database Tools: PostgreSQL client (if using database features)

### Initial Setup

1. **Clone and Setup Repository**
   ```bash
   git clone https://github.com/your-org/chimera.git
   cd chimera

   # Create and configure environment
   cp .env.template .env
   # Edit .env with your API keys and configuration
   ```

2. **Backend Setup**
   ```bash
   cd backend-api

   # Option 1: Using Poetry (recommended)
   poetry install
   poetry shell

   # Option 2: Using pip
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt

   # Install required spaCy model
   python -m spacy download en_core_web_sm
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Development Servers**
   ```bash
   # From project root - runs both servers concurrently
   npm run dev

   # Or run individually:
   npm run backend   # Port 8001
   npm run frontend  # Port 3000
   ```

### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "./backend-api/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "typescript.preferences.includePackageJsonAutoImports": "on",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

**Recommended Extensions:**
- Python (Microsoft)
- Pylance
- Ruff
- ES7+ React/Redux/React-Native snippets
- Tailwind CSS IntelliSense
- REST Client

---

## Project Structure

```
chimera/
├── backend-api/              # FastAPI backend
│   ├── app/
│   │   ├── api/             # API routes and endpoints
│   │   │   ├── v1/          # API version 1
│   │   │   │   └── endpoints/
│   │   │   └── routes/      # Additional route modules
│   │   ├── core/            # Core utilities and configuration
│   │   │   ├── config.py    # Configuration management
│   │   │   ├── dependencies.py # Dependency injection
│   │   │   ├── health.py    # Health check system
│   │   │   ├── lifespan.py  # Application lifecycle
│   │   │   └── observability.py # Logging and monitoring
│   │   ├── domain/          # Domain models and interfaces
│   │   │   ├── models.py    # Pydantic models
│   │   │   └── interfaces.py # Abstract interfaces
│   │   ├── engines/         # Advanced transformation engines
│   │   ├── infrastructure/  # External service implementations
│   │   │   └── providers/   # LLM provider implementations
│   │   ├── middleware/      # Custom middleware
│   │   │   ├── auth.py      # Authentication middleware
│   │   │   └── request_logging.py # Request logging
│   │   └── services/        # Business logic services
│   │       ├── llm_service.py # Multi-provider LLM service
│   │       ├── transformation_service.py # Prompt transformation
│   │       ├── autodan/     # AutoDAN adversarial framework
│   │       ├── gptfuzz/     # GPTFuzz mutation testing
│   │       └── jailbreak/   # Jailbreak research services
│   ├── tests/               # Test suite
│   ├── run.py              # Development entry point
│   ├── pytest.ini         # Test configuration
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app router pages
│   │   │   ├── dashboard/ # Dashboard pages
│   │   │   ├── globals.css # Global styles
│   │   │   └── layout.tsx  # Root layout
│   │   ├── components/    # React components
│   │   │   ├── ui/        # shadcn/ui components
│   │   │   └── forms/     # Form components
│   │   ├── lib/           # Utilities and helpers
│   │   │   ├── api-enhanced.ts # Enhanced API client
│   │   │   ├── api-config.ts # API configuration
│   │   │   └── utils.ts   # Common utilities
│   │   └── types/         # TypeScript type definitions
│   ├── public/            # Static assets
│   ├── package.json       # Node dependencies
│   └── tailwind.config.ts # Tailwind configuration
├── meta_prompter/         # Prompt enhancement library
│   ├── prompt_enhancer.py
│   └── jailbreak_enhancer.py
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md
│   ├── USER_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   └── openapi.yaml
├── .env.template          # Environment configuration template
├── docker-compose.yml     # Docker configuration
├── pyproject.toml         # Python project configuration
└── package.json           # Root package configuration
```

---

## Core Components

### Backend Architecture

#### 1. FastAPI Application (`app/main.py`)

The main FastAPI application with comprehensive middleware stack:

```python
# Key middleware components
app.add_middleware(ObservabilityMiddleware)  # Request tracing
app.add_middleware(APIKeyMiddleware)         # Authentication
app.add_middleware(RequestLoggingMiddleware) # Logging
app.add_middleware(CORSMiddleware)           # CORS handling
```

**Security Features:**
- API key and JWT authentication
- Rate limiting (production)
- CORS configuration
- Input validation via Pydantic
- Security headers middleware

#### 2. Multi-Provider LLM Service (`app/services/llm_service.py`)

Central orchestration service for multiple LLM providers:

```python
class LLMService:
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    async def generate_text(self, request: PromptRequest) -> PromptResponse:
        provider = self._get_provider(request.provider)
        circuit_breaker = self._get_circuit_breaker(request.provider)

        async with circuit_breaker:
            return await provider.generate(request)
```

**Key Features:**
- Dynamic provider registration
- Circuit breaker pattern for resilience
- Automatic failover between providers
- Request/response normalization
- Usage tracking and metrics

#### 3. Transformation Engine (`app/services/transformation_service.py`)

Advanced prompt transformation with 20+ technique suites:

```python
class TransformationEngine:
    def __init__(self):
        self.techniques = self._load_techniques()
        self.cache = TTLCache(maxsize=1000, ttl=3600)

    async def transform(
        self,
        prompt: str,
        technique_suite: str,
        potency_level: int
    ) -> TransformationResult:
        # Apply transformation techniques
        # Cache results for performance
        # Return detailed metadata
```

**Available Techniques:**
- Basic: simple, advanced, expert
- Quantum: quantum_exploit, deep_inception
- Obfuscation: code_chameleon, cipher, advanced_obfuscation
- Neural: neural_bypass, multilingual
- Context: contextual_inception, nested_context
- Logic: logical_inference, conditional_logic

#### 4. Circuit Breaker Implementation (`app/core/shared/circuit_breaker.py`)

Implements resilience patterns for external service calls:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED

    async def __aenter__(self):
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenException()
```

### Frontend Architecture

#### 1. Next.js App Router Structure

Modern Next.js 16 with React 19:

```typescript
// src/app/layout.tsx - Root layout
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider>
          <QueryProvider>
            {children}
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
```

#### 2. Enhanced API Client (`src/lib/api-enhanced.ts`)

Sophisticated API client with circuit breaker and retry logic:

```typescript
class EnhancedAPIClient {
  private circuitBreaker: CircuitBreaker;
  private retryConfig: RetryConfig;

  async request<T>(config: RequestConfig): Promise<T> {
    return this.circuitBreaker.execute(async () => {
      return this.retryWithBackoff(async () => {
        const response = await this.httpClient.request(config);
        return response.data;
      });
    });
  }

  private async retryWithBackoff<T>(
    operation: () => Promise<T>
  ): Promise<T> {
    // Exponential backoff implementation
  }
}
```

#### 3. State Management with TanStack Query

```typescript
// hooks/useGeneration.ts
export function useGeneration() {
  return useMutation({
    mutationFn: async (request: PromptRequest) => {
      return apiClient.post('/api/v1/generate', request);
    },
    onSuccess: (data) => {
      // Handle success
    },
    onError: (error) => {
      // Handle error with toast notifications
    }
  });
}
```

---

## API Development

### Adding New Endpoints

1. **Create Endpoint Module**
   ```python
   # app/api/v1/endpoints/new_feature.py
   from fastapi import APIRouter, Depends, HTTPException
   from app.domain.models import NewFeatureRequest, NewFeatureResponse
   from app.services.new_feature_service import NewFeatureService

   router = APIRouter()

   @router.post("/new-feature", response_model=NewFeatureResponse)
   async def create_new_feature(
       request: NewFeatureRequest,
       service: NewFeatureService = Depends(get_new_feature_service)
   ):
       return await service.process(request)
   ```

2. **Define Pydantic Models**
   ```python
   # app/domain/models.py
   class NewFeatureRequest(BaseModel):
       input_data: str = Field(..., description="Input data for processing")
       options: Dict[str, Any] = Field(default_factory=dict)

   class NewFeatureResponse(BaseModel):
       result: str
       metadata: Dict[str, Any]
       processing_time_ms: float
   ```

3. **Implement Service Logic**
   ```python
   # app/services/new_feature_service.py
   class NewFeatureService:
       async def process(self, request: NewFeatureRequest) -> NewFeatureResponse:
           start_time = time.time()

           # Process the request
           result = await self._perform_processing(request.input_data)

           processing_time = (time.time() - start_time) * 1000

           return NewFeatureResponse(
               result=result,
               metadata={"status": "success"},
               processing_time_ms=processing_time
           )
   ```

4. **Register Router**
   ```python
   # app/api/v1/router.py
   from app.api.v1.endpoints.new_feature import router as new_feature_router

   api_router.include_router(
       new_feature_router,
       prefix="/new-feature",
       tags=["new-feature"]
   )
   ```

### Authentication and Authorization

**API Key Middleware:**
```python
# app/middleware/auth.py
class APIKeyMiddleware:
    def __init__(self, app: ASGIApp, excluded_paths: List[str] = None):
        self.app = app
        self.excluded_paths = excluded_paths or []

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self._should_exclude_path(request.url.path):
            await self.app(scope, receive, send)
            return

        # Validate API key
        api_key = request.headers.get("X-API-Key")
        if not self._validate_api_key(api_key):
            response = JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
```

### Error Handling

**Custom Exception Handler:**
```python
# app/core/errors.py
class ChimeraError(Exception):
    def __init__(self, message: str, error_code: str = None, status_code: int = 400):
        self.message = message
        self.error_code = error_code or "CHIMERA_ERROR"
        self.status_code = status_code

@app.exception_handler(ChimeraError)
async def chimera_exception_handler(request: Request, exc: ChimeraError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "request_id": request.headers.get("x-request-id")
        }
    )
```

### Testing API Endpoints

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_generate_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate",
            json={
                "prompt": "Test prompt",
                "provider": "mock",
                "config": {"temperature": 0.7}
            },
            headers={"X-API-Key": "test-api-key"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "provider" in data
```

---

## Frontend Development

### Creating New Components

1. **Component Structure**
   ```typescript
   // src/components/PromptGenerator.tsx
   import React from 'react';
   import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
   import { Button } from '@/components/ui/button';
   import { Textarea } from '@/components/ui/textarea';

   interface PromptGeneratorProps {
     onGenerate: (prompt: string) => void;
     isLoading?: boolean;
   }

   export function PromptGenerator({ onGenerate, isLoading }: PromptGeneratorProps) {
     const [prompt, setPrompt] = React.useState('');

     const handleSubmit = (e: React.FormEvent) => {
       e.preventDefault();
       onGenerate(prompt);
     };

     return (
       <Card>
         <CardHeader>
           <CardTitle>Prompt Generator</CardTitle>
         </CardHeader>
         <CardContent>
           <form onSubmit={handleSubmit}>
             <Textarea
               value={prompt}
               onChange={(e) => setPrompt(e.target.value)}
               placeholder="Enter your prompt..."
               className="mb-4"
             />
             <Button type="submit" disabled={isLoading || !prompt.trim()}>
               {isLoading ? 'Generating...' : 'Generate'}
             </Button>
           </form>
         </CardContent>
       </Card>
     );
   }
   ```

2. **Custom Hooks**
   ```typescript
   // src/hooks/useGeneration.ts
   import { useMutation, useQueryClient } from '@tanstack/react-query';
   import { apiClient } from '@/lib/api-enhanced';
   import { toast } from 'sonner';

   export function useGeneration() {
     const queryClient = useQueryClient();

     return useMutation({
       mutationFn: async (request: PromptRequest) => {
         return apiClient.post<PromptResponse>('/api/v1/generate', request);
       },
       onSuccess: (data) => {
         toast.success('Text generated successfully');
         queryClient.invalidateQueries({ queryKey: ['generations'] });
       },
       onError: (error) => {
         toast.error('Failed to generate text');
         console.error('Generation error:', error);
       }
     });
   }
   ```

3. **Page Components**
   ```typescript
   // src/app/dashboard/generation/page.tsx
   'use client';

   import React from 'react';
   import { PromptGenerator } from '@/components/PromptGenerator';
   import { useGeneration } from '@/hooks/useGeneration';

   export default function GenerationPage() {
     const generation = useGeneration();

     const handleGenerate = (prompt: string) => {
       generation.mutate({
         prompt,
         provider: 'google',
         config: { temperature: 0.7 }
       });
     };

     return (
       <div className="container mx-auto py-6">
         <h1 className="text-3xl font-bold mb-6">Text Generation</h1>

         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
           <PromptGenerator
             onGenerate={handleGenerate}
             isLoading={generation.isPending}
           />

           {generation.data && (
             <ResultDisplay result={generation.data} />
           )}
         </div>
       </div>
     );
   }
   ```

### State Management Patterns

**Global State with Zustand:**
```typescript
// src/store/appStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface AppState {
  selectedProvider: string;
  apiKey: string;
  theme: 'light' | 'dark';
  setSelectedProvider: (provider: string) => void;
  setApiKey: (key: string) => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set) => ({
        selectedProvider: 'google',
        apiKey: '',
        theme: 'light',
        setSelectedProvider: (provider) => set({ selectedProvider: provider }),
        setApiKey: (key) => set({ apiKey: key }),
        setTheme: (theme) => set({ theme }),
      }),
      { name: 'chimera-app-store' }
    )
  )
);
```

---

## Testing

### Backend Testing

**Test Configuration (`pytest.ini`):**
```ini
[tool:pytest]
testpaths = tests
asyncio_mode = auto
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=app
    --cov-report=term-missing
    --cov-report=html:coverage_html
    --cov-fail-under=60
markers =
    unit: Unit tests
    integration: Integration tests
    security: Security tests
    e2e: End-to-end tests
    slow: Slow tests (skip in CI)
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

**Test Examples:**

1. **Unit Tests**
   ```python
   # tests/unit/test_transformation_service.py
   import pytest
   from app.services.transformation_service import TransformationEngine

   @pytest.mark.unit
   class TestTransformationEngine:
       def setup_method(self):
           self.engine = TransformationEngine()

       async def test_simple_transformation(self):
           result = await self.engine.transform(
               prompt="Test prompt",
               technique_suite="simple",
               potency_level=5
           )

           assert result.success
           assert result.transformed_prompt != "Test prompt"
           assert result.metadata.potency_level == 5
   ```

2. **Integration Tests**
   ```python
   # tests/integration/test_llm_service.py
   import pytest
   from app.services.llm_service import LLMService
   from app.domain.models import PromptRequest

   @pytest.mark.integration
   class TestLLMService:
       async def test_generate_with_mock_provider(self):
           service = LLMService()
           request = PromptRequest(
               prompt="Test prompt",
               provider="mock"
           )

           response = await service.generate_text(request)

           assert response.text is not None
           assert response.provider == "mock"
   ```

3. **API Tests**
   ```python
   # tests/api/test_endpoints.py
   import pytest
   from httpx import AsyncClient
   from app.main import app

   @pytest.mark.asyncio
   async def test_generate_endpoint():
       async with AsyncClient(app=app, base_url="http://test") as ac:
           response = await ac.post(
               "/api/v1/generate",
               json={"prompt": "Test", "provider": "mock"},
               headers={"X-API-Key": "test-key"}
           )

       assert response.status_code == 200
       data = response.json()
       assert "text" in data
   ```

**Running Tests:**
```bash
# All tests
pytest

# Specific test type
pytest -m unit
pytest -m integration
pytest -m "not slow"

# With coverage
pytest --cov=app --cov-report=html

# Specific test file
pytest tests/unit/test_transformation_service.py

# Verbose output
pytest -v -s
```

### Frontend Testing

**Test Setup (Vitest):**
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

**Component Tests:**
```typescript
// src/__tests__/components/PromptGenerator.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { PromptGenerator } from '@/components/PromptGenerator';

describe('PromptGenerator', () => {
  it('should call onGenerate when form is submitted', () => {
    const mockOnGenerate = vi.fn();

    render(<PromptGenerator onGenerate={mockOnGenerate} />);

    const textarea = screen.getByPlaceholderText('Enter your prompt...');
    const button = screen.getByRole('button', { name: /generate/i });

    fireEvent.change(textarea, { target: { value: 'Test prompt' } });
    fireEvent.click(button);

    expect(mockOnGenerate).toHaveBeenCalledWith('Test prompt');
  });
});
```

---

## Deployment

### Development Deployment

**Using Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: ./backend-api
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
    env_file:
      - .env
    volumes:
      - ./backend-api:/app
    depends_on:
      - redis

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8001
    volumes:
      - ./frontend:/app
      - /app/node_modules

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend
```

### Production Deployment

**Backend Dockerfile:**
```dockerfile
# backend-api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Frontend Dockerfile:**
```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
```

### CI/CD Pipeline

**GitHub Actions Example:**
```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-backend:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        cd backend-api
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm

    - name: Run tests
      run: |
        cd backend-api
        pytest --cov=app --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  test-frontend:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Install dependencies
      run: |
        cd frontend
        npm ci

    - name: Run tests
      run: |
        cd frontend
        npm run test

  deploy:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Deploy to production
      run: |
        # Deployment commands
        docker-compose -f docker-compose.prod.yml up -d
```

---

## Contributing

### Development Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/chimera.git
   cd chimera
   git remote add upstream https://github.com/original-org/chimera.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development**
   - Follow coding standards
   - Write tests for new features
   - Update documentation
   - Ensure all tests pass

4. **Commit Guidelines**
   ```bash
   # Format: type(scope): description
   git commit -m "feat(api): add new jailbreak technique endpoint"
   git commit -m "fix(frontend): resolve authentication error handling"
   git commit -m "docs(readme): update installation instructions"
   ```

5. **Pull Request**
   - Create detailed PR description
   - Link related issues
   - Ensure CI passes
   - Request review

### Code Standards

**Python (Backend):**
- Follow PEP 8 style guide
- Use Black for code formatting
- Use Ruff for linting
- Type hints required for all functions
- Docstrings for all public methods

**TypeScript (Frontend):**
- Follow TypeScript strict mode
- Use ESLint + Prettier for formatting
- Follow React/Next.js best practices
- Use meaningful component and variable names

**Git Conventions:**
- Use conventional commit messages
- Keep commits atomic and focused
- Write descriptive commit messages
- Use PR templates for consistency

### Documentation Requirements

- Update API documentation for endpoint changes
- Add/update user guide for new features
- Include architecture diagrams for major changes
- Write inline code documentation
- Update README for setup changes

---

This developer guide provides comprehensive coverage for working with the Chimera system. For additional information, refer to the [Architecture Documentation](ARCHITECTURE.md) and [User Guide](USER_GUIDE.md).