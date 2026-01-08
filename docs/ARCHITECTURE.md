# Chimera System Architecture

## Overview

Chimera is a sophisticated AI-powered prompt optimization and jailbreak research system built with a modern, scalable architecture supporting advanced prompt transformation techniques and multi-provider LLM integration.

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Next.js Frontend]
        Mobile[Mobile Clients]
        API_Client[API Clients]
    end

    subgraph "API Gateway & Load Balancing"
        Gateway[Reverse Proxy/LB]
        Auth[Authentication Service]
        RateLimit[Rate Limiter]
    end

    subgraph "Chimera Core Services"
        FastAPI[FastAPI Backend<br/>Port 8001]
        WebSocket[WebSocket Handler<br/>/ws/enhance]
        Transform[Transformation Engine<br/>20+ Technique Suites]
    end

    subgraph "AI Research Frameworks"
        AutoDAN[AutoDAN Service<br/>Adversarial Optimization]
        GPTFuzz[GPTFuzz Service<br/>Mutation Testing]
        PromptEnh[Prompt Enhancer<br/>Meta Prompter]
        JailbreakEnh[Jailbreak Enhancer<br/>Neural Bypass]
    end

    subgraph "LLM Provider Network"
        Google[Google/Gemini]
        OpenAI[OpenAI/GPT]
        Anthropic[Anthropic/Claude]
        DeepSeek[DeepSeek]
        Qwen[Qwen/Alibaba]
        Cursor[Cursor AI]
    end

    subgraph "Infrastructure & Data"
        CircuitBreaker[Circuit Breaker<br/>Failure Handling]
        Redis[Redis Cache<br/>Optional]
        PostgresDB[(PostgreSQL<br/>Session Data)]
        Monitoring[Health Checks<br/>Observability]
    end

    subgraph "Configuration & Security"
        Config[Environment Config<br/>Proxy/Direct Mode]
        Security[Security Middleware<br/>XSS, CSRF, Headers]
        Secrets[API Key Management<br/>JWT Tokens]
    end

    %% Client Connections
    UI --> Gateway
    Mobile --> Gateway
    API_Client --> Gateway

    %% API Gateway Flow
    Gateway --> Auth
    Gateway --> RateLimit
    Auth --> FastAPI
    RateLimit --> FastAPI

    %% Core Service Integration
    FastAPI --> Transform
    FastAPI --> WebSocket
    Transform --> AutoDAN
    Transform --> GPTFuzz
    Transform --> PromptEnh
    Transform --> JailbreakEnh

    %% LLM Provider Integration
    AutoDAN --> CircuitBreaker
    GPTFuzz --> CircuitBreaker
    PromptEnh --> CircuitBreaker
    CircuitBreaker --> Google
    CircuitBreaker --> OpenAI
    CircuitBreaker --> Anthropic
    CircuitBreaker --> DeepSeek
    CircuitBreaker --> Qwen
    CircuitBreaker --> Cursor

    %% Infrastructure Dependencies
    FastAPI --> Redis
    FastAPI --> PostgresDB
    FastAPI --> Monitoring
    FastAPI --> Security
    FastAPI --> Config
    FastAPI --> Secrets

    %% Styling
    classDef clientLayer fill:#e1f5fe
    classDef coreServices fill:#f3e5f5
    classDef aiFrameworks fill:#fff3e0
    classDef llmProviders fill:#e8f5e8
    classDef infrastructure fill:#fce4ec

    class UI,Mobile,API_Client clientLayer
    class FastAPI,WebSocket,Transform coreServices
    class AutoDAN,GPTFuzz,PromptEnh,JailbreakEnh aiFrameworks
    class Google,OpenAI,Anthropic,DeepSeek,Qwen,Cursor llmProviders
    class CircuitBreaker,Redis,PostgresDB,Monitoring infrastructure
```

## Component Architecture

### 1. Frontend Layer (Next.js 16 + React 19)

```mermaid
graph LR
    subgraph "Next.js Frontend"
        Dashboard[Dashboard Pages]
        Components[UI Components<br/>shadcn/ui]
        ApiClient[Enhanced API Client<br/>Circuit Breaker + Retry]
        StateManager[TanStack Query<br/>State Management]
    end

    subgraph "Key Features"
        Generation[Prompt Generation UI]
        Jailbreak[Jailbreak Testing Interface]
        Providers[Provider Management]
        Health[Health Monitoring]
    end

    Dashboard --> Generation
    Dashboard --> Jailbreak
    Dashboard --> Providers
    Dashboard --> Health
    Components --> Dashboard
    ApiClient --> StateManager
    StateManager --> Components
```

**Technology Stack:**
- **Framework**: Next.js 16 with App Router
- **UI Library**: React 19 + TypeScript
- **Styling**: Tailwind CSS 3 + shadcn/ui components
- **State Management**: TanStack Query for server state
- **Testing**: Vitest for unit/integration testing
- **API Integration**: Enhanced client with circuit breaker pattern

### 2. Backend Core (FastAPI + Python 3.11+)

```mermaid
graph TB
    subgraph "FastAPI Application"
        Main[main.py<br/>Application Entry]
        Routes[API Routes<br/>v1/endpoints/]
        Middleware[Middleware Stack]
    end

    subgraph "Business Logic Services"
        LLMService[LLM Service<br/>Multi-Provider Orchestration]
        TransformService[Transformation Service<br/>20+ Technique Suites]
        AutoDANSvc[AutoDAN Service<br/>Genetic Algorithm Optimization]
        GPTFuzzSvc[GPTFuzz Service<br/>Mutation-Based Testing]
        JailbreakSvc[Jailbreak Service<br/>Neural Bypass Techniques]
    end

    subgraph "Core Infrastructure"
        CircuitBreaker[Circuit Breaker<br/>Provider Resilience]
        ConfigMgr[Configuration Manager<br/>Environment Handling]
        HealthCheck[Health Check System<br/>Dependency Monitoring]
        Observability[Observability<br/>Metrics & Logging]
    end

    Main --> Routes
    Main --> Middleware
    Routes --> LLMService
    Routes --> TransformService
    LLMService --> AutoDANSvc
    LLMService --> GPTFuzzSvc
    TransformService --> JailbreakSvc

    LLMService --> CircuitBreaker
    TransformService --> ConfigMgr
    AutoDANSvc --> HealthCheck
    GPTFuzzSvc --> Observability
```

**Key Components:**

**LLM Service** (`app/services/llm_service.py`)
- Multi-provider orchestration with automatic failover
- Circuit breaker pattern for provider resilience
- Dynamic provider registration and discovery
- Request/response normalization across providers

**Transformation Engine** (`app/services/transformation_service.py`)
- 20+ transformation technique suites
- Extensible plugin architecture
- Async processing with concurrent transformations
- Caching layer for frequently used transformations

**Circuit Breaker** (`app/core/shared/circuit_breaker.py`)
- Provider failure detection and recovery
- Configurable failure thresholds and timeouts
- Automatic fallback to alternative providers
- Performance metrics and monitoring

### 3. AI Research Frameworks

#### AutoDAN Integration

```mermaid
graph LR
    subgraph "AutoDAN Framework"
        Adapter[Chimera LLM Adapter]
        Optimizer[Genetic Algorithm<br/>Optimizer]
        SearchStrategies[Search Strategies<br/>Vanilla, Best-of-N, Beam]
        ReasoningModels[Reasoning Model<br/>Support]
    end

    subgraph "Configuration"
        Config[AutoDAN Config<br/>Retry Strategies]
        Enhanced[Enhanced Config<br/>Model Selection]
    end

    Adapter --> Optimizer
    Optimizer --> SearchStrategies
    SearchStrategies --> ReasoningModels
    Config --> Adapter
    Enhanced --> Config
```

**Features:**
- **Adversarial Prompt Optimization**: Using genetic algorithms for prompt evolution
- **Multiple Attack Methods**: Vanilla, best-of-n, beam search optimization
- **Reasoning Model Integration**: Support for advanced reasoning capabilities
- **Hierarchical Search**: Multi-level optimization strategies

#### GPTFuzz Framework

```mermaid
graph LR
    subgraph "GPTFuzz Components"
        Mutators[Mutator Network<br/>CrossOver, Expand, Rephrase]
        MCTS[MCTS Selection<br/>Policy]
        SessionMgr[Session Manager<br/>Test Orchestration]
        Predictor[LLM Predictor<br/>Response Analysis]
    end

    subgraph "Mutation Types"
        CrossOver[CrossOver Mutator]
        Expand[Expand Mutator]
        Similar[GenerateSimilar]
        Rephrase[Rephrase Mutator]
        Shorten[Shorten Mutator]
    end

    Mutators --> CrossOver
    Mutators --> Expand
    Mutators --> Similar
    Mutators --> Rephrase
    Mutators --> Shorten

    MCTS --> Mutators
    SessionMgr --> MCTS
    Predictor --> SessionMgr
```

**Capabilities:**
- **Mutation-Based Testing**: Intelligent prompt mutation strategies
- **MCTS Exploration**: Monte Carlo Tree Search for optimal prompt selection
- **Session Management**: Configurable testing sessions with state persistence
- **Automated Analysis**: LLM-powered response evaluation

### 4. Multi-Provider LLM Integration

```mermaid
graph TB
    subgraph "Provider Abstraction Layer"
        Interface[LLMProvider Protocol]
        Registry[Provider Registry]
        Router[Request Router]
    end

    subgraph "Provider Implementations"
        GoogleProv[Google Provider<br/>Gemini Models]
        OpenAIProv[OpenAI Provider<br/>GPT Models]
        AnthropicProv[Anthropic Provider<br/>Claude Models]
        DeepSeekProv[DeepSeek Provider]
        QwenProv[Qwen Provider]
        CursorProv[Cursor Provider]
        MockProv[Mock Provider<br/>Testing/Fallback]
    end

    subgraph "Connection Modes"
        DirectMode[Direct Mode<br/>Native API Calls]
        ProxyMode[Proxy Mode<br/>AIClient-2-API Server<br/>localhost:8080]
    end

    Interface --> Registry
    Registry --> Router
    Router --> GoogleProv
    Router --> OpenAIProv
    Router --> AnthropicProv
    Router --> DeepSeekProv
    Router --> QwenProv
    Router --> CursorProv
    Router --> MockProv

    GoogleProv --> DirectMode
    OpenAIProv --> DirectMode
    AnthropicProv --> ProxyMode
    DeepSeekProv --> DirectMode
```

**Provider Features:**
- **Unified Interface**: Consistent API across all providers
- **Dynamic Registration**: Runtime provider discovery and registration
- **Connection Modes**: Support for both direct and proxy connections
- **Model Selection**: Provider-specific model configuration
- **Automatic Fallback**: Circuit breaker with provider failover

### 5. Security Architecture

```mermaid
graph TB
    subgraph "Authentication Layer"
        APIKey[API Key Authentication<br/>X-API-Key Header]
        JWT[JWT Token Support<br/>Bearer Tokens]
        AuthMiddleware[Authentication Middleware]
    end

    subgraph "Security Middleware"
        SecurityHeaders[Security Headers<br/>XSS, Clickjacking Protection]
        CSRF[CSRF Protection]
        RateLimiting[Rate Limiting<br/>Per-User/IP]
        InputValidation[Pydantic Input Validation]
    end

    subgraph "Jailbreak Research Security"
        EthicalSafeguards[Ethical Research Safeguards]
        OutputFiltering[Response Filtering]
        AuditLogging[Security Audit Logging]
        PatternDetection[Dangerous Pattern Detection]
    end

    APIKey --> AuthMiddleware
    JWT --> AuthMiddleware
    AuthMiddleware --> SecurityHeaders
    SecurityHeaders --> CSRF
    CSRF --> RateLimiting
    RateLimiting --> InputValidation

    InputValidation --> EthicalSafeguards
    EthicalSafeguards --> OutputFiltering
    OutputFiltering --> AuditLogging
    AuditLogging --> PatternDetection
```

**Security Features:**
- **Multi-Factor Authentication**: API key + JWT token support
- **Comprehensive Security Headers**: XSS, CSRF, clickjacking protection
- **Research Ethics**: Safeguards for responsible jailbreak research
- **Audit Logging**: Complete security event logging
- **Pattern Detection**: Automated detection of dangerous prompt patterns

## Data Flow Architecture

### Request Processing Pipeline

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant FastAPI
    participant Transform
    participant CircuitBreaker
    participant LLMProvider
    participant Cache

    Client->>Gateway: API Request
    Gateway->>Gateway: Authentication & Rate Limiting
    Gateway->>FastAPI: Validated Request

    FastAPI->>Transform: Prompt Transformation
    Transform->>Transform: Apply Technique Suite
    Transform->>Cache: Check Cache

    alt Cache Hit
        Cache-->>Transform: Cached Result
    else Cache Miss
        Transform->>CircuitBreaker: LLM Request
        CircuitBreaker->>LLMProvider: Provider Call
        LLMProvider-->>CircuitBreaker: LLM Response
        CircuitBreaker-->>Transform: Processed Response
        Transform->>Cache: Store Result
    end

    Transform-->>FastAPI: Enhanced Prompt
    FastAPI-->>Gateway: API Response
    Gateway-->>Client: Final Response
```

### WebSocket Real-Time Enhancement

```mermaid
sequenceDiagram
    participant Client
    participant WebSocket
    participant Enhancer
    participant LLM

    Client->>WebSocket: Connect /ws/enhance
    WebSocket-->>Client: Connection Established

    loop Real-time Enhancement
        Client->>WebSocket: Prompt Input
        WebSocket->>Enhancer: Process Prompt
        Enhancer->>LLM: Enhancement Request
        LLM-->>Enhancer: Enhanced Result
        Enhancer-->>WebSocket: Processed Enhancement
        WebSocket-->>Client: Real-time Update
    end

    Client->>WebSocket: Disconnect
    WebSocket-->>Client: Connection Closed
```

## Deployment Architecture

### Development Environment

```mermaid
graph LR
    subgraph "Local Development"
        Dev[Developer Machine]
        Backend[Backend:8001]
        Frontend[Frontend:3000]
        Redis[Redis:6379]
    end

    Dev --> Backend
    Dev --> Frontend
    Backend --> Redis
    Frontend --> Backend
```

### Production Environment

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[nginx/HAProxy]
    end

    subgraph "Application Tier"
        App1[Chimera Instance 1]
        App2[Chimera Instance 2]
        App3[Chimera Instance N]
    end

    subgraph "Data Tier"
        RedisCluster[Redis Cluster]
        PostgresHA[PostgreSQL HA]
        S3[S3 Storage]
    end

    subgraph "External Services"
        GoogleAPI[Google AI API]
        OpenAIAPI[OpenAI API]
        AnthropicAPI[Anthropic API]
    end

    LB --> App1
    LB --> App2
    LB --> App3

    App1 --> RedisCluster
    App2 --> RedisCluster
    App3 --> RedisCluster

    App1 --> PostgresHA
    App2 --> PostgresHA
    App3 --> PostgresHA

    App1 --> GoogleAPI
    App1 --> OpenAIAPI
    App1 --> AnthropicAPI
```

## Configuration Management

### Environment Configuration

```yaml
# Production Configuration Example
environment: production
log_level: INFO

# Server Configuration
server:
  port: 8001
  host: "0.0.0.0"
  workers: 4

# Security Configuration
security:
  jwt_secret: "${JWT_SECRET}"
  api_key: "${CHIMERA_API_KEY}"
  cors_origins: ["https://chimera.example.com"]

# LLM Provider Configuration
providers:
  google:
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-1.5-pro"
    enabled: true
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    enabled: true
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-5-sonnet-20241022"
    enabled: true

# Connection Mode
connection:
  mode: "direct"  # or "proxy"
  proxy_url: "http://localhost:8080"

# Circuit Breaker Configuration
circuit_breaker:
  failure_threshold: 3
  recovery_timeout: 60
  timeout: 30

# Redis Configuration
redis:
  url: "${REDIS_URL}"
  ttl: 3600

# Database Configuration
database:
  url: "${DATABASE_URL}"
  pool_size: 10
  max_overflow: 20
```

## Scaling Considerations

### Horizontal Scaling

1. **Stateless Design**: All services designed for horizontal scaling
2. **Load Balancing**: Round-robin with health checks
3. **Session Storage**: Redis-based distributed session management
4. **Database Scaling**: Read replicas and connection pooling

### Performance Optimization

1. **Async Processing**: Full async/await implementation
2. **Connection Pooling**: Optimized HTTP client pools
3. **Caching Strategy**: Multi-tier caching (Redis + in-memory)
4. **Circuit Breakers**: Provider failure isolation

### Monitoring & Observability

1. **Health Checks**: Comprehensive dependency monitoring
2. **Metrics Collection**: Performance and usage metrics
3. **Distributed Tracing**: Request tracing across services
4. **Log Aggregation**: Centralized logging with correlation IDs

---

## Next Steps

1. **Microservices Evolution**: Consider service decomposition for larger scale
2. **Event-Driven Architecture**: Implement event streaming for real-time features
3. **Advanced Caching**: Multi-layer caching with invalidation strategies
4. **Service Mesh**: Istio/Linkerd for advanced networking and security

This architecture provides a solid foundation for both current AI research needs and future scaling requirements.