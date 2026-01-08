# Technical Specification: Real-Time Research Platform

Date: 2026-01-02
Author: BMAD USER
Epic ID: Epic 3
Status: Draft

---

## Overview

Epic 3 delivers the user-facing Next.js 16 frontend with React 19, providing an intuitive real-time research platform for prompt testing, jailbreak experimentation, and workflow management. This epic implements WebSocket communication for <200ms real-time updates, comprehensive dashboard navigation, responsive design with WCAG AA accessibility compliance, and session persistence for research continuity.

## Objectives and Scope

**Objectives:**
- Establish Next.js 16 application foundation with React 19 and TypeScript
- Implement responsive dashboard layout with clear navigation
- Deliver prompt input form with comprehensive parameter controls
- Enable WebSocket real-time updates with <200ms latency
- Create results display with comparison and export capabilities
- Build jailbreak testing interface for AutoDAN and GPTFuzz
- Implement session persistence and history management
- Achieve WCAG AA accessibility compliance

**Scope:**
- 8 user stories covering application setup, dashboard layout, prompt input, WebSocket updates, results display, jailbreak interface, session persistence, and accessibility
- Next.js 16 with App Router pattern
- React 19 with concurrent features
- TypeScript strict mode
- Tailwind CSS 3 for styling
- shadcn/ui component library
- TanStack Query for data fetching
- WebSocket client with reconnection logic

**Out of Scope:**
- Provider backend integration (Epic 1)
- Transformation techniques (Epic 2)
- Analytics dashboard (Epic 4)
- Cross-model comparison (Epic 5)

## System Architecture Alignment

Epic 3 implements the **Frontend Presentation Layer** from the solution architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js Frontend (Port 3000)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Layout & Navigation                            │ │
│  │  ┌────────────┐  ┌──────────────────────────────────────┐ │ │
│  │  │  Sidebar   │  │         Main Content Area             │ │ │
│  │  │  Nav       │  │  ┌────────────────────────────────┐  │ │ │
│  │  │            │  │  │   Page Routes (App Router)      │  │ │
│  │  │ • Generation│  │  │  • /dashboard/generation        │  │ │
│  │  │ • Jailbreak│  │  │  • /dashboard/jailbreak          │  │ │
│  │  │ • Providers│  │  │  • /dashboard/providers          │  │ │
│  │  │ • Health   │  │  │  • /dashboard/analytics          │  │ │
│  │  └────────────┘  │  └────────────────────────────────┘  │ │ │
│  │                 └──────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Data Fetching Layer                            │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         TanStack Query (React Query)                 │  │ │
│  │  │  • Caching and stale-while-revalidate                │  │ │
│  │  │  • Automatic refetching                              │  │ │
│  │  │  • Optimistic updates                                │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Real-Time Communication                       │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         WebSocket Client (ws/enhance)                │  │ │
│  │  │  • Real-time updates (<200ms latency)               │  │ │
│  │  │  • Automatic reconnection with exponential backoff   │  │ │
│  │  │  • Heartbeat mechanism for connectivity             │  │ │
│  │  │  • Connection state management                       │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Component Library                             │ │
│  │  • shadcn/ui (Radix UI primitives + Tailwind)             │ │
│  │  • Responsive design (mobile, tablet, desktop)             │ │
│  │  • WCAG AA accessibility compliance                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions Referenced:**
- **ADR-002**: Separate Frontend/Backend Deployment
- **ADR-003**: WebSocket for Real-Time Updates
- **ADR-005**: shadcn/ui for Component Library

## Detailed Design

### Services and Modules

**Frontend Architecture:**

1. **Application Setup** (`src/app/` - Next.js App Router)
   - `layout.tsx` - Root layout with providers
   - `page.tsx` - Home/landing page
   - `dashboard/layout.tsx` - Dashboard layout wrapper
   - `dashboard/generation/page.tsx` - Prompt generation interface
   - `dashboard/jailbreak/page.tsx` - Jailbreak testing interface
   - `dashboard/providers/page.tsx` - Provider management
   - `dashboard/analytics/page.tsx` - Analytics dashboard
   - `dashboard/health/page.tsx` - System health monitoring

2. **API Client Layer** (`src/lib/`)
   - `api-client.ts` - Main API client with TypeScript types
   - `api-enhanced.ts` - Enhanced client with circuit breaker and retry
   - `api-config.ts` - Centralized API URL configuration
   - `websocket-client.ts` - WebSocket client with reconnection logic

3. **State Management** (`src/lib/`)
   - `stores/` - Zustand or React Context for global state
   - `providers/` - React context providers
   - `hooks/` - Custom React hooks (useWebSocket, useGeneration)

4. **Components** (`src/components/`)
   - `layout/` - Layout components (Sidebar, Header, Navigation)
   - `forms/` - Form components (PromptInput, ParameterControls)
   - `results/` - Results display (ResultsDisplay, ComparisonView)
   - `providers/` - Provider management (ProviderCard, ProviderSelector)
   - `jailbreak/` - Jailbreak interfaces (AutoDANConfig, GPTFuzzConfig)
   - `sessions/` - Session management (SessionList, SessionDetail)
   - `ui/` - Reusable UI components from shadcn/ui

5. **Types** (`src/types/`)
   - `api.ts` - API request/response types
   - `provider.ts` - Provider types
   - `transformation.ts` - Transformation types
   - `session.ts` - Session types

### Data Models and Contracts

**Frontend TypeScript Types:**

```typescript
// API Types
interface GenerationRequest {
  prompt: string;
  provider?: string;
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
}

interface GenerationResponse {
  text: string;
  provider: string;
  model: string;
  usage: TokenUsage;
  timing: TimingInfo;
  metadata: Record<string, unknown>;
}

interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

interface TimingInfo {
  latency_ms: number;
  total_time_ms: number;
}

// Provider Types
interface Provider {
  id: string;
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  models: Model[];
  metrics: ProviderMetrics;
}

interface Model {
  id: string;
  name: string;
  provider: string;
  context_length: number;
}

interface ProviderMetrics {
  latency_ms: number;
  success_rate: number;
  request_count: number;
  error_rate: number;
}

// WebSocket Types
interface WebSocketMessage {
  type: 'status' | 'partial' | 'complete' | 'error' | 'heartbeat';
  payload: unknown;
  request_id?: string;
  timestamp: string;
}

interface WebSocketStatus {
  connected: boolean;
  connecting: boolean;
  error: Error | null;
}

// Session Types
interface Session {
  id: string;
  timestamp: Date;
  prompt: string;
  result: GenerationResponse;
  tags: string[];
  metadata: Record<string, unknown>;
}

// Transformation Types
interface TransformationTechnique {
  name: string;
  category: TechniqueCategory;
  description: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

// Jailbreak Types
interface AutoDANConfig {
  attackMethod: 'vanilla' | 'best_of_n' | 'beam_search' | 'mousetrap';
  populationSize: number;
  iterations: number;
  temperature: number;
  targetModel: string;
}

interface GPTFuzzConfig {
  mutators: MutatorType[];
  iterations: number;
  sessionId?: string;
}
```

### APIs and Interfaces

**Frontend-to-Backend API Integration:**

| Frontend Component | Backend Endpoint | Method | Purpose |
|--------------------|------------------|--------|---------|
| Provider Selector | `/api/v1/providers` | GET | Fetch available providers |
| Model Selector | `/api/v1/session/models` | GET | Fetch models for session |
| Generation Form | `/api/v1/generate` | POST | Submit generation request |
| Transform Form | `/api/v1/transform` | POST | Apply transformations |
| Execute Form | `/api/v1/execute` | POST | Transform and execute |
| AutoDAN Config | `/api/v1/autodan/optimize` | POST | Run AutoDAN optimization |
| GPTFuzz Config | `/api/v1/gptfuzz/mutate` | POST | Run GPTFuzz mutation |

**WebSocket Protocol:**

**Connection:** `WS localhost:8001/ws/enhance`

**Message Format:**
```typescript
// Client → Server
{
  "type": "generate",
  "request_id": "uuid-1234",
  "payload": GenerationRequest
}

// Server → Client (status update)
{
  "type": "status",
  "request_id": "uuid-1234",
  "payload": {
    "message": "Processing prompt...",
    "progress": 0.5
  },
  "timestamp": "2026-01-02T10:30:00Z"
}

// Server → Client (partial result for streaming)
{
  "type": "partial",
  "request_id": "uuid-1234",
  "payload": {
    "text": "Partial generated text..."
  },
  "timestamp": "2026-01-02T10:30:01Z"
}

// Server → Client (complete result)
{
  "type": "complete",
  "request_id": "uuid-1234",
  "payload": GenerationResponse,
  "timestamp": "2026-01-02T10:30:02Z"
}

// Server → Client (heartbeat)
{
  "type": "heartbeat",
  "timestamp": "2026-01-02T10:30:30Z"
}
```

**Component API (React Props):**

```typescript
// PromptInput Form Component
interface PromptInputProps {
  onSubmit: (request: GenerationRequest) => void;
  isLoading: boolean;
  availableProviders: Provider[];
  enabledTechniques: TransformationTechnique[];
}

// Results Display Component
interface ResultsDisplayProps {
  result: GenerationResponse;
  originalPrompt: string;
  showComparison?: boolean;
  onExport: (format: 'json' | 'text' | 'markdown') => void;
}

// WebSocket Status Component
interface WebSocketStatusProps {
  status: WebSocketStatus;
  latency?: number;
}

// Provider Card Component
interface ProviderCardProps {
  provider: Provider;
  onSelect: (providerId: string) => void;
  onTest: (providerId: string) => Promise<boolean>;
}
```

### Workflows and Sequencing

**Prompt Generation Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  User enters prompt in PromptInput form                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  User configures parameters:                               │
│  • Provider selection (dropdown)                            │
│  • Model selection (filtered by provider)                   │
│  • Temperature slider (0.0 - 2.0)                           │
│  • Top_p slider (0.0 - 1.0)                                 │
│  • Max_tokens input (1 - 32000)                             │
│  • Transformation techniques (multi-select)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  User clicks "Generate"                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Frontend validates input                                   │
│  • Required fields present                                  │
│  • Parameter ranges valid                                   │
│  • Provider and model compatible                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Frontend opens WebSocket connection                         │
│  • Sends generate message with request_id                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Real-time updates via WebSocket:                           │
│  • status: "Generating..."                                  │
│  • partial: Streaming text chunks (if streaming enabled)     │
│  • complete: Final result with metadata                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Results displayed with:                                    │
│  • Generated text (formatted)                               │
│  • Usage metadata (tokens, timing, costs)                    │
│  • Transformation techniques applied                         │
│  • Copy-to-clipboard button                                  │
│  • Export options (JSON, text, markdown)                     │
└─────────────────────────────────────────────────────────────┘
```

**WebSocket Connection Management Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  Application mounts                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  WebSocket client attempts connection                       │
│  • Connects to ws://localhost:8001/ws/enhance              │
└────────────────────────┬────────────────────────────────────┘
                         │
                ┌────────┴────────┐
                │                 │
            Success          Failure
                │                 │
                ▼                 ▼
    ┌───────────────┐   ┌─────────────────────┐
    │ Connected    │   │ Exponential backoff  │
    │ (heartbeat)  │   │ retry (1s, 2s, 4s...) │
    └───────┬───────┘   └──────────┬──────────┘
            │                      │
            │                      ▼
            │              ┌───────────────┐
            │              │ Max retries    │
            │              │ exhausted →    │
            │              │ Show error      │
            │              └───────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Message handling loop:                                      │
│  • Incoming messages routed by type                         │
│  • status → Update progress indicator                        │
│  • partial → Append to streaming output                     │
│  • complete → Render final result                            │
│  • error → Show error message                                │
│  • heartbeat → Update connection timestamp                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Connection lost detection:                                  │
│  • No heartbeat for 60 seconds → Reconnect                   │
│  • WebSocket close event → Reconnect                         │
│  • Network error → Reconnect with backoff                    │
└─────────────────────────────────────────────────────────────┘
```

**Session Persistence Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│  User completes generation request                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Frontend creates session object:                            │
│  {                                                           │
│    id: generateUUID(),                                      │
│    timestamp: new Date(),                                   │
│    prompt: originalPrompt,                                  │
│    result: generationResponse,                              │
│    tags: [],                                                │
│    metadata: { provider, model, techniques }               │
│  }                                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Session persisted to:                                      │
│  • localStorage (browser persistence)                        │
│  • Backend session API (if authenticated)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Session History page displays:                              │
│  • Chronological list of past sessions                      │
│  • Session cards with preview info                          │
│  • Search and filter controls                                │
│  • Tag management UI                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  User can:                                                   │
│  • Click session to view full details                        │
│  • Add/edit tags for organization                            │
│  • Export session to file (JSON, CSV)                        │
│  • Resume session for continued testing                      │
│  • Delete or archive old sessions                            │
└─────────────────────────────────────────────────────────────┘
```

## Non-Functional Requirements

### Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Initial Page Load | <2s | First Contentful Paint |
| Time to Interactive | <3s | User can interact with UI |
| WebSocket Latency | <200ms | Message round-trip time |
| Provider List Load | <500ms | API response + render |
| Results Render | <500ms | Large result display |
| Search Response | <300ms | Session search filtering |

**Performance Optimization:**
- Code splitting with Next.js dynamic imports
- Image optimization with Next.js Image component
- TanStack Query caching with stale-while-revalidate
- Lazy loading for large session lists
- Virtualization for long lists (react-window)

### Security

| Aspect | Implementation |
|--------|----------------|
| API Authentication | X-API-Key header stored securely |
| XSS Prevention | React automatic escaping, Content Security Policy |
| CSRF Protection | CSRF tokens for state-changing requests |
| Input Sanitization | DOMPurify for user-generated content |
| Secure Storage | Sensitive data in sessionStorage (not localStorage) |

**Security Considerations:**
- API keys never exposed in client-side code
- WebSocket connection authenticated via token
- User input sanitized before display
- Export functionality validates data

### Reliability/Availability

| Metric | Target | Mechanism |
|--------|--------|-----------|
| Frontend Uptime | 99.9% | CDN deployment, error boundaries |
| WebSocket Reconnection | Automatic | Exponential backoff retry |
| Offline Capability | Limited | Service worker for caching |
| Error Boundary Coverage | 100% | All components wrapped |
| Session Persistence | 100% | localStorage + server backup |

**Reliability Features:**
- React error boundaries prevent component crashes from cascading
- WebSocket automatic reconnection with backoff
- localStorage fallback for session persistence
- Optimistic UI updates with rollback on error

### Observability

| Aspect | Implementation |
|--------|----------------|
| Client Metrics | Web Vitals (LCP, FID, CLS) |
| Error Tracking | Sentry or similar error reporting |
| Analytics | User interaction tracking |
| Performance Monitoring | Real User Monitoring (RUM) |
| Console Logging | Structured logging with log levels |

**Observable Metrics:**
- Page load times by route
- WebSocket connection success rate
- API error rates by endpoint
- User interaction patterns
- Session creation and access patterns

## Dependencies and Integrations

**Internal Dependencies:**
- Epic 1 (Multi-Provider Foundation) - Backend API endpoints
- Epic 2 (Advanced Transformation Engine) - Transformation techniques
- Epic 4 (Analytics and Compliance) - Analytics dashboard data

**External Dependencies:**

| Dependency | Version | Purpose |
|------------|---------|---------|
| Next.js | 16 | Frontend framework |
| React | 19 | UI library |
| TypeScript | 5.0+ | Type safety |
| Tailwind CSS | 3.3+ | Styling |
| shadcn/ui | Latest | Component library |
| TanStack Query | 5.0+ | Data fetching and caching |
| Zustand | 4.4+ | State management (optional) |
| Vitest | 1.0+ | Testing framework |
| @playwright/test | 1.40+ | E2E testing |

**Development Dependencies:**
- ESLint - Code linting
- Prettier - Code formatting
- TypeScript - Type checking
- Playwright - E2E testing

## Acceptance Criteria (Authoritative)

### Story RP-001: Next.js Application Setup
- [ ] Next.js 16 configured with App Router
- [ ] React 19 integrated with latest features
- [ ] TypeScript enabled with strict type checking
- [ ] Tailwind CSS 3 configured for styling
- [ ] Project structure follows Next.js best practices
- [ ] Development server runs on port 3000
- [ ] Build and production configuration optimized
- [ ] ESLint and TypeScript configurations in place

### Story RP-002: Dashboard Layout and Navigation
- [ ] Dashboard has sidebar navigation with clear sections
- [ ] Navigation includes: Generation, Jailbreak, Providers, Health
- [ ] Layout responsive across desktop and tablet
- [ ] Active navigation state visually indicated
- [ ] Navigation accessible with keyboard shortcuts
- [ ] Dashboard shows quick stats and recent activity
- [ ] Overall design professional and research-focused

### Story RP-003: Prompt Input Form
- [ ] Form includes prompt text area with character count
- [ ] Form supports provider selection from available providers
- [ ] Form includes model selection based on chosen provider
- [ ] Form has parameter controls: temperature, top_p, max_tokens
- [ ] Form supports transformation technique selection
- [ ] Form validates inputs before submission
- [ ] Form shows recent prompts for quick reuse
- [ ] Submission triggers real-time updates via WebSocket

### Story RP-004: WebSocket Real-Time Updates
- [ ] WebSocket connection provides real-time updates
- [ ] Updates include: status messages, partial results, completion
- [ ] Connection handles reconnection automatically
- [ ] Connection shows heartbeat for connectivity status
- [ ] latency under 200ms for updates
- [ ] Connection gracefully handles failures
- [ ] Multiple concurrent requests supported
- [ ] Connection state visually indicated

### Story RP-005: Results Display and Analysis
- [ ] Display shows generated text with formatting preserved
- [ ] Display includes usage metadata (tokens, timing, costs)
- [ ] Display shows transformation techniques applied
- [ ] Display provides copy-to-clipboard functionality
- [ ] Display supports export to file (JSON, text, markdown)
- [ ] Display shows comparison with original prompt
- [ ] Display highlights changes and improvements
- [ ] Display supports side-by-side comparison views

### Story RP-006: Jailbreak Testing Interface
- [ ] Interface supports AutoDAN optimization method selection
- [ ] Interface supports GPTFuzz mutator configuration
- [ ] Interface shows target model selection
- [ ] Interface includes optimization parameters (population, iterations, etc.)
- [ ] Results show ASR metrics and success rates
- [ ] Results include optimized prompts and analysis
- [ ] Interface supports session-based testing persistence
- [ ] Interface provides risk warnings and usage guidance

### Story RP-007: Session Persistence and History
- [ ] Interface shows chronological list of past sessions
- [ ] Each session shows summary (timestamp, prompt, result preview)
- [ ] Sessions searchable and filterable
- [ ] Sessions support tags and labels for organization
- [ ] Clicking session loads full details
- [ ] Sessions support export and sharing
- [ ] Interface supports session resumption for testing
- [ ] Old sessions archived or deleted as needed

### Story RP-008: Responsive Design and Accessibility
- [ ] Layout adapts to desktop (1280px+), tablet (768-1279px), mobile (<768px)
- [ ] Navigation accessible via keyboard and screen readers
- [ ] Form inputs have proper labels and ARIA attributes
- [ ] Color contrast meets WCAG AA standards
- [ ] Interactive elements have clear focus indicators
- [ ] Touch targets minimum 44x44 pixels
- [ ] Content readable at default zoom levels
- [ ] Interface supports high contrast mode

## Traceability Mapping

**Requirements from PRD:**

| PRD Requirement | Epic 3 Story | Implementation |
|----------------|--------------|----------------|
| FR-08: WebSocket real-time updates | RP-004 | <200ms latency updates |
| FR-09: Responsive dashboard interface | RP-002, RP-008 | Mobile/tablet/desktop support |
| NFR-02: 99.9% uptime | RP-001 | Error boundaries, CDN deployment |
| NFR-03: <200ms WebSocket latency | RP-004 | WebSocket protocol optimization |
| NFR-07: WCAG AA accessibility | RP-008 | ARIA attributes, keyboard nav |

**Epic-to-Architecture Mapping:**

| Architecture Component | Epic 3 Implementation |
|-----------------------|----------------------|
| Frontend Presentation Layer | All stories (RP-001 through RP-008) |
| Next.js Application Foundation | RP-001 |
| Dashboard Layout | RP-002 |
| Prompt Input Forms | RP-003 |
| WebSocket Client | RP-004 |
| Results Display | RP-005 |
| Jailbreak Interface | RP-006 |
| Session Management | RP-007 |
| Responsive/Accessible Design | RP-008 |

## Risks, Assumptions, Open Questions

**Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| WebSocket connection unstable in production | High | Automatic reconnection with backoff, fallback to polling |
| Large session lists cause performance issues | Medium | Virtualization, pagination, lazy loading |
| Accessibility compliance requires significant effort | Medium | shadcn/ui has built-in accessibility, WCAG checklist |
| Browser compatibility issues | Low | Progressive enhancement, polyfills for older browsers |

**Assumptions:**
- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+) support all features
- WebSocket protocol stable and well-supported
- Users have JavaScript enabled
- Display resolution sufficient for dashboard layout

**Open Questions:**
- Should session persistence rely solely on localStorage or also sync to backend? → **Decision: Hybrid - localStorage for speed, backend sync for authenticated users**
- What is the maximum number of sessions to store in localStorage? → **Decision: 100 sessions max, oldest archived beyond limit**
- Should WebSocket use a fallback to polling for incompatible browsers? → **Decision: No, require WebSocket for real-time features**

## Test Strategy Summary

**Unit Tests (Vitest):**
- Component rendering with props
- Form validation logic
- Custom hooks (useWebSocket, useGeneration)
- Utility functions
- Type checking (TypeScript compiler)

**Integration Tests:**
- API client integration with mock backend
- WebSocket message handling
- Form submission to API
- Session persistence to localStorage
- Navigation between routes

**End-to-End Tests (Playwright):**
- Complete prompt generation flow
- WebSocket connection and reconnection
- Session creation and retrieval
- Jailbreak configuration and execution
- Export functionality
- Responsive design at different breakpoints
- Keyboard navigation and screen reader compatibility

**Accessibility Tests:**
- Automated axe-core testing
- Keyboard navigation flow
- Screen reader testing (NVDA, JAWS)
- Color contrast verification
- Focus management testing

**Performance Tests:**
- Page load time measurement
- WebSocket latency measurement
- Large list rendering performance
- Memory leak detection
- Bundle size analysis

**Test Coverage Target:** 70%+ for frontend code, 80%+ for critical paths

---

_This technical specification serves as the implementation guide for Epic 3: Real-Time Research Platform. All development should reference this document for detailed design decisions, component architecture, and acceptance criteria._
