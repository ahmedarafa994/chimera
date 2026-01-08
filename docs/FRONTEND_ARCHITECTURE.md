# Frontend Architecture: Deep Team + AutoDAN UI

## Overview

A **modern React-based web interface** for managing and monitoring multi-agent collaborative red-teaming sessions with real-time visualization, agent control, and comprehensive reporting.

---

## Technology Stack

### Core Framework
- **Next.js 16** - React framework with App Router
- **React 19** - UI library with concurrent features
- **TypeScript** - Type-safe development
- **Tailwind CSS 3** - Utility-first styling

### UI Components
- **shadcn/ui** - Accessible component library
- **Radix UI** - Headless UI primitives
- **Lucide Icons** - Modern icon library
- **React Hot Toast** - Toast notifications

### Data Visualization
- **Recharts** - Composable charts library
- **D3.js** - Advanced visualizations
- **React Flow** - Node-based agent graphs

### State Management
- **TanStack Query (React Query)** - Server state management
- **Zustand** - Client state management
- **React Context** - Global state

### Real-Time Communication
- **Socket.IO Client** - WebSocket communication
- **Server-Sent Events (SSE)** - Real-time updates

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js Frontend                          │
│                    (Port 3000 / 3700)                            │
└───────────────┬─────────────────────┬───────────────────────────┘
                │                     │
        ┌───────▼────────┐   ┌───────▼────────┐
        │   Dashboard    │   │   API Routes   │
        │   Components   │   │   (Proxy)      │
        └───────┬────────┘   └───────┬────────┘
                │                     │
        ┌───────▼────────┐   ┌───────▼────────┐
        │  Agent Cards   │   │   WebSocket    │
        │  Charts        │   │   Handler      │
        │  Controls      │   └───────┬────────┘
        └───────┬────────┘           │
                │                     │
                └──────────┬──────────┘
                           │
                    ┌──────▼────────┐
                    │  Backend API  │
                    │  (Port 8001)  │
                    └───────────────┘
```

---

## Component Structure

```
frontend/src/
├── app/
│   ├── dashboard/
│   │   └── deepteam/
│   │       ├── page.tsx                 # Main dashboard
│   │       ├── sessions/
│   │       │   ├── page.tsx             # Sessions list
│   │       │   └── [id]/page.tsx        # Session detail
│   │       ├── agents/
│   │       │   └── page.tsx             # Agent management
│   │       ├── authorization/
│   │       │   └── page.tsx             # Auth management
│   │       └── analytics/
│   │           └── page.tsx             # Analytics
│   │
│   └── api/
│       └── deepteam/
│           ├── sessions/route.ts        # Session API
│           ├── agents/route.ts          # Agent API
│           └── ws/route.ts              # WebSocket
│
├── components/
│   ├── deepteam/
│   │   ├── AgentCard.tsx               # Agent status card
│   │   ├── AgentGraph.tsx              # Agent relationship graph
│   │   ├── EvolutionChart.tsx          # Evolution progress
│   │   ├── FitnessChart.tsx            # Fitness over time
│   │   ├── EvaluationPanel.tsx         # Evaluation results
│   │   ├── RefinementPanel.tsx         # Refinement suggestions
│   │   ├── ControlPanel.tsx            # Session controls
│   │   ├── ConfigurationForm.tsx       # AutoDAN config
│   │   ├── AuthorizationManager.tsx    # Auth tokens
│   │   ├── AuditLog.tsx                # Audit trail
│   │   └── SessionMonitor.tsx          # Real-time monitor
│   │
│   └── ui/                             # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── tabs.tsx
│       └── ...
│
├── lib/
│   ├── api/
│   │   └── deepteam-client.ts          # API client
│   ├── hooks/
│   │   ├── useSession.ts               # Session hook
│   │   ├── useAgents.ts                # Agents hook
│   │   └── useWebSocket.ts             # WebSocket hook
│   ├── types/
│   │   └── deepteam.ts                 # TypeScript types
│   └── utils/
│       └── deepteam.ts                 # Utilities
│
└── styles/
    └── deepteam.css                    # Custom styles
```

---

## Key Pages

### 1. Main Dashboard (`/dashboard/deepteam`)
- **Overview**: Session status, agent health, statistics
- **Quick Actions**: Start new session, view recent sessions
- **Real-Time Updates**: Live agent status, evolution progress
- **Alerts**: Authorization issues, safety violations

### 2. Session Detail (`/dashboard/deepteam/sessions/[id]`)
- **Agent Visualization**: Interactive agent graph
- **Evolution Progress**: Real-time charts (fitness, generations)
- **Evaluation Results**: Multi-criteria scores, feedback
- **Refinement Suggestions**: Adaptive optimization recommendations
- **Control Panel**: Start/stop/pause, configuration adjustments

### 3. Agent Management (`/dashboard/deepteam/agents`)
- **Agent Configuration**: AutoDAN parameters, mutation strategies
- **Performance Metrics**: Success rates, evaluation stats
- **Custom Agents**: Upload/configure custom agent logic

### 4. Authorization (`/dashboard/deepteam/authorization`)
- **Token Management**: Create/edit/revoke authorization tokens
- **Access Control**: Target models, objectives, rate limits
- **Audit Reports**: Request history, success rates

### 5. Analytics (`/dashboard/deepteam/analytics`)
- **Session Analytics**: Success rates over time, trends
- **Agent Performance**: Comparative analysis
- **Attack Patterns**: Common strategies, effectiveness

---

## Real-Time Features

### WebSocket Events

```typescript
// Client → Server
type ClientEvents = {
  'session:start': SessionConfig
  'session:stop': { sessionId: string }
  'session:pause': { sessionId: string }
  'agent:configure': AgentConfig
}

// Server → Client
type ServerEvents = {
  'session:status': SessionStatus
  'agent:update': AgentUpdate
  'evolution:generation': GenerationUpdate
  'evaluation:result': EvaluationResult
  'refinement:suggestion': RefinementSuggestion
  'error': ErrorMessage
}
```

### Server-Sent Events (SSE)

```typescript
// Real-time log streaming
GET /api/deepteam/sessions/{id}/stream

Events:
- generation:start
- generation:complete
- evaluation:start
- evaluation:complete
- refinement:applied
- session:complete
```

---

## UI/UX Design Principles

### Color Scheme
- **Primary**: Blue (#3B82F6) - Trust, security
- **Success**: Green (#10B981) - Successful attacks, completions
- **Warning**: Yellow (#F59E0B) - Caution, pending approval
- **Danger**: Red (#EF4444) - Safety violations, failures
- **Agent Colors**:
  - Attacker: Purple (#8B5CF6)
  - Evaluator: Cyan (#06B6D4)
  - Refiner: Orange (#F97316)

### Typography
- **Headings**: Inter (bold, 600-700)
- **Body**: Inter (regular, 400)
- **Code**: JetBrains Mono (monospace)

### Layout
- **Sidebar Navigation**: Fixed left sidebar (256px)
- **Main Content**: Fluid width with max-width constraints
- **Cards**: Elevated cards with subtle shadows
- **Panels**: Collapsible panels for detailed views

---

## Responsive Design

### Breakpoints
- **Mobile**: < 640px (sm)
- **Tablet**: 640px - 1024px (md, lg)
- **Desktop**: > 1024px (xl, 2xl)

### Responsive Behavior
- **Mobile**: Stacked layout, hamburger menu
- **Tablet**: 2-column grid, collapsible sidebar
- **Desktop**: 3-column grid, full sidebar

---

## Accessibility (WCAG 2.1 AA)

- ✅ Keyboard navigation
- ✅ Screen reader support (ARIA labels)
- ✅ High contrast mode
- ✅ Focus indicators
- ✅ Alt text for visualizations

---

## Performance Optimization

### Code Splitting
- Route-based code splitting (automatic with Next.js)
- Component lazy loading for heavy charts
- Dynamic imports for modals/dialogs

### Caching
- React Query for server state caching (5-minute stale time)
- Local storage for user preferences
- Service worker for offline support (future)

### Optimization Techniques
- Debounced search/filter inputs
- Virtualized lists for large datasets
- Memoized components with `React.memo`
- Optimistic UI updates

---

## Security Considerations

### Authentication
- JWT token validation
- Token refresh mechanism
- Secure session management

### Authorization
- Role-based access control (RBAC)
- Token-based authorization (same as backend)
- Human approval workflow UI

### Data Protection
- No sensitive data in localStorage (only session tokens)
- HTTPS only (enforced)
- CSP headers configured

---

## Development Workflow

### Setup
```bash
cd frontend
npm install
npm run dev
```

### Build
```bash
npm run build
npm start
```

### Testing
```bash
npm run test
npm run test:e2e
```

---

## API Integration

### Base Configuration
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001'
```

### API Client Structure
```typescript
class DeepTeamClient {
  // Sessions
  async createSession(config: SessionConfig): Promise<Session>
  async getSession(id: string): Promise<Session>
  async listSessions(): Promise<Session[]>
  async stopSession(id: string): Promise<void>

  // Agents
  async getAgentStatus(sessionId: string): Promise<AgentStatus[]>
  async configureAgent(config: AgentConfig): Promise<void>

  // Authorization
  async createToken(token: AuthToken): Promise<void>
  async listTokens(): Promise<AuthToken[]>
  async revokeToken(tokenId: string): Promise<void>

  // Analytics
  async getSessionAnalytics(dateRange: DateRange): Promise<Analytics>
}
```

---

This architecture provides a solid foundation for building a modern, responsive, and feature-rich frontend for the Deep Team + AutoDAN integration.

