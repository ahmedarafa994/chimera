# ADR 0002: Frontend Architecture with Next.js 16 and React 19

## Status

Accepted

## Date

2026-01-01

## Context

The Chimera frontend needs to provide a modern, responsive, and performant user interface for AI security research tools. Key requirements include:

- Real-time updates for long-running operations (AutoDAN, GPTFuzz)
- Complex state management across multiple features
- Server-side rendering for SEO and initial load performance
- TypeScript for type safety
- Tailwind CSS for consistent styling
- Support for concurrent features (React 19)

## Decision

We will use **Next.js 16** with **React 19** and the following architecture:

### Technology Stack

| Layer | Technology |
|-------|------------|
| Framework | Next.js 16 (App Router) |
| UI Library | React 19 |
| State Management | Zustand + TanStack Query |
| Styling | Tailwind CSS v4 (OKLCH colors) |
| Type System | TypeScript 5.x (strict mode) |
| Testing | Vitest + Playwright |
| WebSocket | Native WebSocket with reconnection |

### Directory Structure

```
frontend/src/
├── app/                    # Next.js App Router pages
│   ├── dashboard/         # Dashboard routes
│   │   ├── autodan/
│   │   ├── jailbreak/
│   │   └── metrics/
│   └── api/               # API routes (proxies)
├── components/            # React components
│   ├── enhanced/          # Enhanced UI components
│   ├── jailbreak/         # Jailbreak-specific
│   ├── model-selector/    # Model selection
│   └── layout/            # Layout components
├── hooks/                 # Custom React hooks
├── lib/                   # Utilities and services
│   ├── api/               # API client
│   ├── stores/            # Zustand stores
│   └── websocket/         # WebSocket manager
└── types/                 # TypeScript types
```

### State Management Strategy

1. **Server State (TanStack Query)**
   - API responses
   - Caching and invalidation
   - Background refetching

2. **Client State (Zustand)**
   - UI state (modals, sidebar)
   - User preferences
   - Form state

3. **URL State (Next.js)**
   - Page parameters
   - Search filters
   - Navigation state

### Component Patterns

```typescript
// Server Component (default)
export default async function Dashboard() {
  const data = await fetchData();
  return <DashboardView data={data} />;
}

// Client Component
'use client';
export function InteractivePanel() {
  const [state, setState] = useState();
  return <Panel onChange={setState} />;
}
```

### Real-time Architecture

```
┌─────────────┐     WebSocket     ┌─────────────┐
│   Browser   │◄─────────────────►│   Backend   │
└─────────────┘                   └─────────────┘
       │                                 │
       │ useWebSocket()                  │
       ▼                                 │
┌─────────────┐                          │
│ WS Manager  │◄─────── Events ──────────┘
└─────────────┘
       │
       │ Zustand Store
       ▼
┌─────────────┐
│  UI Update  │
└─────────────┘
```

## Consequences

### Positive

- **Performance**: Server components reduce JavaScript bundle
- **Type Safety**: End-to-end TypeScript coverage
- **Developer Experience**: Hot reload, excellent tooling
- **SEO**: Server-side rendering support
- **Concurrent Features**: React 19 transitions and Suspense

### Negative

- **Learning Curve**: App Router patterns differ from Pages Router
- **Complexity**: Server/Client component boundaries
- **Bundle Size**: Large framework despite tree-shaking

### Migration Notes

- All `any` types being replaced with proper types
- Using `unknown` with type guards for external data
- Strict TypeScript configuration enforced

## Alternatives Considered

1. **Vite + React**: Faster builds but no SSR out of box
2. **Remix**: Good alternative but smaller ecosystem
3. **Vue/Nuxt**: Different paradigm, team expertise in React

## References

- [Next.js 16 Documentation](https://nextjs.org/docs)
- [React 19 Release Notes](https://react.dev/blog)
- [Zustand Best Practices](https://zustand-demo.pmnd.rs/)
