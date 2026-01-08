# Chimera Frontend

Modern React-based frontend for the Chimera adversarial prompting and red teaming platform.

## Technology Stack

- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript 5.5+
- **UI Library**: React 19
- **Styling**: Tailwind CSS v4 with OKLCH color system
- **State Management**: TanStack Query (React Query)
- **Components**: shadcn/ui with custom glass-morphism design system
- **Real-time**: WebSocket connections with auto-reconnect

## Quick Start

### Prerequisites
- Node.js 18+ (recommend using nvm)
- npm or yarn

### Installation

```bash
# From project root
npm run install:all

# Or directly in frontend directory
cd frontend
npm install
```

### Development

```bash
# Start development server
npm run dev

# The frontend will be available at http://localhost:3000
```

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── dashboard/          # Main dashboard views
│   │   │   ├── autodan/        # AutoDAN interface
│   │   │   ├── deepteam/       # DeepTeam agent dashboard
│   │   │   ├── gptfuzz/        # GPTFuzz mutation testing
│   │   │   ├── jailbreak/      # Jailbreak generation
│   │   │   ├── models/         # Model selection & config
│   │   │   ├── providers/      # LLM provider management
│   │   │   └── techniques/     # Transformation techniques
│   │   ├── api/                # API route handlers
│   │   └── layout.tsx          # Root layout with providers
│   ├── components/             # React components
│   │   ├── ui/                 # Base UI primitives (shadcn)
│   │   ├── enhanced/           # Enhanced UI with animations
│   │   ├── providers/          # Provider components
│   │   ├── model-selector/     # Model selection components
│   │   ├── deepteam/           # DeepTeam-specific components
│   │   └── autodan/            # AutoDAN-specific components
│   ├── hooks/                  # Custom React hooks
│   │   ├── use-socket.ts       # WebSocket management
│   │   ├── use-optimistic-mutation.ts  # TanStack mutation helpers
│   │   └── use-autodan-*.ts    # AutoDAN-specific hooks
│   ├── providers/              # React context providers
│   │   ├── unified-model-provider.tsx  # Global model state
│   │   ├── chimera-provider.tsx        # Main app provider
│   │   └── query-provider.tsx          # TanStack Query setup
│   ├── lib/                    # Utility libraries
│   │   ├── api/                # API client and endpoints
│   │   └── utils.ts            # General utilities
│   └── contexts/               # Additional React contexts
├── public/                     # Static assets
├── next.config.ts              # Next.js configuration
├── tailwind.config.ts          # Tailwind CSS configuration
└── tsconfig.json               # TypeScript configuration
```

## Key Features

### 1. Multi-Provider Model Selection
Unified model selector supporting Google Gemini, OpenAI, Anthropic Claude, and DeepSeek with real-time health monitoring.

### 2. Real-time WebSocket Updates
Live updates for prompt generation, technique execution, and system health via WebSocket connections.

### 3. Glass-morphism Design System
Modern UI with backdrop blur effects, OKLCH color gradients, and subtle animations.

### 4. Performance Optimizations
- Lazy loading for heavy components
- Route prefetching
- Image optimization
- TanStack Query caching with deduplication

### 5. Research Dashboards
- **AutoDAN Dashboard**: Adversarial prompt optimization
- **GPTFuzz Interface**: Mutation-based jailbreak testing
- **DeepTeam Agents**: Multi-agent security testing
- **HouYi Optimizer**: Intent-aware optimization

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server with hot reload |
| `npm run build` | Create production build |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm test` | Run Vitest test suite |
| `npx vitest --run` | Run tests once |

## Environment Variables

Create a `.env.local` file based on `.env.example`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_WS_URL=ws://localhost:8001
NEXT_PUBLIC_ENABLE_DEVTOOLS=true
```

## Testing

```bash
# Run all tests
npx vitest --run

# Watch mode
npx vitest

# With coverage
npx vitest --coverage
```

## Browser Support

- Chrome/Edge 90+
- Firefox 90+
- Safari 15+

## Related Documentation

- [Architecture Overview](../docs/ARCHITECTURE.md)
- [User Guide](../docs/USER_GUIDE.md)
- [Developer Guide](../docs/DEVELOPER_GUIDE.md)
- [API Reference](../docs/openapi.yaml)
