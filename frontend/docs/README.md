# Project Chimera Frontend

A Next.js 16 dashboard for the Chimera AI Prompt Transformation Engine.

## Prerequisites

- Node.js 18+
- npm or yarn
- Backend API running on port 8001 (see `backend-api/README.md`)

## Quick Start

### 1. Start the Backend First

```bash
cd backend-api
pip install -r requirements.txt
python run.py
```

The backend will start on `http://localhost:8001`

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:3000`

### 3. Open the Dashboard

Navigate to [http://localhost:3000/dashboard](http://localhost:3000/dashboard)

## Environment Configuration

Create a `.env.local` file (or copy from `.env.local.example`):

```env
# API Configuration - Backend runs on port 8001
NEXT_PUBLIC_API_URL=http://localhost:8001/api/v1

# Development settings
NEXT_PUBLIC_DEV_MODE=true
```

## Dashboard Panels

| Panel | Route | Description |
|-------|-------|-------------|
| Overview | `/dashboard` | System status and quick access |
| Execution | `/dashboard/execution` | Transform + LLM execution |
| Generation | `/dashboard/generation` | Direct LLM interaction |
| Jailbreak | `/dashboard/jailbreak` | Advanced jailbreak generation |
| Providers | `/dashboard/providers` | LLM provider management |
| Techniques | `/dashboard/techniques` | 40+ transformation techniques |
| Metrics | `/dashboard/metrics` | Real-time system analytics |

## Troubleshooting

### "Failed to fetch" Error

This error occurs when the frontend cannot connect to the backend API.

**Quick Fix:**
1. Ensure the backend is running on port 8001
2. Check `.env.local` has the correct API URL
3. Clear cache and restart:
   ```bash
   rm -rf .next
   npm run dev
   ```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed debugging steps.

### Turbopack HMR Issues

If Hot Module Replacement isn't working:

```bash
# Use Webpack instead of Turbopack
npm run dev -- --no-turbo
```

## Development

### Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   └── dashboard/          # Dashboard routes
│   ├── components/             # React components
│   │   ├── ui/                 # shadcn/ui components
│   │   ├── layout/             # Layout components
│   │   ├── execution-panel.tsx
│   │   ├── generation-panel.tsx
│   │   ├── jailbreak-generator.tsx
│   │   ├── metrics-dashboard.tsx
│   │   ├── providers-panel.tsx
│   │   └── techniques-explorer.tsx
│   ├── lib/                    # Utilities and API client
│   │   ├── api.ts
│   │   ├── api-enhanced.ts     # Main API client
│   │   └── utils.ts
│   ├── providers/              # React context providers
│   └── types/                  # TypeScript types
├── .env.local                  # Environment variables
├── next.config.ts              # Next.js configuration
└── package.json
```

### Tech Stack

- **Framework:** Next.js 16 with App Router
- **UI:** shadcn/ui + Tailwind CSS
- **State:** TanStack Query (React Query)
- **HTTP Client:** Axios
- **Forms:** React Hook Form + Zod

## API Endpoints

The frontend connects to these backend endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/providers` | GET | List providers |
| `/api/v1/techniques` | GET | List techniques |
| `/api/v1/transform` | POST | Transform prompt |
| `/api/v1/execute` | POST | Transform + Execute |
| `/api/v1/generate` | POST | Direct LLM generation |
| `/api/v1/generation/jailbreak/generate` | POST | Jailbreak generation |
| `/api/v1/metrics` | GET | System metrics |

## Building for Production

```bash
npm run build
npm start
```

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [shadcn/ui](https://ui.shadcn.com/)
- [TanStack Query](https://tanstack.com/query)