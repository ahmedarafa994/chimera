# Tech Context

## Technologies Used

### Backend
-   **Language**: Python 3.11+
-   **Framework**: FastAPI
-   **Validation**: Pydantic v2
-   **Database**: PostgreSQL with Alembic migrations
-   **Caching/Rate Limiting**: Redis
-   **Observability**: OpenTelemetry
-   **Package Management**: Poetry

### Frontend
-   **Framework**: Next.js 15 (App Router)
-   **Language**: TypeScript
-   **Styling**: Tailwind CSS
-   **UI Components**: Shadcn UI, Lucide React
-   **Data Fetching**: TanStack Query (React Query), Axios
-   **Proxying**: Undici (for high-timeout backend requests)

### Adversarial Tooling
-   **Engines**: AutoDAN, HotFlip, Pandora, GCG (Greedy Coordinate Gradient)
-   **Evaluation**: Custom behavior analysis and risk scoring logic

## Development Setup
-   **Environment Variables**: Managed via `.env` files (see `.env.example`).
-   **Build System**: `npm run install:all` for root, frontend, and backend dependencies.
-   **Dev Loop**: `npm run dev` for full-stack hot-reload.
-   **Testing**: `pytest` for backend, `vitest` for frontend, `playwright` for E2E.

## Technical Constraints
1.  **High Latency AI Operations**: Adversarial engines like AutoDAN can take several minutes to complete, requiring high timeouts (600s) across the proxy and backend.
2.  **Security Sensitivity**: The tool itself handles adversarial prompts, necessitating robust internal security to prevent misuse or accidental exposure.
3.  **Model Rate Limits**: Testing against external LLM providers is subject to their respective rate limits and safety filters.
4.  **Browser Timeouts**: Standard browser timeouts must be bypassed using the Next.js proxy for long-running tasks.

## Dependencies
-   **Backend**: `fastapi`, `pydantic`, `pydantic-settings`, `sqlalchemy`, `alembic`, `redis`, `opentelemetry-api`.
-   **Frontend**: `next`, `react`, `react-dom`, `lucide-react`, `@tanstack/react-query`, `axios`, `undici`.
