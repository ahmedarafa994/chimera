# Repository Guidelines

## Project Structure & Module Organization
Chimera is split into Python services and a Next.js UI. `backend-api/app` contains FastAPI routers, services, and `config/` for settings; Alembic migrations live in `backend-api/alembic`. The adversarial tooling shared by agents sits under `meta_prompter/`, while the orchestrator UI is in `frontend/src` with shared UI primitives under `frontend/src/components`. Shared automation, port health checks, and compose helpers are inside `scripts/`. Test assets live both in `backend-api/tests` (API unit/integration) and top-level `tests/` for scenario and security suites. Docs and architecture notes are under `docs/` and the root `*_REPORT.md` files.

## Build, Test, and Development Commands
Run `poetry install && npm run install:all` when provisioning a new machine; the helper script installs root, frontend, and backend dependencies (Python 3.11+ required). Use `npm run dev` (alias of `npm run dev:backend` + `dev:frontend`) for a full-stack hot-reload loop, or `npm run start` for production-like execution. Backend-only work can use `poetry run chimera-dev` or `npm run dev:backend`. Frontend builds run with `npm run build:frontend`, while standalone lint/build/test cycles happen inside `frontend` via `npm run lint`, `npm run build`, and `npx vitest`. Before releasing artifacts, execute `node scripts/check-ports.js`, `node scripts/health-check.js`, and `docker-compose up -d` as needed.

## Coding Style & Naming Conventions
Python code is formatted with Black (100-char lines) and linted by Ruff; enforce typing everywhere outside tests, prefer `snake_case` modules, and treat routers/services as `PascalCase` classes. Frontend TypeScript follows ESLint + Next rules, Tailwind utility classes, and React components in `PascalCase.tsx`. Keep shared configs in `.env.example`, never commit real secrets, and run `pre-commit run --all-files` before pushing.

## Testing Guidelines
Backend API suites use `pytest` with strict markers; run `poetry run pytest --cov backend-api/app --cov meta_prompter` and keep coverage above 80% (enforced via `.coveragerc`). Security and scenario suites under `tests/` follow the same `test_*.py` naming; add new markers when tests are slow/integration-heavy. UI units live in `frontend` and should run via `npx vitest --run`. Cross-browser smoke tests rely on Playwright, so execute `npx playwright test` after major UI changes.

## Commit & Pull Request Guidelines
Match existing history: start subjects with a lowercase conventional prefix (`feat:`, `fix:`, `refactor:`) and append ticket identifiers when applicable (`P2-FIX-008`). PRs must describe the motivation, link issues, list validation commands, and include screenshots or API traces for UI/API work. Block PRs until `poetry run pytest`, `npm run lint`, and relevant Playwright suites are green; mention any skipped coverage explicitly.

## Security & Configuration Tips
Copy `.env.example` files into `.env` per module, keeping secrets in your local vault. When exposing services, prefer `docker-compose.prod.yml` and validate via `npm run health`. Retain logs in `logs/` for audits and never check sensitive exports into Git.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
