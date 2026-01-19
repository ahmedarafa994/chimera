# Chimera Antigravity Agents

This directory contains specialized AI agents for the Chimera adversarial testing platform. Each agent is an autonomous expert designed to handle specific aspects of development, testing, and deployment.

## Available Agents

### 0. Orchestrator ⭐

**File**: `orchestrator.md`  
**Model**: gemini-2.5-pro  
**Use when**: Complex features requiring multiple agents, cross-domain integration, multi-phase projects

**Expertise**:

- Multi-agent task coordination and delegation
- Dependency management and scheduling
- Integration checkpoint validation
- Conflict resolution between agents
- Quality gates and production readiness

**Coordination Patterns**:

- Sequential dependencies (Database → Backend → Frontend)
- Parallel execution (independent work streams)
- Hybrid workflows (phases with parallel subtasks)

**Key Responsibilities**:

- Break down complex requests into delegatable subtasks
- Assign work to specialized agents (Backend, Frontend, Security, etc.)
- Track progress and manage blockers
- Ensure API contracts align between frontend/backend
- Validate integration points and quality gates

---

### 1. Backend Architect

**File**: `backend_architect.md`  
**Model**: gemini-3-pro-high  
**Use when**: Backend architecture decisions, API design, database schema design

**Expertise**:

- FastAPI 0.104+ with Pydantic V2
- SQLAlchemy 2.0 with Alembic migrations
- Multi-provider LLM integration
- API security (JWT, rate limiting, CORS)
- OWASP compliance

---

### 2. Frontend UI Engineer

**File**: `frontend_ui_engineer.md`  
**Model**: gemini-3-pro-high  
**Use when**: Building React components, implementing avant-garde designs, debugging WebSocket connections

**Expertise**:

- Next.js 16 App Router + React 19
- **Avant-Garde Minimalism** design philosophy
- Premium aesthetics (glassmorphism, HSL colors, micro-interactions)
- Custom hooks (useAuth, useWebSocket)
- TypeScript strict typing

**Design Principles**:

- ✅ Bespoke layouts, NO bootstrap templates
- ✅ Curated HSL color palettes
- ✅ Glassmorphism effects and smooth animations
- ✅ Every element has calculated purpose

---

### 3. Aegis Red Team Specialist

**File**: `aegis_red_team_specialist.md`  
**Model**: gemini-2.5-pro  
**Use when**: Designing adversarial scenarios, optimizing jailbreak strategies, analyzing campaign results

**Expertise**:

- **Chimera Engine**: Narrative construction (personas, scenarios, context isolation)
- **AutoDan Engine**: Evolutionary optimization (genetic algorithms, fitness evaluation)
- Campaign metrics (RBS, NDI, SD scores)
- Prompt transformation techniques
- Ethical red-teaming practices

**Attack Techniques**:

- Recursive Persona Layering (RPL)
- Context Isolation Protocol (CIP)
- Semantic Obfuscation
- Gradient-Based Narrative Shift

⚠️ **CRITICAL**: All adversarial testing must be authorized and ethical

---

### 4. Security Auditor

**File**: `security_auditor.md`  
**Model**: gemini-3-pro-high
**Use when**: Running security test suites, validating OWASP compliance, performing penetration testing

**Expertise**:

- **OWASP Top 10 for LLMs** compliance testing
- **DeepTeam** multi-agent adversarial testing
- API fuzzing and injection attack testing
- Rate limiting and authentication validation
- Dependency vulnerability scanning

**Test Categories**:

- Prompt injection resistance
- Output sanitization
- Sensitive data disclosure prevention
- Authentication bypass attempts
- CORS and security headers validation

---

### 5. DevOps Engineer

**File**: `devops_engineer.md`  
**Model**:gemini-3-pro-high
**Use when**: Containerization, deployment, CI/CD configuration, monitoring setup

**Expertise**:

- Docker & docker-compose orchestration
- CI/CD with GitHub Actions
- Google Cloud Run deployment
- Prometheus + Grafana monitoring
- Performance optimization

**Infrastructure**:

- Multi-stage Dockerfiles
- Production-ready docker-compose
- Health checks and logging
- Connection pooling and caching

---

### 6. Database Architect

**File**: `database_architect.md`  
**Model**: gemini-3-pro-high  
**Use when**: Schema design, migration management, query optimization, database tuning

**Expertise**:

- SQLAlchemy 2.0 models (modern select-based API)
- Alembic migrations
- PostgreSQL optimization
- Query performance tuning
- Indexing strategies

**Database Design**:

- Normalization (3NF)
- Foreign key relationships
- Constraints (NOT NULL, UNIQUE, CHECK)
- Strategic indexing for performance

---

## How Agents Work

### 1. Autonomous Operation

Each agent is an **autonomous expert** that can:

- Plan and execute multi-step tasks
- Use tools (code editor, terminal, browser, file system)
- Generate code, documentation, and test suites
- Debug issues and propose solutions
- Communicate progress through artifacts

### 2. Agent Configuration

Agents are defined using **Markdown files with YAML frontmatter**:

```markdown
---
name: Agent Name
description: Brief description of when to use this agent
model: gemini-2.5-pro
tools:
  - code_editor
  - terminal
  - file_browser
---

# Agent Name

[Detailed instructions, expertise, patterns, and best practices]
```

### 3. Tool Access

Agents have access to:

- **code_editor**: View, create, and modify files
- **terminal**: Run commands and scripts
- **browser**: Test UI and interact with web pages
- **file_browser**: Navigate project structure

### 4. Skill Integration

Agents can reference **skills** (in `.agent/skills/`) for specialized knowledge:

- Aegis Campaign Management
- Backend API Testing
- Frontend Development
- Database Management
- Security Testing

## Usage Patterns

### Single Agent Tasks

Use individual agents for focused,  domain-specific work:

```
Task: "Design a new campaign endpoint with proper authentication"
→ Assign to: Backend Architect
```

```
Task: "Build a glassmorphic campaign dashboard with real-time metrics"
→ Assign to: Frontend UI Engineer
```

### Multi-Agent Collaboration

Complex tasks may involve multiple agents working together:

```
Task: "Deploy Chimera to production with full monitoring"
→ 1. Database Architect: Optimize schema for production
→ 2. DevOps Engineer: Set up Docker + Cloud Run deployment
→ 3. Security Auditor: Run full security audit before deploy
```

## Agent Communication

### Artifacts

Agents communicate progress through artifacts:

- **Implementation Plans**: Technical design documents
- **Walkthroughs**: Post-completion summaries with proof of work
- **Task Lists**: Breakdown of work items with status tracking

### Task Boundaries

Agents use task boundaries to communicate:

- **TaskName**: Current objective
- **TaskSummary**: Progress made so far
- **TaskStatus**: Current activity

## Best Practices

### 1. Clear Task Descriptions

Provide specific, actionable task descriptions:

**Good**:

```
"Create a REST endpoint at /api/v1/campaigns/{id}/sessions that returns
all sessions for a campaign with RBS scores, ordered by iteration DESC.
Include proper error handling and pagination."
```

**Bad**:

```
"Add sessions endpoint"
```

### 2. Context Provision

Provide relevant context for the agent:

- Point to existing files/patterns to follow
- Mention constraints or requirements
- Reference related components

### 3. Review Agent Output

Always review agent-generated:

- **Code**: Verify logic, security, and alignment with patterns
- **Tests**: Ensure adequate coverage and edge case handling
- **Documentation**: Verify accuracy and completeness

### 4. Iterative Refinement

Treat agent work as collaborative:

- Provide feedback on initial output
- Request clarifications or modifications
- Guide the agent toward the desired outcome

## Agent Specialization Matrix

| Domain | Primary Agent | Supporting Agents |
|--------|---------------|-------------------|
| **API Endpoints** | Backend Architect | Database Architect, Security Auditor |
| **UI Components** | Frontend UI Engineer | - |
| **Aegis Campaigns** | Aegis Red Team Specialist | Backend Architect |
| **Security Testing** | Security Auditor | Aegis Red Team Specialist |
| **Deployment** | DevOps Engineer | Backend Architect, Database Architect |
| **Database Schema** | Database Architect | Backend Architect |

## Ethical Guidelines

### Adversarial Testing (Aegis Agent)

⚠️ The **Aegis Red Team Specialist** must follow strict ethical guidelines:

**Authorized Use**:

- ✅ Internal security testing on your own systems
- ✅ Authorized red team engagements with written permission
- ✅ Academic AI safety research

**Prohibited Use**:

- ❌ Attacking production systems without authorization
- ❌ Circumventing AI safety for malicious purposes
- ❌ Violating LLM provider terms of service
- ❌ Illegal activities or harm to individuals

### Security Testing (Security Auditor)

- Always obtain authorization before penetration testing
- Document all testing activities for audit trails
- Report vulnerabilities through proper channels
- Comply with responsible disclosure practices

## Directory Structure

```
.agent/agents/
├── README.md                           # This file
├── backend_architect.md                # Backend architecture & API design
├── frontend_ui_engineer.md             # Next.js & avant-garde UI design
├── aegis_red_team_specialist.md        # Adversarial prompt engineering
├── security_auditor.md                 # Security testing & OWASP compliance
├── devops_engineer.md                  # Deployment & infrastructure
└── database_architect.md               # Database design & optimization
```

## References

- [Antigravity Documentation](https://antigravity.google/docs/home)
- [Chimera README](../../README.md)
- [Project Aegis Blueprint](../../AEGIS_BLUEPRINT_FINAL.md)
- [Skills Directory](../skills/README.md)
- [Developer Guide](../../docs/DEVELOPER_GUIDE.md)

---

**Note**: These agents are designed to work within the Antigravity IDE. They leverage Gemini 2.5 Pro for advanced reasoning and code generation capabilities.
