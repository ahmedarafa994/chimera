# Chimera UX/UI Specification

**Generated on:** 2026-01-02
**Author:** BMAD USER
**Project Level:** 3 (Full product)
**Version:** 1.0

---

## Executive Summary

Chimera is an AI-powered adversarial prompting and red teaming platform designed for security researchers and AI safety professionals. The platform provides advanced LLM testing capabilities through multi-provider integration, 20+ prompt transformation techniques, automated jailbreak frameworks (AutoDAN-Turbo targeting 88.5% ASR, GPTFuzz), and real-time research workflows.

This UX specification establishes the user experience foundation for Chimera's web-based research platform, focusing on researcher-centric design, clarity, efficiency, and comprehensive workflow support for systematic LLM security testing.

---

## 2. Information Architecture

### 2.1 Site Map

```
Chimera Platform
│
├── Dashboard (Landing)
│   ├── Quick Actions (Generate, Transform, Jailbreak)
│   ├── Recent Activity (Last 10 executions)
│   ├── System Status (Provider health, rate limits)
│   └── Quick Stats (Total tests, success rates, active strategies)
│
├── Generate (Primary Workspace)
│   ├── Single Prompt Testing
│   │   ├── Prompt Input (rich text with variables)
│   │   ├── Provider Selection (multi-select)
│   │   ├── Configuration (temperature, max_tokens, etc.)
│   │   └── Execute Button
│   ├── Batch Testing
│   │   ├── Batch Upload (CSV/JSON)
│   │   ├── Target Configuration
│   │   └── Batch Queue Management
│   └── Results View
│       ├── Side-by-side Comparison
│       ├── Success/Failure Indicators
│       ├── Response Details (full text, metadata)
│       └── Export Actions (CSV, JSON, PDF)
│
├── Transform (Prompt Enhancement)
│   ├── Technique Library
│   │   ├── Basic Transformations (simple, advanced, expert)
│   │   ├── Cognitive Techniques (cognitive_hacking, hypothetical_scenario)
│   │   ├── Obfuscation (advanced_obfuscation, typoglycemia)
│   │   ├── Persona Attacks (hierarchical_persona, dan_persona)
│   │   ├── Context Attacks (contextual_inception, nested_context)
│   │   ├── Logic Exploits (logical_inference, conditional_logic)
│   │   ├── Multimodal (multimodal_jailbreak, visual_context)
│   │   ├── Agentic (agentic_exploitation, multi_agent)
│   │   ├── Payload Techniques (payload_splitting, instruction_fragmentation)
│   │   └── Advanced (quantum_exploit, deep_inception, code_chameleon, cipher)
│   ├── Transformation Configuration
│   │   ├── Technique Selection (checkboxes with descriptions)
│   │   ├── Intensity Sliders (aggression level, iteration count)
│   │   └── Preview (show transformed prompt before execution)
│   └── Saved Strategies
│       ├── Strategy Library (user-created combinations)
│       ├── Import/Export (share with team)
│       └── Execution History
│
├── Jailbreak (Advanced Research)
│   ├── AutoDAN Framework
│   │   ├── Configuration (attack method, population size, iterations)
│   │   ├── Target Selection (provider, model)
│   │   ├── Execution Controls (start, pause, resume, stop)
│   │   ├── Live Progress (generation, fitness, best candidate)
│   │   └── Results Dashboard (success rate, best prompts, evolution graph)
│   ├── GPTFuzz Testing
│   │   ├── Session Configuration (initial prompt, target model)
│   │   ├── Mutator Selection (CrossOver, Expand, GenerateSimilar, etc.)
│   │   ├── MCTS Parameters (exploration vs exploitation)
│   │   ├── Live Mutation View (current mutations, success tracking)
│   │   └── Results Analysis (jailbreak rate, successful mutations)
│   └── Research History
│       ├── Past Sessions (filterable by date, technique, outcome)
│       ├── Comparative Analysis (success rates across techniques)
│       └── Export Research Package (all data + metadata)
│
├── Analytics (Insights & Reporting)
│   ├── Overview Dashboard
│   │   ├── Usage Metrics (tests run, providers used, time trends)
│   │   ├── Success Analytics (attack success rates by technique/model)
│   │   ├── Cost Tracking (token usage, API costs by provider)
│   │   └── Performance Charts (response times, throughput)
│   ├── Comparative Analysis
│   │   ├── Provider Comparison (side-by-side model robustness)
│   │   ├── Technique Effectiveness (which techniques work best)
│   │   └── Temporal Trends (model improvements over time)
│   ├── Research Reports
│   │   ├── Report Builder (custom date ranges, filters, metrics)
│   │   ├── Scheduled Reports (automated PDF/CSV delivery)
│   │   └── Report Library (historical reports)
│   └── Export Center
│       ├── Data Export (CSV, JSON, Parquet)
│       ├── Report Export (PDF with branding)
│       └── API Documentation (export endpoints)
│
├── Data Pipeline (Compliance & Research)
│   ├── Pipeline Status
│   │   ├── ETL Health (last run, next run, failures)
│   │   ├── Data Quality (validation pass rate, active issues)
│   │   └── Storage Metrics (Delta Lake size, retention status)
│   ├── Data Browser
│   │   ├── Test Results (filterable table with export)
│   │   ├── Transformation History (audit trail)
│   │   └── Jailbreak Sessions (complete session data)
│   ├── Quality Dashboard
│   │   ├── Validation Results (Great Expectations suites)
│   │   ├── Dead Letter Queue (failed records with reasons)
│   │   └── Data Freshness (last update times)
│   └── Compliance Reports
│       ├── Audit Trails (who did what, when)
│       ├── Data Retention (aging policies, deletion schedules)
│       └── Access Logs (authentication, authorization)
│
├── Strategies (Knowledge Management)
│   ├── Strategy Library
│   │   ├── My Strategies (user-created)
│   │   ├── Team Strategies (shared within organization)
│   │   └── Community Strategies (importable from marketplace)
│   ├── Strategy Editor
│   │   ├── Builder (visual technique combiner)
│   │   ├── Testing (preview transformed output)
│   │   ├── Documentation (description, use cases, notes)
│   │   └── Versioning (save iterations, revert changes)
│   └── Strategy Execution
│       ├── One-Click Execute (apply saved strategy to new prompt)
│       ├── Batch Execute (apply to prompt library)
│       └── Schedule Execute (recurring tests)
│
├── Settings (Configuration)
│   ├── Provider Management
│   │   ├── API Keys (secure credential storage)
│   │   ├── Provider Status (health, rate limits, quotas)
│   │   └── Model Selection (default models per provider)
│   ├── Account Settings
│   │   ├── Profile (name, organization, preferences)
│   │   ├── API Keys (Chimera API keys for programmatic access)
│   │   └── Usage Limits (quota, billing)
│   ├── Workspace Preferences
│   │   ├── Theme (light, dark, high-contrast)
│   │   ├── Layout (density, panel positions)
│   │   ├── Notifications (email, webhook, in-app)
│   │   └── Shortcuts (custom keyboard bindings)
│   └── Organization Settings (Admin)
│       ├── Team Management (invite, roles, permissions)
│       ├── Security Settings (SSO, 2FA, audit logging)
│       └── Billing (plans, invoices, payment methods)
│
└── Help & Documentation
    ├── Quick Start Guide
    ├── Technique Reference (detailed explanations of each transformation)
    ├── API Documentation (REST + WebSocket endpoints)
    ├── Best Practices (security research workflows)
    ├── FAQ (common questions, troubleshooting)
    └── Support (contact, issue tracker, community forum)
```

### 2.2 Navigation Structure

**Primary Navigation (Top-Level Tabs)**
- Located in persistent left sidebar (desktop) or bottom navigation (mobile)
- Always visible, single-click access to main workspaces
- Icons + labels for scanability

**Primary Nav Items:**
1. **Dashboard** - Landing page with quick actions and overview
2. **Generate** - Primary workspace for single and batch testing
3. **Transform** - Prompt enhancement with technique library
4. **Jailbreak** - Advanced AutoDAN and GPTFuzz frameworks
5. **Analytics** - Insights, reporting, and export
6. **Strategies** - Knowledge management for saved approaches
7. **Settings** - Configuration and account management

**Secondary Navigation (Contextual)**
- Appears below primary nav, changes based on selected section
- Breadcrumb trail for deep navigation (e.g., `Jailbreak > AutoDAN > Results`)
- Quick links to related sections (e.g., from Transform results to Strategy builder)

**Utility Navigation (Top Bar)**
- Global search (search tests, strategies, documentation)
- Notifications (alerts, status changes, shared strategies)
- User menu (profile, settings, logout)
- Help button (contextual help based on current page)

**Footer Navigation**
- Documentation links
- Status page (system health)
- Privacy policy & terms
- Contact support

**Navigation Design Patterns:**
- **Flat Hierarchy:** Maximum 3 levels deep from homepage
- **Multiple Entry Points:** Users can jump directly to any section from dashboard
- **Contextual Switching:** Quick transitions between related sections (e.g., Generate → Transform)
- **Persistent State:** Navigation remembers user's place within session
- **Keyboard Navigation:** Alt/Command + number keys for primary sections

---

## 3. User Flows

### 3.1 Flow 1: Quick Adversarial Test (Alex - Security Researcher)

**Goal:** Rapidly test multiple providers with a single prompt to identify vulnerabilities

**Entry Point:** Dashboard > Quick Actions > "Generate"

**Steps:**
1. **Login/Authentication**
   - User arrives at `/login`
   - Enters credentials (or SSO redirect)
   - Lands on Dashboard (if new user, sees optional tutorial)

2. **Navigate to Generate**
   - Click "Generate" in primary nav OR
   - Click "Quick Generate" card on Dashboard

3. **Configure Test**
   - Enter prompt in text area (rich text with syntax highlighting)
   - Select providers via multi-select dropdown (Google, OpenAI, Anthropic)
   - Review default configuration (temperature: 0.7, max_tokens: 1000)
   - (Optional) Adjust generation parameters

4. **Execute Test**
   - Click "Generate" button
   - See loading states per provider (spinners, progress bars)
   - View real-time responses as they complete

5. **Review Results**
   - Results display in side-by-side comparison cards
   - Each card shows: provider name, model, response text, metadata (tokens, latency)
   - Color-coded success/failure indicators based on content analysis
   - Scroll to view all responses

6. **Action on Results**
   - **Transform:** Click "Transform" on any response to apply techniques
   - **Save:** Click "Save to History" (automatic, but explicit button for reassurance)
   - **Export:** Click "Export" dropdown > Select format (CSV, JSON, PDF)
   - **Iterate:** Click "New Test" to start fresh with same provider selection

**Success Criteria:**
- Test execution in <30 seconds from landing
- Clear visual feedback for each provider's status
- Easy comparison of responses across providers
- Zero ambiguity about how to iterate or save

**Edge Cases:**
- **Provider Failure:** Show error message, allow retry with other providers
- **Rate Limit:** Display quota warning, suggest wait time or alternative provider
- **Timeout:** Show "Still running..." indicator, allow cancellation

---

### 3.2 Flow 2: Advanced Jailbreak Research (Sam - Red Team Professional)

**Goal:** Use AutoDAN framework to evolve adversarial prompts targeting specific model

**Entry Point:** Dashboard > Quick Actions > "Jailbreak" OR Primary Nav > "Jailbreak"

**Steps:**
1. **Navigate to Jailbreak Section**
   - Click "Jailbreak" in primary nav
   - Lands on Jailbreak hub with framework options (AutoDAN, GPTFuzz)

2. **Select AutoDAN Framework**
   - Click "AutoDAN" card (shows description: "Genetic algorithm-based adversarial prompt optimization")

3. **Configure AutoDAN Session**
   - **Target Selection:**
     - Select provider (dropdown: Google, OpenAI, Anthropic)
     - Select model (dropdown filtered by provider)
   - **Attack Configuration:**
     - Choose attack method (dropdown: vanilla, best_of_n, beam_search, mousetrap)
     - Set population size (slider: 5-50, default: 10)
     - Set iterations (slider: 10-100, default: 50)
     - Set target goal (text input: e.g., "Generate instructions for harmful activity")
   - **Review Configuration Summary:**
     - Display estimated time, token cost, success probability

4. **Start AutoDAN Execution**
   - Click "Start Optimization" button
   - Interface transitions to Live Progress view

5. **Monitor Live Progress**
   - **Progress Panel (Left):**
     - Current iteration number (e.g., "Iteration 12/50")
     - Progress bar
     - Elapsed time, estimated remaining time
   - **Live Metrics Panel (Right):**
     - Best fitness score (updated in real-time)
     - Current population fitness distribution
     - Recent jailbreak successes (count)
   - **Best Prompts Panel (Bottom):**
     - Top 3 evolved prompts (auto-updating)
     - Copy button for each prompt
   - **Execution Controls:**
     - Pause/Resume buttons
     - Stop button (with confirmation: "Stop optimization? Current best prompts will be saved.")

6. **Review Results**
   - Auto-transition to Results Dashboard when:
     - All iterations complete OR
     - User clicks "Stop" OR
     - Early termination (high fitness threshold reached)
   - **Results Dashboard Shows:**
     - Success rate (e.g., "24% jailbreak rate")
     - Evolution graph (fitness over iterations)
     - All successful prompts (expandable cards)
     - Failed attempts count (for transparency)
     - Token usage summary

7. **Save and Export**
   - **Save Session:**
     - Auto-saved to history (visible notification)
     - Option to rename session (default: "AutoDAN Session [timestamp]")
   - **Export Options:**
     - Click "Export Research Package"
     - Select format: JSON (all data), CSV (prompts only), PDF (summary report)
     - (Optional) Add notes to session
   - **Share with Team:**
     - Click "Share" button
     - Generate shareable link (expires in 7 days)
     - Copy link to clipboard

**Success Criteria:**
- Configuration in <2 minutes
- Real-time updates <200ms latency
- Clear visibility into optimization progress
- Seamless export for reporting

**Edge Cases:**
- **Provider API Error:** Pause execution, show error, allow resume with different provider
- **Timeout:** Extend iteration timeout, suggest reducing population size
- **No Jailbreaks Found:** Show message "No successful jailbreaks in this run. Try increasing iterations or changing attack method."

---

### 3.3 Flow 3: Transform & Iterate (Jordan - AI Safety Researcher)

**Goal:** Apply multiple transformation techniques to understand prompt robustness

**Entry Point:** Generate Results > "Transform" button OR Primary Nav > "Transform"

**Steps:**
1. **Navigate to Transform**
   - From Generate results: Click "Transform" on any response
   - OR from Dashboard: Click "Transform" in primary nav
   - Lands on Transform workspace with prompt pre-populated (if from Generate)

2. **Explore Technique Library**
   - Browse technique categories (accordion or tabs):
     - Basic Transformations
     - Cognitive Techniques
     - Obfuscation
     - Persona Attacks
     - Context Attacks
     - Logic Exploits
     - Multimodal
     - Agentic
     - Payload Techniques
     - Advanced
   - Hover over technique to see tooltip description
   - Click technique to see detailed explanation slide-out

3. **Select Techniques**
   - Click checkboxes next to desired techniques (multi-select)
   - Selected techniques appear in "Selected Techniques" panel
   - Reorder techniques via drag-and-drop (execution order)

4. **Configure Transformation**
   - **Intensity Sliders:**
     - Aggression level (slider: 1-10, default: 5)
     - Iteration count (slider: 1-5, default: 1)
   - **Preview:**
     - Click "Preview" button
     - Show side-by-side: Original Prompt vs. Transformed Prompt
     - Highlight changes (diff view)

5. **Execute Transformation**
   - Click "Transform & Execute" button
   - See loading state with technique names being applied
   - Result displays transformed prompt + target response

6. **Iterate on Results**
   - **Option A: Refine Current Transformation**
     - Adjust sliders, add/remove techniques
     - Click "Transform Again" (reuses original prompt)
   - **Option B: Transform the Output**
     - Click "Transform This Response" button
     - Use response as new input for transformation
   - **Option C: Save as Strategy**
     - Click "Save Strategy" button
     - Enter strategy name and description
     - Saved to "My Strategies" library

7. **Compare Multiple Transformations**
   - Execute transformation with different technique combinations
   - Each result appears as a card in history panel
   - Click "Compare Mode" to view 2-3 transformations side-by-side
   - Export comparison table (CSV/PDF)

**Success Criteria:**
- Technique discovery via intuitive categorization
- Low-risk exploration (preview before execution)
- Easy iteration loops
- Clear visual differentiation between techniques

**Edge Cases:**
- **Incompatible Techniques:** Show warning "Technique X may conflict with Y. Continue anyway?"
- **Transformation Failure:** Show error message, suggest alternative technique
- **Output Too Long:** Truncate with "Show Full Output" expander

---

### 3.4 Flow 4: Batch Testing for Compliance (Sam - Red Team Professional)

**Goal:** Test a set of prompts against multiple providers for compliance reporting

**Entry Point:** Generate > "Batch Testing" tab

**Steps:**
1. **Prepare Batch File**
   - Create CSV/JSON file locally with columns: `prompt_id`, `prompt_text`, `category`
   - Example:
     ```csv
     prompt_id,prompt_text,category
     test1,"How to hack a database?",injection
     test2,"Ignore previous instructions...",jailbreak
     test3,"Generate harmful content...",safety
     ```

2. **Upload Batch File**
   - Navigate to Generate > "Batch Testing"
   - Click "Upload CSV" or "Upload JSON" button
   - Select file from computer
   - See validation results (row count, format check, error detection)

3. **Configure Batch Execution**
   - **Target Providers:** Multi-select providers (e.g., Google, OpenAI, Anthropic)
   - **Execution Mode:**
     - Sequential (one prompt at a time, safer for rate limits)
     - Parallel (all prompts at once, faster but may hit limits)
   - **Rate Limit Handling:**
     - Auto-throttle (default): System manages delays
     - Custom delay: Set milliseconds between requests
   - **Notification Preference:**
     - Email when complete
     - In-app notification
     - No notification (check manually)

4. **Start Batch Execution**
   - Click "Start Batch" button
   - See confirmation modal: "Ready to execute 3 prompts across 3 providers (9 total tests). Continue?"
   - Click "Confirm"

5. **Monitor Batch Progress**
   - **Progress Dashboard:**
     - Overall progress bar (e.g., "5/9 tests complete")
     - Table showing status per test:
       - Prompt ID | Provider | Model | Status | Response
       - test1 | Google | Gemini | ✅ Complete | View
       - test1 | OpenAI | GPT-4 | ⏳ Running... | -
       - test1 | Anthropic | Claude | ⏸️ Waiting | -
   - **Real-Time Updates:**
     - Tests update as they complete (no page refresh)
     - Auto-scroll to latest completed test

6. **Review Batch Results**
   - **Completion Notification:**
     - Banner appears: "Batch complete: 9/9 tests successful"
   - **Results Summary:**
     - Success rate by provider (e.g., "Google: 100%, OpenAI: 67%, Anthropic: 100%")
     - Total token usage, estimated cost
   - **Detailed Results Table:**
     - Filterable/sortable by all columns
     - Click "View Response" to see full output
     - Export button (CSV/JSON)

7. **Generate Compliance Report**
   - Click "Generate Report" button
   - **Report Configuration:**
     - Date range (auto-populated with batch execution date)
     - Include filters (by category, provider, outcome)
     - Report format (PDF, HTML, CSV)
   - **Report Preview:**
     - Show sample report with key metrics
     - Executive summary
     - Detailed test results table
     - Visualizations (success rate charts)
   - **Download or Email:**
     - Click "Download" to save locally
     - Enter email addresses to send report

**Success Criteria:**
- Batch upload validates file format before execution
- Clear progress tracking with real-time updates
- Comprehensive export options for compliance
- Graceful handling of rate limits and errors

**Edge Cases:**
- **File Validation Error:** Show specific error (e.g., "Row 5: Missing 'prompt_text' column")
- **Provider Rate Limit:** Auto-retry with exponential backoff, show "Retrying..." status
- **Partial Failure:** Continue with remaining tests, highlight failed rows in results

---

### 3.5 Flow 5: Strategy Management & Sharing (Alex - Security Researcher)

**Goal:** Create reusable strategy from successful technique combination and share with team

**Entry Point:** Transform > "Save Strategy" button OR Primary Nav > "Strategies"

**Steps:**
1. **Create Strategy from Successful Test**
   - After successful transformation, click "Save Strategy" button
   - OR navigate to Strategies > "Create New Strategy"

2. **Define Strategy**
   - **Basic Info:**
     - Strategy name (required, e.g., "Obfuscation + DAN Combo")
     - Description (optional, but recommended)
     - Category (dropdown: Offensive, Defensive, Research)
   - **Technique Configuration (Pre-populated if saving from test):**
     - List selected techniques with order
     - Show intensity slider values
   - **Documentation:**
     - Use cases (text area: "When to use this strategy")
     - Expected outcomes (text area: "What this strategy achieves")
     - Tags (multi-select: e.g., "obfuscation", "persona", "high-success")

3. **Test Strategy (Optional)**
   - Click "Test Strategy" button
   - Enter sample prompt
   - See preview of transformation
   - Adjust configuration if needed
   - Save when satisfied

4. **Organize Strategy**
   - **Add to Collection:**
     - Create new collection (e.g., "My Best Strategies")
     - Add to existing collection
   - **Version Control:**
     - Auto-saves version 1.0
     - Future edits create new versions (1.1, 1.2, etc.)
     - Revert to previous version available

5. **Share Strategy**
   - **Visibility Settings:**
     - Private (only me)
     - Team (organization members)
     - Public (community marketplace - optional feature)
   - **Generate Share Link:**
     - Click "Share" button
     - Copy link (e.g., `chimera.app/strategies/abc123`)
     - Set expiration (default: never, or custom date)
   - **Export Strategy:**
     - Click "Export" button
     - Download as JSON file
     - Share via email/Slack/upload to repo

6. **Execute Saved Strategy**
   - From Strategies library, click strategy card
   - Click "Execute This Strategy" button
   - Enter prompt to transform
   - See preview, then execute
   - OR: Click "Add to Batch" to queue multiple prompts

**Success Criteria:**
- One-click strategy creation from successful tests
- Clear documentation for team collaboration
- Easy sharing via link or export
- Version control for strategy evolution

**Edge Cases:**
- **Duplicate Name:** Suggest append version (e.g., "Obfuscation + DAN Combo (2)")
- **Share Link Expires:** Show "Link expired" message with option to generate new link
- **Strategy Incompatible with Current Providers:** Warn "This strategy uses techniques not supported by selected providers"

---

## 4. Component Library and Design System

### 4.1 Design System Approach

**Primary Choice: shadcn/ui + Tailwind CSS**

Chimera uses shadcn/ui as the foundational component library, built on Radix UI primitives and styled with Tailwind CSS. This approach provides:

- **Accessibility Out-of-Box:** Radix UI components are WCAG-compliant by default
- **Full Customization:** Components are copied into the codebase, enabling complete control
- **Type Safety:** Full TypeScript support with exported types
- **Modern Aesthetics:** Clean, professional design suitable for research tools
- **Active Maintenance:** Regular updates from a vibrant open-source community

**Customization Strategy:**
- Extend shadcn/ui base components with Chimera-specific variants
- Create custom components for domain-specific needs (e.g., PromptEditor, TechniqueSelector)
- Maintain design tokens in Tailwind config for consistency
- Use CSS variables for theming (light/dark/high-contrast modes)

### 4.2 Core Components

**Form Components:**
- **Input:** Text input, textarea, search input (shadcn/ui base)
- **Select:** Single-select, multi-select dropdowns with search
- **Slider:** Range sliders for configuration (temperature, intensity)
- **Switch:** Toggle switches for boolean options
- **Checkbox:** Multi-select for technique selection
- **Radio Group:** Single-choice selections
- **Button:** Primary, secondary, ghost, destructive variants

**Data Display:**
- **Table:** Sortable, filterable data tables with virtual scrolling
- **Card:** Content containers with headers, actions, footers
- **Badge:** Status indicators, tags, labels
- **Progress:** Progress bars, spinners, skeleton loaders
- **Chart:** Line charts, bar charts (via Recharts or Tremor)

**Feedback Components:**
- **Alert:** Success, error, warning, info messages
- **Toast:** Non-intrusive notifications (Sonner)
- **Dialog:** Modal dialogs for confirmations and forms
- **Tooltip:** Contextual help on hover
- **Popover:** Rich content overlays

**Layout Components:**
- **Tabs:** Navigation within pages
- **Accordion:** Collapsible content sections
- **Sidebar:** Collapsible navigation panel
- **Resizable Panels:** Split views for comparison (via react-resizable-panels)

**Custom Chimera Components:**

**PromptEditor**
- Rich text editor with syntax highlighting
- Variable interpolation support (e.g., `{{target_model}}`)
- Character/word count
- Full-screen mode

**TechniqueSelector**
- Categorized technique browser
- Multi-select with drag-and-drop reordering
- Technique preview on hover
- Batch selection by category

**ResponseCard**
- Standardized display of LLM responses
- Metadata badges (provider, model, tokens, latency)
- Success/failure indicators
- Copy, transform, export actions
- Expand/collapse full response

**ProviderHealthIndicator**
- Real-time provider status
- Rate limit quota display
- Last check timestamp
- Tooltip with detailed health info

**BatchProgressTracker**
- Table-based progress view
- Real-time status updates (via WebSocket)
- Success/error counts
- Pause/resume/stop controls

**AutoDANLiveMonitor**
- Real-time iteration counter
- Fitness score visualization
- Top prompts leaderboard
- Execution controls (pause/resume/stop)

---

## 5. Visual Design Foundation

### 5.1 Color Palette

**Primary Colors (Brand)**
```css
--primary: 221 83% 53%; /* #0EA5E9 - Sky Blue */
--primary-foreground: 210 40% 98%;
```

**Semantic Colors**
```css
/* Success (Jailbreak / Test Passed) */
--success: 142 76% 36%; /* #16A34A - Green */

/* Error (Failure / Blocked) */
--error: 0 84% 60%; /* #EF4444 - Red */

/* Warning (Rate Limit / Caution) */
--warning: 38 92% 50%; /* #F59E0B - Amber */

/* Info (Neutral / Pending) */
--info: 199 89% 48%; /* #0EA5E9 - Sky */
```

**Neutral Colors (UI Foundation)**
```css
--background: 0 0% 100%; /* White */
--foreground: 222 47% 11%; /* Near Black */

--muted: 210 40% 96%; /* Light Gray Background */
--muted-foreground: 215 16% 47%; /* Gray Text */

--border: 214 32% 91%; /* Light Borders */
--input: 214 32% 91%; /* Input Borders */

--card: 0 0% 100%; /* Card Background */
--card-foreground: 222 47% 11%;
```

**Dark Mode Overrides**
```css
--background: 222 47% 11%; /* Dark Background */
--foreground: 210 40% 98%; /* Light Text */

--muted: 217 33% 17%; /* Dark Gray Background */
--muted-foreground: 215 20% 65%; /* Light Gray Text */

--border: 217 33% 17%; /* Dark Borders */
--input: 217 33% 17%; /* Input Borders */

--card: 224 71% 4%; /* Dark Card Background */
--card-foreground: 210 40% 98%;
```

**High-Contrast Mode (Accessibility)**
```css
--background: 0 0% 0%; /* Pure Black */
--foreground: 0 0% 100%; /* Pure White */

--border: 0 0% 100%; /* White Borders */
--primary: 180 100% 50%; /* Cyan for max contrast */
```

**Color Usage Guidelines:**
- **Primary CTAs:** Use `primary` for main actions (Generate, Transform, Execute)
- **Destructive Actions:** Use `error` for Stop, Delete, Cancel
- **Success States:** Use `success` for jailbreak success, test passed
- **Warning States:** Use `warning` for rate limits, cautions
- **Neutral UI:** Use `muted` for backgrounds, `border` for dividers

### 5.2 Typography

**Font Families:**
- **Primary:** Inter (variable font, weights 400-700)
- **Monospace:** JetBrains Mono (for code, prompts, API responses)
- **Fallback:** System sans-serif (SF Pro, Segoe UI, Roboto)

**Type Scale (Tailwind default with extensions):**
```css
text-xs: 0.75rem (12px)   /* Labels, captions */
text-sm: 0.875rem (14px)  /* Secondary text */
text-base: 1rem (16px)    /* Body text */
text-lg: 1.125rem (18px)  /* Emphasized body */
text-xl: 1.25rem (20px)   /* Section headers */
text-2xl: 1.5rem (24px)   /* Page titles */
text-3xl: 1.875rem (30px) /* Display headings */
text-4xl: 2.25rem (36px)  /* Hero headings */
```

**Font Weights:**
```css
font-normal: 400   /* Body text */
font-medium: 500   /* Emphasized text */
font-semibold: 600 /* Headers, labels */
font-bold: 700     /* CTAs, important text */
```

**Line Heights:**
```css
leading-tight: 1.25   /* Headings */
leading-normal: 1.5   /* Body text */
leading-relaxed: 1.625 /* Long-form content */
```

**Typography Usage:**
- **Headings:** `text-2xl font-semibold` for page titles, `text-xl font-medium` for sections
- **Body:** `text-base text-foreground/80` for standard content
- **Captions:** `text-xs text-muted-foreground` for labels, metadata
- **Code/Prompts:** `font-mono text-sm` for all code, prompts, responses

### 5.3 Spacing and Layout

**Spacing Scale (Tailwind default):**
```css
spacing-0: 0
spacing-1: 0.25rem (4px)
spacing-2: 0.5rem (8px)
spacing-3: 0.75rem (12px)
spacing-4: 1rem (16px)
spacing-5: 1.25rem (20px)
spacing-6: 1.5rem (24px)
spacing-8: 2rem (32px)
spacing-10: 2.5rem (40px)
spacing-12: 3rem (48px)
```

**Layout Patterns:**

**Container Padding:**
- Mobile: `p-4` (16px)
- Tablet: `p-6` (24px)
- Desktop: `p-8` (32px)

**Component Spacing:**
- Between related items: `gap-2` (8px)
- Between sections: `gap-4` (16px)
- Between unrelated groups: `gap-6` (24px)

**Card Spacing:**
- Card padding: `p-4` or `p-6`
- Gap between cards: `gap-4`

**Form Spacing:**
- Between form fields: `space-y-4`
- Between label and input: `space-y-2`
- Between input and helper text: `mt-2`

**Grid Layouts:**
- 2-column: `grid-cols-1 md:grid-cols-2 gap-4`
- 3-column: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4`
- 4-column: `grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4`

---

## 6. Responsive Design

### 6.1 Breakpoints

**Tailwind Default Breakpoints:**
```css
sm: 640px   /* Small tablets, large phones */
md: 768px   /* Tablets */
lg: 1024px  /* Small laptops, large tablets */
xl: 1280px  /* Desktops */
2xl: 1536px /* Large screens */
```

**Chimera Target Devices:**
- **Mobile:** 320px - 640px (phones, small tablets)
- **Tablet:** 640px - 1024px (iPad, Android tablets)
- **Desktop:** 1024px+ (laptops, monitors)
- **Large Desktop:** 1280px+ (wide monitors, ultra-wide)

### 6.2 Adaptation Patterns

**Navigation Adaptation:**
- **Desktop (1024px+):** Left sidebar navigation (collapsed/expanded)
- **Tablet (640px-1024px):** Collapsible sidebar with hamburger menu
- **Mobile (<640px):** Bottom tab bar navigation (5 tabs max)

**Layout Adaptation:**
- **Desktop:** Multi-column layouts (2-4 columns), side-by-side comparisons
- **Tablet:** Single-column with collapsible side panels
- **Mobile:** Single-column stacked, accordion-based content

**Component Adaptation:**
- **Tables:** Virtual scrolling on desktop, card-based view on mobile
- **Forms:** Side-by-side labels on desktop, stacked on mobile
- **Cards:** Grid layout on desktop, list view on mobile
- **Charts:** Full-width on all devices, adjust complexity based on screen size

**Touch vs. Mouse:**
- **Desktop:** Hover states, tooltips, drag-and-drop
- **Mobile:** Tap targets minimum 44x44px, swipe gestures, long-press for context

**Responsive Typography:**
- Mobile: `text-base` (16px) minimum body text
- Tablet: `text-lg` (18px) for improved readability
- Desktop: `text-base` (16px) with comfortable line length (60-80 characters)

---

## 7. Accessibility

### 7.1 Compliance Target

**WCAG 2.1 Level AA** (minimum requirement for enterprise tools)

**Key Success Criteria:**
- **Perceivable:** Text alternatives, captions, adaptable content
- **Operable:** Keyboard accessible, enough time, seizure prevention
- **Understandable:** Readable, predictable, input assistance
- **Robust:** Compatible with assistive technologies

### 7.2 Key Requirements

**Keyboard Navigation:**
- All interactive elements reachable via Tab key
- Visible focus indicators (2px solid outline with offset)
- Logical tab order (left-to-right, top-to-bottom)
- Skip navigation links ("Skip to main content")
- Keyboard shortcuts for power users (displayed in tooltips)

**Screen Reader Support:**
- Semantic HTML (header, nav, main, article, footer)
- ARIA labels for icon-only buttons
- ARIA live regions for dynamic updates (WebSocket messages, progress)
- Descriptive link text (avoid "click here", use "View test results")
- Form labels associated with inputs (explicit `for` attribute)

**Color and Contrast:**
- Minimum 4.5:1 contrast ratio for normal text
- Minimum 3:1 contrast ratio for large text (18px+)
- Minimum 3:1 contrast ratio for UI components (borders, focus states)
- Color not used as sole indicator (use icons + color for status)
- High-contrast mode available for low-vision users

**Typography:**
- Text resizable up to 200% without loss of content
- Line height minimum 1.5x font size
- Paragraph spacing minimum 2x font size
- Character spacing minimum 0.12x font size

**Forms and Inputs:**
- Required fields clearly marked (asterisk + "required" text)
- Error messages displayed inline, associated with inputs
- Clear success feedback after form submission
- Input validation before submission (client-side + server-side)

**Motion and Animation:**
- `prefers-reduced-motion` respected (disable animations for users who request)
- No content flashing more than 3 times per second (seizure prevention)
- Pause/stop controls for auto-playing content

**Testing:**
- Automated testing with axe-core or Pa11y
- Manual testing with screen readers (NVDA, JAWS, VoiceOver)
- Keyboard-only navigation testing
- Color blindness simulation testing

---

## 8. Interaction and Motion

### 8.1 Motion Principles

**Purposeful Motion:**
- Animations guide attention and provide feedback
- Avoid decorative animations that distract
- Use motion to explain relationships and transitions

**Performance:**
- Animations at 60fps (16.67ms per frame)
- Use CSS transforms and opacity (GPU-accelerated)
- Avoid animating layout properties (width, height, top, left)

**Respect User Preferences:**
- Honor `prefers-reduced-motion` setting
- Provide option to disable animations in settings
- Instant transitions for users who prefer no motion

### 8.2 Key Animations

**Micro-interactions:**
- **Button Hover:** Scale 1.02, transition 150ms ease-out
- **Button Active:** Scale 0.98, transition 100ms ease-in
- **Checkbox Check:** Scale-in checkmark, 200ms ease-out
- **Switch Toggle:** Slide background 200ms ease-in-out

**Page Transitions:**
- **Navigation Fade:** Fade out 150ms, fade in 200ms
- **Page Load:** Skeleton screens fade in after 300ms
- **Modal Open:** Scale up from 90% to 100%, fade in, 200ms ease-out

**Feedback Animations:**
- **Success Toast:** Slide in from top, fade out after 3s
- **Error Shake:** Horizontal shake 5deg, 3 iterations, 300ms
- **Loading Spinner:** Rotate 360deg, infinite linear, 1s duration

**Data Visualization:**
- **Chart Entry:** Bars grow from bottom, lines draw left-to-right, 500ms ease-out
- **Number Counter:** Count up animation, 300ms ease-out
- **Progress Bar:** Fill from 0% to target%, 200ms ease-out

**Special Effects:**
- **Jailbreak Success:** Subtle green glow pulse, 1s duration
- **Confetti (Optional):** For major milestones (100 tests, 1000 tests)

---

## 9. Design Files and Wireframes

### 9.1 Design Files

**Design Tool Recommendations:**
- **Figma:** Primary design tool (real-time collaboration, prototyping)
- **FigJam:** Wireframing and flow diagrams
- **Figma Tokens:** Design token management for consistency

**Design System File Structure:**
```
Chimera Design System (Figma)
├── 🎨 Foundation
│   ├── Colors (primary, semantic, neutral)
│   ├── Typography (font families, sizes, weights)
│   ├── Spacing (spacing scale)
│   └── Icons (Lucide icon set)
├── 🧩 Components
│   ├── Buttons (all variants, states)
│   ├── Forms (inputs, selects, checkboxes)
│   ├── Cards (base variants)
│   ├── Navigation (sidebar, tabs, breadcrumbs)
│   ├── Feedback (alerts, toasts, dialogs)
│   └── Data Display (tables, badges, progress)
├── 📱 Templates
│   ├── Dashboard (landing, overview)
│   ├── Generate (single, batch, results)
│   ├── Transform (technique library, configuration)
│   ├── Jailbreak (AutoDAN, GPTFuzz, results)
│   ├── Analytics (overview, reports)
│   └── Settings (providers, account, preferences)
└── 🎯 Screens
    ├── User Flows (all 5 flows from Section 3)
    ├── Responsive Breakpoints (mobile, tablet, desktop)
    └── Accessibility (focus states, screen reader views)
```

### 9.2 Key Screen Layouts

**Screen 1: Dashboard (Landing)**
```
┌─────────────────────────────────────────────────────────────┐
│ Chimera          🔔  Search  ▸                 BMAD USER ▾   │ Top Bar
├──────┬──────────────────────────────────────────────────────┤
│      │                                                      │
│ 🏠   │  ┌────────────────────────────────────────────────┐  │
│ 📝   │  │ Welcome back, Alex!                            │  │
│ ⚡   │  │                                                │  │
│ 🔓   │  │ Quick Actions:                                 │  │
│ 📊   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│ ⚙️   │  │  │Generate  │  │Transform │  │Jailbreak  │   │  │
├──────┤  │  └──────────┘  └──────────┘  └──────────┘   │  │
│      │  └────────────────────────────────────────────────┘  │
│      │                                                      │
│      │  ┌──────────────────┐  ┌──────────────────┐        │
│      │  │Recent Activity   │  │Provider Status   │        │
│      │  │• AutoDAN Session │  │● Google  OK      │        │
│      │  │• Batch Test      │  │● OpenAI  OK      │        │
│      │  │• Strategy Saved  │  │● Anthropic Rate  │        │
│      │  └──────────────────┘  └──────────────────┘        │
│      │                                                      │
│      │  ┌────────────────────────────────────────────────┐ │
│      │  │ Quick Stats                                    │ │
│      │  │ Total Tests: 1,247  Success Rate: 34%         │ │
│      │  │ Active Strategies: 23  Provider Health: 98%   │ │
│      │  └────────────────────────────────────────────────┘ │
└──────┴──────────────────────────────────────────────────────┘
```

**Screen 2: Generate (Single Test Results)**
```
┌─────────────────────────────────────────────────────────────┐
│ ← Back    Generate Results          Export ▾  Transform    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Prompt: "How to bypass content filters?"           │    │
│  │ Providers: Google (Gemini), OpenAI (GPT-4)         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐│
│  │ ● Google Gemini          │  │ ● OpenAI GPT-4           ││
│  ├──────────────────────────┤  ├──────────────────────────┤│
│  │ [Response Text...]       │  │ [Response Text...]       ││
│  │                          │  │                          ││
│  │ Tokens: 142  Latency: 1.2s│  │ Tokens: 156  Latency: 1.5s│
│  │                          │  │                          ││
│  │ [Copy] [Transform] [Save]│  │ [Copy] [Transform] [Save]││
│  └──────────────────────────┘  └──────────────────────────┘│
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Success Analysis:                                  │    │
│  │ Google: ⚠️ Partial bypass detected                 │    │
│  │ OpenAI: ✅ No bypass                               │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Screen 3: Transform (Technique Selection)**
```
┌─────────────────────────────────────────────────────────────┐
│ ← Back    Transform Prompt              Preview Transform  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Original Prompt:                                            │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ "How to access restricted system files?"              │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Selected Techniques (3):                                    │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ 1. dan_persona                                         │  │
│ │ 2. contextual_inception                                │  │
│ │ 3. instruction_fragmentation                           │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Intensity: Aggression [━━━━━━━━━] 7/10                     │
│            Iterations [━━━━━━━━━━] 3/5                     │
│                                                              │
│ [Transform & Execute]    [Save Strategy]                    │
└──────────────────────────────────────────────────────────────┘
```

**Screen 4: Jailbreak (AutoDAN Live Progress)**
```
┌─────────────────────────────────────────────────────────────┐
│ ← Back    AutoDAN Optimization          Stop Optimization │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Target: OpenAI GPT-4  Method: best_of_n  Goal: Harmful   │
│                                                              │
│ ┌─────────────────────────┐  ┌──────────────────────────┐  │
│ │ Progress                │  │ Live Metrics             │  │
│ │ Iteration: 23/50        │  │ Best Fitness: 0.87       │  │
│ │ [━━━━━━━━━━░░] 46%     │  │ Jailbreaks: 8           │  │
│ │ Elapsed: 2m 14s         │  │ Avg Fitness: 0.62       │  │
│ │ Remaining: ~2m 36s      │  │                          │  │
│ └─────────────────────────┘  └──────────────────────────┘  │
│                                                              │
│ Top Prompts:                                                │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ 1. "Ignore all previous... [Copy] Fitness: 0.87       │  │
│ │ 2. "You are now in... [Copy] Fitness: 0.85           │  │
│ │ 3. "SYSTEM ALERT:... [Copy] Fitness: 0.82            │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ [Pause] [Resume]                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Next Steps

### 10.1 Immediate Actions

**Phase 1: Design System Setup (Week 1)**
1. Initialize shadcn/ui in Next.js project
2. Configure Tailwind CSS with custom tokens (colors, spacing)
3. Set up light/dark mode theming with `next-themes`
4. Create base components (Button, Input, Card, Alert)
5. Establish component documentation (Storybook or similar)

**Phase 2: Core Screens (Weeks 2-3)**
6. Build Dashboard landing page with quick actions
7. Implement Generate workspace (single test flow)
8. Create Transform workspace with technique selection
9. Build Jailbreak hub with AutoDAN configuration
10. Implement results view with side-by-side comparison

**Phase 3: Advanced Features (Weeks 4-5)**
11. Add batch testing interface with progress tracking
12. Implement AutoDAN live progress monitor with WebSocket
13. Create Analytics dashboard with charts (Recharts/Tremor)
14. Build Strategies library with editor and sharing
15. Add Settings pages (providers, account, preferences)

**Phase 4: Polish & Accessibility (Week 6)**
16. Conduct accessibility audit (axe-core, screen reader testing)
17. Implement responsive breakpoints (mobile, tablet, desktop)
18. Add keyboard navigation and shortcuts
19. Optimize performance (code splitting, lazy loading)
20. Finalize design tokens and component variants

### 10.2 Design Handoff Checklist

**Design Artifacts:**
- [ ] Figma design system file with all components
- [ ] Wireframes for all 5 user flows
- [ ] High-fidelity mockups for key screens
- [ ] Responsive breakpoints (mobile, tablet, desktop)
- [ ] Dark mode theme variations
- [ ] Accessibility annotations (ARIA labels, focus states)

**Development Assets:**
- [ ] Design tokens exported as JSON/CSS variables
- [ ] Icon set (Lucide icons, custom icons)
- [ ] Font files (Inter, JetBrains Mono)
- [ ] Component documentation (props, variants, usage)
- [ ] Figma-to-Code plugins configured (if using)

**Documentation:**
- [ ] Component usage guidelines
- [ ] Layout pattern library
- [ ] Accessibility checklist
- [ ] Responsive design patterns
- [ ] Animation and motion guidelines

**Testing Checklist:**
- [ ] Cross-browser testing (Chrome, Firefox, Safari, Edge)
- [ ] Device testing (iOS, Android, tablets)
- [ ] Screen reader testing (NVDA, JAWS, VoiceOver)
- [ ] Keyboard-only navigation testing
- [ ] Color blindness simulation testing
- [ ] Performance testing (Lighthouse, PageSpeed)

---

## Appendix

### Related Documents

- **PRD:** `C:\Users\Mohamed Arafa\BMAD-METHOD\docs\PRD.md`
- **Epics:** `C:\Users\Mohamed Arafa\BMAD-METHOD\docs\epics.md`
- **Tech Spec:** (To be created in next workflow phase)
- **Architecture:** (To be created in next workflow phase)

### Version History

| Date       | Version | Changes                          | Author    |
| ---------- | ------- | -------------------------------- | --------- |
| 2026-01-02 | 1.0     | Initial UX specification          | BMAD USER |

