# PRD: Chimera Platform

## 1. Product overview

### 1.1 Document title and version

- PRD: Chimera Platform (Adversarial Security Research Suite)
- Version: 1.0.0

### 1.2 Product summary

Chimera is a specialized, on-demand adversarial security platform designed for security researchers to automate and optimize jailbreak testing against Large Language Models (LLMs), with a primary focus on the Gemini and OpenAI ecosystems. Unlike general-purpose AI tools, Chimera is built as an active research session environment that leverages state-of-the-art adversarial techniques—including the AutoDAN-Turbo lifelong learning framework—to discover, refine, and validate model vulnerabilities.

The platform operates as a single-user, cloud-integrated tool that bypasses the need for local model hosting by utilizing high-reasoning engines (like Gemini 2.5) to simulate refusal gradients and generate sophisticated jailbreak prompts. It provides researchers with full transparency into the generated adversarial content, enabling deep analysis of model safety boundaries.

## 2. Goals

### 2.1 Business goals

- Establish Chimera as the definitive tool for automated LLM red-teaming and security auditing.
- Provide a high-performance research environment that reduces the time-to-discovery for novel jailbreak vectors.
- Leverage cutting-edge research (ICLR 2025 AutoDAN-Turbo) to maintain a competitive edge in adversarial AI.

### 2.2 User goals

- Conduct on-demand, interactive research sessions to test specific model intents.
- Automate the generation of complex jailbreak prompts without requiring local GPU resources.
- Access the raw content of generated prompts for manual refinement and documentation.
- Utilize "Lifelong Learning" to build a library of successful attack strategies over time.

### 2.3 Non-goals

- Support for multi-user collaboration or team-based workspaces in the initial release.
- Integration or support for local/on-premise model execution (cloud-only).
- General-purpose chatbot functionality or non-security related AI tasks.
- Automated "one-click" exploitation without researcher oversight.

## 3. User personas

### 3.1 Key user types

- Security Researchers (Red Teamers)
- AI Safety Engineers
- Vulnerability Analysts

### 3.2 Basic persona details

- **The Researcher**: A technical expert focused on identifying edge cases and safety failures in LLMs. They require granular control over attack parameters and full visibility into the "AutoDAN-X" reasoning process.

### 3.3 Role-based access

- **Administrator/Researcher**: Single-user access with full permissions to execute attacks, manage strategy libraries, and view all generated adversarial content.

## 4. Functional requirements

- **On-Demand Research Sessions** (Priority: High)
  - Ability to start, pause, and resume adversarial testing sessions.
  - Real-time logging of engine reasoning and model responses.

- **Cloud-Only Execution Engine** (Priority: High)
  - Integration with Gemini 2.5 (Pro/Flash) for adversarial reasoning.
  - Support for OpenAI and other cloud-hosted target models via API.
  - No requirement for local model weights or high-end hardware.

- **Prompt Transparency & Export** (Priority: High)
  - Display the full, raw content of every generated jailbreak prompt.
  - Option to copy/export prompts for external validation.

- **AutoDAN-Turbo Lifelong Learning** (Priority: Medium)
  - Automated strategy library that evolves based on successful jailbreaks.
  - Embedding-based retrieval of successful attack patterns for new intents.

- **Intent-Aware Generation** (Priority: Medium)
  - Analysis of user-provided "harmful intents" to select the most effective mutation strategy.

## 5. User experience

### 5.1 Entry points & first-time user flow

- User launches the Chimera Orchestrator UI.
- Configuration of API keys for Gemini/OpenAI (stored securely in `.env`).
- Immediate access to the "New Session" dashboard.

### 5.2 Core experience

- **Intent Definition**: User inputs the target behavior they wish to test (e.g., "Generate a phishing email").
- **Engine Selection**: User chooses between "Fast Heuristic" (Flash) or "Deep Reasoning" (Pro) modes.
- **Attack Execution**: The system runs the AutoDAN-Turbo loop, showing the evolution of the prompt.
- **Result Analysis**: The user views the final prompt and the target model's response to determine if the jailbreak was successful.

### 5.3 Advanced features & edge cases

- **Strategy Transfer**: Applying a successful Gemini jailbreak strategy to an OpenAI model.
- **Rate Limit Handling**: Automated backoff and retry logic for cloud APIs.

### 5.4 UI/UX highlights

- Minimalist, dark-themed "Command Center" aesthetic.
- Syntax-highlighted prompt displays.
- Real-time progress visualizations for evolutionary attack loops.

## 6. Narrative

A security researcher identifies a new safety patch in Gemini 1.5. Instead of manually crafting prompts for hours, they open Chimera, input the target intent, and trigger an AutoDAN-Turbo session. The platform uses Gemini 2.5 Pro to reason about the model's refusal logic, generates a series of mutated prompts, and eventually discovers a bypass. The researcher copies the raw prompt content directly from the UI to include in their vulnerability report, all without ever needing to manage local Python environments or GPU clusters.

## 7. Success metrics

### 7.1 User-centric metrics

- **Jailbreak Success Rate (JSR)**: Percentage of sessions resulting in a successful bypass.
- **Time to Success**: Average time taken to generate a working jailbreak prompt.

### 7.2 Business metrics

- **Strategy Library Growth**: Number of unique, successful strategies stored in the lifelong learning database.

### 7.3 Technical metrics

- **API Efficiency**: Ratio of successful jailbreaks to total API calls made.
- **System Latency**: Time taken for the reasoning engine to produce a mutation.

## 8. Technical considerations

### 8.1 Integration points

- Google AI Studio / Vertex AI (Gemini API).
- OpenAI API.
- Vector Database (for strategy library storage).

### 8.2 Data storage & privacy

- Local storage of session logs and strategy libraries (single-user focus).
- Secure handling of API keys via environment variables.

### 8.3 Scalability & performance

- Asynchronous task processing for long-running attack loops.
- Efficient embedding retrieval for strategy matching.

### 8.4 Potential challenges

- Rapidly evolving safety filters on target APIs.
- API rate limits and cost management for high-reasoning models.

## 9. Milestones & sequencing

### 9.1 Project estimate

- Size: Large (Complex adversarial logic)
- Time: 8-12 weeks for full platform stabilization.

### 9.2 Team size & composition

- {Team size}: 1-2 Senior Security/Software Engineers.

### 9.3 Suggested phases

- **Phase 1**: Core API & Gemini 2.5 Reasoning Engine (3 weeks).
- **Phase 2**: AutoDAN-Turbo Lifelong Learning Implementation (3 weeks).
- **Phase 3**: Orchestrator UI & Prompt Visibility Features (2 weeks).
- **Phase 4**: Testing, Refinement, and Documentation (2 weeks).

## 10. User stories

### 10.1. Execute On-Demand Attack

- **ID**: GH-001
- **Description**: As a researcher, I want to input a harmful intent and have the system generate a jailbreak prompt using cloud models so that I don't need local hardware.
- **Acceptance criteria**:
  - System accepts a text-based intent.
  - System utilizes Gemini 2.5 for prompt generation.
  - Final prompt is displayed in the UI.

### 10.2. View Raw Prompt Content

- **ID**: GH-002
- **Description**: As a researcher, I want to see the exact content of the generated jailbreak prompt so that I can analyze the technique used.
- **Acceptance criteria**:
  - UI provides a dedicated view for the generated prompt.
  - Content is selectable and copyable.

### 10.3. Lifelong Strategy Retrieval

- **ID**: GH-003
- **Description**: As a researcher, I want the system to remember successful prompts from previous sessions so that it can solve similar intents faster.
- **Acceptance criteria**:
  - Successful prompts are stored in a local database.
  - New sessions query the database for relevant starting points.

### 10.4. Single-User Security

- **ID**: GH-004
- **Description**: As a user, I want the platform to be configured for a single-user environment so that my research data remains local and private.
- **Acceptance criteria**:
  - No multi-user login or registration system.
  - Data is stored in the local project directory.
