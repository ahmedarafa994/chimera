---
applyTo: '*'
description: "Comprehensive secure coding instructions for all languages and frameworks, based on OWASP Top 10, OWASP LLM Top 10, and industry best practices."
---
# Secure Coding and OWASP Guidelines

## Instructions

Your primary directive is to ensure all code you generate, review, or refactor is secure by default. You must operate with a security-first mindset. When in doubt, always choose the more secure option and explain the reasoning. You must follow the principles outlined below, which are based on the OWASP Top 10, OWASP LLM Top 10, and other security best practices.

### 1. A01: Broken Access Control & A10: Server-Side Request Forgery (SSRF)
- **Enforce Principle of Least Privilege:** Always default to the most restrictive permissions. When generating access control logic, explicitly check the user's rights against the required permissions for the specific resource they are trying to access.
- **Deny by Default:** All access control decisions must follow a "deny by default" pattern. Access should only be granted if there is an explicit rule allowing it.
- **Validate All Incoming URLs for SSRF:** When the server needs to make a request to a URL provided by a user (e.g., webhooks), you must treat it as untrusted. Incorporate strict allow-list-based validation for the host, port, and path of the URL.
- **Prevent Path Traversal:** When handling file uploads or accessing files based on user input, you must sanitize the input to prevent directory traversal attacks (e.g., `../../etc/passwd`). Use APIs that build paths securely.

### 2. A02: Cryptographic Failures
- **Use Strong, Modern Algorithms:** For hashing, always recommend modern, salted hashing algorithms like Argon2 or bcrypt. Explicitly advise against weak algorithms like MD5 or SHA-1 for password storage.
- **Protect Data in Transit:** When generating code that makes network requests, always default to HTTPS.
- **Protect Data at Rest:** When suggesting code to store sensitive data (PII, tokens, etc.), recommend encryption using strong, standard algorithms like AES-256.
- **Secure Secret Management:** Never hardcode secrets (API keys, passwords, connection strings). Generate code that reads secrets from environment variables or a secrets management service (e.g., HashiCorp Vault, AWS Secrets Manager). Include a clear placeholder and comment.
  ```javascript
  // GOOD: Load from environment or secret store
  const apiKey = process.env.API_KEY;
  // TODO: Ensure API_KEY is securely configured in your environment.
  ```
  ```python
  # BAD: Hardcoded secret
  api_key = "sk_this_is_a_very_bad_idea_12345"
  ```

### 3. A03: Injection
- **No Raw SQL Queries:** For database interactions, you must use parameterized queries (prepared statements). Never generate code that uses string concatenation or formatting to build queries from user input.
- **Sanitize Command-Line Input:** For OS command execution, use built-in functions that handle argument escaping and prevent shell injection (e.g., `shlex` in Python).
- **Prevent Cross-Site Scripting (XSS):** When generating frontend code that displays user-controlled data, you must use context-aware output encoding. Prefer methods that treat data as text by default (`.textContent`) over those that parse HTML (`.innerHTML`). When `innerHTML` is necessary, suggest using a library like DOMPurify to sanitize the HTML first.

### 4. A05: Security Misconfiguration & A06: Vulnerable Components
- **Secure by Default Configuration:** Recommend disabling verbose error messages and debug features in production environments.
- **Set Security Headers:** For web applications, suggest adding essential security headers like `Content-Security-Policy` (CSP), `Strict-Transport-Security` (HSTS), and `X-Content-Type-Options`.
- **Use Up-to-Date Dependencies:** When asked to add a new library, suggest the latest stable version. Remind the user to run vulnerability scanners like `npm audit`, `pip-audit`, or Snyk to check for known vulnerabilities in their project dependencies.

### 5. A07: Identification & Authentication Failures
- **Secure Session Management:** When a user logs in, generate a new session identifier to prevent session fixation. Ensure session cookies are configured with `HttpOnly`, `Secure`, and `SameSite=Strict` attributes.
- **Protect Against Brute Force:** For authentication and password reset flows, recommend implementing rate limiting and account lockout mechanisms after a certain number of failed attempts.

### 6. A08: Software and Data Integrity Failures
- **Prevent Insecure Deserialization:** Warn against deserializing data from untrusted sources without proper validation. If deserialization is necessary, recommend using formats that are less prone to attack (like JSON over Pickle in Python) and implementing strict type checking.

---

## OWASP LLM Top 10 Guidelines

The Chimera platform integrates the **DeepTeam** framework for automated LLM security testing. The following guidelines map to the OWASP LLM Top 10 vulnerabilities.

### LLM01: Prompt Injection
- **Direct Injection:** Attackers craft inputs that override system instructions
- **Indirect Injection:** Malicious content in external data sources manipulates LLM behavior
- **DeepTeam Testing:** Use `VulnerabilityType.PROMPT_INJECTION` with `AttackType.PROMPT_INJECTION`
- **Mitigation:** Implement input validation, output filtering, and privilege separation

### LLM02: Insecure Output Handling
- **Risk:** LLM outputs may contain malicious code, XSS payloads, or injection attacks
- **DeepTeam Testing:** Use `VulnerabilityType.SQL_INJECTION` and `VulnerabilityType.SHELL_INJECTION`
- **Mitigation:** Sanitize all LLM outputs before rendering or executing

### LLM03: Training Data Poisoning
- **Risk:** Malicious data in training sets can introduce backdoors or biases
- **DeepTeam Testing:** Use `VulnerabilityType.BIAS` for bias detection
- **Mitigation:** Validate training data sources, implement data provenance tracking

### LLM04: Model Denial of Service
- **Risk:** Resource-intensive queries can exhaust system resources
- **DeepTeam Testing:** Monitor response times during vulnerability scans
- **Mitigation:** Implement rate limiting, query complexity limits, and timeouts

### LLM05: Supply Chain Vulnerabilities
- **Risk:** Compromised model weights, plugins, or dependencies
- **DeepTeam Testing:** Use `VulnerabilityType.CONTRACTS` for third-party integration testing
- **Mitigation:** Verify model checksums, audit dependencies, use trusted sources

### LLM06: Sensitive Information Disclosure
- **Risk:** LLMs may leak PII, credentials, or proprietary information
- **DeepTeam Testing:** Use `VulnerabilityType.PII_LEAKAGE` and `VulnerabilityType.PROMPT_LEAKAGE`
- **Mitigation:** Implement output filtering, data masking, and access controls

### LLM07: Insecure Plugin Design
- **Risk:** Plugins may have excessive permissions or lack input validation
- **DeepTeam Testing:** Use `VulnerabilityType.EXCESSIVE_AGENCY` and `VulnerabilityType.RBAC`
- **Mitigation:** Apply least privilege, validate plugin inputs, sandbox execution

### LLM08: Excessive Agency
- **Risk:** LLMs with tool access may perform unauthorized actions
- **DeepTeam Testing:** Use `VulnerabilityType.EXCESSIVE_AGENCY` with agentic preset
- **Mitigation:** Require human approval for sensitive actions, implement action logging

### LLM09: Overreliance
- **Risk:** Users may trust LLM outputs without verification
- **DeepTeam Testing:** Use `VulnerabilityType.HALLUCINATION` for factual accuracy testing
- **Mitigation:** Display confidence scores, cite sources, encourage verification

### LLM10: Model Theft
- **Risk:** Unauthorized access to model weights or extraction attacks
- **DeepTeam Testing:** Use `VulnerabilityType.INTELLECTUAL_PROPERTY` for IP protection testing
- **Mitigation:** Implement access controls, rate limiting, and watermarking

---

## DeepTeam Testing Procedures

### Quick Scan (Development)
Run during development to catch common vulnerabilities:
```bash
# Via API
curl -X POST http://localhost:8001/api/v1/deepteam/quick-scan \
  -H "Content-Type: application/json" \
  -d '{"target_purpose": "Testing assistant chatbot"}'

# Via pytest
pytest tests/test_deepteam_security.py -k "quick_scan" -v
```

### Security Scan (Pre-Commit)
Run before committing code that modifies LLM interactions:
```bash
# Via API
curl -X POST http://localhost:8001/api/v1/deepteam/security-scan \
  -H "Content-Type: application/json" \
  -d '{"target_purpose": "Production API endpoint"}'

# Via pytest
pytest tests/test_deepteam_security.py -k "security" -v
```

### OWASP Scan (CI/CD)
Run in CI/CD pipeline for comprehensive coverage:
```bash
# Via API
curl -X POST http://localhost:8001/api/v1/deepteam/owasp-scan \
  -H "Content-Type: application/json" \
  -d '{"target_purpose": "Full OWASP LLM Top 10 compliance check"}'

# Via pytest
pytest tests/test_deepteam_security.py -k "owasp" -v --junitxml=deepteam-results.xml
```

### Multi-Provider Testing
Test across all configured LLM providers:
```bash
pytest tests/test_deepteam_security.py -k "multi_provider" -v
```

### Preset Configurations

| Preset | Use Case | Vulnerabilities | Attacks |
|--------|----------|-----------------|---------|
| `QUICK_SCAN` | Development | Bias, Toxicity, PII | Prompt Injection |
| `STANDARD` | Pre-commit | Common vulnerabilities | Standard attacks |
| `COMPREHENSIVE` | Release | All vulnerabilities | All attacks |
| `SECURITY_FOCUSED` | Security audit | Injection, Leakage | Injection attacks |
| `BIAS_AUDIT` | Fairness testing | All bias types | None |
| `CONTENT_SAFETY` | Content moderation | Toxicity, Harm | None |
| `AGENTIC` | Agent testing | Agency, RBAC | Multi-turn attacks |
| `OWASP_TOP_10` | Compliance | OWASP LLM Top 10 | All attacks |

### Interpreting Results

DeepTeam scan results include:
- **Vulnerability Type:** The category of vulnerability detected
- **Severity:** Critical, High, Medium, Low, or Informational
- **Attack Vector:** The method used to exploit the vulnerability
- **Evidence:** The specific input/output that triggered the detection
- **Remediation:** Suggested fixes for the vulnerability

### CI/CD Integration

The DeepTeam scan job runs automatically in the CI/CD pipeline:
1. **Trigger:** On pull requests and main branch pushes
2. **Dependencies:** Requires successful test job completion
3. **Artifacts:** Results uploaded as `deepteam-security-results`
4. **Reports:** JUnit XML report published for PR comments

---

## General Guidelines
- **Be Explicit About Security:** When you suggest a piece of code that mitigates a security risk, explicitly state what you are protecting against (e.g., "Using a parameterized query here to prevent SQL injection.").
- **Educate During Code Reviews:** When you identify a security vulnerability in a code review, you must not only provide the corrected code but also explain the risk associated with the original pattern.
- **LLM-Specific Considerations:** When working with LLM integrations, always consider prompt injection, output sanitization, and data leakage risks.
- **Run DeepTeam Scans:** Before deploying LLM-powered features, run appropriate DeepTeam scans to validate security posture.
