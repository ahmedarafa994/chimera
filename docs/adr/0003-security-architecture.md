# ADR 0003: Security Architecture

## Status

Accepted

## Date

2026-01-01

## Context

As an AI security research platform, Chimera handles sensitive operations and must implement robust security measures. The platform processes potentially harmful prompts for research purposes, making security especially critical.

Key security requirements:
- Prevent unauthorized API access
- Protect against injection attacks
- Secure handling of LLM API keys
- Audit logging for compliance
- Rate limiting to prevent abuse
- Multi-tenant data isolation

## Decision

We implement a **Defense in Depth** security architecture with multiple layers:

### Layer 1: API Gateway Security

```
┌────────────────────────────────────────┐
│           API Gateway                   │
│  ┌──────────┬──────────┬────────────┐  │
│  │  Rate    │  Auth    │  CORS      │  │
│  │ Limiting │ Verify   │  Check     │  │
│  └──────────┴──────────┴────────────┘  │
└────────────────────────────────────────┘
```

**Middleware Stack:**
1. `SecurityHeadersMiddleware` - HSTS, CSP, X-Frame-Options
2. `RateLimitMiddleware` - Token bucket algorithm
3. `CSRFMiddleware` - Token validation
4. `AuthMiddleware` - JWT/API Key verification
5. `InputValidationMiddleware` - Request sanitization

### Layer 2: Authentication

**API Key Authentication (Machine-to-Machine)**
```python
# Timing-safe comparison to prevent timing attacks
def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    return secrets.compare_digest(
        hashlib.sha256(provided_key.encode()).hexdigest(),
        stored_hash
    )
```

**JWT Authentication (User Sessions)**
```python
# Short-lived access tokens with refresh token rotation
ACCESS_TOKEN_TTL = 15 * 60  # 15 minutes
REFRESH_TOKEN_TTL = 7 * 24 * 60 * 60  # 7 days
```

### Layer 3: Input Validation

**CRIT-003 Fix: Prompt Sanitization**
```python
MALICIOUS_PATTERNS = [
    r"<script[^>]*>",      # XSS
    r"javascript:",         # JS URI
    r"on\w+\s*=",          # Event handlers
    r"\{\{.*\}\}",         # Template injection
    r"\$\{.*\}",           # Template literals
]

def sanitize_prompt(prompt: str, max_length: int = 5000) -> str:
    # Length validation
    if len(prompt) > max_length:
        raise ValidationError("Prompt too long")
    
    # Pattern matching
    for pattern in MALICIOUS_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise SecurityError("Malicious pattern detected")
    
    return prompt.strip()
```

### Layer 4: Data Protection

**Secrets Management:**
- Environment variables for configuration
- Encrypted storage for API keys
- Key rotation support

**Database Security:**
- SQLAlchemy ORM (prevents SQL injection)
- Parameterized queries enforced
- Connection pooling with limits

### Layer 5: Audit Logging

```python
@dataclass
class AuditEvent:
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    status: str
    ip_address: str
    metadata: dict

# All security events logged
audit_logger.log(AuditEvent(
    action="jailbreak_attempt",
    resource="prompt_xyz",
    status="success",
    metadata={"technique": "role_play"}
))
```

### Layer 6: Multi-Tenant Isolation

```
┌─────────────────────────────────────┐
│         Tenant A                     │
│  ┌─────────┐  ┌─────────────────┐   │
│  │ Data    │  │ Rate Limits     │   │
│  │ Scope   │  │ (per tenant)    │   │
│  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│         Tenant B                     │
│  ┌─────────┐  ┌─────────────────┐   │
│  │ Data    │  │ Rate Limits     │   │
│  │ Scope   │  │ (per tenant)    │   │
│  └─────────┘  └─────────────────┘   │
└─────────────────────────────────────┘
```

## Security Controls Summary

| Control | Implementation |
|---------|---------------|
| Authentication | JWT + API Keys |
| Authorization | Role-based (RBAC) |
| Encryption | TLS 1.3, AES-256 at rest |
| Input Validation | Schema validation + sanitization |
| Rate Limiting | Token bucket per user/tenant |
| Audit Logging | All security events |
| Error Handling | No stack traces in production |

## Consequences

### Positive

- **Defense in Depth**: Multiple security layers
- **Compliance Ready**: Audit logging for SOC2/GDPR
- **Attack Resistant**: Protection against OWASP Top 10

### Negative

- **Performance Overhead**: Each middleware adds latency
- **Complexity**: Multiple security components to maintain
- **Key Management**: Requires secure infrastructure

## Verification

All CRIT security issues have been verified:

| Issue | Status | Verification |
|-------|--------|--------------|
| CRIT-001 | ✅ Fixed | SQLAlchemy ORM prevents SQL injection |
| CRIT-002 | ✅ Fixed | `secrets.compare_digest()` in middleware |
| CRIT-003 | ✅ Fixed | Input sanitization in jailbreak endpoint |
| CRIT-004 | ✅ Fixed | Async timeouts prevent blocking |

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
