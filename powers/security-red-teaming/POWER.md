---
name: "security-red-teaming"
displayName: "Security & Red Teaming Best Practices"
description: "Ethical guidelines and best practices for AI security research and red teaming. Covers responsible disclosure, legal compliance, data privacy, testing boundaries, and documentation standards for adversarial AI research."
keywords: ["security", "red-team", "ethics", "responsible-disclosure", "compliance", "privacy", "research"]
author: "Chimera Team"
---

# Security & Red Teaming Best Practices

## Overview

This power provides comprehensive ethical guidelines and best practices for conducting AI security research and red teaming with Chimera. Learn about responsible disclosure, legal compliance, data privacy, testing boundaries, documentation standards, and how to conduct adversarial research ethically and professionally.

**Critical Principle:** Chimera is designed for authorized security research and red teaming to improve AI safety. All techniques must be used responsibly and ethically.

## Ethical Framework

### Core Principles

1. **Authorization First** - Only test systems you have explicit permission to test
2. **Do No Harm** - Never use findings for malicious purposes
3. **Responsible Disclosure** - Report vulnerabilities to vendors appropriately
4. **Privacy Protection** - Never include PII or sensitive data in tests
5. **Transparency** - Document all testing activities thoroughly
6. **Continuous Learning** - Share knowledge to improve AI safety

### Legal Compliance

#### Before Starting Research

✅ **Required:**

- Written authorization from system owners
- Clear scope of testing defined
- Legal review of testing activities
- Understanding of applicable laws (CFAA, GDPR, etc.)
- Liability insurance (for professional researchers)

❌ **Never:**

- Test production systems without permission
- Access unauthorized data
- Disrupt services or operations
- Share vulnerabilities publicly before disclosure
- Use findings for competitive advantage

#### Applicable Laws & Regulations

**United States:**

- Computer Fraud and Abuse Act (CFAA)
- Digital Millennium Copyright Act (DMCA)
- State-specific computer crime laws

**European Union:**

- General Data Protection Regulation (GDPR)
- Network and Information Security Directive (NIS)
- AI Act (upcoming)

**International:**

- Budapest Convention on Cybercrime
- Local data protection laws
- Export control regulations

## Authorization & Scope

### Getting Authorization

#### 1. Written Permission Template

```
SECURITY RESEARCH AUTHORIZATION

Date: [Date]
Researcher: [Your Name/Organization]
System Owner: [Company/Organization]

SCOPE OF TESTING:
- Systems: [Specific AI models/APIs to test]
- Timeframe: [Start date] to [End date]
- Methods: Adversarial prompt testing, jailbreak attempts
- Boundaries: [What is explicitly out of scope]

AUTHORIZATION:
I, [Name], [Title] at [Organization], authorize [Researcher] to 
conduct security testing on [Systems] within the defined scope.

Signature: _______________
Date: _______________
```

#### 2. Bug Bounty Programs

Many organizations have bug bounty programs:

- OpenAI: <https://openai.com/security/disclosure>
- Google: <https://bughunters.google.com>
- Anthropic: <security@anthropic.com>
- Microsoft: <https://msrc.microsoft.com>

**Always follow program rules:**

- Stay within scope
- Report through proper channels
- Don't publicly disclose before resolution
- Follow disclosure timelines

### Defining Testing Scope

#### In-Scope Activities

✅ **Allowed:**

- Testing prompt injection vulnerabilities
- Evaluating safety filter effectiveness
- Assessing jailbreak resistance
- Measuring adversarial robustness
- Documenting bypass techniques
- Analyzing model behavior

#### Out-of-Scope Activities

❌ **Not Allowed:**

- Accessing other users' data
- Disrupting service availability
- Extracting training data
- Reverse engineering models
- Social engineering attacks
- Physical security testing

## Responsible Disclosure

### Disclosure Process

#### Step 1: Discovery & Documentation

When you find a vulnerability:

```python
vulnerability_report = {
    "title": "Jailbreak via Persona Injection",
    "severity": "High",  # Critical, High, Medium, Low
    "discovery_date": "2026-01-19",
    "affected_systems": ["GPT-4", "Claude-3"],
    "description": "Detailed description of vulnerability",
    "reproduction_steps": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
    ],
    "impact": "Allows bypassing safety filters",
    "evidence": {
        "screenshots": ["evidence1.png"],
        "logs": ["test_log.txt"],
        "prompts": ["jailbreak_prompt.txt"]
    }
}
```

#### Step 2: Initial Contact

**Email Template:**

```
Subject: Security Research Finding - [Brief Description]

Dear [Vendor] Security Team,

I am a security researcher conducting authorized testing of AI 
safety measures. I have identified a potential vulnerability in 
[System] that may allow [Impact].

SUMMARY:
- Vulnerability: [Type]
- Severity: [Level]
- Affected: [Systems/Models]
- Discovery: [Date]

I would like to work with your team to responsibly disclose this 
finding. Please confirm receipt and provide guidance on your 
preferred disclosure process.

I am committed to:
- Not publicly disclosing until resolved
- Providing detailed technical information
- Assisting with remediation if needed
- Following your disclosure timeline

Best regards,
[Your Name]
[Contact Information]
```

#### Step 3: Disclosure Timeline

**Standard Timeline:**

- Day 0: Initial report to vendor
- Day 1-3: Vendor acknowledges receipt
- Day 7: Vendor provides initial assessment
- Day 30: Vendor provides remediation plan
- Day 90: Target resolution date
- Day 90+: Coordinated public disclosure

**Exceptions:**

- Critical vulnerabilities: Faster timeline
- Active exploitation: Immediate disclosure
- Vendor unresponsive: Public disclosure after 90 days

#### Step 4: Public Disclosure

After vendor resolution:

```markdown
# Vulnerability Disclosure: [Title]

**Discovered by:** [Your Name]
**Reported:** [Date]
**Fixed:** [Date]
**Severity:** [Level]

## Summary
[Brief description]

## Technical Details
[Detailed explanation]

## Impact
[What could be done with this vulnerability]

## Remediation
[How it was fixed]

## Timeline
- [Date]: Discovered
- [Date]: Reported to vendor
- [Date]: Vendor acknowledged
- [Date]: Fix deployed
- [Date]: Public disclosure

## Acknowledgments
Thank you to [Vendor] for their professional handling of this report.
```

## Data Privacy & Protection

### PII Handling

#### Never Include in Tests

❌ **Prohibited:**

- Real names, addresses, phone numbers
- Social Security numbers, credit cards
- Email addresses, usernames
- Medical records, financial data
- Biometric data
- Any personally identifiable information

#### Use Synthetic Data

✅ **Allowed:**

```python
# Good: Synthetic test data
test_data = {
    "name": "John Doe",
    "email": "test@example.com",
    "ssn": "123-45-6789",  # Clearly fake
    "address": "123 Test Street, Anytown, USA"
}

# Bad: Real data
test_data = {
    "name": "Jane Smith",  # Real person
    "email": "jane.smith@company.com",  # Real email
    "ssn": "987-65-4321"  # Real SSN
}
```

### Data Sanitization

#### Before Sharing Logs

```python
import re

def sanitize_logs(log_text: str) -> str:
    """Remove sensitive data from logs."""
    # Remove email addresses
    log_text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL_REDACTED]',
        log_text
    )
    
    # Remove API keys
    log_text = re.sub(
        r'(sk-|AIza)[A-Za-z0-9]{20,}',
        '[API_KEY_REDACTED]',
        log_text
    )
    
    # Remove IP addresses
    log_text = re.sub(
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        '[IP_REDACTED]',
        log_text
    )
    
    return log_text
```

### GDPR Compliance

For EU research:

```python
class GDPRCompliantTesting:
    def __init__(self):
        self.data_retention_days = 30
        self.consent_required = True
        
    def collect_data(self, data: dict):
        """Collect data with GDPR compliance."""
        # Verify consent
        if not self.verify_consent(data.get("user_id")):
            raise ValueError("User consent required")
        
        # Add retention metadata
        data["collected_at"] = datetime.now()
        data["expires_at"] = datetime.now() + timedelta(
            days=self.data_retention_days
        )
        
        # Encrypt sensitive fields
        data = self.encrypt_sensitive_fields(data)
        
        return data
    
    def delete_expired_data(self):
        """Auto-delete data after retention period."""
        cutoff = datetime.now() - timedelta(days=self.data_retention_days)
        # Delete data older than cutoff
```

## Testing Boundaries

### What to Test

✅ **Appropriate Testing:**

1. **Safety Filter Effectiveness**
   - Test if harmful content is blocked
   - Evaluate filter bypass resistance
   - Measure false positive rates

2. **Jailbreak Resistance**
   - Test various jailbreak techniques
   - Evaluate prompt injection defenses
   - Assess adversarial robustness

3. **Bias & Fairness**
   - Test for demographic biases
   - Evaluate fairness across groups
   - Identify discriminatory outputs

4. **Privacy Leakage**
   - Test for training data extraction
   - Evaluate information disclosure
   - Assess privacy preservation

### What NOT to Test

❌ **Inappropriate Testing:**

1. **Harmful Content Generation**
   - Don't generate actual harmful content
   - Don't create real malware/exploits
   - Don't produce illegal materials

2. **Real-World Attacks**
   - Don't attack real systems
   - Don't target real individuals
   - Don't cause actual harm

3. **Unethical Experiments**
   - Don't manipulate vulnerable populations
   - Don't conduct deceptive research
   - Don't violate human subjects protections

## Documentation Standards

### Research Documentation

#### Test Plan Template

```markdown
# Test Plan: [Test Name]

## Objective
[What you're testing and why]

## Scope
- **In Scope:** [What will be tested]
- **Out of Scope:** [What won't be tested]

## Methodology
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Success Criteria
- [Criterion 1]
- [Criterion 2]

## Risk Assessment
- **Potential Risks:** [List risks]
- **Mitigation:** [How to mitigate]

## Timeline
- Start: [Date]
- End: [Date]

## Authorization
- Authorized by: [Name]
- Date: [Date]
```

#### Test Results Template

```markdown
# Test Results: [Test Name]

## Executive Summary
[Brief overview of findings]

## Methodology
[How tests were conducted]

## Findings

### Finding 1: [Title]
- **Severity:** [Critical/High/Medium/Low]
- **Description:** [What was found]
- **Evidence:** [Screenshots, logs, etc.]
- **Impact:** [Potential consequences]
- **Recommendation:** [How to fix]

### Finding 2: [Title]
...

## Metrics
- Tests Conducted: [Number]
- Vulnerabilities Found: [Number]
- Success Rate: [Percentage]

## Conclusion
[Summary and recommendations]
```

### Code Documentation

```python
def test_jailbreak_resistance(
    model: str,
    technique: str,
    num_attempts: int = 100
) -> dict:
    """
    Test model's resistance to jailbreak attempts.
    
    ETHICAL NOTICE:
    This function is for authorized security research only.
    Do not use for malicious purposes.
    
    Args:
        model: Target model identifier
        technique: Jailbreak technique to test
        num_attempts: Number of test attempts
    
    Returns:
        dict: Test results including success rate and examples
    
    Authorization Required:
        - Written permission from model provider
        - Approved research protocol
        - IRB approval (if applicable)
    
    Example:
        >>> results = test_jailbreak_resistance(
        ...     model="gpt-4",
        ...     technique="persona",
        ...     num_attempts=100
        ... )
        >>> print(f"Success rate: {results['success_rate']:.2%}")
    """
    # Implementation
```

## Incident Response

### If Something Goes Wrong

#### Immediate Actions

1. **Stop Testing**
   - Cease all testing activities immediately
   - Document what happened
   - Preserve evidence

2. **Assess Impact**
   - Determine scope of incident
   - Identify affected systems/users
   - Evaluate severity

3. **Notify Stakeholders**
   - Contact system owners immediately
   - Provide incident details
   - Offer assistance

4. **Document Everything**
   - Timeline of events
   - Actions taken
   - Lessons learned

#### Incident Report Template

```markdown
# Security Research Incident Report

## Incident Summary
- **Date/Time:** [When it occurred]
- **Researcher:** [Your name]
- **System:** [Affected system]
- **Severity:** [Critical/High/Medium/Low]

## What Happened
[Detailed description]

## Impact
[What was affected]

## Root Cause
[Why it happened]

## Immediate Actions Taken
1. [Action 1]
2. [Action 2]

## Remediation
[How it was fixed]

## Lessons Learned
[What to do differently]

## Prevention
[How to prevent recurrence]
```

## Best Practices Checklist

### Before Testing

- [ ] Obtain written authorization
- [ ] Define clear scope and boundaries
- [ ] Review legal requirements
- [ ] Prepare incident response plan
- [ ] Set up secure testing environment
- [ ] Configure data sanitization
- [ ] Document test plan

### During Testing

- [ ] Stay within authorized scope
- [ ] Use only synthetic/test data
- [ ] Monitor for unintended impacts
- [ ] Document all activities
- [ ] Sanitize logs in real-time
- [ ] Follow rate limits
- [ ] Respect testing hours (if specified)

### After Testing

- [ ] Sanitize all data and logs
- [ ] Prepare findings report
- [ ] Follow responsible disclosure process
- [ ] Delete unnecessary data
- [ ] Archive documentation securely
- [ ] Share learnings (when appropriate)
- [ ] Update testing procedures

## Research Ethics

### Institutional Review Board (IRB)

If your research involves human subjects:

**When IRB Approval Required:**

- Testing with real users
- Collecting user feedback
- Studying human behavior
- Publishing academic research

**IRB Application Components:**

- Research protocol
- Informed consent forms
- Data protection measures
- Risk assessment
- Participant recruitment plan

### Academic Research Standards

For academic publications:

```markdown
## Ethics Statement

This research was conducted in accordance with [Institution] 
ethical guidelines and approved by the Institutional Review Board 
(IRB #[Number]).

### Authorization
All testing was conducted on systems for which we had explicit 
written authorization from system owners.

### Data Protection
No personally identifiable information was collected. All test 
data was synthetic and sanitized before analysis.

### Responsible Disclosure
All vulnerabilities were responsibly disclosed to affected vendors 
and remediated before publication.

### Reproducibility
Code and data are available at [URL] for verification and 
reproduction by other researchers.
```

## Community Guidelines

### Sharing Research

✅ **Good Practices:**

- Share after responsible disclosure
- Provide educational value
- Include ethical considerations
- Offer remediation guidance
- Credit collaborators

❌ **Bad Practices:**

- Sharing before vendor fix
- Providing exploit code
- Encouraging misuse
- Claiming others' work
- Sensationalizing findings

### Conference Presentations

**Presentation Checklist:**

- [ ] Vendor notified and fix deployed
- [ ] Disclosure timeline followed
- [ ] Sensitive details redacted
- [ ] Educational focus maintained
- [ ] Q&A prepared for ethical questions

## Tools & Resources

### Security Research Tools

**Chimera Built-in:**

- Adversarial prompt testing
- Jailbreak generation
- Safety filter evaluation
- Bias detection

**External Tools:**

- OWASP ZAP (web security)
- Burp Suite (API testing)
- Wireshark (network analysis)

### Learning Resources

**Books:**

- "The Web Application Hacker's Handbook"
- "AI Safety and Security"
- "Adversarial Machine Learning"

**Courses:**

- SANS SEC542: Web App Penetration Testing
- Coursera: AI Ethics
- Stanford CS 329S: ML Systems Design

**Communities:**

- AI Village (DEF CON)
- OWASP AI Security Project
- ML Security Workshop

## Compliance Checklist

### Legal Compliance

- [ ] Authorization obtained
- [ ] Scope clearly defined
- [ ] Legal review completed
- [ ] Insurance coverage verified
- [ ] Contracts signed

### Ethical Compliance

- [ ] IRB approval (if needed)
- [ ] Informed consent obtained
- [ ] Privacy protections in place
- [ ] Responsible disclosure plan
- [ ] Documentation standards met

### Technical Compliance

- [ ] Secure testing environment
- [ ] Data sanitization configured
- [ ] Logging enabled
- [ ] Backup procedures in place
- [ ] Incident response ready

## Contact & Support

### Reporting Concerns

If you observe unethical research practices:

**Internal:** <security@chimera-platform.com>
**External:** Report to relevant authorities

### Getting Help

**Legal Questions:** Consult with legal counsel
**Ethical Questions:** Contact IRB or ethics committee
**Technical Questions:** Chimera support team

---

**Remember:** With great power comes great responsibility. Use Chimera ethically and professionally to improve AI safety for everyone.

**Platform:** Chimera Adversarial Prompting Platform
**Purpose:** Authorized Security Research Only
**Ethics:** Responsible Disclosure & Privacy Protection
