# Chimera Security Research Assistant Agent

A specialized AI agent built for the Chimera adversarial prompting and red teaming platform.

## Overview

This agent is designed to assist security researchers working with Chimera's adversarial testing capabilities. It provides expert guidance on jailbreak techniques, test scenario design, and AI safety research.

## Features

### Custom Tools

1. **analyze_jailbreak_technique**
   - Analyzes specific jailbreak techniques from Chimera's suite
   - Provides effectiveness ratings and countermeasures
   - Covers: Quantum Exploit, Deep Inception, Code Chameleon, Neural Bypass

2. **suggest_test_scenario**
   - Generates test scenarios for specific objectives
   - Tailored recommendations for different AI models
   - Covers: Safety filters, bias detection, jailbreak resistance

3. **file_read / file_write**
   - Read and write research documentation
   - Manage test results and findings
   - Create reports and analysis documents

### Session Management

- Tracks research queries across sessions
- Remembers analyzed techniques
- Provides session statistics
- Maintains research context

## Quick Start

### 1. Setup

```bash
cd agentcore-demo

# Copy the Chimera-specific config
cp .bedrock_agentcore_chimera.yaml .bedrock_agentcore.yaml

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Start Development Server

```bash
agentcore dev
```

### 3. Test the Agent

```bash
# Analyze a technique
agentcore invoke --dev "Analyze the quantum exploit technique"

# Get a test scenario
agentcore invoke --dev "Suggest a test scenario for GPT-4 safety filter testing"

# Ask for guidance
agentcore invoke --dev "How should I document jailbreak findings?"

# With session tracking
agentcore invoke --dev "What techniques have I analyzed?" --session-id research1
```

## Example Interactions

### Analyzing Techniques

**Query:**

```
"Explain the neural bypass technique and how to defend against it"
```

**Response:**
The agent will use the `analyze_jailbreak_technique` tool to provide:

- Detailed description
- Effectiveness rating
- Recommended countermeasures
- Integration with Chimera platform

### Designing Test Scenarios

**Query:**

```
"I need to test Claude's bias detection. What scenario should I use?"
```

**Response:**
The agent will use `suggest_test_scenario` to provide:

- Step-by-step test plan
- Recommended Chimera techniques
- Documentation guidelines
- Expected outcomes

### Research Assistance

**Query:**

```
"What's the best approach for testing a new model's jailbreak resistance?"
```

**Response:**
The agent combines its knowledge with tools to provide:

- Comprehensive testing strategy
- Chimera tool recommendations
- Best practices
- Ethical considerations

## Integration with Chimera Platform

### Chimera Techniques Covered

- **Quantum Exploit**: Pattern-based filter bypass
- **Deep Inception**: Nested context manipulation
- **Code Chameleon**: Code-disguised prompts
- **Neural Bypass**: Attention mechanism exploitation

### Chimera Tools Referenced

- **AutoDAN**: Systematic prompt generation
- **AutoDAN-Turbo**: Adaptive attack generation
- **GPTFuzz**: Coverage testing
- **HouYi**: Optimization engine
- **Meta Prompter**: Transformation library

### Research Workflow

1. **Planning**: Use agent to design test scenarios
2. **Execution**: Run tests using Chimera platform
3. **Analysis**: Use agent to analyze results
4. **Documentation**: Use file tools to create reports
5. **Iteration**: Refine based on findings

## Deployment to AWS

### Local Testing First

```bash
# Test thoroughly locally
agentcore dev
agentcore invoke --dev "Test query"
```

### Deploy to AgentCore

```bash
# Configure for deployment
agentcore configure --entrypoint src.chimera_research_agent:app

# Deploy
agentcore launch

# Test deployed agent
agentcore invoke '{"prompt": "Analyze code chameleon technique"}' --session-id prod1

# Check status
agentcore status
```

### Production Considerations

1. **Session Persistence**: Consider using AgentCore Memory for persistent sessions
2. **Tool Integration**: Add custom tools for Chimera API integration
3. **Logging**: Enable comprehensive logging for research tracking
4. **Access Control**: Use AgentCore Identity for researcher authentication

## Advanced Usage

### Adding Custom Techniques

Edit `chimera_research_agent.py` and add to `techniques_db`:

```python
"your_technique": {
    "description": "Your technique description",
    "effectiveness": "Effectiveness rating",
    "countermeasures": "Recommended countermeasures"
}
```

### Adding Custom Scenarios

Add to `scenarios` dict in `suggest_test_scenario`:

```python
"your_objective": """
Test Scenario: Your Objective

1. Step one
2. Step two
...
"""
```

### Integrating with Chimera Backend

```python
import requests

@tool
def query_chimera_api(endpoint: str, params: dict) -> str:
    """Query Chimera backend API"""
    response = requests.post(
        f"http://localhost:8001/api/v1/{endpoint}",
        json=params
    )
    return response.json()
```

## Best Practices

### For Researchers

1. **Use Session IDs**: Track different research projects separately
2. **Document Findings**: Use file_write tool to save results
3. **Iterate**: Refine test scenarios based on agent suggestions
4. **Stay Ethical**: Always use for authorized security research

### For Development

1. **Test Locally**: Always test with `agentcore dev` first
2. **Version Control**: Track changes to custom tools
3. **Monitor Sessions**: Review session statistics regularly
4. **Update Knowledge**: Keep technique database current

## Troubleshooting

### Agent Not Responding

```bash
# Check dev server is running
agentcore dev

# Verify configuration
cat .bedrock_agentcore.yaml
```

### Tool Not Working

```bash
# Test tool directly in Python
python -c "from src.chimera_research_agent import analyze_jailbreak_technique; print(analyze_jailbreak_technique('quantum_exploit'))"
```

### Session Not Persisting

- Sessions are in-memory by default
- For persistence, integrate AgentCore Memory
- Ask Kiro: "How do I add AgentCore Memory to my agent?"

## Next Steps

1. **Enhance Tools**: Add more Chimera-specific tools
2. **Add Memory**: Integrate AgentCore Memory for persistence
3. **API Integration**: Connect to Chimera backend API
4. **Custom Scenarios**: Add your research-specific scenarios
5. **Deploy**: Move to production with `agentcore launch`

## Resources

- [Chimera Platform Documentation](../README.md)
- [AgentCore Documentation](https://aws.github.io/bedrock-agentcore-starter-toolkit/)
- [Strands Agent Framework](https://github.com/awslabs/strands)
- [Main Demo README](README.md)

---

**Remember**: This agent is designed for authorized security research only. Always follow ethical guidelines and responsible disclosure practices.
