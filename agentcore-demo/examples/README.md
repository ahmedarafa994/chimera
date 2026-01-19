# AgentCore Examples

This directory contains practical examples demonstrating various AgentCore capabilities.

## Examples Overview

### 1. Streaming Agent (`streaming_agent.py`)

Demonstrates real-time streaming responses with tool usage.

**Features:**

- Async streaming with `agent.stream_async()`
- Calculator tool integration
- Real-time event streaming

**Usage:**

```bash
# Update config to use streaming agent
agentcore configure --entrypoint examples.streaming_agent:app

# Start dev server
agentcore dev

# Test with a math question
agentcore invoke --dev "What is 25 * 47 + 123?"
```

### 2. Session Management Agent (`session_agent.py`)

Shows how to maintain conversation state across multiple requests.

**Features:**

- Session ID tracking
- In-memory conversation history
- Context-aware responses

**Usage:**

```bash
# Configure
agentcore configure --entrypoint examples.session_agent:app

# Deploy
agentcore launch

# Start a conversation
agentcore invoke '{"prompt": "Hello, my name is Alice"}' --session-id user1

# Continue the conversation (remembers context)
agentcore invoke '{"prompt": "What is my name?"}' --session-id user1

# Different session (no shared context)
agentcore invoke '{"prompt": "What is my name?"}' --session-id user2
```

## Running Examples Locally

1. **Install dependencies:**

```bash
cd agentcore-demo
pip install -r requirements.txt
```

1. **Choose an example and configure:**

```bash
agentcore configure --entrypoint examples.streaming_agent:app
```

1. **Start dev server:**

```bash
agentcore dev
```

1. **Test in another terminal:**

```bash
agentcore invoke --dev "Your test message"
```

## Deploying Examples to AWS

1. **Configure for deployment:**

```bash
agentcore configure --entrypoint examples.streaming_agent:app
```

1. **Deploy:**

```bash
agentcore launch
```

1. **Test deployed agent:**

```bash
agentcore invoke '{"prompt": "Hello from AWS!"}'
```

1. **Clean up:**

```bash
agentcore destroy --dry-run  # Preview
agentcore destroy            # Actually destroy
```

## Adding More Examples

To add your own example:

1. Create a new Python file in this directory
2. Follow the pattern:

```python
import os
os.environ["BYPASS_TOOL_CONSENT"] = "true"

from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp

agent = Agent(...)
app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    # Your logic here
    pass

if __name__ == "__main__":
    app.run()
```

1. Configure and test:

```bash
agentcore configure --entrypoint examples.your_example:app
agentcore dev
```

## Next Steps

- **Add Memory**: Integrate AgentCore Memory for persistent storage
- **Add Gateway**: Connect MCP tools via AgentCore Gateway
- **Add Tools**: Use strands_tools for file operations, web browsing, etc.
- **Production Deploy**: Use CDK/Terraform for infrastructure as code

## Resources

- [AgentCore Documentation](https://aws.github.io/bedrock-agentcore-starter-toolkit/)
- [Strands Agent Framework](https://github.com/awslabs/strands)
- [Main README](../README.md)
- [Quick Start Guide](../QUICK_START.md)
