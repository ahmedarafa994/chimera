# AgentCore Demo Agent

A simple demonstration of AWS Bedrock AgentCore with Strands agent framework.

## Setup

1. Install dependencies:

```bash
cd agentcore-demo
pip install -r requirements.txt
```

## Local Development

1. Start the development server:

```bash
agentcore dev
```

1. In another terminal, test the agent:

```bash
# Simple text (auto-wrapped in JSON)
agentcore invoke --dev "Hello, what can you help me with?"

# Or explicit JSON
agentcore invoke --dev '{"prompt": "What is AWS?"}'
```

## How It Works

- **src/main.py**: Main agent code with Strands Agent wrapped in BedrockAgentCoreApp
- **.bedrock_agentcore.yaml**: Configuration file specifying the entrypoint
- **requirements.txt**: Python dependencies

## Next Steps

1. **Add tools**: Import tools from `strands_tools` (file_read, file_write, etc.)
2. **Add memory**: Use AgentCore Memory for persistent conversations
3. **Add gateway**: Expose MCP tools via AgentCore Gateway
4. **Deploy**: Use `agentcore configure` and `agentcore launch` to deploy to AWS

## Deployment

When ready to deploy to AWS:

```bash
# Configure deployment
agentcore configure --entrypoint src.main:app

# Deploy to AWS
agentcore launch

# Test deployed agent
agentcore invoke '{"prompt": "Hello from the cloud!"}'

# Check status
agentcore status

# Stop session (free resources)
agentcore stop-session

# Destroy all resources
agentcore destroy --dry-run  # Preview first
agentcore destroy            # Actually destroy
```
