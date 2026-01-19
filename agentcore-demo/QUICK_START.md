# AgentCore Quick Start Guide

## What You Just Installed

The **aws-agentcore** Kiro Power gives you:

- 📚 **Documentation search** via MCP tools
- 🛠️ **CLI commands** for agent development
- 🚀 **Local dev workflow** with hot-reload
- ☁️ **AWS deployment** capabilities

## Your Demo Project Structure

```
agentcore-demo/
├── src/
│   └── main.py              # Your agent code
├── .bedrock_agentcore.yaml  # AgentCore config
├── requirements.txt         # Python dependencies
└── README.md                # Full documentation
```

## Try It Now (3 Steps)

### 1. Install Dependencies

```bash
cd agentcore-demo
pip install -r requirements.txt
```

### 2. Start Dev Server

```bash
agentcore dev
```

This starts a local server at <http://localhost:8080> with hot-reload.

### 3. Test Your Agent (in another terminal)

```bash
# Simple text
agentcore invoke --dev "Hello, what can you help me with?"

# JSON format
agentcore invoke --dev '{"prompt": "Explain AWS in one sentence"}'
```

## Using the Power's MCP Tools

### Search Documentation

Ask me: "Search AgentCore docs for deployment"
I'll use: `kiroPowers("aws-agentcore", "search_agentcore_docs", {"query": "deployment"})`

### Get Memory CLI Commands

Ask me: "How do I create AgentCore Memory?"
I'll use: `kiroPowers("aws-agentcore", "manage_agentcore_memory", {})`

### Get Gateway Info

Ask me: "How do I deploy an MCP Gateway?"
I'll use: `kiroPowers("aws-agentcore", "manage_agentcore_gateway", {})`

## Next Steps

1. **Modify the agent**: Edit `src/main.py` and add tools
2. **Add Memory**: Create persistent conversation memory
3. **Add Gateway**: Expose MCP tools to your agent
4. **Deploy to AWS**: Use `agentcore launch` when ready

## Common Commands

```bash
# Development
agentcore dev                    # Start dev server
agentcore invoke --dev "text"    # Test locally

# Memory Management
agentcore memory create my_mem   # Create memory resource
agentcore memory list            # List all memories
agentcore memory get <id>        # Get memory details

# Gateway Management
agentcore gateway create my_gw   # Create gateway
agentcore gateway list           # List gateways

# Deployment
agentcore configure              # Configure for deployment
agentcore launch                 # Deploy to AWS
agentcore status                 # Check deployment status
agentcore invoke '{"prompt":"hi"}'  # Test deployed agent
agentcore stop-session           # Stop active session
agentcore destroy --dry-run      # Preview destruction
agentcore destroy                # Delete all resources
```

## Getting Help

- **Search docs**: Ask me to search AgentCore documentation
- **CLI help**: Run `agentcore --help` or `agentcore <command> --help`
- **Steering files**: Ask me to read specific steering guides
