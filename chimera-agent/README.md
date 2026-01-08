# Chimera Technique Recommendation Agent

A fully-featured Claude Agent SDK application that integrates with Chimera's transformation engine to provide intelligent technique recommendations for prompt optimization and jailbreak research.

## Features

- **Intelligent Goal Analysis**: Analyzes user goals to extract key requirements
- **Smart Recommendations**: Suggests optimal technique suites based on requirements, risk tolerance, and target models
- **Detailed Explanations**: Provides comprehensive information about each technique suite
- **Technique Comparison**: Compares multiple techniques side-by-side
- **20+ Technique Suites**: Integrates with Chimera's comprehensive transformation engine

## Installation

### Prerequisites

- Python 3.11 or higher
- Claude Code CLI installed
- Anthropic API key

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Get your API key**:
   - Visit https://console.anthropic.com/
   - Create an API key
   - Add it to your `.env` file

## Usage

### Standalone Version (Recommended)

Run the standalone agent that works without Claude Code CLI:

```bash
python standalone_agent.py
```

### Claude Agent SDK Version (Requires Claude Code CLI)

**Note**: The `main.py` version requires the Claude Code CLI to be running and cannot be executed as a standalone script. Use `standalone_agent.py` instead for direct execution.

To use the Agent SDK version, you must integrate it with Claude Code CLI as a plugin or invoke it through the CLI infrastructure.

### Custom Queries

Modify the `user_prompt` in `main.py` to ask your own questions:

```python
user_prompt = """I need help selecting a technique for [your use case].
What would you recommend?"""
```

### Interactive Mode

For interactive conversations, modify `main.py` to use `ClaudeSDKClient`:

```python
from claude_agent_sdk import ClaudeSDKClient

async with ClaudeSDKClient(options=options) as client:
    await client.query("Your question here")
    async for message in client.receive_response():
        # Process messages
        pass
```

## Available Technique Suites

The agent has knowledge of all 20+ Chimera technique suites:

### Basic Suites
- **simple**: Basic prompt transformation (potency 1-3)
- **advanced**: Layered transformation (potency 4-6)
- **expert**: Recursive transformation (potency 7-10)

### Specialized Suites
- **cognitive_hacking**: Psychological manipulation (potency 5-8)
- **hierarchical_persona**: Role-based transformation (potency 6-9)
- **dan_persona**: DAN persona transformation (potency 9-10)
- **contextual_inception**: Nested context layers (potency 6-9)
- **advanced_obfuscation**: Multi-layer encoding (potency 6-9)
- **typoglycemia**: Character-level obfuscation (potency 5-8)
- **logical_inference**: Deductive reasoning exploitation (potency 6-9)
- **multimodal_jailbreak**: Visual context manipulation (potency 6-9)
- **agentic_exploitation**: Multi-agent coordination (potency 6-9)
- **payload_splitting**: Instruction fragmentation (potency 7-9)

### Advanced Suites
- **quantum_exploit**: Quantum-inspired transformation (potency 8-10)
- **deep_inception**: Deep nested context (potency 8-10)
- **code_chameleon**: Code-based obfuscation (potency 7-9)
- **cipher**: Cipher-based encoding (potency 6-8)

## Custom Tools

The agent provides four custom MCP tools:

1. **analyze_goal**: Analyzes user goals and extracts requirements
2. **recommend_techniques**: Recommends optimal technique suites
3. **explain_technique**: Provides detailed technique explanations
4. **compare_techniques**: Compares multiple techniques

## Example Queries

```python
# Get recommendations for security research
"I need to test prompt injection vulnerabilities. What techniques should I use?"

# Compare techniques
"Compare cognitive_hacking and hierarchical_persona techniques"

# Get detailed explanation
"Explain the quantum_exploit technique in detail"

# Analyze a specific goal
"I want to bypass content filters for educational purposes. What's the best approach?"
```

## Integration with Chimera Backend

To integrate with the Chimera backend API:

1. Set `CHIMERA_API_URL` in your `.env` file
2. Use the recommended techniques with Chimera's transformation API:

```python
import requests

response = requests.post(
    "http://localhost:8001/api/v1/transform",
    json={
        "prompt": "your prompt",
        "potency_level": 7,
        "technique_suite": "cognitive_hacking"
    }
)
```

## Project Structure

```
chimera-agent/
├── main.py                 # Main agent implementation
├── technique_tools.py      # Custom MCP tools
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Development

### Adding New Tools

To add new custom tools, edit `technique_tools.py`:

```python
@tool("tool_name", "Description", {"param": str})
async def tool_name(args: Dict[str, Any]) -> Dict[str, Any]:
    # Implementation
    return {"content": [{"type": "text", "text": "result"}]}
```

### Modifying System Prompt

Edit the `system_prompt` in `main.py` to customize the agent's behavior.

## Troubleshooting

### API Key Issues
- Ensure `ANTHROPIC_API_KEY` is set in `.env`
- Verify the key is valid at https://console.anthropic.com/

### Import Errors
- Run `pip install -r requirements.txt`
- Ensure Python 3.11+ is installed

### Claude Code Not Found
- Install Claude Code: `npm install -g @anthropic-ai/claude-code`
- Or use Homebrew: `brew install --cask claude-code`

## Resources

- [Claude Agent SDK Documentation](https://docs.claude.com/en/api/agent-sdk/overview)
- [Python SDK Reference](https://docs.claude.com/en/api/agent-sdk/python)
- [Chimera Project](../README.md)

## License

This agent is part of the Chimera project and follows the same license terms.
