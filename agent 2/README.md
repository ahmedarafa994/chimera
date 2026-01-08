# Automated Prompt Quality Auditor Agent

An AI-powered agent built with the Claude Agent SDK that monitors and improves prompt quality across the Chimera AI system.

## Features

- **Prompt Fetching**: Retrieves recent prompts from the Chimera backend API
- **Quality Analysis**: Analyzes prompts for clarity, completeness, and effectiveness
- **Metrics Generation**: Provides detailed quality scores and metrics
- **Improvement Suggestions**: Offers actionable recommendations for prompt optimization
- **Comprehensive Reports**: Generates audit reports with quality insights

## Prerequisites

- Python 3.11 or higher
- Claude Code CLI installed
- Anthropic API key

## Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   - `ANTHROPIC_API_KEY`: Get from https://console.anthropic.com/
   - `CHIMERA_API_KEY`: Your Chimera API key (optional)

## Usage

Run the agent:

```bash
python main.py
```

The agent will:
1. Fetch recent prompts from the Chimera system
2. Analyze each prompt for quality metrics
3. Generate a comprehensive quality report
4. Provide specific improvement suggestions

## Quality Metrics

The agent evaluates prompts based on:

- **Clarity Score**: How clear and unambiguous the prompt is
- **Completeness Score**: Whether sufficient context is provided
- **Effectiveness Score**: Likelihood of producing desired outcomes
- **Overall Score**: Weighted average of all metrics

## Custom Tools

The agent includes three custom MCP tools:

1. **fetch_prompts**: Retrieves prompts from Chimera API
2. **analyze_prompt_quality**: Analyzes individual prompt quality
3. **generate_quality_report**: Creates comprehensive audit reports

## Integration with Chimera

To integrate with the Chimera backend:

1. Ensure the Chimera backend is running on `http://localhost:8001`
2. Set the `CHIMERA_API_KEY` in your `.env` file
3. Update the `fetch_prompts` tool in `main.py` to call actual API endpoints

## Development

The agent uses:
- **Claude Agent SDK v0.1.16**: For agent orchestration
- **Custom MCP Tools**: For prompt analysis functionality
- **Async/Await**: For efficient I/O operations

## Architecture

```
main.py
├── Custom Tools (@tool decorators)
│   ├── fetch_prompts
│   ├── analyze_prompt_quality
│   └── generate_quality_report
├── SDK MCP Server (create_sdk_mcp_server)
└── Agent Configuration (ClaudeAgentOptions)
```

## Next Steps

1. **Connect to Real API**: Replace mock data in `fetch_prompts` with actual Chimera API calls
2. **Enhance Analysis**: Add more sophisticated quality metrics
3. **Automate Scheduling**: Set up periodic audits using cron or task scheduler
4. **Add Persistence**: Store audit results in a database
5. **Create Dashboard**: Build a web interface for viewing audit results

## Resources

- [Claude Agent SDK Documentation](https://docs.claude.com/en/api/agent-sdk/python)
- [Chimera Project Documentation](../README.md)
- [MCP Protocol](https://modelcontextprotocol.io/)

## License

Part of the Chimera project.
