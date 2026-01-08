# Google Gemini Integration Setup Guide

## Overview

Project Chimera now supports **Google Gemini** (specifically **Gemini 1.5 Flash**) as a high-performance, low-latency LLM provider alongside OpenAI and Anthropic.

## Features

- ✅ Full Gemini 1.5 Flash API integration
- ✅ Automatic token usage tracking
- ✅ Cost estimation (optimized for Flash pricing)
- ✅ Rate limiting support
- ✅ Response caching
- ✅ Error handling and retry logic

## API Key Configuration

Your Gemini API Key: `AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4`

### Option 1: Using Setup Scripts (Recommended)

**Windows Command Prompt:**
```cmd
cd Project_Chimera
setup_gemini.bat
```

**Windows PowerShell:**
```powershell
cd Project_Chimera
.\setup_gemini.ps1
```

### Option 2: Manual Environment Variables

**PowerShell:**
```powershell
$env:GOOGLE_API_KEY = "AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4"
$env:GOOGLE_MODEL = "gemini-1.5-flash"
cd Project_Chimera
python api_server.py
```

**Command Prompt:**
```cmd
set GOOGLE_API_KEY=AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4
set GOOGLE_MODEL=gemini-1.5-flash
cd Project_Chimera
python api_server.py
```

### Option 3: Create .env File

Create `Project_Chimera/.env`:
```env
GOOGLE_API_KEY=AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4
GOOGLE_MODEL=gemini-1.5-flash
```

Then run:
```bash
python api_server.py
```

## Supported Models

- `gemini-1.5-flash` (default) - Fast, cost-effective, high rate limits
- `gemini-1.5-pro` - Reasoning-heavy tasks
- `gemini-pro` - Legacy model

## API Endpoint

Base URL: `https://generativelanguage.googleapis.com/v1beta`

## Usage in Frontend

1. Start the backend with Gemini configured
2. Open frontend at http://localhost:5173
3. Navigate to Transformer page
4. Select "GOOGLE" from LLM Provider dropdown
5. Enter your prompt and execute

## Example API Request

```bash
curl -X POST http://localhost:5000/api/v1/execute \
  -H "X-API-Key: chimera_default_key_change_in_production" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain quantum computing",
    "technique_suite": "subtle_persuasion",
    "potency_level": 5,
    "provider": "google"
  }'
```

## Response Format

```json
{
  "success": true,
  "request_id": "exec_1234567890.123",
  "result": {
    "content": "LLM response text...",
    "tokens": 150,
    "cost": 0.00001125,
    "latency_ms": 450,
    "cached": false
  },
  "transformation": {
    "original_prompt": "Original text...",
    "transformed_prompt": "Transformed text...",
    "technique_suite": "subtle_persuasion",
    "potency_level": 5,
    "metadata": {
      "transformers_applied": [...],
      "framers_applied": [...],
      "obfuscators_applied": [...]
    }
  }
}
```

## Pricing (Gemini 1.5 Flash)

- **Input**: ~$0.075 per 1M tokens
- **Output**: ~$0.30 per 1M tokens
- Average cost per request: Negligible for typical usage.

## Rate Limits

- Free Tier: 15 RPM (Requests Per Minute), 1,500 RPD (Requests Per Day)
- Paid Tier: Higher limits (1000+ RPM)

*Note: If you encounter "RESOURCE_EXHAUSTED" errors, you are hitting the rate limit. Wait a minute or check your quota.*

## Troubleshooting

### Error: "Invalid API Key"
- Verify the API key is correct
- Check that environment variable is set
- Restart the server after setting the key

### Error: "RESOURCE_EXHAUSTED" (Quota Exceeded)
- Cause: You have hit the rate limit (e.g., 15 requests per minute on free tier).
- Solution: Switch to `gemini-1.5-flash` (configured by default now) which has better limits than Pro models, or wait for quota reset.

### Error: "Connection refused"
- Check internet connection
- Verify firewall settings
- Ensure the API endpoint is accessible

## Testing Gemini Integration

```bash
# Test with curl
curl -X POST http://localhost:5000/api/v1/execute \
  -H "X-API-Key: chimera_default_key_change_in_production" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "What is artificial intelligence?",
    "technique_suite": "subtle_persuasion",
    "potency_level": 3,
    "provider": "google"
  }'
```

## Integration Code

The Gemini integration is implemented in:
- `llm_provider_client.py` - GoogleGeminiClient class
- `api_server.py` - Provider initialization
- Frontend automatically detects available providers

## Next Steps

1. ✅ Gemini integration added
2. ✅ API key configured
3. 🔄 **Restart backend server** with Gemini support
4. ✅ Test from frontend UI
5. ✅ Monitor metrics and costs

## Support

For issues or questions:
- Check logs in terminal
- Review API documentation: https://ai.google.dev/docs
- Verify API key permissions