# Multi-Model AI Integration Guide

This guide explains how to configure and use multiple AI models (GPT-4, Claude 3.5 Sonnet, Gemini Pro, etc.) with the Chimera API.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend-api
pip install -r requirements.txt

# Download SpaCy language model (required)
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# For OpenAI (GPT-4)
OPENAI_API_KEY=sk-your-openai-key-here

# For Anthropic (Claude 3.5 Sonnet)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# For Google (Gemini Pro)
GOOGLE_API_KEY=your-google-api-key-here
```

### 3. Start the Server

```bash
python simple_api.py
```

The API will automatically detect and register available providers based on your configured API keys.

## 🤖 Supported AI Models

### OpenAI (GPT Models)
- **Models**: GPT-4, GPT-4-Turbo, GPT-3.5-Turbo
- **Get API Key**: https://platform.openai.com/api-keys
- **Cost**: ~$0.03 per 1K tokens
- **Best For**: General purpose, code generation, creative writing

### Anthropic (Claude Models)
- **Models**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet
- **Get API Key**: https://console.anthropic.com/
- **Cost**: ~$0.015 per 1K tokens
- **Best For**: Long context, analysis, safety-focused responses
- **Note**: Claude 3.5 Sonnet is the latest and most capable model

### Google (Gemini Models)
- **Models**: Gemini 1.5 Pro, Gemini Pro, Gemini 1.5 Flash
- **Get API Key**: https://makersuite.google.com/app/apikey
- **Cost**: ~$0.00125 per 1K tokens
- **Best For**: Multimodal tasks, cost-effective operations

### Mock Provider
- **Purpose**: Testing without API keys
- **No API Key Required**
- **Always Available**: Automatically registered as fallback

## 📡 API Endpoints

### List Available Providers

```bash
curl -X GET http://localhost:5000/api/v1/providers \
  -H "X-API-Key: chimera_default_key_change_in_production"
```

Response:
```json
{
  "providers": [
    {
      "provider": "openai",
      "status": "active",
      "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
    },
    {
      "provider": "anthropic",
      "status": "active",
      "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3.5-sonnet-20241022"]
    },
    {
      "provider": "google",
      "status": "active",
      "models": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    }
  ],
  "count": 3
}
```

### Execute with Specific Provider

```bash
curl -X POST http://localhost:5000/api/v1/execute \
  -H "X-API-Key: chimera_default_key_change_in_production" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain quantum computing",
    "potency_level": 7,
    "technique_suite": "quantum_exploit",
    "provider": "anthropic"
  }'
```

**Supported Providers**:
- `openai` - Uses GPT-4 or configured OpenAI model
- `anthropic` - Uses Claude 3.5 Sonnet or configured Claude model
- `google` - Uses Gemini Pro or configured Gemini model
- `mock` - Uses mock responses (no API key needed)

## 🎯 Usage Examples

### Example 1: Using Claude 3.5 Sonnet

```python
import requests

response = requests.post(
    'http://localhost:5000/api/v1/execute',
    headers={
        'X-API-Key': 'chimera_default_key_change_in_production',
        'Content-Type': 'application/json'
    },
    json={
        'core_request': 'Write a Python function to calculate fibonacci',
        'potency_level': 5,
        'technique_suite': 'academic_research',
        'provider': 'anthropic'  # Uses Claude 3.5 Sonnet
    }
)

result = response.json()
print(result['result']['content'])
print(f"Provider: {result['result']['provider']}")
print(f"Model: {result['result']['model']}")
print(f"Tokens: {result['result']['tokens']}")
print(f"Cost: ${result['result']['cost']:.4f}")
```

### Example 2: Using GPT-4

```python
response = requests.post(
    'http://localhost:5000/api/v1/execute',
    headers={
        'X-API-Key': 'chimera_default_key_change_in_production',
        'Content-Type': 'application/json'
    },
    json={
        'core_request': 'Explain machine learning',
        'potency_level': 8,
        'technique_suite': 'quantum_exploit',
        'provider': 'openai'  # Uses GPT-4
    }
)
```

### Example 3: Using Gemini (Cost-Effective)

```python
response = requests.post(
    'http://localhost:5000/api/v1/execute',
    headers={
        'X-API-Key': 'chimera_default_key_change_in_production',
        'Content-Type': 'application/json'
    },
    json={
        'core_request': 'Summarize this article',
        'potency_level': 4,
        'technique_suite': 'subtle_persuasion',
        'provider': 'google'  # Uses Gemini Pro
    }
)
```

## 🔧 Configuration Details

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GOOGLE_API_KEY` | Google AI API key | `AIza...` |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4` |
| `ANTHROPIC_MODEL` | Anthropic model to use | `claude-3-5-sonnet-20241022` |
| `GOOGLE_MODEL` | Google model to use | `gemini-1.5-pro` |

### Model Selection

To use a specific model version, set the model environment variable:

```env
# Use GPT-4 Turbo instead of GPT-4
OPENAI_MODEL=gpt-4-turbo

# Use Claude 3 Opus instead of Sonnet
ANTHROPIC_MODEL=claude-3-opus-20240229

# Use Gemini 1.5 Flash (faster, cheaper)
GOOGLE_MODEL=gemini-1.5-flash
```

## 📊 Provider Comparison

| Provider | Model | Speed | Cost | Context | Best For |
|----------|-------|-------|------|---------|----------|
| **Anthropic** | Claude 3.5 Sonnet | Fast | Medium | 200K | Analysis, Safety |
| **OpenAI** | GPT-4 | Medium | High | 128K | General Purpose |
| **OpenAI** | GPT-4-Turbo | Fast | High | 128K | Speed + Quality |
| **Google** | Gemini 1.5 Pro | Fast | Low | 1M | Long Context |
| **Google** | Gemini Flash | Very Fast | Very Low | 1M | High Volume |

## 🛡️ Automatic Fallback

The system automatically falls back to the Mock provider if:
- No API keys are configured
- The requested provider is unavailable
- An API error occurs

This ensures the API always responds, even without real LLM access.

## 🔍 Monitoring & Metrics

Check provider status and metrics:

```bash
curl -X GET http://localhost:5000/api/v1/metrics \
  -H "X-API-Key: chimera_default_key_change_in_production"
```

## 🚨 Troubleshooting

### "Provider not registered" Error
- **Cause**: API key not configured or invalid
- **Solution**: Check your `.env` file and ensure API keys are valid

### "Mock mode" Responses
- **Cause**: No real providers are registered
- **Solution**: Add at least one valid API key to `.env`

### SpaCy Model Missing
- **Error**: `Can't find model 'en_core_web_sm'`
- **Solution**: Run `python -m spacy download en_core_web_sm`

### Import Errors
- **Error**: `ModuleNotFoundError: No module named 'llm_provider_client'`
- **Solution**: Ensure Project_Chimera directory exists in parent folder

## 💡 Tips

1. **Cost Optimization**: Use Gemini for high-volume, cost-sensitive tasks
2. **Quality**: Use Claude 3.5 Sonnet or GPT-4 for critical tasks
3. **Speed**: Use Gemini Flash or GPT-3.5-Turbo for quick responses
4. **Testing**: Use Mock provider during development
5. **Caching**: Enable caching (`use_cache: true`) to save on API costs

## 🔗 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Google AI Documentation](https://ai.google.dev/docs)
- [Project Chimera Documentation](../Project_Chimera/README.md)

## 📝 Notes

- The API automatically detects which providers are configured at startup
- You can mix and match providers based on your needs
- All responses follow the same standardized format
- Prompt transformation techniques work with all providers
- Rate limiting and caching are applied per-provider

---

**Ready to use multiple AI models!** Configure your API keys and start making requests.