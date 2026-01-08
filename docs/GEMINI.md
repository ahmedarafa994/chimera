# Gemini API Integration Guide

This guide explains how to use Google's Gemini models through OpenAI's Python client library.

## Overview

The code you provided demonstrates how to use Gemini's **experimental thinking/reasoning mode** (similar to OpenAI's o1 models) through the OpenAI-compatible API endpoint.

## Code Explanation

```python
from openai import OpenAI

# Initialize the OpenAI client with Gemini's endpoint
client = OpenAI(
    api_key="GEMINI_API_KEY",  # Replace with your actual Google API key
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Make a chat completion request with reasoning enabled
response = client.chat.completions.create(
    model="gemini-2.5-pro",           # Use Gemini 2.5 Pro model
    reasoning_effort="high",          # Enable deep reasoning (experimental)
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain to me how AI works"}
    ]
)

print(response.choices[0].message)
```

## Key Components

### 1. **API Endpoint**
- Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
- This is Google's OpenAI-compatible endpoint for Gemini models

### 2. **Model Selection**
- `gemini-2.5-pro`: Latest Gemini Pro model with advanced capabilities
- Other options: `gemini-1.5-pro`, `gemini-1.5-flash`

### 3. **Reasoning Effort** (Experimental Feature)
- `reasoning_effort`: Controls the depth of thinking before responding
- Values: `"low"`, `"medium"`, `"high"`
- Higher values = more deliberate, step-by-step reasoning
- Similar to OpenAI's o1 reasoning models

### 4. **Message Format**
Standard OpenAI chat format with roles:
- `system`: Sets the assistant's behavior
- `user`: User's input
- `assistant`: Model's responses

## Setup Instructions

### 1. Get Your API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or select a project
3. Generate an API key
4. Copy the key

### 2. Install Required Package
```bash
pip install openai
```

### 3. Set Up Environment Variable (Recommended)
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-api-key-here"

# Windows (CMD)
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY=your-api-key-here
```

### 4. Use in Your Code
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

## Complete Working Example

```python
import os
from openai import OpenAI

# Initialize client
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Example 1: Basic chat completion
response = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)

# Example 2: With reasoning (experimental)
response = client.chat.completions.create(
    model="gemini-2.5-pro",
    reasoning_effort="high",
    messages=[
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "Solve: If x + 5 = 12, what is x?"}
    ]
)
print(response.choices[0].message.content)

# Example 3: With temperature and max_tokens
response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    temperature=0.9,
    max_tokens=100
)
print(response.choices[0].message.content)

# Example 4: Streaming responses
stream = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[
        {"role": "user", "content": "Count from 1 to 5"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `gemini-2.5-pro` | Latest, most capable | Complex reasoning, long context |
| `gemini-1.5-pro` | Advanced model | General tasks, multimodal |
| `gemini-1.5-flash` | Fast, efficient | Quick responses, high volume |

## Parameters Reference

```python
response = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[...],

    # Optional parameters
    temperature=0.7,          # 0.0-2.0, controls randomness
    max_tokens=1024,          # Maximum tokens to generate
    top_p=0.95,              # Nucleus sampling
    frequency_penalty=0.0,    # -2.0 to 2.0
    presence_penalty=0.0,     # -2.0 to 2.0
    reasoning_effort="high",  # "low", "medium", "high" (experimental)
    stream=False,             # Enable streaming
    n=1,                     # Number of completions
)
```

## Integration with Project Chimera

To integrate this with your existing project:

1. **Update `.env` file:**
```env
GEMINI_API_KEY=your-actual-api-key-here
```

2. **Create a test script:**
```python
# test_gemini_openai.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-pro",
    reasoning_effort="high",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain to me how AI works"}
    ]
)

print(response.choices[0].message.content)
```

3. **Run the test:**
```bash
python test_gemini_openai.py
```

## Reasoning Mode Details

The `reasoning_effort` parameter is **experimental** and enables the model to:
- Think step-by-step before answering
- Show its reasoning process (in some cases)
- Provide more accurate, well-thought-out responses
- Take longer to respond (trade-off: speed vs. quality)

**Use Cases for High Reasoning:**
- Complex math problems
- Logical puzzles
- Code debugging
- Multi-step planning
- Critical analysis

## Error Handling

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

try:
    response = client.chat.completions.create(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

except RateLimitError:
    print("Rate limit exceeded. Please wait and try again.")
except APIConnectionError:
    print("Failed to connect to the API.")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Comparison: Native SDK vs OpenAI Client

### Native Google GenAI SDK:
```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')
response = model.generate_content("Explain AI")
```

### OpenAI Client (Your Code):
```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
response = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[{"role": "user", "content": "Explain AI"}]
)
```

**Benefits of OpenAI Client:**
- ✅ Familiar API for OpenAI users
- ✅ Easy to switch between providers
- ✅ Same code structure across models
- ✅ Access to reasoning_effort parameter

**Benefits of Native SDK:**
- ✅ Full access to Gemini-specific features
- ✅ Better documentation for Gemini
- ✅ Direct Google support

## Additional Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [OpenAI Python Client](https://github.com/openai/openai-python)
- [Gemini Pricing](https://ai.google.dev/pricing)

## Troubleshooting

### Issue: "Invalid API Key"
- Verify your API key is correct
- Check if the key is enabled for the Gemini API
- Ensure no extra spaces in the key

### Issue: "Model not found"
- Verify model name spelling: `gemini-2.5-pro`
- Check if you have access to that model version
- Try `gemini-1.5-pro` or `gemini-1.5-flash`

### Issue: "Rate limit exceeded"
- Wait before making more requests
- Implement exponential backoff
- Consider upgrading your quota

### Issue: Reasoning mode not working
- `reasoning_effort` is experimental and may not work in all regions
- Try without it first to verify basic connectivity
- Check the API documentation for updates

## Next Steps

1. Get your Gemini API key from Google AI Studio
2. Add it to your `.env` file
3. Install the OpenAI package: `pip install openai`
4. Run the example code
5. Experiment with different `reasoning_effort` values
6. Compare responses with and without reasoning mode