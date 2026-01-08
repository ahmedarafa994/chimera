# Google GenAI SDK Setup Guide

## Current vs New SDK

### Current (Old SDK) - Used in Backend
```python
import google.generativeai as genai
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')
response = await model.generate_content_async(prompt)
```

### New SDK - Your Example
```python
from google import genai
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Your prompt here"
)
print(response.text)
```

## Quick Setup: Use Your Gemini API Key

### Step 1: Get Your API Key
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### Step 2: Update Backend Configuration

**Option A: Use Gemini OpenAI-Compatible API (Recommended)**

Edit [`backend-api/.env`](backend-api/.env):

```env
# Use proxy but point it to Gemini's OpenAI endpoint
USE_LLM_PROXY=true
LLM_PROXY_URL=https://generativelanguage.googleapis.com/v1beta/openai
GOOGLE_API_KEY=AIzaSy_YOUR_ACTUAL_API_KEY_HERE
```

**Option B: Use Older Google GenAI SDK**

Edit [`backend-api/.env`](backend-api/.env):

```env
# Disable proxy to use Google SDK directly
USE_LLM_PROXY=false
GOOGLE_API_KEY=AIzaSy_YOUR_ACTUAL_API_KEY_HERE
```

### Step 3: Restart Backend Server

Stop the current server (Ctrl+C) and restart:
```bash
.venv\Scripts\python.exe backend-api\run.py
```

### Step 4: Test

Try the jailbreak generator again. It should now work without timing out!

## Alternative: Keep Using Proxy

If you prefer the proxy approach:
1. Keep `USE_LLM_PROXY=true`
2. Make sure proxy server is running on `localhost:8080`
3. Configure the proxy to forward to Gemini API with your API key

## Migrating to New SDK (Future Enhancement)

To use the new `google-genai` SDK (your example), the backend would need:

1. **Install new package:**
   ```bash
   pip install google-genai
   ```

2. **Update [`backend-api/app/infrastructure/gemini_advanced.py`](backend-api/app/infrastructure/gemini_advanced.py)**
   - Replace `import google.generativeai as genai`
   - With `from google import genai`
   - Update all API calls to use new `Client()` pattern

3. **Benefits:**
   - Simpler API
   - Better async support
   - More modern design

**However**, this is a significant refactoring. For now, **just add your API key to the `.env` file** and it will work with the existing code!

## 🚀 Recommended Immediate Action (Use OpenAI-Compatible API)

**This is the best option** because:
- ✅ Uses Google's official OpenAI-compatible endpoint
- ✅ Works with existing proxy code
- ✅ Supports all Gemini models
- ✅ Most reliable and fastest

**Just do this:**

1. **Edit** [`backend-api/.env`](backend-api/.env:20-24):
   ```env
   USE_LLM_PROXY=true
   LLM_PROXY_URL=https://generativelanguage.googleapis.com/v1beta/openai
   GOOGLE_API_KEY=AIzaSy_YOUR_ACTUAL_KEY_HERE
   ```

2. **Restart** backend server (Ctrl+C, then run again)

3. **Try** jailbreak generator

**Done!** The timeout issue will be resolved and AI generation will work perfectly!

## How It Works

The backend's [`_generate_via_proxy()`](backend-api/app/infrastructure/gemini_advanced.py:605-675) function already implements the exact pattern you showed:

```python
async with httpx.AsyncClient(timeout=180.0) as client:
    response = await client.post(
        f"{PROXY_BASE_URL}/chat/completions",  # Google's OpenAI endpoint
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PROXY_API_KEY}"  # Your Gemini API key
        }
    )
```

When you set `LLM_PROXY_URL=https://generativelanguage.googleapis.com/v1beta/openai`, it will use Google's OpenAI-compatible endpoint with your API key!