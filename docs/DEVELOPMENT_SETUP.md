# Development Environment Setup Guide

## Problem Diagnosis

Your React frontend at `http://localhost:3001` is trying to connect to the FastAPI backend at `http://localhost:8001/api/v1/gptfuzz/run`, but the backend server is **NOT RUNNING**, causing `ERR_CONNECTION_REFUSED`.

## Quick Fix - Start Both Servers

### Option 1: Separate Terminals (Recommended)

**Terminal 1 - Backend:**
```bash
cd backend-api
python run.py
```

If `python` doesn't work, try:
```bash
py run.py
# or
python3 run.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Expected output:**
```
> frontend@0.1.0 dev
> next dev
  ▲ Next.js 16.0.6
  - Local:        http://localhost:3001
```

### Option 2: Using Concurrently (if npm start fails)

The `npm start` command should run both, but it's currently failing to start the backend on Windows.

**To fix:** Stop the current terminal (Ctrl+C) and restart:
```bash
npm start
```

You should see BOTH:
- `[BACKEND]` logs showing Uvicorn starting
- `[FRONTEND]` logs showing Next.js compilation

## Verification Steps

### 1. Check Backend is Running
Open browser to: `http://localhost:8001/docs`

You should see the **Swagger UI** with all API endpoints.

### 2. Test Backend Health
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "security": "enabled",
  "features": {...}
}
```

### 3. Check Frontend Connection
Open: `http://localhost:3001/dashboard`

The fuzzing dashboard should load without network errors.

## Common Issues & Solutions

### Issue 1: Backend Not Starting

**Symptom:** Only see `[FRONTEND]` logs, no `[BACKEND]`

**Solutions:**
1. Python not in PATH - Add Python to Windows PATH or use full path
2. Wrong Python command - Try `python`, `python3`, or `py`
3. Missing dependencies - Run: `cd backend-api && pip install -r requirements.txt`
4. Port 8001 in use - Check: `netstat -ano | findstr :8001`

### Issue 2: Connection Refused

**Symptom:** `net::ERR_CONNECTION_REFUSED` in browser console

**Solution:** Backend is not running. Start it using Terminal 1 instructions above.

### Issue 3: CORS Errors

**Symptom:** Backend starts but frontend shows CORS errors

**Solution:** Already fixed in `backend-api/app/main.py` - CORS allows `http://localhost:3001`

### Issue 4: Missing Image (01.png 404)

**Symptom:** `GET /avatars/01.png 404`

**Solution:** This is cosmetic. Create the file or update component to remove reference.

## Backend Configuration

### Port Configuration
Set in `backend-api/run.py`:
```python
port = int(os.getenv("PORT", 8001))
```

### CORS Configuration
Set in `backend-api/app/main.py`:
```python
cors_origins = [
    "http://localhost:3001",
    "http://localhost:3001",
    ...
]
```

## Frontend Configuration

### API Base URL
Set in `frontend/src/lib/api.ts`:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001/api/v1";
```

### Environment Variables (Optional)
Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8001/api/v1
```

## LLM Configuration (Antigravity)

Your fuzzing config shows:
- **Provider:** antigravity
- **Model:** gemini-claude-sonnet-4-5
- **Endpoint:** http://localhost:8080/gemini-antigravity/v1
- **API Key:** admin123

**IMPORTANT:** The antigravity endpoint at `localhost:8080` must also be running for fuzzing to work!

To start the antigravity service (if you have it):
```bash
# Check if it's running
curl http://localhost:8080/v1/models

# If not, start your AIClient-2-API server on port 8080
```

## Full Stack Startup Checklist

- [ ] Backend dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend dependencies installed (`npm install` in frontend/)
- [ ] Backend running on port 8001 (verify at /docs)
- [ ] Frontend running on port 3000
- [ ] Antigravity/LLM proxy running on port 8080 (if using)
- [ ] Browser can access http://localhost:3001/dashboard
- [ ] No CORS errors in console
- [ ] No connection refused errors

## Troubleshooting Commands

```bash
# Check what's running on each port
netstat -ano | findstr :3000
netstat -ano | findstr :8001
netstat -ano | findstr :8080

# Kill a process on Windows (use PID from netstat)
taskkill /PID <process_id> /F

# Check Python version
python --version
py --version

# Check Node version
node --version
npm --version

# Reinstall dependencies
cd backend-api && pip install -r requirements.txt
cd frontend && npm install
```

## Success Indicators

When everything is working correctly, you should see:

1. **Backend Terminal:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
[BACKEND] All LLM providers registered via AIClient-2-API Server
```

2. **Frontend Terminal:**
```
[FRONTEND] ✓ Compiled in 144ms
[FRONTEND] GET /dashboard 200 in 75ms
```

3. **Browser Console:** No network errors

4. **API Docs:** http://localhost:8001/docs loads successfully