# Troubleshooting Guide - Project Chimera Frontend

## "Failed to fetch" TypeError during HMR

This error typically occurs when the frontend cannot connect to the backend API server. Here's a systematic debugging approach:

### Quick Fixes

1. **Ensure the backend is running:**
   ```bash
   # From the backend-api directory
   cd backend-api
   python run.py
   ```
   The backend should be running on `http://localhost:8000`

2. **Clear Next.js cache and restart:**
   ```bash
   # From the frontend directory
   cd frontend
   rm -rf .next
   rm -rf node_modules/.cache
   npm run dev
   ```

3. **Check if port 8000 is available:**
   ```bash
   # Windows
   netstat -ano | findstr :8000

   # Linux/Mac
   lsof -i :8000
   ```

### Common Issues

#### 1. Backend Server Not Running
**Symptoms:** "Cannot connect to backend server" toast messages
**Solution:** Start the backend server:
```bash
cd backend-api
pip install -r requirements.txt
python run.py
```

#### 2. Wrong API Port Configuration
**Symptoms:** Network errors, 404 responses
**Solution:** The backend runs on port 8000. Check your `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

#### 3. CORS Issues
**Symptoms:** CORS errors in browser console
**Solution:** The backend should have CORS configured. Check `backend-api/.env`:
```env
BACKEND_CORS_ORIGINS=["http://localhost:3000"]
```

#### 4. Turbopack HMR WebSocket Issues
**Symptoms:** HMR not working, page not refreshing
**Solution:**
- Try using Webpack instead of Turbopack:
  ```bash
  npm run dev -- --no-turbo
  ```
- Or clear the cache:
  ```bash
  rm -rf .next
  npm run dev
  ```

#### 5. Port Conflicts
**Symptoms:** "Address already in use" errors
**Solution:**
```bash
# Kill process on port 3000 (frontend)
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac
kill -9 $(lsof -t -i:3000)

# Kill process on port 8000 (backend)
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
kill -9 $(lsof -t -i:8000)
```

### Debugging Steps

1. **Check browser DevTools Network tab:**
   - Look for failed requests (red entries)
   - Check the status codes (404, 500, CORS errors)
   - Verify the request URL is correct (`localhost:8000/api/v1/...`)

2. **Test API directly:**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Providers list
   curl http://localhost:8000/api/v1/providers

   # Techniques list
   curl http://localhost:8000/api/v1/techniques
   ```

3. **Check terminal output:**
   - Frontend terminal: Look for compilation errors
   - Backend terminal: Look for request logs and errors

4. **Verify environment variables:**
   ```bash
   # Create .env.local if it doesn't exist
   cp .env.local.example .env.local
   ```

### Clean Restart Procedure

```bash
# 1. Stop all running processes (Ctrl+C in terminals)

# 2. Clean frontend
cd frontend
rm -rf .next
rm -rf node_modules/.cache

# 3. Clean backend (optional)
cd ../backend-api
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 4. Start backend first
python run.py

# 5. In a new terminal, start frontend
cd ../frontend
npm run dev
```

### Turbopack vs Webpack

Next.js 16 uses Turbopack by default. If you experience HMR issues:

**Turbopack (default):**
- Faster compilation
- May have occasional HMR issues
- Use: `npm run dev`

**Webpack (fallback):**
- More stable HMR
- Slower compilation
- Use: `npm run dev -- --no-turbo`

### API Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/providers` | GET | List providers |
| `/api/v1/techniques` | GET | List techniques |
| `/api/v1/transform` | POST | Transform prompt |
| `/api/v1/execute` | POST | Transform + Execute |
| `/api/v1/generate` | POST | Direct LLM generation |
| `/api/v1/generation/jailbreak/generate` | POST | Jailbreak generation |
| `/api/v1/metrics` | GET | System metrics |

### Still Having Issues?

1. Check the backend logs for errors
2. Verify all dependencies are installed:
   ```bash
   # Frontend
   cd frontend && npm install

   # Backend
   cd backend-api && pip install -r requirements.txt
   ```
3. Try a different browser or incognito mode
4. Check firewall settings aren't blocking localhost connections