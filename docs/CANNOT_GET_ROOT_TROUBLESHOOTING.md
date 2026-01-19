# Cannot GET / - Root URL Troubleshooting Guide

This comprehensive guide helps you diagnose and fix the "Cannot GET /" error when accessing the Chimera application.

## Quick Diagnosis

### What error are you seeing?

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| "Cannot GET /" on `localhost:8001` | Accessing backend directly | Go to `http://localhost:3001` instead |
| "Cannot GET /" on `localhost:3001` | Frontend not running | Start frontend: `cd frontend && npm run dev` |
| Browser shows blank page | JavaScript error | Check browser console (F12) |
| "Network Error" in browser | Backend not accessible | Verify backend: `curl http://localhost:8001/health` |
| "CORS Error" in console | Cross-origin blocked | Check backend CORS settings |
| Connection refused | Service not running | Start the appropriate service |

### Quick Verification Commands

```powershell
# Check if ports are in use
netstat -ano | findstr :3000
netstat -ano | findstr :8001

# Test backend health
curl http://localhost:8001/health

# Test frontend
curl http://localhost:3001

# Check running processes
tasklist | findstr "node python"
```

---

## Section 1: Development Mode Solutions

### Step 1: Start Backend First

```powershell
cd backend-api
py run.py
```

Wait for this message:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### Step 2: Verify Backend Health

```powershell
curl http://localhost:8001/health
```

Expected response:
```json
{"status": "healthy"}
```

If you don't get this response:
- Check for Python errors in the terminal
- Verify Python dependencies: `pip install -r requirements.txt`
- Check if port 8001 is already in use

### Step 3: Start Frontend

In a new terminal:
```powershell
cd frontend
npm run dev
```

Wait for:
```
✓ Ready in X.Xs
➜ Local: http://localhost:3001
```

### Step 4: Access the Application

Open your browser to: **http://localhost:3001**

You should be automatically redirected to `/dashboard`.

---

## Section 2: Common Development Errors

### Error: "Cannot GET /" on localhost:3001

**Causes:**
1. Frontend dev server not running
2. Next.js build error
3. Cached browser state

**Solutions:**

```powershell
# 1. Start the frontend dev server
cd frontend
npm run dev

# 2. If there are build errors, try:
npm run build

# 3. Clear Next.js cache and restart
Remove-Item -Recurse -Force .next
npm run dev

# 4. Clear browser cache (Ctrl+Shift+R in browser)
```

### Error: "Cannot GET /" on localhost:8001

**This is expected behavior!**

The backend API (FastAPI) doesn't serve HTML pages. Access the frontend at `http://localhost:3001` instead.

The backend provides:
- `http://localhost:8001/health` - Health check endpoint
- `http://localhost:8001/docs` - API documentation (Swagger)
- `http://localhost:8001/api/v1/*` - API endpoints

### Error: Port Already in Use

```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Find process using port 8001
netstat -ano | findstr :8001

# Kill process by PID (replace XXXX with actual PID)
taskkill /PID XXXX /F
```

### Error: Node.js or Python Not Found

```powershell
# Verify Node.js installation
node --version  # Should show v18.x or higher

# Verify Python installation
py --version    # Should show Python 3.10+

# If not installed, download from:
# Node.js: https://nodejs.org/
# Python: https://python.org/
```

### Error: Dependencies Not Installed

```powershell
# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd backend-api
pip install -r requirements.txt
```

---

## Section 3: Production Mode Solutions (Docker)

### Step 1: Build Production Images

```powershell
# Build with production compose file
docker-compose -f docker-compose.prod.yml build
```

### Step 2: Start Services

```powershell
docker-compose -f docker-compose.prod.yml up -d
```

### Step 3: Check Container Status

```powershell
docker ps
```

All containers should show "healthy" status.

### Step 4: View Logs

```powershell
# Frontend logs
docker logs chimera-frontend

# Backend logs
docker logs chimera-backend-api

# Follow logs in real-time
docker logs -f chimera-frontend
```

### Common Docker Issues

#### Issue: Frontend returns 404 or Cannot GET /

**Cause:** Frontend built in development mode, not production.

**Solution:** Use the production Dockerfile:
```yaml
# In docker-compose.prod.yml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile.prod  # Use production Dockerfile
```

#### Issue: "Connection refused" from frontend to backend

**Cause:** Environment variable uses Docker internal hostname.

**Solution:** Use browser-accessible URL:
```yaml
# Wrong (Docker internal):
NEXT_PUBLIC_CHIMERA_API_URL=http://backend-api:8001/api/v1

# Correct (Browser accessible):
NEXT_PUBLIC_CHIMERA_API_URL=http://localhost:8001/api/v1
```

---

## Section 4: Environment Variables

### Frontend Environment Variables

Create or update `frontend/.env.local`:

```env
# API Connection Mode (use "proxy" for development)
NEXT_PUBLIC_API_MODE=proxy

# Backend API URL (browser-accessible)
NEXT_PUBLIC_CHIMERA_API_URL=http://localhost:8001/api/v1

# API Key
NEXT_PUBLIC_CHIMERA_API_KEY=admin123
```

### Backend Environment Variables

The backend uses defaults but you can override in `backend-api/.env`:

```env
ENVIRONMENT=development
PORT=8001
HOST=0.0.0.0
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3001,http://localhost:8080
```

---

## Section 5: Verification Checklist

### Development Mode Checklist

- [ ] Python installed and accessible (`py --version`)
- [ ] Node.js installed and accessible (`node --version`)
- [ ] Backend dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Backend running on port 8001 (`curl localhost:8001/health`)
- [ ] Frontend running on port 3000
- [ ] Browser accessing `http://localhost:3001` (not 8001)
- [ ] No CORS errors in browser console
- [ ] `.env.local` configured correctly

### Production Mode Checklist

- [ ] Docker installed and running
- [ ] Production Dockerfile exists (`frontend/Dockerfile.prod`)
- [ ] Environment variables use localhost URLs (not Docker hostnames)
- [ ] All containers show "healthy" status
- [ ] No build errors in container logs
- [ ] Frontend accessible at `http://localhost:3001`
- [ ] Backend health check passes

---

## Section 6: Architecture Overview

```
┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │
│   Browser           │────▶│   Frontend          │
│   (localhost:3001)  │     │   (Next.js :3000)   │
│                     │     │                     │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                                       │ API calls
                                       │
                            ┌──────────▼──────────┐
                            │                     │
                            │   Backend           │
                            │   (FastAPI :8001)   │
                            │                     │
                            └─────────────────────┘
```

**Important:** Always access the application through the frontend (`localhost:3001`), not the backend (`localhost:8001`).

---

## Section 7: Getting Help

If you're still experiencing issues:

1. **Check the logs** - Both frontend and backend terminals show detailed errors
2. **Review browser console** - Press F12 and check for JavaScript errors
3. **Verify network requests** - In browser DevTools, check the Network tab
4. **Run health checks** - `npm run health` from the project root
5. **Check related documentation:**
   - `FRONTEND_TROUBLESHOOTING.md` - Network error details
   - `DEVELOPMENT_SETUP.md` - Initial setup instructions
   - `FULLSTACK_INTEGRATION_GUIDE.md` - Architecture overview
