# Frontend Troubleshooting Guide

This guide helps resolve common development issues with the Chimera frontend.

## Port Configuration Issues

### Problem: Authentication 404 Errors
**Symptoms:**
- Console shows `POST http://localhost:8003/auth/login 404 (Not Found)`
- Login attempts fail immediately
- Frontend can't connect to backend APIs

**Root Cause:**
Environment variables pointing to wrong backend port.

**Solution:**
1. Check `.env.local` configuration:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8001  # CORRECT
   NEXT_PUBLIC_WS_URL=ws://localhost:8001     # CORRECT
   ```

2. Verify backend is running on correct port:
   ```bash
   # Check if backend is running on port 8001 (correct)
   netstat -ano | findstr :8001

   # Test health endpoint
   curl http://localhost:8001/health
   ```

3. Restart frontend dev server after changing environment variables:
   ```bash
   npm run dev
   ```

## Service Health Issues

### Problem: "Service Unavailable" Messages
**Symptoms:**
- AuthProvider shows "Authentication service is currently unavailable"
- Console shows network errors or timeouts

**Diagnosis:**
1. Check backend health:
   ```bash
   curl -v http://localhost:8001/health
   ```

2. Check backend logs for errors
3. Verify environment variables are correct

**Solution:**
1. Restart backend service:
   ```bash
   cd backend-api
   python run.py
   ```

2. Clear browser cache and local storage
3. Restart frontend dev server

## Browser Extension Conflicts

### Problem: JavaScript Runtime Errors
**Symptoms:**
- Console shows `copilot.b68e6a51.js:15 Uncaught TypeError: v[b] is not a function`
- Multiple browser extension related errors

**Solution:**
1. **Temporary Fix:** Disable browser extensions during development
2. **Permanent Fix:** Use browser profiles for development:
   ```bash
   # Chrome with clean profile
   chrome --user-data-dir=/tmp/chrome-dev --disable-extensions

   # Firefox with clean profile
   firefox -CreateProfile dev
   firefox -P dev
   ```

## WebSocket Connection Issues

### Problem: WebSocket Connection Failures
**Symptoms:**
- Console shows `WebSocket connection to 'ws://localhost:8081/' failed`
- Real-time features not working

**Solution:**
1. Check WebSocket URL configuration:
   ```bash
   # Should be port 8001, not 8081 or 8003
   NEXT_PUBLIC_WS_URL=ws://localhost:8001
   ```

2. Verify backend WebSocket endpoints are working
3. Check for firewall or proxy issues

## Development Environment Setup

### Quick Health Check Script
Save this as `scripts/health-check.sh`:

```bash
#!/bin/bash
echo "=== Chimera Frontend Health Check ==="

echo "1. Checking backend health..."
if curl -f -s http://localhost:8001/health > /dev/null; then
    echo "✅ Backend health OK"
else
    echo "❌ Backend health FAILED"
    exit 1
fi

echo "2. Checking API endpoints..."
if curl -f -s http://localhost:8001/api/v1/health > /dev/null; then
    echo "✅ API endpoints OK"
else
    echo "❌ API endpoints FAILED"
    exit 1
fi

echo "3. Checking environment variables..."
if [ "$NEXT_PUBLIC_API_URL" = "http://localhost:8001" ]; then
    echo "✅ Environment variables OK"
else
    echo "❌ Environment variables incorrect"
    echo "Expected: http://localhost:8001"
    echo "Actual: $NEXT_PUBLIC_API_URL"
    exit 1
fi

echo "=== All checks passed! ==="
```

### Quick Fix Script
Save this as `scripts/fix-config.sh`:

```bash
#!/bin/bash
echo "=== Applying Chimera Frontend Fixes ==="

echo "1. Updating environment variables..."
sed -i 's/localhost:8003/localhost:8001/g' frontend/.env.local

echo "2. Clearing Next.js cache..."
rm -rf frontend/.next

echo "3. Restarting dev server..."
cd frontend && npm run dev

echo "=== Configuration fixed! ==="
```

## Common Port Conflicts

| Service | Port | Purpose |
|---------|------|---------|
| Frontend Dev Server | 3000 | Next.js development server |
| Backend API | 8001 | Chimera FastAPI backend (CORRECT) |
| Provider Sync | 8003 | Real-time provider synchronization |
| External Proxy | 8080 | Optional external API proxy |

## Getting Help

1. Check the main [README.md](../README.md) for setup instructions
2. Review [CLAUDE.md](../CLAUDE.md) for architecture details
3. Check recent git commits for configuration changes
4. Ask team members about recent infrastructure changes

## Preventive Measures

1. **Use environment validation:**
   - Add startup checks for required environment variables
   - Validate port availability before starting services

2. **Automate health checks:**
   - Add pre-commit hooks to validate configuration
   - Include health checks in CI/CD pipeline

3. **Document port assignments:**
   - Keep port assignments documented and centralized
   - Use port discovery instead of hardcoded ports where possible