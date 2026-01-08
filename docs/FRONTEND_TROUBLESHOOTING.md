# Frontend Network Error Troubleshooting Guide

## Quick Links

- **"Cannot GET /" Error?** → See [CANNOT_GET_ROOT_TROUBLESHOOTING.md](./CANNOT_GET_ROOT_TROUBLESHOOTING.md) for detailed solutions.

---

## "Cannot GET /" Error - Quick Reference

### Error Diagnosis Flowchart

```
Start: You see "Cannot GET /"
       │
       ├── Accessing localhost:8001?
       │   └── Yes → Backend doesn't serve HTML
       │             Go to http://localhost:3000 instead
       │
       ├── Accessing localhost:3000?
       │   ├── Is frontend dev server running?
       │   │   └── No → Run: cd frontend && npm run dev
       │   │
       │   ├── Is Next.js build broken?
       │   │   └── Check terminal for build errors
       │   │
       │   └── Using Docker?
       │       └── Check: docker logs chimera-frontend
       │
       └── Not sure what's wrong?
           └── See CANNOT_GET_ROOT_TROUBLESHOOTING.md
```

### Quick Fix Commands

```powershell
# Development: Start both services
cd backend-api; py run.py
# (in new terminal)
cd frontend; npm run dev
# Access: http://localhost:3000

# Docker: Rebuild and restart
docker-compose down
docker-compose up --build -d
# Check: docker logs chimera-frontend
```

---

## Issue
Frontend displays "Network Error" when trying to connect to the backend API, even after CORS and authentication fixes have been applied.

## Root Causes
1. **Browser cache** - Old failed requests are cached
2. **Service Worker** - Next.js development server service worker holding stale connections
3. **localStorage** - Old API configuration stored
4. **Hot Module Replacement** - Dev server needs full restart

## Solution Steps

### Step 1: Clear Browser Cache
1. Open DevTools (F12 or Right-click → Inspect)
2. Go to **Application** tab
3. Clear the following:
   - **Local Storage** → Clear all `localhost:3000` entries
   - **Session Storage** → Clear all
   - **Cookies** → Clear all for `localhost:3000`
   - **Cache Storage** → Delete all caches

### Step 2: Hard Refresh Browser
- **Windows/Linux**: `Ctrl + Shift + R` or `Ctrl + F5`
- **Mac**: `Cmd + Shift + R`

### Step 3: Restart Frontend Development Server
```bash
# Stop the current dev server (Ctrl+C in terminal)
cd frontend
npm run dev
```

### Step 4: Verify Backend is Running
Check that the backend server is running on port 8001:
```bash
curl http://localhost:8001/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "...",
  "version": "1.0.0",
  "environment": "development",
  "services": {
    "llm_service": "available",
    "transformation_service": "available"
  }
}
```

### Step 5: Test CORS Manually
```bash
curl -X OPTIONS http://localhost:8001/api/v1/generation/jailbreak/generate \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -v
```

Look for these headers in the response:
- `access-control-allow-origin: http://localhost:3000`
- `access-control-allow-methods: ...POST...`
- `access-control-allow-credentials: true`

### Step 6: Check Frontend API Configuration
Open browser console and run:
```javascript
localStorage.getItem('chimera_api_config')
```

Should show:
```json
{
  "mode": "proxy",
  "proxyUrl": "http://localhost:8001/api/v1",
  "proxyApiKey": "admin123",
  ...
}
```

If it shows `"chimera"` mode or wrong URL, clear it:
```javascript
localStorage.removeItem('chimera_api_config')
```

Then refresh the page.

### Step 7: Verify Network Request
1. Open DevTools → **Network** tab
2. Try generating a jailbreak
3. Look for the request to `/api/v1/generation/jailbreak/generate`
4. Check:
   - **Request URL**: Should be `http://localhost:8001/api/v1/generation/jailbreak/generate`
   - **Request Method**: Should be `POST`
   - **Status Code**: Should be `200 OK` (not 401, 403, or CORS error)
   - **Response Headers**: Should include CORS headers

### Step 8: Check Console for Errors
Look in the browser console for any of these errors:
- ❌ `CORS policy` errors → Backend CORS not configured correctly
- ❌ `401 Unauthorized` → Endpoint needs to be added to excluded paths
- ❌ `Network Error` → Frontend can't reach backend
- ❌ `ERR_CONNECTION_REFUSED` → Backend not running

## Common Issues & Fixes

### Issue: "CORS policy: No 'Access-Control-Allow-Origin' header"
**Fix**: Backend CORS middleware order issue
- CORS middleware must be added **first** in `backend-api/app/main.py`
- Check line 21-62 to ensure CORS is before other middleware

### Issue: "401 Unauthorized"
**Fix**: Endpoint not in excluded paths
- Add endpoint to `excluded_paths` in `backend-api/app/main.py` line 70-87

### Issue: "Network Error" or "ERR_NETWORK"
**Fix**: Multiple possible causes
1. Backend not running → Start with `.venv\Scripts\python.exe backend-api\run.py`
2. Wrong URL in frontend → Check `api-config.ts` (should be port 8001)
3. Firewall blocking → Allow localhost connections
4. Browser cache → Clear and hard refresh

### Issue: "422 Unprocessable Entity"
**Fix**: Mode mismatch
- Frontend sending invalid `mode` value
- Should be `"proxy"` or `"direct"`, not `"chimera"`
- Fixed in `frontend/src/lib/api-config.ts`

## Verification Checklist

✅ Backend running on port 8001
✅ Frontend running on port 3000
✅ CORS configured correctly (check backend logs)
✅ Endpoint in excluded paths (no auth required)
✅ Browser cache cleared
✅ localStorage cleared
✅ Hard refresh performed
✅ Network tab shows correct request URL
✅ No console errors

## Still Not Working?

### Option 1: Nuclear Option - Complete Reset
```bash
# Stop both servers
# Clear all browser data
# Delete node_modules and reinstall
cd frontend
rm -rf node_modules .next
npm install
npm run dev

# Restart backend
cd ../
.venv\Scripts\python.exe backend-api\run.py
```

### Option 2: Check Firewall/Antivirus
Some security software blocks localhost connections:
- Temporarily disable antivirus
- Check Windows Firewall settings
- Try accessing from different browser

### Option 3: Use Different Port
If port 8001 is problematic:
1. Change backend port in `backend-api/run.py`
2. Update `NEXT_PUBLIC_CHIMERA_API_URL` in `frontend/.env.local`
3. Restart both servers

## Environment Files

### Backend `.env`
```env
CHIMERA_API_KEY=admin123
ENVIRONMENT=development
API_CONNECTION_MODE=proxy
```

### Frontend `.env.local`
```env
NEXT_PUBLIC_API_MODE=proxy
NEXT_PUBLIC_CHIMERA_API_URL=http://localhost:8001/api/v1
NEXT_PUBLIC_CHIMERA_API_KEY=admin123
```

## Success Indicators

When everything works, you should see:
1. ✅ Frontend loads without errors
2. ✅ Connection status shows "Connected"
3. ✅ Can generate jailbreaks successfully
4. ✅ Network tab shows 200 responses
5. ✅ No CORS errors in console
6. ✅ No authentication errors

## Contact Points

If still experiencing issues:
- Check backend logs in terminal
- Check browser console for errors
- Verify network requests in DevTools
- Ensure both servers are running