# Chimera Deployment Checklist

Use this checklist to ensure successful deployment of the Chimera application.

---

## Pre-Deployment Checklist

### Environment Configuration

- [ ] **Environment variables configured correctly**
  - `NEXT_PUBLIC_API_MODE` set to `proxy`
  - `NEXT_PUBLIC_CHIMERA_API_URL` points to correct backend URL
  - `NEXT_PUBLIC_CHIMERA_API_KEY` is set and secured
  - Production URLs use `https://` for security

- [ ] **API keys secured**
  - API keys not committed to version control
  - Production keys different from development
  - Keys stored in secure environment variables or secrets manager

- [ ] **CORS origins configured**
  - Production domain added to `ALLOWED_ORIGINS`
  - No wildcard (`*`) in production
  - Include both `http` and `https` if needed

### Build Verification

- [ ] **Frontend builds successfully**
  ```bash
  cd frontend
  npm run build
  ```
  - No TypeScript errors
  - No missing dependencies
  - Build output created in `.next` directory

- [ ] **Backend starts without errors**
  ```bash
  cd backend-api
  py run.py
  ```
  - No import errors
  - All dependencies installed
  - Configuration validated

- [ ] **Docker images build successfully**
  ```bash
  docker-compose -f docker-compose.prod.yml build
  ```
  - All stages complete
  - No build errors
  - Images tagged correctly

### Security Review

- [ ] **Debug mode disabled in production**
  - `NEXT_PUBLIC_ENABLE_DEBUG=false`
  - `ENVIRONMENT=production`
  - Swagger docs disabled (`/docs` returns 404)

- [ ] **Secure headers configured**
  - HSTS enabled
  - X-Frame-Options set
  - Content-Security-Policy configured

- [ ] **Rate limiting enabled**
  - API rate limits configured
  - Abuse prevention active

---

## Deployment Steps

### Step 1: Build Production Images

```bash
# Pull latest code
git pull origin main

# Build Docker images
docker-compose -f docker-compose.prod.yml build --no-cache
```

### Step 2: Start Services

```bash
# Start in detached mode
docker-compose -f docker-compose.prod.yml up -d

# Check container status
docker ps
```

### Step 3: Verify Health Endpoints

```bash
# Backend health
curl http://localhost:8001/health

# Expected: {"status":"healthy"}
```

### Step 4: Run Smoke Tests

```bash
# Run automated smoke tests
./tests/smoke-test.sh

# Or manually verify:
curl -I http://localhost:3000
curl -I http://localhost:3000/dashboard
```

---

## Post-Deployment Checklist

### Functionality Verification

- [ ] **Root URL loads correctly**
  - `http://localhost:3000` returns 200
  - Redirects to `/dashboard`
  - No JavaScript errors in console

- [ ] **Dashboard accessible**
  - `/dashboard` route works
  - Page renders correctly
  - Navigation functions properly

- [ ] **API calls succeed**
  - Network requests complete
  - Expected data returned
  - No CORS errors

- [ ] **No console errors**
  - Browser console is clean
  - No unhandled exceptions
  - No failed network requests

- [ ] **Health endpoints return 200**
  - Backend: `GET /health` → 200
  - Frontend: `GET /` → 200

### Performance Verification

- [ ] **Page load times acceptable**
  - Initial load < 3 seconds
  - Subsequent navigation < 1 second
  - API responses < 500ms

- [ ] **Resource usage normal**
  - Memory usage stable
  - CPU usage reasonable
  - No memory leaks

### Monitoring Setup

- [ ] **Logging configured**
  - Application logs flowing
  - Error tracking active
  - Log retention configured

- [ ] **Alerting configured**
  - Health check alerts set up
  - Error rate alerts configured
  - Resource usage alerts active

---

## Rollback Procedure

If deployment fails:

```bash
# Stop current deployment
docker-compose -f docker-compose.prod.yml down

# Rollback to previous version
git checkout <previous-commit>

# Rebuild and redeploy
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

---

## Quick Reference Commands

```bash
# View logs
docker logs chimera-frontend
docker logs chimera-backend-api

# Restart services
docker-compose -f docker-compose.prod.yml restart

# Stop services
docker-compose -f docker-compose.prod.yml down

# Check resource usage
docker stats

# Enter container shell
docker exec -it chimera-frontend sh
docker exec -it chimera-backend-api bash
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Container won't start | Check `docker logs <container>` |
| Health check failing | Verify environment variables |
| CORS errors | Update `ALLOWED_ORIGINS` |
| 404 on routes | Check Next.js build output |
| Slow performance | Review resource limits |

For detailed troubleshooting, see `CANNOT_GET_ROOT_TROUBLESHOOTING.md`.
