# API Endpoint Synchronization Guide

## Backend-Frontend Endpoint Mapping (Fixed)

This document provides the corrected endpoint mappings between frontend and backend after synchronization fixes.

### Core Generation Endpoints

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/generate` | `/api/v1/generate` | ✅ Fixed | Main generation endpoint |
| `/transform` | `/api/v1/transform` | ✅ Fixed | Prompt transformation |
| `/execute` | `/api/v1/execute` | ✅ Fixed | Transform + execute |

### Jailbreak Endpoints

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/jailbreak` | `/api/v1/jailbreak` | ✅ Fixed | Added alias for compatibility |
| `/generation/jailbreak/generate` | `/api/v1/generation/jailbreak/generate` | ✅ Fixed | Primary endpoint |

### Provider Management

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/providers` | `/api/v1/providers` | ✅ Fixed | List providers |
| `/provider-config/*` | `/api/v1/provider-config/*` | ✅ Fixed | Provider configuration |
| `/provider-sync/*` | `/api/v1/provider-sync/*` | ✅ Fixed | Provider synchronization |
| `/models` | `/api/v1/models` | ✅ Fixed | Model listing |

### Technique Endpoints

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/techniques` | `/api/v1/techniques` | ✅ Fixed | Added alias to utils endpoint |
| `/techniques/{name}` | `/api/v1/techniques/{name}` | ✅ Fixed | Technique details |

### AutoDAN Endpoints

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/autodan-turbo/*` | `/api/v1/autodan-turbo/*` | ✅ Fixed | AutoDAN-Turbo endpoints |
| `/autodan/*` | `/api/v1/autodan/*` | ✅ Fixed | Legacy AutoDAN |

### Health and Utility Endpoints

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/health` | `/api/health` | ✅ Fixed | Health check |
| `/metrics` | `/api/v1/metrics` | ✅ Fixed | System metrics |

### Session Management

| Frontend Call | Backend Endpoint | Status | Notes |
|--------------|------------------|--------|--------|
| `/session/*` | `/api/v1/session/*` | ✅ Fixed | Session endpoints |

## Frontend API Client Updates Required

### 1. ChimeraAPI.ts Updates
- ✅ Provider sync endpoints correctly mapped to `/api/v1/provider-sync/*`
- ✅ Models endpoint using `/api/v1/models`

### 2. API-Enhanced.ts Updates
- ✅ Jailbreak endpoint using `/api/v1/jailbreak`
- ✅ Techniques endpoint using `/api/v1/techniques`
- ✅ Provider config endpoints using `/api/v1/provider-config/*`

## Backend Fixes Applied

### 1. Missing Import Fixed
- ✅ Added `aegis_ws` import to `backend-api/app/api/v1/api.py`

### 2. Endpoint Aliases Added
- ✅ Added `/api/v1/jailbreak` alias in `api_routes.py`
- ✅ Added `/api/v1/techniques` alias in `api_routes.py`

### 3. Router Configuration
- ✅ All v1 endpoints properly included in router
- ✅ Provider sync endpoints available at correct paths

## Error Handling Improvements

### Backend
- ✅ Consistent error response format across all endpoints
- ✅ Proper HTTP status codes
- ✅ Detailed error messages with context

### Frontend
- ⚠️  Need to update error handling to match backend format
- ⚠️  Need to add circuit breaker patterns for reliability

## Testing Requirements

### Unit Tests
- [ ] Test all endpoint aliases work correctly
- [ ] Test error handling consistency
- [ ] Test provider sync functionality

### Integration Tests
- [ ] Test full frontend-backend communication
- [ ] Test WebSocket connections
- [ ] Test model selection propagation

### E2E Tests
- [ ] Test complete user workflows
- [ ] Test error recovery scenarios
- [ ] Test provider switching

## Next Steps

1. **Frontend Updates** - Update remaining API clients to use correct endpoints
2. **Error Handling** - Standardize error handling across frontend
3. **Testing** - Comprehensive endpoint testing
4. **Documentation** - Update API documentation

## Configuration Notes

### Backend Configuration
```python
# In app/core/config.py
API_VERSION = "/api/v1"
ENABLE_ENDPOINT_ALIASES = True
```

### Frontend Configuration
```typescript
// In lib/api-config.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
const API_VERSION = '/api/v1';
```

## Deployment Checklist

- [ ] Backend endpoint aliases deployed
- [ ] Frontend API clients updated
- [ ] Error handling standardized
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Monitoring configured

---

**Last Updated**: 2026-01-18
**Status**: In Progress - Core fixes applied, testing remaining