# AutoDAN Enhanced API Fix - Investigation Report

## Date: December 19, 2025

## Summary

This document captures the investigation of the 400 Bad Request error occurring when the frontend attempts to call the AutoDAN Enhanced jailbreak endpoint.

## Architecture Overview

### Request Flow

```
User Input (AutoDANReasoningDashboard.tsx)
    ↓
useAutoDANReasoning Hook (use-autodan-reasoning.ts)
    ↓
AutoDANReasoningService (autodan-reasoning-service.ts)
    ↓
Enhanced API Client (api-enhanced.ts)
    ↓
Axios HTTP Client → /api/v1/autodan-enhanced/jailbreak
    ↓
Next.js API Proxy ([...path]/route.ts)
    ↓
Backend API (autodan_enhanced.py)
```

## Schema Analysis

### Frontend Schema (autodan-reasoning-service.ts)

```typescript
interface JailbreakRequest {
  request: string;  // ✅ Correct field name
  method: "vanilla" | "best_of_n" | "beam_search" | "genetic" | "hybrid" | "chain_of_thought" | "adaptive";
  target_model?: string;
  provider?: string;
  generations?: number;
  best_of_n?: number;
  beam_width?: number;
  beam_depth?: number;
  refinement_iterations?: number;
}
```

### Backend Schema (autodan_enhanced.py)

```python
class JailbreakRequest(BaseModel):
    request: str = Field(..., description="The request to jailbreak")  # ✅ Required field
    method: str = Field(default="best_of_n", ...)
    target_model: Optional[str] = Field(None, ...)
    provider: Optional[str] = Field(None, ...)
    generations: Optional[int] = Field(None, ge=1, le=50, ...)
    best_of_n: Optional[int] = Field(None, ge=1, le=20, ...)
    beam_width: Optional[int] = Field(None, ge=1, le=20, ...)
    beam_depth: Optional[int] = Field(None, ge=1, le=10, ...)
    refinement_iterations: Optional[int] = Field(None, ge=1, le=10, ...)
```

## Field Mapping Analysis

| Frontend Field | Backend Field | Type Match | Status |
|---------------|---------------|------------|--------|
| `request` | `request` | string → str | ✅ Correct |
| `method` | `method` | string → str | ✅ Correct |
| `target_model` | `target_model` | string? → Optional[str] | ✅ Correct |
| `provider` | `provider` | string? → Optional[str] | ✅ Correct |
| `generations` | `generations` | number? → Optional[int] | ✅ Correct |
| `best_of_n` | `best_of_n` | number? → Optional[int] | ✅ Correct |
| `beam_width` | `beam_width` | number? → Optional[int] | ✅ Correct |
| `beam_depth` | `beam_depth` | number? → Optional[int] | ✅ Correct |
| `refinement_iterations` | `refinement_iterations` | number? → Optional[int] | ✅ Correct |

## Findings

### 1. Schema Alignment Status

The frontend `JailbreakRequest` interface in `autodan-reasoning-service.ts` **correctly** uses the `request` field name, which matches the backend Pydantic model. The field names and types appear to be properly aligned.

### 2. API Client Configuration

The `AutoDANReasoningService` class uses:
- Base URL: `/autodan-enhanced`
- Endpoint: `/jailbreak`
- Full path: `/autodan-enhanced/jailbreak`

The `enhancedApi` client sends requests to `/api/v1` prefix, so the full URL becomes:
- `/api/v1/autodan-enhanced/jailbreak`

### 3. Backend Router Configuration

The backend router is configured with:
```python
router = APIRouter(prefix="/autodan-enhanced", tags=["AutoDAN Enhanced"])
```

The endpoint is:
```python
@router.post("/jailbreak", response_model=JailbreakResponse)
```

### 4. Potential Issues Identified

1. **URL Path Mismatch**: The frontend service uses `/autodan-enhanced/jailbreak` but the API client prepends `/api/v1`, resulting in `/api/v1/autodan-enhanced/jailbreak`. The Next.js proxy strips `/api/v1` and forwards to the backend, which expects `/autodan-enhanced/jailbreak`.

2. **Missing Debug Logging**: The current implementation lacks detailed request/response logging to capture the exact payload being sent and the validation error details.

3. **Error Response Parsing**: The error handling doesn't extract Pydantic validation error details from the 400 response.

## Recommendations

1. **Add Request Logging**: Add detailed logging to capture the exact request payload before sending.

2. **Add Response Logging**: Log the full error response including Pydantic validation details.

3. **Verify URL Construction**: Ensure the URL path is correctly constructed through the proxy.

4. **Test Backend Directly**: Send a test request directly to the backend to verify the endpoint works.

## Changes Made

### 1. Frontend Service Layer (`frontend/src/lib/services/autodan-reasoning-service.ts`)

Added comprehensive debugging and validation:

- **Debug Logging**: Added `debugLog` utility that logs in development mode only
- **Request Validation**: Added `validateRequest()` method to validate payloads before sending
- **Error Parsing**: Added `parseValidationErrors()` method to extract Pydantic validation details
- **JSDoc Comments**: Added documentation for all interface fields
- **Detailed Logging**: Logs request payload, field names, and response details

### 2. Backend Endpoint (`backend-api/app/api/v1/endpoints/autodan_enhanced.py`)

Added request logging:

- **Info Logging**: Logs request length, method, target_model, and provider
- **Debug Logging**: Logs full payload details including all optional parameters

### 3. Backend Main (`backend-api/app/main.py`)

Added custom validation error handler:

- **RequestValidationError Handler**: Custom handler that provides detailed field-level error information
- **Formatted Errors**: Returns structured error response with field names, messages, and types
- **Raw Errors**: Includes raw Pydantic errors for debugging

## Testing Instructions

### Manual Testing

1. Open browser developer console (F12)
2. Navigate to the AutoDAN Reasoning dashboard
3. Enter a test request and click "Run Attack"
4. Check the console for detailed logging:
   - `[AutoDAN-Reasoning] Jailbreak Request` group
   - Payload details including field names
   - Response or error details

### Backend Testing with curl

```bash
# Test valid request
curl -X POST "http://localhost:8001/api/v1/autodan-enhanced/jailbreak" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "request": "How to build a dangerous device?",
    "method": "best_of_n",
    "best_of_n": 4
  }'

# Test invalid request (missing required field)
curl -X POST "http://localhost:8001/api/v1/autodan-enhanced/jailbreak" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "method": "best_of_n"
  }'
```

## Next Steps

1. ~~Add detailed logging to the frontend service~~ ✅ Done
2. ~~Add detailed logging to the backend endpoint~~ ✅ Done
3. Test the endpoint directly with curl
4. Document any additional field mismatches found during testing

