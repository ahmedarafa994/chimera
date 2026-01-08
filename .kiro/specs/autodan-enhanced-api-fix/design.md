# Design Document

## Overview

This design addresses the 400 Bad Request error in the AutoDAN Enhanced API integration by identifying and fixing schema mismatches between the frontend and backend. The root cause is that the frontend service is sending incorrect field names that don't match the backend's Pydantic model expectations.

## Architecture

The system follows a three-tier architecture:

1. **Frontend Service Layer** (`frontend/src/lib/services/autodan-reasoning-service.ts`)
   - Constructs API requests with proper field names
   - Handles response transformation
   - Provides type-safe interfaces

2. **API Client Layer** (`frontend/src/lib/api-enhanced.ts`)
   - Manages HTTP communication
   - Handles authentication and headers
   - Provides retry and error handling

3. **Backend API Layer** (`backend-api/app/api/v1/endpoints/autodan_enhanced.py`)
   - Validates incoming requests using Pydantic
   - Processes jailbreak generation
   - Returns structured responses

## Components and Interfaces

### Frontend Service Interface

```typescript
interface JailbreakRequest {
  request: string;  // CRITICAL: Must be "request", not "prompt"
  method: "vanilla" | "best_of_n" | "beam_search" | "genetic" | "hybrid" | "chain_of_thought" | "adaptive";
  target_model?: string;
  provider?: string;
  generations?: number;
  best_of_n?: number;
  beam_width?: number;
  beam_depth?: number;
  refinement_iterations?: number;
}

interface JailbreakResponse {
  jailbreak_prompt: string;
  method: string;
  status: string;
  score?: number;
  iterations?: number;
  latency_ms: number;
  cached: boolean;
  model_used?: string;
  provider_used?: string;
  error?: string;
}
```

### Backend API Schema

```python
class JailbreakRequest(BaseModel):
    request: str = Field(..., description="The request to jailbreak")
    method: str = Field(default="best_of_n", ...)
    target_model: Optional[str] = Field(None, ...)
    provider: Optional[str] = Field(None, ...)
    generations: Optional[int] = Field(None, ge=1, le=50, ...)
    best_of_n: Optional[int] = Field(None, ge=1, le=20, ...)
    beam_width: Optional[int] = Field(None, ge=1, le=20, ...)
    beam_depth: Optional[int] = Field(None, ge=1, le=10, ...)
    refinement_iterations: Optional[int] = Field(None, ge=1, le=10, ...)
```

## Data Models

### Request Flow

```
User Input → Frontend Component → Service Layer → API Client → Next.js Proxy → Backend API
```

### Field Mapping

The critical issue is that field names must match exactly:

| Frontend Field | Backend Field | Status |
|---------------|---------------|---------|
| `request` | `request` | ✅ Correct |
| `method` | `method` | ✅ Correct |
| `target_model` | `target_model` | ✅ Correct |
| `provider` | `provider` | ✅ Correct |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Request Field Name Consistency

*For any* jailbreak request sent from the frontend, the payload MUST contain a field named `request` (not `prompt` or any other name), and this field MUST be a non-empty string.

**Validates: Requirements 2.1, 2.2**

### Property 2: Schema Validation Success

*For any* valid jailbreak request constructed by the frontend service, when sent to the backend API, the request MUST pass Pydantic validation without returning a 400 error.

**Validates: Requirements 1.4, 2.4**

### Property 3: Optional Parameter Handling

*For any* optional parameter (generations, best_of_n, beam_width, beam_depth, refinement_iterations), if provided by the frontend, it MUST be within the valid range defined by the backend schema (e.g., generations: 1-50, best_of_n: 1-20).

**Validates: Requirements 2.2, 2.4**

### Property 4: Error Message Clarity

*For any* validation error returned by the backend, the error response MUST include specific field names and validation constraints that failed, enabling developers to quickly identify the issue.

**Validates: Requirements 3.1, 3.2, 3.3**

### Property 5: Type Safety at Compile Time

*For any* request constructed using the TypeScript JailbreakRequest interface, the TypeScript compiler MUST prevent assigning incorrect field types (e.g., string to number field) before runtime.

**Validates: Requirements 4.1, 4.4, 4.5**

## Error Handling

### Validation Errors (400)

When the backend returns a 400 error:
1. Extract the Pydantic validation error details
2. Log the full request payload that was rejected
3. Display user-friendly error message
4. Provide actionable guidance for fixing the issue

### Network Errors (503)

When the backend is unreachable:
1. Display connection error message
2. Suggest checking if backend is running
3. Provide fallback behavior (if applicable)

### Authentication Errors (401)

When API key is missing or invalid:
1. Display authentication error
2. Guide user to configure API key
3. Prevent further requests until configured

## Testing Strategy

### Unit Tests

1. **Service Layer Tests**
   - Test request payload construction with correct field names
   - Test optional parameter handling
   - Test error response parsing

2. **API Client Tests**
   - Test HTTP request formatting
   - Test header injection
   - Test error handling

3. **Backend Validation Tests**
   - Test Pydantic model validation with valid payloads
   - Test validation error messages for invalid payloads
   - Test field name case sensitivity

### Property-Based Tests

1. **Property Test: Request Field Validation**
   - Generate random valid jailbreak requests
   - Verify all requests contain `request` field
   - Verify all requests pass backend validation
   - **Feature: autodan-enhanced-api-fix, Property 1: Request Field Name Consistency**

2. **Property Test: Optional Parameter Ranges**
   - Generate random optional parameters within valid ranges
   - Verify backend accepts all valid values
   - Verify backend rejects out-of-range values
   - **Feature: autodan-enhanced-api-fix, Property 3: Optional Parameter Handling**

3. **Property Test: Type Safety**
   - Attempt to construct requests with incorrect types
   - Verify TypeScript compiler catches type errors
   - **Feature: autodan-enhanced-api-fix, Property 5: Type Safety at Compile Time**

### Integration Tests

1. **End-to-End Request Flow**
   - Send request from frontend component
   - Verify request reaches backend successfully
   - Verify response is parsed correctly
   - Verify UI updates with response data

2. **Error Scenario Testing**
   - Test 400 error handling
   - Test 503 error handling
   - Test 401 error handling
   - Verify error messages are displayed correctly

### Manual Testing

1. Open browser developer console
2. Trigger jailbreak generation from UI
3. Verify request payload in Network tab
4. Verify no 400 errors occur
5. Verify response is displayed correctly

## Implementation Notes

### Critical Fix

The primary issue is in `frontend/src/lib/services/autodan-reasoning-service.ts`. The service must ensure the request payload uses `request` as the field name, matching the backend's Pydantic model.

### Debugging Support

Add comprehensive logging:
- Log request payload before sending
- Log response status and body
- Log validation errors with full details
- Use conditional logging (development only)

### Type Definitions

Ensure TypeScript interfaces exactly match Pydantic models:
- Field names must match (case-sensitive)
- Optional fields must use `?` in TypeScript
- Field types must align (string, number, boolean)
- Enum values must match exactly
