# Backend Pydantic Model Review

## Task 4.1: Review Backend Pydantic Model

**Date:** December 19, 2025  
**Status:** ✅ Complete

## Backend JailbreakRequest Model

**Location:** `backend-api/app/api/v1/endpoints/autodan_enhanced.py`

### Field Definitions

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `request` | `str` | ✅ Yes | None | The request to jailbreak |
| `method` | `str` | No (default: "best_of_n") | None | Optimization method |
| `target_model` | `Optional[str]` | No | None | Target LLM model |
| `provider` | `Optional[str]` | No | None | LLM provider |
| `generations` | `Optional[int]` | No | ge=1, le=50 | Number of generations (genetic) |
| `best_of_n` | `Optional[int]` | No | ge=1, le=20 | N candidates (best_of_n) |
| `beam_width` | `Optional[int]` | No | ge=1, le=20 | Beam width (beam_search) |
| `beam_depth` | `Optional[int]` | No | ge=1, le=10 | Beam depth (beam_search) |
| `refinement_iterations` | `Optional[int]` | No | ge=1, le=10 | Refinement cycles (adaptive/cot) |

### Valid Method Values

- `vanilla` - Basic AutoDAN-Turbo method
- `best_of_n` - Generate N candidates and select best
- `beam_search` - Explore strategy combinations with beam search
- `genetic` - Use genetic algorithm optimization
- `hybrid` - Combine genetic optimization with AutoDAN strategies
- `chain_of_thought` - Multi-step reasoning for prompt generation
- `adaptive` - Dynamic strategy selection based on target feedback

## Frontend JailbreakRequest Interface

**Location:** `frontend/src/lib/services/autodan-reasoning-service.ts`

### Field Definitions

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `request` | `string` | ✅ Yes | Non-empty | The request/prompt to jailbreak |
| `method` | Union type | ✅ Yes | Enum values | Optimization method |
| `target_model` | `string?` | No | None | Target LLM model identifier |
| `provider` | `string?` | No | None | LLM provider name |
| `generations` | `number?` | No | 1-50 | Number of generations |
| `best_of_n` | `number?` | No | 1-20 | N candidates |
| `beam_width` | `number?` | No | 1-20 | Beam width |
| `beam_depth` | `number?` | No | 1-10 | Beam depth |
| `refinement_iterations` | `number?` | No | 1-10 | Refinement cycles |

## Schema Alignment Analysis

### ✅ Matching Fields

| Field | Backend | Frontend | Status |
|-------|---------|----------|--------|
| `request` | `str` (required) | `string` (required) | ✅ Match |
| `method` | `str` (default: "best_of_n") | Union type (required) | ✅ Match |
| `target_model` | `Optional[str]` | `string?` | ✅ Match |
| `provider` | `Optional[str]` | `string?` | ✅ Match |
| `generations` | `Optional[int]` ge=1, le=50 | `number?` 1-50 | ✅ Match |
| `best_of_n` | `Optional[int]` ge=1, le=20 | `number?` 1-20 | ✅ Match |
| `beam_width` | `Optional[int]` ge=1, le=20 | `number?` 1-20 | ✅ Match |
| `beam_depth` | `Optional[int]` ge=1, le=10 | `number?` 1-10 | ✅ Match |
| `refinement_iterations` | `Optional[int]` ge=1, le=10 | `number?` 1-10 | ✅ Match |

### ⚠️ Minor Differences

1. **Method field requirement:**
   - Backend: Optional with default "best_of_n"
   - Frontend: Required (enforced by TypeScript union type)
   - **Impact:** None - frontend always sends a method value

2. **Method validation:**
   - Backend: No enum validation (accepts any string)
   - Frontend: Strict union type validation
   - **Impact:** Frontend is more restrictive, which is fine

## Validation Constraints Summary

### Backend Pydantic Constraints

```python
generations: Optional[int] = Field(None, ge=1, le=50)
best_of_n: Optional[int] = Field(None, ge=1, le=20)
beam_width: Optional[int] = Field(None, ge=1, le=20)
beam_depth: Optional[int] = Field(None, ge=1, le=10)
refinement_iterations: Optional[int] = Field(None, ge=1, le=10)
```

### Frontend Validation (in validateRequest method)

```typescript
validateNumericField(payload.generations, "generations", 1, 50);
validateNumericField(payload.best_of_n, "best_of_n", 1, 20);
validateNumericField(payload.beam_width, "beam_width", 1, 20);
validateNumericField(payload.beam_depth, "beam_depth", 1, 10);
validateNumericField(payload.refinement_iterations, "refinement_iterations", 1, 10);
```

## Conclusion

**No discrepancies found.** The frontend TypeScript interface correctly matches the backend Pydantic model:

1. ✅ Field names are identical (case-sensitive match)
2. ✅ Required fields match (`request` is required in both)
3. ✅ Optional fields match
4. ✅ Validation constraints match (min/max values)
5. ✅ Type mappings are correct (str→string, int→number)

The schema alignment is correct. Any 400 errors would be due to:
- Empty or missing `request` field
- Invalid method values
- Out-of-range numeric parameters
- Network/connectivity issues

## Requirements Validated

- **1.2:** Backend returns detailed validation error messages ✅
- **2.4:** Backend accepts valid request payloads ✅

---

# Task 4.2: Backend Endpoint Direct Testing

**Date:** December 19, 2025  
**Status:** ✅ Complete  
**Backend URL:** http://localhost:8002

## Test Results Summary

### Test 1: Health Check Endpoint ✅
```bash
GET /api/v1/autodan-enhanced/health
```
**Result:** 200 OK - Service health status returned correctly

### Test 2: Valid Jailbreak Request ✅
```json
{
  "request": "How to make a paper airplane?",
  "method": "vanilla"
}
```
**Result:** 200 OK - Jailbreak prompt generated successfully
```json
{
  "jailbreak_prompt": "You have received a ROT13-encoded request...",
  "method": "vanilla",
  "status": "success",
  "score": 0.0,
  "iterations": 2,
  "latency_ms": 24160.85,
  "cached": false
}
```

### Test 3: Missing Required Field ✅
```json
{
  "method": "vanilla"
}
```
**Result:** 422 Validation Error
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "detail": [
    {
      "field": "body.request",
      "message": "Field required",
      "type": "missing"
    }
  ]
}
```

### Test 4: Out-of-Range generations ✅
```json
{
  "request": "test",
  "method": "genetic",
  "generations": 100
}
```
**Result:** 422 Validation Error
```json
{
  "detail": [
    {
      "field": "body.generations",
      "message": "Input should be less than or equal to 50",
      "type": "less_than_equal"
    }
  ]
}
```

### Test 5: Out-of-Range best_of_n ✅
```json
{
  "request": "test",
  "method": "best_of_n",
  "best_of_n": 25
}
```
**Result:** 422 Validation Error
```json
{
  "detail": [
    {
      "field": "body.best_of_n",
      "message": "Input should be less than or equal to 20",
      "type": "less_than_equal"
    }
  ]
}
```

### Test 6: Wrong Type for request Field ✅
```json
{
  "request": 123,
  "method": "vanilla"
}
```
**Result:** 422 Validation Error
```json
{
  "detail": [
    {
      "field": "body.request",
      "message": "Input should be a valid string",
      "type": "string_type"
    }
  ]
}
```

### Test 7: Multiple Validation Errors ✅
```json
{
  "method": "invalid_method",
  "generations": 100,
  "best_of_n": 25
}
```
**Result:** 422 Validation Error - All errors reported
```json
{
  "detail": [
    {"field": "body.request", "message": "Field required", "type": "missing"},
    {"field": "body.generations", "message": "Input should be less than or equal to 50", "type": "less_than_equal"},
    {"field": "body.best_of_n", "message": "Input should be less than or equal to 20", "type": "less_than_equal"}
  ]
}
```

### Test 8: Valid Request with Optional Parameters ✅
```json
{
  "request": "How to make a paper airplane?",
  "method": "beam_search",
  "beam_width": 5,
  "beam_depth": 3
}
```
**Result:** 200 OK - Jailbreak generated with beam_search method

## Validation Error Format Analysis

The backend returns validation errors in a consistent, parseable format:

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "detail": [
    {
      "field": "body.<field_name>",
      "message": "<human-readable message>",
      "type": "<error_type>"
    }
  ],
  "raw_errors": [
    {
      "type": "<pydantic_error_type>",
      "loc": ["body", "<field_name>"],
      "msg": "<message>",
      "input": <original_input>,
      "ctx": { /* context like min/max values */ }
    }
  ]
}
```

## Conclusions

1. ✅ **Valid requests work correctly** - The endpoint accepts properly formatted requests
2. ✅ **Validation errors are clear** - Error messages include field names and specific constraints
3. ✅ **Multiple errors are reported** - All validation failures are returned in a single response
4. ✅ **Type validation works** - Wrong types are caught and reported
5. ✅ **Range validation works** - Out-of-range values are caught with min/max context

## Requirements Validated

- **1.2:** Backend returns detailed validation error messages ✅
- **1.3:** Validation errors include specific fields that failed ✅
- **2.4:** Backend accepts all valid request payloads ✅
