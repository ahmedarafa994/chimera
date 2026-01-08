# Implementation Plan: AutoDAN Enhanced API Fix

## Overview

This plan addresses the 400 Bad Request error by fixing schema mismatches between the frontend and backend. The primary issue is incorrect field names in the request payload. We'll verify the current implementation, add debugging, fix any mismatches, and add tests to prevent regression.

## Tasks

- [x] 1. Investigate and document the current error
  - Add detailed logging to capture the exact request payload being sent
  - Capture the backend's validation error response
  - Document the specific field mismatches
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Fix frontend service layer
  - [x] 2.1 Update JailbreakRequest interface to match backend schema
    - Ensure `request` field is used (not `prompt`)
    - Verify all optional fields match backend names
    - Add JSDoc comments documenting field requirements
    - _Requirements: 2.1, 2.2, 4.1_

  - [ ]* 2.2 Write unit tests for request payload construction
    - Test that payload contains `request` field
    - Test optional parameter inclusion
    - Test field name correctness
    - _Requirements: 2.1, 2.2_

  - [x] 2.3 Add request validation before sending
    - Validate required fields are present
    - Validate optional fields are within valid ranges
    - Log validation errors before sending
    - _Requirements: 2.3, 3.5_

- [x] 3. Enhance error handling and debugging
  - [x] 3.1 Add comprehensive request/response logging
    - Log request payload in development mode
    - Log response status and body
    - Log validation errors with details
    - _Requirements: 1.1, 3.1, 3.2, 3.5_

  - [x] 3.2 Improve error message display
    - Parse Pydantic validation errors
    - Display field-specific error messages
    - Provide actionable guidance for users
    - _Requirements: 3.1, 3.3, 3.4_

  - [ ]* 3.3 Write unit tests for error handling
    - Test 400 error parsing
    - Test error message display
    - Test validation error extraction
    - _Requirements: 3.1, 3.3_

- [x] 4. Verify backend API endpoint
  - [x] 4.1 Review backend Pydantic model
    - Confirm field names and types
    - Verify validation constraints
    - Document any discrepancies
    - _Requirements: 1.2, 2.4_

  - [x] 4.2 Test backend endpoint directly
    - Send test requests using curl or Postman
    - Verify successful requests work
    - Verify validation errors are clear
    - _Requirements: 1.2, 1.3, 2.4_

  - [ ]* 4.3 Write backend validation tests
    - Test valid request acceptance
    - Test invalid request rejection
    - Test validation error messages
    - _Requirements: 1.2, 1.3, 2.4_

- [x] 5. Integration testing
  - [x] 5.1 Test end-to-end request flow
    - Send request from frontend UI
    - Verify request reaches backend
    - Verify response is received and parsed
    - Verify UI updates correctly
    - _Requirements: 2.4, 3.4_

  - [ ]* 5.2 Write integration tests
    - Test successful jailbreak generation
    - Test error scenarios (400, 503, 401)
    - Test with various method types
    - _Requirements: 2.4, 3.4_

- [-] 6. Type safety improvements
  - [-] 6.1 Sync TypeScript types with backend schema
    - Update JailbreakRequest interface
    - Update JailbreakResponse interface
    - Add type guards for validation
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 6.2 Add compile-time type checking tests
    - Test that incorrect types are caught
    - Test that required fields are enforced
    - Test that optional fields work correctly
    - _Requirements: 4.4, 4.5_

- [ ] 7. Documentation and cleanup
  - [ ] 7.1 Update API documentation
    - Document correct request format
    - Document all available fields
    - Add examples of valid requests
    - _Requirements: 1.3, 3.3_

  - [ ] 7.2 Add inline code comments
    - Comment critical field mappings
    - Document validation logic
    - Explain error handling flow
    - _Requirements: 3.3, 3.5_

- [ ] 8. Final verification
  - Ensure all tests pass
  - Verify no 400 errors occur in browser console
  - Test with multiple method types (vanilla, best_of_n, beam_search, etc.)
  - Verify error messages are clear and actionable
  - _Requirements: 1.4, 2.4, 3.3, 3.4_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Focus on fixing the immediate 400 error first (tasks 1-2)
- Then improve error handling and debugging (task 3)
- Finally add comprehensive testing (tasks 4-6)
- The primary fix is ensuring the frontend sends `request` field instead of any other field name
