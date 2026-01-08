# Requirements Document

## Introduction

This specification addresses the 400 Bad Request error occurring when the frontend attempts to call the AutoDAN Enhanced jailbreak endpoint. The error indicates a mismatch between the frontend request payload and the backend API expectations.

## Glossary

- **Frontend_Client**: The Next.js React application that sends API requests
- **Backend_API**: The FastAPI Python service that processes jailbreak requests
- **API_Proxy**: The Next.js API route that forwards requests to the Backend_API
- **Request_Payload**: The JSON data sent from Frontend_Client to Backend_API
- **Validation_Error**: An error returned when Request_Payload does not match expected schema

## Requirements

### Requirement 1: API Request Validation

**User Story:** As a developer, I want to understand why the API request is failing, so that I can fix the integration issue.

#### Acceptance Criteria

1. WHEN the Frontend_Client sends a request to `/api/v1/autodan-enhanced/jailbreak`, THE System SHALL log the complete request payload for debugging
2. WHEN the Backend_API receives an invalid request, THE System SHALL return a detailed validation error message
3. WHEN a validation error occurs, THE System SHALL include the specific fields that failed validation
4. THE Frontend_Client SHALL send all required fields as defined in the Backend_API schema
5. THE Frontend_Client SHALL send field names that exactly match the Backend_API schema

### Requirement 2: Request/Response Schema Alignment

**User Story:** As a developer, I want the frontend and backend schemas to be aligned, so that API calls succeed without validation errors.

#### Acceptance Criteria

1. WHEN the Frontend_Client constructs a JailbreakRequest, THE System SHALL include the `request` field (not `prompt`)
2. WHEN the Frontend_Client sends optional parameters, THE System SHALL use the correct field names as defined in the Backend_API
3. THE Frontend_Client SHALL validate request payloads before sending to prevent 400 errors
4. THE Backend_API SHALL accept all valid request payloads from the Frontend_Client
5. WHEN field names differ between frontend and backend, THE System SHALL transform them correctly

### Requirement 3: Error Handling and Debugging

**User Story:** As a developer, I want clear error messages when API calls fail, so that I can quickly identify and fix issues.

#### Acceptance Criteria

1. WHEN a 400 Bad Request error occurs, THE System SHALL display the validation error details in the browser console
2. WHEN the Backend_API rejects a request, THE System SHALL log the rejected payload for debugging
3. THE Frontend_Client SHALL display user-friendly error messages for validation failures
4. THE System SHALL distinguish between network errors and validation errors
5. WHEN debugging is enabled, THE System SHALL log request/response payloads

### Requirement 4: Type Safety

**User Story:** As a developer, I want TypeScript types that match the backend schema, so that I catch errors at compile time.

#### Acceptance Criteria

1. THE Frontend_Client SHALL define TypeScript interfaces that match Backend_API Pydantic models
2. WHEN the Backend_API schema changes, THE System SHALL update frontend TypeScript types
3. THE Frontend_Client SHALL use type-safe API client methods
4. THE System SHALL prevent sending requests with incorrect field types
5. THE TypeScript compiler SHALL catch schema mismatches before runtime
