#!/bin/bash
# =============================================================================
# Chimera Smoke Test Script
# Runs basic health checks after deployment
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3001}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8001}"
TIMEOUT=10

# Test results
PASSED=0
FAILED=0

# Helper functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

test_url() {
    local url=$1
    local expected_status=$2
    local description=$3

    log_info "Testing: $description"

    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$url" 2>/dev/null)

    if [ "$response" == "$expected_status" ]; then
        log_pass "$description (HTTP $response)"
        return 0
    else
        log_fail "$description (Expected $expected_status, got $response)"
        return 1
    fi
}

test_json_response() {
    local url=$1
    local key=$2
    local description=$3

    log_info "Testing: $description"

    response=$(curl -s --max-time $TIMEOUT "$url" 2>/dev/null)

    if echo "$response" | grep -q "\"$key\""; then
        log_pass "$description"
        return 0
    else
        log_fail "$description (Key '$key' not found in response)"
        return 1
    fi
}

# =============================================================================
# Run Tests
# =============================================================================

echo ""
echo "========================================"
echo "   CHIMERA SMOKE TEST SUITE"
echo "========================================"
echo ""
echo "Frontend URL: $FRONTEND_URL"
echo "Backend URL:  $BACKEND_URL"
echo ""
echo "----------------------------------------"
echo ""

# Test 1: Frontend root URL
test_url "$FRONTEND_URL" "200" "Frontend root URL returns 200"

# Test 2: Frontend dashboard route
test_url "$FRONTEND_URL/dashboard" "200" "Dashboard route returns 200"

# Test 3: Frontend jailbreak route
test_url "$FRONTEND_URL/dashboard/jailbreak" "200" "Jailbreak route returns 200"

# Test 4: Backend health endpoint
test_url "$BACKEND_URL/health" "200" "Backend health endpoint returns 200"

# Test 5: Backend health response contains status
test_json_response "$BACKEND_URL/health" "status" "Backend health response contains 'status'"

# Test 6: Backend root endpoint (API info)
test_url "$BACKEND_URL/" "200" "Backend root endpoint returns 200"

# Test 7: API v1 health endpoint
test_url "$BACKEND_URL/api/v1/health" "200" "API v1 health endpoint returns 200"

# Test 8: CORS preflight check
log_info "Testing: CORS preflight request"
cors_response=$(curl -s -o /dev/null -w "%{http_code}" \
    -X OPTIONS \
    -H "Origin: $FRONTEND_URL" \
    -H "Access-Control-Request-Method: POST" \
    --max-time $TIMEOUT \
    "$BACKEND_URL/api/v1/generation/jailbreak/generate" 2>/dev/null)

if [ "$cors_response" == "200" ]; then
    log_pass "CORS preflight returns 200"
else
    log_fail "CORS preflight failed (HTTP $cors_response)"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "----------------------------------------"
echo ""
echo "RESULTS SUMMARY"
echo ""
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All smoke tests passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}Some smoke tests failed. Please investigate.${NC}"
    echo ""
    exit 1
fi
