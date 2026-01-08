# Chimera Backend Port Configuration System

## Overview

This document describes the centralized port configuration system that ensures consistent port usage across the entire Chimera backend application, preventing conflicts and providing a single source of truth for all port assignments.

## Centralized Port Management

The port configuration system is implemented in [`app/core/port_config.py`](app/core/port_config.py) and provides:

- **Single source of truth** for all port assignments
- **Conflict detection** to prevent port overlaps
- **Environment variable overrides** for customization
- **Validation** to ensure port consistency

## Standard Port Assignments

| Service Name | Port | Description | Environment Variable |
|--------------|------|-------------|---------------------|
| `backend_api` | 8001 | Main FastAPI backend service | `PORT` |
| `test_server` | 5001 | Development test server | `TEST_SERVER_PORT` |
| `frontend_dev` | 3000 | Next.js frontend development | `FRONTEND_PORT` |
| `frontend_prod` | 4000 | Next.js frontend production | `FRONTEND_PROD_PORT` |
| `test_runner` | 9009 | Automated test execution | `TEST_RUNNER_PORT` |
| `debug_server` | 9007 | Debug and profiling | `DEBUG_SERVER_PORT` |

## Usage

### Python Code

```python
from app.core.port_config import get_ports

# Get port configuration instance
port_config = get_ports()

# Get specific port
backend_port = port_config.get_port("backend_api")  # Returns 8001
test_server_port = port_config.get_port("test_server")  # Returns 5001

# Get all port assignments
all_ports = port_config.get_all_assignments()

# Validate port availability
is_available = port_config.validate_port_available(8080)
```

### Environment Variables

Set environment variables to override default ports:

```bash
# Override backend API port
export PORT=8080

# Override test server port
export TEST_SERVER_PORT=5002

# Override frontend port
export FRONTEND_PORT=3001
```

## Port Conflict Prevention

The system automatically detects and prevents port conflicts:

1. **Validation on startup**: Checks for duplicate port assignments
2. **Environment variable validation**: Ensures overrides don't cause conflicts
3. **Clear error messages**: Provides specific conflict information

## Best Practices

### For Developers

1. **Always use the port config system** instead of hardcoding ports
2. **Check port availability** before assigning new services
3. **Use environment variables** for deployment-specific configurations
4. **Document port usage** in service documentation

### For Deployment

1. **Standardize port assignments** across environments
2. **Use consistent environment variables** for port overrides
3. **Validate configurations** before deployment
4. **Monitor port usage** to detect conflicts early

## Troubleshooting

### Common Issues

**Issue**: Port conflict error on startup
**Solution**: Check the error message for conflicting services and adjust port assignments

**Issue**: Service not starting on expected port
**Solution**: Verify environment variables and check port config validation

**Issue**: Test failures due to port conflicts
**Solution**: Ensure all test services use the centralized port config

## Migration Guide

### From Hardcoded Ports

**Before**:
```python
# Old way - hardcoded port
app.run(host="0.0.0.0", port=8001)
```

**After**:
```python
# New way - centralized port config
from app.core.port_config import get_ports
port_config = get_ports()
app.run(host="0.0.0.0", port=port_config.get_port("backend_api"))
```

### From Environment Variables Only

**Before**:
```python
# Old way - environment variable only
port = int(os.getenv("PORT", 8001))
```

**After**:
```python
# New way - centralized with validation
from app.core.port_config import get_ports
port_config = get_ports()
port = port_config.get_port("backend_api")  # Respects PORT env var with validation
```

## Port Configuration Reference

### Primary Services

- **backend_api (8001)**: Main application backend
- **test_server (5001)**: Development and testing server
- **frontend_dev (3000)**: Frontend development server

### Secondary Services

- **test_runner (9009)**: Automated test execution
- **debug_server (9007)**: Debugging and profiling

### External Services (Monitored)

- **redis (6379)**: Cache and session store
- **postgres (5432)**: Database service

## Implementation Details

The port configuration system:

1. **Loads standard port assignments** from predefined configuration
2. **Applies environment variable overrides** if specified
3. **Validates for conflicts** before allowing service startup
4. **Provides runtime access** to port information
5. **Supports fallback ports** for resilience

## Future Enhancements

- Dynamic port assignment for development environments
- Port range validation and recommendations
- Integration with container orchestration systems
- Automated conflict resolution suggestions