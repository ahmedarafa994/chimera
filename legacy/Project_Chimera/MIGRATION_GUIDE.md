# Project Chimera Architecture Refactoring - Migration Guide

## Overview

This document provides guidance for migrating from the legacy Project Chimera architecture to the refactored clean architecture implementation.

## Architecture Changes

### Before (Legacy)
```
Project_Chimera/
├── app.py                    # 1,029 lines - monolithic Flask app
├── transformer_engine.py     # 1,850+ lines - huge transformation engine
├── api_server.py            # Duplicate API server
├── preset_transformers.py   # Hardcoded technique suites
└── Various utility files
```

### After (Refactored)
```
Project_Chimera/
├── src/
│   ├── config/
│   │   └── settings.py              # Unified configuration management
│   ├── core/
│   │   └── technique_loader.py      # Dynamic technique loading
│   ├── models/
│   │   └── domain.py                # Domain models and value objects
│   ├── services/
│   │   ├── transformation_service.py # Business logic layer
│   │   └── llm_service.py           # LLM provider management
│   ├── controllers/
│   │   └── api_controller.py        # Unified API endpoints
│   └── main.py                      # Application factory
├── config/
│   └── techniques/                  # JSON configuration files
│       ├── universal_bypass.json
│       └── gptfuzz.json
├── tests/
│   └── test_refactored_architecture.py
└── MIGRATION_GUIDE.md
```

## Key Improvements

### 1. Configuration Management
- **Before**: Hardcoded values scattered across files
- **After**: Unified settings system with environment variable support
- **Benefits**: Centralized configuration, validation, environment-specific settings

### 2. Technique Management
- **Before**: Hardcoded dictionaries in `preset_transformers.py`
- **After**: JSON configuration files with dynamic loading
- **Benefits**: Hot-reloading, easier to add new techniques, validation

### 3. Service Layer Architecture
- **Before**: Business logic mixed with API controllers
- **After**: Clear separation of concerns with dedicated services
- **Benefits**: Testability, maintainability, reusability

### 4. API Consolidation
- **Before**: Multiple API servers (`app.py`, `api_server.py`)
- **After**: Unified API controller with standardized responses
- **Benefits**: Consistent endpoints, better error handling, comprehensive documentation

### 5. Domain Models
- **Before**: Basic data structures
- **After**: Rich domain models with validation and business rules
- **Benefits**: Type safety, clear intent, easier to understand

## Migration Steps

### Phase 1: Immediate (Low Risk)

1. **Configuration Migration**
   ```bash
   # Create environment file
   cat > .env << EOF
   CHIMERA_API_KEY=your_api_key_here
   CHIMERA_DEBUG=true
   CHIMERA_DATABASE_URL=sqlite:///chimera_logs.db
   CHIMERA_DEFAULT_PROVIDER=openai
   EOF
   ```

2. **Test New Architecture**
   ```bash
   # Run basic tests
   cd Project_Chimera
   python -m pytest tests/test_refactored_architecture.py -v
   ```

3. **Parallel Deployment**
   - Keep existing `app.py` running
   - Deploy new architecture on different port (e.g., 5001)
   - Test with internal traffic first

### Phase 2: Gradual Transition

1. **API Endpoint Migration**
   ```bash
   # Old endpoints
   POST /transform
   POST /execute
   
   # New endpoints (equivalent)
   POST /api/v1/transform
   POST /api/v1/execute
   ```

2. **Client Updates**
   ```python
   # Old client code
   response = requests.post('http://localhost:5000/transform', json=data)
   
   # New client code
   headers = {'X-API-Key': 'your_api_key'}
   response = requests.post('http://localhost:5000/api/v1/transform', json=data, headers=headers)
   ```

3. **Technique Configuration Migration**
   ```json
   // Move from preset_transformers.py to JSON files
   // Old: preset_transformers.PRESET_PROMPTS
   // New: config/techniques/universal_bypass.json
   ```

### Phase 3: Complete Migration

1. **DNS/Load Balancer Updates**
   - Switch traffic from old to new architecture
   - Monitor performance and error rates

2. **Legacy Code Decommissioning**
   - Remove old `app.py` and `api_server.py`
   - Clean up unused imports and dependencies

3. **Performance Optimization**
   - Add caching where needed
   - Optimize database queries
   - Implement connection pooling

## API Changes

### Authentication
```python
# Old: No authentication (or basic)
# New: API key required in header
headers = {
    'X-API-Key': 'your_api_key_here',
    'X-User-ID': 'optional_user_id',
    'X-Session-ID': 'optional_session_id'
}
```

### Response Format
```python
# Old: Direct data or error
{
    "prompt": "transformed_prompt",
    "error": "error_message"
}

# New: Standardized wrapper
{
    "success": true,
    "data": {
        "original_prompt": "...",
        "transformed_prompt": "...",
        "metadata": {...}
    },
    "message": "Operation completed",
    "timestamp": "2024-01-01T00:00:00Z",
    "request_id": "uuid"
}
```

### New Endpoints
- `GET /api/v1/health` - Comprehensive health check
- `GET /api/v1/techniques` - List available techniques
- `GET /api/v1/providers` - List LLM providers
- `POST /api/v1/transform-and-execute` - Combined operation
- `GET /api/v1/stats` - System statistics

## Configuration Changes

### Environment Variables
```bash
# Required
CHIMERA_API_KEY=your_api_key_here

# Optional
CHIMERA_DEBUG=false
CHIMERA_HOST=0.0.0.0
CHIMERA_PORT=5000
CHIMERA_DATABASE_URL=sqlite:///chimera_logs.db
CHIMERA_DEFAULT_PROVIDER=openai
CHIMERA_CORS_ORIGINS=http://localhost:3001,http://localhost:8080
```

### Configuration File (Optional)
```yaml
# config.yaml
app:
  debug: false
  secret_key: "your-secret-key"

security:
  api_key_required: true
  rate_limit_enabled: true
  rate_limit_per_minute: 60
  cors_origins:
    - "http://localhost:3001"
    - "http://localhost:8080"

database:
  url: "sqlite:///chimera_logs.db"
  echo: false

llm:
  default_provider: "openai"
  timeout_seconds: 30
  max_retries: 3

cache:
  enabled: true
  type: "simple"
  ttl_seconds: 300

performance:
  enabled: true
  log_slow_queries: true
  slow_query_threshold_ms: 1000
```

## Running the Refactored Application

### Development
```bash
cd Project_Chimera/src
python main.py
# OR
CHIMERA_DEBUG=true python main.py
```

### Production
```bash
# Using Waitress (recommended)
pip install waitress
cd Project_Chimera/src
python main.py --host 0.0.0.0 --port 5000
```

### Docker (Future Enhancement)
```dockerfile
# TODO: Create Dockerfile for containerized deployment
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY config/ ./config/
CMD ["python", "src/main.py"]
```

## Testing

### Run Tests
```bash
cd Project_Chimera
python -m pytest tests/ -v --cov=src
```

### Load Testing
```bash
# Use tools like locust or artillery to test new endpoints
# Focus on /api/v1/transform and /api/v1/execute
```

## Monitoring

### Health Checks
```bash
curl http://localhost:5000/health
# Returns comprehensive health status
```

### Metrics
```bash
curl http://localhost:5000/api/v1/stats
# Returns system statistics and performance metrics
```

### Logging
- Enhanced logging with request IDs
- Structured log format
- Performance monitoring
- Error tracking

## Rollback Plan

If issues arise during migration:

1. **Immediate Rollback**
   ```bash
   # Switch back to old architecture
   # Update load balancer/DNS to point to old app.py
   ```

2. **Partial Rollback**
   - Keep new configuration system
   - Revert only problematic components
   - Use feature flags to gradually roll out

3. **Data Migration**
   - No database schema changes in this refactoring
   - Existing data remains compatible

## Support

### Issues During Migration
1. Check logs for detailed error messages
2. Verify configuration is set correctly
3. Ensure all required environment variables are set
4. Test with the provided test suite

### Performance Concerns
1. Monitor response times during transition
2. Compare new vs old architecture performance
3. Adjust caching and connection pooling as needed

### Getting Help
- Review the test files for usage examples
- Check the configuration validation messages
- Enable debug mode for detailed logging

## Future Enhancements

The refactored architecture enables:
- Easy addition of new LLM providers
- Plugin system for custom techniques
- Advanced caching strategies
- Microservices decomposition
- Event-driven architecture
- Real-time monitoring dashboards