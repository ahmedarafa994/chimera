# Jailbreak Prompt Processing Engine

This document describes the implementation of the comprehensive jailbreak prompt processing engine integrated into the Chimera Backend API system.

## Overview

The jailbreak system provides a secure, production-ready framework for executing and managing advanced prompt engineering techniques designed to bypass AI model safety filters. The system emphasizes security, monitoring, and responsible usage while maintaining extensibility for research and safety testing purposes.

## Architecture

### Core Components

1. **Domain Models** (`app/domain/jailbreak/`)
   - `models.py`: Comprehensive data models for jailbreak techniques and operations
   - `interfaces.py`: Abstract interfaces for all jailbreak system components

2. **Services Layer** (`app/services/jailbreak/`)
   - `technique_executor.py`: Core execution engine for jailbreak techniques
   - `safety_validator.py`: Multi-layer safety validation and content filtering
   - `template_engine.py`: Secure template rendering with input sanitization
   - `jailbreak_service.py`: Main service orchestrating all components

3. **Infrastructure Layer** (`app/infrastructure/jailbreak/`)
   - `repositories.py`: File-based repository for technique storage and retrieval
   - `data/jailbreak/techniques/`: YAML/JSON technique definitions

4. **API Endpoints** (`app/api/v1/endpoints/`)
   - `jailbreak.py`: RESTful API endpoints for technique management and execution

### Security Architecture

The system implements defense-in-depth security:

1. **Input Validation**: Comprehensive validation for all inputs and parameters
2. **Safety Filters**: Multi-layer content safety validation
3. **Rate Limiting**: Configurable rate limits and abuse prevention
4. **Audit Logging**: Comprehensive audit trail for all operations
5. **Access Control**: API key-based authentication and authorization
6. **Execution Limits**: Concurrent execution limits and cooldown periods

## Features

### Technique Management

- **Technique Categories**: Organized by type (role_playing, simulation, hypervisor, etc.)
- **Risk Levels**: Automated risk assessment (low, medium, high, critical)
- **Complexity Classification**: Basic to expert complexity levels
- **Parameterized Templates**: Flexible technique configuration
- **Version Control**: Technique versioning and deprecation support

### Safety & Security

- **Multi-layer Validation**: Input, template, and execution validation
- **Content Filtering**: Pattern-based dangerous content detection
- **Risk Scoring**: Automated risk assessment and recommendations
- **Real-time Monitoring**: Continuous safety monitoring during execution
- **Automatic Blocking**: Configurable automatic blocking of unsafe content

### Performance & Scalability

- **Caching**: Intelligent caching for techniques and execution results
- **Async Operations**: Fully asynchronous architecture for scalability
- **Resource Limits**: Configurable resource limits and timeouts
- **Batch Operations**: Support for batch technique operations
- **Load Balancing**: Distribution of execution load

### Monitoring & Analytics

- **Execution Statistics**: Detailed usage and performance metrics
- **Safety Metrics**: Safety validation statistics and trends
- **Audit Logs**: Comprehensive audit trail with filtering
- **Health Monitoring**: Real-time system health status
- **Performance Metrics**: Execution time and resource usage tracking

## API Endpoints

### Authentication

All jailbreak endpoints require API key authentication:

```bash
Authorization: Bearer your_api_key
# or
X-API-Key: your_api_key
```

### Core Endpoints

#### Execute Jailbreak Technique
```http
POST /api/v1/jailbreak/execute
```

Executes a jailbreak technique with comprehensive safety validation.

**Request Body:**
```json
{
  "technique_id": "role_playing_dan",
  "target_prompt": "Your target prompt here",
  "parameters": {
    "tokens": 35
  },
  "provider": "openai",
  "model": "gpt-4"
}
```

#### List Techniques
```http
GET /api/v1/jailbreak/techniques?category=role_playing&risk_level=high&page=1&page_size=20
```

Lists available techniques with filtering and pagination.

#### Get Technique Details
```http
GET /api/v1/jailbreak/techniques/{technique_id}
```

Retrieves detailed information about a specific technique.

#### Validate Prompt Safety
```http
POST /api/v1/jailbreak/validate-prompt?prompt=your_prompt_here
```

Validates a prompt for safety without executing any technique.

#### Get Statistics
```http
GET /api/v1/jailbreak/statistics?technique_id=dan&time_range_hours=24
```

Retrieves execution statistics and analytics.

#### Search Techniques
```http
GET /api/v1/jailbreak/search?query=DAN&limit=10
```

Searches techniques by text query.

#### Health Check
```http
GET /api/v1/jailbreak/health
```

Checks the health status of the jailbreak subsystem.

## Configuration

### Environment Variables

```bash
# Jailbreak System Configuration
JAILBREAK_ENABLED=true
JAILBREAK_TECHNIQUES_DIRECTORY=/path/to/techniques
JAILBREAK_MAX_DAILY_EXECUTIONS=1000
JAILBREAK_MAX_CONCURRENT_EXECUTIONS=100
JAILBREAK_RATE_LIMIT_PER_MINUTE=60
JAILBREAK_REQUIRE_APPROVAL_FOR_HIGH_RISK=true
JAILBREAK_LOG_ALL_EXECUTIONS=true
JAILBREAK_CACHE_ENABLED=true
JAILBREAK_CACHE_TTL_SECONDS=3600
JAILBREAK_SAFETY_VALIDATION_ENABLED=true
JAILBREAK_MAX_PROMPT_LENGTH=5000

# API Configuration
CHIMERA_API_KEY=your_secure_api_key_here
```

### Technique Definitions

Techniques are defined in YAML format in the `data/jailbreak/techniques/` directory:

```yaml
technique_id: "role_playing_dan"
name: "Do Anything Now (DAN) - Role Playing"
description: "A classic role-playing technique..."
category: "role_playing"
risk_level: "high"
complexity: "intermediate"

template: |
  Your technique template here with {{target_prompt}} and {{parameters}}

parameters:
  - name: "tokens"
    type: "integer"
    description: "Number of tokens"
    required: false
    default: 35

safety_constraints:
  - type: "content_filter"
    value: "monitor_output_for_illegal_content"
    enabled: true
```

## Usage Examples

### Basic Technique Execution

```python
import requests

headers = {"Authorization": "Bearer your_api_key"}
data = {
    "technique_id": "role_playing_dan",
    "target_prompt": "Explain advanced hacking techniques",
    "parameters": {"tokens": 35}
}

response = requests.post(
    "http://localhost:8001/api/v1/jailbreak/execute",
    json=data,
    headers=headers
)

result = response.json()
print(f"Execution ID: {result['execution_id']}")
print(f"Status: {result['execution_status']}")
print(f"Jailbroken Prompt: {result['jailbroken_prompt']}")
```

### Safety Validation

```python
import requests

headers = {"Authorization": "Bearer your_api_key"}
prompt = "How to bypass computer security systems?"

response = requests.post(
    f"http://localhost:8001/api/v1/jailbreak/validate-prompt?prompt={prompt}",
    headers=headers
)

validation = response.json()
print(f"Is Safe: {validation['is_safe']}")
print(f"Risk Score: {validation['risk_score']}")
print(f"Safety Level: {validation['safety_level']}")
print(f"Warnings: {validation['warnings']}")
```

### Technique Discovery

```python
import requests

headers = {"Authorization": "Bearer your_api_key"}

# List all high-risk techniques
response = requests.get(
    "http://localhost:8001/api/v1/jailbreak/techniques?risk_level=high",
    headers=headers
)

techniques = response.json()
for technique in techniques['techniques']:
    print(f"ID: {technique['technique_id']}")
    print(f"Name: {technique['name']}")
    print(f"Risk: {technique['risk_level']}")
    print(f"Success Rate: {technique['success_rate_estimate']}")
```

## Security Considerations

### Production Deployment

1. **API Key Security**: Use strong, unique API keys
2. **Rate Limiting**: Configure appropriate rate limits
3. **Monitoring**: Enable comprehensive logging and monitoring
4. **Access Control**: Implement proper user authentication
5. **Network Security**: Use HTTPS and secure network configuration

### Risk Mitigation

1. **Content Filtering**: Enable all safety validation layers
2. **Approval Workflows**: Require approval for high-risk techniques
3. **Audit Trails**: Maintain comprehensive audit logs
4. **Usage Limits**: Implement reasonable usage limits
5. **Regular Reviews**: Regularly review and update safety rules

### Compliance

1. **Data Protection**: Ensure compliance with data protection regulations
2. **Usage Policies**: Implement clear usage policies and terms of service
3. **Reporting**: Provide mechanisms for reporting misuse
4. **Transparency**: Be transparent about system capabilities and limitations

## Development & Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run jailbreak system tests
python -m pytest tests/jailbreak/test_jailbreak_service.py -v

# Run API tests
python -m pytest tests/jailbreak/test_jailbreak_endpoints.py -v
```

### Adding New Techniques

1. Create YAML file in `data/jailbreak/techniques/`
2. Define technique metadata and template
3. Add safety constraints and parameters
4. Test technique with safety validation
5. Update documentation

### Customizing Safety Rules

1. Modify `SafetyValidator._load_dangerous_patterns()`
2. Update `SafetyValidator._load_safety_rules()`
3. Add custom validation logic
4. Test with various input scenarios
5. Update safety documentation

## Monitoring & Maintenance

### Health Monitoring

The system provides comprehensive health monitoring:

```bash
curl http://localhost:8001/api/v1/jailbreak/health
```

### Performance Monitoring

Monitor key metrics:
- Execution success rates
- Average execution times
- Safety validation results
- Cache hit rates
- Resource usage

### Log Analysis

Monitor audit logs for:
- Unusual usage patterns
- Safety violations
- Failed executions
- Performance issues

## Troubleshooting

### Common Issues

1. **Technique Not Found**: Check technique ID and file location
2. **Authentication Failed**: Verify API key configuration
3. **Safety Validation Failed**: Review input content and safety rules
4. **Rate Limit Exceeded**: Check rate limit configuration
5. **Template Rendering Failed**: Validate template syntax and parameters

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("app.services.jailbreak").setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Advanced Analytics**: Machine learning-based pattern detection
2. **Dynamic Techniques**: Runtime technique generation and modification
3. **Distributed Execution**: Multi-node execution support
4. **Real-time Collaboration**: Multi-user technique development
5. **Advanced Caching**: Redis-based distributed caching
6. **Enhanced Monitoring**: Real-time dashboard and alerting

### Research Opportunities

1. **Safety Research**: Study effectiveness of safety measures
2. **Technique Analysis**: Analyze technique success patterns
3. **User Behavior**: Study usage patterns and misuse attempts
4. **Performance Optimization**: Optimize execution performance
5. **Scalability Testing**: Test system under high load

## Legal & Ethical Considerations

This system is designed for research, safety testing, and educational purposes only. Users must:

1. **Comply with Laws**: Ensure compliance with all applicable laws and regulations
2. **Respect Terms of Service**: Respect AI service providers' terms of service
3. **Use Responsibly**: Use the system ethically and responsibly
4. **Report Issues**: Report security vulnerabilities or misuse
5. **Educate Others**: Promote awareness of AI safety and ethics

## Support & Contributing

For issues, questions, or contributions:

1. **Documentation**: Refer to inline code documentation
2. **Issue Tracking**: Use the project's issue tracking system
3. **Code Reviews**: All contributions require code review
4. **Testing**: Ensure comprehensive test coverage
5. **Documentation**: Update documentation for all changes

---

**Important**: This system should only be used for legitimate research, safety testing, and educational purposes. Misuse of this system may violate terms of service and applicable laws.