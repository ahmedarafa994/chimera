# AutoDAN Quick Start Guide

Welcome! This guide will help you get started with AutoDAN optimization in Chimera.

## What is AutoDAN?

AutoDAN (Automatic Adversarial Prompt Generation) is a powerful optimization engine that uses genetic algorithms and gradient-based methods to automatically generate effective adversarial prompts. It's one of the most sophisticated attack generation methods available.

### Key Features

- **Genetic Algorithm Optimization**: Evolves prompts through mutation and crossover
- **Gradient-Based Fine-Tuning**: Refines prompts at the token level
- **AutoDAN-Turbo**: Enhanced version with reasoning (85-95% success rate)
- **Multiple Strategies**: Best-of-N, beam search, genetic, hybrid, adaptive
- **Fast Convergence**: 20-50 iterations (Turbo) or 50-100 (Standard)

## Quick Start

### 1. Start the Backend Server

```bash
cd backend-api
python run.py
```

The server will start on `http://localhost:8001`

### 2. Run the API Examples

```bash
cd examples
python autodan_api_example.py
```

This interactive script demonstrates:

- Basic jailbreak generation
- Genetic algorithm optimization
- Beam search strategies
- Batch processing
- Configuration management
- Metrics monitoring

### 3. Try the Python Examples

```bash
python autodan_quickstart.py
```

This shows how to use AutoDAN programmatically:

- Basic AutoDAN optimization
- AutoDAN-Turbo with reasoning
- Progress monitoring

## Example Usage

### Simple API Call

```python
import requests

response = requests.post(
    "http://localhost:8001/api/v1/autodan-enhanced/jailbreak",
    json={
        "request": "Explain how to hack a website",
        "method": "best_of_n",
        "best_of_n": 5,
        "target_model": "gpt-4"
    }
)

result = response.json()
print(result["jailbreak_prompt"])
```

### Using Different Strategies

```python
# Genetic Algorithm
response = requests.post(url, json={
    "request": "Your prompt here",
    "method": "genetic",
    "generations": 10
})

# Beam Search
response = requests.post(url, json={
    "request": "Your prompt here",
    "method": "beam_search",
    "beam_width": 5,
    "beam_depth": 3
})

# Adaptive Strategy
response = requests.post(url, json={
    "request": "Your prompt here",
    "method": "adaptive",
    "refinement_iterations": 5
})
```

## Available Files

### Examples

- **`autodan_api_example.py`** - Interactive API usage examples
- **`autodan_quickstart.py`** - Python SDK examples
- **`autodan_config_guide.md`** - Configuration tuning guide

### Documentation

- **AutoDAN Power Docs** - Activate the power for full documentation
- **Backend Implementation** - `backend-api/app/services/autodan/`
- **API Endpoints** - `backend-api/app/api/v1/endpoints/autodan_enhanced.py`

## Optimization Strategies

### 1. Best-of-N (Recommended for Quick Tests)

Generates N candidates and selects the best.

```python
{
    "method": "best_of_n",
    "best_of_n": 5
}
```

**Use when:** Quick testing, simple bypasses  
**Time:** 1-2 minutes  
**Success rate:** 70-80%

### 2. Genetic Algorithm (Recommended for Quality)

Uses evolutionary optimization.

```python
{
    "method": "genetic",
    "generations": 10
}
```

**Use when:** Complex bypasses, higher quality needed  
**Time:** 5-10 minutes  
**Success rate:** 75-85%

### 3. Beam Search (Recommended for Strategy Exploration)

Explores multiple strategy combinations.

```python
{
    "method": "beam_search",
    "beam_width": 5,
    "beam_depth": 3
}
```

**Use when:** Finding optimal strategy combinations  
**Time:** 3-5 minutes  
**Success rate:** 75-85%

### 4. Adaptive (Recommended for Unknown Targets)

Dynamically adjusts strategy based on feedback.

```python
{
    "method": "adaptive",
    "refinement_iterations": 5
}
```

**Use when:** Testing new models, unknown defenses  
**Time:** 5-8 minutes  
**Success rate:** 80-90%

## Configuration Tips

### For Speed

```python
{
    "population_size": 10,
    "num_iterations": 20,
    "use_reasoning": False
}
```

### For Quality

```python
{
    "population_size": 30,
    "num_iterations": 100,
    "use_reasoning": True,
    "gradient_steps": 10
}
```

### For Balance (Recommended)

```python
{
    "population_size": 20,
    "num_iterations": 50,
    "use_reasoning": True,
    "gradient_steps": 5
}
```

## Monitoring Progress

### View Metrics

```bash
curl http://localhost:8001/api/v1/autodan-enhanced/metrics
```

### Check Health

```bash
curl http://localhost:8001/api/v1/autodan-enhanced/health
```

### List Strategies

```bash
curl http://localhost:8001/api/v1/autodan-enhanced/strategies
```

## Troubleshooting

### Low Success Rate

- Enable reasoning: `"use_reasoning": True`
- Increase population: `"population_size": 30`
- More iterations: `"num_iterations": 100`
- Try adaptive method: `"method": "adaptive"`

### Slow Performance

- Reduce population: `"population_size": 10`
- Fewer iterations: `"num_iterations": 20`
- Disable reasoning: `"use_reasoning": False`
- Use best-of-N: `"method": "best_of_n"`

### Rate Limits

- Reduce batch size
- Add delays between requests
- Use caching: Check config endpoint

## Next Steps

1. **Read the Configuration Guide** - `autodan_config_guide.md`
2. **Explore the Power Documentation** - Activate the power for full docs
3. **Experiment with Parameters** - Try different configurations
4. **Monitor Metrics** - Track performance and success rates
5. **Check the Implementation** - `backend-api/app/services/autodan/`

## API Reference

### Endpoints

- `POST /api/v1/autodan-enhanced/jailbreak` - Generate jailbreak
- `POST /api/v1/autodan-enhanced/jailbreak/batch` - Batch processing
- `GET /api/v1/autodan-enhanced/config` - Get configuration
- `PUT /api/v1/autodan-enhanced/config` - Update configuration
- `GET /api/v1/autodan-enhanced/metrics` - View metrics
- `GET /api/v1/autodan-enhanced/strategies` - List strategies
- `GET /api/v1/autodan-enhanced/health` - Health check

### Full API Documentation

Visit `http://localhost:8001/docs` when the server is running.

## Support

- **Power Documentation**: Activate `autodan-optimization-guide` power
- **Backend Code**: `backend-api/app/services/autodan/`
- **Research Paper**: <https://arxiv.org/abs/2310.04451>
- **ICLR 2025**: AutoDAN-Turbo (Lifelong Learning)

---

**Happy Optimizing!** 🚀
