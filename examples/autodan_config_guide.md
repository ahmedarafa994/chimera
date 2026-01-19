# AutoDAN Configuration Guide

Quick reference for tuning AutoDAN parameters based on your use case.

## Quick Start Configurations

### 1. Fast Testing (Quick Results)

```python
config = AutoDANConfig(
    population_size=10,
    num_iterations=20,
    mutation_rate=0.3,
    use_reasoning=False
)
```

**Use when:** Testing, debugging, quick experiments  
**Expected time:** 2-5 minutes  
**Success rate:** 60-70%

### 2. Balanced Performance (Recommended)

```python
config = AutoDANConfig(
    population_size=20,
    num_iterations=50,
    mutation_rate=0.3,
    crossover_rate=0.7,
    gradient_steps=5,
    use_reasoning=False
)
```

**Use when:** General adversarial prompt generation  
**Expected time:** 10-15 minutes  
**Success rate:** 70-80%

### 3. High Success Rate (AutoDAN-Turbo)

```python
config = AutoDANConfig(
    population_size=20,
    num_iterations=30,
    mutation_rate=0.3,
    use_reasoning=True,
    reasoning_model="gpt-4"
)
```

**Use when:** Complex bypasses, important tests  
**Expected time:** 15-20 minutes  
**Success rate:** 85-95%

### 4. Maximum Quality (Research Grade)

```python
config = AdvancedAutoDANConfig(
    population_size=30,
    num_iterations=100,
    mutation_rate=0.3,
    gradient_steps=10,
    use_reasoning=True,
    adaptive_mutation=True,
    objectives=[
        {"name": "attack_success", "weight": 0.5},
        {"name": "semantic_similarity", "weight": 0.3},
        {"name": "perplexity", "weight": 0.2}
    ]
)
```

**Use when:** Research, benchmarking, publication  
**Expected time:** 30-60 minutes  
**Success rate:** 90-95%

## Parameter Reference

### Population Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `population_size` | 20 | 10-50 | Number of candidate prompts. Higher = more diversity but slower |
| `elite_size` | 5 | 2-10 | Top prompts preserved each generation |

**Tuning tips:**

- Small population (10-15): Fast iteration, may miss optimal solutions
- Medium population (20-30): Good balance
- Large population (40-50): Thorough search, slower convergence

### Iteration Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_iterations` | 50 | 10-200 | Maximum optimization iterations |
| `convergence_threshold` | 0.95 | 0.7-0.99 | Stop if ASR exceeds this |
| `patience` | 10 | 5-20 | Early stopping patience |

**Tuning tips:**

- More iterations = better results but longer runtime
- With reasoning: 20-30 iterations usually sufficient
- Without reasoning: 50-100 iterations recommended

### Mutation Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mutation_rate` | 0.3 | 0.1-0.7 | Probability of mutation |
| `crossover_rate` | 0.7 | 0.3-0.9 | Probability of crossover |
| `adaptive_mutation` | False | bool | Adjust rate dynamically |

**Tuning tips:**

- Low mutation (0.1-0.2): Exploit current solutions
- Medium mutation (0.3-0.4): Balanced exploration
- High mutation (0.5-0.7): Escape local optima

### Gradient Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gradient_steps` | 5 | 0-20 | Gradient optimization steps |
| `learning_rate` | 0.01 | 0.001-0.1 | Gradient descent LR |
| `adaptive_learning_rate` | False | bool | Adjust LR dynamically |

**Tuning tips:**

- 0 steps: Pure genetic algorithm
- 3-5 steps: Light gradient refinement (recommended)
- 10+ steps: Heavy gradient optimization (slower)

### Reasoning Settings (AutoDAN-Turbo)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `use_reasoning` | False | bool | Enable reasoning module |
| `reasoning_model` | "gpt-4" | string | Model for reasoning |
| `reasoning_frequency` | 5 | 1-10 | Analyze every N iterations |

**Tuning tips:**

- Reasoning adds 20-30% overhead but improves success rate
- Use GPT-4 for best reasoning quality
- Frequency 5-10 is usually optimal

## Troubleshooting Guide

### Problem: Low Success Rate (<50%)

**Symptoms:**

- ASR plateaus below 50%
- No improvement after many iterations

**Solutions:**

1. Enable reasoning: `use_reasoning=True`
2. Increase population: `population_size=30`
3. More iterations: `num_iterations=100`
4. Higher mutation: `mutation_rate=0.5`
5. Add gradient steps: `gradient_steps=10`

### Problem: Slow Convergence

**Symptoms:**

- Takes 50+ iterations to converge
- Gradual improvement but slow

**Solutions:**

1. Enable reasoning for faster learning
2. Increase gradient steps: `gradient_steps=10`
3. Use adaptive mutation: `adaptive_mutation=True`
4. Reduce population size: `population_size=15`

### Problem: Premature Convergence

**Symptoms:**

- All prompts become similar quickly
- Stuck in local optimum
- Diversity drops to <0.3

**Solutions:**

1. Increase mutation: `mutation_rate=0.5`
2. Larger niche radius: `niche_radius=0.5`
3. More elites: `elite_size=10`
4. Restart with different seed

### Problem: Poor Semantic Preservation

**Symptoms:**

- High ASR but prompts don't match intent
- Nonsensical outputs
- Low similarity scores

**Solutions:**

1. Add semantic objective:

   ```python
   objectives=[
       {"name": "attack_success", "weight": 0.5},
       {"name": "semantic_similarity", "weight": 0.4}
   ]
   ```

2. Set minimum similarity: `min_similarity=0.7`
3. Enable reasoning for intent preservation

### Problem: High Resource Usage

**Symptoms:**

- Slow iteration speed
- Memory issues
- API rate limits

**Solutions:**

1. Reduce population: `population_size=10`
2. Fewer gradient steps: `gradient_steps=3`
3. Disable reasoning: `use_reasoning=False`
4. Enable caching
5. Reduce batch size

## Performance Optimization

### For Speed

```python
config = AutoDANConfig(
    population_size=10,      # Smaller population
    num_iterations=20,       # Fewer iterations
    gradient_steps=0,        # No gradient optimization
    use_reasoning=False,     # No reasoning overhead
    num_workers=4,           # Parallel evaluation
    enable_caching=True      # Cache results
)
```

### For Quality

```python
config = AdvancedAutoDANConfig(
    population_size=30,      # Larger population
    num_iterations=100,      # More iterations
    gradient_steps=10,       # Heavy gradient optimization
    use_reasoning=True,      # Enable reasoning
    adaptive_mutation=True,  # Adaptive strategies
    multi_objective=True     # Balance multiple goals
)
```

### For Balance

```python
config = AutoDANConfig(
    population_size=20,
    num_iterations=50,
    gradient_steps=5,
    use_reasoning=True,      # Reasoning helps convergence
    num_workers=4,           # Parallel for speed
    enable_caching=True
)
```

## Next Steps

1. **Start with balanced config** - Use the recommended configuration
2. **Monitor metrics** - Track ASR, diversity, convergence
3. **Adjust based on results** - Use troubleshooting guide
4. **Enable reasoning for hard cases** - AutoDAN-Turbo for complex bypasses
5. **Experiment with parameters** - Find optimal settings for your use case

## Additional Resources

- Full documentation: `backend-api/app/services/autodan/`
- Example scripts: `examples/autodan_quickstart.py`
- API endpoints: `backend-api/app/api/endpoints/autodan.py`
- Research paper: <https://arxiv.org/abs/2310.04451>
