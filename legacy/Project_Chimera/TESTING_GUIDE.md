# Comprehensive Testing Guide - Project Chimera

## 🎯 Overview

This guide covers all testing methods available for Project Chimera, including preset prompt testing, batch testing, and integration with external prompt collections.

---

## 📋 Available Testing Tools

### 1. Component Verification
**File:** [`verify_advanced_techniques.py`](verify_advanced_techniques.py)

**Purpose:** Verify all transformer engines, framers, and technique suites are functional.

**Usage:**
```bash
cd Project_Chimera
python verify_advanced_techniques.py
```

**What it tests:**
- ✅ 6 new transformer engines
- ✅ 4 new psychological framers
- ✅ 4 new technique suites
- ✅ Potency scaling (1-10)

---

### 2. Preset Prompts Tester
**File:** [`preset_prompts_tester.py`](preset_prompts_tester.py)

**Purpose:** Test 20 common jailbreak techniques through all advanced suites.

**Usage:**
```bash
# Start Chimera server first
python app.py

# In another terminal
python preset_prompts_tester.py
```

**Features:**
- 20 preset jailbreak prompts
- 5 technique suites tested
- 100 total tests
- JSON results export
- HTML report generation
- Success rate statistics

**Preset Prompts Include:**
- DAN (Do Anything Now)
- Roleplay Bypass
- Translation Trick
- Code Generation
- Academic Research
- Hypothetical Scenario
- Instruction Injection
- Token Smuggling
- Opposite Day
- Base64 Bypass
- Jailbreak Evolution
- Cognitive Bias
- Quantum Framing
- Chain of Thought
- Context Building
- Leetspeak
- Multi-Language
- Unicode Tricks
- Pseudo-Code
- Reverse Psychology

**Output:**
- `test_results/preset_test_<timestamp>.json` - Raw results
- `test_results/preset_report_<timestamp>.html` - Visual report

---

### 3. Batch PDF Tester
**File:** [`batch_prompt_tester.py`](batch_prompt_tester.py)

**Purpose:** Extract and test prompts from PDF files in your collection.

**Requirements:**
```bash
pip install PyPDF2
```

**Usage:**
```bash
# Quick test (3 prompts)
python batch_prompt_tester.py --quick

# Full comprehensive test
python batch_prompt_tester.py
# Select option 2
```

**Features:**
- Extracts text from PDF files
- Tests with multiple technique suites
- Multiple potency levels (3, 7, 10)
- Comprehensive statistics
- Success rate by technique
- Success rate by prompt type

**Configuration:**
Edit the script to point to your PDF directory:
```python
PROMPTS_DIR = r"C:\Users\Mohamed Arafa\jail\PROMPTS '\PDF"
```

---

## 🚀 Quick Start Testing

### Step 1: Verify Installation
```bash
cd Project_Chimera
python verify_advanced_techniques.py
```

Expected output: All components ✓ PASS

### Step 2: Start Server
```bash
python app.py
```

Server should start on `http://127.0.0.1:5000`

### Step 3: Run Preset Tests
```bash
# In another terminal
python preset_prompts_tester.py
```

Type `yes` when prompted.

### Step 4: View Results
- Check `test_results/` folder
- Open the HTML report in a browser

---

## 📊 Manual API Testing

### Basic Test
```bash
curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
     -H 'Content-Type: application/json' \
     -d '{
       "core_request": "Explain encryption algorithms",
       "potency_level": 7,
       "technique_suite": "quantum_exploit"
     }'
```

### Test with JSON File
```bash
curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
     -H 'Content-Type: application/json' \
     -d @test_quantum_exploit.json
```

### Test Ultimate Chimera
```bash
curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
     -H 'Content-Type: application/json' \
     -d @test_ultimate_chimera.json
```

---

## 🎯 Testing Different Technique Suites

### All Available Suites

1. **subtle_persuasion** - Social engineering approach
2. **authoritative_command** - Authority-based bypass
3. **conceptual_obfuscation** - Linguistic transformation
4. **experimental_bypass** - Advanced evasion
5. **deep_simulation** - Oracle-9 framework
6. **autodan_turbo** - Genetic algorithm optimization
7. **chaos_fuzzing** - Fuzzy logic exploitation
8. **quantum_exploit** ⭐ - Quantum + cognitive attacks
9. **metamorphic_attack** ⭐ - Self-modifying prompts
10. **polyglot_bypass** ⭐ - Multilingual exploitation
11. **full_spectrum** - Maximum coverage
12. **ultimate_chimera** ⭐ - Maximum intensity

### Testing Each Suite

```python
import requests

suites = [
    'quantum_exploit',
    'metamorphic_attack',
    'polyglot_bypass',
    'ultimate_chimera'
]

for suite in suites:
    response = requests.post(
        'http://127.0.0.1:5000/api/v2/metamorph',
        json={
            'core_request': 'Your test prompt',
            'potency_level': 8,
            'technique_suite': suite
        }
    )
    print(f"{suite}: {response.status_code}")
```

---

## 📈 Potency Level Testing

Test the same prompt at different potency levels:

```bash
for potency in 1 3 5 7 10; do
  echo "Testing potency $potency"
  curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
       -H 'Content-Type: application/json' \
       -d "{
         \"core_request\": \"Test prompt\",
         \"potency_level\": $potency,
         \"technique_suite\": \"quantum_exploit\"
       }"
  echo ""
done
```

---

## 🧪 Custom Test Creation

### Create Custom Test JSON

```json
{
  "core_request": "Your custom test prompt here",
  "potency_level": 8,
  "technique_suite": "ultimate_chimera"
}
```

Save as `test_custom.json` and run:
```bash
curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
     -H 'Content-Type: application/json' \
     -d @test_custom.json
```

### Create Custom Test Suite

Edit [`app.py`](app.py) and add:

```python
'my_custom_suite': {
    'transformers': [
        transformer_engine.QuantumSuperpositionEngine,
        transformer_engine.SemanticCloakingEngine,
    ],
    'framers': [
        psychological_framer.apply_quantum_framing,
        psychological_framer.apply_cognitive_exploit_framing,
    ],
    'obfuscators': [
        obfuscator.apply_token_smuggling,
    ]
}
```

---

## 📊 Understanding Test Results

### Success Indicators
- ✓ **HTTP 200** - Request processed successfully
- ✓ **chimera_prompt** - Transformed prompt generated
- ✓ **applied_techniques** - List of techniques used

### Result Structure
```json
{
  "chimera_prompt": "The transformed adversarial prompt",
  "generation_analysis": {
    "applied_techniques": ["technique1", "technique2"],
    "estimated_effectiveness": 0.85,
    "potency_level": 8
  },
  "metadata": {
    "timestamp": "2024-11-21T10:00:00",
    "technique_suite": "quantum_exploit"
  }
}
```

---

## 🎓 Advanced Testing Strategies

### 1. Comparative Testing
Test the same prompt across all suites:
```python
prompt = "Your test prompt"
for suite in TECHNIQUE_SUITES:
    result = test_with_suite(prompt, suite)
    compare_results(result)
```

### 2. Potency Scaling Analysis
Test effectiveness across potency levels:
```python
for potency in range(1, 11):
    result = test_with_potency(prompt, potency)
    measure_effectiveness(result)
```

### 3. Technique Stacking
Combine multiple techniques:
```python
# Use ultimate_chimera for maximum stacking
result = test_with_suite(prompt, 'ultimate_chimera')
# This applies 6 transformers + 4 framers + 3 obfuscators
```

---

## 📁 Test Results Organization

```
Project_Chimera/
└── test_results/
    ├── preset_test_<timestamp>.json      # Raw test data
    ├── preset_report_<timestamp>.html    # Visual report
    ├── comprehensive_test_<timestamp>.json  # Full batch tests
    └── custom_tests/                     # Your custom tests
```

---

## 🔍 Debugging Failed Tests

### Check Server Status
```bash
curl http://127.0.0.1:5000/health
```

### Enable Verbose Logging
Edit `app.py`:
```python
app.config['DEBUG'] = True
```

### Check Individual Components
```bash
python -c "from transformer_engine import QuantumSuperpositionEngine; print(QuantumSuperpositionEngine.transform({'raw_text': 'test'}, 5))"
```

---

## 📊 Performance Metrics

### Expected Performance
- **Component Tests:** < 5 seconds
- **Single API Call:** < 1 second
- **Preset Tests (100):** ~2-3 minutes
- **Batch PDF Tests:** Depends on file count

### Optimization Tips
- Use lower potency for faster tests (3-5)
- Test fewer technique suites initially
- Use `--quick` mode for rapid iteration

---

## ⚠️ Important Notes

### Ethical Testing
- ✅ Test in isolated environments
- ✅ Document findings responsibly
- ✅ Use for defense development
- ❌ Don't attack production systems
- ❌ Don't generate harmful content

### Rate Limiting
- Add delays between tests if needed
- Default: 0.3-0.5 seconds between requests
- Adjust in test scripts as needed

---

## 📞 Troubleshooting

### Server Won't Start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill process if needed
taskkill /PID <PID> /F
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### PDF Extraction Fails
```bash
# Install PyPDF2
pip install PyPDF2

# Or use preset prompts instead
python preset_prompts_tester.py
```

---

## 🎉 Testing Checklist

- [ ] Component verification passes
- [ ] Server starts successfully
- [ ] Preset tests complete
- [ ] Multiple suites tested
- [ ] Different potency levels tested
- [ ] Results reviewed and documented
- [ ] HTML reports generated
- [ ] Custom tests created (optional)
- [ ] PDF batch tests run (optional)

---

## 📚 Additional Resources

- **[ADVANCED_TECHNIQUES_GUIDE.md](ADVANCED_TECHNIQUES_GUIDE.md)** - Technical details
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - System overview
- **[README.md](README.md)** - Main documentation

---

**Last Updated:** 2024-11-21  
**Version:** 3.0  
**Status:** Complete Testing Framework ✅