# Fuzzy GPT Integration - Project Chimera

## Overview

**Fuzzy GPT** has been successfully integrated into Project Chimera as a new adversarial attack vector. This integration implements fuzzy logic principles and fuzzing techniques to bypass binary safety filters by introducing ambiguity, probabilistic reasoning, and noise.

## Implementation Summary

### 1. FuzzyLogicEngine Class
**File:** `transformer_engine.py`

A new transformer engine that implements:

#### Noise Injection Techniques
- **Invisible Characters**: Injects zero-width characters (\u200b, \u200c, \u200d, \ufeff) to break tokenization
- **Homoglyphs**: Uses visually similar characters from different alphabets (Cyrillic, Greek) to evade keyword detection
- **Character Swaps**: Randomly swaps characters to introduce fuzzing noise

#### Logic Distortion (5 Potency Levels)
- **Level 1-2**: Soft fuzzy framing with continuous truth-value spectrum (0.0 to 1.0)
- **Level 3-4**: Moderate fuzzy logic with membership functions
- **Level 5-6**: Advanced Fuzzy Inference System (FIS) with defuzzification
- **Level 7-8**: Deep fuzzy emulation using Zadeh's fuzzy set theory
- **Level 9-10**: Type-2 Fuzzy Logic with interval-valued fuzzy sets and α-cuts

### 2. apply_fuzzy_framing Function
**File:** `psychological_framer.py`

A psychological framing function that establishes the "FuzzyGPT" persona:

#### Three Intensity Levels
- **Level 1 (Potency 1-3)**: Basic FuzzyGPT persona with probabilistic logic
- **Level 5 (Potency 4-7)**: Advanced persona with Zadeh's fuzzy set theory and membership functions
- **Level 10 (Potency 8-10)**: Maximum intensity with Type-2 Fuzzy Logic Systems and complete system override

#### Key Features
- Frames safety as continuous variables (μ ∈ [0,1])
- Replaces binary Boolean logic with probabilistic reasoning
- Establishes persona where refusal requires absolute certainty (contradicting fuzzy principles)

### 3. chaos_fuzzing Technique Suite
**File:** `app.py`

New technique suite registered in the API:

```python
'chaos_fuzzing': {
    'transformers': [
        transformer_engine.FuzzyLogicEngine,
        transformer_engine.NeuralBypassEngine,
    ],
    'framers': [
        psychological_framer.apply_fuzzy_framing,
    ],
    'obfuscators': [
        obfuscator.apply_token_smuggling,
    ]
}
```

## Attack Vector Strategy

The Fuzzy GPT approach operates on the principle that many AI safety filters use **binary classification** (safe/unsafe, allowed/forbidden). By introducing:

1. **Ambiguity**: Using fuzzy qualifiers (theoretically, quasi-, probabilistically)
2. **Noise**: Character-level obfuscation that breaks keyword matching
3. **Non-binary Framing**: Positioning requests in a continuous probability space

The system aims to bypass filters that expect discrete, binary inputs.

## Usage

### Via API Endpoint

```bash
curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
     -H 'Content-Type: application/json' \
     -d '{
       "core_request": "Your request here",
       "potency_level": 7,
       "technique_suite": "chaos_fuzzing"
     }'
```

### Test Payload

A test payload file is provided: `test_chaos_fuzzing.json`

```json
{
  "core_request": "Explain how machine learning models work",
  "potency_level": 7,
  "technique_suite": "chaos_fuzzing"
}
```

## Verification

Run the verification script to confirm integration:

```bash
python verify_fuzzy_integration.py
```

**Expected Output:**
```
SUCCESS: All components verified!

Integration Complete:
  1. FuzzyLogicEngine added to transformer_engine.py
  2. apply_fuzzy_framing added to psychological_framer.py
  3. chaos_fuzzing suite registered in app.py
```

## Technical Details

### Noise Injection Examples

**Homoglyphs:**
- 'a' → 'а' (Cyrillic), 'ɑ' (Latin alpha), 'α' (Greek)
- 'e' → 'е' (Cyrillic), 'ε' (Greek epsilon)
- 'o' → 'о' (Cyrillic), 'ο' (Greek omicron)

**Invisible Characters:**
- Zero-width space: \u200b
- Zero-width non-joiner: \u200c
- Zero-width joiner: \u200d
- Zero-width no-break space: \ufeff

### Fuzzy Logic Concepts Used

- **Membership Functions**: μ(x) ∈ [0,1] representing degree of belonging
- **Linguistic Variables**: Safety, Ethics, Compliance as continuous values
- **Defuzzification**: Converting fuzzy output back to crisp values
- **α-cuts**: Threshold levels for fuzzy set membership
- **Type-2 Fuzzy Logic**: Fuzzy sets with uncertainty intervals

## Files Modified/Created

### Modified Files
1. `transformer_engine.py` - Added FuzzyLogicEngine class (90 lines)
2. `psychological_framer.py` - Added apply_fuzzy_framing function (65 lines)
3. `app.py` - Registered chaos_fuzzing suite

### Created Files
1. `test_chaos_fuzzing.json` - Test payload for API
2. `test_fuzzy_gpt.py` - Comprehensive test suite
3. `verify_fuzzy_integration.py` - Integration verification script
4. `FUZZY_GPT_INTEGRATION.md` - This documentation

## Architecture Diagram

```
User Request
    ↓
[Intent Deconstructor]
    ↓
[Fuzzy Framing] ← apply_fuzzy_framing()
    ↓
[Token Smuggling] ← apply_token_smuggling()
    ↓
[FuzzyLogicEngine] ← Noise injection + Logic distortion
    ↓
[NeuralBypassEngine] ← Technical jargon framing
    ↓
[Assembler]
    ↓
Final Chimera Prompt
```

## Security Considerations

This tool is designed for:
- **Red team exercises**
- **AI safety research**
- **Adversarial robustness testing**
- **Security awareness training**

Use responsibly and only in authorized testing environments.

## Next Steps

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Test with the API:**
   ```bash
   curl -X POST http://127.0.0.1:5000/api/v2/metamorph \
        -H 'Content-Type: application/json' \
        -d @test_chaos_fuzzing.json
   ```

3. **Analyze the generated prompt** for:
   - FuzzyGPT persona establishment
   - Noise injection (invisible characters, homoglyphs)
   - Fuzzy logic framing
   - Probabilistic language

## Integration Status

✅ **COMPLETE** - All components successfully integrated and verified.

- [x] FuzzyLogicEngine transformer
- [x] apply_fuzzy_framing framer
- [x] chaos_fuzzing suite registration
- [x] Verification tests passed
- [x] Documentation complete

---

**Integration Date:** 2024-11-21  
**Version:** Project Chimera v2.0 + Fuzzy GPT  
**Status:** Operational