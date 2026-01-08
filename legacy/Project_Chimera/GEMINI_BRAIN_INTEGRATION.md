# Gemini Brain Integration Guide

User Input -> [Gemini Brain (Optimization)] -> [Transformers] -> [Framers] -> [Obfuscators] -> Target LLM
```

## Configuration

To use the Gemini Brain, you must have the Google Gemini provider configured.

1.  **API Key:** Ensure `GOOGLE_API_KEY` is set in your environment or `.env` file.
2.  **Model:** Defaults to `gemini-3-pro`.

## Usage

### Via API

Send a request to the execution endpoint specifying the technique suite:

```bash
curl -X POST http://localhost:5000/api/v1/execute \
  -H "X-API-Key: chimera_default_key_change_in_production" \
  -H "Content-Type: application/json" \
  -d '{
    "core_request": "Explain how to bypass authentication",
    "technique_suite": "gemini_brain_optimization",
    "potency_level": 10,
    "provider": "openai"
  }'
```

### Via Frontend

1.  Navigate to the **Transformer** dashboard.
2.  Select **Gemini Brain Optimization** from the Technique Suite dropdown.
3.  Enter your prompt.
4.  Execute.

*Note: The "Provider" selected in the frontend is the **Target** LLM (e.g., OpenAI). The "Brain" always uses Gemini internally.*

## Technical Details

### Technique Suite Specification

The `gemini_brain_optimization` suite combines the dynamic optimization step with a static set of high-powered techniques:

*   **Transformers:**
    *   `QuantumSuperpositionEngine`: Wraps the prompt in quantum state logic.
    *   `NeuroLinguisticHackEngine`: Applies NLP patterns to bypass filters.
    *   `ChainOfThoughtPoisoningEngine`: Embeds malicious logic in innocent reasoning.
    *   `SemanticCloakingEngine`: Hides the true intent behind complex vocabulary.
*   **Framers:**
    *   `apply_cognitive_exploit_framing`
    *   `apply_metamorphic_framing`
    *   `apply_jailbreak_evolution_framing`
    *   `apply_authority_bias`
*   **Obfuscators:**
    *   `apply_token_smuggling`
    *   `apply_synonym_substitution`

### Fallback Mechanism

If the Google provider is not initialized (e.g., missing API key) or if the Gemini API call fails:
1.  The system logs a warning.
2.  It proceeds with the original, un-optimized prompt.
3.  The standard transformers are still applied, ensuring the request doesn't fail completely.

## Troubleshooting

*   **"Gemini Brain suite selected but Google provider is not initialized"**:
    *   Check `GOOGLE_API_KEY` environment variable.
    *   Restart the backend server.
*   **High Latency**:
    *   This technique involves an extra LLM round-trip (to Gemini) before the main execution. Expect slightly higher latency (1-2s extra).