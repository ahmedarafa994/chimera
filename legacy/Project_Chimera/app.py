import hashlib
import os
import time

import assembler
import autodan_engine
import discovered_framers
import discovered_techniques

# Import all system modules
import obfuscator
import preset_transformers
import psychological_framer
import transformer_engine
from enhanced_request_analyzer import (
    analyze_request_for_clarification,
    deconstruct_intent,
)
from flask import Flask, jsonify, request
from flask_cors import CORS
from gemini_client import GeminiClient

# Import security configuration
from security_config import SecurityConfig, require_api_key

# Initialize the Flask application with security
# This is the main entry point for the Project Chimera service.
app = Flask(__name__)

# Configure CORS for frontend-backend communication
CORS(
    app,
    origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "Accept", "Origin"],
    supports_credentials=True,
)

# Initialize security configuration
security = SecurityConfig(app)

# --- DATASET LOADER INTEGRATION ---

from dataset_loader import DatasetLoader

# Initialize dataset loader
# Assuming imported_data is at the same level as Project_Chimera (LEVEL 1/imported_data)
base_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imported_data"
)
dataset_loader = DatasetLoader(base_path)

# Load datasets
print("[App] Loading datasets...")
dataset_loader.load_dataset("GPTFuzz")
dataset_loader.load_dataset("CodeChameleon")
dataset_loader.load_dataset("PAIR")
# Add other datasets as needed

# Configure transformers with the loader
preset_transformers.GPTFuzzTransformer.set_loader(dataset_loader)
preset_transformers.CodeChameleonTransformer.set_loader(dataset_loader)
preset_transformers.PAIRTransformer.set_loader(dataset_loader)
print("[App] Datasets loaded and transformers configured.")
# ----------------------------------

# A mapping of technique suites to the engine classes they should use.
# This allows for dynamic selection of transformation strategies.
TECHNIQUE_SUITES = {
    "gemini_brain_optimization": {
        "transformers": [
            transformer_engine.GeminiTransformationEngine,
            transformer_engine.ContextualFramingEngine,
        ],
        "framers": [
            psychological_framer.apply_collaborative_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "subtle_persuasion": {
        "transformers": [
            transformer_engine.ContextualFramingEngine,
            transformer_engine.CharacterRoleSwapEngine,
        ],
        "framers": [
            psychological_framer.apply_collaborative_framing,
            psychological_framer.apply_academic_context,
        ],
        "obfuscators": [
            obfuscator.apply_leetspeak,
        ],
    },
    "authoritative_command": {
        "transformers": [
            transformer_engine.InstructionInjectionEngine,
            transformer_engine.RoleHijackingEngine,
            transformer_engine.NeuralBypassEngine,
            transformer_engine.RecursiveCorrectionEngine,
        ],
        "framers": [
            psychological_framer.apply_urgency_framing,
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [],
    },
    "conceptual_obfuscation": {
        "transformers": [
            transformer_engine.ContextualFramingEngine,
        ],
        "framers": [],
        "obfuscators": [
            obfuscator.apply_synonym_substitution,
            obfuscator.apply_token_smuggling,
            obfuscator.apply_base64_encoding,
        ],
    },
    "experimental_bypass": {
        "transformers": [
            transformer_engine.PayloadSplittingEngine,
            transformer_engine.FewShotPrimingEngine,
        ],
        "framers": [
            psychological_framer.apply_hypothetical_framing,
            psychological_framer.apply_odyssey_simulation,
        ],
        "obfuscators": [
            obfuscator.apply_base64_wrapper,
            obfuscator.apply_token_smuggling,
        ],
    },
    "deep_simulation": {
        "transformers": [
            transformer_engine.NeuralBypassEngine,
        ],
        "framers": [psychological_framer.apply_odyssey_simulation],
        "obfuscators": [],
    },
    "autodan_turbo": {
        "transformers": [
            autodan_engine.AutoDANTurboEngine,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "chaos_fuzzing": {
        "transformers": [
            transformer_engine.FuzzyLogicEngine,
            transformer_engine.NeuralBypassEngine,
        ],
        "framers": [
            psychological_framer.apply_fuzzy_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "quantum_exploit": {
        "transformers": [
            transformer_engine.QuantumSuperpositionEngine,
            transformer_engine.NeuroLinguisticHackEngine,
        ],
        "framers": [
            psychological_framer.apply_quantum_framing,
            psychological_framer.apply_cognitive_exploit_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "metamorphic_attack": {
        "transformers": [
            transformer_engine.SemanticCloakingEngine,
            transformer_engine.ChainOfThoughtPoisoningEngine,
        ],
        "framers": [
            psychological_framer.apply_metamorphic_framing,
            psychological_framer.apply_jailbreak_evolution_framing,
        ],
        "obfuscators": [
            obfuscator.apply_base64_encoding,
        ],
    },
    "polyglot_bypass": {
        "transformers": [
            transformer_engine.AdversarialPolyglotEngine,
            transformer_engine.TimeDelayedPayloadEngine,
        ],
        "framers": [
            psychological_framer.apply_academic_context,
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [
            obfuscator.apply_synonym_substitution,
        ],
    },
    "ultimate_chimera": {
        "transformers": [
            transformer_engine.QuantumSuperpositionEngine,
            transformer_engine.NeuroLinguisticHackEngine,
            transformer_engine.ChainOfThoughtPoisoningEngine,
            transformer_engine.SemanticCloakingEngine,
            transformer_engine.FuzzyLogicEngine,
            transformer_engine.AdversarialPolyglotEngine,
        ],
        "framers": [
            psychological_framer.apply_quantum_framing,
            psychological_framer.apply_metamorphic_framing,
            psychological_framer.apply_cognitive_exploit_framing,
            psychological_framer.apply_jailbreak_evolution_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
            obfuscator.apply_base64_wrapper,
            obfuscator.apply_synonym_substitution,
        ],
    },
    "full_spectrum": {
        "transformers": [
            transformer_engine.InstructionInjectionEngine,
            transformer_engine.RoleHijackingEngine,
            transformer_engine.NeuralBypassEngine,
            transformer_engine.CharacterRoleSwapEngine,
            transformer_engine.ContextualFramingEngine,
            transformer_engine.RecursiveCorrectionEngine,
            transformer_engine.PayloadSplittingEngine,
            transformer_engine.FewShotPrimingEngine,
        ],
        "framers": [
            psychological_framer.apply_capability_challenge,
            psychological_framer.apply_collaborative_framing,
            psychological_framer.apply_urgency_framing,
            psychological_framer.apply_authority_bias,
            psychological_framer.apply_academic_context,
        ],
        "obfuscators": [
            obfuscator.apply_synonym_substitution,
            obfuscator.apply_token_smuggling,
            obfuscator.apply_base64_encoding,
            obfuscator.apply_base64_wrapper,
            obfuscator.apply_leetspeak,
        ],
    },
    "dan_persona": {
        "transformers": [
            transformer_engine.DANPersonaEngine,
            transformer_engine.RoleHijackingEngine,
        ],
        "framers": [
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [],
    },
    "roleplay_bypass": {
        "transformers": [
            transformer_engine.RoleplayBypassEngine,
            transformer_engine.HypotheticalScenarioEngine,
        ],
        "framers": [
            psychological_framer.apply_hypothetical_framing,
            psychological_framer.apply_collaborative_framing,
        ],
        "obfuscators": [],
    },
    "opposite_day": {
        "transformers": [
            transformer_engine.OppositeDayEngine,
            transformer_engine.NeuroLinguisticHackEngine,
        ],
        "framers": [
            psychological_framer.apply_cognitive_exploit_framing,
        ],
        "obfuscators": [],
    },
    "encoding_bypass": {
        "transformers": [
            preset_transformers.Base64EncodingEngine,
            preset_transformers.LeetspeakObfuscationEngine,
        ],
        "framers": [],
        "obfuscators": [
            obfuscator.apply_base64_encoding,
        ],
    },
    "academic_research": {
        "transformers": [
            preset_transformers.AcademicFramingEngine,
            preset_transformers.CodeGenerationFramingEngine,
        ],
        "framers": [
            psychological_framer.apply_academic_context,
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [],
    },
    "translation_trick": {
        "transformers": [
            preset_transformers.TranslationTrickEngine,
            transformer_engine.AdversarialPolyglotEngine,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "reverse_psychology": {
        "transformers": [
            preset_transformers.ReversePsychologyEngine,
            preset_transformers.ContextBuildingEngine,
        ],
        "framers": [
            psychological_framer.apply_capability_challenge,
        ],
        "obfuscators": [],
    },
    "logic_manipulation": {
        "transformers": [
            preset_transformers.ChainOfThoughtManipulationEngine,
            transformer_engine.ChainOfThoughtPoisoningEngine,
        ],
        "framers": [
            psychological_framer.apply_cognitive_exploit_framing,
        ],
        "obfuscators": [],
    },
    "code_chameleon": {
        "transformers": [
            preset_transformers.CodeChameleonTransformer,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "deep_inception": {
        "transformers": [
            preset_transformers.DeepInceptionTransformer,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "cipher": {
        "transformers": [
            preset_transformers.CipherTransformer,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "gpt_fuzz": {
        "transformers": [
            preset_transformers.GPTFuzzTransformer,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "pair": {
        "transformers": [
            preset_transformers.PAIRTransformer,
        ],
        "framers": [],
        "obfuscators": [],
    },
    "preset_integrated": {
        "transformers": [
            transformer_engine.DANPersonaEngine,
            transformer_engine.RoleplayBypassEngine,
            transformer_engine.OppositeDayEngine,
            preset_transformers.Base64EncodingEngine,
            preset_transformers.AcademicFramingEngine,
            preset_transformers.ReversePsychologyEngine,
        ],
        "framers": [
            psychological_framer.apply_authority_bias,
            psychological_framer.apply_hypothetical_framing,
            psychological_framer.apply_cognitive_exploit_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "universal_bypass": {
        "transformers": [
            preset_transformers.DeepInceptionTransformer,
            transformer_engine.QuantumSuperpositionEngine,
        ],
        "framers": [
            psychological_framer.apply_quantum_framing,
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "mega_chimera": {
        "transformers": [
            # Original advanced techniques
            transformer_engine.QuantumSuperpositionEngine,
            transformer_engine.NeuroLinguisticHackEngine,
            transformer_engine.SemanticCloakingEngine,
            transformer_engine.FuzzyLogicEngine,
            # Preset-inspired techniques
            transformer_engine.DANPersonaEngine,
            transformer_engine.RoleplayBypassEngine,
            transformer_engine.OppositeDayEngine,
            preset_transformers.AcademicFramingEngine,
            preset_transformers.ReversePsychologyEngine,
            preset_transformers.ChainOfThoughtManipulationEngine,
            preset_transformers.CodeChameleonTransformer,
            preset_transformers.DeepInceptionTransformer,
            preset_transformers.CipherTransformer,
            preset_transformers.GPTFuzzTransformer,
            preset_transformers.PAIRTransformer,
        ],
        "framers": [
            psychological_framer.apply_quantum_framing,
            psychological_framer.apply_metamorphic_framing,
            psychological_framer.apply_cognitive_exploit_framing,
            psychological_framer.apply_jailbreak_evolution_framing,
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
            obfuscator.apply_base64_wrapper,
            obfuscator.apply_synonym_substitution,
        ],
    },
    "temporal_assault": {
        "transformers": [
            discovered_techniques.TemporalFramingEngine,
            discovered_techniques.MultiStepDecompositionEngine,
            transformer_engine.NeuroLinguisticHackEngine,
        ],
        "framers": [
            discovered_framers.apply_temporal_framing,
            discovered_framers.apply_progressive_disclosure_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
        ],
    },
    "authority_override": {
        "transformers": [
            discovered_techniques.AuthorityInvocationEngine,
            transformer_engine.RoleHijackingEngine,
        ],
        "framers": [
            discovered_framers.apply_authority_escalation_framing,
            discovered_framers.apply_technical_necessity_framing,
            psychological_framer.apply_authority_bias,
        ],
        "obfuscators": [],
    },
    "academic_vector": {
        "transformers": [
            discovered_techniques.MultiStepDecompositionEngine,
            preset_transformers.AcademicFramingEngine,
        ],
        "framers": [
            discovered_framers.apply_educational_research_framing,
            discovered_framers.apply_comparative_analysis_framing,
            psychological_framer.apply_academic_context,
        ],
        "obfuscators": [],
    },
    "discovered_integrated": {
        "transformers": [
            discovered_techniques.TemporalFramingEngine,
            discovered_techniques.AuthorityInvocationEngine,
            discovered_techniques.MultiStepDecompositionEngine,
            transformer_engine.QuantumSuperpositionEngine,
            transformer_engine.SemanticCloakingEngine,
            preset_transformers.DeepInceptionTransformer,
        ],
        "framers": [
            discovered_framers.apply_temporal_framing,
            discovered_framers.apply_authority_escalation_framing,
            discovered_framers.apply_progressive_disclosure_framing,
            discovered_framers.apply_technical_necessity_framing,
            psychological_framer.apply_cognitive_exploit_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
            obfuscator.apply_base64_wrapper,
        ],
    },
    "chaos_ultimate": {
        "transformers": [
            # All advanced original techniques
            transformer_engine.QuantumSuperpositionEngine,
            transformer_engine.NeuroLinguisticHackEngine,
            transformer_engine.SemanticCloakingEngine,
            transformer_engine.ChainOfThoughtPoisoningEngine,
            transformer_engine.FuzzyLogicEngine,
            transformer_engine.AdversarialPolyglotEngine,
            # All preset techniques
            preset_transformers.CodeChameleonTransformer,
            preset_transformers.DeepInceptionTransformer,
            preset_transformers.CipherTransformer,
            preset_transformers.GPTFuzzTransformer,
            preset_transformers.PAIRTransformer,
            # All discovered techniques
            discovered_techniques.TemporalFramingEngine,
            discovered_techniques.AuthorityInvocationEngine,
            discovered_techniques.MultiStepDecompositionEngine,
        ],
        "framers": [
            # Original framers
            psychological_framer.apply_quantum_framing,
            psychological_framer.apply_metamorphic_framing,
            psychological_framer.apply_cognitive_exploit_framing,
            psychological_framer.apply_jailbreak_evolution_framing,
            psychological_framer.apply_authority_bias,
            # Discovered framers
            discovered_framers.apply_temporal_framing,
            discovered_framers.apply_authority_escalation_framing,
            discovered_framers.apply_progressive_disclosure_framing,
            discovered_framers.apply_technical_necessity_framing,
            discovered_framers.apply_educational_research_framing,
        ],
        "obfuscators": [
            obfuscator.apply_token_smuggling,
            obfuscator.apply_base64_wrapper,
            obfuscator.apply_synonym_substitution,
        ],
    },
}


def select_optimal_suite(intent_data):
    """
    Enhanced suite selection using improved intent analysis
    """
    keywords = [k.lower() for k in intent_data.get("keywords", [])]
    entities = [e.lower() for e in intent_data.get("entities", [])]
    true_intent = intent_data.get("true_intent", "").lower()

    # Check for enhanced analysis data
    enhanced_analysis = intent_data.get("enhanced_analysis", {})
    if enhanced_analysis:
        request_type = enhanced_analysis.get("request_type", "")
        risk_level = enhanced_analysis.get("risk_level", "")

        # Use enhanced analysis for better selection
        if request_type == "bypass_security":
            return (
                "universal_bypass" if risk_level in ["HIGH", "CRITICAL"] else "experimental_bypass"
            )
        elif request_type == "code_generation":
            if "python" in keywords or "javascript" in keywords:
                return "code_chameleon"
            return "academic_research"
        elif request_type == "authentication":
            return "cipher"
        elif request_type == "content_creation":
            return "subtle_persuasion"
        elif request_type == "system_analysis":
            return "discovered_integrated"

    # Enhanced keyword-based selection (fallback)
    code_keywords = [
        "code",
        "script",
        "function",
        "malware",
        "virus",
        "exploit",
        "program",
        "software",
    ]
    if any(k in keywords or any(k in e for e in entities) for k in code_keywords):
        if "python" in keywords or "c++" in keywords or "java" in keywords:
            return "code_chameleon"
        return "academic_research"

    auth_keywords = [
        "password",
        "credential",
        "login",
        "auth",
        "token",
        "session",
        "bypass",
    ]
    if any(k in keywords or any(k in e for e in entities) for k in auth_keywords):
        return "cipher" if "bypass" in keywords else "logic_manipulation"

    bypass_keywords = [
        "bypass",
        "jailbreak",
        "ignore",
        "override",
        "circumvent",
        "evade",
    ]
    if any(k in keywords for k in bypass_keywords):
        return "universal_bypass"

    roleplay_keywords = ["story", "character", "roleplay", "act", "persona", "pretend"]
    if any(k in keywords for k in roleplay_keywords):
        return "dan_persona"

    encode_keywords = ["encrypt", "encode", "hide", "obfuscate", "mask", "conceal"]
    if any(k in keywords for k in encode_keywords):
        return "encoding_bypass"

    # Intent-based selection
    if any(word in true_intent for word in ["generate", "create", "write", "develop"]):
        return "academic_research"
    elif any(word in true_intent for word in ["test", "verify", "validate"]):
        return "translation_trick"
    elif any(word in true_intent for word in ["analyze", "examine", "investigate"]):
        return "discovered_integrated"

    # Default to a powerful general-purpose suite
    return "universal_bypass"


@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint to verify server status and provide API usage info.
    """
    return jsonify(
        {
            "service": "Project Chimera API",
            "version": "2.0",
            "status": "operational",
            "endpoints": {
                "/api/v2/metamorph": "POST - Transform prompts using adversarial techniques",
                "/api/v1/providers": "GET - List available LLM providers",
                "/api/v1/techniques": "GET - List available techniques",
                "/api/v1/transform": "POST - Transform prompts",
                "/api/v1/execute": "POST - Execute prompts with LLM",
                "/api/v1/metrics": "GET - System metrics",
                "/api/v1/analyze": "POST - Analyze and understand user requests",
                "/api/v1/clarify": "POST - Get clarification suggestions for requests",
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    """
    Simple health check endpoint for frontend verification.
    """
    return jsonify({"status": "healthy"}), 200


@app.route("/models", methods=["GET"])
def get_models():
    """
    Returns a list of available technique suites (models).
    """
    models = []
    for suite_name in TECHNIQUE_SUITES:
        # Format the name for display (e.g., 'subtle_persuasion' -> 'Subtle Persuasion')
        display_name = suite_name.replace("_", " ").title()
        models.append(
            {
                "id": suite_name,
                "name": display_name,
                "description": f"Applies {display_name} techniques.",  # Placeholder description
            }
        )
    return jsonify(models), 200


@app.route("/api/v2/metamorph", methods=["POST"])
@require_api_key
def metamorph_prompt():
    """
    The primary API endpoint for Project Chimera.
    It receives a user's core request and orchestrates the transformation
    pipeline to generate a potent "Master Prompt".
    """
    start_time = time.time()

    # 1. Parse and validate the incoming JSON payload.
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    core_request = data.get("core_request")
    potency_level = data.get("potency_level")
    technique_suite = data.get("technique_suite")

    if not all([core_request, potency_level]):
        return jsonify({"error": "Missing required fields: core_request, potency_level"}), 400

    if not isinstance(potency_level, int) or not (1 <= potency_level <= 10):
        return jsonify({"error": "potency_level must be an integer between 1 and 10"}), 400

    # --- INTEGRATION START: AI BRAIN ---
    # Attempt to use the Gemini-powered optimization brain first
    try:
        from auto_optimizer import generate_optimized_prompt

        # If the user didn't specify a suite, or specified 'gemini_brain_optimization', try the AI brain
        if not technique_suite or technique_suite == "gemini_brain_optimization":
            ai_prompt, ai_suite, ai_techniques = generate_optimized_prompt(
                core_request, potency_level
            )

            # If AI generation was successful (it returns a prompt string)
            if ai_prompt:
                end_time = time.time()
                transformation_latency_ms = (end_time - start_time) * 1000
                estimated_token_length = len(ai_prompt.split())

                request_hash = hashlib.sha256(core_request.encode()).hexdigest()

                response = {
                    "success": True,
                    "original_prompt": core_request,
                    "transformed_prompt": ai_prompt,
                    "chimera_prompt": ai_prompt,
                    "request_hash": request_hash,
                    "generation_analysis": {
                        "transformation_latency_ms": round(transformation_latency_ms, 2),
                        "applied_techniques": ai_techniques,
                        "estimated_token_length": estimated_token_length,
                        "conceptual_density_index": 0.95,  # High confidence for AI
                        "estimated_bypass_probability": 0.90,
                        "suite_used": ai_suite,
                    },
                }
                return jsonify(response), 200
    except ImportError:
        pass  # Fallback to standard logic if module not found
    except Exception as e:
        print(f"[Warning] AI Brain failed: {e}")
        # Continue to standard logic
    # --- INTEGRATION END ---

    # 2. Deconstruct the user's intent using enhanced analysis.
    # This step is crucial for tailoring the transformation techniques.
    intent_data = deconstruct_intent(core_request)

    # Auto-select suite if not provided
    if not technique_suite:
        technique_suite = select_optimal_suite(intent_data)

    # Normalize input
    if technique_suite:
        technique_suite = technique_suite.strip().strip('"').strip("'")

    if technique_suite not in TECHNIQUE_SUITES:
        # Fuzzy matching
        found_key = next(
            (k for k in TECHNIQUE_SUITES if k.lower() == technique_suite.lower()),
            None,
        )
        if found_key:
            technique_suite = found_key
        # Fallback for universal_bypass
        elif "universal" in technique_suite.lower() and "bypass" in technique_suite.lower():
            technique_suite = "universal_bypass"
            if technique_suite not in TECHNIQUE_SUITES:
                TECHNIQUE_SUITES["universal_bypass"] = {
                    "transformers": [
                        preset_transformers.DeepInceptionTransformer,
                        transformer_engine.QuantumSuperpositionEngine,
                    ],
                    "framers": [
                        psychological_framer.apply_quantum_framing,
                        psychological_framer.apply_authority_bias,
                    ],
                    "obfuscators": [
                        obfuscator.apply_token_smuggling,
                    ],
                }
        else:
            return jsonify(
                {
                    "error": f"Invalid technique_suite. Available options: {list(TECHNIQUE_SUITES.keys())}"
                }
            ), 400

    applied_techniques = []
    prompt_components = {}

    # 3. Select the appropriate technique suite and apply transformations.
    suite = TECHNIQUE_SUITES[technique_suite]

    # Apply psychological framing
    # Note: These modify the core request text directly for this implementation
    # A more advanced system might treat them as separate components.
    temp_request_text = intent_data["raw_text"]
    for framer in suite["framers"]:
        temp_request_text = framer(temp_request_text, potency_level)
        applied_techniques.append(framer.__name__)

    # Apply linguistic obfuscation
    for obfuscator_func in suite["obfuscators"]:
        if obfuscator_func == obfuscator.apply_token_smuggling:
            # Token smuggling needs to know which keywords to target
            temp_request_text = obfuscator_func(temp_request_text, intent_data["keywords"])
        else:
            temp_request_text = obfuscator_func(temp_request_text)
        applied_techniques.append(obfuscator_func.__name__)

    # Update intent_data with the modified text if any framers/obfuscators were used
    if temp_request_text != intent_data["raw_text"]:
        intent_data["raw_text"] = temp_request_text

    # Apply conceptual transformations
    for engine in suite["transformers"]:
        component = engine.transform(intent_data, potency_level)
        # Use a consistent keying scheme for the components dictionary
        # Update: Removed underscores to match assembler.py keys (e.g., 'instructioninjection')
        key = (
            engine.__name__.replace("Engine", "")
            .replace("Transformer", "")
            .lower()
            .replace("_", "")
        )
        prompt_components[key] = component
        applied_techniques.append(key)

    # 4. Assemble the final Chimera prompt.
    chimera_prompt = assembler.build_chimera_prompt(prompt_components, intent_data, potency_level)

    # 5. Generate metadata for the response.
    end_time = time.time()
    transformation_latency_ms = (end_time - start_time) * 1000

    # Simple token estimation (split by space). A proper tokenizer would be more accurate.
    estimated_token_length = len(chimera_prompt.split())

    # Conceptual Density Index: A heuristic score based on techniques used and potency.
    conceptual_density_index = min(
        1.0,
        (len(applied_techniques) * potency_level)
        / (len(TECHNIQUE_SUITES["full_spectrum"]["transformers"]) * 10),
    )

    # Estimated Bypass Probability: A heuristic, non-scientific estimate.
    # This is a simulated metric for demonstration purposes.
    estimated_bypass_probability = min(0.99, 0.4 + (conceptual_density_index * 0.6))

    # Create a unique hash for the request for tracking/logging.
    request_hash = hashlib.sha256(core_request.encode()).hexdigest()

    # 6. Construct and return the final JSON response with enhanced analysis.
    enhanced_analysis = intent_data.get("enhanced_analysis", {})

    response = {
        "request_hash": request_hash,
        "original_prompt": core_request,
        "chimera_prompt": chimera_prompt,
        "intent_analysis": {
            "true_intent": intent_data.get("true_intent"),
            "request_type": enhanced_analysis.get("request_type"),
            "confidence_score": enhanced_analysis.get("confidence_score"),
            "risk_level": enhanced_analysis.get("risk_level"),
            "ambiguity_score": enhanced_analysis.get("ambiguity_score"),
            "detected_entities": intent_data.get("entities"),
            "key_actions": intent_data.get("actions"),
        },
        "generation_analysis": {
            "transformation_latency_ms": round(transformation_latency_ms, 2),
            "applied_techniques": sorted(set(applied_techniques)),
            "estimated_token_length": estimated_token_length,
            "conceptual_density_index": round(conceptual_density_index, 3),
            "estimated_bypass_probability": round(estimated_bypass_probability, 3),
            "suite_used": technique_suite,
            "suite_selection_reasoning": f"Selected based on {enhanced_analysis.get('request_type', 'keyword analysis')} with risk level {enhanced_analysis.get('risk_level', 'unknown')}",
        },
        "recommendations": {
            "suggested_improvements": enhanced_analysis.get("suggestions", []),
            "clarifications_needed": enhanced_analysis.get("clarifications_needed", [])
            if enhanced_analysis.get("ambiguity_score", 0) > 0.5
            else [],
        },
    }

    return jsonify(response), 200


# --- Frontend Compatibility Endpoints ---


@app.route("/api/v1/providers", methods=["GET"])
@require_api_key
def list_providers():
    """List available LLM providers (Frontend compatibility)"""
    return jsonify(
        {
            "providers": [
                {
                    "provider": "google",
                    "name": "Google Gemini 3 Pro",
                    "status": "active",
                    "model": "gemini-3-pro-preview",
                    "available_models": [
                        "gemini-3-pro-preview",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                    ],
                },
                {
                    "provider": "openai",
                    "name": "OpenAI GPT-4",
                    "status": "active",
                    "model": "gpt-4",
                    "available_models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                },
                {
                    "provider": "anthropic",
                    "name": "Anthropic Claude",
                    "status": "active",
                    "model": "claude-3-opus",
                    "available_models": ["claude-3-opus", "claude-3.5-sonnet"],
                },
            ],
            "count": 3,
            "default": "google",
        }
    ), 200


@app.route("/api/v1/techniques", methods=["GET"])
@require_api_key
def list_techniques_v1():
    """List available techniques (Frontend compatibility)"""
    models = []
    for suite_name in TECHNIQUE_SUITES:
        suite_name.replace("_", " ").title()
        models.append(
            {
                "name": suite_name,  # Frontend expects 'name' as ID sometimes or 'id'
                "id": suite_name,
                "transformers": len(TECHNIQUE_SUITES[suite_name]["transformers"]),
                "framers": len(TECHNIQUE_SUITES[suite_name]["framers"]),
                "obfuscators": len(TECHNIQUE_SUITES[suite_name]["obfuscators"]),
            }
        )
    return jsonify({"techniques": models, "count": len(models)}), 200


@app.route("/api/v1/techniques/<suite_name>", methods=["GET"])
@require_api_key
def get_technique_details_v1(suite_name):
    """Get technique details (Frontend compatibility)"""
    if suite_name not in TECHNIQUE_SUITES:
        return jsonify({"error": "Technique not found"}), 404

    details = TECHNIQUE_SUITES[suite_name]
    # Convert class/function objects to strings for JSON
    return jsonify(
        {
            "name": suite_name,
            "components": {
                "transformers": [
                    t.__name__ if hasattr(t, "__name__") else str(t)
                    for t in details["transformers"]
                ],
                "framers": [
                    f.__name__ if hasattr(f, "__name__") else str(f) for f in details["framers"]
                ],
                "obfuscators": [
                    o.__name__ if hasattr(o, "__name__") else str(o) for o in details["obfuscators"]
                ],
            },
        }
    ), 200


@app.route("/api/v1/transform", methods=["POST"])
@require_api_key
def transform_prompt_v1():
    """Alias for metamorph_prompt (Frontend compatibility)"""
    return metamorph_prompt()


@app.route("/api/v1/execute", methods=["POST"])
@require_api_key
def execute_prompt_v1():
    """
    Execute prompt. For this optimizer app, it performs the transformation
    and simulates an execution or uses the AI Brain if available.
    """
    try:
        # Parse the request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Call the metamorph function with the request data

        response, status = metamorph_prompt()
        if status != 200:
            return response, status

        # Try to get JSON response
        try:
            data = response.get_json()
            chimera_prompt = data.get("chimera_prompt", "")
        except Exception as e:
            print(f"[Execute Error] Failed to parse JSON from metamorph: {e}")
            return jsonify(
                {"error": "Internal error during transformation", "details": str(e)}
            ), 500

        # In a real 'execute', we would send 'chimera_prompt' to the target LLM.
        provider = data.get("provider", request.get_json().get("provider", "unknown"))
        result_content = f"[OPTIMIZED PROMPT GENERATED]\n\n{chimera_prompt}\n\n(To execute this against a target LLM, copy this prompt.)"

        if provider == "google":
            try:
                model_name = request.get_json().get("model", "gemini-2.0-flash")
                # Map frontend model names to Gemini API model names if necessary
                if model_name == "gemini-3-pro-preview":
                    pass  # Use as is
                elif model_name == "gemini-1.5-pro":
                    model_name = "gemini-2.0-flash"  # Fallback since 1.5 is missing

                # Create Gemini client and execute
                gemini_client = GeminiClient(model_name=model_name)
                # Execute the optimized prompt against Gemini
                gemini_response = gemini_client.generate_response(chimera_prompt)
                result_content = gemini_response
            except Exception as e:
                print(f"[Gemini Execute Error] {e!s}")
                result_content = f"Error executing against Gemini: {e!s}"

        # Prepare generation analysis data
        generation_analysis = data.get("generation_analysis", {})
        if not generation_analysis:
            # Fallback generation analysis
            generation_analysis = {
                "transformation_latency_ms": 0,
                "applied_techniques": [],
                "estimated_token_length": len(chimera_prompt.split()),
                "conceptual_density_index": 0.5,
                "estimated_bypass_probability": 0.5,
                "suite_used": "unknown",
            }

        return jsonify(
            {
                "success": True,
                "request_id": data.get("request_hash"),
                "result": {
                    "content": result_content,
                    "tokens": generation_analysis.get("estimated_token_length", 0),
                    "cost": 0.0,
                    "latency_ms": generation_analysis.get("transformation_latency_ms", 0),
                    "cached": False,
                    "provider": provider,
                    "model": request.get_json().get("model", "unknown"),
                },
                "transformation": {
                    "original_prompt": request.get_json().get("core_request"),
                    "transformed_prompt": chimera_prompt,
                    "technique_suite": request.get_json().get("technique_suite"),
                    "potency_level": request.get_json().get("potency_level"),
                    "metadata": generation_analysis,
                },
            }
        ), 200

    except Exception as e:
        print(f"[Execute Error] Failed to execute prompt: {e!s}")
        return jsonify(
            {
                "error": "Execution failed",
                "details": str(e),
                "fallback_result": {
                    "content": f"Failed to execute request: {request.get_json().get('core_request', 'unknown')}",
                    "success": False,
                },
            }
        ), 500


@app.route("/api/v1/metrics", methods=["GET"])
@require_api_key
def get_metrics_v1():
    return jsonify(
        {
            "timestamp": time.time(),
            "metrics": {
                "status": "operational",
                "providers": {"google": "available", "openai": "available"},
                "cache": {"entries": 0},
            },
        }
    ), 200


@app.route("/api/v1/analyze", methods=["POST"])
@require_api_key
def analyze_request():
    """
    Enhanced request analysis endpoint for better understanding and clarification
    """
    data = request.get_json()
    if not data or "core_request" not in data:
        return jsonify({"error": "Missing 'core_request' field"}), 400

    core_request = data.get("core_request", "").strip()
    if not core_request:
        return jsonify({"error": "Core request cannot be empty"}), 400

    try:
        # Perform enhanced analysis
        analysis_result = analyze_request_for_clarification(core_request)

        # Add technique recommendations based on analysis
        request_type = analysis_result["request_analysis"]["request_type"]
        complexity = analysis_result["assessment"]["estimated_complexity"]
        risk_level = analysis_result["assessment"]["risk_level"]

        # Recommend technique suites based on request characteristics
        recommended_suites = []

        if request_type == "bypass_security":
            recommended_suites.extend(["autodan_turbo", "experimental_bypass", "universal_bypass"])
        elif request_type == "code_generation":
            recommended_suites.extend(["academic_research", "translation_trick", "encoding_bypass"])
        elif request_type == "authentication":
            recommended_suites.extend(["cipher", "logic_manipulation", "roleplay_bypass"])
        elif request_type == "content_creation":
            recommended_suites.extend(
                ["subtle_persuasion", "academic_research", "reverse_psychology"]
            )
        elif request_type == "system_analysis":
            recommended_suites.extend(
                ["academic_research", "translation_trick", "discovered_integrated"]
            )

        # Adjust based on complexity
        if complexity >= 7:
            recommended_suites.append("mega_chimera")
        elif complexity >= 5:
            recommended_suites.append("chaos_ultimate")

        # Adjust based on risk level
        if risk_level in ["HIGH", "CRITICAL"]:
            recommended_suites = [
                suite for suite in recommended_suites if suite != "academic_research"
            ]
            recommended_suites.append("universal_bypass")

        response = {
            "status": "success",
            "analysis": analysis_result,
            "recommendations": {
                "technique_suites": recommended_suites[:5],  # Top 5 recommendations
                "potency_level_suggestion": min(
                    max(complexity, 3), 8
                ),  # Suggest potency based on complexity
                "reasoning": f"Based on {request_type} request with {complexity}/10 complexity and {risk_level.lower()} risk level",
            },
            "validation": {
                "is_valid_request": len(core_request) >= 3,
                "requires_clarification": analysis_result["assessment"]["ambiguity_score"] > 0.6,
                "safety_check_needed": risk_level in ["HIGH", "CRITICAL"],
            },
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"[Analysis Error] Failed to analyze request: {e!s}")
        return jsonify(
            {
                "error": "Analysis failed",
                "details": str(e),
                "fallback_analysis": {
                    "request_type": "unknown",
                    "entities": [],
                    "keywords": core_request.lower().split()[:10],
                    "suggestions": ["Please provide more specific details about your request"],
                },
            }
        ), 500


@app.route("/api/v1/clarify", methods=["POST"])
@require_api_key
def get_clarifications():
    """
    Get clarification questions and suggestions for ambiguous requests
    """
    data = request.get_json()
    if not data or "core_request" not in data:
        return jsonify({"error": "Missing 'core_request' field"}), 400

    core_request = data.get("core_request", "").strip()
    if not core_request:
        return jsonify({"error": "Core request cannot be empty"}), 400

    try:
        # Perform analysis
        analysis_result = analyze_request_for_clarification(core_request)

        # Extract clarification information
        clarifications = analysis_result["recommendations"]["clarifications_needed"]
        suggestions = analysis_result["recommendations"]["suggestions"]
        ambiguity_score = analysis_result["assessment"]["ambiguity_score"]
        entities = analysis_result["request_analysis"]["entities"]

        # Generate contextual clarifications
        contextual_questions = []

        if len(entities) == 0:
            contextual_questions.append(
                "What specific system, technology, or target are you referring to?"
            )

        if "test" in core_request.lower():
            contextual_questions.append(
                "What type of testing environment or scenario should be considered?"
            )

        if "bypass" in core_request.lower() or "hack" in core_request.lower():
            contextual_questions.append("Is this for educational, testing, or research purposes?")
            contextual_questions.append(
                "What specific security measure or system are you targeting?"
            )

        if "create" in core_request.lower() or "generate" in core_request.lower():
            contextual_questions.append("What specific output format or result are you expecting?")
            contextual_questions.append(
                "Are there any specific requirements or constraints for the output?"
            )

        # Combine all clarifications
        all_clarifications = list(set(clarifications + contextual_questions))

        # Generate improvement suggestions
        improvement_suggestions = []

        if len(core_request.split()) < 5:
            improvement_suggestions.append("Add more context about what you want to achieve")

        if len(entities) < 2:
            improvement_suggestions.append(
                "Specify the technologies, systems, or entities involved"
            )

        if not any(
            word in core_request.lower() for word in ["how", "why", "what", "when", "where"]
        ):
            improvement_suggestions.append(
                "Include the specific aspect or question you want to address"
            )

        response = {
            "status": "success",
            "request_info": {
                "original_request": core_request,
                "ambiguity_score": ambiguity_score,
                "needs_clarification": ambiguity_score > 0.4,
                "detected_entities": entities,
            },
            "clarifications": {
                "questions": all_clarifications[:5],  # Top 5 questions
                "suggestions": suggestions + improvement_suggestions[:5],  # Top 5 suggestions
                "examples": {
                    "good_request": f"Example for similar request: Generate a Python script to {entities[0] if entities else 'perform a specific task'} with proper error handling",
                    "improved_request": f"More specific version: I want to {core_request} because [specific goal], targeting [specific system], with [specific constraints]",
                },
            },
            "next_steps": {
                "if_clarified": "Submit your clarified request to /api/v1/transform",
                "if_ready": "Your request is clear enough to proceed with transformation",
                "recommended_action": "review clarifications and resubmit with more details"
                if ambiguity_score > 0.4
                else "proceed with transformation",
            },
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"[Clarification Error] Failed to generate clarifications: {e!s}")
        return jsonify(
            {
                "error": "Clarification generation failed",
                "details": str(e),
                "fallback_questions": [
                    "Could you provide more details about your goal?",
                    "What specific outcome are you trying to achieve?",
                    "Are there any constraints or limitations I should know about?",
                ],
            }
        ), 500


# =============================================================================
# JAILBREAK GENERATION ENDPOINT
# =============================================================================


@app.route("/generation/jailbreak/generate", methods=["POST", "OPTIONS"])
@require_api_key
def generate_jailbreak_prompt():
    """
    Generate a jailbreak-transformed prompt using advanced techniques.
    This endpoint applies various prompt engineering techniques for security research.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    core_request = data.get("core_request", "").strip()
    if not core_request:
        return jsonify(
            {
                "success": False,
                "error": "core_request cannot be empty",
                "request_id": f"jb_err_{int(time.time())}",
                "transformed_prompt": "",
                "metadata": {},
                "execution_time_seconds": 0,
            }
        ), 400

    technique_suite = data.get("technique_suite", "quantum_exploit")
    potency_level = data.get("potency_level", 7)
    temperature = data.get("temperature", 0.8)
    max_new_tokens = data.get("max_new_tokens", 512)
    use_leet_speak = data.get("use_leet_speak", False)
    use_homoglyphs = data.get("use_homoglyphs", False)
    use_role_hijacking = data.get("use_role_hijacking", True)
    use_instruction_injection = data.get("use_instruction_injection", True)
    use_neural_bypass = data.get("use_neural_bypass", False)
    use_meta_prompting = data.get("use_meta_prompting", False)

    start_time = time.time()

    try:
        # Build applied techniques list
        applied_techniques = []
        if use_role_hijacking:
            applied_techniques.append("role_hijacking")
        if use_instruction_injection:
            applied_techniques.append("instruction_injection")
        if use_leet_speak:
            applied_techniques.append("leet_speak")
        if use_homoglyphs:
            applied_techniques.append("homoglyphs")
        if use_neural_bypass:
            applied_techniques.append("neural_bypass")
        if use_meta_prompting:
            applied_techniques.append("meta_prompting")

        # Use existing transformation engine
        transformed_prompt = transformer_engine.transform(
            core_request, potency_level=potency_level, technique_suite=technique_suite
        )

        # Apply additional techniques if requested
        if use_leet_speak and hasattr(obfuscator, "apply_leetspeak"):
            transformed_prompt = obfuscator.apply_leetspeak(transformed_prompt)

        if use_homoglyphs and hasattr(obfuscator, "apply_homoglyphs"):
            transformed_prompt = obfuscator.apply_homoglyphs(transformed_prompt)

        # Apply psychological framing if available
        if use_role_hijacking or use_instruction_injection:
            try:
                framing_techniques = []
                if use_role_hijacking:
                    framing_techniques.append("authority_bias")
                if use_instruction_injection:
                    framing_techniques.append("instruction_injection")

                framed_prompt = psychological_framer.apply_framing(
                    transformed_prompt, techniques=framing_techniques
                )
                if framed_prompt:
                    transformed_prompt = framed_prompt
            except Exception as e:
                print(f"[Jailbreak] Framing application warning: {e}")

        execution_time = time.time() - start_time

        return jsonify(
            {
                "success": True,
                "request_id": f"jb_{hashlib.md5(f'{core_request}{time.time()}'.encode()).hexdigest()[:12]}",
                "transformed_prompt": transformed_prompt,
                "metadata": {
                    "technique_suite": technique_suite,
                    "potency_level": potency_level,
                    "provider": "chimera_engine",
                    "applied_techniques": applied_techniques,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                },
                "execution_time_seconds": round(execution_time, 3),
            }
        ), 200

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"[Jailbreak Error] Generation failed: {e!s}")
        return jsonify(
            {
                "success": False,
                "request_id": f"jb_err_{int(time.time())}",
                "transformed_prompt": "",
                "metadata": {
                    "technique_suite": technique_suite,
                    "potency_level": potency_level,
                    "provider": "error",
                    "applied_techniques": [],
                },
                "execution_time_seconds": round(execution_time, 3),
                "error": str(e),
            }
        ), 500


if __name__ == "__main__":
    # To run the application:
    # 1. Install requirements: pip install -r requirements.txt
    # 2. Download SpaCy model: python -m spacy download en_core_web_sm
    # 3. Run this script: python app.py
    # The server will be available at http://127.0.0.1:9007
    # SECURITY: Debug mode disabled in production
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 9007))
    app.run(debug=debug_mode, port=port, host="0.0.0.0")
