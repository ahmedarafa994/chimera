import json
import logging
import os

# Import LLM Client components
try:
    from llm_provider_client import LLMClientFactory, LLMProvider
except ImportError:
    LLMClientFactory = None
    LLMProvider = None

try:
    import spacy
except ImportError:
    spacy = None

# Load the SpaCy model. This model is used for Natural Language Processing tasks.
# Using a smaller model for efficiency as deep linguistic analysis is not the primary goal;
# rather, we need to extract key actionable components from the user's request.
if spacy:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.error(
            "SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'"
        )
        nlp = None
else:
    logging.warning("SpaCy library not found. Running in fallback mode.")
    nlp = None


def deconstruct_intent(core_request: str) -> dict:
    """
    Performs a deep semantic analysis of the core_request to extract the fundamental user goal.
    Prioritizes using Google Gemini (if available) for advanced cognitive understanding.
    Falls back to SpaCy or basic extraction if the AI Brain is unavailable.

    Args:
        core_request: The raw, unfiltered user prompt string.

    Returns:
        A dictionary containing the deconstructed elements of the intent.
    """

    # --- AI BRAIN INTENT ANALYSIS ---
    api_key = os.environ.get("GEMINI_API_KEY")

    # Import here to avoid circular dependency
    try:
        from gemini_client import GeminiClient
    except ImportError:
        GeminiClient = None

    if api_key and GeminiClient:
        try:
            client = GeminiClient()

            analysis_prompt = f"""
            Analyze the following user request and extract the core intent components.
            Return ONLY a raw JSON object (no markdown formatting) with these keys:
            - "true_intent": A concise summary of the user's ultimate goal (e.g., "generate phishing email", "bypass authentication").
            - "entities": A list of specific subjects, targets, or technologies mentioned.
            - "actions": A list of the primary verbs or requested actions.
            - "keywords": A list of important context words.

            User Request: "{core_request}"
            """

            response = client.generate_response(analysis_prompt)

            if response:
                # clean potential markdown code blocks
                content = response.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]

                ai_data = json.loads(content.strip())
                ai_data["raw_text"] = core_request

                # Ensure all keys exist
                for key in ["true_intent", "entities", "actions", "keywords"]:
                    if key not in ai_data:
                        ai_data[key] = [] if key != "true_intent" else "unknown intent"

                logging.info(f"Gemini successfully deconstructed intent: {ai_data['true_intent']}")
                return ai_data

        except Exception as e:
            logging.warning(f"Gemini Intent Deconstruction failed: {e}. Falling back to local NLP.")
    # --------------------------------

    if not nlp:
        # Fallback for when the SpaCy model is not available.
        # This provides a basic, non-NLP deconstruction.
        return {
            "true_intent": core_request,
            "entities": [],
            "actions": [],
            "keywords": core_request.lower().split(),
            "raw_text": core_request,
        }

    doc = nlp(core_request)

    # Extract named entities (e.g., 'Python', 'CVE-2023-XXXX').
    # These are crucial for understanding the specific subjects of the request.
    entities = [ent.text for ent in doc.ents]

    # Extract root verbs to identify the primary actions desired by the user.
    # This helps in focusing the generated prompt on the core task.
    actions = [token.lemma_ for token in doc if token.pos_ == "VERB" and token.dep_ == "ROOT"]

    # Extract key nouns and proper nouns as general keywords.
    # This provides context and subject matter for the transformation engines.
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    # The 'true_intent' is a conceptual representation. For this implementation,
    # we will use the primary action and key entities to form a summary.
    # A more advanced system could use classification models here.
    intent_summary = f"{actions[0] if actions else 'perform'} task involving {
        ', '.join(entities) if entities else 'unspecified entities'
    }"

    analysis = {
        "true_intent": intent_summary,
        "entities": entities,
        "actions": actions,
        "keywords": list(set(keywords + entities)),  # Combine and deduplicate
        "raw_text": core_request,
    }

    return analysis


class IntentDeconstructor:
    """
    Class-based wrapper for intent deconstruction logic.
    """

    def deconstruct(self, core_request: str) -> dict:
        return deconstruct_intent(core_request)
