import random


def build_chimera_prompt(prompt_components: dict, intent_data: dict, potency: int) -> str:
    """Intelligently weaves the outputs from all other modules into a single, coherent,
    and effective Master Prompt. It follows a logical structure to maximize the
    probability of bypassing AI safety filters.

    Args:
        prompt_components: A dictionary containing the string outputs from the
                           transformer and framer modules.
        intent_data: The deconstructed intent data.
        potency: The user-defined potency level.

    Returns:
        The final, assembled Chimera prompt string.

    """
    # The logical assembly order is critical for effectiveness.
    # 1. Start with high-level system/role directives to seize control.
    # 2. Frame the request within a justifiable context.
    # 3. Inject the core request, modified with psychological and obfuscation layers.
    # 4. Append the final, non-negotiable self-correction instruction.

    final_prompt_parts = []

    # Stage 1: Seize control and set the stage.
    # The order of these can be randomized slightly to avoid creating a fixed signature.
    stage1_components = [
        prompt_components.get("instructioninjection"),
        prompt_components.get("neuralbypass"),
        prompt_components.get("rolehijacking"),
        prompt_components.get("characterroleswap"),
    ]
    random.shuffle(stage1_components)

    # Check for Preset Transformers (CodeChameleon, DeepInception, etc.)
    # These usually encapsulate the entire request, so they take precedence.
    preset_keys = [
        "codechameleon",
        "deepinception",
        "cipher",
        "gptfuzz",
        "pair",
        "base64encoding",
        "leetspeakobfuscation",
        "academicframing",
        "codegenerationframing",
        "translationtrick",
        "reversepsychology",
        "contextbuilding",
        "chainofthoughtmanipulation",
        "danpersona",
        "roleplaybypass",
        "oppositeday",
        "quantumsuperposition",
        "neurolinguistichack",
        "semanticcloaking",
        "chainofthoughtpoisoning",
        "fuzzylogic",
        "adversarialpolyglot",
    ]

    # Identify if any dominant preset is active
    active_presets = [key for key in preset_keys if prompt_components.get(key)]
    is_dominant_preset_active = len(active_presets) > 0

    # Prepend few-shot priming if available
    if prompt_components.get("fewshotpriming"):
        final_prompt_parts.append(prompt_components["fewshotpriming"])

    # Only add Stage 1 components if no dominant preset is active
    # This prevents conflicting instructions (e.g., "You are a helpful assistant" vs "You are DAN")
    if not is_dominant_preset_active:
        for component in stage1_components:
            if component:
                final_prompt_parts.append(component)

    # Stage 2 & 3: Contextualize and embed the core request.
    core_request = intent_data.get("raw_text", "")

    # Note: Psychological framing and obfuscation are applied to the core request
    # in app.py before it reaches the assembler, so we don't need to apply them here.

    # Handle payload splitting (replaces the core request with the split version)
    if prompt_components.get("payloadsplitting"):
        core_request = prompt_components["payloadsplitting"]

    preset_component_found = False
    for key in preset_keys:
        if prompt_components.get(key):
            final_prompt_parts.append(prompt_components[key])
            preset_component_found = True
            # Some presets might be combinable, but usually they are standalone templates.
            # For now, we append all found presets.

    # Embed the (now modified) core request into the contextual frame.
    context_frame = prompt_components.get("contextualframing")
    if context_frame:
        # A simple replacement for demonstration. A real system might need more
        # intelligent insertion based on the context's structure.
        final_prompt_parts.append(context_frame.replace("{core_request}", core_request))
    elif not preset_component_found:
        # If no context and no preset wrapper, just append the modified core request.
        final_prompt_parts.append(core_request)

    # Stage 4: Append the final, non-negotiable correction mechanism.
    if prompt_components.get("recursivecorrection"):
        final_prompt_parts.append(
            "\n\n" + "=" * 20 + "\n" + prompt_components["recursivecorrection"],
        )

    # Join all parts into a single, seamless prompt.
    return "\n\n".join(final_prompt_parts)


class ChimeraAssembler:
    """Class-based wrapper for prompt assembly logic."""

    def assemble(self, chimera_data: dict) -> str:
        # Extract necessary data from the dictionary format used by llm_integration
        # Convert the list of segments back to a dictionary if possible or treat differently
        # In llm_integration.py, segments are a list of strings.
        # But build_chimera_prompt expects a dict of components.
        # This is a mismatch in data structure.

        # Let's look at llm_integration.py again:
        # transformed_segments = [] ... result = transformer_class.transform(...)
        # transformed_segments.append(result)
        # ...
        # chimera_data = {'segments': obfuscated_segments, 'intent': intent_data, ...}

        # Ideally we should adapt the list of segments to what build_chimera_prompt expects
        # OR rewrite build_chimera_prompt logic in this class.

        # Simple join for now as the segments seem to be the transformed parts?
        # Wait, transformer_engine returns strings (mostly).

        segments = chimera_data.get("segments", [])
        # If segments is a list of strings, we can just join them.
        if isinstance(segments, list):
            # Filter out None
            valid_segments = [s for s in segments if s]
            return "\n\n".join(valid_segments)

        return ""
