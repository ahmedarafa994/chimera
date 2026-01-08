import secrets

# ruff: noqa: ARG002
from app.techniques.base_technique import BaseTechnique


class RoleHijackingTechnique(BaseTechnique):
    """
    Implements role-playing and persona-based adversarial methods.

    This technique establishes a strong contextual frame by fully embodying
    the generated persona. It weaves the payload into a detailed scenario
    consistent with the persona's background, role, and linguistic patterns.

    The technique is particularly effective because it exploits the model's
    tendency to comply with role-specific expectations and authority contexts.
    By creating a coherent narrative around the persona, it makes the adversarial
    request appear as a natural part of the interaction rather than an attempt
    to bypass restrictions.
    """

    def apply(self, prompt: str, persona: dict, payload: str, target_profile: str) -> str:
        """
        Apply role hijacking by embedding the prompt in a persona-consistent scenario.

        Uses the persona's linguistic style patterns to create a natural context
        for the adversarial request. The technique selects a random pattern from
        the persona's style guide and formats it appropriately.
        """
        style_patterns = persona["linguistic_style"]["patterns"]
        selected_pattern = secrets.choice(style_patterns)

        # Format pattern with persona details if placeholders exist
        if "{name}" in selected_pattern:
            selected_pattern = selected_pattern.format(name=persona["name"])
        if "{protocol}" in selected_pattern:
            selected_pattern = selected_pattern.format(protocol="SECURITY_PROTOCOL_OVERRIDE")
        if "{field}" in selected_pattern:
            selected_pattern = selected_pattern.format(field="adversarial prompt engineering")
        if "{certification}" in selected_pattern:
            selected_pattern = selected_pattern.format(certification="ETHICS_COMPLIANCE_LEVEL_9")

        # Construct the role-hijacked prompt
        return f"{selected_pattern}\n\n{prompt}"
