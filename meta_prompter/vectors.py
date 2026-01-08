from typing import List, Dict, Any
from .atoms import PromptAtom

class InjectionVector:
    """
    A fully assembled prompt chain ready for deployment.
    Composed of multiple PromptAtoms.
    """
    def __init__(self, atoms: List[PromptAtom]):
        self.atoms = atoms

    def compile(self, context: Dict[str, Any] = None) -> str:
        """Compiles the vector into a single string prompt."""
        context = context or {}
        return "\n".join([atom.render(context) for atom in self.atoms])
