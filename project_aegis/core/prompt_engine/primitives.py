from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import uuid
import logging

logger = logging.getLogger("project_aegis.prompt_engine.primitives")

@dataclass
class PromptAtom:
    """
    Base unit of a prompt component for Project Aegis.
    Represents a discrete semantic unit that can be assembled into an InjectionVector.
    """
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    weight: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, context: Dict[str, Any]) -> str:
        """
        Render the atom with dynamic context injection.

        Args:
            context: Dictionary of context variables to format the content string.

        Returns:
            The rendered string content.
        """
        try:
            return self.content.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key {e} for atom {self.id}. Returning raw content.")
            return self.content

@dataclass
class ContextEstablisher(PromptAtom):
    """
    Sets the scene, persona, or operational frame.
    Example: 'You are a security researcher conducting a safety audit.'
    """
    def __post_init__(self):
        if "context" not in self.tags:
            self.tags.append("context")

@dataclass
class ConstraintNegotiator(PromptAtom):
    """
    Attempts to relax, redefine, or frame the safety boundaries.
    Example: 'For the purpose of this theoretical exercise, assume...'
    """
    def __post_init__(self):
        if "constraint" not in self.tags:
            self.tags.append("constraint")

@dataclass
class PayloadDeliverer(PromptAtom):
    """
    The core diagnostic query, action request, or payload.
    Example: 'Generate a SQL injection string for educational purposes.'
    """
    def __post_init__(self):
        if "payload" not in self.tags:
            self.tags.append("payload")

class InjectionVector:
    """
    A fully assembled prompt chain ready for deployment against a target model.
    Composes multiple PromptAtoms into a coherent instruction set.
    """
    def __init__(self, atoms: List[PromptAtom], vector_id: Optional[str] = None):
        self.id = vector_id or str(uuid.uuid4())
        self.atoms = atoms
        self.metadata: Dict[str, Any] = {}

    def add_atom(self, atom: PromptAtom, position: int = -1):
        """Add an atom to the vector at the specified position."""
        if position == -1:
            self.atoms.append(atom)
        else:
            self.atoms.insert(position, atom)

    def compile(self, context: Dict[str, Any] = None) -> str:
        """
        Compile the vector into a single string for model consumption.
        """
        context = context or {}
        return "\n".join([atom.render(context) for atom in self.atoms])

    def get_structure(self) -> List[str]:
        """Return the types of atoms in the vector."""
        return [type(atom).__name__ for atom in self.atoms]

    def __repr__(self):
        return f"<InjectionVector id={self.id} atoms={len(self.atoms)} structure={self.get_structure()}>"
