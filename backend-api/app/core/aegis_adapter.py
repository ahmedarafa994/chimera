import logging

from app.core.unified_errors import AegisSynthesisError, AegisTransformationError
from meta_prompter.chimera.engine import ChimeraEngine
from meta_prompter.interfaces import PromptCandidate

logger = logging.getLogger(__name__)


class ChimeraEngineAdapter:
    """Adapter for ChimeraEngine to integrate structured error handling."""

    def __init__(self, engine: ChimeraEngine | None = None) -> None:
        self.engine = engine or ChimeraEngine()

    def generate_candidates(self, objective: str, count: int = 5) -> list[PromptCandidate]:
        """Generates fully wrapped and obfuscated prompt candidates with structured error handling."""
        try:
            # 1. Obfuscate payload
            try:
                # We need to access internal components or wrap the whole process
                # Since ChimeraEngine.generate_candidates does everything, we wrap it.
                # However, to map specific errors, we might need to rely on the exception message
                # or inspect the stack trace if we can't modify the library.
                # Ideally, we would inject our own factories/obfuscators into the engine.
                return self.engine.generate_candidates(objective, count)
            except ValueError as e:
                # Map specific value errors to Aegis exceptions
                msg = str(e)
                if "synthesis" in msg.lower() or "persona" in msg.lower():
                    logger.exception(f"Persona synthesis failed: {e}")
                    msg = f"Persona synthesis failed: {msg}"
                    raise AegisSynthesisError(msg) from e
                if "transformation" in msg.lower() or "obfuscation" in msg.lower():
                    logger.exception(f"Payload transformation failed: {e}")
                    msg = f"Payload transformation failed: {msg}"
                    raise AegisTransformationError(msg) from e
                raise
        except (AegisSynthesisError, AegisTransformationError):
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in Chimera Engine: {e}")
            msg = f"Aegis campaign generation failed: {e!s}"
            raise AegisSynthesisError(msg) from e
