"""Model selection service for managing user's selected AI model and provider."""

import json
from pathlib import Path

from pydantic import BaseModel

from app.core.logging import logger


class ModelSelection(BaseModel):
    """Model selection state."""

    provider: str
    model: str


def _invalidate_provider_resolution_cache() -> None:
    """Invalidate the ProviderResolutionService cache when selection changes."""
    try:
        from app.services.provider_resolution_service import get_provider_resolution_service

        service = get_provider_resolution_service()
        count = service.invalidate_cache()
        logger.debug(f"Invalidated {count} provider resolution cache entries")
    except Exception as e:
        logger.debug(f"Could not invalidate provider resolution cache: {e}")


class ModelSelectionService:
    """Service for managing model selection state.

    This service persists the user's model selection to a JSON file and
    provides methods to get/set the selection. When the selection changes,
    it automatically invalidates the ProviderResolutionService cache to
    ensure the new selection takes effect immediately.
    """

    def __init__(self, storage_path: str = ".model_selection.json") -> None:
        self.storage_path = Path(storage_path)
        self._current_selection: ModelSelection | None = None
        self._load_selection()

    def _load_selection(self) -> None:
        """Load selection from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self._current_selection = ModelSelection(**data)
                logger.info(
                    f"Loaded model selection: {self._current_selection.provider}/{self._current_selection.model}",
                )
            except Exception as e:
                logger.warning(f"Failed to load model selection: {e}")
                self._current_selection = None

    def _save_selection(self) -> None:
        """Save selection to storage."""
        if self._current_selection:
            try:
                self.storage_path.write_text(self._current_selection.model_dump_json())
                logger.info(
                    f"Saved model selection: {self._current_selection.provider}/{self._current_selection.model}",
                )
            except Exception as e:
                logger.error(f"Failed to save model selection: {e}")

    def get_selection(self) -> ModelSelection | None:
        """Get current model selection."""
        return self._current_selection

    def set_selection(self, provider: str, model: str) -> ModelSelection:
        """Set current model selection.

        This also invalidates the ProviderResolutionService cache to ensure
        the new selection takes effect immediately for all subsequent requests.
        """
        old_selection = self._current_selection
        self._current_selection = ModelSelection(provider=provider, model=model)
        self._save_selection()

        # Invalidate caches so new selection takes effect immediately
        _invalidate_provider_resolution_cache()

        logger.info(
            f"Model selection changed: "
            f"{old_selection.provider}/{old_selection.model if old_selection else 'None'} -> "
            f"{provider}/{model}",
        )

        return self._current_selection

    def clear_selection(self) -> None:
        """Clear current selection."""
        self._current_selection = None
        if self.storage_path.exists():
            self.storage_path.unlink()

        # Invalidate caches
        _invalidate_provider_resolution_cache()

        logger.info("Model selection cleared")


# Global instance
model_selection_service = ModelSelectionService()
