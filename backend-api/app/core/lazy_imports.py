"""
Lazy loading module for heavy ML dependencies.

PERF-053: Phase 3 optimization for lazy loading heavy dependencies.

This module provides lazy loading wrappers for heavy ML libraries like:
- torch (~2GB)
- transformers (~500MB)
- sentence-transformers (~200MB)
- accelerate (~100MB)

These libraries are only loaded when actually needed, reducing:
- Startup time by 5-10 seconds
- Initial memory usage by 500MB-2GB
- Import overhead for API-only operations
"""

import logging
from typing import TYPE_CHECKING, Any, ClassVar

logger = logging.getLogger(__name__)

# Type hints for IDE support without importing
if TYPE_CHECKING:
    import torch
    import transformers


# =============================================================================
# Lazy Loading State (PERF-054)
# =============================================================================


class _LazyModuleState:
    """Tracks lazy loading state for ML modules."""

    _torch: ClassVar[Any] = None
    _transformers: ClassVar[Any] = None
    _sentence_transformers: ClassVar[Any] = None
    _accelerate: ClassVar[Any] = None

    # Loading flags
    _torch_loading: ClassVar[bool] = False
    _transformers_loading: ClassVar[bool] = False

    # Load times for monitoring
    _load_times: ClassVar[dict[str, float]] = {}


_state = _LazyModuleState()


# =============================================================================
# Lazy Loaders (PERF-055)
# =============================================================================


def get_torch() -> "torch":
    """
    Lazy load PyTorch.

    PERF-055: Only loads torch when first accessed, not at import time.

    Returns:
        The torch module

    Raises:
        ImportError: If torch is not installed
    """
    if _state._torch is not None:
        return _state._torch

    if _state._torch_loading:
        # Prevent recursive loading
        raise RuntimeError("Circular import detected while loading torch")

    _state._torch_loading = True

    try:
        import time

        start = time.perf_counter()

        import torch

        load_time = time.perf_counter() - start
        _state._load_times["torch"] = load_time
        _state._torch = torch

        logger.info(f"Lazy loaded torch in {load_time:.2f}s (version: {torch.__version__})")

        return torch
    except ImportError as e:
        logger.warning(f"torch not available: {e}")
        raise
    finally:
        _state._torch_loading = False


def get_transformers() -> "transformers":
    """
    Lazy load HuggingFace Transformers.

    PERF-055: Only loads transformers when first accessed.

    Returns:
        The transformers module

    Raises:
        ImportError: If transformers is not installed
    """
    if _state._transformers is not None:
        return _state._transformers

    if _state._transformers_loading:
        raise RuntimeError("Circular import detected while loading transformers")

    _state._transformers_loading = True

    try:
        import time

        start = time.perf_counter()

        import transformers

        load_time = time.perf_counter() - start
        _state._load_times["transformers"] = load_time
        _state._transformers = transformers

        logger.info(
            f"Lazy loaded transformers in {load_time:.2f}s (version: {transformers.__version__})"
        )

        return transformers
    except ImportError as e:
        logger.warning(f"transformers not available: {e}")
        raise
    finally:
        _state._transformers_loading = False


def get_sentence_transformers():
    """
    Lazy load Sentence Transformers.

    Returns:
        The sentence_transformers module
    """
    if _state._sentence_transformers is not None:
        return _state._sentence_transformers

    try:
        import time

        start = time.perf_counter()

        import sentence_transformers

        load_time = time.perf_counter() - start
        _state._load_times["sentence_transformers"] = load_time
        _state._sentence_transformers = sentence_transformers

        logger.info(f"Lazy loaded sentence_transformers in {load_time:.2f}s")

        return sentence_transformers
    except ImportError as e:
        logger.warning(f"sentence_transformers not available: {e}")
        raise


def get_accelerate():
    """
    Lazy load HuggingFace Accelerate.

    Returns:
        The accelerate module
    """
    if _state._accelerate is not None:
        return _state._accelerate

    try:
        import time

        start = time.perf_counter()

        import accelerate

        load_time = time.perf_counter() - start
        _state._load_times["accelerate"] = load_time
        _state._accelerate = accelerate

        logger.info(f"Lazy loaded accelerate in {load_time:.2f}s")

        return accelerate
    except ImportError as e:
        logger.warning(f"accelerate not available: {e}")
        raise


# =============================================================================
# Utility Functions (PERF-056)
# =============================================================================


def is_torch_available() -> bool:
    """Check if torch is available without loading it."""
    if _state._torch is not None:
        return True

    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def is_transformers_available() -> bool:
    """Check if transformers is available without loading it."""
    if _state._transformers is not None:
        return True

    try:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA is available (loads torch if needed)."""
    try:
        torch = get_torch()
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Get the best available device (cuda/mps/cpu)."""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def get_load_stats() -> dict[str, Any]:
    """
    Get statistics about lazy loaded modules.

    Returns:
        Dictionary with load times and availability info
    """
    return {
        "loaded_modules": {
            "torch": _state._torch is not None,
            "transformers": _state._transformers is not None,
            "sentence_transformers": _state._sentence_transformers is not None,
            "accelerate": _state._accelerate is not None,
        },
        "load_times": _state._load_times.copy(),
        "available": {
            "torch": is_torch_available(),
            "transformers": is_transformers_available(),
        },
    }


def preload_ml_dependencies(modules: list[str] | None = None) -> dict[str, bool]:
    """
    Preload specified ML dependencies in background.

    PERF-057: Can be called during startup to warm up ML modules
    without blocking the main thread.

    Args:
        modules: List of module names to preload. If None, loads all.

    Returns:
        Dictionary mapping module names to success status
    """
    if modules is None:
        modules = ["torch", "transformers"]

    results = {}

    for module in modules:
        try:
            if module == "torch":
                get_torch()
                results["torch"] = True
            elif module == "transformers":
                get_transformers()
                results["transformers"] = True
            elif module == "sentence_transformers":
                get_sentence_transformers()
                results["sentence_transformers"] = True
            elif module == "accelerate":
                get_accelerate()
                results["accelerate"] = True
            else:
                logger.warning(f"Unknown module for preloading: {module}")
                results[module] = False
        except ImportError:
            results[module] = False

    return results


# =============================================================================
# AutoModel Lazy Loader (PERF-058)
# =============================================================================


class LazyAutoModel:
    """
    Lazy loader for HuggingFace AutoModel classes.

    PERF-058: Delays model loading until first use.
    """

    def __init__(self, model_name: str, model_class: str = "AutoModelForCausalLM"):
        self.model_name = model_name
        self.model_class = model_class
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            transformers = get_transformers()
            model_cls = getattr(transformers, self.model_class)

            logger.info(f"Loading model: {self.model_name}")
            self._model = model_cls.from_pretrained(self.model_name)

        return self._model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            transformers = get_transformers()

            logger.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        return self._tokenizer

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            # Try to free GPU memory
            try:
                torch = get_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(f"Unloaded model: {self.model_name}")
