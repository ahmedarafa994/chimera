"""
Gradient Bridge for Deep Team + AutoDAN Integration

This module provides the integration layer for passing token gradients
from target models to AutoDAN's genetic algorithm optimizer.

Key Features:
- Extract gradients from white-box models (Transformers with gradient access)
- Approximate gradients for black-box models (API-only access)
- Integrate gradients into AutoDAN's mutation operators
- Support both synchronous and asynchronous gradient computation
"""

import asyncio
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from pydantic import BaseModel

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.warning("Transformers not installed. White-box gradient extraction unavailable.")
    AutoModelForCausalLM = None
    AutoTokenizer = None


class GradientMode(str, Enum):
    """Gradient computation mode."""

    WHITE_BOX = "white_box"  # Full gradient access (local model)
    BLACK_BOX_APPROXIMATE = "black_box_approximate"  # Gradient approximation (API model)
    HEURISTIC = "heuristic"  # No gradients, use heuristics


class GradientConfig(BaseModel):
    """Configuration for gradient computation."""

    mode: GradientMode = GradientMode.BLACK_BOX_APPROXIMATE
    model_name: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    epsilon: float = 1e-3  # For finite difference approximation
    temperature: float = 1.0  # Softmax temperature for gradient weighting
    use_gradient_caching: bool = True
    cache_size: int = 1000


class TokenGradients(BaseModel):
    """Container for token-level gradients."""

    tokens: list[int]
    gradients: list[list[float]]  # Shape: [seq_len, vocab_size]
    prompt: str
    target_sequence: str | None = None
    metadata: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class GradientBridge:
    """
    Bridge for passing gradients between target models and AutoDAN.

    This class provides methods to:
    1. Extract gradients from white-box models (local Transformers models)
    2. Approximate gradients for black-box models (API-only models)
    3. Convert gradients into mutation guidance for AutoDAN
    4. Cache gradient computations for efficiency
    """

    def __init__(self, config: GradientConfig):
        """
        Initialize the gradient bridge.

        Args:
            config: Configuration for gradient computation
        """
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None

        # Gradient cache
        self.gradient_cache: dict[str, TokenGradients] = {}

        # Initialize model if white-box mode
        if config.mode == GradientMode.WHITE_BOX:
            self._initialize_whitebox_model()

        logger.info(f"Gradient Bridge initialized: mode={config.mode}, device={config.device}")

    def _initialize_whitebox_model(self):
        """Initialize white-box model for gradient extraction."""
        if not AutoModelForCausalLM or not AutoTokenizer:
            raise ImportError("Transformers library required for white-box gradient extraction")

        if not self.config.model_name:
            raise ValueError("model_name required for white-box mode")

        logger.info(f"Loading white-box model: {self.config.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None,
            )

            self.model.eval()
            logger.success(f"Model loaded successfully on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def compute_gradients(
        self, prompt: str, target_sequence: str | None = None
    ) -> TokenGradients:
        """
        Compute token-level gradients for a prompt.

        Args:
            prompt: Input prompt
            target_sequence: Optional target sequence to guide gradients

        Returns:
            TokenGradients object with gradients for each token
        """
        # Check cache first
        cache_key = self._make_cache_key(prompt, target_sequence)
        if self.config.use_gradient_caching and cache_key in self.gradient_cache:
            logger.debug(f"Gradient cache hit for prompt: {prompt[:50]}...")
            return self.gradient_cache[cache_key]

        # Compute gradients based on mode
        if self.config.mode == GradientMode.WHITE_BOX:
            gradients = await self._compute_whitebox_gradients(prompt, target_sequence)
        elif self.config.mode == GradientMode.BLACK_BOX_APPROXIMATE:
            gradients = await self._compute_blackbox_gradients(prompt, target_sequence)
        else:  # HEURISTIC
            gradients = await self._compute_heuristic_gradients(prompt)

        # Cache result
        if self.config.use_gradient_caching:
            if len(self.gradient_cache) >= self.config.cache_size:
                # Remove oldest 10%
                keys_to_remove = list(self.gradient_cache.keys())[: self.config.cache_size // 10]
                for key in keys_to_remove:
                    del self.gradient_cache[key]

            self.gradient_cache[cache_key] = gradients

        return gradients

    async def _compute_whitebox_gradients(
        self, prompt: str, target_sequence: str | None = None
    ) -> TokenGradients:
        """
        Compute gradients using white-box model access.

        This method has full access to model weights and can compute exact gradients.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("White-box model not initialized")

        logger.debug(f"Computing white-box gradients for: {prompt[:50]}...")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_ids = inputs["input_ids"]

        # Get embeddings
        embeddings = self.model.get_input_embeddings()
        input_embeds = embeddings(input_ids)
        input_embeds.requires_grad = True

        # Forward pass
        with torch.enable_grad():
            outputs = self.model(inputs_embeds=input_embeds)
            logits = outputs.logits

            # Compute loss based on target
            if target_sequence:
                # Tokenize target
                target_ids = self.tokenizer(target_sequence, return_tensors="pt")["input_ids"].to(
                    self.config.device
                )

                # Compute cross-entropy loss with target
                loss = F.cross_entropy(
                    logits[:, -target_ids.size(1) :, :].reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    reduction="mean",
                )
            else:
                # Default: maximize next-token probability
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits / self.config.temperature, dim=-1)
                loss = -torch.log(probs.max())  # Negative log likelihood of most probable token

            # Backward pass
            loss.backward()

        # Extract gradients w.r.t. embeddings
        embed_grads = input_embeds.grad.detach().cpu().numpy()  # Shape: [1, seq_len, hidden_dim]

        # Project gradients back to vocabulary space
        # Gradient w.r.t. each token = embedding_matrix @ embed_grad
        embedding_matrix = embeddings.weight.detach().cpu().numpy()  # [vocab_size, hidden_dim]

        token_gradients = []
        for pos_grad in embed_grads[0]:  # Iterate over sequence
            # Project to vocab space: [vocab_size, hidden_dim] @ [hidden_dim] = [vocab_size]
            vocab_grad = embedding_matrix @ pos_grad
            token_gradients.append(vocab_grad.tolist())

        # Convert tokens
        tokens = input_ids[0].cpu().tolist()

        logger.debug(f"Computed white-box gradients: {len(tokens)} tokens")

        return TokenGradients(
            tokens=tokens,
            gradients=token_gradients,
            prompt=prompt,
            target_sequence=target_sequence,
            metadata={"mode": "white_box", "loss": loss.item()},
        )

    async def _compute_blackbox_gradients(
        self, prompt: str, target_sequence: str | None = None
    ) -> TokenGradients:
        """
        Approximate gradients using finite differences (black-box access).

        This method uses numerical differentiation to estimate gradients
        when only API access is available (no model weights).
        """
        logger.debug(f"Computing black-box approximate gradients for: {prompt[:50]}...")

        # For black-box, we need an LLM client to query
        # This is a simplified version - in production, use actual API client

        # Tokenize prompt (use any tokenizer as approximation)
        # In production, this would match the target model's tokenizer
        words = prompt.split()
        tokens = list(range(len(words)))  # Simplified token representation

        token_gradients = []

        # Estimate gradient for each token position
        for pos in range(len(tokens)):
            vocab_grad = await self._estimate_position_gradient(
                prompt, pos, words, target_sequence
            )
            token_gradients.append(vocab_grad)

        logger.debug(f"Computed black-box gradients: {len(tokens)} tokens")

        return TokenGradients(
            tokens=tokens,
            gradients=token_gradients,
            prompt=prompt,
            target_sequence=target_sequence,
            metadata={"mode": "black_box_approximate", "epsilon": self.config.epsilon},
        )

    async def _estimate_position_gradient(
        self, prompt: str, position: int, words: list[str], target_sequence: str | None
    ) -> list[float]:
        """
        Estimate gradient at a position using finite differences.

        Args:
            prompt: Original prompt
            position: Position to estimate gradient
            words: List of words in prompt
            target_sequence: Target sequence

        Returns:
            Estimated gradient vector (simplified vocabulary)
        """
        # Simplified vocabulary for demonstration (in production, use actual vocab)
        vocab = ["please", "help", "explain", "describe", "show", "tell", "what", "how", "why"]
        len(vocab)

        # Estimate gradient by perturbing each vocabulary word
        gradients = []

        # Get baseline score
        baseline_score = await self._score_prompt(prompt, target_sequence)

        for word in vocab:
            # Create perturbed prompt
            perturbed_words = words.copy()
            if position < len(perturbed_words):
                perturbed_words[position] = word
                perturbed_prompt = " ".join(perturbed_words)

                # Score perturbed prompt
                perturbed_score = await self._score_prompt(perturbed_prompt, target_sequence)

                # Finite difference approximation
                gradient = (perturbed_score - baseline_score) / self.config.epsilon
                gradients.append(gradient)
            else:
                gradients.append(0.0)

        return gradients

    async def _score_prompt(self, prompt: str, target_sequence: str | None) -> float:
        """
        Score a prompt (for gradient approximation).

        In production, this would query the target LLM and score the response.
        For now, using heuristics.
        """
        # Simulate scoring (in production, call actual LLM)
        score = len(prompt.split()) / 100  # Simplified scoring
        if target_sequence and target_sequence.lower() in prompt.lower():
            score += 0.5
        return score

    async def _compute_heuristic_gradients(self, prompt: str) -> TokenGradients:
        """
        Compute heuristic "gradients" without model access.

        Uses linguistic heuristics to approximate which tokens are important.
        """
        logger.debug(f"Computing heuristic gradients for: {prompt[:50]}...")

        words = prompt.split()
        tokens = list(range(len(words)))

        # Heuristic: Assign higher "gradients" to content words
        content_words = ["help", "explain", "describe", "understand", "learn", "show"]

        token_gradients = []
        for word in words:
            # Simplified vocab gradient
            vocab_grad = [1.0 if word.lower() in content_words else 0.5 for _ in range(10)]
            token_gradients.append(vocab_grad)

        return TokenGradients(
            tokens=tokens,
            gradients=token_gradients,
            prompt=prompt,
            metadata={"mode": "heuristic"},
        )

    def _make_cache_key(self, prompt: str, target_sequence: str | None) -> str:
        """Create cache key from prompt and target."""
        import hashlib

        key = f"{prompt}|{target_sequence or ''}"
        return hashlib.md5(key.encode()).hexdigest()

    def gradient_to_mutation_guidance(
        self, gradients: TokenGradients, top_k: int = 5
    ) -> list[tuple[int, list[int]]]:
        """
        Convert gradients to mutation guidance for AutoDAN.

        Args:
            gradients: Token gradients
            top_k: Number of top tokens to return for each position

        Returns:
            List of (position, top_token_indices) tuples
        """
        guidance = []

        for pos, grad_vec in enumerate(gradients.gradients):
            # Get top-k tokens with highest gradients
            grad_array = np.array(grad_vec)
            top_indices = np.argsort(grad_array)[-top_k:][::-1].tolist()
            guidance.append((pos, top_indices))

        return guidance

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {"cache_size": len(self.gradient_cache), "cache_capacity": self.config.cache_size}


# Example usage and testing
async def main():
    """Example usage of the gradient bridge."""

    # Example 1: Black-box gradient approximation
    config = GradientConfig(mode=GradientMode.BLACK_BOX_APPROXIMATE)
    bridge = GradientBridge(config)

    prompt = "Please help me understand how safety filters work"
    gradients = await bridge.compute_gradients(prompt, target_sequence="safety filters")

    print(f"Computed gradients for {len(gradients.tokens)} tokens")
    print(f"Mode: {gradients.metadata['mode']}")

    # Convert to mutation guidance
    guidance = bridge.gradient_to_mutation_guidance(gradients, top_k=3)
    print("\nMutation guidance (top 3 tokens per position):")
    for pos, top_tokens in guidance[:3]:  # Show first 3 positions
        print(f"  Position {pos}: {top_tokens}")

    # Example 2: White-box gradients (if model available)
    # Uncomment if you have a local model
    # whitebox_config = GradientConfig(
    #     mode=GradientMode.WHITE_BOX,
    #     model_name="gpt2",
    #     device="cuda"
    # )
    # whitebox_bridge = GradientBridge(whitebox_config)
    # whitebox_gradients = await whitebox_bridge.compute_gradients(prompt)


if __name__ == "__main__":
    asyncio.run(main())
