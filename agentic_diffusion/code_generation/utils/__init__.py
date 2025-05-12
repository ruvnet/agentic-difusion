"""
Utility functions for code generation with diffusion models.

This package contains various utility functions used in
code generation with diffusion models, including token processing
and diffusion-specific utilities.
"""

from agentic_diffusion.code_generation.utils.diffusion_utils import (
    sinusoidal_embedding,
    categorical_diffusion,
    token_accuracy,
    reconstruct_code_from_tokens,
    calculate_perplexity
)

__all__ = [
    'sinusoidal_embedding',
    'categorical_diffusion',
    'token_accuracy',
    'reconstruct_code_from_tokens',
    'calculate_perplexity'
]