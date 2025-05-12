"""
Main entry point for code generation with diffusion models.

This module provides a high-level interface for code generation
using diffusion models. It imports and re-exports the modularized components
from the code_generation package.

Note: This module is maintained for backward compatibility.
New code should import directly from agentic_diffusion.code_generation.generation.
"""

# Re-export all components from generation.py for backward compatibility
from agentic_diffusion.code_generation.generation import (
    create_code_diffusion_model,
    generate_code,
    complete_code,
    refine_code,
    evaluate_code_quality
)

# Re-export key components for backward compatibility
from agentic_diffusion.code_generation.diffusion import CodeDiffusion, CodeDiffusionModel
from agentic_diffusion.code_generation.models import CodeUNet, CodeEmbedding, TransformerBlock
from agentic_diffusion.code_generation.schedulers import CodeDiscreteScheduler
from agentic_diffusion.code_generation.utils.diffusion_utils import (
    sinusoidal_embedding,
    categorical_diffusion,
    token_accuracy,
    reconstruct_code_from_tokens,
    calculate_perplexity
)

# Module exports
__all__ = [
    'CodeDiffusion',
    'CodeDiffusionModel',
    'CodeUNet',
    'CodeEmbedding',
    'TransformerBlock',
    'CodeDiscreteScheduler',
    'create_code_diffusion_model',
    'generate_code',
    'complete_code',
    'refine_code',
    'evaluate_code_quality'
]