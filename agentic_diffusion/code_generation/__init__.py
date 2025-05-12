"""
Code generation module using diffusion models.

This package provides diffusion-based code generation capabilities,
with a modular architecture for flexibility and extensibility.
"""

# Export main components from diffusion module
from agentic_diffusion.code_generation.diffusion.code_diffusion import CodeDiffusion
from agentic_diffusion.code_generation.diffusion.diffusion_model import CodeDiffusionModel

# Export model components
from agentic_diffusion.code_generation.models.code_unet import CodeUNet, CodeClassifierFreeGuidanceUNet
from agentic_diffusion.code_generation.models.embeddings import CodeEmbedding, TimestepEmbedding
from agentic_diffusion.code_generation.models.blocks import TransformerBlock, CrossAttentionBlock, ResidualBlock

# Export scheduler components
from agentic_diffusion.code_generation.schedulers.discrete_scheduler import CodeDiscreteScheduler

# Export utility functions
from agentic_diffusion.code_generation.utils.diffusion_utils import (
    sinusoidal_embedding,
    categorical_diffusion,
    token_accuracy,
    reconstruct_code_from_tokens,
    calculate_perplexity
)

# Export high-level functions
from agentic_diffusion.code_generation.generation import (
    create_code_diffusion_model,
    generate_code,
    complete_code,
    refine_code,
    evaluate_code_quality
)

__all__ = [
    # Diffusion models
    'CodeDiffusion',
    'CodeDiffusionModel',
    
    # Neural network models
    'CodeUNet',
    'CodeClassifierFreeGuidanceUNet',
    'CodeEmbedding',
    'TimestepEmbedding',
    'TransformerBlock',
    'CrossAttentionBlock',
    'ResidualBlock',
    
    # Schedulers
    'CodeDiscreteScheduler',
    
    # Utility functions
    'sinusoidal_embedding',
    'categorical_diffusion',
    'token_accuracy',
    'reconstruct_code_from_tokens',
    'calculate_perplexity',
    
    # High-level functions
    'create_code_diffusion_model',
    'generate_code',
    'complete_code',
    'refine_code',
    'evaluate_code_quality'
]

# Version
__version__ = '0.1.0'