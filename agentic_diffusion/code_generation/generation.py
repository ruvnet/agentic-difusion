"""
High-level code generation API functions.

This module provides a high-level API for code generation with diffusion models,
offering functions for generating, completing, and refining code.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Import required components
from agentic_diffusion.code_generation.diffusion.code_diffusion import CodeDiffusion

# Setup logging
logger = logging.getLogger(__name__)

def create_code_diffusion_model(
    vocab_size: int = 10000,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_timesteps: int = 1000,
    device: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None
) -> CodeDiffusion:
    """
    Create a code diffusion model with specified parameters.
    
    Args:
        vocab_size: Size of the code token vocabulary
        embedding_dim: Dimension of token embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_timesteps: Number of diffusion timesteps
        device: Device to use for computation
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Initialized CodeDiffusion instance
    """
    logger.info(f"Creating code diffusion model with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    # Create the code diffusion model
    model = CodeDiffusion(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_timesteps=num_timesteps,
        device=device,
        config_path=config_path,
        checkpoint_path=checkpoint_path
    )
    
    return model

def generate_code(
    specification: str,
    language: str = "python",
    partial_code: Optional[str] = None,
    model: Optional[CodeDiffusion] = None,
    max_length: int = 512,
    num_samples: int = 5,
    guidance_scale: float = 1.5,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Generate code from a natural language specification.
    
    This is a high-level function that provides an easy-to-use interface
    for code generation, handling model creation if needed.
    
    Args:
        specification: Text description of the code to generate
        language: Programming language to generate
        partial_code: Optional partial code to complete
        model: Optional pre-initialized CodeDiffusion model
        max_length: Maximum code length to generate
        num_samples: Number of candidates to generate and select from
        guidance_scale: Scale for classifier-free guidance
        temperature: Sampling temperature (lower = more deterministic)
        **kwargs: Additional parameters passed to the model
        
    Returns:
        Generated code as a string
    """
    # Create a model if not provided
    if model is None:
        logger.info("Creating default code diffusion model")
        model = create_code_diffusion_model(**kwargs)
    
    # Generate code
    generated_code = model.generate_code(
        specification=specification,
        language=language,
        partial_code=partial_code,
        max_length=max_length,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        temperature=temperature
    )
    
    return generated_code

def complete_code(
    partial_code: str,
    language: str = "python",
    context: Optional[str] = None,
    model: Optional[CodeDiffusion] = None,
    **kwargs
) -> str:
    """
    Complete partial code using diffusion-based generation.
    
    Args:
        partial_code: Code to complete
        language: Programming language of the code
        context: Optional additional context or specification
        model: Optional pre-initialized CodeDiffusion model
        **kwargs: Additional parameters passed to generate_code
        
    Returns:
        Completed code as a string
    """
    # Create a specification from the context or infer it from partial code
    if context:
        specification = context
    else:
        # Generate a generic specification based on the language
        specification = f"Complete the following {language} code"
    
    # Use the generate_code function for completion
    return generate_code(
        specification=specification,
        language=language,
        partial_code=partial_code,
        model=model,
        **kwargs
    )

def refine_code(
    code: str,
    specification: str,
    language: str = "python",
    model: Optional[CodeDiffusion] = None,
    num_iterations: int = 3,
    **kwargs
) -> str:
    """
    Refine existing code to improve its quality or match a specification better.
    
    Args:
        code: Existing code to refine
        specification: Target specification for refinement
        language: Programming language of the code
        model: Optional pre-initialized CodeDiffusion model
        num_iterations: Number of refinement iterations
        **kwargs: Additional parameters passed to model
        
    Returns:
        Refined code as a string
    """
    # Create a model if not provided
    if model is None:
        model = create_code_diffusion_model(**kwargs)
    
    # Use the model's refine_code method if available
    if hasattr(model, 'refine_code'):
        refined_code = model.refine_code(
            initial_code=code,
            num_iterations=num_iterations,
            language=language,
            **kwargs
        )
    else:
        # Fall back to using generate_code with partial code
        refined_code = generate_code(
            specification=specification,
            language=language,
            partial_code=code,
            model=model,
            **kwargs
        )
    
    return refined_code

def evaluate_code_quality(
    code: str,
    specification: str = "",
    language: str = "python",
    model: Optional[CodeDiffusion] = None
) -> Dict[str, float]:
    """
    Evaluate the quality of generated code.
    
    Args:
        code: Code to evaluate
        specification: Original specification for relevance evaluation
        language: Programming language of the code
        model: Optional pre-initialized CodeDiffusion model
        
    Returns:
        Dictionary of quality metrics
    """
    # Create a model if not provided (needed for reward models)
    if model is None:
        model = create_code_diffusion_model()
    
    # Use the model's evaluate_code_quality method
    metrics = model.evaluate_code_quality(
        code=code,
        specification=specification,
        language=language
    )
    
    return metrics