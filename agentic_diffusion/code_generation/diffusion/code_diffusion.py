"""
High-level interface for code generation with diffusion models.

This module provides the CodeDiffusion class, which serves as the main
interface for generating code using diffusion models. It handles model
configuration, loading, and provides convenient methods for different
code generation tasks.
"""

import os
import logging
import time
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from agentic_diffusion.code_generation.diffusion.diffusion_model import CodeDiffusionModel
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.utils.diffusion_utils import (
    token_accuracy,
    calculate_perplexity
)

class CodeDiffusion:
    """
    High-level interface for code generation using diffusion models.
    
    This class provides an easy-to-use interface for code generation tasks,
    including generating new code, completing partial code, and refining
    existing code using diffusion models.
    """
    
    def __init__(
        self,
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
    ):
        """
        Initialize the CodeDiffusion interface.
        
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
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing CodeDiffusion with embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
            # Override parameters from config if available
            vocab_size = self.config.get("vocab_size", vocab_size)
            embedding_dim = self.config.get("embedding_dim", embedding_dim)
            hidden_dim = self.config.get("hidden_dim", hidden_dim)
            num_layers = self.config.get("num_layers", num_layers)
            num_heads = self.config.get("num_heads", num_heads)
            dropout = self.config.get("dropout", dropout)
            num_timesteps = self.config.get("num_timesteps", num_timesteps)
            
            # Update device if specified in config
            if "device" in self.config:
                self.device = self.config.get("device")
                self.logger.info(f"Config override - Using device: {self.device}")
        else:
            self.config = {
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "num_timesteps": num_timesteps,
                "device": self.device
            }
        
        # Create the diffusion model
        self.model = CodeDiffusionModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_timesteps=num_timesteps,
            device=self.device
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        
        # Tokenizer cache for different languages
        self.tokenizers = {}
        
        # Keep track of total tokens generated
        self.total_tokens_generated = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        import json
        import yaml
        
        try:
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    config = json.load(f)
            elif config_path.endswith((".yaml", ".yml")):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                self.logger.warning(f"Unsupported config file format: {config_path}")
                config = {}
                
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model weights if they exist in the checkpoint
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
                self.logger.info(f"Loaded model weights from {checkpoint_path}")
            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.logger.info(f"Loaded model state dict from {checkpoint_path}")
            else:
                # Try direct loading
                self.model.load_state_dict(checkpoint)
                self.logger.info(f"Loaded model weights directly from {checkpoint_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
    
    def _get_tokenizer(self, language: str) -> CodeTokenizer:
        """
        Get a tokenizer for the specified language.
        
        Args:
            language: Programming language
            
        Returns:
            Code tokenizer for the language
        """
        if language not in self.tokenizers:
            self.tokenizers[language] = CodeTokenizer(language=language)
        return self.tokenizers[language]
    
    def generate_code(
        self,
        specification: str,
        language: str = "python",
        partial_code: Optional[str] = None,
        max_length: int = 512,
        num_samples: int = 5,
        guidance_scale: float = 1.5,
        temperature: float = 0.7,
        batch_size: int = 4,
        precision: str = "float32",
        device: Optional[str] = None,
        use_rewards: bool = True,
        num_iterations: int = 1
    ) -> str:
        """
        Generate code from a natural language specification.
        
        Args:
            specification: Text description of the code to generate
            language: Programming language to generate
            partial_code: Optional partial code to complete
            max_length: Maximum code length to generate
            num_samples: Number of candidates to generate and select from
            guidance_scale: Scale for classifier-free guidance
            temperature: Sampling temperature (lower = more deterministic)
            batch_size: Number of samples to generate (for API compatibility)
            precision: Precision to use for computation (e.g., "float32", "float16")
            device: Device to use for generation (overrides instance device if provided)
            use_rewards: Whether to use reward models for quality assessment
            num_iterations: Number of generation iterations for refinement
            
        Returns:
            Generated code as a string
        """
        self.logger.info(f"Generating {language} code for specification: {specification[:50]}...")
        
        # Get tokenizer for the language
        tokenizer = self._get_tokenizer(language)
        
        # Record generation start time
        start_time = time.time()
        
        # Prepare parameters for the model's generate method
        generation_kwargs = {
            'specification': specification,
            'language': language,
            'partial_code': partial_code,
            'tokenizer': tokenizer,
            'max_length': max_length,
            'guidance_scale': guidance_scale,
            'temperature': temperature
        }
        
        # Determine the actual number of samples to use
        # Use batch_size if specified, otherwise use num_samples
        actual_samples = batch_size if 'batch_size' in locals() else num_samples
        generation_kwargs['num_samples'] = actual_samples
        
        # Generate code using the diffusion model
        # Generate code using the diffusion model
        try:
            generated_code = self.model.generate(**generation_kwargs)
        except Exception as e:
            self.logger.error(f"Error in diffusion model generation: {e}")
            
            # Check specifically for dimension mismatch errors
            if isinstance(e, AssertionError) and "embedding dimension" in str(e):
                self.logger.error("Detected dimension mismatch in UNet model. This is likely due to incompatible dimensions in condition blocks.")
                raise RuntimeError("Dimension mismatch in diffusion model. Please ensure the CodeUNet has been updated to handle dynamic dimensions.") from e
            
            # Check for other common issues
            if "CUDA out of memory" in str(e):
                self.logger.error("CUDA out of memory error. Try reducing batch_size or max_length.")
                raise RuntimeError("GPU memory exhausted during code generation. Please reduce batch size or model parameters.") from e
                
            # Fallback with minimal parameters
            try:
                self.logger.info("Attempting fallback with minimal parameters...")
                generated_code = self.model.generate(
                    specification=specification,
                    language=language,
                    partial_code=partial_code,
                    tokenizer=tokenizer
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback generation also failed: {fallback_error}")
                raise RuntimeError("Diffusion model failed to generate valid code after fallback attempt") from fallback_error
        # Record generation end time
        generation_time = time.time() - start_time
        
        # Estimate token count for tracking
        token_count = len(generated_code.split())
        self.total_tokens_generated += token_count
        
        self.logger.info(
            f"Generated {token_count} tokens in {generation_time:.2f}s "
            f"({token_count/generation_time:.2f} tokens/sec)"
        )
        
        return generated_code
    
    def complete_code(
        self,
        partial_code: str,
        language: str = "python",
        context: Optional[str] = None,
        max_length: int = 512,
        **kwargs
    ) -> str:
        """
        Complete partial code using the diffusion model.
        
        Args:
            partial_code: Code to complete
            language: Programming language of the code
            context: Optional additional context or specification
            max_length: Maximum code length for the completed code
            **kwargs: Additional parameters for code generation
            
        Returns:
            Completed code as a string
        """
        # Create a specification from the context or infer it from partial code
        if context:
            specification = context
        else:
            # Generate a generic specification based on the language
            specification = f"Complete the following {language} code"
        
        # Use the generate_code method with partial code
        return self.generate_code(
            specification=specification,
            language=language,
            partial_code=partial_code,
            max_length=max_length,
            **kwargs
        )
    
    def refine_code(
        self,
        initial_code: str,
        specification: str = None,
        language: str = "python",
        num_iterations: int = 3,
        **kwargs
    ) -> str:
        """
        Refine existing code through multiple iterations of the diffusion process.
        
        Args:
            initial_code: Code to refine
            specification: Description of the desired refinement
            language: Programming language of the code
            num_iterations: Number of refinement iterations
            **kwargs: Additional parameters for code generation
            
        Returns:
            Refined code as a string
        """
        self.logger.info(f"Refining {language} code with {num_iterations} iterations")
        
        # Start with the initial code
        current_code = initial_code
        
        # Create a default specification if none provided
        if specification is None:
            specification = f"Refine and improve the following {language} code"
        
        # Get tokenizer for the language
        tokenizer = self._get_tokenizer(language)
        
        # Perform multiple refinement iterations
        for iteration in range(num_iterations):
            self.logger.info(f"Refinement iteration {iteration+1}/{num_iterations}")
            
            # Create a noised version of the current code
            # This involves tokenizing, adding noise, and refining
            tokens = tokenizer.tokenize(current_code) if hasattr(tokenizer, 'tokenize') else current_code.split()
            
            # Use partial_code as a starting point with reduced noise
            # This keeps the original structure while allowing for refinements
            noise_level = 0.3 + (0.3 * (iteration / num_iterations))  # Increase noise gradually
            
            # Use the original specification with the current code as partial
            refined_code = self.generate_code(
                specification=specification,
                language=language,
                partial_code=current_code,
                temperature=0.8 - (0.1 * iteration),  # Reduce temperature with iterations
                guidance_scale=1.5 + (0.2 * iteration),  # Increase guidance with iterations
                **kwargs
            )
            
            # Check if the refinement improved the code
            if self.evaluate_code_quality(refined_code, specification, language).get("combined", 0) > \
               self.evaluate_code_quality(current_code, specification, language).get("combined", 0):
                current_code = refined_code
                self.logger.info("Refinement improved code quality")
            else:
                self.logger.info("Refinement did not improve code quality, keeping previous version")
        
        return current_code
    
    def evaluate_code_quality(
        self,
        code: str,
        specification: str = "",
        language: str = "python"
    ) -> Dict[str, float]:
        """
        Evaluate the quality of code.
        
        Args:
            code: Code to evaluate
            specification: Original specification for relevance evaluation
            language: Programming language of the code
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Use the model's internal quality evaluation method
        reward = self.model._compute_quality_reward(code, specification, language)
        metrics["combined"] = reward
        
        # Add more detailed metrics if available
        try:
            from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward
            from agentic_diffusion.code_generation.rewards.quality_reward import QualityReward
            from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward
            
            syntax_reward = SyntaxReward()
            quality_reward = QualityReward()
            relevance_reward = RelevanceReward()
            
            metrics["syntax"] = syntax_reward.evaluate(code, language)
            metrics["quality"] = quality_reward.evaluate(code, language)
            
            if hasattr(relevance_reward, 'evaluate_with_reference'):
                metrics["relevance"] = relevance_reward.evaluate_with_reference(code, specification, language)
            else:
                metrics["relevance"] = relevance_reward.evaluate(code, reference=specification, language=language)
                
        except ImportError as e:
            self.logger.warning(f"Could not import reward models for detailed evaluation: {e}")
        
        return metrics
    def generate(
        self,
        specification: str,
        language: Optional[str] = None,
        partial_code: Optional[str] = None,
        batch_size: int = 4,
        precision: str = "float32",
        device: Optional[str] = None,
        guidance_scale: float = 1.5,
        temperature: float = 0.7,
        use_rewards: bool = True,
        max_length: int = 512,
        num_iterations: int = 1,
        **kwargs
    ) -> str:
        """
        Compatibility method that delegates to generate_code.
        
        This method provides compatibility with the common generate interface
        used across different code generation models.
        
        Args:
            specification: Text description of the code to generate
            language: Programming language to generate (defaults to Python if None)
            partial_code: Optional partial code to complete
            **kwargs: Additional parameters passed to generate_code
            
        Returns:
            Generated code as a string
        """
        # Default to Python if language is None
        actual_language = language if language is not None else "python"
        
        # Extract supported parameters from kwargs
        generation_kwargs = {
            'specification': specification,
            'language': actual_language,
            'partial_code': partial_code
        }
        
        # Copy over other supported parameters from kwargs
        supported_params = [
            'max_length', 'num_samples', 'guidance_scale',
            'temperature', 'batch_size', 'precision', 'device',
            'use_rewards', 'num_iterations'
        ]
        
        for param in supported_params:
            if param in kwargs:
                generation_kwargs[param] = kwargs[param]
        
        # Delegate to the main generate_code method
        try:
            return self.generate_code(**generation_kwargs)
        except Exception as e:
            self.logger.error(f"Error in generate method: {e}")
            
            # Check for specific error types to provide better error messages
            if isinstance(e, AssertionError) and "embedding dimension" in str(e):
                self.logger.error("Detected dimension mismatch in UNet model. This is likely due to incompatible dimensions in condition blocks.")
                raise RuntimeError("Dimension mismatch in diffusion model. Please ensure the CodeUNet has been updated to handle dynamic dimensions.") from e
            
            # Try one more time with minimal parameters if the error isn't a dimension mismatch
            try:
                self.logger.info("Attempting fallback with minimal parameters...")
                return self.generate_code(
                    specification=specification,
                    language=actual_language,
                    partial_code=partial_code
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback generation also failed: {fallback_error}")
                raise RuntimeError("Diffusion model failed to generate valid code after fallback attempt") from fallback_error
    
    def sample(
        self,
        specification: str,
        language: Optional[str] = None,
        partial_code: Optional[str] = None,
        batch_size: int = 4,
        precision: str = "float32",
        device: Optional[str] = None,
        guidance_scale: float = 1.5,
        temperature: float = 0.7,
        use_rewards: bool = True,
        max_length: int = 512,
        num_iterations: int = 1,
        **kwargs
    ) -> str:
        """
        Compatibility method that delegates to generate_code.
        
        This method provides compatibility with sample-based code generation interfaces.
        It handles tokenizer parameters that might be passed from other generators.
        
        Args:
            specification: Text description of the code to generate
            language: Programming language to generate (defaults to Python if None)
            partial_code: Optional partial code to complete
            **kwargs: Additional parameters passed to generate_code
            
        Returns:
            Generated code as a string
        """
        # Default to Python if language is None
        actual_language = language if language is not None else "python"
        
        # Remove tokenizer parameter if it exists since our generate_code
        # method gets the tokenizer internally
        if "tokenizer" in kwargs:
            self.logger.debug("Removing tokenizer parameter from kwargs for compatibility")
            kwargs.pop("tokenizer")
            
        # Extract supported parameters from kwargs
        generation_kwargs = {
            'specification': specification,
            'language': actual_language,
            'partial_code': partial_code
        }
        
        # Copy over other supported parameters from kwargs
        supported_params = [
            'max_length', 'num_samples', 'guidance_scale',
            'temperature', 'batch_size', 'precision', 'device',
            'use_rewards', 'num_iterations'
        ]
        
        for param in supported_params:
            if param in kwargs:
                generation_kwargs[param] = kwargs[param]
        
        # Delegate to the main generate_code method
        # Delegate to the main generate_code method
        try:
            return self.generate_code(**generation_kwargs)
        except Exception as e:
            self.logger.error(f"Error in sample method: {e}")
            
            # Check for specific error types to provide better error messages
            if isinstance(e, AssertionError) and "embedding dimension" in str(e):
                self.logger.error("Detected dimension mismatch in UNet model. This is likely due to incompatible dimensions in condition blocks.")
                raise RuntimeError("Dimension mismatch in diffusion model. Please ensure the CodeUNet has been updated to handle dynamic dimensions.") from e
            
            # Try one more time with minimal parameters if the error isn't a dimension mismatch
            try:
                self.logger.info("Attempting fallback with minimal parameters...")
                return self.generate_code(
                    specification=specification,
                    language=actual_language,
                    partial_code=partial_code
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback generation also failed: {fallback_error}")
                raise RuntimeError("Diffusion model failed to generate valid code after fallback attempt") from fallback_error
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save the model to a checkpoint file.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Create checkpoint dictionary
        checkpoint = {
            "model": self.model.state_dict(),
            "config": self.config,
            "total_tokens_generated": self.total_tokens_generated
        }
        
        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved model checkpoint to {checkpoint_path}")
        self.logger.info(f"Saved model checkpoint to {checkpoint_path}")