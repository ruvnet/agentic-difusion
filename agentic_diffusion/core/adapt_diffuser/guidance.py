"""
Reward-based guidance for AdaptDiffuser model.

This module implements guidance mechanisms that use reward models to
steer the diffusion sampling process toward higher quality solutions.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Union, Dict, Any, Callable, Tuple

from agentic_diffusion.core.common_types import RewardModelProtocol, TaskEmbeddingModelProtocol

# Configure logging
logger = logging.getLogger(__name__)


class AdaptDiffuserGuidance:
    """
    Guidance mechanisms for AdaptDiffuser reward-based sampling.
    
    This class implements various methods for computing guidance directions
    based on reward models and task embeddings.
    """
    
    def __init__(
        self,
        reward_model: Optional[RewardModelProtocol] = None,
        task_embedding_model: Optional[TaskEmbeddingModelProtocol] = None,
        device: str = None
    ):
        """
        Initialize guidance handler.
        
        Args:
            reward_model: Model for computing rewards and gradients
            task_embedding_model: Model for encoding task descriptions
            device: Device to use for computations
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_model = reward_model
        self.task_embedding_model = task_embedding_model
        
        # Configuration for adaptive guidance
        self.default_cfg = {
            "min_step_percent": 0.1,  # Don't guide in first 10% of steps (too noisy)
            "max_step_percent": 0.9,  # Don't guide in last 10% of steps (details)
            "guidance_scale_min": 0.1,  # Minimum guidance scale
            "guidance_scale_max": 5.0,  # Maximum guidance scale
            "warmup_steps": 100,       # Steps before using full guidance
            "use_exponent": True,      # Use exponential scaling
            "adaptive_factor": 2.0     # Scaling factor
        }
    
    def encode_task(
        self,
        task: Union[str, torch.Tensor, None]
    ) -> Optional[torch.Tensor]:
        """
        Encode a task description into an embedding.
        
        Args:
            task: Task description or embedding
            
        Returns:
            Task embedding or None if not available
        """
        if task is None:
            return None
            
        if isinstance(task, torch.Tensor):
            return task.to(self.device)
            
        if self.task_embedding_model is None:
            logger.warning("Task embedding model not available for task encoding")
            return None
            
        try:
            task_embedding = self.task_embedding_model.encode(task)
            return task_embedding.to(self.device)
        except Exception as e:
            logger.error(f"Error encoding task: {e}")
            return None
    
    def compute_reward_gradient(
        self,
        samples: torch.Tensor,
        task: Optional[Union[str, torch.Tensor]] = None,
        noise_level: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient of reward with respect to samples.
        
        Args:
            samples: Input samples to compute gradient for
            task: Optional task identifier or embedding
            noise_level: Optional noise level tensor
            
        Returns:
            Gradient tensor of shape matching samples
            
        Raises:
            ValueError: If reward model is not available
        """
        if self.reward_model is None:
            raise ValueError("Reward model not available for gradient computation")
        
        # Create copy of samples that requires grad
        samples_with_grad = samples.detach().clone().to(self.device)
        samples_with_grad.requires_grad_(True)
        
        # Encode task if needed
        task_embedding = self.encode_task(task)
        
        try:
            # Forward pass through reward model
            if task_embedding is not None:
                reward = self.reward_model.compute_reward(samples_with_grad, task_embedding)
            else:
                reward = self.reward_model.compute_reward(samples_with_grad)
            
            # Compute gradient
            reward.sum().backward()
            
            # Get gradient
            grad = samples_with_grad.grad.detach()
            
            # Apply noise level scaling if provided
            if noise_level is not None:
                # Scale gradient inversely with noise level
                # Higher noise = smaller gradient, lower noise = larger gradient
                scale = 1.0 / (1.0 + noise_level)
                grad = grad * scale.view(-1, 1, 1, 1)
            
            return grad
            
        except Exception as e:
            logger.error(f"Error computing reward gradient: {e}")
            # Return zero gradient in case of error
            return torch.zeros_like(samples)
    
    def compute_adaptive_guidance_scale(
        self,
        step: int,
        num_steps: int,
        base_scale: float,
        min_percent: float = None,
        max_percent: float = None,
        warmup_steps: int = None,
        adaptive_factor: float = None,
        use_exponent: bool = None
    ) -> float:
        """
        Compute adaptive guidance scale based on denoising step.
        
        Args:
            step: Current denoising step
            num_steps: Total denoising steps
            base_scale: Base guidance scale
            min_percent: Minimum percent of steps to apply guidance
            max_percent: Maximum percent of steps to apply guidance
            warmup_steps: Steps before using full guidance
            adaptive_factor: Scaling factor for adaptation
            use_exponent: Whether to use exponential scaling
            
        Returns:
            Adjusted guidance scale
        """
        # Use defaults if not provided
        min_percent = min_percent if min_percent is not None else self.default_cfg["min_step_percent"]
        max_percent = max_percent if max_percent is not None else self.default_cfg["max_step_percent"]
        warmup_steps = warmup_steps if warmup_steps is not None else self.default_cfg["warmup_steps"]
        adaptive_factor = adaptive_factor if adaptive_factor is not None else self.default_cfg["adaptive_factor"]
        use_exponent = use_exponent if use_exponent is not None else self.default_cfg["use_exponent"]
        
        # Convert to ratio (0 = start, 1 = end)
        step_ratio = 1.0 - (step / max(1, num_steps - 1))
        
        # Only apply guidance between min and max percentages
        if step_ratio < min_percent or step_ratio > max_percent:
            return 0.0
        
        # Normalize to 0-1 within active range
        norm_ratio = (step_ratio - min_percent) / (max_percent - min_percent)
        
        # Apply warmup if in early steps
        if num_steps > warmup_steps and step < warmup_steps:
            warmup_scale = step / warmup_steps
            base_scale = base_scale * warmup_scale
        
        # Apply adaptive scaling based on step ratio
        if use_exponent:
            # Exponential scaling - stronger in middle, weaker at ends
            # exp(-(x-0.5)^2 / 0.25) gives a bell curve centered at 0.5
            bell_curve = torch.exp(torch.tensor(-((norm_ratio - 0.5) ** 2) / 0.25))
            scale = base_scale * (1.0 + adaptive_factor * bell_curve.item())
        else:
            # Linear scaling - strongest in middle
            scale = base_scale * (1.0 + adaptive_factor * norm_ratio * (1.0 - norm_ratio))
        
        return scale
    
    def combine_guidance_signals(
        self,
        reward_grad: torch.Tensor,
        task_grad: Optional[torch.Tensor] = None,
        prior_grad: Optional[torch.Tensor] = None,
        reward_scale: float = 1.0,
        task_scale: float = 0.5,
        prior_scale: float = 0.3
    ) -> torch.Tensor:
        """
        Combine multiple guidance signals.
        
        Args:
            reward_grad: Gradient from reward model
            task_grad: Gradient from task conditioning
            prior_grad: Gradient from prior knowledge
            reward_scale: Weight for reward gradient
            task_scale: Weight for task gradient
            prior_scale: Weight for prior gradient
            
        Returns:
            Combined gradient tensor
        """
        # Start with reward gradient
        combined_grad = reward_grad * reward_scale
        
        # Add task gradient if available
        if task_grad is not None:
            combined_grad = combined_grad + task_grad * task_scale
            
        # Add prior gradient if available
        if prior_grad is not None:
            combined_grad = combined_grad + prior_grad * prior_scale
            
        return combined_grad
    
    def guide_samples(
        self,
        samples: torch.Tensor,
        step: int,
        num_steps: int,
        task: Optional[Union[str, torch.Tensor]] = None,
        noise_level: Optional[torch.Tensor] = None,
        base_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Apply guidance to samples during denoising.
        
        Args:
            samples: Current denoised samples
            step: Current denoising step
            num_steps: Total denoising steps
            task: Optional task identifier or embedding
            noise_level: Optional noise level tensor
            base_scale: Base guidance scale
            
        Returns:
            Guided samples
        """
        # Compute adaptive guidance scale
        adaptive_scale = self.compute_adaptive_guidance_scale(
            step=step,
            num_steps=num_steps,
            base_scale=base_scale
        )
        
        # If scale is zero, return unmodified samples
        if adaptive_scale <= 0:
            return samples
        
        try:
            # Compute reward gradient
            reward_grad = self.compute_reward_gradient(
                samples=samples,
                task=task,
                noise_level=noise_level
            )
            
            # Apply guidance
            guided_samples = samples + adaptive_scale * reward_grad
            
            # Log guidance application
            if step % max(1, num_steps // 10) == 0:
                grad_norm = torch.norm(reward_grad.view(reward_grad.size(0), -1), dim=1).mean().item()
                logger.debug(f"Step {step}/{num_steps}: Applied guidance with scale {adaptive_scale:.4f}, gradient norm {grad_norm:.4f}")
                
            return guided_samples
            
        except Exception as e:
            logger.error(f"Error applying guidance: {e}")
            return samples  # Return unmodified samples in case of error