"""
Noise scheduling for discrete diffusion models.

This module provides schedulers specifically designed for discrete
token-based diffusion processes in code generation models.
"""

import torch
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any

from agentic_diffusion.core.noise_schedules import NoiseScheduler

class CodeDiscreteScheduler(NoiseScheduler):
    """
    Noise scheduler for discrete token diffusion.
    
    This scheduler adapts diffusion noise schedules to work with discrete
    tokens (code) instead of continuous values, handling the categorical
    nature of code tokens.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the discrete scheduler.
        
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            beta_schedule: Schedule type ('linear', 'cosine', 'sqrt', 'squared')
            variance_type: Type of variance to use
            clip_sample: Whether to clip samples to a valid range
            device: Device to use for computation
        """
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the noise schedule
        self._init_noise_schedule()
        
    def _init_noise_schedule(self) -> None:
        """
        Initialize the noise schedule parameters.
        """
        # Create the beta schedule
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32
            )
        elif self.beta_schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        elif self.beta_schedule == "sqrt":
            self.betas = torch.linspace(
                self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps, dtype=torch.float32
            ) ** 2
        elif self.beta_schedule == "squared":
            self.betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32
            ) ** 2
        else:
            raise ValueError(f"Unsupported beta schedule: {self.beta_schedule}")
        
        # Move tensors to device
        self.betas = self.betas.to(self.device)
        
        # Calculate alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        
        # Calculate derived values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance calculations (q(x_{t-1} | x_t, x_0))
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Clamping for stable log
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        # Posterior mean coefficient calculations
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples at specified timesteps.
        
        Args:
            original_samples: Original samples (token indices)
            noise: Noise to add
            timesteps: Timesteps at which to add noise
            
        Returns:
            Noisy samples
        """
        # Ensure timesteps are in the correct range
        timesteps = torch.clamp(timesteps, 0, self.num_timesteps - 1)
        
        # Get alphas and betas for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        
        # Add noise based on the diffusion formula
        noisy_samples = sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise
        
        # For discrete tokens, we need to handle this differently
        # This is a categorical diffusion process, so we'll use a sampling approach
        noise_level = sqrt_one_minus_alphas_cumprod_t.squeeze(-1)
        
        # Apply categorical sampling
        # For each token, with probability (1-noise_level), keep original token
        # With probability noise_level, sample a random token
        mask = torch.rand_like(original_samples.float()) < noise_level
        
        # Create random indices for noise
        random_indices = torch.randint_like(original_samples, 0, original_samples.shape[-1])
        
        # Apply mask to choose between original and random tokens
        noisy_samples = torch.where(mask, random_indices, original_samples)
        
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        predict_epsilon: bool = True,
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a denoising step.
        
        Args:
            model_output: Output from model
            timestep: Current timestep
            sample: Current noisy sample
            predict_epsilon: Whether model predicts noise (epsilon) or x0
            generator: Optional random generator
            
        Returns:
            Dictionary containing the prev_sample and other prediction outputs
        """
        # Ensure timestep is valid
        timestep = min(timestep, self.num_timesteps - 1)
        
        # Get alpha values for current timestep
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod_prev[timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = self.alphas[timestep]
        current_beta_t = self.betas[timestep]
        
        # Compute predicted original sample
        if predict_epsilon:
            # Model predicts noise
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        else:
            # Model directly predicts x0
            pred_original_sample = model_output
        
        # For discrete tokens, we need to handle this as a categorical distribution
        # For each position, we have logits over vocabulary
        vocab_size = model_output.shape[-1]
        
        # Get the predicted token distribution
        pred_token_logits = pred_original_sample.reshape(-1, vocab_size)
        pred_token_probs = torch.softmax(pred_token_logits, dim=-1)
        
        # Sample from the distribution
        if generator is not None:
            pred_tokens = torch.multinomial(pred_token_probs, num_samples=1, generator=generator)
        else:
            pred_tokens = torch.multinomial(pred_token_probs, num_samples=1)
        
        # Reshape back to original shape
        pred_tokens = pred_tokens.reshape(sample.shape)
        
        # Compute the posterior mean and variance
        # For categorical data, we use a different approach than continuous diffusion
        # We'll use the predicted tokens as the mean of a categorical distribution
        variance = 0
        
        # Compute the previous noisy sample based on pred_tokens
        prev_sample = self._get_prev_sample(
            pred_tokens, timestep, sample, generator=generator
        )
        
        return {
            "prev_sample": prev_sample,
            "pred_original_sample": pred_tokens,
            "variance": variance
        }
    
    def _get_prev_sample(
        self,
        pred_original_sample: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Get the previous sample in the diffusion process.
        
        Args:
            pred_original_sample: Predicted original sample (x_0)
            timestep: Current timestep
            sample: Current noisy sample (x_t)
            generator: Optional random generator
            
        Returns:
            Previous sample (x_{t-1})
        """
        # Get the timestep values
        timestep = min(timestep, self.num_timesteps - 1)
        prev_timestep = timestep - 1
        
        # Special case for the first step
        if prev_timestep < 0:
            return pred_original_sample
        
        # Get alpha values for current and previous timesteps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod_prev[timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = self.alphas[timestep]
        current_beta_t = self.betas[timestep]
        
        # Compute the coefficients for the posterior mean
        # q(x_{t-1} | x_t, x_0) distribution
        posterior_mean_coef1 = self.posterior_mean_coef1[timestep]
        posterior_mean_coef2 = self.posterior_mean_coef2[timestep]
        
        # Calculate posterior mean
        posterior_mean = posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * sample
        
        # For discrete tokens, this becomes a categorical distribution
        # We'll sample from a distribution weighted by the posterior mean
        # This is a heuristic approach for discrete diffusion
        posterior_variance = self.posterior_variance[timestep]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[timestep]
        
        # Get noise to add for stochastic sampling
        noise = torch.randn_like(sample)
        if generator is not None:
            noise = noise.to(sample.device).normal_(generator=generator)
        
        # Compute the previous sample
        # For discrete tokens, we use a mixture approach
        # With probability (1-noise_level), use the predicted token
        # With probability noise_level, use the current token
        noise_level = current_beta_t
        
        # Create a mask for mixing
        if generator is not None:
            mask = torch.rand_like(sample.float(), generator=generator) < noise_level
        else:
            mask = torch.rand_like(sample.float()) < noise_level
        
        # Mix predicted tokens and current tokens
        prev_sample = torch.where(mask, sample, pred_original_sample)
        
        return prev_sample