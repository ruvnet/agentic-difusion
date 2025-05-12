"""
Extension of noise schedulers for the Agentic Diffusion system.

This module implements the missing methods needed for the noise schedulers to
meet the 90% test coverage requirement.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Union, Optional, Dict, Any, List

from agentic_diffusion.core.noise_schedules import NoiseScheduler


def extend_noise_scheduler():
    """
    Extends the NoiseScheduler class with additional methods for complete coverage.
    
    This function adds all the required methods to the NoiseScheduler class to ensure
    it passes the test coverage requirements.
    """
    
    # Add extract_into_tensor as an alias of extract for test compatibility
    def extract_into_tensor(self, a: torch.Tensor, t: Union[torch.Tensor, int], broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Alias for the extract method to maintain compatibility with tests.
        """
        # Handle case where t is an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        return self.extract(a, t, broadcast_shape)
    
    # Add signal_to_noise_ratio method
    def signal_to_noise_ratio(self, t: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Calculate the signal-to-noise ratio for given timesteps.
        
        Args:
            t: Timesteps tensor of shape [batch_size] or integer
            
        Returns:
            Signal-to-noise ratio for each timestep
        """
        # Handle case where t is an integer
        if isinstance(t, int):
            # Extract alpha_cumprod at the specified timestep
            alpha_t = self.alphas_cumprod[t]
        else:
            # Extract alphas_cumprod at the specified timesteps
            alpha_t = torch.stack([self.alphas_cumprod[i] for i in t])
        
        # Calculate SNR = alpha / (1 - alpha)
        snr = alpha_t / (1.0 - alpha_t)
        
        return snr
    
    # Add predict_v method
    def predict_v(self, x_t: torch.Tensor, t: Union[torch.Tensor, int], noise: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity (v) from noise and x_t.
        
        The velocity is an alternative parameterization used in some diffusion papers.
        
        Args:
            x_t: Noisy data at timestep t
            t: Timesteps tensor or integer
            noise: Predicted noise
            
        Returns:
            Predicted velocity
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Extract alphas_cumprod at the specified timesteps
        alpha_cumprod = self.extract(self.alphas_cumprod, t, x_t.shape)
        
        # Calculate velocity using the formula from the paper
        # v = sqrt(1 - alpha) * noise - sqrt(alpha) * (x_t - sqrt(alpha) * x_0) / sqrt(1 - alpha)
        # Simplified: v = sqrt(1 - alpha) * noise - sqrt(alpha) * (x_t - predict_x0) / sqrt(1 - alpha)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cumprod)
        sqrt_alpha = torch.sqrt(alpha_cumprod)
        
        # First calculate x_0 prediction
        x_0_pred = self.predict_start_from_noise(x_t, t, noise)
        
        # Then calculate velocity
        velocity = sqrt_one_minus_alpha * noise - sqrt_alpha * (x_t - sqrt_alpha * x_0_pred) / sqrt_one_minus_alpha
        
        return velocity
    
    # Add v_prediction_from_noise_prediction method
    def v_prediction_from_noise_prediction(self, noise: torch.Tensor, t: Union[torch.Tensor, int], x_t: torch.Tensor) -> torch.Tensor:
        """
        Convert noise prediction to velocity prediction.
        
        Args:
            noise: Predicted noise
            t: Timesteps tensor or integer
            x_t: Noisy data at timestep t
            
        Returns:
            Predicted velocity
        """
        return self.predict_v(x_t, t, noise)
    
    # Add noise_prediction_from_v_prediction method
    def noise_prediction_from_v_prediction(self, v: torch.Tensor, t: Union[torch.Tensor, int], x_t: torch.Tensor) -> torch.Tensor:
        """
        Convert velocity prediction to noise prediction.
        
        Args:
            v: Predicted velocity
            t: Timesteps tensor or integer
            x_t: Noisy data at timestep t
            
        Returns:
            Predicted noise
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Extract alphas_cumprod at the specified timesteps
        alpha_cumprod = self.extract(self.alphas_cumprod, t, x_t.shape)
        
        # Calculate noise from velocity using the inverse of the velocity formula
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cumprod)
        sqrt_alpha = torch.sqrt(alpha_cumprod)
        
        # Start with x_0 prediction
        x_0_pred = (x_t - sqrt_one_minus_alpha * v) / sqrt_alpha
        
        # Calculate noise prediction
        noise = (x_t - sqrt_alpha * x_0_pred) / sqrt_one_minus_alpha
        
        return noise
    
    # Add predict_start_from_noise_and_v method
    def predict_start_from_noise_and_v(self, x_t: torch.Tensor, t: Union[torch.Tensor, int], v: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from velocity and x_t.
        
        Args:
            x_t: Noisy data at timestep t
            t: Timesteps tensor or integer
            v: Velocity prediction
            
        Returns:
            Predicted x_0
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Extract alphas_cumprod at the specified timesteps
        alpha_cumprod = self.extract(self.alphas_cumprod, t, x_t.shape)
        
        # Calculate x_0 using the velocity formula
        sqrt_alpha = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cumprod)
        
        # x_0 = (x_t - sqrt(1 - alpha) * v) / sqrt(alpha)
        x_0_pred = (x_t - sqrt_one_minus_alpha * v) / sqrt_alpha
        
        return x_0_pred
    
    
    # Add model_predictions method
    def model_predictions(self, model_output: torch.Tensor, t: Union[torch.Tensor, int], x_t: torch.Tensor,
                          return_all: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                                             Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Make model predictions and convert to required representations.
        
        Args:
            model_output: Raw output from the model (typically noise prediction)
            t: Timesteps tensor or integer
            x_t: Noisy data at timestep t
            return_all: Whether to return all predictions (x_0, noise, and velocity)
            
        Returns:
            pred_x_0: Predicted clean data
            pred_noise: Predicted noise
            pred_v: Predicted velocity (only if return_all=True)
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Assume model predicts noise by default
        pred_noise = model_output
        
        # Predict x_0 from noise
        pred_x_0 = self.predict_start_from_noise(x_t, t, pred_noise)
        
        # Compute velocity prediction if needed
        if return_all:
            pred_v = self.predict_v(x_t, t, pred_noise)
            return pred_x_0, pred_noise, pred_v
        
        return pred_x_0, pred_noise
    
    # Add ddpm_step method that handles integer timesteps
    def ddpm_step(self, model_output: torch.Tensor, t: Union[torch.Tensor, int], x_t: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute a single DDPM sampling step.
        
        Args:
            model_output: Raw output from the model (typically noise prediction)
            t: Timesteps tensor or integer
            x_t: Noisy data at timestep t
            
        Returns:
            mean: Posterior mean
            variance: Posterior variance
            log_variance: Log of posterior variance
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Get the predicted x_0
        pred_x_0 = self.predict_start_from_noise(x_t, t, model_output)
        
        # Calculate the posterior distribution parameters
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_0, x_t, t
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance

    # Add step method that handles integer timesteps
    def step(self, model_output: torch.Tensor, t: Union[torch.Tensor, int], x_t: torch.Tensor, 
             noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Take a step in the reverse diffusion process.
        
        Args:
            model_output: Raw output from the model (typically noise prediction)
            t: Timesteps tensor or integer
            x_t: Noisy data at timestep t
            noise: Optional random noise for the step
            
        Returns:
            Predicted x_{t-1}
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Get posterior parameters
        mean, variance, log_variance = self.ddpm_step(model_output, t, x_t)
        
        # No noise for t=0
        if noise is None:
            noise = torch.zeros_like(x_t)
            if torch.min(t) > 0:
                noise = torch.randn_like(x_t)
        
        # Apply DDPM step formula
        x_t_minus_1 = mean + torch.exp(0.5 * log_variance) * noise
        
        return x_t_minus_1
    
    # Add ddim_step method that handles integer timesteps
    def ddim_step(self, model_output: torch.Tensor, t: Union[torch.Tensor, int], next_t: Union[torch.Tensor, int], 
                 x_t: torch.Tensor, eta: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a single DDIM sampling step.
        
        Args:
            model_output: Raw output from the model (typically noise prediction)
            t: Current timesteps tensor or integer
            next_t: Next timesteps tensor or integer
            x_t: Noisy data at timestep t
            eta: Controls the amount of stochasticity (0 = deterministic, 1 = full stochasticity)
            
        Returns:
            x_{t-1}: Predicted data at timestep t-1
            pred_x_0: Predicted clean data
        """
        # Convert t and next_t to tensors if they're integers
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
        if isinstance(next_t, int):
            next_t = torch.tensor([next_t], device=self.device)
            
        # Get predicted x_0
        pred_x_0 = self.predict_start_from_noise(x_t, t, model_output)
        
        # Extract alphas for current and next timesteps
        alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_next = self.extract(self.alphas_cumprod, next_t, x_t.shape)
        
        # Compute sigma for DDIM
        sigma = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod_t) * 
                                (1 - alpha_cumprod_t / alpha_cumprod_next))
        
        # Get noise
        noise = torch.randn_like(x_t)
        
        # Compute DDIM step
        c1 = torch.sqrt(alpha_cumprod_next / alpha_cumprod_t)
        c2 = torch.sqrt(1 - alpha_cumprod_next - sigma**2)
        
        # Compute prediction for x_{t-1}
        x_t_minus_1 = c1 * x_t + c2 * (pred_x_0 - c1 * x_t) / torch.sqrt(1 - alpha_cumprod_t)
        
        # Add noise if eta > 0
        if eta > 0:
            x_t_minus_1 = x_t_minus_1 + sigma * noise
        
        return x_t_minus_1, pred_x_0
    
    # Add posterior_sample method that handles integer timesteps
    def posterior_sample(self, x_0: torch.Tensor, x_t: torch.Tensor, t: Union[torch.Tensor, int], 
                         noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the posterior distribution q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Clean data
            x_t: Noisy data at timestep t
            t: Timesteps tensor or integer
            noise: Optional random noise for the step
            
        Returns:
            Sample from the posterior distribution
        """
        # Convert t to tensor if it's an integer
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device)
            
        # Get posterior parameters
        mean, variance, log_variance = self.q_posterior_mean_variance(x_0, x_t, t)
        
        # Sample from the posterior
        if noise is None:
            noise = torch.randn_like(x_t)
        
        # No noise for t=0
        if torch.min(t) == 0:
            return mean
        
        x_t_minus_1 = mean + torch.exp(0.5 * log_variance) * noise
        
        return x_t_minus_1
    
    # Add the methods to the NoiseScheduler class's prototype
    NoiseScheduler.extract_into_tensor = extract_into_tensor
    NoiseScheduler.signal_to_noise_ratio = signal_to_noise_ratio
    NoiseScheduler.predict_v = predict_v
    NoiseScheduler.v_prediction_from_noise_prediction = v_prediction_from_noise_prediction
    NoiseScheduler.noise_prediction_from_v_prediction = noise_prediction_from_v_prediction
    NoiseScheduler.predict_start_from_noise_and_v = predict_start_from_noise_and_v
    NoiseScheduler.model_predictions = model_predictions
    NoiseScheduler.ddpm_step = ddpm_step
    NoiseScheduler.step = step
    NoiseScheduler.ddim_step = ddim_step
    NoiseScheduler.posterior_sample = posterior_sample
    
    return NoiseScheduler