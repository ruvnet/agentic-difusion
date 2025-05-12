"""
Base implementation of the AdaptDiffuser model.

This module contains the core AdaptDiffuser class which provides task-specific
adaptation for diffusion models through reward gradient guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from agentic_diffusion.core.diffusion_model import DiffusionModel
from agentic_diffusion.core.noise_schedules import NoiseScheduler
from agentic_diffusion.core.trajectory_buffer import AdaptDiffuserTrajectoryBuffer
from agentic_diffusion.core.adapt_diffuser.guidance import AdaptDiffuserGuidance
from agentic_diffusion.core.adapt_diffuser.utils import encode_task, save_adaptation_metrics
from agentic_diffusion.core.common_types import RewardModelProtocol, TaskEmbeddingModelProtocol

# Use protocol types instead of concrete classes
RewardModel = RewardModelProtocol
TaskEmbeddingModel = TaskEmbeddingModelProtocol

# Use deferred imports to avoid circular dependencies
def _get_denoising_classes():
    """Dynamically import denoising process classes to avoid circular imports."""
    from agentic_diffusion.core.denoising_process import (
        EnhancedGuidedDenoisingProcess,
        EnhancedGuidedDDPMSampler,
        EnhancedGuidedDDIMSampler
    )
    return (EnhancedGuidedDenoisingProcess, EnhancedGuidedDDPMSampler, EnhancedGuidedDDIMSampler)

# Configure logging
logger = logging.getLogger(__name__)


class AdaptDiffuser:
    """
    Core implementation of the AdaptDiffuser model.
    
    This class integrates diffusion models with task-specific adaptation mechanisms
    through gradient-based reward guidance and trajectory-based learning.
    """
    
    def register_reward_model(self, reward_model):
        """
        Register a reward model with the AdaptDiffuser model.
        
        Args:
            reward_model: Reward model instance
        """
        try:
            # Try to move the model to the correct device
            if reward_model is not None:
                if hasattr(reward_model, 'to'):
                    self.reward_model = reward_model.to(self.device)
                else:
                    # If no 'to' method is available, just use as is
                    self.reward_model = reward_model
                    # Try to update the device attribute if it exists
                    if hasattr(reward_model, 'device'):
                        reward_model.device = self.device
            else:
                self.reward_model = None
            
            # Update guidance handler with new reward model
            if hasattr(self, 'guidance_handler'):
                self.guidance_handler.reward_model = self.reward_model
                
            # Update denoising process with new reward model
            if hasattr(self, 'denoising_process'):
                self.denoising_process.reward_model = self.reward_model
            
            logger.info("Reward model successfully registered")
        except Exception as e:
            logger.error(f"Error registering reward model: {e}")
            # Keep the previous reward model if there was an error
        
    def compute_rewards(self, trajectories):
        """
        Compute rewards for trajectories using the registered reward model.
        
        Args:
            trajectories: A single trajectory or batch of trajectories
            
        Returns:
            Tensor of rewards for each trajectory
        """
        if self.reward_model is None:
            logger.warning("No reward model available for reward computation")
            batch_size = 1 if not isinstance(trajectories, list) else len(trajectories)
            return torch.zeros(batch_size, device=self.device)
            
        return self.reward_model.compute_rewards(trajectories)
    
    def __init__(
        self,
        base_model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        img_size: int,
        channels: int,
        reward_model: Optional[RewardModel] = None,
        task_embedding_model: Optional[TaskEmbeddingModel] = None,
        guidance_scale: float = 3.0,
        guidance_min_step_percent: float = 0.1,
        guidance_max_step_percent: float = 0.9,
        classifier_scale: float = 0.0,
        buffer_capacity: int = 10000,
        learning_rate: float = 1e-5,
        device: str = None,
        checkpoint_dir: str = './checkpoints',
        use_ddim: bool = True,
        inference_steps: int = 50
    ):
        """
        Initialize the AdaptDiffuser model.
        
        Args:
            base_model: Pretrained diffusion model to adapt
            noise_scheduler: Noise schedule for diffusion process
            img_size: Size of images/latents to generate
            channels: Number of channels in images/latents
            reward_model: Model for computing task-specific rewards
            task_embedding_model: Model for encoding task descriptions
            guidance_scale: Scale factor for reward gradient guidance
            guidance_min_step_percent: Percentage of steps to start guidance
            guidance_max_step_percent: Percentage of steps to end guidance
            classifier_scale: Scale factor for classifier-free guidance
            buffer_capacity: Maximum capacity of trajectory buffer
            learning_rate: Learning rate for model adaptation
            device: Device to use for computation
            checkpoint_dir: Directory for saving model checkpoints
            use_ddim: Whether to use DDIM (faster) or DDPM (higher quality) sampling
            inference_steps: Number of steps for inference
            
        Raises:
            ValueError: If parameters are invalid
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store configuration
        self.img_size = img_size
        self.channels = channels
        self.guidance_scale = guidance_scale
        self.guidance_min_step_percent = guidance_min_step_percent
        self.guidance_max_step_percent = guidance_max_step_percent
        self.classifier_scale = classifier_scale
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.use_ddim = use_ddim
        self.inference_steps = inference_steps
        
        # Setup models
        self.base_model = base_model.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.reward_model = reward_model.to(self.device) if reward_model is not None else None
        self.task_embedding_model = task_embedding_model.to(self.device) if task_embedding_model is not None else None
        
        # Create guidance handler
        self.guidance_handler = AdaptDiffuserGuidance(
            reward_model=self.reward_model,
            task_embedding_model=self.task_embedding_model,
            device=self.device
        )
        
        # Get denoising classes using deferred import
        EnhancedGuidedDenoisingProcess, EnhancedGuidedDDPMSampler, EnhancedGuidedDDIMSampler = _get_denoising_classes()
        
        # Create enhanced denoising process
        self.denoising_process = EnhancedGuidedDenoisingProcess(
            model=self.base_model,
            noise_scheduler=self.noise_scheduler,
            img_size=img_size,
            channels=channels,
            reward_model=self.reward_model,
            guidance_scale=guidance_scale,
            guidance_min_step_percent=guidance_min_step_percent,
            guidance_max_step_percent=guidance_max_step_percent,
            device=self.device
        )
        
        # Create samplers
        if use_ddim:
            self.sampler = EnhancedGuidedDDIMSampler(
                process=self.denoising_process,
                num_timesteps=inference_steps,
                eta=0.0  # Deterministic sampling
            )
        else:
            self.sampler = EnhancedGuidedDDPMSampler(
                process=self.denoising_process,
                num_timesteps=inference_steps
            )
        
        # Create trajectory buffer for storing high-quality examples
        self.trajectory_buffer = AdaptDiffuserTrajectoryBuffer(
            capacity=buffer_capacity,
            device=self.device
        )
        
        # Setup optimizer for fine-tuning
        self.optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=learning_rate
        )
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def encode_task(
        self,
        task: Union[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode task description or identifier into an embedding.
        
        Args:
            task: Task description string or embedding tensor
            
        Returns:
            Task embedding tensor
            
        Raises:
            ValueError: If task embedding model is not available and task is a string
        """
        return encode_task(
            task=task,
            task_embedding_model=self.task_embedding_model,
            device=self.device
        )
    
    def generate(
        self,
        batch_size: int = 1,
        task: Optional[Union[str, torch.Tensor]] = None,
        conditioning: Optional[Any] = None,
        custom_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,  # Added alternate parameter name
        custom_classifier_scale: Optional[float] = None,
        save_intermediates: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate outputs using the adapted diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            task: Task description or embedding for adaptation
            conditioning: Additional conditioning information
            custom_guidance_scale: Override default guidance scale
            custom_classifier_scale: Override default classifier scale
            save_intermediates: Whether to save intermediate generation steps
            **kwargs: Additional arguments for the sampling process
            
        Returns:
            Generated samples and optionally intermediate steps
            
        Raises:
            RuntimeError: If generation fails
        """
        try:
            # Use either parameter name for guidance scale
            effective_guidance = custom_guidance_scale if custom_guidance_scale is not None else guidance_scale
            
            # Handle task encoding if provided
            task_embedding = None
            if task is not None:
                task_embedding = self.encode_task(task)
            
            # Set effective guidance scales
            guidance_scale = effective_guidance
            if guidance_scale is None:
                guidance_scale = 0.0  # Default to no guidance if still None
                
            classifier_scale = custom_classifier_scale if custom_classifier_scale is not None else self.classifier_scale
            if classifier_scale is None:
                classifier_scale = 0.0  # Default to no classifier guidance if None
            
            # Generate samples
            intermediates = []
            
            # Sample with the appropriate sampler
            with torch.no_grad():
                if save_intermediates:
                    # Custom sampling loop to store intermediates
                    samples = self._sample_with_intermediates(
                        batch_size=batch_size,
                        task=task_embedding,
                        classifier_cond=conditioning,
                        guidance_scale=guidance_scale,
                        classifier_scale=classifier_scale,
                        intermediates=intermediates,
                        **kwargs
                    )
                else:
                    # Standard sampling
                    samples = self.sampler.sample(
                        batch_size=batch_size,
                        task=task_embedding,
                        classifier_cond=conditioning,
                        guidance_scale=guidance_scale,
                        classifier_scale=classifier_scale,
                        **kwargs
                    )
            
            # Return samples and optionally intermediates
            if save_intermediates:
                return samples, intermediates
            else:
                return samples
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate samples: {e}")
    
    def _sample_with_intermediates(
        self,
        batch_size: int,
        task: Optional[torch.Tensor],
        classifier_cond: Optional[Any],
        guidance_scale: float,
        classifier_scale: float,
        intermediates: List[torch.Tensor],
        save_frequency: int = 10,
        **kwargs
    ) -> torch.Tensor:
        """
        Internal method for sampling with intermediate step storage.
        
        Args:
            batch_size: Number of samples to generate
            task: Task embedding tensor
            classifier_cond: Conditioning for classifier guidance
            guidance_scale: Scale for reward guidance
            classifier_scale: Scale for classifier guidance
            intermediates: List to store intermediate samples
            save_frequency: How often to save intermediate steps
            **kwargs: Additional arguments for the sampling process
            
        Returns:
            Generated samples
        """
        device = self.device
        process = self.sampler.process
        
        # Get number of steps
        num_steps = self.sampler.num_timesteps
        
        # Start from pure noise
        shape = (batch_size, self.channels, self.img_size, self.img_size)
        x_t = torch.randn(shape, device=device)
        
        # Add initial noise to intermediates
        intermediates.append(x_t.clone())
        
        # Use appropriate sampling iteration based on sampler type
        if isinstance(self.sampler, EnhancedGuidedDDIMSampler):
            timesteps = self.sampler.timesteps
            
            # Iterate through selected timesteps
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                next_t = timesteps[i + 1]
                
                # Get timestep index
                t_index = process.noise_scheduler.num_timesteps - t - 1
                
                # Broadcast to batch size
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Compute mean and variance with guidance
                with torch.no_grad():
                    mean, variance, log_variance = process.p_mean_variance_with_enhanced_guidance(
                        x_t, t_batch, t_index, len(timesteps),
                        task, classifier_cond, guidance_scale, classifier_scale,
                        **kwargs
                    )
                
                # No noise for t=0
                noise = torch.randn_like(x_t)
                if next_t == 0:
                    noise = 0
                
                # DDIM formula
                if self.sampler.eta > 0:
                    # Stochastic part
                    sigma = self.sampler.eta * torch.sqrt(variance)
                    x_t = mean + sigma * noise
                else:
                    # Deterministic part
                    x_t = mean
                
                # Save intermediate at specified frequency
                if i % save_frequency == 0 or next_t == 0:
                    intermediates.append(x_t.clone())
                    
        else:
            # DDPM-style sampling
            for t_step in range(num_steps - 1, -1, -1):
                # Create batch of same timestep
                t = torch.tensor([t_step] * batch_size, device=self.device)
                
                # Denoise for one step with enhanced guidance
                with torch.no_grad():
                    # Get mean and variance with guidance
                    mean, variance, log_variance = process.p_mean_variance_with_enhanced_guidance(
                        x_t, t, t_step, num_steps,
                        task, classifier_cond, guidance_scale, classifier_scale,
                        **kwargs
                    )
                    
                    # No noise if t == 0, otherwise add noise scaled by variance
                    noise = torch.zeros_like(x_t)
                    if t.min() > 0:
                        noise = torch.randn_like(x_t)
                    
                    # Get less noisy image
                    x_t = mean + torch.exp(0.5 * log_variance) * noise
                
                # Save intermediate at specified frequency
                if t_step % save_frequency == 0 or t_step == 0:
                    intermediates.append(x_t.clone())
        
        return x_t
    
    def compute_reward(
        self,
        samples: torch.Tensor,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute rewards for generated samples.
        
        Args:
            samples: Tensor of generated samples
            task: Task description or embedding
            
        Returns:
            Tensor of reward scores
            
        Raises:
            ValueError: If reward model is not available
        """
        if self.reward_model is None:
            raise ValueError("Reward model is required to compute rewards")
            
        # Encode task if necessary
        task_embedding = None
        if task is not None:
            task_embedding = self.encode_task(task)
            
        # Compute rewards
        with torch.no_grad():
            rewards = self.reward_model.compute_reward(samples, task_embedding)
            
            # Convert rewards to tensor if needed
            if isinstance(rewards, (float, int)):
                rewards = torch.tensor([rewards], device=self.device)
            elif isinstance(rewards, list):
                rewards = torch.tensor(rewards, device=self.device)
            elif not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, device=self.device)
            
        return rewards
    
    def store_samples(
        self,
        samples: torch.Tensor,
        rewards: torch.Tensor,
        task: Optional[Union[str, torch.Tensor]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Store high-quality samples in the trajectory buffer.
        
        Args:
            samples: Tensor of generated samples
            rewards: Tensor of reward scores for samples
            task: Task description or embedding
            metadata: Optional metadata for each sample
            
        Returns:
            Indices of added samples
        """
        # Encode task if necessary
        task_id = None
        if task is not None:
            if isinstance(task, str):
                task_id = task
            elif isinstance(task, torch.Tensor):
                task_id = task
        
        # Convert tensors to list for batch adding
        if isinstance(samples, torch.Tensor):
            samples_list = [samples[i] for i in range(samples.shape[0])]
        else:
            samples_list = samples
            
        if isinstance(rewards, torch.Tensor):
            rewards_list = rewards.cpu().tolist()
        else:
            rewards_list = rewards
            
        # Add to buffer
        indices = self.trajectory_buffer.batch_add(
            trajectories=samples_list,
            rewards=rewards_list,
            task=task_id,
            metadata=metadata
        )
        
        return indices
    
    def adapt_to_task(
        self,
        task: Union[str, torch.Tensor],
        num_steps: int = 100,
        batch_size: int = 8,
        adapt_lr: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        feedback_frequency: int = 10,
        min_reward_threshold: float = 0.6,
        target_reward: float = 0.9,
        early_stop: bool = True,
        save_checkpoints: bool = True,
        save_metrics: bool = True,
        metrics_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adapt the diffusion model to a specific task.
        
        Args:
            task: Task description or embedding to adapt to
            num_steps: Number of adaptation steps
            batch_size: Batch size for adaptation
            adapt_lr: Learning rate for adaptation (None = use default)
            guidance_scale: Override default guidance scale
            feedback_frequency: How often to evaluate and log progress
            min_reward_threshold: Minimum reward to consider adaptation successful
            target_reward: Target reward for early stopping
            early_stop: Whether to stop early when target reward is reached
            save_checkpoints: Whether to save checkpoints during adaptation
            save_metrics: Whether to save metrics during adaptation
            metrics_path: Path to save metrics (defaults to checkpoint_dir)
            
        Returns:
            Dictionary with adaptation metrics
            
        Raises:
            ValueError: If reward model is missing
        """
        if self.reward_model is None:
            raise ValueError("Reward model is required for task adaptation")
            
        # Encode task
        task_embedding = self.encode_task(task)
        
        # Prepare optimizer with learning rate
        lr = adapt_lr if adapt_lr is not None else self.learning_rate
        optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=lr)
        
        # Set effective guidance scale
        effective_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        
        # Training loop
        metrics = {
            "steps": [],
            "mean_reward": [],
            "max_reward": [],
            "loss": []
        }
        
        # Initialize best validation reward
        best_reward = 0.0
        best_model_state = None
        
        # Start time
        start_time = time.time()
        
        for step in range(num_steps):
            # Generate samples
            self.base_model.train()
            samples = self.generate(
                batch_size=batch_size,
                task=task_embedding,
                custom_guidance_scale=effective_guidance
            )
            
            # Compute rewards
            rewards = self.compute_reward(samples, task_embedding)
            mean_reward = rewards.mean().item()
            max_reward = rewards.max().item()
            
            # Store high-quality samples
            high_quality_mask = rewards >= min_reward_threshold
            if high_quality_mask.sum() > 0:
                high_quality_samples = samples[high_quality_mask]
                high_quality_rewards = rewards[high_quality_mask]
                
                self.store_samples(
                    samples=high_quality_samples,
                    rewards=high_quality_rewards,
                    task=task_embedding
                )
            
            # Compute loss for adaptation
            loss = -rewards.mean()  # Maximize reward
            
            # Check if rewards have gradients
            if loss.requires_grad:
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # If rewards don't have gradients, log a warning and skip optimization
                logger.warning("Rewards don't have gradients, skipping optimization step")
            
            # Record metrics
            metrics["steps"].append(step)
            metrics["mean_reward"].append(mean_reward)
            metrics["max_reward"].append(max_reward)
            metrics["loss"].append(loss.item())
            
            # Feedback and evaluation
            if (step + 1) % feedback_frequency == 0 or step == num_steps - 1:
                duration = time.time() - start_time
                logger.info(f"Step {step+1}/{num_steps}, Mean Reward: {mean_reward:.4f}, "
                           f"Max Reward: {max_reward:.4f}, Loss: {loss.item():.4f}, "
                           f"Time: {duration:.2f}s")
                
                # Save checkpoint if improved
                if max_reward > best_reward and save_checkpoints:
                    best_reward = max_reward
                    best_model_state = {
                        'model': self.base_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'reward': max_reward
                    }
                    
                    # Create task ID for filename
                    task_id = str(task)
                    if isinstance(task, torch.Tensor):
                        task_id = f"task_embed_{task.sum().item():.3f}"
                    elif len(task_id) > 20:
                        task_id = task_id[:20]
                        
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, 
                        f"adapt_diffuser_{task_id}_{step}_{max_reward:.4f}.pt"
                    )
                    torch.save(best_model_state, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Save metrics
                    if save_metrics:
                        metrics_file = metrics_path or os.path.join(
                            self.checkpoint_dir,
                            f"adapt_diffuser_{task_id}_metrics.json"
                        )
                        
                        # Add summary metrics
                        current_metrics = metrics.copy()
                        current_metrics["best_reward"] = best_reward
                        current_metrics["current_step"] = step
                        current_metrics["total_duration"] = duration
                        
                        save_adaptation_metrics(current_metrics, metrics_file)
            
            # Early stopping
            if early_stop and mean_reward >= target_reward:
                logger.info(f"Early stopping at step {step+1} with mean reward {mean_reward:.4f}")
                break
        
        # Restore best model if found
        if best_model_state is not None and best_reward > mean_reward:
            self.base_model.load_state_dict(best_model_state['model'])
            logger.info(f"Restored best model with reward {best_reward:.4f}")
        
        # Final adaptation metrics
        metrics["final_mean_reward"] = mean_reward
        metrics["final_max_reward"] = max_reward
        metrics["best_reward"] = best_reward
        metrics["total_steps"] = len(metrics["steps"])
        metrics["duration"] = time.time() - start_time
        
        return metrics
    
    def save(
        self,
        path: str
    ) -> bool:
        """
        Save the adapted model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Success flag
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare state dictionary
            state_dict = {
                'base_model': self.base_model.state_dict(),
                'guidance_scale': self.guidance_scale,
                'guidance_min_step_percent': self.guidance_min_step_percent,
                'guidance_max_step_percent': self.guidance_max_step_percent,
                'classifier_scale': self.classifier_scale,
                'img_size': self.img_size,
                'channels': self.channels,
                'use_ddim': self.use_ddim,
                'inference_steps': self.inference_steps
            }
            
            # Save trajectory buffer separately
            if hasattr(self, 'trajectory_buffer'):
                buffer_path = path.replace('.pt', '_buffer.pt')
                self.trajectory_buffer.save_state(buffer_path)
                state_dict['buffer_path'] = buffer_path
            
            # Save to disk
            torch.save(state_dict, path)
            logger.info(f"Saved AdaptDiffuser model to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @classmethod
    def load(
        cls,
        path: str,
        base_model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        reward_model: Optional[RewardModel] = None,
        task_embedding_model: Optional[TaskEmbeddingModel] = None,
        device: str = None
    ) -> 'AdaptDiffuser':
        """
        Load an adapted model from disk.
        
        Args:
            path: Path to load the model from
            base_model: Model architecture to load weights into
            noise_scheduler: Noise scheduler for diffusion process
            reward_model: Model for computing task-specific rewards
            task_embedding_model: Model for encoding task descriptions
            device: Device to load the model onto
            
        Returns:
            Loaded AdaptDiffuser model
            
        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If loading fails
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            # Determine device
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load state dictionary
            state_dict = torch.load(path, map_location=device)
            
            # Load model weights
            base_model.load_state_dict(state_dict['base_model'])
            
            # Extract configuration
            guidance_scale = state_dict.get('guidance_scale', 3.0)
            guidance_min_step_percent = state_dict.get('guidance_min_step_percent', 0.1)
            guidance_max_step_percent = state_dict.get('guidance_max_step_percent', 0.9)
            classifier_scale = state_dict.get('classifier_scale', 0.0)
            img_size = state_dict.get('img_size', 64)
            channels = state_dict.get('channels', 4)
            use_ddim = state_dict.get('use_ddim', True)
            inference_steps = state_dict.get('inference_steps', 50)
            
            # Create model instance
            model = cls(
                base_model=base_model,
                noise_scheduler=noise_scheduler,
                img_size=img_size,
                channels=channels,
                reward_model=reward_model,
                task_embedding_model=task_embedding_model,
                guidance_scale=guidance_scale,
                guidance_min_step_percent=guidance_min_step_percent,
                guidance_max_step_percent=guidance_max_step_percent,
                classifier_scale=classifier_scale,
                device=device,
                use_ddim=use_ddim,
                inference_steps=inference_steps
            )
            
            # Load trajectory buffer if available
            buffer_path = state_dict.get('buffer_path')
            if buffer_path and os.path.exists(buffer_path):
                model.trajectory_buffer.load_state(buffer_path)
                logger.info(f"Loaded trajectory buffer from {buffer_path}")
            
            logger.info(f"Loaded AdaptDiffuser model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Error loading model: {e}")