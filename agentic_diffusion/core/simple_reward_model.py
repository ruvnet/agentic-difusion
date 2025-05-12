"""
Simple reward model implementation for AdaptDiffuser.

This module provides a basic implementation of the RewardModel interface
for testing the 'adapt' and 'improve' commands.
"""

import torch
import numpy as np
from typing import Optional

from agentic_diffusion.core.reward_functions import RewardModel


class SimpleRewardModel(RewardModel):
    """
    A simple reward model implementation for testing AdaptDiffuser.
    
    This model computes rewards based on basic statistical properties
    of the trajectories to provide meaningful scores for testing purposes.
    """
    
    def __init__(
        self,
        base_reward: float = 0.5,
        noise_scale: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize the simple reward model.
        
        Args:
            base_reward: Base reward value (default: 0.5)
            noise_scale: Scale of random noise to add (default: 0.1)
            device: Device to compute on
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_reward = base_reward
        self.noise_scale = noise_scale
        
    def compute_reward(self, samples: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute reward based on samples and optional task embedding.
        
        This implements a simple but sensible reward function that:
        1. Computes the L2 norm of samples
        2. Adds a task-dependent component if task is provided
        3. Adds a small amount of noise for variation
        4. Ensures rewards are in a reasonable range [0.1, 0.9]
        
        Args:
            samples: Batch of samples/trajectories
            task: Optional task embedding
            
        Returns:
            Tensor of rewards
        """
        # Move samples to the correct device
        samples = samples.to(self.device)
        
        # Calculate L2 norm (sample complexity) - normalized to [0, 1]
        if samples.dim() > 2:
            flat_samples = samples.view(samples.shape[0], -1)
        else:
            flat_samples = samples
            
        # Compute norms and normalize to [0, 1] using sigmoid
        norms = torch.norm(flat_samples, p=2, dim=-1)
        norm_factor = torch.sigmoid(1.0 - norms / (torch.mean(norms) + 1e-8))
        
        # Start with base reward
        rewards = torch.ones(samples.shape[0], device=self.device) * self.base_reward
        
        # Add norm-based component (30% weight)
        rewards = rewards * 0.7 + norm_factor * 0.3
        
        # Add task-dependent component if task is provided (10% weight)
        if task is not None and isinstance(task, torch.Tensor):
            task = task.to(self.device)
            # Use task vector features to create a task-specific component
            if task.dim() == 1:
                # Single task embedding
                task_factor = torch.sigmoid(task.sum() / task.shape[0])
                task_component = task_factor.repeat(samples.shape[0])
            else:
                # Batch of task embeddings
                task_factor = torch.sigmoid(task.sum(dim=1) / task.shape[1])
                if task.shape[0] == 1:
                    task_component = task_factor.repeat(samples.shape[0])
                else:
                    task_component = task_factor
                    
            rewards = rewards * 0.9 + task_component * 0.1
        
        # Add small random noise for variation
        noise = torch.randn(rewards.shape, device=self.device) * self.noise_scale
        rewards = rewards + noise
        
        # Clip to ensure rewards are in sensible range
        rewards = torch.clamp(rewards, 0.1, 0.9)
        
        return rewards


class AdaptDiffuserTestRewardModel(RewardModel):
    """
    A test-specific reward model for AdaptDiffuser.
    
    This model tracks iteration progress and produces increasingly higher
    rewards to demonstrate the 'improve' functionality.
    """
    
    def __init__(
        self,
        initial_reward: float = 0.5,
        improvement_rate: float = 0.05,
        device: Optional[str] = None
    ):
        """
        Initialize the test reward model.
        
        Args:
            initial_reward: Starting reward value (default: 0.5)
            improvement_rate: Rate of improvement per call (default: 0.05)
            device: Device to compute on
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") 
        self.initial_reward = initial_reward
        self.improvement_rate = improvement_rate
        self.calls = 0
        
    def compute_reward(self, samples: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute reward based on samples and optional task embedding.
        
        This implements a reward function that increases with each call
        to demonstrate improvement over iterations.
        
        Args:
            samples: Batch of samples/trajectories
            task: Optional task embedding
            
        Returns:
            Tensor of rewards
        """
        # Move samples to the correct device
        samples = samples.to(self.device)
        
        # Calculate base reward with gradual improvement
        self.calls += 1
        current_reward = min(0.9, self.initial_reward + self.calls * self.improvement_rate)
        
        # Add variance between samples
        batch_size = samples.shape[0]
        rewards = torch.ones(batch_size, device=self.device) * current_reward
        
        # Add variation between samples (±10%)
        variation = (torch.rand(batch_size, device=self.device) - 0.5) * 0.2
        rewards = rewards + variation
        
        # Ensure rewards are in a sensible range
        rewards = torch.clamp(rewards, 0.1, 0.95)
        
        return rewards