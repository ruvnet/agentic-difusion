"""
Test-specific models for AdaptDiffuser testing.

This module contains implementations of models used specifically for testing
the AdaptDiffuser framework, including simplified versions of the core components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from agentic_diffusion.core.noise_schedules import NoiseScheduler


class TaskEmbeddingManager:
    """
    Manager for task embeddings used in AdaptDiffuser.
    
    This class handles the creation, storage, and retrieval of task embeddings
    for use with AdaptDiffuser models.
    """
    
    def __init__(self, embedding_dim=64, device=None):
        """
        Initialize the TaskEmbeddingManager.
        
        Args:
            embedding_dim: Dimension of task embeddings
            device: Device to store embeddings on
        """
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task_embeddings = {}
        
    def get_embedding(self, task_name: str) -> torch.Tensor:
        """
        Get embedding for a task, creating it if it doesn't exist.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task embedding tensor
        """
        if task_name not in self.task_embeddings:
            # Create new random embedding
            embedding = torch.randn(1, self.embedding_dim, device=self.device)
            self.task_embeddings[task_name] = embedding
        return self.task_embeddings[task_name]
        
    def register_embedding(self, task_name: str, embedding: torch.Tensor):
        """
        Register a pre-computed embedding for a task.
        
        Args:
            task_name: Name of the task
            embedding: Pre-computed embedding tensor
        """
        self.task_embeddings[task_name] = embedding.to(self.device)
        
    def list_tasks(self) -> List[str]:
        """
        List all registered task names.
        
        Returns:
            List of task names
        """
        return list(self.task_embeddings.keys())


class TaskRewardModel(nn.Module):
    """
    Reward model for computing task-specific rewards.
    
    This model computes rewards for trajectories based on task embeddings
    and can be used for guiding the adaptation process.
    """
    
    def __init__(self, default_reward=None, device=None):
        """
        Initialize the TaskRewardModel.
        
        Args:
            default_reward: Default reward function if task-specific one not available
            device: Device to run computation on
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_reward = default_reward
        self.task_reward_functions = {}
        
    def register_reward_function(self, task_name: str, reward_fn: Callable):
        """
        Register a reward function for a specific task.
        
        Args:
            task_name: Name of the task
            reward_fn: Reward function that takes trajectory and returns scalar reward
        """
        self.task_reward_functions[task_name] = reward_fn
        
    def compute_reward(self, trajectory: torch.Tensor, task=None) -> torch.Tensor:
        """
        Compute reward for a trajectory based on task.
        
        Args:
            trajectory: Input trajectory or sample
            task: Task name or embedding
            
        Returns:
            Reward value
        """
        # Handle task specification
        task_name = None
        if isinstance(task, str):
            task_name = task
        
        # Use task-specific reward function if available
        if task_name is not None and task_name in self.task_reward_functions:
            return self.task_reward_functions[task_name](trajectory)
        
        # Fall back to default reward function
        if self.default_reward is not None:
            return self.default_reward(trajectory)
            
        # Last resort: return zero reward
        return torch.zeros(trajectory.shape[0], device=self.device)


class AdaptDiffuserModel(nn.Module):
    """
    Neural network model for AdaptDiffuser.
    
    This class wraps the diffusion model with task adaptation capabilities,
    providing a unified interface for the AdaptDiffuser framework.
    """
    
    def __init__(
        self,
        noise_pred_net: nn.Module,
        noise_scheduler: NoiseScheduler,
        reward_model: TaskRewardModel,
        trajectory_dim: int = 64,
        adaptation_rate: float = 0.01
    ):
        """
        Initialize the AdaptDiffuserModel.
        
        Args:
            noise_pred_net: Network for predicting noise
            noise_scheduler: Scheduler for diffusion process
            reward_model: Model for computing task-specific rewards
            trajectory_dim: Dimension of trajectory embeddings
            adaptation_rate: Rate for adaptation updates
        """
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.noise_scheduler = noise_scheduler
        self.reward_model = reward_model
        self.trajectory_dim = trajectory_dim
        self.adaptation_rate = adaptation_rate
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(noise_pred_net.parameters(), lr=adaptation_rate)
        
        # Initialize task embeddings
        self.task_embeddings = {}
        
    def forward(self, x, timesteps, task=None, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            timesteps: Timestep tensor
            task: Optional task identifier or embedding
            **kwargs: Additional arguments
            
        Returns:
            Model output
        """
        # Handle task if provided
        if task is not None:
            if isinstance(task, str) and task not in self.task_embeddings:
                # Create new task embedding
                self.task_embeddings[task] = torch.randn(1, self.trajectory_dim)
        
        # Forward through noise prediction network
        return self.noise_pred_net(x, timesteps, **kwargs)
        
    def sample(self, shape, task=None, guidance_scale=1.0):
        """
        Sample from the model.
        
        Args:
            shape: Shape of the samples to generate
            task: Optional task identifier or embedding
            guidance_scale: Scale for guidance
            
        Returns:
            Generated samples
        """
        # Simple sampling implementation
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(*shape, device=device)
        
        # Setup timesteps
        timesteps = list(range(self.noise_scheduler.num_timesteps))[::-1]
        
        # Sampling loop
        for t in timesteps:
            # Convert t to tensor for compatibility with scheduler methods
            t_tensor = torch.tensor([t], device=device)
            with torch.no_grad():
                predicted_noise = self(x, t_tensor, task=task)
                x = self.noise_scheduler.step(predicted_noise, t_tensor, x)
                
        return x
    
    def compute_gradients(self, trajectory, reward):
        """
        Compute gradients for adaptation.
        
        Args:
            trajectory: Input trajectory
            reward: Reward value
            
        Returns:
            Dictionary of gradients
        """
        # Setup
        self.noise_pred_net.zero_grad()
        
        # Ensure trajectory is a tensor
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
            
        # Add batch dimension if not present
        if len(trajectory.shape) == 1:
            trajectory = trajectory.unsqueeze(0)
            
        # Add noise to trajectory
        timestep = torch.randint(0, self.noise_scheduler.num_timesteps, (1,))
        noise = torch.randn_like(trajectory)
        # Call add_noise with the correct number of arguments
        noisy_trajectory, _ = self.noise_scheduler.add_noise(trajectory, timestep)
        
        # Predict noise
        predicted_noise = self.noise_pred_net(noisy_trajectory, timestep)
        
        # Ensure reward is a tensor
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)
            
        # Compute loss (minimize negative reward)
        loss = -reward * F.mse_loss(predicted_noise, noise)
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in self.noise_pred_net.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
                
        return gradients
    
    def apply_gradients(self, gradients):
        """
        Apply gradients for adaptation.
        
        Args:
            gradients: Dictionary of gradients
        """
        # Apply gradients and update
        for name, param in self.noise_pred_net.named_parameters():
            if name in gradients:
                param.grad = gradients[name]
                
        self.optimizer.step()
        
    def adapt_to_task(self, task, trajectories, rewards, steps=10):
        """
        Adapt model to a specific task.
        
        Args:
            task: Task identifier
            trajectories: List of example trajectories
            rewards: List of rewards for trajectories
            steps: Number of adaptation steps
            
        Returns:
            Dictionary of adaptation metrics
        """
        # Initialize metrics
        metrics = {
            "loss": [],
            "reward_improvement": 0.0,
            "steps_completed": 0
        }
        
        # Convert to tensors
        if not isinstance(trajectories[0], torch.Tensor):
            trajectories = [torch.tensor(t) for t in trajectories]
            
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards)
            
        # Initial reward evaluation
        initial_reward = rewards.mean().item()
        
        # Adaptation loop
        for step in range(steps):
            # Compute and apply gradients for each trajectory
            step_loss = 0.0
            for traj, reward in zip(trajectories, rewards):
                gradients = self.compute_gradients(traj, reward)
                self.apply_gradients(gradients)
                step_loss += -reward.item()  # Loss is negative reward
                
            # Record metrics
            avg_loss = step_loss / len(trajectories)
            metrics["loss"].append(avg_loss)
            metrics["steps_completed"] = step + 1
            
        # Final reward evaluation
        final_rewards = [self.reward_model.compute_reward(t, task).item() for t in trajectories]
        final_reward = sum(final_rewards) / len(final_rewards)
        metrics["reward_improvement"] = final_reward - initial_reward
        
        return metrics
    
    def self_evolve(self, task, discriminator, trajectory_buffer, iterations=5, trajectories_per_iter=10):
        """
        Self-evolve model using generated trajectories.
        
        Args:
            task: Task identifier
            discriminator: Discriminator for filtering trajectories
            trajectory_buffer: Buffer for storing trajectories
            iterations: Number of evolution iterations
            trajectories_per_iter: Number of trajectories per iteration
            
        Returns:
            Dictionary of evolution metrics
        """
        metrics = {
            "iterations_completed": 0,
            "trajectories_generated": 0,
            "trajectories_accepted": 0,
            "buffer_size": trajectory_buffer.size()
        }
        
        # Evolution loop
        for iteration in range(iterations):
            # Generate trajectories
            shape = (trajectories_per_iter, self.trajectory_dim)
            generated_trajectories = self.sample(shape, task=task)
            metrics["trajectories_generated"] += trajectories_per_iter
            
            # Filter using discriminator
            filtered_trajectories, filtered_rewards = discriminator.filter_trajectories(
                generated_trajectories, task=task
            )
            metrics["trajectories_accepted"] += len(filtered_trajectories)
            
            # Add to buffer
            for traj, reward in zip(filtered_trajectories, filtered_rewards):
                trajectory_buffer.add(traj, reward, task=task)
            
            # Update from buffer
            if trajectory_buffer.size(task=task) > 0:
                update_metrics = self.update_from_buffer(task, trajectory_buffer)
                
            # Update iteration counter
            metrics["iterations_completed"] += 1
            
        # Final buffer size
        metrics["final_buffer_size"] = trajectory_buffer.size()
        
        return metrics
    
    def update_from_buffer(self, task, trajectory_buffer, batch_size=8):
        """
        Update model using samples from trajectory buffer.
        
        Args:
            task: Task identifier
            trajectory_buffer: Buffer containing trajectories
            batch_size: Batch size for update
            
        Returns:
            Dictionary of update metrics
        """
        # Sample from buffer
        trajectories, rewards, _, _ = trajectory_buffer.sample(
            batch_size=min(batch_size, trajectory_buffer.size(task=task)),
            task=task
        )
        
        # Perform adaptation
        metrics = {
            "loss": 0.0,
            "samples_used": len(trajectories)
        }
        
        if len(trajectories) == 0:
            return metrics
            
        # Combine into batch
        batch_trajectories = torch.stack(trajectories)
        batch_rewards = torch.tensor(rewards)
        
        # Compute loss and update
        loss = 0.0
        for i in range(len(batch_trajectories)):
            gradients = self.compute_gradients(batch_trajectories[i], batch_rewards[i])
            self.apply_gradients(gradients)
            loss -= batch_rewards[i]  # Loss is negative reward
            
        metrics["loss"] = loss.item() / len(batch_trajectories)
        
        return metrics


class AdaptDiffuserDiscriminator(nn.Module):
    """
    Discriminator model for filtering generated trajectories.
    
    This model evaluates the quality of generated trajectories and
    filters out low-quality examples for improved adaptation.
    """
    
    def __init__(
        self,
        trajectory_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2
    ):
        """
        Initialize the discriminator.
        
        Args:
            trajectory_dim: Dimension of trajectory embeddings
            hidden_dim: Hidden dimension for network
            n_layers: Number of network layers
        """
        super().__init__()
        
        # Build network
        layers = [nn.Linear(trajectory_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, trajectory):
        """
        Compute quality score for trajectory.
        
        Args:
            trajectory: Input trajectory
            
        Returns:
            Quality score between 0 and 1
        """
        return self.net(trajectory)
    
    def filter_trajectories(self, trajectories, task=None, threshold=0.5):
        """
        Filter trajectories based on quality.
        
        Args:
            trajectories: Batch of trajectories
            task: Optional task identifier (not used in basic implementation)
            threshold: Quality threshold
            
        Returns:
            Filtered trajectories and their quality scores
        """
        with torch.no_grad():
            # Compute quality scores
            if len(trajectories.shape) == 3:  # Batch, sequence, features
                flat_trajectories = trajectories.reshape(trajectories.shape[0], -1)
            else:
                flat_trajectories = trajectories
                
            quality_scores = self.forward(flat_trajectories).squeeze()
            
            # Filter based on threshold
            indices = torch.where(quality_scores >= threshold)[0]
            filtered_trajectories = [trajectories[i] for i in indices]
            filtered_scores = quality_scores[indices].tolist()
            
            return filtered_trajectories, filtered_scores