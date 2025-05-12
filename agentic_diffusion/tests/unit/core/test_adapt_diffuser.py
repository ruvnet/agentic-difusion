"""
Unit tests for the AdaptDiffuser implementation.

These tests verify the functionality of the AdaptDiffuser model, its integration with the
guided denoising process, and the enhanced trajectory buffer.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.core.adapt_diffuser import (
    AdaptDiffuserModel, AdaptDiffuserDiscriminator, TaskEmbeddingManager, TaskRewardModel
)
from agentic_diffusion.core.denoising_process import GuidedDenoisingProcess, GuidedDDPMSampler
from agentic_diffusion.core.trajectory_buffer import AdaptDiffuserTrajectoryBuffer
from agentic_diffusion.core.noise_schedules import LinearNoiseScheduler


class TestAdaptDiffuser:
    """Test suite for AdaptDiffuser and related components."""
    
    @pytest.fixture
    def dummy_noise_pred_net(self):
        """Create a simple network for testing."""
        class DummyNetwork(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim + 1, 128),  # +1 for timestep embedding
                    nn.ReLU(),
                    nn.Linear(128, dim)
                )
                
            def forward(self, x, t, **kwargs):
                # Ensure t is a tensor
                if isinstance(t, int):
                    t = torch.tensor([t], device=x.device)
                
                # Check if x is already batched, if not add batch dimension
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)
                
                # Expand time embedding to match batch size of x
                t_emb = t.float().unsqueeze(1) / 1000.0  # Simple timestep embedding
                t_emb = t_emb.expand(x.shape[0], -1)  # Expand to match batch dimension
                
                # Flatten input if it has more than 2 dimensions
                x_flat = x.reshape(x.shape[0], -1)
                
                # Concatenate along feature dimension
                inputs = torch.cat([x_flat, t_emb], dim=1)
                
                return self.net(inputs).reshape(x.shape)
                
        return DummyNetwork()
    
    @pytest.fixture
    def noise_scheduler(self):
        """Create a noise scheduler for testing."""
        return LinearNoiseScheduler(num_timesteps=100)
    
    @pytest.fixture
    def reward_model(self):
        """Create a reward model for testing."""
        def default_reward_fn(trajectory):
            # Simple reward based on L2 norm (smaller norm = higher reward)
            # Handle case where trajectory is 1D (not batched)
            if len(trajectory.shape) == 1:
                norm = torch.norm(trajectory, p=2)
            else:
                norm = torch.norm(trajectory, p=2, dim=1).mean()
                
            reward = 1.0 / (1.0 + norm)
            return reward
            
        reward_model = TaskRewardModel(
            default_reward=default_reward_fn
        )
        return reward_model
    
    @pytest.fixture
    def adapt_diffuser_model(self, dummy_noise_pred_net, noise_scheduler, reward_model):
        """Create an AdaptDiffuser model for testing."""
        return AdaptDiffuserModel(
            noise_pred_net=dummy_noise_pred_net,
            noise_scheduler=noise_scheduler,
            reward_model=reward_model,
            trajectory_dim=64,
            adaptation_rate=0.01
        )
    
    @pytest.fixture
    def trajectory_buffer(self):
        """Create a trajectory buffer for testing."""
        return AdaptDiffuserTrajectoryBuffer(
            capacity=100,
            alpha=0.6,
            beta=0.4
        )
    
    @pytest.fixture
    def discriminator(self):
        """Create a discriminator for testing."""
        return AdaptDiffuserDiscriminator(
            trajectory_dim=64,
            hidden_dim=128,
            n_layers=2
        )
    
    def test_adapt_diffuser_initialization(self, adapt_diffuser_model):
        """Test that AdaptDiffuser model is initialized correctly."""
        assert adapt_diffuser_model is not None
        assert adapt_diffuser_model.noise_pred_net is not None
        assert adapt_diffuser_model.noise_scheduler is not None
        assert adapt_diffuser_model.reward_model is not None
        assert adapt_diffuser_model.trajectory_dim == 64
        assert adapt_diffuser_model.adaptation_rate == 0.01
        assert isinstance(adapt_diffuser_model.optimizer, torch.optim.Optimizer)
        assert len(adapt_diffuser_model.task_embeddings) == 0
    
    def test_adapt_diffuser_forward(self, adapt_diffuser_model):
        """Test forward pass of AdaptDiffuser model."""
        # Create test input
        batch_size = 2
        trajectory = torch.randn(batch_size, 64)
        timesteps = torch.tensor([10, 50])
        
        # Test without task
        output = adapt_diffuser_model.forward(trajectory, timesteps)
        assert output.shape == trajectory.shape
        
        # Test with task as string
        output = adapt_diffuser_model.forward(trajectory, timesteps, task="test_task")
        assert output.shape == trajectory.shape
        assert "test_task" in adapt_diffuser_model.task_embeddings
        
        # Test with task as tensor
        task_tensor = torch.randn(1, 64)
        output = adapt_diffuser_model.forward(trajectory, timesteps, task=task_tensor)
        assert output.shape == trajectory.shape
    
    def test_adapt_diffuser_sample(self, adapt_diffuser_model):
        """Test sampling from AdaptDiffuser model."""
        # Test sampling without task
        samples = adapt_diffuser_model.sample((2, 64))
        assert samples.shape == (2, 64)
        
        # Test sampling with task
        samples = adapt_diffuser_model.sample((2, 64), task="test_task", guidance_scale=0.5)
        assert samples.shape == (2, 64)
    
    def test_compute_and_apply_gradients(self, adapt_diffuser_model):
        """Test gradient computation and application."""
        # Create test trajectory
        trajectory = torch.randn(1, 64)
        reward = 0.7
        
        # Initial parameter values
        initial_params = [p.clone().detach() for p in adapt_diffuser_model.noise_pred_net.parameters()]
        
        # Compute and apply gradients
        gradients = adapt_diffuser_model.compute_gradients(trajectory, reward)
        adapt_diffuser_model.apply_gradients(gradients)
        
        # Check that parameters have changed
        for i, p in enumerate(adapt_diffuser_model.noise_pred_net.parameters()):
            assert not torch.allclose(p, initial_params[i]), f"Parameter {i} did not change"
    
    def test_adapt_to_task(self, adapt_diffuser_model):
        """Test adaptation to a specific task."""
        # Create test data
        trajectories = [torch.randn(64) for _ in range(5)]
        rewards = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Adapt to task
        metrics = adapt_diffuser_model.adapt_to_task(
            task="test_adaptation",
            trajectories=trajectories,
            rewards=rewards,
            steps=3
        )
        
        # Check metrics
        assert "loss" in metrics
        assert "reward_improvement" in metrics
        assert "steps_completed" in metrics
        assert metrics["steps_completed"] == 3
    
    def test_self_evolve(self, adapt_diffuser_model, discriminator, trajectory_buffer):
        """Test self-evolution through synthetic data."""
        # Mock discriminator filter_trajectories to avoid computation
        orig_filter = discriminator.filter_trajectories
        discriminator.filter_trajectories = MagicMock(return_value=(
            [torch.randn(64) for _ in range(2)],
            [0.7, 0.8]
        ))
        
        # Mock update_from_buffer for faster testing
        adapt_diffuser_model.update_from_buffer = MagicMock(return_value={"loss": 0.1})
        
        # Run self-evolution
        metrics = adapt_diffuser_model.self_evolve(
            task="test_evolve",
            discriminator=discriminator,
            trajectory_buffer=trajectory_buffer,
            iterations=2,
            trajectories_per_iter=5
        )
        
        # Check metrics
        assert "iterations_completed" in metrics
        assert "trajectories_generated" in metrics
        assert "trajectories_accepted" in metrics
        assert metrics["iterations_completed"] == 2
        assert metrics["trajectories_generated"] == 10
        assert metrics["trajectories_accepted"] > 0
        
        # Restore original method
        discriminator.filter_trajectories = orig_filter
    
    def test_enhance_trajectory_buffer(self, trajectory_buffer):
        """Test the enhanced trajectory buffer."""
        # Add trajectories for different tasks
        traj1 = torch.randn(64)
        traj2 = torch.randn(64)
        traj3 = torch.randn(64)
        
        # Add to buffer
        trajectory_buffer.add(traj1, 0.7, task="task1")
        trajectory_buffer.add(traj2, 0.8, task="task1")
        trajectory_buffer.add(traj3, 0.9, task="task2")
        
        # Test size
        assert trajectory_buffer.size() == 3
        assert trajectory_buffer.size(task="task1") == 2
        assert trajectory_buffer.size(task="task2") == 1
        
        # Test get_task_trajectories
        task1_trajectories, task1_rewards = trajectory_buffer.get_task_trajectories("task1")
        assert len(task1_trajectories) == 2
        assert len(task1_rewards) == 2
        assert task1_rewards[0] > task1_rewards[1]  # Sorted by reward
        
        # Test sampling
        sampled_trajectories, sampled_rewards, indices, weights = trajectory_buffer.sample(2, task="task1")
        assert len(sampled_trajectories) == 2
        assert len(sampled_rewards) == 2
        assert len(indices) == 2
        
        # Test updating priorities
        trajectory_buffer.update_priorities([indices[0]], [1.5])
        
        # Test clear with task filter
        removed = trajectory_buffer.clear(task="task2")
        assert removed == 1
        assert trajectory_buffer.size() == 2
        assert trajectory_buffer.size(task="task2") == 0
    
    def test_guided_denoising_process(self, adapt_diffuser_model, noise_scheduler):
        """Test the guided denoising process."""
        # Create guided denoising process
        process = GuidedDenoisingProcess(
            model=adapt_diffuser_model,
            noise_scheduler=noise_scheduler,
            img_size=8,  # Small size for testing
            channels=1,
            reward_model=adapt_diffuser_model.reward_model,
            guidance_scale=0.5
        )
        
        # Create sampler
        sampler = GuidedDDPMSampler(process, num_timesteps=5)
        
        # Test sampling without guidance
        samples = sampler.sample(batch_size=2, guidance_scale=0.0)
        assert samples.shape == (2, 1, 8, 8)
        
        # Test sampling with guidance
        task_embedding = torch.randn(1, 64)
        samples = sampler.sample(batch_size=1, task=task_embedding, guidance_scale=0.5)
        assert samples.shape == (1, 1, 8, 8)