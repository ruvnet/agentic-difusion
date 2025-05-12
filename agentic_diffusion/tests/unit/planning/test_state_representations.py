"""
Unit tests for state representation components used in planning.

This module tests the integration between state representations and AdaptDiffuser's
capabilities, ensuring proper encoding, decoding, and reward computation.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.planning.state_representations import (
    StateEncoder,
    StateRewardModel,
    TaskAdaptiveStateRepresentation
)
from agentic_diffusion.core.adapt_diffuser.utils import encode_task


class TestStateEncoder:
    """Tests for the StateEncoder class."""
    
    def test_initialization(self):
        """Test that the StateEncoder initializes correctly."""
        encoder = StateEncoder(
            state_dim=4,
            latent_dim=8,
            hidden_dims=[16, 32],
            device="cpu"
        )
        
        assert encoder.state_dim == 4
        assert encoder.latent_dim == 8
        assert len(encoder.encoder) > 0
        assert len(encoder.decoder) > 0
    
    def test_encode_decode(self):
        """Test encoding and decoding functionality."""
        encoder = StateEncoder(
            state_dim=4,
            latent_dim=8,
            device="cpu"
        )
        
        # Create a test state
        state = torch.randn(4)
        
        # Encode the state
        latent = encoder.encode(state)
        
        # Check latent dimensions
        assert latent.shape == torch.Size([8])
        
        # Decode the latent
        reconstructed = encoder.decode(latent)
        
        # Check reconstructed dimensions
        assert reconstructed.shape == torch.Size([4])
    
    def test_batch_processing(self):
        """Test that the encoder can process batches of states."""
        encoder = StateEncoder(
            state_dim=4,
            latent_dim=8,
            device="cpu"
        )
        
        # Create a batch of states
        batch_size = 10
        states = torch.randn(batch_size, 4)
        
        # Encode the batch
        latents = encoder.encode_batch(states)
        
        # Check dimensions
        assert latents.shape == torch.Size([batch_size, 8])
        
        # Decode the batch
        reconstructed = encoder.decode_batch(latents)
        
        # Check dimensions
        assert reconstructed.shape == torch.Size([batch_size, 4])


class TestStateRewardModel:
    """Tests for the StateRewardModel class."""
    
    def test_initialization(self):
        """Test that the StateRewardModel initializes correctly."""
        reward_model = StateRewardModel(
            state_dim=4,
            task_dim=16,
            hidden_dims=[32, 32],
            device="cpu"
        )
        
        assert reward_model.state_dim == 4
        assert reward_model.task_dim == 16
        assert len(reward_model.model) > 0
    
    def test_compute_reward(self):
        """Test reward computation."""
        reward_model = StateRewardModel(
            state_dim=4,
            task_dim=16,
            device="cpu"
        )
        
        # Create a test state and task
        state = torch.randn(4)
        task = torch.randn(16)
        
        # Compute reward
        reward = reward_model.compute_reward(state, task)
        
        # Check reward shape and range
        assert reward.shape == torch.Size([1])
        assert 0.0 <= reward.item() <= 1.0  # Assuming sigmoid activation
    
    def test_batch_reward_computation(self):
        """Test reward computation for batches."""
        reward_model = StateRewardModel(
            state_dim=4,
            task_dim=16,
            device="cpu"
        )
        
        # Create batch of states and task
        batch_size = 10
        states = torch.randn(batch_size, 4)
        task = torch.randn(16)
        
        # Compute rewards
        rewards = reward_model.compute_reward(states, task)
        
        # Check rewards shape
        assert rewards.shape == torch.Size([batch_size])
    
    def test_reward_gradient(self):
        """Test reward gradient computation."""
        reward_model = StateRewardModel(
            state_dim=4,
            task_dim=16,
            device="cpu"
        )
        
        # Create a test state and task
        state = torch.randn(4, requires_grad=True)
        task = torch.randn(16)
        
        # Compute reward gradient
        grad = reward_model.compute_reward_gradient(state, task)
        
        # Check gradient shape
        assert grad.shape == torch.Size([4])


class TestTaskAdaptiveStateRepresentation:
    """Tests for the TaskAdaptiveStateRepresentation class."""
    
    @pytest.fixture
    def state_representation(self):
        """Create a TaskAdaptiveStateRepresentation instance for testing."""
        encoder = StateEncoder(
            state_dim=4,
            latent_dim=8,
            device="cpu"
        )
        
        reward_model = StateRewardModel(
            state_dim=8,  # Using latent dimension
            task_dim=16,
            device="cpu"
        )
        
        return TaskAdaptiveStateRepresentation(
            state_encoder=encoder,
            reward_model=reward_model,
            device="cpu"
        )
    
    def test_encoding_with_task_conditioning(self, state_representation):
        """Test state encoding with task conditioning."""
        # Create test state and task
        state = torch.randn(4)
        task = torch.randn(16)
        
        # Encode state with task conditioning
        latent = state_representation.encode_states(state, task)
        
        # Check latent dimensions
        assert latent.shape == torch.Size([8])
        
        # Test batch encoding
        batch_size = 5
        states = torch.randn(batch_size, 4)
        
        latents = state_representation.encode_states(states, task)
        
        # Check batch latent dimensions
        assert latents.shape == torch.Size([batch_size, 8])
    
    def test_reward_computation(self, state_representation):
        """Test reward computation with state representation."""
        # Create test state and task
        state = torch.randn(4)
        task = torch.randn(16)
        
        # Encode state
        latent = state_representation.encode_states(state, task)
        
        # Compute reward
        reward = state_representation.compute_rewards(latent, task)
        
        # Check reward shape
        assert reward.shape == torch.Size([1])
        
        # Test batch reward computation
        batch_size = 5
        states = torch.randn(batch_size, 4)
        
        # Encode states
        latents = state_representation.encode_states(states, task)
        
        # Compute rewards
        rewards = state_representation.compute_rewards(latents, task)
        
        # Check rewards shape
        assert rewards.shape == torch.Size([batch_size])
    
    def test_reward_gradient_computation(self, state_representation):
        """Test reward gradient computation."""
        # Create test state and task
        state = torch.randn(4)
        task = torch.randn(16)
        
        # Encode state
        latent = state_representation.encode_states(state, task)
        latent.requires_grad_(True)
        
        # Compute reward gradient
        gradient = state_representation.compute_reward_gradients(latent, task)
        
        # Check gradient shape
        assert gradient.shape == torch.Size([1, 8])
        
        # Test batch gradient computation
        batch_size = 3
        latents = torch.randn(batch_size, 8, requires_grad=True)
        
        # Compute gradients
        gradients = state_representation.compute_reward_gradients(latents, task)
        
        # Check gradients shape
        assert gradients.shape == torch.Size([batch_size, 8])
    
    def test_task_adaptation(self, state_representation):
        """Test task adaptation capabilities."""
        # Create demonstration states and rewards
        num_demos = 10
        demo_states = torch.randn(num_demos, 4)
        demo_rewards = torch.rand(num_demos)
        
        # Create task through adaptation
        with patch.object(state_representation.reward_model, 'update_weights') as mock_update:
            state_representation.adapt_to_demonstrations(
                demo_states=demo_states,
                demo_rewards=demo_rewards,
                num_steps=5,
                learning_rate=1e-4
            )
            
            # Check that update_weights was called
            assert mock_update.called