"""
Unit tests for the AdaptDiffuser adaptation implementation.
"""

import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from agentic_diffusion.adaptation.adapt_diffuser_adaptation import (
    SyntheticExpertGenerator,
    AdaptDiffuserDiscriminator,
    AdaptDiffuserAdaptation
)
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser


class MockAdaptDiffuser:
    """Mock AdaptDiffuser for testing."""
    
    def __init__(self):
        self.img_size = 32
        self.channels = 3
        
        # Mock task embedding model
        class MockTaskEmbedding:
            def __init__(self):
                self.embedding_dim = 16
            
            def encode(self, task):
                if isinstance(task, str):
                    # Simple hash-based encoding
                    task_hash = hash(task) % 10000
                    embedding = torch.zeros(16)
                    for i in range(16):
                        embedding[i] = ((task_hash >> i) & 1) * 2 - 1
                    return embedding
                return task
        
        self.task_embedding_model = MockTaskEmbedding()
        
    def encode_task(self, task):
        return self.task_embedding_model.encode(task)
        
    def generate(self, batch_size=1, task=None, custom_guidance_scale=None, conditioning=None):
        shape = (batch_size, self.channels, self.img_size, self.img_size)
        return torch.randn(shape)
        
    def compute_reward(self, trajectories, task=None):
        # Simple mock reward function
        if isinstance(trajectories, list):
            batch_size = len(trajectories)
        else:
            batch_size = trajectories.shape[0]
        return torch.rand(batch_size)
        
    def adapt_to_task(self, task, num_steps=100, batch_size=8, **kwargs):
        return {"loss": 0.5 - 0.01 * num_steps}


@pytest.fixture
def mock_adapt_diffuser():
    """Fixture for mock AdaptDiffuser."""
    return MockAdaptDiffuser()


@pytest.fixture
def synthetic_generator(mock_adapt_diffuser):
    """Fixture for SyntheticExpertGenerator."""
    return SyntheticExpertGenerator(
        adapt_diffuser=mock_adapt_diffuser,
        generation_steps=5,
        refinement_iterations=2,
        quality_threshold=0.6,
        batch_size=4
    )


@pytest.fixture
def discriminator():
    """Fixture for AdaptDiffuserDiscriminator."""
    return AdaptDiffuserDiscriminator(
        input_dim=32 * 32 * 3,
        task_embedding_dim=16,
        hidden_dims=[64, 32],
        dropout_rate=0.1
    )


@pytest.fixture
def adaptation_mechanism(mock_adapt_diffuser, discriminator, synthetic_generator, tmp_path):
    """Fixture for AdaptDiffuserAdaptation."""
    checkpoint_dir = os.path.join(tmp_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return AdaptDiffuserAdaptation(
        adapt_diffuser=mock_adapt_diffuser,
        discriminator=discriminator,
        synthetic_expert_generator=synthetic_generator,
        refinement_steps=3,
        adaptation_rate=0.1,
        quality_threshold=0.6,
        memory_capacity=10,
        checkpoint_dir=checkpoint_dir
    )


class TestSyntheticExpertGenerator:
    """Tests for SyntheticExpertGenerator."""
    
    def test_init(self, synthetic_generator, mock_adapt_diffuser):
        """Test initialization."""
        assert synthetic_generator.adapt_diffuser == mock_adapt_diffuser
        assert synthetic_generator.generation_steps == 5
        assert synthetic_generator.refinement_iterations == 2
        assert synthetic_generator.quality_threshold == 0.6
        assert synthetic_generator.batch_size == 4
    
    def test_generate_synthetic_data(self, synthetic_generator):
        """Test generating synthetic data."""
        samples, rewards = synthetic_generator.generate_synthetic_data(
            task="test_task",
            num_samples=2
        )
        
        assert isinstance(samples, list)
        assert isinstance(rewards, list)
        assert len(samples) <= 2  # May be less if quality threshold filters some out
        
        # Check at least one sample was generated
        assert len(samples) > 0
        
        # Check sample shape
        assert samples[0].shape == (3, 32, 32)
        
        # Check rewards
        for reward in rewards:
            assert isinstance(reward, float)
            assert 0 <= reward <= 1


class TestAdaptDiffuserDiscriminator:
    """Tests for AdaptDiffuserDiscriminator."""
    
    def test_init(self, discriminator):
        """Test initialization."""
        assert discriminator.use_task_conditioning is True
        assert isinstance(discriminator.model, torch.nn.Sequential)
    
    def test_forward_pass(self, discriminator):
        """Test forward pass."""
        batch_size = 3
        samples = torch.randn(batch_size, 3, 32, 32)
        task_embedding = torch.randn(16)
        
        scores = discriminator(samples, task_embedding)
        
        assert scores.shape == (batch_size, 1)
        assert torch.all((scores >= 0) & (scores <= 1))
    
    def test_evaluate_quality(self, discriminator):
        """Test quality evaluation."""
        samples = [torch.randn(3, 32, 32) for _ in range(3)]
        task_embedding = torch.randn(16)
        
        scores = discriminator.evaluate_quality(samples, task_embedding)
        
        assert scores.shape == (3, 1)
        assert torch.all((scores >= 0) & (scores <= 1))
    
    def test_filter_trajectories(self, discriminator):
        """Test trajectory filtering."""
        samples = [torch.randn(3, 32, 32) for _ in range(5)]
        task_embedding = torch.randn(16)
        
        # Mock the evaluate_quality method to return controlled values
        original_evaluate = discriminator.evaluate_quality
        discriminator.evaluate_quality = MagicMock(
            return_value=torch.tensor([[0.9], [0.3], [0.7], [0.2], [0.8]])
        )
        
        filtered, scores = discriminator.filter_trajectories(
            samples, 
            task_embedding,
            threshold=0.5
        )
        
        # Restore original method
        discriminator.evaluate_quality = original_evaluate
        
        assert len(filtered) == 3  # Should have 3 samples with score >= 0.5
        assert len(scores) == 3
        assert all(score >= 0.5 for score in scores)


class TestAdaptDiffuserAdaptation:
    """Tests for AdaptDiffuserAdaptation."""
    
    def test_init(self, adaptation_mechanism, mock_adapt_diffuser, discriminator, synthetic_generator):
        """Test initialization."""
        assert adaptation_mechanism.adapt_diffuser == mock_adapt_diffuser
        assert adaptation_mechanism.discriminator == discriminator
        assert adaptation_mechanism.synthetic_expert_generator == synthetic_generator
        assert adaptation_mechanism.refinement_steps == 3
        assert adaptation_mechanism.adaptation_rate == 0.1
        assert adaptation_mechanism.quality_threshold == 0.6
        assert adaptation_mechanism.memory_capacity == 10
        assert isinstance(adaptation_mechanism.memory_buffer, list)
    
    def test_adapt_to_task(self, adaptation_mechanism):
        """Test adaptation to a task."""
        metrics = adaptation_mechanism._adapt_to_task(
            task="test_task",
            num_steps=5,
            batch_size=2
        )
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics
    
    def test_memory_buffer(self, adaptation_mechanism):
        """Test memory buffer operations."""
        # Store trajectories
        for i in range(3):
            trajectory = torch.randn(3, 32, 32)
            reward = 0.5 + i * 0.2
            adaptation_mechanism.store_trajectory(trajectory, reward, "test_task")
        
        assert len(adaptation_mechanism.memory_buffer) == 3
        
        # Sample from memory
        samples, rewards, tasks = adaptation_mechanism._sample_from_memory(batch_size=2)
        
        assert len(samples) == 2
        assert len(rewards) == 2
        assert len(tasks) == 2
        
        # Rewards should be sorted in descending order
        assert rewards[0] > rewards[1]
    
    def test_state_saving_loading(self, adaptation_mechanism, tmp_path):
        """Test saving and loading state."""
        save_path = os.path.join(tmp_path, "adapt_diffuser_test.pkl")
        
        # Store something in the buffer
        for i in range(3):
            trajectory = torch.randn(3, 32, 32)
            reward = 0.5 + i * 0.2
            adaptation_mechanism.store_trajectory(trajectory, reward, "test_task")
        
        # Save state
        success = adaptation_mechanism.save_state(save_path)
        assert success is True
        assert os.path.exists(save_path)
        
        # Create new mechanism
        new_mechanism = AdaptDiffuserAdaptation(
            adapt_diffuser=MockAdaptDiffuser(),
            checkpoint_dir=os.path.join(tmp_path, "new_checkpoints")
        )
        
        # Load state
        success = new_mechanism.load_state(save_path)
        assert success is True
        
        # Check parameters were loaded
        assert new_mechanism.adaptation_rate == adaptation_mechanism.adaptation_rate
        assert new_mechanism.quality_threshold == adaptation_mechanism.quality_threshold
    
    def test_adapt_with_code(self, adaptation_mechanism):
        """Test adapting code."""
        original_code = "def example(): return 42"
        
        adapted_code = adaptation_mechanism.adapt(
            code=original_code,
            feedback={"task": "improve_code"},
            language="python"
        )
        
        # In our implementation, this should return the original code 
        # since we don't have a real code adaptation model
        assert adapted_code is not None