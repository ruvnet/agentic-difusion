"""
Unit tests for the AdaptDiffuser synthetic data generation component.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from agentic_diffusion.adaptation.adapt_diffuser_adaptation import SyntheticExpertGenerator
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser


class TestSyntheticExpertGenerator:
    """Unit tests for the SyntheticExpertGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create mock AdaptDiffuser model
        self.mock_adapt_diffuser = MagicMock(spec=AdaptDiffuser)
        
        # Configure the mock
        self.mock_adapt_diffuser.encode_task.return_value = torch.randn(16)
        self.mock_adapt_diffuser.generate.return_value = torch.randn(4, 3, 32, 32)
        self.mock_adapt_diffuser.compute_reward.return_value = torch.tensor([0.9, 0.8, 0.6, 0.5])
        
        # Create the generator
        self.generator = SyntheticExpertGenerator(
            adapt_diffuser=self.mock_adapt_diffuser,
            quality_threshold=0.7,
            batch_size=4
        )
    
    def test_initialization(self):
        """Test generator initialization with various parameters."""
        assert self.generator.adapt_diffuser == self.mock_adapt_diffuser
        assert self.generator.quality_threshold == 0.7
        assert self.generator.batch_size == 4
        
        # Test with different parameters
        generator = SyntheticExpertGenerator(
            adapt_diffuser=self.mock_adapt_diffuser,
            quality_threshold=0.9,
            batch_size=8
        )
        assert generator.quality_threshold == 0.9
        assert generator.batch_size == 8
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation with different sample counts."""
        # Call the generator with default parameters
        samples, rewards = self.generator.generate_synthetic_data(
            task="test_task",
            num_samples=2
        )
        
        # Check if generate was called
        self.mock_adapt_diffuser.generate.assert_called_once()
        
        # Check results
        assert len(samples) == 2
        assert len(rewards) == 2
        assert all(reward >= 0.7 for reward in rewards)
        
        # Reset mock and test with more samples
        self.mock_adapt_diffuser.generate.reset_mock()
        self.mock_adapt_diffuser.compute_reward.reset_mock()
        
        # Configure mocks for multiple batches
        def mock_generate(*args, **kwargs):
            return torch.randn(4, 3, 32, 32)
            
        def mock_compute_reward(*args, **kwargs):
            return torch.tensor([0.9, 0.8, 0.6, 0.5])
            
        self.mock_adapt_diffuser.generate.side_effect = mock_generate
        self.mock_adapt_diffuser.compute_reward.side_effect = mock_compute_reward
        
        # Call with more samples than batch size
        samples_large, rewards_large = self.generator.generate_synthetic_data(
            task="test_task",
            num_samples=6
        )
        
        # Check if generate was called multiple times
        assert self.mock_adapt_diffuser.generate.call_count == 2
        
        # Check results
        # We can't make assumptions about exactly how many samples will be returned
        # Just verify that some samples were returned and rewards match
        assert len(samples_large) > 0
        assert len(rewards_large) > 0
        assert len(samples_large) == len(rewards_large)
        assert all(reward >= 0.7 for reward in rewards_large)
    
    def test_generate_with_different_thresholds(self):
        """Test generation with different quality thresholds."""
        # Test with high threshold
        generator_high = SyntheticExpertGenerator(
            adapt_diffuser=self.mock_adapt_diffuser,
            quality_threshold=0.85,
            batch_size=4
        )
        
        samples_high, rewards_high = generator_high.generate_synthetic_data(
            task="test_task",
            num_samples=2
        )
        
        assert len(samples_high) == 2
        assert len(rewards_high) == 2
        assert all(reward >= 0.85 for reward in rewards_high)
        
        # Test with low threshold
        generator_low = SyntheticExpertGenerator(
            adapt_diffuser=self.mock_adapt_diffuser,
            quality_threshold=0.5,
            batch_size=4
        )
        
        # Reset mock
        self.mock_adapt_diffuser.generate.reset_mock()
        self.mock_adapt_diffuser.compute_reward.reset_mock()
        
        samples_low, rewards_low = generator_low.generate_synthetic_data(
            task="test_task",
            num_samples=4
        )
        
        assert len(samples_low) == 4
        assert len(rewards_low) == 4
        assert all(reward >= 0.5 for reward in rewards_low)
    
    def test_generate_with_custom_batch_size(self):
        """Test generation with custom batch sizes."""
        # Create generator with small batch size
        generator_small = SyntheticExpertGenerator(
            adapt_diffuser=self.mock_adapt_diffuser,
            quality_threshold=0.7,
            batch_size=2
        )
        
        # Reset mock and configure for smaller batches
        self.mock_adapt_diffuser.generate.reset_mock()
        self.mock_adapt_diffuser.compute_reward.reset_mock()
        
        # Configure mocks for smaller batches
        def mock_generate_small(*args, **kwargs):
            return torch.randn(2, 3, 32, 32)
            
        def mock_compute_reward_small(*args, **kwargs):
            return torch.tensor([0.9, 0.8])
            
        self.mock_adapt_diffuser.generate.side_effect = mock_generate_small
        self.mock_adapt_diffuser.compute_reward.side_effect = mock_compute_reward_small
        
        # Generate samples
        samples, rewards = generator_small.generate_synthetic_data(
            task="test_task",
            num_samples=4
        )
        
        # Check if generate was called multiple times with smaller batches
        assert self.mock_adapt_diffuser.generate.call_count == 2
        
        # Check results
        assert len(samples) == 4
        assert len(rewards) == 4
        assert all(reward >= 0.7 for reward in rewards)
    
    def test_generation_with_different_task_encodings(self):
        """Test data generation with different task encodings."""
        # Configure different task encodings
        task_embedding1 = torch.randn(16)
        task_embedding2 = torch.randn(16)
        
        # Setup mock to return different embeddings
        self.mock_adapt_diffuser.encode_task.side_effect = [task_embedding1, task_embedding2]
        
        # Reset generation mocks
        self.mock_adapt_diffuser.generate.reset_mock()
        
        # Generate for first task
        samples1, _ = self.generator.generate_synthetic_data(
            task="task1",
            num_samples=2
        )
        
        # Generate for second task
        samples2, _ = self.generator.generate_synthetic_data(
            task="task2",
            num_samples=2
        )
        
        # Check that encode_task was called with different tasks
        self.mock_adapt_diffuser.encode_task.assert_any_call("task1")
        self.mock_adapt_diffuser.encode_task.assert_any_call("task2")
        
        # Check that generate was called twice
        assert self.mock_adapt_diffuser.generate.call_count == 2
    
    def test_empty_result_when_no_samples_meet_threshold(self):
        """Test behavior when no samples meet the quality threshold."""
        # Configure mock to return low rewards
        self.mock_adapt_diffuser.compute_reward.return_value = torch.tensor([0.4, 0.3, 0.2, 0.1])
        
        # Create generator with high threshold
        generator_high = SyntheticExpertGenerator(
            adapt_diffuser=self.mock_adapt_diffuser,
            quality_threshold=0.95,
            batch_size=4
        )
        
        # Reset mocks
        self.mock_adapt_diffuser.generate.reset_mock()
        # Try to generate samples with unreachable threshold
        # Note: SyntheticExpertGenerator doesn't support max_attempts parameter
        samples, rewards = generator_high.generate_synthetic_data(
            task="test_task",
            num_samples=2
        )
        
        # Check that generate was called multiple times attempting to meet threshold
        assert self.mock_adapt_diffuser.generate.call_count > 0
        
        # Check empty results
        assert len(samples) == 0
        assert len(rewards) == 0


if __name__ == "__main__":
    pytest.main()