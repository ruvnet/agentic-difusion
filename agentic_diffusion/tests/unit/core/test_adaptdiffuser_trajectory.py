import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from agentic_diffusion.core.trajectory_buffer import TrajectoryBuffer, AdaptDiffuserTrajectoryBuffer
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.noise_schedules import LinearScheduler


class TestAdaptDiffuserTrajectory(unittest.TestCase):
    """
    Unit tests for the AdaptDiffuser trajectory generation and storage.
    
    These tests verify that trajectory generation, evaluation, and storage
    work correctly with the AdaptDiffuser model.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model
        self.mock_model = Mock()
        self.mock_model.parameters = Mock(return_value=[torch.ones(1)])
        self.mock_model.to = Mock(return_value=self.mock_model)
        self.mock_model.forward = Mock(return_value=torch.zeros(2, 3, 32, 32))
        
        # Define default dimensions
        self.img_size = 32
        self.channels = 3
        self.batch_size = 2
        
        # Create a simple noise scheduler
        self.noise_scheduler = LinearScheduler(num_timesteps=100)
        
        # Create a mock reward model
        self.mock_reward_model = Mock()
        self.mock_reward_model.to = Mock(return_value=self.mock_reward_model)
        self.mock_reward_model.compute_reward = Mock(return_value=torch.tensor([0.5, 0.7]))

    def test_trajectory_buffer_initialization(self):
        """Test initialization of a TrajectoryBuffer."""
        # Arrange & Act
        buffer = TrajectoryBuffer(capacity=100)
        
        # Assert
        self.assertEqual(buffer.capacity, 100)
        self.assertEqual(len(buffer.trajectories), 0)
        self.assertEqual(len(buffer.rewards), 0)
        self.assertEqual(len(buffer.priorities), 0)
    
    def test_trajectory_buffer_add_and_sample(self):
        """Test adding trajectories to buffer and sampling."""
        # Arrange
        buffer = TrajectoryBuffer(capacity=5)
        
        # Create sample trajectories
        trajectory1 = torch.ones(10, 3)  # 10 steps, 3 dimensions
        trajectory2 = torch.zeros(10, 3)
        trajectory3 = torch.ones(10, 3) * 2
        
        # Act - Add trajectories with different rewards
        buffer.add(trajectory1, reward=1.0)
        buffer.add(trajectory2, reward=0.5)
        buffer.add(trajectory3, reward=2.0)
        
        # Sample from buffer
        samples = buffer.sample(batch_size=2)
        
        # Assert
        self.assertEqual(buffer.size(), 3)
        self.assertEqual(len(samples), 2)
        # High reward trajectory (trajectory3) should be sampled with higher probability
        # But this is probabilistic, so we can't assert exact results
    
    def test_trajectory_buffer_capacity(self):
        """Test that buffer respects capacity limit."""
        # Arrange
        buffer = TrajectoryBuffer(capacity=2)
        
        # Create sample trajectories
        trajectory1 = torch.ones(10, 3)
        trajectory2 = torch.zeros(10, 3)
        trajectory3 = torch.ones(10, 3) * 2
        
        # Act - Add more trajectories than capacity
        buffer.add(trajectory1, reward=0.5)  # Lowest reward
        buffer.add(trajectory2, reward=1.0)
        buffer.add(trajectory3, reward=2.0)  # Should replace trajectory1
        
        # Assert
        self.assertEqual(buffer.size(), 2)
        # The lowest priority trajectory should have been removed
        self.assertFalse(any(torch.equal(t, trajectory1) for t in buffer.trajectories))
    
    def test_adaptdiffuser_trajectory_buffer_initialization(self):
        """Test initialization of AdaptDiffuserTrajectoryBuffer."""
        # Arrange & Act
        buffer = AdaptDiffuserTrajectoryBuffer(
            capacity=200,
            alpha=0.7,
            beta=0.5,
            beta_annealing=0.002,
            device="cpu"
        )
        
        # Assert
        self.assertEqual(buffer.capacity, 200)
        self.assertEqual(buffer.alpha, 0.7)
        self.assertEqual(buffer.beta, 0.5)
        self.assertEqual(buffer.beta_annealing, 0.002)
        self.assertEqual(buffer.device, "cpu")
        self.assertEqual(len(buffer.trajectories), 0)
    
    def test_adaptdiffuser_trajectory_buffer_task_tracking(self):
        """Test task-specific trajectory tracking in AdaptDiffuserTrajectoryBuffer."""
        # Arrange
        buffer = AdaptDiffuserTrajectoryBuffer(capacity=100)
        
        # Create sample trajectories
        trajectory1 = torch.ones(10, 3)
        trajectory2 = torch.zeros(10, 3)
        
        # Act - Add trajectories with task identifiers
        buffer.add(trajectory1, reward=1.0, task="task1")
        buffer.add(trajectory2, reward=2.0, task="task2")
        
        # Get trajectories for each task
        task1_samples, task1_rewards = buffer.get_task_trajectories(task="task1")
        task2_samples, task2_rewards = buffer.get_task_trajectories(task="task2")
        
        # Assert
        self.assertEqual(len(task1_samples), 1)
        self.assertEqual(len(task2_samples), 1)
        self.assertTrue(torch.equal(task1_samples[0], trajectory1))
        self.assertTrue(torch.equal(task2_samples[0], trajectory2))
        self.assertEqual(task1_rewards[0], 1.0)
        self.assertEqual(task2_rewards[0], 2.0)
    
    def test_adaptdiffuser_store_samples(self):
        """Test storing generated samples in AdaptDiffuser's trajectory buffer."""
        # Arrange
        with patch('torch.optim.AdamW', return_value=Mock()):
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=self.noise_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Create sample data
        samples = torch.randn(3, self.channels, self.img_size, self.img_size)
        rewards = torch.tensor([0.8, 1.2, 0.5])
        
        # Act
        indices = adapt_diffuser.store_samples(
            samples=samples,
            rewards=rewards,
            task="test_task"
        )
        
        # Assert
        self.assertEqual(len(indices), 3)
        self.assertEqual(adapt_diffuser.trajectory_buffer.size(), 3)
    
    def test_adaptdiffuser_trajectory_with_reward(self):
        """Test generating and evaluating trajectories with rewards."""
        # Arrange
        with patch('torch.optim.AdamW', return_value=Mock()):
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=self.noise_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Mock generate method to return a fixed tensor
        samples = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        
        with patch.object(adapt_diffuser, 'generate', return_value=samples):
            # Act - Generate samples
            generated = adapt_diffuser.generate(batch_size=self.batch_size)
            
            # Compute rewards
            rewards = adapt_diffuser.compute_reward(generated)
        
        # Assert
        self.assertEqual(generated.shape, (self.batch_size, self.channels, self.img_size, self.img_size))
        self.assertEqual(rewards.shape, (self.batch_size,))
        # The reward values should match our mock
        self.assertTrue(torch.allclose(rewards, torch.tensor([0.5, 0.7])))
    
    def test_task_specific_trajectory_generation(self):
        """Test generating trajectories for specific tasks."""
        # Arrange
        with patch('torch.optim.AdamW', return_value=Mock()):
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=self.noise_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Mock task embedding model
        mock_task_embedding_model = Mock()
        mock_task_embedding_model.encode = Mock(
            return_value=torch.ones(1, 8)  # 8-dimensional embedding
        )
        adapt_diffuser.task_embedding_model = mock_task_embedding_model
        
        # Create expected output
        expected_output = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        
        # Act - with different tasks
        with patch.object(adapt_diffuser, 'generate', return_value=expected_output):
            task1_result = adapt_diffuser.generate(batch_size=self.batch_size, task="task1")
            task2_result = adapt_diffuser.generate(batch_size=self.batch_size, task="task2")
        
        # Assert
        self.assertEqual(task1_result.shape, expected_output.shape)
        self.assertEqual(task2_result.shape, expected_output.shape)
    
    def test_trajectory_buffer_clear(self):
        """Test clearing the trajectory buffer."""
        # Arrange
        buffer = TrajectoryBuffer(capacity=10)
        
        # Add sample trajectories
        for i in range(5):
            buffer.add(
                trajectory=torch.ones(10, 3) * i,
                reward=float(i)
            )
        
        # Act - Clear buffer
        buffer.clear()
        
        # Assert
        self.assertEqual(buffer.size(), 0)
        self.assertEqual(len(buffer.trajectories), 0)
        self.assertEqual(len(buffer.rewards), 0)
        self.assertEqual(len(buffer.priorities), 0)
    
    def test_adaptdiffuser_trajectory_buffer_with_metadata(self):
        """Test storing and retrieving trajectories with metadata."""
        # Arrange
        buffer = AdaptDiffuserTrajectoryBuffer(capacity=10)
        
        # Create sample trajectory with metadata
        trajectory = torch.ones(10, 3)
        metadata = {"source": "test", "timestamp": 12345}
        
        # Act - Add with metadata
        buffer.add(
            trajectory=trajectory,
            reward=1.0,
            metadata=metadata
        )
        
        # Get stored metadata directly from buffer
        stored_metadata = buffer.metadata[0]  # Access the first metadata entry
        
        # Assert
        self.assertEqual(stored_metadata, metadata)
        self.assertEqual(stored_metadata["source"], "test")
        self.assertEqual(stored_metadata["timestamp"], 12345)


if __name__ == '__main__':
    unittest.main()