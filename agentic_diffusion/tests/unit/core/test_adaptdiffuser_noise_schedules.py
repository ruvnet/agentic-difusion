import unittest
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.noise_schedules import (
    NoiseScheduler,
    LinearScheduler,
    CosineScheduler,
    SigmoidScheduler
)


class TestAdaptDiffuserNoiseSchedules(unittest.TestCase):
    """
    Unit tests for noise schedules and their integration with AdaptDiffuser.
    
    These tests verify that noise schedules are properly integrated with AdaptDiffuser
    and that they behave as expected under different configurations.
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
        
        # Create a mock reward model
        self.mock_reward_model = Mock()
        self.mock_reward_model.compute_reward = Mock(return_value=torch.tensor([0.5, 0.7]))

    def test_linear_scheduler_integration(self):
        """Test integration of LinearScheduler with AdaptDiffuser."""
        # Arrange
        linear_scheduler = LinearScheduler(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        with patch('torch.optim.AdamW', return_value=Mock()):
            # Create AdaptDiffuser with linear scheduler
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=linear_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Assert
        self.assertEqual(adapt_diffuser.noise_scheduler, linear_scheduler)
        self.assertEqual(adapt_diffuser.noise_scheduler.num_timesteps, 100)
        self.assertIsInstance(adapt_diffuser.noise_scheduler, LinearScheduler)
        
        # Verify betas are correctly set up
        self.assertEqual(linear_scheduler.betas.shape, (100,))
        self.assertAlmostEqual(linear_scheduler.betas[0].item(), 0.0001, delta=1e-5)
        self.assertAlmostEqual(linear_scheduler.betas[-1].item(), 0.02, delta=1e-5)

    def test_cosine_scheduler_integration(self):
        """Test integration of CosineScheduler with AdaptDiffuser."""
        # Arrange
        cosine_scheduler = CosineScheduler(
            num_timesteps=100,
            s=0.008
        )
        
        with patch('torch.optim.AdamW', return_value=Mock()):
            # Create AdaptDiffuser with cosine scheduler
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=cosine_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Assert
        self.assertEqual(adapt_diffuser.noise_scheduler, cosine_scheduler)
        self.assertEqual(adapt_diffuser.noise_scheduler.num_timesteps, 100)
        self.assertIsInstance(adapt_diffuser.noise_scheduler, CosineScheduler)
        
        # Verify betas have proper shape and values are within valid range
        self.assertEqual(cosine_scheduler.betas.shape, (100,))
        self.assertTrue(torch.all(cosine_scheduler.betas >= 0))
        self.assertTrue(torch.all(cosine_scheduler.betas <= 1))

    def test_sigmoid_scheduler_integration(self):
        """Test integration of SigmoidScheduler with AdaptDiffuser."""
        # Arrange
        sigmoid_scheduler = SigmoidScheduler(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        with patch('torch.optim.AdamW', return_value=Mock()):
            # Create AdaptDiffuser with sigmoid scheduler
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=sigmoid_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Assert
        self.assertEqual(adapt_diffuser.noise_scheduler, sigmoid_scheduler)
        self.assertEqual(adapt_diffuser.noise_scheduler.num_timesteps, 100)
        self.assertIsInstance(adapt_diffuser.noise_scheduler, SigmoidScheduler)
        
        # Verify betas have proper shape and values
        self.assertEqual(sigmoid_scheduler.betas.shape, (100,))
        self.assertTrue(torch.all(sigmoid_scheduler.betas >= 0))
        self.assertTrue(torch.all(sigmoid_scheduler.betas <= 1))

    def test_forward_process_multiple_schedules(self):
        """Test forward diffusion process with multiple scheduler types."""
        # Create schedulers
        linear_scheduler = LinearScheduler(num_timesteps=100)
        cosine_scheduler = CosineScheduler(num_timesteps=100)
        sigmoid_scheduler = SigmoidScheduler(num_timesteps=100)
        
        schedulers = [linear_scheduler, cosine_scheduler, sigmoid_scheduler]
        
        # Test sample data
        x_0 = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        t = torch.tensor([50] * self.batch_size)
        
        for scheduler in schedulers:
            # Create AdaptDiffuser with scheduler
            with patch('torch.optim.AdamW', return_value=Mock()):
                adapt_diffuser = AdaptDiffuser(
                    base_model=self.mock_model,
                    noise_scheduler=scheduler,
                    img_size=self.img_size,
                    channels=self.channels,
                    reward_model=self.mock_reward_model
                )
            
            # Apply forward process using scheduler's q_sample
            noise = torch.randn_like(x_0)
            x_t = scheduler.q_sample(x_0, t, noise)
            
            # Check output shape and type
            self.assertEqual(x_t.shape, x_0.shape)
            self.assertIsInstance(x_t, torch.Tensor)
            
            # Verify that the output is not the same as the input (noise was added)
            self.assertFalse(torch.allclose(x_t, x_0))

    def test_noise_scheduler_settings_propagation(self):
        """Test that noise scheduler settings are properly propagated."""
        # Arrange - Create scheduler with specific settings
        num_timesteps = 200
        beta_start = 0.00025
        beta_end = 0.015
        
        linear_scheduler = LinearScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        with patch('torch.optim.AdamW', return_value=Mock()):
            # Create AdaptDiffuser with this scheduler
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=linear_scheduler,
                img_size=self.img_size,
                channels=self.channels,
                reward_model=self.mock_reward_model
            )
        
        # Assert settings were properly propagated
        self.assertEqual(adapt_diffuser.noise_scheduler.num_timesteps, num_timesteps)
        self.assertEqual(adapt_diffuser.noise_scheduler.beta_start, beta_start)
        self.assertEqual(adapt_diffuser.noise_scheduler.beta_end, beta_end)

    def test_edge_cases_noise_scheduler(self):
        """Test edge cases and special scenarios with noise schedulers."""
        # Test with minimum number of timesteps
        min_timesteps = 2
        linear_scheduler_min = LinearScheduler(num_timesteps=min_timesteps)
        
        # Verify scheduler properties
        self.assertEqual(linear_scheduler_min.num_timesteps, min_timesteps)
        self.assertEqual(linear_scheduler_min.betas.shape, (min_timesteps,))
        
        # Test with very small beta values
        tiny_beta = 1e-10
        linear_scheduler_tiny = LinearScheduler(
            num_timesteps=100,
            beta_start=tiny_beta,
            beta_end=tiny_beta * 10
        )
        
        # Verify beta values
        self.assertAlmostEqual(linear_scheduler_tiny.betas[0].item(), tiny_beta, delta=1e-11)
        
        # Test with identical beta_start and beta_end
        beta_value = 0.001
        linear_scheduler_same = LinearScheduler(
            num_timesteps=100, 
            beta_start=beta_value,
            beta_end=beta_value
        )
        
        # Verify all betas are the same
        self.assertTrue(torch.allclose(
            linear_scheduler_same.betas,
            torch.ones_like(linear_scheduler_same.betas) * beta_value
        ))

    def test_scheduler_derived_properties(self):
        """Test derived properties of noise schedulers."""
        # Create scheduler
        num_timesteps = 100
        linear_scheduler = LinearScheduler(num_timesteps=num_timesteps)
        
        # Check that alphas, alphas_cumprod, etc. are properly derived
        self.assertEqual(linear_scheduler.betas.shape, (num_timesteps,))
        self.assertEqual(linear_scheduler.alphas.shape, (num_timesteps,))
        self.assertEqual(linear_scheduler.alphas_cumprod.shape, (num_timesteps,))
        self.assertEqual(linear_scheduler.alphas_cumprod_prev.shape, (num_timesteps,))
        
        # Check specific properties
        self.assertTrue(torch.all(linear_scheduler.alphas == 1 - linear_scheduler.betas))
        self.assertTrue(torch.all(linear_scheduler.alphas <= 1))
        self.assertTrue(torch.all(linear_scheduler.alphas >= 0))
        
        # Verify alphas_cumprod is correctly calculated as cumprod of alphas
        alphas_cumprod_manual = torch.cumprod(linear_scheduler.alphas, dim=0)
        self.assertTrue(torch.allclose(linear_scheduler.alphas_cumprod, alphas_cumprod_manual))


if __name__ == '__main__':
    unittest.main()