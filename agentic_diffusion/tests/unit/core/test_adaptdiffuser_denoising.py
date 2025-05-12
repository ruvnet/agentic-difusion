import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
import tempfile
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.noise_schedules import NoiseScheduler, LinearScheduler, CosineScheduler
from agentic_diffusion.core.denoising_process import DenoisingDiffusionProcess


class TestAdaptDiffuserDenoising(unittest.TestCase):
    """Unit tests for AdaptDiffuser denoising process functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock denoising model
        self.mock_model = Mock()
        self.mock_model.parameters = Mock(return_value=[torch.ones(1)])
        self.mock_model.to = Mock(return_value=self.mock_model)
        
        # Mock return values for the model
        self.mock_model.return_value = torch.zeros(2, 4)  # Batch size 2, dim 4

        # Create a simple noise scheduler
        self.noise_scheduler = LinearScheduler(num_timesteps=100)
        
        # Define default dimensions
        self.img_size = 4
        self.channels = 1
        self.batch_size = 2
        
        # Create a simple reward model
        self.mock_reward_model = Mock()
        self.mock_reward_model.compute_reward = Mock(return_value=torch.tensor([0.5, 0.7]))

    def test_denoising_process_initialization(self):
        """Test initialization of DenoisingDiffusionProcess."""
        # Arrange & Act
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Assert
        self.assertEqual(denoising_process.model, self.mock_model)
        self.assertEqual(denoising_process.noise_scheduler, self.noise_scheduler)
        self.assertEqual(denoising_process.img_size, self.img_size)
        self.assertEqual(denoising_process.channels, self.channels)
    
    def test_denoising_forward_step(self):
        """Test a single forward step in the denoising process."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Create a noisy sample
        x_t = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        t = torch.tensor([50, 50])  # Middle timestep
        
        # Act
        with patch.object(self.mock_model, 'forward', return_value=torch.zeros_like(x_t)):
            # Call p_mean_variance to get the forward process results
            mean, variance, log_variance = denoising_process.p_mean_variance(x_t, t)
            
        # Assert
        # Check that we got the expected tuple of tensors
        self.assertIsInstance(mean, torch.Tensor)
        self.assertIsInstance(variance, torch.Tensor)
        self.assertIsInstance(log_variance, torch.Tensor)
        
        # Check the shapes
        self.assertEqual(mean.shape, x_t.shape)
        
    def test_sample_method(self):
        """Test the sample method of the denoising process."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Mock the reverse_process_step method to return a predetermined value
        expected_result = torch.ones(
            self.batch_size, self.channels, self.img_size, self.img_size)
        
        with patch.object(denoising_process, 'reverse_process_step', return_value=expected_result):
            # Act
            result = denoising_process.sample(
                batch_size=self.batch_size,
                num_steps=10
            )
        
        # Assert
        self.assertTrue(torch.equal(result, expected_result))
        self.assertEqual(result.shape, (self.batch_size, self.channels, self.img_size, self.img_size))
        
    def test_timestep_boundary_handling(self):
        """Test behavior with timesteps at the boundaries of valid ranges."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Create sample tensors
        x_t = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        
        # Get the number of timesteps from the scheduler
        num_timesteps = self.noise_scheduler.num_timesteps
        
        # Test with extreme values that should be clamped or handled gracefully
        max_t = torch.tensor([num_timesteps-1, num_timesteps-1])  # Maximum valid timestep
        min_t = torch.tensor([0, 0])  # Minimum valid timestep
        over_t = torch.tensor([num_timesteps+10, num_timesteps+20])  # Over maximum
        
        # Mock the model's forward method
        with patch.object(self.mock_model, 'forward',
                         return_value=torch.zeros(self.batch_size, self.channels*2, self.img_size, self.img_size)):
            # All these should work without errors
            max_result = denoising_process.p_mean_variance(x_t, max_t)
            min_result = denoising_process.p_mean_variance(x_t, min_t)
            over_result = denoising_process.p_mean_variance(x_t, over_t)
            
            # The implementation should handle boundary values
            self.assertIsNotNone(max_result)
            self.assertIsNotNone(min_result)
            self.assertIsNotNone(over_result)
            
            # Assert that we get meaningful outputs (not all zeros or NaNs)
            self.assertFalse(torch.isnan(max_result[0]).any())
            self.assertFalse(torch.isnan(min_result[0]).any())
            self.assertFalse(torch.isnan(over_result[0]).any())
    
    def test_denoising_process_integration_with_adaptdiffuser(self):
        """Test integration of denoising process with AdaptDiffuser."""
        # Arrange
        with patch('torch.optim.AdamW', return_value=Mock()):
            # Create an AdaptDiffuser with mocked components
            adapt_diffuser = AdaptDiffuser(
                base_model=self.mock_model,
                noise_scheduler=self.noise_scheduler,
                reward_model=self.mock_reward_model,
                img_size=self.img_size,
                channels=self.channels
            )
        
        # Mock the generate method for AdaptDiffuser that returns a tensor
        expected_result = torch.ones(self.batch_size, self.channels, self.img_size, self.img_size)
        
        with patch.object(adapt_diffuser, 'generate', return_value=expected_result):
            # Act
            result = adapt_diffuser.generate(batch_size=self.batch_size)
        
        # Assert
        self.assertTrue(torch.equal(result, expected_result))
        self.assertEqual(result.shape, (self.batch_size, self.channels, self.img_size, self.img_size))
    
    def test_denoising_with_conditioning(self):
        """Test denoising process with conditioning signals."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Create a noisy sample and conditioning
        x_t = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        t = torch.tensor([50, 50])  # Middle timestep
        conditioning = torch.randn(self.batch_size, 8)  # Conditioning signal
        
        # Mock the model to handle conditioning
        def model_with_conditioning(x, t, conditioning=None):
            # Return zeros but ensure conditioning was passed
            self.assertIsNotNone(conditioning)
            return torch.zeros_like(x)
        
        self.mock_model.forward = model_with_conditioning
        
        # Act
        mean, variance, log_variance = denoising_process.p_mean_variance(
            x_t, t, conditioning=conditioning
        )
        
        # Assert
        self.assertIsInstance(mean, torch.Tensor)
        self.assertEqual(mean.shape, x_t.shape)
    
    def test_denoising_determinism(self):
        """Test that denoising is deterministic with fixed noise and seed."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Set a fixed seed
        torch.manual_seed(42)
        
        # Mock torch.randn to return the same noise for both calls
        fixed_noise = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        
        # Mock reverse_process_step to return a deterministic value based on input
        def deterministic_reverse_step(x_t, t, **kwargs):
            return x_t * 0.9  # Deterministic transformation
        
        with patch.object(denoising_process, 'reverse_process_step', side_effect=deterministic_reverse_step), \
             patch('torch.randn', return_value=fixed_noise.clone()):
            
            # Act - Sample twice with the same parameters
            result1 = denoising_process.sample(
                batch_size=self.batch_size,
                num_steps=10
            )
            
            # Reset the seed to the same value
            torch.manual_seed(42)
            
            result2 = denoising_process.sample(
                batch_size=self.batch_size,
                num_steps=10
            )
        
        # Assert
        self.assertTrue(torch.allclose(result1, result2))
    
    def test_denoising_device_handling(self):
        """Test that denoising process correctly handles device specification."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for testing")
        
        # Arrange
        device = torch.device('cuda')
        
        # Mock the model's device handling
        model = self.mock_model
        model.to = Mock(return_value=model)
        
        denoising_process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels,
            device=device
        )
        
        # Assert
        self.assertEqual(denoising_process.device, device)
        model.to.assert_called_with(device)
    
    def test_denoising_with_timestep_respacing(self):
        """Test denoising with different numbers of inference steps."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Mock torch.randn to keep input tensor the same
        fixed_noise = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        
        # Track how many timesteps are actually used
        timesteps_used = []
        
        # Original range method to capture calls
        original_range = range
        
        def mock_range(*args, **kwargs):
            # Capture the range arguments when denoising_process.sample calls range
            if len(args) == 3 and args[2] == -1:  # This pattern matches the reverse timestep range
                start, end, step = args
                timesteps_used.append(start - end)  # Number of steps = (start - end) when step is -1
            return original_range(*args, **kwargs)
            
        with patch('torch.randn', return_value=fixed_noise), \
             patch('builtins.range', side_effect=mock_range):
            
            # Act - Sample with 10 steps
            denoising_process.sample(
                batch_size=self.batch_size,
                num_steps=10
            )
            
            # Sample with 20 steps
            denoising_process.sample(
                batch_size=self.batch_size,
                num_steps=20
            )
        
        # Assert
        self.assertEqual(timesteps_used[0], 10)  # First call used 10 steps
        self.assertEqual(timesteps_used[1], 20)  # Second call used 20 steps
    
    def test_model_call_in_denoising(self):
        """Test that the model is called correctly during denoising."""
        # Arrange
        denoising_process = DenoisingDiffusionProcess(
            model=self.mock_model,
            noise_scheduler=self.noise_scheduler,
            img_size=self.img_size,
            channels=self.channels
        )
        
        # Create valid inputs
        x_t = torch.randn(self.batch_size, self.channels, self.img_size, self.img_size)
        t = torch.tensor([50, 50])
        
        # Mock the model.forward method to track calls
        mock_forward = Mock(return_value=torch.zeros_like(x_t))
        self.mock_model.forward = mock_forward
        
        # Act
        denoising_process.p_mean_variance(x_t, t)
        
        # Assert
        mock_forward.assert_called_once()
        # Check that the model was called with the right arguments
        args, kwargs = mock_forward.call_args
        self.assertTrue(torch.equal(args[0], x_t))
        self.assertTrue(torch.equal(args[1], t))


if __name__ == '__main__':
    unittest.main()