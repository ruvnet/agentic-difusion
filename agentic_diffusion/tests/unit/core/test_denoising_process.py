"""
Tests for the DenoisingDiffusionProcess class in the Agentic Diffusion system.

These tests verify the DenoisingDiffusionProcess class correctly implements both
forward and reverse diffusion processes.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from agentic_diffusion.core.denoising_process import (
    DenoisingDiffusionProcess,
    DDPMSampler,
    DDIMSampler
)
from agentic_diffusion.core.noise_schedules import LinearScheduler
from agentic_diffusion.core.diffusion_model import DiffusionModel


class MockModel(DiffusionModel):
    """Mock diffusion model for testing."""
    
    def __init__(self):
        super().__init__()
        self.hidden_dim = 64
    
    def forward(self, x, t, **kwargs):
        """Mock forward method that returns noise of same shape as input."""
        return torch.randn_like(x)
    
    def sample(self, shape, **kwargs):
        """Mock sample method that returns tensor of specified shape."""
        return torch.randn(shape)


class TestDenoisingDiffusionProcess:
    """Tests for the DenoisingDiffusionProcess class."""
    
    def test_initialization(self):
        """Test proper initialization of the DenoisingDiffusionProcess."""
        # Arrange
        model = MockModel()
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        
        # Act
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Assert
        assert process.model == model
        assert process.noise_scheduler == noise_scheduler
        assert process.img_size == 32
        assert process.channels == 3
        assert process.device == "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_forward_process(self):
        """Test q(x_t | x_0) forward process."""
        # Arrange
        model = MockModel()
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Create clean image batch
        x_0 = torch.randn(2, 3, 32, 32)
        
        # Act - Apply forward process at different timesteps
        t_early = torch.tensor([100, 100])
        t_mid = torch.tensor([500, 500])
        t_late = torch.tensor([900, 900])
        
        # Generate noisy images at different timesteps with fixed noise for deterministic testing
        noise = torch.randn_like(x_0)
        x_early = process.forward_process(x_0, t_early, noise)
        x_mid = process.forward_process(x_0, t_mid, noise)
        x_late = process.forward_process(x_0, t_late, noise)
        
        # Assert
        # Verify shape is preserved
        assert x_early.shape == x_0.shape
        assert x_mid.shape == x_0.shape
        assert x_late.shape == x_0.shape
        
        # Verify noise levels increase with timestep
        # Calculate mean squared difference from original
        diff_early = torch.mean((x_early - x_0) ** 2)
        diff_mid = torch.mean((x_mid - x_0) ** 2)
        diff_late = torch.mean((x_late - x_0) ** 2)
        
        # Later timesteps should have more noise (larger difference from original)
        assert diff_early < diff_mid < diff_late
    
    def test_reverse_process_single_step(self):
        """Test p(x_{t-1} | x_t) reverse process for a single denoising step."""
        # Arrange
        model = Mock()
        # Mock the noise prediction to just return zero noise (makes testing simpler)
        model.forward = Mock(return_value=torch.zeros(2, 3, 32, 32))
        
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Create a noisy image batch at timestep t
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([500, 500])
        
        # Act - Apply single reverse process step
        with torch.no_grad():
            x_t_minus_1 = process.reverse_process_step(x_t, t)
        
        # Assert
        # Verify shape is preserved
        assert x_t_minus_1.shape == x_t.shape
        
        # Model should have been called with the correct arguments
        model.forward.assert_called_once()
        args, kwargs = model.forward.call_args
        assert torch.all(torch.eq(args[0], x_t))
        assert torch.all(torch.eq(args[1], t))
    
    def test_sample_loop(self):
        """Test the sampling loop for generating images."""
        # Arrange
        model = Mock()
        # Mock the model to return predictable noise
        model.forward.return_value = torch.zeros(2, 3, 32, 32)  # Predict zero noise
        
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Act - Sample images
        samples = process.sample(batch_size=2)
        
        # Assert
        # Verify output shape
        assert samples.shape == (2, 3, 32, 32)
        
        # Model should have been called multiple times during sampling
        assert model.forward.call_count > 0
    
    def test_p_mean_variance(self):
        """Test computation of posterior mean and variance."""
        # Arrange
        model = Mock()
        # Mock the model to return predictable noise
        model.forward.return_value = torch.zeros(2, 3, 32, 32)  # Predict zero noise
        
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Create a noisy image batch at timestep t
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([500, 500])
        
        # Act - Calculate posterior mean and variance
        mean, variance, log_variance = process.p_mean_variance(x_t, t)
        
        # Assert
        # Verify shape
        assert mean.shape == x_t.shape
        assert variance.shape == (2, 1, 1, 1)  # Broadcasting dimensions
        assert log_variance.shape == (2, 1, 1, 1)
        
        # Model should have been called correctly
        model.forward.assert_called_once()
        args, kwargs = model.forward.call_args
        assert torch.all(torch.eq(args[0], x_t))
        assert torch.all(torch.eq(args[1], t))
    
    def test_training_step(self):
        """Test training step that computes the diffusion loss."""
        # Arrange
        model = Mock()
        # Mock the model to return a fixed noise prediction
        fixed_noise = torch.randn(2, 3, 32, 32) * 0.1  # Small noise prediction for testing
        model.forward.return_value = fixed_noise
        
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Create training batch
        x_0 = torch.randn(2, 3, 32, 32)
        
        # Act - Apply training step
        with patch.object(torch, "randint", return_value=torch.tensor([500, 500])):
            with patch.object(torch, "randn_like", return_value=torch.ones_like(x_0)):
                loss, metrics = process.training_step(x_0)
        
        # Assert
        # Verify loss is calculated
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() > 0  # Should be positive since prediction doesn't match noise
        
        # Metrics should be dictionary with SNR values
        assert isinstance(metrics, dict)
        assert "mean_snr" in metrics
    
    def test_conditioning(self):
        """Test that conditioning information is properly passed to the model."""
        # Arrange
        model = Mock()
        model.forward.return_value = torch.zeros(2, 3, 32, 32)  # Predict zero noise
        
        noise_scheduler = LinearScheduler(num_timesteps=1000)
        process = DenoisingDiffusionProcess(
            model=model,
            noise_scheduler=noise_scheduler,
            img_size=32,
            channels=3
        )
        
        # Create noisy image and conditioning information
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([500, 500])
        conditioning = torch.randn(2, 64)  # Mock conditioning embedding
        
        # Act - Call reverse process with conditioning
        process.reverse_process_step(x_t, t, conditioning=conditioning)
        
        # Assert
        # Model should have been called with conditioning in kwargs
        model.forward.assert_called_once()
        args, kwargs = model.forward.call_args
        assert "conditioning" in kwargs
        assert torch.all(torch.eq(kwargs["conditioning"], conditioning))


class TestDDPMSampler:
    """Tests for the DDPMSampler class."""
    
    def test_initialization(self):
        """Test proper initialization of the DDPM sampler."""
        # Arrange
        process = Mock()
        
        # Act
        sampler = DDPMSampler(
            process=process,
            num_timesteps=1000
        )
        
        # Assert
        assert sampler.process == process
        assert sampler.num_timesteps == 1000
    
    def test_sample(self):
        """Test sampling using the DDPM algorithm."""
        # Arrange
        process = Mock()
        process.img_size = 32
        process.channels = 3
        process.device = "cpu"
        
        # Create a mock sample method for the process
        mock_samples = torch.randn(2, 3, 32, 32)
        process.sample.return_value = mock_samples
        
        sampler = DDPMSampler(
            process=process,
            num_timesteps=10  # Smaller steps for testing
        )
        
        # Act
        samples = sampler.sample(batch_size=2)
        
        # Assert
        # Verify process.sample was called correctly
        process.sample.assert_called_once()
        # Verify the batch_size was passed correctly
        args, kwargs = process.sample.call_args
        assert kwargs["batch_size"] == 2
        assert kwargs["num_steps"] == 10
        # Verify sample shape
        assert samples.shape == (2, 3, 32, 32)


class TestDDIMSampler:
    """Tests for the DDIMSampler class."""
    
    def test_initialization(self):
        """Test proper initialization of the DDIM sampler."""
        # Arrange
        process = Mock()
        
        # Act
        sampler = DDIMSampler(
            process=process,
            num_timesteps=1000,
            eta=0.0  # Deterministic sampling
        )
        
        # Assert
        assert sampler.process == process
        assert sampler.num_timesteps == 1000
        assert sampler.eta == 0.0
    
    def test_sample(self):
        """Test sampling using the DDIM algorithm."""
        # Arrange
        process = Mock()
        process.img_size = 32
        process.channels = 3
        process.device = "cpu"
        
        # Mock methods needed for DDIM sampling
        process.model = Mock()
        process.model.forward.return_value = torch.zeros(2, 3, 32, 32)
        
        process.noise_scheduler = Mock()
        process.noise_scheduler.alphas_cumprod = torch.linspace(0.99, 0.01, 1000)
        
        # Mock needed process methods
        def mock_p_mean_variance(x, t, **kwargs):
            return x, torch.ones((2, 1, 1, 1)) * 0.1, torch.ones((2, 1, 1, 1)) * -2.3
            
        process.p_mean_variance.side_effect = mock_p_mean_variance
        
        sampler = DDIMSampler(
            process=process,
            num_timesteps=10,  # Smaller steps for testing
            eta=0.0  # Deterministic sampling
        )
        
        # Act
        samples = sampler.sample(batch_size=2)
        
        # Assert
        # Verify correct methods were called
        assert process.p_mean_variance.call_count > 0
        
        # Verify sample shape
        assert samples.shape == (2, 3, 32, 32)
    
    def test_eta_effect(self):
        """Test that different eta values produce different samples."""
        # This test verifies that stochastic sampling (eta > 0)
        # differs from deterministic sampling (eta = 0)
        
        # Arrange
        process = Mock()
        process.img_size = 32
        process.channels = 3
        process.device = "cpu"
        
        # Set up a reproducible random seed
        torch.manual_seed(42)
        
        # Mock methods needed for DDIM sampling - use real noise
        process.model = Mock()
        process.model.forward.return_value = torch.randn(2, 3, 32, 32) * 0.1
        
        process.noise_scheduler = Mock()
        process.noise_scheduler.alphas_cumprod = torch.linspace(0.99, 0.01, 1000)
        
        # Mock needed process methods
        def mock_p_mean_variance(x, t, **kwargs):
            return x, torch.ones((2, 1, 1, 1)) * 0.1, torch.ones((2, 1, 1, 1)) * -2.3
            
        process.p_mean_variance.side_effect = mock_p_mean_variance
        
        # Create two samplers with different eta values
        deterministic_sampler = DDIMSampler(
            process=process,
            num_timesteps=10,
            eta=0.0  # Deterministic
        )
        
        stochastic_sampler = DDIMSampler(
            process=process,
            num_timesteps=10,
            eta=0.8  # Stochastic
        )
        
        # Act - sample with same seed but different eta
        torch.manual_seed(42)
        deterministic_samples = deterministic_sampler.sample(batch_size=2)
        
        torch.manual_seed(42)
        stochastic_samples = stochastic_sampler.sample(batch_size=2)
        
        # Assert - samples should be different
        assert not torch.allclose(deterministic_samples, stochastic_samples)