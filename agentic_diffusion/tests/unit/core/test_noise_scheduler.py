import pytest
import torch
import numpy as np
from agentic_diffusion.core.noise_schedules import LinearScheduler, CosineScheduler, SigmoidScheduler

class TestNoiseScheduler:
    """Test suite for the NoiseScheduler class."""
    
    def test_initialization_linear(self):
        """Test that LinearScheduler initializes correctly."""
        # Arrange
        beta_start = 0.0001
        beta_end = 0.02
        num_timesteps = 1000
        
        # Act
        scheduler = LinearScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            num_timesteps=num_timesteps
        )
        
        # Assert
        assert scheduler.beta_start == beta_start
        assert scheduler.beta_end == beta_end
        assert scheduler.num_timesteps == num_timesteps
        assert scheduler.betas.shape == (num_timesteps,)
        assert scheduler.alphas.shape == (num_timesteps,)
        assert scheduler.alphas_cumprod.shape == (num_timesteps,)
        
        # Check that betas increase monotonically
        assert torch.all(scheduler.betas[1:] >= scheduler.betas[:-1])
        
        # Check that alphas decrease monotonically
        assert torch.all(scheduler.alphas[1:] <= scheduler.alphas[:-1])
        
        # Check that alphas_cumprod decrease monotonically
        assert torch.all(scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1])
        
        # Check ranges
        assert torch.all(scheduler.betas >= beta_start)
        assert torch.all(scheduler.betas <= beta_end)
        assert torch.all(scheduler.alphas <= 1.0)
        assert torch.all(scheduler.alphas >= 1.0 - beta_end)
        assert torch.all(scheduler.alphas_cumprod <= 1.0)
        assert torch.all(scheduler.alphas_cumprod >= 0.0)
    
    def test_initialization_cosine(self):
        """Test that CosineScheduler initializes correctly."""
        # Arrange
        s = 0.008  # Cosine scheduler uses an 's' parameter instead of beta values
        num_timesteps = 1000
        
        # Act
        scheduler = CosineScheduler(
            s=s,
            num_timesteps=num_timesteps
        )
        
        # Assert
        assert scheduler.s == s
        # Check that alphas_cumprod follows a cosine schedule
        # Verify scheduler has expected properties
        assert scheduler.num_timesteps == num_timesteps
        assert scheduler.betas.shape == (num_timesteps,)
        assert scheduler.alphas.shape == (num_timesteps,)
        assert scheduler.alphas_cumprod.shape == (num_timesteps,)
        
        # Check that values are in expected ranges
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
        assert torch.all(scheduler.alphas >= 0)
        assert torch.all(scheduler.alphas <= 1)
        assert torch.all(scheduler.alphas_cumprod >= 0)
        assert torch.all(scheduler.alphas_cumprod <= 1)
    
    def test_initialization_sigmoid(self):
        """Test that SigmoidScheduler initializes correctly."""
        # Arrange
        beta_start = 0.0001
        beta_end = 0.02
        sigmoid_scale = 10.0
        num_timesteps = 1000
        
        # Act
        scheduler = SigmoidScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            sigmoid_scale=sigmoid_scale,
            num_timesteps=num_timesteps
        )
        
        # Assert
        assert scheduler.beta_start == beta_start
        assert scheduler.beta_end == beta_end
        assert scheduler.sigmoid_scale == sigmoid_scale
        # Check that betas follow a sigmoid pattern (S-shaped)
        first_quarter = scheduler.betas[num_timesteps//4] - scheduler.betas[0]
        mid_section = scheduler.betas[3*num_timesteps//4] - scheduler.betas[num_timesteps//4]
        last_quarter = scheduler.betas[-1] - scheduler.betas[3*num_timesteps//4]
        
        # In a sigmoid schedule, the middle section changes faster than the ends
        assert mid_section > first_quarter
        assert mid_section > last_quarter
    
    def test_parameter_boundaries(self):
        """Test parameter boundaries and their effects."""
        # Test small beta_start
        scheduler_small_start = LinearScheduler(
            beta_start=1e-6,
            beta_end=0.02,
            num_timesteps=1000
        )
        assert scheduler_small_start.betas[0] < 0.0001
        
        # Test large beta_end
        scheduler_large_end = LinearScheduler(
            beta_start=0.0001,
            beta_end=0.1,
            num_timesteps=1000
        )
        assert scheduler_large_end.betas[-1] > 0.05
        
        # Test beta_end equal to beta_start - should still work but with flat schedule
        scheduler_equal = LinearScheduler(
            beta_start=0.01,
            beta_end=0.01,
            num_timesteps=1000
        )
        assert torch.allclose(scheduler_equal.betas, torch.full((1000,), 0.01))
    
    def test_add_noise(self):
        """Test adding noise to a sample using the scheduler."""
        # Arrange
        scheduler = LinearScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            num_timesteps=1000
        )
        
        batch_size = 2
        channels = 3
        height = width = 8
        x_start = torch.randn(batch_size, channels, height, width)
        
        # Test multiple timesteps to verify different noise levels
        for t in [0, 500, 999]:
            # Convert to tensor if not already
            time = torch.tensor([t, t])
            
            # Act
            noisy_x, noise = scheduler.add_noise(x_start, time)
            
            # Assert
            assert noisy_x.shape == x_start.shape
            assert noise.shape == x_start.shape
            assert torch.all(torch.isfinite(noisy_x))
            assert torch.all(torch.isfinite(noise))
            
            # For t > 0, ensure the noisy sample is different from the original
            if t > 0:
                # Compute the difference between x_start and noisy_x
                diff = (noisy_x - x_start).abs().mean().item()
                
                # The difference should increase with timestep
                if t == 500:
                    assert diff > 0.1  # Significant difference at t=500
                elif t == 999:
                    assert diff > 0.5  # Large difference at t=999
                elif t == 500:
                    # At t=500, should be mixed
                    assert noise_contribution > 0.01 * signal_power
                    assert signal_power > 0.01 * noise_contribution
    
    def test_reverse_step(self):
        """Test the reverse diffusion step (denoising)."""
        # Arrange
        scheduler = LinearScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            num_timesteps=1000
        )
        
        batch_size = 2
        channels = 3
        height = width = 8
        x_start = torch.randn(batch_size, channels, height, width)
        
        # First add noise to get a noisy sample
        t = torch.tensor([500, 500])
        noisy_x, noise = scheduler.add_noise(x_start, t)
        
        # Predict the noise (in real system this would be done by a model)
        predicted_noise = noise  # Perfect prediction for testing
        
        # Act
        # Perform a reverse step
        x_prev = scheduler.reverse_step(noisy_x, t, predicted_noise)
        
        # Assert
        assert x_prev.shape == x_start.shape
        assert torch.all(torch.isfinite(x_prev))
        
        # The denoised sample should be closer to x_start than the noisy sample
        noisy_error = torch.norm(noisy_x - x_start)
        denoised_error = torch.norm(x_prev - x_start)
        
        assert denoised_error < noisy_error
    
    def test_sample_timesteps(self):
        """Test sampling of timesteps."""
        # Arrange
        num_timesteps = 1000
        scheduler = LinearScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            num_timesteps=num_timesteps
        )
        
        batch_size = 100
        
        # Act
        timesteps = scheduler.sample_timesteps(batch_size)
        
        # Assert
        assert timesteps.shape == (batch_size,)
        assert torch.all(timesteps >= 0)
        assert torch.all(timesteps < num_timesteps)
        
        # Check that the distribution is roughly uniform
        # (divide range into bins and check roughly equal counts)
        bins = 10
        bin_size = num_timesteps // bins
        for i in range(bins):
            bin_count = ((timesteps >= i * bin_size) & (timesteps < (i + 1) * bin_size)).sum()
            # Allow some variation due to randomness
            assert bin_count > 0  # Should have at least some samples in each bin
    
    def test_sample_random_noise(self):
        """Test sampling of random noise."""
        # Arrange
        scheduler = LinearScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            num_timesteps=1000
        )
        
        batch_size = 2
        channels = 3
        height = width = 8
        shape = (batch_size, channels, height, width)
        
        # Act
        noise = scheduler.sample_random_noise(shape)
        
        # Assert
        assert noise.shape == shape
        assert torch.all(torch.isfinite(noise))
        
        # Check that noise follows a standard normal distribution
        mean = noise.mean().item()
        std = noise.std().item()
        
        assert abs(mean) < 0.1  # Mean should be close to 0
        assert abs(std - 1.0) < 0.1  # Std should be close to 1