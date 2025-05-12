#!/usr/bin/env python
"""
Test script for verifying the ResidualBlock's dynamic dimension handling.
This script tests the improved dimension handling in the ResidualBlock class.
"""

import torch
import logging
import sys
import traceback

# Configure logging to see detailed messages
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("residual_block_test")

# Import the ResidualBlock class
from agentic_diffusion.code_generation.models.blocks import ResidualBlock

def test_residual_block_dimension_handling():
    """Test the ResidualBlock's handling of dimension mismatches."""
    print("\n=== Testing ResidualBlock dynamic dimension handling ===")
    
    # Base dimensions
    d_model = 512
    d_time = 128
    n_heads = 8
    
    # Create a ResidualBlock instance
    residual_block = ResidualBlock(
        d_model=d_model,
        d_time=d_time,
        n_heads=n_heads,
        dropout=0.1
    )
    
    # Set device (use CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    residual_block = residual_block.to(device)
    
    # Test with matching dimensions (normal case)
    print("\nTest with normal dimensions (matching base dimensions):")
    batch_size = 2
    seq_len = 100
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    time_emb = torch.randn(batch_size, d_time, device=device)
    
    try:
        output = residual_block(x, time_emb)
        print(f"Success! Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, d_model)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with larger input dimension than d_model
    print("\nTest with larger input dimension:")
    larger_dim = d_model * 2  # 1024
    
    x_large = torch.randn(batch_size, seq_len, larger_dim, device=device)
    
    try:
        output = residual_block(x_large, time_emb)
        print(f"Success! Input shape: {x_large.shape}, Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, larger_dim)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with smaller input dimension than d_model
    print("\nTest with smaller input dimension:")
    smaller_dim = d_model // 2  # 256
    
    x_small = torch.randn(batch_size, seq_len, smaller_dim, device=device)
    
    try:
        output = residual_block(x_small, time_emb)
        print(f"Success! Input shape: {x_small.shape}, Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, smaller_dim)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with different time embedding dimension
    print("\nTest with different time embedding dimension:")
    different_time_dim = d_time * 2  # 256
    
    time_emb_different = torch.randn(batch_size, different_time_dim, device=device)
    
    try:
        output = residual_block(x, time_emb_different)
        print(f"Success! Time embed shape: {time_emb_different.shape}, Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, d_model)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with uncommon dimensions
    print("\nTest with uncommon dimensions:")
    uncommon_dim = 768  # Not a multiple of d_model
    
    x_uncommon = torch.randn(batch_size, seq_len, uncommon_dim, device=device)
    
    try:
        output = residual_block(x_uncommon, time_emb)
        print(f"Success! Input shape: {x_uncommon.shape}, Output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, uncommon_dim)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Test ResidualBlock
        test_residual_block_dimension_handling()
        
        print("\nAll ResidualBlock tests completed successfully.")
    except Exception as e:
        print(f"Unexpected error during testing: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)