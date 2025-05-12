#!/usr/bin/env python3
"""
Test script for the ResidualBlock fix.

This script creates tensors with various shapes and runs them through
the ResidualBlock to verify our dimension-adaptive fix works correctly.
"""

import torch
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import the ResidualBlock class
from agentic_diffusion.code_generation.models.blocks import ResidualBlock

def test_residual_block():
    """Test the ResidualBlock with different input shapes."""
    logger = logging.getLogger("test_residual_block")
    logger.info("Testing ResidualBlock with different tensor shapes")
    
    # Test parameters
    d_time = 512
    device = torch.device("cpu")
    
    # Create test tensors with different shapes
    batch_size = 4
    seq_len = 64
    dimensions = [256, 512, 768, 1024, 2048]
    
    # Create a time embedding tensor
    time_emb = torch.randn(batch_size, d_time).to(device)
    
    success = True
    # Test each dimension
    for d_model in dimensions:
        try:
            logger.info(f"Testing ResidualBlock with d_model={d_model}")
            
            # Create input tensor [batch_size, seq_len, d_model]
            x = torch.randn(batch_size, seq_len, d_model).to(device)
            
            # Create the ResidualBlock
            block = ResidualBlock(
                d_model=d_model,
                d_time=d_time,
                n_heads=8,
                dropout=0.1
            )
            
            # Run the forward pass
            output = block(x, time_emb)
            
            # Verify output shape
            expected_shape = (batch_size, seq_len, d_model)
            if output.shape == expected_shape:
                logger.info(f"✅ Output shape is correct: {output.shape}")
            else:
                logger.error(f"❌ Output shape mismatch: {output.shape} != {expected_shape}")
                success = False
                
        except Exception as e:
            logger.error(f"❌ Error testing dimension {d_model}: {e}")
            success = False
    
    # Test handling different input/time dimensions
    logger.info("Testing cross-dimensional handling...")
    
    # Create the ResidualBlock with d_model=512
    block = ResidualBlock(
        d_model=512,
        d_time=d_time,
        n_heads=8,
        dropout=0.1
    )
    
    # Create input tensor with dimension 1024
    x_1024 = torch.randn(batch_size, seq_len, 1024).to(device)
    
    try:
        # Run the forward pass
        output = block(x_1024, time_emb)
        
        # Verify output shape
        if output.shape == (batch_size, seq_len, 1024):
            logger.info(f"✅ Cross-dimensional test passed: input shape {x_1024.shape} -> output shape {output.shape}")
        else:
            logger.error(f"❌ Cross-dimensional test failed: output shape {output.shape}")
            success = False
            
    except Exception as e:
        logger.error(f"❌ Cross-dimensional test error: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    success = test_residual_block()
    sys.exit(0 if success else 1)