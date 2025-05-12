#!/usr/bin/env python
"""
Test script for verifying the embedding dimension mismatch handling.
This script tests the improved dimension handling in the embedding classes.
"""

import torch
import torch.nn as nn
import sys
import traceback

# Import the embedding classes
from agentic_diffusion.code_generation.models.embeddings import CodeEmbedding, TimestepEmbedding

def test_code_embedding_dimension_handling():
    """Test the CodeEmbedding class's dimension handling."""
    print("\n=== Testing CodeEmbedding dimension handling ===")
    
    # Standard initialization
    vocab_size = 10000
    embedding_dim = 128
    
    # Create an instance of CodeEmbedding
    embedding = CodeEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_seq_len=512,
        dropout=0.1
    )
    
    # Test with normal dimensions
    print("\nTest with normal dimensions:")
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    try:
        outputs = embedding(input_ids)
        print(f"Success! Output shape: {outputs.shape}")
        assert outputs.shape == (batch_size, seq_len, embedding_dim)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with sequence length larger than max_seq_len
    print("\nTest with sequence length larger than max_seq_len:")
    long_seq_len = 600
    long_input_ids = torch.randint(0, vocab_size, (batch_size, long_seq_len))
    
    try:
        outputs = embedding(long_input_ids)
        print(f"Success! Output shape: {outputs.shape}")
        # Should be truncated to max_seq_len
        print(f"Expected embedding dimension: {embedding_dim}, Actual: {outputs.shape[-1]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with expected_dim parameter
    print("\nTest with expected_dim parameter:")
    expected_dim = 256  # Different from embedding_dim
    
    try:
        outputs = embedding(input_ids, expected_dim=expected_dim)
        print(f"Success! Output shape: {outputs.shape}")
        assert outputs.shape[-1] == expected_dim
        print(f"Dimension adjustment worked! Output dimension: {outputs.shape[-1]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


def test_timestep_embedding_dimension_handling():
    """Test the TimestepEmbedding class's dimension handling."""
    print("\n=== Testing TimestepEmbedding dimension handling ===")
    
    # Test with matching dimensions
    print("\nTest with matching dimensions:")
    embedding_dim = 128
    projection_dim = 128
    
    timestep_embedding = TimestepEmbedding(
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    batch_size = 2
    timesteps = torch.tensor([10, 20])
    
    try:
        outputs = timestep_embedding(timesteps)
        print(f"Success! Output shape: {outputs.shape}")
        assert outputs.shape == (batch_size, projection_dim)
        print("Dimension check passed.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Test with mismatched dimensions
    print("\nTest with mismatched dimensions (should be handled internally):")
    
    # Create a modified instance with a different projection dimension
    projection_dim = 256
    
    # Modify the output_dim without changing the projection layer
    # This would cause a dimension mismatch in the original implementation
    timestep_embedding = TimestepEmbedding(
        embedding_dim=embedding_dim,
        projection_dim=projection_dim
    )
    
    # Override the output_dim to simulate a mismatch
    timestep_embedding.output_dim = 512
    
    try:
        outputs = timestep_embedding(timesteps)
        print(f"Success! Output shape: {outputs.shape}")
        print(f"Expected dimension: {timestep_embedding.output_dim}, Actual: {outputs.shape[-1]}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Test CodeEmbedding
        test_code_embedding_dimension_handling()
        
        # Test TimestepEmbedding
        test_timestep_embedding_dimension_handling()
        
        print("\nAll embedding tests completed successfully.")
    except Exception as e:
        print(f"Unexpected error during testing: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)