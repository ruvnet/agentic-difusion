"""
Unit tests for the code diffusion model implementation.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from agentic_diffusion.code_generation.code_diffusion import (
    CodeEmbedding,
    TransformerBlock,
    CodeUNet,
    CodeDiffusionModel,
    CodeDiscreteScheduler
)
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.core.noise_schedules import LinearScheduler


class TestCodeEmbedding:
    """Test suite for CodeEmbedding class."""
    
    def test_init(self):
        """Test initialization of CodeEmbedding."""
        vocab_size = 1000
        embedding_dim = 128
        
        embedding = CodeEmbedding(vocab_size, embedding_dim)
        
        assert embedding.vocab_size == vocab_size
        assert embedding.embedding_dim == embedding_dim
        assert isinstance(embedding.token_embedding, torch.nn.Embedding)
        assert embedding.token_embedding.num_embeddings == vocab_size
        assert embedding.token_embedding.embedding_dim == embedding_dim
    
    def test_forward(self):
        """Test forward pass of CodeEmbedding."""
        vocab_size = 1000
        embedding_dim = 128
        batch_size = 4
        seq_len = 16
        
        embedding = CodeEmbedding(vocab_size, embedding_dim)
        
        # Create random token indices
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Get embeddings
        embeddings = embedding(tokens)
        
        # Check shapes
        assert embeddings.shape == (batch_size, seq_len, embedding_dim)
        
        # Check that different tokens get different embeddings
        if tokens[0, 0] != tokens[0, 1]:
            assert not torch.allclose(embeddings[0, 0], embeddings[0, 1])


class TestTransformerBlock:
    """Test suite for TransformerBlock class."""
    
    def test_init(self):
        """Test initialization of TransformerBlock."""
        embedding_dim = 128
        num_heads = 4
        hidden_dim = 256
        dropout = 0.1
        time_embedding_dim = 64
        
        block = TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout, time_embedding_dim)
        
        assert block.embedding_dim == embedding_dim
        assert isinstance(block.attention, torch.nn.MultiheadAttention)
        assert isinstance(block.ff, torch.nn.Sequential)
        assert isinstance(block.time_proj, torch.nn.Linear)
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)
    
    def test_forward(self):
        """Test forward pass of TransformerBlock."""
        embedding_dim = 128
        num_heads = 4
        hidden_dim = 256
        dropout = 0.1
        time_embedding_dim = 64
        
        batch_size = 4
        seq_len = 16
        
        block = TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout, time_embedding_dim)
        
        # Create random embeddings and time embeddings
        x = torch.randn(batch_size, seq_len, embedding_dim)
        time_emb = torch.randn(batch_size, time_embedding_dim)
        
        # Forward pass
        output = block(x, time_emb)
        
        # Check shape preservation
        assert output.shape == x.shape


class TestCodeUNet:
    """Test suite for CodeUNet class."""
    
    def test_init(self):
        """Test initialization of CodeUNet."""
        vocab_size = 1000
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 4
        num_heads = 4
        dropout = 0.1
        time_embedding_dim = 64
        
        model = CodeUNet(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            time_embedding_dim=time_embedding_dim
        )
        
        assert isinstance(model.token_embedding, CodeEmbedding)
        assert isinstance(model.time_embedding, torch.nn.Sequential)
        assert len(model.blocks) == num_layers
        assert isinstance(model.norm, torch.nn.LayerNorm)
        assert isinstance(model.out_proj, torch.nn.Linear)
    
    def test_sinusoidal_embedding(self):
        """Test sinusoidal embedding generation."""
        vocab_size = 1000
        embedding_dim = 128
        hidden_dim = 256
        
        model = CodeUNet(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        
        batch_size = 4
        t = torch.randint(0, 1000, (batch_size,))
        
        emb = model.get_sinusoidal_embedding(t)
        
        assert emb.shape == (batch_size, embedding_dim)
        
        # Different timesteps should get different embeddings
        if t[0] != t[1]:
            assert not torch.allclose(emb[0], emb[1])
    
    def test_forward(self):
        """Test forward pass of CodeUNet."""
        vocab_size = 1000
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 4
        
        model = CodeUNet(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        batch_size = 4
        seq_len = 16
        
        # Create random token indices and timesteps
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        t = torch.randint(0, 1000, (batch_size,))
        
        # Forward pass
        logits = model(x, t)
        
        # Check output shape
        assert logits.shape == (batch_size, seq_len, vocab_size)


class TestCodeDiffusionModel:
    """Test suite for CodeDiffusionModel class."""
    
    def test_init(self):
        """Test initialization of CodeDiffusionModel."""
        vocab_size = 1000
        max_seq_len = 512
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 4
        
        model = CodeDiffusionModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        assert model.vocab_size == vocab_size
        assert model.max_seq_len == max_seq_len
        assert model.embedding_dim == embedding_dim
        assert isinstance(model.model, CodeUNet)
        assert model.mask_token_id == vocab_size - 1
    
    def test_forward(self):
        """Test forward pass of CodeDiffusionModel."""
        vocab_size = 1000
        max_seq_len = 512
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 2  # Smaller model for faster tests
        
        model = CodeDiffusionModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        batch_size = 2
        seq_len = 16
        
        # Create random token indices and timesteps
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        t = torch.randint(0, 1000, (batch_size,))
        
        # Forward pass
        logits = model(x, t)
        
        # Check output shape
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_noising_process(self):
        """Test the noising process for code tokens."""
        vocab_size = 1000
        max_seq_len = 512
        embedding_dim = 128
        
        model = CodeDiffusionModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            noise_scheduler=LinearScheduler(num_timesteps=100)
        )
        
        batch_size = 4
        seq_len = 16
        
        # Create random token indices and timesteps
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test with different timesteps (noise levels)
        for t_val in [0, 50, 99]:
            t = torch.full((batch_size,), t_val)
            
            # Apply noise
            x_t = model.noising_process(x_0, t)
            
            # Check shape
            assert x_t.shape == x_0.shape
            
            # Calculate percentage of tokens that were changed
            changed = (x_t != x_0).float().mean().item()
            
            # At t=0, no tokens should change
            if t_val == 0:
                assert changed < 0.1
            # At t=max, most tokens should change
            elif t_val == 99:
                assert changed > 0.5
    
    def test_training_step(self):
        """Test training step of CodeDiffusionModel."""
        vocab_size = 1000
        max_seq_len = 512
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 2  # Smaller model for faster tests
        
        model = CodeDiffusionModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            noise_scheduler=LinearScheduler(num_timesteps=100)
        )
        
        batch_size = 2
        seq_len = 16
        
        # Create a batch of data
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch = {"x": x}
        
        # Perform training step
        result = model.training_step(batch)
        
        # Check results
        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].ndim == 0  # Scalar tensor
        assert "x_t" in result
        assert result["x_t"].shape == x.shape
    
    @pytest.mark.parametrize("guidance_scale", [1.0, 1.5])
    @pytest.mark.parametrize("temperature", [0.7, 1.0])
    def test_sample(self, guidance_scale, temperature):
        """Test sampling from CodeDiffusionModel."""
        vocab_size = 1000
        max_seq_len = 512
        embedding_dim = 128
        hidden_dim = 256
        num_layers = 2  # Smaller model for faster tests
        
        model = CodeDiffusionModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            noise_scheduler=LinearScheduler(num_timesteps=10)  # Use fewer steps for testing
        )
        
        batch_size = 2
        seq_len = 16
        
        # Sample from the model
        samples = model.sample(
            [batch_size, seq_len],
            guidance_scale=guidance_scale,
            temperature=temperature
        )
        
        # Check results
        assert isinstance(samples, list)
        assert len(samples) == batch_size
        for sample in samples:
            assert isinstance(sample, list)
            assert len(sample) == seq_len
            # All tokens should be valid indices
            assert all(0 <= token < vocab_size for token in sample)


class TestCodeDiscreteScheduler:
    """Test suite for CodeDiscreteScheduler class."""
    
    def test_init(self):
        """Test initialization of CodeDiscreteScheduler."""
        num_timesteps = 1000
        mask_token_id = 0
        
        scheduler = CodeDiscreteScheduler(
            num_timesteps=num_timesteps,
            mask_token_id=mask_token_id
        )
        
        assert scheduler.num_timesteps == num_timesteps
        assert scheduler.mask_token_id == mask_token_id
        assert scheduler.noise_rates.shape == (num_timesteps,)
        assert scheduler.noise_rates[0] == 0.0
        assert scheduler.noise_rates[-1] == 1.0
    
    def test_add_noise(self):
        """Test add_noise method of CodeDiscreteScheduler."""
        num_timesteps = 100
        mask_token_id = 0
        vocab_size = 1000
        
        scheduler = CodeDiscreteScheduler(
            num_timesteps=num_timesteps,
            mask_token_id=mask_token_id
        )
        
        batch_size = 4
        seq_len = 16
        
        # Create random token indices
        x_0 = torch.randint(1, vocab_size, (batch_size, seq_len))
        
        # Test with different timesteps (noise levels)
        for t_val in [0, 50, 99]:
            t = torch.full((batch_size,), t_val)
            
            # Add noise
            x_t = scheduler.add_noise(x_0, t)
            
            # Check shape
            assert x_t.shape == x_0.shape
            
            # Calculate percentage of tokens that were changed
            changed = (x_t != x_0).float().mean().item()
            
            # At t=0, few tokens should change
            if t_val == 0:
                assert changed < 0.1
            # At t=max, most tokens should change
            elif t_val == 99:
                assert changed > 0.5
    
    def test_remove_noise(self):
        """Test remove_noise method of CodeDiscreteScheduler."""
        num_timesteps = 100
        mask_token_id = 0
        vocab_size = 1000
        
        scheduler = CodeDiscreteScheduler(
            num_timesteps=num_timesteps,
            mask_token_id=mask_token_id
        )
        
        batch_size = 4
        seq_len = 16
        
        # Create noisy token indices
        x_t = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create model predictions (logits)
        pred = torch.randn(batch_size, seq_len, vocab_size)
        
        # Test with different timesteps
        for t_val in [0, 50, 99]:
            t = torch.full((batch_size,), t_val)
            
            # Remove noise
            x_t_minus_1 = scheduler.remove_noise(x_t, t, pred)
            
            # Check shape
            assert x_t_minus_1.shape == x_t.shape
            
            # Tokens should be valid indices
            assert torch.all((0 <= x_t_minus_1) & (x_t_minus_1 < vocab_size))


@pytest.mark.integration
def test_code_diffusion_end_to_end():
    """
    Integration test for the full code diffusion pipeline.
    
    This test verifies that the entire code diffusion model can be
    trained and used for generation.
    """
    # Model parameters
    vocab_size = 1000
    max_seq_len = 64
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    
    # Create model
    model = CodeDiffusionModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        noise_scheduler=LinearScheduler(num_timesteps=10)  # Use fewer steps for testing
    )
    
    # Create synthetic training data
    batch_size = 2
    seq_len = 32
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Perform a few training steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for _ in range(2):
        batch = {"x": x}
        result = model.training_step(batch)
        loss = result["loss"]
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Generate a sample
    generated_tokens = model.sample(
        [1, seq_len],
        temperature=1.0,
        guidance_scale=1.0
    )[0]
    
    # Verify the sample has the correct length and valid tokens
    assert len(generated_tokens) == seq_len
    assert all(0 <= token < vocab_size for token in generated_tokens)
    
    # Test with code tokenizer
    tokenizer = CodeTokenizer(language="python")
    code_sample = "def hello_world():\n    print('Hello, world!')"
    tokens = tokenizer.tokenize(code_sample)
    
    # Verify token processing and model functionality are compatible
    token_indices = [hash(token) % (vocab_size - 1) for token in tokens]
    x_code = torch.tensor([token_indices], device=model.device)
    t = torch.zeros(1, device=model.device)
    
    # Process through model
    with torch.no_grad():
        logits = model(x_code, t)
    
    # Verify output shape
    assert logits.shape == (1, len(token_indices), vocab_size)