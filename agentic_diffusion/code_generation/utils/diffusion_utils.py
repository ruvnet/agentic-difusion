"""
Utility functions for code diffusion models.

This module provides utility functions for working with diffusion models
applied to code generation, including embedding generation, diffusion operations,
and evaluation metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union, Any

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for diffusion models.
    
    Args:
        timesteps: Tensor of timestep values of shape [batch_size]
        dim: Embedding dimension
        
    Returns:
        Tensor of shape [batch_size, dim] containing timestep embeddings
    """
    # Create embedding lookup table
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    
    # Create the actual embeddings
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    # Handle odd dimensions
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
    return emb

def categorical_diffusion(
    x_0: torch.Tensor,
    t: torch.Tensor,
    num_classes: int,
    noise_schedule: torch.Tensor
) -> torch.Tensor:
    """
    Apply categorical diffusion process to discrete token sequences.
    
    Args:
        x_0: Original token indices of shape [batch_size, seq_len]
        t: Timestep values of shape [batch_size]
        num_classes: Number of token classes (vocabulary size)
        noise_schedule: Alpha cumulative product schedule for diffusion
        
    Returns:
        Noisy token indices of shape [batch_size, seq_len]
    """
    batch_size, seq_len = x_0.shape
    device = x_0.device
    
    # Get the appropriate noise level based on timestep
    # Normalize t to index into noise_schedule
    noise_level = torch.gather(noise_schedule, 0, t).view(-1, 1, 1)
    
    # One-hot encode the original tokens
    x_onehot = F.one_hot(x_0, num_classes).float()
    
    # Create random noise for categorical distribution
    noise = torch.randn_like(x_onehot)
    
    # Apply noise to one-hot vectors based on noise level
    noisy_logits = noise_level * x_onehot + (1 - noise_level) * noise
    
    # Sample from the resulting categorical distribution
    # We can either sample or take the argmax depending on noise level
    if torch.rand(1).item() < 0.5:  # Randomize between sampling and argmax
        x_t = torch.argmax(noisy_logits, dim=-1)
    else:
        # Sample from categorical distribution (more randomness)
        probs = F.softmax(noisy_logits, dim=-1)
        x_t = torch.multinomial(probs.view(-1, num_classes), 1).view(batch_size, seq_len)
    
    return x_t

def token_accuracy(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Calculate token-level accuracy for code generation.
    
    Args:
        predicted_tokens: Predicted token indices of shape [batch_size, seq_len]
        target_tokens: Target token indices of shape [batch_size, seq_len]
        ignore_index: Index to ignore in the calculation
        
    Returns:
        Token accuracy as a float
    """
    # Mask for valid tokens
    mask = (target_tokens != ignore_index).float()
    
    # Calculate accuracy on valid tokens
    correct = (predicted_tokens == target_tokens).float() * mask
    accuracy = correct.sum() / (mask.sum() + 1e-8)
    
    return accuracy.item()

def reconstruct_code_from_tokens(
    token_indices: List[int],
    tokenizer: Any
) -> str:
    """
    Reconstruct code from token indices.
    
    Args:
        token_indices: List of token indices
        tokenizer: Tokenizer with detokenization capability
        
    Returns:
        Reconstructed code as a string
    """
    try:
        # First try to convert indices to tokens
        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            tokens = tokenizer.convert_ids_to_tokens(token_indices)
        elif hasattr(tokenizer, 'idx_to_token'):
            tokens = [tokenizer.idx_to_token(idx) for idx in token_indices]
        else:
            # Fallback option
            tokens = [f"<{idx}>" for idx in token_indices]
            
        # Then detokenize
        if hasattr(tokenizer, 'detokenize'):
            code = tokenizer.detokenize(tokens)
        else:
            # Simple joining as fallback
            code = "".join(tokens)
            
        return code
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error reconstructing code: {e}")
        return "".join([f"<{idx}>" for idx in token_indices])

def calculate_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Calculate perplexity for a language model.
    
    Args:
        logits: Model logits of shape [batch_size, seq_len, vocab_size]
        targets: Target token indices of shape [batch_size, seq_len]
        ignore_index: Index to ignore in the calculation
        
    Returns:
        Perplexity for each sequence in the batch
    """
    # Flatten the logits and targets
    batch_size, seq_len = targets.shape
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Create a mask for valid tokens
    mask = (targets != ignore_index).float()
    
    # Calculate cross-entropy loss
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
    
    # Reshape loss and mask to batch dimensions
    loss = loss.view(batch_size, seq_len)
    mask = mask.view(batch_size, seq_len)
    
    # Calculate perplexity per sequence
    seq_loss = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    perplexity = torch.exp(seq_loss)
    
    return perplexity