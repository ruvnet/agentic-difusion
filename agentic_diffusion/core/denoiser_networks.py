"""
Denoiser network implementations for diffusion models.

This module contains different implementations of denoiser networks
that are used to predict noise during the diffusion process.
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union


class DenoiserNetwork(nn.Module):
    """Base class for denoiser networks used in diffusion models."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        """
        Initialize denoiser network.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, noise_level: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through denoiser network.
        
        Args:
            x: Input data
            noise_level: Current noise level
            condition: Optional conditioning information
            
        Returns:
            Predicted noise
        """
        raise NotImplementedError("Subclasses must implement forward")


class TransformerBlock(nn.Module):
    """Transformer block with self-attention."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float):
        """
        Initialize transformer block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x)
        )
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float):
        """
        Initialize cross-attention block.
        
        Args:
            dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross-attention block.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            context: Context tensor for conditioning [batch_size, context_len, dim]
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Cross-attention with residual connection
        attn_out, _ = self.attention(
            query=self.norm1(x),
            key=self.norm2(context),
            value=self.norm2(context)
        )
        x = x + self.dropout(attn_out)
        
        return x


class UnconditionalDenoiser(DenoiserNetwork):
    """Denoiser network for unconditional generation."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        """
        Initialize unconditional denoiser.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(embedding_dim, hidden_dim, num_layers, num_heads, dropout)
        
        # Noise level embedding
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder layers
        self.encoder = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor, noise_level: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through unconditional denoiser.
        
        Args:
            x: Input data [batch_size, *data_dims, embedding_dim]
            noise_level: Current noise level [batch_size]
            condition: Ignored for unconditional model
            
        Returns:
            Predicted noise [batch_size, *data_dims, embedding_dim]
        """
        # Flatten data dimensions for transformer
        batch_size = x.shape[0]
        data_dims = x.shape[1:-1]
        seq_len = np.prod(data_dims).astype(int)
        
        x_flat = x.view(batch_size, seq_len, self.embedding_dim)
        
        # Project input to hidden dimension
        h = self.input_proj(x_flat)
        
        # Embed noise level
        noise_emb = self.noise_embedding(noise_level.view(-1, 1))
        
        # Add noise embedding to input
        h = h + noise_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for layer in self.encoder:
            h = layer(h)
        
        # Project back to embedding dimension
        output = self.output_proj(h)
        
        # Reshape back to input shape
        output = output.view(batch_size, *data_dims, self.embedding_dim)
        
        return output


class ConditionalDenoiser(DenoiserNetwork):
    """Denoiser network that supports conditional generation."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        """
        Initialize conditional denoiser.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(embedding_dim, hidden_dim, num_layers, num_heads, dropout)
        
        # Noise level embedding
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Self-attention layers
        self.self_attention = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor, noise_level: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through conditional denoiser.
        
        Args:
            x: Input data [batch_size, *data_dims, embedding_dim]
            noise_level: Current noise level [batch_size]
            condition: Conditioning information [batch_size, cond_dims, embedding_dim]
            
        Returns:
            Predicted noise [batch_size, *data_dims, embedding_dim]
        """
        # Flatten data dimensions for transformer
        batch_size = x.shape[0]
        data_dims = x.shape[1:-1]
        seq_len = np.prod(data_dims).astype(int)
        
        x_flat = x.view(batch_size, seq_len, self.embedding_dim)
        
        # Project input to hidden dimension
        h = self.input_proj(x_flat)
        
        # Embed noise level
        noise_emb = self.noise_embedding(noise_level.view(-1, 1))
        
        # Add noise embedding to input
        h = h + noise_emb.unsqueeze(1)
        
        # Process condition
        if condition is not None:
            # Flatten condition dimensions if needed
            cond_dims = condition.shape[1:-1]
            cond_seq_len = np.prod(cond_dims).astype(int)
            cond_flat = condition.view(batch_size, cond_seq_len, self.embedding_dim)
            
            # Project condition to hidden dimension
            cond_emb = self.condition_embedding(cond_flat)
            
            # Apply interleaved self-attention and cross-attention
            for self_attn, cross_attn in zip(self.self_attention, self.cross_attention):
                h = self_attn(h)
                h = cross_attn(h, cond_emb)
        else:
            # Apply only self-attention if no condition is provided
            for layer in self.self_attention:
                h = layer(h)
        
        # Project back to embedding dimension
        output = self.output_proj(h)
        
        # Reshape back to input shape
        output = output.view(batch_size, *data_dims, self.embedding_dim)
        
        return output