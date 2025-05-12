"""
Neural network blocks for code diffusion models.

This module provides building blocks for the neural networks used in
code diffusion models, including transformer blocks and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any

class TransformerBlock(nn.Module):
    """
    Transformer block for code diffusion models.
    
    This module implements a standard transformer block with self-attention
    and feed-forward layers, with residual connections and layer normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        prenorm: bool = True
    ):
        """
        Initialize the transformer block.
        
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
            activation: Activation function
            prenorm: Whether to use pre-normalization or post-normalization
        """
        super().__init__()
        
        # Set dimensions
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.prenorm = prenorm
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            self._get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _get_activation_fn(self, activation: str) -> nn.Module:
        """
        Get activation function by name.
        
        Args:
            activation: Name of the activation function
            
        Returns:
            Activation function module
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "silu" or activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            is_causal: Whether to use causal masking (for autoregressive)
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Pre-normalization architecture
        if self.prenorm:
            # Self-attention with pre-normalization
            norm_x = self.norm1(x)
            attn_output, _ = self.self_attention(
                query=norm_x,
                key=norm_x,
                value=norm_x,
                key_padding_mask=mask,
                is_causal=is_causal
            )
            x = x + self.dropout(attn_output)
            
            # Feed-forward with pre-normalization
            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + self.dropout(ff_output)
        
        # Post-normalization architecture
        else:
            # Self-attention with post-normalization
            attn_output, _ = self.self_attention(
                query=x,
                key=x,
                value=x,
                key_padding_mask=mask,
                is_causal=is_causal
            )
            x = self.norm1(x + self.dropout(attn_output))
            
            # Feed-forward with post-normalization
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning in diffusion models.
    
    This module implements a cross-attention mechanism that allows
    conditioning the diffusion model on external information.
    """
    
    def __init__(
        self,
        d_model: int,
        d_context: int,
        n_heads: int,
        dropout: float = 0.1,
        prenorm: bool = True
    ):
        """
        Initialize the cross-attention block.
        
        Args:
            d_model: Hidden dimension of the main model
            d_context: Hidden dimension of the context
            n_heads: Number of attention heads
            dropout: Dropout rate
            prenorm: Whether to use pre-normalization or post-normalization
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_context = d_context
        self.n_heads = n_heads
        self.prenorm = prenorm
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layer for context if dimensions don't match
        if d_model != d_context:
            self.context_projection = nn.Linear(d_context, d_model)
        else:
            self.context_projection = nn.Identity()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the cross-attention block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            context: Context tensor of shape [batch_size, context_len, d_context]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Project context to match model dimension
        context = self.context_projection(context)
        
        # Apply cross-attention
        if self.prenorm:
            # Pre-normalization
            norm_x = self.norm1(x)
            attn_output, _ = self.cross_attention(
                query=norm_x,
                key=context,
                value=context,
                key_padding_mask=mask
            )
            x = x + self.dropout(attn_output)
        else:
            # Post-normalization
            attn_output, _ = self.cross_attention(
                query=x,
                key=context,
                value=context,
                key_padding_mask=mask
            )
            x = self.norm1(x + self.dropout(attn_output))
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block for U-Net architecture in diffusion models.
    
    This module combines transformer blocks with residual connections
    and timestep conditioning for diffusion models.
    """
    
    def __init__(
        self,
        d_model: int,
        d_time: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize the residual block.
        
        Args:
            d_model: Hidden dimension
            d_time: Dimension of timestep embeddings
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Time projection
        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_model),
            nn.SiLU()
        )
        
        # First transformer block
        self.transformer1 = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Second transformer block
        self.transformer2 = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Skip connection
        self.skip_connection = nn.Identity()
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            time_emb: Timestep embedding of shape [batch_size, d_time]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Project timestep embedding
        time_proj = self.time_proj(time_emb)
        
        # Expand timestep embedding to add to each position
        time_proj = time_proj.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Store input for residual connection
        residual = x
        
        # Apply transformer blocks
        x = self.transformer1(x, mask)
        x = x + time_proj  # Add timestep embedding
        x = self.transformer2(x, mask)
        
        # Apply residual connection
        x = residual + self.skip_connection(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        return x