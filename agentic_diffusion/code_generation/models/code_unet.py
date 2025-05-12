"""
U-Net model for code diffusion.

This module provides the CodeUNet implementation, which is a U-Net
architecture adapted for code generation using diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any

from agentic_diffusion.code_generation.models.embeddings import CodeEmbedding, TimestepEmbedding
from agentic_diffusion.code_generation.models.blocks import TransformerBlock, ResidualBlock, CrossAttentionBlock

class CodeUNet(nn.Module):
    """
    U-Net architecture for code diffusion.
    
    This model adapts the U-Net architecture, traditionally used in image diffusion,
    for code generation with transformer-based blocks. It includes timestep conditioning
    and optional specification conditioning.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        condition_dim: Optional[int] = None,
        num_downsamples: int = 2
    ):
        """
        Initialize the code U-Net model.
        
        Args:
            vocab_size: Size of the code token vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            condition_dim: Dimension of condition embeddings (if None, no conditioning)
            num_downsamples: Number of downsampling operations in the U-Net
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.condition_dim = condition_dim
        self.num_downsamples = num_downsamples
        
        # Token embedding
        self.token_embedding = CodeEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Timestep embedding
        self.time_embedding = TimestepEmbedding(
            embedding_dim=embedding_dim,
            projection_dim=hidden_dim
        )
        
        # Condition embedding (for specification)
        self.use_conditioning = condition_dim is not None
        if self.use_conditioning:
            self.condition_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    d_model=hidden_dim,
                    d_context=condition_dim,
                    n_heads=num_heads,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        
        # Initial projection from embedding dim to hidden dim
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Encoder blocks (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_downsamples):
            # Each encoder level has layers and downsampling
            level_blocks = nn.ModuleList([
                ResidualBlock(
                    d_model=current_dim,
                    d_time=hidden_dim,
                    n_heads=num_heads,
                    dropout=dropout
                ) for _ in range(num_layers // num_downsamples)
            ])
            
            # Downsampling with stride-2 attention
            downsample = nn.Sequential(
                nn.LayerNorm(current_dim),
                nn.Linear(current_dim, current_dim * 2)
            )
            
            self.encoder_blocks.append(nn.ModuleDict({
                'blocks': level_blocks,
                'downsample': downsample
            }))
            
            current_dim *= 2
        
        # Middle block (bottleneck)
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(
                d_model=current_dim,
                d_time=hidden_dim,
                n_heads=num_heads * 2,
                dropout=dropout
            ) for _ in range(2)
        ])
        
        # Decoder blocks (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_downsamples):
            # Upsample to match encoder level
            upsample = nn.Sequential(
                nn.LayerNorm(current_dim),
                nn.Linear(current_dim, current_dim // 2)
            )
            
            # After upsampling, the dimension is halved
            current_dim //= 2
            
            # Each decoder level has layers and receives skip connection from encoder
            level_blocks = nn.ModuleList([
                ResidualBlock(
                    d_model=current_dim,
                    d_time=hidden_dim,
                    n_heads=num_heads,
                    dropout=dropout
                ) for _ in range(num_layers // num_downsamples)
            ])
            
            self.decoder_blocks.append(nn.ModuleDict({
                'blocks': level_blocks,
                'upsample': upsample
            }))
        
        # Final layer to produce logits for each token
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the code U-Net model.
        
        Args:
            x: Token indices of shape [batch_size, seq_len]
            t: Timestep values of shape [batch_size]
            condition: Optional condition embedding of shape [batch_size, condition_dim]
            mask: Optional attention mask
            
        Returns:
            Token logits of shape [batch_size, seq_len, vocab_size]
        """
        # Get batch size and sequence length
        batch_size, seq_len = x.shape
        
        # Embed tokens
        token_embeds = self.token_embedding(x)
        
        # Embed timesteps
        time_embeds = self.time_embedding(t)
        
        # Project to hidden dimension
        h = self.input_projection(token_embeds)
        
        # Store skip connections for U-Net
        skip_connections = []
        
        # Encoder path
        for encoder_level in self.encoder_blocks:
            # Apply blocks
            for block in encoder_level['blocks']:
                h = block(h, time_embeds, mask)
            
            # Store for skip connection
            skip_connections.append(h)
            
            # Downsample
            h = encoder_level['downsample'](h)
        
        # Middle blocks
        for block in self.middle_blocks:
            h = block(h, time_embeds, mask)
        
        # Decoder path
        for i, decoder_level in enumerate(self.decoder_blocks):
            # Upsample
            h = decoder_level['upsample'](h)
            
            # Add skip connection from encoder
            skip_idx = len(skip_connections) - i - 1
            h = h + skip_connections[skip_idx]
            
            # Apply blocks
            for block in decoder_level['blocks']:
                h = block(h, time_embeds, mask)
            
            # Apply conditioning if available
            if self.use_conditioning and condition is not None:
                condition_idx = i % len(self.condition_blocks)
                h = self.condition_blocks[condition_idx](h, condition)
        
        # Final layer to produce token logits
        logits = self.final_layer(h)
        
        return logits
    
    def get_model_size(self) -> int:
        """
        Calculate the number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())

class CodeClassifierFreeGuidanceUNet(nn.Module):
    """
    U-Net with classifier-free guidance for code generation.
    
    This model extends the CodeUNet to support classifier-free guidance,
    which allows controlled generation without a separate classifier.
    """
    
    def __init__(self, model: CodeUNet):
        """
        Initialize the guided U-Net.
        
        Args:
            model: Base CodeUNet model
        """
        super().__init__()
        
        self.model = model
        self.condition_dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        
        Args:
            x: Token indices of shape [batch_size, seq_len]
            t: Timestep values of shape [batch_size]
            condition: Condition embedding of shape [batch_size, condition_dim]
            guidance_scale: Scale for classifier-free guidance
            mask: Optional attention mask
            
        Returns:
            Token logits of shape [batch_size, seq_len, vocab_size]
        """
        # If guidance scale is 1.0 or no condition, use regular forward pass
        if guidance_scale == 1.0 or condition is None:
            return self.model(x, t, condition, mask)
        
        # Get prediction with condition
        cond_logits = self.model(x, t, condition, mask)
        
        # Get prediction without condition
        uncond_logits = self.model(x, t, None, mask)
        
        # Apply classifier-free guidance
        guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        
        return guided_logits