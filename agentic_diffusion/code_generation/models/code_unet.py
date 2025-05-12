"""
U-Net model for code diffusion.

This module provides the CodeUNet implementation, which is a U-Net
architecture adapted for code generation using diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from agentic_diffusion.code_generation.models.embeddings import CodeEmbedding, TimestepEmbedding
from agentic_diffusion.code_generation.models.blocks import TransformerBlock, ResidualBlock, CrossAttentionBlock, DynamicLayerNorm

logger = logging.getLogger(__name__)

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
            # Create condition blocks for different dimensions in the U-Net
            self.condition_blocks = nn.ModuleDict()
            
            # Calculate the dimensions at each level of the decoder path
            decoder_dims = []
            current_dim = hidden_dim
            for i in range(num_downsamples):
                # Track dimensions along the encoder path (will be mirrored in decoder)
                decoder_dims.append(current_dim)
                current_dim *= 2
                
            # Add the bottleneck dimension
            decoder_dims.append(current_dim)
            
            # Create condition blocks for each dimension in the decoder path
            for i, dim in enumerate(reversed(decoder_dims)):
                dim_key = str(dim)
                self.condition_blocks[dim_key] = nn.ModuleList([
                    CrossAttentionBlock(
                        d_model=dim,
                        d_context=condition_dim,
                        n_heads=num_heads,
                        dropout=dropout
                    ) for _ in range(num_layers // num_downsamples)
                ])
                logger.info(f"Created condition blocks for dimension {dim}")
        
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
        
        # Final layer to produce logits for each token - using a flexible approach
        self.final_norm = DynamicLayerNorm(hidden_dim)
        self.final_projection = nn.ModuleDict({
            str(hidden_dim): nn.Linear(hidden_dim, vocab_size)
        })
        
        # Pre-initialize for common dimensions in the U-Net
        for scale in [1, 2, 4]:
            dim = hidden_dim * scale
            if dim != hidden_dim:
                self.final_projection[str(dim)] = nn.Linear(dim, vocab_size)
                logger.info(f"Pre-initialized final projection from {dim} to {vocab_size}")
    
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
                # Get the current dimension
                current_dim = h.size(-1)
                
                # Get or create appropriate condition blocks for this dimension
                condition_blocks = self._get_or_create_condition_block(current_dim, self.condition_dim)
                
                # Use the appropriate condition block for this dimension
                block_idx = i % len(condition_blocks)
                h = condition_blocks[block_idx](h, condition)
        
        # Get the final hidden dimension
        final_dim = h.size(-1)
        
        # Apply layer normalization with the correct dimension
        h_norm = self.final_norm(h)
        
        # Apply final projection with the correct dimension
        dim_key = str(final_dim)
        if dim_key not in self.final_projection:
            logger.info(f"Creating new final projection from {final_dim} to {self.vocab_size}")
            self.final_projection[dim_key] = nn.Linear(final_dim, self.vocab_size).to(h.device)
        
        # Get logits using the appropriate projection
        logits = self.final_projection[dim_key](h_norm)
        
        return logits
    
    def get_model_size(self) -> int:
        """
        Calculate the number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
        
    def _get_or_create_condition_block(self, dimension: int, condition_dim: int) -> nn.Module:
        """
        Get an existing condition block for the given dimension or create a new one.
        
        Args:
            dimension: Hidden dimension for the condition block
            condition_dim: Dimension of the condition embedding
            
        Returns:
            Appropriate condition block for the dimension
        """
        dim_key = str(dimension)
        
        # Create new condition blocks for this dimension if needed
        if dim_key not in self.condition_blocks:
            logger.info(f"Creating new condition blocks for dimension {dimension}")
            self.condition_blocks[dim_key] = nn.ModuleList([
                CrossAttentionBlock(
                    d_model=dimension,
                    d_context=self.condition_dim,
                    n_heads=self.num_heads,
                    dropout=0.1
                ) for _ in range(self.num_layers // self.num_downsamples)
            ]).to(next(self.parameters()).device)
        
        return self.condition_blocks[dim_key]


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