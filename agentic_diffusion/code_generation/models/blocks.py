"""
Neural network blocks for code diffusion models.

This module provides building blocks for the neural networks used in
code diffusion models, including transformer blocks and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

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
        
        # Layer normalization with dynamic support
        self.norm1 = DynamicLayerNorm(d_model)
        self.norm2 = DynamicLayerNorm(d_model)
        
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
            
            # Ensure inputs have proper dimensions for batched attention
            query = norm_x
            key = norm_x
            value = norm_x
            
            # Check dimensions for debugging
            logger.debug(f"Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
            
            # Handle any potential dimension issues with batching
            if len(query.shape) == 3 and len(key.shape) == 2:
                # Expand 2D tensors to 3D for batched attention
                key = key.unsqueeze(0).expand(query.size(0), -1, -1)
                value = value.unsqueeze(0).expand(query.size(0), -1, -1)
            
            attn_output, _ = self.self_attention(
                query=query,
                key=key,
                value=value,
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
        
        # Layer normalization with dynamic support
        self.norm1 = DynamicLayerNorm(d_model)
        
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
            
            # Ensure inputs have proper dimensions for batched attention
            query = norm_x
            key = context
            value = context
            
            # Handle any potential dimension issues with batching
            if len(query.shape) == 3 and len(key.shape) == 2:
                # Expand 2D tensors to 3D for batched attention
                key = key.unsqueeze(0).expand(query.size(0), -1, -1)
                value = value.unsqueeze(0).expand(query.size(0), -1, -1)
            
            attn_output, _ = self.cross_attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=mask
            )
            x = x + self.dropout(attn_output)
        else:
            # Post-normalization
            
            # Ensure inputs have proper dimensions for batched attention
            query = x
            key = context
            value = context
            
            # Handle any potential dimension issues with batching
            if len(query.shape) == 3 and len(key.shape) == 2:
                # Expand 2D tensors to 3D for batched attention
                key = key.unsqueeze(0).expand(query.size(0), -1, -1)
                value = value.unsqueeze(0).expand(query.size(0), -1, -1)
            
            attn_output, _ = self.cross_attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=mask
            )
            x = self.norm1(x + self.dropout(attn_output))
        
        return x

class FeatureProjection(nn.Module):
    """
    Feature projection module for handling dimension changes.
    
    This module handles projection between different feature dimensions in the U-Net.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Initialize the projection layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection layer.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        return self.projection(x)

class DynamicLayerNorm(nn.Module):
    """
    Dynamic layer normalization that can handle different input dimensions.
    
    This is a flexible LayerNorm implementation that can adjust to different
    feature dimensions at runtime, useful for U-Net architectures where the
    feature dimension changes between encoder and decoder.
    """
    def __init__(self, initial_normalized_shape: int = 512):
        """
        Initialize with a default normalized shape.
        
        Args:
            initial_normalized_shape: Initial dimension for normalization
        """
        super().__init__()
        self.initial_normalized_shape = initial_normalized_shape
        
        # Initialize with common dimensions used in U-Net
        self.norms = nn.ModuleDict()
        
        # Create norms for common dimensions
        common_dims = [initial_normalized_shape, initial_normalized_shape*2, initial_normalized_shape*4]
        for dim in common_dims:
            dim_key = str(dim)
            self.norms[dim_key] = nn.LayerNorm(dim)
            logger.info(f"Pre-initialized LayerNorm for dimension {dim}")
            
        # Pre-initialize norms for common dimension values that might be used
        additional_dimensions = [256, 768, 1024, 2048, 4096]
        for dim in additional_dimensions:
            if dim not in common_dims:
                dim_key = str(dim)
                self.norms[dim_key] = nn.LayerNorm(dim)
                logger.info(f"Pre-initialized LayerNorm for dimension {dim} (special case)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that dynamically selects or creates the appropriate LayerNorm.
        
        Args:
            x: Input tensor of shape [..., feature_dim]
            
        Returns:
            Normalized tensor with the same shape
        """
        # Get the feature dimension (last dimension)
        feature_dim = x.size(-1)
        dim_key = str(feature_dim)
        
        # Create a norm for this dimension if it doesn't exist
        if dim_key not in self.norms:
            logger.info(f"Creating new LayerNorm for dimension {feature_dim}")
            self.norms[dim_key] = nn.LayerNorm(feature_dim).to(x.device)
        
        # Apply the appropriate norm
        return self.norms[dim_key](x)
class ResidualBlock(nn.Module):
    """
    Residual block for U-Net architecture in diffusion models.
    
    This module combines transformer blocks with residual connections
    and timestep conditioning for diffusion models. It includes dynamic
    handling of feature dimensions to support the U-Net architecture.
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
        
        # Store dimensions
        self.d_model = d_model
        # Pre-initialize time projections for standard dimensions
        # This ensures we have projections ready for the common cases
        self.time_proj = nn.ModuleDict()
        
        # Pre-create common time dimension projections
        common_output_dims = [
            d_model,         # Base case
            d_model*2,       # Upsampling
            256, 512, 768, 1024, 2048, 4096  # Common dimensions in transformer models
        ]
        
        for output_dim in common_output_dims:
            key = f"{d_time}_{output_dim}"
            self.time_proj[key] = nn.Sequential(
                nn.Linear(d_time, output_dim),
                nn.SiLU()
            )
            logger.info(f"Pre-initialized time projection from {d_time} to {output_dim}")
        
        # Transformer blocks
        self.transformer1 = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.transformer2 = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Dynamic layer normalization for variable feature dimensions
        self.norm = DynamicLayerNorm(d_model)
        
        # Feature projections for each supported dimension change
        # These will handle projections between different dimensions
        self.input_projections = nn.ModuleDict()
        
        # Common dimensions in the U-Net architecture (d_model, d_model*2, d_model*4, etc.)
        common_scales = [1, 2, 4]
        
        # Create input projections (from dim to d_model)
        for scale in common_scales:
            dim = d_model * scale
            if dim != d_model:
                # Input projection (from larger dimension to d_model)
                in_key = str(dim)
                self.input_projections[in_key] = FeatureProjection(dim, d_model)
                
                # Output projection (from d_model back to larger dimension)
                out_key = f"{d_model}_{dim}"
                self.input_projections[out_key] = FeatureProjection(d_model, dim)
                
        # Time projection for each dimension
        self.time_projections = nn.ModuleDict({
            str(d_model): nn.Identity()  # No projection needed for base dimension
        })
        
        # Pre-initialize time projections for common dimensions
        common_time_dims = [d_time, d_time*2]
        for time_dim in common_time_dims:
            if time_dim != d_model:
                key = f"{time_dim}_{d_model}"
                self.time_projections[key] = nn.Sequential(
                    nn.Linear(time_dim, d_model),
                    nn.SiLU()
                )
                logger.info(f"Pre-initialized time projection from {time_dim} to {d_model}")
        
        # Debug info
        logger.info(f"Initialized ResidualBlock with d_model={d_model}, d_time={d_time}")
    
    def _get_or_create_input_projection(self, input_dim: int) -> nn.Module:
        """
        Get an existing input projection or create a new one if needed.
        
        Args:
            input_dim: Input dimension
            
        Returns:
            Projection module for the input
        """
        dim_key = str(input_dim)
        
        # If the dimension matches d_model, no projection needed
        if input_dim == self.d_model:
            return nn.Identity()
        
        # If a projection already exists, use it
        if dim_key in self.input_projections:
            return self.input_projections[dim_key]
        
        # Create a new projection
        logger.info(f"Creating new input projection from {input_dim} to {self.d_model}")
        projection = FeatureProjection(input_dim, self.d_model).to(next(self.parameters()).device)
        self.input_projections[dim_key] = projection
        return projection
    def _get_or_create_time_projection(self, time_dim: int, target_dim: int) -> nn.Module:
        """
        Get an existing time projection or create a new one if needed.
        
        Args:
            time_dim: Time embedding dimension
            target_dim: Target dimension for projection
            
        Returns:
            Projection module for the time embedding
        """
        key = f"{time_dim}_{target_dim}"
        
        # If a projection already exists, use it
        if key in self.time_proj:
            return self.time_proj[key]
        
        # If dimensions match, use a simple activation
        if time_dim == target_dim:
            logger.info(f"Creating identity time projection (dimensions match: {time_dim})")
            projection = nn.Sequential(
                nn.Identity(),
                nn.SiLU()
            ).to(next(self.parameters()).device)
        else:
            # Create a new projection for mismatched dimensions
            logger.info(f"Creating new time projection from {time_dim} to {target_dim}")
            projection = nn.Sequential(
                nn.Linear(time_dim, target_dim),
                nn.SiLU()
            ).to(next(self.parameters()).device)
        
        # Add the new projection to our ModuleDict
        self.time_proj[key] = projection
        return projection
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the residual block.
            Forward pass of the residual block.
            
            Args:
                x: Input tensor of shape [batch_size, seq_len, d_model]
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            time_emb: Time embedding tensor of shape [batch_size, d_time]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Project input to d_model if necessary
        if input_dim != self.d_model:
            input_projection = self._get_or_create_input_projection(input_dim)
            x_projected = input_projection(x)
            logger.debug(f"Projected input from {input_dim} to {self.d_model}")
        else:
            x_projected = x
        
        # Store original input for residual connection
        residual = x
        
        # Project time embedding to match the input dimension
        time_embed_dim = time_emb.size(-1)
        
        # The target dimension for time embedding should match x_projected's dimension
        target_dim = x_projected.size(-1)
        
        # Log time embedding dimensions for debugging
        logger.debug(f"Time embedding dimension: {time_embed_dim}, target dimension: {target_dim}")
        
        # Create the key for our ModuleDict
        time_proj_key = f"{time_embed_dim}_{target_dim}"
        
        # If we don't have this projection yet, create it
        if time_proj_key not in self.time_proj:
            logger.info(f"Creating new time projection from {time_embed_dim} to {target_dim}")
            self.time_proj[time_proj_key] = nn.Sequential(
                nn.Linear(time_embed_dim, target_dim),
                nn.SiLU()
            ).to(x.device)
        
        # Apply the time projection
        time_embed = self.time_proj[time_proj_key](time_emb)
        
        # Expand time embedding to match sequence dimension
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        # First transformer block with dynamic layer norm
        h = self.transformer1(x_projected, mask)
        
        # Add time embedding - ensuring dimensions match
        if h.size(-1) != time_embed.size(-1):
            logger.warning(f"Shape mismatch: h {h.shape}, time_embed {time_embed.shape}")
            new_time_proj_key = f"{time_embed_dim}_{h.size(-1)}"
            if new_time_proj_key not in self.time_proj:
                logger.info(f"Creating emergency time projection from {time_embed_dim} to {h.size(-1)}")
                self.time_proj[new_time_proj_key] = nn.Sequential(
                    nn.Linear(time_embed_dim, h.size(-1)),
                    nn.SiLU()
                ).to(x.device)
            time_embed = self.time_proj[new_time_proj_key](time_emb)
            time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)
        
        h = h + time_embed
        
        # Second transformer block
        h = self.transformer2(h, mask)
        
        # Project back to original input dimension if needed
        if input_dim != self.d_model:
            # Create a projection for d_model to input_dim if it doesn't exist
            out_key = f"{self.d_model}_{input_dim}"
            if out_key not in self.input_projections:
                logger.info(f"Creating new output projection from {self.d_model} to {input_dim}")
                self.input_projections[out_key] = FeatureProjection(self.d_model, input_dim).to(x.device)
            
            # Project the transformed representation to the original dimension
            h_orig_dim = self.input_projections[out_key](h)
            logger.debug(f"Projected output from {self.d_model} to original dimension {input_dim}")
            
            # Ensure we have the correct LayerNorm for the original dimension
            dim_key = str(input_dim)
            if dim_key not in self.norm.norms:
                logger.info(f"Creating LayerNorm for dimension {input_dim}")
                self.norm.norms[dim_key] = nn.LayerNorm(input_dim).to(x.device)
            
            # Add residual connection after projection
            output = residual + h_orig_dim
            
            # Apply normalization with the correct dimension
            output = self.norm.norms[dim_key](output)
        else:
            # Same dimension case - just add residual and apply normalization
            output = residual + h
            output = self.norm(output)
        
        logger.debug(f"ResidualBlock output shape: {output.shape}")
        return output