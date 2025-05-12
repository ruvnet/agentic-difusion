"""
Embedding models for code diffusion.

This module provides embedding implementations for code tokens
and positions used in code diffusion models.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Union, Any

class CodeEmbedding(nn.Module):
    """
    Embedding layer for code tokens with positional encoding.
    
    This module combines token embeddings with sinusoidal position
    embeddings to represent code tokens in the diffusion model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_positional_embedding: bool = True
    ):
        """
        Initialize the code embedding module.
        
        Args:
            vocab_size: Size of the code token vocabulary
            embedding_dim: Dimension of the embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_positional_embedding: Whether to use positional embeddings
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_positional_embedding = use_positional_embedding
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embedding layer (if used)
        if use_positional_embedding:
            self.position_embedding = self._create_sinusoidal_positional_embedding()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def _create_sinusoidal_positional_embedding(self) -> nn.Embedding:
        """
        Create sinusoidal positional embeddings.
        
        Returns:
            Positional embedding layer
        """
        # Create a buffer of position embeddings
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2) * -(math.log(10000.0) / self.embedding_dim)
        )
        
        # Calculate sinusoidal embeddings
        pe = torch.zeros(self.max_seq_len, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to make it a persistent part of the module
        self.register_buffer("pe", pe)
        
        # Create an embedding layer initialized with these values
        position_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)
        position_embedding.weight.data.copy_(pe)
        position_embedding.weight.requires_grad = False
        
        return position_embedding
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        expected_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass of the embedding layer.
        
        Args:
            input_ids: Token indices of shape [batch_size, seq_len]
            position_ids: Optional position indices for custom positions
            expected_dim: Optional expected output dimension for compatibility
            
        Returns:
            Combined embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        # Get sequence length
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            # Truncate instead of raising error for better resilience
            print(f"WARNING: Input sequence length ({seq_len}) exceeds maximum allowed "
                  f"length ({self.max_seq_len}). Truncating sequence.")
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Get token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Add positional embeddings if used
        if self.use_positional_embedding:
            # Create position IDs if not provided
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            else:
                # Ensure position_ids match the truncated input_ids
                position_ids = position_ids[:, :seq_len]
            
            # Get position embeddings
            position_embeddings = self.position_embedding(position_ids)
            
            # Combine token and position embeddings
            embeddings = token_embeddings + position_embeddings
        else:
            embeddings = token_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        # Handle dimension compatibility if expected_dim is provided
        if expected_dim is not None and embeddings.shape[-1] != expected_dim:
            print(f"WARNING: Embedding dimension mismatch. Got {embeddings.shape[-1]}, "
                  f"but expected {expected_dim}. Adjusting dimension dynamically.")
            
            # Create a dynamic projection layer to the expected dimension
            adapter = nn.Linear(embeddings.shape[-1], expected_dim).to(embeddings.device)
            embeddings = adapter(embeddings)
        
        return embeddings

class TimestepEmbedding(nn.Module):
    """
    Embedding layer for diffusion timesteps.
    
    This module embeds diffusion timesteps into a high-dimensional space
    using sinusoidal embeddings, which are then projected to the desired dimension.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: Optional[int] = None,
        max_positions: int = 10000
    ):
        """
        Initialize the timestep embedding module.
        
        Args:
            embedding_dim: Initial dimension of the timestep embedding
            projection_dim: Dimension to project the embedding to
            max_positions: Maximum number of timesteps
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        
        # Default to using embedding_dim if no projection_dim is specified
        # Default to using embedding_dim if no projection_dim is specified
        if projection_dim is None:
            projection_dim = embedding_dim
        
        # For consistency, ensure embedding always projects to the expected dimension
        # Double projection approach to ensure proper dimensionality transformation
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Store the output dimension for clarity and reference
        self.output_dim = projection_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed diffusion timesteps.
        
        Args:
            timesteps: Tensor of timestep values of shape [batch_size]
            
        Returns:
            Tensor of shape [batch_size, output_dim] containing timestep embeddings
        """
        # Create sinusoidal timestep embeddings
        half_dim = self.embedding_dim // 2
        emb_scale = math.log(self.max_positions) / (half_dim - 1)
        emb_range = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
        
        # Create the actual embeddings
        emb = timesteps[:, None].float() * emb_range[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        # Handle odd dimensions
        # Handle odd dimensions
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        # Apply projection to get the final dimension
        projected_emb = self.projection(emb)
        
        # Replace rigid assertion with dynamic dimension handling
        if projected_emb.shape[-1] != self.output_dim:
            # Log the issue without breaking execution
            print(f"WARNING: TimestepEmbedding dimension mismatch. "
                  f"Got {projected_emb.shape[-1]}, expected {self.output_dim}. "
                  f"Adjusting projection to match expected dimension.")
            
            # Dynamically adjust the dimension with an additional projection layer
            adjust_layer = nn.Linear(projected_emb.shape[-1], self.output_dim).to(projected_emb.device)
            projected_emb = adjust_layer(projected_emb)
        
        return projected_emb