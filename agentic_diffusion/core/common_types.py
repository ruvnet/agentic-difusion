"""
Common type definitions and utilities to avoid circular imports.

This module contains shared type definitions, abstract base classes, and
utility functions used across the core diffusion modules.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Protocol, TypeVar, runtime_checkable

import torch

# Define type variables for more specific type hints
T = TypeVar('T')
ModelOutput = TypeVar('ModelOutput')

# Type aliases for improved readability
BatchTensor = torch.Tensor  # Shape: [batch_size, ...]
TimestepTensor = torch.Tensor  # Shape: [batch_size]
TaskEmbedding = Union[str, torch.Tensor]

# Trajectory types
Trajectory = torch.Tensor  # Shape: [num_timesteps, state_dim]
TrajectoryBatch = List[torch.Tensor]  # List of trajectories

@runtime_checkable
class RewardModelProtocol(Protocol):
    """Protocol defining the interface for reward models to avoid circular imports."""
    
    def compute_rewards(self, trajectories: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Compute rewards for a batch of trajectories."""
        ...
    
    def update(self, trajectories: List[torch.Tensor], rewards: torch.Tensor) -> None:
        """Update the reward model based on new trajectory-reward pairs."""
        ...
    
    def reset(self) -> None:
        """Reset the reward model to its initial state."""
        ...
        
    # Legacy methods for backward compatibility
    def compute_reward(self, samples: torch.Tensor, task_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rewards for samples.
        
        This is a compatibility method that maps to compute_rewards internally.
        """
        if hasattr(self, 'compute_rewards'):
            return self.compute_rewards(samples)
        ...
        
    def compute_reward_gradient(self, samples: torch.Tensor, task: Optional[Union[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute gradient of reward with respect to samples.
        
        Default implementation returns zeros.
        """
        return torch.zeros_like(samples)

@runtime_checkable
class TaskEmbeddingModelProtocol(Protocol):
    """Protocol defining the interface for task embedding models to avoid circular imports."""
    
    def encode(self, task_description: str) -> torch.Tensor:
        """Encode a task description into an embedding tensor."""
        ...

# Type aliases for improved readability
BatchTensor = torch.Tensor  # Shape: [batch_size, ...]
TimestepTensor = torch.Tensor  # Shape: [batch_size]
TaskEmbedding = Union[str, torch.Tensor]

# Trajectory types
Trajectory = torch.Tensor  # Shape: [num_timesteps, state_dim]
TrajectoryBatch = List[torch.Tensor]  # List of trajectories