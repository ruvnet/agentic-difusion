"""
Action space definitions for planning with AdaptDiffuser.

This module contains classes and functions for representing action spaces
in planning tasks, supporting the integration with AdaptDiffuser.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from abc import ABC, abstractmethod
import gym
from gym import spaces

# Configure logging
logger = logging.getLogger(__name__)


class ActionSpace(ABC):
    """
    Base class for action spaces in AdaptDiffuser planning.
    
    This abstract class defines the interface for action spaces, which transform
    actions between their native representation and a normalized representation
    suitable for diffusion models.
    """
    
    def __init__(
        self,
        action_dim: int,
        device: str = None
    ):
        """
        Initialize the action space.
        
        Args:
            action_dim: Dimension of the action space
            device: Device to use for computation
        """
        self.action_dim = action_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def normalize(self, action: Any) -> torch.Tensor:
        """
        Normalize an action to [-1, 1] range.
        
        Args:
            action: The action in its native form
            
        Returns:
            Normalized action tensor
        """
        pass
    
    @abstractmethod
    def denormalize(self, normalized_action: torch.Tensor) -> Any:
        """
        Convert a normalized action back to its native form.
        
        Args:
            normalized_action: Normalized action tensor
            
        Returns:
            Action in its native form
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: int = 1) -> Any:
        """
        Sample random actions from the space.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            Sampled actions
        """
        pass


class ContinuousActionSpace(ActionSpace):
    """
    Continuous action space for AdaptDiffuser planning.
    
    This class handles continuous actions with specified bounds.
    """
    
    def __init__(
        self,
        low: Union[float, np.ndarray, List[float]],
        high: Union[float, np.ndarray, List[float]],
        device: str = None
    ):
        """
        Initialize the continuous action space.
        
        Args:
            low: Lower bounds of the action space
            high: Upper bounds of the action space
            device: Device to use for computation
        """
        # Convert to numpy arrays if needed
        if isinstance(low, (float, int)):
            low = np.array([low])
        if isinstance(high, (float, int)):
            high = np.array([high])
            
        if isinstance(low, list):
            low = np.array(low)
        if isinstance(high, list):
            high = np.array(high)
            
        action_dim = low.shape[0]
        super().__init__(action_dim, device)
        
        # Store bounds as tensors
        self.low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(high, dtype=torch.float32, device=self.device)
        
        # Compute scaling factors for normalization
        self.scale = (self.high - self.low) / 2.0
        self.offset = (self.high + self.low) / 2.0
    
    def normalize(self, action: Union[np.ndarray, torch.Tensor, List[float]]) -> torch.Tensor:
        """
        Normalize an action to [-1, 1] range.
        
        Args:
            action: The action in its native range
            
        Returns:
            Normalized action tensor in [-1, 1] range
        """
        if isinstance(action, list):
            action = np.array(action)
            
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            
        # Ensure action has batch dimension
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Apply linear scaling
        normalized = (action - self.offset) / self.scale
        
        # Clip to ensure range [-1, 1]
        normalized = torch.clamp(normalized, -1.0, 1.0)
        
        return normalized
    
    def denormalize(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """
        Convert a normalized action back to its native range.
        
        Args:
            normalized_action: Normalized action tensor in [-1, 1] range
            
        Returns:
            Action in its native range
        """
        # Ensure normalized action has batch dimension
        if normalized_action.dim() == 1:
            normalized_action = normalized_action.unsqueeze(0)
            
        # Apply inverse scaling
        action = normalized_action * self.scale + self.offset
        
        # Clip to ensure within bounds
        action = torch.clamp(action, self.low, self.high)
        
        return action
    
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample random actions from the space.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            Sampled actions in native range
        """
        # Sample uniform random actions
        shape = (batch_size, self.action_dim)
        normalized = torch.rand(shape, device=self.device) * 2.0 - 1.0  # [-1, 1]
        
        # Convert to native range
        actions = self.denormalize(normalized)
        
        return actions


class DiscreteActionSpace(ActionSpace):
    """
    Discrete action space for AdaptDiffuser planning.
    
    This class handles discrete actions by converting them to one-hot encodings.
    """
    
    def __init__(
        self,
        n_actions: int,
        device: str = None
    ):
        """
        Initialize the discrete action space.
        
        Args:
            n_actions: Number of discrete actions
            device: Device to use for computation
        """
        super().__init__(n_actions, device)
        self.n_actions = n_actions
    
    def normalize(self, action: Union[int, np.ndarray, torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Convert discrete actions to one-hot encoding.
        
        Args:
            action: Discrete action indices
            
        Returns:
            One-hot encoded actions, scaled to [-1, 1] range
        """
        if isinstance(action, (int, np.integer)):
            action = [action]
            
        if isinstance(action, list):
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            
        # Ensure action has batch dimension
        if action.dim() == 0:
            action = action.unsqueeze(0)
            
        # Convert to one-hot
        one_hot = torch.zeros(action.shape[0], self.n_actions, device=self.device)
        one_hot.scatter_(1, action.unsqueeze(1), 1.0)
        
        # Scale to [-1, 1] range (optional, but consistent with continuous spaces)
        normalized = one_hot * 2.0 - 1.0
        
        return normalized
    
    def denormalize(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized actions back to discrete indices.
        
        Args:
            normalized_action: Normalized action tensor
            
        Returns:
            Discrete action indices
        """
        # Ensure normalized action has batch dimension
        if normalized_action.dim() == 1:
            normalized_action = normalized_action.unsqueeze(0)
            
        # Convert from [-1, 1] scale back to [0, 1]
        probabilities = (normalized_action + 1.0) / 2.0
        
        # Get the most likely action
        indices = torch.argmax(probabilities, dim=1)
        
        return indices
    
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample random actions from the space.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            Sampled discrete action indices
        """
        # Sample random indices
        indices = torch.randint(0, self.n_actions, (batch_size,), device=self.device)
        
        return indices


class GymActionSpaceAdapter(ActionSpace):
    """
    Adapter for Gym action spaces to work with AdaptDiffuser.
    
    This class wraps Gym action spaces (Box, Discrete, MultiDiscrete, etc.)
    and provides appropriate conversions.
    """
    
    def __init__(
        self,
        gym_space: gym.Space,
        device: str = None
    ):
        """
        Initialize the Gym action space adapter.
        
        Args:
            gym_space: Gym action space to adapt
            device: Device to use for computation
        """
        self.gym_space = gym_space
        
        if isinstance(gym_space, spaces.Box):
            action_dim = np.prod(gym_space.shape)
            self.space_type = "continuous"
            
            # Create internal continuous space
            self.internal_space = ContinuousActionSpace(
                low=gym_space.low,
                high=gym_space.high,
                device=device
            )
            
        elif isinstance(gym_space, spaces.Discrete):
            action_dim = gym_space.n
            self.space_type = "discrete"
            
            # Create internal discrete space
            self.internal_space = DiscreteActionSpace(
                n_actions=gym_space.n,
                device=device
            )
            
        elif isinstance(gym_space, spaces.MultiDiscrete):
            action_dim = sum(gym_space.nvec)
            self.space_type = "multidiscrete"
            
            # Store information for each dimension
            self.nvec = gym_space.nvec
            self.dims = len(self.nvec)
            
            # Calculate offsets for flattened one-hot encoding
            self.action_dims = list(self.nvec)
            self.action_offsets = [0]
            for dim in self.action_dims[:-1]:
                self.action_offsets.append(self.action_offsets[-1] + dim)
                
        else:
            raise ValueError(f"Unsupported Gym space type: {type(gym_space)}")
            
        super().__init__(action_dim, device)
    
    def normalize(self, action: Any) -> torch.Tensor:
        """
        Normalize a Gym action to diffusion model format.
        
        Args:
            action: The action in Gym format
            
        Returns:
            Normalized action tensor
        """
        if self.space_type in ["continuous", "discrete"]:
            return self.internal_space.normalize(action)
            
        elif self.space_type == "multidiscrete":
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.long, device=self.device)
                
            # Ensure action has batch dimension
            if action.dim() == 1:
                action = action.unsqueeze(0)
                
            batch_size = action.shape[0]
            
            # Create flattened one-hot encoding
            normalized = torch.full((batch_size, self.action_dim), -1.0, device=self.device)
            
            for i in range(self.dims):
                offset = self.action_offsets[i]
                indices = action[:, i]
                normalized[torch.arange(batch_size), offset + indices] = 1.0
                
            return normalized
    
    def denormalize(self, normalized_action: torch.Tensor) -> Any:
        """
        Convert a normalized action back to Gym format.
        
        Args:
            normalized_action: Normalized action tensor
            
        Returns:
            Action in Gym format
        """
        if self.space_type in ["continuous", "discrete"]:
            return self.internal_space.denormalize(normalized_action)
            
        elif self.space_type == "multidiscrete":
            # Ensure normalized action has batch dimension
            if normalized_action.dim() == 1:
                normalized_action = normalized_action.unsqueeze(0)
                
            batch_size = normalized_action.shape[0]
            
            # Convert to probabilities
            probs = (normalized_action + 1.0) / 2.0
            
            # Extract actions for each dimension
            action = torch.zeros((batch_size, self.dims), dtype=torch.long, device=self.device)
            
            for i in range(self.dims):
                offset = self.action_offsets[i]
                dim_size = self.action_dims[i]
                
                dim_probs = probs[:, offset:offset + dim_size]
                action[:, i] = torch.argmax(dim_probs, dim=1)
                
            return action
    
    def sample(self, batch_size: int = 1) -> Any:
        """
        Sample random actions from the space.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            Sampled actions in Gym format
        """
        if self.space_type in ["continuous", "discrete"]:
            return self.internal_space.sample(batch_size)
            
        elif self.space_type == "multidiscrete":
            # Sample each dimension independently
            action = torch.zeros((batch_size, self.dims), dtype=torch.long, device=self.device)
            
            for i in range(self.dims):
                action[:, i] = torch.randint(0, self.action_dims[i], (batch_size,), device=self.device)
                
            return action


class HybridActionSpace(ActionSpace):
    """
    Hybrid action space mixing continuous and discrete actions.
    
    This class allows combining different types of action spaces, which is useful
    for tasks with both continuous and discrete control aspects.
    """
    
    def __init__(
        self,
        action_spaces: List[ActionSpace],
        device: str = None
    ):
        """
        Initialize the hybrid action space.
        
        Args:
            action_spaces: List of component action spaces
            device: Device to use for computation
        """
        self.action_spaces = action_spaces
        self.num_spaces = len(action_spaces)
        
        # Calculate total action dimension
        total_dim = sum(space.action_dim for space in action_spaces)
        
        # Calculate offsets for each subspace
        self.dim_offsets = [0]
        for space in action_spaces[:-1]:
            self.dim_offsets.append(self.dim_offsets[-1] + space.action_dim)
            
        super().__init__(total_dim, device)
    
    def normalize(self, actions: List[Any]) -> torch.Tensor:
        """
        Normalize a list of actions from each subspace.
        
        Args:
            actions: List of actions, one for each subspace
            
        Returns:
            Combined normalized action tensor
        """
        if len(actions) != self.num_spaces:
            raise ValueError(f"Expected {self.num_spaces} actions, got {len(actions)}")
            
        # Normalize each action component
        normalized_actions = [
            self.action_spaces[i].normalize(actions[i])
            for i in range(self.num_spaces)
        ]
        
        # Ensure all have batch dimension and same batch size
        batch_sizes = [a.shape[0] for a in normalized_actions]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
            
        batch_size = batch_sizes[0]
        
        # Create combined action tensor
        combined = torch.full((batch_size, self.action_dim), -1.0, device=self.device)
        
        # Fill in each component
        for i, action in enumerate(normalized_actions):
            offset = self.dim_offsets[i]
            dim = self.action_spaces[i].action_dim
            combined[:, offset:offset + dim] = action
            
        return combined
    
    def denormalize(self, normalized_action: torch.Tensor) -> List[Any]:
        """
        Convert a normalized action back to a list of native actions.
        
        Args:
            normalized_action: Combined normalized action tensor
            
        Returns:
            List of denormalized actions, one for each subspace
        """
        # Ensure action has batch dimension
        if normalized_action.dim() == 1:
            normalized_action = normalized_action.unsqueeze(0)
            
        # Split into component actions
        actions = []
        for i in range(self.num_spaces):
            offset = self.dim_offsets[i]
            dim = self.action_spaces[i].action_dim
            
            component = normalized_action[:, offset:offset + dim]
            denormalized = self.action_spaces[i].denormalize(component)
            actions.append(denormalized)
            
        return actions
    
    def sample(self, batch_size: int = 1) -> List[Any]:
        """
        Sample random actions from each subspace.
        
        Args:
            batch_size: Number of actions to sample
            
        Returns:
            List of sampled actions, one for each subspace
        """
        return [space.sample(batch_size) for space in self.action_spaces]


class ActionEncoder:
    """
    Encoder for actions in planning tasks with AdaptDiffuser.
    
    This class handles encoding actions for input to diffusion models,
    including sequences of actions for trajectory planning.
    """
    
    def __init__(
        self,
        action_space: ActionSpace,
        max_sequence_length: int = 10,
        device: str = None
    ):
        """
        Initialize the action encoder.
        
        Args:
            action_space: The action space to encode
            max_sequence_length: Maximum length of action sequences
            device: Device to use for computation
        """
        self.action_space = action_space
        self.max_sequence_length = max_sequence_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode_action(self, action: Any) -> torch.Tensor:
        """
        Encode a single action.
        
        Args:
            action: Action to encode
            
        Returns:
            Encoded action tensor
        """
        return self.action_space.normalize(action)
    
    def decode_action(self, encoded_action: torch.Tensor) -> Any:
        """
        Decode an encoded action.
        
        Args:
            encoded_action: Encoded action tensor
            
        Returns:
            Decoded action
        """
        return self.action_space.denormalize(encoded_action)
    
    def encode_sequence(self, actions: List[Any]) -> torch.Tensor:
        """
        Encode a sequence of actions.
        
        Args:
            actions: List of actions to encode
            
        Returns:
            Encoded action sequence tensor
        """
        # Truncate if too long
        if len(actions) > self.max_sequence_length:
            actions = actions[:self.max_sequence_length]
            
        # Normalize each action
        normalized_actions = [self.action_space.normalize(a) for a in actions]
        
        # Ensure all have batch dimension
        for i, a in enumerate(normalized_actions):
            if a.dim() == 1:
                normalized_actions[i] = a.unsqueeze(0)
                
        # Pad to max length with -1 (neutral value in normalized space)
        action_dim = self.action_space.action_dim
        sequence_length = len(normalized_actions)
        batch_size = normalized_actions[0].shape[0]
        
        padded_sequence = torch.full(
            (batch_size, self.max_sequence_length, action_dim),
            -1.0,
            device=self.device
        )
        
        for i, action in enumerate(normalized_actions):
            padded_sequence[:, i, :] = action
            
        return padded_sequence
    
    def decode_sequence(self, encoded_sequence: torch.Tensor) -> List[Any]:
        """
        Decode an encoded action sequence.
        
        Args:
            encoded_sequence: Encoded action sequence tensor
            
        Returns:
            List of decoded actions
        """
        # Ensure sequence has batch dimension
        if encoded_sequence.dim() == 2:
            encoded_sequence = encoded_sequence.unsqueeze(0)
            
        batch_size, seq_length, _ = encoded_sequence.shape
        
        # Decode each step in the sequence
        actions = []
        for i in range(seq_length):
            action_encoded = encoded_sequence[:, i, :]
            action = self.action_space.denormalize(action_encoded)
            actions.append(action)
            
        return actions