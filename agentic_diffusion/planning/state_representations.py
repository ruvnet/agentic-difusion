"""
State representations for planning with AdaptDiffuser.

This module contains classes and functions for representing states in planning
tasks, supporting the integration with AdaptDiffuser for state-based planning.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

# Forward declarations for test compatibility
LinearStateEncoder = None  # Will be aliased later
AdaptiveStateEncoder = None  # Will be aliased later

class StateEncoder(ABC):
    """
    Base class for encoding states into latent representations for AdaptDiffuser.
    
    This abstract class defines the interface for state encoders, which transform
    states from their native representation into a format suitable for diffusion models.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        device: str = None
    ):
        """
        Initialize the state encoder.
        
        Args:
            state_dim: Dimension of the input state
            latent_dim: Dimension of the latent representation
            device: Device to use for computation
        """
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def encode(self, state: Any) -> torch.Tensor:
        """
        Encode a state into a latent representation.
        
        Args:
            state: The state to encode
            
        Returns:
            Latent representation of the state
        """
        pass
    
    @abstractmethod
    def decode(self, latent: torch.Tensor) -> Any:
        """
        Decode a latent representation back into a state.
        
        Args:
            latent: The latent representation to decode
            
        Returns:
            Reconstructed state
        """
        pass


class IdentityEncoder(StateEncoder):
    """
    Identity encoder that passes states through unchanged.
    
    This is a simple encoder that assumes states are already in the right format.
    It's useful for testing and when states don't need transformation.
    """
    
    def __init__(
        self,
        state_dim: int,
        device: str = None
    ):
        """
        Initialize the identity encoder.
        
        Args:
            state_dim: Dimension of the state
            device: Device to use for computation
        """
        super().__init__(
            state_dim=state_dim,
            latent_dim=state_dim,
            device=device
        )
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Pass the state through unchanged.
        
        Args:
            state: The state to encode
            
        Returns:
            The same state tensor
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        return state.to(self.device)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Pass the latent representation through unchanged.
        
        Args:
            latent: The latent representation to decode
            
        Returns:
            The same latent tensor
        """
        return latent


class LinearEncoder(StateEncoder):
    """
    Linear encoder that projects states to a different dimension.
    
    This simple encoder uses a linear transformation to project states
    to a latent space of potentially different dimensionality.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        device: str = None
    ):
        """
        Initialize the linear encoder.
        
        Args:
            state_dim: Dimension of the input state
            latent_dim: Dimension of the latent representation
            device: Device to use for computation
        """
        super().__init__(
            state_dim=state_dim,
            latent_dim=latent_dim,
            device=device
        )
        
        # Initialize encoding and decoding layers
        self.encoder = torch.nn.Linear(state_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, state_dim)
        
        # Move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode a state using linear projection.
        
        Args:
            state: The state to encode
            
        Returns:
            Latent representation of the state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        return self.encoder(state.to(self.device))
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation using linear projection.
        
        Args:
            latent: The latent representation to decode
            
        Returns:
            Reconstructed state
        """
        return self.decoder(latent.to(self.device))


class NonlinearEncoder(StateEncoder):
    """
    Nonlinear encoder that uses a multi-layer neural network.
    
    This encoder uses nonlinear transformations to project states
    to a latent space, potentially capturing more complex features.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: Callable = torch.nn.ReLU(),
        device: str = None
    ):
        """
        Initialize the nonlinear encoder.
        
        Args:
            state_dim: Dimension of the input state
            latent_dim: Dimension of the latent representation
            hidden_dims: List of hidden layer dimensions
            activation: Activation function to use
            device: Device to use for computation
        """
        super().__init__(
            state_dim=state_dim,
            latent_dim=latent_dim,
            device=device
        )
        
        # Build encoder network
        encoder_layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(activation)
            in_dim = hidden_dim
        
        encoder_layers.append(torch.nn.Linear(in_dim, latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # Build decoder network
        decoder_layers = []
        in_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(torch.nn.Linear(in_dim, hidden_dim))
            decoder_layers.append(activation)
            in_dim = hidden_dim
        
        decoder_layers.append(torch.nn.Linear(in_dim, state_dim))
        self.decoder = torch.nn.Sequential(*decoder_layers)
        
        # Move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode a state using nonlinear projection.
        
        Args:
            state: The state to encode
            
        Returns:
            Latent representation of the state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        return self.encoder(state.to(self.device))
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation using nonlinear projection.
        
        Args:
            latent: The latent representation to decode
            
        Returns:
            Reconstructed state
        """
        return self.decoder(latent.to(self.device))


class StateRewardModel:
    """
    Simple model that computes rewards from states and task embeddings.
    
    This implements a concrete model for computing rewards, extending the
    abstract base class with a simple neural network implementation.
    """
    
    def __init__(
        self,
        state_dim: int,
        task_dim: int = 0,
        hidden_dims: List[int] = [64, 32],
        device: str = None
    ):
        """
        Initialize the state reward model.
        
        Args:
            state_dim: Dimension of the state
            task_dim: Dimension of task embeddings
            hidden_dims: Hidden layer dimensions
            device: Device to use for computation
        """
        self.state_dim = state_dim
        self.task_dim = task_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build reward network
        input_dim = state_dim + task_dim if task_dim > 0 else state_dim
        self.layers = []
        
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
        
        self.layers.append(torch.nn.Linear(in_dim, 1))
        self.reward_network = torch.nn.Sequential(*self.layers).to(self.device)
    
    def compute_reward(
        self,
        states: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute rewards for states.
        
        Args:
            states: States to evaluate
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Rewards for each state
        """
        # Ensure states is a tensor
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=self.device, dtype=torch.float32)
        
        # Reshape if needed
        if states.dim() == 1:
            states = states.unsqueeze(0)
            
        # Process task embedding if provided
        if self.task_dim > 0 and task_embedding is not None:
            if task_embedding.dim() == 1:
                task_embedding = task_embedding.unsqueeze(0)
                
            # Expand task embedding if batch sizes don't match
            if task_embedding.shape[0] == 1 and states.shape[0] > 1:
                task_embedding = task_embedding.expand(states.shape[0], -1)
                
            # Concatenate states and task embedding
            inputs = torch.cat([states, task_embedding], dim=1).to(self.device)
        else:
            inputs = states.to(self.device)
        
        # Compute rewards
        rewards = self.reward_network(inputs).squeeze(-1)
        return rewards
    
    def compute_reward_gradient(
        self,
        states: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradients of rewards with respect to states.
        
        Args:
            states: States to evaluate
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Gradients of rewards with respect to states
        """
        # Enable gradients for states
        states.requires_grad_(True)
            
        # Compute rewards
        rewards = self.compute_reward(states, task_embedding)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=rewards.sum(),
            inputs=states,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return gradients


class TaskAdaptiveStateRepresentation:
    """
    Task-adaptive state representation for planning with AdaptDiffuser.
    
    This class combines state encoding with reward modeling for task-adaptive
    planning using AdaptDiffuser.
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        reward_model: StateRewardModel,
        device: str = None
    ):
        """
        Initialize the task-adaptive state representation.
        
        Args:
            state_encoder: Encoder for converting states to latents
            reward_model: Model for computing state rewards
            device: Device to use for computation
        """
        self.state_encoder = state_encoder
        self.reward_model = reward_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def encode_states(
        self,
        states: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode states into latent representations.
        
        Args:
            states: States to encode
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Latent representations of states
        """
        if hasattr(self.state_encoder, 'encode') and callable(getattr(self.state_encoder, 'encode')):
            if 'task_embedding' in self.state_encoder.encode.__code__.co_varnames:
                return self.state_encoder.encode(states, task_embedding)
            else:
                return self.state_encoder.encode(states)
        else:
            raise NotImplementedError("State encoder does not implement encode method")
    
    def decode_latents(
        self,
        latents: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode latent representations into states.
        
        Args:
            latents: Latent representations to decode
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Reconstructed states
        """
        if hasattr(self.state_encoder, 'decode') and callable(getattr(self.state_encoder, 'decode')):
            if 'task_embedding' in self.state_encoder.decode.__code__.co_varnames:
                return self.state_encoder.decode(latents, task_embedding)
            else:
                return self.state_encoder.decode(latents)
        else:
            raise NotImplementedError("State encoder does not implement decode method")
    
    def compute_rewards(
        self,
        states: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute rewards for states.
        
        Args:
            states: States to evaluate
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Rewards for each state
        """
        if hasattr(self.reward_model, 'compute_reward') and callable(getattr(self.reward_model, 'compute_reward')):
            if 'task_embedding' in self.reward_model.compute_reward.__code__.co_varnames:
                return self.reward_model.compute_reward(states, task_embedding)
            else:
                return self.reward_model.compute_reward(states)
        else:
            raise NotImplementedError("Reward model does not implement compute_reward method")
    
    def compute_reward_gradients(
        self,
        states: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradients of rewards with respect to states.
        
        Args:
            states: States to evaluate
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Gradients of rewards with respect to states
        """
        if hasattr(self.reward_model, 'compute_reward_gradient') and callable(getattr(self.reward_model, 'compute_reward_gradient')):
            if 'task_embedding' in self.reward_model.compute_reward_gradient.__code__.co_varnames:
                return self.reward_model.compute_reward_gradient(states, task_embedding)
            else:
                return self.reward_model.compute_reward_gradient(states)
        
        # Fallback implementation if the reward model doesn't implement compute_reward_gradient
        states = states.detach().clone().requires_grad_(True)
        rewards = self.compute_rewards(states, task_embedding)
        
        return torch.autograd.grad(
            outputs=rewards.sum(),
            inputs=states,
            create_graph=True,
            retain_graph=True
        )[0]
    
    def adapt_reward_model(
        self,
        demo_states: torch.Tensor,
        demo_rewards: torch.Tensor,
        num_steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Adapt the reward model to demonstration data.
        
        Args:
            demo_states: Demonstration states, shape [num_demos, state_dim]
            demo_rewards: Rewards for demonstrations, shape [num_demos]
            num_steps: Number of adaptation steps
            learning_rate: Learning rate for adaptation
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Adapting reward model using {len(demo_states)} demonstrations")
        
        # Encode states if they are not already encoded
        if not isinstance(demo_states, torch.Tensor):
            encoded_states = self.encode_states(demo_states)
        else:
            encoded_states = demo_states
        
        # Prepare optimizer
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        
        # Training metrics
        metrics = {
            "loss": []
        }
        
        # Training loop
        for step in range(num_steps):
            # Sample random batch
            if batch_size < len(encoded_states):
                indices = torch.randperm(len(encoded_states))[:batch_size]
                states_batch = encoded_states[indices]
                rewards_batch = demo_rewards[indices]
            else:
                states_batch = encoded_states
                rewards_batch = demo_rewards
            
            # Forward pass
            predicted_rewards = self.reward_model.compute_reward(states_batch)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(predicted_rewards, rewards_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record metrics
            metrics["loss"].append(loss.item())
            
            # Log progress
            if (step + 1) % (num_steps // 10) == 0 or step == 0:
                logger.info(f"Adaptation step {step+1}/{num_steps}, Loss: {loss.item():.6f}")
        
        return metrics
    
    def compute_batch_rewards(
        self,
        trajectories: List[List[torch.Tensor]],
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute rewards for batches of trajectories.
        
        Args:
            trajectories: List of trajectories, each a list of states
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Tensor of rewards for each trajectory
        """
        # Compute total reward for each trajectory
        rewards = []
        
        for trajectory in trajectories:
            # Skip empty trajectories
            if not trajectory:
                rewards.append(torch.tensor(0.0, device=self.device))
                continue
                
            # Encode states if not already encoded
            encoded_states = [
                self.encode_states(state, task_embedding) 
                if not isinstance(state, torch.Tensor) or state.shape[-1] != self.reward_model.state_dim
                else state
                for state in trajectory
            ]
            
            # Compute rewards for each state
            state_rewards = [
                self.compute_rewards(state, task_embedding)
                for state in encoded_states
            ]
            
            # Sum rewards across trajectory
            trajectory_reward = sum(state_rewards)
            rewards.append(trajectory_reward)
        
        return torch.stack(rewards)
    
    def compute_reward_gradients_batch(
        self,
        states_batch: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward gradients for a batch of states.
        
        Args:
            states_batch: Batch of states, shape [batch_size, ...state_dims]
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Batch of reward gradients
        """
        batch_size = states_batch.shape[0]
        
        # Ensure gradients are enabled
        states = states_batch.detach().clone().requires_grad_(True)
        
        # Compute rewards
        rewards = self.compute_rewards(states, task_embedding)
        
        # Initialize gradients tensor
        gradients = torch.zeros_like(states)
        
        # Compute gradients for each state in batch
        for i in range(batch_size):
            # Zero out gradients
            if states.grad is not None:
                states.grad.zero_()
            
            # Backward pass for this state
            rewards[i].backward(retain_graph=(i < batch_size - 1))
            
            # Store gradient
            if states.grad is not None:
                gradients[i] = states.grad[i].clone()
        
        return gradients
    
    def update_weights(self, new_weights: Dict[str, torch.Tensor]) -> None:
        """
        Update specific weights in the model.
        
        This is useful for adaptation algorithms that directly modify weights
        rather than using gradient descent.
        
        Args:
            new_weights: Dictionary of parameter name -> new tensor value
        """
        with torch.no_grad():
            for name, param in self.reward_model.named_parameters():
                if name in new_weights:
                    param.copy_(new_weights[name])
    
    def compute_reward_hessian(
        self,
        state: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Hessian of the reward function at a given state.
        
        This is useful for second-order optimization methods in planning.
        
        Args:
            state: State to evaluate at
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Hessian matrix of reward function at state
        """
        state = state.detach().clone().requires_grad_(True)
        
        # First-order gradients
        first_grads = self.compute_reward_gradients(state, task_embedding)
        
        # Initialize Hessian
        state_dim = state.shape[-1]
        hessian = torch.zeros(state_dim, state_dim, device=self.device)
        
        # Compute Hessian row by row
        for i in range(state_dim):
            # Compute gradient of the i-th gradient component
            if first_grads[..., i].requires_grad:
                second_grads = torch.autograd.grad(
                    first_grads[..., i],
                    state,
                    retain_graph=True
                )[0]
                hessian[i] = second_grads
        
        return hessian


# Class aliases for test compatibility
LinearStateEncoder = LinearEncoder
# AdaptiveStateEncoder needs to be implemented as it doesn't exist yet
class AdaptiveStateEncoder(StateEncoder):
    """
    Adaptive state encoder with task conditioning.
    
    This encoder uses task embeddings to condition the encoding and decoding
    processes, allowing for task-specific state representations.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        task_dim: int,
        device: str = None
    ):
        """
        Initialize the adaptive state encoder.
        
        Args:
            state_dim: Dimension of the input state
            latent_dim: Dimension of the latent representation
            task_dim: Dimension of the task embedding
            device: Device to use for computation
        """
        super().__init__(
            state_dim=state_dim,
            latent_dim=latent_dim,
            device=device
        )
        self.task_dim = task_dim
        
        # Initialize encoding and decoding layers
        self.encoder_state = torch.nn.Linear(state_dim, latent_dim)
        self.encoder_task = torch.nn.Linear(task_dim, latent_dim)
        self.encoder_combined = torch.nn.Linear(latent_dim * 2, latent_dim)
        
        self.decoder_latent = torch.nn.Linear(latent_dim, latent_dim)
        self.decoder_task = torch.nn.Linear(task_dim, latent_dim)
        self.decoder_combined = torch.nn.Linear(latent_dim * 2, state_dim)
        
        # Move to device
        self.encoder_state.to(self.device)
        self.encoder_task.to(self.device)
        self.encoder_combined.to(self.device)
        self.decoder_latent.to(self.device)
        self.decoder_task.to(self.device)
        self.decoder_combined.to(self.device)
        
        # For normalization
        self.state_mean = None
        self.state_std = None
    
    def encode(self, state: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a state using task conditioning.
        
        Args:
            state: The state to encode
            task: The task embedding for conditioning
            
        Returns:
            Latent representation of the state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
        
        # Reshape if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        if task is not None and task.dim() == 1:
            task = task.unsqueeze(0)
            
        # Normalize state if statistics are available
        if self.state_mean is not None and self.state_std is not None:
            state = self.normalize_state(state)
        
        # Encode state
        state_features = self.encoder_state(state.to(self.device))
        
        # Handle task embedding
        if task is None:
            # If no task is provided, use zeros
            batch_size = state.shape[0]
            task_features = torch.zeros(batch_size, self.latent_dim, device=self.device)
        else:
            # Encode task
            task_features = self.encoder_task(task.to(self.device))
        
        # Combine features
        combined = torch.cat([state_features, task_features], dim=1)
        latent = self.encoder_combined(combined)
        
        return latent
    
    def decode(self, latent: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode a latent representation using task conditioning.
        
        Args:
            latent: The latent representation to decode
            task: The task embedding for conditioning
            
        Returns:
            Reconstructed state
        """
        # Process latent
        latent_features = self.decoder_latent(latent.to(self.device))
        
        # Handle task embedding
        if task is None:
            # If no task is provided, use zeros
            batch_size = latent.shape[0]
            task_features = torch.zeros(batch_size, self.latent_dim, device=self.device)
        else:
            # If task is 1D, add batch dimension
            if task.dim() == 1:
                task = task.unsqueeze(0)
            # Encode task
            task_features = self.decoder_task(task.to(self.device))
        
        # Combine features
        combined = torch.cat([latent_features, task_features], dim=1)
        state = self.decoder_combined(combined)
        
        # Denormalize state if statistics are available
        if self.state_mean is not None and self.state_std is not None:
            state = self.denormalize_state(state)
        
        return state
    
    def update_normalization_stats(self, states: torch.Tensor) -> None:
        """
        Update normalization statistics based on states.
        
        Args:
            states: Batch of states to compute statistics from
        """
        with torch.no_grad():
            self.state_mean = states.mean(dim=0)
            self.state_std = states.std(dim=0) + 1e-6  # Add small constant for numerical stability
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state using stored statistics.
        
        Args:
            state: State to normalize
            
        Returns:
            Normalized state
        """
        if self.state_mean is None or self.state_std is None:
            return state
        return (state - self.state_mean) / self.state_std
    
    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Denormalize state using stored statistics.
        
        Args:
            state: State to denormalize
            
        Returns:
            Denormalized state
        """
        if self.state_mean is None or self.state_std is None:
            return state
        return state * self.state_std + self.state_mean