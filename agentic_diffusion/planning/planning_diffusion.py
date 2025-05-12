"""
Planning diffusion module with AdaptDiffuser integration.

This module contains the implementation of diffusion-based planning capabilities
that leverage AdaptDiffuser for adaptive trajectory generation based on reward signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from abc import ABC, abstractmethod

from agentic_diffusion.core.diffusion_model import DiffusionModel
from agentic_diffusion.core.noise_schedules import NoiseScheduler
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.guidance import AdaptDiffuserGuidance
from agentic_diffusion.core.adapt_diffuser.utils import (
    encode_task,
    compute_reward_statistics,
    save_adaptation_metrics
)
from agentic_diffusion.planning.state_representations import (
    StateEncoder,
    StateRewardModel,
    TaskAdaptiveStateRepresentation
)
from agentic_diffusion.planning.action_space import (
    ActionSpace,
    ActionEncoder
)
from agentic_diffusion.planning.plan_validator import PlanValidator

# Configure logging
logger = logging.getLogger(__name__)


class PlanningDiffusionModel(DiffusionModel):
    """
    Diffusion model specialized for planning tasks.
    
    This model extends the base diffusion model with state-action specific capabilities
    for generating plans and trajectories.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 512, 512, 512],
        dropout: float = 0.1,
        state_conditioned: bool = True,
        task_conditioned: bool = True,
        task_dim: int = 0,
        activation: str = "silu",
        device: str = None
    ):
        """
        Initialize the planning diffusion model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            time_embedding_dim: Dimension for time step embeddings
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout rate for regularization
            state_conditioned: Whether to condition on states
            task_conditioned: Whether to condition on task embeddings
            task_dim: Dimension of task embeddings
            activation: Activation function to use
            device: Device to place model on
        """
        # Determine input/output dimensions
        if state_conditioned:
            input_dim = state_dim + action_dim
            output_dim = action_dim
        else:
            input_dim = action_dim
            output_dim = action_dim
            
        # Call parent initializer
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            time_embedding_dim=time_embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            device=device
        )
        
        # Store additional configuration
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_conditioned = state_conditioned
        self.task_conditioned = task_conditioned
        self.task_dim = task_dim
        
        # Additional conditioning if needed
        if task_conditioned and task_dim > 0:
            # Add task embedding layer
            self.task_projection = nn.Linear(task_dim, time_embedding_dim)
            
            # Modify first layer to include task conditioning
            first_hidden = hidden_dims[0]
            self.input_projection = nn.Linear(
                input_dim + time_embedding_dim * 2,  # Include task embedding
                first_hidden
            )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        task_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (noisy actions or state-action pairs)
            timesteps: Diffusion timesteps
            state: Optional state tensor for conditioning
            task_embed: Optional task embedding for conditioning
            
        Returns:
            Predicted noise or signal
            
        Raises:
            ValueError: If conditioning inputs are missing when required
        """
        # Check for required conditioning
        if self.state_conditioned and state is None:
            raise ValueError("State conditioning is required but state is None")
            
        if self.task_conditioned and task_embed is None and self.task_dim > 0:
            raise ValueError("Task conditioning is required but task_embed is None")
            
        # Create time embeddings
        temb = self.get_time_embedding(timesteps)
        
        # Handle task embedding if provided
        if self.task_conditioned and task_embed is not None and self.task_dim > 0:
            task_embedding = self.task_projection(task_embed)
            # Combine time and task embeddings
            combined_embedding = torch.cat([temb, task_embedding], dim=1)
        else:
            combined_embedding = temb
        
        # Handle state conditioning
        if self.state_conditioned and state is not None:
            # Ensure state has same batch dimension as x
            if state.shape[0] != x.shape[0]:
                if state.shape[0] == 1:
                    # Broadcast single state to batch
                    state = state.expand(x.shape[0], -1)
                else:
                    raise ValueError(f"State batch size {state.shape[0]} doesn't match input {x.shape[0]}")
                    
            # Concatenate state and action
            x = torch.cat([state, x], dim=1)
        
        # Pass through network
        h = self.input_projection(torch.cat([x, combined_embedding], dim=1))
        h = self.activation_fn(h)
        
        # Hidden layers with time embedding influence
        for i, (projection, projection_temb) in enumerate(zip(self.projection_layers, self.projection_time_embed)):
            h_temb = self.activation_fn(projection_temb(combined_embedding))
            h = self.activation_fn(projection(h) + h_temb)
            h = self.dropout(h)
        
        # Output projection
        output = self.output_projection(h)
        
        # If state conditioned, only return the action part
        if self.state_conditioned:
            output = output[:, :self.action_dim]
            
        return output


class TrajectoryModel:
    """
    Model for trajectory generation using diffusion.
    
    This class combines state and action representations with a diffusion
    model to generate trajectories.
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        action_space: ActionSpace,
        diffusion_model: PlanningDiffusionModel,
        noise_scheduler: NoiseScheduler,
        max_trajectory_length: int = 10,
        device: str = None
    ):
        """
        Initialize the trajectory model.
        
        Args:
            state_encoder: Encoder for state representations
            action_space: Action space for planning
            diffusion_model: Diffusion model for trajectories
            noise_scheduler: Noise schedule for diffusion
            max_trajectory_length: Maximum length of trajectories
            device: Device to place model on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_encoder = state_encoder
        self.action_space = action_space
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.max_trajectory_length = max_trajectory_length
        
        # Create action encoder
        self.action_encoder = ActionEncoder(
            action_space=action_space,
            max_sequence_length=max_trajectory_length,
            device=self.device
        )
    
    def encode_trajectory(
        self,
        states: List[Any],
        actions: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a trajectory of states and actions.
        
        Args:
            states: List of states in the trajectory
            actions: List of actions in the trajectory
            
        Returns:
            Tuple of (encoded states, encoded actions)
        """
        # Encode states
        encoded_states = torch.stack([
            self.state_encoder.encode(state) for state in states
        ])
        
        # Encode actions
        encoded_actions = self.action_encoder.encode_sequence(actions)
        
        return encoded_states, encoded_actions
    
    def decode_trajectory(
        self,
        encoded_states: torch.Tensor,
        encoded_actions: torch.Tensor
    ) -> Tuple[List[Any], List[Any]]:
        """
        Decode a trajectory of encoded states and actions.
        
        Args:
            encoded_states: Tensor of encoded states
            encoded_actions: Tensor of encoded actions
            
        Returns:
            Tuple of (decoded states, decoded actions)
        """
        # Decode states
        decoded_states = [
            self.state_encoder.decode(encoded_states[i])
            for i in range(encoded_states.shape[0])
        ]
        
        # Decode actions
        decoded_actions = self.action_encoder.decode_sequence(encoded_actions)
        
        return decoded_states, decoded_actions
    
    def generate_trajectory(
        self,
        initial_state: Any,
        task: Optional[Union[str, torch.Tensor]] = None,
        num_steps: int = 50,
        temperature: float = 1.0,
        guidance_scale: float = 3.0
    ) -> Tuple[List[Any], List[Any]]:
        """
        Generate a trajectory from an initial state.
        
        Args:
            initial_state: Starting state for trajectory
            task: Optional task description or embedding
            num_steps: Number of denoising steps
            temperature: Sampling temperature
            guidance_scale: Scale for guidance strength
            
        Returns:
            Tuple of (states, actions) forming a trajectory
        """
        # Encode initial state
        encoded_state = self.state_encoder.encode(initial_state).unsqueeze(0)
        
        # Encode task if provided
        task_embedding = None
        if task is not None:
            if isinstance(task, str) and hasattr(self.diffusion_model, 'task_embedding_model'):
                task_embedding = self.diffusion_model.task_embedding_model(task)
            elif isinstance(task, torch.Tensor):
                task_embedding = task.to(self.device)
                
        # Generate a sequence of actions using diffusion
        batch_size = 1
        shape = (batch_size, self.max_trajectory_length, self.action_space.action_dim)
        
        # Start with random noise
        x_t = torch.randn(shape, device=self.device) * temperature
        
        # Set up scheduler
        timesteps = self.noise_scheduler.get_sampling_timesteps(num_steps)
        
        # Denoising loop
        for t in timesteps:
            # Create batch of timesteps
            t_batch = torch.full((batch_size,), t, device=self.device)
            
            # Get model prediction (noise or signal) conditioned on state and task
            with torch.no_grad():
                model_output = self.diffusion_model(
                    x=x_t,
                    timesteps=t_batch,
                    state=encoded_state,
                    task_embed=task_embedding
                )
            
            # Update sample with scheduler update
            x_t = self.noise_scheduler.step(model_output, t, x_t)
        
        # Denormalize actions
        actions = self.action_encoder.decode_sequence(x_t)
        
        # Generate states from initial state and actions
        states = [initial_state]
        for action in actions:
            # Simple state transition for now
            # In a real system, this would use a dynamics model or simulator
            new_state = self._apply_action_to_state(states[-1], action)
            states.append(new_state)
        
        return states, actions
    
    def _apply_action_to_state(self, state: Any, action: Any) -> Any:
        """
        Apply an action to a state to get the next state.
        
        This is a placeholder that would be replaced with actual dynamics.
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            Next state after applying action
        """
        # Simple example of state transition
        if isinstance(state, torch.Tensor) and isinstance(action, torch.Tensor):
            # Apply simple dynamics (example only)
            return state + 0.1 * action
        else:
            # Default behavior for unknown types
            return state


class AdaptivePlanner:
    """
    Adaptive planner using AdaptDiffuser for trajectory generation.
    
    This class integrates the AdaptDiffuser framework with planning-specific components
    to generate adaptive plans and trajectories.
    """
    
    def __init__(
        self,
        state_representation: TaskAdaptiveStateRepresentation,
        action_space: ActionSpace,
        diffusion_model: PlanningDiffusionModel,
        noise_scheduler: NoiseScheduler,
        validator: Optional[PlanValidator] = None,
        max_trajectory_length: int = 10,
        guidance_scale: float = 3.0,
        guidance_min_step_percent: float = 0.1,
        guidance_max_step_percent: float = 0.9,
        learning_rate: float = 1e-5,
        checkpoint_dir: str = './checkpoints/planning',
        device: str = None,
        inference_steps: int = 50
    ):
        """
        Initialize the adaptive planner.
        
        Args:
            state_representation: State encoder and reward model
            action_space: Action space for planning
            diffusion_model: Diffusion model for trajectories
            noise_scheduler: Noise schedule for diffusion
            validator: Optional plan validator
            max_trajectory_length: Maximum length of trajectories
            guidance_scale: Scale for reward gradient guidance
            guidance_min_step_percent: Percentage of steps to start guidance
            guidance_max_step_percent: Percentage of steps to end guidance
            learning_rate: Learning rate for adaptation
            checkpoint_dir: Directory for checkpoints
            device: Device to place model on
            inference_steps: Number of steps for inference
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store components
        self.state_representation = state_representation
        self.action_space = action_space
        self.diffusion_model = diffusion_model.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.validator = validator
        self.max_trajectory_length = max_trajectory_length
        
        # Store configuration
        self.guidance_scale = guidance_scale
        self.guidance_min_step_percent = guidance_min_step_percent
        self.guidance_max_step_percent = guidance_max_step_percent
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.inference_steps = inference_steps
        
        # Create AdaptDiffuser instance
        self.adapt_diffuser = AdaptDiffuser(
            base_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            img_size=max_trajectory_length,
            channels=action_space.action_dim,
            reward_model=state_representation.reward_model,
            guidance_scale=guidance_scale,
            guidance_min_step_percent=guidance_min_step_percent,
            guidance_max_step_percent=guidance_max_step_percent,
            learning_rate=learning_rate,
            device=self.device,
            checkpoint_dir=checkpoint_dir,
            inference_steps=inference_steps
        )
        
        # Create action encoder
        self.action_encoder = ActionEncoder(
            action_space=action_space,
            max_sequence_length=max_trajectory_length,
            device=self.device
        )
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def plan(
        self,
        initial_state: Any,
        task: Optional[Union[str, torch.Tensor]] = None,
        num_samples: int = 10,
        temperature: float = 1.0,
        custom_guidance_scale: Optional[float] = None,
        return_all_samples: bool = False
    ) -> Union[Tuple[List[Any], List[Any]], List[Tuple[List[Any], List[Any]]]]:
        """
        Generate plans using adaptive diffusion.
        
        Args:
            initial_state: Starting state for planning
            task: Optional task description or embedding
            num_samples: Number of candidate plans to generate
            temperature: Sampling temperature
            custom_guidance_scale: Optional override for guidance scale
            return_all_samples: Whether to return all samples or just the best
            
        Returns:
            Best trajectory (states, actions) or list of all trajectories
        """
        # Encode task if provided
        task_embedding = None
        if task is not None:
            task_embedding = encode_task(
                task=task,
                task_embedding_model=getattr(self.diffusion_model, 'task_embedding_model', None),
                device=self.device
            )
        
        # Encode initial state
        encoded_state = self.state_representation.encode_states(
            initial_state, task_embedding
        ).unsqueeze(0)
        
        # Generate a batch of candidate action sequences
        batch_size = num_samples
        shape = (batch_size, self.max_trajectory_length, self.action_space.action_dim)
        
        # Generate action sequences using AdaptDiffuser
        conditioning = {"state": encoded_state}
        
        # Set effective guidance scale
        guidance = custom_guidance_scale if custom_guidance_scale is not None else self.guidance_scale
        
        # Generate samples
        encoded_actions = self.adapt_diffuser.generate(
            batch_size=batch_size,
            task=task_embedding,
            conditioning=conditioning,
            custom_guidance_scale=guidance,
            temperature=temperature
        )
        
        # Decode all action sequences
        all_action_sequences = []
        for i in range(batch_size):
            sequence = encoded_actions[i].unsqueeze(0)  # Add batch dimension
            actions = self.action_encoder.decode_sequence(sequence)
            all_action_sequences.append(actions)
        
        # Generate state sequences from initial state and actions
        all_trajectories = []
        for actions in all_action_sequences:
            states = [initial_state]
            for action in actions:
                # Apply action to get next state
                next_state = self._apply_action_to_state(states[-1], action)
                states.append(next_state)
            all_trajectories.append((states, actions))
        
        # If validator provided, score and rank trajectories
        if self.validator is not None and not return_all_samples:
            # Convert trajectories to tensor format for validation
            state_trajectories = []
            action_trajectories = []
            
            for states, actions in all_trajectories:
                # Encode states and actions
                encoded_states = torch.stack([
                    self.state_representation.encode_states(state, task_embedding)
                    for state in states[:-1]  # Exclude last state which has no action
                ])
                
                encoded_actions_seq = torch.stack([
                    self.action_space.normalize(action)
                    for action in actions
                ])
                
                state_trajectories.append(encoded_states)
                action_trajectories.append(encoded_actions_seq)
            
            # Pad to same length if needed
            max_len = max(s.shape[0] for s in state_trajectories)
            padded_states = []
            padded_actions = []
            
            for states, actions in zip(state_trajectories, action_trajectories):
                if states.shape[0] < max_len:
                    padding = torch.zeros(
                        (max_len - states.shape[0], states.shape[1]),
                        device=self.device
                    )
                    padded_states.append(torch.cat([states, padding], dim=0))
                else:
                    padded_states.append(states)
                    
                if actions.shape[0] < max_len:
                    padding = torch.zeros(
                        (max_len - actions.shape[0], actions.shape[1]),
                        device=self.device
                    )
                    padded_actions.append(torch.cat([actions, padding], dim=0))
                else:
                    padded_actions.append(actions)
            
            # Convert to batched tensor
            state_batch = torch.stack(padded_states)
            action_batch = torch.stack(padded_actions)
            
            # Validate and get scores
            _, validity_info = self.validator.validate(
                state_batch, action_batch, task_embedding
            )
            
            # Get scores (higher is better)
            if 'scores' in validity_info:
                scores = validity_info['scores']
            else:
                # Use default scoring if no scores provided
                rewards = self.state_representation.compute_rewards(
                    state_batch[:, -1], task_embedding  # Use final states
                )
                scores = rewards - self.validator.compute_violation_penalties(
                    state_batch, action_batch, task_embedding
                )
            
            # Get best trajectory
            best_idx = scores.argmax().item()
            best_trajectory = all_trajectories[best_idx]
            
            return best_trajectory
        
        # Return all samples or just the first if no validator
        if return_all_samples:
            return all_trajectories
        else:
            return all_trajectories[0]
    
    def adapt_to_task(
        self,
        task: Union[str, torch.Tensor],
        initial_states: List[Any] = None,
        demo_trajectories: Optional[List[Tuple[List[Any], List[Any]]]] = None,
        num_steps: int = 100,
        batch_size: int = 8,
        adapt_lr: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        feedback_frequency: int = 10,
        min_reward_threshold: float = 0.6,
        save_checkpoints: bool = True
    ) -> Dict[str, Any]:
        """
        Adapt the planner to a specific task.
        
        Args:
            task: Task description or embedding to adapt to
            initial_states: List of initial states for training
            demo_trajectories: Optional demonstration trajectories
            num_steps: Number of adaptation steps
            batch_size: Batch size for adaptation
            adapt_lr: Learning rate for adaptation (None = use default)
            guidance_scale: Override default guidance scale
            feedback_frequency: How often to evaluate and log progress
            min_reward_threshold: Minimum reward to consider adaptation successful
            save_checkpoints: Whether to save checkpoints during adaptation
            
        Returns:
            Dictionary with adaptation metrics
            
        Raises:
            ValueError: If no initial states or demonstrations provided
        """
        if initial_states is None and demo_trajectories is None:
            raise ValueError("Either initial_states or demo_trajectories must be provided")
        
        # Encode task
        task_embedding = encode_task(
            task=task,
            task_embedding_model=getattr(self.diffusion_model, 'task_embedding_model', None),
            device=self.device
        )
        
        # If demo trajectories provided, use them for learning
        if demo_trajectories is not None:
            logger.info("Using demonstration trajectories for adaptation")
            
            # Process demonstration trajectories
            encoded_trajectories = []
            for states, actions in demo_trajectories:
                # Encode states
                encoded_states = torch.stack([
                    self.state_representation.encode_states(state, task_embedding)
                    for state in states[:-1]  # Exclude last state which has no action
                ])
                
                # Encode actions
                encoded_actions = torch.stack([
                    self.action_space.normalize(action)
                    for action in actions
                ])
                
                # Store with states as conditioning
                encoded_trajectories.append((encoded_states, encoded_actions))
            
            # Prepare training data
            for step in range(num_steps):
                # Sample random demonstrations
                indices = torch.randint(0, len(encoded_trajectories), (batch_size,))
                batch_states = []
                batch_actions = []
                
                for idx in indices:
                    states, actions = encoded_trajectories[idx]
                    batch_states.append(states)
                    batch_actions.append(actions)
                
                # TODO: Implement training with demonstrations
                # This would require custom adaptation logic beyond AdaptDiffuser
        
        # Use AdaptDiffuser's task adaptation
        return self.adapt_diffuser.adapt_to_task(
            task=task_embedding,
            num_steps=num_steps,
            batch_size=batch_size,
            adapt_lr=adapt_lr,
            guidance_scale=guidance_scale,
            feedback_frequency=feedback_frequency,
            min_reward_threshold=min_reward_threshold,
            save_checkpoints=save_checkpoints
        )
    
    def _apply_action_to_state(self, state: Any, action: Any) -> Any:
        """
        Apply an action to a state to get the next state.
        
        This is a placeholder that would be replaced with actual dynamics.
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            Next state after applying action
        """
        # Simple example of state transition
        if isinstance(state, torch.Tensor) and isinstance(action, torch.Tensor):
            # Apply simple dynamics (example only)
            return state + 0.1 * action
        else:
            # Default behavior for unknown types
            return state
    
    def save(
        self,
        path: str
    ) -> bool:
        """
        Save the adaptive planner to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            Success flag
        """
        return self.adapt_diffuser.save(path)
    
    @classmethod
    def load(
        cls,
        path: str,
        state_representation: TaskAdaptiveStateRepresentation,
        action_space: ActionSpace,
        diffusion_model: PlanningDiffusionModel,
        noise_scheduler: NoiseScheduler,
        validator: Optional[PlanValidator] = None,
        device: str = None
    ) -> 'AdaptivePlanner':
        """
        Load an adaptive planner from disk.
        
        Args:
            path: Path to load the model from
            state_representation: State encoder and reward model
            action_space: Action space for planning
            diffusion_model: Diffusion model for trajectories
            noise_scheduler: Noise schedule for diffusion
            validator: Optional plan validator
            device: Device to load the model onto
            
        Returns:
            Loaded adaptive planner
        """
        # Load AdaptDiffuser model
        adapt_diffuser = AdaptDiffuser.load(
            path=path,
            base_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            reward_model=state_representation.reward_model,
            device=device
        )
        
        # Extract configuration
        guidance_scale = adapt_diffuser.guidance_scale
        guidance_min_step_percent = adapt_diffuser.guidance_min_step_percent
        guidance_max_step_percent = adapt_diffuser.guidance_max_step_percent
        learning_rate = adapt_diffuser.learning_rate
        checkpoint_dir = adapt_diffuser.checkpoint_dir
        inference_steps = adapt_diffuser.inference_steps
        max_trajectory_length = adapt_diffuser.img_size
        
        # Create new planner with loaded configuration
        planner = cls(
            state_representation=state_representation,
            action_space=action_space,
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            validator=validator,
            max_trajectory_length=max_trajectory_length,
            guidance_scale=guidance_scale,
            guidance_min_step_percent=guidance_min_step_percent,
            guidance_max_step_percent=guidance_max_step_percent,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            device=device,
            inference_steps=inference_steps
        )
        
        # Replace AdaptDiffuser instance
        planner.adapt_diffuser = adapt_diffuser
        
        return planner