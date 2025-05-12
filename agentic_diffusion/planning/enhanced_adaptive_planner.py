"""
Enhanced adaptive planner implementation for AdaptDiffuser.

This module provides the enhanced adaptive planning capabilities based on AdaptDiffuser,
with support for multiple guidance methods, trajectory optimization, and multi-task adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from agentic_diffusion.core.diffusion_model import DiffusionModel
from agentic_diffusion.core.noise_schedules import NoiseScheduler
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.guidance import AdaptDiffuserGuidance
from agentic_diffusion.core.adapt_diffuser.multi_task import MultiTaskAdaptDiffuser
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
from agentic_diffusion.planning.planning_diffusion import AdaptivePlanner
from agentic_diffusion.planning.guidance_strategies import (
    GuidanceStrategy,
    ClassifierFreeGuidance,
    ConstraintGuidance,
    MultiObjectiveGuidance,
    ProgressiveGuidance
)

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedAdaptivePlanner(AdaptivePlanner):
    """
    Enhanced adaptive planner with advanced trajectory generation capabilities.
    
    This planner extends the base AdaptivePlanner with additional guidance strategies,
    hierarchical planning, and multi-objective optimization.
    """
    
    def __init__(
        self,
        state_representation: TaskAdaptiveStateRepresentation,
        action_space: ActionSpace,
        diffusion_model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        validator: Optional[PlanValidator] = None,
        max_trajectory_length: int = 10,
        guidance_scale: float = 3.0,
        guidance_min_step_percent: float = 0.1,
        guidance_max_step_percent: float = 0.9,
        learning_rate: float = 1e-5,
        checkpoint_dir: str = './checkpoints/planning',
        device: str = None,
        inference_steps: int = 50,
        guidance_strategy: Union[GuidanceStrategy, str] = GuidanceStrategy.REWARD_GRADIENT,
        multi_objective_weights: Optional[Dict[str, float]] = None,
        dynamics_model: Optional[Callable] = None,
        use_hierarchical_planning: bool = False,
        segment_length: int = 5,
        use_multi_task: bool = False,
        num_tasks: int = 1,
        task_dim: int = 16
    ):
        """
        Initialize the enhanced adaptive planner.
        
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
            guidance_strategy: Strategy for guiding diffusion
            multi_objective_weights: Weights for multi-objective optimization
            dynamics_model: Optional model for state transitions
            use_hierarchical_planning: Whether to use hierarchical planning
            segment_length: Length of trajectory segments for hierarchical planning
            use_multi_task: Whether to use multi-task adaptation
            num_tasks: Number of tasks for multi-task model
            task_dim: Dimension of task embeddings
        """
        # Call parent initializer
        super().__init__(
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
        
        # Store additional configuration
        self.guidance_strategy = guidance_strategy if isinstance(guidance_strategy, GuidanceStrategy) else GuidanceStrategy(guidance_strategy)
        self.multi_objective_weights = multi_objective_weights or {}
        self.dynamics_model = dynamics_model
        self.use_hierarchical_planning = use_hierarchical_planning
        self.segment_length = segment_length
        self.use_multi_task = use_multi_task
        
        # Create adapt diffuser based on configuration
        if self.use_multi_task:
            # Replace with multi-task adapt diffuser
            self.adapt_diffuser = MultiTaskAdaptDiffuser(
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
                inference_steps=inference_steps,
                num_tasks=num_tasks,
                task_dim=task_dim
            )
        else:
            # Set up custom guidance based on strategy
            guidance_cls = self._get_guidance_class(self.guidance_strategy)
            
            # Create custom guidance
            guidance = guidance_cls(
                reward_model=state_representation.reward_model,
                guidance_scale=guidance_scale,
                device=self.device
            )
            
            # Create AdaptDiffuser with custom guidance
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
                inference_steps=inference_steps,
                guidance=guidance
            )
    
    def _get_guidance_class(self, strategy: GuidanceStrategy) -> type:
        """
        Get the appropriate guidance class for the given strategy.
        
        Args:
            strategy: The guidance strategy to use
            
        Returns:
            The guidance class for the strategy
            
        Raises:
            ValueError: If the strategy is not recognized
        """
        if strategy == GuidanceStrategy.REWARD_GRADIENT:
            # Standard reward gradient guidance
            return AdaptDiffuserGuidance
        elif strategy == GuidanceStrategy.CLASSIFIER_FREE:
            # Classifier-free guidance
            return ClassifierFreeGuidance
        elif strategy == GuidanceStrategy.CONSTRAINT_GUIDED:
            # Constraint-guided diffusion
            return ConstraintGuidance
        elif strategy == GuidanceStrategy.MULTI_OBJECTIVE:
            # Multi-objective guidance
            return MultiObjectiveGuidance
        elif strategy == GuidanceStrategy.PROGRESSIVE:
            # Progressive guidance
            return ProgressiveGuidance
        else:
            raise ValueError(f"Unknown guidance strategy: {strategy}")
            
    def plan_hierarchical(
        self,
        initial_state: Any,
        task: Optional[Union[str, torch.Tensor]] = None,
        num_samples: int = 10,
        temperature: float = 1.0,
        custom_guidance_scale: Optional[float] = None,
        max_segments: int = 5
    ) -> Tuple[List[Any], List[Any]]:
        """
        Generate plans hierarchically by planning segments sequentially.
        
        Args:
            initial_state: Starting state for planning
            task: Optional task description or embedding
            num_samples: Number of candidate plans to generate per segment
            temperature: Sampling temperature
            custom_guidance_scale: Optional override for guidance scale
            max_segments: Maximum number of segments to generate
            
        Returns:
            Complete trajectory (states, actions)
        """
        if not self.use_hierarchical_planning:
            logger.warning("Hierarchical planning requested but not enabled in initialization. Using regular planning.")
            return self.plan(
                initial_state=initial_state,
                task=task,
                num_samples=num_samples,
                temperature=temperature,
                custom_guidance_scale=custom_guidance_scale
            )
        
        # Encode task if provided
        task_embedding = None
        if task is not None:
            task_embedding = encode_task(
                task=task,
                task_embedding_model=getattr(self.diffusion_model, 'task_embedding_model', None),
                device=self.device
            )
        
        # Start with initial state
        current_state = initial_state
        all_states = [current_state]
        all_actions = []
        
        # Generate plan segment by segment
        for segment_idx in range(max_segments):
            logger.info(f"Planning segment {segment_idx + 1}/{max_segments}")
            
            # Generate this segment using regular planning
            segment_states, segment_actions = self.plan(
                initial_state=current_state,
                task=task_embedding,
                num_samples=num_samples,
                temperature=temperature,
                custom_guidance_scale=custom_guidance_scale,
                segment_length=self.segment_length
            )
            
            # Update current state to the last state in the segment
            current_state = segment_states[-1]
            
            # Add segment to overall trajectory (skip the first state as it's duplicate)
            all_states.extend(segment_states[1:])
            all_actions.extend(segment_actions)
            
            # Check if task is complete (e.g., reached goal state or max length)
            if self._check_termination(current_state, task_embedding):
                logger.info(f"Task completed after {segment_idx + 1} segments")
                break
        
        return all_states, all_actions
    
    def _check_termination(self, state: Any, task_embedding: Optional[torch.Tensor]) -> bool:
        """
        Check if the current state represents task completion.
        
        Args:
            state: Current state
            task_embedding: Task embedding
            
        Returns:
            True if task is complete, False otherwise
        """
        # Default implementation - can be overridden in subclasses
        return False
        
    def adapt_to_task_with_demonstrations(
        self,
        task: Union[str, torch.Tensor],
        demo_trajectories: List[Tuple[List[Any], List[Any]]],
        num_steps: int = 100,
        batch_size: int = 8,
        adapt_lr: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        feedback_frequency: int = 10,
        min_reward_threshold: float = 0.6,
        save_checkpoints: bool = True,
        noise_level: float = 0.05,
        alpha_kl: float = 0.1
    ) -> Dict[str, Any]:
        """
        Adapt the planner to a specific task using demonstration trajectories.
        
        Args:
            task: Task description or embedding to adapt to
            demo_trajectories: Demonstration trajectories (states, actions)
            num_steps: Number of adaptation steps
            batch_size: Batch size for adaptation
            adapt_lr: Learning rate for adaptation (None = use default)
            guidance_scale: Override default guidance scale
            feedback_frequency: How often to evaluate and log progress
            min_reward_threshold: Minimum reward to consider adaptation successful
            save_checkpoints: Whether to save checkpoints during adaptation
            noise_level: Level of noise to add during training
            alpha_kl: Weight for KL divergence term
            
        Returns:
            Dictionary with adaptation metrics
        """
        logger.info("Adapting to task with demonstration trajectories")
        
        # Encode task
        task_embedding = encode_task(
            task=task,
            task_embedding_model=getattr(self.diffusion_model, 'task_embedding_model', None),
            device=self.device
        )
        
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
            
            # Store as (states, actions) pair
            encoded_trajectories.append((encoded_states, encoded_actions))
        
        # Initialize metrics
        metrics = {
            "loss_history": [],
            "reward_history": [],
            "kl_history": []
        }
        
        # Get optimizer
        lr = adapt_lr if adapt_lr is not None else self.learning_rate
        optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=lr)
        
        # Training loop
        for step in range(num_steps):
            # Sample random demonstrations
            indices = torch.randint(0, len(encoded_trajectories), (batch_size,))
            
            # Prepare batch
            batch_states = []
            batch_actions = []
            
            for idx in indices:
                states, actions = encoded_trajectories[idx]
                
                # Handle sequences of different lengths
                if states.shape[0] > self.max_trajectory_length:
                    # Randomly select a continuous segment
                    start_idx = torch.randint(0, states.shape[0] - self.max_trajectory_length, (1,)).item()
                    states = states[start_idx:start_idx + self.max_trajectory_length]
                    actions = actions[start_idx:start_idx + self.max_trajectory_length]
                elif states.shape[0] < self.max_trajectory_length:
                    # Pad sequences
                    pad_len = self.max_trajectory_length - states.shape[0]
                    state_padding = torch.zeros(pad_len, states.shape[1], device=self.device)
                    action_padding = torch.zeros(pad_len, actions.shape[1], device=self.device)
                    states = torch.cat([states, state_padding], dim=0)
                    actions = torch.cat([actions, action_padding], dim=0)
                
                batch_states.append(states)
                batch_actions.append(actions)
            
            # Convert to tensors
            states_tensor = torch.stack(batch_states)
            actions_tensor = torch.stack(batch_actions)
            
            # Prepare for diffusion with teacher forcing
            for t_idx, timestep in enumerate(self.noise_scheduler.get_training_timesteps()):
                # Add noise to target actions
                noisy_actions, noise = self.noise_scheduler.add_noise(
                    actions_tensor, 
                    timestep, 
                    noise=None
                )
                
                # Get batch timesteps
                t_batch = torch.full((batch_size,), timestep, device=self.device)
                
                # Forward pass through model to predict noise/signal
                optimizer.zero_grad()
                pred = self.diffusion_model(
                    x=noisy_actions,
                    timesteps=t_batch,
                    state=states_tensor.view(batch_size, -1),
                    task_embed=task_embedding
                )
                
                # Compute loss
                if self.noise_scheduler.prediction_type == "noise":
                    target = noise
                else:
                    target = actions_tensor
                
                # Main diffusion loss
                diff_loss = F.mse_loss(pred, target)
                
                # Add KL regularization to stay close to prior
                kl_loss = torch.tensor(0.0, device=self.device)
                if alpha_kl > 0:
                    # Compute KL divergence to prior
                    prior_mean = torch.zeros_like(pred)
                    prior_var = torch.ones_like(pred)
                    pred_var = torch.ones_like(pred) * noise_level
                    
                    kl_div = torch.log(prior_var / pred_var) + \
                            (pred_var + (pred - prior_mean).pow(2)) / prior_var - 1
                    kl_loss = 0.5 * kl_div.sum(dim=-1).mean()
                
                # Total loss
                loss = diff_loss + alpha_kl * kl_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Log metrics
                if t_idx == 0 and step % feedback_frequency == 0:
                    with torch.no_grad():
                        # Evaluate reward on clean actions
                        rewards = self.state_representation.compute_rewards(
                            states_tensor[:, -1],  # Use final states
                            task_embedding
                        )
                        avg_reward = rewards.mean().item()
                    
                    metrics["loss_history"].append(diff_loss.item())
                    metrics["kl_history"].append(kl_loss.item())
                    metrics["reward_history"].append(avg_reward)
                    
                    logger.info(
                        f"Step {step}/{num_steps}, "
                        f"Loss: {diff_loss.item():.4f}, "
                        f"KL: {kl_loss.item():.4f}, "
                        f"Reward: {avg_reward:.4f}"
                    )
                    
                    if avg_reward >= min_reward_threshold:
                        logger.info(f"Reached reward threshold {min_reward_threshold}")
                        
                        if save_checkpoints:
                            checkpoint_path = os.path.join(
                                self.checkpoint_dir,
                                f"adapted_model_step_{step}_reward_{avg_reward:.2f}.pt"
                            )
                            self.save(checkpoint_path)
            
            # Optional checkpoint saving
            if save_checkpoints and step > 0 and step % (num_steps // 5) == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"adapted_model_step_{step}.pt"
                )
                self.save(checkpoint_path)
        
        # Final evaluation
        with torch.no_grad():
            # Generate samples with adapted model
            samples = self.adapt_diffuser.generate(
                batch_size=4,
                task=task_embedding,
                conditioning={"state": batch_states[0].unsqueeze(0)},
                guidance_scale=guidance_scale or self.guidance_scale
            )
            metrics["samples"] = samples
            
            # Save final checkpoint
            if save_checkpoints:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"adapted_model_final.pt"
                )
                self.save(checkpoint_path)
        
        # Save adaptation metrics for analysis
        save_adaptation_metrics(
            metrics=metrics,
            save_dir=self.checkpoint_dir,
            prefix="demo_adaptation"
        )
        
        return metrics
        
    def multi_objective_plan(
        self,
        initial_state: Any,
        task: Optional[Union[str, torch.Tensor]] = None,
        objective_dict: Dict[str, Callable] = None,
        weights: Dict[str, float] = None,
        num_samples: int = 10,
        temperature: float = 1.0,
        custom_guidance_scale: Optional[float] = None
    ) -> Union[Tuple[List[Any], List[Any]], List[Tuple[List[Any], List[Any]]]]:
        """
        Generate plans optimizing multiple objectives simultaneously.
        
        Args:
            initial_state: Starting state for planning
            task: Optional task description or embedding
            objective_dict: Dictionary of objective functions
            weights: Weights for each objective
            num_samples: Number of candidate plans to generate
            temperature: Sampling temperature
            custom_guidance_scale: Optional override for guidance scale
            
        Returns:
            Best trajectory (states, actions) or list of all trajectories
        """
        if self.guidance_strategy != GuidanceStrategy.MULTI_OBJECTIVE:
            logger.warning(
                "Multi-objective planning requested but planner was not initialized with "
                "MULTI_OBJECTIVE guidance strategy. Results may be suboptimal."
            )
        
        # Use provided weights or fall back to initialized weights
        weights = weights or self.multi_objective_weights
        
        # If no objective dictionary provided, use a default
        if objective_dict is None:
            # Default to using reward model as single objective
            objective_dict = {
                "reward": lambda states, actions, task_emb: \
                    self.state_representation.compute_rewards(states, task_emb)
            }
        
        # Set up multi-objective guidance
        if hasattr(self.adapt_diffuser.guidance, "set_objectives"):
            self.adapt_diffuser.guidance.set_objectives(objective_dict, weights)
        else:
            logger.warning(
                "Adapt diffuser guidance does not support setting objectives. "
                "Using default reward guidance."
            )
        
        # Generate plans using the parent method
        return self.plan(
            initial_state=initial_state,
            task=task,
            num_samples=num_samples,
            temperature=temperature,
            custom_guidance_scale=custom_guidance_scale
        )
    
    def plan_with_dynamics(
        self,
        initial_state: Any,
        task: Optional[Union[str, torch.Tensor]] = None,
        num_samples: int = 10,
        temperature: float = 1.0,
        custom_guidance_scale: Optional[float] = None,
        use_validator: bool = True
    ) -> Tuple[List[Any], List[Any]]:
        """
        Generate plans using a dynamics model for accurate state transitions.
        
        Args:
            initial_state: Starting state for planning
            task: Optional task description or embedding
            num_samples: Number of candidate plans to generate
            temperature: Sampling temperature
            custom_guidance_scale: Optional override for guidance scale
            use_validator: Whether to use validator for selecting best trajectory
            
        Returns:
            Best trajectory (states, actions)
        """
        if self.dynamics_model is None:
            logger.warning(
                "Planning with dynamics requested but no dynamics model provided. "
                "Using simple state transitions."
            )
            return self.plan(
                initial_state=initial_state,
                task=task,
                num_samples=num_samples,
                temperature=temperature,
                custom_guidance_scale=custom_guidance_scale
            )
        
        # Generate action sequences
        candidates = self.plan(
            initial_state=initial_state,
            task=task,
            num_samples=num_samples,
            temperature=temperature,
            custom_guidance_scale=custom_guidance_scale,
            return_all_samples=True
        )
        
        # Process all candidates using accurate dynamics
        processed_candidates = []
        for states, actions in candidates:
            # Use only the initial state and actions
            accurate_states = [initial_state]
            
            # Apply dynamics model to get accurate states
            for action in actions:
                next_state = self.dynamics_model(accurate_states[-1], action)
                accurate_states.append(next_state)
            
            processed_candidates.append((accurate_states, actions))
        
        # Choose best candidate if validator provided
        if self.validator is not None and use_validator:
            # Encode task if provided
            task_embedding = None
            if task is not None:
                task_embedding = encode_task(
                    task=task,
                    task_embedding_model=getattr(self.diffusion_model, 'task_embedding_model', None),
                    device=self.device
                )
            
            # Score each candidate
            best_score = float('-inf')
            best_candidate = processed_candidates[0]
            
            for states, actions in processed_candidates:
                # Encode states and actions
                encoded_states = torch.stack([
                    self.state_representation.encode_states(state, task_embedding)
                    for state in states[:-1]  # Exclude last state which has no action
                ])
                
                encoded_actions = torch.stack([
                    self.action_space.normalize(action)
                    for action in actions
                ])
                
                # Compute rewards for final state
                reward = self.state_representation.compute_rewards(
                    self.state_representation.encode_states(states[-1], task_embedding),
                    task_embedding
                ).item()
                
                # Compute constraint penalties
                penalty = self.validator.compute_violation_penalties(
                    encoded_states.unsqueeze(0),
                    encoded_actions.unsqueeze(0),
                    task_embedding
                ).item()
                
                # Compute score as reward minus penalty
                score = reward - penalty
                
                if score > best_score:
                    best_score = score
                    best_candidate = (states, actions)
            
            return best_candidate
        else:
            # Return first candidate if no validator
            return processed_candidates[0]
    
    def apply_action_with_dynamics(self, state: Any, action: Any) -> Any:
        """
        Apply an action to a state using the dynamics model if available.
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            Next state after applying action
        """
        if self.dynamics_model is not None:
            return self.dynamics_model(state, action)
        else:
            return super()._apply_action_to_state(state, action)