"""
Guidance strategies for adaptive planning in diffusion models.

This module defines various guidance strategies for controlling the
diffusion trajectory generation process.
"""

import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum

from agentic_diffusion.core.adapt_diffuser.guidance import AdaptDiffuserGuidance
from agentic_diffusion.planning.state_representations import StateRewardModel


class GuidanceStrategy(Enum):
    """
    Enumeration of guidance strategies for adaptive planning.
    """
    REWARD_GRADIENT = "reward_gradient"  # Standard reward gradient guidance
    CLASSIFIER_FREE = "classifier_free"  # Classifier-free guidance
    CONSTRAINT_GUIDED = "constraint_guided"  # Constraint-based guidance
    MULTI_OBJECTIVE = "multi_objective"  # Multi-objective optimization
    PROGRESSIVE = "progressive"  # Progressive distillation guidance


class ClassifierFreeGuidance(AdaptDiffuserGuidance):
    """
    Classifier-free guidance for AdaptDiffuser.
    
    This guidance approach adapts the classifier-free guidance technique
    from image diffusion models to trajectory generation.
    """
    
    def __init__(
        self,
        reward_model: StateRewardModel,
        guidance_scale: float = 3.0,
        device: str = None
    ):
        """
        Initialize the classifier-free guidance.
        
        Args:
            reward_model: Model for computing rewards
            guidance_scale: Scale factor for guidance
            device: Device to use for computation
        """
        super().__init__(reward_model, guidance_scale, device)
    
    def guide(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        task: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance to the diffusion process.
        
        Args:
            x: Current noise level tensor
            t: Current timestep
            conditioning: Dictionary of conditioning inputs
            task: Optional task embedding
            
        Returns:
            Guidance gradients to apply
        """
        # Standard implementation would run the model twice - once with
        # conditioning and once without, then combine the results
        # This is just a sketch of the approach
        
        # Extract state conditioning
        state = conditioning.get("state", None)
        
        # Apply standard reward gradient guidance
        gradient = super().guide(x, t, conditioning, task)
        
        # In a full implementation, we would run the model without conditioning
        # and compute a weighted combination of conditioned and unconditioned outputs
        
        return gradient


class ConstraintGuidance(AdaptDiffuserGuidance):
    """
    Constraint-based guidance for AdaptDiffuser.
    
    This guidance incorporates explicit constraints into the diffusion process.
    """
    
    def __init__(
        self,
        reward_model: StateRewardModel,
        guidance_scale: float = 3.0,
        device: str = None,
        constraint_functions: List[Callable] = None,
        constraint_weights: List[float] = None
    ):
        """
        Initialize the constraint-guided diffusion.
        
        Args:
            reward_model: Model for computing rewards
            guidance_scale: Scale factor for guidance
            device: Device to use for computation
            constraint_functions: List of constraint functions
            constraint_weights: Weights for each constraint
        """
        super().__init__(reward_model, guidance_scale, device)
        self.constraint_functions = constraint_functions or []
        
        # Set uniform weights if not provided
        if constraint_weights is None and constraint_functions:
            constraint_weights = [1.0] * len(constraint_functions)
        
        self.constraint_weights = constraint_weights or []
    
    def guide(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        task: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply constraint-guided diffusion.
        
        Args:
            x: Current noise level tensor
            t: Current timestep
            conditioning: Dictionary of conditioning inputs
            task: Optional task embedding
            
        Returns:
            Guidance gradients to apply
        """
        # Start with reward gradient
        gradient = super().guide(x, t, conditioning, task)
        
        # If no constraints, just return reward gradient
        if not self.constraint_functions:
            return gradient
        
        # Extract state conditioning
        state = conditioning.get("state", None)
        
        if state is None:
            # Cannot apply constraints without state
            return gradient
        
        # Apply each constraint
        constraint_gradients = []
        for constraint_fn, weight in zip(self.constraint_functions, self.constraint_weights):
            # Compute constraint violation probability
            x_detached = x.detach().requires_grad_(True)
            violation_prob = constraint_fn(state, x_detached, task)
            
            # Compute gradient of constraint violation
            if torch.is_tensor(violation_prob) and violation_prob.requires_grad:
                constraint_grad = torch.autograd.grad(
                    violation_prob.sum(),
                    x_detached,
                    create_graph=False,
                    retain_graph=True
                )[0]
                
                # Apply weight and add to list
                constraint_gradients.append(weight * constraint_grad)
        
        # Combine constraint gradients with reward gradient
        if constraint_gradients:
            combined_constraint_grad = sum(constraint_gradients)
            # Apply negative constraint gradient to avoid violations
            gradient = gradient - combined_constraint_grad
        
        return gradient


class MultiObjectiveGuidance(AdaptDiffuserGuidance):
    """
    Multi-objective guidance for AdaptDiffuser.
    
    This guidance optimizes for multiple objectives simultaneously.
    """
    
    def __init__(
        self,
        reward_model: StateRewardModel,
        guidance_scale: float = 3.0,
        device: str = None
    ):
        """
        Initialize multi-objective guidance.
        
        Args:
            reward_model: Model for computing rewards
            guidance_scale: Scale factor for guidance
            device: Device to use for computation
        """
        super().__init__(reward_model, guidance_scale, device)
        self.objectives = {}
        self.weights = {}
    
    def set_objectives(
        self,
        objective_dict: Dict[str, Callable],
        weights: Dict[str, float]
    ) -> None:
        """
        Set the objective functions and their weights.
        
        Args:
            objective_dict: Dictionary of objective functions
            weights: Weights for each objective
        """
        self.objectives = objective_dict
        self.weights = weights
    
    def guide(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        task: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-objective guidance.
        
        Args:
            x: Current noise level tensor
            t: Current timestep
            conditioning: Dictionary of conditioning inputs
            task: Optional task embedding
            
        Returns:
            Guidance gradients to apply
        """
        # If no custom objectives set, use default reward guidance
        if not self.objectives:
            return super().guide(x, t, conditioning, task)
        
        # Extract state conditioning
        state = conditioning.get("state", None)
        
        if state is None:
            # Cannot apply objectives without state
            return super().guide(x, t, conditioning, task)
        
        # Compute gradients for each objective
        objective_gradients = {}
        for name, objective_fn in self.objectives.items():
            # Detach x and enable gradients
            x_detached = x.detach().requires_grad_(True)
            
            # Compute objective value
            obj_value = objective_fn(state, x_detached, task)
            
            # Compute gradient
            if torch.is_tensor(obj_value) and obj_value.requires_grad:
                obj_grad = torch.autograd.grad(
                    obj_value.sum(),
                    x_detached,
                    create_graph=False,
                    retain_graph=True
                )[0]
                
                # Store gradient
                objective_gradients[name] = obj_grad
        
        # Combine gradients with weights
        combined_gradient = torch.zeros_like(x)
        for name, gradient in objective_gradients.items():
            weight = self.weights.get(name, 1.0)
            combined_gradient = combined_gradient + weight * gradient
        
        # Apply guidance scale
        return self.guidance_scale * combined_gradient


class ProgressiveGuidance(AdaptDiffuserGuidance):
    """
    Progressive guidance for AdaptDiffuser.
    
    This guidance gradually changes its behavior during the diffusion process.
    """
    
    def __init__(
        self,
        reward_model: StateRewardModel,
        guidance_scale: float = 3.0,
        device: str = None,
        guidance_schedule: Optional[Callable] = None
    ):
        """
        Initialize progressive guidance.
        
        Args:
            reward_model: Model for computing rewards
            guidance_scale: Scale factor for guidance
            device: Device to use for computation
            guidance_schedule: Function to compute guidance scale based on timestep
        """
        super().__init__(reward_model, guidance_scale, device)
        
        # Default schedule: linear decrease from max to min
        if guidance_schedule is None:
            guidance_schedule = lambda t, max_t: 1.0 - 0.8 * (t / max_t)
            
        self.guidance_schedule = guidance_schedule
    
    def guide(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Dict[str, torch.Tensor],
        task: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply progressive guidance.
        
        Args:
            x: Current noise level tensor
            t: Current timestep
            conditioning: Dictionary of conditioning inputs
            task: Optional task embedding
            
        Returns:
            Guidance gradients to apply
        """
        # Compute base guidance
        base_gradient = super().guide(x, t, conditioning, task)
        
        # Apply schedule based on normalized timestep
        max_timestep = 1000  # Typical max timestep in diffusion
        t_norm = t.float() / max_timestep
        
        # Compute adaptive scale
        adaptive_scale = self.guidance_schedule(t_norm.mean().item(), 1.0)
        
        # Apply adaptive scaling
        return adaptive_scale * self.guidance_scale * base_gradient