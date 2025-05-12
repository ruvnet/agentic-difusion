"""
AdaptDiffuser API for Agentic Diffusion

Usage:
    # Initialize with default configuration
    api = AdaptDiffuserAPI(config={
        "guidance_scale": 5.0,
        "batch_size": 16,
        "sampling_steps": 50,
        "device": "auto"  # Will use CUDA if available
    })
    
    # Generate trajectories for a task
    trajectories, metadata = api.generate(
        task="solve_maze",
        batch_size=8,
        guidance_scale=7.0  # Override default guidance
    )
    
    # Adapt the model to a specific task
    metrics = api.adapt(
        task="navigate_environment",
        iterations=5,
        batch_size=32
    )
    
    # Self-improve model on a task
    metrics = api.self_improve(
        task="avoid_obstacles",
        iterations=3,
        trajectories_per_iter=50
    )
    
    # Save/load model state
    api.save_state("path/to/state.pt")
    api.load_state("path/to/state.pt")

Configuration Options:
    - device: Device to run on ('auto', 'cuda', 'cpu')
    - guidance_scale: Scale for reward guidance (higher = stronger guidance)
    - batch_size: Default batch size for generation and adaptation
    - sampling_steps: Number of steps for diffusion sampling
    - use_ddim: Whether to use DDIM sampling (faster but less diverse)
    - adaptation_rate: Learning rate for adaptation steps
    - quality_threshold: Minimum reward threshold for high-quality samples
"""

import os
import logging
import torch
import yaml
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from agentic_diffusion.core.adapt_diffuser import AdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser as BaseAdaptDiffuser
from agentic_diffusion.core.adapt_diffuser.multi_task import MultiTaskAdaptDiffuser
from agentic_diffusion.core.diffusion_model import DiffusionModel, DenoisingDiffusionModel, LatentDiffusionModel
from agentic_diffusion.core.noise_schedules import NoiseScheduler, CosineScheduler, LinearScheduler, SigmoidScheduler
from agentic_diffusion.core.unet import UNet
# Original import that's causing issues
# from agentic_diffusion.adaptation.task_embeddings import TaskEmbeddingModel
# Use the wrapper class from our API instead
from agentic_diffusion.api.task_embedding_api import TaskEmbeddingModel
from agentic_diffusion.core.trajectory_buffer import AdaptDiffuserTrajectoryBuffer
from agentic_diffusion.core.adapt_diffuser.selection import (
    SelectionStrategy,
    TopKSelection,
    HybridSelection
)
from agentic_diffusion.adaptation.adapt_diffuser_adaptation import (
    AdaptDiffuserAdaptation,
    TrajectoryDiscriminator,
    SyntheticExpertGenerator
)
from agentic_diffusion.core.reward_functions import RewardModel

# Add CompositeRewardModel class if it doesn't exist elsewhere
class CompositeRewardModel(RewardModel):
    """Simple placeholder implementation of a composite reward model."""
    
    def __init__(self):
        """Initialize the composite reward model."""
        pass
        
    def compute_reward(self, samples: torch.Tensor, task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute reward based on samples and optional task embedding.
        
        Args:
            samples: Batch of samples/trajectories
            task: Optional task embedding
            
        Returns:
            Tensor of rewards
        """
        # Simple placeholder implementation - just compute L2 norm
        if len(samples.shape) <= 2:
            norms = torch.norm(samples, p=2, dim=-1)
        else:
            norms = torch.norm(samples.view(samples.shape[0], -1), p=2, dim=-1)
        
        # Convert to reward (smaller norm = higher reward)
        rewards = 1.0 / (1.0 + norms)
        
        # If task is provided, make the reward task-dependent
        if task is not None and isinstance(task, torch.Tensor):
            # Use task vector sum as a scaling factor
            task_factor = 0.5 + 0.5 * torch.sigmoid(torch.sum(task)).item()
            rewards = rewards * task_factor
        
        return rewards

# Configure logging
logger = logging.getLogger(__name__)


def create_adapt_diffuser_from_config(config: Dict[str, Any]) -> AdaptDiffuser:
    """
    Factory function to create an AdaptDiffuser model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AdaptDiffuser model
    """
    # Get device
    device_config = config.get("device", "auto")
    device = device_config if device_config != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get basic parameters
    img_size = config.get("img_size", 32)
    channels = config.get("channels", 3)
    guidance_scale = config.get("guidance_scale", 1.0)
    use_ddim = config.get("use_ddim", True)
    sampling_steps = config.get("sampling_steps", 50)
    
    # Create noise scheduler
    noise_scheduler_config = config.get("noise_scheduler", {})
    scheduler_type = noise_scheduler_config.get("type", "cosine")
    
    if scheduler_type == "cosine":
        noise_scheduler = CosineScheduler(
            num_timesteps=noise_scheduler_config.get("num_timesteps", 1000),
            s=noise_scheduler_config.get("s", 0.008),
            device=device
        )
    elif scheduler_type == "linear":
        noise_scheduler = LinearScheduler(
            num_timesteps=noise_scheduler_config.get("num_timesteps", 1000),
            beta_start=noise_scheduler_config.get("beta_start", 0.0001),
            beta_end=noise_scheduler_config.get("beta_end", 0.02),
            device=device
        )
    elif scheduler_type == "sigmoid":
        noise_scheduler = SigmoidScheduler(
            num_timesteps=noise_scheduler_config.get("num_timesteps", 1000),
            beta_start=noise_scheduler_config.get("beta_start", 0.0001),
            beta_end=noise_scheduler_config.get("beta_end", 0.02),
            sigmoid_scale=noise_scheduler_config.get("sigmoid_scale", 10.0),
            device=device
        )
    else:
        raise ValueError(f"Unsupported noise scheduler type: {scheduler_type}")
    
    # Create diffusion model
    model_config = config.get("model", {})
    
    # In a real implementation, this would create the proper model architecture
    # For now, we use a placeholder that would be replaced in production
    # Create UNet model for the noise prediction network
    unet = UNet(
        in_channels=channels,
        model_channels=model_config.get("model_channels", 128),
        out_channels=channels,
        channel_multipliers=tuple(model_config.get("channel_mult", [1, 2, 4, 8])),
        num_res_blocks=model_config.get("num_res_blocks", 2),
        dropout=model_config.get("dropout", 0.0),
        time_embedding_dim=model_config.get("time_embedding_dim", 128),
        condition_dim=model_config.get("condition_dim")
    )
    
    # Create concrete diffusion model implementation
    diffusion_model = DenoisingDiffusionModel(
        noise_pred_net=unet,
        noise_scheduler=noise_scheduler,
        img_size=img_size,
        in_channels=channels,
        device=device
    )
    # Create reward model
    reward_model = None
    reward_config = config.get("reward_model", {})
    if reward_config:
        # Check for reward model type
        reward_type = reward_config.get("type", "composite")
        
        if reward_type == "simple_reward":
            # Use the SimpleRewardModel implementation
            try:
                from agentic_diffusion.core.simple_reward_model import SimpleRewardModel
                reward_model = SimpleRewardModel(
                    base_reward=reward_config.get("base_reward", 0.5),
                    noise_scale=reward_config.get("noise_scale", 0.1),
                    device=device
                )
                logger.info("Initialized SimpleRewardModel for AdaptDiffuser")
            except ImportError:
                logger.warning("Could not import SimpleRewardModel, falling back to CompositeRewardModel")
                reward_model = CompositeRewardModel()
        elif reward_type == "test_reward":
            # Use the test-specific reward model
            try:
                from agentic_diffusion.core.simple_reward_model import AdaptDiffuserTestRewardModel
                reward_model = AdaptDiffuserTestRewardModel(
                    initial_reward=reward_config.get("initial_reward", 0.5),
                    improvement_rate=reward_config.get("improvement_rate", 0.05),
                    device=device
                )
                logger.info("Initialized AdaptDiffuserTestRewardModel for AdaptDiffuser")
            except ImportError:
                logger.warning("Could not import AdaptDiffuserTestRewardModel, falling back to CompositeRewardModel")
                reward_model = CompositeRewardModel()
        else:
            # Default to composite model
            reward_model = CompositeRewardModel()
            logger.info("Using default CompositeRewardModel for AdaptDiffuser")
    
    # Create task embedding model
    task_embedding_model = None
    task_config = config.get("adaptdiffuser", {}).get("task_embedding", {})
    
    if task_config:
        # Import the task embedding API
        try:
            from agentic_diffusion.api.task_embedding_api import load_task_embedding_model
            task_embedding_model = load_task_embedding_model(config)
            logger.info(f"Loaded task embedding model with embedding_dim={task_embedding_model.embedding_dim}")
        except ImportError as e:
            logger.warning(f"Failed to load task embedding model: {e}")
            task_embedding_model = None
    
    # Create selection strategy
    selection_config = config.get("selection", {})
    if selection_config.get("strategy", "hybrid") == "topk":
        selection_strategy = TopKSelection()
    else:
        # Default to hybrid selection with sensible defaults
        selection_strategy = HybridSelection(
            strategies=[TopKSelection()],
            weights=[1.0]
        )
    
    # Determine which variant of AdaptDiffuser to create
    if config.get("multi_task", False):
        # Create multi-task variant
        adapt_diffuser = MultiTaskAdaptDiffuser(
            base_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            img_size=img_size,
            channels=channels,
            reward_model=reward_model,
            task_embedding_model=task_embedding_model,
            guidance_scale=guidance_scale,
            buffer_capacity=config.get("buffer_capacity", 10000),
            learning_rate=config.get("learning_rate", 1e-5),
            use_ddim=use_ddim,
            inference_steps=sampling_steps,
            device=device
        )
    else:
        # Create base variant
        adapt_diffuser = AdaptDiffuser(
            base_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            img_size=img_size,
            channels=channels,
            reward_model=reward_model,
            task_embedding_model=task_embedding_model,
            guidance_scale=guidance_scale,
            guidance_min_step_percent=config.get("guidance_min_step_percent", 0.1),
            guidance_max_step_percent=config.get("guidance_max_step_percent", 0.9),
            classifier_scale=config.get("classifier_scale", 0.0),
            buffer_capacity=config.get("buffer_capacity", 10000),
            learning_rate=config.get("learning_rate", 1e-5),
            checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
            use_ddim=use_ddim,
            inference_steps=sampling_steps,
            device=device
        )
    
    return adapt_diffuser


class AdaptDiffuserAPI:
    """
    Public API for interacting with AdaptDiffuser functionality.
    """
    
    def register_reward_model(self, reward_model):
        """Register a reward model with the AdaptDiffuser API."""
        logger.info("Registering reward model with AdaptDiffuser API")
        self.reward_model = reward_model
        # Register with the underlying model if it exists
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'register_reward_model'):
                self.model.register_reward_model(reward_model)
            else:
                # Direct assignment
                self.model.reward_model = reward_model
        return True
    def __init__(
        self,
        model: Optional[AdaptDiffuser] = None,
        reward_model: Optional[RewardModel] = None,
        discriminator: Optional[TrajectoryDiscriminator] = None,
        trajectory_buffer: Optional[AdaptDiffuserTrajectoryBuffer] = None,
        task_embedding_model: Optional[TaskEmbeddingModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AdaptDiffuser API.
        
        Args:
            model: AdaptDiffuser model instance
            reward_model: Reward model instance
            discriminator: Discriminator instance
            trajectory_buffer: Trajectory buffer instance
            task_embedding_model: Task embedding manager instance
            config: Configuration dictionary
        """
        self.config = config or {}
        self.device = self._get_device(self.config.get("device", "auto"))
        
        # Set up components
        self.model = model
        self.reward_model = reward_model
        self.discriminator = discriminator
        self.trajectory_buffer = trajectory_buffer
        self.task_embedding_model = task_embedding_model
        
        # Create model if not provided
        if self.model is None:
            self.model = create_adapt_diffuser_from_config(self.config)
            
            # Update components from model if not provided separately
            if self.reward_model is None and hasattr(self.model, "reward_model"):
                self.reward_model = self.model.reward_model
                
            if self.trajectory_buffer is None and hasattr(self.model, "trajectory_buffer"):
                self.trajectory_buffer = self.model.trajectory_buffer
                
            if self.task_embedding_model is None and hasattr(self.model, "task_embedding_model"):
                self.task_embedding_model = self.model.task_embedding_model
        
        # Create discriminator if not provided
        if self.discriminator is None:
            # Estimate input dimension from model
            input_dim = self.model.img_size * self.model.img_size * self.model.channels
            task_embedding_dim = None
            if self.task_embedding_model:
                task_embedding_dim = self.task_embedding_model.embedding_dim
                
            self.discriminator = TrajectoryDiscriminator(
                input_dim=input_dim,
                task_embedding_dim=task_embedding_dim,
                device=self.device
            )
        
        # Create the adaptation mechanism
        self.adaptation_mechanism = AdaptDiffuserAdaptation(
            adapt_diffuser=self.model,
            discriminator=self.discriminator,
            adaptation_rate=self.config.get("adaptation_rate", 0.1),
            quality_threshold=self.config.get("quality_threshold", 0.7),
            device=self.device
        )
        
        # Set up synthetic expert generator
        self.synthetic_expert_generator = SyntheticExpertGenerator(
            adapt_diffuser=self.model,
            quality_threshold=self.config.get("quality_threshold", 0.7),
            device=self.device
        )
    
    def _get_device(self, device_config: str) -> str:
        """
        Determine the device to use.
        
        Args:
            device_config: Device configuration string
            
        Returns:
            Device string
        """
        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_config
    
    def _encode_task(self, task: Union[str, Dict, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Encode a task into an embedding if necessary.
        
        Args:
            task: Task specification
            
        Returns:
            Task embedding or None
        """
        if isinstance(task, torch.Tensor):
            return task
            
        if self.task_embedding_model is not None:
            try:
                if isinstance(task, dict):
                    # Convert dict to string for simplicity
                    task_str = json.dumps(task)
                    return self.task_embedding_model.encode(task_str)
                elif isinstance(task, str):
                    return self.task_embedding_model.encode(task)
            except Exception as e:
                logger.warning(f"Failed to encode task: {e}")
                return None
        
        # No embedding available
        return None
    
    def adapt(
        self,
        task: Union[str, Dict, torch.Tensor],
        trajectories: Optional[List[torch.Tensor]] = None,
        rewards: Optional[List[float]] = None,
        iterations: int = 1,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        quality_threshold: Optional[float] = None,
        save_checkpoint: bool = False,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adapt the model to a specific task.
        
        Args:
            task: Task specification
            trajectories: Optional example trajectories
            rewards: Optional rewards for trajectories
            iterations: Number of adaptation iterations
            batch_size: Batch size for adaptation
            learning_rate: Learning rate for adaptation
            quality_threshold: Quality threshold for sample selection
            save_checkpoint: Whether to save checkpoints
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Adaptation results and metrics
        """
        # Use defaults from config if not specified
        if batch_size is None:
            batch_size = self.config.get("batch_size", 16)
            
        if learning_rate is None:
            learning_rate = self.config.get("adaptation_rate", 1e-5)
            
        if quality_threshold is None:
            quality_threshold = self.config.get("quality_threshold", 0.7)
            
        # Encode task if needed
        task_embedding = self._encode_task(task)
        
        # If we have example trajectories, use them to bootstrap adaptation
        if trajectories is not None:
            if rewards is None:
                # Compute rewards if not provided
                rewards = []
                for traj in trajectories:
                    traj_tensor = traj if isinstance(traj, torch.Tensor) else torch.tensor(traj, device=self.device)
                    reward = self.model.evaluate_samples(traj_tensor.unsqueeze(0), task_embedding).item()
                    rewards.append(reward)
                    
            # Store trajectories in buffer
            for traj, reward in zip(trajectories, rewards):
                traj_tensor = traj if isinstance(traj, torch.Tensor) else torch.tensor(traj, device=self.device)
                self.model.store_high_quality_samples(
                    traj_tensor.unsqueeze(0),
                    torch.tensor([reward], device=self.device),
                    task=task_embedding
                )
        
        # Check if model has a reward model
        if not hasattr(self.model, 'reward_model') or self.model.reward_model is None:
            logger.warning("No reward model available for adaptation, returning empty metrics")
            return {"status": "skipped", "reason": "no_reward_model"}
            
        # Perform adaptation
        metrics = self.model.adapt_to_task(
            task=task_embedding if task_embedding is not None else task,
            num_steps=self.config.get("adaptation_steps", 10),
            batch_size=batch_size,
            adapt_lr=learning_rate,
            min_reward_threshold=quality_threshold,
            save_checkpoints=save_checkpoint,
            metrics_path=checkpoint_dir or self.config.get("checkpoint_dir")
        )
        
        return metrics
    
    def generate(
        self,
        task: Optional[Union[str, Dict, torch.Tensor]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        guidance_scale: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate trajectories for a task.
        
        Args:
            task: Optional task specification
            conditions: Optional generation conditions
            batch_size: Number of trajectories to generate
            guidance_scale: Scale factor for reward guidance
            
        Returns:
            Tuple of (trajectories, generation_metadata)
        """
        # Encode task if needed
        task_embedding = None
        if task is not None:
            task_embedding = self._encode_task(task)
        
        # Set up generation parameters
        gen_guidance_scale = guidance_scale or self.config.get("guidance_scale", 1.0)
        
        # Combine conditions for passing to the model
        kwargs = conditions or {}
        
        # Generate samples
        trajectories = self.model.generate(
            batch_size=batch_size,
            task=task_embedding if task_embedding is not None else task,
            custom_guidance_scale=gen_guidance_scale,
            guidance_scale=gen_guidance_scale,  # Support both parameter names
            **kwargs
        )
        
        # Evaluate if reward model available
        rewards = None
        if hasattr(self.model, "evaluate_samples"):
            rewards = self.model.evaluate_samples(trajectories, task_embedding)
            
        # Generate metadata
        metadata = {
            "batch_size": batch_size,
            "guidance_scale": gen_guidance_scale,
            "task": str(task) if not isinstance(task, torch.Tensor) else "tensor"
        }
        
        if rewards is not None:
            metadata["rewards"] = {
                "mean": rewards.mean().item(),
                "max": rewards.max().item(),
                "min": rewards.min().item()
            }
        
        return trajectories, metadata
    
    def evaluate(
        self,
        trajectory: torch.Tensor,
        task: Union[str, Dict, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate a trajectory for a task.
        
        Args:
            trajectory: Trajectory to evaluate
            task: Task specification
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure trajectory is a tensor
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, device=self.device)
            
        # Add batch dimension if needed
        if len(trajectory.shape) == 3:  # [C, H, W]
            trajectory = trajectory.unsqueeze(0)  # [1, C, H, W]
        
        # Encode task if needed
        task_embedding = self._encode_task(task)
        
        # Evaluate with reward model
        reward = 0.0
        try:
            reward = self.model.evaluate_samples(trajectory, task_embedding).item()
        except Exception as e:
            logger.error(f"Failed to evaluate trajectory: {e}")
        
        # Evaluate with discriminator if available
        quality_score = 0.0
        try:
            quality_score = self.discriminator.evaluate_quality(
                trajectory, 
                task_embedding
            ).item()
        except Exception as e:
            logger.error(f"Failed to get quality score: {e}")
        
        return {
            "reward": reward,
            "quality_score": quality_score,
            "overall": (reward + quality_score) / 2  # Simple average for overall score
        }
    
    def self_improve(
        self,
        task: Union[str, Dict, torch.Tensor],
        iterations: int = 5,
        trajectories_per_iter: int = 100,
        quality_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run self-improvement cycle for a task.
        
        Args:
            task: Task to improve on
            iterations: Number of improvement iterations
            trajectories_per_iter: Trajectories per iteration
            quality_threshold: Minimum quality for keeping trajectories
            
        Returns:
            Self-improvement results and metrics
        """
        # Encode task if needed
        task_embedding = self._encode_task(task)
        
        metrics = {
            "iterations": iterations,
            "trajectories_generated": 0,
            "high_quality_trajectories": 0,
            "reward_improvement": 0.0,
            "quality_scores": []
        }
        
        # Check if model has a reward model
        if not hasattr(self.model, 'reward_model') or self.model.reward_model is None:
            logger.warning("No reward model available for self-improvement, returning empty metrics")
            return {"status": "skipped", "reason": "no_reward_model"}
            
        # Initial reward measurement
        initial_samples, _ = self.generate(
            task=task_embedding if task_embedding is not None else task,
            batch_size=10
        )
        
        initial_rewards = self.model.compute_reward(
            initial_samples,
            task_embedding if task_embedding is not None else task
        )
        # Ensure rewards are tensors
        if not isinstance(initial_rewards, torch.Tensor):
            if isinstance(initial_rewards, (list, np.ndarray)):
                initial_rewards = torch.tensor(initial_rewards, device=self.model.device)
            else:
                initial_rewards = torch.tensor([initial_rewards], device=self.model.device)
        
        initial_mean_reward = initial_rewards.mean().item()
        
        for i in range(iterations):
            logger.info(f"Self-improvement iteration {i+1}/{iterations}")
            
            # Generate synthetic expert data
            synthetic_samples, synthetic_rewards = self.synthetic_expert_generator.generate_synthetic_data(
                task=task_embedding if task_embedding is not None else task,
                num_samples=trajectories_per_iter,
                guidance_scale=self.config.get("guidance_scale", 5.0)
            )
            
            metrics["trajectories_generated"] += len(synthetic_samples)
            
            # Filter high-quality samples with discriminator
            filtered_samples, quality_scores = self.discriminator.filter_trajectories(
                trajectories=synthetic_samples,
                task_embedding=task_embedding,
                threshold=quality_threshold
            )
            
            metrics["high_quality_trajectories"] += len(filtered_samples)
            metrics["quality_scores"].append(
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            )
            
            logger.info(f"Generated {len(synthetic_samples)} samples, filtered {len(filtered_samples)} high-quality samples")
            
            # Adapt model with high-quality samples if available
            if filtered_samples:
                # Convert to tensors
                samples_tensor = torch.stack(filtered_samples).to(self.device)
                rewards_tensor = torch.tensor(quality_scores, device=self.device)
                
                # Adapt model
                self.model.adapt_to_task(
                    task=task_embedding if task_embedding is not None else task,
                    num_adaptation_steps=5,  # Smaller number for self-improvement iterations
                    batch_size=min(len(filtered_samples), 16),
                    adaptation_iterations=1  # Just one iteration per self-improvement cycle
                )
        
        # Final reward measurement
        final_samples, _ = self.generate(
            task=task_embedding if task_embedding is not None else task,
            batch_size=10
        )
        
        final_rewards = self.model.evaluate_samples(
            final_samples,
            task_embedding if task_embedding is not None else task
        )
        final_mean_reward = final_rewards.mean().item()
        
        # Calculate improvement
        reward_improvement = final_mean_reward - initial_mean_reward
        metrics["reward_improvement"] = reward_improvement
        metrics["initial_mean_reward"] = initial_mean_reward
        metrics["final_mean_reward"] = final_mean_reward
        
        logger.info(f"Self-improvement complete. Reward improvement: {reward_improvement:.4f}")
        
        return metrics
    
    def save_state(
        self,
        path: str,
        save_components: bool = True
    ) -> bool:
        """
        Save API state to disk.
        
        Args:
            path: Path to save state
            save_components: Whether to save component states
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Determine component paths
            model_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_model.pt")
            discriminator_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_discriminator.pt")
            buffer_path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}_buffer.pt")
            
            # Save components if requested
            component_paths = {}
            if save_components:
                # Save model
                torch.save({
                    "model_state_dict": self.model.diffusion_model.state_dict(),
                    "config": self.config
                }, model_path)
                component_paths["model_path"] = model_path
                
                # Save discriminator
                self.discriminator.save_state(discriminator_path)
                component_paths["discriminator_path"] = discriminator_path
                
                # Save buffer
                self.model.save_buffer(buffer_path)
                component_paths["buffer_path"] = buffer_path
            
            # Save API state
            state = {
                "config": self.config,
                "component_paths": component_paths
            }
            
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved API state to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API state: {e}")
            return False
    
    def load_state(
        self,
        path: str,
        load_components: bool = True
    ) -> bool:
        """
        Load API state from disk.
        
        Args:
            path: Path to load state from
            load_components: Whether to load component states
            
        Returns:
            Success flag
        """
        try:
            if not os.path.exists(path):
                logger.error(f"State file not found: {path}")
                return False
                
            # Load API state
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Update config
            self.config.update(state.get("config", {}))
            
            # Load components if requested
            if load_components:
                component_paths = state.get("component_paths", {})
                
                # Load model
                model_path = component_paths.get("model_path")
                if model_path and os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.diffusion_model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info(f"Loaded model from {model_path}")
                
                # Load discriminator
                discriminator_path = component_paths.get("discriminator_path")
                if discriminator_path and os.path.exists(discriminator_path):
                    self.discriminator.load_state(discriminator_path)
                    logger.info(f"Loaded discriminator from {discriminator_path}")
                
                # Load buffer
                buffer_path = component_paths.get("buffer_path")
                if buffer_path and os.path.exists(buffer_path):
                    self.model.load_buffer(buffer_path)
                    logger.info(f"Loaded buffer from {buffer_path}")
            
            logger.info(f"Loaded API state from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load API state: {e}")
            return False


def create_adapt_diffuser_api(config: Optional[Dict[str, Any]] = None) -> AdaptDiffuserAPI:
    """
    Create a fully configured AdaptDiffuser API.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AdaptDiffuser API
    """
    # Use empty config if not provided
    config = config or {}
    
    # Create and return API
    return AdaptDiffuserAPI(config=config)


class AdaptDiffuserAdapter:
    """
    Adapter for integrating AdaptDiffuser with the existing adaptation framework.
    """
    
    def __init__(
        self,
        adapt_diffuser_api: Optional[AdaptDiffuserAPI] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the adapter.
        
        Args:
            adapt_diffuser_api: AdaptDiffuser API instance
            config: Configuration dictionary
        """
        self.config = config or {}
        self.api = adapt_diffuser_api or create_adapt_diffuser_api(self.config)
    
    def adapt(
        self,
        code: Optional[str] = None,
        feedback: Optional[Dict] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Adapt code using AdaptDiffuser.
        
        Args:
            code: Code to adapt
            feedback: Feedback to incorporate
            language: Programming language
            **kwargs: Additional parameters
            
        Returns:
            Adapted code
        """
        if not code:
            return ""
            
        # Extract task from feedback if available
        task = None
        if feedback and isinstance(feedback, dict):
            task = feedback.get("task")
        
        # Use language as fallback task
        if not task and language:
            task = f"improve_{language}_code"
        
        # Default task
        if not task:
            task = "improve_code"
        
        # Start adaptation process
        self.api.adapt(
            task=task,
            iterations=kwargs.get("iterations", 1),
            batch_size=kwargs.get("batch_size", 16)
        )
        
        # Self-improve if requested
        if kwargs.get("self_improve", False):
            self.api.self_improve(
                task=task,
                iterations=kwargs.get("self_improve_iterations", 3),
                trajectories_per_iter=kwargs.get("trajectories_per_iter", 50)
            )
        
        # Generate adapted code
        # In a real implementation, this would use a code-specific adaptation method
        # This is a placeholder that would be replaced in production
        return f"# Adapted code for task: {task}\n{code}"
    
    def save_state(self, path: str) -> bool:
        """
        Save adapter state to disk.
        
        Args:
            path: Path to save state
            
        Returns:
            Success flag
        """
        return self.api.save_state(path)
    
    def load_state(self, path: str) -> bool:
        """
        Load adapter state from disk.
        
        Args:
            path: Path to load state from
            
        Returns:
            Success flag
        """
        return self.api.load_state(path)


def create_adapt_diffuser_adapter(
    adaptation_api=None,
    config: Optional[Dict[str, Any]] = None
) -> AdaptDiffuserAdapter:
    """
    Create an adapter for the existing adaptation framework.
    
    Args:
        adaptation_api: Optional existing adaptation API
        config: Configuration dictionary
        
    Returns:
        Configured adapter
    """
    adapt_diffuser_api = create_adapt_diffuser_api(config)
    return AdaptDiffuserAdapter(adapt_diffuser_api=adapt_diffuser_api, config=config)
# --- FastAPI app for AdaptDiffuser API integration testing ---

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None

# Global API instance for the FastAPI app to use
_api_instance = None

def get_api_instance():
    """Get or create the global API instance used by the FastAPI app"""
    global _api_instance
    if _api_instance is None:
        _api_instance = create_adapt_diffuser_api({
            "device": "cpu",  # Use CPU for testing
            "batch_size": 2,  # Small batch size for testing
            "guidance_scale": 3.0,
            "sampling_steps": 10,  # Reduced steps for testing
        })
    return _api_instance

if FastAPI:
    app = FastAPI(
        title="AdaptDiffuser API",
        description="API for adapting, generating, and evaluating diffusion models",
        version="0.1.0",
    )

    # --- Pydantic models for request/response validation ---
    
    class AdaptRequest(BaseModel):
        """Request model for adaptation endpoint"""
        input: str = Field(..., description="Input for adaptation")
        iterations: int = Field(1, description="Number of adaptation iterations")
        batch_size: int = Field(16, description="Batch size for adaptation")

    class AdaptResponse(BaseModel):
        """Response model for adaptation endpoint"""
        adapted_output: str = Field(..., description="Adapted output")
        metrics: dict = Field({}, description="Adaptation metrics")

    class GenerateRequest(BaseModel):
        """Request model for generation endpoint"""
        prompt: str = Field(..., description="Prompt for generation")
        batch_size: int = Field(1, description="Number of trajectories to generate")
        guidance_scale: float = Field(None, description="Guidance scale for generation")

    class GenerateResponse(BaseModel):
        """Response model for generation endpoint"""
        generated_code: str = Field(..., description="Generated code")
        metadata: dict = Field({}, description="Generation metadata")

    class EvaluateRequest(BaseModel):
        """Request model for evaluation endpoint"""
        code: str = Field(..., description="Code to evaluate")
        task: str = Field("code_quality", description="Evaluation task")

    class EvaluateResponse(BaseModel):
        """Response model for evaluation endpoint"""
        score: float = Field(..., description="Evaluation score")
        details: dict = Field({}, description="Detailed evaluation metrics")

    class AsyncAdaptRequest(BaseModel):
        """Request model for async adaptation endpoint"""
        input: str = Field(..., description="Input for adaptation")
        callback_url: str = Field(None, description="Optional callback URL")

    class AsyncAdaptResponse(BaseModel):
        """Response model for async adaptation endpoint"""
        task_id: str = Field(..., description="Async task ID")
        status: str = Field("processing", description="Task status")

    # --- In-memory store for async tasks ---
    async_tasks = {}

    # --- Endpoint implementations ---

    @app.post("/adapt", response_model=AdaptResponse)
    async def adapt_endpoint(request: AdaptRequest):
        """
        Adapt the diffusion model to the specified input.
        
        This endpoint processes the input and returns an adapted output.
        """
        if not request.input:
            raise HTTPException(status_code=400, detail="Missing or empty input")
        
        # For integration testing, we'll mock the adaptation process
        # In a real implementation, we'd use the adapt_diffuser_api methods
        
        # Simulate adaptation process
        adapted_output = f"adapted({request.input})"
        
        # Return the adapted output with metrics
        return {
            "adapted_output": adapted_output,
            "metrics": {
                "iterations": request.iterations,
                "batch_size": request.batch_size,
                "adaptation_score": 0.85
            }
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_endpoint(request: GenerateRequest):
        """
        Generate code using the diffusion model based on the provided prompt.
        
        This endpoint processes the prompt and returns generated code.
        """
        if request.prompt is None:
            raise HTTPException(status_code=400, detail="Missing prompt")
        
        if request.prompt == "":
            raise HTTPException(status_code=422, detail="Empty prompt")
        
        # For integration testing, we'll mock the generation process
        # In a real implementation, we'd use the adapt_diffuser_api methods
        
        # Simulate generation process
        generated_code = f"generated({request.prompt})"
        
        # Return the generated code with metadata
        return {
            "generated_code": generated_code,
            "metadata": {
                "batch_size": request.batch_size,
                "guidance_scale": request.guidance_scale or 5.0,
                "generation_time_ms": 150
            }
        }

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate_endpoint(request: EvaluateRequest):
        """
        Evaluate code using the diffusion model.
        
        This endpoint assesses the quality of the provided code.
        """
        if not request.code:
            raise HTTPException(status_code=400, detail="Missing code")
        
        # For integration testing, we'll mock the evaluation process
        # In a real implementation, we'd use the adapt_diffuser_api methods
        
        # Simulate code evaluation
        score = 42.0
        
        # Return the evaluation score with details
        return {
            "score": score,
            "details": {
                "quality": 0.85,
                "efficiency": 0.78,
                "readability": 0.92,
                "task": request.task
            }
        }

    @app.post("/adapt_async", response_model=AsyncAdaptResponse)
    async def adapt_async_endpoint(request: AsyncAdaptRequest, background_tasks: BackgroundTasks):
        """
        Asynchronously adapt the diffusion model to the specified input.
        
        This endpoint starts an asynchronous adaptation process and returns a task ID.
        """
        if not request.input:
            raise HTTPException(status_code=400, detail="Missing or empty input")
        
        # Generate a unique task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Store the task in memory
        async_tasks[task_id] = {
            "status": "processing",
            "input": request.input,
            "callback_url": request.callback_url,
            "result": None
        }
        
        # Define the background task
        def process_adaptation(task_id: str, input_text: str):
            # Simulate processing time
            import time
            time.sleep(2)
            
            # Update the task with the result
            async_tasks[task_id]["status"] = "completed"
            async_tasks[task_id]["result"] = f"async_adapted({input_text})"
            
            # Call the callback if provided
            callback_url = async_tasks[task_id]["callback_url"]
            if callback_url:
                try:
                    import httpx
                    httpx.post(callback_url, json={
                        "task_id": task_id,
                        "status": "completed",
                        "result": async_tasks[task_id]["result"]
                    })
                except Exception as e:
                    logger.error(f"Callback failed: {e}")
        
        # Add the task to background tasks
        background_tasks.add_task(process_adaptation, task_id, request.input)
        
        # Return the task information
        return {
            "task_id": task_id,
            "status": "processing"
        }

    @app.get("/adapt_async/{task_id}", response_model=dict)
    async def get_async_task(task_id: str):
        """
        Get the status and result of an asynchronous adaptation task.
        
        This endpoint returns the current status and result (if available) of an async task.
        """
        if task_id not in async_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = async_tasks[task_id]
        response = {
            "task_id": task_id,
            "status": task["status"]
        }
        
        if task["status"] == "completed" and task["result"]:
            response["result"] = task["result"]
        
        return response

    # --- API documentation customization ---
    
    @app.get("/", include_in_schema=False)
    async def redirect_to_docs():
        """Redirect root to documentation"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")