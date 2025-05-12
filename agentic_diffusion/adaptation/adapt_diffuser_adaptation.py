import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.core.trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)

class SyntheticExpertGenerator:
    """
    Generates synthetic expert data by optimizing for high rewards.
    """
    def __init__(
        self,
        adapt_diffuser: AdaptDiffuser,
        batch_size: int = 8,
        refinement_iterations: int = 5,
        quality_threshold: float = 0.7,
        gradient_scale: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the synthetic expert data generator.
        
        Args:
            adapt_diffuser: AdaptDiffuser model
            batch_size: Batch size for generation
            refinement_iterations: Number of refinement iterations
            quality_threshold: Quality threshold for filtering samples
            gradient_scale: Scale for reward gradients
            device: Device to use
        """
        self.adapt_diffuser = adapt_diffuser
        self.batch_size = batch_size
        self.refinement_iterations = refinement_iterations
        self.quality_threshold = quality_threshold
        self.gradient_scale = gradient_scale
        self.device = device
        
    def generate_synthetic_data(
        self,
        task: torch.Tensor,
        num_samples: int = 10,
        guidance_scale: float = 5.0
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Generate synthetic expert data by optimizing for high rewards.
        
        Args:
            task: Task embedding
            num_samples: Number of high-quality samples to generate
            guidance_scale: Guidance scale for generation
            
        Returns:
            Tuple of (high-quality samples, reward scores)
        """
        high_quality_samples = []
        high_quality_rewards = []
        best_samples = []
        best_rewards = []
        
        # Encode task if necessary
        task_embedding = None
        if task is not None:
            task_embedding = self.adapt_diffuser.encode_task(task)
        
        # Generate in batches until we have enough high-quality samples
        batches_needed = max(1, num_samples // self.batch_size)
        
        for batch_idx in range(batches_needed * 2):  # Allow extra batches if needed
            if len(high_quality_samples) >= num_samples:
                break
                
            # Generate initial batch with high guidance
            samples = self.adapt_diffuser.generate(
                batch_size=self.batch_size,
                task=task_embedding,
                custom_guidance_scale=guidance_scale
            )
            
            # Apply refinement iterations
            for _ in range(self.refinement_iterations):
                # Compute rewards for current samples
                rewards = self.adapt_diffuser.compute_reward(samples, task_embedding)
                
                # Use reward gradients to refine samples
                gradients = self._compute_reward_gradients(samples, rewards, task_embedding)
                samples = self._refine_samples_with_gradients(samples, gradients)
            
            # Final evaluation
            final_rewards = self.adapt_diffuser.compute_reward(samples, task_embedding)
            
            # Convert final_rewards to tensor if needed
            if isinstance(final_rewards, (float, int)):
                final_rewards = torch.tensor([final_rewards], device=samples.device)
            elif isinstance(final_rewards, list):
                final_rewards = torch.tensor(final_rewards, device=samples.device)
            elif not isinstance(final_rewards, torch.Tensor):
                final_rewards = torch.tensor(final_rewards, device=samples.device)
            
            # Track the best samples from each batch, regardless of threshold
            if final_rewards.dim() == 0:  # If it's a scalar
                best_samples.append(samples[0])
                best_rewards.append(final_rewards.item())
            else:
                for i in range(min(len(samples), len(final_rewards))):
                    best_samples.append(samples[i])
                    best_rewards.append(final_rewards[i].item())
            
            # Filter high-quality samples based on threshold
            if final_rewards.dim() == 0:  # If it's a scalar
                if final_rewards.item() >= self.quality_threshold:
                    high_quality_samples.append(samples[0])
                    high_quality_rewards.append(final_rewards.item())
            else:
                for i in range(min(len(samples), len(final_rewards))):
                    if final_rewards[i].item() >= self.quality_threshold:
                        high_quality_samples.append(samples[i])
                        high_quality_rewards.append(final_rewards[i].item())
            
            logger.info(f"Batch {batch_idx+1}: Generated {len(high_quality_samples)}/{num_samples} high-quality samples")
        
        # If we didn't find enough high-quality samples, use the best ones we found
        if len(high_quality_samples) == 0 and len(best_samples) > 0:
            logger.warning(f"No samples met the quality threshold of {self.quality_threshold}. Using the best samples found.")
            # Sort best samples by reward
            sorted_pairs = sorted(zip(best_samples, best_rewards), key=lambda x: x[1], reverse=True)
            best_samples, best_rewards = zip(*sorted_pairs)
            # Take up to num_samples
            high_quality_samples = list(best_samples[:num_samples])
            high_quality_rewards = list(best_rewards[:num_samples])
        
        return high_quality_samples, high_quality_rewards
    
    def _compute_reward_gradients(
        self,
        samples: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradients of reward with respect to samples.
        
        Args:
            samples: Tensor of shape (batch_size, dim)
            rewards: Optional precomputed rewards
            task_embedding: Optional task embedding
            
        Returns:
            Gradient tensor of same shape as samples
        """
        # Make samples require gradients
        samples_with_grad = samples.detach().clone().requires_grad_(True)
        
        # Compute rewards with gradients
        if hasattr(self.adapt_diffuser, 'reward_model') and self.adapt_diffuser.reward_model is not None:
            new_rewards = self.adapt_diffuser.reward_model.compute_reward(samples_with_grad, task_embedding)
            
            # Convert rewards to tensor if needed
            if isinstance(new_rewards, (float, int)):
                new_rewards = torch.tensor([new_rewards], device=samples.device, requires_grad=True)
            elif isinstance(new_rewards, list):
                new_rewards = torch.tensor(new_rewards, device=samples.device, requires_grad=True)
            elif not isinstance(new_rewards, torch.Tensor):
                new_rewards = torch.tensor(new_rewards, device=samples.device, requires_grad=True)
            elif not new_rewards.requires_grad:
                new_rewards = new_rewards.detach().clone().requires_grad_(True)
        else:
            logger.warning("No reward model available for gradient computation")
            return torch.zeros_like(samples)
        
        # Compute gradient of mean reward
        reward_mean = new_rewards.mean()
        reward_mean.backward()
        
        # Return gradients
        if samples_with_grad.grad is None:
            # If gradients couldn't be computed, return zeros
            logger.warning("Failed to compute gradients, returning zeros")
            return torch.zeros_like(samples)
        else:
            gradients = samples_with_grad.grad.clone()
            return gradients
    
    def _refine_samples_with_gradients(
        self,
        samples: torch.Tensor,
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine samples using reward gradients.
        
        Args:
            samples: Tensor of shape (batch_size, dim)
            gradients: Gradient tensor of same shape as samples
            
        Returns:
            Refined samples
        """
        # Add scaled gradients to samples
        refined_samples = samples + self.gradient_scale * gradients
        
        # Normalize if embeddings should be normalized
        if hasattr(self.adapt_diffuser, 'normalized_embeddings') and self.adapt_diffuser.normalized_embeddings:
            refined_samples = F.normalize(refined_samples, p=2, dim=-1)
            
        return refined_samples


class AdaptationMechanism:
    """Base class for adaptation mechanisms."""
    
    def __init__(self, adapt_diffuser: AdaptDiffuser):
        """
        Initialize adaptation mechanism.
        
        Args:
            adapt_diffuser: AdaptDiffuser model
        """
        self.adapt_diffuser = adapt_diffuser
        
    def adapt_to_task(
        self,
        task: torch.Tensor,
        num_adaptation_steps: int,
        learning_rate: float = 1e-4,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Adapt model to a task.
        
        Args:
            task: Task embedding or description
            num_adaptation_steps: Number of adaptation steps
            learning_rate: Learning rate for adaptation
            batch_size: Batch size for adaptation
            
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement adapt_to_task")


class TrajectoryDiscriminator(nn.Module):
    """
    Discriminator for filtering high-quality trajectories.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        task_embedding_dim: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trajectory discriminator.
        
        Args:
            input_dim: Dimension of input trajectories
            hidden_dim: Hidden dimension of discriminator
            task_embedding_dim: Dimension of task embeddings (optional)
            device: Device to use
        """
        super().__init__()
        self.device = device
        self.to(device)
        
        # Input layers
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task-conditional layers
        self.use_task_embedding = task_embedding_dim is not None
        if self.use_task_embedding:
            self.task_encoder = nn.Sequential(
                nn.Linear(task_embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            
            # Fusion layer
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        
        # Quality prediction
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        trajectories: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            trajectories: Trajectories to evaluate
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Quality scores [0-1]
        """
        # Encode trajectories
        traj_features = self.trajectory_encoder(trajectories)
        
        # Incorporate task embedding if available
        if self.use_task_embedding and task_embedding is not None:
            task_features = self.task_encoder(task_embedding)
            
            # Expand task features if needed
            if task_features.dim() == 1 and traj_features.dim() > 1:
                task_features = task_features.unsqueeze(0).expand(traj_features.size(0), -1)
                
            # Fuse features
            combined_features = self.fusion(torch.cat([traj_features, task_features], dim=-1))
        else:
            combined_features = traj_features
            
        # Predict quality
        quality = self.quality_head(combined_features)
        
        return quality
    
    def evaluate_quality(
        self,
        samples: Union[torch.Tensor, List[torch.Tensor]],
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluate quality of samples.
        
        Args:
            samples: Tensor or list of samples to evaluate
            task_embedding: Optional task embedding for conditioning
            
        Returns:
            Quality scores [0-1]
        """
        self.eval()
        
        # Check for empty list
        if isinstance(samples, list) and len(samples) == 0:
            logger.warning("Empty sample list provided to evaluate_quality")
            return torch.tensor([], device=self.device)
        
        with torch.no_grad():
            if isinstance(samples, list):
                samples = torch.stack(samples).to(self.device)
            else:
                samples = samples.to(self.device)
                
            if task_embedding is not None:
                task_embedding = task_embedding.to(self.device)
                
            quality_scores = self(samples, task_embedding)
        
        return quality_scores
    
    def filter_trajectories(
        self,
        trajectories: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Filter trajectories based on quality scores.
        
        Args:
            trajectories: Tensor or list of trajectories to filter
            task_embedding: Optional task embedding for conditioning
            threshold: Quality threshold [0-1]
            
        Returns:
            Tuple of (filtered_trajectories, quality_scores)
        """
        # Handle empty trajectories
        if isinstance(trajectories, list) and len(trajectories) == 0:
            logger.warning("Empty trajectory list provided to filter_trajectories")
            return [], []
            
        # Evaluate quality
        quality_scores = self.evaluate_quality(trajectories, task_embedding)
        
        # Handle empty quality scores
        if quality_scores.numel() == 0:
            logger.warning("No quality scores returned from evaluate_quality")
            return [], []
        
        # Filter trajectories
        if isinstance(trajectories, torch.Tensor):
            mask = quality_scores.squeeze() >= threshold
            if mask.numel() == 0:
                return [], []
            filtered_trajectories = [trajectories[i] for i in range(len(trajectories)) if mask[i]]
            filtered_scores = [quality_scores[i].item() for i in range(len(quality_scores)) if mask[i]]
        else:
            # Handle list of trajectories
            mask = quality_scores.squeeze() >= threshold
            if mask.numel() == 0:
                return [], []
            filtered_trajectories = [trajectories[i] for i in range(len(trajectories)) if mask[i]]
            filtered_scores = [quality_scores[i].item() for i in range(len(quality_scores)) if mask[i]]
        
        return filtered_trajectories, filtered_scores
    
    def save_state(self, path: str) -> bool:
        """
        Save the discriminator state to disk.
        
        Args:
            path: Path to save the state
            
        Returns:
            Success flag
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        return True
    
    def load_state(self, path: str) -> bool:
        """
        Load the discriminator state from disk.
        
        Args:
            path: Path to load the state from
            
        Returns:
            Success flag
        """
        if not os.path.exists(path):
            logger.error(f"State file does not exist: {path}")
            return False
            
        self.load_state_dict(torch.load(path, map_location=self.device))
        return True


class AdaptDiffuserAdaptation(AdaptationMechanism):
    """
    Adaptation mechanism for AdaptDiffuser.
    """
    def __init__(
        self,
        adapt_diffuser: AdaptDiffuser,
        synthetic_expert_generator: Optional[SyntheticExpertGenerator] = None,
        discriminator: Optional[TrajectoryDiscriminator] = None,
        adaptation_rate: float = 0.1,
        quality_threshold: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize AdaptDiffuser adaptation mechanism.
        
        Args:
            adapt_diffuser: AdaptDiffuser model
            synthetic_expert_generator: Synthetic expert data generator
            discriminator: Trajectory discriminator
            adaptation_rate: Learning rate for adaptation
            quality_threshold: Quality threshold for filtering samples
            device: Device to use
        """
        super().__init__(adapt_diffuser)
        
        # Store configuration parameters
        self.adaptation_rate = adaptation_rate
        self.quality_threshold = quality_threshold
        
        # Create synthetic expert generator if not provided
        input_dim = adapt_diffuser.embedding_dim if hasattr(adapt_diffuser, 'embedding_dim') else 128
        device = device or (adapt_diffuser.device if hasattr(adapt_diffuser, 'device') else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Create synthetic expert generator if not provided
        if synthetic_expert_generator is None:
            self.synthetic_expert_generator = SyntheticExpertGenerator(
                adapt_diffuser=adapt_diffuser,
                device=device
            )
        else:
            self.synthetic_expert_generator = synthetic_expert_generator
            
        # Create discriminator if not provided
        if discriminator is None:
            task_embedding_dim = adapt_diffuser.task_embedding_dim if hasattr(adapt_diffuser, 'task_embedding_dim') else None
            self.discriminator = TrajectoryDiscriminator(
                input_dim=input_dim,
                task_embedding_dim=task_embedding_dim,
                device=device
            )
        else:
            self.discriminator = discriminator
            
        self.device = device
        
    def adapt_to_task(
        self,
        task: torch.Tensor,
        num_adaptation_steps: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        guidance_scale: float = 5.0,
        quality_threshold: float = 0.7,
        min_reward_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Adapt model to a task.
        
        Args:
            task: Task embedding or description
            num_adaptation_steps: Number of adaptation steps
            learning_rate: Learning rate for adaptation
            batch_size: Batch size for adaptation
            guidance_scale: Guidance scale for generation
            quality_threshold: Quality threshold for discriminator
            min_reward_threshold: Minimum reward threshold for storing samples
            
        Returns:
            Dictionary of metrics
        """
        # Encode task if necessary
        task_embedding = None
        if not torch.is_tensor(task):
            task_embedding = self.adapt_diffuser.encode_task(task)
        else:
            task_embedding = task
            
        # Initialize metrics
        metrics = {
            "steps": [],
            "mean_rewards": [],
            "max_rewards": [],
            "min_rewards": [],
            "losses": [],
            "high_quality_samples": 0,
            "total_samples": 0,
            "adaptation_time": 0.0
        }
        
        # Main adaptation loop
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        for step in range(num_adaptation_steps):
            # Generate synthetic expert data
            synthetic_samples, synthetic_rewards = self.synthetic_expert_generator.generate_synthetic_data(
                task=task_embedding,
                num_samples=batch_size,
                guidance_scale=guidance_scale
            )
            
            metrics["total_samples"] += len(synthetic_samples)
            
            # Filter high-quality samples with discriminator
            filtered_samples, quality_scores = self.discriminator.filter_trajectories(
                trajectories=synthetic_samples,
                task_embedding=task_embedding,
                threshold=quality_threshold
            )
            
            metrics["high_quality_samples"] += len(filtered_samples)
            
            # Adapt model with examples
            if filtered_samples:
                adaptation_metrics = self.adapt_diffuser.adapt_with_examples(
                    samples=filtered_samples,
                    task=task_embedding,
                    learning_rate=learning_rate
                )
                
                # Update metrics
                metrics["mean_rewards"].append(adaptation_metrics.get("mean_reward", 0))
                metrics["max_rewards"].append(adaptation_metrics.get("max_reward", 0))
                metrics["min_rewards"].append(adaptation_metrics.get("min_reward", 0))
                metrics["losses"].append(adaptation_metrics.get("loss", 0))
            else:
                logger.warning(f"Step {step+1}/{num_adaptation_steps}: No high-quality samples found")
                metrics["mean_rewards"].append(0)
                metrics["max_rewards"].append(0)
                metrics["min_rewards"].append(0)
                metrics["losses"].append(0)
                
            metrics["steps"].append(step)
            
        end_time.record()
        torch.cuda.synchronize()
        
        metrics["adaptation_time"] = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        return metrics

# Helper function to create a standard adaptation mechanism
def create_adaptation_mechanism(
    adapt_diffuser: AdaptDiffuser,
    mechanism_type: str = "standard",
    adaptation_rate: float = 0.1,
    quality_threshold: float = 0.7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> AdaptationMechanism:
    """
    Create an adaptation mechanism of the specified type.
    
    Args:
        adapt_diffuser: AdaptDiffuser model
        mechanism_type: Type of adaptation mechanism
        adaptation_rate: Learning rate for adaptation
        quality_threshold: Quality threshold for filtering samples
        device: Device to use
        
    Returns:
        Adaptation mechanism
    """
    if mechanism_type == "standard":
        return AdaptDiffuserAdaptation(
            adapt_diffuser=adapt_diffuser,
            adaptation_rate=adaptation_rate,
            quality_threshold=quality_threshold,
            device=device
        )
    else:
        raise ValueError(f"Unknown adaptation mechanism type: {mechanism_type}")
