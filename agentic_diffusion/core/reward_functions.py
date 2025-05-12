import logging
import math
import random
from typing import Optional, List, Union

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

class RewardModel(nn.Module):
    """
    Base class for reward models used with AdaptDiffuser.
    
    A reward model evaluates the quality of a generated trajectory or sample
    with respect to a given task.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the reward model.
        
        Args:
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        self.to(device)
        
    def compute_reward(
        self,
        trajectories: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute rewards for trajectories.
        
        Args:
            trajectories: Tensor of shape (batch_size, dim) containing trajectories
            task_embedding: Optional tensor of shape (task_dim,) containing task embedding
            
        Returns:
            Tensor of shape (batch_size,) containing rewards
        """
        raise NotImplementedError("Subclasses must implement compute_reward")
    
    def batch_compute_reward(
        self,
        trajectory_batches: List[torch.Tensor],
        task_embedding: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Compute rewards for batches of trajectories.
        
        Args:
            trajectory_batches: List of tensors, each of shape (batch_size, dim)
            task_embedding: Optional tensor of shape (task_dim,) containing task embedding
            
        Returns:
            List of tensors, each of shape (batch_size,) containing rewards
        """
        return [self.compute_reward(batch, task_embedding) for batch in trajectory_batches]


class CosineRewardModel(RewardModel):
    """
    Reward model based on cosine similarity to a target.
    """
    
    def __init__(
        self,
        target_embedding: torch.Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize cosine reward model.
        
        Args:
            target_embedding: Target embedding to compare trajectories against
            device: Device to run the model on
        """
        super().__init__(device)
        self.register_buffer("target_embedding", target_embedding.to(device))
        
    def compute_reward(
        self,
        trajectories: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity rewards.
        
        Args:
            trajectories: Tensor of shape (batch_size, dim) containing trajectories
            task_embedding: Optional tensor containing task embedding (not used)
            
        Returns:
            Tensor of shape (batch_size,) containing rewards
        """
        # Ignore task_embedding, use registered target
        trajectories = trajectories.to(self.device)
        
        # Normalize inputs (just in case)
        trajectories_norm = torch.nn.functional.normalize(trajectories, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(self.target_embedding, p=2, dim=0)
        
        # Compute cosine similarity
        # Scale to [0, 1] range (cosine is in [-1, 1])
        cosine_sim = torch.matmul(trajectories_norm, target_norm)
        rewards = (cosine_sim + 1) / 2
        
        return rewards


class TaskConditionedRewardModel(RewardModel):
    """
    Neural network reward model conditioned on task embeddings.
    """
    
    def __init__(
        self,
        trajectory_dim: int,
        task_dim: int,
        hidden_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize task-conditioned reward model.
        
        Args:
            trajectory_dim: Dimension of trajectory embeddings
            task_dim: Dimension of task embeddings
            hidden_dim: Dimension of hidden layers
            device: Device to run the model on
        """
        super().__init__(device)
        
        # Trajectory encoder
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Task encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion and reward prediction
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_reward(
        self,
        trajectories: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute rewards based on task-conditioned model.
        
        Args:
            trajectories: Tensor of shape (batch_size, dim) containing trajectories
            task_embedding: Tensor of shape (task_dim,) containing task embedding
            
        Returns:
            Tensor of shape (batch_size,) containing rewards
        """
        if task_embedding is None:
            raise ValueError("Task embedding is required for TaskConditionedRewardModel")
        
        trajectories = trajectories.to(self.device)
        task_embedding = task_embedding.to(self.device)
        
        # Encode trajectories
        trajectory_features = self.trajectory_encoder(trajectories)
        
        # Encode task
        task_features = self.task_encoder(task_embedding)
        
        # Expand task features if needed (for batched trajectories)
        if task_features.dim() == 1 and trajectory_features.dim() > 1:
            task_features = task_features.unsqueeze(0).expand(trajectory_features.size(0), -1)
        
        # Combine features
        combined = torch.cat([trajectory_features, task_features], dim=-1)
        
        # Predict rewards
        rewards = self.fusion(combined).squeeze(-1)
        
        return rewards


class CompositeRewardModel(RewardModel):
    """
    A composite reward model that combines multiple reward models with weights.
    """
    
    def __init__(
        self,
        reward_models_or_config: Union[List[RewardModel], dict],
        weights: Optional[List[float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize composite reward model.
        
        Args:
            reward_models_or_config: List of reward models to combine or config dict
            weights: Optional list of weights for each model (default: equal weights)
            device: Device to run the model on
        """
        super().__init__(device)
        
        # Handle config dict case
        if isinstance(reward_models_or_config, dict):
            config = reward_models_or_config
            # Create default empty lists when loading from config
            self.reward_models = []
            self.weights = config.get('weights', [])
            
            # Models would need to be registered separately when using config
            logger.info(f"Initialized CompositeRewardModel from config (models must be registered separately)")
        else:
            self.reward_models = reward_models_or_config
            
            # If weights not provided, use equal weights
            if weights is None:
                weights = [1.0 / len(self.reward_models)] * len(self.reward_models)
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
            assert len(self.reward_models) == len(self.weights), "Number of models and weights must match"
            
            logger.info(f"Initialized CompositeRewardModel with {len(self.reward_models)} models")
        
    def compute_reward(
        self,
        trajectories: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted combination of rewards from all models.
        
        Args:
            trajectories: Tensor of shape (batch_size, dim) containing trajectories
            task_embedding: Optional tensor containing task embedding
            
        Returns:
            Tensor of shape (batch_size,) containing combined rewards
        """
        # Get rewards from each model
        all_rewards = []
        for model in self.reward_models:
            try:
                reward = model.compute_reward(trajectories, task_embedding)
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, device=self.device)
                all_rewards.append(reward)
            except Exception as e:
                logger.warning(f"Error in reward model: {e}")
                # If a model fails, use zeros for its contribution
                batch_size = trajectories.shape[0] if hasattr(trajectories, 'shape') else 1
                all_rewards.append(torch.zeros(batch_size, device=self.device))
        
        # Apply weights and sum
        weighted_rewards = torch.zeros_like(all_rewards[0])
        for reward, weight in zip(all_rewards, self.weights):
            weighted_rewards += reward * weight
            
        return weighted_rewards


class SimpleRewardModel(RewardModel):
    """
    A simple reward model for AdaptDiffuser that returns fixed or parameterized rewards.
    Useful for testing and development purposes.
    """
    
    def __init__(
        self,
        reward_value_or_config: Union[float, dict] = 0.7,
        noise_scale: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize simple reward model.
        
        Args:
            reward_value_or_config: Base reward value [0-1] or config dict
            noise_scale: Scale of random noise to add to rewards
            device: Device to run the model on
        """
        super().__init__(device)
        
        # Handle config dict case
        if isinstance(reward_value_or_config, dict):
            config = reward_value_or_config
            self.reward_value = config.get('reward_value', 0.7)
            self.noise_scale = config.get('noise_scale', 0.05)
        else:
            self.reward_value = reward_value_or_config
            self.noise_scale = noise_scale
        
        logger.info(f"Initialized SimpleRewardModel with reward_value={self.reward_value}, noise_scale={self.noise_scale}")
        
    def compute_reward(
        self,
        trajectories: Union[torch.Tensor, np.ndarray, List[float], float],
        task_embedding: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, float]:
        """
        Compute rewards with simple fixed value plus noise.
        
        Args:
            trajectories: Tensor, array, list, or float representing trajectories
            task_embedding: Optional tensor containing task embedding (not used)
            
        Returns:
            Tensor or float containing rewards
        """
        # Handle different input types
        if isinstance(trajectories, torch.Tensor):
            # For tensor input, return tensor rewards
            batch_size = trajectories.shape[0] if trajectories.dim() > 0 else 1
            noise = torch.randn(batch_size, device=self.device) * self.noise_scale
            rewards = torch.full((batch_size,), self.reward_value, device=self.device) + noise
            rewards = torch.clamp(rewards, 0.0, 1.0)
            return rewards
        elif isinstance(trajectories, np.ndarray):
            # For numpy array, return numpy rewards
            batch_size = trajectories.shape[0] if trajectories.ndim > 0 else 1
            noise = np.random.randn(batch_size) * self.noise_scale
            rewards = np.full((batch_size,), self.reward_value) + noise
            rewards = np.clip(rewards, 0.0, 1.0)
            return rewards
        elif isinstance(trajectories, list):
            # For list input, return list of rewards
            batch_size = len(trajectories)
            rewards = [min(1.0, max(0.0, self.reward_value + random.gauss(0, self.noise_scale)))
                      for _ in range(batch_size)]
            return torch.tensor(rewards, device=self.device)
        else:
            # For single input, return single reward
            reward = min(1.0, max(0.0, self.reward_value + random.gauss(0, self.noise_scale)))
            return reward


class AdaptDiffuserTestRewardModel(RewardModel):
    """
    A test reward model for AdaptDiffuser that returns increasingly better scores
    for successive evaluation calls, which allows testing iterative improvement.
    """
    
    def __init__(
        self,
        initial_reward_or_config: Union[float, dict] = 0.5,
        improvement_rate: float = 0.05,
        noise_scale: float = 0.02,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize test reward model.
        
        Args:
            initial_reward_or_config: Initial reward value [0-1] or config dict
            improvement_rate: Rate of improvement per evaluation
            noise_scale: Scale of random noise to add to rewards
            device: Device to run the model on
        """
        super().__init__(device)
        
        # Handle config dict case
        if isinstance(initial_reward_or_config, dict):
            config = initial_reward_or_config
            self.initial_reward = config.get('initial_reward', 0.5)
            self.improvement_rate = config.get('improvement_rate', 0.05)
            self.noise_scale = config.get('noise_scale', 0.02)
        else:
            self.initial_reward = initial_reward_or_config
            self.improvement_rate = improvement_rate
            self.noise_scale = noise_scale
            
        self.eval_count = 0
        
        logger.info(f"Initialized AdaptDiffuserTestRewardModel with initial_reward={self.initial_reward}, improvement_rate={self.improvement_rate}")
        
    def compute_reward(
        self,
        trajectories: Union[torch.Tensor, np.ndarray, List[float], float],
        task_embedding: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, float]:
        """
        Compute rewards for test purposes with progressive improvement.
        
        Args:
            trajectories: Tensor, array, list, or float representing trajectories
            task_embedding: Optional tensor containing task embedding (not used)
            
        Returns:
            Tensor or float containing rewards that improve with successive calls
        """
        # Update evaluation counter
        self.eval_count += 1
        
        # Calculate reward with improvement and noise
        base_reward = min(0.95, self.initial_reward + self.improvement_rate * math.log(1 + self.eval_count))
        
        # Handle different input types
        if isinstance(trajectories, torch.Tensor):
            # For tensor input, return tensor rewards
            batch_size = trajectories.shape[0] if trajectories.dim() > 0 else 1
            noise = torch.randn(batch_size, device=self.device) * self.noise_scale
            rewards = torch.full((batch_size,), base_reward, device=self.device) + noise
            rewards = torch.clamp(rewards, 0.0, 1.0)
            return rewards
        elif isinstance(trajectories, np.ndarray):
            # For numpy array, return numpy rewards
            batch_size = trajectories.shape[0] if trajectories.ndim > 0 else 1
            noise = np.random.randn(batch_size) * self.noise_scale
            rewards = np.full((batch_size,), base_reward) + noise
            rewards = np.clip(rewards, 0.0, 1.0)
            return rewards
        elif isinstance(trajectories, list):
            # For list input, return list of rewards
            batch_size = len(trajectories)
            rewards = [min(1.0, max(0.0, base_reward + random.gauss(0, self.noise_scale))) 
                      for _ in range(batch_size)]
            return torch.tensor(rewards, device=self.device)
        else:
            # For single input, return single reward
            reward = min(1.0, max(0.0, base_reward + random.gauss(0, self.noise_scale)))
            return reward