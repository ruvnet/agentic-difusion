"""
Trajectory selection strategies for AdaptDiffuser.

This module implements various methods for selecting high-quality synthetic data
from the trajectory buffer for model adaptation.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union, Any

from agentic_diffusion.core.trajectory_buffer import AdaptDiffuserTrajectoryBuffer

# Configure logging
logger = logging.getLogger(__name__)


class SelectionStrategy:
    """Base class for trajectory selection strategies."""
    
    def select(
        self, 
        buffer: AdaptDiffuserTrajectoryBuffer,
        task: Optional[Union[str, torch.Tensor]] = None,
        batch_size: int = 32
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Select trajectories from buffer.
        
        Args:
            buffer: Trajectory buffer to select from
            task: Optional task identifier or embedding
            batch_size: Number of trajectories to select
            
        Returns:
            Tuple of (selected_trajectories, rewards)
        """
        raise NotImplementedError("Selection strategy must implement select method")


class TopKSelection(SelectionStrategy):
    """
    Select top-K trajectories based on reward.
    
    This strategy simply selects the K trajectories with highest rewards.
    """
    
    def select(
        self, 
        buffer: AdaptDiffuserTrajectoryBuffer,
        task: Optional[Union[str, torch.Tensor]] = None,
        batch_size: int = 32
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Select top-K trajectories based on reward.
        
        Args:
            buffer: Trajectory buffer to select from
            task: Optional task identifier or embedding
            batch_size: Number of trajectories to select (K)
            
        Returns:
            Tuple of (selected_trajectories, rewards)
        """
        return buffer.get_task_trajectories(task, limit=batch_size)


class TemperatureSelection(SelectionStrategy):
    """
    Select trajectories based on softmax temperature sampling.
    
    This strategy uses a temperature parameter to control the randomness
    of selection, where higher temperature leads to more exploration.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize temperature selection.
        
        Args:
            temperature: Sampling temperature (higher = more uniform)
        """
        self.temperature = temperature
    
    def select(
        self, 
        buffer: AdaptDiffuserTrajectoryBuffer,
        task: Optional[Union[str, torch.Tensor]] = None,
        batch_size: int = 32
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Select trajectories using temperature sampling.
        
        Args:
            buffer: Trajectory buffer to select from
            task: Optional task identifier or embedding
            batch_size: Number of trajectories to select
            
        Returns:
            Tuple of (selected_trajectories, rewards)
        """
        trajectories, rewards, _ = buffer.temperature_sample(
            batch_size=batch_size,
            temperature=self.temperature,
            task=task
        )
        return trajectories, rewards


class DiversitySelection(SelectionStrategy):
    """
    Select diverse trajectories using clustering.
    
    This strategy aims to select a diverse set of trajectories by
    clustering them and selecting representatives from each cluster.
    """
    
    def __init__(self, n_clusters: int = 5, device: str = None):
        """
        Initialize diversity selection.
        
        Args:
            n_clusters: Number of clusters to form
            device: Device to perform computations on
        """
        self.n_clusters = n_clusters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def select(
        self, 
        buffer: AdaptDiffuserTrajectoryBuffer,
        task: Optional[Union[str, torch.Tensor]] = None,
        batch_size: int = 32
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Select diverse trajectories using clustering.
        
        Args:
            buffer: Trajectory buffer to select from
            task: Optional task identifier or embedding
            batch_size: Number of trajectories to select
            
        Returns:
            Tuple of (selected_trajectories, rewards)
        """
        # Get all trajectories for the task
        trajectories, rewards = buffer.get_task_trajectories(task)
        
        if not trajectories:
            logger.warning(f"No trajectories found for task {task}")
            return [], []
        
        # If we have fewer trajectories than batch_size, return all
        if len(trajectories) <= batch_size:
            return trajectories, rewards
        
        # Flatten trajectories for clustering
        flat_trajectories = [t.flatten().cpu().numpy() for t in trajectories]
        
        try:
            # Import sklearn only when needed to avoid dependency issues
            from sklearn.cluster import KMeans
            
            # Adjust n_clusters if needed
            actual_clusters = min(self.n_clusters, len(trajectories))
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=actual_clusters)
            cluster_labels = kmeans.fit_predict(flat_trajectories)
            
            # Select top trajectories from each cluster
            selected_indices = []
            
            # Get indices of trajectories in each cluster
            cluster_indices = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_indices:
                    cluster_indices[label] = []
                cluster_indices[label].append(i)
            
            # Select top trajectories from each cluster based on reward
            per_cluster = max(1, batch_size // actual_clusters)
            remaining = batch_size
            
            for label, indices in cluster_indices.items():
                # Sort indices by reward (descending)
                sorted_indices = sorted(indices, key=lambda i: rewards[i], reverse=True)
                # Take top per_cluster trajectories
                to_take = min(per_cluster, len(sorted_indices), remaining)
                selected_indices.extend(sorted_indices[:to_take])
                remaining -= to_take
            
            # If we still have remaining slots, fill with highest reward trajectories
            if remaining > 0:
                # Get all indices not already selected, sorted by reward
                remaining_indices = [i for i in range(len(trajectories)) if i not in selected_indices]
                remaining_indices = sorted(remaining_indices, key=lambda i: rewards[i], reverse=True)
                # Add the top remaining ones
                selected_indices.extend(remaining_indices[:remaining])
            
            # Return selected trajectories and rewards
            selected_trajectories = [trajectories[i] for i in selected_indices]
            selected_rewards = [rewards[i] for i in selected_indices]
            
            return selected_trajectories, selected_rewards
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to top-K selection")
            return TopKSelection().select(buffer, task, batch_size)


class HybridSelection(SelectionStrategy):
    """
    Hybrid selection strategy combining multiple methods.
    
    This strategy combines different selection methods with specified weights.
    """
    
    def __init__(
        self,
        strategies: List[SelectionStrategy],
        weights: List[float],
    ):
        """
        Initialize hybrid selection.
        
        Args:
            strategies: List of selection strategies
            weights: Weights for each strategy (must sum to 1.0)
            
        Raises:
            ValueError: If weights don't sum to 1.0 or lengths don't match
        """
        if len(strategies) != len(weights):
            raise ValueError("Number of strategies must match number of weights")
            
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
            
        self.strategies = strategies
        self.weights = weights
    
    def select(
        self, 
        buffer: AdaptDiffuserTrajectoryBuffer,
        task: Optional[Union[str, torch.Tensor]] = None,
        batch_size: int = 32
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Select trajectories using multiple strategies.
        
        Args:
            buffer: Trajectory buffer to select from
            task: Optional task identifier or embedding
            batch_size: Number of trajectories to select
            
        Returns:
            Tuple of (selected_trajectories, rewards)
        """
        # Calculate how many trajectories to select with each strategy
        counts = []
        remaining = batch_size
        
        for i, weight in enumerate(self.weights[:-1]):
            count = int(batch_size * weight)
            counts.append(count)
            remaining -= count
        
        # Assign remaining to last strategy
        counts.append(remaining)
        
        # Select trajectories using each strategy
        all_trajectories = []
        all_rewards = []
        
        for strategy, count in zip(self.strategies, counts):
            if count <= 0:
                continue
                
            trajectories, rewards = strategy.select(buffer, task, count)
            all_trajectories.extend(trajectories)
            all_rewards.extend(rewards)
        
        return all_trajectories, all_rewards