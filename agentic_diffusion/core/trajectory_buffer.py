"""
Trajectory buffer for storing and sampling high-quality trajectories.

This module contains classes for storing and sampling trajectories based on their 
reward values for prioritized experience replay. The AdaptDiffuserTrajectoryBuffer extends
this with task-specific trajectory storage and prioritization mechanisms.
"""

import torch
import numpy as np
import logging
import os
from typing import List, Union, Optional, Tuple, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class TrajectoryBuffer:
    """Buffer for storing and sampling high-quality trajectories."""
    
    def __init__(self, capacity: int):
        """
        Initialize trajectory buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
            
        Raises:
            ValueError: If capacity is not a positive integer
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity}")
        
        self.capacity = capacity
        self.trajectories = []
        self.rewards = []
        self.priorities = []
    
    def add(self, trajectory: torch.Tensor, reward: float):
        """
        Add trajectory to buffer.
        
        Args:
            trajectory: Trajectory data (tensor)
            reward: Reward value for trajectory
        """
        # If buffer is full, remove lowest priority trajectory
        if len(self.trajectories) >= self.capacity:
            idx = np.argmin(self.priorities)
            self.trajectories.pop(idx)
            self.rewards.pop(idx)
            self.priorities.pop(idx)
        
        # Add new trajectory
        self.trajectories.append(trajectory)
        self.rewards.append(reward)
        
        # Calculate priority based on reward (exponential prioritization)
        priority = np.exp(reward)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, replace: bool = False) -> List[torch.Tensor]:
        """
        Sample trajectories from buffer based on priorities.
        
        Args:
            batch_size: Number of trajectories to sample
            replace: Whether to sample with replacement
            
        Returns:
            List of sampled trajectories
        """
        # Return empty list if buffer is empty
        if len(self.trajectories) == 0:
            return []
        
        # Ensure batch_size is not larger than buffer size when sampling without replacement
        if not replace:
            batch_size = min(batch_size, len(self.trajectories))
        
        # Calculate sampling probabilities based on priorities
        probs = np.array(self.priorities) / sum(self.priorities)
        
        # Sample indices
        indices = np.random.choice(
            len(self.trajectories),
            size=batch_size,
            replace=replace,
            p=probs
        )
        
        # Return sampled trajectories
        return [self.trajectories[idx] for idx in indices]
    
    def size(self) -> int:
        """
        Get current buffer size.
        
        Returns:
            Number of trajectories in buffer
        """
        return len(self.trajectories)
    
    def clear(self):
        """Clear all trajectories from buffer."""
        self.trajectories = []
        self.rewards = []
        self.priorities = []


class AdaptDiffuserTrajectoryBuffer:
    """
    Memory structure for storing and retrieving high-quality trajectories for AdaptDiffuser.
    
    This buffer implements prioritized experience replay with task-specific filtering
    and importance sampling for efficient model adaptation.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.001,
        device: str = None
    ):
        """
        Initialize the trajectory buffer.
        
        Args:
            capacity: Maximum number of trajectories to store
            alpha: Priority exponent for sampling
            beta: Importance sampling exponent (corrects PER bias)
            beta_annealing: Rate to anneal beta toward 1
            device: Device to store tensors on
            
        Raises:
            ValueError: If capacity is not a positive integer
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity}")
            
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Storage structures
        self.trajectories = []  # List of trajectories
        self.rewards = []  # Corresponding rewards
        self.priorities = []  # Priority values (derived from rewards)
        self.task_ids = []  # Associated task identifiers
        self.metadata = []  # Additional metadata for each trajectory
        
        # Task-specific indexing for fast retrieval
        self.task_indices = {}  # Maps task_id to list of indices
        
        # Counter for updates to track beta annealing
        self.updates = 0
    
    def add(
        self,
        trajectory: torch.Tensor,
        reward: float,
        task: Optional[Union[str, torch.Tensor]] = None,
        priority: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a trajectory to the buffer.
        
        Args:
            trajectory: Trajectory data to store
            reward: Reward value for the trajectory
            task: Optional task identifier or embedding
            priority: Optional explicit priority value
            metadata: Optional additional information
            
        Returns:
            Index of the added trajectory
        """
        # Determine task identifier
        task_id = None
        if task is not None:
            if isinstance(task, torch.Tensor):
                # Use tensor hash for task embedding
                task_id = str(torch.sum(task).item())
            else:
                task_id = str(task)
        
        # Compute priority if not provided
        if priority is None:
            # Use reward with offset to ensure positive priorities
            priority = max(abs(reward) + 1e-6, 1e-6) ** self.alpha
        
        # If buffer is full, remove lowest priority trajectory
        if len(self.trajectories) >= self.capacity:
            idx = np.argmin(self.priorities)
            
            # Update task indices
            old_task = self.task_ids[idx]
            if old_task and old_task in self.task_indices:
                self.task_indices[old_task].remove(idx)
                
                # If no more trajectories for this task, remove the key
                if not self.task_indices[old_task]:
                    del self.task_indices[old_task]
            
            # Remove trajectory
            self.trajectories.pop(idx)
            self.rewards.pop(idx)
            self.priorities.pop(idx)
            self.task_ids.pop(idx)
            self.metadata.pop(idx)
            
            # Update indices for trajectories after the removed one
            for t_id, indices in self.task_indices.items():
                self.task_indices[t_id] = [i if i < idx else i - 1 for i in indices]
        
        # Add new trajectory
        self.trajectories.append(trajectory)
        self.rewards.append(reward)
        self.priorities.append(priority)
        self.task_ids.append(task_id)
        self.metadata.append(metadata or {})
        
        # Update task indices
        if task_id:
            if task_id not in self.task_indices:
                self.task_indices[task_id] = []
            self.task_indices[task_id].append(len(self.trajectories) - 1)
        
        # Return index of added trajectory
        return len(self.trajectories) - 1
    
    def sample(
        self,
        batch_size: int,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[float], List[int], np.ndarray]:
        """
        Sample trajectories based on priorities.
        
        Args:
            batch_size: Number of trajectories to sample
            task: Optional task to filter by
            
        Returns:
            Tuple of (trajectories, rewards, indices, weights)
        """
        # Return empty results if buffer is empty
        if len(self.trajectories) == 0:
            return [], [], [], np.array([])
        
        # Filter by task if specified
        if task is not None:
            task_id = None
            if isinstance(task, torch.Tensor):
                task_id = str(torch.sum(task).item())
            else:
                task_id = str(task)
                
            if task_id in self.task_indices:
                valid_indices = self.task_indices[task_id]
            else:
                logger.warning(f"No trajectories found for task {task_id}")
                return [], [], [], np.array([])
        else:
            valid_indices = list(range(len(self.trajectories)))
        
        # Cap batch size
        batch_size = min(batch_size, len(valid_indices))
        if batch_size == 0:
            return [], [], [], np.array([])
        
        # Get priorities for valid indices
        valid_priorities = np.array([self.priorities[i] for i in valid_indices])
        probs = valid_priorities / valid_priorities.sum()
        
        # Sample trajectories based on priorities
        sample_indices = np.random.choice(
            len(valid_indices),
            size=batch_size,
            replace=batch_size > len(valid_indices),
            p=probs
        )
        
        # Map to original indices
        original_indices = [valid_indices[i] for i in sample_indices]
        
        # Compute importance sampling weights to correct bias
        current_beta = min(1.0, self.beta + self.updates * self.beta_annealing)
        weights = (1.0 / (len(self.trajectories) * probs[sample_indices])) ** current_beta
        
        # Normalize weights
        weights = weights / weights.max()
        
        # Increment update counter for beta annealing
        self.updates += 1
        
        # Collect sampled data
        sampled_trajectories = [self.trajectories[i] for i in original_indices]
        sampled_rewards = [self.rewards[i] for i in original_indices]
        
        return sampled_trajectories, sampled_rewards, original_indices, weights
    
    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float]
    ) -> None:
        """
        Update trajectory priorities.
        
        Args:
            indices: Indices of trajectories to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = max(priority ** self.alpha, 1e-6)
            else:
                logger.warning(f"Index {idx} out of range for priority update")
    
    def get_task_trajectories(
        self,
        task: Union[str, torch.Tensor],
        limit: Optional[int] = None,
        min_reward: Optional[float] = None
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Get trajectories for a specific task.
        
        Args:
            task: Task identifier or embedding
            limit: Maximum number of trajectories to return
            min_reward: Minimum reward threshold
            
        Returns:
            Tuple of (trajectories, rewards)
        """
        # Resolve task ID
        task_id = None
        if isinstance(task, torch.Tensor):
            task_id = str(torch.sum(task).item())
        else:
            task_id = str(task)
        
        # Get indices for the task
        if task_id not in self.task_indices:
            logger.warning(f"No trajectories found for task {task_id}")
            return [], []
        
        task_indices = self.task_indices[task_id]
        
        # Filter by reward if specified
        if min_reward is not None:
            task_indices = [idx for idx in task_indices if self.rewards[idx] >= min_reward]
        
        # Sort by reward (descending)
        task_indices = sorted(task_indices, key=lambda idx: self.rewards[idx], reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            task_indices = task_indices[:limit]
        
        # Collect trajectories and rewards
        task_trajectories = [self.trajectories[idx] for idx in task_indices]
        task_rewards = [self.rewards[idx] for idx in task_indices]
        
        return task_trajectories, task_rewards
    
    def clear(
        self,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> int:
        """
        Clear trajectories from buffer.
        
        Args:
            task: Optional task to clear trajectories for.
                 If None, clears all trajectories.
                 
        Returns:
            Number of trajectories removed
        """
        if task is None:
            # Clear all trajectories
            count = len(self.trajectories)
            self.trajectories = []
            self.rewards = []
            self.priorities = []
            self.task_ids = []
            self.metadata = []
            self.task_indices = {}
            return count
        
        # Clear trajectories for specific task
        task_id = None
        if isinstance(task, torch.Tensor):
            task_id = str(torch.sum(task).item())
        else:
            task_id = str(task)
            
        if task_id not in self.task_indices:
            return 0
            
        indices_to_remove = sorted(self.task_indices[task_id], reverse=True)
        count = len(indices_to_remove)
        
        # Remove trajectories in reverse order to avoid index shifting
        for idx in indices_to_remove:
            self.trajectories.pop(idx)
            self.rewards.pop(idx)
            self.priorities.pop(idx)
            self.task_ids.pop(idx)
            self.metadata.pop(idx)
        
        # Rebuild task indices
        self.task_indices = {}
        for i, task_id in enumerate(self.task_ids):
            if task_id:
                if task_id not in self.task_indices:
                    self.task_indices[task_id] = []
                self.task_indices[task_id].append(i)
        
        return count
    
    def save_state(
        self,
        path: str
    ) -> bool:
        """
        Save buffer state to disk.
        
        Args:
            path: Path to save state
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare state dictionary
            state_dict = {
                "trajectories": self.trajectories,
                "rewards": self.rewards,
                "priorities": self.priorities,
                "task_ids": self.task_ids,
                "metadata": self.metadata,
                "capacity": self.capacity,
                "alpha": self.alpha,
                "beta": self.beta,
                "updates": self.updates
            }
            
            # Save to disk
            torch.save(state_dict, path)
            logger.info(f"Saved trajectory buffer state to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trajectory buffer: {e}")
            return False
    
    def load_state(
        self,
        path: str
    ) -> bool:
        """
        Load buffer state from disk.
        
        Args:
            path: Path to load state from
            
        Returns:
            Success flag
        """
        try:
            if not os.path.exists(path):
                logger.error(f"State file not found: {path}")
                return False
                
            # Load state dictionary
            state_dict = torch.load(path, map_location=self.device)
            
            # Restore state
            self.trajectories = state_dict["trajectories"]
            self.rewards = state_dict["rewards"]
            self.priorities = state_dict["priorities"]
            self.task_ids = state_dict["task_ids"]
            self.metadata = state_dict.get("metadata", [{} for _ in self.trajectories])
            self.capacity = state_dict.get("capacity", self.capacity)
            self.alpha = state_dict.get("alpha", self.alpha)
            self.beta = state_dict.get("beta", self.beta)
            self.updates = state_dict.get("updates", 0)
            
            # Rebuild task indices
            self.task_indices = {}
            for i, task_id in enumerate(self.task_ids):
                if task_id:
                    if task_id not in self.task_indices:
                        self.task_indices[task_id] = []
                    self.task_indices[task_id].append(i)
                    
            logger.info(f"Loaded trajectory buffer state from {path} with {len(self.trajectories)} trajectories")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trajectory buffer: {e}")
            return False
    
    def size(
        self,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> int:
        """
        Get current buffer size.
        
        Args:
            task: Optional task to filter by
            
        Returns:
            Number of trajectories in buffer
        """
        if task is None:
            return len(self.trajectories)
            
        task_id = None
        if isinstance(task, torch.Tensor):
            task_id = str(torch.sum(task).item())
        else:
            task_id = str(task)
            
        if task_id in self.task_indices:
            return len(self.task_indices[task_id])
        return 0
        
    def filter_by_quality(
        self,
        quality_threshold: float,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> int:
        """
        Filter trajectories based on reward quality.
        
        Args:
            quality_threshold: Minimum reward threshold to keep
            task: Optional task to filter by
            
        Returns:
            Number of trajectories removed
        """
        if task is not None:
            task_id = None
            if isinstance(task, torch.Tensor):
                task_id = str(torch.sum(task).item())
            else:
                task_id = str(task)
                
            if task_id not in self.task_indices:
                logger.warning(f"No trajectories found for task {task_id}")
                return 0
                
            # Get indices of trajectories for the specified task
            indices = self.task_indices[task_id]
            
            # Find low-quality trajectories
            indices_to_remove = [idx for idx in indices if self.rewards[idx] < quality_threshold]
        else:
            # Find low-quality trajectories across all tasks
            indices_to_remove = [i for i, r in enumerate(self.rewards) if r < quality_threshold]
            
        # Remove trajectories in reverse order to avoid index shifting
        removed_count = 0
        for idx in sorted(indices_to_remove, reverse=True):
            try:
                task_id = self.task_ids[idx]
                
                # Update task indices
                if task_id and task_id in self.task_indices:
                    self.task_indices[task_id].remove(idx)
                    
                    # If no more trajectories for this task, remove the key
                    if not self.task_indices[task_id]:
                        del self.task_indices[task_id]
                
                # Remove trajectory
                self.trajectories.pop(idx)
                self.rewards.pop(idx)
                self.priorities.pop(idx)
                self.task_ids.pop(idx)
                self.metadata.pop(idx)
                
                removed_count += 1
                
                # Update indices for trajectories after the removed one
                for t_id, indices in self.task_indices.items():
                    self.task_indices[t_id] = [i if i < idx else i - 1 for i in indices]
            except IndexError:
                logger.error(f"Failed to remove trajectory at index {idx}")
                
        logger.info(f"Filtered out {removed_count} low-quality trajectories below threshold {quality_threshold}")
        return removed_count
    
    def merge(
        self,
        other_buffer: 'AdaptDiffuserTrajectoryBuffer',
        only_high_quality: bool = True,
        quality_threshold: float = 0.7
    ) -> int:
        """
        Merge trajectories from another buffer into this one.
        
        Args:
            other_buffer: Buffer to merge from
            only_high_quality: Whether to only merge high-quality trajectories
            quality_threshold: Quality threshold for merging when only_high_quality is True
            
        Returns:
            Number of trajectories merged
        """
        if not isinstance(other_buffer, AdaptDiffuserTrajectoryBuffer):
            logger.error("Can only merge with another AdaptDiffuserTrajectoryBuffer")
            return 0
            
        merged_count = 0
        
        # For each task in the other buffer
        for task_id in other_buffer.task_indices:
            # Get indices for this task
            indices = other_buffer.task_indices[task_id]
            
            # Filter by quality if needed
            if only_high_quality:
                indices = [idx for idx in indices if other_buffer.rewards[idx] >= quality_threshold]
                
            # Add trajectories to this buffer
            for idx in indices:
                trajectory = other_buffer.trajectories[idx]
                reward = other_buffer.rewards[idx]
                metadata = other_buffer.metadata[idx]
                
                # Add to this buffer
                self.add(trajectory, reward, task_id, None, metadata)
                merged_count += 1
                
        logger.info(f"Merged {merged_count} trajectories from another buffer")
        return merged_count
    
    def get_statistics(
        self,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compute statistics about stored trajectories.
        
        Args:
            task: Optional task to filter by
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "count": 0,
            "mean_reward": 0.0,
            "median_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "std_reward": 0.0,
            "task_distribution": {}
        }
        
        if len(self.rewards) == 0:
            return stats
            
        # Filter by task if specified
        if task is not None:
            task_id = None
            if isinstance(task, torch.Tensor):
                task_id = str(torch.sum(task).item())
            else:
                task_id = str(task)
                
            if task_id in self.task_indices:
                indices = self.task_indices[task_id]
                rewards = [self.rewards[idx] for idx in indices]
            else:
                logger.warning(f"No trajectories found for task {task_id}")
                return stats
        else:
            rewards = self.rewards
            indices = range(len(rewards))
            
        # Compute statistics
        rewards_array = np.array(rewards)
        stats["count"] = len(rewards)
        stats["mean_reward"] = float(np.mean(rewards_array))
        stats["median_reward"] = float(np.median(rewards_array))
        stats["min_reward"] = float(np.min(rewards_array))
        stats["max_reward"] = float(np.max(rewards_array))
        stats["std_reward"] = float(np.std(rewards_array))
        
        # Compute task distribution
        if task is None:
            task_counts = {}
            for task_id in self.task_ids:
                if task_id:
                    task_counts[task_id] = task_counts.get(task_id, 0) + 1
                    
            stats["task_distribution"] = task_counts
            
        return stats
    def optimize_buffer_for_adaptation(
        self,
        task: Union[str, torch.Tensor],
        min_quality: float = 0.5,
        max_size: Optional[int] = None,
        diversity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize buffer contents for model adaptation.
        
        This method filters and organizes the buffer to ensure high-quality,
        diverse trajectories are available for model adaptation.
        
        Args:
            task: Task identifier or embedding
            min_quality: Minimum reward threshold to keep
            max_size: Maximum number of trajectories for the task
            diversity_threshold: Optional similarity threshold for removing redundant samples
            
        Returns:
            Stats about optimization process
        """
        stats = {
            "initial_count": 0,
            "filtered_by_quality": 0,
            "filtered_by_diversity": 0,
            "final_count": 0
        }
        
        # Get task ID
        task_id = None
        if isinstance(task, torch.Tensor):
            task_id = str(torch.sum(task).item())
        else:
            task_id = str(task)
            
        # Check if task exists in buffer
        if task_id not in self.task_indices:
            logger.warning(f"No trajectories found for task {task_id}")
            return stats
            
        # Get initial count
        stats["initial_count"] = len(self.task_indices[task_id])
        
        # 1. Filter by quality
        filtered_count = self.filter_by_quality(min_quality, task)
        stats["filtered_by_quality"] = filtered_count
        
        # 2. If diversity threshold provided, remove redundant samples
        if diversity_threshold is not None and diversity_threshold > 0:
            # Get all trajectories for the task
            task_trajectories, task_rewards = self.get_task_trajectories(task)
            
            if len(task_trajectories) > 1:
                # Convert to numpy arrays for easier processing
                flat_trajectories = [t.flatten().cpu().numpy() for t in task_trajectories]
                
                # Compute pairwise cosine similarities
                from sklearn.metrics.pairwise import cosine_similarity
                try:
                    similarities = cosine_similarity(flat_trajectories)
                    
                    # Set self-similarities to 0
                    np.fill_diagonal(similarities, 0)
                    
                    # Find redundant pairs
                    to_remove = set()
                    for i in range(len(similarities)):
                        if i in to_remove:
                            continue
                            
                        for j in range(i+1, len(similarities)):
                            if j in to_remove:
                                continue
                                
                            # If similarity exceeds threshold, remove the lower reward one
                            if similarities[i, j] > diversity_threshold:
                                if task_rewards[i] >= task_rewards[j]:
                                    to_remove.add(j)
                                else:
                                    to_remove.add(i)
                                    break  # Break as i is now removed
                    
                    # Get indices in original buffer
                    indices_to_remove = [self.task_indices[task_id][i] for i in to_remove]
                    
                    # Remove redundant samples
                    for idx in sorted(indices_to_remove, reverse=True):
                        try:
                            # Update task indices
                            self.task_indices[task_id].remove(idx)
                            
                            # Remove trajectory
                            self.trajectories.pop(idx)
                            self.rewards.pop(idx)
                            self.priorities.pop(idx)
                            self.task_ids.pop(idx)
                            self.metadata.pop(idx)
                            
                            stats["filtered_by_diversity"] += 1
                            
                            # Update indices for trajectories after the removed one
                            for t_id, indices in self.task_indices.items():
                                self.task_indices[t_id] = [i if i < idx else i - 1 for i in indices]
                        except Exception as e:
                            logger.error(f"Error removing sample: {e}")
                            
                except ImportError:
                    logger.warning("scikit-learn not available, skipping diversity filtering")
                except Exception as e:
                    logger.error(f"Error in diversity filtering: {e}")
        
        # 3. If max_size provided, keep only the top max_size trajectories
        if max_size is not None and task_id in self.task_indices:
            if len(self.task_indices[task_id]) > max_size:
                # Get all trajectories for the task
                task_trajectories, task_rewards = self.get_task_trajectories(task)
                
                # Sort by reward (descending)
                sorted_indices = sorted(
                    range(len(task_rewards)),
                    key=lambda i: task_rewards[i],
                    reverse=True
                )
                
                # Keep only the top max_size indices
                to_keep = sorted_indices[:max_size]
                to_remove = sorted_indices[max_size:]
                
                # Map to actual buffer indices
                indices_to_remove = [self.task_indices[task_id][i] for i in to_remove]
                
                # Remove excess samples
                for idx in sorted(indices_to_remove, reverse=True):
                    try:
                        # Update task indices
                        self.task_indices[task_id].remove(idx)
                        
                        # Remove trajectory
                        self.trajectories.pop(idx)
                        self.rewards.pop(idx)
                        self.priorities.pop(idx)
                        self.task_ids.pop(idx)
                        self.metadata.pop(idx)
                        
                        # Update indices for trajectories after the removed one
                        for t_id, indices in self.task_indices.items():
                            self.task_indices[t_id] = [i if i < idx else i - 1 for i in indices]
                    except Exception as e:
                        logger.error(f"Error removing sample: {e}")
        
        # Get final count
        stats["final_count"] = len(self.task_indices.get(task_id, []))
        
        logger.info(f"Optimized buffer for task {task_id}: {stats}")
        return stats
    
    def batch_add(
        self,
        trajectories: List[torch.Tensor],
        rewards: List[float],
        task: Optional[Union[str, torch.Tensor]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add multiple trajectories to the buffer.
        
        Args:
            trajectories: List of trajectory tensors
            rewards: List of reward values
            task: Optional task identifier or embedding
            metadata: Optional list of metadata dictionaries
            
        Returns:
            List of indices of added trajectories
        """
        if len(trajectories) != len(rewards):
            raise ValueError("trajectories and rewards must have the same length")
            
        if metadata is not None and len(metadata) != len(trajectories):
            raise ValueError("metadata must have the same length as trajectories")
            
        added_indices = []
        
        # Process task ID once
        task_id = None
        if task is not None:
            if isinstance(task, torch.Tensor):
                task_id = str(torch.sum(task).item())
            else:
                task_id = str(task)
        
        # Add trajectories
        for i, (trajectory, reward) in enumerate(zip(trajectories, rewards)):
            meta = metadata[i] if metadata is not None else None
            idx = self.add(trajectory, reward, task_id, None, meta)
            added_indices.append(idx)
            
        return added_indices
        
    def temperature_sample(
        self,
        batch_size: int,
        temperature: float = 1.0,
        task: Optional[Union[str, torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[float], List[int]]:
        """
        Sample trajectories using softmax temperature.
        
        Args:
            batch_size: Number of trajectories to sample
            temperature: Softmax temperature (higher = more uniform)
            task: Optional task to filter by
            
        Returns:
            Tuple of (trajectories, rewards, indices)
        """
        # Return empty results if buffer is empty
        if len(self.trajectories) == 0:
            return [], [], []
        
        # Filter by task if specified
        if task is not None:
            task_id = None
            if isinstance(task, torch.Tensor):
                task_id = str(torch.sum(task).item())
            else:
                task_id = str(task)
                
            if task_id in self.task_indices:
                valid_indices = self.task_indices[task_id]
            else:
                logger.warning(f"No trajectories found for task {task_id}")
                return [], [], []
        else:
            valid_indices = list(range(len(self.trajectories)))
        
        # Cap batch size
        batch_size = min(batch_size, len(valid_indices))
        if batch_size == 0:
            return [], [], []
        
        # Get rewards for valid indices
        valid_rewards = np.array([self.rewards[i] for i in valid_indices])
        
        # Apply temperature scaling
        if temperature <= 0:
            # With zero temperature, just take the highest rewards
            sorted_idxs = np.argsort(-valid_rewards)
            sample_indices = sorted_idxs[:batch_size]
        else:
            # Apply softmax with temperature
            scaled_rewards = valid_rewards / temperature
            # Shift by max for numerical stability
            scaled_rewards = scaled_rewards - np.max(scaled_rewards)
            exp_rewards = np.exp(scaled_rewards)
            probs = exp_rewards / np.sum(exp_rewards)
            
            # Sample indices
            sample_indices = np.random.choice(
                len(valid_indices),
                size=batch_size,
                replace=batch_size > len(valid_indices),
                p=probs
            )
        
        # Map to original indices
        original_indices = [valid_indices[i] for i in sample_indices]
        
        # Collect sampled data
        sampled_trajectories = [self.trajectories[i] for i in original_indices]
        sampled_rewards = [self.rewards[i] for i in original_indices]
        
        return sampled_trajectories, sampled_rewards, original_indices