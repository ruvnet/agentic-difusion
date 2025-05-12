"""
Utility functions for AdaptDiffuser models.

This module provides helper functions for working with AdaptDiffuser
models, including metrics computation, serialization, and data processing.
"""

import torch
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def compute_reward_statistics(
    rewards: Union[List[float], torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    Compute statistics for reward values.
    
    Args:
        rewards: Collection of reward values
        
    Returns:
        Dictionary with statistics (mean, median, min, max, std)
    """
    # Convert to numpy array for consistent processing
    if isinstance(rewards, torch.Tensor):
        rewards_np = rewards.detach().cpu().numpy()
    elif isinstance(rewards, list):
        rewards_np = np.array(rewards)
    else:
        rewards_np = rewards
        
    # Compute statistics
    stats = {
        "mean": float(np.mean(rewards_np)),
        "median": float(np.median(rewards_np)),
        "min": float(np.min(rewards_np)),
        "max": float(np.max(rewards_np)),
        "std": float(np.std(rewards_np)),
        "count": len(rewards_np)
    }
    
    return stats


def save_adaptation_metrics(
    metrics: Dict[str, Any],
    path: str,
    include_plots: bool = True,
    plot_dir: Optional[str] = None
) -> bool:
    """
    Save adaptation metrics to disk.
    
    Args:
        metrics: Dictionary of adaptation metrics
        path: Path to save the metrics JSON
        include_plots: Whether to generate and save plots
        plot_dir: Directory to save plots (defaults to same directory as metrics)
        
    Returns:
        Success flag
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert non-serializable objects to lists or strings
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_metrics[key] = value
            else:
                # Try to convert to string for non-standard types
                serializable_metrics[key] = str(value)
        
        # Save metrics to JSON
        with open(path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        logger.info(f"Saved adaptation metrics to {path}")
        
        # Generate and save plots if requested
        if include_plots and 'steps' in metrics:
            plot_path = plot_dir or os.path.dirname(path)
            os.makedirs(plot_path, exist_ok=True)
            
            base_filename = os.path.splitext(os.path.basename(path))[0]
            
            # Plot reward curve
            if 'mean_reward' in metrics and len(metrics['steps']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['steps'], metrics['mean_reward'], label='Mean Reward')
                if 'max_reward' in metrics:
                    plt.plot(metrics['steps'], metrics['max_reward'], label='Max Reward')
                plt.xlabel('Steps')
                plt.ylabel('Reward')
                plt.title('Adaptation Reward Curve')
                plt.legend()
                plt.grid(True)
                reward_plot_path = os.path.join(plot_path, f"{base_filename}_rewards.png")
                plt.savefig(reward_plot_path)
                plt.close()
                
                logger.info(f"Saved reward plot to {reward_plot_path}")
            
            # Plot loss curve
            if 'loss' in metrics and len(metrics['steps']) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['steps'], metrics['loss'])
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Adaptation Loss Curve')
                plt.grid(True)
                loss_plot_path = os.path.join(plot_path, f"{base_filename}_loss.png")
                plt.savefig(loss_plot_path)
                plt.close()
                
                logger.info(f"Saved loss plot to {loss_plot_path}")
                
        return True
        
    except Exception as e:
        logger.error(f"Failed to save adaptation metrics: {e}")
        return False


def load_adaptation_metrics(
    path: str
) -> Dict[str, Any]:
    """
    Load adaptation metrics from disk.
    
    Args:
        path: Path to load the metrics JSON from
        
    Returns:
        Dictionary of adaptation metrics
        
    Raises:
        FileNotFoundError: If metrics file not found
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
        
    try:
        with open(path, 'r') as f:
            metrics = json.load(f)
            
        logger.info(f"Loaded adaptation metrics from {path}")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to load adaptation metrics: {e}")
        raise RuntimeError(f"Error loading metrics: {e}")


def encode_task(
    task: Union[str, torch.Tensor],
    task_embedding_model: Optional[Any] = None,
    device: str = None
) -> torch.Tensor:
    """
    Encode task description or identifier into an embedding.
    
    Args:
        task: Task description string or embedding tensor
        task_embedding_model: Model for encoding task descriptions
        device: Device to place tensor on
        
    Returns:
        Task embedding tensor
        
    Raises:
        ValueError: If task_embedding_model is not available and task is a string
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(task, torch.Tensor):
        # Already an embedding
        return task.to(device)
    elif isinstance(task, str):
        # Text description needs encoding
        if task_embedding_model is None:
            raise ValueError("Task embedding model required to encode task descriptions")
        
        with torch.no_grad():
            embedding = task_embedding_model.encode(task)
        return embedding.to(device)
    else:
        raise ValueError(f"Unsupported task type: {type(task)}")


def calculate_adaptive_guidance_schedule(
    num_steps: int,
    max_scale: float,
    schedule_type: str = "triangular",
    min_step_percent: float = 0.1,
    max_step_percent: float = 0.9
) -> List[float]:
    """
    Calculate an adaptive guidance scale schedule.
    
    Args:
        num_steps: Total number of denoising steps
        max_scale: Maximum guidance scale
        schedule_type: Type of schedule ("triangular", "linear", "cosine")
        min_step_percent: Percentage of steps to start guidance
        max_step_percent: Percentage of steps to end guidance
        
    Returns:
        List of guidance scales for each step
    """
    # Convert percentages to step indices
    start_idx = int(min_step_percent * num_steps)
    end_idx = int(max_step_percent * num_steps)
    active_steps = end_idx - start_idx
    
    # Create empty schedule
    schedule = [0.0] * num_steps
    
    # No guidance outside the active window
    if active_steps <= 0:
        return schedule
    
    # Fill active window based on schedule type
    if schedule_type == "triangular":
        # Triangular schedule with peak in the middle
        mid_point = start_idx + active_steps // 2
        for i in range(start_idx, end_idx):
            normalized_pos = abs(i - mid_point) / (active_steps / 2)
            schedule[i] = max_scale * (1 - normalized_pos)
            
    elif schedule_type == "linear":
        # Linear ramp-up and ramp-down
        mid_point = start_idx + active_steps // 2
        for i in range(start_idx, mid_point):
            # Ramp up
            normalized_pos = (i - start_idx) / (mid_point - start_idx)
            schedule[i] = max_scale * normalized_pos
            
        for i in range(mid_point, end_idx):
            # Ramp down
            normalized_pos = (i - mid_point) / (end_idx - mid_point)
            schedule[i] = max_scale * (1 - normalized_pos)
            
    elif schedule_type == "cosine":
        # Cosine schedule
        for i in range(start_idx, end_idx):
            # Map to [0, π]
            theta = np.pi * (i - start_idx) / active_steps
            # Shifted cosine gives 0->1->0
            schedule[i] = max_scale * 0.5 * (1 + np.cos(theta - np.pi))
            
    else:
        # Default: constant schedule within window
        for i in range(start_idx, end_idx):
            schedule[i] = max_scale
            
    return schedule