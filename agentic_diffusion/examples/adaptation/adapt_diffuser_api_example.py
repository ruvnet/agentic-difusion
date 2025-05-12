"""
Example of using the AdaptDiffuserAPI.

This example demonstrates how to use the AdaptDiffuserAPI for adaptive diffusion
with various tasks and self-improvement.
"""

import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI, create_adapt_diffuser_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def visualize_trajectory(trajectory: torch.Tensor, title: str = "Generated Trajectory"):
    """
    Visualize a generated trajectory.
    
    Args:
        trajectory: Trajectory tensor
        title: Plot title
    """
    # Ensure trajectory is on CPU and convert to numpy
    trajectory_np = trajectory.cpu().numpy()
    
    # Handle different trajectory shapes
    if len(trajectory_np.shape) == 4:  # [B, C, H, W]
        trajectory_np = trajectory_np[0]  # Take first batch element
    
    if trajectory_np.shape[0] == 3:  # [C, H, W] with RGB channels
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(trajectory_np, (1, 2, 0)))  # Change to [H, W, C] for imshow
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        # For other types of trajectories, visualize as heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(trajectory_np[0], cmap='viridis')  # Use first channel
        plt.colorbar(label='Value')
        plt.title(title)
        plt.tight_layout()
        plt.show()


def run_simple_example():
    """Run a simple example of using AdaptDiffuserAPI."""
    logger.info("Starting simple AdaptDiffuser example")
    
    # Create API with default configuration
    config = {
        "guidance_scale": 3.0,
        "batch_size": 4,
        "sampling_steps": 50,
        "img_size": 32,
        "channels": 3,
        "device": "auto"  # Will use CUDA if available
    }
    
    api = create_adapt_diffuser_api(config)
    
    # Define a simple task
    task = "navigate_maze"
    
    logger.info(f"Generating initial trajectories for task: {task}")
    
    # Generate initial trajectories
    trajectories, metadata = api.generate(
        task=task,
        batch_size=4,
        guidance_scale=2.0  # Override default guidance
    )
    
    logger.info(f"Generated {len(trajectories)} trajectories with metadata: {metadata}")
    
    # Visualize a trajectory
    if trajectories is not None and len(trajectories) > 0:
        visualize_trajectory(trajectories[0], f"Initial Trajectory for {task}")
    
    # Adapt the model to the task
    logger.info(f"Adapting model to task: {task}")
    adaptation_metrics = api.adapt(
        task=task,
        iterations=2,
        batch_size=8
    )
    
    logger.info(f"Adaptation metrics: {adaptation_metrics}")
    
    # Generate improved trajectories after adaptation
    logger.info("Generating trajectories after adaptation")
    improved_trajectories, improved_metadata = api.generate(
        task=task,
        batch_size=4
    )
    
    if improved_trajectories is not None and len(improved_trajectories) > 0:
        visualize_trajectory(improved_trajectories[0], f"Improved Trajectory for {task}")
    
    # Compare metrics
    if "rewards" in metadata and "rewards" in improved_metadata:
        initial_reward = metadata["rewards"]["mean"]
        improved_reward = improved_metadata["rewards"]["mean"]
        improvement = (improved_reward - initial_reward) / abs(initial_reward) * 100
        
        logger.info(f"Reward improvement: {improvement:.2f}%")
        logger.info(f"Initial reward: {initial_reward:.4f}")
        logger.info(f"Improved reward: {improved_reward:.4f}")


def run_self_improvement_example():
    """Run an example demonstrating self-improvement."""
    logger.info("Starting self-improvement example")
    
    # Create API with configuration for self-improvement
    config = {
        "guidance_scale": 5.0,
        "batch_size": 8,
        "sampling_steps": 40,
        "img_size": 32,
        "channels": 3,
        "quality_threshold": 0.6,
        "device": "auto"
    }
    
    api = create_adapt_diffuser_api(config)
    
    # Define task
    task = "avoid_obstacles"
    
    # Run self-improvement
    logger.info(f"Running self-improvement for task: {task}")
    
    self_improve_metrics = api.self_improve(
        task=task,
        iterations=3,
        trajectories_per_iter=20,
        quality_threshold=0.65
    )
    
    logger.info(f"Self-improvement metrics: {self_improve_metrics}")
    
    # Generate final trajectories
    logger.info("Generating trajectories after self-improvement")
    final_trajectories, final_metadata = api.generate(
        task=task,
        batch_size=4,
        guidance_scale=7.0  # Higher guidance for final generation
    )
    
    if final_trajectories is not None and len(final_trajectories) > 0:
        visualize_trajectory(final_trajectories[0], f"Self-Improved Trajectory for {task}")
    
    # Report improvement results
    initial_reward = self_improve_metrics.get("initial_mean_reward", 0)
    final_reward = self_improve_metrics.get("final_mean_reward", 0)
    
    logger.info(f"Initial reward: {initial_reward:.4f}")
    logger.info(f"Final reward after self-improvement: {final_reward:.4f}")
    logger.info(f"Reward improvement: {self_improve_metrics.get('reward_improvement', 0):.4f}")


def run_multi_task_example():
    """Run an example demonstrating multi-task adaptation."""
    logger.info("Starting multi-task adaptation example")
    
    # Create API with multi-task configuration
    config = {
        "multi_task": True,
        "guidance_scale": 4.0,
        "batch_size": 8,
        "sampling_steps": 40,
        "img_size": 32,
        "channels": 3,
        "device": "auto"
    }
    
    api = create_adapt_diffuser_api(config)
    
    # Define multiple tasks
    tasks = ["navigate_maze", "avoid_obstacles", "reach_goal"]
    
    # Initial performance on all tasks
    initial_performance = {}
    
    for task in tasks:
        logger.info(f"Generating initial trajectories for task: {task}")
        trajectories, metadata = api.generate(task=task, batch_size=4)
        
        if "rewards" in metadata:
            initial_performance[task] = metadata["rewards"]["mean"]
            logger.info(f"Initial performance on {task}: {initial_performance[task]:.4f}")
    
    # Adapt to each task sequentially
    for task in tasks:
        logger.info(f"Adapting to task: {task}")
        adaptation_metrics = api.adapt(
            task=task,
            iterations=2,
            batch_size=8
        )
        
        logger.info(f"Adaptation metrics for {task}: {adaptation_metrics}")
    
    # Evaluate final performance on all tasks
    final_performance = {}
    
    for task in tasks:
        logger.info(f"Generating final trajectories for task: {task}")
        trajectories, metadata = api.generate(task=task, batch_size=4)
        
        if "rewards" in metadata:
            final_performance[task] = metadata["rewards"]["mean"]
            
            initial = initial_performance.get(task, 0)
            final = final_performance[task]
            improvement = (final - initial) / abs(initial) * 100 if initial != 0 else 0
            
            logger.info(f"Task: {task}")
            logger.info(f"  Initial performance: {initial:.4f}")
            logger.info(f"  Final performance: {final:.4f}")
            logger.info(f"  Improvement: {improvement:.2f}%")


def save_and_load_example():
    """Demonstrate saving and loading model state."""
    logger.info("Starting save/load state example")
    
    # Create API
    config = {
        "guidance_scale": 3.0,
        "batch_size": 4,
        "sampling_steps": 30,
    }
    
    api = create_adapt_diffuser_api(config)
    
    # Define task
    task = "navigate_maze"
    
    # Adapt model
    logger.info(f"Adapting model to task: {task}")
    api.adapt(
        task=task,
        iterations=1,
        batch_size=4
    )
    
    # Generate trajectories with adapted model
    logger.info("Generating trajectories with adapted model")
    before_trajectories, before_metadata = api.generate(
        task=task,
        batch_size=2
    )
    
    # Save state
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "adaptdiffuser_state.json")
    
    logger.info(f"Saving model state to {save_path}")
    success = api.save_state(save_path)
    
    if success:
        logger.info("Successfully saved model state")
        
        # Create new API instance
        new_api = create_adapt_diffuser_api({})
        
        # Load state
        logger.info(f"Loading model state from {save_path}")
        success = new_api.load_state(save_path)
        
        if success:
            logger.info("Successfully loaded model state")
            
            # Generate trajectories with loaded model
            logger.info("Generating trajectories with loaded model")
            after_trajectories, after_metadata = new_api.generate(
                task=task,
                batch_size=2
            )
            
            # Compare results
            if ("rewards" in before_metadata and 
                "rewards" in after_metadata):
                
                before_reward = before_metadata["rewards"]["mean"]
                after_reward = after_metadata["rewards"]["mean"]
                
                logger.info(f"Original model reward: {before_reward:.4f}")
                logger.info(f"Loaded model reward: {after_reward:.4f}")
                logger.info(f"Difference: {after_reward - before_reward:.4f}")
        else:
            logger.error("Failed to load model state")
    else:
        logger.error("Failed to save model state")


if __name__ == "__main__":
    print("\n=== Simple AdaptDiffuser Example ===\n")
    run_simple_example()
    
    print("\n=== Self-Improvement Example ===\n")
    run_self_improvement_example()
    
    print("\n=== Multi-Task Adaptation Example ===\n")
    run_multi_task_example()
    
    print("\n=== Save and Load Example ===\n")
    save_and_load_example()