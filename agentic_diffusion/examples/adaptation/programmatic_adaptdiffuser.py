"""
Programmatic usage examples for AdaptDiffuser API.

This script demonstrates how to use the AdaptDiffuser API programmatically
in your own Python applications, without using the command-line interface.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from agentic_diffusion.api.adapt_diffuser_api import (
    AdaptDiffuserAPI,
    create_adapt_diffuser_api,
    create_adapt_diffuser_adapter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def visualize_trajectory(trajectory: torch.Tensor, title: str = None):
    """Visualize a trajectory tensor."""
    # Ensure trajectory is on CPU and convert to numpy
    if isinstance(trajectory, torch.Tensor):
        trajectory_np = trajectory.cpu().detach().numpy()
    else:
        trajectory_np = trajectory
    
    # Handle different trajectory shapes
    if len(trajectory_np.shape) == 4:  # [B, C, H, W]
        trajectory_np = trajectory_np[0]  # Take first batch element
    
    if trajectory_np.shape[0] == 3:  # [C, H, W] with RGB channels
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(trajectory_np, (1, 2, 0)))  # Change to [H, W, C] for imshow
        plt.title(title or "Trajectory Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        # For other types of trajectories, visualize as heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(trajectory_np[0], cmap='viridis')  # Use first channel
        plt.colorbar(label='Value')
        plt.title(title or "Trajectory Visualization")
        plt.tight_layout()
        plt.show()


def example_1_basic_generation():
    """Demonstrate basic trajectory generation."""
    print("\n=== Example 1: Basic Trajectory Generation ===")
    
    # Create API with default configuration
    api = create_adapt_diffuser_api()
    
    # Define a task
    task = "navigate_maze"
    
    # Generate trajectories
    print(f"Generating trajectories for task: {task}")
    trajectories, metadata = api.generate(
        task=task,
        batch_size=2,
        guidance_scale=2.0
    )
    
    # Print metadata
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Metadata: {metadata}")
    
    # Visualize a trajectory
    if trajectories is not None and len(trajectories) > 0:
        visualize_trajectory(trajectories[0], f"Generated Trajectory for {task}")
    
    return api, trajectories, metadata


def example_2_adaptation():
    """Demonstrate model adaptation to a specific task."""
    print("\n=== Example 2: Model Adaptation ===")
    
    # Create API with custom configuration
    config = {
        "guidance_scale": 3.0,
        "batch_size": 8,
        "sampling_steps": 40,
        "device": "auto"
    }
    
    api = create_adapt_diffuser_api(config)
    
    # Define task
    task = "avoid_obstacles"
    
    # Generate initial trajectories to establish baseline
    print(f"Generating initial trajectories for task: {task}")
    initial_trajectories, initial_metadata = api.generate(
        task=task,
        batch_size=4
    )
    
    initial_reward = 0
    if "rewards" in initial_metadata:
        initial_reward = initial_metadata["rewards"]["mean"]
        print(f"Initial mean reward: {initial_reward:.4f}")
    
    # Adapt model to task
    print(f"Adapting model to task: {task}")
    adaptation_metrics = api.adapt(
        task=task,
        iterations=3,
        batch_size=8,
        learning_rate=1e-5
    )
    
    print(f"Adaptation metrics: {adaptation_metrics}")
    
    # Generate trajectories after adaptation
    print("Generating trajectories after adaptation")
    adapted_trajectories, adapted_metadata = api.generate(
        task=task,
        batch_size=4
    )
    
    # Compare rewards
    if "rewards" in adapted_metadata:
        adapted_reward = adapted_metadata["rewards"]["mean"]
        improvement = ((adapted_reward - initial_reward) / abs(initial_reward)) * 100 if initial_reward != 0 else 0
        
        print(f"Initial mean reward: {initial_reward:.4f}")
        print(f"Adapted mean reward: {adapted_reward:.4f}")
        print(f"Improvement: {improvement:.2f}%")
    
    # Visualize before/after
    if initial_trajectories is not None and adapted_trajectories is not None:
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        if isinstance(initial_trajectories[0], torch.Tensor):
            traj = initial_trajectories[0].cpu().detach().numpy()
        else:
            traj = initial_trajectories[0]
        if traj.shape[0] == 3:  # RGB image
            plt.imshow(np.transpose(traj, (1, 2, 0)))
        else:
            plt.imshow(traj[0], cmap='viridis')
        plt.title("Before Adaptation")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        if isinstance(adapted_trajectories[0], torch.Tensor):
            traj = adapted_trajectories[0].cpu().detach().numpy()
        else:
            traj = adapted_trajectories[0]
        if traj.shape[0] == 3:  # RGB image
            plt.imshow(np.transpose(traj, (1, 2, 0)))
        else:
            plt.imshow(traj[0], cmap='viridis')
        plt.title("After Adaptation")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return api, adaptation_metrics


def example_3_self_improvement():
    """Demonstrate self-improvement capabilities."""
    print("\n=== Example 3: Self-Improvement ===")
    
    # Create API
    config = {
        "guidance_scale": 5.0,
        "quality_threshold": 0.6,
        "device": "auto"
    }
    
    api = create_adapt_diffuser_api(config)
    
    # Define task
    task = "reach_goal"
    
    # Run self-improvement
    print(f"Running self-improvement for task: {task}")
    improvement_metrics = api.self_improve(
        task=task,
        iterations=2,
        trajectories_per_iter=15,
        quality_threshold=0.65
    )
    
    print(f"Self-improvement metrics: {improvement_metrics}")
    
    # Generate final trajectories
    print("Generating trajectories after self-improvement")
    final_trajectories, final_metadata = api.generate(
        task=task,
        batch_size=2,
        guidance_scale=7.0  # Higher guidance for final generation
    )
    
    # Report results
    if "initial_mean_reward" in improvement_metrics and "final_mean_reward" in improvement_metrics:
        initial = improvement_metrics["initial_mean_reward"]
        final = improvement_metrics["final_mean_reward"]
        improvement = ((final - initial) / abs(initial)) * 100 if initial != 0 else 0
        
        print(f"Initial reward: {initial:.4f}")
        print(f"Final reward: {final:.4f}")
        print(f"Improvement: {improvement:.2f}%")
    
    # Visualize final trajectory
    if final_trajectories is not None and len(final_trajectories) > 0:
        visualize_trajectory(final_trajectories[0], f"Self-Improved Trajectory for {task}")
    
    return api, improvement_metrics


def example_4_save_and_load():
    """Demonstrate saving and loading model state."""
    print("\n=== Example 4: Save and Load Model State ===")
    
    # Create API
    api = create_adapt_diffuser_api()
    
    # Define task and adapt
    task = "navigate_maze"
    
    print(f"Adapting model to task: {task}")
    api.adapt(task=task, iterations=1)
    
    # Create save directory
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "adaptdiffuser_example.json")
    
    # Save state
    print(f"Saving model state to {save_path}")
    success = api.save_state(save_path)
    
    if success:
        print("Successfully saved model state")
        
        # Create new API instance
        new_api = create_adapt_diffuser_api()
        
        # Load state
        print(f"Loading model state from {save_path}")
        load_success = new_api.load_state(save_path)
        
        if load_success:
            print("Successfully loaded model state")
            
            # Generate from both APIs to compare
            original_trajectories, _ = api.generate(task=task, batch_size=1)
            loaded_trajectories, _ = new_api.generate(task=task, batch_size=1)
            
            # Visualize comparison
            if original_trajectories is not None and loaded_trajectories is not None:
                plt.figure(figsize=(15, 7))
                
                plt.subplot(1, 2, 1)
                if isinstance(original_trajectories[0], torch.Tensor):
                    traj = original_trajectories[0].cpu().detach().numpy()
                else:
                    traj = original_trajectories[0]
                if traj.shape[0] == 3:  # RGB image
                    plt.imshow(np.transpose(traj, (1, 2, 0)))
                else:
                    plt.imshow(traj[0], cmap='viridis')
                plt.title("Original Model")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                if isinstance(loaded_trajectories[0], torch.Tensor):
                    traj = loaded_trajectories[0].cpu().detach().numpy()
                else:
                    traj = loaded_trajectories[0]
                if traj.shape[0] == 3:  # RGB image
                    plt.imshow(np.transpose(traj, (1, 2, 0)))
                else:
                    plt.imshow(traj[0], cmap='viridis')
                plt.title("Loaded Model")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
        else:
            print("Failed to load model state")
    else:
        print("Failed to save model state")


def example_5_adapter_integration():
    """Demonstrate integration with the adaptation framework."""
    print("\n=== Example 5: Integration with Adaptation Framework ===")
    
    # Create adapter
    adapter = create_adapt_diffuser_adapter()
    
    # Create sample code to adapt
    code = """
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b
    """
    
    # Provide feedback
    feedback = {
        "task": "optimize_calculation_functions",
        "quality": 0.6,
        "suggestions": ["Add error handling", "Optimize for large numbers"]
    }
    
    # Adapt code
    print("Adapting code:")
    print(code)
    print("\nFeedback:", feedback)
    
    adapted_code = adapter.adapt(
        code=code,
        feedback=feedback,
        language="python",
        self_improve=True
    )
    
    print("\nAdapted code:")
    print(adapted_code)


if __name__ == "__main__":
    try:
        # Run examples
        example_1_basic_generation()
        example_2_adaptation()
        example_3_self_improvement()
        example_4_save_and_load()
        example_5_adapter_integration()
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()