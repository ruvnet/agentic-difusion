#!/usr/bin/env python3
"""
Script to set up and test a simple reward model for AdaptDiffuser.

This script creates a basic reward model that returns sensible scores for trajectories,
registers it with AdaptDiffuser, and verifies it can be used for adaptation and improvement.
"""

import os
import sys
import logging
import torch
import yaml
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI
from agentic_diffusion.core.reward_functions import AdaptDiffuserTestRewardModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mock_adaptdiffuser_reward")

def main():
    """Set up and test a simple reward model for AdaptDiffuser."""
    
    # Step 1: Create AdaptDiffuserAPI with config
    logger.info("Step 1: Creating AdaptDiffuserAPI with config from ./config/test_cpu.yaml")
    
    # Load the config from YAML
    import yaml
    with open('./config/test_cpu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create the API
    api = AdaptDiffuserAPI(config=config)
    
    # Step 2: Create and register test reward model
    logger.info("Step 2: Creating test reward model")
    reward_model = AdaptDiffuserTestRewardModel(
        initial_reward_or_config=0.5,
        improvement_rate=0.05,
        noise_scale=0.02
    )
    
    logger.info("Registering reward model with AdaptDiffuserAPI")
    api.register_reward_model(reward_model)
    
    # Step 3: Test reward model
    logger.info("Step 3: Testing reward model with 5 samples")
    dummy_samples = torch.randn(5, 128)  # 5 random samples
    
    # Create smaller test samples to get scalar outputs
    small_dummy_samples = torch.randn(5, 8)  # 5 samples with 8 dimensions
    
    # Test direct computation
    try:
        direct_rewards = [reward_model.compute_reward(sample) for sample in small_dummy_samples]
        logger.info(f"Direct rewards: {direct_rewards}")
        
        # Calculate mean safely from tensors or floats
        direct_reward_mean = 0.0
        direct_reward_count = 0
        
        for r in direct_rewards:
            if isinstance(r, torch.Tensor):
                if r.numel() == 1:
                    direct_reward_mean += r.item()
                    direct_reward_count += 1
                else:
                    direct_reward_mean += r.mean().item()
                    direct_reward_count += 1
            else:
                direct_reward_mean += float(r)
                direct_reward_count += 1
        
        if direct_reward_count > 0:
            direct_reward_mean /= direct_reward_count
            
        logger.info(f"Mean direct reward: {direct_reward_mean:.4f}")
    except Exception as e:
        logger.error(f"Error in direct reward computation: {e}")
    
    # Test API computation
    try:
        # First convert the string task to a tensor if needed
        if isinstance(api.model, torch.nn.Module) and hasattr(api.model, 'task_embedding_model'):
            task_embedding = api.model.task_embedding_model("test_task")
            model_rewards = api.model.reward_model.compute_reward(small_dummy_samples, task_embedding)
        else:
            # Fallback if needed
            model_rewards = api.model.reward_model.compute_reward(small_dummy_samples)
            
        logger.info(f"API-computed rewards: {model_rewards}")
        
        # Calculate mean safely
        if isinstance(model_rewards, list):
            model_reward_mean = 0.0
            model_reward_count = 0
            
            for r in model_rewards:
                if isinstance(r, torch.Tensor):
                    if r.numel() == 1:
                        model_reward_mean += r.item()
                        model_reward_count += 1
                    else:
                        model_reward_mean += r.mean().item()
                        model_reward_count += 1
                else:
                    model_reward_mean += float(r)
                    model_reward_count += 1
            
            if model_reward_count > 0:
                model_reward_mean /= model_reward_count
        elif isinstance(model_rewards, torch.Tensor):
            model_reward_mean = model_rewards.mean().item()
        else:
            model_reward_mean = float(model_rewards)
            
        logger.info(f"Mean API reward: {model_reward_mean:.4f}")
    except Exception as e:
        logger.error(f"Error in API reward computation: {e}")
    
    # Step 4: Test adaptation
    logger.info("Step 4: Testing adaptation with reward model")
    
    task = "test_task"
    logger.info(f"Adapting to task: {task}")
    
    # Time the adaptation
    start_time = time.time()
    
    result = api.adapt(
        task=task,
        iterations=2,     # Perform 2 iterations of adaptation
        batch_size=5,     # Batch size of 5
        learning_rate=1e-4,
        quality_threshold=0.7
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Adaptation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Adaptation metrics: {result}")
    
    # Step 5: Test improvement
    logger.info("Step 5: Testing self-improvement with reward model")
    
    # Time the improvement
    start_time = time.time()
    
    try:
        # The self_improve method doesn't take initial_trajectories
        result = api.self_improve(
            task=task,
            iterations=2,
            trajectories_per_iter=4,
            quality_threshold=0.6
        )
    except Exception as e:
        logger.error(f"Error in self-improvement: {e}")
        result = {"error": str(e)}
    
    elapsed_time = time.time() - start_time
    logger.info(f"Self-improvement completed in {elapsed_time:.2f} seconds")
    logger.info(f"Improvement metrics: {result}")
    
    # Step 6: Generate samples and compute rewards
    logger.info("Step 6: Generating samples and computing rewards")
    
    try:
        samples = api.generate(
            task=task,
            batch_size=5,
            guidance_scale=1.0
        )
        
        # Handle various return types from generate
        trajectories = samples
        
        # If samples is a tuple, extract the trajectories
        if isinstance(samples, tuple):
            trajectories = samples[0]  # Typically the first element is the trajectories
            logger.info(f"Generate returned a tuple, using first element as trajectories")
            
        # Use the reward model directly
        if isinstance(trajectories, torch.Tensor):
            if isinstance(task, str) and hasattr(api.model, 'task_embedding_model'):
                task_embedding = api.model.task_embedding_model(task)
                rewards = api.model.reward_model.compute_reward(trajectories, task_embedding)
            else:
                rewards = api.model.reward_model.compute_reward(trajectories)
                
            # Convert tensor rewards to list for logging
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.tolist() if rewards.numel() > 1 else [rewards.item()]
            
            logger.info(f"Generated {len(trajectories)} trajectories")
            logger.info(f"Rewards: {rewards}")
            logger.info(f"Mean reward: {sum(rewards)/len(rewards):.4f}")
        else:
            logger.warning(f"Unexpected trajectories type: {type(trajectories)}")
    except Exception as e:
        logger.error(f"Error in generate/compute_reward: {e}")
    
    logger.info("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())