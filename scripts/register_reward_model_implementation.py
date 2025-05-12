#!/usr/bin/env python3
"""
Script to register a reward model with the AdaptDiffuser API.

This script provides functionality to register various reward models
and optionally test their functionality.
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
import random

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI
from agentic_diffusion.core.reward_functions import SimpleRewardModel, AdaptDiffuserTestRewardModel, CompositeRewardModel


def create_reward_model(args):
    """
    Create a reward model based on the provided arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        An instance of a reward model
    """
    if args.test_mode:
        logger.info("Creating AdaptDiffuserTestRewardModel for testing")
        config = {
            "initial_reward": args.initial_reward,
            "improvement_rate": args.improvement_rate
        }
        return AdaptDiffuserTestRewardModel(config)
    elif args.composite:
        logger.info("Creating CompositeRewardModel")
        config = {
            "components": [
                {
                    "type": "simple",
                    "base_reward": args.base_reward,
                    "noise_scale": args.noise_scale
                },
                {
                    "type": "test",
                    "initial_reward": args.initial_reward,
                    "improvement_rate": args.improvement_rate
                }
            ],
            "weights": [0.7, 0.3]
        }
        return CompositeRewardModel(config)
    else:
        logger.info("Creating SimpleRewardModel")
        config = {
            "base_reward": args.base_reward,
            "noise_scale": args.noise_scale
        }
        return SimpleRewardModel(config)


def test_reward_model(reward_model, num_samples=5):
    """
    Test the reward model with sample trajectories.
    
    Args:
        reward_model: The reward model to test
        num_samples: Number of sample trajectories to test
        
    Returns:
        True if tests pass, False otherwise
    """
    logger.info(f"Testing reward model with {num_samples} samples")
    
    # Create dummy trajectories (random tensors for testing)
    trajectories = [
        torch.randn(10, 5)  # Random trajectory with 10 steps and 5 dimensions
        for _ in range(num_samples)
    ]
    
    # Test direct reward calculation
    direct_rewards = []
    for trajectory in trajectories:
        reward = reward_model.compute_reward(trajectory)
        direct_rewards.append(reward)
    
    # Test batch reward calculation
    model_rewards = reward_model.compute_rewards(trajectories)
    
    # Log results
    logger.info(f"Direct rewards: {direct_rewards}")
    logger.info(f"Model rewards: {model_rewards}")
    logger.info(f"Mean direct reward: {sum(direct_rewards)/len(direct_rewards):.4f}")
    logger.info(f"Mean model reward: {sum(model_rewards)/len(model_rewards):.4f}")
    
    # Check that rewards are within expected range (0 to 1)
    for reward in direct_rewards + model_rewards:
        if reward < 0 or reward > 1:
            logger.error(f"Reward {reward} is outside the expected range [0, 1]")
            return False
    
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Register a reward model with AdaptDiffuser API")
    parser.add_argument("--config", default="./config/test_cpu.yaml", help="Path to configuration file")
    parser.add_argument("--test-mode", action="store_true", help="Use test reward model")
    parser.add_argument("--composite", action="store_true", help="Use composite reward model")
    parser.add_argument("--run-test", action="store_true", help="Run tests after registration")
    parser.add_argument("--base-reward", type=float, default=0.6, help="Base reward for SimpleRewardModel")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Noise scale for SimpleRewardModel")
    parser.add_argument("--initial-reward", type=float, default=0.5, help="Initial reward for TestRewardModel")
    parser.add_argument("--improvement-rate", type=float, default=0.05, help="Improvement rate for TestRewardModel")
    parser.add_argument("--num-test-samples", type=int, default=5, help="Number of test samples")
    
    args = parser.parse_args()
    
    # Create AdaptDiffuser API
    logger.info(f"Creating AdaptDiffuserAPI from config: {args.config}")
    
    # Load configuration from YAML file
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration file loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        return 1
    
    # Create API with loaded config
    api = AdaptDiffuserAPI(config=config)
    
    # Create reward model
    reward_model = create_reward_model(args)
    
    # Register reward model
    logger.info("Registering reward model with AdaptDiffuser API")
    success = api.register_reward_model(reward_model)
    
    if success:
        logger.info("Reward model successfully registered")
    else:
        logger.error("Failed to register reward model")
        return 1
    
    # Run tests if requested
    if args.run_test:
        if test_reward_model(reward_model, args.num_test_samples):
            logger.info("Reward model registered and tested successfully")
        else:
            logger.error("Reward model tests failed")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())