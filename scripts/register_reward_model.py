#!/usr/bin/env python3
"""
Script to register a reward model for AdaptDiffuser.

This script creates and registers a reward model for use with the AdaptDiffuser
model. It can be used to test the adapt and improve commands with a real reward model.
"""

import argparse
import logging
import os
import sys
import yaml

import torch
import numpy as np

# Add parent directory to path to import agentic_diffusion module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI
from agentic_diffusion.core.reward_functions import SimpleRewardModel, AdaptDiffuserTestRewardModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('register_reward_model')


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except (yaml.YAMLError, FileNotFoundError) as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        sys.exit(1)


def register_reward_model(config_path, test_mode=False):
    """Register a reward model with AdaptDiffuser.
    
    Args:
        config_path: Path to the configuration file.
        test_mode: Whether to use the test-specific reward model.
        
    Returns:
        Tuple of (AdaptDiffuserAPI, RewardModel)
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Create reward model
    if test_mode:
        logger.info("Creating AdaptDiffuserTestRewardModel for test mode")
        reward_model = AdaptDiffuserTestRewardModel(config.get("adaptdiffuser", {}).get("reward_model", {}))
    else:
        logger.info("Creating SimpleRewardModel")
        reward_model = SimpleRewardModel(config.get("adaptdiffuser", {}).get("reward_model", {}))
    
    # Initialize AdaptDiffuser API with reward model
    logger.info("Initializing AdaptDiffuser API with custom reward model")
    api = AdaptDiffuserAPI(config)
    
    # Register reward model with AdaptDiffuser
    api.model.register_reward_model(reward_model)
    logger.info("Reward model successfully registered with AdaptDiffuser model")
    
    return api, reward_model


def test_reward_model(api, num_samples=5):
    """Test the reward model on random trajectories.
    
    Args:
        api: AdaptDiffuserAPI instance.
        num_samples: Number of test samples to generate.
    """
    logger.info(f"Testing reward model with {num_samples} samples")
    
    # Generate some random trajectories
    trajectories = []
    for _ in range(num_samples):
        # Create a simple trajectory with random states
        state_dim = api.model.state_dim if hasattr(api.model, 'state_dim') else 16
        trajectory = torch.randn(10, state_dim)  # 10 timesteps
        trajectories.append(trajectory)
    
    # Compute rewards for the trajectories
    rewards = api.model.compute_rewards(trajectories)
    
    # Log the results
    logger.info(f"Generated rewards: {rewards.tolist()}")
    logger.info(f"Mean reward: {rewards.mean().item():.4f}")


def update_config_file(config_path, test_mode=False):
    """Update the config file to include reward model settings."""
    try:
        # Load the existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize adaptdiffuser section if it doesn't exist
        if "adaptdiffuser" not in config:
            config["adaptdiffuser"] = {}
        
        # Add or update reward model configuration
        reward_model_type = "test_reward" if test_mode else "simple_reward"
        config["adaptdiffuser"]["reward_model"] = {
            "type": reward_model_type,
            "base_reward": 0.6,
            "noise_scale": 0.05,
            "initial_reward": 0.5,
            "improvement_rate": 0.05
        }
        
        # Write the updated config back to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Updated configuration file {config_path} with reward model settings")
        return True
    except Exception as e:
        logger.error(f"Failed to update config file: {e}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Register a reward model for AdaptDiffuser")
    parser.add_argument("--config", default="./config/test_cpu.yaml", help="Path to configuration file")
    parser.add_argument("--test-mode", action="store_true", help="Use test-specific reward model")
    parser.add_argument("--run-test", action="store_true", help="Run a test of the reward model")
    parser.add_argument("--update-config", action="store_true", help="Update the config file with reward model settings")
    
    args = parser.parse_args()
    
    try:
        # Update the config file if requested
        if args.update_config:
            update_config_file(args.config, args.test_mode)
        
        # Register the reward model
        api, reward_model = register_reward_model(args.config, args.test_mode)
        
        # Test the reward model if requested
        if args.run_test:
            test_reward_model(api)
            
        logger.info("Reward model registration completed successfully")
        
    except Exception as e:
        logger.error(f"Error registering reward model: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())