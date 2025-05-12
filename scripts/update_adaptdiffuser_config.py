#!/usr/bin/env python3
"""
Script to update a configuration file with reward model settings.

This script adds reward model configuration to an AdaptDiffuser config file,
enabling the use of reward models in adapt and improve commands.
"""

import sys
import os
import argparse
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None


def save_config(config, config_path):
    """Save configuration to a YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        return False


def update_config_with_reward_model(config, reward_model_type):
    """
    Update configuration with reward model settings.
    
    Args:
        config: Configuration dictionary
        reward_model_type: Type of reward model to use
        
    Returns:
        Updated configuration dictionary
    """
    if "adaptdiffuser" not in config:
        config["adaptdiffuser"] = {}
    
    # Create reward model configuration
    if reward_model_type == "simple":
        reward_config = {
            "type": "simple",
            "base_reward": 0.6,
            "noise_scale": 0.1
        }
    elif reward_model_type == "test":
        reward_config = {
            "type": "test",
            "initial_reward": 0.5,
            "improvement_rate": 0.05
        }
    elif reward_model_type == "composite":
        reward_config = {
            "type": "composite",
            "components": [
                {
                    "type": "simple",
                    "base_reward": 0.6,
                    "noise_scale": 0.1
                },
                {
                    "type": "test",
                    "initial_reward": 0.5,
                    "improvement_rate": 0.05
                }
            ],
            "weights": [0.7, 0.3]
        }
    else:
        logger.warning(f"Unknown reward model type: {reward_model_type}, using simple model")
        reward_config = {
            "type": "simple",
            "base_reward": 0.6,
            "noise_scale": 0.1
        }
    
    # Update configuration
    config["adaptdiffuser"]["reward_model"] = reward_config
    logger.info(f"Updated configuration with {reward_model_type} reward model")
    
    # Add task embedding model configuration
    task_embedding_config = {
        "type": "simple",
        "embedding_dim": 128,
        "seed": 42
    }
    
    # Add paths for task embedding
    if "adaptdiffuser_paths" not in config:
        config["adaptdiffuser_paths"] = {}
    
    config["adaptdiffuser_paths"]["task_embedding_model"] = "agentic_diffusion.adaptation.task_embeddings"
    config["adaptdiffuser"]["task_embedding"] = task_embedding_config
    
    logger.info("Added task embedding model configuration")
    
    return config


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Update configuration with reward model settings")
    parser.add_argument("--config", default="./config/test_cpu.yaml", help="Path to configuration file")
    parser.add_argument("--reward-model", default="simple", choices=["simple", "test", "composite"], help="Type of reward model to use")
    parser.add_argument("--output", help="Output path for the updated configuration")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return 1
    
    # Update configuration with reward model settings
    config = update_config_with_reward_model(config, args.reward_model)
    
    # Save configuration
    output_path = args.output or args.config
    if not save_config(config, output_path):
        return 1
    
    logger.info(f"Successfully updated configuration with {args.reward_model} reward model")
    return 0


if __name__ == "__main__":
    sys.exit(main())