#!/usr/bin/env python3
"""
Test script for AdaptDiffuser with a reward model.

This script tests the functionality of AdaptDiffuser's 'adapt' and 'improve' commands
with a simple test reward model that returns sensible scores for trajectories.
"""

import os
import sys
import logging
import subprocess
import json
import time
import torch
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI
from agentic_diffusion.core.reward_functions import AdaptDiffuserTestRewardModel

# Configure logging
logger = logging.getLogger("test_adaptdiffuser_with_reward")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create output directories
output_dir = "test_results/adaptdiffuser_with_reward"
log_dir = f"{output_dir}/logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
logger.info(f"Created output directories: {output_dir}, {log_dir}")

def run_command(command, log_file=None):
    """Run a command and return its output and return code."""
    start_time = time.time()
    
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                command, 
                stdout=f,
                stderr=subprocess.PIPE,
                shell=True
            )
            _, stderr = process.communicate()
    else:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        _, stderr = process.communicate()
    
    return_code = process.returncode
    elapsed_time = time.time() - start_time
    
    return return_code, stderr.decode('utf-8'), elapsed_time

def register_test_reward_model():
    """Register test reward model with AdaptDiffuser API."""
    logger.info("Registering test reward model")
    
    # Create a test script that registers the reward model
    script_content = """
import os
import sys
import logging
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_diffusion.api.adapt_diffuser_api import AdaptDiffuserAPI
from agentic_diffusion.core.reward_functions import AdaptDiffuserTestRewardModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__main__)

# Create AdaptDiffuserAPI
logger.info("Creating AdaptDiffuserAPI from config: ./config/test_cpu.yaml")
api = AdaptDiffuserAPI.from_config('./config/test_cpu.yaml')

# Create test reward model
logger.info("Creating AdaptDiffuserTestRewardModel for testing")
reward_model = AdaptDiffuserTestRewardModel(initial_reward=0.5, improvement_rate=0.05)

# Register reward model
logger.info("Registering reward model with AdaptDiffuser API")
api.register_reward_model(reward_model)
logger.info("Reward model successfully registered")

# Test reward model
logger.info("Testing reward model with 5 samples")
dummy_samples = torch.randn(5, 128)  # 5 random samples
direct_rewards = [reward_model.compute_reward(sample) for sample in dummy_samples]
model_rewards = api.compute_rewards(dummy_samples, "test_task")

logger.info(f"Direct rewards: {direct_rewards}")
logger.info(f"Model rewards: {model_rewards}")
logger.info(f"Mean direct reward: {sum(direct_rewards)/len(direct_rewards):.4f}")
logger.info(f"Mean model reward: {sum(model_rewards)/len(model_rewards):.4f}")
logger.info("Reward model registered and tested successfully")
"""
    
    # Write the script to a temporary file
    with open("scripts/register_test_reward.py", "w") as f:
        f.write(script_content)
    
    # Run the script
    return_code, stderr, elapsed_time = run_command("python -m scripts.register_test_reward")
    
    if return_code == 0:
        logger.info(f"Command completed in {elapsed_time:.2f}s with return code {return_code}")
    else:
        logger.error(f"Command failed with return code {return_code}")
    
    if stderr:
        logger.warning(f"Standard Error:\n{stderr}")
    
    return return_code == 0

def test_adapt_command():
    """Test the 'adapt' command with the registered reward model."""
    logger.info("Running AdaptDiffuser adapt test")
    
    # Command to test adapt
    command = "python -m agentic_diffusion adapt --config ./config/test_cpu.yaml --reward-model test --task test_task --iterations 10"
    log_file = f"{log_dir}/adapt_command.log"
    
    # Run the command
    return_code, stderr, elapsed_time = run_command(command, log_file)
    
    logger.info(f"Adapt command completed in {elapsed_time:.2f}s with return code {return_code}")
    
    if stderr:
        logger.warning(f"Standard Error:\n{stderr}")
    
    if return_code == 0:
        logger.info("Adapt command executed successfully")
    else:
        logger.error("Adapt command failed")
    
    return return_code == 0

def test_improve_command():
    """Test the 'improve' command with the registered reward model."""
    logger.info("Running AdaptDiffuser improve test")
    
    # Command to test improve
    command = "python -m agentic_diffusion improve --config ./config/test_cpu.yaml --reward-model test --task test_task --quality-threshold 0.7 --iterations 2"
    log_file = f"{log_dir}/improve_command.log"
    
    # Run the command
    return_code, stderr, elapsed_time = run_command(command, log_file)
    
    logger.info(f"Improve command completed in {elapsed_time:.2f}s with return code {return_code}")
    
    if stderr:
        logger.warning(f"Standard Error:\n{stderr}")
    
    if return_code == 0:
        logger.info("Improve command executed successfully")
    else:
        logger.error("Improve command failed")
    
    return return_code == 0

def main():
    """Run tests for AdaptDiffuser with reward model."""
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {
            "register_reward_model": False,
            "adapt_command": False,
            "improve_command": False
        },
        "all_passed": False
    }
    
    # Step 1: Register reward model
    logger.info("\n=== Step 1: Registering Reward Model ===")
    results["tests"]["register_reward_model"] = register_test_reward_model()
    
    # Step 2: Test adapt command
    logger.info("\n=== Step 2: Testing 'adapt' Command ===")
    results["tests"]["adapt_command"] = test_adapt_command()
    
    if not results["tests"]["adapt_command"]:
        logger.error("Adapt command test failed. Continuing with improve test.")
    
    # Step 3: Test improve command
    logger.info("\n=== Step 3: Testing 'improve' Command ===")
    results["tests"]["improve_command"] = test_improve_command()
    
    if not results["tests"]["improve_command"]:
        logger.error("Improve command test failed.")
    
    # Check if all tests passed
    results["all_passed"] = all(results["tests"].values())
    
    # Save results
    with open(f"{output_dir}/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/test_results.json")
    
    if not results["all_passed"]:
        logger.error("Some tests failed. Check results for details.")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()