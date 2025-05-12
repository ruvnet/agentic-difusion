"""
Example of using the AdaptDiffuserAdaptation for self-evolving adaptation.

This example demonstrates how to use AdaptDiffuserAdaptation to adapt a model
to new tasks without requiring expert data.
"""

import os
import torch
import numpy as np
import logging
import argparse
from typing import Dict, List, Optional, Tuple

from agentic_diffusion.core.adapt_diffuser.base import AdaptDiffuser
from agentic_diffusion.adaptation.adapt_diffuser_adaptation import (
    AdaptDiffuserAdaptation,
    AdaptDiffuserDiscriminator,
    SyntheticExpertGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_trajectories(
    batch_size: int = 5,
    channels: int = 3,
    img_size: int = 32
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Create dummy trajectories for demonstration purposes.
    
    Args:
        batch_size: Number of trajectories to create
        channels: Number of channels in the trajectories
        img_size: Size of each trajectory image
        
    Returns:
        Tuple of (trajectories, rewards)
    """
    trajectories = []
    rewards = []
    
    for i in range(batch_size):
        # Create random trajectory with values between -1 and 1
        trajectory = torch.rand((channels, img_size, img_size)) * 2 - 1
        
        # Assign synthetic reward (higher index = higher reward for demonstration)
        reward = 0.5 + i * 0.1
        
        trajectories.append(trajectory)
        rewards.append(reward)
    
    return trajectories, rewards


def main(
    adaptation_rate: float = 0.1,
    quality_threshold: float = 0.7,
    save_path: str = './checkpoints/adaptation',
    device: str = None
):
    """
    Run the AdaptDiffuser adaptation example.
    
    Args:
        adaptation_rate: Learning rate for adaptation
        quality_threshold: Threshold for high-quality samples
        save_path: Path to save checkpoints
        device: Device to use (defaults to CUDA if available)
    """
    # Use CUDA if available
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Create directories for checkpoints
    os.makedirs(save_path, exist_ok=True)
    
    # Simulation parameters
    channels = 3
    img_size = 32
    embedding_dim = 16
    
    # Create a simple mock AdaptDiffuser model
    class MockAdaptDiffuser(AdaptDiffuser):
        def __init__(self):
            self.img_size = img_size
            self.channels = channels
            
            # Mock task embedding model
            class MockTaskEmbedding:
                def __init__(self):
                    self.embedding_dim = embedding_dim
                
                def encode(self, task):
                    if isinstance(task, str):
                        # Simple hash-based encoding
                        task_hash = hash(task) % 10000
                        embedding = torch.zeros(embedding_dim)
                        for i in range(embedding_dim):
                            embedding[i] = ((task_hash >> i) & 1) * 2 - 1
                        return embedding
                    return task
            
            self.task_embedding_model = MockTaskEmbedding()
            
        def encode_task(self, task):
            return self.task_embedding_model.encode(task)
            
        def generate(self, batch_size=1, task=None, custom_guidance_scale=None, conditioning=None):
            shape = (batch_size, self.channels, self.img_size, self.img_size)
            return torch.randn(shape)
            
        def compute_reward(self, trajectories, task=None):
            # Simple mock reward function
            batch_size = trajectories.shape[0]
            return torch.rand(batch_size)
            
        def adapt_to_task(self, task, num_steps=100, batch_size=8, **kwargs):
            logger.info(f"Adapting to task: {task} for {num_steps} steps")
            return {"loss": 0.5 - 0.01 * num_steps}
    
    # Create mock AdaptDiffuser model
    adapt_diffuser = MockAdaptDiffuser()
    
    # Create adaptation mechanism
    adaptation = AdaptDiffuserAdaptation(
        adapt_diffuser=adapt_diffuser,
        adaptation_rate=adaptation_rate,
        quality_threshold=quality_threshold,
        checkpoint_dir=save_path,
        device=device
    )
    
    # Create synthetic expert generator
    generator = SyntheticExpertGenerator(
        adapt_diffuser=adapt_diffuser,
        quality_threshold=quality_threshold,
        device=device
    )
    
    # Generate some synthetic data
    logger.info("Generating synthetic expert data")
    synthetic_samples, synthetic_rewards = generator.generate_synthetic_data(
        task="example_task",
        num_samples=10
    )
    
    logger.info(f"Generated {len(synthetic_samples)} synthetic samples with rewards: {synthetic_rewards}")
    
    # Adapt to a task
    logger.info("Adapting to a task")
    metrics = adaptation._adapt_to_task(
        task="example_task",
        num_steps=10,
        batch_size=2
    )
    
    logger.info(f"Adaptation metrics: {metrics}")
    
    # Create dummy code for adaptation
    example_code = """
    def example_function(x, y):
        # This is an example function
        return x + y
    """
    
    # Adapt code
    logger.info("Adapting code example")
    adapted_code = adaptation.adapt(
        code=example_code,
        feedback={"task": "improve_code_example"},
        language="python"
    )
    
    logger.info(f"Original code:\n{example_code}")
    logger.info(f"Adapted code:\n{adapted_code}")
    
    # Create trajectories and adapt
    logger.info("Creating dummy trajectories")
    trajectories, rewards = create_dummy_trajectories()
    
    logger.info("Adapting to trajectories")
    adaptation._adapt_to_trajectories(trajectories)
    
    # Save and load state
    save_file = os.path.join(save_path, "adaptation_state.pkl")
    logger.info(f"Saving adaptation state to {save_file}")
    adaptation.save_state(save_file)
    
    logger.info(f"Loading adaptation state from {save_file}")
    adaptation.load_state(save_file)
    
    logger.info("Example completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaptDiffuser Adaptation Example")
    parser.add_argument("--adaptation_rate", type=float, default=0.1, help="Learning rate for adaptation")
    parser.add_argument("--quality_threshold", type=float, default=0.7, help="Threshold for high-quality samples")
    parser.add_argument("--save_path", type=str, default="./checkpoints/adaptation", help="Path to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use (defaults to CUDA if available)")
    
    args = parser.parse_args()
    main(
        adaptation_rate=args.adaptation_rate,
        quality_threshold=args.quality_threshold,
        save_path=args.save_path,
        device=args.device
    )