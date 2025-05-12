#!/usr/bin/env python3
"""
End-to-end test of the code generation pipeline with the fixed ResidualBlock.

This script tests the actual diffusion model directly to verify the tensor
shape mismatch fix is working correctly.
"""

import logging
import os
import sys
import torch
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_code_generation")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_test():
    """Test the fixed code generation pipeline with minimal example."""
    try:
        logger.info("Loading required modules...")
        from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
        from agentic_diffusion.code_generation.syntax_model import SyntaxModel
        from agentic_diffusion.code_generation.code_generator import CodeGenerator
        from agentic_diffusion.code_generation.diffusion.code_diffusion import CodeDiffusion
        from agentic_diffusion.code_generation.models.code_unet import CodeUNet
        
        # Create tokenizer and syntax model
        logger.info("Creating tokenizer and syntax model...")
        tokenizer = CodeTokenizer(language="python")
        syntax_model = SyntaxModel()
        
        # Create minimal code UNet for testing
        logger.info("Creating minimal CodeUNet...")
        code_unet = CodeUNet(
            vocab_size=10000,  # Vocabulary size for tokenizer
            embedding_dim=256,  # Must match the expected dimension
            hidden_dim=256,    # Small hidden dimension for testing
            num_layers=2,      # Minimal layers
            num_heads=4,       # Minimal attention heads
            dropout=0.1,
            condition_dim=256,  # Must match embedding_dim for cross-attention
            num_downsamples=1  # Minimal downsampling
        )
        
        # Create diffusion model
        logger.info("Creating CodeDiffusion model...")
        diffusion_model = CodeDiffusion(
            vocab_size=10000,
            embedding_dim=256,  # Must match CodeUNet's embedding_dim
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            num_timesteps=100,  # Reduced for testing
            device="cpu"  # Force CPU for testing
        )
        
        # Create code generator
        logger.info("Creating CodeGenerator...")
        code_generator = CodeGenerator(
            # We'll use the default parameters since we don't need to pass the models directly
            # This will use the same parameters as in the production code
            tokenizer=tokenizer,
            syntax_model=syntax_model,
            diffusion_model=diffusion_model
        )
        
        # Generate code with a simple prompt
        prompt = "Write a function that checks if a number is prime"
        language = "python"
        
        logger.info(f"Generating code with prompt: '{prompt}'")
        
        # Run with small parameters to make it quick
        code = code_generator.generate_code(
            specification=prompt,
            language=language,
            batch_size=1,
            max_length=64,  # Very short for quick testing
            num_iterations=1
        )
        
        if code:
            logger.info("Successfully generated code")
            logger.info("Generated code snippet:")
            print(code)
            return True
        else:
            logger.error("Code generation returned empty result")
            return False
    
    except Exception as e:
        logger.error(f"Error testing code generation: {e}")
        logger.error(f"Exception details: {str(e)}")
        # Print full stack trace for debugging
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the fixed code generation pipeline")
    args = parser.parse_args()
    
    success = run_test()
    sys.exit(0 if success else 1)