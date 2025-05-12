#!/usr/bin/env python3
"""
Verification script for the dimension mismatch fix in code generation.

This script tests the error handling in the code generation pipeline to verify
that our fixes properly handle dimension mismatches and provide meaningful error messages.
"""

import os
import sys
import logging
import argparse
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("verify_code_generation_fix")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_fixed_code_generation():
    """Test the fixed code generation pipeline and verify error handling."""
    try:
        logger.info("Loading required modules...")
        from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
        from agentic_diffusion.code_generation.syntax_model import SyntaxModel
        from agentic_diffusion.code_generation.code_generator import CodeGenerator
        from agentic_diffusion.code_generation.diffusion.code_diffusion import CodeDiffusion
        from agentic_diffusion.code_generation.models.code_unet import CodeUNet
        from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
        
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
        
        # Create API with the diffusion model
        logger.info("Creating CodeGenerationAPI...")
        api = CodeGenerationAPI(diffusion_model=diffusion_model, config={
            "batch_size": 1,  # Small batch size for faster testing
            "num_iterations": 1,
            "max_length": 64,  # Short max length for testing
        })
        
        # Generate code with a simple prompt
        prompt = "Write a function that checks if a number is prime"
        language = "python"
        
        logger.info(f"Generating code with prompt: '{prompt}'")
        
        # Run with small parameters to make it quick
        code, metadata = api.generate_code(
            specification=prompt,
            language=language
        )
        
        if code:
            logger.info("Successfully generated code")
            logger.info("Generated code snippet:")
            print(code)
            return True
        else:
            if metadata and "error" in metadata:
                logger.error(f"Code generation failed with error: {metadata['error']['message']}")
                logger.error(f"Error details: {metadata['error']['details']}")
            else:
                logger.error("Code generation returned empty result without error metadata")
            return False
    
    except Exception as e:
        logger.error(f"Unexpected error during test: {e}")
        # Print full stack trace for debugging
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test the error handling specifically with a misconfigured model."""
    try:
        logger.info("Testing error handling with mismatched dimensions...")
        from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
        from agentic_diffusion.code_generation.syntax_model import SyntaxModel
        from agentic_diffusion.code_generation.code_generator import CodeGenerator
        from agentic_diffusion.code_generation.diffusion.code_diffusion import CodeDiffusion
        from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
        
        # Create tokenizer and syntax model
        tokenizer = CodeTokenizer(language="python")
        syntax_model = SyntaxModel()
        
        # Create diffusion model with deliberately mismatched dimensions
        # This should trigger our error handling
        diffusion_model = CodeDiffusion(
            vocab_size=10000,
            embedding_dim=256,
            hidden_dim=256,
            num_layers=3,  # More layers to cause potential dimension issues
            num_heads=4,
            dropout=0.1,
            num_timesteps=50,
            device="cpu"
        )
        
        # Create API
        api = CodeGenerationAPI(diffusion_model=diffusion_model)
        
        # Try to generate code which might trigger errors
        prompt = "Write a recursive function"
        
        try:
            code, metadata = api.generate_code(
                specification=prompt,
                language="python"
            )
            
            if code is None and metadata and "error" in metadata:
                logger.info("Successfully detected error and provided metadata:")
                logger.info(f"Error type: {metadata['error']['type']}")
                logger.info(f"Error message: {metadata['error']['message']}")
                logger.info(f"Error details: {metadata['error']['details']}")
                return True
            else:
                logger.warning("No error detected or properly handled")
                return False
                
        except Exception as e:
            logger.error(f"Unexpected exception not properly handled: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up error handling test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify fixes to code generation error handling")
    parser.add_argument("--test-type", choices=["fixed", "error-handling", "both"], 
                        default="both", help="Type of test to run")
    args = parser.parse_args()
    
    success = True
    
    if args.test_type in ["fixed", "both"]:
        logger.info("Running test with fixed code generation...")
        success = test_fixed_code_generation() and success
        
    if args.test_type in ["error-handling", "both"]:
        logger.info("Running error handling test...")
        success = test_error_handling() and success
    
    if success:
        logger.info("All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("One or more tests failed")
        sys.exit(1)