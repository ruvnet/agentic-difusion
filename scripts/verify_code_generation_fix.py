#!/usr/bin/env python3
"""
Verification script for the fixed code generation pipeline.

This script tests the code generation with a simple prompt to verify
the fix for the tensor shape mismatch error in the ResidualBlock.
"""

import argparse
import logging
import sys
import os
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("verify_code_generation_fix")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def run_verification(use_mock=False):
    """Run verification of code generation with the fixed ResidualBlock."""
    from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
    from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
    from agentic_diffusion.code_generation.syntax_model import SyntaxModel
    
    # Test parameters
    specification = "Write a function to check if a string is a palindrome"
    language = "python"
    
    if use_mock:
        logger.info("Using a mock diffusion model for testing")
        
        # Import after setting up paths
        from unittest.mock import MagicMock
        
        # Create mock output
        mock_code = '''
def is_palindrome(s):
    """
    Check if a string is a palindrome (reads the same forward and backward).
    
    Args:
        s (str): The string to check
        
    Returns:
        bool: True if the string is a palindrome, False otherwise
    """
    # Normalize the string: convert to lowercase and remove non-alphanumeric chars
    s = ''.join(c.lower() for c in s if c.isalnum())
    
    # Check if the string is equal to its reverse
    return s == s[::-1]
'''
        # Create a mock diffusion model
        mock_diffusion_model = MagicMock()
        mock_diffusion_model.generate.return_value = mock_code
        mock_diffusion_model.evaluate_code_quality.return_value = {
            "syntax_score": 0.95,
            "quality_score": 0.95,
            "relevance_score": 0.9,
            "overall_score": 0.93,
            "complexity": "low"
        }
        
        # Create API instance with the mock diffusion model
        api = CodeGenerationAPI(
            diffusion_model=mock_diffusion_model,
            config={
                "default_language": "python",
                "batch_size": 1,
                "precision": "float32",
                "adaptation_type": "hybrid"
            }
        )
        
        try:
            # Generate code
            code, metadata = api.generate_code(
                specification=specification,
                language=language,
                custom_parameters={"max_length": 512}
            )
            
            logger.info(f"Successfully generated code with mock")
            logger.info(f"Generated code snippet:\n{code[:200]}...")
            quality_metrics = metadata.get("quality", {})
            if quality_metrics:
                logger.info(f"Quality metrics: {quality_metrics}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate code with mock: {e}")
            return False
    else:
        logger.info("Using the real CodeDiffusion model")
        
        # Create a real diffusion model
        try:
            from agentic_diffusion.code_generation.diffusion.code_diffusion import CodeDiffusion
            from agentic_diffusion.code_generation.models.code_unet import CodeUNet
            
            # Create a minimal diffusion model
            tokenizer = CodeTokenizer(language="python")
            syntax_model = SyntaxModel()
            
            # Try to create a minimal diffusion model for testing
            logger.info("Creating minimal CodeDiffusion model for test")
            diffusion_model = CodeDiffusion(
                model=CodeUNet(
                    d_model=256,  # Use minimal size for testing
                    num_layers=2,
                    context_dim=128
                ),
                tokenizer=tokenizer,
                syntax_model=syntax_model
            )
            
            # Create API instance
            api = CodeGenerationAPI(
                diffusion_model=diffusion_model,
                config={
                    "default_language": "python",
                    "batch_size": 1,
                    "precision": "float32",
                    "adaptation_type": "hybrid"
                }
            )
            
            start_time = time.time()
            
            try:
                # Generate code
                code, metadata = api.generate_code(
                    specification=specification,
                    language=language,
                    custom_parameters={
                        "max_length": 128,  # Smaller for testing
                        "batch_size": 1,  # Small batch size to avoid OOM
                        "num_iterations": 1  # Minimal iterations for test
                    }
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Successfully generated code in {elapsed:.2f} seconds")
                quality_metrics = metadata.get("quality", {})
                if quality_metrics:
                    logger.info(f"Quality metrics: {quality_metrics}")
                logger.info(f"Generated code snippet:\n{code[:200]}...")
                
                return True
            except Exception as e:
                logger.error(f"Failed to generate code with real model: {e}")
                logger.error(f"Error details: {str(e)}")
                return False
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify code generation with fix")
    parser.add_argument("--mock", action="store_true", help="Use mock diffusion model")
    args = parser.parse_args()
    
    success = run_verification(use_mock=args.mock)
    sys.exit(0 if success else 1)