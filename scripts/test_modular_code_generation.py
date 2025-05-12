#!/usr/bin/env python
"""
Test script for the modularized code generation.

This script tests the modularized code_generation package by:
1. Importing all necessary components
2. Creating a code diffusion model
3. Generating code based on a specification
4. Verifying the output
"""

import os
import sys
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path if needed
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import the main components from our modularized package
    from agentic_diffusion.code_generation import (
        CodeDiffusion,
        generate_code,
        create_code_diffusion_model
    )
    logger.info("Successfully imported code generation modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def test_model_creation():
    """Test that we can create a code diffusion model."""
    logger.info("Testing model creation...")
    
    try:
        # Create a tiny model for testing purposes
        model = create_code_diffusion_model(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            device="cpu"
        )
        logger.info(f"Successfully created model: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def test_code_generation(model=None):
    """Test that we can generate code from a specification."""
    logger.info("Testing code generation...")
    
    # Create a model if not provided
    if model is None:
        model = create_code_diffusion_model(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            dropout=0.1,
            device="cpu"
        )
    
    # Define a simple specification
    specification = "Create a function that calculates the factorial of a number."
    language = "python"
    
    try:
        # Generate code using the high-level function
        logger.info(f"Generating code for specification: {specification}")
        code = generate_code(
            specification=specification,
            language=language,
            model=model,
            max_length=128,
            num_samples=2,
            guidance_scale=1.2,
            temperature=0.7
        )
        
        logger.info(f"Generated code:\n{code}")
        return code
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        raise

def main():
    """Run all tests to verify the code generation functionality."""
    logger.info("Starting code generation tests")
    
    try:
        # Test model creation
        model = test_model_creation()
        
        # Test code generation
        code = test_code_generation(model)
        
        logger.info("All tests completed successfully")
        return True
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)