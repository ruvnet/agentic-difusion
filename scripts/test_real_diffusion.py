#!/usr/bin/env python3
"""
Test script for the updated code diffusion model using real diffusion principles.

This script demonstrates the code generation capabilities using the new
implementation that replaces template-based generation with actual diffusion.
"""

import os
import sys
import time
import torch
import numpy as np

# Add the parent directory to the path to import agentic_diffusion
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.syntax_model import SyntaxModel
from agentic_diffusion.code_generation.code_generator import CodeGenerator
from agentic_diffusion.api.code_generation_api import create_code_generation_api


def test_direct_diffusion():
    """Test the CodeDiffusion model directly."""
    print("\n----- Testing CodeDiffusion Model Directly -----")
    
    # Initialize the diffusion model
    diffusion_model = CodeDiffusion(
        vocab_size=5000,
        embedding_dim=256,
        max_seq_len=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        num_timesteps=100
    )
    
    # Generate code using the diffusion model
    specification = "Write a Python function to check if a number is prime."
    start_time = time.time()
    code = diffusion_model.generate(
        specification=specification,
        language="python",
        guidance_scale=1.5,
        temperature=0.7,
        batch_size=2,
        use_rewards=True
    )
    elapsed = time.time() - start_time
    
    print(f"Generated code in {elapsed:.2f} seconds:")
    print("-" * 40)
    print(code)
    print("-" * 40)


def test_via_code_generator():
    """Test code generation using the CodeGenerator with diffusion model."""
    print("\n----- Testing via CodeGenerator -----")
    
    # Initialize components
    tokenizer = CodeTokenizer(language="python")
    syntax_model = SyntaxModel()
    diffusion_model = CodeDiffusion(
        vocab_size=5000,
        embedding_dim=256,
        max_seq_len=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        num_timesteps=100
    )
    
    # Create the code generator
    code_generator = CodeGenerator(
        tokenizer=tokenizer,
        syntax_model=syntax_model,
        diffusion_model=diffusion_model
    )
    
    # Generate code
    specification = "Write a recursive function to calculate factorial."
    start_time = time.time()
    code = code_generator.generate_code(
        specification=specification,
        language="python",
        batch_size=3,
        guidance_scale=1.5,
        temperature=0.7,
        use_rewards=True,
        num_iterations=2
    )
    elapsed = time.time() - start_time
    
    print(f"Generated code in {elapsed:.2f} seconds:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    
    # Evaluate the generated code
    metrics = code_generator.evaluate_quality(code, specification, "python")
    print("Quality metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}")


def test_via_api():
    """Test code generation using the CodeGenerationAPI."""
    print("\n----- Testing via CodeGenerationAPI -----")
    
    # Initialize the diffusion model
    diffusion_model = CodeDiffusion(
        vocab_size=5000,
        embedding_dim=256,
        max_seq_len=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        num_timesteps=100
    )
    
    # Create the API
    config = {
        "default_language": "python",
        "adaptation_type": "hybrid",
        "batch_size": 3,
        "guidance_scale": 1.5,
        "temperature": 0.7,
        "use_rewards": True,
        "max_length": 256,
        "num_iterations": 2
    }
    api = create_code_generation_api(diffusion_model, config)
    
    # Generate code
    specification = "Create a function to sort a dictionary by values."
    start_time = time.time()
    code, metadata = api.generate_code(
        specification=specification,
        language="python"
    )
    elapsed = time.time() - start_time
    
    print(f"Generated code in {elapsed:.2f} seconds:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    
    print("Performance metrics:")
    for k, v in metadata["performance"].items():
        print(f"- {k}: {v}")
    
    print("\nQuality metrics:")
    for k, v in metadata["quality"].items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")
    
    # Test multi-language support
    print("\nTesting JavaScript generation:")
    js_code, _ = api.generate_code(
        specification="Create a function to sort an array of objects by a property",
        language="javascript"
    )
    print(js_code)


if __name__ == "__main__":
    # Run the tests
    test_direct_diffusion()
    test_via_code_generator()
    test_via_api()