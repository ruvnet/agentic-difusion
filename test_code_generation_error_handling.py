#!/usr/bin/env python
"""
Test script for verifying error handling in code generation modules.
This script tests the improved error handling for dimension mismatches and other failures.
"""

import sys
import traceback
from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
from agentic_diffusion.code_generation.code_generator import CodeGenerator
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.syntax_model import SyntaxModel

class MockDiffusionModel:
    """Mock diffusion model that raises dimension mismatch errors for testing."""
    
    def __init__(self, error_type="dimension_mismatch"):
        self.error_type = error_type
    
    def generate(self, specification, language, partial_code, **kwargs):
        """Mock generate method that raises errors for testing."""
        if self.error_type == "dimension_mismatch":
            raise AssertionError("Embedding dimension mismatch: expected 128 but got 64")
        elif self.error_type == "empty_output":
            return ""
        elif self.error_type == "none_output":
            return None
        else:
            raise RuntimeError("Failed to generate code with unspecified error")
    
    def sample(self, specification, language, partial_code, **kwargs):
        """Mock sample method that raises errors for testing."""
        if self.error_type == "dimension_mismatch":
            raise ValueError("Expected tensor shape [32, 64, 128] but got [32, 64, 256]")
        elif self.error_type == "empty_output":
            return ""
        elif self.error_type == "none_output":
            return None
        else:
            raise RuntimeError("Failed to sample code with unspecified error")


def test_direct_code_generator():
    """Test error handling directly in the CodeGenerator class."""
    print("\n=== Testing CodeGenerator error handling ===")
    
    # Initialize components
    tokenizer = CodeTokenizer(language="python")
    syntax_model = SyntaxModel()
    
    # Test different error types
    error_types = ["dimension_mismatch", "empty_output", "none_output", "generic_error"]
    
    for error_type in error_types:
        print(f"\nTesting error type: {error_type}")
        diffusion_model = MockDiffusionModel(error_type=error_type)
        code_generator = CodeGenerator(tokenizer, syntax_model, diffusion_model)
        
        try:
            code = code_generator.generate_code(
                specification="Create a function that calculates factorial",
                language="python",
                batch_size=1,
                num_iterations=2  # Try multiple iterations
            )
            print(f"Result: {'Success' if code else 'Empty result'}")
        except Exception as e:
            print(f"Expected error occurred: {type(e).__name__}: {str(e)}")


def test_api_error_handling():
    """Test error handling through the API layer."""
    print("\n=== Testing API error handling ===")
    
    # Test different error types
    error_types = ["dimension_mismatch", "empty_output", "none_output", "generic_error"]
    
    for error_type in error_types:
        print(f"\nTesting error type: {error_type}")
        diffusion_model = MockDiffusionModel(error_type=error_type)
        
        api = CodeGenerationAPI(diffusion_model, config={
            "batch_size": 1,
            "num_iterations": 2,
            "use_rewards": False
        })
        
        code, metadata = api.generate_code(
            specification="Create a function that sorts a list",
            language="python"
        )
        
        if code is None and metadata:
            print(f"API properly handled error: {metadata['error']['type']}")
            print(f"Error message: {metadata['error']['message']}")
            print(f"Error details: {metadata['error']['details']}")
            if "suggested_fix" in metadata["error"]:
                print(f"Suggested fix: {metadata['error']['suggested_fix']}")
        else:
            print("Expected an error but received a result:", code)


if __name__ == "__main__":
    try:
        # Test direct code generator error handling
        test_direct_code_generator()
        
        # Test API layer error handling
        test_api_error_handling()
        
        print("\nAll tests completed.")
    except Exception as e:
        print(f"Unexpected error during testing: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)