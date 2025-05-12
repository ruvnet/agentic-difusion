#!/usr/bin/env python
"""
Complete test script for code generation using the improved dimension mismatch handling.
This script demonstrates the full code generation pipeline with error handling fixes.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("code_generation_test")

# Import necessary components
from agentic_diffusion.api.code_generation_api import CodeGenerationAPI
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.syntax_model import SyntaxModel
from agentic_diffusion.code_generation.code_generator import CodeGenerator

# Mock implementation for testing
class CodeDiffusionModelMock:
    """Mock diffusion model for testing the code generation API."""
    
    def __init__(self, behavior="success", dimension_check=True):
        """
        Initialize the mock diffusion model.
        
        Args:
            behavior: The behavior to simulate ("success", "dimension_error", "empty_output")
            dimension_check: Whether to perform dimension checks (simulating real models)
        """
        self.behavior = behavior
        self.dimension_check = dimension_check
        self.generation_count = 0
    
    def generate(self, specification, language, partial_code=None, **kwargs):
        """Mock implementation of generate method."""
        self.generation_count += 1
        logger.info(f"Mock diffusion model generate() called ({self.generation_count})")
        
        if self.behavior == "success":
            # Successful generation with dummy code
            return self._get_sample_code(specification, language)
        elif self.behavior == "dimension_error" and self.dimension_check:
            if self.generation_count < 2:  # Only fail on first attempt
                raise AssertionError("Embedding dimension mismatch: expected 128 but got 64")
            else:
                return self._get_sample_code(specification, language)
        elif self.behavior == "empty_output":
            if self.generation_count < 2:  # Only fail on first attempt
                return ""
            else:
                return self._get_sample_code(specification, language)
        else:
            return self._get_sample_code(specification, language)
    
    def sample(self, specification, language, partial_code=None, **kwargs):
        """Mock implementation of sample method."""
        logger.info("Mock diffusion model sample() called")
        
        if self.behavior == "success":
            return self._get_sample_code(specification, language)
        elif self.behavior == "dimension_error" and self.dimension_check:
            raise ValueError("Expected tensor shape [32, 64, 128] but got [32, 64, 256]")
        elif self.behavior == "empty_output":
            return ""
        else:
            return self._get_sample_code(specification, language)
    
    def _get_sample_code(self, specification, language):
        """Generate sample code based on specification."""
        if language == "python":
            return """
def factorial(n):
    \"\"\"Calculate the factorial of a number.\"\"\"
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

def main():
    number = 5
    result = factorial(number)
    print(f"The factorial of {number} is {result}")

if __name__ == "__main__":
    main()
"""
        elif language == "javascript":
            return """
function factorial(n) {
    // Calculate the factorial of a number
    if (n === 0 || n === 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

function main() {
    const number = 5;
    const result = factorial(number);
    console.log(`The factorial of ${number} is ${result}`);
}

main();
"""
        else:
            return f"// Generated code for {language}\nfunction example() {{ }}"
    
    def evaluate_code_quality(self, code, specification, language):
        """Mock implementation of code quality evaluation."""
        return {
            "syntax_score": 0.95,
            "quality_score": 0.85,
            "relevance_score": 0.9,
            "overall_score": 0.9,
            "syntax_correct": True
        }


def run_code_generation_test(behavior="success", with_error_handling=True):
    """
    Run a test of the code generation API with the specified behavior.
    
    Args:
        behavior: The behavior to simulate ("success", "dimension_error", "empty_output")
        with_error_handling: Whether to enable enhanced error handling
        
    Returns:
        tuple: (generated_code, metadata, elapsed_time)
    """
    logger.info(f"Testing code generation with behavior '{behavior}'")
    
    # Create mock diffusion model
    diffusion_model = CodeDiffusionModelMock(behavior=behavior)
    
    # Create API instance
    api_config = {
        "batch_size": 1,
        "precision": "float32",
        "device": "cpu",
        "guidance_scale": 1.5,
        "temperature": 0.7,
        "use_rewards": True,
        "max_length": 512,
        "num_iterations": 2,  # Try multiple iterations for better error handling
        "default_language": "python"
    }
    
    api = CodeGenerationAPI(diffusion_model, config=api_config)
    
    # Test specification
    specification = "Create a function that calculates the factorial of a number"
    language = "python"
    
    # Track timing
    start_time = time.time()
    
    # Generate code
    code, metadata = api.generate_code(
        specification=specification,
        language=language
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    return code, metadata, elapsed_time


def main():
    """Main function to run the code generation tests."""
    parser = argparse.ArgumentParser(description="Test code generation with error handling")
    parser.add_argument(
        "--behavior", 
        choices=["success", "dimension_error", "empty_output"],
        default="success",
        help="The behavior to simulate"
    )
    parser.add_argument(
        "--disable-error-handling",
        action="store_true",
        help="Disable enhanced error handling"
    )
    parser.add_argument(
        "--output-dir",
        default="test_results",
        help="Directory to save test results"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run the test
        with_error_handling = not args.disable_error_handling
        
        print(f"\n=== Testing code generation with behavior '{args.behavior}' ===")
        print(f"Enhanced error handling: {'enabled' if with_error_handling else 'disabled'}")
        
        code, metadata, elapsed_time = run_code_generation_test(
            behavior=args.behavior,
            with_error_handling=with_error_handling
        )
        
        # Display results
        print(f"\nGeneration completed in {elapsed_time:.2f} seconds")
        
        if code:
            print("\nGenerated Code:")
            print("=" * 40)
            print(code)
            print("=" * 40)
        else:
            print("\nNo code generated.")
        
        if metadata:
            print("\nMetadata:")
            print("-" * 40)
            if "error" in metadata:
                print(f"Error type: {metadata['error']['type']}")
                print(f"Error message: {metadata['error']['message']}")
                print(f"Error details: {metadata['error']['details']}")
                if "suggested_fix" in metadata["error"]:
                    print(f"Suggested fix: {metadata['error']['suggested_fix']}")
            elif "quality" in metadata:
                print(f"Syntax score: {metadata['quality']['syntax_score']:.2f}")
                print(f"Quality score: {metadata['quality']['quality_score']:.2f}")
                print(f"Relevance score: {metadata['quality']['relevance_score']:.2f}")
                print(f"Overall score: {metadata['quality']['overall_score']:.2f}")
            print("-" * 40)
        
        # Save results to file
        results_file = os.path.join(
            args.output_dir, 
            f"code_generation_{args.behavior}_results.json"
        )
        
        with open(results_file, "w") as f:
            json.dump({
                "behavior": args.behavior,
                "with_error_handling": with_error_handling,
                "elapsed_time": elapsed_time,
                "code": code,
                "metadata": metadata
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return 0  # Success
    
    except Exception as e:
        logger.error(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        return 1  # Error


if __name__ == "__main__":
    sys.exit(main())