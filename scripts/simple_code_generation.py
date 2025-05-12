#!/usr/bin/env python3
"""
Simple script to generate code using the standard diffusion approach.
"""

import argparse
import logging
import sys
import time

from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion
from agentic_diffusion.api.code_generation_api import create_code_generation_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("code_generation")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate code using standard diffusion"
    )
    
    parser.add_argument(
        "--specification",
        type=str,
        default="Write a function to calculate the factorial of a number using recursion",
        help="Code specification"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Target programming language"
    )
    
    return parser.parse_args()

def generate_code_standard_diffusion(specification, language):
    """Generate code using the standard diffusion approach."""
    print("\n" + "=" * 80)
    print("GENERATING CODE WITH STANDARD DIFFUSION")
    print("=" * 80)
    
    logger.info(f"Generating {language} code with standard diffusion approach")
    
    start_time = time.time()
    
    # Initialize the API
    diffusion_model = CodeDiffusion()
    diffusion_api = create_code_generation_api(diffusion_model)
    
    # Generate code
    try:
        code, metadata = diffusion_api.generate_code(
            specification=specification,
            language=language
        )
        
        elapsed_time = time.time() - start_time
        
        # Print the code
        print("\nSTANDARD DIFFUSION GENERATED CODE:")
        print(f"Language: {language}")
        print(f"Time: {elapsed_time:.2f}s\n")
        print(code)
        
        # Evaluate the code
        try:
            quality_metrics = diffusion_api.evaluate_code(code, language)
            
            print("\nQUALITY METRICS:")
            for key, value in quality_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Error evaluating code: {e}")
        
        return code
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        return None

def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate code with standard diffusion
    generated_code = generate_code_standard_diffusion(
        specification=args.specification,
        language=args.language
    )
    
    if generated_code is None:
        logger.error("Code generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()