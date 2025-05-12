#!/usr/bin/env python3
"""
Example demonstrating the hybrid LLM + diffusion code generation approach.

This example shows how to use the hybrid approach to generate code with
improved quality compared to the standard diffusion-only approach.
"""

import logging
import time
import argparse
from typing import Dict, Any

from agentic_diffusion.api.code_generation_api import create_code_generation_api
from agentic_diffusion.api.hybrid_llm_diffusion_api import create_hybrid_llm_diffusion_api
from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hybrid_example")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Hybrid LLM + Diffusion vs Standard Diffusion"
    )
    
    parser.add_argument(
        "--specification",
        type=str,
        default="Write a function to calculate the Fibonacci sequence up to n terms",
        help="Code specification"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Target programming language"
    )
    
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        help="LLM provider for hybrid approach"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4",
        help="LLM model for hybrid approach"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of diffusion refinement iterations"
    )
    
    return parser.parse_args()


def print_separator(title=None):
    """Print a separator with optional title."""
    print("\n" + "=" * 80)
    if title:
        print(title)
        print("=" * 80)


def format_time(seconds):
    """Format time in seconds to a readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.2f} s"


def generate_code_standard_diffusion(specification, language):
    """Generate code using the standard diffusion approach."""
    print_separator("GENERATING CODE WITH STANDARD DIFFUSION")
    logger.info(f"Generating {language} code with standard diffusion approach")
    
    start_time = time.time()
    
    # Initialize the API
    diffusion_model = CodeDiffusion()
    diffusion_api = create_code_generation_api(diffusion_model)
    
    # Generate code
    code, metadata = diffusion_api.generate_code(
        specification=specification,
        language=language
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Standard diffusion generation completed in {format_time(elapsed_time)}")
    
    # Print the code
    print("\nSTANDARD DIFFUSION GENERATED CODE:")
    print(f"Language: {language}")
    print(f"Time: {format_time(elapsed_time)}\n")
    print(code)
    
    # Evaluate the code
    quality_metrics = diffusion_api.evaluate_code(code, language)
    
    print("\nQUALITY METRICS:")
    for key, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    return code, quality_metrics, elapsed_time


def generate_code_hybrid_approach(specification, language, llm_provider, llm_model, iterations):
    """Generate code using the hybrid LLM + diffusion approach."""
    print_separator("GENERATING CODE WITH HYBRID LLM + DIFFUSION")
    logger.info(f"Generating {language} code with hybrid approach")
    
    start_time = time.time()
    
    # Configure the hybrid approach
    config = {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "refinement_iterations": iterations,
        "temperature": 0.7
    }
    
    # Initialize the API
    hybrid_api = create_hybrid_llm_diffusion_api(config)
    
    # Generate code
    code, metadata = hybrid_api.generate_code(
        specification=specification,
        language=language
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Hybrid generation completed in {format_time(elapsed_time)}")
    
    # Print the code
    print("\nHYBRID LLM + DIFFUSION GENERATED CODE:")
    print(f"Language: {language}")
    print(f"LLM: {llm_provider}/{llm_model}")
    print(f"Refinement iterations: {iterations}")
    print(f"Time: {format_time(elapsed_time)}\n")
    print(code)
    
    # Show quality improvement
    quality_metrics = hybrid_api.evaluate_code(code, language)
    
    print("\nQUALITY METRICS:")
    for key, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    # Print timing breakdown
    if "timing" in metadata:
        print("\nTIMING BREAKDOWN:")
        for key, value in metadata["timing"].items():
            print(f"  {key}: {format_time(value)}")
    
    # Print quality improvement
    if "quality" in metadata and "quality_improvement_percentage" in metadata["quality"]:
        improvement = metadata["quality"]["quality_improvement_percentage"]
        print(f"\nQUALITY IMPROVEMENT: {improvement:.2f}%")
    
    return code, quality_metrics, elapsed_time, metadata


def compare_approaches(standard_result, hybrid_result):
    """Compare the standard and hybrid approaches."""
    standard_code, standard_metrics, standard_time = standard_result
    hybrid_code, hybrid_metrics, hybrid_time, hybrid_metadata = hybrid_result
    
    print_separator("COMPARISON: STANDARD DIFFUSION vs HYBRID APPROACH")
    
    # Calculate quality improvement
    if standard_metrics["overall"] > 0:
        quality_improvement = (
            (hybrid_metrics["overall"] - standard_metrics["overall"]) / 
            standard_metrics["overall"]
        ) * 100
    else:
        quality_improvement = 0.0
    
    # Calculate time difference
    time_ratio = hybrid_time / standard_time if standard_time > 0 else 0
    
    print("QUALITY COMPARISON:")
    print(f"  Standard diffusion: {standard_metrics['overall']:.3f}")
    print(f"  Hybrid approach:    {hybrid_metrics['overall']:.3f}")
    print(f"  Improvement:        {quality_improvement:.2f}%")
    
    print("\nTIMING COMPARISON:")
    print(f"  Standard diffusion: {format_time(standard_time)}")
    print(f"  Hybrid approach:    {format_time(hybrid_time)}")
    print(f"  Time ratio:         {time_ratio:.2f}x")
    
    print("\nSUMMARY:")
    if quality_improvement >= 15:
        print(f"  The hybrid approach achieved a significant quality improvement of {quality_improvement:.2f}%")
    elif quality_improvement > 0:
        print(f"  The hybrid approach achieved a modest quality improvement of {quality_improvement:.2f}%")
    else:
        print(f"  The hybrid approach did not improve quality (change: {quality_improvement:.2f}%)")
    
    print(f"  The hybrid approach took {time_ratio:.2f}x longer than standard diffusion")
    
    # Compute breakdown of where time was spent in hybrid approach
    if "timing" in hybrid_metadata:
        llm_time = hybrid_metadata["timing"].get("llm_generation_time", 0)
        diffusion_time = hybrid_metadata["timing"].get("diffusion_refinement_time", 0)
        
        if hybrid_time > 0:
            llm_percent = (llm_time / hybrid_time) * 100
            diffusion_percent = (diffusion_time / hybrid_time) * 100
            
            print(f"\nHYBRID TIME BREAKDOWN:")
            print(f"  LLM generation:      {format_time(llm_time)} ({llm_percent:.1f}%)")
            print(f"  Diffusion refinement: {format_time(diffusion_time)} ({diffusion_percent:.1f}%)")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate code with standard diffusion
    standard_result = generate_code_standard_diffusion(
        specification=args.specification,
        language=args.language
    )
    
    # Generate code with hybrid approach
    hybrid_result = generate_code_hybrid_approach(
        specification=args.specification,
        language=args.language,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        iterations=args.iterations
    )
    
    # Compare the approaches
    compare_approaches(standard_result, hybrid_result)


if __name__ == "__main__":
    main()
